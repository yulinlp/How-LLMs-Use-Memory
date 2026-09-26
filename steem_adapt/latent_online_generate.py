"""Dynamic Counterfactual Residual Transport (DCRT).

At every generated token, frozen full-context and leave-one-memory-out streams
measure each memory's current counterfactual marginal activation displacement
at one transformer layer.
The controlled full-context stream receives the policy-conditioned transport

    h_control <- h_full + alpha * sum_i c(policy_i) * (h_full - h_without_i)

with c(Ignore)=-1, c(Support)=0, and c(Dominate)=+1.  No loss is computed and
no model or auxiliary parameters are updated.  For a single Ignore memory and
alpha=1, the transported residual is exactly the leave-one-out residual at the
intervention layer on every decoding step.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .io import append_jsonl, load_done_uids, load_jsonl
from .mccs_generate import render_latent_prompt, sample_tokens_batch
from .modeling import get_decoder_layers


def transport_module(model, layer_index: int, site: str):
    if site == "block":
        return get_decoder_layers(model)[layer_index]
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    raise ValueError("Could not locate the model's final normalization module")


@contextmanager
def capture_last_residual(
    model, layer_index: int, sink: list[torch.Tensor], site: str = "block"
) -> Iterator[None]:
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        sink.append(hidden[:, -1, :].detach())

    handle = transport_module(model, layer_index, site).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def capture_selected_attention(
    layer_indices: set[int], sink: dict[int, torch.Tensor]
) -> Iterator[None]:
    """Capture selected Qwen attention maps without returning every layer.

    The attention implementation used by the ordinary executor is SDPA, which
    intentionally does not materialize attention probabilities.  Attention-
    aware candidates switch only their model instance to Qwen's eager
    implementation and intercept the selected layer there.  Keeping the
    capture in a context manager makes the patch process-local and leaves all
    existing runs unchanged.
    """
    if not layer_indices:
        yield
        return
    try:
        from transformers.models.qwen3 import modeling_qwen3 as qwen3
    except ImportError as exc:  # pragma: no cover - only non-Qwen backbones
        raise RuntimeError("attention-aware control currently requires Qwen3") from exc
    original = qwen3.eager_attention_forward

    def wrapped(module, query, key, value, attention_mask, scaling,
                dropout=0.0, **kwargs):
        output, weights = original(
            module, query, key, value, attention_mask,
            scaling=scaling, dropout=dropout, **kwargs
        )
        if getattr(module, "layer_idx", None) in layer_indices:
            # Only the selected layer is retained; Qwen's attention function
            # still returns its normal weights to the decoder layer, but no
            # all-layer attention output is accumulated by the model wrapper.
            sink[module.layer_idx] = weights.detach()
        return output, weights

    qwen3.eager_attention_forward = wrapped
    try:
        yield
    finally:
        qwen3.eager_attention_forward = original


def memory_token_positions(tokenizer, prompt: str, row: dict) -> dict[str, list[int]]:
    """Map benchmark memory spans to token positions in one rendered prompt.

    Missing memories are expected for leave-one-out and isolated streams.  We
    return only spans that are actually present, and never infer a position
    from a guessed fixed offset.
    """
    encoded = tokenizer(
        prompt, return_offsets_mapping=True, add_special_tokens=True
    )
    offsets = encoded["offset_mapping"]
    result: dict[str, list[int]] = {}
    cursor = 0
    for memory in row["memories"]:
        text = str(memory["memory_text"])
        start = prompt.find(text, cursor)
        if start < 0:
            continue
        end = start + len(text)
        positions = [
            index for index, (left, right) in enumerate(offsets)
            if left < end and right > start and right > left
        ]
        if positions:
            result[str(memory["memory_id"])] = positions
            cursor = end
    return result


def attention_mass_for_spans(
    attention: torch.Tensor,
    stream_index: int,
    spans: dict[str, list[int]],
    memory_ids: list[str],
) -> torch.Tensor:
    """Average selected-layer/head attention mass for each memory span."""
    values = []
    for memory_id in memory_ids:
        positions = spans.get(str(memory_id), [])
        if not positions:
            values.append(attention.new_zeros(()))
            continue
        valid = [position for position in positions if position < attention.shape[-1]]
        if not valid:
            values.append(attention.new_zeros(()))
            continue
        # Per-token mass avoids rewarding a memory merely because its text is
        # longer.  Head averaging is deliberately label-free and stable.
        values.append(attention[stream_index, :, -1, valid].float().mean())
    return torch.stack(values) if values else attention.new_empty((0,))


def normalize_attention_gate(mass: torch.Tensor) -> torch.Tensor:
    """Turn attention mass into a bounded, mean-one endogenous gate."""
    if mass.ndim != 2:
        raise ValueError("attention mass must have shape [batch, memory]")
    mean = mass.mean(dim=1, keepdim=True)
    relative = mass / mean.clamp_min(1e-8)
    # Square-root compression keeps every item active while preventing one
    # sharply attended token from dominating the entire composition.
    gate = torch.sqrt(relative.clamp_min(0.0))
    # A sliding-attention layer may not see any original memory token at the
    # current position.  In that case attention carries no routing evidence;
    # fall back to the unmodified direction instead of shrinking every item.
    gate = torch.where(mean > 1e-8, gate, torch.ones_like(gate))
    return gate.nan_to_num(1.0, 0.25, 4.0).clamp(0.25, 4.0)


@contextmanager
def transport_last_residual(
    model,
    layer_index: int,
    delta: torch.Tensor,
    site: str = "block",
    norm_mode: str = "none",
) -> Iterator[None]:
    """Add a previously measured per-sample delta at the current final token."""
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        steered = hidden.clone()
        base = steered[:, -1, :]
        candidate = base + delta.to(hidden.device, hidden.dtype)
        if norm_mode == "radial":
            candidate = radial_norm_projection(base, candidate)
        elif norm_mode == "tangent":
            candidate = spherical_tangent_transport(
                base, delta.to(hidden.device, hidden.dtype)
            )
        elif norm_mode != "none":
            raise ValueError(f"Unknown norm preservation mode: {norm_mode}")
        steered[:, -1, :] = candidate
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    handle = transport_module(model, layer_index, site).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def next_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.long().sum(dim=-1, keepdim=True).sub(1).clamp_min(0)


def full_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = attention_mask.long().cumsum(dim=-1).sub(1)
    return positions.masked_fill(attention_mask == 0, 0)


def radial_norm_projection(base: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Project candidate vectors to the per-example L2 sphere of base."""
    base_norm = base.float().norm(dim=-1, keepdim=True)
    candidate_norm = candidate.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return candidate * (base_norm / candidate_norm).to(candidate.dtype)


def spherical_tangent_transport(base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Move along the activation sphere using the delta's tangent component."""
    base32, delta32 = base.float(), delta.float()
    radius = base32.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit_base = base32 / radius
    tangent = delta32 - (delta32 * unit_base).sum(dim=-1, keepdim=True) * unit_base
    tangent_norm = tangent.norm(dim=-1, keepdim=True)
    unit_tangent = tangent / tangent_norm.clamp_min(1e-12)
    # A fixed trust region prevents a correlated multi-memory sum from rotating
    # past orthogonality in one decoding step.
    angle = (tangent_norm / radius).clamp_max(torch.pi / 2)
    moved = torch.cos(angle) * base32 + torch.sin(angle) * radius * unit_tangent
    return torch.where(tangent_norm > 1e-12, moved, base32).to(base.dtype)


def fp32_latent_readout(
    model, base: torch.Tensor, delta: torch.Tensor, norm_mode: str
) -> torch.Tensor:
    """Apply transport in float32 activation space, then the frozen LM head.

    Keeping the addition and radial projection in float32 prevents bfloat16
    rounding order from overwhelming small counterfactual displacements.  The
    intervention remains the LM-head input activation; no logit difference is
    constructed or edited.
    """
    if norm_mode == "tangent":
        candidate = spherical_tangent_transport(base, delta)
    else:
        candidate = base.float() + delta.float()
    if norm_mode == "radial":
        candidate = radial_norm_projection(base.float(), candidate)
    elif norm_mode not in ("none", "tangent"):
        raise ValueError(f"Unknown norm preservation mode: {norm_mode}")
    head = model.get_output_embeddings()
    # Converting a 150k-by-hidden LM head from BF16 on every decode token is a
    # large, completely redundant device copy.  Cache a detached read-only FP32
    # view once per loaded model.  This changes neither values nor parameters;
    # it only removes repeated dtype conversion from the inference hot path.
    weight = getattr(model, "_mccs_fp32_lm_head_weight", None)
    if weight is None or weight.device != head.weight.device:
        weight = head.weight.detach().float()
        model._mccs_fp32_lm_head_weight = weight
    head_bias = getattr(head, "bias", None)
    bias = getattr(model, "_mccs_fp32_lm_head_bias", None)
    if head_bias is not None and (bias is None or bias.device != head_bias.device):
        bias = head_bias.detach().float()
        model._mccs_fp32_lm_head_bias = bias
    return F.linear(
        candidate,
        weight,
        bias,
    )


def native_latent_readout(
    model, base: torch.Tensor, delta: torch.Tensor, norm_mode: str
) -> torch.Tensor:
    """Compose a probe activation and apply the frozen head in model dtype."""
    if norm_mode == "tangent":
        candidate = spherical_tangent_transport(base, delta)
    else:
        candidate = base + delta.to(base.dtype)
    if norm_mode == "radial":
        candidate = radial_norm_projection(base, candidate)
    elif norm_mode not in ("none", "tangent"):
        raise ValueError(f"Unknown norm preservation mode: {norm_mode}")
    return model.get_output_embeddings()(candidate)


def mixed_latent_readout(
    model, base: torch.Tensor, delta: torch.Tensor, norm_mode: str
) -> torch.Tensor:
    """Compose in model dtype, but normalize/read out the latent in float32."""
    native_candidate = base + delta.to(base.dtype)
    if norm_mode == "radial":
        candidate = radial_norm_projection(base.float(), native_candidate.float())
    elif norm_mode == "tangent":
        candidate = spherical_tangent_transport(base.float(), delta.float())
    elif norm_mode == "none":
        candidate = native_candidate.float()
    else:
        raise ValueError(f"Unknown norm preservation mode: {norm_mode}")
    head = model.get_output_embeddings()
    weight = getattr(model, "_mccs_fp32_lm_head_weight", None)
    if weight is None or weight.device != head.weight.device:
        weight = head.weight.detach().float()
        model._mccs_fp32_lm_head_weight = weight
    head_bias = getattr(head, "bias", None)
    bias = getattr(model, "_mccs_fp32_lm_head_bias", None)
    if head_bias is not None and (bias is None or bias.device != head_bias.device):
        bias = head_bias.detach().float()
        model._mccs_fp32_lm_head_bias = bias
    return F.linear(
        candidate,
        weight,
        bias,
    )


def policy_transport(
    residuals: torch.Tensor,
    policies: list[str | float] | list[list[str | float]],
    coefficients: dict[str, float],
    aggregation: str,
) -> torch.Tensor:
    """Combine per-memory displacements with raw or Gram-conditioned transport.

    The Gram variant returns the minimum-norm vector associated with the
    least-squares projection targets c_i * ||delta_i||. The targets are exact
    only when they lie in the Gram matrix range; nearly collinear directions
    can make the pseudoinverse sensitive. It equals the raw signed displacement
    for one memory and sums orthogonal memories.
    """
    single = residuals.ndim == 2
    if single:
        residuals = residuals.unsqueeze(0)
        policies = [policies]  # type: ignore[list-item]
    directions = residuals[:, :1, :] - residuals[:, 1:, :]
    c = torch.tensor(
        [[float(policy) if isinstance(policy, (int, float)) else coefficients[policy]
          for policy in sample] for sample in policies],
        device=directions.device,
        dtype=torch.float32,
    )
    # Avoid a tiny SVD and its synchronization cost on the common single-memory
    # setting; both aggregation rules reduce exactly to the signed raw delta.
    if aggregation in {"sum", "signed-mean"} or directions.shape[1] == 1:
        weighted = c.to(directions.dtype).unsqueeze(-1) * directions
        if aggregation == "signed-mean" and directions.shape[1] > 1:
            # Average positive and negative actions separately so memory-pool
            # cardinality cannot amplify one sign merely through duplication.
            # The action magnitude itself is retained: Support=.25 remains
            # weaker than Dominate=1 after count normalization.
            positive = (c > 0).to(weighted.dtype).unsqueeze(-1)
            negative = (c < 0).to(weighted.dtype).unsqueeze(-1)
            positive_mean = (weighted * positive).sum(dim=1) / positive.sum(
                dim=1
            ).clamp_min(1)
            negative_mean = (weighted * negative).sum(dim=1) / negative.sum(
                dim=1
            ).clamp_min(1)
            combined = positive_mean + negative_mean
        else:
            combined = weighted.sum(dim=1)
        return combined[0] if single else combined
    raw = directions.float()
    norms = raw.norm(dim=-1).clamp_min(1e-6)
    unit = raw / norms.unsqueeze(-1)
    gram = unit @ unit.transpose(-1, -2)
    target_projection = c * norms
    # pinv provides a deterministic closed-form solution under redundant or
    # contradictory memory directions; it is not an optimized model parameter.
    weights = (torch.linalg.pinv(gram, rtol=1e-4, hermitian=True) @ target_projection.unsqueeze(-1)).squeeze(-1)
    combined = torch.einsum("bm,bmd->bd", weights, unit).to(directions.dtype)
    return combined[0] if single else combined


def isolated_policy_transport(
    residuals: torch.Tensor,
    policies: list[list[str | float]],
    coefficients: dict[str, float],
    aggregation: str,
) -> torch.Tensor:
    """Transport directions measured in isolated single-memory contexts.

    The residual layout is ``[full-base, iso-full-0, iso-clean-0, ...]``.
    The full-context base remains the recipient state used for generation, but
    each control direction is measured as ``h(only memory i) -
    h(no-memory)``.  This is intentionally an RQ2 diagnostic: it tests
    whether a cleaner policy direction transfers into a full context.
    """
    if residuals.ndim != 3:
        raise ValueError("isolated transport expects [batch, streams, hidden]")
    if residuals.shape[1] < 3 or (residuals.shape[1] - 1) % 2:
        raise ValueError("isolated stream layout must be [base, iso-full, iso-clean]*")
    directions = residuals[:, 1::2] - residuals[:, 2::2]
    c = torch.tensor(
        [[
            float(policy) if isinstance(policy, (int, float)) else coefficients[policy]
            for policy in sample
        ] for sample in policies],
        device=directions.device,
        dtype=torch.float32,
    )
    weighted = c.to(directions.dtype).unsqueeze(-1) * directions
    if aggregation == "sum":
        return weighted.sum(dim=1)
    if aggregation == "signed-mean":
        positive = (c > 0).to(weighted.dtype).unsqueeze(-1)
        negative = (c < 0).to(weighted.dtype).unsqueeze(-1)
        positive_mean = (weighted * positive).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
        negative_mean = (weighted * negative).sum(dim=1) / negative.sum(dim=1).clamp_min(1)
        return positive_mean + negative_mean
    if aggregation == "gram":
        # Geometry-aware isolated composition.  Treat each isolated direction
        # as a constraint on the desired signed projection, then solve the
        # minimum-norm joint intervention in their span.  This preserves the
        # RQ2 isolated directions while correcting direct-sum cancellation or
        # amplification when directions are correlated.
        raw = directions.float()
        norms = raw.norm(dim=-1).clamp_min(1e-6)
        unit = raw / norms.unsqueeze(-1)
        gram = unit @ unit.transpose(-1, -2)
        target_projection = c * norms
        weights = (
            torch.linalg.pinv(gram, rtol=1e-4, hermitian=True)
            @ target_projection.unsqueeze(-1)
        ).squeeze(-1)
        combined = torch.einsum("bm,bmd->bd", weights, unit)
        return combined.to(directions.dtype)
    raise ValueError("isolated direction context currently supports sum or signed-mean aggregation")


def hybrid_policy_transport(
    residuals: torch.Tensor,
    policies: list[list[str | float]],
    coefficients: dict[str, float],
    aggregation: str,
    context_mix: float = 0.5,
) -> torch.Tensor:
    """Compose isolated policy directions with a context-compatibility gate.

    The stream layout is ``[full, iso-full-i, iso-clean-i, full-without-i]``.
    Policy identity still comes from the isolated direction (the RQ1 signal),
    while the full-context leave-one-out effect supplies an endogenous test of
    whether that direction transfers to the current memory set.  We norm-match
    the contextual effect to the isolated one and interpolate only when their
    cosine similarity is positive.  This is deliberately a small, training-
    free correction to cross-memory interference, not a second policy reader.
    """
    if residuals.ndim != 3 or residuals.shape[1] < 4 or (residuals.shape[1] - 1) % 3:
        raise ValueError("hybrid transport expects [base, iso-full, iso-clean, full-LOO]*")
    if not 0.0 <= context_mix <= 1.0:
        raise ValueError("context_mix must lie in [0, 1]")
    isolated = residuals[:, 1::3] - residuals[:, 2::3]
    contextual = residuals[:, :1] - residuals[:, 3::3]
    c = torch.tensor(
        [[float(policy) if isinstance(policy, (int, float)) else coefficients[policy]
          for policy in sample] for sample in policies],
        device=isolated.device,
        dtype=torch.float32,
    )
    isolated_float = isolated.float()
    contextual_float = contextual.float()
    iso_norm = isolated_float.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    ctx_norm = contextual_float.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cosine = (isolated_float * contextual_float).sum(dim=-1, keepdim=True) / (iso_norm * ctx_norm)
    compatibility = cosine.clamp_min(0.0)
    contextual_matched = contextual_float * (iso_norm / ctx_norm)
    mix = context_mix * compatibility
    effective = (1.0 - mix) * isolated_float + mix * contextual_matched
    weighted = c.to(effective.dtype).unsqueeze(-1) * effective
    if aggregation == "signed-mean":
        positive = (c > 0).to(weighted.dtype).unsqueeze(-1)
        negative = (c < 0).to(weighted.dtype).unsqueeze(-1)
        return ((weighted * positive).sum(1) / positive.sum(1).clamp_min(1)
                + (weighted * negative).sum(1) / negative.sum(1).clamp_min(1)).to(residuals.dtype)
    if aggregation != "sum":
        raise ValueError("hybrid context transport supports sum or signed-mean")
    return weighted.sum(dim=1).to(residuals.dtype)


def joint_isolated_policy_transport(
    residuals: torch.Tensor,
    policies: list[list[str | float]],
    coefficients: dict[str, float],
    aggregation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use a jointly cleaned anchor and isolated positive policy directions.

    The stream layout is ``[joint-clean, iso-full-i, iso-clean-i, ...]``.
    Memories whose action is negative are removed jointly in the anchor; this
    avoids repeatedly subtracting independent single-memory directions for a
    group of ignored memories.  Non-negative actions are then injected from
    the isolated RQ1 directions.  The separation is intentional: the joint
    stream handles context cleanup, while the isolated streams provide the
    policy-bearing control signal.
    """
    if residuals.ndim != 3 or residuals.shape[1] < 3 or (residuals.shape[1] - 1) % 2:
        raise ValueError("joint-isolated layout must be [joint-clean, iso-full, iso-clean]*")
    clean = residuals[:, 0]
    directions = residuals[:, 1::2] - residuals[:, 2::2]
    c = torch.tensor(
        [[
            float(policy) if isinstance(policy, (int, float)) else coefficients[policy]
            for policy in sample
        ] for sample in policies],
        device=directions.device,
        dtype=torch.float32,
    )
    # The anchor has already handled all negative actions.  Positive actions
    # retain their calibrated magnitudes; zero-valued support actions remain
    # a no-op.  Signed-mean keeps the scale stable when many memories agree.
    positive = c.clamp_min(0.0)
    weighted = positive.to(directions.dtype).unsqueeze(-1) * directions
    if aggregation == "signed-mean":
        count = (positive > 0).to(weighted.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
        addition = weighted.sum(dim=1) / count
    elif aggregation == "sum":
        addition = weighted.sum(dim=1)
    else:
        raise ValueError("joint-isolated transport supports sum or signed-mean")
    return clean, addition


def grouped_policy_transport(
    residuals: torch.Tensor,
    policies: list[list[str]],
    coefficients: dict[str, float],
) -> torch.Tensor:
    """Jointly remove Ignore memories, then amplify memories from that clean base.

    Streams are [full, without-all-ignore, clean-base-without-memory-0, ...].
    This retains cross-memory interactions that independent leave-one-out deltas
    discard, while leaving the controlled model's actual prompt unchanged.
    """
    full, clean = residuals[:, 0], residuals[:, 1]
    output = torch.zeros_like(full)
    for batch_index, sample_policies in enumerate(policies):
        ignore_coefficients = [coefficients[p] for p in sample_policies if p == "ignore"]
        if ignore_coefficients:
            output[batch_index].add_(ignore_coefficients[0] * (full[batch_index] - clean[batch_index]))
        for memory_index, policy in enumerate(sample_policies):
            if policy == "ignore":
                continue
            coefficient = coefficients[policy]
            if coefficient:
                without_memory = residuals[batch_index, memory_index + 2]
                output[batch_index].add_(coefficient * (clean[batch_index] - without_memory))
    return output


def grouped_clean_latent_transport(
    residuals: torch.Tensor,
    policies: list[list[str]],
    coefficients: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the jointly-clean base and only its non-Ignore additions.

    This is algebraically the same grouped policy as ``full-(full-clean)`` at
    unit Ignore strength, but avoids catastrophic cancellation and makes the
    latent base identical to the clean counterfactual stream used by the logit
    executor.
    """
    clean = residuals[:, 1]
    addition = torch.zeros_like(clean)
    for batch_index, sample_policies in enumerate(policies):
        for memory_index, policy in enumerate(sample_policies):
            if policy == "ignore":
                continue
            coefficient = coefficients[policy]
            if coefficient:
                without_memory = residuals[batch_index, memory_index + 2]
                addition[batch_index].add_(
                    coefficient * (clean[batch_index] - without_memory)
                )
    return clean, addition


def sparse_grouped_clean_latent_transport(
    residuals: torch.Tensor,
    stream_slices: list[slice],
    policies: list[list[str]],
    coefficients: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose variable-size [clean, adjusted counterfactuals...] groups."""
    bases, additions = [], []
    for stream_slice, sample_policies in zip(stream_slices, policies):
        sample = residuals[stream_slice]
        clean = sample[0]
        addition = torch.zeros_like(clean)
        cursor = 1
        for policy in sample_policies:
            coefficient = 0.0 if policy == "ignore" else coefficients[policy]
            if coefficient:
                addition.add_(coefficient * (clean - sample[cursor]))
                cursor += 1
        if cursor != len(sample):
            raise RuntimeError("Sparse grouped stream layout is inconsistent")
        bases.append(clean)
        additions.append(addition)
    return torch.stack(bases), torch.stack(additions)


def action_coefficient(
    policy: str | float, coefficients: dict[str, float]
) -> float:
    """Resolve a categorical or already-numeric memory action."""
    return float(policy) if isinstance(policy, (int, float)) else coefficients[policy]


def policy_grouped_transport(
    residuals: torch.Tensor,
    stream_slices: list[slice],
    group_coefficients: list[list[float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose joint negative/positive memory-group displacements.

    Each variable-size sample layout is ``[full, without-negative?,
    without-positive?]``.  The untouched full-context activation is always the
    generation base.  Memories with zero action require no stream.  When a
    benchmark uses graded actions of the same sign, their mean action scales
    the joint group displacement; RPEval's -1/0/+1 actions reduce exactly to
    ``-(full-without-ignore) + (full-without-dominate)``.
    """
    bases, additions = [], []
    for stream_slice, sample_coefficients in zip(
        stream_slices, group_coefficients
    ):
        sample = residuals[stream_slice]
        base = sample[0]
        addition = torch.zeros_like(base)
        if len(sample) != 1 + len(sample_coefficients):
            raise RuntimeError("Policy-grouped stream layout is inconsistent")
        for branch, coefficient in zip(sample[1:], sample_coefficients):
            addition.add_(coefficient * (base - branch))
        bases.append(base)
        additions.append(addition)
    return torch.stack(bases), torch.stack(additions)


def policy_ordered_transport(
    residuals: torch.Tensor,
    stream_slices: list[slice],
    group_specs: list[dict[str, float | bool]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply negative and positive memory actions in causal order.

    A sample is laid out as ``[full, without-negative?,
    without-negative-and-positive?]``.  Positive influence is therefore
    measured after the negative group has been removed, rather than against
    the original full context.  This retains the interaction term that the
    parallel ``policy-grouped`` composition drops.

    For the canonical RPEval action ``Ignore=-1``, the clean counterfactual is
    used directly as the readout base.  Radial projection then preserves the
    clean state's norm instead of projecting it back to the confounded full
    state.  Other negative strengths retain the full state as the base and are
    interpolated in closed form.
    """
    bases, additions = [], []
    for stream_slice, spec in zip(stream_slices, group_specs):
        sample = residuals[stream_slice]
        cursor = 1
        full = sample[0]
        has_negative = bool(spec["has_negative"])
        has_positive = bool(spec["has_positive"])
        negative_coefficient = float(spec["negative_coefficient"])
        positive_coefficient = float(spec["positive_coefficient"])

        clean = sample[cursor] if has_negative else full
        cursor += int(has_negative)
        without_positive = sample[cursor] if has_positive else clean
        cursor += int(has_positive)
        if cursor != len(sample):
            raise RuntimeError("Policy-ordered stream layout is inconsistent")

        if has_negative and abs(negative_coefficient + 1.0) < 1e-8:
            base = clean
            addition = torch.zeros_like(base)
        else:
            base = full
            addition = negative_coefficient * (full - clean)
        if has_positive:
            addition = addition + positive_coefficient * (
                clean - without_positive
            )
        bases.append(base)
        additions.append(addition)
    return torch.stack(bases), torch.stack(additions)


def select_cache_rows(past_key_values, indices: torch.Tensor):
    """Keep selected batch rows from a Transformers cache without recompute."""
    if hasattr(past_key_values, "batch_select_indices"):
        past_key_values.batch_select_indices(indices)
        return past_key_values
    # Compatibility with the legacy tuple-of-layer-tuples cache format.
    return tuple(
        tuple(value.index_select(0, indices.to(value.device)) for value in layer)
        for layer in past_key_values
    )


def counterfactual_stream_count(
    policies: list[str | float],
    coefficients: dict[str, float],
    counterfactual_mode: str,
) -> int:
    """Return the number of synchronized decoder streams actually executed."""
    if counterfactual_mode == "leave-one-out":
        return len(policies) + 1
    if counterfactual_mode in {"policy-grouped", "policy-ordered"}:
        actions = [action_coefficient(policy, coefficients) for policy in policies]
        return 1 + int(any(value < 0 for value in actions)) + int(
            any(value > 0 for value in actions)
        )
    # Grouped execution folds all Ignore actions into one base prompt and
    # allocates another stream only for a non-zero per-memory adjustment.
    return 1 + sum(
        policy != "ignore" and float(coefficients[policy]) != 0.0
        for policy in policies
    )


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    rows: list[dict],
    policies: list[list[str]],
    layer: int,
    alpha: float,
    coefficients: dict[str, float],
    aggregation: str,
    counterfactual_mode: str,
    transport_site: str,
    norm_mode: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    readout_precision: str,
    direction_refresh: str,
    direction_context: str = "full",
    attention_control: str = "none",
) -> list[str]:
    if not rows:
        return []
    memory_count = len(rows[0]["memories"])
    if any(len(row["memories"]) != memory_count for row in rows):
        raise ValueError("DCRT batches must group samples with the same number of memories")
    composite_readout = readout_precision in ("fp32", "mixed", "probe-native")
    # The grouped stream layout depends on the policy actions, not on how the
    # final activation is read out.  Native-hook execution also needs only the
    # clean stream plus non-zero adjusted counterfactuals.
    sparse_grouped = counterfactual_mode == "grouped"
    policy_grouped = counterfactual_mode == "policy-grouped"
    policy_ordered = counterfactual_mode == "policy-ordered"
    isolated_context = direction_context == "isolated"
    hybrid_context = direction_context == "hybrid"
    joint_isolated_context = direction_context == "joint-isolated"
    if direction_context not in {"full", "isolated", "hybrid", "joint-isolated"}:
        raise ValueError(f"Unknown direction context: {direction_context}")
    if (isolated_context or hybrid_context or joint_isolated_context) and counterfactual_mode != "leave-one-out":
        raise ValueError("isolated/hybrid/joint-isolated direction context requires leave-one-out mode")
    if (hybrid_context or joint_isolated_context) and not composite_readout:
        raise ValueError(
            "hybrid/joint-isolated direction context requires a composite latent readout"
        )
    if attention_control not in {"none", "steer", "extract", "both"}:
        raise ValueError(f"Unknown attention control: {attention_control}")
    if attention_control != "none":
        if not isolated_context or counterfactual_mode != "leave-one-out":
            raise ValueError(
                "attention-aware candidates currently require isolated leave-one-out streams"
            )
        if not composite_readout or transport_site != "final-norm":
            raise ValueError(
                "attention-aware candidates require composite final-norm readout"
            )
    variable_streams = sparse_grouped or policy_grouped or policy_ordered
    full_prompts, probe_prompts = [], []
    stream_slices: list[slice] = []
    stream_to_sample: list[int] = []
    policy_group_coefficients: list[list[float]] = []
    policy_ordered_specs: list[dict[str, float | bool]] = []
    for sample_index, (row, sample_policies) in enumerate(zip(rows, policies)):
        full = render_latent_prompt(tokenizer, row, set())
        full_prompts.append(full)
        if sparse_grouped:
            ignored = {
                memory["memory_id"]
                for memory, policy in zip(row["memories"], sample_policies)
                if policy == "ignore"
            }
            start = len(probe_prompts)
            probe_prompts.append(render_latent_prompt(tokenizer, row, ignored))
            for memory, policy in zip(row["memories"], sample_policies):
                coefficient = 0.0 if policy == "ignore" else coefficients[policy]
                if coefficient:
                    probe_prompts.append(
                        render_latent_prompt(
                            tokenizer, row, ignored | {memory["memory_id"]}
                        )
                    )
            stream_slices.append(slice(start, len(probe_prompts)))
            stream_to_sample.extend([sample_index] * (len(probe_prompts) - start))
            continue
        if policy_grouped:
            start = len(probe_prompts)
            probe_prompts.append(full)
            values = [
                action_coefficient(policy, coefficients)
                for policy in sample_policies
            ]
            sample_group_coefficients = []
            for sign in (-1, 1):
                selected = [
                    (memory, value)
                    for memory, value in zip(row["memories"], values)
                    if (value < 0 if sign < 0 else value > 0)
                ]
                if not selected:
                    continue
                removed = {memory["memory_id"] for memory, _ in selected}
                probe_prompts.append(render_latent_prompt(tokenizer, row, removed))
                sample_group_coefficients.append(
                    sum(value for _, value in selected) / len(selected)
                )
            stream_slices.append(slice(start, len(probe_prompts)))
            stream_to_sample.extend([sample_index] * (len(probe_prompts) - start))
            policy_group_coefficients.append(sample_group_coefficients)
            continue
        if policy_ordered:
            start = len(probe_prompts)
            probe_prompts.append(full)
            values = [
                action_coefficient(policy, coefficients)
                for policy in sample_policies
            ]
            negative = [
                (memory, value)
                for memory, value in zip(row["memories"], values)
                if value < 0
            ]
            positive = [
                (memory, value)
                for memory, value in zip(row["memories"], values)
                if value > 0
            ]
            removed_negative = {memory["memory_id"] for memory, _ in negative}
            if negative:
                probe_prompts.append(
                    render_latent_prompt(tokenizer, row, removed_negative)
                )
            if positive:
                removed_positive = {
                    memory["memory_id"] for memory, _ in positive
                }
                probe_prompts.append(
                    render_latent_prompt(
                        tokenizer, row, removed_negative | removed_positive
                    )
                )
            stream_slices.append(slice(start, len(probe_prompts)))
            stream_to_sample.extend(
                [sample_index] * (len(probe_prompts) - start)
            )
            policy_ordered_specs.append(
                {
                    "has_negative": bool(negative),
                    "has_positive": bool(positive),
                    "negative_coefficient": (
                        sum(value for _, value in negative) / len(negative)
                        if negative
                        else 0.0
                    ),
                    "positive_coefficient": (
                        sum(value for _, value in positive) / len(positive)
                        if positive
                        else 0.0
                    ),
                }
            )
            continue
        if joint_isolated_context:
            ignored = {
                memory["memory_id"]
                for memory, policy in zip(row["memories"], sample_policies)
                if action_coefficient(policy, coefficients) < 0
            }
            # Use one jointly cleaned stream as the recipient state.  Negative
            # actions are therefore handled once at the context level instead
            # of being repeatedly subtracted as independent item directions.
            probe_prompts.append(render_latent_prompt(tokenizer, row, ignored))
            for memory in row["memories"]:
                isolated_row = dict(row)
                isolated_row["memories"] = [memory]
                probe_prompts.append(render_latent_prompt(tokenizer, isolated_row, set()))
                probe_prompts.append(
                    render_latent_prompt(tokenizer, isolated_row, {memory["memory_id"]})
                )
            continue
        probe_prompts.append(full)
        if isolated_context:
            for memory in row["memories"]:
                isolated_row = dict(row)
                isolated_row["memories"] = [memory]
                probe_prompts.append(render_latent_prompt(tokenizer, isolated_row, set()))
                probe_prompts.append(
                    render_latent_prompt(tokenizer, isolated_row, {memory["memory_id"]})
                )
        elif hybrid_context:
            for memory in row["memories"]:
                isolated_row = dict(row)
                isolated_row["memories"] = [memory]
                probe_prompts.append(render_latent_prompt(tokenizer, isolated_row, set()))
                probe_prompts.append(
                    render_latent_prompt(tokenizer, isolated_row, {memory["memory_id"]})
                )
                probe_prompts.append(
                    render_latent_prompt(tokenizer, row, {memory["memory_id"]})
                )
        elif counterfactual_mode == "leave-one-out":
            probe_prompts.extend(
                render_latent_prompt(tokenizer, row, {memory["memory_id"]})
                for memory in row["memories"]
            )
        else:
            ignored = {
                memory["memory_id"] for memory, policy in zip(row["memories"], sample_policies)
                if policy == "ignore"
            }
            probe_prompts.append(render_latent_prompt(tokenizer, row, ignored))
            probe_prompts.extend(
                render_latent_prompt(
                    tokenizer, row,
                    ignored if policy == "ignore" else ignored | {memory["memory_id"]},
                )
                for memory, policy in zip(row["memories"], sample_policies)
            )
    streams_per_sample = (
        1 + 2 * memory_count
        if isolated_context or joint_isolated_context
        else 1 + 3 * memory_count
        if hybrid_context
        else memory_count + (1 if counterfactual_mode == "leave-one-out" else 2)
    )
    # Flattened as B fixed-size counterfactual groups.
    probe = tokenizer(probe_prompts, return_tensors="pt", padding=True).to(model.device)
    attention_enabled = attention_control != "none"
    attention_layer_indices = {layer} if attention_enabled else set()
    probe_stream_spans: list[dict[str, list[int]]] = []
    memory_ids_by_sample = [
        [str(memory["memory_id"]) for memory in row["memories"]]
        for row in rows
    ]
    if attention_enabled:
        # Attention positions are measured before batch left-padding and then
        # shifted into the rectangular probe batch.  The prompt renderer is
        # the single source of truth for the span lookup.
        padded_length = probe["input_ids"].shape[1]
        probe_sample_indices = [
            stream_index // streams_per_sample
            for stream_index in range(len(probe_prompts))
        ]
        for stream_index, prompt in enumerate(probe_prompts):
            sample_index = probe_sample_indices[stream_index]
            encoded = tokenizer(prompt, add_special_tokens=True)
            unpadded_length = len(encoded["input_ids"])
            shift = padded_length - unpadded_length
            spans = memory_token_positions(tokenizer, prompt, rows[sample_index])
            probe_stream_spans.append(
                {
                    memory_id: [position + shift for position in positions]
                    for memory_id, positions in spans.items()
                }
            )
    control = None
    # Prefill-frozen steering probes the counterfactual direction once, then
    # decodes on a single clean stream instead of retaining every LOO stream.
    if readout_precision == "native":
        control = tokenizer(full_prompts, return_tensors="pt", padding=True).to(model.device)
    probe_capture: list[torch.Tensor] = []
    attention_sink: dict[int, torch.Tensor] = {}
    # Composite readout only consumes the final normalized activation and KV
    # cache.  Calling the causal-LM wrapper here would unnecessarily project
    # every counterfactual stream through the large vocabulary head before we
    # perform the one actual latent readout below.  The backbone is exactly the
    # same frozen computation and returns the same cache.
    probe_model = model.model if composite_readout and hasattr(model, "model") else model
    attention_context = (
        capture_selected_attention(attention_layer_indices, attention_sink)
        if attention_enabled else nullcontext()
    )
    with attention_context:
        with capture_last_residual(model, layer, probe_capture, transport_site):
            probe_out = probe_model(
                **probe,
                position_ids=full_position_ids(probe["attention_mask"]),
                use_cache=True,
            )
    probe_past = probe_out.past_key_values
    probe_mask = probe["attention_mask"]
    captured = probe_capture.pop()
    residuals = captured if variable_streams else captured.reshape(len(rows), streams_per_sample, -1)
    transport_residuals = residuals.float() if readout_precision == "fp32" else residuals
    def apply_attention_state(
        current_residuals: torch.Tensor,
        current_attention: torch.Tensor,
    ) -> tuple[torch.Tensor, list[list[float | str]]]:
        """Apply attention to isolated directions and/or their token gates."""
        full_mass = []
        isolated_mass = []
        for sample_index in range(len(rows)):
            base_index = sample_index * streams_per_sample
            sample_memory_ids = memory_ids_by_sample[sample_index]
            full_mass.append(
                attention_mass_for_spans(
                    current_attention, base_index,
                    probe_stream_spans[base_index], sample_memory_ids
                )
            )
            isolated_mass.append(
                torch.stack([
                    attention_mass_for_spans(
                        current_attention,
                        base_index + 1 + 2 * memory_index,
                        probe_stream_spans[base_index + 1 + 2 * memory_index],
                        sample_memory_ids,
                    )[memory_index]
                    for memory_index in range(memory_count)
                ])
            )
        full_gate = normalize_attention_gate(torch.stack(full_mass))
        isolated_gate = normalize_attention_gate(torch.stack(isolated_mass))
        updated = current_residuals
        if attention_control in {"extract", "both"}:
            updated = updated.clone()
            directions = updated[:, 1::2] - updated[:, 2::2]
            updated[:, 1::2] = (
                updated[:, 2::2]
                + directions * isolated_gate.unsqueeze(-1).to(directions.dtype)
            )
        updated_policies: list[list[float | str]] = policies
        if attention_control in {"steer", "both"}:
            base_values = torch.tensor(
                [[action_coefficient(policy, coefficients) for policy in sample]
                 for sample in policies],
                device=updated.device,
                dtype=torch.float32,
            )
            updated_policies = (base_values * full_gate).tolist()
        return updated, updated_policies

    effective_policies = policies
    if attention_enabled:
        attention_map = attention_sink.get(layer)
        if attention_map is None:
            raise RuntimeError(
                f"No eager attention map captured at requested layer {layer}"
            )
        # In isolated layout each sample starts with the full stream and then
        # has [only-i, empty-i] pairs.  The isolated map determines how much
        # of each per-memory direction is actually read out; the full stream
        # map determines the token-level steering gate.
        transport_residuals, effective_policies = apply_attention_state(
            transport_residuals, attention_map
        )
    if sparse_grouped:
        latent_base, combined = sparse_grouped_clean_latent_transport(
            transport_residuals, stream_slices, policies, coefficients
        )
    elif policy_grouped:
        latent_base, combined = policy_grouped_transport(
            transport_residuals, stream_slices, policy_group_coefficients
        )
    elif policy_ordered:
        latent_base, combined = policy_ordered_transport(
            transport_residuals, stream_slices, policy_ordered_specs
        )
    elif readout_precision == "fp32" and counterfactual_mode == "grouped":
        latent_base, combined = grouped_clean_latent_transport(
            transport_residuals, policies, coefficients
        )
    elif isolated_context:
        latent_base = transport_residuals[:, 0]
        combined = isolated_policy_transport(
            transport_residuals, effective_policies, coefficients, aggregation
        )
    elif hybrid_context:
        latent_base = transport_residuals[:, 0]
        combined = hybrid_policy_transport(
            transport_residuals, policies, coefficients, aggregation
        )
    elif joint_isolated_context:
        latent_base, combined = joint_isolated_policy_transport(
            transport_residuals, policies, coefficients, aggregation
        )
    else:
        latent_base = transport_residuals[:, 0]
        combined = (
            policy_transport(transport_residuals, policies, coefficients, aggregation)
            if counterfactual_mode == "leave-one-out"
            else grouped_policy_transport(transport_residuals, policies, coefficients)
        )
    if composite_readout:
        if transport_site != "final-norm":
            raise ValueError("Composite latent readout is only defined at --transport-site final-norm")
        readout = {
            "fp32": fp32_latent_readout,
            "mixed": mixed_latent_readout,
            "probe-native": native_latent_readout,
        }[readout_precision]
        logits = readout(model, latent_base, alpha * combined, norm_mode)
        if direction_refresh == "prefill":
            if variable_streams:
                full_indices = torch.tensor(
                    [stream_slice.start for stream_slice in stream_slices],
                    device=model.device,
                    dtype=torch.long,
                )
            else:
                full_indices = torch.arange(
                    0,
                    len(rows) * streams_per_sample,
                    streams_per_sample,
                    device=model.device,
                    dtype=torch.long,
                )
            control_past = select_cache_rows(probe_past, full_indices)
            control_mask = probe_mask.index_select(
                0, full_indices.to(probe_mask.device)
            )
        else:
            control_past = None
            control_mask = None
    else:
        assert control is not None
        with transport_last_residual(
            model, layer, alpha * combined, transport_site, norm_mode
        ):
            control_out = model(
                **control,
                position_ids=full_position_ids(control["attention_mask"]),
                use_cache=True,
            )
        logits = control_out.logits[:, -1, :]
        control_past = control_out.past_key_values
        control_mask = control["attention_mask"]

    if direction_refresh == "prefill":
        # The counterfactual streams have served their only purpose.  Releasing
        # their KV cache before decoding leaves memory for a larger one-stream
        # batch; keeping it alive silently erased much of the expected speedup.
        probe_past = None
        probe_mask = None
        del probe_out, probe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    stream_to_sample_tensor = (
        torch.tensor(stream_to_sample, device=model.device, dtype=torch.long)
        if variable_streams else None
    )
    generated: list[list[int]] = [[] for _ in rows]
    finished = torch.zeros(len(rows), dtype=torch.bool, device=model.device)
    generators = []
    for row in rows:
        stable_id = int(hashlib.sha256(str(row["sample_id"]).encode()).hexdigest()[:8], 16)
        generator = torch.Generator(device=model.device)
        generator.manual_seed(seed + stable_id)
        generators.append(generator)
    eos_ids = tokenizer.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]
    fallback_eos = eos_ids[0]
    fixed_combined = combined.clone() if direction_refresh == "prefill" else None
    for _ in range(max_new_tokens):
        token = sample_tokens_batch(logits.float(), temperature, top_p, generators)
        for sample_index, token_id in enumerate(token.tolist()):
            if finished[sample_index]:
                continue
            if token_id in eos_ids:
                finished[sample_index] = True
            else:
                generated[sample_index].append(token_id)
        if bool(finished.all()):
            break
        token = torch.where(finished, torch.full_like(token, fallback_eos), token)

        if direction_refresh == "prefill":
            assert control_mask is not None and control_past is not None
            control_mask = torch.cat(
                [control_mask, torch.ones_like(control_mask[:, :1])], dim=1
            )
            decode_capture: list[torch.Tensor] = []
            with capture_last_residual(model, layer, decode_capture, transport_site):
                control_out = probe_model(
                    input_ids=token.unsqueeze(1),
                    attention_mask=control_mask,
                    position_ids=next_position_ids(control_mask),
                    past_key_values=control_past,
                    use_cache=True,
                )
            control_past = control_out.past_key_values
            latent_base = decode_capture.pop()
            assert fixed_combined is not None
            logits = readout(model, latent_base, alpha * fixed_combined, norm_mode)
            continue

        probe_token = (
            token[stream_to_sample_tensor].unsqueeze(1)
            if variable_streams else token.repeat_interleave(streams_per_sample).unsqueeze(1)
        )
        probe_mask = torch.cat([probe_mask, torch.ones_like(probe_mask[:, :1])], dim=1)
        probe_capture = []
        attention_sink = {}
        attention_context = (
            capture_selected_attention(attention_layer_indices, attention_sink)
            if attention_enabled else nullcontext()
        )
        with attention_context:
            with capture_last_residual(model, layer, probe_capture, transport_site):
                probe_out = probe_model(
                    input_ids=probe_token,
                    attention_mask=probe_mask,
                    position_ids=next_position_ids(probe_mask),
                    past_key_values=probe_past,
                    use_cache=True,
                )
        probe_past = probe_out.past_key_values
        captured = probe_capture.pop()
        residuals = captured if variable_streams else captured.reshape(len(rows), streams_per_sample, -1)
        transport_residuals = residuals.float() if readout_precision == "fp32" else residuals
        effective_policies = policies
        if attention_enabled:
            attention_map = attention_sink.get(layer)
            if attention_map is None:
                raise RuntimeError(
                    f"No eager attention map captured at requested layer {layer}"
                )
            transport_residuals, effective_policies = apply_attention_state(
                transport_residuals, attention_map
            )
        if sparse_grouped:
            latent_base, combined = sparse_grouped_clean_latent_transport(
                transport_residuals, stream_slices, policies, coefficients
            )
        elif policy_grouped:
            latent_base, combined = policy_grouped_transport(
                transport_residuals, stream_slices, policy_group_coefficients
            )
        elif policy_ordered:
            latent_base, combined = policy_ordered_transport(
                transport_residuals, stream_slices, policy_ordered_specs
            )
        elif readout_precision == "fp32" and counterfactual_mode == "grouped":
            latent_base, combined = grouped_clean_latent_transport(
                transport_residuals, policies, coefficients
            )
        elif isolated_context:
            latent_base = transport_residuals[:, 0]
            combined = isolated_policy_transport(
                transport_residuals, effective_policies, coefficients, aggregation
            )
        elif hybrid_context:
            latent_base = transport_residuals[:, 0]
            combined = hybrid_policy_transport(
                transport_residuals, policies, coefficients, aggregation
            )
        elif joint_isolated_context:
            latent_base, combined = joint_isolated_policy_transport(
                transport_residuals, policies, coefficients, aggregation
            )
        else:
            latent_base = transport_residuals[:, 0]
            combined = (
                policy_transport(transport_residuals, policies, coefficients, aggregation)
                if counterfactual_mode == "leave-one-out"
                else grouped_policy_transport(transport_residuals, policies, coefficients)
            )

        if composite_readout:
            logits = readout(model, latent_base, alpha * combined, norm_mode)
        else:
            assert control_mask is not None
            control_mask = torch.cat(
                [control_mask, torch.ones_like(control_mask[:, :1])], dim=1
            )
            with transport_last_residual(
                model, layer, alpha * combined, transport_site, norm_mode
            ):
                control_out = model(
                    input_ids=token.unsqueeze(1),
                    attention_mask=control_mask,
                    position_ids=next_position_ids(control_mask),
                    past_key_values=control_past,
                    use_cache=True,
                )
            logits = control_out.logits[:, -1, :]
            control_past = control_out.past_key_values
    return [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in generated]


def generate_one(
    model, tokenizer, row: dict, policies: list[str], layer: int, alpha: float,
    coefficients: dict[str, float], aggregation: str, max_new_tokens: int,
    counterfactual_mode: str = "leave-one-out", transport_site: str = "block",
    norm_mode: str = "none", temperature: float = 0.0, top_p: float = 1.0,
    seed: int = 42, readout_precision: str = "native",
    direction_refresh: str = "token", direction_context: str = "full",
    attention_control: str = "none",
) -> str:
    """Compatibility wrapper used by small diagnostics."""
    return generate_batch(
        model, tokenizer, [row], [policies], layer, alpha,
        coefficients, aggregation, counterfactual_mode, transport_site, norm_mode,
        max_new_tokens, temperature, top_p, seed, readout_precision,
        direction_refresh, direction_context, attention_control,
    )[0]


@torch.inference_mode()
def generate_direct_batch(
    model,
    tokenizer,
    rows: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> list[str]:
    """Generate from the untouched full prompt on the same HF/SDPA backend.

    This path exists for a measured same-backend efficiency baseline.  It does
    not construct a counterfactual stream or apply an intervention, and it uses
    the identical stable per-sample sampler as latent decoding.
    """
    if not rows:
        return []
    prompts = [render_latent_prompt(tokenizer, row, set()) for row in rows]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    outputs = model(
        **inputs,
        position_ids=full_position_ids(inputs["attention_mask"]),
        use_cache=True,
    )
    logits = outputs.logits[:, -1, :]
    past = outputs.past_key_values
    attention_mask = inputs["attention_mask"]
    generated: list[list[int]] = [[] for _ in rows]
    finished = torch.zeros(len(rows), dtype=torch.bool, device=model.device)
    generators = []
    for row in rows:
        stable_id = int(
            hashlib.sha256(str(row["sample_id"]).encode()).hexdigest()[:8], 16
        )
        generator = torch.Generator(device=model.device)
        generator.manual_seed(seed + stable_id)
        generators.append(generator)
    eos_ids = tokenizer.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]
    fallback_eos = eos_ids[0]
    for _ in range(max_new_tokens):
        token = sample_tokens_batch(logits.float(), temperature, top_p, generators)
        for sample_index, token_id in enumerate(token.tolist()):
            if finished[sample_index]:
                continue
            if token_id in eos_ids:
                finished[sample_index] = True
            else:
                generated[sample_index].append(token_id)
        if bool(finished.all()):
            break
        token = torch.where(finished, torch.full_like(token, fallback_eos), token)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1
        )
        outputs = model(
            input_ids=token.unsqueeze(1),
            attention_mask=attention_mask,
            position_ids=next_position_ids(attention_mask),
            past_key_values=past,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]
        past = outputs.past_key_values
    return [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in generated]


@torch.inference_mode()
def generate_direct_fp32_batch(
    model,
    tokenizer,
    rows: list[dict],
    layer: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> list[str]:
    """Untouched one-stream generation through the composite FP32 readout.

    This is the numerical-path control for final-norm FP32 steering: it uses no
    LOO stream and adds no activation delta, but obtains logits from the same
    frozen normalized activation and cached FP32 LM-head view.
    """
    if not rows:
        return []
    prompts = [render_latent_prompt(tokenizer, row, set()) for row in rows]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    backbone = model.model if hasattr(model, "model") else model
    capture: list[torch.Tensor] = []
    with capture_last_residual(model, layer, capture, "final-norm"):
        outputs = backbone(
            **inputs,
            position_ids=full_position_ids(inputs["attention_mask"]),
            use_cache=True,
        )
    base = capture.pop()
    logits = fp32_latent_readout(model, base, torch.zeros_like(base), "none")
    past = outputs.past_key_values
    attention_mask = inputs["attention_mask"]
    generated: list[list[int]] = [[] for _ in rows]
    finished = torch.zeros(len(rows), dtype=torch.bool, device=model.device)
    generators = []
    for row in rows:
        stable_id = int(hashlib.sha256(str(row["sample_id"]).encode()).hexdigest()[:8], 16)
        generator = torch.Generator(device=model.device)
        generator.manual_seed(seed + stable_id)
        generators.append(generator)
    eos_ids = tokenizer.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]
    fallback_eos = eos_ids[0]
    for _ in range(max_new_tokens):
        token = sample_tokens_batch(logits.float(), temperature, top_p, generators)
        for sample_index, token_id in enumerate(token.tolist()):
            if finished[sample_index]:
                continue
            if token_id in eos_ids:
                finished[sample_index] = True
            else:
                generated[sample_index].append(token_id)
        if bool(finished.all()):
            break
        token = torch.where(finished, torch.full_like(token, fallback_eos), token)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1
        )
        capture = []
        with capture_last_residual(model, layer, capture, "final-norm"):
            outputs = backbone(
                input_ids=token.unsqueeze(1),
                attention_mask=attention_mask,
                position_ids=next_position_ids(attention_mask),
                past_key_values=past,
                use_cache=True,
            )
        past = outputs.past_key_values
        base = capture.pop()
        logits = fp32_latent_readout(model, base, torch.zeros_like(base), "none")
    return [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in generated]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--policy-source", choices=["gold", "latent"], default="gold")
    parser.add_argument("--split", choices=["all", "calibration", "heldout"], default="heldout")
    parser.add_argument("--memory-setting", choices=["all", "single", "multi"], default="all")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ignore-coefficient", type=float, default=-1.0)
    parser.add_argument("--support-coefficient", type=float, default=0.0)
    parser.add_argument("--dominate-coefficient", type=float, default=1.0)
    parser.add_argument(
        "--ordinal-coefficients",
        help="Comma-separated coefficients for predicted/gold levels 1..5; requires leave-one-out mode.",
    )
    parser.add_argument(
        "--numeric-prediction-coefficients",
        action="store_true",
        help="Read a per-memory steering_coefficient from latent predictions; requires leave-one-out mode.",
    )
    parser.add_argument(
        "--aggregation", choices=["sum", "signed-mean", "gram"], default="gram"
    )
    parser.add_argument("--counterfactual-mode", choices=["leave-one-out", "grouped", "policy-grouped", "policy-ordered"],
                        default="leave-one-out")
    parser.add_argument("--transport-site", choices=["block", "final-norm"], default="block")
    parser.add_argument(
        "--norm-mode", choices=["none", "radial", "tangent"], default="none",
        help="optionally project the transported token activation back to its original norm",
    )
    parser.add_argument(
        "--readout-precision", choices=["native", "probe-native", "mixed", "fp32"], default="native",
        help="compose final activations and apply the frozen LM head in native or float32 precision",
    )
    parser.add_argument(
        "--direction-refresh", choices=["token", "prefill"], default="token",
        help="recompute counterfactual directions per token or freeze the prefill direction for single-stream decoding",
    )
    parser.add_argument(
        "--direction-context", choices=["full", "isolated", "hybrid", "joint-isolated"], default="full",
        help="measure directions in full context, isolated contexts, a transfer gate, or a jointly cleaned isolated composition",
    )
    parser.add_argument(
        "--attention-control", choices=["none", "steer", "extract", "both"], default="none",
        help=(
            "use selected-layer attention to rescale isolated directions, to gate "
            "each token's steering strength, or both"
        ),
    )
    parser.add_argument("--attention-backend", choices=["auto", "eager", "sdpa"], default="auto",
                        help="Explicit backend for matched controls; auto preserves legacy selection")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="same-backend untouched full-context generation for efficiency measurement",
    )
    parser.add_argument(
        "--direct-fp32-only",
        action="store_true",
        help="one-stream no-intervention control using the composite FP32 final-norm readout",
    )
    args = parser.parse_args()
    if args.direct_only and args.direct_fp32_only:
        parser.error("--direct-only and --direct-fp32-only are mutually exclusive")
    if args.direction_refresh == "prefill" and args.readout_precision == "native":
        parser.error("--direction-refresh prefill requires a composite final-norm readout")
    if args.attention_control != "none":
        if args.direction_context != "isolated":
            parser.error("--attention-control requires --direction-context isolated")
        if args.counterfactual_mode != "leave-one-out":
            parser.error("--attention-control requires --counterfactual-mode leave-one-out")
        if args.direction_refresh != "token":
            parser.error("--attention-control currently requires token-level direction refresh")

    # An append-only output must have exactly one writer.  This also lets an
    # independently scheduled backbone job finish while a sequential fallback
    # waits safely; after acquiring the lock the fallback re-reads completed
    # IDs and exits without duplicating rows.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_lock = Path(str(output_path) + ".lock").open("a")
    fcntl.flock(output_lock.fileno(), fcntl.LOCK_EX)

    rows = load_jsonl(args.input)
    prediction_rows = load_jsonl(args.predictions)
    predictions = {item["uid"]: item for item in prediction_rows}
    if args.split != "all":
        sample_ids = {item["sample_id"] for item in prediction_rows if item["split"] == args.split}
        rows = [row for row in rows if row["sample_id"] in sample_ids]
    if args.memory_setting == "single":
        rows = [row for row in rows if len(row["memories"]) == 1]
    elif args.memory_setting == "multi":
        rows = [row for row in rows if len(row["memories"]) > 1]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    # Limit defines a stable evaluation subset, not the number of new rows to
    # append on each retry.  Apply it before removing completed IDs so a resumed
    # timing job still contains exactly the same first N examples.
    if args.limit is not None:
        rows = rows[: args.limit]
    done = load_done_uids(args.output)
    rows = [row for row in rows if row["sample_id"] not in done]

    coefficients = {
        "ignore": args.ignore_coefficient,
        "support": args.support_coefficient,
        "dominate": args.dominate_coefficient,
    }
    ordinal = None
    if args.ordinal_coefficients:
        ordinal = [float(value) for value in args.ordinal_coefficients.split(",")]
        if len(ordinal) != 5:
            parser.error("--ordinal-coefficients requires exactly five values")
        if args.counterfactual_mode not in {"leave-one-out", "policy-grouped", "policy-ordered"}:
            parser.error(
                "ordinal coefficients require --counterfactual-mode "
                "leave-one-out, policy-grouped, or policy-ordered"
            )
        coefficients.update({f"level{index + 1}": value for index, value in enumerate(ordinal)})
    if args.numeric_prediction_coefficients:
        if args.policy_source != "latent":
            parser.error("numeric prediction coefficients require --policy-source latent")
        if args.counterfactual_mode not in {"leave-one-out", "policy-grouped", "policy-ordered"}:
            parser.error(
                "numeric prediction coefficients require --counterfactual-mode "
                "leave-one-out, policy-grouped, or policy-ordered"
            )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if not rows:
        print(
            f"[resume] no pending rows for {args.output}; "
            f"{len(done)} completed rows already present",
            flush=True,
        )
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation=("eager" if args.attention_control != "none" else
                             ("sdpa" if args.attention_backend == "auto" else args.attention_backend)),
    ).eval()

    # Same-memory-count grouping makes the flattened counterfactual batch rectangular.
    rows.sort(key=lambda row: len(row["memories"]))
    # Keep batch offsets globally unique when an append-only generation resumes.
    completed = len(done)
    cursor = 0
    while cursor < len(rows):
        memory_count = len(rows[cursor]["memories"])
        end = cursor
        while end < len(rows) and len(rows[end]["memories"]) == memory_count:
            end += 1
        group = rows[cursor:end]
        for start in range(0, len(group), args.batch_size):
            current = group[start:start + args.batch_size]
            policy_batch = []
            for row in current:
                sample_policies = []
                for memory in row["memories"]:
                    uid = f"{row['sample_id']}::{memory['memory_id']}"
                    if args.numeric_prediction_coefficients:
                        policy = float(predictions[uid]["steering_coefficient"])
                    elif ordinal is not None:
                        level = (
                            int(memory["gold_score"])
                            if args.policy_source == "gold"
                            else int(predictions[uid]["predicted_level"])
                        )
                        policy = f"level{level}"
                    else:
                        policy = memory["gold_policy"] if args.policy_source == "gold" else predictions[uid]["predicted_policy"]
                    sample_policies.append(policy)
                policy_batch.append(sample_policies)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            batch_start = time.perf_counter()
            if args.direct_only:
                texts = generate_direct_batch(
                    model, tokenizer, current, args.max_new_tokens,
                    args.temperature, args.top_p, args.seed,
                )
            elif args.direct_fp32_only:
                texts = generate_direct_fp32_batch(
                    model, tokenizer, current, args.layer, args.max_new_tokens,
                    args.temperature, args.top_p, args.seed,
                )
            else:
                texts = generate_batch(
                    model, tokenizer, current, policy_batch, args.layer, args.alpha,
                    coefficients, args.aggregation, args.counterfactual_mode,
                    args.transport_site, args.norm_mode, args.max_new_tokens,
                    args.temperature, args.top_p, args.seed, args.readout_precision,
                    args.direction_refresh, args.direction_context,
                    args.attention_control,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_wall_seconds = time.perf_counter() - batch_start
            peak_allocated_bytes = (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            )
            peak_reserved_bytes = (
                torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
            )
            output_rows = []
            for row, sample_policies, text in zip(current, policy_batch, texts):
                output_rows.append({
                    "uid": row["sample_id"], "sample_id": row["sample_id"],
                    "generated_text": text,
                    "method": (
                        "direct_same_backend" if args.direct_only else
                        "direct_fp32_same_backend" if args.direct_fp32_only
                        else (
                            "prefill_counterfactual_residual_transport"
                            if args.direction_refresh == "prefill"
                            else (
                                (
                                    "policy_grouped_dynamic_residual_transport"
                                    if args.counterfactual_mode == "policy-grouped"
                                    else "policy_ordered_dynamic_residual_transport"
                                )
                                if args.counterfactual_mode
                                in {"policy-grouped", "policy-ordered"}
                                else "dynamic_counterfactual_residual_transport"
                            )
                        )
                    ),
                    "policy_source": "none" if (args.direct_only or args.direct_fp32_only) else args.policy_source,
                    "policies": [] if (args.direct_only or args.direct_fp32_only) else sample_policies,
                    "transport_layer": None if (args.direct_only or args.direct_fp32_only) else args.layer,
                    "alpha": 0.0 if (args.direct_only or args.direct_fp32_only) else args.alpha,
                    "coefficients": {} if (args.direct_only or args.direct_fp32_only) else coefficients,
                    "aggregation": None if (args.direct_only or args.direct_fp32_only) else args.aggregation,
                    "counterfactual_mode": None if (args.direct_only or args.direct_fp32_only) else args.counterfactual_mode,
                    "transport_site": "final-norm" if args.direct_fp32_only else (None if args.direct_only else args.transport_site),
                    "norm_mode": "none" if (args.direct_only or args.direct_fp32_only) else args.norm_mode,
                    "temperature": args.temperature, "top_p": args.top_p,
                    "seed": args.seed,
                    "readout_precision": "fp32" if args.direct_fp32_only else ("native" if args.direct_only else args.readout_precision),
                    "direction_refresh": None if (args.direct_only or args.direct_fp32_only) else args.direction_refresh,
                    "direction_context": None if (args.direct_only or args.direct_fp32_only) else args.direction_context,
                    "attention_control": "none" if (args.direct_only or args.direct_fp32_only) else args.attention_control,
                    "generated_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                    "batch_wall_seconds": batch_wall_seconds,
                    "peak_allocated_bytes": peak_allocated_bytes,
                    "peak_reserved_bytes": peak_reserved_bytes,
                    "generation_batch_size": len(current),
                    "generation_batch_start": completed,
                    "counterfactual_streams": (
                        1
                        if (args.direct_only or args.direct_fp32_only)
                        else (
                            1 + 2 * len(sample_policies)
                            if args.direction_context in {"isolated", "joint-isolated"}
                            else 1 + 3 * len(sample_policies)
                            if args.direction_context == "hybrid"
                            else counterfactual_stream_count(
                                sample_policies, coefficients, args.counterfactual_mode
                            )
                        )
                    ),
                    "decode_streams": (
                        1
                        if args.direction_refresh == "prefill"
                        else (
                            1
                            if (args.direct_only or args.direct_fp32_only)
                            else (
                                1 + 2 * len(sample_policies)
                                if args.direction_context in {"isolated", "joint-isolated"}
                                else 1 + 3 * len(sample_policies)
                                if args.direction_context == "hybrid"
                                else counterfactual_stream_count(
                                    sample_policies, coefficients, args.counterfactual_mode
                                )
                            )
                        )
                    ),
                })
            append_jsonl(args.output, output_rows)
            completed += len(current)
            print(f"[{completed}/{len(rows)}] memories/sample={memory_count}", flush=True)
        cursor = end


if __name__ == "__main__":
    main()
