from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class SteeringSpec:
    vector: torch.Tensor
    alpha: float
    layers: list[int] | None = None
    prompt_only: bool = False  # Default: steer ALL positions (prefill + decode), matching AdaSteer


@dataclass
class AdaptiveSteeringSpec:
    """Adaptive steering: compute alpha per-sample based on projection distance to anchors.

    At the probe_layer, project the current hidden state onto (high_anchor - low_anchor).
    Compute alpha dynamically based on distance to anchors.

    Matches AdaSteer: two probe layers, adaptive alpha computed at each prefill step,
    then applied to all positions including decode.
    """
    vector: torch.Tensor
    alpha_base: float
    low_anchors: torch.Tensor   # (num_layers, D)
    high_anchors: torch.Tensor  # (num_layers, D)
    probe_layers: list[int] | None = None  # which layers to compute adaptive alpha; default [5, 13]
    layers: list[int] | None = None
    prompt_only: bool = False  # steer ALL positions
    scale: float = 100.0       # normalization scale for projection distance


@dataclass
class BatchDynamicSteeringSpec:
    """One layer-wise steering direction per item in the active batch."""
    vectors: torch.Tensor  # (batch, layers, hidden)
    alpha: float
    layers: list[int]
    prompt_only: bool = False
    preserve_norm: bool = True
    token_position: str = "all"  # "all" or only the final token during prefill


def get_decoder_layers(model) -> torch.nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("Could not locate decoder layers for this model architecture")


def _get_model_device(model):
    """Get device from model, handling both transformers and vLLM internal models."""
    if hasattr(model, "device"):
        return model.device
    # vLLM: model may not have .device, try to infer from parameters
    try:
        return next(model.parameters()).device
    except StopIteration:
        # Fallback: try to find a submodule with parameters
        for module in model.modules():
            try:
                return next(module.parameters()).device
            except StopIteration:
                continue
    # Last resort: CUDA if available, else CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_vectors(vectors: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denom = vectors.norm(dim=-1, keepdim=True).clamp_min(eps)
    return vectors / denom


def last_nonpad_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.long().sum(dim=1).sub(1).clamp_min(0)


@contextmanager
def fixed_steering_hooks(model, spec: SteeringSpec) -> Iterator[None]:
    """Fixed-alpha steering hooks.

    When prompt_only=False (default, matching AdaSteer), the steering vector
    is added to ALL token positions in ALL forward passes (both prefill and decode).
    When prompt_only=True, only add during prefill (seq_len > 1).
    """
    layers = get_decoder_layers(model)
    steer_layers = spec.layers if spec.layers is not None else list(range(len(layers)))
    model_device = _get_model_device(model)
    vector = spec.vector.to(model_device)
    if vector.ndim == 2:
        vector_by_layer = vector
    elif vector.ndim == 1:
        vector_by_layer = vector.unsqueeze(0).repeat(len(layers), 1)
    else:
        raise ValueError(f"Expected steering vector shape (layers, hidden) or (hidden,), got {tuple(vector.shape)}")

    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Only skip decode steps if prompt_only is True
            if spec.prompt_only and hidden.shape[1] == 1:
                return output
            delta = spec.alpha * vector_by_layer[layer_idx].to(hidden.device, dtype=hidden.dtype)
            steered = hidden + delta.view(1, 1, -1)
            if isinstance(output, tuple):
                return (steered, *output[1:])
            return steered

        return hook

    try:
        for layer_idx in steer_layers:
            handles.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def batch_dynamic_steering_hooks(model, spec: BatchDynamicSteeringSpec) -> Iterator[None]:
    """Apply sample-specific, layer-specific directions without fitted weights."""
    layers = get_decoder_layers(model)
    vectors = spec.vectors.to(_get_model_device(model))
    if vectors.ndim != 3 or vectors.shape[1] != len(layers):
        raise ValueError(f"Expected (batch,{len(layers)},hidden), got {tuple(vectors.shape)}")
    if spec.token_position not in {"all", "last"}:
        raise ValueError(f"token_position must be 'all' or 'last', got {spec.token_position!r}")
    handles = []

    def make_hook(layer_index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if spec.prompt_only and hidden.shape[1] == 1:
                return output
            if hidden.shape[0] != vectors.shape[0]:
                raise ValueError(f"steering batch {vectors.shape[0]} != hidden batch {hidden.shape[0]}")
            direction = vectors[:, layer_index].to(hidden.device, hidden.dtype).unsqueeze(1)
            # A counterfactual signature is measured at the final prompt token.
            # Applying it to every prompt position destroys its exact residual-space
            # interpretation, so transport only the matching position when requested.
            if spec.token_position == "last" and hidden.shape[1] > 1:
                original = hidden[:, -1:, :]
                transported = original + spec.alpha * direction
                if spec.preserve_norm:
                    original_norm = original.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    new_norm = transported.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    transported = transported * (original_norm / new_norm).to(transported.dtype)
                steered = hidden.clone()
                steered[:, -1:, :] = transported
            else:
                steered = hidden + spec.alpha * direction
                if spec.preserve_norm:
                    original_norm = hidden.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    new_norm = steered.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    steered = steered * (original_norm / new_norm).to(steered.dtype)
            if isinstance(output, tuple):
                return (steered, *output[1:])
            return steered
        return hook

    try:
        for layer_index in spec.layers:
            handles.append(layers[layer_index].register_forward_hook(make_hook(layer_index)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def adaptive_steering_hooks(model, spec: AdaptiveSteeringSpec) -> Iterator[None]:
    """Adaptive steering hooks matching AdaSteer's approach.

    Key differences from fixed steering:
    1. Alpha is computed dynamically at probe layers during prefill
    2. Steering is applied at ALL positions (prefill + decode)
    3. Two probe layers (default [5, 13]) compute separate alpha components,
       matching AdaSteer's dual-probe design
    """
    layers = get_decoder_layers(model)
    steer_layers = spec.layers if spec.layers is not None else list(range(len(layers)))
    model_device = _get_model_device(model)
    vector = spec.vector.to(model_device)
    if vector.ndim == 2:
        vector_by_layer = vector
    elif vector.ndim == 1:
        vector_by_layer = vector.unsqueeze(0).repeat(len(layers), 1)
    else:
        raise ValueError(f"Expected steering vector shape (layers, hidden) or (hidden,), got {tuple(vector.shape)}")

    # Setup anchor directions for each probe layer
    probe_layers = spec.probe_layers or [5, 13]
    anchor_data = {}
    model_device = _get_model_device(model)
    for pl in probe_layers:
        if pl < spec.low_anchors.shape[0]:
            low_a = spec.low_anchors[pl].to(model_device).float()
            high_a = spec.high_anchors[pl].to(model_device).float()
            direction = high_a - low_a
            direction = direction / direction.norm().clamp_min(1e-6)
            anchor_data[pl] = {"low": low_a, "high": high_a, "direction": direction}

    # Alpha will be computed during prefill
    computed_alpha: list[float] = [0.0]
    first_prefill_done: list[bool] = [False]

    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output

            # Skip decode steps only if prompt_only is True
            if spec.prompt_only and hidden.shape[1] == 1:
                return output

            # During prefill, compute adaptive alpha at probe layers
            if layer_idx in anchor_data and hidden.shape[1] > 1:
                ad = anchor_data[layer_idx]
                hs_last = hidden[:, -1, :].float()  # (B, D)

                # Project onto anchor direction
                dis_low = hs_last - ad["low"].unsqueeze(0)
                proj_low = torch.matmul(dis_low, ad["direction"])

                dis_high = hs_last - ad["high"].unsqueeze(0)
                proj_high = torch.matmul(dis_high, ad["direction"])

                # Normalize by distance between anchors
                diff = (proj_low - proj_high).abs().clamp_min(1e-6)
                proj_normalized = proj_low / diff * spec.scale

                # Compute alpha: positive alpha_base pushes toward high anchor
                # Matching AdaSteer: alpha = alpha_base * (normalized_distance - offset)
                if layer_idx == probe_layers[0]:
                    # First probe layer (layer 5 in AdaSteer)
                    alpha = spec.alpha_base * (proj_normalized - spec.scale * 1.4)
                    computed_alpha[0] = alpha.item()
                elif layer_idx == probe_layers[-1]:
                    # Second probe layer (layer 13 in AdaSteer)
                    alpha = -0.06 * (proj_normalized - spec.scale * 0.5)
                    # Clamp like AdaSteer
                    alpha = max(-0.6, min(0.4, alpha.item()))
                    # Add to existing alpha from first probe
                    computed_alpha[0] += alpha

                first_prefill_done[0] = True

            # Apply steering with computed alpha (or 0 if not yet computed)
            delta = computed_alpha[0] * vector_by_layer[layer_idx].to(hidden.device, dtype=hidden.dtype)
            steered = hidden + delta.view(1, 1, -1)
            if isinstance(output, tuple):
                return (steered, *output[1:])
            return steered

        return hook

    try:
        for layer_idx in steer_layers:
            handles.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
        yield
    finally:
        for handle in handles:
            handle.remove()
