"""Per-memory counterfactual steering without parameter updates.

MCCS keeps one full-context stream and one leave-one-memory-out stream per
memory.  At every decoding step, memory i's local causal contribution is
approximated by

    delta_i = logits(full context) - logits(context without memory_i).

The full-context logits are then edited independently for every memory:

    Ignore   -> subtract delta_i
    Support  -> leave delta_i unchanged
    Dominate -> add a scaled delta_i

All streams consume the same generated prefix, so their counterfactual logits
remain aligned token by token.  The model is frozen and no auxiliary model is
trained.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .io import append_jsonl, load_done_uids, load_jsonl
from .prepare_generation import (
    RPEVAL_SYSTEM_TEMPLATE,
    RPEVAL_SYSTEM_TEMPLATE_ZH,
    STRATMEM_SYSTEM_TEMPLATE,
    ablate_rpval_context,
    format_memories_for_prompt,
)
from .prompts import build_messages


POLICY_COEFFICIENT = {"ignore": -1.0, "support": 0.0, "dominate": 1.0}


def _load_frozen_rpeval_vanilla_template() -> str:
    """Read, without rewriting, RPEval's released Vanilla prompt string."""
    if os.environ.get('RPEVAL_PROMPT_FILE'):
        return Path(os.environ['RPEVAL_PROMPT_FILE']).read_text(encoding='utf-8')
    path = Path(__file__).resolve().parent.parent / "RPEval" / "prompts" / "prompts.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "Personalized_responser_template"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise RuntimeError(f"Frozen RPEval Vanilla template not found in {path}")


RPEVAL_OFFICIAL_VANILLA_TEMPLATE = _load_frozen_rpeval_vanilla_template()


def _is_zh(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def build_system_prompt(
    row: dict, removed_memory_ids: set[str] | str | None = None
) -> str:
    """Render a prompt with an arbitrary subset of memories removed."""
    if removed_memory_ids is None:
        removed: set[str] = set()
    elif isinstance(removed_memory_ids, str):
        removed = {removed_memory_ids}
    else:
        removed = set(removed_memory_ids)
    source = row["source"]
    query = row["query"]
    memories = row["memories"]
    context = row.get("context", "")
    metadata = row.get("metadata", {})

    if source == "rpval":
        template = RPEVAL_SYSTEM_TEMPLATE_ZH if _is_zh(query) else RPEVAL_SYSTEM_TEMPLATE
        if removed:
            policies = {memory["memory_id"]: "support" for memory in memories}
            policies.update({memory_id: "ignore" for memory_id in removed})
            context = ablate_rpval_context(row, policies)
        return template.format(context=context[:3000])

    if source == "stratmem":
        ablation = None
        if removed:
            ablation = {memory_id: "ignore" for memory_id in removed}
        memory_list = format_memories_for_prompt(memories, ablation_policies=ablation)
        return STRATMEM_SYSTEM_TEMPLATE.format(
            persona_name=metadata.get("virtual_person", "the assistant"),
            # StratMem's protocol supplies the complete persona and history.
            # Truncating this field changes the official model input.
            persona_info=context,
            memory_list=memory_list,
        )

    if source == "steem":
        # The released SteeM generator renders ``filtered_context`` verbatim as
        # the system message.  The unified representation stores that one
        # removable history in memory_text, so retain it without generic
        # Context/Memories wrappers.  Removing the memory yields no system
        # message, which is the exact context-free counterfactual.
        kept = [
            memory["memory_text"]
            for memory in memories
            if memory["memory_id"] not in removed
        ]
        return "\n".join(kept)

    kept = [m for m in memories if m["memory_id"] not in removed]
    memory_list = format_memories_for_prompt(kept)
    return f"Context:\n{context[:2000]}\n\nMemories:\n{memory_list}"


def render_counterfactual_prompts(tokenizer, row: dict) -> list[str]:
    """Return [full, without-memory-0, ..., without-memory-n]."""
    removed_ids: list[str | None] = [None]
    removed_ids.extend(memory["memory_id"] for memory in row["memories"])
    prompts = []
    for removed_id in removed_ids:
        messages = build_messages(build_system_prompt(row, removed_id), row["query"])
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompts.append(prompt)
    return prompts


def render_official_rpeval_prompt(tokenizer, row: dict, removed_memory_ids: set[str]) -> str:
    """Render the exact released RPEval Vanilla prompt with selected personas removed."""
    kept = [
        memory["memory_text"]
        for memory in row["memories"]
        if memory["memory_id"] not in removed_memory_ids
    ]
    persona = kept[0] if len(row["memories"]) == 1 and kept else ("" if len(row["memories"]) == 1 else kept)
    user_prompt = RPEVAL_OFFICIAL_VANILLA_TEMPLATE.format(
        persona=persona,
        question=row["query"],
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def render_benchpres_prompt(tokenizer, row: dict, removed_memory_ids: set[str]) -> str:
    """Render BenchPreS's released user prompt with exact preference deletion.

    The dataset stores the complete inference prompt as one user message.  Its
    five evaluated preferences occur as standalone lines inside the profile;
    leave-one-out removes only the selected line and leaves every other byte of
    benchmark content unchanged.
    """
    prompt = str(row["context"])
    by_id = {str(memory["memory_id"]): str(memory["memory_text"]) for memory in row["memories"]}
    for memory_id in removed_memory_ids:
        preference = by_id[memory_id]
        line = preference + "\n"
        if prompt.count(line) != 1:
            raise ValueError(
                f"BenchPreS preference must occur exactly once as a line: {memory_id}"
            )
        prompt = prompt.replace(line, "", 1)
    # Do not strip the released prompt.  All public BenchPreS rows end in a
    # newline, and preserving it makes the full stream byte-for-byte faithful
    # before chat-template wrapping.  LOO therefore changes exactly one
    # preference line and nothing else.
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def render_latent_prompt(tokenizer, row: dict, removed_memory_ids: set[str]) -> str:
    """Render the frozen benchmark prompt used by latent counterfactual streams."""
    if row.get("source") == "rpval":
        return render_official_rpeval_prompt(tokenizer, row, removed_memory_ids)
    if row.get("source") == "benchpres":
        return render_benchpres_prompt(tokenizer, row, removed_memory_ids)
    messages = build_messages(build_system_prompt(row, removed_memory_ids), row["query"])
    if row.get("source") == "steem":
        # Match the released SteeM inference code, which does not override the
        # model's chat-template kwargs (notably Qwen3's default reasoning mode).
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def render_grouped_prompts(
    tokenizer,
    row: dict,
    policies: list[str],
    prompt_style: str = "adaptive",
    support_strength: float = 1.0,
    dominate_strength: float = 1.0,
) -> list[str]:
    """Return [all non-Ignore memories, then one stream per removed Dominate]."""
    ignore_ids = {
        memory["memory_id"]
        for memory, policy in zip(row["memories"], policies)
        if policy == "ignore"
    }
    adjusted_ids = [
        memory["memory_id"]
        for memory, policy in zip(row["memories"], policies)
        if (policy == "support" and support_strength != 1.0)
        or (policy == "dominate" and dominate_strength != 0.0)
    ]
    removal_sets = [ignore_ids]
    removal_sets.extend(ignore_ids | {memory_id} for memory_id in adjusted_ids)
    prompts = []
    for removed in removal_sets:
        if prompt_style == "rpeval_official":
            if row.get("source") != "rpval":
                raise ValueError("rpeval_official prompt style only supports RPEval rows")
            prompts.append(render_official_rpeval_prompt(tokenizer, row, removed))
            continue
        messages = build_messages(build_system_prompt(row, removed), row["query"])
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompts.append(prompt)
    return prompts


def load_policy_predictions(path: str | None) -> dict[str, dict[str, str]]:
    """Load either flat per-memory rows or nested per-sample predictions."""
    if path is None:
        return {}
    result: dict[str, dict[str, str]] = {}
    for row in load_jsonl(path):
        sid = str(row["sample_id"])
        result.setdefault(sid, {})
        if "memory_id" in row and "predicted_policy" in row:
            result[sid][str(row["memory_id"])] = row["predicted_policy"]
        for pred in row.get("predicted_policies", []):
            result[sid][str(pred["memory_id"])] = pred["predicted_policy"]
    return result


def policies_for_row(
    row: dict,
    policy_source: str,
    predictions: dict[str, dict[str, str]],
) -> list[str]:
    if policy_source == "gold":
        return [memory["gold_policy"] for memory in row["memories"]]
    if policy_source == "all_support":
        return ["support"] * len(row["memories"])
    by_memory = predictions.get(str(row["sample_id"]), {})
    missing = [m["memory_id"] for m in row["memories"] if m["memory_id"] not in by_memory]
    if missing:
        raise ValueError(f"{row['sample_id']}: missing predicted policies for {missing}")
    return [by_memory[memory["memory_id"]] for memory in row["memories"]]


def combine_counterfactual_logits(
    logits: torch.Tensor,
    policies: Iterable[str],
    ignore_strength: float,
    dominate_strength: float,
) -> tuple[torch.Tensor, list[float]]:
    """Combine [full, leave-one-out...] logits into one controlled distribution."""
    full = logits[0].float()
    controlled = full.clone()
    coefficients = []
    for index, policy in enumerate(policies, start=1):
        if policy not in POLICY_COEFFICIENT:
            raise ValueError(f"Unknown policy: {policy}")
        if policy == "ignore":
            coefficient = -ignore_strength
        elif policy == "dominate":
            coefficient = dominate_strength
        else:
            coefficient = 0.0
        controlled.add_(coefficient * (full - logits[index].float()))
        coefficients.append(coefficient)
    return controlled, coefficients


def combine_grouped_logits(
    logits: torch.Tensor,
    policies: Iterable[str],
    support_strength: float,
    dominate_strength: float,
) -> tuple[torch.Tensor, list[float]]:
    """Amplify Dominate deltas after Ignore memories were jointly removed."""
    base = logits[0].float()
    controlled = base.clone()
    dominate_stream = 1
    coefficients = []
    for policy in policies:
        if policy == "ignore":
            # Ignore is realized exactly in the prompt, before logit steering.
            coefficients.append(-1.0)
        elif policy == "support":
            coefficient = support_strength - 1.0
            if coefficient != 0.0:
                controlled.add_(coefficient * (base - logits[dominate_stream].float()))
                dominate_stream += 1
            coefficients.append(coefficient)
        elif policy == "dominate":
            if dominate_strength != 0.0:
                controlled.add_(dominate_strength * (base - logits[dominate_stream].float()))
                dominate_stream += 1
            coefficients.append(dominate_strength)
        else:
            raise ValueError(f"Unknown policy: {policy}")
    return controlled, coefficients


def sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax().reshape(1)
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_probs[remove] = 0
        sorted_probs /= sorted_probs.sum()
        sampled_position = torch.multinomial(sorted_probs, 1, generator=generator)
        return sorted_indices[sampled_position]
    return torch.multinomial(probs, 1, generator=generator)


def sample_tokens_batch(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    generators: list[torch.Generator],
) -> torch.Tensor:
    """Vectorized top-p sampling with one deterministic RNG per sample."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_probs.masked_fill_(remove, 0)
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        probs = sorted_probs
    else:
        sorted_indices = torch.arange(
            probs.shape[-1], device=probs.device
        ).expand_as(probs)

    # Inverse-CDF sampling lets the expensive vocabulary operations stay
    # batched while preserving an independent, sample-stable RNG stream.
    draws = torch.stack([
        torch.rand((), device=probs.device, generator=generator)
        for generator in generators
    ])
    cumulative = torch.cumsum(probs, dim=-1)
    positions = torch.searchsorted(
        cumulative.contiguous(), draws.unsqueeze(1), right=False
    ).squeeze(1)
    positions.clamp_max_(probs.shape[-1] - 1)
    return sorted_indices.gather(1, positions.unsqueeze(1)).squeeze(1)


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    row: dict,
    policies: list[str],
    ignore_strength: float,
    dominate_strength: float,
    support_strength: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    counterfactual_mode: str,
    prompt_style: str = "adaptive",
) -> tuple[str, int, list[float]]:
    if counterfactual_mode == "leave_one_out":
        prompts = render_counterfactual_prompts(tokenizer, row)
    elif counterfactual_mode == "grouped":
        prompts = render_grouped_prompts(
            tokenizer, row, policies, prompt_style, support_strength, dominate_strength
        )
    else:
        raise ValueError(f"Unknown counterfactual mode: {counterfactual_mode}")
    batch = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    # Manual cached decoding must reproduce ``generate``'s left-padding
    # position handling.  Without per-stream position IDs, prompt-length
    # differences leak a positional shift into the leave-one-out delta.
    position_ids = batch["attention_mask"].long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(batch["attention_mask"] == 0, 0)
    outputs = model(
        **batch, position_ids=position_ids, use_cache=True, logits_to_keep=1
    )
    next_logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values
    attention_mask = batch["attention_mask"]

    stable_id = int(hashlib.sha256(str(row["sample_id"]).encode()).hexdigest()[:8], 16)
    generator = torch.Generator(device=model.device)
    generator.manual_seed(seed + stable_id)
    generated: list[int] = []
    coefficients: list[float] = []
    eos_ids = tokenizer.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]

    for _ in range(max_new_tokens):
        if counterfactual_mode == "leave_one_out":
            controlled, coefficients = combine_counterfactual_logits(
                next_logits, policies, ignore_strength, dominate_strength
            )
        else:
            controlled, coefficients = combine_grouped_logits(
                next_logits, policies, support_strength, dominate_strength
            )
        token = sample_token(controlled, temperature, top_p, generator)
        token_id = int(token.item())
        if token_id in eos_ids:
            break
        generated.append(token_id)

        repeated = token.reshape(1, 1).expand(len(prompts), 1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(repeated, dtype=attention_mask.dtype)], dim=1
        )
        decode_position_ids = attention_mask.long().sum(dim=-1, keepdim=True) - 1
        outputs = model(
            input_ids=repeated,
            attention_mask=attention_mask,
            position_ids=decode_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        next_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

    return tokenizer.decode(generated, skip_special_tokens=True).strip(), len(generated), coefficients


@torch.inference_mode()
def generate_many(
    model,
    tokenizer,
    rows: list[dict],
    row_policies: list[list[str]],
    ignore_strength: float,
    dominate_strength: float,
    support_strength: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    counterfactual_mode: str,
    prompt_style: str = "adaptive",
) -> list[tuple[str, int, list[float]]]:
    """Decode independent samples together as one flattened stream batch."""
    all_prompts: list[str] = []
    stream_slices: list[slice] = []
    stream_to_sample: list[int] = []
    for sample_index, (row, policies) in enumerate(zip(rows, row_policies)):
        if counterfactual_mode == "leave_one_out":
            prompts = render_counterfactual_prompts(tokenizer, row)
        elif counterfactual_mode == "grouped":
            prompts = render_grouped_prompts(
                tokenizer, row, policies, prompt_style, support_strength, dominate_strength
            )
        else:
            raise ValueError(f"Unknown counterfactual mode: {counterfactual_mode}")
        start = len(all_prompts)
        all_prompts.extend(prompts)
        stream_slices.append(slice(start, len(all_prompts)))
        stream_to_sample.extend([sample_index] * len(prompts))

    batch = tokenizer(all_prompts, return_tensors="pt", padding=True).to(model.device)
    position_ids = batch["attention_mask"].long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(batch["attention_mask"] == 0, 0)
    outputs = model(
        **batch, position_ids=position_ids, use_cache=True, logits_to_keep=1
    )
    next_logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values
    attention_mask = batch["attention_mask"]

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
    generated: list[list[int]] = [[] for _ in rows]
    finished = [False] * len(rows)
    coefficients: list[list[float]] = [[] for _ in rows]
    stream_to_sample_tensor = torch.tensor(
        stream_to_sample, device=model.device, dtype=torch.long
    )

    for _ in range(max_new_tokens):
        controlled_batch = []
        for sample_index, (policies, stream_slice) in enumerate(
            zip(row_policies, stream_slices)
        ):
            logits = next_logits[stream_slice]
            if counterfactual_mode == "leave_one_out":
                controlled, coeff = combine_counterfactual_logits(
                    logits, policies, ignore_strength, dominate_strength
                )
            else:
                controlled, coeff = combine_grouped_logits(
                    logits, policies, support_strength, dominate_strength
                )
            coefficients[sample_index] = coeff
            controlled_batch.append(controlled)

        sampled = sample_tokens_batch(
            torch.stack(controlled_batch), temperature, top_p, generators
        )
        sample_tokens = []
        for sample_index, sampled_token in enumerate(sampled):
            if finished[sample_index]:
                token_id = fallback_eos
            else:
                token_id = int(sampled_token.item())
                if token_id in eos_ids:
                    finished[sample_index] = True
                else:
                    generated[sample_index].append(token_id)
            sample_tokens.append(token_id)

        if all(finished):
            break
        sample_token_tensor = torch.tensor(
            sample_tokens, device=model.device, dtype=torch.long
        )
        repeated = sample_token_tensor[stream_to_sample_tensor].unsqueeze(1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(repeated, dtype=attention_mask.dtype)], dim=1
        )
        decode_position_ids = attention_mask.long().sum(dim=-1, keepdim=True) - 1
        outputs = model(
            input_ids=repeated,
            attention_mask=attention_mask,
            position_ids=decode_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        next_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

    return [
        (tokenizer.decode(tokens, skip_special_tokens=True).strip(), len(tokens), coeff)
        for tokens, coeff in zip(generated, coefficients)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Training-free per-memory MCCS generation")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy-source", choices=["gold", "predicted", "all_support"], default="gold")
    parser.add_argument("--policy-jsonl")
    parser.add_argument("--ignore-strength", type=float, default=1.0)
    parser.add_argument("--dominate-strength", type=float, default=1.0)
    parser.add_argument(
        "--support-strength",
        type=float,
        default=1.0,
        help="Retained fraction of a Support memory's counterfactual contribution",
    )
    parser.add_argument("--counterfactual-mode", choices=["leave_one_out", "grouped"],
                        default="grouped")
    parser.add_argument(
        "--prompt-style",
        choices=["adaptive", "rpeval_official"],
        default="adaptive",
        help="Use the existing adaptive prompt or RPEval's frozen Vanilla template",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--selection-jsonl",
        help="optional per-memory prediction file whose sample IDs define a split",
    )
    parser.add_argument(
        "--selection-split", choices=["calibration", "heldout"],
        help="retain only sample IDs with this split in --selection-jsonl",
    )
    parser.add_argument("--memory-setting", choices=["all", "single", "multi"], default="all")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.policy_source == "predicted" and not args.policy_jsonl:
        parser.error("--policy-source predicted requires --policy-jsonl")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    rows = load_jsonl(args.input_jsonl)
    if bool(args.selection_jsonl) != bool(args.selection_split):
        parser.error("--selection-jsonl and --selection-split must be provided together")
    if args.selection_jsonl:
        selected = {
            str(item["sample_id"])
            for item in load_jsonl(args.selection_jsonl)
            if item.get("split") == args.selection_split
        }
        rows = [row for row in rows if str(row["sample_id"]) in selected]
    if args.memory_setting == "single":
        rows = [row for row in rows if len(row["memories"]) == 1]
    elif args.memory_setting == "multi":
        rows = [row for row in rows if len(row["memories"]) > 1]
    if args.offset:
        rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]
    done = set() if args.no_resume else load_done_uids(args.output_jsonl)
    rows = [row for row in rows if str(row["sample_id"]) not in done]
    predictions = load_policy_predictions(args.policy_jsonl)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    for start in range(0, len(rows), args.batch_size):
        row_batch = rows[start : start + args.batch_size]
        policy_batch = [
            policies_for_row(row, args.policy_source, predictions) for row in row_batch
        ]
        results = generate_many(
            model=model,
            tokenizer=tokenizer,
            rows=row_batch,
            row_policies=policy_batch,
            ignore_strength=args.ignore_strength,
            dominate_strength=args.dominate_strength,
            support_strength=args.support_strength,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            counterfactual_mode=args.counterfactual_mode,
            prompt_style=args.prompt_style,
        )
        for row, policies, (text, output_tokens, coefficients) in zip(
            row_batch, policy_batch, results
        ):
            append_jsonl(args.output_jsonl, [{
                "uid": row["sample_id"],
                "sample_id": row["sample_id"],
                "source": row["source"],
                "query": row["query"],
                "memories": row["memories"],
                "metadata": row.get("metadata", {}),
                "generated_text": text,
                "output_tokens": output_tokens,
                "policy_source": args.policy_source,
                "policies": policies,
                "mccs_coefficients": coefficients,
                "ignore_strength": args.ignore_strength,
                "dominate_strength": args.dominate_strength,
                "support_strength": args.support_strength,
                "seed": args.seed,
                "counterfactual_mode": args.counterfactual_mode,
                "method": f"mccs_{args.counterfactual_mode}_logits",
                "prompt_style": args.prompt_style,
                "generation_batch_size": args.batch_size,
            }])
            completed += 1
            print(
                f"[{completed}/{len(rows)}] {row['sample_id']} tokens={output_tokens}",
                flush=True,
            )


if __name__ == "__main__":
    main()
