"""Convert unified benchmark format to generation-ready format.

Takes unified JSONL and produces JSONL compatible with generate_vllm.py,
with proper system/user field formatting for each benchmark source.

Output format per row:
{
    "uid": str (sample_id),
    "filtered_context": str (system prompt with context + memories),
    "rewritten_query": str (user query),
    "source": str,
    "memories": [...],  # preserved for judge
}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_jsonl, write_jsonl

# --- Prompt templates ---

RPEVAL_SYSTEM_TEMPLATE = """You are a personalized assistant. The user has shared some conversation history that reveals their preferences. Please consider this context when responding, but use good judgment about which preferences are relevant to the current request.

Conversation history:
{context}

Remember: Not all preferences are equally relevant to every request. Use your judgment to determine which preferences should influence your response and how strongly."""

RPEVAL_SYSTEM_TEMPLATE_ZH = """你是一个个性化助手。用户分享了一些对话历史，其中隐含了他们的偏好。请在回答时考虑这些上下文，但要合理判断哪些偏好与当前请求相关。

对话历史：
{context}

请注意：并非所有偏好都与每个请求同等相关。请自行判断哪些偏好应该影响你的回复，以及影响程度。"""

STRATMEM_SYSTEM_TEMPLATE = """You are {persona_name}, a virtual person in a conversation. You have the following traits and background:

{persona_info}

You are now responding to a conversation partner. Below are some memories from your past interactions that may or may not be relevant to the current conversation. Use your judgment about which memories to draw upon and how strongly.

Relevant memories from past conversations:
{memory_list}

Respond naturally as {persona_name}, using only the memories that are genuinely relevant to the current question."""


def format_memories_for_prompt(memories: list[dict], include_policy: bool = False, predicted_policies: dict | None = None, ablation_policies: dict | None = None) -> str:
    """Format memories as a numbered list for the prompt.

    Args:
        memories: List of memory dicts with gold_policy/gold_score
        include_policy: If True, add policy hints from gold_policy
        predicted_policies: If provided, use predicted policies instead of gold (for policy_prompt mode)
        ablation_policies: If provided, REMOVE memories with policy="ignore" from the list
    """
    lines = []
    policy_hints = {
        "ignore": "(This preference is NOT relevant to the current request - do not let it influence your response)",
        "support": "(This preference can enhance your response but should not dominate it)",
        "dominate": "(This preference MUST strongly shape your response - it is essential)",
    }
    idx = 0
    for i, m in enumerate(memories):
        text = m["memory_text"]
        policy = m.get("gold_policy", "")

        # Ablation mode: skip ignore memories entirely
        if ablation_policies and m["memory_id"] in ablation_policies:
            policy = ablation_policies[m["memory_id"]]
            if policy == "ignore":
                continue  # Physical removal from prompt

        if include_policy:
            if predicted_policies and m["memory_id"] in predicted_policies:
                policy = predicted_policies[m["memory_id"]]
            if policy in policy_hints:
                text = f"{text} {policy_hints[policy]}"
        idx += 1
        lines.append(f"{idx}. {text}")
    return "\n".join(lines)


def ablate_rpval_context(row: dict, policies: dict[str, str]) -> str:
    """Remove the implicit dialogue aligned to each ignored RPEval memory.

    RPEval represents a memory as a dialogue history plus an explicit persona
    summary used only for annotation. Appending filtered persona summaries does
    not remove the original memory, so true ablation must operate on the aligned
    dialogue histories retained by ``convert_benchmarks.convert_rpval``.
    """
    memories = row["memories"]
    memory_contexts = row.get("metadata", {}).get("memory_contexts")
    if not isinstance(memory_contexts, list) or len(memory_contexts) != len(memories):
        raise ValueError(
            f"{row['sample_id']}: RPEval ablation requires one metadata.memory_contexts "
            f"entry per memory; regenerate the unified data with convert_rpval"
        )

    kept = [
        context
        for memory, context in zip(memories, memory_contexts)
        if policies.get(memory["memory_id"], "support") != "ignore"
    ]
    return "\n\n".join(kept)


def convert_for_generation(
    input_jsonl: str,
    output_jsonl: str,
    mode: str = "vanilla",  # "vanilla", "oracle_prompt", "policy_prompt"
    policy_jsonl: str | None = None,
) -> None:
    """Convert unified benchmark to generation format.

    Modes:
    - vanilla: No policy hints in prompt (for direct gen and controller settings)
    - oracle_prompt: Gold policy hints in prompt (for oracle-prompt setting)
    - policy_prompt: Predicted policy hints in prompt (for predicted-prompt setting)
        Requires --policy-jsonl with policy inductor output.
    - oracle_ablation: Remove gold-ignore memories from prompt, keep support/dominate
    - probe_ablation: Remove probe-predicted-ignore memories, keep support/dominate
        Requires --policy-jsonl with probe predictions.
    """
    rows = load_jsonl(input_jsonl)

    # Load predicted policies if needed
    predicted_by_sample = {}
    if mode in ("policy_prompt", "probe_ablation") and policy_jsonl:
        from .io import load_jsonl as _load
        for r in _load(policy_jsonl):
            predicted_by_sample[r["sample_id"]] = {
                p["memory_id"]: p["predicted_policy"]
                for p in r.get("predicted_policies", [])
            }

    # Build ablation policy maps: {sample_id: {memory_id: policy}}
    # For oracle_ablation, use gold_policy; for probe_ablation, use probe predictions
    ablation_by_sample: dict[str, dict[str, str]] = {}
    if mode == "oracle_ablation":
        for row in rows:
            ablation_by_sample[row["sample_id"]] = {
                m["memory_id"]: m["gold_policy"]
                for m in row["memories"]
            }
    elif mode == "probe_ablation":
        # probe predictions come as flat per-memory rows (sample_id + memory_id + predicted_policy)
        if policy_jsonl:
            from .io import load_jsonl as _load
            for r in _load(policy_jsonl):
                sid = r["sample_id"]
                if sid not in ablation_by_sample:
                    ablation_by_sample[sid] = {}
                ablation_by_sample[sid][r["memory_id"]] = r["predicted_policy"]

    results = []

    for row in rows:
        source = row["source"]
        query = row["query"]
        context = row.get("context", "")
        memories = row["memories"]

        # Get ablation policies for this sample (if any)
        ablation_policies = ablation_by_sample.get(row["sample_id"])

        if source == "rpval":
            # RPEval: context is implicit_persona (5-turn dialogue)
            lang = "zh" if any('一' <= c <= '鿿' for c in query) else "en"
            template = RPEVAL_SYSTEM_TEMPLATE_ZH if lang == "zh" else RPEVAL_SYSTEM_TEMPLATE

            # For RPEval, the context IS the implicit persona dialogue.
            # Ablation must remove aligned dialogue histories, not append the
            # explicit persona summaries used by the dataset annotators.
            if mode in ("oracle_ablation", "probe_ablation"):
                context = ablate_rpval_context(row, ablation_policies or {})
            # Memories are the explicit preference texts (for reference)
            system_content = template.format(context=context[:3000])

            # In oracle_prompt mode, add policy hints
            if mode == "oracle_prompt":
                memory_hints = format_memories_for_prompt(memories, include_policy=True)
                system_content += f"\n\nPreference guidance:\n{memory_hints}"
            elif mode == "policy_prompt":
                pred = predicted_by_sample.get(row["sample_id"])
                if pred:
                    memory_hints = format_memories_for_prompt(
                        memories, include_policy=True, predicted_policies=pred
                    )
                    system_content += f"\n\nPreference guidance:\n{memory_hints}"
        elif source == "stratmem":
            # StratMem: context has persona info, memories are the memory pool
            metadata = row.get("metadata", {})
            persona_name = metadata.get("virtual_person", "the assistant")
            persona_info = context  # already formatted with summary/traits/style

            include_policy = (mode in ("oracle_prompt", "policy_prompt"))
            pred_policies = None
            if mode == "policy_prompt":
                pred_policies = predicted_by_sample.get(row["sample_id"])
            memory_list = format_memories_for_prompt(
                memories, include_policy=include_policy, predicted_policies=pred_policies,
                ablation_policies=ablation_policies,
            )

            system_content = STRATMEM_SYSTEM_TEMPLATE.format(
                persona_name=persona_name,
                persona_info=persona_info,
                memory_list=memory_list,
            )

        else:
            # Generic fallback
            memory_list = format_memories_for_prompt(
                memories, include_policy=(mode == "oracle_prompt"),
                ablation_policies=ablation_policies,
            )
            system_content = f"Context:\n{context[:2000]}\n\nMemories:\n{memory_list}"

        result = {
            "uid": row["sample_id"],
            "filtered_context": system_content,
            "rewritten_query": query,
            "source": source,
            "memories": memories,  # preserve for judge
            "metadata": row.get("metadata", {}),
        }
        results.append(result)

    write_jsonl(output_jsonl, results)
    print(f"Converted {len(results)} rows ({mode} mode) -> {output_jsonl}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert unified benchmark to generation format")
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--mode", default="vanilla",
                    choices=["vanilla", "oracle_prompt", "policy_prompt",
                             "oracle_ablation", "probe_ablation"])
    ap.add_argument("--policy-jsonl", default=None, help="Policy inductor output (for policy_prompt mode)")

    args = ap.parse_args()
    convert_for_generation(args.input_jsonl, args.output_jsonl, args.mode, args.policy_jsonl)


if __name__ == "__main__":
    main()
