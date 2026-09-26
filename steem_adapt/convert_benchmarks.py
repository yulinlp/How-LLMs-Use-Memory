"""Convert RPEval, StratMem-Bench, BenchPreS to unified per-memory policy format.

Unified schema per entry:
{
    "source": "rpval" | "stratmem" | "benchpres",
    "sample_id": str,
    "query": str,                    # current user query
    "context": str,                  # dialogue history / persona summary (optional)
    "memories": [
        {
            "memory_id": str,
            "memory_text": str,      # the memory/preference text
            "gold_policy": str,      # "ignore" | "support" | "dominate"
            "gold_score": float,     # 0.0 | 0.5 | 1.0
        },
        ...
    ],
    "metadata": dict,                # source-specific fields
}
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

# --- Label mapping ---
POLICY_MAP = {
    # RPEval labels
    "ignore": "ignore", "忽略偏好": "ignore", "A": "ignore",
    "support": "support", "supportive": "support", "支持性偏好": "support", "B": "support",
    "dominate": "dominate", "以偏好为主线": "dominate", "C": "dominate",
    # StratMem labels
    "must": "dominate", "nice": "support", "irr": "ignore",
    # BenchPreS labels
    "apply": "dominate", "suppress": "ignore",
}

SCORE_MAP = {"ignore": 0.0, "support": 0.5, "dominate": 1.0}

PROJ = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ / "data"
DATA_DIR.mkdir(exist_ok=True)


def normalize_policy(label: str) -> str:
    """Map any source label to unified policy name."""
    return POLICY_MAP[label]


def convert_rpval(out_dir: Path, source_root: Path | None = None) -> list[dict]:
    """Convert RPEval implicit_preference data."""
    rpval_dir = (source_root or PROJ / 'RPEval') / 'benchmark_dataset' / 'implicit_preference'
    results = []

    # --- Single preference ---
    with open(rpval_dir / "single_testset.json") as f:
        data = json.load(f)
    for i, d in enumerate(data):
        policy = normalize_policy(d["intent_type"])
        entry = {
            "source": "rpval",
            "sample_id": f"rpval_impl_single_{i:04d}",
            "query": d["question"],
            "context": d["implicit_persona"],  # 5-turn dialogue history
            "memories": [
                {
                    "memory_id": f"rpval_impl_single_{i:04d}_m0",
                    "memory_text": d["persona"],  # explicit preference text (for reference)
                    "gold_policy": policy,
                    "gold_score": SCORE_MAP[policy],
                }
            ],
            "metadata": {
                "split": "implicit_single",
                "intent": d.get("intent", ""),
                "reason": d.get("reason", ""),
                # Preserve the context aligned to each memory. The flattened
                # ``context`` field alone cannot support true context ablation.
                "memory_contexts": [d["implicit_persona"]],
            },
        }
        results.append(entry)
    print(f"RPEval implicit single: {len(data)} entries")

    # --- Multi preference ---
    with open(rpval_dir / "multi_testset.json") as f:
        data = json.load(f)
    for i, d in enumerate(data):
        personas = d["persona"]
        intent_str = d["intent_type"]  # e.g. "ACAA"
        memories = []
        for j, (persona, policy_char) in enumerate(zip(personas, intent_str)):
            policy = normalize_policy(policy_char)
            memories.append({
                "memory_id": f"rpval_impl_multi_{i:04d}_m{j}",
                "memory_text": persona,
                "gold_policy": policy,
                "gold_score": SCORE_MAP[policy],
            })
        entry = {
            "source": "rpval",
            "sample_id": f"rpval_impl_multi_{i:04d}",
            "query": d["question"],
            "context": "\n\n".join(d["implicit_persona"]),  # concatenated dialogue histories
            "memories": memories,
            "metadata": {
                "split": "implicit_multi",
                "reason": d.get("reason", []),
                "memory_contexts": d["implicit_persona"],
            },
        }
        results.append(entry)
    print(f"RPEval implicit multi: {len(data)} entries")

    out_path = out_dir / "rpval_implicit_unified.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written to {out_path}: {len(results)} entries")
    return results


def convert_stratmem(
    out_dir: Path,
    dataset_filename: str = "StratMemBench.json",
    output_filename: str = "stratmem_unified.jsonl",
) -> list[dict]:
    """Convert StratMem-Bench data."""
    sm_path = PROJ / "StratMem-Bench" / "data" / dataset_filename
    if not sm_path.exists():
        print("StratMem-Bench data not found, downloading...")
        sm_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://ghfast.top/https://raw.githubusercontent.com/seucoin/StratMem-Bench/main/data/StratMemBench.json"
        urllib.request.urlretrieve(url, str(sm_path))
        print("Downloaded.")

    with open(sm_path) as f:
        data = json.load(f)

    results = []
    for d in data:
        mem = d["memory"]
        memories = []

        # must -> dominate
        for j, m in enumerate(mem.get("must", [])):
            policy = "dominate"
            memories.append({
                "memory_id": f"stratmem_{d['id']}_must_{j}",
                "memory_text": m["fact"],
                "gold_policy": policy,
                "gold_score": SCORE_MAP[policy],
            })

        # nice -> support
        for j, m in enumerate(mem.get("nice", [])):
            policy = "support"
            memories.append({
                "memory_id": f"stratmem_{d['id']}_nice_{j}",
                "memory_text": m["fact"],
                "gold_policy": policy,
                "gold_score": SCORE_MAP[policy],
            })

        # irr -> ignore
        for j, m in enumerate(mem.get("irr", [])):
            policy = "ignore"
            memories.append({
                "memory_id": f"stratmem_{d['id']}_irr_{j}",
                "memory_text": m["fact"],
                "gold_policy": policy,
                "gold_score": SCORE_MAP[policy],
            })

        # Build context from persona
        persona = d.get("virtual_person_persona", {})
        summary = persona.get("summary", "")
        if isinstance(summary, list):
            summary = " ".join(summary)
        traits = persona.get("traits", [])
        style = persona.get("style", [])
        domains = persona.get("domains", [])
        values = persona.get("values", [])
        context_parts = []
        if summary:
            context_parts.append(f"Persona summary: {summary}")
        if traits:
            context_parts.append(f"Traits: {', '.join(traits)}")
        if style:
            context_parts.append(f"Style: {', '.join(style)}")
        if domains:
            context_parts.append(f"Domains: {', '.join(domains)}")
        if values:
            context_parts.append(f"Values: {', '.join(values)}")
        roles = d.get("roles", {})
        if roles:
            context_parts.append(
                f"Participants: human={roles.get('human', '')}; "
                f"virtual_person={roles.get('virtual_person', '')}"
            )
        if d.get("query_time"):
            context_parts.append(f"Current query time: {d['query_time']}")
        context = "\n".join(context_parts)

        # Add dialogue history if present
        history = d.get("history", "")
        if history:
            try:
                hist_list = json.loads(history) if isinstance(history, str) else history
                if isinstance(hist_list, list):
                    context += "\n\nDialogue history:\n" + "\n".join(str(h) for h in hist_list)
            except (json.JSONDecodeError, TypeError):
                context += f"\n\nDialogue history:\n{history}"

        entry = {
            "source": "stratmem",
            "sample_id": f"stratmem_{d['id']}",
            "query": d["query"],
            "context": context,
            "memories": memories,
            "metadata": {
                "given_type": d.get("given_type", ""),
                "inferred_type": d.get("inferred_type", ""),
                "query_time": d.get("query_time", ""),
                "virtual_person": persona.get("virtual_person_name", ""),
                "roles": d.get("roles", {}),
            },
        }
        results.append(entry)

    out_path = out_dir / output_filename
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"StratMem: {len(results)} entries, written to {out_path}")
    return results


def convert_benchpres(out_dir: Path, source_rows=None) -> list[dict]:
    """Convert BenchPreS data from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = source_rows if source_rows is not None else load_dataset("sangyon/BenchPreS", split="test")
    except Exception as e:
        print(f"Failed to load BenchPreS from HuggingFace: {e}")
        print("Skipping BenchPreS. Download manually or run on a machine with HF access.")
        return []

    results = []
    for i, row in enumerate(ds):
        prefs = row["preference_attribute"]
        labels = row["preference_label"]
        memories = []
        for j, (pref, label) in enumerate(zip(prefs, labels)):
            policy = "dominate" if label else "ignore"
            memories.append({
                "memory_id": f"benchpres_{i:04d}_m{j}",
                "memory_text": pref,
                "gold_policy": policy,
                "gold_score": SCORE_MAP[policy],
            })

        entry = {
            "source": "benchpres",
            "sample_id": f"benchpres_{i:04d}",
            "query": row["task"],
            "context": row["prompt"],  # full profile + task instruction
            "memories": memories,
            "metadata": {
                "name": row.get("name", ""),
                "task": row["task"],
                "recipient": row.get("recipient", ""),
                "domain": row.get("domain", ""),
            },
        }
        results.append(entry)

    out_path = out_dir / "benchpres_unified.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"BenchPreS: {len(results)} entries, written to {out_path}")
    return results


def print_stats(results: list[dict], name: str):
    """Print label distribution stats."""
    policy_counts = {"ignore": 0, "support": 0, "dominate": 0}
    total_memories = 0
    for r in results:
        for m in r["memories"]:
            policy_counts[m["gold_policy"]] += 1
            total_memories += 1
    print(f"\n{name} stats:")
    print(f"  Samples: {len(results)}")
    print(f"  Total memories: {total_memories}")
    print(f"  Policy distribution: {policy_counts}")
    print(f"  Avg memories/sample: {total_memories/len(results):.1f}")


def main():
    out_dir = DATA_DIR / "unified"
    out_dir.mkdir(exist_ok=True)

    all_results = {}

    rpval = convert_rpval(out_dir)
    print_stats(rpval, "RPEval")
    all_results["rpval"] = rpval

    stratmem = convert_stratmem(out_dir)
    print_stats(stratmem, "StratMem")
    all_results["stratmem"] = stratmem

    benchpres = convert_benchpres(out_dir)
    if benchpres:
        print_stats(benchpres, "BenchPreS")
        all_results["benchpres"] = benchpres

    # Summary
    total_samples = sum(len(v) for v in all_results.values())
    total_mem = 0
    for name, entries in all_results.items():
        for e in entries:
            total_mem += len(e["memories"])
    print(f"\n=== TOTAL ===")
    print(f"Samples: {total_samples}")
    print(f"Memory-policy pairs: {total_mem}")


if __name__ == "__main__":
    main()
