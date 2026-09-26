from __future__ import annotations

import argparse
import random
from typing import Any

from .io import load_json_or_jsonl, write_jsonl
from .prompts import (
    build_user_query,
    load_control_instructions,
    make_query_id,
    mask_memory_reliance_tag,
    pick_instruction,
)


def expand_samples(
    samples: list[dict[str, Any]],
    levels: list[int],
    instruction_path: str | None,
    seed: int,
    include_uncontrolled: bool,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    instructions = load_control_instructions(instruction_path)
    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        query = str(sample.get("query", "")).strip()
        context = str(sample.get("context_prompt") or sample.get("full_context") or "").strip()
        if not query or not context:
            continue
        query_id = sample.get("query_id") or make_query_id(sample)
        base = {
            "source_index": idx,
            "query_id": query_id,
            "domain": sample.get("domain"),
            "topic": sample.get("topic"),
            "subject": sample.get("subject"),
            "directory_index": sample.get("directory_index"),
            "event_id": sample.get("event_id"),
            "event_time_index": sample.get("event_time_index"),
            "task": sample.get("task"),
            "target": sample.get("target"),
            "essential": sample.get("essential") or sample.get("essential_artifacts") or [],
            "query": query,
            "context": context,
            "filtered_context": context,
        }
        if include_uncontrolled:
            rows.append(
                {
                    **base,
                    "uid": f"{query_id}::none",
                    "control_level": None,
                    "target_dependence_score": None,
                    "rewritten_query": query,
                }
            )
        for level in levels:
            instruction = pick_instruction(instructions, level, rng)
            tag_masked_instruction = mask_memory_reliance_tag(instruction)
            rows.append(
                {
                    **base,
                    "uid": f"{query_id}::md{level}",
                    "control_level": level,
                    "target_dependence_score": level,
                    "control_instruction": instruction,
                    "rewritten_query": build_user_query(query, instruction),
                    "tag_masked_instruction": tag_masked_instruction,
                    "tag_masked_query": build_user_query(query, tag_masked_instruction),
                }
            )
    return rows


def split_by_project(samples: list[dict[str, Any]], seed: int, extract_frac: float):
    """Split samples into two sets by directory_index (project) to avoid leakage.

    Samples from the same project never appear in both sets.
    Returns (extract_samples, eval_samples).
    """
    rng = random.Random(seed)
    by_project: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        by_project.setdefault(str(s.get("directory_index")), []).append(s)
    projects = sorted(by_project.keys())
    rng.shuffle(projects)
    n_extract = int(len(projects) * extract_frac)
    extract_projects = set(projects[:n_extract])
    extract_samples, eval_samples = [], []
    for p in projects:
        target = extract_samples if p in extract_projects else eval_samples
        target.extend(by_project[p])
    # Shuffle within each split so subsequent capping samples across projects
    rng.shuffle(extract_samples)
    rng.shuffle(eval_samples)
    return extract_samples, eval_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SteeM contexts as eval JSONL with 1-5 memory-dependence targets.")
    parser.add_argument("--input", required=True, help="SteeM JSON/JSONL input, e.g. sampled_contexts.json or all_contexts.json.gz")
    parser.add_argument("--output", required=True, help="Output JSONL (single split)")
    parser.add_argument("--extract-output", default=None, help="If set, split by project: write direction-extraction set here")
    parser.add_argument("--eval-output", default=None, help="If set with --extract-output, write eval set here")
    parser.add_argument("--levels", default="1,2,3,4,5", help="Comma-separated target MD levels")
    parser.add_argument("--extract-levels", default="1,5", help="Levels for the extraction split (default 1,5)")
    parser.add_argument("--limit", type=int, default=None, help="Limit total input samples (before split)")
    parser.add_argument("--num-extract", type=int, default=None, help="Cap number of source samples in extraction split")
    parser.add_argument("--num-eval", type=int, default=None, help="Cap number of source samples in eval split")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--control-instruct-json", default=None)
    parser.add_argument("--include-uncontrolled", action="store_true")
    args = parser.parse_args()

    samples = load_json_or_jsonl(args.input)
    if args.limit is not None:
        samples = samples[: args.limit]
    print(f"Loaded {len(samples)} source samples")

    if args.extract_output and args.eval_output:
        # Split by project to avoid train/eval leakage
        extract_src, eval_src = split_by_project(samples, args.seed, extract_frac=0.5)
        if args.num_extract is not None:
            extract_src = extract_src[: args.num_extract]
        if args.num_eval is not None:
            eval_src = eval_src[: args.num_eval]
        extract_levels = [int(x) for x in args.extract_levels.split(",") if x.strip()]
        eval_levels = [int(x) for x in args.levels.split(",") if x.strip()]
        extract_rows = expand_samples(extract_src, extract_levels, args.control_instruct_json, args.seed, args.include_uncontrolled)
        eval_rows = expand_samples(eval_src, eval_levels, args.control_instruct_json, args.seed, args.include_uncontrolled)
        write_jsonl(args.extract_output, extract_rows)
        write_jsonl(args.eval_output, eval_rows)
        print(f"Wrote {len(extract_rows)} extraction rows (levels={extract_levels}) to {args.extract_output}")
        print(f"Wrote {len(eval_rows)} eval rows (levels={eval_levels}) to {args.eval_output}")
        # Verify no project overlap
        ep = set(str(r.get("directory_index")) for r in extract_rows)
        vp = set(str(r.get("directory_index")) for r in eval_rows)
        print(f"Project overlap (must be 0): {len(ep & vp)}")
    else:
        levels = [int(x) for x in args.levels.split(",") if x.strip()]
        rows = expand_samples(samples, levels, args.control_instruct_json, args.seed, args.include_uncontrolled)
        write_jsonl(args.output, rows)
        print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
