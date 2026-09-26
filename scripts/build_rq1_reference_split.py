#!/usr/bin/env python3
"""Build the fixed query-group split used by the RQ1 diagnostics.

The unit of splitting is the complete RPEval ``sample_id`` group.  We keep
all memory items belonging to a sample on the same side of the split and
select a small, stratified reference set without looking at activations.
The resulting manifest is deliberately independent of any particular
backbone so that every model uses the same query groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


POLICIES = ("ignore", "support", "dominate")
REGIMES = ("single", "multi")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_rank(seed: int, group_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"rq1-reference|{seed}|{group_id}".encode()).hexdigest()
    return digest, group_id


def group_records(rows: list[dict]) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for row in rows:
        group_id = str(row["sample_id"])
        memories = row.get("memories", [])
        regime = str(row.get("kind") or ("single" if len(memories) == 1 else "multi"))
        if regime not in REGIMES:
            raise ValueError(f"Unexpected regime for {group_id}: {regime}")
        labels = tuple(str(memory["gold_policy"]) for memory in memories)
        if group_id in groups:
            previous = groups[group_id]
            if previous["regime"] != regime or previous["labels"] != labels:
                raise ValueError(f"Inconsistent duplicate group: {group_id}")
            continue
        if not set(labels) <= set(POLICIES):
            raise ValueError(f"Unexpected policy label in {group_id}: {labels}")
        groups[group_id] = {
            "group_id": group_id,
            "regime": regime,
            "labels": labels,
            "n_memories": len(memories),
        }
    return groups


def choose_groups(
    candidates: list[str],
    records: dict[str, dict],
    count: int,
    seed: int,
    min_groups_per_policy: int = 2,
) -> list[str]:
    ordered = sorted(candidates, key=lambda group: stable_rank(seed, group))
    rank = {group: index for index, group in enumerate(ordered)}
    selected: list[str] = []
    selected_counts = Counter()

    # First guarantee repeated group-level coverage for every policy.  This is
    # needed by group-wise leave-one-group-out reference validation below; a
    # one-off memory item would disappear entirely in one of the folds.
    while any(selected_counts[policy] < min_groups_per_policy for policy in POLICIES):
        eligible = [
            group for group in ordered
            if group not in selected
            and any(
                selected_counts[policy] < min_groups_per_policy
                for policy in set(records[group]["labels"])
            )
        ]
        if not eligible:
            break
        chosen = max(
            eligible,
            key=lambda group: (
                sum(
                    selected_counts[policy] < min_groups_per_policy
                    for policy in set(records[group]["labels"])
                ),
                -rank[group],
            ),
        )
        selected.append(chosen)
        selected_counts.update(set(records[chosen]["labels"]))

    for group in ordered:
        if len(selected) >= count:
            break
        if group not in selected:
            selected.append(group)
    if len(selected) != count:
        raise ValueError(f"Could not choose {count} groups from {len(candidates)}")
    missing = [
        policy for policy in POLICIES
        if selected_counts[policy] < min_groups_per_policy
    ]
    if missing:
        raise ValueError(f"Reference set does not cover policies twice: {missing}")
    return selected


def build_manifest(rows: list[dict], fraction: float, seed: int) -> dict:
    records = group_records(rows)
    all_groups = sorted(records)
    if not all_groups:
        raise ValueError("No query groups found")
    reference_count = round(len(all_groups) * fraction)
    reference_count = max(6, min(reference_count, len(all_groups) - 6))

    by_regime = {
        regime: sorted(
            [group for group in all_groups if records[group]["regime"] == regime]
        )
        for regime in REGIMES
    }
    single_count = round(reference_count * len(by_regime["single"]) / len(all_groups))
    single_count = max(3, min(single_count, reference_count - 3))
    multi_count = reference_count - single_count

    selected = []
    selected.extend(choose_groups(by_regime["single"], records, single_count, seed))
    selected.extend(choose_groups(by_regime["multi"], records, multi_count, seed + 1))
    reference_groups = sorted(selected)
    final_groups = sorted(set(all_groups) - set(reference_groups))

    def row_counts(group_ids: list[str]) -> dict:
        counts = Counter()
        for group_id in group_ids:
            record = records[group_id]
            for label in record["labels"]:
                counts[(record["regime"], label)] += 1
        return {
            regime: {policy: counts[(regime, policy)] for policy in POLICIES}
            for regime in REGIMES
        }

    manifest = {
        "manifest_version": 1,
        "purpose": "RQ1 frozen-activation layer dynamics",
        "benchmark": "RPEval implicit preference",
        "group_key": "sample_id",
        "selection": {
            "fraction": fraction,
            "seed": seed,
            "reference_groups_are_complete_query_groups": True,
            "stratified_by": "single_vs_multi",
            "policy_coverage_required_within_each_regime": True,
            "minimum_reference_groups_per_policy_within_each_regime": 2,
            "selection_does_not_use_activations": True,
        },
        "n_total_groups": len(all_groups),
        "n_reference_groups": len(reference_groups),
        "n_final_test_groups": len(final_groups),
        "reference_groups_by_regime": {
            regime: sum(records[group]["regime"] == regime for group in reference_groups)
            for regime in REGIMES
        },
        "reference_memory_counts": row_counts(reference_groups),
        "final_test_memory_counts": row_counts(final_groups),
        "reference_group_policy_counts": {
            regime: {
                policy: sum(
                    records[group]["regime"] == regime
                    and policy in set(records[group]["labels"])
                    for group in reference_groups
                )
                for policy in POLICIES
            }
            for regime in REGIMES
        },
        "reference_group_ids": reference_groups,
        "final_test_group_ids": final_groups,
    }
    for regime in REGIMES:
        observed = {
            label
            for group in reference_groups
            if records[group]["regime"] == regime
            for label in records[group]["labels"]
        }
        if observed != set(POLICIES):
            raise ValueError(f"{regime} reference groups miss labels: {observed}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/unified/rpval_implicit_unified.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/latent_mccs/rq1_layer_dynamics/rq1_reference_split.json"),
    )
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = build_manifest(read_jsonl(args.input), args.fraction, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "n_total_groups", "n_reference_groups", "n_final_test_groups",
        "reference_groups_by_regime", "reference_memory_counts",
    )}, ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
