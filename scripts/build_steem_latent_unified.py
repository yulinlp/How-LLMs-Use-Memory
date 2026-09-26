"""Convert Tag-masked SteeM rows to one-memory latent-controller format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval_split_tag_masked.jsonl")
    parser.add_argument("--output", default="data/unified/steem_tag_masked_unified.jsonl")
    args = parser.parse_args()
    rows = []
    for line in Path(args.input).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        level = int(row["target_dependence_score"])
        policy = "ignore" if level == 1 else "dominate" if level == 5 else "support"
        sample_id = str(row["uid"])
        rows.append({
            "source": "steem",
            "sample_id": sample_id,
            "query": row["tag_masked_query"],
            # The complete supplied history is the single removable memory.
            "context": "",
            "memories": [{
                "memory_id": f"{sample_id}::history",
                "memory_text": row["filtered_context"],
                "gold_policy": policy,
                "gold_score": level,
            }],
            "metadata": {
                "gold_level": level,
                "query_id": row["query_id"],
                "task": row["task"],
                "domain": row["domain"],
                "target": row["target"],
            },
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
