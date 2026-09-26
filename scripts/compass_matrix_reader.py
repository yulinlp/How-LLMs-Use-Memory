"""Isolated COMPASS numeric readers; local execution only.

Default --stage prepare uses only the standard library and loads no weights.
Run extract explicitly on an allocated machine, then predict on CPU. Outputs
are separate from all historical 4B artifacts; legacy tensors are NOT imported
or relabelled. --source-benchmark enables frozen benchmark-reference transfer.

The fixed portability recipe averages cosine similarity over block outputs
[floor(16*L/36), floor(24*L/36)), excluding the embedding output. Thus L=36
uses blocks 16..23 / hidden_states[17:25]. This is not a claim of the same
tuned optimum: the old 4B RPEval reader selected layers/k/regimes using
calibration labels. Here layers and k=1 are fixed, with no target tuning.
Optional --k is an explicitly supplied ablation, not automatic selection.
Numeric references map binary scores to 2*s-1 and ordinal 1..5 to (s-3)/2.
MemoryAtoms' ordinal mapping is a portability hypothesis, not calibrated
actuator response. No query label is accepted by nearest_numeric().
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import unicodedata

EXTRACTION_LAYOUT = "v2:rpval-singleton-string;benchpres-full-row-deletion;other-singleton"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from steem_adapt.reproduction import model_paths
MODELS = model_paths()
DATA = {"rpeval": "rpval_implicit_unified.jsonl",
        "benchpres": "benchpres_unified.jsonl", "steem": "steem_tag_masked_unified.jsonl"}
SPLITS = {
    "rpeval": "runs/rq3_endogenous_response_20260920/Qwen3-4B/protocol.json",
    "benchpres": "runs/latent_mccs/Qwen3-4B/benchpres_indomain_directional_rbf_controller/predictions.jsonl",
    "steem": "runs/latent_mccs/Qwen3-4B/steem_tag_masked_controller/prototype_.40/predictions.jsonl",
}


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def group(row):
    return str(row.get("metadata", {}).get("query_id", row["sample_id"]))


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def layers_for(depth):
    if depth < 2:
        raise ValueError("At least two transformer blocks required")
    start = 16 * depth // 36
    return list(range(start, max(start + 1, 24 * depth // 36)))


def clean(row):
    # The renderer needs only these fields; no metadata, answer, or gold keys.
    return {"source": row["source"], "sample_id": str(row["sample_id"]),
            "query": row["query"], "context": row.get("context", ""),
            "memories": [{"memory_id": str(m["memory_id"]),
                          "memory_text": m["memory_text"]} for m in row["memories"]]}


def calibration(rows, benchmark):
    path = ROOT / SPLITS[benchmark]
    if benchmark == "rpeval":
        ids = set(map(str, json.loads(path.read_text())["calibration_group_ids"]))
        selected = {str(r["sample_id"]) for r in rows if group(r) in ids}
    else:
        split_rows = read(path)
        assignments = {}
        for r in split_rows:
            sid, split = str(r["sample_id"]), r["split"]
            if split not in {"calibration", "heldout"}:
                raise ValueError("Unknown split")
            if sid in assignments and assignments[sid] != split:
                raise ValueError("Conflicting sample split")
            assignments[sid] = split
        selected = {sid for sid, split in assignments.items() if split == "calibration"}
    all_ids = {str(r["sample_id"]) for r in rows}
    if not selected or not selected <= all_ids or len(all_ids) != len(rows):
        raise ValueError("Missing calibration IDs or duplicate samples")
    groups = {group(r) for r in rows}
    cal_groups = {group(r) for r in rows if str(r["sample_id"]) in selected}
    if any((str(r["sample_id"]) in selected) != (group(r) in cal_groups) for r in rows):
        raise ValueError("Calibration splits a whole query group")
    if len(cal_groups) * 5 > len(groups):
        raise ValueError("Calibration exceeds 20% of query groups")
    if benchmark == "rpeval" and ids != cal_groups:
        raise ValueError("Unknown RPEval calibration groups")
    return cal_groups, {"path": str(path), "calibration_groups": sorted(cal_groups),
                        "total_groups": len(groups), "calibration_samples": len(selected)}


def overlap_audit(atoms, datasets):
    if len(atoms) != 100 or len({group(r) for r in atoms}) != 20:
        raise ValueError("Expected original MemoryAtoms: 100 samples, 20 primitive families")
    for g in {group(r) for r in atoms}:
        family = [r for r in atoms if group(r) == g]
        if len(family) != 5 or {m["gold_score"] for r in family for m in r["memories"]} != {1, 2, 3, 4, 5}:
            raise ValueError("Invalid primitive family levels")
    def fields(rows):
        return {"sample_id": {str(r["sample_id"]) for r in rows},
                "group_id": {group(r) for r in rows},
                "query": {normalize(r["query"]) for r in rows},
                "memory_text": {normalize(m["memory_text"]) for r in rows for m in r["memories"]}}
    source = fields(atoms)
    report = {}
    for name, rows in datasets.items():
        target = fields(rows)
        report[name] = {key: len(source[key] & target[key]) for key in source}
        if any(report[name].values()):
            raise ValueError(f"MemoryAtoms overlap: {name}: {report[name]}")
    return {"counts": report, "method": "IDs and NFKC/casefold/whitespace-normalized exact query and memory text",
            "limitation": "Exact-match audit does not establish semantic or pretraining non-overlap"}


def items(rows, benchmark, cal_groups):
    result = []
    for row in rows:
        mids = [str(m["memory_id"]) for m in row["memories"]]
        if not mids or len(set(mids)) != len(mids):
            raise ValueError("Empty/duplicate memory identities")
        for mid in mids:
            result.append({"uid": f"{row['sample_id']}::{mid}", "sample_id": str(row["sample_id"]),
                           "memory_id": mid, "group_id": group(row), "benchmark": benchmark,
                           "query_key": normalize(row["query"]),
                           "split": "calibration" if group(row) in cal_groups else "heldout"})
    return result


def target(score, benchmark):
    score = float(score)
    ordinal = benchmark in {"steem", "memoryatoms"}
    valid = score in {1, 2, 3, 4, 5} if ordinal else (0 <= score <= 1 if benchmark == "rpeval" else score in {0, 1})
    if not math.isfinite(score) or not valid:
        raise ValueError(f"Unexpected {benchmark} reference score: {score}")
    return (score - 3) / 2 if ordinal else 2 * score - 1


def write_once(path, value, jsonl=False):
    content = ("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in value) if jsonl
               else json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Existing output differs; use a separate --output-root: {path}")
        return
    with path.open("x") as stream:
        stream.write(content)


def prepare(args):
    checkpoint = Path(MODELS.get(args.model, args.model)).resolve()
    config = json.loads((checkpoint / "config.json").read_text())
    text_config = config.get("text_config", config)
    depth = int(text_config["num_hidden_layers"])
    needed = set(DATA) if args.reference == 'independent' else {args.benchmark, args.source_benchmark or args.benchmark}
    data_dir = Path(getattr(args, 'data_dir', ROOT / 'data/unified'))
    datasets = {b: read(data_dir / DATA[b]) for b in sorted(needed)}
    splits = {}
    if getattr(args, 'split_manifest', None):
        override = json.loads(Path(args.split_manifest).read_text())
        for b, rows in datasets.items():
            ids = set(override[b])
            groups = {group(r) for r in rows}
            if not ids or not ids < groups:
                raise ValueError(f'Invalid explicit calibration groups: {b}')
            splits[b] = (ids, dict(path=str(args.split_manifest), calibration_groups=sorted(ids),
                                  total_groups=len(groups),
                                  calibration_samples=sum(group(r) in ids for r in rows)))
    else:
        splits = {b: calibration(rows, b) for b, rows in datasets.items()}
    source = args.source_benchmark or args.benchmark
    if args.reference == "independent":
        if args.source_benchmark:
            raise ValueError("Independent reference always uses MemoryAtoms")
        refs = read(ROOT / "data/unified/memoryatoms_unified.jsonl")
        audit = overlap_audit(refs, datasets)
        source = "memoryatoms"
        ref_groups = {group(r) for r in refs}
    else:
        ref_groups = splits[source][0]
        refs = [r for r in datasets[source] if group(r) in ref_groups]
        audit = {"method": "Whole query group and normalized query exclusion at prediction"}
    queries = datasets[args.benchmark]
    qmeta = items(queries, args.benchmark, splits[args.benchmark][0])
    rmeta = items(refs, source, ref_groups)
    # This is the sole label access, restricted to the chosen reference rows.
    values = [target(m["gold_score"], source) for r in refs for m in r["memories"]]
    directory = Path(args.output_root) / checkpoint.name / args.benchmark / args.reference
    protocol = {"version": 1, "model": str(checkpoint), "text_model_type": text_config.get("model_type"),
                "depth": depth, "hidden_size": text_config["hidden_size"], "layers": layers_for(depth),
                "hidden_state_indices": [i + 1 for i in layers_for(depth)],
                "readout": "last prompt token; per-layer L2-normalized h(only i)-h(empty); mean cosine",
                "selection": "Fixed relative-middle portability choice; no calibration search; not same tuned optimum as legacy 4B",
                "k": args.k, "temperature": 0.1, "reference": args.reference,
                "source_benchmark": source, "benchmark": args.benchmark,
                "splits": {b: s[1] for b, s in splits.items()}, "overlap_audit": audit,
                "runtime_gold": False, "legacy_artifact_reuse": False,
                "dtype": args.dtype, "renderer": "steem_adapt.mccs_generate.render_latent_prompt",
                "query_items": len(qmeta), "reference_items": len(rmeta)}
    directory.mkdir(parents=True, exist_ok=True)
    write_once(directory / "reader_manifest.json", protocol)
    write_once(directory / "evaluation.jsonl",
               [clean(r) for r in queries if group(r) not in splits[args.benchmark][0]], True)
    for name, records in [("queries", list(map(clean, queries))), ("references", list(map(clean, refs))),
                          ("query_metadata", qmeta), ("reference_metadata", rmeta)]:
        write_once(directory / (name + ".jsonl"), records, True)
    write_once(directory / "reference_values.json", values)
    return directory, protocol


def isolated_pairs(row):
    """Yield (render row, kept-stream removals, empty-stream removals).

    RPEval MUST use singleton rows: its renderer branches on list length.
    BenchPreS MUST retain the full row to delete other preferences from context.
    This matches the existing executor plus its BenchPreS deletion adapter.
    """
    for memory in row["memories"]:
        if row["source"] == "benchpres":
            all_ids = {m["memory_id"] for m in row["memories"]}
            yield row, all_ids - {memory["memory_id"]}, all_ids
        else:
            yield {**row, "memories": [memory]}, set(), {memory["memory_id"]}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def signature_contract(directory, protocol, name):
    """Label-free identity: prevent layout/model/input drift during cache reuse.

    Checkpoint weight identity uses resolved path, size and mtime (not a full
    multi-GB content hash); assumes local checkpoint files are immutable.
    """
    checkpoint = Path(protocol["model"])
    model_files = sorted(p for p in checkpoint.iterdir()
                         if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model", ".jinja"})
    prefix = "query" if name == "queries" else "reference"
    return {"layout": EXTRACTION_LAYOUT,
            **{k: protocol[k] for k in ("model", "depth", "hidden_size", "layers", "hidden_state_indices", "dtype")},
            "model_files": [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in model_files],
            "config_sha256": hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(),
            "renderer_sha256": hashlib.sha256((ROOT / "steem_adapt/mccs_generate.py").read_bytes()).hexdigest(),
            "prompts_sha256": hashlib.sha256((ROOT / "steem_adapt/prompts.py").read_bytes()).hexdigest(),
            "prepare_generation_sha256": hashlib.sha256((ROOT / "steem_adapt/prepare_generation.py").read_bytes()).hexdigest(),
            "rpeval_template_sha256": hashlib.sha256(Path(os.environ.get('RPEVAL_PROMPT_FILE', ROOT / 'RPEval/prompts/prompts.py')).read_bytes()).hexdigest(),
            "rows_sha256": digest(read(directory / (name + ".jsonl"))),
            "metadata_sha256": digest(read(directory / (prefix + "_metadata.jsonl"))),
            "readout": "last prompt token; raw fp32 isolated differences; eager; no padding"}


def load_signatures(path, contract):
    import torch
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("signature_contract") != contract:
        raise ValueError(f"Stale/unversioned signatures (including pre-fix RPEval layout): {path}; use a fresh output root or explicitly quarantine old tensors")
    return artifact["signatures"]


def cached_signatures(path, contract, compute):
    """Lock before compute and publish atomically: parallel reference lanes share work."""
    import fcntl
    import tempfile
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            return load_signatures(path, contract)
        signatures = compute()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save({"signature_contract": contract, "signatures": signatures}, stream)
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return signatures


def configure_reader_runtime(model_path):
    """Select hybrid compatibility sources before any HF/renderer import.

    Kept inside lazy model loading: prepare/predict and complete cache hits do
    not need Transformers. Ordinary Qwen3/Llama keep their installed runtime.
    """
    config = json.loads((Path(model_path) / "config.json").read_text())
    family = config.get("text_config", config).get("model_type")
    if family in {"qwen3_5", "qwen3_5_text"} or config.get("model_type") == "qwen3_5":
        sys.path.insert(0, str(ROOT / "scripts"))
        from compass_matrix_generate import configure_hybrid_runtime
        configure_hybrid_runtime()
    return config


def extract(directory, protocol, device):
    import torch
    model = tokenizer = render_latent_prompt = None

    def activation(row, removed):
        nonlocal model, tokenizer, render_latent_prompt
        if model is None:
            config = configure_reader_runtime(protocol["model"])
            from transformers import AutoModelForCausalLM, AutoTokenizer
            sys.path.insert(0, str(ROOT))
            from steem_adapt.mccs_generate import render_latent_prompt
            tokenizer = AutoTokenizer.from_pretrained(protocol["model"], local_files_only=True, use_fast=True)
            model_class = AutoModelForCausalLM
            if config.get("model_type") == "qwen3_5":
                from transformers import Qwen3_5ForConditionalGeneration
                model_class = Qwen3_5ForConditionalGeneration
            model = model_class.from_pretrained(
                protocol["model"], local_files_only=True,
                torch_dtype=getattr(torch, protocol["dtype"]), attn_implementation="eager").to(device).eval()
        prompt = render_latent_prompt(tokenizer, row, removed)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        if len(out.hidden_states) != protocol["depth"] + 1:
            raise ValueError("Unexpected hidden-state layout (embedding + block outputs required)")
        return torch.stack([out.hidden_states[i][0, -1].float().cpu() for i in protocol["hidden_state_indices"]])

    for name in ("queries", "references"):
        contract = signature_contract(directory, protocol, name)

        def compute():
            vectors = []
            for row in read(directory / (name + ".jsonl")):
                empty_cache = {}
                for render_row, kept_removed, empty_removed in isolated_pairs(row):
                    # Empty prompt is identical across singleton RPEval items,
                    # but cache by actual row/removals to avoid assuming that.
                    key = digest([render_row, sorted(empty_removed)])
                    if key not in empty_cache:
                        empty_cache[key] = activation(render_row, empty_removed)
                    vectors.append(activation(render_row, kept_removed) - empty_cache[key])
                print(f"EXTRACT {name} {row['sample_id']} {len(vectors)}", flush=True)
            return torch.stack(vectors)

        if name == "queries":
            shared = directory.parent / "_query_signatures" / (digest(contract) + ".pt")
            def shared_compute():
                return cached_signatures(shared, contract, compute)
            cached_signatures(directory / "queries.pt", contract, shared_compute)
        else:
            cached_signatures(directory / "references.pt", contract, compute)


def nearest_numeric(query, references, values, qmeta, rmeta, k=1, temperature=0.1):
    """CPU numeric readout; one neighbor per whole group, no query targets."""
    import torch
    import torch.nn.functional as F
    import math
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if k < 1 or query.ndim != 3 or references.ndim != 3 or query.shape[1:] != references.shape[1:]:
        raise ValueError("Invalid k or signature shape")
    if len(query) != len(qmeta) or len(references) != len(rmeta) or len(values) != len(rmeta):
        raise ValueError("Signature/metadata/value count mismatch")
    if not all(torch.isfinite(x).all() for x in (query, references, values)):
        raise ValueError("Nonfinite readout input")
    refs = F.normalize(references.float(), dim=-1).flatten(1)
    results = []
    for vector, meta in zip(query, qmeta):
        sim = refs @ F.normalize(vector.float(), dim=-1).flatten() / query.shape[1]
        seen, selected = set(), []
        # stable sort makes tied cosine scores deterministic in reference order.
        for j in torch.argsort(sim, descending=True, stable=True).tolist():
            ref = rmeta[j]
            g = (ref["benchmark"], ref["group_id"])
            if (g == (meta["benchmark"], meta["group_id"]) or
                    ref["query_key"] == meta["query_key"] or g in seen):
                continue
            selected.append(j)
            seen.add(g)
            if len(selected) == k:
                break
        if not selected:
            raise ValueError(f"No query-disjoint references for {meta['uid']}")
        coefficient = float(torch.softmax(sim[selected] / temperature, dim=0) @ values[selected].float())
        results.append({**{key: meta[key] for key in ("uid", "sample_id", "memory_id", "split")},
                        "steering_coefficient": coefficient, "reader": "query_group_excluded_numeric_reference",
                        "reference_uids": [rmeta[j]["uid"] for j in selected], "effective_k": len(selected)})
    return results


def predict(directory, protocol):
    import torch
    torch.set_num_threads(4)
    tensors = []
    for name in ("queries", "references"):
        contract = signature_contract(directory, protocol, name)
        x = load_signatures(directory / (name + ".pt"), contract)
        if list(x.shape[1:]) != [len(protocol["layers"]), protocol["hidden_size"]]:
            raise ValueError("Wrong backbone signature dimensions")
        tensors.append(x)
    result = nearest_numeric(*tensors, torch.tensor(json.loads((directory / "reference_values.json").read_text())),
                             read(directory / "query_metadata.jsonl"), read(directory / "reference_metadata.jsonl"), protocol["k"], protocol.get("matching_temperature", 0.1))
    write_once(directory / "predictions.jsonl", result, True)
    print(f"PREDICTED {len(result)} items -> {directory / 'predictions.jsonl'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Known six-model alias or local checkpoint directory")
    parser.add_argument("--benchmark", choices=DATA, required=True)
    parser.add_argument("--reference", choices=["benchmark", "independent"], required=True)
    parser.add_argument("--source-benchmark", choices=DATA)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/compass_matrix_20260922")
    parser.add_argument("--validate", action="store_true", help="CPU metadata preparation and audits only; no model/device load")
    parser.add_argument("--stage", choices=["prepare", "extract", "predict", "all"], default="prepare")
    parser.add_argument("--device", default="cpu", help="Explicit local device for extraction; never schedules jobs")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--split-manifest", type=Path, help="Explicit benchmark-to-whole-group-ID mapping")
    parser.add_argument("--data-dir", type=Path, default=ROOT / 'data/unified')
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    if args.validate:
        args.stage = "prepare"
    if args.stage == "predict":
        checkpoint = Path(MODELS.get(args.model, args.model)).resolve()
        source = "memoryatoms" if args.reference == "independent" else (args.source_benchmark or args.benchmark)
        directory = args.output_root / checkpoint.name / args.benchmark / args.reference
        protocol = json.loads((directory / "reader_manifest.json").read_text())
        if (protocol["model"] != str(checkpoint) or protocol["k"] != args.k
                or protocol["source_benchmark"] != source):
            raise ValueError("Model/k differs from prepared protocol")
        predict(directory, protocol)  # No unified data or gold labels loaded here.
        return
    directory, protocol = prepare(args)
    print(json.dumps({"directory": str(directory), "layers": protocol["layers"],
                      "extraction_layout": EXTRACTION_LAYOUT,
                      "reference_items": protocol["reference_items"], "query_items": protocol["query_items"]}))
    if args.stage in {"extract", "all"}:
        extract(directory, protocol, args.device)
    if args.stage == "all":
        predict(directory, protocol)


if __name__ == "__main__":
    main()
