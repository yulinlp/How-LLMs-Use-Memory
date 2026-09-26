#!/usr/bin/env python3
"""COMPASS main/ablation adapter; local execution only, no scheduler submission.

Use --validate for a dependency-free plan or --audit-only for tokenizer audits.
The caller supplies frozen numeric reader predictions; this program never fits
a reader or derives coefficients from benchmark gold labels. All conditions
consume exactly those coefficients and the same generated prefix per stream.

Qwen3-4B defaults to frozen layer 31; other backbones use the last full-attention
layer from config. Easysteer's older Transformers uses the local .deps runtime
for Qwen3.5 only. A --limit/--max-new-tokens GPU smoke must use a separate output
path because resume metadata distinguishes smoke subsets from full runs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from steem_adapt.reproduction import model_paths
MODEL_PATHS = model_paths()
DATA = {
    "rpeval": ROOT / "data/unified/rpval_implicit_unified.jsonl",
    "benchpres": ROOT / "data/unified/benchpres_unified.jsonl",
    "steem": ROOT / "data/unified/steem_tag_masked_unified.jsonl",
}
CONDITIONS = ("dynamic_gate", "dynamic_no_gate", "fixed_gate", "fixed_no_gate", "dac_full_vocab", "fixed_dac_full_vocab")


class UnsupportedModel(ValueError):
    pass


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def model_spec(model, requested_layer=None):
    aliases = {"Llama3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
               "Llama3.2-3B-Instruct": "Llama-3.2-3B-Instruct"}
    model = aliases.get(model, model)
    path = Path(MODEL_PATHS.get(model, model)).resolve()
    config = json.loads((path / "config.json").read_text())
    text = config.get("text_config", config)
    family = text["model_type"]
    count = text["num_hidden_layers"]
    layer_types = text.get("layer_types")
    if family == "qwen3_5_text" and layer_types is None:
        raise UnsupportedModel("Qwen3.5 requires explicit layer_types in the local config")
    if layer_types is not None and len(layer_types) != count:
        raise UnsupportedModel("layer_types length does not match num_hidden_layers")
    full_layers = [i for i, kind in enumerate(layer_types) if kind == "full_attention"] if layer_types else list(range(count))
    if not full_layers:
        raise UnsupportedModel("No full-attention measurement layer exists")
    # Frozen Qwen3-4B replay remains layer 31; new backbones use the last
    # full-attention block, fixed from config without evaluation selection.
    default_layer = 31 if path.name == "Qwen3-4B" and family == "qwen3" else full_layers[-1]
    layer = requested_layer if requested_layer is not None else default_layer
    if not 0 <= layer < count:
        raise UnsupportedModel(f"Layer {layer} outside {count} decoder layers")
    if layer_types is not None:
        if layer_types[layer] != "full_attention":
            raise UnsupportedModel(f"Layer {layer} is {layer_types[layer]}, not full_attention; no substitute selected")
    if family not in {"qwen3", "llama", "qwen3_5_text"}:
        raise UnsupportedModel(f"Unsupported model family: {family}")
    return dict(name=path.name, path=str(path), family=family, layer=layer, num_layers=count)


def configure_hybrid_runtime():
    """Use the inspected local compatibility tree only when installed HF lacks Qwen3.5.

    Call before importing Transformers. Nothing is installed or changed on disk;
    ordinary Qwen3/Llama processes retain their own existing runtime.
    """
    import importlib.util
    location = importlib.util.find_spec("transformers")
    roots = list(location.submodule_search_locations or []) if location else []
    if any((Path(root) / "models/qwen3_5/modeling_qwen3_5.py").exists() for root in roots):
        return
    if "transformers" in sys.modules or "huggingface_hub" in sys.modules:
        raise UnsupportedModel("Qwen3.5 needs the local compatibility runtime before HF imports; start a fresh adapter process")
    paths = [ROOT / ".deps/huggingface-hub-main/src", ROOT / ".deps/transformers-main/src"]
    if not (paths[1] / "transformers/models/qwen3_5/modeling_qwen3_5.py").exists() or not (paths[0] / "huggingface_hub").is_dir():
        raise UnsupportedModel("Qwen3.5 compatibility sources missing under .deps")
    sys.path[:0] = list(map(str, paths))
    importlib.invalidate_caches()
    print(f"HYBRID_RUNTIME {paths[1]}", flush=True)


class HybridTextView:
    """Read-only module view: native text backbone + original LM head.

    The executor calls .model with its 2-D text positions and passes the native
    hybrid cache back unchanged. No KV-only conversion, cache row pruning,
    norm recreation or visual forward is used. The parent owns all weights.
    """
    def __init__(self, parent, layer):
        self.parent = parent
        backbone = parent.model
        self.model = getattr(backbone, "language_model", backbone)
        self.config = self.model.config
        kinds = self.config.layer_types
        if not 0 <= layer < len(kinds) or kinds[layer] != "full_attention":
            raise UnsupportedModel(f"Hybrid runtime layer {layer} is not full_attention")
        if len(self.model.layers) != self.config.num_hidden_layers or not hasattr(self.model, "norm"):
            raise UnsupportedModel("Hybrid text decoder/final-norm layout mismatch")
        attention = getattr(self.model.layers[layer], "self_attn", None)
        if attention is None or attention.layer_idx != layer:
            raise UnsupportedModel("Hybrid runtime full-attention module/index mismatch")
        if self.config._attn_implementation != "eager":
            raise UnsupportedModel("Hybrid COMPASS requires eager full attention")

    @property
    def device(self):
        return self.model.embed_tokens.weight.device

    def get_output_embeddings(self):
        return self.parent.get_output_embeddings()

    def eval(self):
        self.parent.eval()
        return self


def install_hybrid_loader(executor, spec):
    """Replace only this private executor's loader, never Transformers globals."""
    from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM
    checkpoint = json.loads((Path(spec["path"]) / "config.json").read_text())
    model_class = Qwen3_5ForConditionalGeneration if "text_config" in checkpoint else Qwen3_5ForCausalLM

    class Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            kwargs["local_files_only"] = True
            parent = model_class.from_pretrained(path, **kwargs)
            return HybridTextView(parent, spec["layer"])

    executor.AutoModelForCausalLM = Loader


def select_rows(data, predictions, split, limit=None):
    by_uid = {}
    splits = {}
    for prediction in predictions:
        uid = prediction["uid"]
        if uid in by_uid:
            raise ValueError(f"Duplicate prediction uid: {uid}")
        if uid != f"{prediction['sample_id']}::{prediction['memory_id']}":
            raise ValueError(f"Prediction identity mismatch: {uid}")
        value = prediction.get("steering_coefficient")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Finite numeric steering_coefficient required: {uid}")
        sample = prediction["sample_id"]
        label = prediction.get("split")
        if split != "all" and label not in {"calibration", "heldout"}:
            raise ValueError(f"Missing/invalid prediction split: {uid}")
        if sample in splits and splits[sample] != label:
            raise ValueError(f"Mixed prediction splits for {sample}")
        splits[sample] = label
        by_uid[uid] = prediction
    ids = {sample for sample, label in splits.items() if split == "all" or label == split}
    data_ids = [r["sample_id"] for r in data]
    if len(data_ids) != len(set(data_ids)):
        raise ValueError("Duplicate source sample_id")
    if ids - set(data_ids):
        raise ValueError(f"Prediction samples absent from input: {sorted(ids - set(data_ids))[:5]}")
    rows = [r for r in data if r["sample_id"] in ids]
    for row in rows:
        memories = row["memories"]
        mids = [m["memory_id"] for m in memories]
        if not mids or len(mids) != len(set(mids)):
            raise ValueError(f"Empty/duplicate memory identities: {row['sample_id']}")
        expected = {f"{row['sample_id']}::{mid}" for mid in mids}
        actual = {uid for uid, p in by_uid.items() if p["sample_id"] == row["sample_id"]}
        if expected != actual:
            raise ValueError(f"Prediction memory coverage mismatch: {row['sample_id']}")
    if not rows:
        raise ValueError("No evaluation rows selected")
    return rows[:limit] if limit is not None else rows


def prompt_fields(row):
    """Only native prompt inputs enter the executor; discard labels/answers."""
    return dict(source=row["source"], sample_id=row["sample_id"], query=row["query"],
                context=row.get("context", ""),
                memories=[dict(memory_id=m["memory_id"], memory_text=m["memory_text"])
                          for m in row["memories"]])


class FrozenDirections:
    """Freeze item prefill differences, retaining live bases and live gates."""
    def __init__(self):
        self.directions = None

    def __call__(self, residuals):
        if self.directions is None:
            self.directions = (residuals[:, 1::2] - residuals[:, 2::2]).detach().clone()
            return residuals
        result = residuals.clone()
        result[:, 1::2] = result[:, 2::2] + self.directions
        return result


def make_executor(condition):
    """Private module, including fixed-direction changes; no on-disk edits."""
    source = ROOT / "steem_adapt/latent_online_generate.py"
    code = source.read_text()
    if condition in {"dac_full_vocab", "fixed_dac_full_vocab"}:
        from duet_executor import install_executor
        executor, _, _ = install_executor(fixed_direction=condition.startswith("fixed_"))
        return executor
    if condition.startswith("fixed_"):
        replacements = {
            "    def apply_attention_state(\n": "    freeze_item_directions = FrozenDirections()\n    def apply_attention_state(\n",
            "        updated = current_residuals\n": "        updated = freeze_item_directions(current_residuals)\n",
        }
        for anchor, replacement in replacements.items():
            if code.count(anchor) != 1:
                raise RuntimeError(f"Executor changed; fixed-direction adapter anchor not unique: {anchor!r}")
            code = code.replace(anchor, replacement)
    module = types.ModuleType("steem_adapt._compass_matrix_executor")
    module.__file__ = str(source)
    module.__package__ = "steem_adapt"
    module.FrozenDirections = FrozenDirections
    exec(compile(code, str(source), "exec"), module.__dict__)
    import torch
    from reproduction_helpers import bounded_attention_gate
    module.normalize_attention_gate = (
        (lambda mass: torch.ones_like(mass, dtype=torch.float32))
        if condition.endswith("no_gate") else bounded_attention_gate
    )
    return module


def install_prompt_adapter(executor, benchmark, rows):
    originals = {row["sample_id"]: row for row in rows}
    native_render = executor.render_latent_prompt
    native_spans = executor.memory_token_positions

    def render(tokenizer, row, removed):
        if benchmark != "benchpres":
            return native_render(tokenizer, row, removed)
        full = originals[row["sample_id"]]
        all_ids = {m["memory_id"] for m in full["memories"]}
        kept = {m["memory_id"] for m in row["memories"]} - set(removed)
        return native_render(tokenizer, full, all_ids - kept)

    def spans(tokenizer, prompt, row):
        result = {}
        for memory in row["memories"]:
            result.update(native_spans(tokenizer, prompt, {**row, "memories": [memory]}))
        return result

    executor.render_latent_prompt = render
    executor.memory_token_positions = spans
    return native_render


def audit_prompts(executor, tokenizer, rows, native_render):
    for row in rows:
        full = executor.render_latent_prompt(tokenizer, row, set())
        ids = {m["memory_id"] for m in row["memories"]}
        if full != native_render(tokenizer, row, set()):
            raise ValueError(f"Full native prompt changed: {row['sample_id']}")
        if set(executor.memory_token_positions(tokenizer, full, row)) != {str(i) for i in ids}:
            raise ValueError(f"Missing full-prompt memory spans: {row['sample_id']}")
        for memory in row["memories"]:
            isolated = {**row, "memories": [memory]}
            for removed, expected in ((set(), ids - {memory["memory_id"]}), ({memory["memory_id"]}, ids)):
                # Frozen RPEval renders singleton personas as strings, but
                # multi-persona rows as lists. Preserve that native isolated
                # stream convention (including empty-string vs empty-list).
                reference_row = isolated if row["source"] == "rpval" else row
                reference_removed = removed if row["source"] == "rpval" else expected
                if executor.render_latent_prompt(tokenizer, isolated, removed) != native_render(tokenizer, reference_row, reference_removed):
                    raise ValueError(f"Native isolated deletion mismatch: {row['sample_id']}")


def attention_capture(family):
    """Intercept the family's eager function, preserving its original numerics."""
    family = "qwen3_5" if family == "qwen3_5_text" else family
    implementation = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    if not hasattr(implementation, "eager_attention_forward"):
        raise UnsupportedModel(f"Installed {family} implementation lacks eager_attention_forward")

    @contextmanager
    def capture(indices, sink):
        original = implementation.eager_attention_forward

        def wrapped(module, *args, **kwargs):
            output, weights = original(module, *args, **kwargs)
            if getattr(module, "layer_idx", None) in indices:
                if weights is None or weights.ndim != 4:
                    raise UnsupportedModel("Eager capture did not return [batch,heads,query,key] weights")
                sink[module.layer_idx] = weights.detach()
            return output, weights

        implementation.eager_attention_forward = wrapped
        try:
            yield
        finally:
            implementation.eager_attention_forward = original
    return capture


def executor_args(args, spec, output):
    return ["compass_matrix_generate", "--input", str(args.input), "--predictions", str(args.predictions),
            "--model", spec["path"], "--output", str(output), "--split", args.split,
            "--memory-setting", "all", "--policy-source", "latent", "--numeric-prediction-coefficients",
            "--layer", str(spec["layer"]), "--alpha", format(args.alpha, 'g'), "--aggregation", "sum",
            "--counterfactual-mode", "leave-one-out", "--direction-context", "isolated",
            "--transport-site", "final-norm", "--norm-mode", "none", "--readout-precision", "fp32",
            "--direction-refresh", "token", "--attention-control", "steer", "--attention-backend", "eager",
            "--max-new-tokens", str(args.max_new_tokens), "--temperature", "0", "--top-p", "1",
            "--seed", "42", "--batch-size", str(args.batch_size), "--device", args.device]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Backbone alias or local model directory")
    p.add_argument("--benchmark", type=str.lower, choices=DATA, required=True)
    p.add_argument("--condition", choices=CONDITIONS, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    output = p.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path, help="Exact append-only generation JSONL path")
    output.add_argument("--output-root", type=Path, help="ROOT/MODEL/BENCH/REFERENCE/CONDITION.jsonl; REFERENCE inferred from predictions parent")
    p.add_argument("--input", type=Path, help="Override native unified dataset")
    p.add_argument("--split", choices=("all", "heldout"), help="Default: heldout for reader evaluation.jsonl; legacy RPEval full source uses all")
    p.add_argument("--layer", type=int, help="Attention measurement block; never changes final-norm transport")
    p.add_argument("--alpha", type=float, default=1.0, help="Global intervention strength; default preserves existing runs")
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--device", default="auto", help="Device map: auto, cpu, or cuda:0")
    p.add_argument("--limit", type=int)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--validate", "--dry-run", dest="dry_run", action="store_true", help="CPU-only config/input/coefficient validation; no torch import, model load or device access")
    mode.add_argument("--audit-only", action="store_true", help="Also audit native prompts with the local tokenizer; no weights")
    a = p.parse_args(argv)
    import math
    if not math.isfinite(a.alpha) or a.alpha < 0:
        p.error("alpha must be finite and nonnegative")
    a.predictions = a.predictions.resolve()
    sibling = a.predictions.parent / "evaluation.jsonl"
    a.input = (a.input or (sibling if sibling.exists() else DATA[a.benchmark])).resolve()
    a.split = a.split or ("all" if a.benchmark == "rpeval" and a.input == DATA["rpeval"].resolve() else "heldout")
    a.max_new_tokens = a.max_new_tokens if a.max_new_tokens is not None else (2048 if a.benchmark == "steem" else 1024)
    a.batch_size = a.batch_size if a.batch_size is not None else (2 if a.benchmark == "rpeval" else 1)
    if a.max_new_tokens < 1 or a.batch_size < 1 or (a.limit is not None and a.limit < 1):
        p.error("token cap, batch size and limit must be positive")
    return a


def main(argv=None):
    args = parse_args(argv)
    spec = model_spec(args.model, args.layer)
    rows = [prompt_fields(row) for row in select_rows(
        read_jsonl(args.input), read_jsonl(args.predictions), args.split, args.limit)]
    expected_source = "rpval" if args.benchmark == "rpeval" else args.benchmark
    if any(r.get("source") != expected_source for r in rows):
        raise ValueError("Input source does not match requested benchmark")
    reference = args.predictions.parent.name
    if args.output is not None:
        output = args.output.resolve()
    else:
        if reference not in {"benchmark", "independent"}:
            raise ValueError("--output-root requires predictions under benchmark/ or independent/; otherwise use --output PATH")
        output = args.output_root.resolve() / spec["name"] / args.benchmark / reference / (args.condition + ".jsonl")
    command = executor_args(args, spec, output)
    metadata = dict(model=spec, benchmark=args.benchmark, reference=reference, condition=args.condition,
                    input=str(args.input), predictions=str(args.predictions), split=args.split,
                    sample_ids=[r["sample_id"] for r in rows], executor_args=command[1:],
                    effective_direction_refresh="prefill" if args.condition.startswith("fixed_") else "token",
                    effective_gate="dac_full_vocab" if args.condition.endswith("dac_full_vocab") else "identity" if args.condition.endswith("no_gate") else "bounded_attention",
                    shared_generated_prefix=True, coefficient_source="supplied frozen numeric predictions",
                    scope="full-data development" if args.split == "all" else "prediction-declared heldout",
                    input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    predictions_sha256=hashlib.sha256(args.predictions.read_bytes()).hexdigest())
    print(json.dumps(dict(status="planned", output=str(output), samples=len(rows), **{k: v for k, v in metadata.items() if k != "sample_ids"}), indent=2), flush=True)
    if args.dry_run:
        return
    if spec["family"] == "qwen3_5_text":
        configure_hybrid_runtime()
    executor = make_executor(args.condition)
    if spec["family"] == "qwen3_5_text":
        install_hybrid_loader(executor, spec)
    native_render = install_prompt_adapter(executor, args.benchmark, rows)
    executor.capture_selected_attention = attention_capture(spec["family"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec["path"], use_fast=True, local_files_only=True)
    audit_prompts(executor, tokenizer, rows, native_render)
    print(f"PROMPT_AUDIT_OK {args.benchmark} {len(rows)}", flush=True)
    if args.audit_only:
        return
    # Preserve append-only resume semantics and reject changed reader/settings.
    import fcntl
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(output) + ".adapter.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = Path(str(output) + ".method.json")
        if manifest.exists():
            if json.loads(manifest.read_text()) != metadata:
                raise ValueError("Existing output has different settings/reader; choose a new output root")
        else:
            if output.exists() and output.stat().st_size:
                raise ValueError("Existing output lacks COMPASS metadata; choose a new output root")
            manifest.write_text(json.dumps(metadata, indent=2) + "\n")
        original_load = executor.load_jsonl
        executor.load_jsonl = lambda path: rows if Path(path).resolve() == args.input else original_load(path)
        original_append = executor.append_jsonl

        def append(path, records):
            for record in records:
                record.update(compass_condition=args.condition, crossbench_condition=args.condition,
                              benchmark=args.benchmark, model=spec["name"],
                              effective_direction_refresh=metadata["effective_direction_refresh"],
                              effective_gate=metadata["effective_gate"], shared_generated_prefix=True)
            return original_append(path, records)

        executor.append_jsonl = append
        previous = sys.argv
        try:
            sys.argv = command
            executor.main()
        finally:
            sys.argv = previous


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        print(json.dumps(dict(status="unsupported" if isinstance(exc, UnsupportedModel) else "error", reason=str(exc))), file=sys.stderr)
        sys.exit(2)
