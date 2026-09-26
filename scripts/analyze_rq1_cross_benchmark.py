#!/usr/bin/env python3
"""Full held-out RQ1 readout with native labels and benchmark-specific groups.

CPU only. Does not generate answers or call judges. Keeps historical RPEval
reports untouched. Repeated splits measure sensitivity, not independent CIs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from build_rq1_reference_split import build_manifest, stable_rank


CONFIG = {
    "rpeval": ("rpval_implicit", ("ignore", "support", "dominate")),
    "benchpres": ("benchpres", ("not_apply", "apply")),
    "steem": ("steem_tag_masked", ("1", "2", "3", "4", "5")),
}


def records_for(rows, benchmark):
    records = []
    for row in rows:
        group = row["metadata"]["query_id"] if benchmark == "steem" else row["sample_id"]
        regime = ("single" if len(row["memories"]) == 1 else "multi") if benchmark == "rpeval" else "all"
        for memory in row["memories"]:
            label = memory["gold_policy"]
            if benchmark == "benchpres":
                if label not in ("ignore", "dominate"):
                    raise ValueError(f"Unexpected BenchPreS label {label}")
                label = "apply" if label == "dominate" else "not_apply"
            elif benchmark == "steem":
                label = str(int(row["metadata"]["gold_level"]))
                if float(memory["gold_score"]) != int(label):
                    raise ValueError("SteeM level metadata mismatch")
            if label not in CONFIG[benchmark][1]:
                raise ValueError(f"Unexpected native label {label}")
            records.append(dict(sample_id=row["sample_id"], memory_id=memory["memory_id"],
                                group_id=str(group), regime=regime, label=label,
                                original_policy=memory["gold_policy"]))
    keys = [(r["sample_id"], r["memory_id"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate memory keys in source data")
    if benchmark == "steem":
        groups = defaultdict(list)
        for r in records:
            groups[r["group_id"]].append(r["label"])
        if any(sorted(labels) != list(CONFIG[benchmark][1]) for labels in groups.values()):
            raise ValueError("SteeM must have five complete native levels per query group")
    return records


def make_split(rows, records, benchmark, fraction, seed):
    if benchmark == "rpeval":
        return build_manifest(rows, fraction, seed)
    labels_by_group = defaultdict(set)
    for r in records:
        labels_by_group[r["group_id"]].add(r["label"])
    ordered = sorted(labels_by_group, key=lambda g: stable_rank(seed, g))
    count = round(len(ordered) * fraction)
    chosen, covered = [], Counter()
    # Label-only stratification ensures leave-one-group-out calibration can
    # always fit every native class. No activation or held-out score is used.
    while any(covered[c] < 2 for c in CONFIG[benchmark][1]):
        candidates = [g for g in ordered if g not in chosen]
        if not candidates:
            raise ValueError("Insufficient groups for native class coverage")
        gain = lambda g: sum(covered[c] < 2 for c in labels_by_group[g])
        best = max(candidates, key=gain)
        if gain(best) == 0:
            raise ValueError("Cannot cover all classes in two groups")
        chosen.append(best)
        covered.update(labels_by_group[best])
    if len(chosen) > count or count >= len(ordered):
        raise ValueError("Requested calibration fraction cannot support group CV")
    for group in ordered:
        if len(chosen) == count:
            break
        if group not in chosen:
            chosen.append(group)
    reference = sorted(chosen)
    return dict(group_key="metadata.query_id" if benchmark == "steem" else "sample_id",
                selection=dict(fraction=fraction, seed=seed, rounding="nearest integer group",
                               minimum_groups_per_class=2, uses_activations=False),
                n_total_groups=len(ordered), n_reference_groups=len(reference),
                n_final_test_groups=len(ordered)-len(reference),
                reference_group_ids=reference,
                final_test_group_ids=sorted(set(ordered)-set(reference)))


def split_indices(records, manifest):
    ref, test = set(manifest["reference_group_ids"]), set(manifest["final_test_group_ids"])
    groups = {r["group_id"] for r in records}
    if ref & test or ref | test != groups:
        raise ValueError("Split overlap or incomplete group coverage")
    return ([i for i, r in enumerate(records) if r["group_id"] in ref],
            [i for i, r in enumerate(records) if r["group_id"] in test])


def validate_artifact(artifact, records, model):
    expected = {(r["sample_id"], r["memory_id"]): r for r in records}
    metadata = artifact["metadata"]
    keys = [(r["sample_id"], r["memory_id"]) for r in metadata]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError("Artifact does not contain exactly all source memories")
    aligned = [expected[key] for key in keys]
    for meta, record in zip(metadata, aligned):
        if "group_id" in meta and str(meta["group_id"]) != record["group_id"]:
            raise ValueError("Artifact group ID differs from source")
        if "gold_policy" in meta and meta["gold_policy"] != record["original_policy"]:
            raise ValueError("Artifact policy differs from source")
    x = artifact["signatures"]
    if x.ndim != 3 or x.shape[0] != len(records) or not torch.isfinite(x).all():
        raise ValueError("Invalid activation tensor")
    if Path(artifact["model"]).name != model or "h(full)-h(without-memory-i)" not in artifact["definition"]:
        raise ValueError("Wrong backbone or non-LOO artifact")
    if model in ("Qwen3-4B", "Qwen3-8B") and x.shape[1] != 36:
        raise ValueError("Missing Qwen3 layers")
    return aligned


def predict(x, labels, regimes, train, query, class_count):
    """Exactly the legacy calibration-centered spherical centroid estimator."""
    center = x[train].mean(dim=0)
    bank = F.normalize(x[train] - center, dim=-1)
    queries = F.normalize(x[query] - center, dim=-1)
    result = torch.empty((len(query), x.shape[1]), dtype=torch.long)
    for regime in sorted(set(regimes)):
        query_pos = [p for p, i in enumerate(query) if regimes[i] == regime]
        if not query_pos:
            continue
        centroids = []
        for label in range(class_count):
            local = [p for p, i in enumerate(train) if regimes[i] == regime and labels[i] == label]
            if not local:
                raise ValueError(f"Calibration fold lacks {regime}/{label}")
            centroids.append(F.normalize(bank[local].mean(dim=0), dim=-1))
        scores = torch.einsum("nld,cld->nlc", queries[query_pos], torch.stack(centroids))
        result[query_pos] = scores.argmax(dim=-1)
    return result


def metrics(prediction, gold, classes, ordinal=False):
    recalls = {}
    for c, label in enumerate(classes):
        mask = gold == c
        recalls[label] = (prediction[mask] == c).float().mean(dim=0).tolist() if mask.any() else None
    present = [v for v in recalls.values() if v is not None]
    result = dict(accuracy=(prediction == gold[:, None]).float().mean(dim=0).tolist(),
                  macro_recall=np.mean(present, axis=0).tolist(), recall=recalls,
                  class_counts={label: int((gold == c).sum()) for c, label in enumerate(classes)})
    if ordinal:
        result["readout_level_mae"] = (prediction - gold[:, None]).abs().float().mean(dim=0).tolist()
    return result


def analyze(path, model, benchmark, rows, manifests, output, threads):
    torch.set_num_threads(threads)
    artifact = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    records = validate_artifact(artifact, records_for(rows, benchmark), model)
    classes = CONFIG[benchmark][1]
    labels = torch.tensor([classes.index(r["label"]) for r in records])
    regimes = [r["regime"] for r in records]
    x = artifact["signatures"].float()
    ref, test = split_indices(records, manifests[0])
    pred = predict(x, labels, regimes, ref, test, len(classes))
    curves = metrics(pred, labels[test], classes, benchmark == "steem")
    cv = torch.empty((len(ref), x.shape[1]), dtype=torch.long)
    for group in manifests[0]["reference_group_ids"]:
        train = [i for i in ref if records[i]["group_id"] != group]
        positions = [p for p, i in enumerate(ref) if records[i]["group_id"] == group]
        query = [ref[p] for p in positions]
        cv[positions] = predict(x, labels, regimes, train, query, len(classes))
    cv_metrics = metrics(cv, labels[ref], classes, benchmark == "steem")
    selected = int(np.argmax(cv_metrics["macro_recall"]))
    peak = int(np.argmax(curves["macro_recall"]))
    regime_curves = {}
    for regime in sorted(set(regimes)):
        positions = [p for p, i in enumerate(test) if regimes[i] == regime]
        regime_curves[regime] = metrics(pred[positions], labels[test][positions], classes, benchmark == "steem")
    resamples = []
    for manifest in manifests[1:]:
        a, b = split_indices(records, manifest)
        p = predict(x, labels, regimes, a, b, len(classes))
        m = metrics(p, labels[b], classes, benchmark == "steem")
        resamples.append(dict(seed=manifest["selection"]["seed"], curves=m,
                              descriptive_peak_layer=int(np.argmax(m["macro_recall"]))+1))
        print(f"  {benchmark}/{model} resample {len(resamples)}/{len(manifests)-1}", flush=True)
    report = dict(benchmark=benchmark, model=model, artifact=str(path),
                  artifact_definition=artifact["definition"], native_classes=classes,
                  audit=dict(exact_source_memory_coverage=True, unique_memory_keys=True,
                             all_values_finite=True, shape=list(x.shape),
                             calibration_groups=len(manifests[0]["reference_group_ids"]),
                             test_groups=len(manifests[0]["final_test_group_ids"]),
                             calibration_memory_rows=len(ref), test_memory_rows=len(test)),
                  curves=curves, regime_curves=regime_curves, calibration_cv=cv_metrics,
                  calibration_selected_layer=selected+1,
                  selected_layer_test_macro_recall=curves["macro_recall"][selected],
                  selected_layer_test_accuracy=curves["accuracy"][selected],
                  descriptive_peak_layer=peak+1, resamples=resamples)
    with (output / f"{benchmark}_{model}_heldout_predictions.jsonl").open("w") as f:
        for p, i in enumerate(test):
            row = {k: records[i][k] for k in ("sample_id", "memory_id", "group_id", "regime", "label")}
            row.update(predicted_labels_by_layer=[classes[c] for c in pred[p].tolist()],
                       calibration_selected_layer=selected+1)
            f.write(json.dumps(row) + "\n")
    return report


def plots(reports, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), constrained_layout=True)
    for ax, benchmark in zip(axes, CONFIG):
        classes = CONFIG[benchmark][1]
        for report in reports:
            if report["benchmark"] != benchmark:
                continue
            y = report["curves"]["macro_recall"]
            line, = ax.plot(range(1, len(y)+1), y, label=report["model"])
            layer = report["calibration_selected_layer"]
            ax.scatter([layer], [y[layer-1]], color=line.get_color(), marker="s", s=24)
        ax.axhline(1/len(classes), color="gray", ls=":", lw=1)
        ax.set(title=f"{benchmark} ({len(classes)} classes)", xlabel="Layer (1-indexed)",
               ylabel="Held-out macro recall", ylim=(0, 1))
        ax.legend(fontsize=8)
    fig.savefig(output / "layer_readout_curves.pdf")
    fig.savefig(output / "layer_readout_curves.png", dpi=180)
    plt.close(fig)
    models = list(dict.fromkeys(r["model"] for r in reports))
    fig, axes = plt.subplots(len(models), 3, figsize=(12, 2.5*len(models)),
                             squeeze=False, constrained_layout=True)
    for report in reports:
        ax = axes[models.index(report["model"]), list(CONFIG).index(report["benchmark"])]
        labels = report["native_classes"]
        matrix = np.asarray([report["curves"]["recall"][c] for c in labels])
        im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis",
                       extent=(.5, matrix.shape[1]+.5, len(labels)-.5, -.5))
        ax.set_yticks(range(len(labels)), labels)
        ax.set(title=f"{report['benchmark']} / {report['model']}", xlabel="Layer")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Per-class recall", shrink=.7)
    fig.savefig(output / "class_recall_profiles.png", dpi=180)
    fig.savefig(output / "class_recall_profiles.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), constrained_layout=True)
    for ax, benchmark in zip(axes, CONFIG):
        for report in reports:
            if report["benchmark"] != benchmark or not report["resamples"]:
                continue
            curves = np.asarray([r["curves"]["macro_recall"] for r in report["resamples"]])
            depths = np.arange(1, curves.shape[1]+1)
            line, = ax.plot(depths, curves.mean(axis=0), label=report["model"])
            ax.fill_between(depths, curves.min(axis=0), curves.max(axis=0),
                            color=line.get_color(), alpha=.13)
        ax.axhline(1/len(CONFIG[benchmark][1]), color="gray", ls=":", lw=1)
        ax.set(title=benchmark, xlabel="Layer", ylabel="Macro recall (split mean and range)", ylim=(0, 1))
        ax.legend(fontsize=8)
    fig.savefig(output / "split_stability.png", dpi=180)
    fig.savefig(output / "split_stability.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", default="runs/latent_mccs/rq1_cross_benchmark")
    parser.add_argument("--models", default="Qwen3-4B,Qwen3-8B")
    parser.add_argument("--benchmarks", default=",".join(CONFIG))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=.05)
    parser.add_argument("--resamples", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--plot-only", action="store_true", help="Render existing report in a matplotlib-enabled environment")
    parser.add_argument("--aligned-compass-splits", action="store_true", help="Use exact frozen COMPASS calibration groups, not a new fractional split")
    args = parser.parse_args()
    if not 0 < args.fraction < 1 or args.resamples < 0 or args.torch_threads < 1:
        parser.error("Require 0 < fraction < 1, resamples >= 0, and threads >= 1")
    output = args.root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plots(json.loads((output / "report.json").read_text())["reports"], output)
        return
    reports = []
    for benchmark in args.benchmarks.split(","):
        dataset, _ = CONFIG[benchmark]
        source = args.root / "data/unified" / f"{dataset}_unified.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        records = records_for(rows, benchmark)
        seeds = [args.seed] + [args.seed+1009*(i+1) for i in range(args.resamples)]
        manifests = [make_split(rows, records, benchmark, args.fraction, seed) for seed in seeds]
        if args.aligned_compass_splits:
            source_manifest=args.root/'runs/compass_matrix_20260922/Qwen3-8B'/benchmark/'benchmark/reader_manifest.json'
            calibration=json.loads(source_manifest.read_text())['splits'][benchmark]['calibration_groups']
            all_groups={r['group_id'] for r in records}
            assert set(calibration)<all_groups
            manifests=[dict(group_key='metadata.query_id' if benchmark=='steem' else 'sample_id',
                selection=dict(source=str(source_manifest),seed=None,rule='Exact COMPASS calibration IDs'),
                reference_group_ids=sorted(calibration),final_test_group_ids=sorted(all_groups-set(calibration)),
                n_total_groups=len(all_groups),n_reference_groups=len(calibration),n_final_test_groups=len(all_groups)-len(calibration))]
        (output / f"{benchmark}_splits.json").write_text(json.dumps(dict(
            source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            manifests=manifests), indent=2)+"\n")
        artifact_prefix = "rpeval" if benchmark == "rpeval" else dataset
        for model in args.models.split(","):
            path = args.root / "runs/latent_mccs" / model / f"{artifact_prefix}_counterfactual_signatures.pt"
            print(f"[analyze] {benchmark}/{model}", flush=True)
            report = analyze(path, model, benchmark, rows, manifests, output, args.torch_threads)
            (output / f"{benchmark}_{model}_report.json").write_text(json.dumps(report, indent=2)+"\n")
            reports.append(report)
    protocol = dict(readout="calibration-centered spherical class centroids; RPEval setting-specific",
                    layer_selection="calibration-only leave-one-complete-group-out macro recall; shallowest tie",
                    test_peaks="descriptive only; never used for model selection",
                    resampling="10 additional group splits by default; not independent confidence intervals",
                    benchmarks="native 3/2/5-class labels; no pooled cross-benchmark accuracy",
                    extraction_caveat="uses existing last-prompt-token full-minus-LOO artifacts; no new extraction",
                    steem_caveat="tag masked, but natural-language reliance instruction retained",
                    benchpres_caveat="prompt-group-disjoint, not task-disjoint",
                    last_layer_caveat="HF hidden_states final entry may include final normalization; not a uniform raw-block hook")
    (output / "report.json").write_text(json.dumps(dict(protocol=protocol, reports=reports), indent=2)+"\n")
    try:
        plots(reports, output)
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        print("Analysis complete. Render figures with --plot-only in an environment with matplotlib.", flush=True)
    print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
