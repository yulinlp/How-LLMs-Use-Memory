<div align="center">

<img src="assets/memory-use-logo-ai.png" width="150" alt="Memory use">

## How Do Language Models Use Memory?<br>From Internal Strategy Readout to Adaptive Control

![ICLR 2027](https://img.shields.io/badge/ICLR_2027-Under_review-315EA8)
[![GitHub stars](https://img.shields.io/github/stars/yulinlp/How-LLMs-Use-Memory?style=social)](https://github.com/yulinlp/How-LLMs-Use-Memory)
[![Last commit](https://img.shields.io/github/last-commit/yulinlp/How-LLMs-Use-Memory)](https://github.com/yulinlp/How-LLMs-Use-Memory/commits)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#installation)

**[English](README.md) | [中文](README_zh.md)**

[Overview](#overview) · [Installation](#installation) · [Experiments](#experiments)

</div>

This repository contains the experiments and code accompanying our paper.

<a id="overview"></a>

## 💡 Overview

Memory-augmented assistants need to decide not only which memories to retrieve,
but also how to use them. We study this second problem through model activations:
can we read out a memory's usage strategy, and can we use these signals to control
the response?

We investigate three research questions:

- **RQ1 — Strategy readout.** Activation differences predict memory-use strategies. On Qwen3-4B and Qwen3-8B, extracting each memory separately improves readout over joint-context extraction.
- **RQ2 — Direct control.** Fixed-direction interventions can improve memory use, but mixed requirements remain difficult to satisfy together. Stronger interventions can also cause repetitive generation.
- **RQ3 — Adaptive control.** These findings motivate **Duet**, which updates each memory's intervention direction during decoding and adjusts its strength using output-distribution sensitivity, without updating model parameters.

<p align="center">
  <img src="assets/research-overview.png" width="100%" alt="Paper overview: strategy readout, direct intervention, and adaptive memory control on Qwen3-4B">
</p>

We evaluate on **RPEval, BenchPreS, and SteeM**, covering discrete usage strategies,
preference applicability, and graded memory reliance. The six backbones are
Qwen3-4B, Qwen3-8B, Qwen3.5-4B, Qwen3.5-9B, Llama-3.1-8B-Instruct, and
Llama-3.2-3B-Instruct.

<a id="installation"></a>

## 🔧 Installation

Run from the repository root:

```bash
git clone https://github.com/yulinlp/How-LLMs-Use-Memory.git
cd How-LLMs-Use-Memory
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduce.txt
python -m pip install -e .
```

Use a CUDA-compatible PyTorch build for benchmark generation. The reproduction
environment pins PyTorch and Transformers; model weights are downloaded separately.

<a id="experiments"></a>

## 🧪 Experiments

Use [reproduce.py](reproduce.py) for **RQ1, RQ2, and Duet's main results**.
All commands below run from the repository root.

### 1. Prepare the data

Obtain the upstream repositories at the versions used in our experiments.
Skip cloning a repository if it is already present.

```bash
git clone https://github.com/XueyangFeng/RPEval.git RPEval
git -C RPEval checkout --detach 0f8bce80a03e6b86426485264b14ae7de49cc8f9
git clone https://github.com/Moore-Tian/SteeM-Memory-Control.git SteeM-Memory-Control
git -C SteeM-Memory-Control checkout --detach fd3e20057a3c6359e1399cbc0258903803ee30d1

python reproduce.py prepare-data
python reproduce.py check-data
```

BenchPreS is loaded from `sangyon/BenchPreS` on Hugging Face. For an offline
download, pass `--benchpres-parquet /path/to/test.parquet` to `prepare-data`.
The script verifies upstream files and the converted inputs against the paper
hashes before saving them to `data/paper/`; it does not silently accept a changed dataset.

| Benchmark | Prepared samples | Calibration groups | Main evaluation samples |
| :--- | ---: | ---: | ---: |
| RPEval | 300 | 15 | 285 |
| BenchPreS | 390 | 20 | 370 |
| SteeM | 1,000 | 10 | 950 |

Splits preserve whole query groups, including SteeM's five reliance levels.
RQ2 uses all 300 RPEval queries. Frozen IDs are in [paper_splits.json](configs/paper_splits.json).

### 2. Select a backbone

```bash
hf download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
```

Use `--model Qwen3-4B` for a checkpoint under `models/`, or pass its local path.
Set `DUET_MODEL_DIR` to use a different model directory. The same commands accept
the other five backbones listed above.

### 3. Run the experiments

**RQ1 — layer-wise strategy readout.** Extract activations, select the layer on
calibration groups, and evaluate on the remaining groups:

```bash
CUDA_VISIBLE_DEVICES=0 python reproduce.py rq1 \
  --model Qwen3-4B --benchmark rpeval
```

Outputs include activation tensors, per-layer predictions, readout metrics, and
a layer-wise plot under `runs/reproduction/rq1/`. Use `--benchmark benchpres` or
`--benchmark steem` for the other datasets. RPEval also saves paired isolated-readout results.

**RQ2 — direct intervention.** Compute oracle-guided directions once at prefill
and apply them at every generation step:

```bash
CUDA_VISIBLE_DEVICES=0 python reproduce.py rq2 \
  --model Qwen3-4B --alpha 1
```

For the strength sweep, repeat with `--alpha` set to
`0.25`, `0.5`, `0.75`, `1`, `1.5`, and `2`. Setting `--alpha 0` gives the
matched no-intervention control.

**Duet — main experiments.** Build the reference reader, predict usage
coefficients, and generate with dynamic directions and adaptive strengths:

```bash
CUDA_VISIBLE_DEVICES=0 python reproduce.py main \
  --model Qwen3-4B --benchmark rpeval --alpha 1
```

Repeat for `benchpres` and `steem`. Add `--plan` to inspect commands without loading
weights. `--limit 2 --max-new-tokens 32` creates a separate generation smoke run.
Existing compatible outputs are resumed; changed inputs or settings require a new `--output` directory.

### 4. Evaluate

Evaluation uses the native benchmark prompts with **Qwen3.8-27B**. Supply an
SGLang endpoint serving `Qwen/Qwen3.8-27B`, with `qwen3` reasoning parsing and
deterministic inference enabled. Set `DUET_JUDGE_API_KEY` if authentication is required.

To score while generating, append these arguments to a `main` or `rq2` command:

```bash
--judge-endpoint http://localhost:8000/v1=8 --concurrency 8
```

Repeat `--judge-endpoint` to distribute requests across services. The runner
waits for scoring and writes metrics after generation completes.

<details>
<summary><b>Score saved responses or retry incomplete judgments</b></summary>

```bash
python reproduce.py judge --benchmark rpeval \
  --input runs/reproduction/main/Qwen3-4B/rpeval/alpha1.jsonl \
  --output runs/reproduction/main/Qwen3-4B/rpeval/alpha1-judge \
  --judge-endpoint http://localhost:8000/v1=8 --concurrency 8

python reproduce.py summarize --benchmark rpeval \
  --input runs/reproduction/main/Qwen3-4B/rpeval/alpha1.jsonl \
  --judge-dir runs/reproduction/main/Qwen3-4B/rpeval/alpha1-judge \
  --output runs/reproduction/main/Qwen3-4B/rpeval/alpha1-metrics.json
```

Judging starts with thinking enabled. Failed or invalid requests receive a
non-thinking retry under the same rubric; each attempt records its settings.
Rerunning the command preserves valid judgments and retries unresolved ones.
Summaries report incomplete coverage explicitly, together with Single, Multi,
All Ignore, Mixed, and repetition statistics for RPEval.

</details>

### Validation

```bash
python -m unittest discover -s tests -p test_reproduction.py -v
python scripts/check_reproduction.py
python scripts/compass_matrix_judge.py --self-test
```

Validation uses a fresh Python environment and an isolated copy of the required
source files. CPU tests cover random tiny-model readout and decoding, all three
input formats, RQ1 analysis, resume behavior, and mocked judge requests. Prepared
benchmark inputs match the paper hashes. These checks do not constitute a new
full-size GPU reproduction of the paper scores. The commands generate Duet and
RQ2 controls, not the five prompting baselines in the main table.

## 📁 File Structure

```text
assets/        README images
configs/       Frozen inputs, upstream versions, and calibration IDs
reproduce.py   Experiment entry point
steem_adapt/   Data processing, prompts, and model interventions
scripts/       Experiment runners, evaluation, and plotting
tests/         Tests
```

## Acknowledgments

Our implementation builds on a modified version of
[EasySteer](https://github.com/ZJU-REAL/EasySteer). The adaptive-strength component
draws on [Dynamic Activation Composition](https://aclanthology.org/2024.blackboxnlp-1.34/).
We thank the authors of the models and benchmarks used in this work.

## License

Original project code is licensed under [Apache-2.0](LICENSE).
Third-party code, datasets and model weights retain their respective licenses;
see [third-party notices](THIRD_PARTY_NOTICES.md).
