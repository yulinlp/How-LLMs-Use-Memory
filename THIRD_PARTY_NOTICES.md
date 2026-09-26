# Third-party acknowledgments and licenses

The implementation builds on EasySteer and related activation-control work.
Third-party notices are retained separately from the license for this project's
original contributions.

| Source | Role | Upstream license |
| :--- | :--- | :--- |
| [EasySteer](https://github.com/ZJU-REAL/EasySteer) | Activation-control implementation reference | Apache-2.0 |
| [AdaSteer](https://github.com/MuyuenLP/AdaSteer) | Adaptive-hook implementation reference | Apache-2.0 |
| [RPEval](https://github.com/XueyangFeng/RPEval) | Benchmark code and native evaluation prompts | MIT for code; CC BY-NC 4.0 for data |
| [SteeM](https://github.com/Moore-Tian/SteeM-Memory-Control) | Benchmark preparation and native evaluation rubrics | MIT |
| [BenchPreS](https://huggingface.co/datasets/sangyon/BenchPreS) | Benchmark downloaded during preparation | See the upstream dataset terms |

Copies of the available upstream license files accompany the release under
`third_party_licenses/`. Dataset files and model weights are not bundled.
Their respective terms continue to apply when downloaded. Benchmark preparation
adapts input formats and preserves the pinned source versions and evaluation
prompts recorded in `configs/`.

The strength-adaptation design draws on
[Dynamic Activation Composition](https://aclanthology.org/2024.blackboxnlp-1.34/).
