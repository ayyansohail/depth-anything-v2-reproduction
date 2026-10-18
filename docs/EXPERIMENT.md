# Stage 4 experiment specification

**Type F: computational analysis.** Branch: `feat/computational-analysis-experiment`.

How much relative-depth accuracy does V2 gain over V1 at matching model sizes,
what inference cost accompanies that gain, and how does the accuracy-cost
tradeoff change across V2 Small, Base, and Large?

See [the experiment guide](../experiment/README.md) for the fixed A/B/B/A GPU
protocol, inherited versus new evidence, Colab workflow, analysis and limitations.
The executable notebook is `notebooks/stage4_computational_analysis.ipynb`.

This replaces the earlier Type C scene-concentration proposal as the primary
Stage 4 experiment. The older `scripts/05_analyze_da2k.py`, scene plots and
pair-level bootstrap outputs remain historical supporting artifacts. They are
not the new experiment's image-cluster uncertainty estimates and are not used
to claim that the training-data change caused an accuracy improvement.

Status: experimental execution and evidence review complete. Colab passed 22 tests and validated 24 blocks / 1,200 timings. See [reviewed findings](../experiment/FINDINGS.md).
No training, fine-tuning, new accuracy evaluation or CPU rerun is planned.
