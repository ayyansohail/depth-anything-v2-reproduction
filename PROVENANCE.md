# PROVENANCE

`OURS` = team-maintained implementation (AI assistance disclosed below); `ADAPTED` = modified upstream code; `REUSED` = upstream code or weights used unchanged; `PAPER` = published reference; `RESULT` = output of the team's runs. These labels describe origin, not proof of correctness.

## Approved scope and assistance

The instructor approved the proposal's pretrained-checkpoint evaluation scope, as confirmed by the team. No training or fine-tuning is claimed. Scope approval does not imply approval of subsequent dataset or experiment changes.

The initial scaffold was AI-assisted (the initial Git record names Claude). Later development and documentation also used AI assistance. The team's existing disclosure states that retained files were reviewed and understood by the team; every member remains responsible for explaining their contribution. This Stage 3 cleanup used Codex for documentation, provenance, and reference-table edits. It did not rerun inference or change evaluation logic.

## Code

| Component | Status | Source and limits |
|---|---|---|
| V1/V2 S/B/L weights and processors | REUSED | Hugging Face model IDs are listed in `src/dav2/models.py`. `from_pretrained()` has no explicit revision, so the exact historical model/processor revisions are not archived. |
| DINOv2/DPT architecture implementation | REUSED | Loaded through the installed `transformers` package. The official [Depth Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2) is a reference, not a vendored implementation. Historical upstream commit and installed package versions are not established by this checkout. |
| `src/dav2/models.py` | OURS (AI-assisted wrapper) | Bilinear output upsampling introduced in local commit `d9b8c0e05e6b380fbefc5777aadb356172896534`. `align_corners=False`; the protocol documents the difference from the official implementation. |
| `src/dav2/metrics.py` | OURS (AI-assisted) | Local NumPy implementation of disparity-space least-squares alignment and depth metrics. Exact parity with the authors' crop, mask, clamp, and alignment routine has not been established. See `docs/EVAL_PROTOCOL.md`. |
| `src/dav2/data.py` | OURS (AI-assisted) | Local dataset readers and subset selection. DA-2K coordinates follow the official [annotation description](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/DA-2K.md): row/column coordinates and point1 nearer. |
| `scripts/*.py`, `tests/` | OURS (AI-assisted) | Evaluation, analysis, diagnostics, and tests; initial scaffold recorded at `aeca79fc675a671faab323c89176c82285a2e17c`. |
| `app/streamlit_app.py` | OURS (AI-assisted) | App wrapper; source presence alone does not establish a working deployment. |

## Data

| Dataset | Source | Recorded use and limits |
|---|---|---|
| NYU Depth V2 | [Ultralytics repackaging](https://github.com/ultralytics/assets/releases/download/v0.0.0/nyu-depth.zip); original dataset: Silberman et al. 2012 | Saved evaluation covers 654 images per model. Exact historical file hashes and equivalence to the authors' test files were not archived. |
| DA-2K | [Official Hugging Face dataset](https://huggingface.co/datasets/depth-anything/DA-2K) | Saved pair records cover 1,033 unique images and 2,068 pairs per model. Dataset revision and annotation checksum for the historical run are not archived. |
| KITTI | [Dataset source](https://www.cvlibs.net/datasets/kitti/) | Proposed originally; no KITTI evaluation result is included. DA-2K provides ordinal evaluation, not a substitute outdoor dense-depth benchmark. |

## Results and references

| Artifact | Origin | Traceability |
|---|---|---|
| Final NYU per-image and summary CSVs | RESULT | Cropped results recorded in commit `74aa4e1d526cb140a5607e7ddd4a6e43fd7b815b`; this is a results-recording commit, not a proven execution-time code revision. |
| DA-2K CSVs | RESULT | Full-scale results recorded in commit `c9042dd0b844d3f4722f1cb45af7b2be693fe515`. Historical execution-time revision and environment remain unrecorded. |
| `results/latency.csv` | RESULT | Reports Tesla T4 hardware; script measures the bare forward pass at a fixed input size, excluding preprocessing and output resizing. |
| `results/paper_reported.csv` | PAPER | V2 references verified against Tables 2 and 3 of [arXiv v2](https://arxiv.org/html/2406.09414v2). Accuracy percentages are stored as fractions. |
| `results/paper_vs_ours.csv` | MIXED (PAPER + RESULT) | References joined to saved results; `gap = ours - paper_value`. Not a fresh evaluation run. |
| `figures/` | DERIVED | Plots from saved result CSVs; not separate experimental measurements. |

Table 3 gives a single V1 baseline of 88.5% without identifying its encoder in that table. V1 references remain blank pending an unambiguous model-specific source; the report's V1-Base reference of 89.4% was not verified. Blank paper values also mean no verified comparison for local RMSE and unpopulated per-scene references, not zero performance.

No exact historical dependency lock, run timestamp, model revision, or dataset checksum has been recovered. A new snapshot must be labeled as a new environment. See `docs/COLAB_STAGE3.md`.

Saved-artifact regeneration: the team returned a Colab ZIP with timestamp 20261018T145617153386Z. Generated comparison/plot outputs were imported; console evidence is in results/stage3_regeneration.log. This is a CSV/plot regeneration step, not a model rerun or historical environment recovery.

Metric-test evidence: results/stage3_metric_tests.log records six passing tests on commit baae1a5b56c587c76fccc9ebc65e4e33c96d7824 in a fresh Colab session. Test-only versions are in requirements.metric-tests.txt; these are not historical inference-environment pins.
