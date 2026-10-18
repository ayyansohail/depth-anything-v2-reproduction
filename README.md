# Depth Anything V2 - reproduction, experiment, deployment

Course project (Intro to Machine Learning). Paper: Yang et al., *Depth Anything V2*, NeurIPS 2024 (arXiv:2406.09414).
Team: Ayyan Sohail, Omar Khan, Maaz Zubairi.

**This project uses the authors' pretrained checkpoints (Small / Base / Large). No model is trained.**
The work is the evaluation pipeline, the metric code, a paper-specific experiment, and a deployed demo.

## Stage 3 status

The instructor approved the proposal's pretrained-evaluation scope (team confirmation).
Saved NYU and DA-2K evaluations are present. Dependency ranges are not a historical
lockfile; environment capture and remaining evidence steps are in
[Colab Stage 3 guide](docs/COLAB_STAGE3.md). Do not install or upgrade packages in
an original surviving results runtime before capturing it.

Known protocol differences are documented in [EVAL_PROTOCOL](docs/EVAL_PROTOCOL.md).
NYU cropping requires the explicit `--eigen-crop` flag; it is not a script default.
The `--limit` option counts images, including on DA-2K, not annotated pairs.
KITTI was proposed but has not been evaluated; DA-2K does not replace its dense
outdoor evaluation. Report corrections are in [Stage 3 report notes](docs/STAGE3_REPORT_NOTES.md).

## Quickstart
```bash
git clone https://github.com/ayyansohail/depth-anything-v2-reproduction.git && cd depth-anything-v2-reproduction
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest -q                                                # metric unit tests (no GPU, no data needed)
python scripts/01_smoke_test.py                          # V2 S/B/L; add --models for V1 too
streamlit run app/streamlit_app.py                       # the deployment
```

## Pipeline (run in this order; Colab/Kaggle T4 for 02-04)
| Step | Command | Output |
|---|---|---|
| Smoke test | `python scripts/01_smoke_test.py` | console check |
| Dense-depth eval (NYU, KITTI optional) | `python scripts/02_eval_depth.py --dataset nyu --data-root data/nyu --eigen-crop [--limit 10]` | `results/nyu_*.csv` |
| DA-2K pair accuracy, V1 and V2 | `python scripts/03_eval_da2k.py --data-root data/da2k [--limit 20]` | `results/da2k_pairs.csv` |
| Latency / params / memory | `python scripts/04_benchmark_latency.py` | `results/latency.csv` |
| Experiment analysis | `python scripts/05_analyze_da2k.py` | `results/da2k_*.csv`, `figures/da2k_gain_by_scene.png` |
| Paper-vs-ours + accuracy-vs-latency (saved CSVs only) | `python scripts/06_make_report_tables.py` | `results/paper_vs_ours.csv`, `figures/accuracy_vs_latency.png` |

## Data setup (data/ is git-ignored)
- **NYU Depth V2, Eigen test split (654 images):** `wget -O data/nyu.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/nyu-depth.zip && unzip data/nyu.zip -d data/nyu` (third-party repackaging; cite it, see PROVENANCE).
- **DA-2K:** `huggingface_hub.snapshot_download(repo_id="depth-anything/DA-2K", repo_type="dataset", local_dir="data/da2k")`.
- **KITTI (optional):** needs a free account at cvlibs.net; see data/README.md for the manifest format.

## Deployment documentation checklist
- Start: `streamlit run app/streamlit_app.py`
- Dependencies: `requirements.txt`
- Input: `.jpg` / `.png` upload. Output: input next to colourised depth map (bright = near), latency in ms, model size, PNG downloads.
- Worked example + screenshot: `docs/` (add before submission)

## Licences
Depth-Anything-V2 Small is Apache-2.0; Base and Large are CC-BY-NC-4.0 (non-commercial). Fine for coursework; do not ship them in a commercial product.

## Docs
`docs/DESIGN.md` diagrams + workflow - `docs/EXPERIMENT.md` the experiment - `docs/EVAL_PROTOCOL.md` metric decisions - `PROVENANCE.md` who wrote what

## Stage 3 validation evidence

- `results/stage3_regeneration.log`: comparison CSV and latency plot regenerated in Colab from saved CSVs.
- `results/stage3_metric_tests.log`: six metric/data-helper tests passed in a fresh Colab session at commit `baae1a5b56c587c76fccc9ebc65e4e33c96d7824`.
- `requirements.metric-tests.txt`: exact versions of the three direct test dependencies in that session; not a full-project or historical inference lock.

No new benchmark run was needed for these checks. The original inference environment remains unarchived.
