# Colab steps to finish Stage 3

The core pretrained evaluation is complete under the instructor-approved proposal.
These steps preserve environment evidence and validate saved artifacts. No model
inference, training, dataset download, or GPU benchmark is required for this cleanup.
The local cleanup did not execute Python, tests, installs, or model workloads.

## 1 Capture versions before changing your runtime

If the original results session is still connected and its packages have not
changed, use that session. Do not restart it, switch its runtime, or install
anything first. Otherwise use an available session and keep CONTEXT as `new`.
A new session cannot recover the exact environment of the earlier run.

Paste this standalone cell into Colab. It needs no repository checkout or package
installation. Change CONTEXT to `original` only if that description is accurate.

```python
from pathlib import Path
from importlib import metadata
import datetime, json, platform, subprocess, sys, zipfile
from google.colab import files

CONTEXT = "new"  # use "original" only for the unchanged results runtime
assert CONTEXT in {"original", "new"}
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out = Path("/content") / f"stage3_environment_{CONTEXT}_{stamp}"
out.mkdir(exist_ok=False)
names = ["torch", "torchvision", "transformers", "huggingface_hub", "numpy",
         "pillow", "pandas", "matplotlib", "tqdm", "streamlit", "pytest", "setuptools"]
versions = {}
for name in names:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"],
                        capture_output=True, text=True, check=True)
(out / "requirements.colab.snapshot.txt").write_text(freeze.stdout, encoding="utf-8")
pins = [f"{name}=={version}" for name, version in versions.items() if version]
(out / "requirements.colab.candidate.txt").write_text("\n".join(pins) + "\n", encoding="utf-8")
info = {"context": CONTEXT, "captured_at_utc": stamp, "python": sys.version,
        "platform": platform.platform(), "packages": versions,
        "historical_result_environment": CONTEXT == "original",
        "note": "Team context declaration; capture time is not the historical run time. "
                "Candidate pins are not an installation-tested lockfile."}
(out / "colab_environment.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
archive = out.with_suffix(".zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for path in sorted(out.iterdir()):
        z.write(path, path.name)
print(json.dumps(info, indent=2))
files.download(str(archive))
```

Send the downloaded ZIP back to the chat, or send the printed package versions
and context. Do not rename candidate pins as a verified historical lockfile.
If the original runtime is gone, record that limitation; a full benchmark rerun
is not needed merely to finish this disclosure.

## 2 Regenerate the table and plot from saved CSVs

Do this after Step 1. The reference edits currently exist in the local checkout;
a fresh GitHub clone will not contain them until they are transferred.
This isolated Colab cell uses only five files uploaded from the local repository:
`results/paper_reported.csv`, `results/nyu_summary.csv`,
`results/da2k_summary.csv`, `results/latency.csv`, and
`scripts/06_make_report_tables.py`. It never opens model weights or datasets.

```python
from google.colab import files
from pathlib import Path
import datetime, re, subprocess, sys, zipfile

required = {"paper_reported.csv", "nyu_summary.csv", "da2k_summary.csv",
            "latency.csv", "06_make_report_tables.py"}
# Reuse a previous successful upload if the old strict filename check failed.
if "uploaded" not in globals():
    uploaded = files.upload()
normalized = {}
for original, content in uploaded.items():
    name = re.sub(r" \(\d+\)(?=\.[^.]+$)", "", Path(original).name)
    if name not in required:
        raise ValueError(f"Unexpected file: {original}")
    if name in normalized:
        raise ValueError(f"Multiple copies of {name}; upload only one copy.")
    normalized[name] = content
    print(f"Using {original} as {name}")
missing = required - set(normalized)
assert not missing, f"Missing files: {sorted(missing)}"
uploaded = normalized
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
work = Path("/content") / f"stage3_saved_artifacts_{stamp}"
(work / "results").mkdir(parents=True)
(work / "figures").mkdir()
(work / "scripts").mkdir()
for name, content in uploaded.items():
    target = work / ("scripts" if name.endswith(".py") else "results") / name
    target.write_bytes(content)
run = subprocess.run([sys.executable, "scripts/06_make_report_tables.py", "--dataset", "nyu"],
                     cwd=work, capture_output=True, text=True)
(work / "regeneration.log").write_text(run.stdout + run.stderr, encoding="utf-8")
print(run.stdout, run.stderr)
assert run.returncode == 0, "Return the log for diagnosis; no model rerun is needed."
archive = work.with_suffix(".zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for path in sorted(work.rglob("*")):
        if path.is_file():
            z.write(path, str(path.relative_to(work)))
files.download(str(archive))
```

This uses pandas and matplotlib already available in many Colab sessions. If an
import fails, send the error; any installation should happen after capture.
Return the generated archive. Existing raw results remain untouched.

## 3 Save a test log in Colab

After Step 1, in a Colab checkout containing `src/` and `tests/`, use:

```python
import os, subprocess, sys
from pathlib import Path

assert Path("src/dav2/metrics.py").exists(), "Change directory to the repository first."
assert Path("tests/test_metrics.py").exists()
env = dict(os.environ)
env["PYTHONPATH"] = str(Path("src").resolve()) + os.pathsep + env.get("PYTHONPATH", "")
run = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_metrics.py"],
                     env=env, capture_output=True, text=True)
Path("stage3_metric_tests.log").write_text(run.stdout + run.stderr, encoding="utf-8")
print(run.stdout, run.stderr)
print("Exit code:", run.returncode)
```

If pytest is missing, capture the environment first, then install it with
`%pip install pytest` and rerun this test cell. Document that this installation
was after capture. This tests NumPy metrics and data helpers; it does not load
models or run inference. Download the log with
`files.download("stage3_metric_tests.log")` and return it.

## 4 Finish traceability and report text

The assistant can incorporate the returned environment/archive/log evidence.
The team should confirm the reason KITTI was omitted and whether that scope
change was approved. Proposal approval does not by itself establish this later
change. No technical run can answer that question.

If historical model/dataset revisions or checksums were not recorded, say so.
Hashes of newly downloaded data cannot establish the exact historical inputs.
Use `docs/STAGE3_REPORT_NOTES.md` to revise the report and include existing RMSE.
Keep claims about full author-protocol parity qualified unless separately verified.

Any future full rerun should use a separate output directory and preserve these
historical results; it is not a prerequisite for the lightweight steps above.

## Returned Colab artifact evidence

The team returned `stage3_saved_artifacts_20261018T145617153386Z.zip` after
running the saved-CSV regeneration cell in Colab. The archive contains the
reference/input CSVs, table-generation script, generated comparison CSV,
accuracy-versus-latency PNG, and a log ending in the successful figure-save
message. Its filename records the wrapper's UTC timestamp, not the original
benchmark run time. No model evaluation was repeated by this step.

The returned comparison CSV and plot have been copied into `results/` and
`figures/`; the console log is saved as `results/stage3_regeneration.log`.
The log contains no traceback. Missing reference values remain intentionally
blank. This archive does not include an environment snapshot or metric-test
log, so neither is established by this regeneration step.

## Returned Colab metric-test evidence

`results/stage3_metric_tests.log` records exit code 0 and **6 passed in 0.26s**
against public repository commit `baae1a5b56c587c76fccc9ebc65e4e33c96d7824`.
The fresh session used Python 3.13.16, pytest 8.4.2, NumPy 2.1.3, and Pillow
11.3.0. These test-only package pins are in `requirements.metric-tests.txt`.
They describe the new metric-test session, not the earlier model runs.

The tests cover alignment invariance, wrong-ordering penalties, invalid-depth
masking, delta1 thresholds, pair ordering/ties, and subset/scene helpers. They
do not establish exact parity with the authors' protocol or rerun benchmarks.
The local Stage 3 cleanup has not changed `src/` or `tests/` relative to that
commit. Saved-artifact regeneration and metric-test evidence are now returned;
no further Colab execution is needed for this documentation cleanup. Exact
historical model-run versions and revisions remain unavailable.
