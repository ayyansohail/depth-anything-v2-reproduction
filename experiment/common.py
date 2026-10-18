"""Shared Stage 4 configuration and evidence validation; no model imports here."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRANCH = "feat/computational-analysis-experiment"
ORDER_A = ["v2-small", "v1-small", "v2-base", "v1-base", "v2-large", "v1-large"]
ORDERS = [ORDER_A, list(reversed(ORDER_A)), list(reversed(ORDER_A)), ORDER_A]
CONFIG = {"seed": 0, "shape": [1, 3, 518, 686], "dtype": "float32",
          "warmup": 10, "iterations": 50, "sweeps": 4,
          "threads": 1, "tf32": False, "cudnn_benchmark": False,
          "input_kind": "seeded synthetic standard-normal tensor; not an accuracy image",
          "measurement": "synchronized host wall-clock; bare forward pass only"}
INPUTS = ["results/nyu_per_image.csv", "results/nyu_summary.csv", "results/da2k_pairs.csv",
          "experiment/legacy/latency_cpu.csv", "experiment/legacy/latency_cuda.csv"]
PACKAGES = ["torch", "transformers", "huggingface_hub", "numpy", "pillow",
            "pandas", "matplotlib", "pytest"]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def package_versions():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def file_hashes():
    return {name: sha256(ROOT / name) for name in INPUTS}


def runtime_identity(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("Select a T4 GPU in Colab. CPU fallback is intentionally disabled.")
    name = torch.cuda.get_device_name(0)
    if "T4" not in name:
        raise RuntimeError(f"Expected a T4; got {name}. Return this output before changing the protocol.")
    boot = Path("/proc/sys/kernel/random/boot_id")
    return {"hostname": platform.node(), "boot_id": boot.read_text().strip() if boot.exists() else None,
            "parent_pid": os.getppid(), "python": platform.python_version(),
            "packages": package_versions(), "gpu": name, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "threads": torch.get_num_threads()}


def expected_blocks():
    return [(sweep + 1, pos + 1, model)
            for sweep, order in enumerate(ORDERS) for pos, model in enumerate(order)]


def validate_samples(rows, sweep, position, model):
    if len(rows) != CONFIG["iterations"]:
        raise ValueError(f"{model}: expected {CONFIG['iterations']} timings, got {len(rows)}")
    seen = set()
    for row in rows:
        key = int(row["iteration"])
        if key in seen or not 1 <= key <= CONFIG["iterations"]:
            raise ValueError("Duplicate or invalid timing iteration")
        seen.add(key)
        if (int(row["sweep"]), int(row["position"]), row["model"]) != (sweep, position, model):
            raise ValueError("Timing row has the wrong sweep, position, or model")
        if not math.isfinite(float(row["latency_ms"])) or float(row["latency_ms"]) <= 0:
            raise ValueError("Timings must be finite and positive")


def validate_manifest(manifest):
    if manifest["config"] != CONFIG or manifest["orders"] != ORDERS:
        raise ValueError("Protocol changed; create a new run, do not reuse this directory")
    if manifest["source_commit"] != git("rev-parse", "HEAD"):
        raise ValueError("Source commit changed during the run")
    if manifest["source_branch"] != BRANCH or git("branch", "--show-current") != BRANCH:
        raise ValueError("Run must execute on the experiment branch")
    if manifest["input_hashes"] != file_hashes():
        raise ValueError("Inherited evidence files changed during the run")
    if git("status", "--porcelain"):
        raise ValueError("The checkout has tracked/untracked changes; use a clean checkout")
