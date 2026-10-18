"""Colab CPU smoke checks for the deployment service; not hosted acceptance."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import platform
import resource
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from app.depth_service import MODEL_REVISIONS, ModelManager, prepare_image, update_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("Use a new output directory; do not overwrite a previous verification")
    args.out_dir.mkdir(parents=True)
    logging.basicConfig(filename=args.out_dir / "service.log", level=logging.INFO)
    import numpy as np
    import torch
    from PIL import Image
    if torch.version.cuda is not None:
        raise RuntimeError("Use the isolated CPU environment from the deployment notebook.")
    if platform.python_version_tuple()[:2] != ("3", "13"):
        raise RuntimeError("Use Python 3.13 to match the planned hosted deployment.")
    packages = {name: metadata.version(name) for name in
                ("torch", "torchvision", "transformers", "huggingface_hub", "numpy", "pillow", "matplotlib", "streamlit", "pytest")}
    manifest = {"context": "Colab CPU application verification, not Streamlit Cloud acceptance",
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                "captured_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "packages": packages, "model_revisions": MODEL_REVISIONS,
                "deployment_config_sha256": hashlib.sha256((ROOT / "app/deployment.json").read_bytes()).hexdigest()}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True)
    (args.out_dir / "requirements.snapshot.txt").write_text(freeze, encoding="utf-8")
    content = args.image.read_bytes()
    prepared = prepare_image(content, args.image.name)
    state, manager, results = {}, ModelManager(), []
    # Test all target models in Colab, without changing the Small-only hosted config.
    sequence = ["v2-small", "v2-base", "v2-large", "v2-small"]
    for number, key in enumerate(sequence, 1):
        print(f"CPU prediction {number}/4: {key}", flush=True)
        result = update_result(state, content, args.image.name, key, manager, submitted=True,
                               on_status=lambda message: print(message, flush=True))
        assert update_result(state, content, args.image.name, key, manager) is result
        assert result["device"] == "cpu" and result["revision"] == MODEL_REVISIONS[key]
        assert result["loaded_this_request"]
        for field in ("input_png", "colour_png", "depth16_png"):
            decoded = Image.open(io.BytesIO(result[field]))
            assert list(decoded.size) == result["size"]
            decoded.load()
            (args.out_dir / f"{number}_{key}_{field}.png").write_bytes(result[field])
        assert Image.open(io.BytesIO(result["colour_png"])).mode == "RGB"
        raw = np.asarray(Image.open(io.BytesIO(result["depth16_png"])))
        assert raw.min() >= 0 and raw.max() <= 65535
        if not result["constant"]:
            assert raw.min() == 0 and raw.max() == 65535
        record = {k: v for k, v in result.items() if not isinstance(v, bytes)}
        record["sequence_position"] = number
        record["process_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        results.append(record)
        (args.out_dir / "prediction_checks.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    manager.clear()
    summary = {"passed": True, "models": list(MODEL_REVISIONS), "predictions": len(sequence),
               "small_reload_after_large": True, "retained_result_check": True,
               "original_size": list(prepared.original_size), "output_size": list(prepared.image.size),
               "note": "Process peak RSS is cumulative, not per-model memory. Colab success does not establish host feasibility."}
    (args.out_dir / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Return the ZIP for review before deploying. This does not enable Base or Large publicly.")


if __name__ == "__main__":
    main()
