"""Colab-only capture, pilot and benchmark. Never invoked by repository imports."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiment.common import (BRANCH, CONFIG, ORDER_A, ORDERS, ROOT, expected_blocks,
                               file_hashes, git, now, read_json, runtime_identity,
                               validate_manifest, validate_samples, write_json)


def configure(torch):
    torch.set_num_threads(CONFIG["threads"])
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def load(key, manifest):
    import torch
    from transformers import AutoModelForDepthEstimation
    from dav2.models import DepthModel, REGISTRY
    revision = manifest["model_revisions"][key]
    model = AutoModelForDepthEstimation.from_pretrained(
        REGISTRY[key], revision=revision, torch_dtype=torch.float32).to("cuda").eval()
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError(f"{key}: expected FP32 model parameters")
    return DepthModel(key, "cuda", None, model)


def make_input(torch):
    generator = torch.Generator(device="cpu").manual_seed(CONFIG["seed"])
    cpu = torch.randn(CONFIG["shape"], generator=generator, dtype=torch.float32)
    checksum = hashlib.sha256(cpu.numpy().tobytes()).hexdigest()
    return cpu.to("cuda"), checksum


def check_output(torch, output):
    if output.ndim != 3 or output.shape[0] != 1 or output.dtype != torch.float32:
        raise RuntimeError(f"Unexpected output: {output.shape}, {output.dtype}")
    if not torch.isfinite(output).all().item():
        raise RuntimeError("Non-finite model output")


def capture(run_dir, torch):
    from huggingface_hub import HfApi
    from dav2.models import REGISTRY
    if run_dir.exists():
        raise FileExistsError("Run directory already exists; choose a fresh directory")
    if git("branch", "--show-current") != BRANCH or git("status", "--porcelain"):
        raise RuntimeError("Capture requires a clean checkout of the experiment branch")
    runtime = runtime_identity(torch)
    revisions = {key: HfApi().model_info(REGISTRY[key]).sha for key in ORDER_A}
    if not all(revisions.values()):
        raise RuntimeError("Could not resolve all model revisions")
    run_dir.mkdir(parents=True)
    manifest = {"schema": 1, "run_id": run_dir.name, "captured_at": now(),
                "source_commit": git("rev-parse", "HEAD"), "source_branch": BRANCH,
                "config": CONFIG, "orders": ORDERS, "runtime": runtime,
                "model_ids": {key: REGISTRY[key] for key in ORDER_A},
                "model_revisions": revisions, "input_hashes": file_hashes(),
                "context": "new Stage 4 timing environment, not the historical accuracy environment"}
    write_json(run_dir / "manifest.json", manifest)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True)
    (run_dir / "requirements.snapshot.txt").write_text(freeze, encoding="utf-8")
    gpu = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    (run_dir / "nvidia-smi.txt").write_text(gpu.stdout + gpu.stderr, encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def pilot(run_dir, manifest, torch):
    if (run_dir / "pilot.json").exists():
        raise FileExistsError("Pilot evidence already exists; preserve it and start a fresh run if needed")
    result = {"passed": False, "started_at": now(), "models": []}
    for key in ORDER_A:
        torch.cuda.empty_cache()
        model = load(key, manifest)
        tensor, checksum = make_input(torch)
        output = model.forward_raw(tensor)
        torch.cuda.synchronize()
        check_output(torch, output)
        result["models"].append({"model": key, "input_shape": list(tensor.shape),
                                 "input_sha256": checksum, "output_shape": list(output.shape),
                                 "dtype": str(output.dtype), "params_m": model.params_m,
                                 "attention_backend": str(getattr(model.model.config, "_attn_implementation", None)),
                                 "revision": manifest["model_revisions"][key]})
        write_json(run_dir / "pilot_progress.json", result)
        print(f"PILOT PASS {key}: shape={tuple(output.shape)}, dtype={output.dtype}", flush=True)
        del output, tensor, model
        gc.collect()
        torch.cuda.empty_cache()
    result.update(passed=True, finished_at=now())
    write_json(run_dir / "pilot.json", result)
    print("All six models passed. Download pilot evidence and return it before benchmarking.")


def benchmark(run_dir, manifest, torch):
    pilot_result = read_json(run_dir / "pilot.json")
    if not pilot_result.get("passed") or [m["model"] for m in pilot_result["models"]] != ORDER_A:
        raise RuntimeError("A successful six-model pilot is required")
    state_path = run_dir / "run_state.json"
    if state_path.exists():
        raise FileExistsError("Benchmark already started; do not mix reruns. Preserve this directory and create a new run.")
    write_json(state_path, {"status": "running", "started_at": now()})
    try:
        for sweep, position, key in expected_blocks():
            print(f"Sweep {sweep}/4, position {position}/6: {key}", flush=True)
            torch.cuda.empty_cache()
            model = load(key, manifest)
            tensor, checksum = make_input(torch)
            pilot_model = next(m for m in pilot_result["models"] if m["model"] == key)
            attention = str(getattr(model.model.config, "_attn_implementation", None))
            if checksum != pilot_model["input_sha256"] or attention != pilot_model["attention_backend"]:
                raise RuntimeError("Input or attention backend changed since pilot")
            check_output(torch, model.forward_raw(tensor))
            for _ in range(CONFIG["warmup"]):
                model.forward_raw(tensor)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            rows = []
            for iteration in range(1, CONFIG["iterations"] + 1):
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = model.forward_raw(tensor)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                del output
                rows.append({"run_id": manifest["run_id"], "sweep": sweep,
                             "order": "A" if sweep in (1, 4) else "B", "position": position,
                             "model": key, "iteration": iteration, "latency_ms": elapsed})
            peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
            check_output(torch, model.forward_raw(tensor))
            validate_samples(rows, sweep, position, key)
            values = [r["latency_ms"] for r in rows]
            block = {"run_id": manifest["run_id"], "sweep": sweep, "position": position,
                     "model": key, "shape": list(tensor.shape), "dtype": "float32",
                     "input_sha256": checksum, "revision": manifest["model_revisions"][key],
                     "attention_backend": attention, "n": len(rows), "warmup": CONFIG["warmup"],
                     "mean_ms": statistics.mean(values), "std_ms": statistics.stdev(values),
                     "median_ms": statistics.median(values), "params_m": model.params_m,
                     "fp32_weights_mib": model.params_m * 1e6 * 4 / (1024 ** 2),
                     "peak_allocated_mib": peak_mib, "completed_at": now()}
            folder = run_dir / "blocks" / f"s{sweep}_p{position}_{key}"
            folder.mkdir(parents=True)
            with (folder / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            write_json(folder / "block.json", block)
            print(json.dumps(block), flush=True)
            del tensor, model
            gc.collect()
            torch.cuda.empty_cache()
        write_json(state_path, {"status": "complete", "blocks": 24, "finished_at": now()})
    except Exception as error:
        write_json(state_path, {"status": "failed", "finished_at": now(), "error": repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["capture", "pilot", "run"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    import torch
    configure(torch)
    if args.phase == "capture":
        capture(args.run_dir, torch)
        return
    manifest = read_json(args.run_dir / "manifest.json")
    validate_manifest(manifest)
    if runtime_identity(torch) != manifest["runtime"]:
        raise RuntimeError("Runtime changed. Start a fresh captured run; do not mix sessions.")
    logging.basicConfig(filename=args.run_dir / f"{args.phase}.log", level=logging.INFO)
    try:
        if args.phase == "pilot":
            pilot(args.run_dir, manifest, torch)
        else:
            benchmark(args.run_dir, manifest, torch)
    except Exception:
        logging.exception("Phase failed")
        raise


if __name__ == "__main__":
    main()
