"""Validate and analyze Stage 4 evidence in Colab; never reruns inference."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment.common import (BRANCH, CONFIG, ORDER_A, ORDERS, ROOT, expected_blocks, file_hashes,
                               read_json, validate_samples, write_json)


B = 2000
SEED = 0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_accuracy(nyu, pairs):
    require(not nyu[["model", "image", "absrel", "rmse", "delta1"]].isna().any().any(), "Missing NYU fields")
    require(set(nyu.model) == {f"v2-{s}" for s in ("small", "base", "large")}, "NYU model set differs")
    require(not nyu.duplicated(["model", "image"]).any(), "Duplicate NYU image/model")
    require(set(nyu.groupby("model").size()) == {654}, "Expected 654 NYU images per model")
    require(np.isfinite(nyu[["absrel", "rmse", "delta1"]].to_numpy()).all(), "Non-finite NYU metric")
    require((nyu.absrel >= 0).all() and (nyu.rmse >= 0).all() and nyu.delta1.between(0, 1).all(), "Invalid NYU metric range")
    require(nyu.groupby("image").model.nunique().eq(3).all(), "NYU models do not share exactly the same images")
    require(set(pairs.model) == set(ORDER_A), "DA-2K model set differs")
    require(not pairs.duplicated(["model", "image", "pair"]).any(), "Duplicate DA-2K pair/model")
    require(set(pairs.groupby("model").size()) == {2068}, "Expected 2068 DA-2K pairs per model")
    require(pairs.correct.isin([0, 1]).all(), "DA-2K correctness must be binary")
    require(not pairs[["model", "scene", "image", "pair", "correct"]].isna().any().any(), "Missing DA-2K fields")
    groups = pairs.groupby(["image", "pair"])
    require(groups.model.nunique().eq(6).all(), "Models are missing paired DA-2K observations")
    require(groups.scene.nunique().eq(1).all(), "Scene labels differ for the same DA-2K pair")
    require(pairs.groupby("image").scene.nunique().eq(1).all(), "Image occurs in multiple scenes")


def paired_cluster_bootstrap(counts, sums, rng, replicates=B):
    """Pair-weighted accuracy; all annotations from a resampled image stay together."""
    counts = np.asarray(counts, dtype=float)
    sums = np.asarray(sums, dtype=float)
    if sums.ndim == 1:
        sums = sums[:, None]
    require(len(counts) == len(sums) and len(counts) > 0 and (counts > 0).all(), "Invalid bootstrap clusters")
    samples = []
    for _ in range(replicates):
        indices = rng.integers(0, len(counts), size=len(counts))
        samples.append(sums[indices].sum(axis=0) / counts[indices].sum())
    return np.asarray(samples)


def accuracy_analysis():
    nyu = pd.read_csv(ROOT / "results/nyu_per_image.csv")
    pairs = pd.read_csv(ROOT / "results/da2k_pairs.csv")
    validate_accuracy(nyu, pairs)
    stored = pd.read_csv(ROOT / "results/nyu_summary.csv")
    for model, part in nyu.groupby("model"):
        for metric in ("absrel", "rmse", "delta1"):
            reference = stored[(stored.model == model) & (stored.metric == metric)]
            require(len(reference) == 1 and np.isclose(part[metric].mean(), reference.ours.iloc[0], atol=1e-12, rtol=0),
                    "NYU stored summary does not match its per-image evidence")
    rng = np.random.default_rng(SEED)
    summaries, gains, nyu_gains = [], [], []
    for metric in ("absrel", "rmse", "delta1"):
        wide = nyu.pivot(index="image", columns="model", values=metric).sort_index()
        samples = paired_cluster_bootstrap(np.ones(len(wide)), wide.to_numpy(), rng)
        for j, model in enumerate(wide.columns):
            lo, hi = np.percentile(samples[:, j], [2.5, 97.5])
            summaries.append({"dataset": "nyu", "model": model, "metric": metric,
                              "value": wide[model].mean(), "ci_lo": lo, "ci_hi": hi, "n_images": len(wide)})
        for small, large in (("small", "base"), ("base", "large"), ("small", "large")):
            a, b = f"v2-{small}", f"v2-{large}"
            difference = samples[:, wide.columns.get_loc(b)] - samples[:, wide.columns.get_loc(a)]
            lo, hi = np.percentile(difference, [2.5, 97.5])
            nyu_gains.append({"from": a, "to": b, "metric": metric,
                              "difference": wide[b].mean() - wide[a].mean(), "ci_lo": lo, "ci_hi": hi})
    wide = pairs.pivot(index=["image", "pair"], columns="model", values="correct").sort_index()
    counts = wide.groupby(level="image").size()
    sums = wide.groupby(level="image").sum().reindex(counts.index)
    samples = paired_cluster_bootstrap(counts.to_numpy(), sums.to_numpy(), rng)
    for j, model in enumerate(sums.columns):
        lo, hi = np.percentile(samples[:, j], [2.5, 97.5])
        summaries.append({"dataset": "da2k", "model": model, "metric": "acc_overall",
                          "value": wide[model].mean(), "ci_lo": lo, "ci_hi": hi, "n_images": len(counts)})
    for size in ("small", "base", "large"):
        a, b = f"v1-{size}", f"v2-{size}"
        differences = samples[:, sums.columns.get_loc(b)] - samples[:, sums.columns.get_loc(a)]
        lo, hi = np.percentile(differences, [2.5, 97.5])
        gains.append({"size": size, "gain_pp": 100 * (wide[b].mean() - wide[a].mean()),
                      "ci_lo_pp": 100 * lo, "ci_hi_pp": 100 * hi, "n_images": len(counts), "n_pairs": len(wide)})
    return pd.DataFrame(summaries), pd.DataFrame(gains), pd.DataFrame(nyu_gains)


def load_timings(run_dir):
    manifest = read_json(run_dir / "manifest.json")
    require(manifest["config"] == CONFIG and manifest["orders"] == ORDERS, "Unexpected timing protocol")
    require(manifest["source_branch"] == BRANCH, "Unexpected source branch")
    require("T4" in manifest["runtime"]["gpu"], "Unexpected timing hardware")
    require(manifest["input_hashes"] == file_hashes(), "Inherited accuracy or legacy timing evidence changed")
    state = read_json(run_dir / "run_state.json")
    require(state["status"] == "complete", "Benchmark is incomplete; preserve it and return its logs")
    pilot = read_json(run_dir / "pilot.json")
    require(pilot["passed"], "Pilot was not successful")
    pilot_models = {m["model"]: m for m in pilot["models"]}
    require(set(pilot_models) == set(ORDER_A) and len(pilot["models"]) == 6, "Incomplete pilot model set")
    blocks, timings = [], []
    expected = expected_blocks()
    require(len(list((run_dir / "blocks").glob("*/block.json"))) == len(expected), "Missing or extra measurement blocks")
    for sweep, position, model in expected:
        folder = run_dir / "blocks" / f"s{sweep}_p{position}_{model}"
        block = read_json(folder / "block.json")
        with (folder / "timings.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        validate_samples(rows, sweep, position, model)
        require((block["sweep"], block["position"], block["model"]) == (sweep, position, model), "Wrong block identity")
        require(all(r["run_id"] == manifest["run_id"] for r in rows) and block["run_id"] == manifest["run_id"], "Mixed run IDs")
        require(all(r["order"] == ("A" if sweep in (1, 4) else "B") for r in rows), "Incorrect timing order labels")
        require(block["shape"] == CONFIG["shape"] and block["dtype"] == CONFIG["dtype"], "Wrong shape or dtype")
        require(block["warmup"] == CONFIG["warmup"] and block["n"] == CONFIG["iterations"], "Wrong iteration counts")
        require(block["revision"] == manifest["model_revisions"][model], "Model revision mismatch")
        require(block["input_sha256"] == pilot_models[model]["input_sha256"], "Synthetic input changed")
        require(block["attention_backend"] == pilot_models[model]["attention_backend"], "Attention backend changed")
        values = np.array([float(r["latency_ms"]) for r in rows])
        for name, value in (("mean_ms", values.mean()), ("median_ms", np.median(values)), ("std_ms", values.std(ddof=1))):
            require(np.isclose(block[name], value, atol=1e-9, rtol=1e-9), "Block summary differs from raw samples")
        require(np.isfinite(block["peak_allocated_mib"]) and block["peak_allocated_mib"] > 0, "Invalid peak memory")
        block["order"] = "A" if sweep in (1, 4) else "B"
        block["p95_ms"] = float(np.percentile(values, 95))
        blocks.append(block)
        timings.extend(rows)
    return manifest, pd.DataFrame(blocks), pd.DataFrame(timings)


def analyze(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest, blocks, timings = load_timings(run_dir)
    accuracy, gains, nyu_gains = accuracy_analysis()
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    timings.to_csv(out / "raw_timings.csv", index=False)
    blocks.to_csv(out / "gpu_blocks.csv", index=False)
    accuracy.to_csv(out / "accuracy_with_image_bootstrap_ci.csv", index=False)
    gains.to_csv(out / "da2k_v2_minus_v1.csv", index=False)
    nyu_gains.to_csv(out / "nyu_size_differences.csv", index=False)
    summary = blocks.groupby("model", sort=False).agg(
        median_of_sweep_medians_ms=("median_ms", "median"),
        min_sweep_median_ms=("median_ms", "min"), max_sweep_median_ms=("median_ms", "max"),
        peak_allocated_mib=("peak_allocated_mib", "max"), params_m=("params_m", "first"))
    summary.to_csv(out / "gpu_summary.csv")
    ratios = []
    for sweep in range(1, 5):
        part = blocks[blocks.sweep == sweep].set_index("model")
        for size in ("small", "base", "large"):
            v1, v2 = part.loc[f"v1-{size}"], part.loc[f"v2-{size}"]
            ratios.append({"sweep": sweep, "order": v1["order"], "size": size,
                           "v1_median_ms": v1.median_ms, "v2_median_ms": v2.median_ms,
                           "v1_over_v2": v1.median_ms / v2.median_ms})
    ratios = pd.DataFrame(ratios)
    ratios.to_csv(out / "v1_v2_latency_ratios.csv", index=False)
    joined = accuracy.merge(summary.reset_index(), on="model", validate="many_to_one")
    joined["accuracy_source"] = "inherited Stage 3 evaluation; historical model revision unavailable"
    joined["timing_source"] = "new synthetic-input GPU benchmark; not remeasured accuracy"
    joined.to_csv(out / "accuracy_cost_comparison.csv", index=False)
    legacy = pd.read_csv(ROOT / "experiment/legacy/latency_cpu.csv")
    require(set(legacy.model) == set(ORDER_A) and not legacy.model.duplicated().any(), "Invalid legacy CPU model rows")
    legacy["context"] = "legacy one-thread CPU; not rerun; CPU hardware model unavailable"
    legacy.to_csv(out / "legacy_cpu_context.csv", index=False)
    for dataset, metric, label, factor in (("nyu", "absrel", "NYU AbsRel (lower is better)", 1),
                                            ("da2k", "acc_overall", "DA-2K pair accuracy (%)", 100)):
        data = joined[(joined.dataset == dataset) & (joined.metric == metric)]
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for row in data.itertuples():
            x = row.median_of_sweep_medians_ms
            y = row.value * factor
            ax.errorbar(x, y, xerr=[[x-row.min_sweep_median_ms], [row.max_sweep_median_ms-x]],
                        yerr=[[(row.value-row.ci_lo)*factor], [(row.ci_hi-row.value)*factor]],
                        fmt="o" if row.model.startswith("v2") else "s", capsize=3,
                        label=row.model)
        ax.set(xlabel="T4 forward latency (ms): median across sweep medians", ylabel=label)
        ax.margins(x=0.15, y=0.15)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(out / f"{dataset}_accuracy_vs_gpu_latency.png", dpi=180)
        plt.close(fig)
    nyu = accuracy[(accuracy.dataset == "nyu") & (accuracy.metric == "absrel")]
    cpu = legacy.merge(nyu, on="model", validate="one_to_one")
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    for row in cpu.itertuples():
        ax.scatter(row.fwd_mean_ms / 1000, row.value, label=row.model)
    ax.set(xlabel="Legacy CPU forward time (s): mean, 10 iterations, one thread", ylabel="Inherited NYU AbsRel")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out / "nyu_accuracy_vs_legacy_cpu.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for size, part in ratios.groupby("size", sort=False):
        ax.plot(part.sweep, part.v1_over_v2, "o-", label=size)
    ax.axhline(1, color="black", linestyle="--")
    ax.set(xticks=[1, 2, 3, 4], xticklabels=["1: A", "2: B", "3: B", "4: A"],
           xlabel="Sweep and execution order", ylabel="V1 / V2 median forward latency")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out / "order_sensitivity.png", dpi=180)
    plt.close(fig)
    notes = ["# Stage 4 analysis evidence", "",
             f"Timing execution commit: {manifest['source_commit']}",
             "Accuracy is inherited; the new timing environment does not recover its historical model revisions.",
             "Horizontal ranges show four within-session sweep medians, not confidence intervals.",
             "Vertical intervals use 2000 paired image-cluster bootstrap replicates, seed 0.",
             "Bootstrap uncertainty concerns these datasets, not transfer to all future scenes.",
             "Synthetic forward input is not a full-pipeline or video benchmark.",
             "The old Stage 3 tensor was 518x518; these timing tensors are 518x686.",
             "Legacy CPU is retained separately; hardware model and individual samples are unavailable.",
             "No equivalence or causal claim is implied by similar computational costs.", ""]
    for row in gains.itertuples():
        part = ratios[ratios['size'] == row.size]
        notes.append(f"- {row.size}: inherited V2-V1 DA-2K gain {row.gain_pp:.3f} pp "
                     f"(image-bootstrap 95% interval {row.ci_lo_pp:.3f} to {row.ci_hi_pp:.3f}); "
                     f"V1/V2 latency ratio spans {part.v1_over_v2.min():.3f} to {part.v1_over_v2.max():.3f}.")
    (out / "FINDINGS.md").write_text("\n".join(notes), encoding="utf-8")
    write_json(out / "validation.json", {"passed": True, "blocks": len(blocks), "timings": len(timings),
                                        "bootstrap_replicates": B, "seed": SEED,
                                        "accuracy_kind": "inherited", "timing_kind": "new GPU synthetic forward"})
    print("Validated 24 blocks / 1200 timings. Analysis written to", out)
    print("\n".join(notes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    analyze(arguments.run_dir)
