"""Run in Colab: pytest -q tests/test_metrics.py tests/test_stage4_analysis.py."""

import numpy as np
import pandas as pd
import pytest

from experiment.common import CONFIG, expected_blocks, validate_samples
from experiment.analyze_stage4 import paired_cluster_bootstrap, validate_accuracy


def samples():
    return [{"sweep": 1, "position": 1, "model": "v2-small", "iteration": i,
             "latency_ms": 2.0 + i / 100} for i in range(1, 51)]


def test_complete_schedule_has_each_model_once_per_sweep():
    schedule = expected_blocks()
    assert len(schedule) == 24
    for sweep in range(1, 5):
        models = [m for s, p, m in schedule if s == sweep]
        assert len(models) == len(set(models)) == 6
    assert [m for s, p, m in schedule if s == 1] == [m for s, p, m in schedule if s == 4]
    assert [m for s, p, m in schedule if s == 2] == list(reversed([m for s, p, m in schedule if s == 1]))


def test_timing_validation_rejects_missing_or_duplicate_iterations():
    rows = samples()
    validate_samples(rows, 1, 1, "v2-small")
    with pytest.raises(ValueError, match="expected"):
        validate_samples(rows[:-1], 1, 1, "v2-small")
    rows[-1]["iteration"] = 1
    with pytest.raises(ValueError, match="Duplicate"):
        validate_samples(rows, 1, 1, "v2-small")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0, -1])
def test_timing_validation_rejects_invalid_durations(bad):
    rows = samples()
    rows[0]["latency_ms"] = bad
    with pytest.raises(ValueError, match="positive"):
        validate_samples(rows, 1, 1, "v2-small")


def test_timing_validation_rejects_wrong_model():
    rows = samples()
    rows[0]["model"] = "v1-small"
    with pytest.raises(ValueError, match="wrong sweep"):
        validate_samples(rows, 1, 1, "v2-small")


def test_image_cluster_resampling_keeps_models_paired():
    counts = np.array([1, 3, 2])
    sums = np.array([[1, 1], [2, 2], [0, 0]])
    results = paired_cluster_bootstrap(counts, sums, np.random.default_rng(0), 100)
    np.testing.assert_array_equal(results[:, 0], results[:, 1])


def test_image_cluster_bootstrap_is_pair_weighted_and_seeded():
    # One successful pair versus nine failed pairs. It must not average image
    # accuracies (0.5): the bootstrap means must be 0, 0.1, or 1 only.
    counts, sums = [1, 9], [[1], [0]]
    first = paired_cluster_bootstrap(counts, sums, np.random.default_rng(7), 200)
    second = paired_cluster_bootstrap(counts, sums, np.random.default_rng(7), 200)
    np.testing.assert_array_equal(first, second)
    assert set(np.round(first[:, 0], 8)) <= {0.0, 0.1, 1.0}
    assert 0.1 in first[:, 0]


def test_saved_accuracy_inputs_are_paired_and_valid():
    from experiment.common import ROOT
    nyu = pd.read_csv(ROOT / "results/nyu_per_image.csv")
    pairs = pd.read_csv(ROOT / "results/da2k_pairs.csv")
    validate_accuracy(nyu, pairs)
    duplicate = pd.concat([pairs, pairs.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_accuracy(nyu, duplicate)
    invalid = pairs.copy()
    invalid.loc[0, "correct"] = 2
    with pytest.raises(ValueError, match="binary"):
        validate_accuracy(nyu, invalid)
    missing = pairs.iloc[1:].copy()
    with pytest.raises(ValueError, match="2068|paired"):
        validate_accuracy(nyu, missing)

@pytest.fixture
def complete_run(tmp_path):
    import csv
    import statistics
    from experiment.common import BRANCH, ORDERS, file_hashes, write_json
    models = list(dict.fromkeys(m for _, _, m in expected_blocks()))
    write_json(tmp_path / "manifest.json", {
        "config": CONFIG, "orders": ORDERS, "source_branch": BRANCH,
        "source_commit": "fixture-not-a-real-run", "run_id": "test-run",
        "runtime": {"gpu": "Tesla T4"}, "input_hashes": file_hashes(),
        "model_revisions": {m: "fixture-revision" for m in models}})
    write_json(tmp_path / "run_state.json", {"status": "complete"})
    write_json(tmp_path / "pilot.json", {"passed": True, "models": [
        {"model": m, "input_sha256": "fixture-input", "attention_backend": "fixture-backend"}
        for m in models]})
    for sweep, position, model in expected_blocks():
        rows = [{"run_id": "test-run", "sweep": sweep, "position": position, "model": model,
                 "order": "A" if sweep in (1, 4) else "B", "iteration": i,
                 "latency_ms": 1 + i / 100} for i in range(1, 51)]
        values = [r["latency_ms"] for r in rows]
        folder = tmp_path / "blocks" / f"s{sweep}_p{position}_{model}"
        folder.mkdir(parents=True)
        with (folder / "timings.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(folder / "block.json", {
            "run_id": "test-run", "sweep": sweep, "position": position, "model": model,
            "shape": CONFIG["shape"], "dtype": CONFIG["dtype"], "warmup": CONFIG["warmup"],
            "n": CONFIG["iterations"], "revision": "fixture-revision",
            "input_sha256": "fixture-input", "attention_backend": "fixture-backend",
            "mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
            "std_ms": statistics.stdev(values), "peak_allocated_mib": 100, "params_m": 25})
    return tmp_path


def test_complete_benchmark_evidence_validates(complete_run):
    from experiment.analyze_stage4 import load_timings
    _, blocks, timings = load_timings(complete_run)
    assert len(blocks) == 24 and len(timings) == 1200


@pytest.mark.parametrize("field,value,reason", [
    ("run_id", "different-session", "Mixed run IDs"),
    ("shape", [1, 3, 518, 518], "shape or dtype"),
    ("revision", "different-model-revision", "revision mismatch"),
    ("mean_ms", 999, "summary differs"),
])
def test_benchmark_rejects_mixed_or_corrupted_blocks(complete_run, field, value, reason):
    from experiment.common import read_json, write_json
    from experiment.analyze_stage4 import load_timings
    path = complete_run / "blocks/s1_p1_v2-small/block.json"
    block = read_json(path)
    block[field] = value
    write_json(path, block)
    with pytest.raises(ValueError, match=reason):
        load_timings(complete_run)


def test_benchmark_rejects_incomplete_run(complete_run):
    from experiment.common import write_json
    from experiment.analyze_stage4 import load_timings
    write_json(complete_run / "run_state.json", {"status": "failed"})
    with pytest.raises(ValueError, match="incomplete"):
        load_timings(complete_run)
