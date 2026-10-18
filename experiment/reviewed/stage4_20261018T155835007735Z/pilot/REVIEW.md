# Pilot review

Accepted for proceeding to the targeted benchmark. This is not a completed
benchmark or Stage 4 finding.

- Source commit: 0d6a6be3eff111dfa81186a5bebd2327c053a2c0, requested experiment branch.
- Colab validation: 22 tests passed in 0.76 seconds.
- All six model pilots passed, using the same seeded 1x3x518x686 FP32 input.
- Output shapes: 1x518x686; outputs finite according to the successful pilot checks.
- All models use eager attention and captured immutable revisions.
- Tesla T4, one torch thread; new environment captured, including torch 2.11.0+cu130.
- All five inherited input checksums match source Git blobs at the execution commit.
- HF authentication and torch_dtype deprecation warnings did not prevent successful loading.

Original ZIP retained as pilot_evidence.zip. Full GPU sweeps and final analysis
are pending. Do not update or restart the active Colab checkout before continuing;
the new timing run remains tied to its captured commit and runtime.
