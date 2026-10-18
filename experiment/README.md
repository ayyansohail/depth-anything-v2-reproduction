# Stage 4 computational analysis (Option F)

Branch: **feat/computational-analysis-experiment**

Question: How much relative-depth accuracy does V2 gain over V1 at matching model
sizes, what inference cost accompanies that gain, and how does the accuracy-cost
tradeoff change across V2 Small, Base, and Large?

Paper connection: the paper offers multiple model sizes and emphasizes improved
relative-depth predictions. We examine accuracy gained versus computational cost
against V1 at the same size, then compare the V2 deployment tradeoffs. This does
not isolate a causal effect of the training recipe: V1 and V2 differ in more
than training data, and their implementations may behave differently.

## Evidence and status

- `legacy/`: teammate's original ZIP and unchanged extracted artifacts. See its README for missing traceability.
- `../results/nyu_per_image.csv` and `../results/da2k_pairs.csv`: inherited accuracy evidence. No new accuracy run is claimed.
- `runs/`: fresh Colab runs, ignored by Git until reviewed and explicitly imported into `reviewed/`.
- New pilot, timings and analysis: **complete and reviewed**. See [FINDINGS.md](FINDINGS.md) and [the reviewed run](reviewed/stage4_20261018T155835007735Z/REVIEW.md): 22 tests passed, 24 blocks and 1,200 timings validated in Colab.

## Start in Colab

Open `notebooks/stage4_computational_analysis.ipynb` from this branch and choose
Runtime > Change runtime type > T4 GPU. Execute setup, validation, capture,
pilot, and pilot-download cells **one at a time**. Return the pilot ZIP to the
assistant. Only after pilot review, set PILOT_REVIEWED to True and run the full
benchmark. Download failures as well as successful evidence before disconnecting.
Do not run the old colab_runner.ipynb or rerun Stage 3 evaluation for this step.

## Fixed new GPU protocol

- Six models, FP32, inference/evaluation mode, batch one, synthetic standard-normal input of shape 1x3x518x686, seed zero.
- Order A: V2-Small, V1-Small, V2-Base, V1-Base, V2-Large, V1-Large. Order B reverses A. Sweeps: A, B, B, A.
- Each block: ten warm-ups, fifty timed forward calls; CUDA synchronized before/after each call.
- Twenty-four blocks / 1,200 timings, saved after each block. Standard deviation uses sample ddof=1; legacy NumPy used ddof=0.
- Measure allocated peak GPU memory in MiB, including loaded weights and active input/output. Not device-wide or reserved memory.
- No autocast, compilation, training, input preprocessing, resizing, app or network overhead in the forward timer.
- Pin HF revisions for the new session. Runtime changes, mixed evidence, missing blocks, non-finite outputs and CPU fallback cause a stop.

If a run fails, download its ZIP and return the log. Do not restart cells with
the same run directory or splice in measurements from another runtime. A rerun
gets a new capture and pilot. Model weights are downloaded only in Colab; they
are never included in evidence ZIPs or committed to Git.

## Analysis

DA-2K analysis resamples image clusters, keeping a sampled image's annotated
pairs and V1/V2 correctness paired, with 2,000 replicates and seed zero. It
estimates pair-weighted accuracy and matching-size gain intervals. NYU uses
paired per-image metrics and reports size differences. Uncertainty pertains to
these test images; it does not establish generalization to all future scenes.

GPU summaries use per-sweep medians, their median and range, and V1/V2 ratios
within each sweep. They do not treat 1,200 calls as independent experiments or
claim statistical equivalence from a small timing difference. Plots distinguish
within-session timing ranges from accuracy bootstrap intervals. All devices
and timing populations are labeled; legacy CPU results stay separate.

The teammate used a real image preprocessed to 518x686; the new input is synthetic
and used for computation only. Older Stage 3 timings used a smaller 518x518
random tensor. Shape explains a workload difference, not the entire measured
timing increase. Legacy decimal MB values must be divided by 1.048576 to obtain
MiB. No further CPU or full predict benchmark is performed.

## Colab commands (the notebook supplies paths and logging)

```bash
python experiment/benchmark_stage4.py --phase capture --run-dir experiment/runs/UNIQUE_RUN
python experiment/benchmark_stage4.py --phase pilot --run-dir experiment/runs/UNIQUE_RUN
# Return pilot evidence before this command:
python experiment/benchmark_stage4.py --phase run --run-dir experiment/runs/UNIQUE_RUN
python experiment/analyze_stage4.py --run-dir experiment/runs/UNIQUE_RUN
```

The notebook setup deliberately keeps the CUDA-enabled torch already installed
in Colab and installs the other direct dependencies. `manifest.json` records
the actual new environment; `requirements.snapshot.txt` is a current snapshot,
not a recovered historical environment or a guaranteed portable lockfile.

The original timing image, CPU hardware model, and historical accuracy model
revisions are unavailable. Therefore new timing and inherited accuracy refer
to matching model IDs, without proof that every historical artifact matches
the exact current revision. This limitation stays visible in generated tables.

## Completion and submission

The returned run meets the completion requirements: successful pilot, full benchmark, passing Colab checks,
validated analysis, and reviewed artifacts committed under `experiment/reviewed/`.
Keep source and results on this branch for submission. Submit its direct URL:
https://github.com/maaz-zubairi/depth-anything-v2-repro/tree/feat/computational-analysis-experiment

No local Python, tests, installation or model execution was used to prepare this
implementation. Colab checks passed for the archived execution. Report
editing and Stage 5 deployment are outside this task.
