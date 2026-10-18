# Stage 3 report corrections

These replacement passages describe the existing evaluation evidence. They do
not assert new experiments or rewrite the submitted PDF. Retain the team's
voice when integrating them into the editable report source.

## Scope and datasets

Our instructor approved the proposal's evaluation-only scope. We use released
relative-depth checkpoints and implement their evaluation, rather than train
or fine-tune teacher or student models. The final NYU evaluation covers 654
images for each V2 model size. The saved DA-2K records cover 1,033 unique images
and 2,068 annotated pairs for each of six V1/V2 model variants.

The proposal originally included NYU and KITTI. The current submission contains
NYU and DA-2K results and no KITTI results. DA-2K extends ordinal evaluation
across diverse scenes but does not replace KITTI's outdoor dense-depth metrics.
The team should add its actual reason for this change and its approval status;
these cannot be inferred from the saved result files.

## Method and evaluation

We load relative inverse-depth checkpoints through Hugging Face Transformers.
They do not output distance in metres. For NYU scoring, we fit per-image scale
and shift to inverse ground-truth depth, apply the documented disparity floor,
and invert before calculating depth metrics. This ground-truth alignment is
part of benchmark scoring; it is not a way to obtain metric distance for a
new uploaded image without ground truth.

Our implementation follows a documented local relative-depth protocol. It is
not an exact replication of the authors' evaluation code. Dataset packaging,
preprocessing, interpolation, masks, and alignment details may differ; their
individual effects have not been quantified. See EVAL_PROTOCOL.md.

The recorded final NYU command explicitly includes `--eigen-crop`. This flag
is optional in the script, not enabled by default. On DA-2K, `--limit` counts
images, not pairs, so any subset must be described using the actual output count.

## Results to include

The local values below are rounded from `results/nyu_summary.csv`. Paper
references are from Table 2 of [Yang et al. 2024, arXiv v2](https://arxiv.org/html/2406.09414v2).

| V2 model | Local AbsRel | Paper AbsRel | Local delta1 | Paper delta1 | Local RMSE (m) |
|---|---:|---:|---:|---:|---:|
| Small | 0.061712 | 0.053 | 0.955580 | 0.973 | 0.296764 |
| Base | 0.057324 | 0.049 | 0.959332 | 0.976 | 0.283095 |
| Large | 0.054759 | 0.045 | 0.960946 | 0.979 | 0.277048 |

RMSE is an additional aligned metric, averaged per image. Do not compare it
with a value from the paper's metric-depth fine-tuning table.

V2 DA-2K overall accuracy is 95.3095%, 97.2437%, and 97.0986% for Small,
Base, and Large respectively. Table 3 reports 95.3%, 97.0%, and 97.1%.
The differences are approximately +0.0095, +0.2437, and -0.0014 percentage
points. These close aggregate scores do not establish identical predictions.

V1 local accuracy is 88.8298%, 89.4101%, and 89.2650% for Small, Base, and
Large. Table 3 gives one V1 baseline of 88.5%, without naming its encoder in
that table. Do not present a V1-Base paper reference of 89.4% without an exact
source. Until model identity is established, leave encoder-specific V1 paper
comparisons blank. The blanket claim that every model differs by only a few
tenths of a percentage point should be removed.

## Debugging interpretation

Replace DA-2K AbsRel references in the anomaly discussion: DA-2K is evaluated
using pairwise accuracy, so the origin of those AbsRel values must be identified
before retaining them. Do not relabel unexplained numbers by guessing.

The crop diagnostic supports the conclusion that enabling the crop resolved
the observed NYU model-ranking anomaly under our protocol. It changes both
alignment pixels and scoring pixels, so it does not isolate which effect
contributed most. The suggested explanation involving larger-model border
uncertainty remains a hypothesis. Bilinear interpolation eliminated the
observed overshoot but was not the main explanation for the ranking anomaly.

## Literature wording and limitations

The synthetic data supervise the teacher; the teacher then pseudo-labels real
images used to train students. Avoid describing student training as using
pseudo-labels on synthetic images only. Also use one consistent checkpoint
description throughout: the evaluated models are relative-depth models,
not metric-head checkpoints.

Versions in requirements.txt are compatibility ranges. An exact historical
Colab environment, model revisions, dataset hashes, and execution-time code
revision have not been archived in the current checkout. Git commits recording
results provide traceability but are not proof of the exact code used at run
time. Add any recovered evidence with its actual context; do not label a fresh
Colab snapshot as the historical environment.

Existing latency results concern a bare forward pass on a fixed 518x518 input.
They exclude image processing, device transfers of uploaded images, output
resizing, visualization, and app overhead. They can support a later computational
experiment, but should not be described as end-to-end app latency.
