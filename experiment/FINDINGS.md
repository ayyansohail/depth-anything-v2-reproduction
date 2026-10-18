# Reviewed Stage 4 findings

**Option F: computational analysis. Branch: feat/computational-analysis-experiment.**

The experiment supports a useful paper-specific finding: the inherited DA-2K
accuracy gain of V2 over V1 is accompanied by similar observed computational
cost at matching model sizes in this T4 test. Within V2, larger models improve
NYU accuracy while requiring substantially more inference time. These are
observational comparisons of released models, not causal training-recipe or
statistical-equivalence claims.

## Question and evidence

How much relative-depth accuracy does V2 gain over V1 at matching model sizes,
what inference cost accompanies that gain, and how does the accuracy-cost
tradeoff change across V2 Small, Base, and Large?

The [reviewed run](reviewed/stage4_20261018T155835007735Z/final/) contains all 24
measurement blocks, 1,200 individual timings, 22 passing Colab tests, captured
environment/model revisions and successful analysis validation. Execution commit:
`0d6a6be3eff111dfa81186a5bebd2327c053a2c0`. The later evidence-import commit does
not replace that recorded execution revision.

New timings: one Tesla T4 session, FP32, batch one, seeded synthetic tensor
1x3x518x686, eager attention, one torch CPU thread, 10 warm-ups and 50 measured
calls per block. A/B/B/A sweeps balance the forward and reversed model orders.
Accuracy is inherited from Stage 3. The teammate's CPU runs remain legacy evidence.

## Matching-size V1 versus V2

| Size | V1 DA-2K accuracy | V2 DA-2K accuracy | Gain (pp), paired image-bootstrap 95% interval | V1 GPU ms | V2 GPU ms |
|---|---:|---:|---|---:|---:|
| Small | 88.83% | 95.31% | 6.48 [5.07, 7.84] | 76.85 | 77.11 |
| Base | 89.41% | 97.24% | 7.83 [6.42, 9.35] | 212.45 | 212.29 |
| Large | 89.26% | 97.10% | 7.83 [6.31, 9.45] | 607.05 | 606.05 |

GPU values above are medians across four sweep medians, not means of all calls.
Accuracy intervals use 2,000 paired image-cluster bootstrap replicates, seed 0,
on 1,033 images and 2,068 annotated pairs. They preserve multiple annotations
from the same image. All three gain intervals exclude zero on this benchmark.
The new summaries also record identical parameter counts and the same maximum
allocated memory at matching sizes. Similar measured costs do not prove exact
latency equivalence or isolate why the accuracy differs.

## Order and session variation

Across all four sweeps, V1/V2 median-latency ratios span 0.993-1.005 (Small),
1.000-1.017 (Base), and 0.999-1.061 (Large). The largest discrepancy occurs in
the first A sweep. In the later three sweeps, ratios are all within about 0.7%
of one. The final A sweep does not reproduce the first A sweep's Large-model
gap, so fixed execution order alone does not explain it.

Several first-sweep timings are lower than later timings. GPU clocks, load or
session drift are plausible explanations, but no profiling or clock-control
experiment isolates a cause. Keep all four sweeps; do not discard the first
post hoc. The teammate's earlier 3-13% V1 GPU disadvantage is not consistently
reproduced by this controlled comparison.

## V2 model-size tradeoff

| V2 size | Inherited NYU AbsRel | T4 forward ms | Legacy CPU forward seconds |
|---|---:|---:|---:|
| Small | 0.061712 | 77.11 | 2.25 |
| Base | 0.057324 | 212.29 | 6.48 |
| Large | 0.054759 | 606.05 | 22.38 |

Small to Base reduces AbsRel by about 7.1%; Base to Large reduces it by about
4.5%. The new GPU median increases about 2.75x and 2.85x respectively. Paired
NYU image-bootstrap intervals for the AbsRel differences exclude zero:
Base minus Small [-0.005434, -0.003392], and Large minus Base
[-0.003297, -0.001834]. This supports improvement on these test images; it does
not establish that the gain is worthwhile for every application.

Small is the lowest-cost V2 option here. Base is an intermediate choice; Large
has the best measured NYU accuracy at the highest cost. No model is universally
best without a latency/quality requirement. DA-2K is a separate metric and
benchmark: Base's slightly higher point accuracy than Large should not be
interpreted as the same model ranking as NYU or as a proven difference.

## Figures and interpretation

- [DA-2K accuracy versus GPU latency](reviewed/stage4_20261018T155835007735Z/final/analysis/da2k_accuracy_vs_gpu_latency.png)
- [NYU accuracy versus GPU latency](reviewed/stage4_20261018T155835007735Z/final/analysis/nyu_accuracy_vs_gpu_latency.png)
- [NYU accuracy versus legacy CPU latency](reviewed/stage4_20261018T155835007735Z/final/analysis/nyu_accuracy_vs_legacy_cpu.png)
- [V1/V2 ratios by sweep and order](reviewed/stage4_20261018T155835007735Z/final/analysis/order_sensitivity.png)

Horizontal ranges are the observed range of four sweep medians, not confidence
intervals. Vertical intervals are image-bootstrap intervals for inherited
accuracy. Use the paired-difference tables to assess gains rather than inferring
significance from whether separate model intervals overlap.

## Limits and completion

One shared T4 session and synthetic input shape cannot establish hardware-wide
performance, throughput of a video pipeline, or end-to-end app latency. Timing
calls within a block are correlated; the analysis does not treat them as
independent experimental runs. There are no new full-predict or CPU measurements.

The original Stage 3 tensor was 518x518, the teammate used a real image processed
to 518x686, and this run uses a synthetic tensor at 518x686. Software stacks also
differ: this run captured torch 2.11.0+cu130, whereas the teammate recorded
2.11.0+cu128. Those differences prevent attributing timing changes to one cause.

The historical accuracy environment and exact model revisions remain unavailable.
New timings are connected to inherited accuracy by model ID, not proof of an
identical historical revision. KITTI was not evaluated. Original teammate
artifacts and missing CPU hardware metadata are retained transparently.

The planned experimental execution and evidence review are complete. The final
course report, presentation, and deployment remain separate tasks. No local
Python, tests, model inference or numerical-analysis reruns were used for this
review. Figures were visually inspected and the returned manifest and pilot
were checked against the earlier pilot archive.
