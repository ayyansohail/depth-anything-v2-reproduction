# Stage 4 analysis evidence

Timing execution commit: 0d6a6be3eff111dfa81186a5bebd2327c053a2c0
Accuracy is inherited; the new timing environment does not recover its historical model revisions.
Horizontal ranges show four within-session sweep medians, not confidence intervals.
Vertical intervals use 2000 paired image-cluster bootstrap replicates, seed 0.
Bootstrap uncertainty concerns these datasets, not transfer to all future scenes.
Synthetic forward input is not a full-pipeline or video benchmark.
The old Stage 3 tensor was 518x518; these timing tensors are 518x686.
Legacy CPU is retained separately; hardware model and individual samples are unavailable.
No equivalence or causal claim is implied by similar computational costs.

- small: inherited V2-V1 DA-2K gain 6.480 pp (image-bootstrap 95% interval 5.072 to 7.843); V1/V2 latency ratio spans 0.993 to 1.005.
- base: inherited V2-V1 DA-2K gain 7.834 pp (image-bootstrap 95% interval 6.419 to 9.353); V1/V2 latency ratio spans 1.000 to 1.017.
- large: inherited V2-V1 DA-2K gain 7.834 pp (image-bootstrap 95% interval 6.308 to 9.453); V1/V2 latency ratio spans 0.999 to 1.061.