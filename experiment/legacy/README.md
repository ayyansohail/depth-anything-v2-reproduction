# Teammate experiment evidence

These files are preserved unchanged from the team's supplied experiment_results.zip.
The original ZIP is retained alongside the extracted files. This is inherited evidence,
not output of the new Stage 4 benchmark. See ../README.md for the current experiment.

The archive contains two GPU aggregate timing CSVs, one CPU CSV, environments,
benchmark/plot scripts, a question, copied NYU accuracy, and a plot. It does not
contain the timing image, individual timing samples, execution commit, or the
RESULTS_LOG.md mentioned by the accompanying report. Scripts depend on the parent
repository and the missing sample image; they are preserved, not invoked by the
new notebook. CPU hardware model and historical model revisions are not recorded.

Legacy CUDA inputs have shape (1, 3, 518, 686). The older Stage 3 benchmark used
(1, 3, 518, 518). Legacy memory is decimal MB; Stage 3 stored MiB. The new
benchmark uses MiB consistently. Full predict measurements are not complete app
latency; separate timing populations can yield an apparent full/forward reversal.

No training or fine-tuning was performed. The legacy environment does not establish
the earlier Stage 3 accuracy environment.
Archive SHA256: `ce706320003c9a531dfb7cdbb8096ad8f70c12d301b131d76d486300d87faf71`
