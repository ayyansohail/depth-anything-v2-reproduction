"""Produce report evidence: paper-vs-ours table + accuracy-vs-latency figure.
Verified references are in results/paper_reported.csv; unavailable references stay blank.
Run:  python scripts/06_make_report_tables.py --dataset nyu
"""
import argparse, os
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="nyu")
args = ap.parse_args()

paper = pd.read_csv("results/paper_reported.csv")
parts = [pd.read_csv(f"results/{args.dataset}_summary.csv")]
if os.path.exists("results/da2k_summary.csv"):
    parts.append(pd.read_csv("results/da2k_summary.csv"))
ours = pd.concat(parts)
tab = paper.merge(ours, on=["model", "dataset", "metric"], how="outer")
tab["gap"] = tab["ours"] - tab["paper_value"]
tab.round(4).to_csv("results/paper_vs_ours.csv", index=False)
print(tab.round(4).to_string(index=False))
print("\nEvery gap needs one sentence of explanation in the report (subset? alignment? resolution?).")

# accuracy vs latency (Type F supporting analysis)
lat = pd.read_csv("results/latency.csv")
acc = ours[(ours.dataset == args.dataset) & (ours.metric == "delta1")][["model", "ours"]]
m = lat.merge(acc, on="model")
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(m["mean_ms"], m["ours"], s=80)
for _, r in m.iterrows():
    ax.annotate(f"{r['model']}\n{r['params_m']}M", (r["mean_ms"], r["ours"]), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.set_xlabel(f"forward latency (ms) on {lat['hardware'].iloc[0]}"); ax.set_ylabel(f"delta1 on {args.dataset}")
plt.tight_layout(); plt.savefig("figures/accuracy_vs_latency.png", dpi=200)
print("saved figures/accuracy_vs_latency.png")
