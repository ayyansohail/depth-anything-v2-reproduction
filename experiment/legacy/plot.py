import pandas as pd, matplotlib.pyplot as plt
acc = pd.read_csv("experiment/accuracy.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, dev in zip(axes, ["cuda", "cpu"]):
    df = pd.read_csv(f"experiment/latency_{dev}.csv").merge(acc, on="model").dropna(subset=["absrel"])
    for ver, style in [("v2", "o-"), ("v1", "s--")]:
        d = df[df.model.str.startswith(ver)]
        if len(d):
            ax.plot(d.fwd_mean_ms, d.absrel, style, label=ver.upper())
            for _, r in d.iterrows():
                ax.annotate(f"{r.model.split('-')[1]} ({r.params_M:.0f}M)", (r.fwd_mean_ms, r.absrel),
                            textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel(f"Forward-pass latency on {dev.upper()} (ms)")
    ax.set_ylabel("NYU AbsRel (lower is better)")
    ax.set_title(dev.upper()); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("experiment/accuracy_vs_latency.png", dpi=200)
