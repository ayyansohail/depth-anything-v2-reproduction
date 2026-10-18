"""Dense-depth evaluation (NYU / KITTI) with the relative-depth protocol.
Run:  python scripts/02_eval_depth.py --dataset nyu --data-root data/nyu --eigen-crop --limit 10  # sanity check
      python scripts/02_eval_depth.py --dataset nyu --data-root data/nyu --eigen-crop             # full 654
Writes results/<dataset>_per_image.csv and results/<dataset>_summary.csv (long format).
"""
import argparse, gc
import pandas as pd, torch
from tqdm import tqdm
from dav2.data import get_dataset
from dav2.metrics import eigen_crop_mask, eval_relative
from dav2.models import V2_KEYS, load_model

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["nyu", "kitti"], default="nyu")
ap.add_argument("--data-root", required=True)
ap.add_argument("--models", nargs="+", default=V2_KEYS)
ap.add_argument("--limit", type=int, default=None, help="evenly spaced subset; omit for full set")
ap.add_argument("--eigen-crop", action="store_true", help="NYU only; check paper/official code first")
ap.add_argument("--out-dir", default="results")
args = ap.parse_args()

ds = get_dataset(args.dataset, args.data_root, args.limit)
print(f"{args.dataset}: evaluating {len(ds)} images x {args.models}")
rows = []
for key in args.models:
    model = load_model(key)
    for i in tqdm(range(len(ds)), desc=key):
        s = ds[i]
        pred = model.predict(s["image"])
        crop = eigen_crop_mask(*pred.shape) if (args.eigen_crop and args.dataset == "nyu") else None
        m = eval_relative(pred, s["depth"], ds.min_depth, ds.max_depth, extra_mask=crop)
        if m is not None:
            rows.append({"model": key, "dataset": args.dataset, "image": s["id"], **m})
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

df = pd.DataFrame(rows)
suffix = f"_n{args.limit}" if args.limit else ""
df.to_csv(f"{args.out_dir}/{args.dataset}_per_image{suffix}.csv", index=False)
summ = (df.groupby("model")[["absrel", "rmse", "delta1"]].mean()
          .reset_index().melt(id_vars="model", var_name="metric", value_name="ours"))
summ["dataset"] = args.dataset + (f"_n{args.limit}" if args.limit else "")
summ["n_images"] = df.groupby("model")["image"].count().reindex(summ["model"]).values
summ.to_csv(f"{args.out_dir}/{args.dataset}_summary{suffix}.csv", index=False)
print(summ.pivot(index="model", columns="metric", values="ours").round(4))
