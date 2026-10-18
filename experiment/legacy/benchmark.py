import sys, os, time, json, gc, glob, platform
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, pandas as pd, torch, transformers
from PIL import Image
from dav2.models import load_model, sync, REGISTRY

device = sys.argv[1]                        # "cuda" or "cpu"
only = sys.argv[2:]                         # optional: e.g. v2-small
keys = only if only else list(REGISTRY)
WARM, RUNS = (10, 50) if device == "cuda" else (2, 10)
OUT = f"experiment/latency_{device}.csv"

imgs = sorted(p for p in glob.glob("assets/samples/*")
              if p.lower().endswith((".jpg", ".jpeg", ".png")))
assert imgs, "no image found in assets/samples"
IMG = imgs[0]
img = Image.open(IMG).convert("RGB")
print("Image:", IMG, img.size)

def time_ms(fn):
    for _ in range(WARM):                   # warm-up, thrown away
        fn()
    sync(device)
    t = []
    for _ in range(RUNS):
        sync(device); s = time.perf_counter()
        fn()
        sync(device); t.append((time.perf_counter() - s) * 1000)
    return np.mean(t), np.std(t), np.median(t)

rows = pd.read_csv(OUT).to_dict("records") if os.path.exists(OUT) else []
done = {r["model"] for r in rows}

for key in keys:
    if key in done:
        print("skip", key); continue
    if device == "cuda":
        torch.cuda.empty_cache()
    m = load_model(key, device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    x = m.processor(images=img, return_tensors="pt")["pixel_values"].to(device)  # not timed
    out = m.forward_raw(x)
    assert torch.isfinite(out).all(), "non-finite output"

    f_mean, f_std, f_med = time_ms(lambda: m.forward_raw(x))          # forward only
    peak = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None
    p_mean, p_std, p_med = time_ms(lambda: m.predict(img))            # full pipeline

    rows.append({
        "model": key, "device": device, "image": os.path.basename(IMG),
        "input_shape": str(tuple(x.shape)), "params_M": m.params_m,
        "fp32_MB": m.params_m * 4,
        "fwd_mean_ms": f_mean, "fwd_std_ms": f_std, "fwd_median_ms": f_med,
        "full_mean_ms": p_mean, "full_std_ms": p_std, "full_median_ms": p_med,
        "peak_mem_MB": peak,
    })
    print(rows[-1])
    pd.DataFrame(rows).to_csv(OUT, index=False)       # saved after every model
    del m, x, out
    gc.collect()

env = {"torch": torch.__version__, "transformers": transformers.__version__,
       "python": platform.python_version(), "threads": torch.get_num_threads(),
       "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
       "image": os.path.basename(IMG), "warmup": WARM, "runs": RUNS}
json.dump(env, open(f"experiment/env_{device}.json", "w"), indent=2)
