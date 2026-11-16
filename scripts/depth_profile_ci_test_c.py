"""Per-depth RMSE differences on test_c (vs GLORYS, deep set) with the same moving-block bootstrap as bootstrap_ci_test_c.py
(seed 0, 30-day blocks, 2000 resamples). Reads only results/scoring/test_c_daily.csv; no targets are read."""
import numpy as np
import pandas as pd

OUT = "results/scoring/depth_profile_ci_test_c.csv"
NDEEP, NB, L = 9140, 2000, 30
rng = np.random.default_rng(0)
depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
PAIRS = [("frozen", "lin"), ("frozen", "xgb"), ("frozen", "unet0")]
df = pd.read_csv("results/scoring/test_c_daily.csv")
mse = {m: df[df["model"] == m].sort_values("date")[[f"sse_{z}" for z in depths]].values / NDEEP for m in df["model"].unique()}
nd = len(mse["clim"])
assert all(len(v) == nd for v in mse.values())
nblk = -(-nd // L)
idx = (rng.integers(0, nd - L + 1, size=(NB, nblk))[:, :, None] + np.arange(L)).reshape(NB, -1)[:, :nd]
rows = []
for k, z in enumerate(depths):
    for a, b in PAIRS:
        pa, pb = np.sqrt(mse[a][:, k].mean()), np.sqrt(mse[b][:, k].mean())
        ra, rb = np.sqrt(mse[a][:, k][idx].mean(1)), np.sqrt(mse[b][:, k][idx].mean(1))
        rel = 100 * (ra - rb) / rb
        rows.append(dict(depth_m=z, a=a, b=b, rmse_a=pa, rmse_b=pb, diff=pa - pb, rel_pct=100 * (pa - pb) / pb,
                         rel_lo=np.percentile(rel, 2.5), rel_hi=np.percentile(rel, 97.5)))
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print("test_c, RMSE change of frozen vs b in % (negative = frozen better), 95% block-bootstrap interval; * excludes 0")
for b in ("lin", "xgb", "unet0"):
    print(f"\n--- frozen vs {b} ---")
    for _, r in out[out["b"] == b].iterrows():
        sig = "*" if (r.rel_lo > 0 or r.rel_hi < 0) else " "
        print(" %4d m  %.3f vs %.3f  %+5.1f%% [%+5.1f, %+5.1f] %s" % (r.depth_m, r.rmse_a, r.rmse_b, r.rel_pct, r.rel_lo, r.rel_hi, sig))
