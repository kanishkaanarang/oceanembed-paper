"""Moving-block bootstrap (30-day blocks, 2000 resamples) over days for RMSE differences between models on the deep set.
Reads only results/scoring/*_daily*.csv (per-day squared-error sums); no targets are read."""
import glob
import os

import numpy as np
import pandas as pd

OUT = "results/scoring/bootstrap_ci.csv"
if os.path.exists(OUT):
    raise SystemExit(f"EXISTS: {OUT} (stop)")
NDEEP, NB, L = 9140, 2000, 30
rng = np.random.default_rng(0)
files = {"test_a": sorted(glob.glob("results/scoring/test_a_daily_rerun_*.csv"))[-1], "test_b": "results/scoring/test_b_daily.csv"}
depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
GROUPS = {"all 15 depths": depths, "0-30 m": [0, 5, 10, 20, 30], "50-200 m": [50, 75, 100, 125, 150, 200],
          "100-150 m": [100, 125, 150], "300-1000 m": [300, 500, 700, 1000]}
PAIRS = [("frozen", "lin"), ("frozen", "xgb"), ("frozen", "unet0"), ("demean", "lin"), ("demean", "frozen")]
rows = []
for per, f in files.items():
    df = pd.read_csv(f)
    models = sorted(df["model"].unique())
    mse = {}
    for m in models:
        d = df[df["model"] == m].sort_values("date")
        mse[m] = d[[f"sse_{z}" for z in depths]].values / NDEEP          # (days, 15)
    nd = len(mse["clim"])
    assert all(len(v) == nd for v in mse.values()), "days differ between models"
    nblk = -(-nd // L)
    idx = (rng.integers(0, nd - L + 1, size=(NB, nblk))[:, :, None] + np.arange(L)).reshape(NB, -1)[:, :nd]
    for gname, gd in GROUPS.items():
        cols = [depths.index(z) for z in gd]
        g = {m: mse[m][:, cols].mean(1) for m in models}
        rm = lambda m, ix=slice(None): np.sqrt(g[m][ix].mean())
        for a, b in PAIRS:
            pt = rm(a) - rm(b)
            ra = np.sqrt(g[a][idx].mean(1))
            rb = np.sqrt(g[b][idx].mean(1))
            diff, rel = ra - rb, 100 * (ra - rb) / rb
            rows.append(dict(period=per, group=gname, a=a, b=b, rmse_a=rm(a), rmse_b=rm(b), diff=pt,
                             diff_lo=np.percentile(diff, 2.5), diff_hi=np.percentile(diff, 97.5),
                             rel_pct=100 * pt / rm(b), rel_lo=np.percentile(rel, 2.5), rel_hi=np.percentile(rel, 97.5)))
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
pd.set_option("display.width", 200)
for per in files:
    print(f"\n=== {per}: RMSE(a) - RMSE(b) in C, 95% block-bootstrap interval; negative = a better ===")
    x = out[out["period"] == per]
    for _, r in x.iterrows():
        sig = "*" if (r.diff_lo > 0 or r.diff_hi < 0) else " "
        print(" %-14s %-6s vs %-6s %+.4f  [%+.4f, %+.4f]  (%+5.1f%% [%+5.1f, %+5.1f]) %s" %
              (r["group"], r.a, r.b, r["diff"], r.diff_lo, r.diff_hi, r.rel_pct, r.rel_lo, r.rel_hi, sig))
print("\n* = interval excludes 0")
