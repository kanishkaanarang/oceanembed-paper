"""EXPLORATORY (not pre-registered): frozen vs XGBoost per depth on test_a and test_b, to check whether the 30-75 m weakness seen on
test_c is also present there. Same method as depth_profile_ci_test_c.py (30-day blocks, 2000 resamples, seed 0 restarted per period).
Reads only per-day squared-error sums (results/scoring/*_daily*.csv); no targets are read."""
import glob

import numpy as np
import pandas as pd

OUT = "results/scoring/depth_profile_xgb_test_ab.csv"
NDEEP, NB, L = 9140, 2000, 30
depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
files = {"test_a": sorted(glob.glob("results/scoring/test_a_daily_rerun_*.csv"))[-1], "test_b": "results/scoring/test_b_daily.csv"}
rows = []
for per, f in files.items():
    rng = np.random.default_rng(0)
    df = pd.read_csv(f)
    mse = {m: df[df["model"] == m].sort_values("date")[[f"sse_{z}" for z in depths]].values / NDEEP for m in ("frozen", "xgb")}
    nd = len(mse["frozen"])
    nblk = -(-nd // L)
    idx = (rng.integers(0, nd - L + 1, size=(NB, nblk))[:, :, None] + np.arange(L)).reshape(NB, -1)[:, :nd]
    for k, z in enumerate(depths):
        pa, pb = np.sqrt(mse["frozen"][:, k].mean()), np.sqrt(mse["xgb"][:, k].mean())
        ra, rb = np.sqrt(mse["frozen"][:, k][idx].mean(1)), np.sqrt(mse["xgb"][:, k][idx].mean(1))
        rel = 100 * (ra - rb) / rb
        rows.append(dict(period=per, depth_m=z, rmse_frozen=pa, rmse_xgb=pb, rel_pct=100 * (pa - pb) / pb,
                         rel_lo=np.percentile(rel, 2.5), rel_hi=np.percentile(rel, 97.5)))
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print("EXPLORATORY: RMSE change of frozen vs XGBoost in % (negative = frozen better), vs GLORYS; * excludes 0")
for per in files:
    print(f"\n--- {per} ---")
    for _, r in out[out["period"] == per].iterrows():
        sig = "*" if (r.rel_lo > 0 or r.rel_hi < 0) else " "
        print(" %4d m  %.3f vs %.3f  %+5.1f%% [%+5.1f, %+5.1f] %s" % (r.depth_m, r.rmse_frozen, r.rmse_xgb, r.rel_pct, r.rel_lo, r.rel_hi, sig))
