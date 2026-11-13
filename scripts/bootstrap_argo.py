"""Moving-block bootstrap (30-day blocks of dates, 2000 resamples) for Argo RMSE differences between models, common set.
Reads only results/scoring/argo_test_*_rows.csv.gz; no new targets are read."""
import os

import numpy as np
import pandas as pd

OUT = "results/scoring/argo_bootstrap_ci.csv"
if os.path.exists(OUT):
    raise SystemExit(f"EXISTS: {OUT} (stop)")
NB, L = 2000, 30
rng = np.random.default_rng(0)
MODELS = ["clim", "glorys", "lin", "xgb", "unet0", "frozen", "demean"]
GROUPS = {"all depths": (1, 1000), "5-50 m": (1, 50), "75-200 m": (75, 200), "300-1000 m": (300, 1000)}
PAIRS = [("frozen", "lin"), ("frozen", "xgb"), ("demean", "lin"), ("demean", "frozen"), ("demean", "glorys"), ("frozen", "glorys")]
rows = []
for per in ("test_a", "test_b"):
    r = pd.read_csv(f"results/scoring/argo_{per}_rows.csv.gz", parse_dates=["day"])
    r = r[np.isfinite(r[MODELS]).all(axis=1)]
    d0, d1 = r.day.min(), r.day.max()
    nd = (d1 - d0).days + 1
    r["t"] = (r.day - d0).dt.days
    nblk = -(-nd // L)
    idx = (rng.integers(0, nd - L + 1, size=(NB, nblk))[:, :, None] + np.arange(L)).reshape(NB, -1)[:, :nd]
    for gname, (lo, hi) in GROUPS.items():
        g = r[(r.depth >= lo) & (r.depth <= hi)]
        n_d = np.bincount(g.t, minlength=nd).astype(float)
        sse = {m: np.bincount(g.t, weights=(g[m] - g["argo"]) ** 2, minlength=nd) for m in MODELS}
        N = n_d[idx].sum(1)
        rm = {m: np.sqrt(sse[m].sum() / n_d.sum()) for m in MODELS}
        for a, b in PAIRS:
            ra, rb = np.sqrt(sse[a][idx].sum(1) / N), np.sqrt(sse[b][idx].sum(1) / N)
            diff, rel = ra - rb, 100 * (ra - rb) / rb
            rows.append(dict(period=per, group=gname, a=a, b=b, rmse_a=rm[a], rmse_b=rm[b], diff=rm[a] - rm[b],
                             diff_lo=np.percentile(diff, 2.5), diff_hi=np.percentile(diff, 97.5),
                             rel_pct=100 * (rm[a] - rm[b]) / rm[b], rel_lo=np.percentile(rel, 2.5), rel_hi=np.percentile(rel, 97.5)))
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
for per in ("test_a", "test_b"):
    print(f"\n=== {per} vs ARGO: RMSE(a) - RMSE(b) in C, 95% block-bootstrap interval; negative = a closer to Argo ===")
    for _, r in out[out["period"] == per].iterrows():
        sig = "*" if (r.diff_lo > 0 or r.diff_hi < 0) else " "
        print(" %-11s %-7s vs %-7s %+.4f  [%+.4f, %+.4f]  (%+5.1f%% [%+5.1f, %+5.1f]) %s" %
              (r["group"], r.a, r.b, r["diff"], r.diff_lo, r.diff_hi, r.rel_pct, r.rel_lo, r.rel_hi, sig))
print("\n* = interval excludes 0")
