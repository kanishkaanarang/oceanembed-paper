"""EXTENSION (exploratory, after the main results): the same correction applied to every estimate. Argo-calibrated additive bias correction (pre-registered in configs/preregistration_argo_correction.md, commit 035f117).
Fit: per-depth mean of (model - Argo) on validation D+A rows, deep-set common rows (XGBoost finite), depth > 0. No correction at 0 m.
Evaluate: Argo RMSE corrected vs uncorrected on test_a/test_b/test_c (paired moving-block bootstrap, 30-day blocks, 2000 resamples,
seed 0 restarted per period; the fitted constants are NOT re-estimated inside the bootstrap). Cost vs GLORYS from existing sums:
SSE' = SSE - 2 c B + c^2 n  (e = prediction - GLORYS, B = sum of e). Reads no new targets."""
import glob

import numpy as np
import pandas as pd

OUT = "results/scoring/argo_correction_all_ci.csv"
NB, L = 2000, 30
MODELS = ["clim", "glorys", "lin", "xgb", "unet0", "frozen", "demean"]
ENS = ["frozen", "demean", "lin", "xgb", "unet0", "clim", "glorys"]
GROUPS = {"all depths": (1, 1000), "5-50 m": (1, 50), "75-200 m": (75, 200), "300-1000 m": (300, 1000)}
PAIRS = [("frozen_c", "lin_c"), ("frozen_c", "xgb_c"), ("frozen_c", "unet0_c"), ("frozen_c", "glorys_c"), ("frozen_c", "clim_c"), ("lin_c", "lin"), ("glorys_c", "glorys")]

v = pd.read_csv("results/scoring/argo_validation_rows.csv.gz", parse_dates=["day"])
v = v[v["mode"].isin(["D", "A"]) & (v.depth > 0)]
v = v[np.isfinite(v[MODELS]).all(axis=1)].copy()
v["depth"] = v.depth.astype(int)
c = {m: v.groupby("depth")["e_" + m].mean() for m in ENS}
cnt = v.groupby("depth").size()
print("FITTED CONSTANTS (C; model - Argo, validation 2018-02..2020-01), subtract from predictions")
print(pd.DataFrame({"n_rows": cnt, "c_frozen": c["frozen"], "c_demean": c["demean"]}).round(3).to_string())

files = {"test_a": "results/scoring/argo_test_a_rows.csv.gz", "test_b": "results/scoring/argo_test_b_rows.csv.gz",
         "test_c": sorted(glob.glob("results/scoring/argo_test_c_rerun_*_rows.csv.gz"))[-1]}
rows = []
for per, f in files.items():
    rng = np.random.default_rng(0)
    r = pd.read_csv(f, parse_dates=["day"])
    r = r[np.isfinite(r[MODELS]).all(axis=1)].copy()
    r["depth"] = r.depth.astype(int)
    sq = {m: r["e_" + m] ** 2 for m in ("glorys",) + tuple(ENS)}
    for m in ENS:
        cm = r.depth.map(c[m])
        assert cm.notna().all(), "depth without a fitted constant"
        sq[m + "_c"] = (r["e_" + m] - cm) ** 2
    d0, d1 = r.day.min(), r.day.max()
    nd = (d1 - d0).days + 1
    t = (r.day - d0).dt.days.values
    nblk = -(-nd // L)
    idx = (rng.integers(0, nd - L + 1, size=(NB, nblk))[:, :, None] + np.arange(L)).reshape(NB, -1)[:, :nd]
    for gname, (lo, hi) in GROUPS.items():
        g = ((r.depth >= lo) & (r.depth <= hi)).values
        n_d = np.bincount(t[g], minlength=nd).astype(float)
        sse = {s: np.bincount(t[g], weights=sq[s].values[g], minlength=nd) for s in sq}
        N = n_d[idx].sum(1)
        rm = {s: np.sqrt(sse[s].sum() / n_d.sum()) for s in sse}
        for a, b in PAIRS:
            ra, rb = np.sqrt(sse[a][idx].sum(1) / N), np.sqrt(sse[b][idx].sum(1) / N)
            rel = 100 * (ra - rb) / rb
            rows.append(dict(period=per, group=gname, a=a, b=b, rmse_a=rm[a], rmse_b=rm[b], rel_pct=100 * (rm[a] - rm[b]) / rm[b],
                             rel_lo=np.percentile(rel, 2.5), rel_hi=np.percentile(rel, 97.5), n_rows=int(n_d.sum())))
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print("\nARGO: RMSE change in % (negative = first is closer to Argo), 95% block-bootstrap; * excludes 0")
for per in files:
    print(f"\n=== {per} (n rows {out[out.period == per].n_rows.iloc[0]}) ===")
    for _, x in out[(out.period == per) & (out.group != "300-1000 m")].iterrows():
        sig = "*" if (x.rel_lo > 0 or x.rel_hi < 0) else " "
        print(" %-11s %-9s vs %-7s %.3f vs %.3f  %+6.1f%% [%+6.1f, %+6.1f] %s" % (x.group, x.a, x.b, x.rmse_a, x.rmse_b, x.rel_pct, x.rel_lo, x.rel_hi, sig))

