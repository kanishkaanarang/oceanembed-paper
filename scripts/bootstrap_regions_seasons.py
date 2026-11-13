"""Block bootstrap (30-day blocks of days, 2000 resamples) for regional and seasonal RMSE differences vs GLORYS (deep set), plus Argo
profile counts per region. Reads only results/scoring/regions_*_daily.csv, regions_n.csv and argo_test_*_rows.csv.gz; no targets are read."""
import os

import numpy as np
import pandas as pd

OUT = "results/scoring/regions_seasons_bootstrap.csv"
if os.path.exists(OUT):
    raise SystemExit(f"EXISTS: {OUT} (stop)")
NB, L = 2000, 30
rng = np.random.default_rng(0)
MODELS = ["clim", "lin", "xgb", "unet0", "frozen", "demean"]
REGS = ["ArabianSea", "BayOfBengal", "Andaman", "Other"]
DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
DG = {"all depths": list(range(15)), "50-200 m": [DEPTHS.index(z) for z in (50, 75, 100, 125, 150, 200)]}
SEAS = ["DJF", "MAM", "JJA", "SON"]
PAIRS = [("frozen", "lin"), ("demean", "frozen")]
ncell = pd.read_csv("results/scoring/regions_n.csv", index_col=0)["n_cells"].to_dict()
ncell["All"] = sum(ncell.values())
cols = [f"sse_{z}" for z in DEPTHS]
rows = []
for per in ("test_a", "test_b"):
    df = pd.read_csv(f"results/scoring/regions_{per}_daily.csv", parse_dates=["date"])
    dates = np.sort(df.date.unique())
    nd = len(dates)
    arr = {m: {} for m in MODELS}
    for m in MODELS:
        for rn in REGS:
            arr[m][rn] = df[(df.model == m) & (df.region == rn)].sort_values("date")[cols].values
        arr[m]["All"] = sum(arr[m][rn] for rn in REGS)
    month = pd.DatetimeIndex(dates).month.values
    smask = {s: (np.array([SEAS[(mm % 12) // 3] for mm in month]) == s).astype(float) for s in SEAS}
    smask["all"] = np.ones(nd)
    nblk = -(-nd // L)
    idx = (rng.integers(0, nd - L + 1, size=(NB, nblk))[:, :, None] + np.arange(L)).reshape(NB, -1)[:, :nd]
    C = np.stack([np.bincount(idx[b], minlength=nd) for b in range(NB)]).astype(float)

    def stat(model, rn, dix, mk):
        S = arr[model][rn][:, dix] * mk[:, None]
        n = ncell[rn] * mk
        pt = np.sqrt((S.sum(0) / n.sum()).mean())
        bs = np.sqrt(((C @ S) / (C @ n)[:, None]).mean(1))
        return pt, bs

    jobs = [(rn, dg, "all") for rn in REGS + ["All"] for dg in DG] + [("All", "all depths", s) for s in SEAS]
    for rn, dg, s in jobs:
        for a, b in PAIRS:
            pa, ba = stat(a, rn, DG[dg], smask[s])
            pb, bb = stat(b, rn, DG[dg], smask[s])
            diff, rel = ba - bb, 100 * (ba - bb) / bb
            rows.append(dict(period=per, region=rn, depths=dg, season=s, a=a, b=b, rmse_a=pa, rmse_b=pb, diff=pa - pb,
                             diff_lo=np.percentile(diff, 2.5), diff_hi=np.percentile(diff, 97.5), rel_pct=100 * (pa - pb) / pb,
                             rel_lo=np.percentile(rel, 2.5), rel_hi=np.percentile(rel, 97.5)))
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
fmt = lambda r: "%+.4f [%+.4f,%+.4f] (%+5.1f%% [%+5.1f,%+5.1f]) %s" % (r["diff"], r.diff_lo, r.diff_hi, r.rel_pct, r.rel_lo, r.rel_hi,
                                                                       "*" if (r.diff_lo > 0 or r.diff_hi < 0) else " ")
for per in ("test_a", "test_b"):
    print(f"\n=== {per} vs GLORYS, REGIONS (all seasons): RMSE(a)-RMSE(b) C, 95% block-bootstrap; negative = a better ===")
    for _, r in out[(out.period == per) & (out.season == "all")].iterrows():
        print(" %-11s %-10s %-6s vs %-6s %s" % (r.region, r.depths, r.a, r.b, fmt(r)))
    print(f"\n=== {per} vs GLORYS, SEASONS (all regions, all depths) ===")
    for _, r in out[(out.period == per) & (out.season != "all")].iterrows():
        print(" %-4s %-6s vs %-6s %s" % (r.season, r.a, r.b, fmt(r)))
print("\n* = interval excludes 0")
# Argo profile counts per region (saved rows; same half-open boxes)
print("\nArgo profiles (unique day, lat, lon) per region, common set:")
for per in ("validation", "test_a", "test_b"):
    r = pd.read_csv(f"results/scoring/argo_{per}_rows.csv.gz", usecols=["day", "lat", "lon"]).drop_duplicates()
    reg = np.where((r.lon >= 45) & (r.lon < 78), "ArabianSea", np.where((r.lon >= 80) & (r.lon < 92) & (r.lat <= 23), "BayOfBengal",
                   np.where((r.lon >= 92) & (r.lon < 100) & (r.lat <= 16), "Andaman", "Other")))
    print(" ", per, dict(pd.Series(reg).value_counts()))
