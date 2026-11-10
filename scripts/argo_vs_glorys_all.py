"""GLORYS vs Argo, profile by profile, 2010-02..2025-12, potential temperature at the 15 target depths.
Target-versus-observation only: no model output is used, so no test period of any model is evaluated."""
import glob
import os

import gsw
import numpy as np
import pandas as pd

OUT = "data/processed/argo_vs_glorys.csv.gz"
if os.path.exists(OUT):
    raise SystemExit(f"EXISTS: {OUT} (stop)")

DEPTHS = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000])
BLOCKS = [("data/processed/glorys_block3.npy", "2010-02-04"),
          ("data/processed/glorys_block4.npy", "2015-10-17"),
          ("data/processed/glorys_block5.npy", "2021-06-28")]
arrs = [(np.load(f, mmap_mode="r"), pd.Timestamp(s)) for f, s in BLOCKS]
print("block shapes:", [a.shape for a, _ in arrs])


def glorys_at(day, i, j):
    for a, s in arrs:
        t = (day - s).days
        if 0 <= t < a.shape[0]:
            return a[t, :, i, j]
    return None


rows = []
for f in sorted(glob.glob("data/interim/argo/argo_*.pkl.gz")):
    df = pd.read_pickle(f)
    for (plat, cyc), p in df.groupby(["PLATFORM_NUMBER", "CYCLE_NUMBER"]):
        lat, lon = float(p.LATITUDE.iloc[0]), float(p.LONGITUDE.iloc[0])
        day = pd.Timestamp(p.TIME.iloc[0]).normalize()
        i, j = int(round((lat - 5) / 0.25)), int(round((lon - 45) / 0.25))
        if not (0 <= i < 101 and 0 <= j < 241):
            continue
        fld = glorys_at(day, i, j)
        if fld is None:
            continue
        p = p.sort_values("PRES").drop_duplicates("PRES")
        p = p[np.isfinite(p.PRES) & np.isfinite(p.PSAL) & np.isfinite(p.TEMP)]
        if len(p) < 2:
            continue
        pres = p.PRES.values
        pt = gsw.pt0_from_t(gsw.SA_from_SP(p.PSAL.values, pres, lon, lat), p.TEMP.values, pres)
        z = -gsw.z_from_p(pres, lat)
        mode = p.DATA_MODE.iloc[0]
        for k, d in enumerate(DEPTHS):
            if d < z.min() or d > z.max() or not np.isfinite(fld[k]):
                continue
            rows.append((day, lat, lon, mode, int(d), float(fld[k]), float(np.interp(d, z, pt))))
    print(os.path.basename(f), "rows so far", len(rows), flush=True)

r = pd.DataFrame(rows, columns=["day", "lat", "lon", "mode", "depth", "glorys", "argo"])
r["diff"] = r["glorys"] - r["argo"]
r["year"] = r["day"].dt.year
r.to_csv(OUT, index=False, compression="gzip")
print("saved", OUT, "| rows", len(r), "| profiles", r.groupby(["day", "lat", "lon"]).ngroups)

da = r[r["mode"].isin(["D", "A"])]
print("\nGLORYS minus Argo by depth, delayed+adjusted modes only, whole period")
print(da.groupby("depth")["diff"].agg(n="size", bias="mean", rmse=lambda x: float(np.sqrt(np.mean(x ** 2)))).round(3).to_string())

e = r[(r.depth == 100) & (r.lon > 75)]
t = e.groupby("year").agg(n=("diff", "size"), argo=("argo", "mean"), glorys=("glorys", "mean"),
                          bias=("diff", "mean"), share_R=("mode", lambda m: float((m == "R").mean())))
print("\n100 m, east of 75E, by year (all modes): mean Argo, mean GLORYS at the same profiles")
print(t.round(3).to_string())

y = r[(r.depth == 100) & (r.year == 2025)]
print("\n100 m in 2025 by data mode")
print(y.groupby("mode")["diff"].agg(n="size", bias="mean").round(3).to_string())
