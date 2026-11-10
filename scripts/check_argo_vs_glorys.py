"""Argo vs GLORYS at the 15 target depths for one validation month (2019-03): tests the gsw conversion and the date/cell alignment."""
import gsw
import numpy as np
import pandas as pd
from argopy import DataFetcher

DEPTHS = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000])
BLOCK4_START = pd.Timestamp("2015-10-17")
g = np.load("data/processed/glorys_block4.npy", mmap_mode="r")
print("block4 shape", g.shape)

df = DataFetcher(src="erddap").region([45, 105, 5, 30, 0, 1000, "2019-03-01", "2019-04-01"]).to_dataframe()
rows, dt = [], []
for (plat, cyc), p in df.groupby(["PLATFORM_NUMBER", "CYCLE_NUMBER"]):
    p = p.sort_values("PRES")
    lat, lon = float(p.LATITUDE.iloc[0]), float(p.LONGITUDE.iloc[0])
    day = pd.Timestamp(p.TIME.iloc[0]).normalize()
    ti, i, j = (day - BLOCK4_START).days, int(round((lat - 5) / 0.25)), int(round((lon - 45) / 0.25))
    if not (0 <= ti < g.shape[0] and 0 <= i < 101 and 0 <= j < 241):
        continue
    pres, sp, t = p.PRES.values, p.PSAL.values, p.TEMP.values
    SA = gsw.SA_from_SP(sp, pres, lon, lat)
    pt = gsw.pt0_from_t(SA, t, pres)
    z = -gsw.z_from_p(pres, lat)
    deep = pres >= 900
    if deep.any():
        dt.append(float(np.mean(t[deep] - pt[deep])))
    fld = g[ti, :, i, j]
    for k, d in enumerate(DEPTHS):
        if d < z.min() or d > z.max() or not np.isfinite(fld[k]):
            continue
        rows.append((plat, cyc, p.DATA_MODE.iloc[0], int(d), float(fld[k] - np.interp(d, z, pt))))

r = pd.DataFrame(rows, columns=["plat", "cyc", "mode", "depth", "diff"])
print("profiles used:", r.groupby(["plat", "cyc"]).ngroups, "of", df.groupby(["PLATFORM_NUMBER", "CYCLE_NUMBER"]).ngroups)
print("mean (in-situ T minus potential T) at >=900 dbar:", round(float(np.mean(dt)), 3), "C")
s = r.groupby("depth")["diff"].agg(n="size", bias="mean", rmse=lambda x: float(np.sqrt(np.mean(x ** 2))))
print("GLORYS minus Argo (potential temperature), all modes")
print(s.round(3).to_string())
print("pooled RMSE all depths:", round(float(np.sqrt((r["diff"] ** 2).mean())), 3), "| n", len(r))
