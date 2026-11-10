"""Write small, committable Argo-vs-GLORYS tables from data/processed/argo_vs_glorys.csv.gz. Target vs observation; no model output."""
import os

import numpy as np
import pandas as pd

OUT1, OUT2 = "results/argo_vs_glorys_by_depth.csv", "results/argo_vs_glorys_by_year_100m.csv"
for f in (OUT1, OUT2):
    if os.path.exists(f):
        raise SystemExit(f"EXISTS: {f} (stop)")

r = pd.read_csv("data/processed/argo_vs_glorys.csv.gz", parse_dates=["day"])
r["year"] = r["day"].dt.year
da = r[r["mode"].isin(["D", "A"])]
by_depth = da.groupby("depth")["diff"].agg(n="size", bias="mean", rmse=lambda x: float(np.sqrt(np.mean(x ** 2))))
by_depth.round(4).to_csv(OUT1)
print("GLORYS minus Argo by depth, D+A, whole period\n", by_depth.round(3).to_string())

e = r[(r.depth == 100) & (r.lon > 75)].copy()
e["season"] = (e.day.dt.month % 12) // 3
e["cell"] = np.floor(e.lat / 2).astype(int).astype(str) + "_" + np.floor(e.lon / 2).astype(int).astype(str) + "_" + e.season.astype(str)
g = e.groupby("cell")
e["n_grp"] = g["argo"].transform("size")
e["argo_anom"] = e["argo"] - g["argo"].transform("mean")
e["glorys_anom"] = e["glorys"] - g["glorys"].transform("mean")
a = e[e.n_grp >= 10]
tab = e.groupby("year").agg(n_profiles=("argo", "size"), argo_mean=("argo", "mean"), glorys_mean=("glorys", "mean"))
tab["bias_all_modes"] = e.groupby("year")["diff"].mean()
tab["n_anom"] = a.groupby("year").size()
tab["argo_anom"] = a.groupby("year")["argo_anom"].mean()
tab["glorys_anom"] = a.groupby("year")["glorys_anom"].mean()
tab["glorys_minus_argo_anom"] = tab["glorys_anom"] - tab["argo_anom"]
tab.round(3).to_csv(OUT2)
print("\n100 m east of 75E by year\n", tab.round(3).to_string())
