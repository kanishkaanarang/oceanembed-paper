"""Is the growing GLORYS-minus-Argo bias real or a sampling artefact? Reads data/processed/argo_vs_glorys.csv.gz; prints only."""
import numpy as np
import pandas as pd

r = pd.read_csv("data/processed/argo_vs_glorys.csv.gz", parse_dates=["day"])
r["year"] = r["day"].dt.year
da = r[r["mode"].isin(["D", "A"])].copy()

print("1) GLORYS-minus-Argo bias at 100 m by year, three longitude bands (n in brackets), D+A only")
b = da[da.depth == 100].copy()
b["band"] = pd.cut(b.lon, [0, 75, 95, 200], labels=["<=75E", "75-95E", ">95E"])
t = b.groupby(["year", "band"], observed=True)["diff"].agg(["mean", "size"]).unstack("band")
print(t.round(2).to_string())

print("\n2) Bias by year at 50, 100, 200, 500 m, whole domain, D+A only")
print(da[da.depth.isin([50, 100, 200, 500])].pivot_table(index="year", columns="depth", values="diff", aggfunc="mean").round(2).to_string())

print("\n3) Bias at 100 m by half-year, 2019-2023, whole domain, D+A only")
h = b[b.year.between(2019, 2023)].copy()
h["half"] = np.where(h.day.dt.month <= 6, "H1", "H2")
print(h.groupby(["year", "half"])["diff"].agg(["mean", "size"]).round(2).to_string())

print("\n4) Sampling-adjusted yearly anomaly at 100 m east of 75E: each profile minus its own 2-degree cell x season mean (groups with >=10 profiles)")
e = r[(r.depth == 100) & (r.lon > 75)].copy()
e["season"] = (e.day.dt.month % 12) // 3
e["cell"] = (np.floor(e.lat / 2).astype(int).astype(str) + "_" + np.floor(e.lon / 2).astype(int).astype(str) + "_" + e.season.astype(str))
e["n_grp"] = e.groupby("cell")["argo"].transform("size")
e = e[e.n_grp >= 10]
e["argo_anom"] = e["argo"] - e.groupby("cell")["argo"].transform("mean")
e["glorys_anom"] = e["glorys"] - e.groupby("cell")["glorys"].transform("mean")
print(e.groupby("year")[["argo_anom", "glorys_anom"]].agg("mean").assign(n=e.groupby("year").size()).round(2).to_string())
