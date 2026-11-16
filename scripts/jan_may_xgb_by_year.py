"""EXPLORATORY (not pre-registered): frozen vs XGBoost at 30-125 m over the same calendar window as test_c (Jan 1 - May 15) in each year.
Descriptive point estimates (no intervals). Reads only per-day squared-error sums in results/scoring/*_daily*.csv."""
import glob

import numpy as np
import pandas as pd

OUT = "results/scoring/jan_may_xgb_by_year.csv"
NDEEP = 9140
depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
show = [30, 50, 75, 100, 125]
parts = [sorted(glob.glob("results/scoring/test_a_daily_rerun_*.csv"))[-1], "results/scoring/test_b_daily.csv", "results/scoring/test_c_daily.csv"]
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
df["date"] = pd.to_datetime(df["date"])
rows = []
for year in range(2020, 2027):
    lo = pd.Timestamp(f"{year}-01-01") if year != 2020 else pd.Timestamp("2020-03-02")
    w = df[(df.date >= lo) & (df.date <= pd.Timestamp(f"{year}-05-15"))]
    out = dict(year=year, n_days=w["date"].nunique())
    for m in ("frozen", "xgb"):
        s = w[w["model"] == m]
        for z in show:
            out[f"{m}_{z}"] = float(np.sqrt(s[f"sse_{z}"].sum() / NDEEP / s["date"].nunique()))
    for z in show:
        out[f"rel_{z}"] = 100 * (out[f"frozen_{z}"] - out[f"xgb_{z}"]) / out[f"xgb_{z}"]
    rows.append(out)
res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)
print("EXPLORATORY: frozen vs XGBoost RMSE change in % over Jan 1 - May 15 of each year (2020: Mar 2 - May 15); + = XGBoost better; vs GLORYS")
print(res[["year", "n_days"] + [f"rel_{z}" for z in show]].round(1).to_string(index=False))
