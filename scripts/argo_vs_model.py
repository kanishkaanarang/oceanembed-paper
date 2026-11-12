"""Final model vs Argo, profile by profile, on the VALIDATION window (dry run for the test evaluation): RMSE and bias against Argo for
climatology, GLORYS and the final model, by depth. Delayed-mode and adjusted Argo only; 0 m dropped (too few profiles)."""
import os

import numpy as np
import pandas as pd

from oceanembed.dataset_anom import AnomData
from oceanembed.glorys import TARGET_DEPTHS

OUT = "results/argo_vs_model_validation.csv"
if os.path.exists(OUT):
    raise SystemExit(f"EXISTS: {OUT} (stop)")

D = AnomData()
days = D.period("validation")
pred = np.load("data/processed/pred_final_validation.npy", mmap_mode="r")
r = pd.read_csv("data/processed/argo_vs_glorys.csv.gz", parse_dates=["day"])
r = r[(r.day >= days[0]) & (r.day <= days[-1]) & r["mode"].isin(["D", "A"]) & (r.depth > 0)].reset_index(drop=True)
r["i"] = np.round((r.lat - 5) / 0.25).astype(int)
r["j"] = np.round((r.lon - 45) / 0.25).astype(int)
r["k"] = r.depth.map({int(d): k for k, d in enumerate(TARGET_DEPTHS)}).astype(int)
r["t"] = (r.day - days[0]).dt.days
r["model"] = pred[r.t.values, r.k.values, r.i.values, r.j.values]
clim = np.full(len(r), np.nan)
for d, g in r.groupby("day"):
    c = D.clim(pd.DatetimeIndex([d]))[0]
    clim[g.index.values] = c[g.k.values, g.i.values, g.j.values]
r["clim"] = clim
r = r[np.isfinite(r.model) & np.isfinite(r.clim)]
print("validation window Argo rows: %d | profiles: %d | days: %s to %s"
      % (len(r), r.groupby(["day", "lat", "lon"]).ngroups, days[0].date(), days[-1].date()))
for c in ("clim", "glorys", "model"):
    r["e_" + c] = r[c] - r["argo"]


def summarize(g):
    mse = {c: float((g["e_" + c] ** 2).mean()) for c in ("clim", "glorys", "model")}
    return pd.Series({"n": len(g), "rmse_clim": np.sqrt(mse["clim"]), "rmse_glorys": np.sqrt(mse["glorys"]),
                      "rmse_model": np.sqrt(mse["model"]), "bias_glorys": g.e_glorys.mean(), "bias_model": g.e_model.mean(),
                      "skill_model_vs_clim": 1 - mse["model"] / mse["clim"], "skill_glorys_vs_clim": 1 - mse["glorys"] / mse["clim"]})


tab = r.groupby("depth").apply(summarize)
tab.loc["pooled"] = summarize(r)
tab.round(3).to_csv(OUT)
print("\nvs Argo, validation window (2018-02 to 2020-01), D+A modes")
print(tab.round(3).to_string())
