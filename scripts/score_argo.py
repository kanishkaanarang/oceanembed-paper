"""Models vs Argo, profile by profile (nearest 0.25-degree cell and day), D+A data modes, 0 m dropped.
Usage: python scripts/score_argo.py PERIOD [PERIOD ...]   (validation | test_a | test_b)
Test periods are opened once for Argo too: a lock is written first; a rerun needs RERUN=1 and is logged."""
import datetime
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from oceanembed.dataset import END
from oceanembed.dataset_anom import AnomData
from oceanembed.glorys import TARGET_DEPTHS

PERIODS = sys.argv[1:]
assert PERIODS and all(p in ("validation", "test_a", "test_b") for p in PERIODS)
MODELS = ["lin", "xgb", "unet0", "frozen", "demean"]
COLS = ["clim", "glorys"] + MODELS
commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
now = datetime.datetime.now().isoformat(timespec="seconds")
rerun = os.environ.get("RERUN") == "1"
tests = [p for p in PERIODS if p.startswith("test")]
if tests and dirty:
    raise SystemExit("tracked files have uncommitted changes - commit first (stop)")
for p in tests:
    if os.path.exists(f"results/scoring/opened_argo_{p}.json") and not rerun:
        raise SystemExit(f"{p} Argo already opened. Rerun only to fix a scoring bug: RERUN=1 (it is logged).")
os.makedirs("results/scoring", exist_ok=True)
D = AnomData()
D.fit_input_clim()
splits = yaml.safe_load(open("configs/splits.yaml"))
rmse = lambda x: float(np.sqrt(np.mean(np.square(x))))
raw = pd.read_csv("data/processed/argo_vs_glorys.csv.gz", parse_dates=["day"])


def table(g):
    rows = {}
    for z, h in list(g.groupby("depth")) + [("pooled", g)]:
        rows[z if z == "pooled" else int(z)] = {"n": len(h), **{c: rmse(h["e_" + c]) for c in COLS},
                                                "bias_glorys": h.e_glorys.mean(), "bias_frozen": h.e_frozen.mean()}
    return pd.DataFrame(rows).T


for P in PERIODS:
    tag = P if not (P.startswith("test") and rerun) else f"{P}_rerun_{now.replace(':', '')}"
    if P.startswith("test"):
        if not rerun:
            json.dump({"period": f"argo_{P}", "opened": now, "commit": commit}, open(f"results/scoring/opened_argo_{P}.json", "w"))
        with open("results/scoring/score_log.csv", "a") as f:
            f.write(f"{now},argo_{P},{commit},rerun={rerun},argo_{tag}\n")
    elif os.path.exists(f"results/scoring/argo_{tag}_rows.csv.gz"):
        raise SystemExit(f"EXISTS: results/scoring/argo_{tag}_rows.csv.gz (stop)")
    s0, s1 = splits[P]
    days = pd.date_range(pd.Timestamp(s0), END if s1 == "END" else pd.Timestamp(s1))
    r = raw[(raw.day >= days[0]) & (raw.day <= days[-1]) & raw["mode"].isin(["D", "A"]) & (raw.depth > 0)].reset_index(drop=True)
    r["i"] = np.round((r.lat - 5) / 0.25).astype(int)
    r["j"] = np.round((r.lon - 45) / 0.25).astype(int)
    r["k"] = r.depth.map({int(d): k for k, d in enumerate(TARGET_DEPTHS)}).astype(int)
    r["t"] = (r.day - days[0]).dt.days
    assert r.i.between(0, 100).all() and r.j.between(0, 240).all()
    for m in MODELS:
        pr = np.load(f"data/processed/pred_{m}_{P}.npy", mmap_mode="r")
        assert pr.shape[0] == len(days), (m, pr.shape)
        r[m] = pr[r.t.values, r.k.values, r.i.values, r.j.values]
    clim = np.full(len(r), np.nan)
    for d, g in r.groupby("day"):
        c = D.clim(pd.DatetimeIndex([d]))[0]
        clim[g.index.values] = c[g.k.values, g.i.values, g.j.values]
    r["clim"] = clim
    for c in COLS:
        r["e_" + c] = r[c] - r["argo"]
    r.drop(columns=["i", "j", "k", "t"]).to_csv(f"results/scoring/argo_{tag}_rows.csv.gz", index=False)
    A = r[np.isfinite(r[["clim", "glorys", "lin", "unet0", "frozen", "demean"]]).all(axis=1)]
    B = A[np.isfinite(A["xgb"])]
    print(f"\n######## {P}: Argo rows {len(r)} | profiles {r.groupby(['day', 'lat', 'lon']).ngroups} | mask-set rows {len(A)} | common (deep) rows {len(B)}")
    print("rows per year:", r.groupby(r.day.dt.year).size().to_dict())
    tb = table(B)
    tb.to_csv(f"results/scoring/argo_{tag}_by_depth.csv")
    print(f"--- {P} vs Argo, COMMON set (all models), RMSE in C ---")
    print(tb.round(3).to_string())
    ta = table(A)
    print(f"--- {P} vs Argo, MASK set (no XGBoost), pooled ---")
    print(ta.loc[["pooled"], ["n"] + [c for c in COLS if c != "xgb"] + ["bias_glorys", "bias_frozen"]].round(3).to_string())
    th = B[(B.depth >= 75) & (B.depth <= 200)]
    yr = {}
    for y, h in th.groupby(th.day.dt.year):
        yr[y] = {"n": len(h), **{c: rmse(h["e_" + c]) for c in ("clim", "glorys", "lin", "frozen", "demean")},
                 "bias_glorys": h.e_glorys.mean(), "bias_frozen": h.e_frozen.mean(), "bias_lin": h.e_lin.mean()}
    print(f"--- {P} thermocline 75-200 m by year (common set) ---")
    print(pd.DataFrame(yr).T.round(3).to_string())
    if P == "validation":
        old = pd.read_csv("results/argo_vs_model_validation.csv", index_col=0).loc["pooled", "rmse_model"]
        print("CHECK frozen pooled RMSE vs Argo, mask set: %.4f  (old argo_vs_model_validation.csv: %.4f)" % (rmse(A["e_frozen"]), old))
