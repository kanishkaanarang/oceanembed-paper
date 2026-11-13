"""Per-day, per-region squared-error sums vs GLORYS on the deep set, for all models. Regions are fixed here BEFORE any test result is seen.
Usage: python scripts/score_regions.py PERIOD [PERIOD ...]   (validation | test_a | test_b)
Test periods: clean tree required, lock written first (opened_regions_*.json), rerun needs RERUN=1, every run logged in score_log.csv."""
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
MODELS = ["clim", "lin", "xgb", "unet0", "frozen", "demean"]
commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
now = datetime.datetime.now().isoformat(timespec="seconds")
rerun = os.environ.get("RERUN") == "1"
tests = [p for p in PERIODS if p.startswith("test")]
if tests and dirty:
    raise SystemExit("tracked files have uncommitted changes - commit first (stop)")
for p in tests:
    if os.path.exists(f"results/scoring/opened_regions_{p}.json") and not rerun:
        raise SystemExit(f"{p} regions already opened. Rerun only to fix a scoring bug: RERUN=1 (logged).")
for p in PERIODS:
    if not rerun and os.path.exists(f"results/scoring/regions_{p}_daily.csv"):
        raise SystemExit(f"EXISTS: results/scoring/regions_{p}_daily.csv (stop)")
D = AnomData()
D.fit_input_clim()
lat = 5 + 0.25 * np.arange(101)[:, None]
lon = 45 + 0.25 * np.arange(241)[None, :]
box = lambda la0, la1, lo0, lo1: (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon < lo1) & D.deep
REG = {"ArabianSea": box(5, 30, 45, 78), "BayOfBengal": box(5, 23, 80, 92), "Andaman": box(5, 16, 92, 100)}
union = REG["ArabianSea"] | REG["BayOfBengal"] | REG["Andaman"]
assert (REG["ArabianSea"].astype(int) + REG["BayOfBengal"] + REG["Andaman"]).max() == 1, "regions overlap"
REG["Other"] = D.deep & ~union
ncell = {k: int(v.sum()) for k, v in REG.items()}
assert sum(ncell.values()) == int(D.deep.sum())
print("deep-set cells per region:", ncell)
nfile = "results/scoring/regions_n.csv"
if not os.path.exists(nfile):
    pd.Series(ncell, name="n_cells").to_csv(nfile)
splits = yaml.safe_load(open("configs/splits.yaml"))
os.makedirs("results/scoring", exist_ok=True)
cols = [f"sse_{z}" for z in TARGET_DEPTHS.astype(int)]
for P in PERIODS:
    tag = P if not (P in tests and rerun) else f"{P}_rerun_{now.replace(':', '')}"
    if P in tests:
        if not rerun:
            json.dump({"period": f"regions_{P}", "opened": now, "commit": commit}, open(f"results/scoring/opened_regions_{P}.json", "w"))
        with open("results/scoring/score_log.csv", "a") as f:
            f.write(f"{now},regions_{P},{commit},rerun={rerun},regions_{tag}_daily.csv\n")
    s0, s1 = splits[P]
    days = pd.date_range(pd.Timestamp(s0), END if s1 == "END" else pd.Timestamp(s1))
    pred = {m: np.load(f"data/processed/pred_{m}_{P}.npy", mmap_mode="r") for m in MODELS if m != "clim"}
    for m, a in pred.items():
        assert a.shape[0] == len(days), (m, a.shape)
    rows = []
    for i, d in enumerate(days):
        g = D.glorys(d).astype(np.float64)
        cl = D.clim(pd.DatetimeIndex([d]))[0]
        for m in MODELS:
            p = cl if m == "clim" else np.asarray(pred[m][i])
            e = p.astype(np.float64) - g
            for rn, mk in REG.items():
                rows.append((d.date().isoformat(), rn, m, *(e[:, mk] ** 2).sum(1)))
        if i % 100 == 0:
            print(P, "scored days:", i + 1, "of", len(days), flush=True)
    df = pd.DataFrame(rows, columns=["date", "region", "model"] + cols)
    df.to_csv(f"results/scoring/regions_{tag}_daily.csv", index=False)
    print("saved", f"results/scoring/regions_{tag}_daily.csv", "commit", commit)
    print(f"\n{P}: pooled RMSE (C) by region (rows) and model (columns), deep set")
    tab = {}
    for rn in REG:
        tab[rn] = {m: np.sqrt((df[(df.region == rn) & (df.model == m)][cols].sum().values / (len(days) * ncell[rn])).mean()) for m in MODELS}
    print(pd.DataFrame(tab).T.round(3).to_string())
    if P == "validation":
        ref = pd.read_csv("results/scoring/validation_sums.csv")
        ref = ref[ref["set"] == "deep"].groupby(["model", "depth_m"])["sse"].sum()
        worst = 0.0
        for m in MODELS:
            mine = df[df.model == m][cols].sum().values
            theirs = np.array([ref[(m, int(z))] for z in TARGET_DEPTHS])
            worst = max(worst, float(np.max(np.abs(mine - theirs) / theirs)))
        print("CHECK regions add up to the saved deep-set totals: max relative difference %.2e (should be ~1e-12)" % worst)
