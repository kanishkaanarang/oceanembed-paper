"""Fit the three learned baselines on TRAINING days only (same recipes as the validation scripts) and save their full-field predictions
for the given periods. Pure inference: no targets are read for validation/test days, except the validation self-check at the end.
Usage: python scripts/predict_baselines.py PERIOD [PERIOD ...]   (validation | test_a | test_b)
Outputs data/processed/pred_{lin,xgb,unet0}_{PERIOD}.npy float32 (n,15,101,241), NaN outside the mask (xgb/lin: outside the deep set / mask)."""
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from oceanembed.dataset import END
from oceanembed.dataset_anom import AnomData
from oceanembed.glorys import TARGET_DEPTHS
from oceanembed.model_sets import ModelSet, device

PERIODS = sys.argv[1:]
assert PERIODS and all(p in ("validation", "test_a", "test_b") for p in PERIODS)
NAMES = ["lin", "xgb", "unet0"]
OUT = {(n, p): f"data/processed/pred_{n}_{p}.npy" for n in NAMES for p in PERIODS}
ex = [f for f in OUT.values() if os.path.exists(f)]
if ex:
    raise SystemExit("EXISTS (stop): " + " ".join(ex))

D = AnomData()
D.fit_input_clim()
tr = D.period("train")
shape = (15, 101, 241)
t0 = time.time()

# ---- linear SLA + SST (recipe of scripts/baseline_sla_sst_regression.py) ----
S = {k: np.zeros(shape) for k in ["n", "s1", "s2", "sy", "s11", "s22", "s12", "s1y", "s2y"]}
for day in tr:
    a = D._anoms(day)
    x1, x2 = a[1].astype(np.float64), a[0].astype(np.float64)
    ok = np.isfinite(x1) & np.isfinite(x2)
    w = (D.mask & ok[None]).astype(np.float64)
    x1, x2 = np.where(ok, x1, 0.0)[None], np.where(ok, x2, 0.0)[None]
    y = D.target(day).astype(np.float64)
    S["n"] += w; S["s1"] += w * x1; S["s2"] += w * x2; S["sy"] += w * y
    S["s11"] += w * x1 * x1; S["s22"] += w * x2 * x2; S["s12"] += w * x1 * x2
    S["s1y"] += w * x1 * y; S["s2y"] += w * x2 * y
n = np.maximum(S["n"], 1)
c11, c22 = S["s11"] - S["s1"] ** 2 / n, S["s22"] - S["s2"] ** 2 / n
c12 = S["s12"] - S["s1"] * S["s2"] / n
d1, d2 = S["s1y"] - S["s1"] * S["sy"] / n, S["s2y"] - S["s2"] * S["sy"] / n
det = c11 * c22 - c12 ** 2
good = det > 1e-9
detg = np.where(good, det, 1.0)
a1 = np.where(good, (c22 * d1 - c12 * d2) / detg, 0.0)
a2 = np.where(good, (c11 * d2 - c12 * d1) / detg, 0.0)
b = (S["sy"] - a1 * S["s1"] - a2 * S["s2"]) / n
print("linear fitted (%.0f s)" % (time.time() - t0), flush=True)

# ---- XGBoost (recipe of scripts/baseline_xgboost.py) ----
ii, jj = np.nonzero(D.deep)
lat, lon = 5 + 0.25 * ii, 45 + 0.25 * jj


def feats(day, sel=slice(None)):
    a = D._anoms(day)
    ph = 2 * np.pi * day.dayofyear / 365.25
    m = len(lat[sel])
    return np.column_stack([a[0][ii[sel], jj[sel]], a[1][ii[sel], jj[sel]], lat[sel], lon[sel],
                            np.full(m, np.sin(ph)), np.full(m, np.cos(ph))]).astype(np.float32)


rng = np.random.default_rng(0)
Xs, Ys = [], []
for day in D.period("train", stride=10):
    sel = rng.choice(len(ii), 1500, replace=False)
    Xs.append(feats(day, sel))
    Ys.append(D.target(day)[:, ii[sel], jj[sel]].T)
X, Y = np.concatenate(Xs), np.concatenate(Ys)
models = []
for k in range(15):
    m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=20, subsample=0.8,
                         tree_method="hist", n_jobs=6, random_state=0)
    m.fit(X, Y[:, k])
    models.append(m)
print("xgboost fitted (%.0f s)" % (time.time() - t0), flush=True)

# ---- single U-Net, seed 0 ----
ms = ModelSet("configs/baseline_unet_seed0.yaml", device())

splits = yaml.safe_load(open("configs/splits.yaml"))
for P in PERIODS:
    s0, s1 = splits[P]
    days = pd.date_range(pd.Timestamp(s0), END if s1 == "END" else pd.Timestamp(s1))
    mm = {n: np.lib.format.open_memmap(OUT[(n, P)], mode="w+", dtype=np.float32, shape=(len(days), 15, 101, 241)) for n in NAMES}
    for s in range(0, len(days), 8):
        chunk = days[s:s + 8]
        clim = D.clim(chunk)
        an_u = ms.predict_anomaly(chunk)
        for q, d in enumerate(chunk):
            i = s + q
            a = D._anoms(d)
            x1 = np.nan_to_num(a[1].astype(np.float64), nan=0.0)[None]
            x2 = np.nan_to_num(a[0].astype(np.float64), nan=0.0)[None]
            mm["lin"][i] = np.where(D.mask, clim[q] + b + a1 * x1 + a2 * x2, np.nan)
            Xd = feats(d)
            an_x = np.zeros(shape)
            for k, m in enumerate(models):
                an_x[k, ii, jj] = m.predict(Xd)
            mm["xgb"][i] = np.where(D.deep[None], clim[q] + an_x, np.nan)
            mm["unet0"][i] = np.where(D.mask, clim[q] + an_u[q], np.nan)
        if (s // 8) % 25 == 0:
            print(P, "days:", min(s + 8, len(days)), flush=True)
    for v in mm.values():
        v.flush()
    print("saved", P, [OUT[(n_, P)] for n_ in NAMES], flush=True)

    if P == "validation":
        ref = {"lin": "sla_sst_regression", "xgb": "xgb", "unet0": "anom_sst_sla"}
        for nm in NAMES:
            sse = np.zeros(15)
            for i, d in enumerate(days):
                sse += (((mm[nm][i] - D.glorys(d))[:, D.deep]) ** 2).sum(axis=1)
            mine = np.sqrt((sse / (len(days) * D.deep.sum())).mean())
            csv = pd.read_csv(f"results/variants/{ref[nm]}_validation.csv")["model_rmse"].values
            print("CHECK %-6s pooled deep RMSE: %.4f   (validation csv: %.4f)" % (nm, mine, np.sqrt((csv ** 2).mean())))
