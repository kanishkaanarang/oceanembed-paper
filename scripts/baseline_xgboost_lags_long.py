"""XGBoost with lagged inputs: the baseline features plus SST and SLA anomalies from 7, 14, 30, 60 and 90 days earlier (past dates only).
Same fixed settings as baseline_xgboost.py. Fitted on TRAINING days only; scored on the VALIDATION window."""
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from oceanembed.dataset import START
from oceanembed.dataset_anom import AnomData
from oceanembed.glorys import TARGET_DEPTHS

OUT = "results/variants/xgb_lags_long_validation.csv"
if os.path.exists(OUT):
    raise SystemExit(f"{OUT} already exists - not overwriting")

LAGS = [7, 14, 30, 60, 90]
NAMES = ["sst", "sla", "lat", "lon", "sin_doy", "cos_doy"] + [f"{c}_lag{L}" for L in LAGS for c in ("sst", "sla")]
D = AnomData()
D.fit_input_clim()
ii, jj = np.nonzero(D.deep)
lat, lon = 5 + 0.25 * ii, 45 + 0.25 * jj


def feats(day, sel=slice(None)):
    assert (day - START).days >= max(LAGS), day
    a = D._anoms(day)
    ph = 2 * np.pi * day.dayofyear / 365.25
    n = len(lat[sel])
    cols = [np.column_stack([a[0][ii[sel], jj[sel]], a[1][ii[sel], jj[sel]], lat[sel], lon[sel],
                             np.full(n, np.sin(ph)), np.full(n, np.cos(ph))])]
    for L in LAGS:
        b = D._anoms(day - pd.Timedelta(days=L))
        cols.append(np.column_stack([b[0][ii[sel], jj[sel]], b[1][ii[sel], jj[sel]]]))
    return np.concatenate(cols, axis=1).astype(np.float32)


rng = np.random.default_rng(0)
t0 = time.time()
Xs, Ys = [], []
for day in [d for d in D.period("train", stride=10) if (d - START).days >= max(LAGS)]:
    sel = rng.choice(len(ii), 1500, replace=False)
    Xs.append(feats(day, sel))
    Ys.append(D.target(day)[:, ii[sel], jj[sel]].T)
X, Y = np.concatenate(Xs), np.concatenate(Ys)
print("training rows %d, features %d (%.0f s)" % (X.shape[0], X.shape[1], time.time() - t0), flush=True)

models = []
for k, z in enumerate(TARGET_DEPTHS.astype(int)):
    m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=20, subsample=0.8,
                         tree_method="hist", n_jobs=6, random_state=0)
    m.fit(X, Y[:, k])
    models.append(m)
    print("fitted depth %d m (%.0f s elapsed)" % (z, time.time() - t0), flush=True)

va = D.period("validation")
sse, sse0, bias, cnt = np.zeros(15), np.zeros(15), np.zeros(15), 0
for s in range(0, len(va), 10):
    days = va[s:s + 10]
    Xv = np.concatenate([feats(d) for d in days])
    Yv = np.concatenate([D.target(d)[:, ii, jj] for d in days], axis=1)
    for k, m in enumerate(models):
        e = m.predict(Xv) - Yv[k]
        sse[k] += (e ** 2).sum()
        sse0[k] += (Yv[k] ** 2).sum()
        bias[k] += e.sum()
    cnt += Xv.shape[0]
rmse, rmse0 = np.sqrt(sse / cnt), np.sqrt(sse0 / cnt)
df = pd.DataFrame({"depth_m": TARGET_DEPTHS.astype(int), "clim_rmse": rmse0, "model_rmse": rmse,
                   "skill": 1 - sse / sse0, "model_bias": bias / cnt})
df.to_csv(OUT, index=False)
base = pd.read_csv("results/variants/xgb_validation.csv")
lin = pd.read_csv("results/variants/sla_sst_regression_validation.csv")
net = pd.read_csv("results/variants/anom_sst_sla_validation.csv")
pool = lambda v: float(np.sqrt((np.asarray(v) ** 2).mean()))
print("RESULT xgboost+lags | pooled RMSE: climatology %.3f | XGBoost with lags %.3f | XGBoost no lags %.3f | linear %.3f | U-Net seed 0 %.3f"
      % (pool(rmse0), pool(rmse), pool(base["model_rmse"]), pool(lin["model_rmse"]), pool(net["model_rmse"])))
print(" depth_m  with_lags  no_lags  linear  U-Net")
for i, z in enumerate(df["depth_m"]):
    print("  %5d   %7.3f  %7.3f  %6.3f  %6.3f" % (z, rmse[i], base["model_rmse"][i], lin["model_rmse"][i], net["model_rmse"][i]))
imp = sorted(zip(NAMES, models[7].feature_importances_), key=lambda t: -t[1])[:6]
print("top features at 100 m:", [(n, round(float(v), 3)) for n, v in imp])
