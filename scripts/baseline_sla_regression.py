"""Linear baseline: per cell and depth, regress the GLORYS anomaly on the sea level anomaly (SLA minus its own
training-period seasonal cycle). Fitted on TRAINING days only, scored on the VALIDATION window; test periods unreachable."""
import os
import time

import numpy as np
import pandas as pd

from oceanembed.dataset import START, Data, design
from oceanembed.glorys import TARGET_DEPTHS

OUT = "results/variants/sla_regression_validation.csv"
if os.path.exists(OUT):
    raise SystemExit(f"{OUT} already exists - not overwriting")
os.makedirs("results/variants", exist_ok=True)

D = Data()
tr, va = D.period("train"), D.period("validation")
P = design(tr[:1]).shape[1]
NCELL = 101 * 241


def sla_day(day):
    return np.asarray(D.sla[int((day - START).days)], dtype=np.float64)


t0 = time.time()
XtX, Xty = np.zeros((P, P)), np.zeros((P, NCELL))
for s in range(0, len(tr), 100):
    days = tr[s:s + 100]
    X = design(days)
    Y = np.stack([np.nan_to_num(sla_day(d), nan=0.0).ravel() for d in days])
    XtX += X.T @ X
    Xty += X.T @ Y
beta_sla = np.linalg.solve(XtX, Xty)
print("SLA seasonal cycle fitted on %d training days (%.0f s)" % (len(tr), time.time() - t0))


def sla_anom(day):
    return sla_day(day) - (design(pd.DatetimeIndex([day])) @ beta_sla).reshape(101, 241)


shape = (15, 101, 241)
n, Sx, Sy, Sxx, Sxy = (np.zeros(shape) for _ in range(5))
for day in tr:
    x = sla_anom(day)
    y = D.target(day).astype(np.float64)                       # anomaly in C, 0 outside the mask
    w = (D.mask & np.isfinite(x)[None]).astype(np.float64)
    x = np.nan_to_num(x, nan=0.0)[None]
    n += w
    Sx += w * x
    Sy += w * y
    Sxx += w * x * x
    Sxy += w * x * y
n1 = np.maximum(n, 1)
vx = Sxx - Sx * Sx / n1
slope = np.where(vx > 1e-12, (Sxy - Sx * Sy / n1) / np.maximum(vx, 1e-12), 0.0)
icpt = (Sy - slope * Sx) / n1
print("regression fitted (%.0f s)" % (time.time() - t0))

sse, sse0, bias, cnt = np.zeros(15), np.zeros(15), np.zeros(15), 0
for day in va:
    x = np.nan_to_num(sla_anom(day), nan=0.0)[None]
    y = D.target(day).astype(np.float64)
    e = (icpt + slope * x - y)[:, D.deep]
    y0 = y[:, D.deep]
    sse += (e ** 2).sum(axis=1)
    sse0 += (y0 ** 2).sum(axis=1)
    bias += e.sum(axis=1)
    cnt += int(D.deep.sum())
rmse, rmse0 = np.sqrt(sse / cnt), np.sqrt(sse0 / cnt)
base = pd.read_csv("results/climatology_baseline_validation.csv")
df = pd.DataFrame({"depth_m": TARGET_DEPTHS.astype(int), "clim_rmse": rmse0, "model_rmse": rmse,
                   "skill": 1 - sse / sse0, "model_bias": bias / cnt})
df.to_csv(OUT, index=False)
pooled0, pooled = np.sqrt((rmse0 ** 2).mean()), np.sqrt((rmse ** 2).mean())
print("RESULT sla_regression | pooled RMSE: climatology %.3f C | regression %.3f C | check vs baseline csv: %.4f C"
      % (pooled0, pooled, np.abs(rmse0 - base["val_rmse"].values).max()))
print(df.round(3).to_string(index=False))
