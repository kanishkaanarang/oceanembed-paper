"""Linear baseline with two predictors: per cell and depth, regress the GLORYS anomaly on the SLA anomaly and the SST
anomaly (each minus its own training-period seasonal cycle). Fitted on TRAINING days only; scored on the VALIDATION window."""
import os
import time

import numpy as np
import pandas as pd

from oceanembed.dataset_anom import AnomData
from oceanembed.glorys import TARGET_DEPTHS

OUT = "results/variants/sla_sst_regression_validation.csv"
if os.path.exists(OUT):
    raise SystemExit(f"{OUT} already exists - not overwriting")

D = AnomData()
D.fit_input_clim()
tr, va = D.period("train"), D.period("validation")
shape = (15, 101, 241)
t0 = time.time()
S = {k: np.zeros(shape) for k in ["n", "s1", "s2", "sy", "s11", "s22", "s12", "s1y", "s2y"]}
for day in tr:
    a = D._anoms(day)
    x1, x2 = a[1].astype(np.float64), a[0].astype(np.float64)          # SLA anomaly, SST anomaly
    ok = np.isfinite(x1) & np.isfinite(x2)
    w = (D.mask & ok[None]).astype(np.float64)
    x1, x2 = np.where(ok, x1, 0.0)[None], np.where(ok, x2, 0.0)[None]
    y = D.target(day).astype(np.float64)
    S["n"] += w
    S["s1"] += w * x1
    S["s2"] += w * x2
    S["sy"] += w * y
    S["s11"] += w * x1 * x1
    S["s22"] += w * x2 * x2
    S["s12"] += w * x1 * x2
    S["s1y"] += w * x1 * y
    S["s2y"] += w * x2 * y
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
print("regression fitted on %d training days (%.0f s)" % (len(tr), time.time() - t0))

sse, sse0, bias, cnt = np.zeros(15), np.zeros(15), np.zeros(15), 0
for day in va:
    a = D._anoms(day)
    x1 = np.nan_to_num(a[1].astype(np.float64), nan=0.0)[None]
    x2 = np.nan_to_num(a[0].astype(np.float64), nan=0.0)[None]
    y = D.target(day).astype(np.float64)
    e = (b + a1 * x1 + a2 * x2 - y)[:, D.deep]
    y0 = y[:, D.deep]
    sse += (e ** 2).sum(axis=1)
    sse0 += (y0 ** 2).sum(axis=1)
    bias += e.sum(axis=1)
    cnt += int(D.deep.sum())
rmse, rmse0 = np.sqrt(sse / cnt), np.sqrt(sse0 / cnt)
df = pd.DataFrame({"depth_m": TARGET_DEPTHS.astype(int), "clim_rmse": rmse0, "model_rmse": rmse,
                   "skill": 1 - sse / sse0, "model_bias": bias / cnt})
os.makedirs("results/variants", exist_ok=True)
df.to_csv(OUT, index=False)
net = pd.read_csv("results/variants/anom_sst_sla_validation.csv")
pooled0, pooled, pooled_net = (np.sqrt((v ** 2).mean()) for v in (rmse0, rmse, net["model_rmse"].values))
print("RESULT sla_sst_regression | pooled RMSE: climatology %.3f C | linear SLA+SST %.3f C | U-Net anomaly SST+SLA %.3f C"
      % (pooled0, pooled, pooled_net))
print(" depth_m  linear_SLA+SST  U-Net  difference (U-Net - linear)")
for i, z in enumerate(df["depth_m"]):
    print("  %5d   %10.3f  %7.3f   %+8.3f" % (z, rmse[i], net["model_rmse"][i], net["model_rmse"][i] - rmse[i]))
