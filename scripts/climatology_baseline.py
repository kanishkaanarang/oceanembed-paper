"""Climatology baseline: harmonic seasonal cycle per cell and depth, fitted on the TRAINING period only,
evaluated on the training period (in-sample) and the VALIDATION period. Test periods are not touched."""
import os

import numpy as np
import pandas as pd
import yaml

from oceanembed.glorys import TARGET_DEPTHS
from oceanembed.glorys_pipeline import block_dates

cfg = yaml.safe_load(open("configs/splits.yaml"))
deep = np.load("data/processed/target_mask_15depths.npy").all(axis=0)   # (101, 241)
NH = 3                    # harmonics: annual, semi-annual, four-monthly
P = 2 * NH + 1
SHAPE = (15, 101, 241)
M = int(np.prod(SHAPE))

BLOCKS = []
for b in (3, 4):          # blocks 3 and 4 hold the training and validation periods
    start, end = block_dates(b)
    BLOCKS.append((pd.date_range(start, end), np.load(f"data/processed/glorys_block{b}.npy", mmap_mode="r")))


def day_chunks(period, size=50):
    t0, t1 = (pd.Timestamp(x) for x in period)
    for dates, arr in BLOCKS:
        sel = np.where((dates >= t0) & (dates <= t1))[0]
        if len(sel) == 0:
            continue
        for s in range(sel[0], sel[-1] + 1, size):
            e = min(s + size, sel[-1] + 1)
            yield dates[s:e], np.asarray(arr[s:e])


def design(dates):
    phase = 2 * np.pi * dates.dayofyear.values / 365.25
    cols = [np.ones(len(dates))]
    for k in range(1, NH + 1):
        cols += [np.sin(k * phase), np.cos(k * phase)]
    return np.stack(cols, axis=1)


def n_days(period):
    return (pd.Timestamp(period[1]) - pd.Timestamp(period[0])).days + 1


XtX = np.zeros((P, P))
Xty = np.zeros((P, M))
count = 0
for dates, y in day_chunks(cfg["train"]):
    X = design(dates)
    Y = np.nan_to_num(y.reshape(len(dates), -1), nan=0.0).astype(np.float64)
    XtX += X.T @ X
    Xty += X.T @ Y
    count += len(dates)
assert count == n_days(cfg["train"]), (count, n_days(cfg["train"]))
beta = np.linalg.solve(XtX, Xty)
print("fitted on %d training days (%s to %s)" % (count, *cfg["train"]))


def evaluate(period):
    sse = np.zeros(15)
    sse0 = np.zeros(15)
    bias = np.zeros(15)
    n = 0
    for dates, y in day_chunks(period):
        X = design(dates)
        pred = (X @ beta).reshape(len(dates), *SHAPE)
        pred0 = beta[0].reshape(1, *SHAPE)
        err = np.where(deep[None, None], pred - y, 0.0)
        err0 = np.where(deep[None, None], pred0 - y, 0.0)
        sse += (err ** 2).sum(axis=(0, 2, 3))
        sse0 += (err0 ** 2).sum(axis=(0, 2, 3))
        bias += err.sum(axis=(0, 2, 3))
        n += len(dates) * int(deep.sum())
    return np.sqrt(sse / n), bias / n, np.sqrt(sse0 / n)


tr_rmse, _, _ = evaluate(cfg["train"])
va_rmse, va_bias, va_rmse0 = evaluate(cfg["validation"])
df = pd.DataFrame({
    "depth_m": TARGET_DEPTHS.astype(int),
    "train_rmse": tr_rmse,
    "val_rmse": va_rmse,
    "val_bias": va_bias,
    "val_rmse_no_season": va_rmse0,
})
print("\nvalidation %s to %s | deep set only | RMSE and bias in C" % tuple(cfg["validation"]))
print("(val_rmse_no_season = the training-period mean with no seasonal cycle)")
print(df.round(3).to_string(index=False))
os.makedirs("results", exist_ok=True)
df.to_csv("results/climatology_baseline_validation.csv", index=False)
print("\nsaved results/climatology_baseline_validation.csv | test periods were not evaluated")
