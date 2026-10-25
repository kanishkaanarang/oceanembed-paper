"""Per-year RMSE of two climatology baselines on TRAINING and VALIDATION days only (test periods untouched).
A: harmonic seasonal cycle. B: harmonic seasonal cycle + linear trend. Both fitted on the training period."""
import os

import numpy as np
import pandas as pd
import yaml

from oceanembed.glorys import TARGET_DEPTHS
from oceanembed.glorys_pipeline import block_dates

cfg = {"train": ["2010-02-04", "2018-12-31"], "validation": ["2019-02-01", "2020-01-31"]}  # frozen copy of the first split used for this diagnostic
deep = np.load("data/processed/target_mask_15depths.npy").all(axis=0)
NH = 3
SHAPE = (15, 101, 241)
M = int(np.prod(SHAPE))
T_REF = pd.Timestamp("2014-07-01")

BLOCKS = []
for b in (3, 4):
    s, e = block_dates(b)
    BLOCKS.append((pd.date_range(s, e), np.load(f"data/processed/glorys_block{b}.npy", mmap_mode="r")))


def in_period(dates, period):
    return np.asarray((dates >= pd.Timestamp(period[0])) & (dates <= pd.Timestamp(period[1])))


def day_chunks(t0, t1, size=50):
    t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
    for dates, arr in BLOCKS:
        sel = np.where((dates >= t0) & (dates <= t1))[0]
        if len(sel) == 0:
            continue
        for s in range(sel[0], sel[-1] + 1, size):
            e = min(s + size, sel[-1] + 1)
            yield dates[s:e], np.asarray(arr[s:e])


def design(dates, trend):
    phase = 2 * np.pi * dates.dayofyear.values / 365.25
    cols = [np.ones(len(dates))]
    for k in range(1, NH + 1):
        cols += [np.sin(k * phase), np.cos(k * phase)]
    if trend:
        cols.append(np.asarray((dates - T_REF).days, dtype=float) / 365.25)
    return np.stack(cols, axis=1)


def fit(trend):
    P = 2 * NH + 1 + (1 if trend else 0)
    XtX = np.zeros((P, P))
    Xty = np.zeros((P, M))
    n = 0
    for dates, y in day_chunks(*cfg["train"]):
        X = design(dates, trend)
        Y = np.nan_to_num(y.reshape(len(dates), -1), nan=0.0).astype(np.float64)
        XtX += X.T @ X
        Xty += X.T @ Y
        n += len(dates)
    return np.linalg.solve(XtX, Xty), n


betaA, nA = fit(False)
betaB, nB = fit(True)
print("fitted on %d training days (%s to %s)" % (nA, *cfg["train"]))

acc = {}
for dates, y in day_chunks(cfg["train"][0], cfg["validation"][1]):
    keep = in_period(dates, cfg["train"]) | in_period(dates, cfg["validation"])
    if not keep.any():
        continue
    dates, y = dates[keep], y[keep]
    errA = np.where(deep[None, None], (design(dates, False) @ betaA).reshape(len(dates), *SHAPE) - y, 0.0)
    errB = np.where(deep[None, None], (design(dates, True) @ betaB).reshape(len(dates), *SHAPE) - y, 0.0)
    yrs = dates.year.values
    isval = in_period(dates, cfg["validation"])
    groups = [(int(Y), yrs == Y) for Y in np.unique(yrs)] + [("val", isval)]
    for key, m in groups:
        if not m.any():
            continue
        d = acc.setdefault(key, {"days": 0, "A": np.zeros(15), "B": np.zeros(15),
                                 "biasA": np.zeros(15), "biasB": np.zeros(15)})
        d["days"] += int(m.sum())
        d["A"] += (errA[m] ** 2).sum(axis=(0, 2, 3))
        d["B"] += (errB[m] ** 2).sum(axis=(0, 2, 3))
        d["biasA"] += errA[m].sum(axis=(0, 2, 3))
        d["biasB"] += errB[m].sum(axis=(0, 2, 3))

ncell = int(deep.sum())
k100 = int(np.where(TARGET_DEPTHS == 100)[0][0])


def mse(d, which):
    return d[which] / (d["days"] * ncell)


rows = []
for key in sorted(k for k in acc if k != "val"):
    d = acc[key]
    rows.append({
        "year": key, "days": d["days"],
        "all_depths_A": np.sqrt(mse(d, "A").mean()), "all_depths_B": np.sqrt(mse(d, "B").mean()),
        "rmse100_A": np.sqrt(mse(d, "A")[k100]), "rmse100_B": np.sqrt(mse(d, "B")[k100]),
        "bias100_A": d["biasA"][k100] / (d["days"] * ncell), "bias100_B": d["biasB"][k100] / (d["days"] * ncell),
    })
df = pd.DataFrame(rows)
print("\nclimatology RMSE by calendar year (C), deep set; years 2010-2018 are in-sample for the fit")
print("2019 holds validation days only (Feb-Dec) and 2020 only January; 2010 starts on 4 Feb")
print("A = seasonal cycle, B = seasonal cycle + linear trend")
print(df.round(3).to_string(index=False))

v = acc["val"]
dv = pd.DataFrame({
    "depth_m": TARGET_DEPTHS.astype(int),
    "rmse_A": np.sqrt(mse(v, "A")), "rmse_B": np.sqrt(mse(v, "B")),
    "bias_A": v["biasA"] / (v["days"] * ncell), "bias_B": v["biasB"] / (v["days"] * ncell),
})
print("\nvalidation %s to %s, %d days" % (*cfg["validation"], v["days"]))
print(dv.round(3).to_string(index=False))
print("all-depth RMSE on validation: A %.3f | B %.3f" % (np.sqrt(mse(v, "A").mean()), np.sqrt(mse(v, "B").mean())))
os.makedirs("results", exist_ok=True)
df.to_csv("results/climatology_by_year.csv", index=False)
dv.to_csv("results/climatology_trend_validation.csv", index=False)
print("\nsaved results/climatology_by_year.csv and results/climatology_trend_validation.csv | test periods untouched")
