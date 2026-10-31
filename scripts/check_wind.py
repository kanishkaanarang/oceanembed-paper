"""Assemble (if needed) and check the daily wind array: range, yearly means, chunk seams, comparison with CCMP."""
import os

import numpy as np
import pandas as pd

from oceanembed.grid import load_ccmp_day
from oceanembed.wind_assemble import assemble_wind
from oceanembed.wind_pipeline import chunk_range, needed_chunks

path = "data/processed/wind_all.npy"
if not os.path.exists(path):
    print("assembling wind ...")
    assemble_wind(path)
arr = np.load(path, mmap_mode="r")
dates = pd.date_range("2010-02-01", "2025-12-31")
assert arr.shape == (len(dates), 2, 101, 241), arr.shape
mask = np.load("data/processed/target_mask_15depths.npy")
ocean, deep = mask[0], mask.all(axis=0)
print("array:", arr.shape, "| %.2f GB" % (os.path.getsize(path) / 1e9))

lo, hi, bad, daily = np.inf, -np.inf, 0, []
for s in range(0, len(dates), 250):
    c = np.asarray(arr[s:s + 250])
    bad += int((~np.isfinite(c)).sum())
    lo, hi = min(lo, float(c.min())), max(hi, float(c.max()))
    daily.append(c[:, :, deep].mean(axis=2))
print("non-finite values:", bad, "| range %.1f to %.1f m/s" % (lo, hi))
df = pd.DataFrame(np.concatenate(daily), index=dates, columns=["u", "v"])
print("\nyearly mean of daily u and v over the deep set (m/s):")
print(df.groupby(df.index.year).mean().round(2).to_string())

print("\nchunk seams: mean |day-to-day change| over ocean cells at the seam day vs 40 neighbouring days")
for k in needed_chunks()[1:]:
    day = chunk_range(k)[0].normalize()
    if day <= dates[1] or day >= dates[-22]:
        continue
    i = int((day - dates[0]).days)
    a = np.asarray(arr[i - 21:i + 21])[:, :, ocean]
    ch = np.abs(np.diff(a, axis=0)).mean(axis=(1, 2))
    print(f"   chunk {k} starts {day.date()}: seam {float(ch[20]):.3f} | typical {float(np.delete(ch, 20).mean()):.3f} m/s")

cc = load_ccmp_day("data/raw/test/CCMP_Wind_Analysis_20220101_V03.1_L4.nc")
i = int((pd.Timestamp("2022-01-01") - dates[0]).days)
for vi, name in ((0, "wind_u"), (1, "wind_v")):
    a, b = np.asarray(arr[i, vi])[ocean], cc[name].values[ocean]
    print(f"2022-01-01 vs CCMP, {name}: correlation {np.corrcoef(a, b)[0, 1]:.3f} | mean difference {np.mean(a - b):+.2f} m/s")
