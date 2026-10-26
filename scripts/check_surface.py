"""Assemble (if needed) and check the SSH array: dates, validity pattern, masks, ranges, yearly means,
and a cross-check of the 2025-minus-2024 pattern against GLORYS at 100 m. Usage: python scripts/check_surface.py ssh"""
import os
import sys

import numpy as np
import pandas as pd

from oceanembed.glorys import TARGET_DEPTHS
from oceanembed.glorys_pipeline import block_dates
from oceanembed.grid import target_grid
from oceanembed.surface_assemble import assemble_surface

product = sys.argv[1] if len(sys.argv) > 1 else "ssh"
path = f"data/processed/{product}_all.npy"
if not os.path.exists(path):
    print("assembling", product, "...")
    assemble_surface(product, path)
arr = np.load(path, mmap_mode="r")
dates = pd.date_range("2010-02-01", "2025-12-31")
assert arr.shape[0] == len(dates), (arr.shape, len(dates))
lat, lon = target_grid()
mask = np.load("data/processed/target_mask_15depths.npy")
ocean, deep = mask[0], mask.all(axis=0)
print("array:", arr.shape, "| dates", dates[0].date(), "to", dates[-1].date(), "| %.2f GB" % (os.path.getsize(path) / 1e9))

ref = np.isfinite(np.asarray(arr[0]))
bad = 0
daily = []
lo, hi = np.inf, -np.inf
for s in range(0, len(dates), 200):
    chunk = np.asarray(arr[s:s + 200])
    v = np.isfinite(chunk)
    bad += int((v != ref[None]).any(axis=(1, 2)).sum())
    daily.append(np.nanmean(chunk[:, deep], axis=1))
    lo, hi = min(lo, float(np.nanmin(chunk))), max(hi, float(np.nanmax(chunk)))
print("days whose validity pattern differs from day 1:", bad, "of", len(dates))
print("valid cells: %d | with GLORYS ocean: %d | GLORYS ocean without SSH: %d | SSH without GLORYS ocean: %d"
      % (ref.sum(), (ref & ocean).sum(), (ocean & ~ref).sum(), (ref & ~ocean).sum()))
print("deep-set cells with SSH values: %d of %d" % ((ref & deep).sum(), deep.sum()))
print("value range over all days: %.3f to %.3f m" % (lo, hi))

s = pd.Series(np.concatenate(daily), index=dates)
print("\nyearly mean sea level anomaly over the deep set (m):")
print(s.groupby(s.index.year).mean().round(3).to_string())

yrs = dates.year.values
dsla = np.asarray(arr[yrs == 2025]).mean(axis=0) - np.asarray(arr[yrs == 2024]).mean(axis=0)
k100 = int(np.where(TARGET_DEPTHS == 100)[0][0])
b5 = np.load("data/processed/glorys_block5.npy", mmap_mode="r")
d5 = pd.date_range(*block_dates(5))
dT = np.asarray(b5[d5.year == 2025, k100]).mean(axis=0) - np.asarray(b5[d5.year == 2024, k100]).mean(axis=0)
sel = deep & np.isfinite(dsla) & np.isfinite(dT)
print("\n2025 minus 2024: spatial correlation between sea level change and GLORYS 100 m temperature change "
      "over %d deep cells: r = %.2f" % (sel.sum(), np.corrcoef(dsla[sel], dT[sel])[0, 1]))
print("sea level change 2025 minus 2024 by region (cm); columns lon 45-60, 60-75, 75-90, 90-105")
for la0, la1 in [(5, 12), (12, 20), (20, 30)]:
    row = []
    for lo0, lo1 in [(45, 60), (60, 75), (75, 90), (90, 105)]:
        m = deep & (lat[:, None] >= la0) & (lat[:, None] < la1) & (lon[None, :] >= lo0) & (lon[None, :] < lo1)
        row.append("%+.1f" % (100 * np.nanmean(dsla[m])) if m.any() else " n/a ")
    print(f"   lat {la0:>2}-{la1:<2}N | " + "   ".join(row))
