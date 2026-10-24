"""Assemble one GLORYS block (if needed) and check it: coverage, validity pattern, yearly statistics.
Usage: python scripts/check_block.py 5"""
import os
import sys

import numpy as np
import pandas as pd

from oceanembed.glorys import TARGET_DEPTHS
from oceanembed.glorys_assemble import assemble_block
from oceanembed.glorys_pipeline import block_dates

block = int(sys.argv[1]) if len(sys.argv) > 1 else 5
path = f"data/processed/glorys_block{block}.npy"
if not os.path.exists(path):
    print("assembling block", block, "...")
    assemble_block(block, path)
arr = np.load(path, mmap_mode="r")
start, end = block_dates(block)
dates = pd.date_range(start, end)
mask = np.load("data/processed/target_mask_15depths.npy")
deep = mask.all(axis=0)
print("array:", arr.shape, "| dates", dates[0].date(), "to", dates[-1].date(), "| file %.2f GB" % (os.path.getsize(path) / 1e9))
assert arr.shape[0] == len(dates)

bad = 0
for s in range(0, arr.shape[0], 100):
    v = np.isfinite(np.asarray(arr[s:s + 100]))
    bad += int((v != mask[None]).any(axis=(1, 2, 3)).sum())
print("days whose validity pattern differs from the reference mask:", bad, "of", arr.shape[0])
print("deep-set cells:", int(deep.sum()))

cols = {}
for z in (0, 100, 300, 1000):
    k = int(np.where(TARGET_DEPTHS == z)[0][0])
    cols[f"T{z}m"] = np.asarray(arr[:, k])[:, deep].mean(axis=1)
df = pd.DataFrame(cols, index=dates)
k100 = int(np.where(TARGET_DEPTHS == 100)[0][0])
x100 = np.asarray(arr[:, k100])[:, deep]
noise = pd.Series(np.abs(np.diff(x100, axis=0)).mean(axis=1), index=dates[1:])
df["noise100m"] = noise
print("\nyearly means over the deep set (C); noise100m = mean |day-to-day change| at 100 m")
print(df.groupby(df.index.year).mean().round(3))
print("days per year:", df.groupby(df.index.year).size().to_dict())
m = noise.groupby(noise.index.to_period("M")).mean()
print("\nmonthly noise at 100 m, %s to %s:" % (m.index[0], m.index[-1]))
print([round(float(v), 3) for v in m.values])
