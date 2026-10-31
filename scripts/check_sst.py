"""Assemble (if needed) and check the SST array: gaps, range, yearly means, month-boundary jumps, and a
date-alignment test against GLORYS at 0 m."""
import os

import numpy as np
import pandas as pd

from oceanembed.glorys_pipeline import block_dates
from oceanembed.sst_assemble import assemble_sst

path = "data/processed/sst_all.npy"
if not os.path.exists(path):
    print("assembling sst ...")
    assemble_sst(path)
arr = np.load(path, mmap_mode="r")
dates = pd.date_range("2010-02-01", "2025-12-31")
assert arr.shape == (len(dates), 101, 241), arr.shape
mask = np.load("data/processed/target_mask_15depths.npy")
ocean, deep = mask[0], mask.all(axis=0)
print("array:", arr.shape, "| %.2f GB" % (os.path.getsize(path) / 1e9))

full = np.asarray(arr)
x = full[:, deep]
ov = full[:, ocean]
print("missing values in the deep set:", int((~np.isfinite(x)).sum()))
print("ocean cells with a missing value on some day: %d of %d" % (int((~np.isfinite(ov)).any(axis=0).sum()), int(ocean.sum())))
print("range over ocean cells: %.2f to %.2f C" % (np.nanmin(ov), np.nanmax(ov)))
del ov

s = pd.Series(x.mean(axis=1), index=dates)
print("\nyearly mean SST over the deep set (C):")
print(s.groupby(s.index.year).mean().round(2).to_string())

ch = np.abs(np.diff(x, axis=0)).mean(axis=1)          # change between day i-1 and day i is ch[i-1]
starts = [i for i in range(1, len(dates)) if dates[i].day == 1]
at = [ch[i - 1] for i in starts]
rest = np.delete(ch, [i - 1 for i in starts])
print("\nmonth boundaries: mean |day-to-day change| at month starts %.3f C | other days %.3f C | largest at a month start %.3f C"
      % (np.mean(at), rest.mean(), max(at)))

blocks = {b: (pd.date_range(*block_dates(b)), np.load(f"data/processed/glorys_block{b}.npy", mmap_mode="r")) for b in (3, 4, 5)}


def glorys0(day):
    for dd, a in blocks.values():
        if dd[0] <= day <= dd[-1]:
            return np.asarray(a[int((day - dd[0]).days), 0])
    raise KeyError(day)


test_days = pd.date_range("2010-03-01", "2025-11-30", freq="45D")
print("\ndate-alignment test: RMS of SST(date) minus GLORYS 0 m(date + lag) over the deep set, mean of %d days" % len(test_days))
res = {}
for lag in range(-3, 4):
    r = []
    for day in test_days:
        xs = full[int((day - dates[0]).days)][deep]
        gy = glorys0(day + pd.Timedelta(days=lag))[deep]
        r.append(np.sqrt(np.mean((xs - gy) ** 2)))
    res[lag] = float(np.mean(r))
    print(f"   lag {lag:+d} d: {res[lag]:.3f} C")
print("best lag:", min(res, key=res.get), "(expected 0)")
