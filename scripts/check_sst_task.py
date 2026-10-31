"""Pipeline output for 2022-01 vs the 3-day test file processed directly, and January vs GLORYS at 0 m."""
import numpy as np
import pandas as pd
import xarray as xr

from oceanembed.glorys_pipeline import block_dates
from oceanembed.sst_pipeline import regrid_days, task_path

v = np.load(task_path("2022-01"))["values"]
print("pipeline file:", v.shape, "| NaN fraction %.3f" % np.isnan(v).mean())
ds = xr.open_dataset("data/raw/test/ostia_3days.nc")
a = ds["analysed_sst"].transpose("time", "latitude", "longitude").values.astype(float)
a = a - 273.15 if np.nanmean(a) > 200 else a
ref = regrid_days(a, ds["latitude"].values.astype(float), ds["longitude"].values.astype(float))
both = np.isfinite(ref) & np.isfinite(v[:3])
print("NaN pattern identical on the 3 test days:", bool(np.array_equal(np.isfinite(ref), np.isfinite(v[:3]))))
print("largest difference vs the 3-day route: %.2e C" % float(np.abs(ref[both] - v[:3][both]).max()))
mask = np.load("data/processed/target_mask_15depths.npy")
deep, ocean = mask.all(axis=0), mask[0]
i0 = (pd.Timestamp("2022-01-01") - block_dates(5)[0]).days
g = np.asarray(np.load("data/processed/glorys_block5.npy", mmap_mode="r")[i0:i0 + 31, 0])
for name, m in (("deep set", deep), ("all GLORYS ocean cells", ocean)):
    k = m[None] & np.isfinite(v) & np.isfinite(g)
    d = v[k] - g[k]
    print("OSTIA minus GLORYS at 0 m, Jan 2022, %s: n=%d | mean %+.2f | RMS %.2f C" % (name, k.sum(), d.mean(), np.sqrt((d ** 2).mean())))
