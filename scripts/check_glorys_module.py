"""Checks for oceanembed.glorys: synthetic tests first, then the real January 2022 file."""
import os

import numpy as np
import xarray as xr

from oceanembed.glorys import TARGET_DEPTHS, depth_weights, glorys_to_target, interpolate_depths, static_mask

PATH = "data/raw/glorys/glorys_2022-01.nc"
# Valid cells per target depth on 2022-01-01, from scripts/inspect_glorys.py (a separate code path)
EXPECTED_DAY0 = [12038, 12038, 11695, 11417, 11089, 10614, 10190, 9929, 9874, 9813, 9723, 9640, 9480, 9363, 9140]

levels = xr.open_dataset(PATH)["depth"].values.astype("float64")
print("native levels: %d | first %.3f m | last %.1f m" % (len(levels), levels[0], levels[-1]))

# Test 1: a linear profile must be recovered exactly (0 m takes the first level's value)
profile = (20.0 - 0.01 * levels).astype(np.float32).reshape(-1, 1, 1)
out = interpolate_depths(profile, levels)[:, 0, 0]
expected = 20.0 - 0.01 * np.where(TARGET_DEPTHS < levels[0], levels[0], TARGET_DEPTHS)
assert np.allclose(out, expected, atol=1e-4), (out, expected)
print("test 1 passed: linear profile recovered at all 15 depths")

# Test 2: seafloor between 266 m and 318 m -> targets up to 200 m valid, 300 m and deeper NaN
p2 = profile.copy()
p2[levels > 300] = np.nan
o2 = interpolate_depths(p2, levels)[:, 0, 0]
assert np.array_equal(np.isfinite(o2), TARGET_DEPTHS <= 200), o2
print("test 2 passed: nothing is filled below the seafloor")

# Test 3: a download that stops above 1000 m must raise, not return NaN quietly
try:
    depth_weights(levels[levels <= 900])
    raise SystemExit("test 3 FAILED: no error for a too-shallow download")
except ValueError:
    print("test 3 passed: too-shallow download is rejected")

# Test 4: a land column stays NaN
land = np.full((len(levels), 1, 1), np.nan, dtype=np.float32)
assert np.isnan(interpolate_depths(land, levels)).all()
print("test 4 passed: land stays NaN")

# Real data
times, out = glorys_to_target(PATH)
print("\nreal file: shape", out.shape, "| dates", str(times[0])[:10], "to", str(times[-1])[:10])
valid = np.isfinite(out)
ref = static_mask(valid)
print("validity pattern identical on all %d days: True" % out.shape[0])
counts = ref.sum(axis=(1, 2)).tolist()
print("valid cells per target depth:", counts)
print("matches the independent count for 2022-01-01:", counts == EXPECTED_DAY0)

v = out[valid]
print("temperature range: %.2f to %.2f C" % (v.min(), v.max()))
print("mean temperature by depth (C):", [round(float(np.nanmean(out[:, k])), 2) for k in range(len(TARGET_DEPTHS))])
d0 = out[0, 0]
print("day 1 at 0 m: min %.2f, max %.2f (previous script said 15.62 and 30.58)" % (np.nanmin(d0), np.nanmax(d0)))

os.makedirs("data/processed", exist_ok=True)
np.save("data/processed/target_mask_15depths.npy", ref)
print("saved data/processed/target_mask_15depths.npy", ref.shape, ref.dtype)
