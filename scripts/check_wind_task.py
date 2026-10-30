"""One pipeline task (time chunk 7, group 26,61) must equal the 3-day test file (2022-01-01 to 03) processed
directly with the same 2x2 area-mean rule."""
import numpy as np
import xarray as xr

from oceanembed.grid import target_grid
from oceanembed.wind_pipeline import VARS, group_targets, task_path

A, B, K = 26, 61, 7
z = np.load(task_path(A, B, K))
sums, counts, days, ti, tj = z["sums"], z["counts"], z["days"], z["ti"], z["tj"]
print("task file: sums", sums.shape, "| days", days[0], "to", days[-1], "| hours per day: first", int(counts[0]),
      "last", int(counts[-1]), "interior all 24:", bool((counts[1:-1] == 24).all()))
lat, lon = target_grid()
_, _, ti2, tj2 = group_targets(A, B)
assert np.array_equal(ti, ti2) and np.array_equal(tj, tj2)

want = np.array(["2022-01-01", "2022-01-02", "2022-01-03"], dtype="datetime64[D]")
idx = np.searchsorted(days, want)
assert (days[idx] == want).all() and (counts[idx] == 24).all()
pipe = sums[idx] / 24.0

ds = xr.open_dataset("data/raw/test/wind_l4_3days.nc")
by_var = []
for v in VARS:
    parts = []
    for dlat in (-0.0625, 0.0625):
        for dlon in (-0.0625, 0.0625):
            sel = ds[v].sel(latitude=lat[ti] + dlat, longitude=lon[tj] + dlon, method="nearest", tolerance=1e-3)
            parts.append(sel.values)
    by_var.append(np.mean(parts, axis=0))
ref = np.stack(by_var, axis=1).reshape(3, 24, 2, len(ti), len(tj)).mean(axis=1)
d = np.abs(pipe - ref)
print("targets in the group:", len(ti), "x", len(tj))
print("largest absolute difference pipeline vs direct: %.2e m/s (mean %.2e)" % (d.max(), d.mean()))
print("wind range in the group: u %.1f to %.1f | v %.1f to %.1f m/s" % (
    pipe[:, 0].min(), pipe[:, 0].max(), pipe[:, 1].min(), pipe[:, 1].max()))
