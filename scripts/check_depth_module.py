import numpy as np
import xarray as xr
from oceanembed.glorys import TARGET_DEPTHS, to_quarter_degree, to_target_depths

ds = xr.open_dataset("data/raw/test/glorys_4days.nc")
da = to_quarter_degree(ds.thetao)
print("quarter-degree block:", dict(da.sizes))

for flag in (False, True):
    out = to_target_depths(da, extrapolate_top=flag)
    print(f"\nextrapolate_top={flag} | shape {out.shape} | dtype {out.dtype}")
    valid = out.notnull().values  # (time, depth, lat, lon)
    same = all(np.array_equal(valid[0], valid[i]) for i in range(1, valid.shape[0]))
    print("same pattern on every day:", same)
    print("cells with data per target depth:", [int(valid[0, k].sum()) for k in range(len(TARGET_DEPTHS))])

# independent check against numpy.interp at one cell that has data down to the deepest level
out = to_target_depths(da, extrapolate_top=False)
col = da.isel(time=0).values  # (depth, lat, lon)
levels = da.depth.values.astype("float64")
iy, ix = np.argwhere(np.isfinite(col[-1]))[0]
print()
for z in (5, 50, 100, 1000):
    k = list(TARGET_DEPTHS).index(z)
    ref = np.interp(z, levels, col[:, iy, ix].astype("float64"))
    got = float(out.isel(time=0).values[k, iy, ix])
    print(f"z={z:>4} m | module {got:.5f} | numpy {ref:.5f} | diff {abs(got - ref):.1e}")
