"""GLORYS onto the 0.25-degree grid: point value vs strict 3x3 box mean (first 3 days of January 2022).
Strict: a cell is valid only if all 9 native cells have data at both bracketing levels."""
import numpy as np
import xarray as xr

from oceanembed.glorys import TARGET_DEPTHS, interpolate_depths
from oceanembed.grid import target_grid

PATH = "data/raw/glorys/glorys_2022-01.nc"
NDAYS = 3

ds = xr.open_dataset(PATH)
lat, lon = target_grid()
assert ds.sizes["latitude"] == 301 and ds.sizes["longitude"] == 721, dict(ds.sizes)
assert np.allclose(ds["latitude"].values[::3], lat, atol=1e-3)
assert np.allclose(ds["longitude"].values[::3], lon, atol=1e-3)
levels = ds["depth"].values.astype("float64")
nat = (ds["thetao"].isel(time=slice(0, NDAYS))
       .transpose("time", "depth", "latitude", "longitude").values.astype("float32"))

# interior target points only (lat index 1..99, lon index 1..239): the file has no margin at the edges
point = interpolate_depths(nat[..., 3:298:3, 3:718:3], levels)
total = np.zeros((NDAYS, len(levels), 99, 239), dtype=np.float32)
for di in (-1, 0, 1):
    for dj in (-1, 0, 1):
        total += nat[..., 3 + di:298 + di:3, 3 + dj:718 + dj:3]
box = interpolate_depths(total / 9.0, levels)   # NaN if any of the 9 cells is NaN

print("interior cells:", point.shape[-2] * point.shape[-1], "| days:", NDAYS)
print("depth | point-valid | box-valid | lost | RMS diff (C) | mean diff box-point (C)")
for k, z in enumerate(TARGET_DEPTHS):
    p, b = point[:, k], box[:, k]
    pv, bv = np.isfinite(p), np.isfinite(b)
    assert not (bv & ~pv).any()
    d = (b - p)[pv & bv]
    print(f"{z:>5.0f} m | {int(pv[0].sum()):>6d} | {int(bv[0].sum()):>6d} | {int((pv & ~bv)[0].sum()):>5d} "
          f"| {np.sqrt(np.mean(d ** 2)):.3f} | {d.mean():+.3f}")
