"""Independent check of the depth interpolation against numpy.interp on random deep cells."""
import numpy as np
import xarray as xr

from oceanembed.glorys import TARGET_DEPTHS, glorys_to_target
from oceanembed.grid import target_grid

PATH = "data/raw/glorys/glorys_2022-01.nc"
times, out = glorys_to_target(PATH)
ds = xr.open_dataset(PATH)
lat, lon = target_grid()
levels = ds["depth"].values.astype("float64")

deep = np.isfinite(out[0]).all(axis=0)          # (lat, lon): cells with all 15 depths
rng = np.random.default_rng(0)
cells = np.argwhere(deep)[rng.choice(int(deep.sum()), 8, replace=False)]
worst = 0.0
for i, j in cells:
    col = (ds["thetao"].isel(time=0)
           .sel(latitude=lat[i], longitude=lon[j], method="nearest", tolerance=1e-3)
           .values.astype("float64"))
    ref = np.interp(TARGET_DEPTHS, levels, col)   # clamps above the first level, like our 0 m rule
    worst = max(worst, float(np.abs(out[0, :, i, j] - ref).max()))
print("cells checked:", len(cells), "| largest difference vs numpy.interp: %.2e C" % worst)
