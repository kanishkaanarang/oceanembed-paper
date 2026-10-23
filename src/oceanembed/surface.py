"""2-D surface fields on a 0.125-degree grid offset by half a cell -> the 0.25-degree target grid.

Each 0.25-degree target cell is exactly tiled by 2x2 native cells, so the 2x2 mean is an area average.
Strict rule: a target value is NaN if any of the four native cells is NaN (nothing is filled).
Downloads need a margin of at least 0.0625 degrees beyond the domain edges.
"""
import numpy as np
import xarray as xr

from .grid import target_grid

H = 0.0625   # half of the 0.125-degree native step


def load_month_2x2(path, var):
    """Return (times, array) with array of shape (time, 101, 241), float32, on the target grid."""
    ds = xr.open_dataset(path)
    da = ds[var]
    if "depth" in da.dims:
        da = da.isel(depth=0, drop=True)
    da = da.transpose("time", "latitude", "longitude")
    lat, lon = target_grid()
    parts = []
    for dlat in (-H, H):
        for dlon in (-H, H):
            # exact selection: raises KeyError if a neighbouring native cell is missing
            sel = da.sel(latitude=lat + dlat, longitude=lon + dlon, method="nearest", tolerance=1e-3)
            parts.append(sel.values.astype("float32"))
    out = np.mean(parts, axis=0).astype("float32")   # NaN if any of the four cells is NaN
    return ds["time"].values, out
