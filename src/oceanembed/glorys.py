"""GLORYS helpers: native 1/12-degree grid -> 0.25-degree target grid and 15 target depths."""

import numpy as np
import xarray as xr

TARGET_DEPTHS = (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000)
STRIDE = 3  # native 1/12 degree -> every 3rd cell is exactly on the 0.25-degree grid


def to_quarter_degree(da):
    """Exact selection of every 3rd native cell (no averaging)."""
    return da.isel(latitude=slice(None, None, STRIDE), longitude=slice(None, None, STRIDE))


def _bracket(levels, z, extrapolate_top):
    """Return (k0, k1, w) so that value = (1 - w) * x[k0] + w * x[k1]; None if z is not allowed."""
    if z > levels[-1]:
        raise ValueError(f"target depth {z} m is below the deepest level {levels[-1]:.1f} m")
    k1 = int(np.searchsorted(levels, z, side="left"))  # first native level >= z
    if k1 == 0:
        if z == levels[0]:
            return 0, 0, 0.0
        if not extrapolate_top:
            return None
        k0, k1 = 0, 1  # linear extrapolation upward from the two shallowest levels
    else:
        k0 = k1 - 1
    w = (z - levels[k0]) / (levels[k1] - levels[k0])
    return k0, k1, float(w)


def to_target_depths(da, targets=TARGET_DEPTHS, extrapolate_top=False):
    """Linear interpolation in depth from native levels to the target depths.

    A target depth is NaN wherever either bracketing native level is NaN (land or seafloor).
    Call to_quarter_degree first so the arrays stay small. Returns float32, same dim order.
    """
    da = da.astype("float32")
    levels = da["depth"].values.astype("float64")
    pieces = []
    for z in targets:
        br = _bracket(levels, float(z), extrapolate_top)
        if br is None:
            v = xr.full_like(da.isel(depth=0, drop=True), np.nan)
        else:
            k0, k1, w = br
            v = (1 - w) * da.isel(depth=k0, drop=True) + w * da.isel(depth=k1, drop=True)
        pieces.append(v.expand_dims(depth=[np.float32(z)]))
    out = xr.concat(pieces, dim="depth").astype("float32")
    return out.transpose(*da.dims)
