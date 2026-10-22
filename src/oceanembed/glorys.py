"""GLORYS thetao -> the 15 target depths on the common 0.25-degree grid, NaN where no honest value exists.

Rules:
- horizontal: exact 0.25-degree grid points (they are native GLORYS points); no averaging
- depth: linear interpolation between the two native levels that bracket each target depth
- a target shallower than the first native level (0 m vs 0.494 m) uses the first level's value
- if either bracketing level has no data (land or below the seafloor) the result is NaN:
  never filled, never extrapolated downward
"""
import numpy as np
import xarray as xr

from .grid import target_grid

TARGET_DEPTHS = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], dtype=float)


def depth_weights(levels, targets=TARGET_DEPTHS):
    """Return (i_lo, i_hi, w): result = (1 - w) * level[i_lo] + w * level[i_hi]."""
    levels = np.asarray(levels, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if levels.ndim != 1 or np.any(np.diff(levels) <= 0):
        raise ValueError("levels must be 1-D and strictly increasing")
    if targets.max() > levels[-1]:
        raise ValueError(
            f"deepest target {targets.max():.0f} m is below the deepest level {levels[-1]:.1f} m; "
            "download deeper (at least to 1100 m)"
        )
    i_hi = np.searchsorted(levels, targets, side="left")
    i_lo = np.where(levels[i_hi] == targets, i_hi, i_hi - 1)
    i_lo = np.maximum(i_lo, 0)
    span = levels[i_hi] - levels[i_lo]
    w = np.zeros(len(targets))
    inner = span > 0
    w[inner] = (targets[inner] - levels[i_lo][inner]) / span[inner]
    return i_lo, i_hi, w


def interpolate_depths(arr, levels, targets=TARGET_DEPTHS):
    """arr: (..., depth, lat, lon), NaN where there is no data. Returns (..., n_targets, lat, lon) float32."""
    i_lo, i_hi, w = depth_weights(levels, targets)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[-3] != len(levels):
        raise ValueError(f"depth axis has {arr.shape[-3]} entries but {len(levels)} levels were given")
    lo = np.take(arr, i_lo, axis=-3)
    hi = np.take(arr, i_hi, axis=-3)
    w = w.astype(np.float32).reshape(-1, 1, 1)
    return ((1.0 - w) * lo + w * hi).astype(np.float32)


def glorys_to_target(path, chunk_days=8):
    """Read a GLORYS thetao file (time, depth, latitude, longitude).
    Returns (times, array) with array of shape (time, 15, 101, 241), float32, on the target grid."""
    ds = xr.open_dataset(path)
    levels = ds["depth"].values.astype("float64")
    lat, lon = target_grid()
    # exact selection: raises KeyError if a target point is not on the native grid
    da = ds["thetao"].sel(latitude=lat, longitude=lon, method="nearest", tolerance=1e-3)
    da = da.transpose("time", "depth", "latitude", "longitude")
    times = ds["time"].values
    n = da.sizes["time"]
    if n > 1 and not np.all(np.diff(times) == np.timedelta64(1, "D")):
        raise ValueError("time axis is not consecutive daily steps")
    out = np.empty((n, len(TARGET_DEPTHS), len(lat), len(lon)), dtype=np.float32)
    for s in range(0, n, chunk_days):
        e = min(s + chunk_days, n)
        out[s:e] = interpolate_depths(da.isel(time=slice(s, e)).values, levels)
    return times, out


def static_mask(valid):
    """valid: (time, depth, lat, lon) boolean. Returns the (depth, lat, lon) mask; raises if it changes."""
    ref = valid[0]
    for i in range(1, valid.shape[0]):
        if not np.array_equal(ref, valid[i]):
            raise AssertionError(f"validity pattern changes on day index {i}")
    return ref
