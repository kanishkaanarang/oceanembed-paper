"""Common 0.25-degree grid and loaders for the surface predictor variables."""
import numpy as np
import xarray as xr

LON_MIN, LON_MAX = 45.0, 105.0
LAT_MIN, LAT_MAX = 5.0, 30.0
STEP = 0.25


def target_grid():
    """Return (lat, lon) 1-D arrays of the common 0.25-degree grid."""
    lat = np.arange(LAT_MIN, LAT_MAX + STEP / 2, STEP)
    lon = np.arange(LON_MIN, LON_MAX + STEP / 2, STEP)
    return lat, lon


def load_oscar_day(path):
    """OSCAR daily file -> cur_u, cur_v on the target grid (dims: lat, lon)."""
    ds = xr.open_dataset(path)
    # lon/lat are plain variables here; make them the index coordinates
    ds = ds.assign_coords(longitude=ds["lon"].values, latitude=ds["lat"].values)
    ds = ds.drop_vars(["lon", "lat"])
    ds = ds[["u", "v"]].transpose("time", "latitude", "longitude").isel(time=0, drop=True)
    ds = ds.rename({"latitude": "lat", "longitude": "lon", "u": "cur_u", "v": "cur_v"})
    lat, lon = target_grid()
    # exact selection: raises KeyError if a target point is not on the native grid
    return ds.sel(lat=lat, lon=lon, method="nearest", tolerance=1e-6).load()


def load_ccmp_day(path):
    """CCMP daily file (4 six-hourly steps) -> wind_u, wind_v daily mean on the target grid."""
    ds = xr.open_dataset(path)[["uwnd", "vwnd"]]
    n = ds.sizes["time"]
    if n != 4:
        raise ValueError(f"expected 4 six-hourly steps, found {n} in {path}")
    # crop first (a 1-degree margin is enough for the interpolation neighbours)
    ds = ds.sel(
        latitude=slice(LAT_MIN - 1, LAT_MAX + 1),
        longitude=slice(LON_MIN - 1, LON_MAX + 1),
    )
    daily = ds.mean("time", skipna=False)  # average the u and v components
    lat, lon = target_grid()
    out = daily.interp(latitude=lat, longitude=lon, method="linear")
    out = out.rename({"latitude": "lat", "longitude": "lon", "uwnd": "wind_u", "vwnd": "wind_v"})
    return out.load()
