"""Timeseries service test for the Copernicus L4 hourly wind: one 16 x 16-cell tile, one full time chunk."""
import math
import os
import subprocess
import time

import numpy as np
import pandas as pd
import xarray as xr
import copernicusmarine

DATASET = "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H"
VARS = ["eastward_wind", "northward_wind"]
TCHUNK = 18567            # hours per time chunk in the timeseries layout (from the chunk check)
K = 3                     # time chunk to test
CELL = 0.125
LAT0, LON0 = -89.9375, -179.9375   # centres of the first cells; verified against the file below
ILAT, ILON, N = 840, 1960, 16      # tile start indices (multiples of 4) and size in cells

z = copernicusmarine.open_dataset(dataset_id=DATASET, variables=VARS, service="timeseries")
la0, lo0 = float(z["latitude"].values[0]), float(z["longitude"].values[0])
tv = z["time"].values
t0 = pd.Timestamp(tv[0])
print("dataset starts", t0, "| first lat %.4f | first lon %.4f | time steps %d" % (la0, lo0, len(tv)))
if abs(la0 - LAT0) > 1e-3 or abs(lo0 - LON0) > 1e-3:
    raise SystemExit("grid origin differs from the assumption; paste the line above")
start = t0 + pd.Timedelta(hours=K * TCHUNK)
end = start + pd.Timedelta(hours=TCHUNK - 1)
if tv[K * TCHUNK] != np.datetime64(start) or tv[K * TCHUNK + TCHUNK - 1] != np.datetime64(end):
    raise SystemExit("time axis is not strictly hourly from the first step; paste the line above")
lat_min, lat_max = LAT0 + ILAT * CELL - 0.01, LAT0 + (ILAT + N - 1) * CELL + 0.01
lon_min, lon_max = LON0 + ILON * CELL - 0.01, LON0 + (ILON + N - 1) * CELL + 0.01
print("tile lat %.3f..%.3f lon %.3f..%.3f | chunk %d: %s to %s" % (lat_min, lat_max, lon_min, lon_max, K, start, end))

OUT, FNAME = "data/raw/test", "wind_tile.nc"
path = os.path.join(OUT, FNAME)
if os.path.exists(path):
    os.remove(path)
cmd = [
    "copernicusmarine", "subset", "--dataset-id", DATASET, "--service", "timeseries",
    "--variable", VARS[0], "--variable", VARS[1],
    "--minimum-longitude", f"{lon_min:.4f}", "--maximum-longitude", f"{lon_max:.4f}",
    "--minimum-latitude", f"{lat_min:.4f}", "--maximum-latitude", f"{lat_max:.4f}",
    "--start-datetime", start.strftime("%Y-%m-%dT%H:%M:%S"), "--end-datetime", end.strftime("%Y-%m-%dT%H:%M:%S"),
    "--output-directory", OUT, "--output-filename", FNAME,
]
t = time.time()
try:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1500, stdin=subprocess.DEVNULL)
except subprocess.TimeoutExpired:
    raise SystemExit("TIMEOUT after 25 minutes: the timeseries route is too slow for this tile")
dt = time.time() - t
print("return code", r.returncode, "| %.0f s" % dt)
print(r.stdout[-300:])
if r.returncode != 0 or not os.path.exists(path):
    print(r.stderr[-600:])
    raise SystemExit("download failed; paste the message above")

ds = xr.open_dataset(path)
print("sizes:", dict(ds.sizes), "| expected time", TCHUNK, "lat/lon", N)
print("lat %.4f..%.4f | lon %.4f..%.4f" % (float(ds.latitude.min()), float(ds.latitude.max()),
                                           float(ds.longitude.min()), float(ds.longitude.max())))
tt = pd.DatetimeIndex(ds["time"].values)
counts = pd.Series(1, index=tt).groupby(tt.normalize()).sum()
print("days touched: %d | hours on the first day: %d | on the last day: %d | full days: %d"
      % (len(counts), counts.iloc[0], counts.iloc[-1], int((counts == 24).sum())))
for v in VARS:
    a = ds[v].values
    print("%s: NaN fraction %.3f | min %.1f | max %.1f m/s" % (v, np.isnan(a).mean(), np.nanmin(a), np.nanmax(a)))
print("file size %.1f MB" % (os.path.getsize(path) / 1e6))
tiles = math.ceil(484 / N) * math.ceil(204 / N)
print("\nprojection (a rough extrapolation): %d tiles x 8 time chunks = %d tile-chunks; one stream at this speed: %.0f hours"
      % (tiles, tiles * 8, tiles * 8 * dt / 3600))
print("the GLORYS pipeline gained roughly 2 to 5 times from grouping tiles and running workers in parallel")
