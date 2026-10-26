"""Copernicus L4 hourly wind: (1) error of a 4-sample daily mean versus all 24 hours, from the file on disk;
(2) chunk layout of both services; (3) timed lazy download of 4 hours per day for 3 days."""
import time

import numpy as np
import pandas as pd
import xarray as xr
import copernicusmarine

DATASET = "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H"
VARS = ["eastward_wind", "northward_wind"]

print("1. daily mean from 4 samples (00, 06, 12, 18 UTC) versus all 24 hours, from the file on disk")
ds = xr.open_dataset("data/raw/test/wind_l4_3days.nc")
for v in VARS:
    diffs = []
    for day in pd.date_range("2022-01-01", periods=3):
        d = ds[v].sel(time=slice(day, day + pd.Timedelta(hours=23)))
        h4 = d.sel(time=[day + pd.Timedelta(hours=h) for h in (0, 6, 12, 18)]).mean("time")
        diffs.append((h4 - d.mean("time")).values)
    diffs = np.stack(diffs)
    print(f"   {v}: RMS difference {np.sqrt(np.mean(diffs ** 2)):.3f} m/s | largest {np.abs(diffs).max():.2f} m/s")


def open_ds(service):
    try:
        return copernicusmarine.open_dataset(dataset_id=DATASET, variables=VARS, service=service)
    except TypeError:
        return copernicusmarine.open_dataset(dataset_id=DATASET, service=service)


print("\n2. chunk layout")
for service in ("geoseries", "timeseries"):
    try:
        da = open_ds(service)[VARS[0]]
        print(f"   {service}: dims {dict(da.sizes)} | preferred_chunks {da.encoding.get('preferred_chunks')} "
              f"| stored dtype {da.encoding.get('dtype')}")
    except Exception as e:
        print(f"   {service}: failed {repr(e)[:200]}")

print("\n3. timed lazy load: 4 hours per day for 3 days (12 time steps), geoseries service")
try:
    z = open_ds("geoseries")
    hours = [pd.Timestamp("2022-01-01") + pd.Timedelta(days=d, hours=h) for d in range(3) for h in (0, 6, 12, 18)]
    sub = z[VARS].sel(latitude=slice(4.8, 30.2), longitude=slice(44.8, 105.2)).sel(time=hours)
    t0 = time.time()
    sub = sub.load()
    dt = time.time() - t0
    print(f"   {len(hours)} time steps loaded in {dt:.0f} s -> {dt / len(hours):.1f} s per step "
          f"(24-hour days earlier: 286 s / 72 steps = 4.0 s per step)")
    steps = 1938 * 4
    print(f"   at this speed, {steps} steps (every 3rd day, 4 samples a day) would take about "
          f"{steps * dt / len(hours) / 3600:.1f} hours on one stream")
    print("   shape", dict(sub.sizes), "| max |u| %.1f m/s" % float(abs(sub[VARS[0]]).max()))
except Exception as e:
    print("   lazy load failed:", repr(e)[:400])
