"""Test OSTIA reprocessed SST: chunk layout, timed 3-day download, grid, units, 0.25-degree area-weighted mean,
and a comparison with GLORYS at 0 m on 2022-01-01. Downloads a few MB."""
import os
import subprocess
import time

import numpy as np
import pandas as pd
import xarray as xr
import copernicusmarine

from oceanembed.glorys_pipeline import block_dates
from oceanembed.grid import target_grid

DATASET, VAR = "METOFFICE-GLO-SST-L4-REP-OBS-SST", "analysed_sst"
OUT, FNAME = "data/raw/test", "ostia_3days.nc"

print("1. chunk layout")
for service in ("geoseries", "timeseries"):
    try:
        try:
            z = copernicusmarine.open_dataset(dataset_id=DATASET, variables=[VAR], service=service)
        except TypeError:
            z = copernicusmarine.open_dataset(dataset_id=DATASET, service=service)
        da = z[VAR]
        print(f"   {service}: dims {dict(da.sizes)} | preferred_chunks {da.encoding.get('preferred_chunks')} "
              f"| dtype {da.encoding.get('dtype')}")
    except Exception as e:
        print(f"   {service}: failed {repr(e)[:200]}")

print("\n2. timed download: 3 days, our region plus margin")
path = os.path.join(OUT, FNAME)
os.makedirs(OUT, exist_ok=True)
if os.path.exists(path):
    os.remove(path)
cmd = [
    "copernicusmarine", "subset", "--dataset-id", DATASET, "--variable", VAR,
    "--minimum-longitude", "44.8", "--maximum-longitude", "105.2",
    "--minimum-latitude", "4.8", "--maximum-latitude", "30.2",
    "--start-datetime", "2022-01-01T00:00:00", "--end-datetime", "2022-01-03T23:59:59",
    "--output-directory", OUT, "--output-filename", FNAME,
]
t = time.time()
r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL)
dt = time.time() - t
print("   return code", r.returncode, "| %.0f s" % dt)
print("  ", r.stdout[-260:].replace("\n", " "))
if r.returncode != 0 or not os.path.exists(path):
    print(r.stderr[-600:])
    raise SystemExit("download failed; paste the message above")

ds = xr.open_dataset(path)
da = ds[VAR]
la, lo = ds["latitude"].values.astype(float), ds["longitude"].values.astype(float)
times = pd.DatetimeIndex(ds["time"].values)
print("\n3. file: dims", dict(ds.sizes), "| times", [str(x) for x in times])
print("   units:", da.attrs.get("units"), "| lat step", np.unique(np.round(np.diff(la), 4)),
      "| first lats", la[:3], "| last lat", la[-1], "| first lons", lo[:3], "| last lon", lo[-1])
a0 = da.isel(time=0).values.astype(float)
if np.nanmean(a0) > 200:
    a0 = a0 - 273.15
    print("   converted from kelvin; mean %.2f C, range %.2f to %.2f, NaN fraction %.3f" % (
        np.nanmean(a0), np.nanmin(a0), np.nanmax(a0), np.isnan(a0).mean()))
else:
    print("   values look like Celsius; range %.2f to %.2f" % (np.nanmin(a0), np.nanmax(a0)))


def axis_weights(centres, targets, half_cell, half_target=0.125):
    W = np.zeros((len(targets), len(centres)))
    for i, tg in enumerate(targets):
        lo_ = np.maximum(centres - half_cell, tg - half_target)
        hi_ = np.minimum(centres + half_cell, tg + half_target)
        W[i] = np.clip(hi_ - lo_, 0, None) / (2 * half_cell)
    return W


lat, lon = target_grid()
hc_lat = float(np.diff(la).mean()) / 2
hc_lon = float(np.diff(lo).mean()) / 2
Wlat, Wlon = axis_weights(la, lat, hc_lat), axis_weights(lo, lon, hc_lon)
print("   weights: native cells spanned per target, lat min %.2f max %.2f | lon min %.2f max %.2f (expect about 5)"
      % (Wlat.sum(1).min(), Wlat.sum(1).max(), Wlon.sum(1).min(), Wlon.sum(1).max()))
valid = np.isfinite(a0)
num = Wlat @ np.where(valid, a0, 0.0) @ Wlon.T
den = Wlat @ valid.astype(float) @ Wlon.T
full = Wlat.sum(1)[:, None] * Wlon.sum(1)[None, :]
cover = den / full
sst = np.where(cover >= 0.5, num / np.maximum(den, 1e-9), np.nan)
strict = cover >= 0.999

mask = np.load("data/processed/target_mask_15depths.npy")
ocean, deep = mask[0], mask.all(axis=0)
print("\n4. 0.25-degree grid: valid if at least half of the cell is covered: %d cells | fully covered: %d" % (
    np.isfinite(sst).sum(), strict.sum()))
print("   GLORYS ocean cells without OSTIA (half-cover rule): %d | deep-set cells without OSTIA: half-cover %d, full-cover %d"
      % ((ocean & ~np.isfinite(sst)).sum(), (deep & ~np.isfinite(sst)).sum(), (deep & ~strict).sum()))
print("   OSTIA cells where GLORYS has no ocean: %d" % (np.isfinite(sst) & ~ocean).sum())

s5, _ = block_dates(5)
i5 = (pd.Timestamp(times[0]).normalize() - s5).days
g0 = np.asarray(np.load("data/processed/glorys_block5.npy", mmap_mode="r")[i5, 0])
for name, m in (("all GLORYS ocean cells", ocean), ("deep set", deep)):
    k = m & np.isfinite(sst) & np.isfinite(g0)
    d = sst[k] - g0[k]
    print("   OSTIA minus GLORYS at 0 m, %s: n=%d | mean %+.2f | RMS %.2f C" % (name, k.sum(), d.mean(), np.sqrt((d ** 2).mean())))
print("\n5. projection (a rough extrapolation): %.0f s for 3 days -> %.0f s per day; 5,810 days on one stream = about %.0f hours"
      % (dt, dt / 3, 5810 * dt / 3 / 3600))
