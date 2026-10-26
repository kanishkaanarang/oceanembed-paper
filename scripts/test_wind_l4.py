"""Test the Copernicus L4 hourly wind product on 3 days: timing, grid, variables, ranges, and a comparison
with the CCMP daily file we already have (2022-01-01). Downloads a few tens of MB."""
import os
import subprocess
import time

import numpy as np
import xarray as xr

from oceanembed.grid import load_ccmp_day, target_grid

DATASET = "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H"
OUT, FNAME = "data/raw/test", "wind_l4_3days.nc"
path = os.path.join(OUT, FNAME)
os.makedirs(OUT, exist_ok=True)
if os.path.exists(path):
    os.remove(path)
cmd = [
    "copernicusmarine", "subset", "--dataset-id", DATASET,
    "--variable", "eastward_wind", "--variable", "northward_wind",
    "--minimum-longitude", "44.8", "--maximum-longitude", "105.2",
    "--minimum-latitude", "4.8", "--maximum-latitude", "30.2",
    "--start-datetime", "2022-01-01T00:00:00", "--end-datetime", "2022-01-03T23:00:00",
    "--output-directory", OUT, "--output-filename", FNAME,
]
t = time.time()
r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL)
print("return code", r.returncode, "| %.0f s for 3 days" % (time.time() - t))
print(r.stdout[-400:])
if r.returncode != 0 or not os.path.exists(path):
    print(r.stderr[-800:])
    raise SystemExit("download failed; paste the message above")

ds = xr.open_dataset(path)
print(ds)
la, lo = ds["latitude"].values, ds["longitude"].values
print("lat step", np.unique(np.round(np.diff(la), 4)), "| first lats", la[:3], "| last lat", la[-1])
print("lon step", np.unique(np.round(np.diff(lo), 4)), "| first lons", lo[:3], "| last lon", lo[-1])
tt = ds["time"].values
print("times: n =", len(tt), "|", str(tt[0]), "to", str(tt[-1]), "| steps", np.unique(np.diff(tt)))
for v in ("eastward_wind", "northward_wind"):
    a = ds[v].values
    fin = np.isfinite(a)
    same = all(np.array_equal(fin[0], fin[i]) for i in range(1, fin.shape[0]))
    print(f"{v}: units {ds[v].attrs.get('units')} | NaN fraction {np.isnan(a).mean():.3f} | "
          f"min {np.nanmin(a):.1f} | max {np.nanmax(a):.1f} | validity pattern same every hour: {same}")
print("file size %.1f MB" % (os.path.getsize(path) / 1e6))

lat, lon = target_grid()
l4 = (ds[["eastward_wind", "northward_wind"]].isel(time=slice(0, 24)).mean("time", skipna=False)
      .interp(latitude=lat, longitude=lon))
cc = load_ccmp_day("data/raw/test/CCMP_Wind_Analysis_20220101_V03.1_L4.nc")
ocean = np.load("data/processed/target_mask_15depths.npy")[0]
print("\nL4 hourly product (daily mean of the first 24 steps) vs CCMP on 2022-01-01, GLORYS ocean cells:")
for a_name, b_name in (("eastward_wind", "wind_u"), ("northward_wind", "wind_v")):
    a, b = l4[a_name].values, cc[b_name].values
    m = ocean & np.isfinite(a) & np.isfinite(b)
    print(f"  {a_name}: cells {int(m.sum())} | correlation {np.corrcoef(a[m], b[m])[0, 1]:.3f} | "
          f"RMS difference {np.sqrt(np.mean((a[m] - b[m]) ** 2)):.2f} m/s | mean difference {np.mean(a[m] - b[m]):+.2f} m/s")
