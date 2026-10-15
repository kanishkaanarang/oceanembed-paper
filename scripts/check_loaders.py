"""Sanity check for the OSCAR and CCMP daily loaders against the raw files."""
import numpy as np
import xarray as xr
from oceanembed.grid import load_oscar_day, load_ccmp_day, target_grid

OSCAR = "data/raw/test/oscar_currents_final_20220101.nc"
CCMP = "data/raw/test/CCMP_Wind_Analysis_20220101_V03.1_L4.nc"

lat, lon = target_grid()
print("target grid:", len(lat), "lat x", len(lon), "lon")

o = load_oscar_day(OSCAR)
c = load_ccmp_day(CCMP)
for name, ds in [("OSCAR", o), ("CCMP", c)]:
    print(name, dict(ds.sizes))
    for v in ds.data_vars:
        a = ds[v].values
        print(f"  {v}: NaN fraction {np.isnan(a).mean():.3f}, min {np.nanmin(a):.2f}, max {np.nanmax(a):.2f}")

# OSCAR: loaded value at (10N, 70E) vs the raw file, index by index
raw = xr.open_dataset(OSCAR)
ilon = int(np.where(raw["lon"].values == 70.0)[0][0])
ilat = int(np.where(raw["lat"].values == 10.0)[0][0])
print("OSCAR u at 10N,70E  raw:", float(raw["u"].values[0, ilon, ilat]),
      " loaded:", float(o.cur_u.sel(lat=10.0, lon=70.0)))

# CCMP: loaded value vs the manual mean of the 4 neighbours over the 4 time steps
rc = xr.open_dataset(CCMP)
manual = float(rc.uwnd.sel(latitude=[9.875, 10.125], longitude=[69.875, 70.125], method="nearest").mean())
print("CCMP uwnd at 10N,70E  manual:", manual, " loaded:", float(c.wind_u.sel(lat=10.0, lon=70.0)))
