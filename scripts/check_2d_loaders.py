"""Check the 2x2 loaders on January 2022 SSH and SSS: structure, masks, usable cells."""
import numpy as np
import xarray as xr

from oceanembed.grid import load_oscar_day
from oceanembed.surface import load_month_2x2

mask = np.load("data/processed/target_mask_15depths.npy")   # (15, 101, 241), True where GLORYS has a target
ocean = mask[0]                  # GLORYS ocean cells at 0 m
deep = mask.all(axis=0)          # cells with all 15 depths: the headline set
print("GLORYS ocean cells:", int(ocean.sum()), "| deep set:", int(deep.sum()))

fields = {}
for name, var in (("ssh", "sla"), ("sss", "sos")):
    times, arr = load_month_2x2(f"data/raw/test/{name}_2022-01.nc", var)
    fields[name] = arr
    v = np.isfinite(arr)
    static = all(np.array_equal(v[0], v[i]) for i in range(1, v.shape[0]))
    d0 = v[0]
    print("=" * 70)
    print(name, var, "| shape", arr.shape, "| times", str(times[0]), "to", str(times[-1]))
    print("validity pattern identical on all days:", static)
    print("valid cells day 1:", int(d0.sum()), "| both with GLORYS:", int((d0 & ocean).sum()),
          "| GLORYS only:", int((ocean & ~d0).sum()), "| %s only:" % name, int((d0 & ~ocean).sum()))
    print("range: %.3f to %.3f" % (np.nanmin(arr), np.nanmax(arr)))

raw = xr.open_dataset("data/raw/test/sss_2022-01.nc")["sos"]
print("\nSSS native: max %.3f | cells equal to 40.000: %d | valid_max attribute: %s | units: %s" % (
    float(raw.max()), int((raw.values == 40.0).sum()), raw.attrs.get("valid_max"), raw.attrs.get("units")))

oscar = np.isfinite(load_oscar_day("data/raw/test/oscar_currents_final_20220101.nc").cur_u.values)
usable = oscar & np.isfinite(fields["ssh"][0]) & np.isfinite(fields["sss"][0])
print("\nusable inputs on 2022-01-01 (OSCAR and SSH and SSS valid):", int(usable.sum()), "cells")
print("  among the GLORYS ocean cells:", int((usable & ocean).sum()), "of", int(ocean.sum()))
print("  among the deep set:", int((usable & deep).sum()), "of", int(deep.sum()))
