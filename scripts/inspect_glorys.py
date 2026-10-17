"""Inspect one GLORYS day: depth levels, ocean mask, interpolation coverage at the
15 target depths, and mask comparison with OSCAR. Read-only."""
import os
import numpy as np
import xarray as xr
from oceanembed.grid import target_grid, load_oscar_day

GLORYS = "data/raw/test/glorys_day.nc"
OSCAR = "data/raw/test/oscar_currents_final_20220101.nc"
TARGET_DEPTHS = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], dtype=float)

for p in (GLORYS, OSCAR):
    if not os.path.exists(p):
        raise SystemExit(f"missing file: {p}")

g = xr.open_dataset(GLORYS)
levels = g.depth.values.astype(float)
assert np.all(np.diff(levels) > 0), "depth levels are not increasing"
print("dims:", dict(g.sizes))
print("n levels:", len(levels))
print("levels (m):", np.round(levels, 1))

lat, lon = target_grid()
t = g.thetao.isel(time=0).sel(latitude=lat, longitude=lon, method="nearest", tolerance=1e-3)
t = t.transpose("depth", "latitude", "longitude")
arr = t.values
print("selected array shape (depth, lat, lon):", arr.shape)
print("max coordinate offset from target grid (deg): lat %.1e, lon %.1e" % (
    np.abs(t.latitude.values - lat).max(), np.abs(t.longitude.values - lon).max()))

valid = ~np.isnan(arr)
ocean = valid[0]
print("surface temperature range: %.2f to %.2f C" % (np.nanmin(arr[0]), np.nanmax(arr[0])))
print("ocean cells at first level:", int(ocean.sum()), "of", ocean.size)
print("cells with data deeper but not at the first level (should be 0):", int((valid[-1] & ~ocean).sum()))

print("\ntarget | bracketing levels (m) | valid | ocean but no data | note")
for d in TARGET_DEPTHS:
    if d < levels[0]:
        i_lo = i_hi = 0
        ok = valid[0]
        note = "above first level: uses first-level value"
    else:
        i_hi = int(np.searchsorted(levels, d, side="left"))
        if i_hi >= len(levels):
            print(f"{d:>5.0f} m | no level at or below this depth in the file")
            continue
        i_lo = i_hi if levels[i_hi] == d else i_hi - 1
        ok = valid[i_lo] & valid[i_hi]
        note = "exact level" if i_lo == i_hi else "interpolated"
    print(f"{d:>5.0f} m | {levels[i_lo]:>8.1f} to {levels[i_hi]:>8.1f} | {int(ok.sum()):>6d} | {int((ocean & ~ok).sum()):>6d} | {note}")

o = load_oscar_day(OSCAR)
b = o.cur_u.notnull().values
print("\nOSCAR valid cells:", int(b.sum()))
print("both:", int((ocean & b).sum()),
      "| GLORYS only:", int((ocean & ~b).sum()),
      "| OSCAR only:", int((~ocean & b).sum()))
