"""Where does the 2025 change at 100 m come from? Offline diagnostics on the assembled block 5."""
import numpy as np
import pandas as pd

from oceanembed.glorys import TARGET_DEPTHS
from oceanembed.glorys_pipeline import block_dates
from oceanembed.grid import target_grid

arr = np.load("data/processed/glorys_block5.npy", mmap_mode="r")
mask = np.load("data/processed/target_mask_15depths.npy")
deep = mask.all(axis=0)
start, end = block_dates(5)
dates = pd.date_range(start, end)
years = dates.year.values
lat, lon = target_grid()
k100 = int(np.where(TARGET_DEPTHS == 100)[0][0])

daily = {int(z): np.asarray(arr[:, k])[:, deep].mean(axis=1) for k, z in enumerate(TARGET_DEPTHS)}
df = pd.DataFrame(daily, index=dates)
yearly = df.groupby(df.index.year).mean()
print("1. yearly mean over the deep set (C), by target depth in m (2021 = Jun 28 to Dec 31 only)")
print(yearly.round(2).to_string())
print("\n   year-to-year change (C), full years only")
print(yearly.loc[2022:].diff().dropna().round(2).to_string())

print("\n2. monthly mean at 100 m and 1000 m, 2024 onward (C)")
m = df[[100, 1000]].groupby(df.index.to_period("M")).mean()
print(m.loc["2024-01":].round(2).to_string())

d = df[100].diff().dropna()
print("\n3. largest day-to-day changes of the domain mean at 100 m (C):")
for t in d.abs().sort_values(ascending=False).head(6).index:
    print("  ", t.date(), "%+.3f" % d[t])
print("   typical |change|: median %.3f, 99th percentile %.3f" % (d.abs().median(), d.abs().quantile(0.99)))

a24 = np.asarray(arr[years == 2024, k100]).mean(axis=0)
a25 = np.asarray(arr[years == 2025, k100]).mean(axis=0)
diff = a25 - a24
print("\n4. 2025 minus 2024 at 100 m, mean by region (C); columns lon 45-60, 60-75, 75-90, 90-105")
for la0, la1 in [(5, 12), (12, 20), (20, 30)]:
    row = []
    for lo0, lo1 in [(45, 60), (60, 75), (75, 90), (90, 105)]:
        sel = deep & (lat[:, None] >= la0) & (lat[:, None] < la1) & (lon[None, :] >= lo0) & (lon[None, :] < lo1)
        row.append("%+.2f" % np.nanmean(diff[sel]) if sel.any() else " n/a ")
    print(f"   lat {la0:>2}-{la1:<2}N | " + "   ".join(row))
