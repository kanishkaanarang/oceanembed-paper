"""Is there a step at the turn of the year in the 100 m domain mean? Offline, uses assembled block 5."""
import numpy as np
import pandas as pd

from oceanembed.glorys import TARGET_DEPTHS
from oceanembed.glorys_pipeline import block_dates

arr = np.load("data/processed/glorys_block5.npy", mmap_mode="r")
mask = np.load("data/processed/target_mask_15depths.npy")
deep = mask.all(axis=0)
start, end = block_dates(5)
dates = pd.date_range(start, end)
k = int(np.where(TARGET_DEPTHS == 100)[0][0])
s = pd.Series(np.asarray(arr[:, k])[:, deep].mean(axis=1), index=dates)

print("mean at 100 m over the deep set: 15-31 Dec vs 1-15 Jan (C)")
for y in (2021, 2022, 2023, 2024):
    a = s[f"{y}-12-15":f"{y}-12-31"].mean()
    b = s[f"{y + 1}-01-01":f"{y + 1}-01-15"].mean()
    print(f"  {y}->{y + 1}: {a:.2f} -> {b:.2f}   change {b - a:+.2f}")

print("\ndaily values, 2024-12-20 to 2025-01-12:")
print(s["2024-12-20":"2025-01-12"].round(2).to_string())
