import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", RuntimeWarning)
K = 7  # index of the 100 m target depth
blocks = [("glorys_block3.npy", "2010-02-04", "2015-10-16"),
          ("glorys_block4.npy", "2015-10-17", "2021-06-27"),
          ("glorys_block5.npy", "2021-06-28", "2025-12-31")]
parts, dates = [], []
for fn, start, end in blocks:
    a = np.load(f"data/processed/{fn}", mmap_mode="r")
    d = pd.date_range(start, periods=a.shape[0], freq="D")
    if d[-1] != pd.Timestamp(end):
        raise SystemExit(f"{fn}: last date {d[-1].date()} but expected {end}")
    parts.append(np.array(a[:, K]))
    dates.append(d)
t100 = np.concatenate(parts)
dates = dates[0].append(dates[1]).append(dates[2])
print("t100 shape:", t100.shape, "| dates", dates[0].date(), "to", dates[-1].date())

mask = np.load("data/processed/target_mask_15depths.npy").astype(bool)
deep, valid100 = mask.all(axis=0), mask[K]
lat, lon = np.linspace(5, 30, 101), np.linspace(45, 105, 241)

years = np.arange(2011, 2026)
ann = {int(y): np.nanmean(t100[dates.year == y], axis=0) for y in years}
dm = {y: float(np.nanmean(ann[y][deep])) for y in ann}
for y in dm:
    print(y, f"{dm[y]:.2f}")
ref_years = [y for y in dm if y <= 2024]
print(f"2011-2024 range: {min(dm[y] for y in ref_years):.2f} to {max(dm[y] for y in ref_years):.2f} | 2025: {dm[2025]:.2f}")
anom = ann[2025] - np.mean([ann[y] for y in ref_years], axis=0)
anom[~valid100] = np.nan
east = np.broadcast_to(lon > 75, deep.shape) & deep
print(f"mean 2025 anomaly: deep set {np.nanmean(anom[deep]):+.2f} C | deep set east of 75E {np.nanmean(anom[east]):+.2f} C")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"width_ratios": [1.35, 1]})
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("#d9d9d9")
im = ax1.pcolormesh(lon, lat, np.ma.masked_invalid(anom), cmap=cmap, vmin=-2.5, vmax=2.5, shading="nearest")
ax1.set_aspect("equal")
ax1.set_xlabel("Longitude (°E)")
ax1.set_ylabel("Latitude (°N)")
ax1.set_title("100 m temperature: 2025 minus mean of 2011-2024", fontsize=10)
fig.colorbar(im, ax=ax1, label="Anomaly (°C)", shrink=0.8)
ax2.plot(list(dm), list(dm.values()), marker="o", color="#1f6fa8")
ax2.plot([2025], [dm[2025]], marker="o", color="#c0392b", markersize=9)
ax2.set_xticks(range(2011, 2026, 2))
ax2.set_xlabel("Year")
ax2.set_ylabel("Deep-set mean at 100 m (°C)")
ax2.set_title("Yearly mean over the deep set", fontsize=10)
ax2.grid(alpha=0.3)
fig.suptitle("GLORYS 100 m temperature, 2025 versus earlier years", fontsize=12)
fig.text(0.5, 0.01, "A feature of the GLORYS reanalysis (2025 lies in its interim production stream). "
         "Not yet checked against Argo; not a finding about the real ocean.",
         ha="center", fontsize=9, color="#a33")
fig.tight_layout(rect=[0, 0.05, 1, 0.94])
out = Path("results/figures")
out.mkdir(parents=True, exist_ok=True)
path = out / "anomaly_2025_100m.png"
if path.exists():
    raise SystemExit(f"{path} already exists - not overwriting")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("saved", path)
