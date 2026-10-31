import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", RuntimeWarning)
K = 7  # 100 m
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
deep = np.load("data/processed/target_mask_15depths.npy").astype(bool).all(axis=0)

daily = pd.Series(np.nanmean(t100[:, deep], axis=1), index=dates)
monthly = daily.resample("MS").mean()
roll = monthly.rolling(12, center=True).mean()

sw = pd.Timestamp("2021-06-28")
before = daily[sw - pd.Timedelta(days=7): sw - pd.Timedelta(days=1)].mean()
after = daily[sw: sw + pd.Timedelta(days=6)].mean()
chg = daily.resample("7D").mean().diff().abs().dropna()
print(f"week before 2021-06-28: {before:.2f} C | week after: {after:.2f} C | step {after - before:+.2f} C")
print(f"typical week-to-week change: median {chg.median():.2f} C | 95th percentile {chg.quantile(0.95):.2f} C")
for y in range(2019, 2026):
    h1 = daily[f"{y}-01-01":f"{y}-06-30"].mean()
    h2 = daily[f"{y}-07-01":f"{y}-12-31"].mean()
    print(y, f"Jan-Jun {h1:.2f} | Jul-Dec {h2:.2f}")

fig, ax = plt.subplots(figsize=(10, 4.4))
ax.plot(monthly.index, monthly.values, color="#b8b8b8", lw=1, label="Monthly mean")
ax.plot(roll.index, roll.values, color="#1f6fa8", lw=2.2, label="12-month running mean")
ax.axvline(pd.Timestamp("2021-07-01"), color="#c0392b", ls="--", lw=1)
ymin, ymax = ax.get_ylim()
ax.text(pd.Timestamp("2021-08-15"), ymin + 0.02 * (ymax - ymin), "GLORYS interim production starts (about Jul 2021)",
        color="#c0392b", fontsize=8, rotation=90, va="bottom")
ax.set_ylabel("Deep-set mean at 100 m (°C)")
ax.set_title("GLORYS temperature at 100 m, deep-set mean, 2010-2025", fontsize=11)
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.grid(alpha=0.3)
fig.text(0.5, 0.0, "GLORYS output; not checked against Argo.", ha="center", fontsize=8, color="#a33")
fig.tight_layout(rect=[0, 0.03, 1, 1])
out = Path("results/figures")
out.mkdir(parents=True, exist_ok=True)
path = out / "monthly_100m_series.png"
if path.exists():
    raise SystemExit(f"{path} already exists - not overwriting")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("saved", path)
