"""Skill of the frozen final model by depth, region and season on the VALIDATION window (dry run for the test evaluation).
skill = 1 - MSE_model / MSE_climatology on deep-set cells. Regions are coarse boxes, not exact basin masks."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oceanembed.dataset_anom import AnomData
from oceanembed.glorys import TARGET_DEPTHS

OUT, FIG = "results/skill_breakdown_validation.csv", "results/figures/skill_map_100m_validation.png"
for f in (OUT, FIG):
    if os.path.exists(f):
        raise SystemExit(f"EXISTS: {f} (stop)")

D = AnomData()
days = D.period("validation")
pred = np.load("data/processed/pred_final_validation.npy", mmap_mode="r")
lat, lon = 5 + 0.25 * np.arange(101)[:, None], 45 + 0.25 * np.arange(241)[None, :]
SN, RN = ["DJF", "MAM", "JJA", "SON"], ["Arabian Sea", "Bay of Bengal", "Andaman Sea", "South of 8N", "Other"]
reg = np.full((101, 241), 4)
reg[(lat < 8) & (lon < 92)] = 3
reg[(lon >= 50) & (lon <= 77) & (lat >= 8) & (lat <= 26)] = 0
reg[(lon >= 80) & (lon < 92) & (lat >= 8) & (lat <= 22)] = 1
reg[(lon >= 92) & (lon <= 100) & (lat >= 5) & (lat <= 18)] = 2
masks = [(reg == k) & D.deep for k in range(5)]
ncell = np.array([m.sum() for m in masks])
print("deep-set cells per region:", dict(zip(RN, ncell.tolist())))

sse, sse0, ndays = np.zeros((4, 5, 15)), np.zeros((4, 5, 15)), np.zeros(4)
c1, c0 = np.zeros((101, 241)), np.zeros((101, 241))
for i, d in enumerate(days):
    s = (d.month % 12) // 3
    e = np.asarray(pred[i]) - D.glorys(d)
    e0 = D.target(d)                      # climatology error = minus the GLORYS anomaly; the sign does not matter here
    ndays[s] += 1
    for k in range(5):
        sse[s, k] += (e[:, masks[k]] ** 2).sum(axis=1)
        sse0[s, k] += (e0[:, masks[k]] ** 2).sum(axis=1)
    c1 += np.where(D.deep, e[7] ** 2, 0.0)
    c0 += np.where(D.deep, e0[7] ** 2, 0.0)

sg = {**{SN[s]: [s] for s in range(4)}, "all": [0, 1, 2, 3]}
rg = {**{RN[k]: [k] for k in range(5)}, "all deep set": [0, 1, 2, 3, 4]}
rows = []
for sn, sl in sg.items():
    for rn, rl in rg.items():
        n = ndays[sl].sum() * ncell[rl].sum()
        if n == 0:
            continue
        a, b = sse[np.ix_(sl, rl)].sum(axis=(0, 1)), sse0[np.ix_(sl, rl)].sum(axis=(0, 1))
        for z in range(15):
            rows.append((sn, rn, int(TARGET_DEPTHS[z]), n, np.sqrt(a[z] / n), np.sqrt(b[z] / n), 1 - a[z] / b[z]))
df = pd.DataFrame(rows, columns=["season", "region", "depth_m", "n", "rmse_model", "rmse_clim", "skill"])
df.to_csv(OUT, index=False)

D5 = [0, 50, 100, 200, 500]
pd.set_option("display.width", 200)
print("\nskill by region and depth (all seasons)")
print(df[(df.season == "all") & df.depth_m.isin(D5)].pivot(index="region", columns="depth_m", values="skill").round(2).to_string())
print("\nskill by season and depth (whole deep set)")
print(df[(df.region == "all deep set") & df.depth_m.isin(D5)].pivot(index="season", columns="depth_m", values="skill").round(2).to_string())
print("\nskill at 100 m by region and season")
print(df[(df.depth_m == 100)].pivot(index="region", columns="season", values="skill")[SN + ["all"]].round(2).to_string())
print("\npooled over depths, all seasons: region | model RMSE | climatology RMSE | skill")
for k, rn in enumerate(RN):
    a, b = sse[:, k].sum(), sse0[:, k].sum()
    n = ndays.sum() * ncell[k] * 15
    print("  %-14s %.3f  %.3f  %.3f" % (rn, np.sqrt(a / n), np.sqrt(b / n), 1 - a / b))

skill = np.where(D.deep & (c0 > 0), 1 - c1 / np.where(c0 > 0, c0, 1), np.nan)
fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
im = ax.imshow(skill, origin="lower", extent=[45, 105, 5, 30], cmap="RdBu", vmin=-0.5, vmax=1.0, aspect="auto")
fig.colorbar(im, ax=ax, label="skill vs climatology (1 = perfect, 0 = no better)")
ax.set_xlabel("longitude (E)")
ax.set_ylabel("latitude (N)")
ax.set_title("Final model, skill at 100 m, validation window (dry run, deep-set cells)")
fig.tight_layout()
fig.savefig(FIG)
print("\nsaved", OUT, "and", FIG)
