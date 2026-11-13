"""Figure: thermocline (75-200 m) bias and RMSE against Argo by calendar year, common set (validation + test windows; training years excluded).
Reads only the saved Argo row files; no targets are read."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results/figures/fig_argo_thermocline_by_year"
if os.path.exists(OUT + ".png") or os.path.exists(OUT + ".pdf"):
    raise SystemExit(f"EXISTS: {OUT}.png/.pdf (stop)")
os.makedirs("results/figures", exist_ok=True)
M = ["clim", "glorys", "lin", "xgb", "unet0", "frozen", "demean"]
r = pd.concat([pd.read_csv(f"results/scoring/argo_{p}_rows.csv.gz", parse_dates=["day"]) for p in ("validation", "test_a", "test_b")])
r = r[np.isfinite(r[M]).all(axis=1) & (r.depth >= 75) & (r.depth <= 200)].copy()
r["year"] = r.day.dt.year
years = sorted(r.year.unique())
n = r.groupby("year").size()
SER = [("glorys", "GLORYS12 (reference)", "#52514e", "--", "D"), ("lin", "Linear SLA+SST", "#1baf7a", "-", "^"),
       ("frozen", "U-Net ensemble", "#2a78d6", "-", "o"), ("demean", "U-Net ensemble, demeaned SLA", "#eb6834", "-", "s")]
bias = {m: [r[r.year == y]["e_" + m].mean() for y in years] for m, *_ in SER}
rmse = {m: [np.sqrt((r[r.year == y]["e_" + m] ** 2).mean()) for y in years] for m, *_ in SER}
print("year  n      " + "  ".join(f"bias_{m:7s}" for m, *_ in SER) + "  " + "  ".join(f"rmse_{m:7s}" for m, *_ in SER))
for i, y in enumerate(years):
    print(y, "%6d " % n[y] + "  ".join("%12.3f" % bias[m][i] for m, *_ in SER) + "  " + "  ".join("%12.3f" % rmse[m][i] for m, *_ in SER))

plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.9), sharex=True)
for ax, data, title, yl in ((axs[0], bias, "(a) Mean bias against Argo (model $-$ Argo)", "Bias (°C)"),
                            (axs[1], rmse, "(b) RMSE against Argo", "RMSE (°C)")):
    for m, lab, col, ls, mk in SER:
        ax.plot(years, data[m], color=col, ls=ls, marker=mk, ms=4.5, lw=1.6, mec="#fcfcfb", mew=0.8, label=lab)
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.set_ylabel(yl)
    ax.grid(axis="y", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xticks(years)
    ax.set_xticklabels([f"{y}\n{n[y] / 1000:.1f}k" for y in years])
axs[0].axhline(0, color="#898781", lw=0.8)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5, handlelength=2.4)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT + ".png", dpi=300)
fig.savefig(OUT + ".pdf")
print("saved", OUT + ".png / .pdf")
