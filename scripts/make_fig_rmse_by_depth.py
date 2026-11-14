"""Figure: RMSE vs GLORYS by depth (deep set) for test_a and test_b, plus per-depth RMSE change of the frozen U-Net ensemble relative to the
linear baseline with 95% block-bootstrap intervals (30-day blocks, 2000 resamples). Reads only results/scoring/*_daily*.csv."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results/figures/fig_rmse_by_depth"
CSV = "results/scoring/depth_profile_ci.csv"
if os.path.exists(OUT + ".png") or os.path.exists(OUT + ".pdf") or os.path.exists(CSV):
    raise SystemExit("EXISTS: figure or csv (stop)")
NDEEP, NB, L = 9140, 2000, 30
rng = np.random.default_rng(0)
DEP = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
cols = [f"sse_{z}" for z in DEP]
files = {"test_a": sorted(glob.glob("results/scoring/test_a_daily_rerun_*.csv"))[-1], "test_b": "results/scoring/test_b_daily.csv"}
res, rows = {}, []
for per, f in files.items():
    df = pd.read_csv(f)
    dd = {m: df[df.model == m].sort_values("date")[cols].values for m in ("clim", "lin", "frozen", "demean")}
    nd = len(dd["clim"])
    nblk = -(-nd // L)
    idx = (rng.integers(0, nd - L + 1, size=(NB, nblk))[:, :, None] + np.arange(L)).reshape(NB, -1)[:, :nd]
    C = np.stack([np.bincount(idx[b], minlength=nd) for b in range(NB)]).astype(float)
    rm = {m: np.sqrt(dd[m].sum(0) / (nd * NDEEP)) for m in dd}
    bs = {m: np.sqrt((C @ dd[m]) / (nd * NDEEP)) for m in ("lin", "frozen")}
    rel_b = 100 * (bs["frozen"] / bs["lin"] - 1)
    rel = 100 * (rm["frozen"] / rm["lin"] - 1)
    lo, hi = np.percentile(rel_b, 2.5, axis=0), np.percentile(rel_b, 97.5, axis=0)
    res[per] = dict(rm=rm, rel=rel, lo=lo, hi=hi)
    for k, z in enumerate(DEP):
        rows.append(dict(period=per, depth_m=z, rmse_clim=rm["clim"][k], rmse_lin=rm["lin"][k], rmse_frozen=rm["frozen"][k],
                         rmse_demean=rm["demean"][k], frozen_vs_lin_pct=rel[k], lo=lo[k], hi=hi[k]))
out = pd.DataFrame(rows)
out.to_csv(CSV, index=False)
print(out.round(3).to_string(index=False))

plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
SER = [("clim", "Climatology", "#898781", ":", "x"), ("lin", "Linear SLA+SST", "#1baf7a", "-", "^"),
       ("frozen", "U-Net ensemble", "#2a78d6", "-", "o"), ("demean", "U-Net ensemble, demeaned SLA", "#eb6834", "-", "s")]
fig, axs = plt.subplots(1, 3, figsize=(7.16, 3.2), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1.1]})
for ax, per, title in ((axs[0], "test_a", "(a) Test A, Mar 2020 - Jun 2021"), (axs[1], "test_b", "(b) Test B, Aug 2021 - Dec 2025")):
    for m, lab, col, ls, mk in SER:
        ax.plot(res[per]["rm"][m], DEP, color=col, ls=ls, marker=mk, ms=3.8, lw=1.4, mec="#fcfcfb", mew=0.6, label=lab)
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xlabel("RMSE vs GLORYS12 (°C)")
ax = axs[2]
for per, mk, ls, lab in (("test_a", "o", "-", "Test A"), ("test_b", "s", "--", "Test B")):
    r = res[per]
    ax.errorbar(r["rel"], DEP, xerr=[r["rel"] - r["lo"], r["hi"] - r["rel"]], color="#2a78d6", marker=mk, ms=3.8, lw=1.2, ls=ls,
                capsize=1.8, elinewidth=0.9, mec="#fcfcfb", mew=0.6, label=lab)
ax.axvline(0, color="#898781", lw=0.8)
ax.set_title("(c) U-Net ensemble vs linear:\nRMSE change (%), 95% CI", fontsize=8, loc="left")
ax.set_xlabel("RMSE change (%)")
ax.legend(loc="lower right", frameon=False, fontsize=7)
for a in axs:
    a.set_yscale("function", functions=(lambda x: np.sqrt(np.clip(x, 0, None)), lambda x: np.asarray(x) ** 2))
    a.set_ylim(1000, 0)
    a.set_yticks([0, 50, 100, 200, 300, 500, 1000])
    a.set_yticklabels(["0", "50", "100", "200", "300", "500", "1000"])
    a.grid(axis="x", color="#e1e0d9", lw=0.6)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
axs[0].set_ylabel("Depth (m), square-root scale")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5, handlelength=2.4)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT + ".png", dpi=300)
fig.savefig(OUT + ".pdf")
print("saved", OUT + ".png / .pdf")
