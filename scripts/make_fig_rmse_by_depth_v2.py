"""Figure v2: RMSE vs GLORYS12 by depth (deep set) for test_a (a) and test_c (b), and the per-depth RMSE change of the frozen U-Net
ensemble relative to the linear baseline with 95% block-bootstrap intervals for test_a, test_b and test_c (c).
Reads only existing results/scoring CSVs: no targets are read and no bootstrap is rerun. Fonts are embedded as TrueType
(pdf.fonttype 42). The plotted RMSEs are checked against the interval files before plotting."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results/figures/fig_rmse_by_depth_v2"
if os.path.exists(OUT + ".pdf") or os.path.exists(OUT + ".png"):
    sys.exit("EXISTS: " + OUT + " (stop)")
S = "results/scoring/"
DEP = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


def rmse_by_depth(path):
    s = pd.read_csv(path)
    g = s[s.set == "deep"].groupby(["model", "depth_m"])[["sse", "n"]].sum()
    return np.sqrt(g.sse / g.n).unstack("model").reindex(DEP)


A = rmse_by_depth(S + "test_a_sums.csv")
C = rmse_by_depth(S + "test_c_sums.csv")
pa = pd.read_csv(S + "depth_profile_ci.csv")
ta = pa[pa.period == "test_a"].set_index("depth_m").reindex(DEP)
tb = pa[pa.period == "test_b"].set_index("depth_m").reindex(DEP)
pc = pd.read_csv(S + "depth_profile_ci_test_c.csv")
pcl = pc[(pc.a == "frozen") & (pc.b == "lin")].set_index("depth_m").reindex(DEP)

for m, col in (("clim", "rmse_clim"), ("lin", "rmse_lin"), ("frozen", "rmse_frozen"), ("demean", "rmse_demean")):
    assert np.allclose(A[m].values, ta[col].values, atol=1e-6), ("test_a mismatch", m)
assert np.allclose(C["frozen"].values, pcl.rmse_a.values, atol=1e-6), "test_c frozen mismatch"
assert np.allclose(C["lin"].values, pcl.rmse_b.values, atol=1e-6), "test_c lin mismatch"
print("checks passed: plotted RMSEs match the interval files (test_a all four models, test_c frozen and linear)")

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.serif": ["STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.6})
SER = [("clim", "Climatology", "#6b6b6b", ":", None), ("lin", "Linear SLA+SST", "#009E73", "--", "^"),
       ("frozen", "U-Net ensemble", "#0072B2", "-", "o"), ("demean", "U-Net ensemble, demeaned SLA", "#D55E00", "-.", "s")]
fig, axs = plt.subplots(1, 3, figsize=(7.16, 3.3), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1.15]})
xmax = 1.05 * max(A.max().max(), C.max().max())
for ax, R, title in ((axs[0], A, "(a) Test A, Mar 2020\u2013Jun 2021"), (axs[1], C, "(b) Test C, Jan\u2013May 2026")):
    for m, lab, col, ls, mk in SER:
        ax.plot(R[m].values, DEP, color=col, ls=ls, marker=mk, ms=3.6, lw=1.3, label=lab)
    ax.set_title(title, loc="left")
    ax.set_xlabel("RMSE vs GLORYS12 (\u00b0C)")
    ax.set_xlim(0.25, xmax)

ax = axs[2]
CH = [("Test A", ta, "frozen_vs_lin_pct", "lo", "hi", "#0072B2", "o", "-", -0.45),
      ("Test B", tb, "frozen_vs_lin_pct", "lo", "hi", "#E69F00", "s", "--", 0.0),
      ("Test C", pcl, "rel_pct", "rel_lo", "rel_hi", "#D55E00", "D", ":", 0.45)]
for lab, d, c, lo, hi, col, mk, ls, off in CH:
    y = np.clip(np.sqrt(np.array(DEP, dtype=float)) + off, 0, None) ** 2
    v = d[c].values
    ax.errorbar(v, y, xerr=[v - d[lo].values, d[hi].values - v], color=col, marker=mk, ms=3.2, lw=1.0, ls=ls,
                capsize=1.5, elinewidth=0.8, label=lab)
ax.axvline(0, color="#6b6b6b", lw=0.8)
ax.set_title("(c) Ensemble vs linear", loc="left")
ax.set_xlabel("RMSE change (%)")
ax.legend(loc="lower right", frameon=False)

for a in axs:
    a.set_yscale("function", functions=(lambda x: np.sqrt(np.clip(x, 0, None)), lambda x: np.asarray(x) ** 2))
    a.set_ylim(1000, 0)
    a.set_yticks([0, 50, 100, 200, 300, 500, 1000])
    a.set_yticklabels(["0", "50", "100", "200", "300", "500", "1000"])
    a.grid(axis="x", color="#dddddd", lw=0.5)
    a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
axs[0].set_ylabel("Depth (m), square-root scale")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, handlelength=2.6)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT + ".png", dpi=300)
fig.savefig(OUT + ".pdf")
print("saved", OUT + ".png / .pdf")
