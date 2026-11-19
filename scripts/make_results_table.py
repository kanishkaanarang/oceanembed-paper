#!/usr/bin/env python
"""Build paper/tables/tab_main.tex from results/scoring files.
Every cell is checked against values copied from docs/results_ledger.md; on any mismatch nothing is written."""
import os, sys
import numpy as np, pandas as pd

R = "results/scoring/"
OUT = "paper/tables/tab_main.tex"
PER = ["test_a", "test_b", "test_c"]
ROWS = {"test_a": "argo_test_a_rows.csv.gz", "test_b": "argo_test_b_rows.csv.gz",
        "test_c": "argo_test_c_rerun_2026-11-16T173108_rows.csv.gz"}
LEARNED = ["lin", "xgb", "unet0", "frozen", "demean"]
LABEL = [("clim", "Climatology"), ("lin", "Linear (SLA+SST)"), ("xgb", "XGBoost"),
         ("unet0", "Single U-Net"), ("frozen", "U-Net ensemble"), ("demean", "Ensemble, demeaned SLA")]

# ledger copies (4 d.p. vs GLORYS and corrected; 3 d.p. raw Argo)
L_G = {"test_a": dict(clim=.9689, lin=.6835, xgb=.6823, unet0=.6895, frozen=.6712, demean=.7433),
       "test_b": dict(clim=1.0680, lin=.7461, xgb=.7409, unet0=.7452, frozen=.7285, demean=.8098),
       "test_c": dict(clim=.9064, lin=.6699, xgb=.6389, unet0=.6419, frozen=.6369, demean=.7052)}
L_A = {"test_a": dict(clim=1.009, lin=.861, xgb=.855, unet0=.848, frozen=.834, demean=.754, glorys=.677),
       "test_b": dict(clim=1.106, lin=.932, xgb=.920, unet0=.917, frozen=.909, demean=.848, glorys=.783),
       "test_c": dict(clim=.817, lin=.831, xgb=.777, unet0=.779, frozen=.773, demean=.649, glorys=.711)}
L_C = {"test_a": dict(frozen=.7569, demean=.7910, lin=.7822, xgb=.7794, unet0=.7757, glorys=.6415),
       "test_b": dict(frozen=.8433, demean=.8788, lin=.8688, xgb=.8626, unet0=.8636, glorys=.7355),
       "test_c": dict(frozen=.6609, demean=.6977, lin=.6974, xgb=.6681, unet0=.6682, glorys=.6683)}

if os.path.exists(OUT):
    sys.exit(f"EXISTS: {OUT} (stop)")

# vs GLORYS: deep set, mean of per-depth MSE over 15 depths
G = {}
for p in PER:
    s = pd.read_csv(f"{R}{p}_sums.csv"); d = s[s.set == "deep"]
    t = d.groupby(["model", "depth_m"])[["sse", "n"]].sum(); t["mse"] = t.sse / t.n
    G[p] = (t.groupby("model").mse.mean() ** 0.5).to_dict()

# vs Argo raw: row-level RMSE on the common set (all e_ columns finite)
E = ["clim", "glorys", "lin", "xgb", "unet0", "frozen", "demean"]
A, NROWS = {}, {}
for p in PER:
    r = pd.read_csv(R + ROWS[p]); c = r.dropna(subset=["e_" + m for m in E])
    A[p] = {m: float(np.sqrt((c["e_" + m] ** 2).mean())) for m in E}; NROWS[p] = len(c)

# vs Argo corrected: from the correction CSVs (absolute RMSEs as printed there)
m1 = pd.read_csv(R + "argo_correction_ci.csv"); m1 = m1[m1.group == "all depths"]
m2 = pd.read_csv(R + "argo_correction_all_ci.csv"); m2 = m2[m2.group == "all depths"]
C = {}
for p in PER:
    x = m1[m1.period == p]; y = m2[m2.period == p]
    C[p] = {"frozen": float(x[(x.a == "frozen_c") & (x.b == "frozen")].rmse_a.iloc[0]),
            "demean": float(x[(x.a == "demean_c") & (x.b == "demean")].rmse_a.iloc[0])}
    for k, nm in [("lin", "lin_c"), ("xgb", "xgb_c"), ("unet0", "unet0_c"), ("glorys", "glorys_c")]:
        C[p][k] = float(y[(y.a == "frozen_c") & (y.b == nm)].rmse_b.iloc[0])

bad = []
for p in PER:
    for m, v in L_G[p].items():
        if abs(G[p][m] - v) > 5.1e-5: bad.append(("GLORYS-ref", p, m, G[p][m], v))
    for m, v in L_A[p].items():
        if round(A[p][m], 3) != v: bad.append(("Argo raw", p, m, A[p][m], v))
    for m, v in L_C[p].items():
        if abs(C[p][m] - v) > 5.1e-5: bad.append(("Argo corrected", p, m, C[p][m], v))
if bad:
    for b in bad: print("MISMATCH", b)
    sys.exit("ledger mismatch: nothing written")
print("all cells match the ledger values; Argo common-set rows:", NROWS)

def col(vals):  # vals: dict model->float; bold the smallest among learned models
    best = min(LEARNED, key=lambda m: vals[m]); return best
lines = []
for key, name in LABEL:
    cells = [name]
    for p in PER:
        b = col(G[p]); cells.append((r"\textbf{%.3f}" if key == b else "%.3f") % G[p][key])
    for p in PER:
        b = col(A[p]); cells.append((r"\textbf{%.3f}" if key == b else "%.3f") % A[p][key])
    for p in PER:
        if key == "clim": cells.append("--")
        else:
            b = col(C[p]); cells.append((r"\textbf{%.3f}" if key == b else "%.3f") % C[p][key])
    lines.append(" & ".join(cells) + r" \\")
gl = ["GLORYS12", "--", "--", "--"] + ["%.3f" % A[p]["glorys"] for p in PER] + ["%.3f" % C[p]["glorys"] for p in PER]
tex = r"""\begin{table*}[t]
\centering
\caption{Pooled RMSE ($^\circ$C) on the deep set. Against GLORYS: mean of the per-depth MSE over 15 depths. Against Argo: all matched profile-depth values at 14 levels (5--1000~m), equally weighted ($%d$, $%d$ and $%d$ values in Tests~A, B and C; Test~C has 507 profiles). The two pooling rules are not comparable in magnitude. Corrected: one additive constant per depth, fitted on validation-window Argo only. Argo is assimilated by GLORYS and is not independent of it. Best learned model in bold per column.}
\label{tab:main}
\begin{tabular}{@{}lccccccccc@{}}
\hline
 & \multicolumn{3}{c}{vs GLORYS} & \multicolumn{3}{c}{vs Argo, raw} & \multicolumn{3}{c}{vs Argo, corrected} \\
\cline{2-4}\cline{5-7}\cline{8-10}
Model & A & B & C & A & B & C & A & B & C \\
\hline
""" % tuple(NROWS[p] for p in PER)
tex += "\n".join(lines[:-0 or None]) + "\n\\hline\n" + " & ".join(gl) + r" \\" + "\n\\hline\n\\end{tabular}\n\\end{table*}\n"
open(OUT, "w").write(tex)
print("wrote", OUT)
