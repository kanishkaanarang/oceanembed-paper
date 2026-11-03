from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REG = "Linear regression on SLA anomaly"
runs = {
    REG: "results/variants/sla_regression_validation.csv",
    "U-Net, raw inputs (SST+SLA+winds)": "results/prelim_model_validation.csv",
    "U-Net, anomaly inputs (SST+SLA+winds)": "results/variants/anom_all_validation.csv",
    "U-Net, anomaly inputs (SST+SLA)": "results/variants/anom_sst_sla_validation.csv",
}
colors = {"Seasonal climatology": "k", REG: "#2a9d8f", "U-Net, raw inputs (SST+SLA+winds)": "#f4a261",
          "U-Net, anomaly inputs (SST+SLA+winds)": "#1f6fa8", "U-Net, anomaly inputs (SST+SLA)": "#7b4fa3"}
missing = [p for p in runs.values() if not Path(p).exists()]
if missing:
    raise SystemExit("not finished yet, missing: " + ", ".join(missing))
dfs = {k: pd.read_csv(p) for k, p in runs.items()}
base = dfs[REG]
clim = float(np.sqrt((base["clim_rmse"] ** 2).mean()))
pooled = {k: float(np.sqrt((d["model_rmse"] ** 2).mean())) for k, d in dfs.items()}
rows = []
print(f"climatology pooled RMSE: {clim:.3f} C")
for k in sorted(pooled, key=pooled.get):
    p = pooled[k]
    row = {"model": k, "pooled_rmse": round(p, 4), "skill_vs_climatology": round(1 - (p / clim) ** 2, 4),
           "skill_vs_regression": round(1 - (p / pooled[REG]) ** 2, 4)}
    rows.append(row)
    print(f"{k:40s} {p:.3f} C | skill vs climatology {row['skill_vs_climatology']:+.3f} | skill vs regression {row['skill_vs_regression']:+.3f}")
best = min((k for k in dfs if k != REG), key=pooled.get)
print(f"\nbest network: {best}\n depth_m  regression  network  difference (network - regression)")
for i, z in enumerate(base["depth_m"]):
    a, b = base["model_rmse"][i], dfs[best]["model_rmse"][i]
    print(f"  {int(z):6d}   {a:8.3f}  {b:8.3f}   {b - a:+8.3f}")

order = sorted(pooled, key=pooled.get, reverse=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})
labels = ["Seasonal climatology"] + order
vals = [clim] + [pooled[k] for k in order]
y = np.arange(len(labels))
ax1.barh(y, vals, color=[colors[k] for k in labels])
for yi, v in zip(y, vals):
    ax1.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9)
ax1.set_yticks(y)
ax1.set_yticklabels(labels, fontsize=9)
ax1.invert_yaxis()
ax1.set_xlim(0, max(vals) * 1.12)
ax1.set_xlabel("Pooled validation RMSE (°C), deep set")
ax1.set_title("All depths pooled", fontsize=10)
yd = np.arange(len(base))
ax2.plot(base["clim_rmse"], yd, "k--", marker="o", ms=4, label="Seasonal climatology")
for k, d in dfs.items():
    ax2.plot(d["model_rmse"], yd, marker="o", ms=3, lw=1.5, color=colors[k], label=k)
ax2.set_yticks(yd)
ax2.set_yticklabels([f"{int(z)} m" for z in base["depth_m"]])
ax2.invert_yaxis()
ax2.set_xlabel("Validation RMSE (°C), deep set")
ax2.set_title("By depth", fontsize=10)
ax2.grid(alpha=0.3)
ax2.legend(fontsize=8, frameon=False, loc="lower right")
fig.suptitle("PRELIMINARY: model comparison on the validation window (Feb 2018 to Jan 2020; trained on 2010-2017)", fontsize=11)
fig.text(0.5, 0.01, "Single run per model; test periods not used; targets are GLORYS (a model reanalysis), not yet checked against Argo.",
         ha="center", fontsize=8, color="#a33")
fig.tight_layout(rect=[0, 0.04, 1, 0.94])
fig.savefig("results/figures/model_comparison.png", dpi=200, bbox_inches="tight")
pd.DataFrame(rows).to_csv("results/model_comparison.csv", index=False)
print("saved results/figures/model_comparison.png and results/model_comparison.csv")
