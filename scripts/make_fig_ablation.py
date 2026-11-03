from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = {
    "All inputs (SST + SLA + winds)": "results/prelim_model_validation.csv",
    "No SST (SLA + winds)": "results/variants/no_sst_validation.csv",
    "No winds (SST + SLA)": "results/variants/no_wind_validation.csv",
    "No SLA (SST + winds)": "results/variants/no_sla_validation.csv",
    "SLA only": "results/variants/sla_only_validation.csv",
    "SST only": "results/variants/sst_only_validation.csv",
}
missing = [p for p in runs.values() if not Path(p).exists()]
if missing:
    raise SystemExit("not finished yet, missing: " + ", ".join(missing))
dfs = {k: pd.read_csv(p) for k, p in runs.items()}
base = next(iter(dfs.values()))
pooled = {k: float(np.sqrt((d["model_rmse"] ** 2).mean())) for k, d in dfs.items()}
clim = float(np.sqrt((base["clim_rmse"] ** 2).mean()))
print(f"climatology: {clim:.3f} C")
rows = []
for k, v in sorted(pooled.items(), key=lambda kv: kv[1]):
    print(f"{k:32s} pooled {v:.3f} C | skill vs climatology {1 - (v / clim) ** 2:.3f}")
    rows.append({"variant": k, "pooled_rmse": round(v, 4), "skill_vs_climatology": round(1 - (v / clim) ** 2, 4)})

order = sorted(pooled, key=pooled.get, reverse=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1, 1.2]})
labels = ["Seasonal climatology"] + order
vals = [clim] + [pooled[k] for k in order]
y = np.arange(len(labels))
ax1.barh(y, vals, color=["#b8b8b8"] + ["#1f6fa8"] * len(order))
for yi, v in zip(y, vals):
    ax1.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9)
ax1.set_yticks(y)
ax1.set_yticklabels(labels)
ax1.invert_yaxis()
ax1.set_xlim(0, max(vals) * 1.12)
ax1.set_xlabel("Pooled validation RMSE (°C), deep set")
ax1.set_title("All depths pooled", fontsize=10)

yd = np.arange(len(base))
ax2.plot(base["clim_rmse"], yd, "k--", marker="o", ms=4, label="Seasonal climatology")
for k, d in dfs.items():
    ax2.plot(d["model_rmse"], yd, marker="o", ms=3, lw=1.4, label=k)
ax2.set_yticks(yd)
ax2.set_yticklabels([f"{int(z)} m" for z in base["depth_m"]])
ax2.invert_yaxis()
ax2.set_xlabel("Validation RMSE (°C), deep set")
ax2.set_title("By depth", fontsize=10)
ax2.legend(fontsize=8, frameon=False, loc="lower right")
ax2.grid(alpha=0.3)
fig.suptitle("PRELIMINARY: which inputs matter (validation Feb 2018 to Jan 2020; trained on 2010-2017)", fontsize=11)
fig.text(0.5, 0.01, "One run per variant (single seed): differences of a few hundredths of a degree may be within run-to-run noise. "
         "Test periods not used.", ha="center", fontsize=8, color="#a33")
fig.tight_layout(rect=[0, 0.04, 1, 0.94])
out = Path("results/figures/ablation_inputs.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
pd.DataFrame(rows).to_csv("results/ablation_summary.csv", index=False)
print("saved", out, "and results/ablation_summary.csv")
