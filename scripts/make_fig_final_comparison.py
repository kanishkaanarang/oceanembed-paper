from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V = "results/variants/"
LIN1, LIN2 = "Linear, SLA anomaly", "Linear, SLA + SST anomalies"
RAW, ALL = "U-Net, raw inputs (SST+SLA+winds)", "U-Net, anomaly inputs (SST+SLA+winds)"
NET = "U-Net, anomaly inputs (SST+SLA), 3 seeds"
single = {LIN1: V + "sla_regression_validation.csv", LIN2: V + "sla_sst_regression_validation.csv",
          RAW: "results/prelim_model_validation.csv", ALL: V + "anom_all_validation.csv"}
seeds = [V + "anom_sst_sla_validation.csv", V + "anom_sst_sla_s1_validation.csv", V + "anom_sst_sla_s2_validation.csv"]
missing = [p for p in list(single.values()) + seeds if not Path(p).exists()]
if missing:
    raise SystemExit("missing: " + ", ".join(missing))


def pool(d, col="model_rmse"):
    return float(np.sqrt((d[col] ** 2).mean()))


dfs = {k: pd.read_csv(p) for k, p in single.items()}
sd = [pd.read_csv(p) for p in seeds]
net = sd[0].copy()
net["model_rmse"] = np.mean([d["model_rmse"].values for d in sd], axis=0)
dfs[NET] = net
seed_pooled = [pool(d) for d in sd]
pooled = {k: pool(d) for k, d in dfs.items()}
pooled[NET] = float(np.mean(seed_pooled))
clim = pool(dfs[LIN1], "clim_rmse")
colors = {"Seasonal climatology": "k", LIN1: "#8fd3c5", LIN2: "#2a9d8f", RAW: "#f4a261", ALL: "#7aa6d6", NET: "#1f6fa8"}

print(f"climatology pooled RMSE: {clim:.3f} C")
rows = []
for k in sorted(pooled, key=pooled.get):
    row = {"model": k, "pooled_rmse": round(pooled[k], 4), "skill_vs_climatology": round(1 - (pooled[k] / clim) ** 2, 4),
           "skill_vs_best_linear": round(1 - (pooled[k] / pooled[LIN2]) ** 2, 4)}
    rows.append(row)
    print(f"{k:46s} {pooled[k]:.3f} C | skill vs climatology {row['skill_vs_climatology']:+.3f} | skill vs linear SLA+SST {row['skill_vs_best_linear']:+.3f}")
print("U-Net seeds pooled:", [round(v, 3) for v in seed_pooled], "| spread (max-min): %.3f C" % (max(seed_pooled) - min(seed_pooled)))

order = sorted(pooled, key=pooled.get, reverse=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
labels = ["Seasonal climatology"] + order
vals = [clim] + [pooled[k] for k in order]
y = np.arange(len(labels))
ax1.barh(y, vals, color=[colors[k] for k in labels])
for yi, k, v in zip(y, labels, vals):
    txt = f"{v:.3f}" if k != NET else f"{v:.3f} (seeds {min(seed_pooled):.3f}-{max(seed_pooled):.3f})"
    ax1.text(v + 0.01, yi, txt, va="center", fontsize=9)
ax1.set_yticks(y)
ax1.set_yticklabels(labels, fontsize=9)
ax1.invert_yaxis()
ax1.set_xlim(0, max(vals) * 1.3)
ax1.set_xlabel("Pooled validation RMSE (°C), deep set")
ax1.set_title("All depths pooled", fontsize=10)

base = dfs[LIN1]
yd = np.arange(len(base))
ax2.plot(base["clim_rmse"], yd, "k--", marker="o", ms=4, label="Seasonal climatology")
for k in (LIN1, LIN2, RAW, NET):
    ax2.plot(dfs[k]["model_rmse"], yd, marker="o", ms=3, lw=1.5, color=colors[k], label=k)
ax2.set_yticks(yd)
ax2.set_yticklabels([f"{int(z)} m" for z in base["depth_m"]])
ax2.invert_yaxis()
ax2.set_xlabel("Validation RMSE (°C), deep set")
ax2.set_title("By depth", fontsize=10)
ax2.grid(alpha=0.3)
ax2.legend(fontsize=8, frameon=False, loc="lower right")
fig.suptitle("PRELIMINARY: model comparison on the validation window (Feb 2018 to Jan 2020; trained on 2010-2017)", fontsize=11)
fig.text(0.5, 0.01, "The U-Net's best epoch is chosen on this same window (slightly optimistic); test periods not used; "
         "target is GLORYS (a model reanalysis), not yet checked against Argo.", ha="center", fontsize=8, color="#a33")
fig.tight_layout(rect=[0, 0.04, 1, 0.94])
fig.savefig("results/figures/model_comparison_final.png", dpi=200, bbox_inches="tight")
pd.DataFrame(rows).to_csv("results/model_comparison_final.csv", index=False)
print("saved results/figures/model_comparison_final.png and results/model_comparison_final.csv")
