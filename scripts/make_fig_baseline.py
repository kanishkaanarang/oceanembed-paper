from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results/climatology_baseline_validation.csv")
y = np.arange(len(df))
h = 0.38
overall = float(np.sqrt((df["val_rmse"] ** 2).mean()))
overall_ns = float(np.sqrt((df["val_rmse_no_season"] ** 2).mean()))
print(f"pooled validation RMSE: {overall:.3f} C (no seasonal cycle: {overall_ns:.3f} C)")

fig, ax = plt.subplots(figsize=(7.5, 5.8))
ax.barh(y - h / 2, df["val_rmse_no_season"], height=h, color="#b8b8b8", label="No seasonal cycle")
ax.barh(y + h / 2, df["val_rmse"], height=h, color="#1f6fa8", label="Seasonal climatology")
ax.set_yticks(y)
ax.set_yticklabels([f"{int(d)} m" for d in df["depth_m"]])
ax.invert_yaxis()
ax.set_xlabel("Validation RMSE (°C)")
ax.set_title("Baseline error by depth, validation period (Feb 2018 to Jan 2020)", fontsize=11)
ax.legend(loc="lower right", bbox_to_anchor=(1, 0.12), fontsize=9, frameon=False)
ax.text(0.98, 0.04, f"Pooled over 15 depths: {overall:.2f} °C", transform=ax.transAxes, ha="right", fontsize=9)
out = Path("results/figures")
out.mkdir(parents=True, exist_ok=True)
path = out / "baseline_rmse_by_depth.png"
if path.exists():
    raise SystemExit(f"{path} already exists - not overwriting")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("saved", path)
