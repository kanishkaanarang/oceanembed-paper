from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

mask = np.load("data/processed/target_mask_15depths.npy").astype(bool)  # (15, 101, 241)
surface = mask[0]
deep = mask.all(axis=0)
shelf = surface & ~deep
print("surface ocean:", int(surface.sum()), "| deep set:", int(deep.sum()), "| shelf:", int(shelf.sum()))
print("expected      : 12038 | 9140 | 2898")

cat = np.zeros(surface.shape, dtype=int)
cat[shelf] = 1
cat[deep] = 2
lat = np.linspace(5, 30, 101)
lon = np.linspace(45, 105, 241)
colors = ["#d9d9d9", "#f2a65a", "#1f6fa8"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.pcolormesh(lon, lat, cat, cmap=ListedColormap(colors), vmin=-0.5, vmax=2.5, shading="nearest")
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title("Study domain on the 0.25° grid (101 × 241 cells)")
ax.set_aspect("equal")
ax.legend(
    handles=[
        Patch(color=colors[2], label=f"Deep set: {int(deep.sum()):,} cells (all 15 depths)"),
        Patch(color=colors[1], label=f"Shelf: {int(shelf.sum()):,} cells"),
        Patch(color=colors[0], label="Land"),
    ],
    loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9, frameon=False,
)
out = Path("results/figures")
out.mkdir(parents=True, exist_ok=True)
path = out / "domain_deep_set.png"
if path.exists():
    raise SystemExit(f"{path} already exists - not overwriting")
fig.savefig(path, dpi=200, bbox_inches="tight")
print("saved", path)
