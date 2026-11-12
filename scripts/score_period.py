"""Score all saved prediction files against GLORYS for one period and save raw error sums (per model, set, year, season, depth).
Usage: python scripts/score_period.py PERIOD   (validation | test_a | test_b)
Test periods are opened ONCE: a lock file is written before any target is read; a rerun needs RERUN=1 and is logged."""
import datetime
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

from oceanembed.dataset import END
from oceanembed.dataset_anom import AnomData
from oceanembed.glorys import TARGET_DEPTHS

PERIOD = sys.argv[1]
assert PERIOD in ("validation", "test_a", "test_b")
os.makedirs("results/scoring", exist_ok=True)
commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
now = datetime.datetime.now().isoformat(timespec="seconds")
rerun = False
sums_out = f"results/scoring/{PERIOD}_sums.csv"
if PERIOD.startswith("test"):
    if dirty:
        raise SystemExit("tracked files have uncommitted changes - commit first (stop)")
    lock = f"results/scoring/opened_{PERIOD}.json"
    if os.path.exists(lock):
        if os.environ.get("RERUN") != "1":
            raise SystemExit(f"{PERIOD} already opened ({lock}). Rerun only to fix a scoring bug: RERUN=1 (it is logged).")
        rerun = True
        sums_out = f"results/scoring/{PERIOD}_sums_rerun_{now.replace(':', '')}.csv"
    else:
        json.dump({"period": PERIOD, "opened": now, "commit": commit}, open(lock, "w"))
elif os.path.exists(sums_out):
    raise SystemExit(f"EXISTS: {sums_out} (stop)")
with open("results/scoring/score_log.csv", "a") as f:
    f.write(f"{now},{PERIOD},{commit},rerun={rerun},{sums_out}\n")

D = AnomData()
D.fit_input_clim()
s0, s1 = yaml.safe_load(open("configs/splits.yaml"))[PERIOD]
days = pd.date_range(pd.Timestamp(s0), END if s1 == "END" else pd.Timestamp(s1))
MODELS = ["clim", "lin", "xgb", "unet0", "frozen", "demean"]
pred = {m: np.load(f"data/processed/pred_{m}_{PERIOD}.npy", mmap_mode="r") for m in MODELS if m != "clim"}
for m, a in pred.items():
    assert a.shape[0] == len(days), (m, a.shape, len(days))
shelf = D.mask & ~D.deep[None]
SEAS = ["DJF", "MAM", "JJA", "SON"]
acc = defaultdict(lambda: np.zeros((3, 15)))
for i, d in enumerate(days):
    g = D.glorys(d).astype(np.float64)
    key = (d.year, SEAS[(d.month % 12) // 3])
    for m in MODELS:
        p = D.clim(pd.DatetimeIndex([d]))[0] if m == "clim" else np.asarray(pred[m][i])
        e = p.astype(np.float64) - g
        ed = e[:, D.deep]
        acc[("deep", m) + key] += np.stack([(ed ** 2).sum(1), ed.sum(1), np.full(15, ed.shape[1], float)])
        if m != "xgb":
            es = np.where(shelf, e, 0.0)
            acc[("shelf", m) + key] += np.stack([(es ** 2).sum((1, 2)), es.sum((1, 2)), shelf.sum((1, 2)).astype(float)])
    if i % 100 == 0:
        print("scored days:", i + 1, "of", len(days), flush=True)
rows = [(*k[:2], *k[2:], int(z), *v[:, j]) for k, v in acc.items() for j, z in enumerate(TARGET_DEPTHS.astype(int))]
df = pd.DataFrame(rows, columns=["set", "model", "year", "season", "depth_m", "sse", "bias_sum", "n"])
df.to_csv(sums_out, index=False)
print("saved", sums_out, "commit", commit)

for st in ("deep", "shelf"):
    x = df[df["set"] == st].groupby(["model", "depth_m"])[["sse", "n"]].sum()
    x["mse"] = x["sse"] / x["n"]
    mse = x["mse"].unstack("model")
    pooled = np.sqrt(mse.mean())
    skill = 1 - mse.mean() / mse["clim"].mean()
    print(f"\n=== {PERIOD} | {st} set | pooled over 15 depths ===")
    for m in MODELS:
        if m in pooled.index:
            print("  %-7s RMSE %.4f C   skill vs climatology %+.3f" % (m, pooled[m], skill[m]))
    print("RMSE by depth (C):")
    print(np.sqrt(mse)[[m for m in MODELS if m in mse.columns]].round(3).to_string())
