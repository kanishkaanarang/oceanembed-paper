"""Validation-window ensemble of the six demeaned-sea-level U-Nets (3 seeds same-day inputs, 3 seeds lags 0/7/14/30 days), scored per depth on
the deep set and compared with the frozen ensemble (results/variants/ensemble_validation.csv). No training; no test periods."""
import os

import numpy as np
import pandas as pd
import torch

from oceanembed.dataset import TARGET_SCALE
from oceanembed.dataset_demean import AnomDemeanData
from oceanembed.final_model import UNet, device
from oceanembed.glorys import TARGET_DEPTHS

OUT = "results/variants/ensemble_demean_validation.csv"
if os.path.exists(OUT):
    raise SystemExit(f"EXISTS: {OUT} (stop)")
dev = device()
D1, D2 = AnomDemeanData(), AnomDemeanData()
D1.lags, D2.lags = [0], [0, 7, 14, 30]
for D in (D1, D2):
    D.fit_input_clim()
    D.fit_demean_std()
va = D1.period("validation")
deep = torch.from_numpy(D1.deep.astype(np.float32))[None, None].to(dev)
SETS = {"unlagged": [f"dm_unlag_s{s}" for s in range(3)], "lagged": [f"dm_lags_s{s}" for s in range(3)]}
models = {}
for kind, names in SETS.items():
    for n in names:
        m = UNet(3 if kind == "unlagged" else 9).to(dev)
        m.load_state_dict(torch.load(f"data/processed/variant_{n}.pt", map_location=dev))
        models[n] = (kind, m.eval())
keys = list(models) + ["ens_unlagged", "ens_lagged", "ens_all6"]
sse, ntot = {k: np.zeros(15) for k in keys}, 0.0
with torch.no_grad():
    for s in range(0, len(va), 8):
        days = va[s:s + 8]
        Xu = torch.from_numpy(np.stack([D1.inputs(d) for d in days])).to(dev)
        Xl = torch.from_numpy(np.stack([D2.inputs(d) for d in days])).to(dev)
        Y = torch.from_numpy(np.stack([D1.target(d) for d in days]) / TARGET_SCALE).to(dev)
        P = {n: m(Xu if kind == "unlagged" else Xl) for n, (kind, m) in models.items()}
        P["ens_unlagged"] = sum(P[n] for n in SETS["unlagged"]) / 3
        P["ens_lagged"] = sum(P[n] for n in SETS["lagged"]) / 3
        P["ens_all6"] = (P["ens_unlagged"] + P["ens_lagged"]) / 2
        for k in keys:
            sse[k] += (((P[k] - Y) * deep) ** 2).sum(dim=(0, 2, 3)).cpu().numpy()
        ntot += float(deep.sum() * len(days))
rm = {k: TARGET_SCALE * np.sqrt(v / ntot) for k, v in sse.items()}
pool = lambda v: float(np.sqrt((np.asarray(v) ** 2).mean()))
fro = pd.read_csv("results/variants/ensemble_validation.csv")
df = pd.DataFrame({"depth_m": TARGET_DEPTHS.astype(int), **rm, "frozen_ens_all6": fro["ens_all6"]})
df["cost_vs_frozen"] = df["ens_all6"] - df["frozen_ens_all6"]
df.to_csv(OUT, index=False)
print("pooled validation RMSE vs GLORYS (deep set)")
for k in keys:
    print("  %-14s %.4f" % (k, pool(rm[k])))
print("  %-14s %.4f   (frozen ensemble)" % ("frozen_all6", pool(fro["ens_all6"])))
print("criterion (1) as a record: demeaned all6 <= 0.6704 ->", "met" if pool(rm["ens_all6"]) <= 0.6704 else "not met")
print(df.round(3)[["depth_m", "ens_unlagged", "ens_lagged", "ens_all6", "frozen_ens_all6", "cost_vs_frozen"]].to_string(index=False))
