"""Counterfactual sensitivity of the frozen final model to a uniform input offset, on the VALIDATION window: add +5 cm to SLA and/or +0.5 C to SST
(all ocean cells, all lags) and report the mean change of the predicted temperature by depth over deep-set cells. No test periods."""
import os

import numpy as np
import pandas as pd
import torch

from oceanembed.dataset import TARGET_SCALE
from oceanembed.dataset_anom import AnomData
from oceanembed.final_model import device, lag_inputs, load_members
from oceanembed.glorys import TARGET_DEPTHS

OUT = "results/offset_sensitivity_validation.csv"
if os.path.exists(OUT):
    raise SystemExit(f"EXISTS: {OUT} (stop)")

dev = device()
D = AnomData()
D.fit_input_clim()
days = D.period("validation")
members = load_members(dev)
d_sla, d_sst = 0.05 / D.istd["sla"], 0.5 / D.istd["sst"]          # SLA is in metres: 0.05 m = 5 cm
ocean = torch.from_numpy(D.ocean.astype(np.float32)).to(dev)
deep = torch.from_numpy(D.deep.astype(np.float32)).to(dev)
cases = {"SLA +5 cm": (d_sla, 0.0), "SST +0.5 C": (0.0, d_sst), "both": (d_sla, d_sst)}


def ens(Xu, Xl):
    return sum(net(Xl if lagged else Xu) for lagged, net in members) / len(members)


def shifted(Xu, Xl, dsla, dsst):
    Xu2, Xl2 = Xu.clone(), Xl.clone()
    Xu2[:, 0] += dsst * ocean
    Xu2[:, 1] += dsla * ocean
    for k in range(4):
        Xl2[:, 2 * k] += dsst * ocean
        Xl2[:, 2 * k + 1] += dsla * ocean
    return Xu2, Xl2


acc, ncell = {c: np.zeros(15) for c in cases}, 0.0
with torch.no_grad():
    for s in range(0, len(days), 8):
        chunk = days[s:s + 8]
        Xu = torch.from_numpy(np.stack([D.inputs(d)[[0, 1, 4]] for d in chunk])).to(dev)
        Xl = torch.from_numpy(np.stack([lag_inputs(D, d) for d in chunk])).to(dev)
        base = ens(Xu, Xl)
        for name, (a, b) in cases.items():
            acc[name] += (TARGET_SCALE * (ens(*shifted(Xu, Xl, a, b)) - base) * deep).sum(dim=(0, 2, 3)).cpu().numpy()
        ncell += float(deep.sum() * len(chunk))
        if (s // 8) % 20 == 0:
            print("validation days done:", min(s + 8, len(days)), flush=True)
df = pd.DataFrame({"depth_m": TARGET_DEPTHS.astype(int), **{k: v / ncell for k, v in acc.items()}})
df["SLA_per_cm"] = df["SLA +5 cm"] / 5
df.round(3).to_csv(OUT, index=False)
print("mean change of the predicted temperature (C) over deep-set cells, validation days")
print(df.round(3).to_string(index=False))
print("reference slopes at 100 m: training-period regression 0.142 C/cm; yearly GLORYS-minus-Argo vs SLA 0.098 C/cm")
