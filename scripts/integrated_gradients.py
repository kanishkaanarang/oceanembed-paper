"""Integrated Gradients for the frozen ensembles on a sample of VALIDATION days (inputs only, no targets).
Baseline: all anomaly channels zero (climatological state), land-mask channel kept. Target per depth: sum over deep-set cells of the squared
predicted anomaly (sign-free). Channels: SST at even positions, SLA at odd positions, mask last; lag index = position // 2.
Output: share of absolute attribution by input (SST/SLA) and by lag, per depth."""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from oceanembed.glorys import TARGET_DEPTHS
from oceanembed.model_sets import ModelSet, device

OUT = "results/ig_validation_by_depth.csv"
if os.path.exists(OUT):
    raise SystemExit(f"EXISTS: {OUT} (stop)")
STRIDE, STEPS = 40, 12
dev = device()
rows = []
t0 = time.time()
for SET, cfg in (("frozen", "configs/final_model.yaml"), ("demean", "configs/final_model_demean.yaml")):
    ms = ModelSet(cfg, dev)
    D = ms.base
    deep = torch.from_numpy(D.deep).to(dev)
    days = D.period("validation", stride=STRIDE)
    print(SET, "validation days sampled:", len(days), flush=True)
    sst_all, sla_all, lagged_lag, comp = [], [], [], []
    al = ((torch.arange(STEPS, device=dev, dtype=torch.float32) + 0.5) / STEPS)[:, None, None, None]
    for di, day in enumerate(days):
        for lagged, net in ms.members:
            x = torch.from_numpy(ms._inputs(lagged, day))[None].to(dev)
            b = x.clone()
            b[:, :-1] = 0
            X = (b + al * (x - b)).requires_grad_(True)
            out = net(X)
            Fs = [(out[:, k][:, deep] ** 2).sum() for k in range(15)]
            grads = [torch.autograd.grad(Fs[k], X, retain_graph=(k < 14))[0].mean(0) for k in range(15)]
            IG = torch.stack(grads) * (x - b)[0]                                  # (15, C, H, W)
            with torch.no_grad():
                Fx = torch.stack([(net(x)[0, k][deep] ** 2).sum() for k in range(15)])
                Fb = torch.stack([(net(b)[0, k][deep] ** 2).sum() for k in range(15)])
                diff = Fx - Fb
                comp.append((((IG.sum((1, 2, 3)) - diff).abs()) / (diff.abs() + 1e-3)).cpu().numpy())
                A = IG.abs().sum((2, 3)).cpu().numpy()                            # (15, C)
            sst, sla = A[:, 0:-1:2], A[:, 1:-1:2]                                 # (15, nlags)
            tot = sst.sum(1) + sla.sum(1)
            sst_all.append(sst.sum(1) / tot)
            sla_all.append(sla.sum(1) / tot)
            if lagged:
                lagged_lag.append((sst + sla) / tot[:, None])                     # (15, 4) share by lag
        print(SET, "day", di + 1, "of", len(days), "(%.0f s)" % (time.time() - t0), flush=True)
    comp = np.concatenate(comp)
    print(SET, "completeness: median relative error %.3f, 90th percentile %.3f" % (np.median(comp), np.percentile(comp, 90)), flush=True)
    sst_s, sla_s, lag_s = np.mean(sst_all, 0), np.mean(sla_all, 0), np.mean(lagged_lag, 0)
    df = pd.DataFrame({"set": SET, "depth_m": TARGET_DEPTHS.astype(int), "share_sst": sst_s, "share_sla": sla_s,
                       "lag0": lag_s[:, 0], "lag7": lag_s[:, 1], "lag14": lag_s[:, 2], "lag30": lag_s[:, 3]})
    rows.append(df)
    print(f"\n{SET}: share of absolute attribution (all six members), and by lag (lagged members only)")
    print(df.drop(columns="set").round(3).to_string(index=False), flush=True)
pd.concat(rows).to_csv(OUT, index=False)
print("saved", OUT)
