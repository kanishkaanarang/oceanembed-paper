"""Predict the full temperature field with a frozen model set for a period and save it. Pure inference: no targets are read, so running it on a
test period is not a look at test results. For the validation period it also scores against GLORYS, as a check of the code.
Usage: python scripts/predict_sets.py SET PERIOD    (SET: frozen | demean; PERIOD: validation | test_a | test_b)"""
import os
import sys

import numpy as np
import pandas as pd
import yaml

from oceanembed.dataset import END
from oceanembed.model_sets import ModelSet, device

SET, PERIOD = sys.argv[1], sys.argv[2]
CFG = {"frozen": "configs/final_model.yaml", "demean": "configs/final_model_demean.yaml"}[SET]
EXPECT = {"frozen": 0.6504, "demean": 0.6677}[SET]
assert PERIOD in ("validation", "test_a", "test_b")
OUT = f"data/processed/pred_{SET}_{PERIOD}.npy"
if os.path.exists(OUT):
    raise SystemExit(f"EXISTS: {OUT} (stop)")

ms = ModelSet(CFG, device())
D = ms.base
t0, t1 = yaml.safe_load(open("configs/splits.yaml"))[PERIOD]
days = pd.date_range(pd.Timestamp(t0), END if t1 == "END" else pd.Timestamp(t1))
print(SET, PERIOD, "days:", len(days), days[0].date(), "to", days[-1].date(), flush=True)
pred = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.float32, shape=(len(days), 15, 101, 241))
for s in range(0, len(days), 8):
    chunk = days[s:s + 8]
    pred[s:s + len(chunk)] = np.where(D.mask[None], D.clim(chunk) + ms.predict_anomaly(chunk), np.nan)
    if (s // 8) % 25 == 0:
        print("predicted days:", min(s + 8, len(days)), flush=True)
pred.flush()
print("saved", OUT, pred.shape)

if PERIOD == "validation":
    sse, n = np.zeros(15), 0
    for i, d in enumerate(days):
        e = (pred[i] - D.glorys(d))[:, D.deep]
        sse += (e ** 2).sum(axis=1)
        n += int(D.deep.sum())
    print("pooled RMSE vs GLORYS on the deep set: %.4f  (expected %.4f)" % (np.sqrt((np.sqrt(sse / n) ** 2).mean()), EXPECT))
