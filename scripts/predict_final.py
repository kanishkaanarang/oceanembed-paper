"""Predict the full temperature field with the frozen final model for the VALIDATION window (dry run for the test evaluation), save it,
then reload it and recompute the pooled deep-set RMSE against GLORYS (expected 0.6504, the ensemble check). Test periods are refused."""
import os
import sys

import numpy as np

from oceanembed.dataset_anom import AnomData
from oceanembed.final_model import device, load_members, predict_anomaly

PERIOD = sys.argv[1] if len(sys.argv) > 1 else "validation"
if PERIOD != "validation":
    raise SystemExit("only the validation window is supported in this dry run")
OUT = f"data/processed/pred_final_{PERIOD}.npy"
if os.path.exists(OUT):
    raise SystemExit(f"{OUT} already exists - not overwriting")

dev = device()
D = AnomData()
D.fit_input_clim()
days = D.period(PERIOD)
members = load_members(dev)
pred = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.float32, shape=(len(days), 15, 101, 241))
for s in range(0, len(days), 8):
    chunk = days[s:s + 8]
    pred[s:s + len(chunk)] = np.where(D.mask[None], D.clim(chunk) + predict_anomaly(D, members, chunk, dev), np.nan)
    if (s // 8) % 20 == 0:
        print("predicted days:", min(s + 8, len(days)), flush=True)
pred.flush()

sse, n = np.zeros(15), 0
for i, d in enumerate(days):
    e = (pred[i] - D.glorys(d))[:, D.deep]
    sse += (e ** 2).sum(axis=1)
    n += int(D.deep.sum())
rmse = np.sqrt(sse / n)
print("saved", OUT, pred.shape)
print("pooled RMSE vs GLORYS on the deep set: %.4f  (expected 0.6504)" % np.sqrt((rmse ** 2).mean()))
print("RMSE at 0 / 100 / 1000 m: %.3f / %.3f / %.3f" % (rmse[0], rmse[7], rmse[14]))
