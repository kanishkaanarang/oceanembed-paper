"""Where is the final model worse than climatology at 100 m (validation window)? Lists the worst cells and compares the north-central
Arabian Sea box (60-70E, 14-22N) with the rest. Prints only."""
import numpy as np
import pandas as pd

from oceanembed.dataset import START
from oceanembed.dataset_anom import AnomData

D = AnomData()
days = D.period("validation")
n = len(days)
pred = np.load("data/processed/pred_final_validation.npy", mmap_mode="r")
z = 7                                                     # 100 m
e, e0 = np.empty((n, 101, 241), np.float32), np.empty((n, 101, 241), np.float32)
for i, d in enumerate(days):
    e[i] = pred[i, z] - D.glorys(d)[z]                    # model minus GLORYS
    e0[i] = D.target(d)[z]                                # climatology error (minus the GLORYS anomaly; sign irrelevant)
deep = D.deep
mse_m, mse_c = (e ** 2).mean(axis=0), (e0 ** 2).mean(axis=0)
skill = np.where(deep & (mse_c > 0), 1 - mse_m / np.where(mse_c > 0, mse_c, 1), np.nan)
bias = e.mean(axis=0)
i0 = (days[0] - START).days
sla_nan = np.isnan(np.asarray(D.sla[i0:i0 + n])).mean(axis=0)
sst_nan = np.isnan(np.asarray(D.sst[i0:i0 + n])).mean(axis=0)
LAT = 5 + 0.25 * np.arange(101)[:, None] + 0 * np.arange(241)[None, :]
LON = 45 + 0.25 * np.arange(241)[None, :] + 0 * np.arange(101)[:, None]

neg = deep & (skill < 0)
print("deep-set cells at 100 m with skill < 0: %d of %d (%.1f%%)" % (neg.sum(), deep.sum(), 100 * neg.sum() / deep.sum()))
print("\n12 worst cells at 100 m")
order = np.argsort(np.where(np.isnan(skill), np.inf, skill), axis=None)[:12]
rows = []
for idx in order:
    r, c = np.unravel_index(idx, skill.shape)
    rows.append((LAT[r, c], LON[r, c], skill[r, c], np.sqrt(mse_m[r, c]), np.sqrt(mse_c[r, c]), bias[r, c], sla_nan[r, c], sst_nan[r, c]))
print(pd.DataFrame(rows, columns=["lat", "lon", "skill", "model_rmse", "clim_rmse", "bias", "sla_nan_frac", "sst_nan_frac"]).round(2).to_string(index=False))

hot = deep & (LON >= 60) & (LON <= 70) & (LAT >= 14) & (LAT <= 22)
arab = deep & (LON >= 50) & (LON <= 77) & (LAT >= 8) & (LAT <= 26) & ~hot
bob = deep & (LON >= 80) & (LON < 92) & (LAT >= 8) & (LAT <= 22)
print("\n100 m, validation window: RMSE in C, skill, mean error (model minus GLORYS)")
for name, m in [("box 60-70E, 14-22N", hot), ("rest of Arabian Sea", arab), ("Bay of Bengal 80-92E", bob)]:
    print("  %-22s cells %4d | clim RMSE %.2f | model RMSE %.2f | skill %.2f | bias %+.2f"
          % (name, m.sum(), np.sqrt(mse_c[m].mean()), np.sqrt(mse_m[m].mean()), 1 - mse_m[m].sum() / mse_c[m].sum(), bias[m].mean()))
seas = np.array([(d.month % 12) // 3 for d in days])
print("\nbox 60-70E, 14-22N by season")
for s, nm in enumerate(["DJF", "MAM", "JJA", "SON"]):
    k = seas == s
    print("  %s | clim RMSE %.2f | model RMSE %.2f | skill %.2f | bias %+.2f"
          % (nm, np.sqrt((e0[k][:, hot] ** 2).mean()), np.sqrt((e[k][:, hot] ** 2).mean()),
             1 - (e[k][:, hot] ** 2).mean() / (e0[k][:, hot] ** 2).mean(), e[k][:, hot].mean()))
