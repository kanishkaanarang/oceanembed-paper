"""Do the inputs carry a long-term trend that a model trained on 2010-2017 could mistake for thermocline warming? Uses the input arrays (all years),
the TRAINING-period GLORYS target and results/argo_vs_glorys_by_year_100m.csv. No model output; prints only."""
import numpy as np
import pandas as pd

from oceanembed.dataset import START
from oceanembed.dataset_anom import AnomData

D = AnomData()
D.fit_input_clim()
n = D.sla.shape[0]
dates = pd.date_range(START, periods=n)
sla, sst = np.empty(n), np.empty(n)
for s in range(0, n, 400):
    sla[s:s + 400] = np.nanmean(np.asarray(D.sla[s:s + 400])[:, D.deep], axis=1)
    sst[s:s + 400] = np.nanmean(np.asarray(D.sst[s:s + 400])[:, D.deep], axis=1)
yrs = (dates - START).days.values / 365.25
print("domain-mean (deep-set cells) trend over 2010-02..2025-12: SLA %+.2f cm/yr | SST %+.3f C/yr"
      % (100 * np.polyfit(yrs, sla, 1)[0], np.polyfit(yrs, sst, 1)[0]))

tr = D.period("train", stride=3)
a_sla, a_t100 = [], []
for d in tr:
    a_sla.append(np.nanmean(D._anoms(d)[1][D.deep]))
    a_t100.append(D.target(d)[7][D.deep].mean())
slope = np.polyfit(a_sla, a_t100, 1)[0]
print("training-period slope: 100 m domain-mean GLORYS anomaly per cm of domain-mean SLA anomaly: %.3f C/cm (rough, mixes all time scales)" % (slope / 100))

train_mean = sla[(dates >= "2010-02-04") & (dates <= "2017-12-31")].mean()
a = pd.read_csv("results/argo_vs_glorys_by_year_100m.csv").set_index("year")
tab = pd.DataFrame({"sla_cm": 100 * pd.Series(sla, index=dates).groupby(dates.year).mean()})
tab["sla_minus_train_cm"] = tab["sla_cm"] - 100 * train_mean
tab["implied_dT100_C"] = tab["sla_minus_train_cm"] * slope / 100
tab["sst_C"] = pd.Series(sst, index=dates).groupby(dates.year).mean()
tab["glorys_minus_argo_100m_east75E"] = a["bias_all_modes"]
print("\nyearly domain means; implied_dT100_C = (SLA minus its 2010-2017 mean) x the training slope; last column from the Argo table")
print(tab.round(2).to_string())
