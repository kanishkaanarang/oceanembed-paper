"""Runs scripts/train_variant.py with anomaly inputs at lags 0, 7, 14 and 30 days (SST and SLA; past dates only; same anomaly definition
as AnomData, whose cached climatology and standard deviations are reused). Training days whose longest lag would fall before the start
of the input arrays are dropped. Usage: python scripts/train_variant_anom_lags.py NAME ch1,ch2,..."""
import runpy

import numpy as np
import pandas as pd

import oceanembed.dataset as ds
from oceanembed.dataset import START
from oceanembed.dataset_anom import AnomData

LAGS = [0, 7, 14, 30]
BASE = ["sst", "sla"]
NEW = [f"{c}_l{L}" for L in LAGS for c in BASE]


class AnomLagData(AnomData):
    def period(self, name, stride=1):
        days = super().period(name, stride)
        return days[np.array([(d - START).days >= max(LAGS) for d in days])]

    def inputs(self, day):
        self.fit_input_clim()
        out = np.zeros((len(NEW) + 1, 101, 241), dtype=np.float32)
        for k, L in enumerate(LAGS):
            assert (day - START).days >= L, day
            a = self._anoms(day - pd.Timedelta(days=L))
            for c, name in enumerate(BASE):          # anomaly channel order is sst, sla, wind_u, wind_v
                z = a[c] / self.istd[name]
                out[2 * k + c] = np.where(np.isfinite(z) & self.ocean, z, 0.0)
        out[-1] = self.ocean.astype(np.float32)
        return out


ds.Data = AnomLagData
ds.CHANNELS = NEW
runpy.run_path("scripts/train_variant.py", run_name="__main__")
