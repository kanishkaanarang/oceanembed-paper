"""Anomaly inputs with the domain-mean sea level removed each day: a uniform sea level offset (for example global mean sea level rise)
carries no thermocline information. Same-day operation, so no future information is used. Optional lags (class attribute `lags`)."""
import json
import os

import numpy as np
import pandas as pd

from .dataset import START
from .dataset_anom import AnomData


class AnomDemeanData(AnomData):
    lags = [0]

    def period(self, name, stride=1):
        days = super().period(name, stride)
        return days[np.array([(d - START).days >= max(self.lags) for d in days])]

    def _sst_sla_dm(self, day):
        a = self._anoms(day)
        sla = a[1].astype(np.float64)
        return a[0], sla - np.nanmean(sla[self.deep])

    def fit_demean_std(self, path="data/processed/input_anom_stats_demean.json"):
        if getattr(self, "dstd", None) is not None:
            return self.dstd
        if os.path.exists(path):
            self.dstd = json.load(open(path))
            return self.dstd
        self.fit_input_clim()
        vals = []
        for d in super().period("train", stride=7):
            v = self._sst_sla_dm(d)[1][self.ocean]
            vals.append(v[np.isfinite(v)])
        self.dstd = {"sla": float(np.concatenate(vals).std())}
        json.dump(self.dstd, open(path, "w"), indent=1)
        return self.dstd

    def inputs(self, day):
        self.fit_input_clim()
        dstd = self.fit_demean_std()
        out = np.zeros((2 * len(self.lags) + 1, 101, 241), dtype=np.float32)
        for k, L in enumerate(self.lags):
            sst, sla = self._sst_sla_dm(day - pd.Timedelta(days=L))
            for c, (field, scale) in enumerate([(sst, self.istd["sst"]), (sla, dstd["sla"])]):
                z = field / scale
                out[2 * k + c] = np.where(np.isfinite(z) & self.ocean, z, 0.0)
        out[-1] = self.ocean.astype(np.float32)
        return out
