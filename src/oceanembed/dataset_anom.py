"""Inputs as anomalies: each input channel minus its own training-period harmonic seasonal cycle (per cell),
divided by a training-period standard deviation. Targets, splits and the test-period guard are inherited from Data."""
import json
import os

import numpy as np
import pandas as pd

from .dataset import CHANNELS, N_IN, NH, START, Data, design


class AnomData(Data):
    def _raw(self, i):
        return {"sst": self.sst[i], "sla": self.sla[i], "wind_u": self.wind[i, 0], "wind_v": self.wind[i, 1]}

    def _anoms(self, day):
        d = design(pd.DatetimeIndex([day]))[0].astype(np.float32)
        clim = np.einsum("p,cpn->cn", d, self.ibeta).reshape(len(CHANNELS), 101, 241)
        raw = self._raw(int((day - START).days))
        return np.stack([np.asarray(raw[name], dtype=np.float32) - clim[c] for c, name in enumerate(CHANNELS)])

    def fit_input_clim(self, path="data/processed/input_clim_beta.npy", stats="data/processed/input_anom_stats.json"):
        if getattr(self, "ibeta", None) is not None:
            return
        if os.path.exists(path) and os.path.exists(stats):
            self.ibeta = np.load(path)
            self.istd = json.load(open(stats))
            return
        tr = self.period("train")
        P, ncell = 2 * NH + 1, 101 * 241
        XtX = np.zeros((P, P))
        Xty = np.zeros((len(CHANNELS), P, ncell))
        for s in range(0, len(tr), 100):
            days = tr[s:s + 100]
            X = design(days)
            XtX += X.T @ X
            raws = [self._raw(int((d - START).days)) for d in days]
            for c, name in enumerate(CHANNELS):
                Y = np.stack([np.nan_to_num(np.asarray(r[name], dtype=np.float64), nan=0.0).ravel() for r in raws])
                Xty[c] += X.T @ Y
        self.ibeta = np.stack([np.linalg.solve(XtX, Xty[c]) for c in range(len(CHANNELS))]).astype(np.float32)
        np.save(path, self.ibeta)
        vals = {name: [] for name in CHANNELS}
        for d in self.period("train", stride=7):
            a = self._anoms(d)
            for c, name in enumerate(CHANNELS):
                v = a[c][self.ocean]
                vals[name].append(v[np.isfinite(v)])
        self.istd = {name: float(np.concatenate(vals[name]).std()) for name in CHANNELS}
        json.dump(self.istd, open(stats, "w"), indent=1)
        print("input anomaly standard deviations:", {k: round(v, 3) for k, v in self.istd.items()})

    def inputs(self, day):
        self.fit_input_clim()
        a = self._anoms(day)
        out = np.zeros((N_IN, 101, 241), dtype=np.float32)
        for c, name in enumerate(CHANNELS):
            z = a[c] / self.istd[name]
            out[c] = np.where(np.isfinite(z) & self.ocean, z, 0.0)
        out[-1] = self.ocean.astype(np.float32)
        return out
