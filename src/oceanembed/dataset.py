"""Date-aligned access to the inputs, the GLORYS target and the climatology, for model training.

Everything is addressed by calendar date. The SLA, wind and SST arrays start on 2010-02-01 (index 0); the GLORYS
blocks start on 2010-02-04 (block 3, index 0). Test periods are deliberately not reachable from this module."""
import json
import os

import numpy as np
import pandas as pd
import yaml

from .glorys_pipeline import block_dates

START = pd.Timestamp("2010-02-01")
END = pd.Timestamp("2025-12-31")
NH = 3
SHAPE = (15, 101, 241)
CHANNELS = ["sst", "sla", "wind_u", "wind_v"]       # plus a land-mask channel
N_IN = len(CHANNELS) + 1
TARGET_SCALE = 2.0      # anomalies are divided by this constant for training (not a data statistic)


def design(days):
    phase = 2 * np.pi * pd.DatetimeIndex(days).dayofyear.values / 365.25
    cols = [np.ones(len(phase))]
    for k in range(1, NH + 1):
        cols += [np.sin(k * phase), np.cos(k * phase)]
    return np.stack(cols, axis=1)


class Data:
    def __init__(self):
        self.cfg = yaml.safe_load(open("configs/splits.yaml"))
        self.mask = np.load("data/processed/target_mask_15depths.npy")
        self.ocean, self.deep = self.mask[0], self.mask.all(axis=0)
        self.sst = np.load("data/processed/sst_all.npy", mmap_mode="r")
        self.sla = np.load("data/processed/ssh_all.npy", mmap_mode="r")
        self.wind = np.load("data/processed/wind_all.npy", mmap_mode="r")
        n = (END - START).days + 1
        assert self.sst.shape == (n, 101, 241) and self.sla.shape == (n, 101, 241), (self.sst.shape, self.sla.shape)
        assert self.wind.shape == (n, 2, 101, 241), self.wind.shape
        self.blocks = []
        for b in (3, 4, 5):
            s, e = block_dates(b)
            a = np.load(f"data/processed/glorys_block{b}.npy", mmap_mode="r")
            assert a.shape[0] == (e - s).days + 1, (b, a.shape)
            self.blocks.append((s, e, a))
        self.beta = None
        self.norm = None

    def period(self, name, stride=1):
        if name not in ("train", "validation"):
            raise PermissionError(f"'{name}' is not available here: test periods are opened once, by a dedicated script")
        t0, t1 = (pd.Timestamp(x) for x in self.cfg[name])
        return pd.date_range(t0, t1)[::stride]

    def glorys(self, day):
        for s, e, a in self.blocks:
            if s <= day <= e:
                return np.asarray(a[int((day - s).days)])
        raise KeyError(f"no GLORYS data for {day.date()}")

    def fit_climatology(self, path="data/processed/clim_beta_train.npy"):
        """Harmonic seasonal cycle per cell and depth, fitted on the training days only."""
        if self.beta is not None:
            return self.beta
        if os.path.exists(path):
            self.beta = np.load(path)
            return self.beta
        t0, t1 = (pd.Timestamp(x) for x in self.cfg["train"])
        P, M = 2 * NH + 1, int(np.prod(SHAPE))
        XtX, Xty = np.zeros((P, P)), np.zeros((P, M))
        for s, e, arr in self.blocks:
            lo, hi = max(s, t0), min(e, t1)
            if lo > hi:
                continue
            i0, i1 = int((lo - s).days), int((hi - s).days) + 1
            for a in range(i0, i1, 50):
                b = min(a + 50, i1)
                days = pd.date_range(s + pd.Timedelta(days=a), periods=b - a)
                X = design(days)
                Y = np.nan_to_num(np.asarray(arr[a:b]).reshape(b - a, -1), nan=0.0).astype(np.float64)
                XtX += X.T @ X
                Xty += X.T @ Y
        self.beta = np.linalg.solve(XtX, Xty).astype(np.float32)
        np.save(path, self.beta)
        return self.beta

    def clim(self, days):
        return (design(days) @ self.fit_climatology()).reshape(len(days), *SHAPE).astype(np.float32)

    def target(self, day):
        """GLORYS anomaly relative to the climatology, in C; 0 outside the mask (never used by the loss)."""
        y = self.glorys(day) - self.clim(pd.DatetimeIndex([day]))[0]
        return np.where(self.mask, y, 0.0).astype(np.float32)

    def fit_norm(self, path="data/processed/norm_stats.json"):
        """Mean and standard deviation of each input over ocean cells, from every 7th TRAINING day."""
        if self.norm is not None:
            return self.norm
        if os.path.exists(path):
            self.norm = json.load(open(path))
            return self.norm
        vals = {c: [] for c in CHANNELS}
        for d in self.period("train", stride=7):
            i = int((d - START).days)
            raw = {"sst": self.sst[i], "sla": self.sla[i], "wind_u": self.wind[i, 0], "wind_v": self.wind[i, 1]}
            for c in CHANNELS:
                a = np.asarray(raw[c])[self.ocean]
                vals[c].append(a[np.isfinite(a)])
        self.norm = {}
        for c in CHANNELS:
            v = np.concatenate(vals[c])
            self.norm[c] = {"mean": float(v.mean()), "std": float(v.std())}
        json.dump(self.norm, open(path, "w"), indent=1)
        return self.norm

    def inputs(self, day):
        """(5, 101, 241): normalised SST, SLA, wind u, wind v (0 on land and where a value is missing) + ocean mask."""
        i = int((day - START).days)
        st = self.fit_norm()
        raw = {"sst": self.sst[i], "sla": self.sla[i], "wind_u": self.wind[i, 0], "wind_v": self.wind[i, 1]}
        out = np.zeros((N_IN, 101, 241), dtype=np.float32)
        for c, name in enumerate(CHANNELS):
            z = (np.asarray(raw[name], dtype=np.float32) - st[name]["mean"]) / st[name]["std"]
            out[c] = np.where(np.isfinite(z) & self.ocean, z, 0.0)
        out[-1] = self.ocean.astype(np.float32)
        return out

    def batch(self, days):
        X = np.stack([self.inputs(d) for d in days])
        Y = np.stack([self.target(d) for d in days]) / TARGET_SCALE
        return X.astype(np.float32), Y.astype(np.float32)
