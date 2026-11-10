"""Seed ensembles on the VALIDATION window. Loads the saved weights of the anomaly-input U-Nets (SST+SLA same day, 3 seeds; SST+SLA at lags
0/7/14/30, 3 seeds), averages their predictions and scores the average per depth on the deep set. No training; test periods unreachable."""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from oceanembed.dataset import START, TARGET_SCALE
from oceanembed.dataset_anom import AnomData
from oceanembed.glorys import TARGET_DEPTHS

OUT = "results/variants/ensemble_validation.csv"
if os.path.exists(OUT):
    raise SystemExit(f"{OUT} already exists - not overwriting")
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
LAGS, WIDTH = [0, 7, 14, 30], 32


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, cin, cout=15, w=WIDTH):
        super().__init__()
        self.e1, self.e2, self.b = Block(cin, w), Block(w, 2 * w), Block(2 * w, 4 * w)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d2, self.d1 = Block(4 * w + 2 * w, 2 * w), Block(2 * w + w, w)
        self.out = nn.Conv2d(w, cout, 1)

    def forward(self, x):
        h, wd = x.shape[-2:]
        x = F.pad(x, (0, (4 - wd % 4) % 4, 0, (4 - h % 4) % 4))
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        b = self.b(self.pool(e2))
        d2 = self.d2(torch.cat([self.up(b), e2], 1))
        d1 = self.d1(torch.cat([self.up(d2), e1], 1))
        return self.out(d1)[..., :h, :wd]


D = AnomData()
D.fit_input_clim()
va = D.period("validation")
DEEP = torch.from_numpy(D.deep.astype(np.float32))[None, None].to(dev)


def lag_inputs(day):
    out = np.zeros((2 * len(LAGS) + 1, 101, 241), dtype=np.float32)
    for k, L in enumerate(LAGS):
        assert (day - START).days >= L, day
        a = D._anoms(day - pd.Timedelta(days=L))
        for c, name in enumerate(["sst", "sla"]):
            z = a[c] / D.istd[name]
            out[2 * k + c] = np.where(np.isfinite(z) & D.ocean, z, 0.0)
    out[-1] = D.ocean.astype(np.float32)
    return out


SETS = {"unlagged": ["anom_sst_sla", "anom_sst_sla_s1", "anom_sst_sla_s2"], "lagged": ["anom_lags", "anom_lags_s1", "anom_lags_s2"]}
models = {}
for kind, names in SETS.items():
    for n in names:
        m = UNet(3 if kind == "unlagged" else 9).to(dev)
        m.load_state_dict(torch.load(f"data/processed/variant_{n}.pt", map_location=dev))
        models[n] = (kind, m.eval())
keys = list(models) + ["ens_unlagged", "ens_lagged", "ens_all6"]
sse, sse0, ntot = {k: np.zeros(15) for k in keys}, np.zeros(15), 0.0
with torch.no_grad():
    for s in range(0, len(va), 8):
        days = va[s:s + 8]
        Xu = torch.from_numpy(np.stack([D.inputs(d)[[0, 1, 4]] for d in days])).to(dev)   # SST, SLA, ocean mask
        Xl = torch.from_numpy(np.stack([lag_inputs(d) for d in days])).to(dev)
        Y = torch.from_numpy(np.stack([D.target(d) for d in days]) / TARGET_SCALE).to(dev)
        P = {n: m(Xu if kind == "unlagged" else Xl) for n, (kind, m) in models.items()}
        P["ens_unlagged"] = sum(P[n] for n in SETS["unlagged"]) / 3
        P["ens_lagged"] = sum(P[n] for n in SETS["lagged"]) / 3
        P["ens_all6"] = (P["ens_unlagged"] + P["ens_lagged"]) / 2
        sse0 += ((Y * DEEP) ** 2).sum(dim=(0, 2, 3)).cpu().numpy()
        for k in keys:
            sse[k] += (((P[k] - Y) * DEEP) ** 2).sum(dim=(0, 2, 3)).cpu().numpy()
        ntot += float(DEEP.sum() * len(days))
        if (s // 8) % 20 == 0:
            print("validation days done:", min(s + 8, len(va)), flush=True)
rm = {k: TARGET_SCALE * np.sqrt(v / ntot) for k, v in sse.items()}
rm0 = TARGET_SCALE * np.sqrt(sse0 / ntot)
pool = lambda v: float(np.sqrt((np.asarray(v) ** 2).mean()))
df = pd.DataFrame({"depth_m": TARGET_DEPTHS.astype(int), "clim_rmse": rm0, **rm})
df.to_csv(OUT, index=False)
print("pooled validation RMSE (deep set), climatology %.3f" % pool(rm0))
for k in keys:
    print("  %-16s %.4f" % (k, pool(rm[k])))
print(df.round(3)[["depth_m", "anom_sst_sla", "ens_unlagged", "ens_lagged", "ens_all6"]].to_string(index=False))
