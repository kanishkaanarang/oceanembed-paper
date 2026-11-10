"""The frozen final model (configs/final_model.yaml): the mean of six U-Nets. Prediction = training-period climatology + TARGET_SCALE * mean of the members' outputs."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .dataset import START, TARGET_SCALE

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


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def lag_inputs(D, day):
    """SST and SLA anomalies (normalised) at lags 0, 7, 14, 30 days plus the ocean mask: (9, 101, 241)."""
    out = np.zeros((2 * len(LAGS) + 1, 101, 241), dtype=np.float32)
    for k, L in enumerate(LAGS):
        assert (day - START).days >= L, day
        a = D._anoms(day - pd.Timedelta(days=L))
        for c, name in enumerate(["sst", "sla"]):
            z = a[c] / D.istd[name]
            out[2 * k + c] = np.where(np.isfinite(z) & D.ocean, z, 0.0)
    out[-1] = D.ocean.astype(np.float32)
    return out


def load_members(dev, cfg_path="configs/final_model.yaml"):
    members = []
    for m in yaml.safe_load(open(cfg_path))["members"]:
        lagged = "lags" in m["file"]
        net = UNet(9 if lagged else 3).to(dev)
        net.load_state_dict(torch.load(m["file"], map_location=dev))
        members.append((lagged, net.eval()))
    return members


@torch.no_grad()
def predict_anomaly(D, members, days, dev):
    """Ensemble-mean GLORYS-anomaly prediction in C, shape (len(days), 15, 101, 241)."""
    Xu = torch.from_numpy(np.stack([D.inputs(d)[[0, 1, 4]] for d in days])).to(dev)     # SST, SLA, ocean mask (same day)
    Xl = torch.from_numpy(np.stack([lag_inputs(D, d) for d in days])).to(dev)
    out = sum(net(Xl if lagged else Xu) for lagged, net in members) / len(members)
    return (TARGET_SCALE * out).cpu().numpy()
