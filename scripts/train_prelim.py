"""Preliminary model: small U-Net on SST, SLA, winds (+ ocean-mask channel) predicting the GLORYS temperature anomaly
relative to the training climatology. Trained on TRAINING days only; early stopping and reporting on the VALIDATION
window (so reported numbers are slightly optimistic). Test periods are unreachable from the data module."""
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from oceanembed.dataset import N_IN, TARGET_SCALE, Data
from oceanembed.glorys import TARGET_DEPTHS

SEED, STRIDE_TRAIN, EPOCHS, PATIENCE, BATCH, LR, WIDTH = 0, 2, 25, 6, 8, 1e-3, 32
torch.manual_seed(SEED)
np.random.seed(SEED)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", dev)

D = Data()
tr, va = D.period("train", stride=STRIDE_TRAIN), D.period("validation")
print("train days:", len(tr), "(every %d-th) | validation days:" % STRIDE_TRAIN, len(va))


def load_set(days):
    X = np.empty((len(days), N_IN, 101, 241), np.float32)
    Y = np.empty((len(days), 15, 101, 241), np.float32)
    for i, d in enumerate(days):
        X[i] = D.inputs(d)
        Y[i] = D.target(d) / TARGET_SCALE
    return torch.from_numpy(X), torch.from_numpy(Y)


t0 = time.time()
Xtr, Ytr = load_set(tr)
Xva, Yva = load_set(va)
print("data loaded in %.0f s | %.1f GB in memory" % (time.time() - t0, (Xtr.numel() + Ytr.numel() + Xva.numel() + Yva.numel()) * 4 / 1e9))
M = torch.from_numpy(D.mask.astype(np.float32))[None].to(dev)             # (1, 15, 101, 241) valid target entries
DEEP = torch.from_numpy(D.deep.astype(np.float32))[None, None].to(dev)    # (1, 1, 101, 241) headline cells


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, cin=N_IN, cout=15, w=WIDTH):
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


def masked_mse(pred, y):
    return (((pred - y) ** 2) * M).sum() / (M.sum() * pred.shape[0])


@torch.no_grad()
def evaluate(model, X, Y):
    model.eval()
    sse, sse0, bias = (torch.zeros(15, device=dev) for _ in range(3))
    se_all = n_all = 0.0
    for s in range(0, len(X), 16):
        x, y = X[s:s + 16].to(dev), Y[s:s + 16].to(dev)
        e = model(x) - y
        se_all += float(((e ** 2) * M).sum())
        n_all += float(M.sum() * len(x))
        ed, yd = e * DEEP, y * DEEP
        sse += (ed ** 2).sum(dim=(0, 2, 3))
        sse0 += (yd ** 2).sum(dim=(0, 2, 3))
        bias += ed.sum(dim=(0, 2, 3))
    n = float(DEEP.sum() * len(X))
    sse, sse0, bias = sse.cpu().numpy(), sse0.cpu().numpy(), bias.cpu().numpy()
    return {"loss": se_all / n_all, "rmse": TARGET_SCALE * np.sqrt(sse / n), "rmse0": TARGET_SCALE * np.sqrt(sse0 / n),
            "bias": TARGET_SCALE * bias / n, "skill": 1 - sse / sse0}


model = UNet().to(dev)
print("parameters: %.2f M" % (sum(p.numel() for p in model.parameters()) / 1e6))
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

best, best_state, bad = 1e9, None, 0
for ep in range(1, EPOCHS + 1):
    t = time.time()
    model.train()
    perm = torch.randperm(len(Xtr))
    tot = 0.0
    for s in range(0, len(perm), BATCH):
        idx = perm[s:s + BATCH]
        x, y = Xtr[idx].to(dev), Ytr[idx].to(dev)
        loss = masked_mse(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += float(loss) * len(idx)
    sched.step()
    r = evaluate(model, Xva, Yva)
    tag = ""
    if r["loss"] < best:
        best, bad = r["loss"], 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        tag = " *best"
    else:
        bad += 1
    print("epoch %2d | train loss %.4f | validation loss %.4f | %.0f s%s" % (ep, tot / len(perm), r["loss"], time.time() - t, tag), flush=True)
    if bad >= PATIENCE:
        print("early stopping")
        break

model.load_state_dict(best_state)
r = evaluate(model, Xva, Yva)
os.makedirs("results/figures", exist_ok=True)
torch.save(best_state, "data/processed/prelim_unet.pt")
base = pd.read_csv("results/climatology_baseline_validation.csv")
df = pd.DataFrame({"depth_m": TARGET_DEPTHS.astype(int), "clim_rmse": r["rmse0"], "model_rmse": r["rmse"],
                   "skill": r["skill"], "model_bias": r["bias"]})
print("\nvalidation window %s to %s, deep set. RMSE in C; skill = 1 - MSE_model / MSE_climatology" % tuple(D.cfg["validation"]))
print(df.round(3).to_string(index=False))
pooled0, pooled = np.sqrt((r["rmse0"] ** 2).mean()), np.sqrt((r["rmse"] ** 2).mean())
print("\npooled over depths: climatology %.3f C | model %.3f C | check vs baseline csv: largest difference %.4f C"
      % (pooled0, pooled, np.abs(r["rmse0"] - base["val_rmse"].values).max()))
df.to_csv("results/prelim_model_validation.csv", index=False)

plt.figure(figsize=(5, 6))
plt.plot(r["rmse0"], TARGET_DEPTHS, "o-", label="seasonal climatology")
plt.plot(r["rmse"], TARGET_DEPTHS, "o-", label="preliminary U-Net")
plt.gca().invert_yaxis()
plt.xlabel("RMSE (C), deep set")
plt.ylabel("depth (m)")
plt.title("PRELIMINARY: validation window Feb 2018 to Jan 2020\ninputs SST, SLA, winds; trained on 2010-2017", fontsize=8)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/prelim_rmse_by_depth.png", dpi=200)
json.dump({"seed": SEED, "stride_train": STRIDE_TRAIN, "epochs_max": EPOCHS, "batch": BATCH, "lr": LR, "width": WIDTH},
          open("results/prelim_model_config.json", "w"))
print("saved results/prelim_model_validation.csv, results/figures/prelim_rmse_by_depth.png")
