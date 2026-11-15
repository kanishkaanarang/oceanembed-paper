"""Build the test_c input arrays (135 days from 2026-01-01) from the monthly files, with integrity checks only."""
import numpy as np
import pandas as pd

N = 135  # 2026-01-01 .. 2026-05-15
months = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
dates = pd.date_range("2026-01-01", periods=N)
assert dates[-1] == pd.Timestamp("2026-05-15")


def load(path_fmt):
    parts = []
    for m in months:
        z = np.load(path_fmt.format(m=m))
        assert str(z["month"]) == m and z["values"].shape[0] == int(z["ndays"])
        parts.append(z["values"])
    return np.concatenate(parts, axis=0)


ssh = load("data/interim/surface/ssh/{m}.npz")
sst = load("data/interim/sst/{m}.npz")
print("SLA days", ssh.shape[0], "| SST days (whole months)", sst.shape[0])
assert ssh.shape[0] == N and sst.shape[0] == 151
sst = sst[:N]
assert ssh.shape == sst.shape == (N, 101, 241)

mask = np.load("data/processed/target_mask_15depths.npy").astype(bool)
deep = mask.all(axis=0)
print("deep cells", int(deep.sum()), "(expect 9140)")
print("non-finite SLA in deep set:", int((~np.isfinite(ssh[:, deep])).sum()), "| SST:", int((~np.isfinite(sst[:, deep])).sum()))
print("SST range (K or C?):", round(float(np.nanmin(sst)), 2), round(float(np.nanmax(sst)), 2))

g = np.load("data/processed/glorys_block6.npy", mmap_mode="r")
g0 = np.asarray(g[:, 0])
sst_c = sst - 273.15 if np.nanmean(sst) > 100 else sst
d = (sst_c - g0)[:, deep]
print("SST minus GLORYS at 0 m, deep set: mean", round(float(np.nanmean(d)), 3), "| RMS", round(float(np.sqrt(np.nanmean(d ** 2))), 3), "(training years: ~0.48 at lag 0)")

np.save("data/processed/ssh_test_c.npy", ssh.astype(np.float32))
np.save("data/processed/sst_test_c.npy", sst.astype(np.float32))
print("saved ssh_test_c.npy and sst_test_c.npy, dates", dates[0].date(), "..", dates[-1].date())
