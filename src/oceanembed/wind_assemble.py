"""Assemble per-task wind results (daily sums and hour counts) into one daily-mean array (days, 2, 101, 241)."""
import os

import numpy as np
import pandas as pd

from .grid import target_grid
from .wind_pipeline import FIRST_DAY, LAST_DAY, make_tasks, needed_chunks, task_path


def assemble_wind(out_path):
    lat, lon = target_grid()
    days = pd.date_range(FIRST_DAY, LAST_DAY)
    nd = len(days)
    d0 = np.datetime64(FIRST_DAY.date(), "D")
    tasks = make_tasks()
    missing = [t for t in tasks if not os.path.exists(task_path(*t))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(tasks)} task files are missing, first: {missing[:3]}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(nd, 2, len(lat), len(lon)))
    filled = np.zeros((len(lat), len(lon)), dtype=int)
    for a, b in sorted({(a, b) for a, b, _ in tasks}):
        S, C, ti0, tj0 = None, np.zeros(nd, dtype=np.int64), None, None
        for k in needed_chunks():
            z = np.load(task_path(a, b, k))
            ti, tj = z["ti"], z["tj"]
            if S is None:
                ti0, tj0 = ti, tj
                S = np.zeros((nd, 2, len(ti), len(tj)), dtype=np.float64)
            elif not (np.array_equal(ti, ti0) and np.array_equal(tj, tj0)):
                raise ValueError(f"group {a},{b}: target indices differ between chunks")
            idx = (z["days"].astype("datetime64[D]") - d0).astype(int)
            keep = (idx >= 0) & (idx < nd)
            S[idx[keep]] += z["sums"][keep]
            C[idx[keep]] += z["counts"][keep]
        if (C != 24).any():
            bad = np.flatnonzero(C != 24)
            raise AssertionError(f"group {a},{b}: {len(bad)} days lack 24 hours, first {days[bad[0]].date()} has {C[bad[0]]}")
        arr[:, :, ti0[:, None], tj0[None, :]] = (S / 24.0).astype(np.float32)
        filled[np.ix_(ti0, tj0)] += 1
    if not (filled == 1).all():
        raise AssertionError(f"cells covered 0 times: {int((filled == 0).sum())}, more than once: {int((filled > 1).sum())}")
    arr.flush()
    del arr
    os.replace(tmp, out_path)
