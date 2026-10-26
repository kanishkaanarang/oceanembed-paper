"""Assemble monthly surface-field result files into one array on disk (time, 101, 241)."""
import os

import numpy as np
import pandas as pd

from .grid import target_grid
from .surface_pipeline import month_range, task_path


def assemble_surface(product, out_path, first="2010-02", last="2025-12"):
    lat, lon = target_grid()
    months = month_range(first, last)
    missing = [m for m in months if not os.path.exists(task_path(product, m))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} month files are missing, first: {missing[:3]}")
    dates = pd.date_range(pd.Timestamp(months[0] + "-01"), pd.Timestamp(months[-1] + "-01") + pd.offsets.MonthEnd(0))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(dates), len(lat), len(lon)))
    pos = 0
    for m in months:
        z = np.load(task_path(product, m))
        n = int(z["ndays"])
        if str(z["month"]) != m or z["values"].shape[0] != n:
            raise ValueError(f"month file {m} is inconsistent")
        arr[pos:pos + n] = z["values"]
        pos += n
    if pos != len(dates):
        raise AssertionError(f"assembled {pos} days, expected {len(dates)}")
    arr.flush()
    del arr
    os.replace(tmp, out_path)
    return dates
