"""Assemble the per-task GLORYS result files of one time block into one array on disk."""
import os

import numpy as np

from .glorys_pipeline import block_dates, make_tasks, task_path
from .grid import target_grid


def assemble_block(block, out_path):
    lat, lon = target_grid()
    start, end = block_dates(block)
    ndays = (end - start).days + 1
    tasks = make_tasks()
    missing = [t for t in tasks if not os.path.exists(task_path(block, *t))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(tasks)} task files are missing, first: {missing[:3]}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(ndays, 15, len(lat), len(lon)))
    filled = np.zeros((len(lat), len(lon)), dtype=int)
    for a, b in tasks:
        z = np.load(task_path(block, a, b))
        if int(z["ndays"]) != ndays or str(z["start"]) != str(start.date()):
            raise ValueError(f"task {a},{b}: wrong time axis ({z['start']}, {z['ndays']} days)")
        i, j = z["lat_idx"], z["lon_idx"]
        arr[:, :, i[:, None], j[None, :]] = z["values"]
        filled[np.ix_(i, j)] += 1
    if not (filled == 1).all():
        raise AssertionError(f"cells covered 0 times: {int((filled == 0).sum())}, more than once: {int((filled > 1).sum())}")
    arr.flush()
    del arr
    os.replace(tmp, out_path)
    return out_path
