"""One partial month of a surface product, same checks as surface_pipeline.run_task but with an explicit last day.
Example: python scripts/run_surface_partial.py --product ssh --month 2026-05 --last 2026-05-15"""
import argparse
import os
import shutil
import time

import numpy as np
import pandas as pd

from oceanembed.surface_pipeline import (MARGIN, PRODUCTS, TMP_ROOT, download_with_retries,
                                         load_month_2x2, task_path)

ap = argparse.ArgumentParser()
ap.add_argument("--product", default="ssh")
ap.add_argument("--month", required=True)
ap.add_argument("--last", required=True)
a = ap.parse_args()

out_path = task_path(a.product, a.month)
if os.path.exists(out_path):
    raise SystemExit(f"EXISTS: {out_path} (stop)")
cfg = PRODUCTS[a.product]
first = pd.Timestamp(a.month + "-01")
last = pd.Timestamp(a.last)
assert first <= last <= first + pd.offsets.MonthEnd(0), "last day must be inside the month"
ndays = (last - first).days + 1
tmp_dir = os.path.join(TMP_ROOT, f"{a.product}_{a.month}_partial")
shutil.rmtree(tmp_dir, ignore_errors=True)
os.makedirs(tmp_dir)
t0 = time.time()
try:
    lat_min, lat_max, lon_min, lon_max = MARGIN
    cmd = ["copernicusmarine", "subset", "--dataset-id", cfg["dataset_id"], "--variable", cfg["var"],
           "--minimum-longitude", str(lon_min), "--maximum-longitude", str(lon_max),
           "--minimum-latitude", str(lat_min), "--maximum-latitude", str(lat_max),
           "--start-datetime", first.strftime("%Y-%m-%dT00:00:00"),
           "--end-datetime", last.strftime("%Y-%m-%dT00:00:00"),
           "--output-directory", tmp_dir, "--output-filename", "raw.nc"]
    ok, msg = download_with_retries(cmd, timeout=1800)
    if not ok:
        raise SystemExit("download failed: " + msg)
    times, arr = load_month_2x2(os.path.join(tmp_dir, "raw.nc"), cfg["var"])
    if arr.shape[0] != ndays:
        raise ValueError(f"expected {ndays} days, got {arr.shape[0]}")
    t = pd.DatetimeIndex(times).normalize()
    if t[0] != first or t[-1] != last:
        raise ValueError(f"time range {t[0]} .. {t[-1]} does not match {first.date()} .. {last.date()}")
    if not np.all(np.diff(t.values) == np.timedelta64(1, "D")):
        raise ValueError("time axis is not consecutive daily steps")
    valid = np.isfinite(arr)
    if not all(np.array_equal(valid[0], valid[i]) for i in range(1, ndays)):
        raise AssertionError("validity pattern changes within the month")
    lo, hi = cfg["limits"]
    if np.nanmin(arr) < lo or np.nanmax(arr) > hi:
        raise ValueError(f"values outside {lo}..{hi}")
    tmp_out = out_path.replace(".npz", ".tmp.npz")
    np.savez(tmp_out, values=arr, month=a.month, ndays=ndays)
    os.replace(tmp_out, out_path)
    print(f"done {a.month} ({ndays} days) in {time.time() - t0:.0f} s | valid cells {int(valid[0].sum())}")
finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)
