"""Monthly pipeline for 2-D surface fields on the 0.125-degree half-cell-offset grid (SSH now; SSS later).
One task = one calendar month of one product: download with a margin, 2x2 area mean onto the 0.25-degree
grid, check, save, delete the raw file. Finished tasks are skipped, so a run can be stopped and restarted."""
import os
import shutil
import time

import numpy as np
import pandas as pd

from .glorys_pipeline import download_with_retries
from .surface import load_month_2x2

PRODUCTS = {
    "ssh": {"dataset_id": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
            "var": "sla", "limits": (-2.0, 2.0)},
}
INTERIM = "data/interim/surface"
TMP_ROOT = "data/raw/surface_tmp"
MARGIN = (4.8, 30.2, 44.8, 105.2)   # lat_min, lat_max, lon_min, lon_max: at least 0.0625 degrees beyond the domain


def month_range(first="2010-02", last="2025-12"):
    return [p.strftime("%Y-%m") for p in pd.period_range(first, last, freq="M")]


def task_path(product, month):
    return f"{INTERIM}/{product}/{month}.npz"


def run_task(product, month):
    t0 = time.time()

    def result(status, message=""):
        return {"product": product, "month": month, "status": status,
                "seconds": time.time() - t0, "message": message}

    out_path = task_path(product, month)
    if os.path.exists(out_path):
        return result("skipped")
    cfg = PRODUCTS[product]
    tmp_dir = os.path.join(TMP_ROOT, f"{product}_{month}")
    try:
        first = pd.Timestamp(month + "-01")
        last = first + pd.offsets.MonthEnd(0)
        ndays = (last - first).days + 1
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir)
        lat_min, lat_max, lon_min, lon_max = MARGIN
        cmd = [
            "copernicusmarine", "subset", "--dataset-id", cfg["dataset_id"], "--variable", cfg["var"],
            "--minimum-longitude", str(lon_min), "--maximum-longitude", str(lon_max),
            "--minimum-latitude", str(lat_min), "--maximum-latitude", str(lat_max),
            "--start-datetime", first.strftime("%Y-%m-%dT00:00:00"),
            "--end-datetime", last.strftime("%Y-%m-%dT00:00:00"),
            "--output-directory", tmp_dir, "--output-filename", "raw.nc",
        ]
        ok, msg = download_with_retries(cmd, timeout=1800)
        if not ok:
            return result("failed", "download: " + msg)

        times, arr = load_month_2x2(os.path.join(tmp_dir, "raw.nc"), cfg["var"])
        if arr.shape[0] != ndays:
            raise ValueError(f"expected {ndays} days, got {arr.shape[0]}")
        t = pd.DatetimeIndex(times).normalize()
        if t[0] != first or t[-1] != last:
            raise ValueError(f"time range {t[0]} .. {t[-1]} does not match {first.date()} .. {last.date()}")
        if ndays > 1 and not np.all(np.diff(t.values) == np.timedelta64(1, "D")):
            raise ValueError("time axis is not consecutive daily steps")
        valid = np.isfinite(arr)
        if not all(np.array_equal(valid[0], valid[i]) for i in range(1, ndays)):
            raise AssertionError("validity pattern changes within the month")
        lo, hi = cfg["limits"]
        if np.nanmin(arr) < lo or np.nanmax(arr) > hi:
            raise ValueError(f"values outside {lo}..{hi}: {np.nanmin(arr):.3f} .. {np.nanmax(arr):.3f}")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp_out = out_path.replace(".npz", ".tmp.npz")
        np.savez(tmp_out, values=arr, month=month, ndays=ndays)
        os.replace(tmp_out, out_path)
        return result("done", f"valid cells {int(valid[0].sum())}")
    except Exception as e:
        return result("failed", repr(e)[:300])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
