"""Monthly pipeline for OSTIA reprocessed SST (0.05 degree, centres offset by 0.025) -> the 0.25-degree grid.
Each target cell is the exact area-weighted mean of the native cells it overlaps. A value is kept if at least
half of the cell's area is covered by valid native cells. Kelvin is converted to Celsius.
One task = one calendar month; finished tasks are skipped."""
import os
import shutil
import time

import numpy as np
import pandas as pd
import xarray as xr

from .glorys_pipeline import download_with_retries
from .grid import target_grid

DATASET, VAR = "METOFFICE-GLO-SST-L4-REP-OBS-SST", "analysed_sst"
INTERIM = "data/interim/sst"
TMP_ROOT = "data/raw/sst_tmp"
MARGIN = (4.8, 30.2, 44.8, 105.2)
MASK_PATH = "data/processed/target_mask_15depths.npy"
MIN_COVER = 0.5


def month_range(first="2010-02", last="2025-12"):
    return [p.strftime("%Y-%m") for p in pd.period_range(first, last, freq="M")]


def task_path(month):
    return f"{INTERIM}/{month}.npz"


def axis_weights(centres, targets, half_target=0.125):
    centres = np.asarray(centres, dtype=float)
    h = float(np.median(np.diff(centres))) / 2
    W = np.zeros((len(targets), len(centres)))
    for i, tg in enumerate(targets):
        lo = np.maximum(centres - h, tg - half_target)
        hi = np.minimum(centres + h, tg + half_target)
        W[i] = np.clip(hi - lo, 0, None) / (2 * h)
    return W


def regrid_days(arr, lat_c, lon_c):
    """arr: (days, nlat, nlon) in Celsius, NaN on land and lakes -> (days, 101, 241) float32."""
    lat, lon = target_grid()
    Wlat, Wlon = axis_weights(lat_c, lat), axis_weights(lon_c, lon)
    full = Wlat.sum(1)[:, None] * Wlon.sum(1)[None, :]
    out = np.full((arr.shape[0], len(lat), len(lon)), np.nan, dtype=np.float32)
    for d in range(arr.shape[0]):
        a = arr[d]
        v = np.isfinite(a)
        num = Wlat @ np.where(v, a, 0.0) @ Wlon.T
        den = Wlat @ v.astype(float) @ Wlon.T
        ok = (den / full) >= MIN_COVER
        o = out[d]
        o[ok] = (num[ok] / den[ok]).astype(np.float32)
    return out


def run_task(month):
    t0 = time.time()

    def result(status, message=""):
        return {"month": month, "status": status, "seconds": time.time() - t0, "message": message}

    out_path = task_path(month)
    if os.path.exists(out_path):
        return result("skipped")
    tmp_dir = os.path.join(TMP_ROOT, month)
    try:
        first = pd.Timestamp(month + "-01")
        last = first + pd.offsets.MonthEnd(0)
        ndays = (last - first).days + 1
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir)
        lat_min, lat_max, lon_min, lon_max = MARGIN
        cmd = [
            "copernicusmarine", "subset", "--dataset-id", DATASET, "--variable", VAR,
            "--minimum-longitude", str(lon_min), "--maximum-longitude", str(lon_max),
            "--minimum-latitude", str(lat_min), "--maximum-latitude", str(lat_max),
            "--start-datetime", first.strftime("%Y-%m-%dT00:00:00"),
            "--end-datetime", last.strftime("%Y-%m-%dT23:59:59"),
            "--output-directory", tmp_dir, "--output-filename", "raw.nc",
        ]
        ok, msg = download_with_retries(cmd, timeout=3000)
        if not ok:
            return result("failed", "download: " + msg)
        ds = xr.open_dataset(os.path.join(tmp_dir, "raw.nc"))
        t = pd.DatetimeIndex(ds["time"].values).normalize()
        if len(t) != ndays or t[0] != first or t[-1] != last:
            raise ValueError(f"time range {t[0]} .. {t[-1]} ({len(t)} steps) does not match {first.date()} .. {last.date()}")
        if ndays > 1 and not np.all(np.diff(t.values) == np.timedelta64(1, "D")):
            raise ValueError("time axis is not consecutive daily steps")
        arr = ds[VAR].transpose("time", "latitude", "longitude").values.astype(np.float64)
        lat_c, lon_c = ds["latitude"].values.astype(np.float64), ds["longitude"].values.astype(np.float64)
        ds.close()
        if np.nanmean(arr) > 200:
            arr = arr - 273.15
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if lo < -3 or hi > 40:
            raise ValueError(f"values outside -3..40 C: {lo:.2f} .. {hi:.2f}")
        out = regrid_days(arr, lat_c, lon_c)
        deep = np.load(MASK_PATH).all(axis=0)
        miss = int((~np.isfinite(out[:, deep])).sum())
        if miss:
            raise AssertionError(f"{miss} deep-set values are missing")
        os.makedirs(INTERIM, exist_ok=True)
        tmp_out = out_path.replace(".npz", ".tmp.npz")
        np.savez(tmp_out, values=out, month=month, ndays=ndays)
        os.replace(tmp_out, out_path)
        return result("done", f"range {lo:.1f}..{hi:.1f} C")
    except Exception as e:
        return result("failed", repr(e)[:300])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
