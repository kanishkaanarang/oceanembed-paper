"""Hourly Copernicus L4 wind (timeseries service) -> daily sums and counts on the 0.25-degree grid.
One task = one group of G x G tiles (each 16 x 16 native cells) for one time chunk (18,567 hours).
Each target cell is the exact 2x2 area mean of native 0.125-degree cells (centres offset by half a cell);
hours are summed per UTC day. Chunk boundaries do not coincide with midnight, so sums and hour counts are
stored and partial days are merged when assembling. Finished tasks are skipped."""
import os
import shutil
import time

import numpy as np
import pandas as pd
import xarray as xr

from .glorys_pipeline import download_with_retries
from .grid import target_grid

DATASET = "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H"
VARS = ["eastward_wind", "northward_wind"]
LAT0, LON0, CELL = -89.9375, -179.9375, 0.125
HALF = CELL / 2
T0 = pd.Timestamp("2007-01-11 00:00:00")
TCH = 18567
CH, G = 16, 2
FIRST_DAY, LAST_DAY = pd.Timestamp("2010-02-01"), pd.Timestamp("2025-12-31")
INTERIM = "data/interim/wind"
TMP_ROOT = "data/raw/wind_tmp"


def chunk_range(k):
    start = T0 + pd.Timedelta(hours=k * TCH)
    return start, start + pd.Timedelta(hours=TCH - 1)


def needed_chunks():
    ks, k = [], 0
    while True:
        start, end = chunk_range(k)
        if start > LAST_DAY + pd.Timedelta(hours=23):
            break
        if end >= FIRST_DAY:
            ks.append(k)
        k += 1
    return ks


def target_native_index():
    """Index of the lower-left native cell of each target point's 2x2 block."""
    lat, lon = target_grid()
    i_lo = np.rint((lat - HALF - LAT0) / CELL).astype(int)
    j_lo = np.rint((lon - HALF - LON0) / CELL).astype(int)
    if not (np.allclose(LAT0 + i_lo * CELL + HALF, lat, atol=1e-6)
            and np.allclose(LON0 + j_lo * CELL + HALF, lon, atol=1e-6)):
        raise ValueError("target points are not midpoints of native cells")
    return i_lo, j_lo


def group_targets(a, b):
    i_lo, j_lo = target_native_index()
    ti = np.where(i_lo // (G * CH) == a)[0]
    tj = np.where(j_lo // (G * CH) == b)[0]
    return i_lo, j_lo, ti, tj


def make_tasks():
    i_lo, j_lo = target_native_index()
    groups_i = np.unique(i_lo // (G * CH))
    groups_j = np.unique(j_lo // (G * CH))
    return [(int(a), int(b), int(k)) for k in needed_chunks() for a in groups_i for b in groups_j]


def task_path(a, b, k):
    return f"{INTERIM}/chunk{k}/g_{a}_{b}.npz"


def run_task(a, b, k):
    t0 = time.time()

    def result(status, message=""):
        return {"a": a, "b": b, "k": k, "status": status, "seconds": time.time() - t0, "message": message}

    out_path = task_path(a, b, k)
    if os.path.exists(out_path):
        return result("skipped")
    tmp_dir = os.path.join(TMP_ROOT, f"k{k}_g{a}_{b}")
    try:
        start, end = chunk_range(k)
        i_lo, j_lo, ti, tj = group_targets(a, b)
        imin, imax = int(i_lo[ti].min()), int(i_lo[ti].max()) + 1
        jmin, jmax = int(j_lo[tj].min()), int(j_lo[tj].max()) + 1
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir)
        cmd = [
            "copernicusmarine", "subset", "--dataset-id", DATASET, "--service", "timeseries",
            "--variable", VARS[0], "--variable", VARS[1],
            "--minimum-longitude", f"{LON0 + jmin * CELL - 0.01:.4f}",
            "--maximum-longitude", f"{LON0 + jmax * CELL + 0.01:.4f}",
            "--minimum-latitude", f"{LAT0 + imin * CELL - 0.01:.4f}",
            "--maximum-latitude", f"{LAT0 + imax * CELL + 0.01:.4f}",
            "--start-datetime", start.strftime("%Y-%m-%dT%H:%M:%S"),
            "--end-datetime", end.strftime("%Y-%m-%dT%H:%M:%S"),
            "--output-directory", tmp_dir, "--output-filename", "raw.nc",
        ]
        ok, msg = download_with_retries(cmd, timeout=2400)
        if not ok:
            return result("failed", "download: " + msg)

        ds = xr.open_dataset(os.path.join(tmp_dir, "raw.nc"))
        times = pd.DatetimeIndex(ds["time"].values)
        if len(times) != TCH or times[0] != start or times[-1] != end:
            raise ValueError(f"time range {times[0]} .. {times[-1]} ({len(times)} steps) does not match "
                             f"{start} .. {end} ({TCH} steps)")
        if not np.all(np.diff(times.values) == np.timedelta64(1, "h")):
            raise ValueError("time axis is not hourly")
        exp_lat = LAT0 + np.arange(imin, imax + 1) * CELL
        exp_lon = LON0 + np.arange(jmin, jmax + 1) * CELL
        if (len(ds["latitude"]) != len(exp_lat) or len(ds["longitude"]) != len(exp_lon)
                or not np.allclose(ds["latitude"].values, exp_lat, atol=1e-3)
                or not np.allclose(ds["longitude"].values, exp_lon, atol=1e-3)):
            raise ValueError("downloaded grid differs from the expected native cells")

        days = times.values.astype("datetime64[D]")
        starts = np.r_[0, np.flatnonzero(np.diff(days.astype("int64")) != 0) + 1]
        counts = np.diff(np.r_[starts, len(times)])
        if (counts[1:-1] != 24).any():
            raise ValueError("an interior day does not have 24 hours")
        rows, cols = (i_lo[ti] - imin)[:, None], (j_lo[tj] - jmin)[None, :]
        sums = np.empty((len(starts), 2, len(ti), len(tj)), dtype=np.float32)
        for vi, v in enumerate(VARS):
            arr = ds[v].transpose("time", "latitude", "longitude").values.astype(np.float64)
            if not np.isfinite(arr).all():
                raise ValueError(f"{v} has non-finite values")
            if np.abs(arr).max() > 60:
                raise ValueError(f"{v} has values above 60 m/s")
            S = np.add.reduceat(arr, starts, axis=0)
            sums[:, vi] = (S[:, rows, cols] + S[:, rows + 1, cols]
                           + S[:, rows, cols + 1] + S[:, rows + 1, cols + 1]) / 4.0
            del arr, S
        ds.close()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp_out = out_path.replace(".npz", ".tmp.npz")
        np.savez(tmp_out, sums=sums, counts=counts.astype(np.int16), days=days[starts], ti=ti, tj=tj)
        os.replace(tmp_out, out_path)
        return result("done", f"{len(starts)} days")
    except Exception as e:
        return result("failed", repr(e)[:300])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
