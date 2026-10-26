"""GLORYS bulk pipeline (timeseries download service).

One task = one group of G x G tiles (each tile 16 x 16 native cells) for one time block (2081 days, the time
chunk of the timeseries layout). A task downloads the group, takes the exact 0.25-degree points inside it,
interpolates to the 15 depths, checks the result against the reference mask, saves a small file and deletes
the raw download. Finished tasks are skipped, so a run can be stopped and restarted at any time.
"""
import os
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import xarray as xr

from .glorys import interpolate_depths
from .grid import target_grid

DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
LAT0, LON0, STEP = -80.0, -180.0, 1 / 12   # native grid origin (index 0) and spacing
CH = 16                                    # lat/lon chunk size of the timeseries layout
TBLOCK = 2081                              # time chunk size of the timeseries layout
T0 = pd.Timestamp("1993-01-01")
G = 2                                      # chunks per side in one request
LAST_DAY = pd.Timestamp("2025-12-31")      # never request beyond the window we need
INTERIM = "data/interim/glorys"
TMP_ROOT = "data/raw/glorys_tmp"
MASK_PATH = "data/processed/target_mask_15depths.npy"


def block_dates(block):
    start = T0 + pd.Timedelta(days=block * TBLOCK)
    end = min(start + pd.Timedelta(days=TBLOCK - 1), LAST_DAY)
    if end < start:
        raise ValueError(f"block {block} starts after {LAST_DAY.date()}")
    return start, end


def native_index(values, origin):
    """Index on the global native grid; every target point must sit exactly on it."""
    n = np.rint((np.asarray(values) - origin) / STEP).astype(int)
    if not np.allclose(origin + n * STEP, values, atol=1e-6):
        raise ValueError("target points are not on the native grid")
    return n


def make_tasks():
    lat, lon = target_grid()
    ck_lat = np.unique(native_index(lat, LAT0) // CH)
    ck_lon = np.unique(native_index(lon, LON0) // CH)
    return [(int(a), int(b)) for a in ck_lat[::G] for b in ck_lon[::G]]


def task_path(block, a, b):
    return f"{INTERIM}/block{block}/g_{a}_{b}.npz"


def _group_points(a, b):
    lat, lon = target_grid()
    n_lat, n_lon = native_index(lat, LAT0), native_index(lon, LON0)
    i_sel = np.where((n_lat // CH >= a) & (n_lat // CH < a + G))[0]
    j_sel = np.where((n_lon // CH >= b) & (n_lon // CH < b + G))[0]
    return lat, lon, i_sel, j_sel


def download_with_retries(cmd, retries=3, timeout=3600):
    last = ""
    for attempt in range(1, retries + 1):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
            if r.returncode == 0:
                return True, ""
            last = r.stderr[-300:]
        except subprocess.TimeoutExpired:
            last = f"timeout after {timeout} s"
        if attempt < retries:
            time.sleep(30 * attempt)
    return False, last


def run_task(block, a, b):
    t0 = time.time()

    def result(status, message=""):
        return {"block": block, "a": a, "b": b, "status": status,
                "seconds": time.time() - t0, "message": message}

    out_path = task_path(block, a, b)
    if os.path.exists(out_path):
        return result("skipped")
    tmp_dir = os.path.join(TMP_ROOT, f"b{block}_g{a}_{b}")
    try:
        start, end = block_dates(block)
        ndays = (end - start).days + 1
        lat, lon, i_sel, j_sel = _group_points(a, b)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir)
        cmd = [
            "copernicusmarine", "subset",
            "--dataset-id", DATASET, "--service", "timeseries", "--variable", "thetao",
            "--minimum-longitude", f"{LON0 + b * CH * STEP - 0.01:.4f}",
            "--maximum-longitude", f"{LON0 + ((b + G) * CH - 1) * STEP + 0.01:.4f}",
            "--minimum-latitude", f"{LAT0 + a * CH * STEP - 0.01:.4f}",
            "--maximum-latitude", f"{LAT0 + ((a + G) * CH - 1) * STEP + 0.01:.4f}",
            "--start-datetime", start.strftime("%Y-%m-%dT00:00:00"),
            "--end-datetime", end.strftime("%Y-%m-%dT00:00:00"),
            "--minimum-depth", "0", "--maximum-depth", "1100",
            "--output-directory", tmp_dir, "--output-filename", "raw.nc",
        ]
        ok, msg = download_with_retries(cmd)
        if not ok:
            return result("failed", "download: " + msg)

        ds = xr.open_dataset(os.path.join(tmp_dir, "raw.nc"))
        times = ds["time"].values
        if ds.sizes["time"] != ndays:
            raise ValueError(f"expected {ndays} days, got {ds.sizes['time']}")
        if pd.Timestamp(times[0]).normalize() != start or pd.Timestamp(times[-1]).normalize() != end:
            raise ValueError(f"time range {times[0]} .. {times[-1]} does not match {start.date()} .. {end.date()}")
        if ndays > 1 and not np.all(np.diff(times) == np.timedelta64(1, "D")):
            raise ValueError("time axis is not consecutive daily steps")
        levels = ds["depth"].values.astype("float64")
        da = (ds["thetao"]
              .sel(latitude=lat[i_sel], longitude=lon[j_sel], method="nearest", tolerance=1e-3)
              .transpose("time", "depth", "latitude", "longitude"))
        out = np.empty((ndays, 15, len(i_sel), len(j_sel)), dtype=np.float32)
        for s in range(0, ndays, 256):
            e = min(s + 256, ndays)
            out[s:e] = interpolate_depths(da.isel(time=slice(s, e)).values, levels)
        ds.close()

        if os.path.exists(MASK_PATH):
            ref = np.load(MASK_PATH)[:, i_sel][:, :, j_sel]
            same = (np.isfinite(out) == ref[None]).all(axis=(1, 2, 3))
            if not same.all():
                bad = np.flatnonzero(~same)
                raise AssertionError(f"validity differs from the reference mask on {len(bad)} days, first index {bad[0]}")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp_out = out_path.replace(".npz", ".tmp.npz")
        np.savez(tmp_out, values=out, lat_idx=i_sel, lon_idx=j_sel, start=str(start.date()), ndays=ndays)
        os.replace(tmp_out, out_path)
        return result("done")
    except Exception as e:
        return result("failed", repr(e)[:300])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
