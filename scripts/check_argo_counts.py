"""Argo coverage over the full domain for a few months: profiles, depth reached, 1-degree cell coverage, data modes. Counts only; no model data."""
import time
import numpy as np
from argopy import DataFetcher

months = [("2019-03-01", "2019-04-01"), ("2022-03-01", "2022-04-01"),
          ("2025-03-01", "2025-04-01"), ("2025-09-01", "2025-10-01")]
for t0_, t1_ in months:
    t0 = time.time()
    try:
        df = DataFetcher(src="erddap").region([45, 105, 5, 30, 0, 1000, t0_, t1_]).to_dataframe()
    except Exception as e:
        print(t0_, "FETCH FAILED:", type(e).__name__, str(e)[:300])
        continue
    g = df.groupby(["PLATFORM_NUMBER", "CYCLE_NUMBER"])
    prof = g.agg(lat=("LATITUDE", "first"), lon=("LONGITUDE", "first"), pmax=("PRES", "max"),
                 mode=("DATA_MODE", "first"))
    cells = prof.assign(ci=np.floor(prof["lat"]).astype(int), cj=np.floor(prof["lon"]).astype(int))
    per_cell = cells.groupby(["ci", "cj"]).size()
    print(f"{t0_[:7]} | {round(time.time() - t0)} s | rows {len(df)} | profiles {len(prof)} | "
          f"reach>=900 dbar {int((prof.pmax >= 900).sum())} | east of 75E {int((prof.lon > 75).sum())} | "
          f"1deg cells with >=1: {len(per_cell)}, >=3: {int((per_cell >= 3).sum())} of 1500 | "
          f"modes {prof['mode'].value_counts().to_dict()}")
