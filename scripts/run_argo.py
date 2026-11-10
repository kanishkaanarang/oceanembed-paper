"""Download Argo profiles (0-1100 dbar) over the domain, one file per month, 2010-02..2025-12. Resumable: finished months are skipped."""
import os
import time

import pandas as pd
from argopy import DataFetcher

OUT = "data/interim/argo"
os.makedirs(OUT, exist_ok=True)
fails = []
for m in pd.period_range("2010-02", "2025-12", freq="M"):
    path = f"{OUT}/argo_{m}.pkl.gz"
    if os.path.exists(path):
        continue
    t0 = m.start_time.strftime("%Y-%m-%d")
    t1 = (m + 1).start_time.strftime("%Y-%m-%d")
    for attempt in range(3):
        try:
            df = DataFetcher(src="erddap").region([45, 105, 5, 30, 0, 1100, t0, t1]).to_dataframe().reset_index(drop=True)
            df.to_pickle(path + ".tmp", compression="gzip")
            os.replace(path + ".tmp", path)
            n = df.groupby(["PLATFORM_NUMBER", "CYCLE_NUMBER"]).ngroups
            print(f"{m} ok | profiles {n} | rows {len(df)} | attempt {attempt + 1}", flush=True)
            break
        except Exception as e:
            print(f"{m} attempt {attempt + 1} failed: {type(e).__name__} {str(e)[:200]}", flush=True)
            time.sleep(20)
    else:
        fails.append(str(m))
print("DONE. failed months:", fails, flush=True)
