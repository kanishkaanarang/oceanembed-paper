"""Argo plan-B smoke test: fetch one month of profiles in a small Arabian Sea box and report what comes back."""
import time
import argopy
from argopy import DataFetcher

print("argopy", argopy.__version__)
box = [60, 70, 10, 20, 0, 1000, "2022-03-01", "2022-04-01"]  # lon, lat, pressure (dbar), dates
t0 = time.time()
try:
    df = DataFetcher(src="erddap").region(box).to_dataframe()
except Exception as e:
    print("FETCH FAILED:", type(e).__name__, str(e)[:500])
    raise SystemExit(1)
print("seconds:", round(time.time() - t0, 1))
print("rows:", len(df))
print("columns:", list(df.columns))
print("profiles:", df.groupby(["PLATFORM_NUMBER", "CYCLE_NUMBER"]).ngroups)
print("PRES min/max:", df["PRES"].min(), df["PRES"].max())
for c in ("TEMP", "PSAL"):
    if c in df.columns:
        print(c, "NaN fraction:", round(df[c].isna().mean(), 4))
print(df.head(3).to_string())
