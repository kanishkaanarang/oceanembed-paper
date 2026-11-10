"""Describe the new multi-observation OI salinity dataset (P7D-c): coordinates (range, step) and variables."""
import pandas as pd
import copernicusmarine

cat = copernicusmarine.describe(contains=["multi-oi"], disable_progress_bar=True)
for prod in cat.products:
    for ds in prod.datasets:
        if "sal_my_multi-oi" not in ds.dataset_id:
            continue
        for v in ds.versions:
            for part in v.parts:
                print(prod.product_id, "|", ds.dataset_id, "| version", v.label, "| part", part.name)
                for cid, (coord, _v, _s) in part.get_coordinates().items():
                    lo, hi = coord.minimum_value, coord.maximum_value
                    if str(cid).lower() == "time":
                        try:
                            lo = pd.to_datetime(lo, unit="ms").date()
                            hi = pd.to_datetime(hi, unit="ms").date()
                        except Exception:
                            pass
                    print(f"  coord {cid}: {lo} to {hi}, step {getattr(coord, 'step', None)}, units {getattr(coord, 'coordinate_unit', None)}")
                for svc in part.services:
                    names = [x.short_name for x in getattr(svc, "variables", [])]
                    if names:
                        print("  variables:", names)
                        break
