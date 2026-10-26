"""Find Copernicus Marine wind datasets: ids, versions, time coverage, variables. Metadata only."""
import pandas as pd
import copernicusmarine

print("searching the catalogue for 'wind_glo_phy' ...")
try:
    cat = copernicusmarine.describe(contains=["wind_glo_phy"], disable_progress_bar=True)
except Exception as e:
    raise SystemExit("describe failed: " + repr(e)[:300])
print("products found:", len(cat.products))
for prod in cat.products:
    print("=" * 72)
    print("product:", getattr(prod, "product_id", "?"), "|", str(getattr(prod, "title", "?"))[:110])
    for ds in prod.datasets:
        print("  dataset:", getattr(ds, "dataset_id", "?"))
        for v in ds.versions:
            for part in v.parts:
                for cid, (coord, _vars, _svcs) in part.get_coordinates().items():
                    if str(cid).lower() == "time" or getattr(coord, "axis", None) == "t":
                        try:
                            t0 = pd.to_datetime(coord.minimum_value, unit="ms").date()
                            t1 = pd.to_datetime(coord.maximum_value, unit="ms").date()
                        except Exception:
                            t0, t1 = coord.minimum_value, coord.maximum_value
                        print(f"     version {v.label} | time {t0} to {t1} | step {getattr(coord, 'step', None)}")
                for svc in part.services:
                    names = [x.short_name for x in getattr(svc, "variables", [])]
                    if names:
                        print("     variables:", names[:14])
                        break
