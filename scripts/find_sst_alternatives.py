"""Find Copernicus Marine daily L4 SST datasets: ids, versions, time coverage, grid step. Metadata only."""
import pandas as pd
import copernicusmarine

cat = copernicusmarine.describe(contains=["sst_glo_sst_l4"], disable_progress_bar=True)
print("products found:", len(cat.products))
for prod in cat.products:
    print("=" * 72)
    print(getattr(prod, "product_id", "?"), "|", str(getattr(prod, "title", "?"))[:100])
    for ds in prod.datasets:
        did = ds.dataset_id
        for v in ds.versions:
            for part in v.parts:
                info = {}
                for cid, (coord, _v, _s) in part.get_coordinates().items():
                    c = str(cid).lower()
                    if c == "time" or getattr(coord, "axis", None) == "t":
                        try:
                            info["time"] = "%s to %s" % (pd.to_datetime(coord.minimum_value, unit="ms").date(),
                                                          pd.to_datetime(coord.maximum_value, unit="ms").date())
                        except Exception:
                            info["time"] = "%s to %s" % (coord.minimum_value, coord.maximum_value)
                    if c in ("latitude", "lat"):
                        info["lat_step"] = getattr(coord, "step", None)
                names = []
                for svc in part.services:
                    names = [x.short_name for x in getattr(svc, "variables", [])]
                    if names:
                        break
                print(f"  {did} | version {v.label} | {info} | vars {names[:6]}")
