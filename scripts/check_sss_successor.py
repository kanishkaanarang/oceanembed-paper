"""List the Copernicus multi-observation sea surface salinity datasets: ids, versions, time coverage, variables."""
import pandas as pd
import copernicusmarine

cat = copernicusmarine.describe(contains=["sss"], disable_progress_bar=True)
for prod in cat.products:
    for ds in prod.datasets:
        if "mob_glo_phy-sss" not in ds.dataset_id:
            continue
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
                print(f"{ds.dataset_id} | version {v.label} | part {part.name} | {info} | vars {names[:6]}")
