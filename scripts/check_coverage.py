"""Check time coverage, version and variables of every data product. Read-only; downloads no data."""
import json
from datetime import datetime, timezone

import copernicusmarine
import earthaccess

COPERNICUS = {
    "GLORYS target": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    "SSS": "cmems_obs-mob_glo_phy-sss_my_multi_P1D",
    "SSH": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
}
NASA = {
    "OSCAR currents": "OSCAR_L4_OC_FINAL_V2.0",
    "CCMP winds": "CCMP_WINDS_10M6HR_L4_V3.1",
}


def to_datetime(value):
    """Catalogue time value (seconds, ms or ns since 1970, ISO string, or datetime) -> UTC datetime.
    Tries units until the year is plausible."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    for divisor in (1.0, 1e3, 1e9):
        try:
            dt = datetime.fromtimestamp(float(value) / divisor, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            continue
        if 1950 <= dt.year <= 2100:
            return dt
    raise ValueError(f"cannot interpret time value {value!r}")


def is_time(coord_id, coord):
    return (
        getattr(coord, "axis", None) == "t"
        or str(coord_id).lower() == "time"
        or str(getattr(coord, "standard_name", "")).lower() == "time"
    )


def check_copernicus(label, dataset_id):
    print("=" * 72)
    print(f"{label} | {dataset_id}")
    try:
        cat = copernicusmarine.describe(dataset_id=dataset_id, disable_progress_bar=True)
    except Exception as e:
        print("  describe() failed:", repr(e))
        return None
    if not cat.products or not cat.products[0].datasets:
        print("  NOT FOUND in the catalogue: check the dataset id")
        return None
    dataset = cat.products[0].datasets[0]
    result = None
    for version in dataset.versions:
        for part in version.parts:
            print(f"  version {version.label} | part {part.name}")
            for coord_id, (coord, _vars, _svcs) in part.get_coordinates().items():
                if is_time(coord_id, coord):
                    start = to_datetime(coord.minimum_value)
                    end = to_datetime(coord.maximum_value)
                    print("    raw min/max:", coord.minimum_value, coord.maximum_value,
                          "| unit:", getattr(coord, "coordinate_unit", None))
                    print("    start:", start.date(), "| end:", end.date())
                    if result is None:
                        result = (start, end)
            for svc in part.services:
                names = [v.short_name for v in getattr(svc, "variables", [])]
                if names:
                    print("    variables:", names)
                    break
    return result


def check_nasa(label, short_name):
    print("=" * 72)
    print(f"{label} | {short_name}")
    try:
        collections = earthaccess.search_datasets(short_name=short_name)
    except Exception as e:
        print("  search failed:", repr(e))
        return
    if not collections:
        print("  NOT FOUND: check the short name")
        return
    for c in collections[:3]:
        umm = c["umm"]
        print("  title:", umm.get("EntryTitle"), "| version:", umm.get("Version"))
        print("  temporal extent (raw):", json.dumps(umm.get("TemporalExtents"))[:500])


results = {}
for label, dataset_id in COPERNICUS.items():
    results[label] = check_copernicus(label, dataset_id)
for label, short_name in NASA.items():
    check_nasa(label, short_name)

found = [r for r in results.values() if r]
if len(found) == len(COPERNICUS):
    print("=" * 72)
    print("Copernicus overlap (GLORYS, SSS, SSH):",
          max(r[0] for r in found).date(), "to", min(r[1] for r in found).date())
