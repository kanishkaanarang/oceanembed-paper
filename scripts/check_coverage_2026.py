"""Read-only: last available date of candidate products for a 2026 test window. Downloads nothing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from check_coverage import check_copernicus

CANDIDATES = {
    "GLORYS12 reprocessed (my)": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    "GLORYS12 interim (myint)": "cmems_mod_glo_phy_myint_0.083deg_P1D-m",
    "GLORYS12 analysis/forecast (anfc, operational)": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
    "DUACS SLA reprocessed (my)": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
    "DUACS SLA near-real-time (nrt)": "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
    "OSTIA SST reprocessed (REP)": "METOFFICE-GLO-SST-L4-REP-OBS-SST",
    "OSTIA SST near-real-time (NRT)": "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
}

summary = {}
for label, dataset_id in CANDIDATES.items():
    summary[label] = check_copernicus(label, dataset_id)

print("\n" + "=" * 72 + "\nSUMMARY (start -> end)")
for label, result in summary.items():
    if result is None:
        print(f"{label:50s} NOT FOUND / failed")
    else:
        print(f"{label:50s} {result[0].date()} -> {result[1].date()}")
