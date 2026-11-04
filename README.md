# OceanEmbed

Estimating daily subsurface ocean temperature (0 to 1000 m) in the North Indian Ocean from surface satellite fields.

**Status (3 November 2026): research in progress. Not published, not submitted, and no test-period results exist.** A paper for IGARSS 2027 is in preparation (submission planned for late December 2026; acceptance is not guaranteed).

## The question

How much of the daily subsurface temperature of the Arabian Sea, the Bay of Bengal and the Andaman Sea can be recovered from variables that satellites observe at the surface, and where does it fail? Models are trained against the GLORYS12 reanalysis and compared with simple baselines under a fixed chronological evaluation.

- Domain: 45 to 105 E, 5 to 30 N, 0.25 degree grid (101 x 241 cells)
- Depths (m): 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000
- Period: daily, 2010-02-01 to 2025-12-31 (the target starts on 2010-02-04)
- Caveat: GLORYS is a model reanalysis, so a model trained on it reproduces GLORYS, not necessarily the real ocean.

## Data (all from Copernicus Marine)

| Role | Product | Version | Processing onto the 0.25 degree grid |
|---|---|---|---|
| Target | GLORYS12V1, cmems_mod_glo_phy_my_0.083deg_P1D-m | 202311 | exact grid points; linear interpolation in depth to 15 levels |
| Input | OSTIA reprocessed SST, METOFFICE-GLO-SST-L4-REP-OBS-SST (product SST_GLO_SST_L4_REP_OBSERVATIONS_010_011) | 202003 | exact area-weighted mean, kelvin to Celsius |
| Input | Sea level anomaly, cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D | 202411 | exact 2 x 2 area mean |
| Input | Hourly winds, cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H (product WIND_GLO_PHY_L4_MY_012_006) | 202211 | exact 2 x 2 area mean; daily mean of hourly values |
| Input | Sea surface salinity | pending | the tested version retires on 24 Nov 2026; a replacement is awaited |

From July 2021 GLORYS is produced in an interim mode that assimilates near-real-time observations.

## Processing rules

- 0 m uses the shallowest native level (0.494 m). A value is missing wherever either bracketing native level is missing; nothing is filled or extrapolated below the seafloor.
- One reference land and seafloor mask (15 x 101 x 241). Of 12,038 ocean cells, 9,140 have values at all 15 depths (the "deep set", used for headline metrics); the other cells are shelf and shallow seas, reported separately.
- Every pipeline task is checked: time axis, grid, value range and mask. Download tasks are resumable.

## Evaluation protocol (configs/splits.yaml, fixed before any model was trained)

| Part | Period | Use |
|---|---|---|
| Train | 2010-02-04 to 2017-12-31 | fit models, normalisation and climatology |
| Validation | 2018-02-01 to 2020-01-31 | tuning and early stopping |
| Test A | 2020-03-02 to 2021-06-27 | headline result, to be evaluated once |
| Test B | 2021-08-01 to the end of the inputs | interim-production target, to be evaluated once |

Gaps of at least 30 days separate the parts. Statistics come from training years only, and the data module cannot return test days.

## Preliminary results (validation window only)

Pooled RMSE over the 15 depths on the deep set; skill = 1 - MSE(model) / MSE(climatology).

| Model | Inputs | RMSE (C) | Skill |
|---|---|---|---|
| Seasonal climatology | day of year | 0.966 | 0 |
| U-Net | raw SST, sea level, winds | 0.775 | +0.356 |
| Linear regression | sea level anomaly | 0.698 | +0.477 |
| Linear regression | sea level + SST anomalies | 0.674 | +0.514 |
| U-Net | SST, sea level, winds (anomalies) | 0.674 | +0.513 |
| U-Net, 3 seeds (0.668, 0.665, 0.668) | SST + sea level (anomalies) | 0.667 | +0.523 |

The U-Net's best epoch is chosen on the validation window, so its numbers are slightly optimistic, and its edge over the linear regression (about 1%) is within that bias. Test periods have not been used.

## Limitations

- The target is a model reanalysis, and no comparison with Argo has been made yet.
- The salinity input is pending. Gradient-boosted trees and per-depth, regional and seasonal analyses are still to be done.
- Similar work exists for the tropical Indian Ocean (Qi et al., 2023) and elsewhere (Chae et al., 2026). No claim of novelty is made.

## Repository layout

    configs/          frozen evaluation split
    src/oceanembed/   pipelines, assembly, data module
    scripts/          run_*.py (downloads), check_*.py (integrity checks), baselines, training, figures
    results/          small result tables and figures
    data/             not tracked (raw, interim, processed)
    environment.yml   environment.lock.yml

## Reproducing

    conda env create -f environment.yml
    conda activate oceanembed-research
    pip install -e . --no-deps

Download pipelines (resumable, finished tasks are skipped): scripts/run_glorys.py, run_surface.py, run_wind.py and run_sst.py, followed by the matching check_*.py scripts. A full run transfers hundreds of GB over several days. Copernicus Marine credentials are needed (copernicusmarine login) and are never committed.

## Use of AI tools

Claude (Anthropic) was used for pipeline code, checks and baselines, planning the split and metrics, literature search (every cited paper is verified by the author) and drafting. The author ran and checked the code and is responsible for all claims.

## Licence and citation

To be decided before any release. Not yet citable.
