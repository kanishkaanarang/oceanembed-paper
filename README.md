# OceanEmbed

Estimating daily subsurface ocean temperature (0 to 1000 m) in the North Indian Ocean from surface satellite fields.

**Status (14 November 2026): experiments finished, paper being written. Not published and not submitted.** A paper for IGARSS 2027 is in preparation (acceptance is not guaranteed). Both models were frozen before any test period was opened, and each test period was scored under a lock file. Every number quoted below is copied from `docs/results_ledger.md`, which is the single source for the paper.

## The question

How much of the daily subsurface temperature of the Arabian Sea, the Bay of Bengal and the Andaman Sea can be recovered from variables that satellites observe at the surface, and where does it fail? Models are trained against the GLORYS12 reanalysis and compared with simple baselines under a fixed chronological evaluation, with GLORYS and with Argo profiles as references.

- Domain: 45 to 105 E, 5 to 30 N, 0.25 degree grid (101 x 241 cells)
- Depths (m): 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000
- Period: daily, 2010-02-01 to 2025-12-31 (the target starts on 2010-02-04)
- Caveat: GLORYS is a model reanalysis that assimilates Argo, so a model trained on it reproduces GLORYS, not necessarily the real ocean, and Argo is not an independent check.

## Data (all from Copernicus Marine)

| Role | Product | Version | Processing onto the 0.25 degree grid |
|---|---|---|---|
| Target | GLORYS12V1, cmems_mod_glo_phy_my_0.083deg_P1D-m | 202311 | exact grid points; linear interpolation in depth to 15 levels |
| Input | OSTIA reprocessed SST, METOFFICE-GLO-SST-L4-REP-OBS-SST (product SST_GLO_SST_L4_REP_OBSERVATIONS_010_011) | 202003 | exact area-weighted mean, kelvin to Celsius |
| Input | Sea level anomaly, cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D | 202411 | exact 2 x 2 area mean |
| Downloaded, not used | Hourly winds, WIND_GLO_PHY_L4_MY_012_006 | 202211 | winds added nothing on the validation window and are not inputs of the final models |
| Dropped | Sea surface salinity | n/a | the tested version retires in November 2026 and no replacement was adopted |
| Validation only | Argo profiles (delayed-mode and adjusted), via argopy | n/a | profile by profile, nearest 0.25 degree cell and day; potential temperature computed with gsw |

From July 2021 GLORYS is produced in an interim mode that assimilates near-real-time observations, so Test B is reported separately from the headline Test A.

## Processing rules

- 0 m uses the shallowest native level (0.494 m). A value is missing wherever either bracketing native level is missing; nothing is filled or extrapolated below the seafloor.
- One reference land and seafloor mask (15 x 101 x 241). Of 12,038 ocean cells, 9,140 have values at all 15 depths (the "deep set", used for headline metrics); the other cells are shelf and shallow seas, reported separately.
- Every pipeline task is checked: time axis, grid, value range and mask. Download tasks are resumable.

## Evaluation protocol (configs/splits.yaml, fixed before any model was trained)

| Part | Period | Use |
|---|---|---|
| Train | 2010-02-04 to 2017-12-31 | fit models, normalisation and climatology |
| Validation | 2018-02-01 to 2020-01-31 | tuning and early stopping |
| Test A | 2020-03-02 to 2021-06-27 | headline result |
| Test B | 2021-08-01 to 2025-12-31 | interim-production target, reported separately |

Gaps of at least 30 days separate the parts. Statistics come from training years only, and the data module cannot return test days. Test periods are scored only by dedicated scripts that write a lock file (`results/scoring/opened_*.json`) before reading any target and record the git commit. Validation was inspected 25 times (score files) before the models were frozen. Test A GLORYS scoring was run twice (the second time only to add a per-day output; the sums were identical). Regional scoring is a further read of the test targets for added output only, and Argo was scored once per test period. No model, input or recipe was changed after a test period was opened.

## Models

Anomalies are inputs minus their training-period harmonic seasonal cycle (three harmonics per cell), divided by a training standard deviation; the target is the GLORYS anomaly. U-Net: width 32, two down-sampling steps, GroupNorm, GELU, about 0.47 million parameters, AdamW, early stopping on validation.

- **Frozen ensemble** (`configs/final_model.yaml`): mean of six U-Nets (three seeds with same-day SST and sea level anomalies, three seeds with lags of 0, 7, 14 and 30 days).
- **Demeaned ensemble** (`configs/final_model_demean.yaml`): the same layout with the daily domain-mean sea level removed from every sea level channel. Pre-registered as co-primary with the frozen ensemble (`configs/preregistration_demean.md` and its addendum).
- **Baselines**: seasonal climatology, per-cell linear regression on sea level and SST anomalies, XGBoost (fixed settings, deep set only), and a single U-Net (seed 0).

## Results (Test A and Test B, pooled RMSE in C over 15 depths, deep set)

| Model | vs GLORYS, Test A | vs GLORYS, Test B | vs Argo, Test A | vs Argo, Test B |
|---|---|---|---|---|
| Seasonal climatology | 0.9689 | 1.0680 | 1.009 | 1.106 |
| Linear regression, SLA + SST | 0.6835 | 0.7461 | 0.861 | 0.932 |
| XGBoost | 0.6823 | 0.7409 | 0.855 | 0.920 |
| Single U-Net (seed 0) | 0.6895 | 0.7452 | 0.848 | 0.917 |
| U-Net ensemble (frozen) | **0.6712** | **0.7285** | 0.834 | 0.909 |
| U-Net ensemble, demeaned | 0.7433 | 0.8098 | **0.754** | **0.848** |
| GLORYS12 itself | n/a | n/a | 0.677 | 0.783 |

What the intervals (30-day block bootstrap, `results/scoring/*bootstrap*.csv`) support:

- Against GLORYS the frozen ensemble beats linear regression by 1.8% (Test A) and 2.4% (Test B), intervals excluding zero. The gain is concentrated at 50 to 200 m. It is **worse** than linear regression at 0 to 10 m in both periods (by 2.7 to 4.8% of RMSE, at most 0.02 C), and below 300 m no model meaningfully improves on climatology.
- By region the gain is clear only in the Bay of Bengal. The validation advantage in the Andaman Sea did not replicate, and Argo coverage there is almost absent in Test B (8 profiles).
- **The ranking flips with the reference.** Against Argo the demeaned ensemble is best of the learned models; against GLORYS it is worst. A bias diagnostic shows its advantage over the frozen ensemble in the thermocline is bias, not pattern: after removing each year's mean bias the two have the same RMSE (1.024 and 1.026 C), and its yearly bias changes sign.
- Models inherit GLORYS's thermocline warm bias against Argo, which grows to 2024.
- Integrated Gradients on validation days shows the networks use SST near the surface (63% of attribution at 0 m) and sea level below about 50 m (72% at 50 m, 83% at 150 m).

## Limitations

- The target is a reanalysis, and Argo, the only real-profile check, is assimilated by it. The paper does not claim two independent validations.
- The gain over a per-cell linear regression is small (about 2 to 3%). The intervals cover day-sampling variability, not training randomness.
- Skill is essentially zero below 300 m, and the U-Net is slightly worse than linear regression near the surface.
- Argo coverage is thin in the Andaman Sea and in 2025; seasonal intervals for Test A are undefined because it spans only 16 months.
- Similar work exists for the tropical Indian Ocean (Qi et al., 2023) and elsewhere (Chae et al., 2026). No claim of novelty is made.

## Repository layout

    configs/          frozen evaluation split, frozen model configs, pre-registration
    models/           frozen weights of both ensembles and their input statistics
    src/oceanembed/   pipelines, data modules, U-Net, model-set loader
    scripts/          run_*.py (downloads), check_*.py (integrity checks), train_*.py, baselines,
                      predict_*.py (inference), score_*.py (scoring with test locks), bootstrap_*.py,
                      integrated_gradients.py, make_fig_*.py
    docs/             results_ledger.md (single source for all numbers and allowed claims)
    results/          small result tables, scoring sums, bootstrap intervals, figures
    data/             not tracked (raw, interim, processed)
    environment.yml   environment.lock.yml

## Reproducing

    conda env create -f environment.yml
    conda activate oceanembed-research
    pip install -e . --no-deps

1. Download pipelines (resumable, finished tasks are skipped): scripts/run_glorys.py, run_surface.py, run_sst.py and run_wind.py, then the matching check_*.py scripts. A full run transfers hundreds of GB over several days. Copernicus Marine credentials are needed (copernicusmarine login) and are never committed.
2. Inference with the frozen weights: `python scripts/predict_sets.py {frozen|demean} {validation|test_a|test_b}` and `python scripts/predict_baselines.py PERIOD ...`.
3. Scoring: `scripts/score_period.py`, `score_argo.py`, `score_regions.py` (test periods are locked and need a clean git tree), then `bootstrap_ci.py`, `bootstrap_argo.py`, `bootstrap_regions_seasons.py`.

Training seeds are fixed, but exact numerical reproduction on different hardware is not guaranteed (the models were trained on an Apple-silicon GPU).

## Use of AI tools

Claude (Anthropic) was used for pipeline code, checks, baselines and scoring scripts, planning the split and metrics, literature search (every cited paper is verified by the author) and drafting. The author ran and checked the code and is responsible for all claims.

## Licence and citation

To be decided before any release. Not yet citable.
