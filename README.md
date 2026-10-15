# OceanEmbed

Reconstruction of subsurface ocean temperature in the North Indian Ocean from five surface satellite-derived variables, using a convolutional neural network.

**Status: research in preparation. Not yet submitted or published.** Numbers from earlier prototypes are not reported here, because they were measured on non-independent splits.

## Problem

Estimate temperature at 15 depths (0 to 1000 m) over the North Indian Ocean (45-105E, 5-30N) at 0.25 degree daily resolution, using only surface inputs:

- sea surface temperature (SST)
- sea surface salinity (SSS)
- sea level anomaly (SLA)
- surface currents (u, v)
- surface winds (u, v)

Training target: GLORYS ocean reanalysis. Independent validation (planned): gridded Argo.

## Evaluation protocol (planned)

- Temporal split: train 2022, validate Jan-Jun 2023, test Jul-Dec 2023. The test set is used once.
- Validation against INCOIS gridded Argo, with a potential vs in-situ temperature conversion.
- Baseline comparison against an XGBoost model with a profile-level split.

## Data sources (to be confirmed when downloaded)

- GLORYS reanalysis (Copernicus Marine)
- Sea surface salinity (Copernicus Marine)
- Sea level anomaly (Copernicus Marine)
- OSCAR surface currents (NASA)
- CCMP surface winds (NASA)
- INCOIS gridded Argo (ERDDAP)

Raw data is not stored in this repository.

## Setup (macOS, Apple Silicon)

    conda env create -f environment.yml
    conda activate oceanembed-research

`environment.lock.yml` records the exact versions used during development.

## Repository layout

    configs/     configuration files
    src/         reusable code (package: oceanembed)
    scripts/     command-line entry points
    notebooks/   exploration and figures
    results/     small result tables and figures
    paper/       manuscript sources
    literature/  reading notes and comparison table

## Reproducing results

To be added once the pipeline and evaluation are complete.

## Citation and license

To be added.
