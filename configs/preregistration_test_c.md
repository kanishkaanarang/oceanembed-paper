# Pre-registration: test_c (prospective period), written 15 Nov 2026 BEFORE any 2026 data is downloaded

## Status at writing
- No 2026 file for GLORYS, DUACS SLA or OSTIA SST exists in this repo or in data/. The newest local data ends 2025-12-31.
- All models are frozen and are not changed by this document: ensemble 1 (configs/final_model.yaml, commit 3826a32), demeaned ensemble (configs/final_model_demean.yaml, commit b562d0f), single U-Net seed 0 (configs/baseline_unet_seed0.yaml). Baselines (climatology, per-cell linear SLA+SST, XGBoost) are fitted on the training period only, exactly as for test_a and test_b.
- test_a and test_b have already been scored and are not used for any decision below.

## Period and products
- test_c = 2026-01-01 .. 2026-05-15 (135 days). The end is set by the last date of the DUACS sea level product.
- Same product families and versions as in training: GLORYS12 cmems_mod_glo_phy_my_0.083deg_P1D-m v202311 (interim production in 2026, as in test_b); DUACS cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D v202411; OSTIA METOFFICE-GLO-SST-L4-REP-OBS-SST v202003. Availability on 2026-11-15: GLORYS to 2026-09-22, DUACS to 2026-05-15, OSTIA to 2026-06-30.
- Near-real-time products are NOT used (different processing stream). If a version label differs from the above when downloading, stop and record it here before continuing.
- Argo: profiles for the same period via argopy, processed profile-by-profile exactly as for test_a/test_b (same code, same quality rules).

## Rules
1. Downloaded to separate files (data/interim/test_c, data/processed/*_test_c.npy). No existing file is modified.
2. Before scoring, only integrity checks are allowed (day counts, dates aligned, NaN pattern, static mask identical, units). No model output is computed before the scoring command.
3. One scoring pass with the existing scripts, under a lock file. Reruns only for code fixes, and every rerun is logged. Any analysis not listed here is labelled exploratory in the paper.
4. Results are reported whatever they are.

## Predictions (written before seeing any 2026 result)
- H1: pooled deep-set RMSE vs GLORYS of ensemble 1 is lower than that of the per-cell linear baseline, with a gain of a few percent at most.
- H2: against Argo, the demeaned ensemble has lower pooled RMSE than ensemble 1.
- H3: no practical skill below 300 m (gains of at most 0.01 C).
- H4: the U-Net is not better than linear at 0-10 m.
A prediction that fails is reported as failed.

## Metrics
Deep set (9,140 cells), pooled and per depth, vs GLORYS and vs Argo: RMSE, bias, paired RMSE differences with 30-day moving-block bootstrap intervals (2000 resamples). With only 135 days these intervals will be wide; this is stated, not hidden. Argo results are reported with profile counts; regional Argo results only where a region has at least 100 profiles.

## Known limitations
- Winter to pre-monsoon only; no monsoon months.
- Argo is assimilated by GLORYS and is not independent of the target.
- The 2026 data existed in the world before the models were frozen; the claim is that this repository had not downloaded or opened it. File timestamps and this commit's date document that.
