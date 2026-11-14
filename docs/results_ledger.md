# Results ledger (numbers copied from the scoring outputs; do not retype from memory)

## Protocol facts
- Models frozen at commit b562d0f before any test period was opened. Validation score files at freeze: 25.
- test_a GLORYS scoring opened at commit a167e6a (lock opened_test_a.json); rerun once with RERUN=1 only to add the per-day output (sums identical, max diff 0.0).
- test_b GLORYS scoring opened once at commit ee00032 (lock opened_test_b.json). Argo test scoring opened once for both periods at f43d1ea (opened_argo_*.json).
- Deep set = 9,140 cells valid at all 15 depths (headline). Shelf cells reported separately; XGBoost is deep-set only.
- Argo: profile level, nearest 0.25 degree cell and day, D+A modes, 0 m dropped. Argo is assimilated by GLORYS: NOT independent.

## Pooled RMSE on the deep set (C) - source: results/scoring/*_sums.csv, argo_*_rows.csv.gz
| reference | period | clim | linear SLA+SST | XGBoost | U-Net seed 0 | frozen ens. | demeaned ens. | GLORYS |
|---|---|---|---|---|---|---|---|---|
| GLORYS | test_a | 0.9689 | 0.6835 | 0.6823 | 0.6895 | 0.6712 | 0.7433 | - |
| GLORYS | test_b | 1.0680 | 0.7461 | 0.7409 | 0.7452 | 0.7285 | 0.8098 | - |
| Argo | test_a | 1.009 | 0.861 | 0.855 | 0.848 | 0.834 | 0.754 | 0.677 |
| Argo | test_b | 1.106 | 0.932 | 0.920 | 0.917 | 0.909 | 0.848 | 0.783 |
Validation (GLORYS): frozen 0.6504, demeaned 0.6677, linear 0.6735, XGBoost 0.6697, single U-Net 0.6679.

## Block-bootstrap intervals (30-day blocks, 2000 resamples; source results/scoring/bootstrap_ci.csv, argo_bootstrap_ci.csv)
- vs GLORYS, frozen vs linear: test_a -1.8% [-2.9,-0.7]; test_b -2.4% [-3.2,-1.6]; at 50-200 m -2.8% and -3.2%; 0-30 m a tie (intervals include 0).
- vs GLORYS, demeaned vs frozen: +10.7% and +11.2% worse (excludes 0).
- vs Argo, frozen vs linear: -3.1% [-5.3,-1.3] and -2.5% [-3.8,-1.3]; at 75-200 m -5.5% both; at 5-50 m frozen is WORSE than linear (+3.8%, +2.7%).
- vs Argo, demeaned vs frozen: -9.7% [-12.8,-7.7] and -6.7% [-8.5,-4.9].
- Intervals cover day-sampling variability only, not training randomness.

## Bias diagnostic (75-200 m vs Argo, common set, years 2018-2025; results/scoring/argo_bias_diagnostics.csv)
- centered RMSE (yearly bias removed): GLORYS 0.894, linear 1.064, frozen 1.026, demeaned 1.024 -> demeaned advantage over frozen is bias, not pattern.
- RMS of yearly biases: GLORYS 0.392, linear 0.531, frozen 0.484, demeaned 0.301; demeaned yearly bias ranges -0.298 to +0.511 (sign changes).

## Allowed claims
- "reproduces GLORYS with RMSE X on a held-out period"; "agrees with Argo profiles with RMSE Y"; ensemble beats linear at 50-200 m (GLORYS) and 75-200 m (Argo) with intervals excluding zero.
## Forbidden claims
- first / novel / state of the art; "validated against two independent datasets"; real-time claims; skill below 300 m; "demeaning removes the drift" (bias changes sign; pattern error unchanged).

## Regional and seasonal (vs GLORYS, deep set; regions fixed before test: ArabianSea 45-78E 5-30N, BayOfBengal 80-92E 5-23N, Andaman 92-100E 5-16N, Other = rest; cells 5984/2431/534/191)
Source: results/scoring/regions_*_daily.csv, regions_seasons_bootstrap.csv. Regional scoring opened once per test period at commit a4f2c3f (third read of test targets, added output only, no model change).
- Frozen vs linear, all depths: BayOfBengal -5.6% [-8.7,-2.4] (test_a), -3.8% [-5.1,-2.6] (test_b); ArabianSea -0.5% [-1.8,+0.7] n.s. (a), -1.9% [-2.8,-1.0] (b); Andaman +0.6% [-2.4,+4.2] n.s. (a), -3.4% [-5.9,-0.6] (b), at 50-200 m n.s. in both (+2.0% [-1.3,+6.4]; -2.4% [-5.3,+1.1]); Other (191 cells) -6.1% and -3.4%.
- Validation Andaman advantage (-12.5%) did NOT replicate on test. Bay of Bengal is the only region with a clear gain in all three periods.
- Demeaned vs frozen vs GLORYS: worse in every region (all significant in test_b; BayOfBengal and Other n.s. in test_a).
- Seasons (test_b only), frozen vs linear: DJF -3.7% [-5.7,-2.2]; MAM -3.5% [-5.6,-1.4]; JJA -0.9% [-2.0,+0.1] n.s.; SON -1.8% [-2.9,-0.9]. test_a seasonal intervals are NaN by construction (each season is one ~3-month stretch): do not use them.
- Argo profiles per region (unique day,lat,lon, common set): validation AS 6657 / BoB 2105 / Andaman 156 / Other 79; test_a AS 1992 / BoB 1412 / Andaman 92 / Other 139; test_b AS 7821 / BoB 2886 / Andaman 8 / Other 6.
- Claim limits: no Argo verification of Andaman skill in test_b (8 profiles); no claim that the model "works best" anywhere except where intervals exclude zero in both periods (Bay of Bengal); seasonal gain absent in JJA, mechanism untested.

## Input attribution (Integrated Gradients; scripts/integrated_gradients.py, results/ig_validation_by_depth.csv)
Setup: 19 VALIDATION days (every 40th), all six members of each ensemble, 12 midpoint steps, baseline = zero anomaly channels (land mask kept), target per depth = sum over deep-set cells of squared predicted anomaly; summary = share of absolute attribution summed over pixels. Completeness: median relative error 0.001, 90th percentile 0.004.
- Frozen ensemble: SST share 0.632 (0 m), 0.595 (20 m), 0.497 (30 m), 0.278 (50 m), 0.186 (100 m), 0.164 (150 m); SLA share 0.722 at 50 m, 0.836 at 150 m, 0.675-0.762 at 500-1000 m.
- Lag shares (lagged members): lag 0 about 0.45-0.55, lag 7 about 0.19, lag 14 about 0.13, lag 30 about 0.14.
- Demeaned ensemble: similar; slightly more SST near the surface (0.664 at 0 m) and more lag-30 weight at 20-50 m (0.178-0.194 vs 0.137-0.149).
- Claim limits: attribution is reliance, not causation or correctness; high SLA share at 500-1000 m does NOT imply skill there; sample of 19 days, domain-level sums only.

## Per-depth gain over linear (vs GLORYS, deep set; frozen ensemble vs linear; results/scoring/depth_profile_ci.csv; results/figures/fig_rmse_by_depth)
Block bootstrap (30-day blocks, 2000 resamples), RMSE change in %, 95% interval.
- test_a: WORSE than linear at 0 m +4.8% [+1.0,+8.3], 5 m +4.5% [+0.6,+8.3], 10 m +3.9% [+0.04,+7.5]; n.s. at 20-30 m; better at 50 m -2.4% [-3.8,-0.5], 75 m -5.6% [-6.8,-4.5], 100 m -3.4% [-5.3,-1.5], 125 m -1.7% [-3.1,-0.01], 150 m -1.8% [-3.2,-0.7]; n.s. at 200 and 300 m; 500-1000 m -1.0 to -1.4%.
- test_b: WORSE than linear at 0 m +3.3% [+1.1,+5.4], 5 m +2.7% [+0.3,+4.8], 10 m +2.7% [+0.5,+4.6]; n.s. at 20-30 m; better at 50 m -4.1%, 75 m -2.9%, 100 m -1.8%, 125 m -3.9%, 150 m -4.1%, 200 m -3.2% (all exclude 0); 300 m -0.6% (borderline), 500 m -2.1%, 700 m -1.4%, 1000 m -1.3%.
- Claim limits: significantly worse than linear at 0-10 m in BOTH periods (2.7-4.8% of RMSE, <= 0.02 C); the pooled 0-30 m "tie" hid this. Gain is concentrated at 50-200 m. Gains at 500-1000 m are significant but <= 0.01 C: do not present as skill.
