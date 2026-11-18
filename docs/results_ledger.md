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

## Prospective test_c (2026-01-01 .. 2026-05-15, 135 days; pre-registered in configs/preregistration_test_c.md)
Sources: results/scoring/test_c_sums.csv, test_c_daily.csv, bootstrap_ci_test_c.csv, argo_test_c_rerun_*_rows.csv.gz, argo_bootstrap_ci_test_c.csv.
Order of events (all in git): pre-registration 9869230 (2026-11-15 16:29:57 IST, before any 2026 file was downloaded); wiring 3f2506b; predictions made after wiring; GLORYS scoring opened once at 3f2506b (lock opened_test_c.json, 2026-11-16 17:21:48); GLORYS bootstrap 25300fc; Argo bootstrap 0e0fd8a. Models frozen at 3826a32 (ensemble) and b562d0f (demeaned); no frozen file was modified.
Disclosure: the first Argo scoring invocation (25300fc) returned 0 rows because the Argo-vs-GLORYS pair file had no 2026 rows; no number was produced. Fixed in 304e3e3 (separate 2026 pair file); scoring rerun with RERUN=1 (logged). Products: GLORYS12 v202311 (interim production in 2026), DUACS v202411, OSTIA REP v202003. Window ends at the DUACS end date; winter to pre-monsoon only.
- Pooled deep-set RMSE vs GLORYS (C): clim 0.9064; linear 0.6699; XGBoost 0.6389; single U-Net 0.6419; frozen ensemble 0.6369; demeaned 0.7052.
- Frozen vs linear (GLORYS): all depths -4.9% [-7.2,-3.0]; 50-200 m -7.9% [-9.8,-6.2]; 100-150 m -9.7% [-11.9,-8.0]; 300-1000 m -1.5% [-2.3,-0.6] (-0.0057 C); 0-30 m WORSE +10.9% [+6.2,+15.8].
- Frozen vs XGBoost (GLORYS): all depths -0.3% [-1.7,+1.8] n.s. (tie); 50-200 m -2.2% [-2.8,-0.5]; 100-150 m -7.2% [-7.7,-6.0]; 0-30 m +11.2% [-0.6,+21.0] n.s. Frozen vs single U-Net: all depths -0.8% [-1.6,+0.1] n.s.
- Demeaned vs frozen (GLORYS): all depths +10.7% [+3.5,+17.0] (worse); demeaned vs linear all depths +5.3% [-0.5,+10.4] n.s.
- vs Argo (common set, 507 D+A profiles of 1,633 in window: D 204, A 303, R 1,126 dropped; 6,895 rows): pooled RMSE clim 0.817; GLORYS 0.711; linear 0.831; XGBoost 0.777; single U-Net 0.779; frozen 0.773; demeaned 0.649. Thermocline 75-200 m (2,508 rows): clim 1.195; GLORYS 1.056; linear 1.256; frozen 1.127; demeaned 0.947; bias GLORYS +0.403, frozen +0.607, linear +0.771.
- Argo intervals: frozen vs linear all depths -7.0% [-10.9,-3.5]; 75-200 m -10.3% [-14.3,-6.7]; 5-50 m WORSE +8.8% [+4.2,+14.7]; frozen vs XGBoost all depths -0.4% [-2.1,+1.5] n.s., 75-200 m -4.1% [-5.0,-2.7], 5-50 m WORSE +19.2% [+9.1,+36.6]; demeaned vs frozen all depths -16.1% [-20.1,-14.4], 5-50 m -18.8% [-27.7,-13.2], 75-200 m -16.0% [-19.7,-14.6]; 300-1000 m frozen vs linear +2.0% [+0.3,+3.9] (+0.0053 C).
- Demeaned vs GLORYS vs Argo: all depths -8.8% [-16.7,-0.2]; 75-200 m -10.3% [-18.2,-2.8]; 5-50 m +0.9% [-8.4,+13.6] n.s.; frozen vs GLORYS +8.8% [+3.7,+17.4].
- Pre-registered predictions: H1 (frozen beats linear, a few percent at most) holds in direction, size -4.9% is about twice test_a/test_b (-1.8%, -2.4%); H2 (demeaned beats frozen vs Argo) holds; H3 (no practical skill below 300 m, <=0.01 C) holds; H4 (U-Net not better than linear at 0-10 m) holds (0-30 m +10.9% worse).
- Replicates in direction on test_a, test_b and test_c: frozen beats linear vs GLORYS; U-Net worse than linear near the surface; demeaned worse vs GLORYS but better vs Argo; no practical skill below 300 m; model ranking changes with the reference.
- Claim limits: about 4.5 date blocks and 507 Argo profiles, intervals cover day-sampling variability only; one season (Jan-May), no regional or seasonal analysis done for test_c; XGBoost vs frozen at single depths (e.g. 50 and 75 m) was seen only in a printed table, not tested with intervals, do not use; do NOT claim the demeaned model beats GLORYS (interval barely excludes 0, one season, GLORYS interim, demeaned gain is bias not pattern); 300-1000 m "models closer to Argo than GLORYS" reflects GLORYS error at depth in this period, not model skill; magnitudes are not comparable across periods without a reason.

## Exploratory: frozen vs XGBoost at 30-125 m (NOT pre-registered; vs GLORYS, deep set; scripts/depth_profile_ci_test_c.py, depth_profile_xgb_test_ab.py, jan_may_xgb_by_year.py)
- test_c per depth (95% block bootstrap, frozen vs XGBoost, + = frozen worse): 30 m +19.2% [+13.1,+26.5]; 50 m +22.1% [+17.2,+30.6]; 75 m +10.5% [+7.6,+16.2]; 100 m -2.7% [-4.5,-0.5]; 125 m -9.5% [-10.3,-8.2]; 150 m -9.2% [-10.7,-6.7]; 200 m -5.1% [-6.0,-2.5].
- Same comparison on the full test_a and test_b periods: 30-75 m within a few percent (test_a 50 m +2.7% [+0.3,+6.2]; test_b all n.s.); 100-200 m frozen better in both (test_a -2.3 to -3.8%; test_b -3.8 to -4.7% at 125-200 m); 500-1000 m frozen better by about 1-2.5% (<0.01 C).
- Jan 1 - May 15 of each year, XGBoost better (+) at 30 m / 50 m: 2020 (from Mar 2) +12.3 / +6.3; 2021 +18.5 / +14.8; 2022 +16.5 / +15.6; 2023 +6.0 / +11.4; 2024 -14.2 / -6.6; 2025 +6.6 / +7.7; 2026 +19.2 / +22.1. At 75 m: -4.6, +1.4, +5.9, +6.5, -6.4, -2.2, +10.5.
- Reading: the 30-50 m disadvantage of the U-Net ensemble is mostly a Jan-May (seasonal) effect, present in 6 of 7 years, reversed in 2024; the full-period ties on test_a/test_b hid it. Hypothesis, untested: XGBoost sees day of year, latitude and longitude, the U-Nets see only SST/SLA anomalies and the mask (no season input).
- Claim limits: single-year point estimates, no intervals; exploratory, chosen after seeing test_c; do not report as a significant seasonal effect; do not claim a cause; models are frozen, no retraining with a season input is allowed (all test periods opened). Safe wording: "XGBoost, which receives the day of year, was better than the ensemble at 30-50 m in January-May of six of seven years (point estimates)".

## Argo-calibrated additive correction (main analysis pre-registered in configs/preregistration_argo_correction.md, commit 035f117; extension added after seeing it, commit 3d1ed25; scripts/argo_correction.py, scripts/argo_correction_all.py; results/scoring/argo_correction_ci.csv, argo_correction_all_ci.csv)
Rule: per-depth mean of (model - Argo) on validation (2018-02..2020-01) D+A deep-set rows, 14 levels (5-1000 m), none at 0 m; subtracted from predictions. Constants (C, frozen ensemble): 5 m -0.002; 10 m 0.029; 20 m 0.140; 30 m 0.226; 50 m 0.292; 75 m 0.385; 100 m 0.501; 125 m 0.510; 150 m 0.415; 200 m 0.257; 300 m 0.126; 500 m 0.032; 700 m 0.023; 1000 m 0.090 (7,168-8,726 rows per depth); demeaned constants within 0.04 C of these. Bootstrap: 30-day blocks, 2000 resamples, seed 0 per period, constants held fixed.
- Main analysis, corrected frozen vs uncorrected frozen vs Argo, all depths: Test A -9.3% [-11.8,-7.8]; Test B -7.2% [-8.4,-6.2]; Test C -14.6% [-16.3,-14.1]. 75-200 m: -11.0% [-13.5,-9.3], -9.1% [-10.8,-7.4], -14.9% [-16.6,-14.5]. 5-50 m: -5.9% [-8.6,-4.7], -4.7% [-5.7,-3.7], -15.0% [-19.0,-13.0].
- Corrected frozen vs uncorrected demeaned (all depths): +0.4% [-0.5,+1.5]; -0.6% [-2.2,+0.8]; +1.9% [+0.1,+4.9]. Corrected demeaned vs uncorrected demeaned: +5.0% [+1.4,+8.8]; +3.6% [+1.2,+5.8]; +7.6% [+4.0,+8.5] (worse: already bias-reduced).
- Cost vs GLORYS (deep set, pooled RMSE, C): frozen 0.6712 -> 0.7163 (A), 0.7285 -> 0.7847 (B), 0.6369 -> 0.6494 (C); demeaned 0.7433 -> 0.8889, 0.8098 -> 0.9529, 0.7052 -> 0.8533.
- Pre-registered verdicts: success rule (>=2 of 3 periods with interval excluding 0) met in all three; P1 holds; P2 FAILS (the correction gives about the same Argo gain as the demeaned ensemble); P3 first clause holds (GLORYS agreement worsens), second clause not tested.
- Extension (exploratory, all estimates corrected the same way, vs Argo), frozen_c vs: linear_c all depths -3.2% [-4.9,-1.9] / -2.9% [-4.0,-1.9] / -5.2% [-8.8,-2.9]; XGBoost_c -2.9% [-4.4,-1.9] / -2.2% [-3.3,-1.2] / -1.1% [-3.0,+0.5] n.s.; single U-Net_c -2.4% [-3.0,-1.5] / -2.4% [-2.8,-1.8] / -1.1% [-2.0,-0.5] (A / B / C). 75-200 m vs linear_c -4.7% [-6.5,-3.0] / -5.5% [-6.6,-4.3] / -7.3% [-11.0,-5.1]; vs XGBoost_c -4.1% [-5.5,-2.8] / -5.1% [-6.1,-3.9] / -3.1% [-4.2,-2.2]. 5-50 m vs linear_c +0.2% [-2.4,+1.5] n.s. / +0.2% [-1.5,+1.5] n.s. / +2.8% [-1.8,+8.2] n.s.; vs XGBoost_c +0.0% n.s. / +2.2% [+0.4,+3.7] / +8.2% [-0.3,+24.5] n.s.
- Same correction applied to others: linear_c vs linear all depths -9.1% [-11.7,-7.5] / -6.8% [-8.1,-5.5] / -16.1% [-17.0,-15.8]; GLORYS_c vs GLORYS -5.3% [-6.6,-4.3] / -6.1% [-6.6,-5.7] / -6.0% [-7.4,-4.8] (75-200 m -7.2% / -8.4% / -7.1%). frozen_c vs GLORYS_c all depths +18.0% [+14.1,+20.3] / +14.7% [+11.5,+17.6] / -1.1% [-6.7,+6.7] n.s.
- Reading: the near-surface deficit of the ensemble against Argo (uncorrected) is bias; after equal correction it ties linear at 5-50 m. The deep ensemble keeps a small consistent advantage over corrected linear (3-5% overall, 5-7% at 75-200 m) and a smaller one over corrected XGBoost and the single U-Net. A one-constant-per-depth correction removes 5-16% of the Argo error of any estimate, including GLORYS.
- Claim limits: exploratory extension chosen after seeing the main result; test periods were already open (not blind); Argo is assimilated by GLORYS; intervals cover day-sampling only and treat the fitted constants as fixed; do NOT claim the corrected model matches or beats GLORYS against Argo (only Test C, n.s.); do not use corrected climatology as a comparator; do not call the correction a new method (mean-bias correction against in-situ data is standard practice).
