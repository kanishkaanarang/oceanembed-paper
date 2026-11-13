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
