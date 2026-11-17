# Pre-registration: Argo-calibrated additive bias correction, written 16 Nov 2026 BEFORE any corrected number is computed

## Status and disclosure
- Models are frozen (ensemble 3826a32, demeaned b562d0f). test_a, test_b and test_c have already been scored; their Argo rows and GLORYS sums have been seen. This analysis is therefore pre-specified but NOT blind to the test data. It uses only validation-window Argo to fit anything, and the paper must say so.

## Method (fixed now)
- Fit set: delayed-mode and adjusted (D+A) Argo rows of the validation window (2018-02-01..2020-01-31), depth > 0, common set (all models finite), from results/scoring/argo_validation_rows.csv.gz.
- Correction: for each ensemble (frozen, demean) and each of the 14 Argo depth levels k, c_k = mean over fit rows of (model - Argo). One additive constant per depth; no region, season, or year terms.
- Applied to predictions: corrected = model - c_k, at every deep-set cell and day (the same constant at all cells of a depth).

## Evaluation (fixed now)
- Argo: pooled and by group (5-50 m, 75-200 m, 300-1000 m) RMSE of corrected vs uncorrected on test_a, test_b and test_c (the same rows files as before), with the paired moving-block bootstrap (30-day blocks, 2000 resamples, seed 0).
- Cost vs GLORYS: computed exactly from existing per-depth sums, SSE' = SSE - 2 c_k B_k + c_k^2 n_k (B_k = bias sum, same sign convention as the sums files); no target is read again.
- Comparison: corrected frozen vs uncorrected demeaned, vs Argo.

## Criteria and predictions (author's, before computing)
- Success for the frozen ensemble: corrected Argo pooled RMSE below uncorrected, with intervals excluding zero, in at least two of the three test periods. Otherwise reported as a negative result.
- P1: corrected frozen improves against Argo in all three periods. P2: the gain is smaller than the demeaned ensemble's gain in test_b and test_c, because yearly bias drifts and a 2018-2020 constant under-corrects later years. P3: agreement with GLORYS worsens, by more where the period's bias differs from the validation bias.
- A prediction that fails is reported as failed.

## Limits to state in the paper
One additive constant per depth cannot correct patterns; Argo is assimilated by GLORYS, so a correction toward Argo moves away from the training target by construction; Argo in 2026 is thin (507 profiles); no uncertainty from the fit set (a single estimate of c_k).

## Extension (added 17 Nov 2026 AFTER seeing the main results above; exploratory, not part of the pre-registration)
- Main-analysis outcome: success rule met in all three periods; P1 held; P2 failed (corrected frozen matches the demeaned ensemble); P3 first clause held.
- Reason for the extension: a per-depth constant can be applied to any estimate, so the fair question is whether the U-Net ensemble's advantage survives when every estimate is corrected the same way.
- Method: identical rule (per-depth mean of model - Argo on validation D+A deep-set rows, no correction at 0 m) applied separately to linear, XGBoost, single U-Net, climatology and GLORYS; evaluation identical to the main analysis; compared pairs: corrected frozen vs corrected linear / XGBoost / single U-Net / GLORYS / climatology, and corrected linear vs linear, corrected GLORYS vs GLORYS. Reported as exploratory.
