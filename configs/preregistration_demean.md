# Pre-registration: demeaned-sea-level model (written 2026-11-12, before any such model was trained; no test period opened)
Finding that motivates it: the frozen final model (commit 3826a32) moves its 100 m prediction by +0.645 C for a uniform +5 cm in the sea level input
(scripts/offset_sensitivity.py). Domain-mean sea level rose +0.43 cm/yr over 2010-2025 and the yearly GLORYS-minus-Argo bias tracks it.
Candidate: same architecture and training, with the daily domain-mean (deep-set cells) removed from every SLA channel (same day only). SST unchanged.
Six models: 3 seeds same-day inputs, 3 seeds lags 0/7/14/30 days; averaged like the frozen ensemble.
Decision rule, fixed now: the demeaned ensemble becomes the PRIMARY model only if (1) validation pooled RMSE vs GLORYS <= 0.6704 C and
(2) its response to a uniform +5 cm SLA offset at 100 m <= 0.13 C. Otherwise the frozen model stays primary and the demeaned ensemble is reported as secondary.
Both are evaluated once on test_a and test_b, against GLORYS and against Argo. Nothing is chosen on test results.
