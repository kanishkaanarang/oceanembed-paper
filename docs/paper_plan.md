# Paper plan (written 16 Nov 2026; source of truth for what the paper is)

## Question
How much does a deep network add over strong simple baselines when estimating daily subsurface ocean temperature (15 depths, 0-1000 m, North Indian Ocean) from SST and sea level, and how much does the answer depend on the reference (GLORYS vs Argo)?

## Evidence
Frozen protocol (configs/splits.yaml); test_a, test_b and a prospective test_c (2026-01-01..2026-05-15, configs/preregistration_test_c.md); baselines climatology, per-cell linear, XGBoost, single U-Net; block-bootstrap intervals; all numbers in docs/results_ledger.md.

## Three messages (one sentence each)
1. The gain over a linear model is small, confined to the thermocline, and replicates in direction on test_a, test_b and test_c.
2. The ranking of models changes with the reference; models inherit GLORYS's thermocline warm bias; the demeaned variant helps against Argo through bias, not pattern.
3. Limits: worse than linear near the surface; no practical skill below 300 m; ties XGBoost overall; XGBoost better at 30-50 m in Jan-May (exploratory, untested cause).

## Budget (about 2,000 words body; measure on the official template)
Abstract 150 | Intro + related work 350 | Data + method 450 | Results 750 | Discussion + limits 250 | Conclusion 70
Table: GLORYS and Argo x Test A/B/C. Figures: RMSE by depth (with intervals); Argo thermocline bias by year.

## Rules
No "first / novel / state of the art"; no "validated against two independent datasets"; no real-time claims; no claim that the demeaned model beats GLORYS; every number from the ledger; every citation verified by the author with a DOI.

## Checks before submission
Number-vs-ledger script; citation check; figures at print size and grayscale; fresh-eyes review; author read-through; external reader if possible; IGARSS portal items (template, page limit, AI-use disclosure, copyright, prior posting, co-author and acknowledgment decisions); clean-checkout reproduction; tag the release commit; descriptive title (not "OceanEmbed").

## Schedule (target submission-ready 22 Dec 2026; hard deadline 10 Jan 2027, verify on the portal)
By 23 Nov: Results and Methods rewritten with test_c, merged table, figures. By 30 Nov: Introduction, related work, bibliography. By 7 Dec: abstract, discussion, template assembly, cut to 4 pages. By 14 Dec: reviews and number check. By 21 Dec: read-through and fixes, freeze.
