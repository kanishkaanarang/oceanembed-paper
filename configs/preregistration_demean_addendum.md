# Addendum to the demeaned-model pre-registration (2026-11-12)
Written after the three same-day demeaned seeds finished (pooled validation RMSE 0.688, 0.684, 0.680 C; the non-demeaned same-day seeds average 0.667) and before the three lagged demeaned seeds finished. No test period opened.
Criterion (2) of the pre-registration (response to a uniform +5 cm SLA offset at 100 m <= 0.13 C) holds by construction for any model whose input has the domain mean removed, so it cannot discriminate between models.
Change: the demeaned ensemble and the frozen ensemble (commit 3826a32) are BOTH reported on test_a and test_b, against GLORYS and against Argo, as co-primary models. No model is designated primary from validation results. Criterion (1) is kept only as a record of the validation cost of demeaning.
This change removes a selection step; it cannot favour either model.
