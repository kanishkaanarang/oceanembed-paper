"""Is the demeaned model's Argo advantage real pattern skill or offset cancellation? Per model: pooled RMSE vs Argo, centered RMSE (each
calendar year's own mean bias removed), RMS and range of the yearly biases. Common set, validation + test windows (training years excluded).
Reads only the saved Argo row files."""
import os

import numpy as np
import pandas as pd

OUT = "results/scoring/argo_bias_diagnostics.csv"
if os.path.exists(OUT):
    raise SystemExit(f"EXISTS: {OUT} (stop)")
M = ["clim", "glorys", "lin", "xgb", "unet0", "frozen", "demean"]
r = pd.concat([pd.read_csv(f"results/scoring/argo_{p}_rows.csv.gz", parse_dates=["day"]) for p in ("validation", "test_a", "test_b")])
r = r[np.isfinite(r[M]).all(axis=1)].copy()
r["year"] = r.day.dt.year
rows = []
for gname, (lo, hi) in {"75-200 m": (75, 200), "5-50 m": (1, 50), "all depths": (1, 1000)}.items():
    g = r[(r.depth >= lo) & (r.depth <= hi)]
    for m in ["glorys", "lin", "frozen", "demean"]:
        by = g.groupby("year")["e_" + m].agg(["mean", "var", "size"])
        by["var"] = g.groupby("year")["e_" + m].var(ddof=0)
        N = by["size"].sum()
        rows.append(dict(group=gname, model=m,
                         rmse=np.sqrt((by["size"] * (by["var"] + by["mean"] ** 2)).sum() / N),
                         centered_rmse=np.sqrt((by["size"] * by["var"]).sum() / N),
                         bias_rms_over_years=np.sqrt((by["mean"] ** 2).mean()),
                         bias_min=by["mean"].min(), bias_max=by["mean"].max()))
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(out.round(3).to_string(index=False))
for gname in out["group"].unique():
    x = out[out["group"] == gname].set_index("model")
    print("\n%-10s demeaned vs frozen:  RMSE %+.1f%%   centered RMSE %+.1f%%   |  demeaned vs linear: RMSE %+.1f%%   centered RMSE %+.1f%%" % (
        gname, 100 * (x.rmse.demean / x.rmse.frozen - 1), 100 * (x.centered_rmse.demean / x.centered_rmse.frozen - 1),
        100 * (x.rmse.demean / x.rmse.lin - 1), 100 * (x.centered_rmse.demean / x.centered_rmse.lin - 1)))
