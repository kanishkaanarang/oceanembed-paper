"""Self-test of oceanembed.dataset: date alignment, input and target sanity, and a reproduction of the
climatology baseline from this code (validation window only)."""
import time

import numpy as np
import pandas as pd

from oceanembed.dataset import CHANNELS, END, START, Data

D = Data()
tr, va = D.period("train"), D.period("validation")
print("train %s to %s: %d days | validation %s to %s: %d days" % (tr[0].date(), tr[-1].date(), len(tr),
                                                                    va[0].date(), va[-1].date(), len(va)))
try:
    D.period("test_a")
    print("ERROR: test period reachable")
except PermissionError:
    print("test periods are not reachable from the data module: OK")

print("\n1. date alignment")
first = pd.Timestamp("2010-02-04")
print("   input index of 2010-02-04 is %d (GLORYS starts that day) | of 2025-12-31 is %d" % ((first - START).days, (END - START).days))
g1, g2 = D.glorys(first), D.glorys(END)
print("   GLORYS valid entries equal the mask on both ends:", bool((np.isfinite(g1) == D.mask).all() and (np.isfinite(g2) == D.mask).all()))
try:
    D.glorys(pd.Timestamp("2010-02-01"))
    print("   ERROR: GLORYS should not exist for 2010-02-01")
except KeyError:
    print("   GLORYS correctly absent for 2010-02-01 to 2010-02-03")

print("\n2. climatology (training only) and input statistics (training days only)")
t = time.time()
beta = D.fit_climatology()
print("   climatology ready in %.0f s | coefficients %s" % (time.time() - t, beta.shape))
st = D.fit_norm()
for c in CHANNELS:
    print("   %-7s mean %8.3f | std %6.3f" % (c, st[c]["mean"], st[c]["std"]))

print("\n3. reproduce the climatology baseline on the validation window (anomaly RMS per depth, deep set)")
t = time.time()
sse, bias, n = np.zeros(15), np.zeros(15), 0
for day in va:
    a = D.target(day)[:, D.deep]
    sse += (a ** 2).sum(axis=1)
    bias += a.sum(axis=1)
    n += a.shape[1]
rms = np.sqrt(sse / n)
base = pd.read_csv("results/climatology_baseline_validation.csv")
diff = np.abs(rms - base["val_rmse"].values)
print("   depth_m  this_code  baseline_csv   mean anomaly (target minus climatology)")
for k in range(15):
    print("   %6d   %8.3f   %10.3f   %+8.3f" % (base["depth_m"][k], rms[k], base["val_rmse"][k], bias[k] / n))
print("   largest difference: %.4f C | %.0f s" % (diff.max(), time.time() - t))

print("\n4. inputs on three sample days: ocean-cell values after normalisation, and missing-value counts")
for day in [pd.Timestamp("2012-07-15"), pd.Timestamp("2018-06-01"), pd.Timestamp("2019-12-31")]:
    x = D.inputs(day)
    i = int((day - START).days)
    miss = {"sst": ~np.isfinite(np.asarray(D.sst[i])), "sla": ~np.isfinite(np.asarray(D.sla[i])),
            "wind_u": ~np.isfinite(np.asarray(D.wind[i, 0])), "wind_v": ~np.isfinite(np.asarray(D.wind[i, 1]))}
    parts = []
    for c, name in enumerate(CHANNELS):
        parts.append("%s %+.2f/%.2f (missing %d)" % (name, x[c][D.ocean].mean(), x[c][D.ocean].std(), int((miss[name] & D.ocean).sum())))
    print("  ", day.date(), "|", " | ".join(parts), "| any NaN:", bool(np.isnan(x).any()))

ok = diff.max() < 0.01
print("\nALL CHECKS PASSED" if ok else "\nCHECK FAILED: the baseline is not reproduced; do not train yet")
