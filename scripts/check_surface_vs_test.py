"""Pipeline output for SSH 2022-01 must equal the test file processed directly."""
import numpy as np

from oceanembed.surface import load_month_2x2

z = np.load("data/interim/surface/ssh/2022-01.npz")
_, ref = load_month_2x2("data/raw/test/ssh_2022-01.nc", "sla")
v = z["values"]
print("shapes:", v.shape, ref.shape)
print("NaN pattern identical:", bool(np.array_equal(np.isnan(v), np.isnan(ref))))
both = np.isfinite(v) & np.isfinite(ref)
print("largest absolute difference: %.2e m over %d values" % (float(np.abs(v[both] - ref[both]).max()), int(both.sum())))
print("valid cells on day 1:", int(np.isfinite(v[0]).sum()))
