"""Compare the pipeline output for one task (block 5) with the January 2022 file processed the other way
(default service, whole domain). Same product, so the numbers should be identical."""
import numpy as np
import pandas as pd

from oceanembed.glorys import glorys_to_target
from oceanembed.glorys_pipeline import block_dates, task_path

BLOCK, A, B = 5, 71, 182
z = np.load(task_path(BLOCK, A, B))
vals, i_idx, j_idx = z["values"], z["lat_idx"], z["lon_idx"]
start, end = block_dates(BLOCK)
print("pipeline file:", vals.shape, "| block", start.date(), "to", end.date(),
      "| lat idx", int(i_idx[0]), "-", int(i_idx[-1]), "| lon idx", int(j_idx[0]), "-", int(j_idx[-1]))

times, jan = glorys_to_target("data/raw/glorys/glorys_2022-01.nc")
offset = (pd.Timestamp(times[0]) - start).days
sub = jan[:, :, i_idx][:, :, :, j_idx]
pipe = vals[offset:offset + jan.shape[0]]
print("January 2022 starts at day offset", offset, "| compared days:", pipe.shape[0])
print("ocean cells at 0 m in this group:", int(np.isfinite(pipe[0, 0]).sum()), "of", pipe.shape[2] * pipe.shape[3])
print("NaN pattern identical:", bool(np.array_equal(np.isnan(sub), np.isnan(pipe))))
both = np.isfinite(sub) & np.isfinite(pipe)
print("largest absolute difference: %.2e C over %d values" % (float(np.abs(sub[both] - pipe[both]).max()), int(both.sum())))
