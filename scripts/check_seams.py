"""Do the assembled GLORYS blocks join without a jump? Compares the change across each seam with typical
day-to-day changes just before it (mean absolute change over the deep set)."""
import numpy as np

from oceanembed.glorys import TARGET_DEPTHS

deep = np.load("data/processed/target_mask_15depths.npy").all(axis=0)
A = {b: np.load(f"data/processed/glorys_block{b}.npy", mmap_mode="r") for b in (3, 4, 5)}


def mad(x, y, k):
    return float(np.abs(np.asarray(x[k])[deep] - np.asarray(y[k])[deep]).mean())


for prev, nxt in ((3, 4), (4, 5)):
    print(f"seam: block {prev} -> block {nxt}")
    for z in (0, 100, 1000):
        k = int(np.where(TARGET_DEPTHS == z)[0][0])
        seam = mad(A[prev][-1], A[nxt][0], k)
        typical = float(np.mean([mad(A[prev][-i - 1], A[prev][-i], k) for i in range(2, 22)]))
        print(f"  {z:>4} m: change across seam {seam:.3f} C | typical day-to-day change before it {typical:.3f} C")
print("(block 4 -> 5 coincides with the start of interim production, so a larger value there is informative, not necessarily a bug)")
