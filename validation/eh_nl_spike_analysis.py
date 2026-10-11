"""
eh_nl_spike_analysis.py -- error hunt 2026-10-05: where do the FEM's
too-high loads at floors <= 0.7 in the nonlinear benchmark come from?
Reads validation/ta_nl_bench_cells_f<floor>.npz (NB_DUMP=1 runs).
"""
import os
import sys

import numpy as np

_VAL = os.path.dirname(os.path.abspath(__file__))
REF_MAX = 0.80          # reference worst load at these floors (pure 0.7985)


def quality(v):
    e = v[:, :, None, :] - v[:, None, :, :]
    lmax = np.sqrt((e ** 2).sum(-1)).reshape(len(v), -1).max(1)
    vol = np.abs(np.linalg.det(v[:, 1:] - v[:, :1])) / 6.0
    return 6.0 * np.sqrt(2.0) * vol / lmax ** 3


def main():
    floors = [f for f in ("1", "0.9", "0.8", "0.7", "0.6")
              if os.path.exists(os.path.join(_VAL, f"ta_nl_bench_cells_f{f}.npz"))]
    D = {f: np.load(os.path.join(_VAL, f"ta_nl_bench_cells_f{f}.npz"))
         for f in floors}
    v0 = D[floors[0]]["verts"]
    q = quality(v0)
    zc = D[floors[0]]["cents"][:, 2]
    zb, zt = zc.min(), zc.max()
    for f in floors:
        u = D[f]["u"]
        hi = u > REF_MAX * 1.05
        print(f"floor {f}: max {u.max():.3f}  p99.9 {np.percentile(u, 99.9):.3f} "
              f"p99 {np.percentile(u, 99):.3f}  cells > {REF_MAX*1.05:.2f}: "
              f"{hi.sum()} of {len(u)}")
        if hi.any():
            idx = np.argsort(u)[::-1][:8]
            for i in idx:
                c = D[f]["cents"][i]
                print(f"    u {u[i]:.3f} at x {c[0]*1e3:6.2f} y {c[1]*1e3:6.2f} "
                      f"z-frac {(c[2]-zb)/(zt-zb):.3f}  quality {q[i]:.4f} "
                      f"(rank {int((q < q[i]).sum())})")
            print(f"    quality of high cells: median {np.median(q[hi]):.4f} "
                  f"vs all {np.median(q):.4f}")
            zf = (zc[hi] - zb) / (zt - zb)
            print(f"    z-fraction of high cells: {np.round(np.percentile(zf, [0, 25, 50, 75, 100]), 3)}")
    if "0.8" in D and "0.7" in D:
        d = D["0.7"]["u"] - D["0.8"]["u"]
        print(f"change 0.8 -> 0.7: median {np.median(d):+.4f}, "
              f"p1 {np.percentile(d, 1):+.4f}, p99 {np.percentile(d, 99):+.4f}")


if __name__ == "__main__":
    main()
