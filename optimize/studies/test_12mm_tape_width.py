"""
test_12mm_tape_width.py -- 2026-09-14, quick one-off test (NOT part of any
search/optimizer -- does not touch cmaes_search.py, opt_config.py, or any
search script)

Question: what happens to the 4.2K-search's best design (16 layers) if the
tape were 12mm wide instead of 4mm -- the lever the ESMA/IRIS paper
comparison (CLAUDE.md known-open-issue #10) flagged as the real difference
between that real 10T magnet and this project's search (ESMA: 2x12mm
tapes face-to-face; this project: always a single 4mm tape).

Two physically-necessary adjustments beyond just setting params.w=0.012,
both disclosed here rather than silently changed:

1. **Ic scaling.** The Shanghai Superconductor CSV is Format B (absolute
   Ic in amps, measured on the real 4mm tape) -- `IcModel`'s own
   `tape_width` parameter is IGNORED for Format B data (see ic_model.py's
   docstring), so simply widening `params.w` would NOT increase the
   modeled critical current at all. `WidthScaledIcModel` below scales
   Ic(B,theta) by the width ratio (12/4 = 3x) as an explicit, disclosed
   first-order assumption -- more superconducting cross-section at the
   same current density/pinning landscape gives proportionally more
   critical current. This is NOT measured data for a 12mm tape (none
   exists in this project); treat the result as "what a linear-scaling
   assumption implies," same epistemic status as the 4.2K ratio-transplant
   model already used elsewhere in this project.
   n(B,theta) is left UNSCALED -- it reflects the pinning landscape's
   uniformity/steepness, not cross-sectional area, so no width-dependence
   is assumed.

2. **Coil gap.** `params.w` is the AXIAL (z) tape width -- 16 layers x
   12mm = 192mm of stack height vs. the original design's 16 x 4mm = 64mm.
   The winner's own gap (34.0mm) cannot physically fit a 192mm-tall
   stack. Gap is raised to the new manufacturing floor
   (N_LAYERS*w/2 + MIN_COIL_GAP_M/2) plus the SAME margin above that
   floor the original design had, so the geometry is actually buildable.
   `a`/`b`/`n_turns` (radial geometry) are left EXACTLY as found by the
   search -- unchanged.

Run:  <env>/bin/python3 optimize/studies/test_12mm_tape_width.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_OPT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_OPT)
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "mesh"),
           os.path.join(_ROOT, "solve"), _OPT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mpi4py import MPI
import params
import ta_safe_current
from ic_extrapolation import make_ic_model

params.mesh_size_min_factor = 0.25
params.mesh_size_max_factor = 0.60
params.mesh_dist_min_factor = 1.50
params.mesh_dist_max_factor = 2.00
params.box_scale = 4.0
params.mesh_z_grading = [0.075, 0.15, 0.55, 0.15, 0.075]
# default alpha (0.30, 0.15) -- 20K regime, not the 4.2K-retuned pair

W_ORIGINAL = 0.004
W_NEW = 0.012
WIDTH_RATIO = W_NEW / W_ORIGINAL

N_LAYERS = 16
MIN_COIL_GAP_M = 0.003  # matches opt_config.py


class WidthScaledIcModel:
    """Wraps a base (measured-tape) Ic model, scaling Ic(B,theta) by
    `ratio` -- see module docstring point 1 for the assumption and its
    disclosed limitation (not measured data)."""
    kind = "width_scaled"

    def __init__(self, base_ic_model, ratio):
        self.base = base_ic_model
        self.ratio = ratio
        self.tape_width = base_ic_model.tape_width * ratio
        self.B_min = base_ic_model.B_min

    def critical_current(self, B_tesla, theta_deg, clip_B=True):
        Ic, frac = self.base.critical_current(B_tesla, theta_deg, clip_B=clip_B)
        return Ic * self.ratio, frac


def gap_floor(n_layers, w):
    return n_layers * w / 2.0 + MIN_COIL_GAP_M / 2.0


if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    # original winner geometry (radial part unchanged)
    a = 31.130821333791943e-3
    b = 40.01565009808927e-3
    n_turns = [550, 550, 584, 584, 17, 17, 507, 507, 507, 507, 613, 613,
               629, 629, 605, 605]
    orig_gap = 33.97411042487954e-3

    orig_floor = gap_floor(N_LAYERS, W_ORIGINAL)
    margin_above_floor = orig_gap - orig_floor
    new_floor = gap_floor(N_LAYERS, W_NEW)
    new_gap = new_floor + margin_above_floor

    print(f"w: {W_ORIGINAL*1e3:.1f}mm -> {W_NEW*1e3:.1f}mm ({WIDTH_RATIO:.1f}x)")
    print(f"stack height: {N_LAYERS*W_ORIGINAL*1e3:.1f}mm -> "
          f"{N_LAYERS*W_NEW*1e3:.1f}mm")
    print(f"gap floor: {orig_floor*1e3:.2f}mm -> {new_floor*1e3:.2f}mm "
          f"(margin above floor preserved: {margin_above_floor*1e3:.3f}mm)")
    print(f"new coil_half_gap: {new_gap*1e3:.2f}mm (was {orig_gap*1e3:.2f}mm)")
    print(f"a={a*1e3:.3f}mm b={b*1e3:.3f}mm n_turns={n_turns} (UNCHANGED)")

    params.w = W_NEW
    params.recompute_derived()

    base_ic = make_ic_model("kim")   # plain 20K Kim-extrapolated model
    ic_model = WidthScaledIcModel(base_ic, WIDTH_RATIO)

    design = dict(a=a, b=b, coil_half_gap=new_gap, n_turns=n_turns)

    out = os.path.join(_OPT, "runs", "ta_safe_margin_4p2K",
                       "winner_12mm_tape_fields.npz")
    print("\nSolving (20K, width-scaled Ic, 3x)...", flush=True)
    r = ta_safe_current.evaluate(design, ic_model, comm, verbose=True,
                                 save_fields_path=out)
    print("\n" + "=" * 70, flush=True)
    print(f"12mm tape @ 20K:  I_op_A={r['I_op_A']:.2f}  "
          f"B_target_T={r['B_target_T']:.3f}  "
          f"ta_worst_margin={r['ta_worst_margin']:.4f}  "
          f"uniformity_pct={r['uniformity_pct']:.3f}  "
          f"hoop_MPa={r['hoop_MPa']:.2f}  tape_km={r['tape_km']:.3f}  "
          f"bisection_exhausted={r['bisection_exhausted']}  "
          f"n_ta_solves={r['n_ta_solves']}", flush=True)
    print(f"\nReference, SAME geometry, 4mm tape @ 20K: I_op_A=44.20  "
          f"B_target_T=4.934  tape_km=1.962", flush=True)
    print(f"Reference, SAME geometry, 4mm tape @ 4.2K: I_op_A=63.06  "
          f"B_target_T=6.987  tape_km=1.962", flush=True)
    print(f"Saved {out}", flush=True)
