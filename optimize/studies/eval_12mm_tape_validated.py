"""
eval_12mm_tape_validated.py -- 2026-09-14

Final, PROPERLY VALIDATED evaluation of the 12mm-tape-width test
(test_12mm_tape_width.py's naive run did not converge at default alpha).
`test_12mm_tape_alpha_sweep.py` found alpha=(0.03,0.01) (the same pair
that fixed the transient/ short-dt problem AND the 4.2K regime) converges
cleanly here too -- 4 independent alpha values <=0.05 all converged to a
consistent worst_margin~1.70-1.89 cluster (raw |dB|/|B| ~1e-3, vs ~0.19
for every alpha>=0.10 tested). Uses the full ta_safe_current.evaluate()
pipeline (not just a fixed-current solve) to get real I_op_A/B_target_T/
uniformity/hoop numbers, with the validated alpha and forced full-length
solving throughout.

Run:  <env>/bin/python3 optimize/studies/eval_12mm_tape_validated.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys

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
from ic_model import NValueModel

params.mesh_size_min_factor = 0.25
params.mesh_size_max_factor = 0.60
params.mesh_dist_min_factor = 1.50
params.mesh_dist_max_factor = 2.00
params.box_scale = 4.0
params.mesh_z_grading = [0.075, 0.15, 0.55, 0.15, 0.075]

# VALIDATED alpha for this regime (test_12mm_tape_alpha_sweep.py) -- NOT
# the project default (0.30,0.15), which does not converge here (rel_err
# ~0.19 across 3 tested defaults-scale values).
params.ta_picard_alpha = 0.03
params.ta_picard_alpha_fine = 0.01

W_NEW = 0.012
WIDTH_RATIO = W_NEW / 0.004
N_LAYERS = 16
MIN_COIL_GAP_M = 0.003


class WidthScaledIcModel:
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
    a = 31.130821333791943e-3
    b = 40.01565009808927e-3
    n_turns = [550, 550, 584, 584, 17, 17, 507, 507, 507, 507, 613, 613,
               629, 629, 605, 605]
    margin_above_floor = 33.97411042487954e-3 - gap_floor(N_LAYERS, 0.004)
    new_gap = gap_floor(N_LAYERS, W_NEW) + margin_above_floor

    params.w = W_NEW  # must be set before evaluate()'s own _apply() call,
                       # which does NOT touch params.w itself

    comm = MPI.COMM_WORLD
    base_ic = make_ic_model("kim")
    ic_model = WidthScaledIcModel(base_ic, WIDTH_RATIO)
    n_model = NValueModel(csv_path=params.n_value_csv_filename)  # UNSCALED

    design = dict(a=a, b=b, coil_half_gap=new_gap, n_turns=n_turns)

    out = os.path.join(_OPT, "runs", "ta_safe_margin_4p2K",
                       "winner_12mm_tape_VALIDATED_fields.npz")
    print(f"Solving (VALIDATED, alpha=({params.ta_picard_alpha},"
         f"{params.ta_picard_alpha_fine}), 20K, 12mm tape, 3x-scaled Ic)...",
         flush=True)
    r = ta_safe_current.evaluate(design, ic_model, comm, verbose=True,
                                 n_model=n_model, min_iters=params.ta_n_picard,
                                 save_fields_path=out)
    print("\n" + "=" * 70, flush=True)
    print(f"12mm tape @ 20K, VALIDATED:  I_op_A={r['I_op_A']:.2f}  "
         f"B_target_T={r['B_target_T']:.3f}  "
         f"ta_worst_margin={r['ta_worst_margin']:.4f}  "
         f"uniformity_pct={r['uniformity_pct']:.3f}  "
         f"hoop_MPa={r['hoop_MPa']:.2f}  tape_km={r['tape_km']:.3f}  "
         f"bisection_exhausted={r['bisection_exhausted']}  "
         f"n_ta_solves={r['n_ta_solves']}", flush=True)
    print(f"\nReference, SAME geometry, 4mm tape @ 20K: I_op_A=44.20  "
         f"B_target_T=4.934", flush=True)
    print(f"Reference, SAME geometry, 4mm tape @ 4.2K: I_op_A=63.06  "
         f"B_target_T=6.987", flush=True)
    print(f"Saved {out}", flush=True)
