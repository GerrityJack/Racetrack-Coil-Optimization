"""
eval_winner_at_20K.py -- 2026-09-14

Re-evaluates the 4.2K-search's best design so far (16 layers, found
2026-09-13/14 overnight) under this project's own native, measured 20K
Ic/n-value model instead of the temperature-scaled 4.2K one -- same
geometry, same mesh tier, same evaluate() pipeline, ONLY the temperature
model changes. Answers directly: how much of this design's 6.99T (at
4.2K) is the temperature, and how much is the geometry itself?

Uses the DEFAULT alpha=(0.30,0.15) and DEFAULT min_iters=25 (NOT the
4.2K-retuned alpha=(0.03,0.01)/forced-full-length) -- 20K is this
project's own well-validated historical regime (n=13-34), where the
default settings are the ones actually validated, not the 4.2K-specific
fix. Matching the search's own medium mesh tier and the same
MARGIN_PERCENTILE-based bisection ta_safe_current.py always uses.

Design: stage6_16layer eval 13 (the 4.2K-search's best-so-far):
    a=31.130821333791943mm b=40.01565009808927mm gap=33.97411042487954mm
    n_turns=[550,550,584,584,17,17,507,507,507,507,613,613,629,629,605,605]
    At 4.2K (matched Ic+n, alpha=(0.03,0.01), forced full-length):
    I_op_A=63.06  B_target_T=6.987  ta_worst_margin=0.962  n_ta_solves=8

Output: optimize/runs/ta_safe_margin_4p2K/winner_at_20K_fields.npz

Run:  <env>/bin/python3 optimize/studies/eval_winner_at_20K.py
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

# force medium mesh tier in-memory (params.py on disk is still the
# 2026-09-03 xdense one-off tier) -- SAME as every search tonight
params.mesh_size_min_factor = 0.25
params.mesh_size_max_factor = 0.60
params.mesh_dist_min_factor = 1.50
params.mesh_dist_max_factor = 2.00
params.box_scale = 4.0
params.mesh_z_grading = [0.075, 0.15, 0.55, 0.15, 0.075]

# DELIBERATELY leave params.ta_picard_alpha / ta_picard_alpha_fine at
# their committed defaults (0.30, 0.15) -- the validated 20K setting,
# not the 4.2K-regime fix.

DESIGN = dict(
    a=31.130821333791943e-3,
    b=40.01565009808927e-3,
    coil_half_gap=33.97411042487954e-3,
    n_turns=[550, 550, 584, 584, 17, 17, 507, 507, 507, 507, 613, 613,
             629, 629, 605, 605],
)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    ic_model = make_ic_model("kim")   # plain 20K Kim-extrapolated model
    # n_model left as None -> ta_safe_current.evaluate() builds the plain
    # 20K NValueModel internally (its own default behaviour)

    out = os.path.join(_OPT, "runs", "ta_safe_margin_4p2K",
                       "winner_at_20K_fields.npz")
    print(f"Re-solving 4.2K-search winner AT 20K: a={DESIGN['a']*1e3:.3f}mm "
          f"b={DESIGN['b']*1e3:.3f}mm gap={DESIGN['coil_half_gap']*1e3:.3f}mm "
          f"n_turns={DESIGN['n_turns']}", flush=True)
    print(f"alpha=({params.ta_picard_alpha}, {params.ta_picard_alpha_fine}) "
          f"[project default, NOT the 4.2K-retuned pair]", flush=True)
    r = ta_safe_current.evaluate(DESIGN, ic_model, comm, verbose=True,
                                 save_fields_path=out)
    print("\n" + "=" * 70, flush=True)
    print(f"AT 20K:  I_op_A={r['I_op_A']:.2f}  B_target_T={r['B_target_T']:.3f}  "
          f"ta_worst_margin={r['ta_worst_margin']:.4f}  "
          f"uniformity_pct={r['uniformity_pct']:.3f}  "
          f"hoop_MPa={r['hoop_MPa']:.2f}  "
          f"bisection_exhausted={r['bisection_exhausted']}  "
          f"n_ta_solves={r['n_ta_solves']}", flush=True)
    print(f"AT 4.2K (search's own result): I_op_A=63.06  B_target_T=6.987  "
          f"ta_worst_margin=0.9618  uniformity_pct=0.130  hoop_MPa=23.70", flush=True)
    ratio = r['B_target_T'] / 6.987
    print(f"\nB_target(20K)/B_target(4.2K) = {ratio:.3f}  "
          f"({'20K is HIGHER' if ratio>1 else '4.2K is HIGHER'} by "
          f"{abs(1-ratio)*100:.1f}%)", flush=True)
    print(f"Saved {out}", flush=True)
