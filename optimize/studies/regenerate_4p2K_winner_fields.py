"""
regenerate_4p2K_winner_fields.py -- 2026-09-13

Re-solves the current best (highest B_target_T) candidate from tonight's
4.2K-model overnight search (optimize/studies/run_4p2K_overnight.py /
ta_safe_margin_search_4p2K.py), writing the field snapshot needed by
transient/validation/plot_quench_location.py -- the search itself only
archives the summary CSV row, not the full per-cell field arrays.

As of this script being written, the leader is from stage 5 (12 layers):
    run_tag=ta_safe_4p2K_20260913_085742 eval=17
    a=32.187mm b=45.254mm gap=26.557mm
    n_turns=[600,600,559,559,37,37,519,519,583,583,601,601]
    B_target_T=6.724  I_op_A=67.04  ta_worst_margin=1.006
Stage 5 was still running when this was captured -- see
optimize/runs/ta_safe_margin_4p2K/stage5_12layer/results.csv for whatever
it ended up finding; re-run this script with an updated DESIGN dict if a
later candidate wins.

Uses the SAME model + solver settings the search itself used (matched
Ic+n 4.2K-scaled models, alpha=(0.03,0.01), forced full-length Picard --
see ta_safe_margin_search_4p2K.py's own docstring for why each of these
is required for a trustworthy result under this project's n~50-60
regime), not a different/inconsistent evaluation path.

Output: optimize/runs/ta_safe_margin_4p2K/winner_fields.npz

Run:  <env>/bin/python3 optimize/studies/regenerate_4p2K_winner_fields.py
"""
import os
import sys

# 2026-09-13: unconstrained multi-threading was found this same session to
# cause severe, unpredictable slowdowns for this solver (one smoke test hit
# 788% CPU and 40+ CPU-minutes without finishing what single-threaded took
# ~6 min) -- set before any BLAS/MUMPS-touching import, matching what
# ta_safe_margin_search_4p2K.py's own main() does for its workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
from ic_temperature_scaling import (Fujikura4p2KScaledIcModel,
                                     Fujikura4p2KScaledNValueModel)

# force medium mesh tier in-memory (params.py on disk is still the
# 2026-09-03 xdense one-off tier) -- SAME as the search itself
params.mesh_size_min_factor = 0.25
params.mesh_size_max_factor = 0.60
params.mesh_dist_min_factor = 1.50
params.mesh_dist_max_factor = 2.00
params.box_scale = 4.0
params.mesh_z_grading = [0.075, 0.15, 0.55, 0.15, 0.075]

# retuned relaxation pair -- required for the 4.2K model's n~50-60 regime
# (default alpha=(0.30,0.15) does not converge there -- see CLAUDE.md's
# "Cryogenic (4.2K)" section)
params.ta_picard_alpha = 0.03
params.ta_picard_alpha_fine = 0.01

DESIGN = dict(
    a=32.18702994017985e-3,
    b=45.253941758558064e-3,
    coil_half_gap=26.557051729786796e-3,
    n_turns=[600, 600, 559, 559, 37, 37, 519, 519, 583, 583, 601, 601],
)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    base_ic = make_ic_model("kim")
    base_n = NValueModel(csv_path=params.n_value_csv_filename)
    ic_model = Fujikura4p2KScaledIcModel(base_ic)
    n_model = Fujikura4p2KScaledNValueModel(base_n)

    out = os.path.join(_OPT, "runs", "ta_safe_margin_4p2K", "winner_fields.npz")
    print(f"Re-solving 4.2K-search winner: a={DESIGN['a']*1e3:.3f}mm "
          f"b={DESIGN['b']*1e3:.3f}mm gap={DESIGN['coil_half_gap']*1e3:.3f}mm "
          f"n_turns={DESIGN['n_turns']}")
    r = ta_safe_current.evaluate(DESIGN, ic_model, comm, verbose=True,
                                 n_model=n_model, min_iters=params.ta_n_picard,
                                 save_fields_path=out)
    print("\n" + "=" * 70)
    print(f"I_op_A={r['I_op_A']:.2f}  B_target_T={r['B_target_T']:.3f}  "
          f"ta_worst_margin={r['ta_worst_margin']:.4f}  "
          f"uniformity_pct={r['uniformity_pct']:.2f}  "
          f"bisection_exhausted={r['bisection_exhausted']}")
    print(f"(search's own reported values: I_op_A=67.04 B_target_T=6.724 "
          f"ta_worst_margin=1.0062 -- should match closely, same model/"
          f"solver settings, mesh non-reproducibility across processes "
          f"gives a small expected difference)")
    print(f"Saved {out}")
