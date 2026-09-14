"""
regenerate_ta_safe_lowcurrent_fields.py -- 2026-09-08

Re-solves the best candidate found by the 2026-09-02/03 T-A-safe-margin
overnight search (optimize/runs/ta_safe_margin/, run_tag
ta_safe_20260903_161304, eval 81) -- the one design in that search that
actually satisfies the T-A local quench margin (>=1.0, worst-cell margin
1.028), at the cost of a much lower field (B_target=4.10T, I_op=52.5A)
than the 10T champion. See CLAUDE.md's "2026-09-02/03" entry and known
open issue #7.

Only the summary CSV row was archived, not the full field snapshot
(coil_centroids/coil_B/J_TA_coil/etc.) needed for spatial plots like
transient/validation/plot_quench_location.py. This re-runs
optimize/ta_safe_current.evaluate() on the exact same design dict, using
its save_fields_path hook to write that snapshot -- same T-A bisection
the search itself already validated for this candidate (worst_margin
should reproduce ~1.03), not a new physics path.

Mutates params.py IN-MEMORY only (ta_safe_current._apply), never written
to disk -- run in its own process, params.py on disk is untouched.

Output: optimize/runs/ta_safe_margin/lowcurrent_design_fields.npz

Run:  <env>/bin/python3 optimize/studies/regenerate_ta_safe_lowcurrent_fields.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_OPT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_OPT)
for _p in (_ROOT, _OPT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mpi4py import MPI
import params
import ta_safe_current
from ic_extrapolation import make_ic_model

# params.py currently has the "xdense" one-off mesh-artifact-investigation
# tier live (committed, per CLAUDE.md 2026-09-03/04 -- 20-30x slower than
# medium per-solve). Force a specific tier IN-MEMORY ONLY here -- never
# written back to params.py on disk, so the committed xdense default is
# untouched. Select via TA_SAFE_MESH_TIER env var (default "medium").
#
#   Tier     DOFs (1/8 domain)   RAM    Time/solve (champion scale)
#   medium   ~48k                ~3GB   ~5s   -- this search's own tier
#   fine     ~70k                ~5GB   ~10s
#
# xdense is NOT offered here: on this design's much larger geometry
# (a~49mm vs champion's 26mm, 3254 turns) an xdense attempt ran past 30
# minutes without finishing a single evaluate() call (7 T-A solves) and
# was killed -- see git history / session log, 2026-09-08.
_TIER = os.environ.get("TA_SAFE_MESH_TIER", "medium")
if _TIER == "medium":
    params.mesh_size_min_factor = 0.25
    params.mesh_size_max_factor = 0.60
    params.mesh_dist_min_factor = 1.50
    params.mesh_dist_max_factor = 2.00
    params.box_scale = 4.0
    params.mesh_z_grading = [0.075, 0.15, 0.55, 0.15, 0.075]
elif _TIER == "fine":
    params.mesh_size_min_factor = 0.10
    params.mesh_size_max_factor = 1.0 / 3.0
    params.mesh_dist_min_factor = 1.00
    params.mesh_dist_max_factor = 3.00
    params.box_scale = 6.0
    params.mesh_z_grading = [0.075, 0.15, 0.55, 0.15, 0.075]
else:
    raise ValueError(f"unknown TA_SAFE_MESH_TIER={_TIER!r} (medium or fine)")

# exact row: optimize/runs/ta_safe_margin/gen19_archive/history_gen19.csv,
# run_tag=ta_safe_20260903_161304, eval=81, B_target_T=4.1024,
# ta_worst_margin=1.0277, I_op_A=52.549
DESIGN = dict(
    a=49.445221432807976e-3,
    b=54.56063243422137e-3,
    coil_half_gap=13.669586146219677e-3,
    n_turns=[1032, 1032, 153, 153, 442, 442],
)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    ic = make_ic_model("kim")
    suffix = "" if _TIER == "medium" else f"_{_TIER}"
    out = os.path.join(_OPT, "runs", "ta_safe_margin",
                       f"lowcurrent_design_fields{suffix}.npz")
    print(f"Mesh tier: {_TIER}")
    r = ta_safe_current.evaluate(DESIGN, ic, comm, verbose=True,
                                 save_fields_path=out)
    print("\n" + "=" * 70)
    print(f"[{_TIER}] I_op_A={r['I_op_A']:.2f}  B_target_T={r['B_target_T']:.3f}  "
          f"ta_worst_margin={r['ta_worst_margin']:.4f}  "
          f"uniformity_pct={r['uniformity_pct']:.2f}")
    print(f"(medium-tier reference: I_op_A=50.39 B_target_T=3.944 "
          f"ta_worst_margin=0.9567, 1/1833 cells<1.0)")
    print(f"Saved {out}")
