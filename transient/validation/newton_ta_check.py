"""
newton_ta_check.py — validates the Newton/Picard-hybrid T-A solver
(transient/newton_ta.py) against the established ground truth, full 6-layer
champion geometry, insulated case (no NI closure yet).

Two checks, both against numbers already established by the (unmodified)
Picard solver in this session's investigation.

NOTE on `converged`: it is intentionally strict -- True ONLY when the real,
formal multi-iteration SCIF-stall criterion fires (a normal, uninterrupted
run). When the outer Jc(B)/n(B) loop has to revert-and-stop (a Newton
failure, or a sudden iteration-count spike signalling numerical trouble
even when SNES itself claims success), `converged` is ALWAYS False, even
if the returned SCIF happens to be accurate -- an earlier attempt to
auto-classify "was this reverted stop actually trustworthy" via a crude
SCIF-delta threshold was itself unreliable (it flagged a value later
confirmed correct as untrustworthy). So case A below is judged on the SCIF
VALUE against Picard's independently-known ground truth, not on
`converged` -- do not "fix" this check to require `converged=True`, that
would be re-introducing the unreliable auto-classifier this file's history
retired.

  A. REGRESSION at dt=600s, I=32.667A, cold start: the Picard solver
     converges cleanly here (k=71, SCIF settling to 172.77 mT). The hybrid
     reaches a comparable SCIF in far fewer outer iterations (each outer
     iteration's inner Newton solve is exact, unlike Picard's damped
     approximation) -- but currently via a revert-and-stop, not the formal
     stall criterion, so `converged=False` is EXPECTED here; the pass/fail
     judgement is entirely on the SCIF value.

  B. THE TARGET FIX at dt=100s, I=32.667A, cold start: Picard did not
     converge even at 1000 iterations here (persistent wandering, std~100 mT
     -- see CLAUDE.md's 2026-08-04 entries). The hybrid at least reaches a
     stopping point without crashing or wandering indefinitely, but there is
     NO ground truth to check its SCIF against (Picard never converged at
     dt=100 at all) -- this check reports status/trajectory only, it does
     not (yet) assert a pass/fail on the SCIF value.

Run:  <env>/bin/python3 transient/validation/newton_ta_check.py
"""

import os
import sys

sys.stdout.reconfigure(line_buffering=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRANS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_TRANS)
for _p in (_TRANS, _ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve"),
           os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def run_case(domain, cell_tags, facet_tags, uniform, ic, nm, I, dt_val,
            label, max_outer=30):
    import ta_solve
    import newton_ta

    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)
    newton_ta.build_layer_newton_problems(ta, verbose=False)

    print(f"\n=== {label}: I={I:.3f} A, dt={dt_val:.1f} s ===")
    info = newton_ta.step(ta, domain, ic, nm, I, dt_val, uniform,
                          max_outer=max_outer, first=True, verbose=True)
    print(f"  RESULT: converged={info['converged']}  "
          f"stop_reason={info['stop_reason']}  "
          f"outer_iters={info['n_outer']}  "
          f"total_inner_SNES_iters={info['total_snes_iters']}  "
          f"SCIF={info['scif_mT']:.2f} mT")
    print(f"  SCIF trajectory (last {len(info['scif_hist_tail'])}): "
          f"{[round(s, 2) for s in info['scif_hist_tail']]}")
    return info


def main():
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    from ic_model import IcModel, NValueModel

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_newtck{os.getpid()}{ext}"
    print("building mesh ...")
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    I = 196.0 / 6

    print("=" * 78)
    print("Newton/Picard-hybrid T-A solver validation")
    print("=" * 78)

    info_600 = run_case(domain, cell_tags, facet_tags, uniform, ic, nm,
                        I, 600.0, "A. REGRESSION vs Picard (k=71, SCIF=172.77 mT)")
    info_100 = run_case(domain, cell_tags, facet_tags, uniform, ic, nm,
                        I, 100.0, "B. TARGET FIX (Picard never converged, "
                        "1000+ iters, std~100 mT)")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    # A: judged on the SCIF VALUE against Picard's independent ground truth
    # -- NOT on `converged`, which is correctly False for any revert-and-stop
    # regardless of accuracy (see module docstring).
    ok_a = abs(info_600["scif_mT"] - 172.77) < 20.0
    print(f"A. dt=600 regression : SCIF={info_600['scif_mT']:.2f} mT "
          f"(Picard ground truth: 172.77 mT)  stop_reason="
          f"{info_600['stop_reason']}  {'PASS' if ok_a else 'FAIL'}")
    # B: no ground truth exists to check against (Picard never converged at
    # dt=100 at all) -- report status only, no pass/fail assertion on SCIF.
    print(f"B. dt=100 target fix : SCIF={info_100['scif_mT']:.2f} mT "
          f"(no ground truth available)  stop_reason="
          f"{info_100['stop_reason']}  STATUS ONLY, not a validated result")
    print("=" * 78)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass
    return 0 if ok_a else 1


if __name__ == "__main__":
    sys.exit(main())
