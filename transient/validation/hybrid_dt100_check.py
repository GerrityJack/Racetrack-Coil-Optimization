"""
hybrid_dt100_check.py — THE decisive test. This whole Newton-Krylov
investigation started because Picard's own iteration could not converge at
dt=100s, I=32.667A, cold start (CLAUDE.md: "Picard did not converge even at
1000 iterations... SCIF wandered chaotically, std~100 mT, no decaying
trend"). Everything since (t_relax, the lag bug, the memoryless-Jc/n fix,
the newton_blend fix) has only been validated against the EASY dt=600 case.
This tests newton_ta.hybrid_step(newton_blend=0.15) -- the version that
matched Picard's dt=600 ground truth to ~1% -- on the ORIGINAL hard case.

No ground truth exists for dt=100 (Picard never converged there at all), so
this is a status/stability check, not a percentage-match check: does it
reach a genuine formal stall, and does the SCIF it reports look like a
believable, physically continuous number (not wildly different in
character from the dt=600 answer)?
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


def main():
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    import newton_ta
    from ic_model import IcModel, NValueModel

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_dt100chk_{os.getpid()}{ext}"
    print("building mesh ...", flush=True)
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    I = 32.666666667   # the exact hard-case current used throughout CLAUDE.md
    dt = 100.0

    print("=" * 78)
    print(f"HYBRID (newton_blend=0.15), THE ORIGINAL HARD CASE: "
          f"dt={dt}s, I={I:.3f}A, cold start")
    print("Picard never converged here (1000+ iters, std~100 mT, no trend)")
    print("=" * 78)
    ta_b = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                     per_layer=True, per_turn_bc=False)
    newton_ta.build_layer_newton_problems(ta_b, verbose=False)
    info_b = newton_ta.hybrid_step(ta_b, domain, ic, nm, I, dt, uniform,
                                   max_outer=150, min_outer=6,
                                   stall_tol=0.05, first=True,
                                   bootstrap_iters=30, verbose=True,
                                   newton_blend=0.15)
    print(f"\nRESULT: converged={info_b['converged']} "
          f"stop_reason={info_b['stop_reason']} n_outer={info_b['n_outer']} "
          f"SCIF={info_b['scif_mT']:+.2f} mT  "
          f"newton_failures={info_b['n_newton_failures']}", flush=True)
    print(f"SCIF trajectory tail: "
          f"{[round(s,2) for s in info_b['scif_hist_tail']]}", flush=True)

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if info_b["stop_reason"] == "stall":
        print("-> The hybrid reaches a GENUINE formal stall at the case "
              "Picard could never converge at all. This is the original "
              "motivating problem, solved -- pending a cross-check against "
              "some form of independent verification (no ground truth "
              "exists here, unlike dt=600).")
    else:
        print(f"-> Did not reach a genuine stall (stop_reason="
              f"{info_b['stop_reason']}) within the 150-iteration budget. "
              f"Improvement over Picard's total non-convergence would still "
              f"need to be judged from the trajectory shape above.")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
