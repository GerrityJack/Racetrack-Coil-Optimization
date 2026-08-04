"""
relax_sweep_diag.py — outer-loop fix attempt #1: does WEAKER Jc/n relaxation
(more, gentler outer iterations) let a warm-started multi-step ramp actually
reach the formal SCIF-stall criterion, where the default relax=0.5 currently
reverts after just 1 outer iteration (see CLAUDE.md's 2026-08-05 entry)?

IMPORTANT CAVEAT ON REUSING THE OLD "relaxation strength didn't matter"
FINDING: CLAUDE.md's step()-docstring comment ("regardless of the Jc/n
relaxation strength tried") is from BEFORE the whole-iteration
snapshot/revert fix -- at that time relaxation was tested against a
DIFFERENT bug (per-layer revert corrupting cross-layer consistency), which
masked whatever effect relaxation alone would have had. This is a clean
re-test under the current, correct revert logic -- not a repeat of a
settled question.

Same setup as spike_check_diag.py: run step 1 (I=49A, dt=150s, cold
bootstrap) once, snapshot the resulting state, then run step 2 (I=98A,
dt=150s, warm-started) repeatedly from that IDENTICAL snapshot at several
jc_n_relax values, with a bigger max_outer budget (60) so a gentler
trajectory has room to actually reach the stall criterion instead of just
being cut off later. spike_check stays ON (already confirmed not a false
alarm in the prior diagnostic).
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

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_relaxsweep{os.getpid()}{ext}"
    print("building mesh ...")
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    from ic_model import IcModel, NValueModel
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)
    newton_ta.build_layer_newton_problems(ta, verbose=False)

    print("=" * 78)
    print("step 1 (I=49A, dt=150s, cold bootstrap) -- reference state for all variants")
    print("=" * 78)
    info1 = newton_ta.step(ta, domain, ic, nm, 49.0, 150.0, uniform,
                          max_outer=30, first=True, bootstrap_iters=30,
                          verbose=False)
    print(f"step1: converged={info1['converged']} stop={info1['stop_reason']} "
          f"SCIF={info1['scif_mT']:.2f} mT")

    T_snap = [T_i.x.array.copy() for T_i in ta["layer_T_fns"]]
    Jc_snap = [fn.x.array.copy() for fn in ta["newton_Jc_fns"]]
    n_snap = [fn.x.array.copy() for fn in ta["newton_n_fns"]]
    A_snap = ta["A_h"].x.array.copy()
    Aprev_snap = ta["A_prev"].x.array.copy()

    def restore():
        for T_i, s in zip(ta["layer_T_fns"], T_snap):
            T_i.x.array[:] = s
            T_i.x.scatter_forward()
        for fn, s in zip(ta["newton_Jc_fns"], Jc_snap):
            fn.x.array[:] = s
        for fn, s in zip(ta["newton_n_fns"], n_snap):
            fn.x.array[:] = s
        ta["A_h"].x.array[:] = A_snap
        ta["A_h"].x.scatter_forward()
        ta["A_prev"].x.array[:] = Aprev_snap
        ta["A_prev"].x.scatter_forward()

    results = []
    for relax in (0.5, 0.3, 0.2, 0.1):
        restore()
        print("\n" + "=" * 78)
        print(f"step 2 (I=98A, dt=150s) -- jc_n_relax={relax}, max_outer=60")
        print("=" * 78)
        info = newton_ta.step(ta, domain, ic, nm, 98.0, 150.0, uniform,
                             max_outer=60, min_outer=3, first=False,
                             verbose=True, spike_check=True,
                             jc_n_relax=relax)
        print(f"  RESULT relax={relax}: converged={info['converged']} "
              f"stop={info['stop_reason']} n_outer={info['n_outer']} "
              f"SCIF={info['scif_mT']:.2f} mT")
        print(f"  SCIF trajectory tail: "
              f"{[round(s,2) for s in info['scif_hist_tail']]}")
        results.append((relax, info))

    print("\n" + "=" * 78)
    print("SWEEP SUMMARY")
    print("=" * 78)
    print(f"{'relax':>6}  {'converged':>9}  {'stop_reason':>24}  "
          f"{'n_outer':>7}  {'SCIF (mT)':>10}")
    for relax, info in results:
        print(f"{relax:6.2f}  {str(info['converged']):>9}  "
              f"{info['stop_reason']:>24}  {info['n_outer']:7d}  "
              f"{info['scif_mT']:10.2f}")

    any_converged = any(info["converged"] for _, info in results)
    print(f"\nAny relax value reached the formal stall criterion: {any_converged}")
    if any_converged:
        best = min((r for r in results if r[1]["converged"]),
                  key=lambda r: r[1]["n_outer"])
        print(f"-> weaker Jc/n relaxation FIXES this step; best: "
              f"relax={best[0]} converged in {best[1]['n_outer']} outer iters")
    else:
        print("-> weaker relaxation ALONE does not fix it in this budget; "
              "the outer-loop fix needs a different lever (see CLAUDE.md "
              "candidate next steps: per-step bootstrap tuning, or a finer "
              "step schedule).")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
