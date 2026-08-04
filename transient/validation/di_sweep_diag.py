"""
di_sweep_diag.py — outer-loop fix attempt #2. relax_sweep_diag.py showed
Jc/n relaxation strength (0.5 down to 0.1) makes essentially NO difference
to step 2's failure (same spike at outer iter 2, same SCIF to within 3 mT
every time) -- so the lever isn't relaxation magnitude. This tests the
other obvious candidate: the SIZE OF THE CURRENT JUMP into a warm-started
step. Same step 1 (I=49A, dt=150s, cold bootstrap) every time; step 2 tried
at several target currents (60/70/80/98A) at the SAME dt=150s, default
relax=0.5, max_outer=60. If smaller jumps converge cleanly while the
original 49->98A jump doesn't, the real fix is a finer current schedule
(more, smaller ramp steps), not a solver-parameter tweak.
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
    params.mesh_filename = f"{root}_disweep{os.getpid()}{ext}"
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
    for I_target in (55.0, 60.0, 70.0, 80.0, 98.0):
        restore()
        print("\n" + "=" * 78)
        print(f"step 2 (I={I_target}A, dt=150s, dI={I_target-49.0:+.1f}A) -- "
              f"relax=0.5 (default), max_outer=60")
        print("=" * 78)
        info = newton_ta.step(ta, domain, ic, nm, I_target, 150.0, uniform,
                             max_outer=60, min_outer=3, first=False,
                             verbose=True, spike_check=True)
        print(f"  RESULT I={I_target}: converged={info['converged']} "
              f"stop={info['stop_reason']} n_outer={info['n_outer']} "
              f"SCIF={info['scif_mT']:.2f} mT")
        results.append((I_target, info))

    print("\n" + "=" * 78)
    print("dI SWEEP SUMMARY")
    print("=" * 78)
    print(f"{'I_target':>9}  {'dI':>7}  {'converged':>9}  {'stop_reason':>24}  "
          f"{'n_outer':>7}  {'SCIF (mT)':>10}")
    for I_target, info in results:
        print(f"{I_target:9.1f}  {I_target-49.0:+7.1f}  "
              f"{str(info['converged']):>9}  {info['stop_reason']:>24}  "
              f"{info['n_outer']:7d}  {info['scif_mT']:10.2f}")

    converged_Is = [I for I, info in results if info["converged"]]
    if converged_Is:
        print(f"\nConverged for dI <= {max(converged_Is)-49.0:+.1f}A "
              f"(smallest failing jump tells the real budget per step)")
    else:
        print("\nNO current jump tested converged -- even the smallest "
              "(dI=+6A) fails. The issue is not step size at dt=150s; "
              "something else about warm-starting a second step is broken.")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
