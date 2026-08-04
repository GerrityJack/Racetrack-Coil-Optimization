"""
high_n_relax_diag.py — outer-loop fix attempt #4: per-layer relaxation.

spike_floor_diag.py isolated the root cause: even the gentlest +6A jump
eventually hits SNES_DIVERGED_DTOL on whichever layer's frozen n is
currently highest (~27 in that test, vs 20-26 on the other five layers).
relax_sweep_diag.py already showed a single UNIFORM relaxation factor makes
no difference -- consistent with the problem being concentrated in ONE
layer that a uniform damping doesn't specially protect.

This tests newton_ta.step()'s new `high_n_relax_factor`/`high_n_threshold`
kwargs: extra-damp (multiply relax by high_n_relax_factor) specifically
whichever layer's current frozen n exceeds high_n_threshold, leaving the
other five layers at the normal relax. Same +6A step (49->55A, dt=150s,
warm-started from an identical step-1 state) as spike_floor_diag.py, with
the spike guard back ON (default) this time -- if this fixes the REAL
underlying instability, the guard should simply never trip.
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
    params.mesh_filename = f"{root}_highnrelax{os.getpid()}{ext}"
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
    print("step 1 (I=49A, dt=150s, cold bootstrap)")
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
    for factor in (0.3, 0.1, 0.03, 0.01):
        restore()
        print("\n" + "=" * 78)
        print(f"step 2 (I=55A, dt=150s) -- high_n_relax_factor={factor}, "
              f"threshold=24.0, max_outer=60, spike_check=FALSE "
              f"(testing the REAL divergence, not the guard)")
        print("=" * 78)
        info = newton_ta.step(ta, domain, ic, nm, 55.0, 150.0, uniform,
                             max_outer=60, min_outer=3, first=False,
                             verbose=True, spike_check=False,
                             high_n_relax_factor=factor,
                             high_n_threshold=24.0)
        print(f"  RESULT factor={factor}: converged={info['converged']} "
              f"stop={info['stop_reason']} n_outer={info['n_outer']} "
              f"SCIF={info['scif_mT']:.2f} mT")
        results.append((factor, info))

    print("\n" + "=" * 78)
    print("PER-LAYER RELAX SWEEP SUMMARY")
    print("=" * 78)
    print(f"{'factor':>7}  {'converged':>9}  {'stop_reason':>24}  "
          f"{'n_outer':>7}  {'SCIF (mT)':>10}")
    for factor, info in results:
        print(f"{factor:7.2f}  {str(info['converged']):>9}  "
              f"{info['stop_reason']:>24}  {info['n_outer']:7d}  "
              f"{info['scif_mT']:10.2f}")

    any_converged = any(info["converged"] for _, info in results)
    print(f"\nAny factor reached the formal stall criterion: {any_converged}")
    if any_converged:
        print("-> PER-LAYER relaxation on the high-n layer FIXES this step. "
              "This is the real lever.")
    else:
        print("-> per-layer relaxation alone does not fix it either -- the "
              "problem is not simply about damping the update speed on the "
              "worst layer. Next lever: fold Jc(B)/n(B) into the Newton "
              "residual itself (bigger rewrite), or accept the outer-loop "
              "instability and design a stopping rule that tolerates it "
              "(risk: no guarantee the SCIF is meaningful when stopped).")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
