"""
t_relax_diag.py — outer-loop fix attempt #5, testing the user's hypothesis:
the original Picard scheme damps BOTH rho/Jc/n AND T itself every iteration
(`T = (1-alpha)*T_old + alpha*T_new`); this Newton-hybrid has only ever
damped Jc/n -- each layer's Newton solve is EXACT and accepted outright.
Four independent Jc/n-relaxation levers (uniform, per-layer, current-jump
size, spike threshold) all made ZERO difference to the outer-loop
divergence, which is inconsistent with "the nonlinearity is just stiff" --
a real damping lever should help SOME amount. This tests the other half of
the original scheme's damping, applied post-hoc to the exact Newton
solution via the new `t_relax` kwarg on newton_ta.step().

Same problem case as spike_floor_diag.py / high_n_relax_diag.py: step 1
(I=49A, dt=150s, cold bootstrap), then step 2 (I=55A, dt=150s, warm-started,
the gentlest +6A jump already shown to fail at default settings), tried at
several t_relax values with spike_check=False (testing the real divergence)
and default Jc/n relax (uniform, unchanged).
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
    params.mesh_filename = f"{root}_trelax{os.getpid()}{ext}"
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
    for tr in (1.0, 0.5, 0.3, 0.15, 0.08):
        restore()
        print("\n" + "=" * 78)
        print(f"step 2 (I=55A, dt=150s) -- t_relax={tr}, max_outer=60, "
              f"spike_check=FALSE")
        print("=" * 78)
        info = newton_ta.step(ta, domain, ic, nm, 55.0, 150.0, uniform,
                             max_outer=60, min_outer=3, first=False,
                             verbose=True, spike_check=False, t_relax=tr)
        print(f"  RESULT t_relax={tr}: converged={info['converged']} "
              f"stop={info['stop_reason']} n_outer={info['n_outer']} "
              f"SCIF={info['scif_mT']:.2f} mT")
        print(f"  SCIF trajectory tail: "
              f"{[round(s,2) for s in info['scif_hist_tail']]}")
        results.append((tr, info))

    print("\n" + "=" * 78)
    print("T-RELAX SWEEP SUMMARY")
    print("=" * 78)
    print(f"{'t_relax':>8}  {'converged':>9}  {'stop_reason':>24}  "
          f"{'n_outer':>7}  {'SCIF (mT)':>10}")
    for tr, info in results:
        print(f"{tr:8.2f}  {str(info['converged']):>9}  "
              f"{info['stop_reason']:>24}  {info['n_outer']:7d}  "
              f"{info['scif_mT']:10.2f}")

    any_converged = any(info["converged"] for _, info in results)
    print(f"\nAny t_relax value reached the formal stall criterion: {any_converged}")
    if any_converged:
        print("-> CONFIRMED: T-level relaxation (the missing half of the "
              "original scheme's double-damping) fixes this. Newton on the "
              "power-law term is fine; the outer coupling needed damping "
              "that was never there.")
    else:
        print("-> T-relaxation alone ALSO doesn't fix it -- neither half of "
              "the original double-damping, tried separately, works here. "
              "Worth checking if n_outer at least improved vs the baseline "
              "(n_outer=2 throughout every prior test).")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
