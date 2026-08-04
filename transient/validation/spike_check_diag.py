"""
spike_check_diag.py — one-off diagnostic: is the iteration-count "spike"
heuristic in newton_ta.step() (calibrated on a single dt=600s case) too
trigger-happy in the warm-started multi-step regime newton_march_check.py
just exercised, where it reverted 3 of 4 ramp steps after only 1 fully
successful outer iteration?

Runs step 1 (I=49A, cold bootstrap) normally, then step 2 (I=98A, warm
start) TWICE: once with the spike check on (reproduces the march's
"reverted after 1 outer iter" result) and once with it disabled and a
larger outer-iteration budget, to see whether outer iteration 2 (14 SNES
iters on layer 0, flagged as a spike) was actually heading toward
divergence or was just a legitimately slower-but-fine solve.

NOT a replacement for ground truth -- there is none for this specific
warm-started (I, dt, history) combination yet. This only asks: does
continuing past the spike point look STABLE (bounded SCIF, no blowup) or
does it show the same cascading-divergence signature CLAUDE.md documents
for the "keep failed layer's old state" mistake?
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
    params.mesh_filename = f"{root}_spikediag{os.getpid()}{ext}"
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
    print("step 1 (I=49A, dt=150s, cold bootstrap) -- normal")
    print("=" * 78)
    info1 = newton_ta.step(ta, domain, ic, nm, 49.0, 150.0, uniform,
                          max_outer=30, first=True, bootstrap_iters=30,
                          verbose=True)
    print(f"step1: converged={info1['converged']} stop={info1['stop_reason']} "
          f"SCIF={info1['scif_mT']:.2f} mT")

    # Snapshot state after step 1 so both step-2 variants start identically.
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

    print("\n" + "=" * 78)
    print("step 2 variant A (I=98A, dt=150s) -- spike_check=True (reproduces march)")
    print("=" * 78)
    infoA = newton_ta.step(ta, domain, ic, nm, 98.0, 150.0, uniform,
                          max_outer=30, first=False, verbose=True,
                          spike_check=True)
    print(f"variant A: converged={infoA['converged']} stop={infoA['stop_reason']} "
          f"SCIF={infoA['scif_mT']:.2f} mT n_outer={infoA['n_outer']}")

    restore()

    print("\n" + "=" * 78)
    print("step 2 variant B (I=98A, dt=150s) -- spike_check=False, max_outer=20")
    print("=" * 78)
    infoB = newton_ta.step(ta, domain, ic, nm, 98.0, 150.0, uniform,
                          max_outer=20, first=False, verbose=True,
                          spike_check=False)
    print(f"variant B: converged={infoB['converged']} stop={infoB['stop_reason']} "
          f"SCIF={infoB['scif_mT']:.2f} mT n_outer={infoB['n_outer']}")
    print(f"variant B SCIF trajectory (tail): "
          f"{[round(s,2) for s in infoB['scif_hist_tail']]}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    tail = infoB["scif_hist_tail"]
    bounded = len(tail) >= 2 and max(abs(x) for x in tail) < 500.0
    trending_ok = infoB["stop_reason"] in ("stall",) or (
        bounded and (len(tail) < 2 or abs(tail[-1] - tail[-2]) < 100.0))
    print(f"variant B stayed bounded (|SCIF|<500mT throughout tail): {bounded}")
    print(f"variant B stop_reason: {infoB['stop_reason']}")
    if infoB["stop_reason"] == "stall":
        print("-> spike check was PREMATURE: without it, this step reaches "
              "the formal stall criterion cleanly. The threshold is too "
              "tight for this regime.")
    elif bounded and infoB["stop_reason"] != "newton_failure_reverted":
        print("-> INCONCLUSIVE-LEANING-OK: ran out of outer-iteration budget "
              "but stayed bounded, no cascade/blowup observed. Consistent "
              "with 'slower to converge here, not actually diverging' -- "
              "would need a larger max_outer to know for sure.")
    else:
        print("-> spike check was JUSTIFIED: without it, the state genuinely "
              "diverges/cascades (unbounded or a Newton failure follows). "
              "Keep spike_check=True as the default.")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
