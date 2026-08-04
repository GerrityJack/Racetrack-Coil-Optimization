"""
spike_floor_diag.py — outer-loop fix attempt #3, and the decisive test.

di_sweep_diag.py showed EVERY tested current jump (even +6A, the gentlest
possible) fails identically via iteration_spike_reverted at outer iter 1.
That is not "hard physics" -- a +6A nudge on a warm-started state should be
trivial. The real cause: newton_ta.step()'s spike check
(`its > max(10, 3*prev_max_iters)`) resets prev_max_iters=None at the START
of every step, so for a warm-started step whose FIRST outer iteration is
anomalously easy (its=1 -- T is already close to the new BC), prev_max_iters
becomes 1, and ANY normal-but-not-trivial iteration 2 (even 11 iterations,
nowhere near the 50-iter SNES cap) trips the "spike" alarm meant to catch
GENUINE divergence.

This tests the gentlest case (I=49->55A, dt=150s, the +6A jump) with the
spike check disabled and a much bigger outer budget, to distinguish two
possibilities:
  (a) it's PURELY a miscalibrated guard -- disabling it lets this trivial
      step reach genuine stall convergence cleanly.
  (b) there's ALSO a real underlying problem -- even the gentlest jump
      eventually hits an outright Newton failure (reason<=0) a few
      iterations later, same as the +49A jump did in spike_check_diag.py.
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
    params.mesh_filename = f"{root}_spikefloor{os.getpid()}{ext}"
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

    print("\n" + "=" * 78)
    print("step 2 (I=55A, dt=150s, dI=+6A) -- spike_check=False, max_outer=50")
    print("=" * 78)
    info2 = newton_ta.step(ta, domain, ic, nm, 55.0, 150.0, uniform,
                          max_outer=50, min_outer=3, first=False,
                          verbose=True, spike_check=False)
    print(f"\nRESULT: converged={info2['converged']} stop={info2['stop_reason']} "
          f"n_outer={info2['n_outer']} SCIF={info2['scif_mT']:.2f} mT")
    print(f"SCIF trajectory tail: "
          f"{[round(s,2) for s in info2['scif_hist_tail']]}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if info2["stop_reason"] == "stall":
        print("(a) PURE MISCALIBRATION CONFIRMED: the gentlest possible jump "
              "reaches genuine stall convergence once the spike guard is out "
              "of the way. Fix: recalibrate the spike floor for warm-started "
              "steps (e.g. do not spike-check outer iteration 1->2, or use "
              "an absolute floor decoupled from a possibly-anomalous "
              "prev_max_iters=1).")
    elif info2["stop_reason"] == "max_outer":
        print("INCONCLUSIVE: ran out of budget without failing OR "
              "converging. Consistent with (a) (just needs more outer "
              "iterations) but not confirmed -- would need a bigger budget.")
    else:
        print(f"(b) REAL PROBLEM CONFIRMED: even the gentlest +6A jump "
              f"eventually hits a genuine failure ({info2['stop_reason']}). "
              f"The spike guard was catching a real issue after all, just "
              f"also mis-firing early on benign jumps. Both are true.")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
