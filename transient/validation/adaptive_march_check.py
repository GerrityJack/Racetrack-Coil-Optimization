"""
adaptive_march_check.py -- does adaptive step-size control alone (on top of
the EXISTING, unmodified, validated per-step Picard machinery,
ta_transient._picard_phase) let the base T-A solver march reliably from
zero-field-cooled all the way to I_design over the full 600s ramp -- the
thing NOTHING tried so far in this project's transient/ history has
actually tested, since every prior multi-step attempt used a fixed,
hand-picked dt schedule instead of a real step-size controller?

Success criteria, in order of importance:
  1. Does it complete the whole ramp without ever hitting the dt floor
     (AdaptiveMarchFailure)? This is the primary question -- "can Picard,
     with genuinely small enough steps, avoid the wandering/non-convergence
     this project has documented at several fixed short dt values?"
  2. How many step-rejects (shrinks) happened in total, and where in the
     ramp did they cluster?
  3. Final on-axis SCIF vs. the established single-dt=600-step reference
     (~+641 mT) -- NOT expected to match exactly (a fine multi-step BDF1
     march is a different, likely MORE accurate, time discretization of
     the same continuous ramp than one coarse 600s step), but should land
     in the same order of magnitude if both are converging to physically
     sensible states.
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
    from adaptive_march import adaptive_march, AdaptiveMarchFailure

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_adaptmarch_{os.getpid()}{ext}"
    print("building mesh ...", flush=True)
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    from ic_model import IcModel, NValueModel
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    I = params.I_design
    t_ramp = params.ramp_duration
    print(f"I_design = {I} A, t_ramp = {t_ramp} s  "
          f"(single-step reference ~+641 mT)", flush=True)

    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)

    print("\n" + "=" * 78)
    print("ADAPTIVE MARCH, insulated base Picard solver, ZFC -> I_design")
    print("=" * 78)
    try:
        history, final_scif = adaptive_march(
            ta, domain, uniform, ic, nm, I, t_ramp, t_hold=0.0,
            dt_init=30.0, dt_min=2.0, dt_max=600.0, grow=1.5, shrink=0.5,
            iters_low=15, max_iters_per_step=150, min_iters_per_step=6,
            scif_tol=0.5, max_rejects_per_step=10, verbose=True)
        failure = None
    except AdaptiveMarchFailure as e:
        history, final_scif, failure = [], float("nan"), str(e)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if failure:
        print(f"FAILED: {failure}")
    else:
        n_steps = len(history)
        n_rejects = sum(h["n_rejects_this_step"] for h in history)
        dts = [h["dt"] for h in history]
        print(f"Completed the full ramp in {n_steps} accepted steps "
              f"({n_rejects} total rejects across all steps).")
        print(f"dt used: min={min(dts):.2f}s  max={max(dts):.2f}s  "
              f"final={dts[-1]:.2f}s")
        print(f"Final on-axis SCIF: {final_scif:+.2f} mT  "
              f"(single-step reference: ~+641 mT)")
        print(f"\nPer-step reject counts (only steps with rejects shown):")
        for h in history:
            if h["n_rejects_this_step"] > 0:
                print(f"  t={h['t']:7.1f}s  dt={h['dt']:6.1f}s  "
                      f"rejects={h['n_rejects_this_step']}  "
                      f"n_iters={h['n_iters']}")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
