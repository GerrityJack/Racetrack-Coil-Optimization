"""
monolithic_step_relax_sweep.py -- the monolithic block-Newton solver
(transient/monolithic_ta.py) diverged on its very first accuracy check:
the raw (step_relax=1.0) block-Newton step, even backtracking-line-search
globalized on the coupled system's raw residual norm, produced an on-axis
SCIF of +11589 mT after step k=0 (vs a ~+641 mT Picard ground truth) and
then failed outright at k=1.

Hypothesis being tested here, not assumed: the raw residual norm the line
search globalizes on does not "see" the SCIF (a near-cancelling difference
of much larger J values, the same sensitivity documented throughout this
project's history for the on-axis SCIF metric) -- so a step that looks
globalized/accepted in the residual-norm sense can still be enormous in the
SCIF-relevant sense. Damping the accepted JOINT (T, A) step
(monolithic_ta.step_relax) tests whether that alone recovers stability,
same dt=600/I=196A/cold-start case as monolithic_accuracy_check.py, one
shared mesh, several step_relax values in one process.
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
    import monolithic_ta

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_monosweep_{os.getpid()}{ext}"
    print("building mesh (shared by every step_relax value) ...", flush=True)
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    from ic_model import IcModel, NValueModel
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    I = params.I_design
    dt = params.ramp_duration
    print(f"I_design = {I} A, ramp_duration = {dt} s  (ground truth ~+641 mT)",
          flush=True)

    results = {}
    for step_relax in (0.3, 0.1, 0.03):
        print("\n" + "=" * 78)
        print(f"MONOLITHIC, step_relax={step_relax}")
        print("=" * 78)
        ta_x = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                         per_layer=True, per_turn_bc=False)
        monolithic_ta.build_monolithic_problem(ta_x, verbose=False)
        info = monolithic_ta.monolithic_step(
            ta_x, domain, ic, nm, I, dt, uniform,
            max_outer=150, min_outer=6, stall_tol=0.05, first=True,
            bootstrap_iters=30, verbose=True, step_relax=step_relax,
            debug=True)
        print(f"-> step_relax={step_relax}: converged={info['converged']} "
              f"stop_reason={info['stop_reason']} n_outer={info['n_outer']} "
              f"SCIF={info['scif_mT']:+.2f} mT  "
              f"tail={[round(s,2) for s in info['scif_hist_tail']]}",
              flush=True)
        results[step_relax] = info

    print("\n" + "=" * 78)
    print("SUMMARY (ground truth ~ +641 mT)")
    print("=" * 78)
    for sr, info in results.items():
        print(f"  step_relax={sr:5.2f}: converged={info['converged']!s:5} "
              f"stop_reason={info['stop_reason']:28s} "
              f"n_outer={info['n_outer']:4d}  SCIF={info['scif_mT']:+9.2f} mT")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
