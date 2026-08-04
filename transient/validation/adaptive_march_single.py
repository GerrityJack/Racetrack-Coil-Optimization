"""
adaptive_march_single.py -- single-dt_init adaptive T-A march, run as its
OWN process.

Exists specifically to control a confound found running dt_init=60/30/15
back-to-back in ONE process (monolithic_step_relax_sweep.py-style reuse of
ta_solve.setup_ta_problem() multiple times): at the identical nominal
configuration (dt=3.75s, t=0, same target current), one chain succeeded and
another failed. PETSc's options database is process-global and this
project's established convention is to run independent configurations as
separate OS processes for exactly this class of risk -- so this script
takes ONE dt_init value (argv[1]) and does nothing else in the process.

Usage: <env python> adaptive_march_single.py <dt_init_seconds>
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
    dt_init = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0

    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    from adaptive_march import adaptive_march, AdaptiveMarchFailure

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_adaptsingle_{int(dt_init)}_{os.getpid()}{ext}"
    print(f"[dt_init={dt_init}s] building mesh (own process, own mesh) ...",
          flush=True)
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    from ic_model import IcModel, NValueModel
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    I = params.I_design
    t_ramp = params.ramp_duration
    print(f"[dt_init={dt_init}s] I_design={I}A t_ramp={t_ramp}s "
          f"(single-step reference ~+641 mT; dt_init=60 same-process "
          f"reference: +828.50 mT)", flush=True)

    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)

    print("\n" + "=" * 78)
    print(f"ADAPTIVE MARCH, dt_init={dt_init}s, OWN PROCESS")
    print("=" * 78)
    try:
        history, final_scif = adaptive_march(
            ta, domain, uniform, ic, nm, I, t_ramp, t_hold=0.0,
            dt_init=dt_init, dt_min=1.0, dt_max=600.0, grow=1.5, shrink=0.5,
            iters_low=15, max_iters_per_step=150, min_iters_per_step=6,
            scif_tol=0.5, max_rejects_per_step=10, verbose=True)
        n_steps = len(history)
        n_rejects = sum(h["n_rejects_this_step"] for h in history)
        dts = [h["dt"] for h in history]
        print("\n" + "=" * 78)
        print(f"SINGLE-PROCESS RESULT dt_init={dt_init}s: {n_steps} steps, "
              f"{n_rejects} rejects, dt range [{min(dts):.2f},{max(dts):.2f}]s, "
              f"final SCIF={final_scif:+.2f} mT", flush=True)
    except AdaptiveMarchFailure as e:
        print("\n" + "=" * 78)
        print(f"SINGLE-PROCESS RESULT dt_init={dt_init}s: FAILED: {e}",
              flush=True)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
