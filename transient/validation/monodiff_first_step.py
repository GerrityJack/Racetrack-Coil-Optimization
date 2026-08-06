"""
monodiff_first_step.py -- first-step smoke test / repro-case runner for
the differentiable-Jacobian monolithic solver (transient/monolithic_ta_diff.py).

Mirrors first_step_diagnostic.py's structure exactly (own mesh per
process, ZFC cold start, arbitrary independent (dt, I) pair) so results
are directly comparable to the canonical repro case's Picard/Newton-hybrid
numbers already on record.

Usage: <env python> monodiff_first_step.py <dt_seconds> <I_target_amps> [max_outer]
"""
import os
import sys

import numpy as np

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
    dt = float(sys.argv[1])
    I_target = float(sys.argv[2])
    max_outer = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    jc_n_relax = float(sys.argv[4]) if len(sys.argv) > 4 else None
    step_relax = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    from ic_model import IcModel, NValueModel
    from entropy_ic_model import EntropyBetaIcModel, HillNModel
    from monolithic_ta_diff import (build_monolithic_problem_diff,
                                    monolithic_diff_step)

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_monodiff_{int(dt)}_{int(I_target*10)}_{os.getpid()}{ext}"
    print(f"[dt={dt}s I={I_target}A] building mesh (own process) ...", flush=True)
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)

    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)

    print("Fitting entropy-max Beta Jc(B) + Hill n(B) models ...", flush=True)
    ic_beta = EntropyBetaIcModel()
    n_hill = HillNModel()
    build_monolithic_problem_diff(ta, ic_beta, n_hill, verbose=True)

    print("\n" + "=" * 78)
    print(f"MONOLITHIC-DIFF FIRST STEP FROM ZFC: dt={dt}s  I_target={I_target}A  "
          f"max_outer={max_outer}")
    print("=" * 78)

    info = monolithic_diff_step(
        ta, domain, ic, nm, I_target, dt, uniform,
        max_outer=max_outer, min_outer=6, stall_tol=0.5, first=True,
        bootstrap_iters=30, verbose=True, jc_n_relax=jc_n_relax,
        step_relax=step_relax)

    finite = (np.all(np.isfinite(ta["A_h"].x.array))
              and all(np.all(np.isfinite(T_i.x.array))
                     for T_i in ta["layer_T_fns"]))

    print("\n" + "=" * 78)
    print(f"RESULT dt={dt}s I_target={I_target}A: "
          f"converged={info['converged']}  n_outer={info['n_outer']}  "
          f"stop_reason={info['stop_reason']}  scif={info['scif_mT']:+.2f} mT  "
          f"finite={finite}  total_snes_iters={info['total_snes_iters']}")
    print("=" * 78)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
