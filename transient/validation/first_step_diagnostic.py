"""
first_step_diagnostic.py -- decoupled single-step convergence test.

adaptive_march.py's linear ramp ties dt and the target current I together
at a FIXED rate (I_next = I_target * (t+dt) / t_ramp), so every case tried
so far confounds two things that could independently explain why small
first steps from zero-field-cooled fail while a large one (dt=60s,
I=19.6A) succeeds:
  (a) a small dt inflates the T-equation's 1/dt forcing coefficient
      (curl(A_h - A_prev)/dt), possibly amplifying a small/noisy signal
  (b) a small target current I means a smaller absolute seed and possibly
      a qualitatively different (harder) regime for the Picard iteration,
      independent of dt

This script runs ONE Picard step from ZFC (T=0, A=0) to an ARBITRARY,
independently-chosen (dt, I_target) pair -- no ramp profile at all -- so
dt and I can be varied one at a time.

Usage: <env python> first_step_diagnostic.py <dt_seconds> <I_target_amps>
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

    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    from ta_transient import _picard_phase, _seed_cold

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_fsdiag_{int(dt)}_{int(I_target*10)}_{os.getpid()}{ext}"
    print(f"[dt={dt}s I={I_target}A] building mesh (own process) ...", flush=True)
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    from ic_model import IcModel, NValueModel
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)

    delta_SC = ta["delta_SC"]
    eps = float(getattr(params, "ta_eps_reg", 1.0))
    B_h = ta["B_fn"]
    coil = ta["coil_cells"]

    J_coil = _seed_cold(ta, uniform, max(I_target, 1e-6))
    B_h.interpolate(ta["curl_expr"])
    B_seed = B_h.x.array.reshape(-1, 3)[coil]
    ta["_rho_prev"] = None
    ta_solve._update_rho(ta, J_coil, B_seed, ic, nm, eps)

    T_amp = I_target / (2.0 * delta_SC)
    ta["T_bot_val"].value = +T_amp
    ta["T_top_val"].value = -T_amp
    ta["dt_const"].value = float(dt)

    print("\n" + "=" * 78)
    print(f"FIRST STEP FROM ZFC: dt={dt}s  I_target={I_target}A  "
          f"(1/dt forcing coeff = {1.0/dt:.4f})")
    print("=" * 78)

    J_coil, n_iters, converged = _picard_phase(
        ta, domain, ic, nm, I_target, dt, J_coil,
        closure=lambda: None, max_iters=150, min_iters=6, scif_tol=0.5,
        label=f"dt={dt}s I={I_target}A", verbose=True)

    finite = (np.all(np.isfinite(ta["A_h"].x.array))
              and all(np.all(np.isfinite(T_i.x.array))
                     for T_i in ta["layer_T_fns"]))

    print("\n" + "=" * 78)
    print(f"RESULT dt={dt}s I_target={I_target}A: "
          f"converged={converged}  n_iters={n_iters}  finite={finite}")
    print("=" * 78)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
