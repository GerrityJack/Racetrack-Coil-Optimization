"""loop_isolation_check2.py -- targeted variant of loop_isolation_check.py:
identical in every respect EXCEPT the order of setting T_bot_val/T_top_val
relative to the A-seed solve, matching every earlier script today's order
(seed A first, THEN set T_bot_val/T_top_val) instead of the official
order (set T_bot_val/T_top_val first, THEN seed A). If this alone
reproduces the wrong ~653.9mT answer, it confirms this ordering is the
actual bug.
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
    import numpy as np
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    from ta_transient import _picard_phase
    from ic_model import IcModel, NValueModel

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_isolate2_{os.getpid()}{ext}"
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    I = params.I_design
    dt = params.ramp_duration
    print(f"I_design={I} A, ramp_duration={dt} s", flush=True)

    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)
    delta_SC = ta["delta_SC"]
    eps_reg = getattr(params, "ta_eps_reg", 1.0)

    # ---- ONLY CHANGE vs loop_isolation_check.py: A-seed FIRST, THEN
    # T_bot_val/T_top_val -- matching every earlier script today's order.
    J_mag_homog = I / (params.t * params.w)
    uniform["J"].x.array[:] = uniform["J_unit_array"] * J_mag_homog
    ta_solve._solve_A(ta, ta["L_seed_form"])

    ta["T_h"].x.array[:] = 0.0
    ta["T_h"].x.scatter_forward()
    for T_i in ta["layer_T_fns"]:
        T_i.x.array[:] = 0.0
        T_i.x.scatter_forward()

    J_mag_SC = I / (delta_SC * params.w)
    J_coil_prev = ta["t_hat_coil"] * J_mag_SC
    ta_solve._update_Js(ta, J_coil_prev)

    ta["B_fn"].interpolate(ta["curl_expr"])
    B_coil_prev = ta["B_fn"].x.array.reshape(-1, 3)[ta["coil_cells"]].copy()

    ta["_rho_prev"] = None
    ta_solve._update_rho(ta, J_coil_prev, B_coil_prev, ic, nm, eps_reg)

    T_amp = I / (2.0 * delta_SC)
    ta["T_bot_val"].value = +T_amp
    ta["T_top_val"].value = -T_amp
    # ---- end reordered block

    ta["dt_const"].value = float(dt)

    print("Seed done (REORDERED: A-seed then BCs), entering _picard_phase ...",
          flush=True)

    J_coil, n_iters, converged = _picard_phase(
        ta, domain, ic, nm, I, dt, J_coil_prev,
        closure=lambda: None, max_iters=150, min_iters=25, scif_tol=0.05,
        label="isolate2", verbose=True)

    J_unif = ta["t_hat_coil"] * (I / (delta_SC * params.w))
    dJs = (J_coil - J_unif) * (delta_SC / ta["Lambda"])
    scif = float(ta_solve.dB_bore_from_dJ(
        ta["coil_centroids"], dJs, ta["coil_vols"])[2] * 1e3)

    print(f"\nRESULT (REORDERED seed, _picard_phase): "
          f"converged={converged} n_iters={n_iters} SCIF={scif:+.3f} mT",
          flush=True)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
