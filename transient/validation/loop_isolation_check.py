"""loop_isolation_check.py -- 2026-08-06, isolating why _picard_phase
(ta_transient.py) and the loop embedded in ta_solve.solve_ta_at_current()
disagree by ~2% (653.9mT vs 641.27mT) at the IDENTICAL problem (dt=600s,
I=196A, default alpha, cold start) despite both being run to genuine,
clean, non-premature convergence.

Builds mesh/ta EXACTLY as accuracy_check_I196.py does, seeds cold exactly
as solve_ta_at_current's own inline cold-start block does (byte-for-byte
copied), then runs _picard_phase (the OTHER implementation) on that same
`ta` object. If this ALSO gives ~653.9mT, the loop function itself is
the source of the discrepancy, not any setup/seeding difference.
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
    params.mesh_filename = f"{root}_isolate_{os.getpid()}{ext}"
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    I = params.I_design
    dt = params.ramp_duration
    print(f"I_design={I} A, ramp_duration={dt} s", flush=True)

    # EXACTLY as accuracy_check_I196.py builds ta for path A.
    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)
    delta_SC = ta["delta_SC"]
    eps_reg = getattr(params, "ta_eps_reg", 1.0)

    # BYTE-FOR-BYTE COPY of solve_ta_at_current's own cold-start block
    # (warm_start=False path), not _seed_cold -- eliminates any chance
    # that _seed_cold itself differs subtly.
    T_amp = I / (2.0 * delta_SC)
    ta["T_bot_val"].value = +T_amp
    ta["T_top_val"].value = -T_amp

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

    # dt_const: solve_ta_at_current relies on setup_ta_problem's own
    # initialization (dt_val = params.ramp_duration) and never touches it
    # again -- match that exactly (set explicitly here anyway for clarity,
    # same value).
    ta["dt_const"].value = float(dt)

    print("Seed done, entering _picard_phase (ta_transient.py's loop, "
          "NOT solve_ta_at_current's) ...", flush=True)

    J_coil, n_iters, converged = _picard_phase(
        ta, domain, ic, nm, I, dt, J_coil_prev,
        closure=lambda: None, max_iters=150, min_iters=25, scif_tol=0.05,
        label="isolate", verbose=True)

    J_unif = ta["t_hat_coil"] * (I / (delta_SC * params.w))
    dJs = (J_coil - J_unif) * (delta_SC / ta["Lambda"])
    scif = float(ta_solve.dB_bore_from_dJ(
        ta["coil_centroids"], dJs, ta["coil_vols"])[2] * 1e3)

    print(f"\nRESULT (_picard_phase, on officially-built ta): "
          f"converged={converged} n_iters={n_iters} SCIF={scif:+.3f} mT",
          flush=True)
    print(f"(For comparison, solve_ta_at_current's own loop on an "
          f"identically-built ta gives +641.27 mT)", flush=True)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
