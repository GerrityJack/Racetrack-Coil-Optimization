"""loop_isolation_check3.py -- calls _seed_cold (the imported function, as
every earlier script today did) instead of manually inlining the seed
logic, otherwise identical to loop_isolation_check2.py (A-seed order,
same alpha, same _picard_phase call). If this reproduces the WRONG
~653.9mT answer, _seed_cold itself (despite looking formula-identical to
the official inline block) is the culprit.
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
    from ta_transient import _picard_phase, _seed_cold
    from ic_model import IcModel, NValueModel

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_isolate4_{os.getpid()}{ext}"
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    I = params.I_design
    dt = params.ramp_duration
    print(f"I_design={I} A, ramp_duration={dt} s", flush=True)

    params.ta_picard_alpha = 0.30
    params.ta_picard_alpha_fine = 0.15

    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)
    delta_SC = ta["delta_SC"]
    eps_reg = getattr(params, "ta_eps_reg", 1.0)

    # ---- calling _seed_cold (imported), as every earlier script today did
    J_coil_prev = _seed_cold(ta, uniform, I)

    ta["B_fn"].interpolate(ta["curl_expr"])
    B_coil_prev = ta["B_fn"].x.array.reshape(-1, 3)[ta["coil_cells"]].copy()

    ta["_rho_prev"] = None
    ta_solve._update_rho(ta, J_coil_prev, B_coil_prev, ic, nm, eps_reg)

    T_amp = I / (2.0 * delta_SC)
    ta["T_bot_val"].value = +T_amp
    ta["T_top_val"].value = -T_amp
    # ---- end

    ta["dt_const"].value = float(dt)

    print(f"DEBUG: J_coil.sum()={J_coil_prev.sum():.6e}  "
          f"rho_fn.sum()={ta['rho_fn'].x.array.sum():.6e}  "
          f"T_bot_val={ta['T_bot_val'].value:.6e}  "
          f"T_top_val={ta['T_top_val'].value:.6e}  "
          f"dt_const={ta['dt_const'].value:.6e}  "
          f"T_amp={T_amp:.6e}  eps={eps_reg:.6e}", flush=True)

    print("Seed done (_seed_cold), entering _picard_phase "
          "(WITH diagnostic closure this time) ...", flush=True)

    J_unif_diag = ta["t_hat_coil"] * (I / (delta_SC * params.w))

    def diag_closure():
        # matches per_layer_diag_check.py's diag_closure exactly: reads
        # B_fn/T state and prints, no state writes -- testing whether this
        # itself is the source of the discrepancy.
        ta["B_fn"].interpolate(ta["curl_expr"])
        J_now = ta_solve._J_from_T(ta, domain)
        dJs_now = (J_now - J_unif_diag) * (delta_SC / ta["Lambda"])
        _ = ta_solve.dB_bore_from_dJ(ta["coil_centroids"], dJs_now,
                                     ta["coil_vols"])[2] * 1e3

    J_coil, n_iters, converged = _picard_phase(
        ta, domain, ic, nm, I, dt, J_coil_prev,
        closure=diag_closure, max_iters=150, min_iters=25, scif_tol=0.05,
        label="isolate4", verbose=True)

    J_unif = ta["t_hat_coil"] * (I / (delta_SC * params.w))
    dJs = (J_coil - J_unif) * (delta_SC / ta["Lambda"])
    scif = float(ta_solve.dB_bore_from_dJ(
        ta["coil_centroids"], dJs, ta["coil_vols"])[2] * 1e3)

    print(f"\nRESULT (_seed_cold, _picard_phase): "
          f"converged={converged} n_iters={n_iters} SCIF={scif:+.3f} mT",
          flush=True)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
