"""
frozen_jcn_diagnostic.py -- THROWAWAY diagnostic, not a production change.

Tests whether re-evaluating Jc(B)/n(B) from the CURRENT B every Picard
iteration (ta_solve._update_rho's normal behaviour) is a material
contributor to the short-dt chaotic-wandering failure documented for
first_step_diagnostic.py, or whether the instability lives entirely in
rho(J)'s power-law nonlinearity (which Newton already linearizes exactly,
per newton_ta.py, without fixing the short-dt problem).

Identical setup to first_step_diagnostic.py (dt/I from argv, insulated
limit, per_layer=True, per_turn_bc=False, cold start from ZFC, eighth-
domain, own mesh in own process) EXCEPT: Jc_vol and n(B) are evaluated
ONCE from the seed (uniform-J) B field and held FIXED for the entire step
-- every subsequent Picard iteration only recomputes Jmag from the
CURRENT J (via the unmodified rho(J) power law) against those frozen
Jc/n arrays, never re-reading B again for the coefficient lookup. This is
a copy of _update_rho's math with Jc_vol/n_arr as fixed inputs instead of
a B-dependent recomputation -- not a modification to ta_solve.py itself.

Usage: <env python> frozen_jcn_diagnostic.py <dt_seconds> <I_target_amps>
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


def _update_rho_frozen_jcn(ta, J_coil, Jc_vol_fixed, n_arr_fixed, eps_reg,
                           relax=None):
    """ta_solve._update_rho's exact math, but Jc_vol/n_arr are FIXED
    (passed in, computed once from the seed B) instead of recomputed from
    the current B. Only Jmag (from the CURRENT J) varies call to call."""
    import params
    delta_SC = ta["delta_SC"]
    Lambda = ta["Lambda"]
    n_hat = ta["n_hat_coil"]

    J_dot_n = np.einsum("ij,ij->i", J_coil, n_hat)
    J_inplane = J_coil - J_dot_n[:, None] * n_hat
    Jmag = np.linalg.norm(J_inplane, axis=-1)

    jr = Jmag / Jc_vol_fixed
    p = float(getattr(params, "ta_floor_smooth_p", 16.0) or 0.0)
    if p > 0:
        j_norm = (eps_reg ** p + jr ** p) ** (1.0 / p)
    else:
        j_norm = np.maximum(jr, eps_reg)

    log_j = np.log(np.maximum(j_norm, 1e-30))
    rho_SC = (1e-4 / Jc_vol_fixed) * np.exp((n_arr_fixed - 1.0) * log_j)
    rho_homog = rho_SC * (delta_SC / Lambda)

    if relax is not None and ta.get("_rho_prev") is not None:
        rho_homog = np.exp((1.0 - relax) * np.log(ta["_rho_prev"])
                           + relax * np.log(rho_homog))
    ta["_rho_prev"] = rho_homog

    ta["rho_fn"].x.array[:] = 0.0
    ta["rho_fn"].x.array[ta["coil_cells"]] = rho_homog
    ta["rho_fn"].x.scatter_forward()
    return Jmag, Jc_vol_fixed, n_arr_fixed


def _picard_phase_frozen_jcn(ta, domain, Jc_vol_fixed, n_arr_fixed, I_now,
                             dt, J_coil, max_iters, min_iters, scif_tol,
                             label, verbose=True):
    """Copy of ta_transient._picard_phase's loop structure, with the
    rho update replaced by the frozen-Jc/n version above. No NI closure
    (closure is a no-op, matching first_step_diagnostic.py's usage)."""
    import params
    import ta_solve

    B_h = ta["B_fn"]
    coil = ta["coil_cells"]
    eps = float(getattr(params, "ta_eps_reg", 1.0))
    alpha_high = float(getattr(params, "ta_picard_alpha", 0.30))
    alpha_low = float(getattr(params, "ta_picard_alpha_fine", 0.15))
    J_unif = ta["t_hat_coil"] * (I_now / (ta["delta_SC"] * params.w))

    alpha = alpha_high
    phase2 = False
    prev_dB_mag = np.inf
    scif_ema = None
    scif_hist = []
    B_h.interpolate(ta["curl_expr"])
    B_prev = B_h.x.array.reshape(-1, 3)[coil].copy()

    converged = False
    n_iters = max_iters
    for k in range(max_iters):
        for i, (T_i, prob) in enumerate(zip(ta["layer_T_fns"],
                                            ta["prob_T_layers"])):
            sol = prob.solve()
            if np.any(np.isnan(sol.x.array)):
                raise RuntimeError(f"NaN in T solve, layer {i}, iter {k}")
            T_i.x.array[:] = (1.0 - alpha) * T_i.x.array + alpha * sol.x.array
            T_i.x.scatter_forward()

        J_coil = ta_solve._J_from_T(ta, domain)
        ta_solve._update_Js(ta, J_coil)
        ta_solve._solve_A(ta, ta["L_A_form"])
        if np.any(np.isnan(ta["A_h"].x.array)):
            raise RuntimeError(f"NaN in A solve, iter {k}")

        B_h.interpolate(ta["curl_expr"])
        B_coil = B_h.x.array.reshape(-1, 3)[coil]
        # THE ONLY SUBSTANTIVE DIFFERENCE FROM _picard_phase: frozen Jc/n
        _update_rho_frozen_jcn(ta, J_coil, Jc_vol_fixed, n_arr_fixed, eps,
                               relax=getattr(params, "ta_rho_relax", 0.5))

        dB = np.linalg.norm((B_coil - B_prev).ravel())
        if not phase2 and k >= 4 and dB >= 0.95 * prev_dB_mag:
            phase2 = True
            alpha = alpha_low
            if verbose:
                print(f"      [{label}] ramp-up done -> alpha={alpha_low}",
                     flush=True)
        prev_dB_mag = dB
        B_prev = B_coil.copy()

        dJs = (J_coil - J_unif) * (ta["delta_SC"] / ta["Lambda"])
        scif = ta_solve.dB_bore_from_dJ(ta["coil_centroids"], dJs,
                                        ta["coil_vols"])[2] * 1e3
        scif_ema = scif if scif_ema is None else 0.8 * scif_ema + 0.2 * scif
        scif_hist.append(scif_ema)
        if len(scif_hist) > 6:
            scif_hist.pop(0)
        stall = (abs(scif_hist[-1] - scif_hist[0])
                if len(scif_hist) == 6 else float("nan"))
        if verbose:
            print(f"      [{label} k={k+1:3d}] SCIF={scif_ema:+9.2f} mT  "
                 f"stall={stall:7.3f}  alpha={alpha:.2f}  |dB|={dB:.3e}",
                 flush=True)
        if (k + 1) >= min_iters and len(scif_hist) == 6:
            if abs(scif_hist[-1] - scif_hist[0]) < scif_tol:
                converged = True
                n_iters = k + 1
                break

    if verbose:
        tag = "conv" if converged else "CAP "
        print(f"      [{label}] {tag} in {n_iters} iters, "
             f"SCIF={scif_hist[-1] if scif_hist else float('nan'):+.2f} mT",
             flush=True)
    return J_coil, n_iters, converged


def main():
    dt = float(sys.argv[1])
    I_target = float(sys.argv[2])

    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    from ta_transient import _seed_cold
    from ic_model import IcModel, NValueModel, angle_with_normal_deg

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_frozenjcn_{int(dt)}_{int(I_target*10)}_{os.getpid()}{ext}"
    print(f"[FROZEN-JCN dt={dt}s I={I_target}A] building mesh (own process) ...",
         flush=True)
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.shanghai_csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)

    delta_SC = ta["delta_SC"]
    eps = float(getattr(params, "ta_eps_reg", 1.0))
    B_h = ta["B_fn"]
    coil = ta["coil_cells"]
    n_hat = ta["n_hat_coil"]

    J_coil = _seed_cold(ta, uniform, max(I_target, 1e-6))
    B_h.interpolate(ta["curl_expr"])
    B_seed = B_h.x.array.reshape(-1, 3)[coil].copy()

    # Jc(B)/n(B) evaluated ONCE from the seed B and held fixed for the
    # ENTIRE step -- this is the frozen-coefficient experiment.
    B_mag_seed = np.linalg.norm(B_seed, axis=-1)
    theta_seed = angle_with_normal_deg(B_seed, n_hat)
    Ic_arr_fixed, _ = ic.critical_current(B_mag_seed, theta_seed)
    Jc_vol_fixed = Ic_arr_fixed / (delta_SC * ic.tape_width)
    n_arr_fixed, _ = nm.n_value(B_mag_seed, theta_seed)
    print(f"[FROZEN-JCN] seed |B| mean={B_mag_seed.mean():.4f} T  "
         f"Jc_vol mean={Jc_vol_fixed.mean():.3e} A/m^2  "
         f"n mean={n_arr_fixed.mean():.2f}  (FIXED for the whole step)",
         flush=True)

    ta["_rho_prev"] = None
    _update_rho_frozen_jcn(ta, J_coil, Jc_vol_fixed, n_arr_fixed, eps)

    T_amp = I_target / (2.0 * delta_SC)
    ta["T_bot_val"].value = +T_amp
    ta["T_top_val"].value = -T_amp
    ta["dt_const"].value = float(dt)

    print("\n" + "=" * 78)
    print(f"FROZEN-JCN FIRST STEP FROM ZFC: dt={dt}s  I_target={I_target}A  "
         f"(Jc/n FIXED at seed-B values for the whole step)")
    print("=" * 78)

    J_coil, n_iters, converged = _picard_phase_frozen_jcn(
        ta, domain, Jc_vol_fixed, n_arr_fixed, I_target, dt, J_coil,
        max_iters=150, min_iters=6, scif_tol=0.5,
        label=f"FROZEN-JCN dt={dt}s I={I_target}A", verbose=True)

    finite = (np.all(np.isfinite(ta["A_h"].x.array))
              and all(np.all(np.isfinite(T_i.x.array))
                     for T_i in ta["layer_T_fns"]))

    print("\n" + "=" * 78)
    print(f"RESULT FROZEN-JCN dt={dt}s I_target={I_target}A: "
         f"converged={converged}  n_iters={n_iters}  finite={finite}")
    print("=" * 78)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
