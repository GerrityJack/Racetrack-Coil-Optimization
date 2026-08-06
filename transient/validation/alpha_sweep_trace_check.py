"""alpha_sweep_trace_check.py -- 2026-08-06, follow-up to the deterministic
failure trace (nondeterminism_investigation_2026-08-05.md's 2026-08-06
continuation): that trace showed the canonical dt=60s/I=19.6A cold-start
case never converges, single-threaded, deterministic -- |dB|/|B| sits at
50-90% EVERY iteration for the full 150-iteration budget, at BOTH the
validated fast (alpha=0.30) and careful (alpha=0.15) relaxation settings.
The two-phase scheme provides no effective damping in this regime at all.

Coordinator-directed next step: get rid of the noise first (forced
single-threaded, exactly as before, no jitter), THEN test whether a
genuinely smaller FIXED relaxation factor actually stabilises the Picard
map here, before concluding anything stronger is needed. This is
DELIBERATELY NOT an adaptive alpha-throttle -- CLAUDE.md explicitly warns
"every adaptive alpha-throttle tried on the sharp-flux-front dataset
misfired" and to not reintroduce one without testing against that
dataset. This keeps the EXACT SAME validated two-phase STRUCTURE
(_picard_phase, completely unmodified -- fast ramp-up until |dB| stops
decreasing, then a fixed slow phase) and only changes the two alpha
VALUES fed into it, via params.ta_picard_alpha / ta_picard_alpha_fine
(which _picard_phase already reads via getattr with these exact names --
confirmed by reading its source -- so this requires no code change to
_picard_phase itself, just setting module attributes before calling it).

Usage: <env python> alpha_sweep_trace_check.py <dt> <I_target>
                    <alpha_high> <alpha_low> <max_iters> <out_json_path>
"""
import os
import sys
import json

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
    dt = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    I_target = float(sys.argv[2]) if len(sys.argv) > 2 else 19.6
    alpha_high = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
    alpha_low = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
    max_iters = int(sys.argv[5]) if len(sys.argv) > 5 else 400
    out_path = sys.argv[6] if len(sys.argv) > 6 else "/tmp/alpha_sweep_result.json"

    import numpy as np
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    from ta_transient import _picard_phase, _seed_cold

    # ONLY change vs. the validated defaults: the two relaxation factors.
    # Same two-phase STRUCTURE, different fixed VALUES -- not an adaptive
    # scheme.
    params.ta_picard_alpha = alpha_high
    params.ta_picard_alpha_fine = alpha_low

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_alphasweep_{os.getpid()}{ext}"
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
    T_amp = I_target / (2.0 * delta_SC)

    J_coil = _seed_cold(ta, uniform, max(I_target, 1e-6))
    ta["B_fn"].interpolate(ta["curl_expr"])
    B_seed = ta["B_fn"].x.array.reshape(-1, 3)[ta["coil_cells"]]
    ta["_rho_prev"] = None
    ta_solve._update_rho(ta, J_coil, B_seed, ic, nm, eps)

    ta["T_bot_val"].value = +T_amp
    ta["T_top_val"].value = -T_amp
    ta["dt_const"].value = float(dt)

    J_unif = ta["t_hat_coil"] * (I_target / (delta_SC * params.w))
    coil = ta["coil_cells"]
    trace = []
    _state = {"B_prev": None}

    def trace_closure():
        ta["B_fn"].interpolate(ta["curl_expr"])
        B_coil = ta["B_fn"].x.array.reshape(-1, 3)[coil]
        B_mag = float(np.linalg.norm(B_coil.ravel()))
        dB_mag = (float(np.linalg.norm((B_coil - _state["B_prev"]).ravel()))
                  if _state["B_prev"] is not None else float("nan"))
        dB_rel = dB_mag / B_mag if (np.isfinite(dB_mag) and B_mag > 0) else float("nan")
        _state["B_prev"] = B_coil.copy()

        J_now = ta_solve._J_from_T(ta, domain)
        dJs = (J_now - J_unif) * (delta_SC / ta["Lambda"])
        scif_raw = float(ta_solve.dB_bore_from_dJ(
            ta["coil_centroids"], dJs, ta["coil_vols"])[2] * 1e3)

        T_max = float(max(T_i.x.array.max() for T_i in ta["layer_T_fns"]))
        T_min = float(min(T_i.x.array.min() for T_i in ta["layer_T_fns"]))
        trace.append(dict(dB_mag=dB_mag, dB_rel=dB_rel, scif_raw_mT=scif_raw,
                          T_max=T_max, T_min=T_min))

    J_coil, n_iters, converged = _picard_phase(
        ta, domain, ic, nm, I_target, dt, J_coil,
        closure=trace_closure, max_iters=max_iters, min_iters=6, scif_tol=0.5,
        label=f"alpha_h={alpha_high}_l={alpha_low}", verbose=False)

    finite = (np.all(np.isfinite(ta["A_h"].x.array))
             and all(np.all(np.isfinite(T_i.x.array))
                    for T_i in ta["layer_T_fns"]))

    dJs_final = (J_coil - J_unif) * (delta_SC / ta["Lambda"])
    scif_final = float(ta_solve.dB_bore_from_dJ(
        ta["coil_centroids"], dJs_final, ta["coil_vols"])[2] * 1e3)

    last10 = trace[-10:] if len(trace) >= 10 else trace
    last10_scif = [r["scif_raw_mT"] for r in last10]
    last10_dB_rel = [r["dB_rel"] for r in last10 if np.isfinite(r["dB_rel"])]
    scif_spread_last10 = (max(last10_scif) - min(last10_scif)
                          if last10_scif else float("nan"))
    dB_rel_last = last10_dB_rel[-1] if last10_dB_rel else float("nan")
    T_max_over_amp_final = trace[-1]["T_max"] / T_amp if trace else float("nan")

    print(f"alpha_high={alpha_high} alpha_low={alpha_low}  converged={converged}  "
          f"n_iters={n_iters}  finite={finite}  scif_final={scif_final:+.3f}mT  "
          f"last10_scif_spread={scif_spread_last10:.4f}mT  "
          f"dB_rel_last={dB_rel_last:.3e}  T_max/T_amp_final={T_max_over_amp_final:.3f}",
          flush=True)

    with open(out_path, "w") as fh:
        json.dump(dict(dt=dt, I_target=I_target, alpha_high=alpha_high,
                       alpha_low=alpha_low, max_iters=max_iters,
                       converged=bool(converged), n_iters=int(n_iters),
                       finite=bool(finite), scif_final_mT=scif_final,
                       scif_spread_last10_mT=scif_spread_last10,
                       dB_rel_last=dB_rel_last,
                       T_max_over_amp_final=T_max_over_amp_final,
                       trace=trace), fh, indent=2)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
