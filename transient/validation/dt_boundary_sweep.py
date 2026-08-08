"""dt_boundary_sweep.py -- 2026-08-06, maps exactly where alpha=(0.03,0.01)
stops working between the confirmed-working dt=60s and the confirmed-
failing dt=30s (per Stage B of the same-day full validation). Single
first step from cold start at I=19.6A, forced full-length, single-
threaded, at each dt point -- same rigor as every other script in
today's validation.

Output: a single .npz under transient/full_validation_plots/data/ with
each dt point's full per-iteration trace plus a summary row, for the
boundary-transition summary plot.
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

OUT_DATA_DIR = os.path.join(_TRANS, "full_validation_plots", "data")

DT_POINTS = [60.0, 55.0, 50.0, 45.0, 40.0, 35.0, 32.0, 30.0]
I_FIXED = 19.6


def run_one(dt, alpha_high, alpha_low, max_iters):
    import numpy as np
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    from ta_transient import _picard_phase, _seed_cold
    from ic_model import IcModel, NValueModel

    params.ta_picard_alpha = alpha_high
    params.ta_picard_alpha_fine = alpha_low

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_dtsweep_{os.getpid()}{ext}"
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)
    delta_SC = ta["delta_SC"]
    eps = float(getattr(params, "ta_eps_reg", 1.0))
    coil = ta["coil_cells"]

    J_coil = _seed_cold(ta, uniform, max(I_FIXED, 1e-6))
    ta["B_fn"].interpolate(ta["curl_expr"])
    B_seed = ta["B_fn"].x.array.reshape(-1, 3)[coil]
    ta["_rho_prev"] = None
    ta_solve._update_rho(ta, J_coil, B_seed, ic, nm, eps)

    T_amp = I_FIXED / (2.0 * delta_SC)
    ta["T_bot_val"].value = +T_amp
    ta["T_top_val"].value = -T_amp
    ta["dt_const"].value = float(dt)

    J_unif = ta["t_hat_coil"] * (I_FIXED / (delta_SC * params.w))
    trace = dict(scif_mT=[], dB_rel=[], T_max_amp=[], T_min_amp=[])
    _prev_B = {"arr": None}

    def closure():
        ta["B_fn"].interpolate(ta["curl_expr"])
        B_now = ta["B_fn"].x.array.reshape(-1, 3)[coil]
        Bn = float(np.linalg.norm(B_now.ravel()))
        dB_rel = (float(np.linalg.norm((B_now - _prev_B["arr"]).ravel())) / Bn
                  if _prev_B["arr"] is not None and Bn > 0 else float("nan"))
        _prev_B["arr"] = B_now.copy()

        J_now = ta_solve._J_from_T(ta, domain)
        dJs = (J_now - J_unif) * (delta_SC / ta["Lambda"])
        scif = float(ta_solve.dB_bore_from_dJ(
            ta["coil_centroids"], dJs, ta["coil_vols"])[2] * 1e3)

        T_max = max(float((T_i.x.array / T_amp).max()) for T_i in ta["layer_T_fns"])
        T_min = min(float((T_i.x.array / T_amp).min()) for T_i in ta["layer_T_fns"])
        trace["scif_mT"].append(scif)
        trace["dB_rel"].append(dB_rel)
        trace["T_max_amp"].append(T_max)
        trace["T_min_amp"].append(T_min)

    J_coil = ta["t_hat_coil"] * (I_FIXED / (delta_SC * params.w))
    J_coil, n_iters, converged = _picard_phase(
        ta, domain, ic, nm, I_FIXED, dt, J_coil,
        closure=closure, max_iters=max_iters, min_iters=max_iters,
        scif_tol=0.5, label=f"dt{dt}", verbose=False)

    finite = (np.all(np.isfinite(ta["A_h"].x.array))
             and all(np.all(np.isfinite(T_i.x.array)) for T_i in ta["layer_T_fns"]))

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass

    return dict(dt=dt, n_iters=n_iters, finite=bool(finite), trace=trace)


def main():
    alpha_high = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
    alpha_low = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    max_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
    out_name = sys.argv[4] if len(sys.argv) > 4 else "dt_boundary_sweep.npz"

    import numpy as np

    results = []
    for dt in DT_POINTS:
        print(f"=== dt={dt} ===", flush=True)
        r = run_one(dt, alpha_high, alpha_low, max_iters)
        t = r["trace"]
        print(f"  dt={dt}  n_iters={r['n_iters']}  finite={r['finite']}  "
              f"final_scif={t['scif_mT'][-1]:+.3f}mT  "
              f"final_T_max={t['T_max_amp'][-1]:.3f}  "
              f"final_T_min={t['T_min_amp'][-1]:.3f}  "
              f"final_dB_rel={t['dB_rel'][-1]:.3e}", flush=True)
        results.append(r)

    os.makedirs(OUT_DATA_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DATA_DIR, out_name)

    save_dict = dict(dt_points=np.array(DT_POINTS), I_fixed=I_FIXED,
                     alpha_high=alpha_high, alpha_low=alpha_low,
                     max_iters=max_iters)
    for r in results:
        dt_key = f"dt{r['dt']:.0f}"
        save_dict[f"{dt_key}__n_iters"] = r["n_iters"]
        save_dict[f"{dt_key}__finite"] = r["finite"]
        for k, v in r["trace"].items():
            save_dict[f"{dt_key}__trace_{k}"] = np.array(v)

    np.savez(out_path, **save_dict)
    print(f"\nSaved dt-boundary sweep data to {out_path}", flush=True)


if __name__ == "__main__":
    main()
