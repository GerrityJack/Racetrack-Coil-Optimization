"""jitter_retry_trace_check.py -- 2026-08-06, deeper accuracy check on the
jitter-retry idea, prompted directly by the user: checking only the final
SCIF is exactly the class of mistake that produced this project's own
false-positive "breakthrough" earlier (monolithic_diff_investigation_
2026-08-05.md Part 3/overnight-continuation -- the SCIF-EMA-stall
diagnostic plateaued and reported converged=True while the RAW residual
was exploding to ~1e95, never decreasing; nobody had checked the raw
signal until then).

This does NOT trust `_picard_phase`'s own `converged` flag alone. Using
its `closure` hook (called once per iteration, before that iteration's
T-solve -- so closure(k) observes state as of the END of iteration k-1),
it independently records, EVERY iteration:
  - dB_mag: |B_coil(k-1) - B_coil(k-2)| -- the SAME raw quantity
    _picard_phase computes internally for its own phase-switch logic,
    recomputed independently here rather than trusted from inside.
  - scif_raw: the INSTANTANEOUS (non-EMA-smoothed) SCIF at this iterate
    -- the stall criterion only ever looks at an EMA of this; if the raw
    signal is still swinging while the EMA has smoothed it flat, that is
    exactly the false-plateau failure mode being checked for here.
  - T_max, T_min: across all layer T functions, a basic sanity/scale
    check independent of SCIF entirely.

After `_picard_phase` returns, the LAST 10 iterations of this
independent trace are inspected: if `converged=True` but the raw SCIF
trace's own range over those last 10 iterations is still large relative
to the reported SCIF value (not just the EMA-smoothed one), that is
flagged explicitly as a suspect convergence, not silently accepted.

Usage: <env python> jitter_retry_trace_check.py <dt> <I_target>
                    <jitter_scale> <base_seed_or_neg1_for_random>
                    <out_json_path>
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
    jitter_scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
    base_seed_arg = int(sys.argv[4]) if len(sys.argv) > 4 else -1
    out_path = sys.argv[5] if len(sys.argv) > 5 else "/tmp/jitter_trace_result.json"

    import numpy as np
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    from ta_transient import _picard_phase, _seed_cold

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_jittrace_{os.getpid()}{ext}"
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

    base_seed = (base_seed_arg if base_seed_arg >= 0
                else int.from_bytes(os.urandom(4), "little"))

    J_coil = _seed_cold(ta, uniform, max(I_target, 1e-6))
    if jitter_scale > 0:
        rng = np.random.default_rng(base_seed)
        for T_i in ta["layer_T_fns"]:
            T_i.x.array[:] += (jitter_scale * T_amp
                               * rng.standard_normal(T_i.x.array.shape[0]))
            T_i.x.scatter_forward()
        J_coil = ta_solve._J_from_T(ta, domain)

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
        closure=trace_closure, max_iters=150, min_iters=6, scif_tol=0.5,
        label="trace", verbose=False)

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

    # Flags a "converged" run as SUSPECT if, over its own last 10 iterations,
    # either (a) the RAW (non-EMA) SCIF is still swinging by >5% of the final
    # value the stall criterion accepted, or (b) the relative field change
    # per iteration (|dB|/|B|) is still above 1e-3 -- both independent of,
    # and not visible to, the internal EMA-based stall check.
    suspect = bool(converged) and (
        abs(scif_spread_last10) > 0.05 * max(abs(scif_final), 1.0)
        or (np.isfinite(dB_rel_last) and dB_rel_last > 1e-3)
    )

    print(f"converged={converged}  n_iters={n_iters}  finite={finite}  "
          f"scif_final={scif_final:+.3f}mT  "
          f"last10_scif_spread={scif_spread_last10:.4f}mT  "
          f"dB_rel_last={dB_rel_last:.3e}  SUSPECT={suspect}", flush=True)

    with open(out_path, "w") as fh:
        json.dump(dict(dt=dt, I_target=I_target, jitter_scale=jitter_scale,
                       base_seed=base_seed, converged=bool(converged),
                       n_iters=int(n_iters), finite=bool(finite),
                       scif_final_mT=scif_final,
                       scif_spread_last10_mT=scif_spread_last10,
                       dB_rel_last=dB_rel_last, suspect=suspect,
                       trace=trace), fh, indent=2)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
