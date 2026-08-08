"""dt_crossing_ramp.py -- 2026-08-07, does a multi-step ramp survive
deliberately crossing the known dt<=32s chaotic boundary mid-sequence?

Adapted directly from full_ramp_run.py (same state-carrying Picard loop,
same alpha=(0.03, 0.01) fix, same forced-full-length methodology) -- the
only change is the per-step SCHEDULE. Steps 0-2 stay at the known-clean
dt=60s to establish a normal baseline; steps 3-4 deliberately drop to
dt=30s (confirmed fully chaotic in the 2026-08-06/07 dt-boundary sweep,
T_max/amp~10.8 there); steps 5-9 return to dt=60s to see whether state
carried forward (A_prev, per-layer T) from the chaotic steps poisons
subsequent clean-dt convergence, or whether it recovers.

Output: a single .npz under transient/full_validation_plots/data/, same
schema as full_ramp_run.py's output (step_summaries, per-iteration trace,
snapshots at SNAPSHOT_STEPS).
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

# steps 0-2: clean dt=60s baseline. steps 3-4: deliberately cross into the
# confirmed-chaotic dt=30s zone. steps 5-9: back to clean dt=60s, to see
# whether the ramp recovers.
SCHEDULE = [
    (60.0, 19.6),
    (60.0, 39.2),
    (60.0, 58.8),
    (30.0, 78.4),
    (30.0, 98.0),
    (60.0, 117.6),
    (60.0, 137.2),
    (60.0, 156.8),
    (60.0, 176.4),
    (60.0, 196.0),
]
SNAPSHOT_STEPS = {2, 4, 5, 9}  # last-clean-before, deepest-into-chaos,
                               # first-recovery-attempt, final


def main():
    alpha_high = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
    alpha_low = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    max_iters_per_step = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    out_name = sys.argv[4] if len(sys.argv) > 4 else "dt_crossing_ramp.npz"

    import numpy as np
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio
    import dolfinx.fem as fem

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
    params.mesh_filename = f"{root}_dtcross_{os.getpid()}{ext}"
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

    layer_dofs = []
    for i in range(params.n_layers):
        idx = ta["layer_cell_idx"][i]
        cells_i = ta["coil_cells"][idx]
        layer_dofs.append(fem.locate_dofs_topological(
            ta["V_T"], domain.topology.dim, cells_i))
    layer_dof_coords = [ta["V_T"].tabulate_dof_coordinates()[d] for d in layer_dofs]

    J_coil = _seed_cold(ta, uniform, max(SCHEDULE[0][1], 1e-6))
    ta["B_fn"].interpolate(ta["curl_expr"])
    B_seed = ta["B_fn"].x.array.reshape(-1, 3)[coil]
    ta["_rho_prev"] = None
    ta_solve._update_rho(ta, J_coil, B_seed, ic, nm, eps)

    trace = dict(cum_iter=[], step=[], scif_mT=[], dB_rel=[],
                T_max_amp=[[] for _ in range(params.n_layers)],
                T_min_amp=[[] for _ in range(params.n_layers)])
    snapshots = {}
    step_summaries = []
    _cum = {"k": 0}

    for step_idx, (dt, I_now) in enumerate(SCHEDULE):
        T_amp = I_now / (2.0 * delta_SC)
        ta["T_bot_val"].value = +T_amp
        ta["T_top_val"].value = -T_amp
        ta["dt_const"].value = float(dt)

        J_coil = ta_solve._J_from_T(ta, domain)
        J_unif = ta["t_hat_coil"] * (I_now / (delta_SC * params.w))

        _prev_B = {"arr": None}

        def closure():
            ta["B_fn"].interpolate(ta["curl_expr"])
            B_now = ta["B_fn"].x.array.reshape(-1, 3)[coil]
            Bn = float(np.linalg.norm(B_now.ravel()))
            dB_rel = (float(np.linalg.norm((B_now - _prev_B["arr"]).ravel())) / Bn
                      if _prev_B["arr"] is not None and Bn > 0 else float("nan"))
            _prev_B["arr"] = B_now.copy()

            J_now_ = ta_solve._J_from_T(ta, domain)
            dJs = (J_now_ - J_unif) * (delta_SC / ta["Lambda"])
            scif = float(ta_solve.dB_bore_from_dJ(
                ta["coil_centroids"], dJs, ta["coil_vols"])[2] * 1e3)

            trace["cum_iter"].append(_cum["k"])
            trace["step"].append(step_idx)
            trace["scif_mT"].append(scif)
            trace["dB_rel"].append(dB_rel)
            for i, T_i in enumerate(ta["layer_T_fns"]):
                vals = T_i.x.array[layer_dofs[i]] / T_amp
                trace["T_max_amp"][i].append(float(vals.max()))
                trace["T_min_amp"][i].append(float(vals.min()))
            _cum["k"] += 1

        J_coil, n_iters, converged = _picard_phase(
            ta, domain, ic, nm, I_now, dt, J_coil,
            closure=closure, max_iters=max_iters_per_step,
            min_iters=max_iters_per_step, scif_tol=0.5,
            label=f"step{step_idx}", verbose=False)

        finite = (np.all(np.isfinite(ta["A_h"].x.array))
                 and all(np.all(np.isfinite(T_i.x.array))
                        for T_i in ta["layer_T_fns"]))

        final_scif = trace["scif_mT"][-1]
        final_T_max = max(trace["T_max_amp"][i][-1] for i in range(params.n_layers))
        final_T_min = min(trace["T_min_amp"][i][-1] for i in range(params.n_layers))
        step_summaries.append(dict(step=step_idx, dt=dt, I_now=I_now,
                                   n_iters=n_iters, finite=bool(finite),
                                   scif_mT=final_scif, T_max_amp=final_T_max,
                                   T_min_amp=final_T_min))
        print(f"  step={step_idx} dt={dt:.0f}s I={I_now:.1f}A  n_iters={n_iters}  "
              f"finite={finite}  scif={final_scif:+.3f}mT  "
              f"T_max/amp={final_T_max:.3f}  T_min/amp={final_T_min:.3f}",
              flush=True)

        if not finite:
            print(f"  ABORTING: non-finite state at step {step_idx}", flush=True)
            break

        if step_idx in SNAPSHOT_STEPS:
            ta["B_fn"].interpolate(ta["curl_expr"])
            snap = dict(
                I_now=I_now, dt=dt,
                coil_centroids=ta["coil_centroids"].copy(),
                B_coil=ta["B_fn"].x.array.reshape(-1, 3)[coil].copy(),
                J_coil=J_coil.copy(),
            )
            for i in range(params.n_layers):
                snap[f"T_layer{i}"] = ta["layer_T_fns"][i].x.array[layer_dofs[i]].copy()
                snap[f"T_layer{i}_coords"] = layer_dof_coords[i].copy()
            snapshots[f"step{step_idx}"] = snap
            print(f"    (snapshot saved for step {step_idx})", flush=True)

        ta["A_prev"].x.array[:] = ta["A_h"].x.array
        ta["A_prev"].x.scatter_forward()

    os.makedirs(OUT_DATA_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DATA_DIR, out_name)

    save_dict = dict(
        schedule=SCHEDULE, alpha_high=alpha_high, alpha_low=alpha_low,
        max_iters_per_step=max_iters_per_step,
        cum_iter=np.array(trace["cum_iter"]),
        step_of_iter=np.array(trace["step"]),
        scif_mT=np.array(trace["scif_mT"]),
        dB_rel=np.array(trace["dB_rel"]),
        n_layers=params.n_layers,
        n_turns=np.array(params.n_turns),
        step_summaries=step_summaries,
        snapshot_steps=sorted(SNAPSHOT_STEPS),
    )
    for i in range(params.n_layers):
        save_dict[f"T_max_amp_layer{i}"] = np.array(trace["T_max_amp"][i])
        save_dict[f"T_min_amp_layer{i}"] = np.array(trace["T_min_amp"][i])
    for key, snap in snapshots.items():
        for k, v in snap.items():
            save_dict[f"{key}__{k}"] = v

    np.savez(out_path, **save_dict)
    print(f"\nSaved dt-crossing ramp data to {out_path}", flush=True)
    print(f"completed_all_steps={len(step_summaries) == len(SCHEDULE)}", flush=True)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
