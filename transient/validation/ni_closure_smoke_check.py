"""ni_closure_smoke_check.py -- 2026-08-07, first test of the NI radial-
current closure (transient/ni_circuit.py) under the validated
alpha=(0.03, 0.01) relaxation fix.

Every multi-step ramp validated so far (full_ramp_run.py,
dt_crossing_ramp.py) used the INSULATED limit only -- ta_solve's plain
T-A Picard loop, per_turn_bc=False, no circuit.update() ever called.
CLAUDE.md flags the NI circuit closure as explicitly out of scope for
everything validated to date. This script is the first attempt to close
that gap.

Two things are combined here for the first time, and this is
deliberately a SMOKE TEST, not a full validation:
  1. per_turn_bc=True (required for the per-bin circuit closure -- the
     alpha fix has only ever been validated under per_turn_bc=False).
  2. transient/ni_circuit.py's own per-bin radial-current solve, which
     has its own separate instability history (a Jacobi-divergence bug
     fixed 2026-08-04, predating the alpha fix entirely).

Methodology matches every other script in this validation effort:
forced full-length (min_iters=max_iters, bypassing ta_transient's own
EMA-based `converged` flag -- tparams.py's default STEP_MIN_ITERS=6 and
STEP_STALL_MT=0.5 are FAR too permissive under this project's own
established lesson that the flag fires 300-1000+ iterations before
genuine settling at short dt/small alpha), single-threaded, raw
diagnostics (T_max/amp per layer, dB_rel, circuit I_z/I_r) not just the
printed summary.

Schedule kept deliberately modest and at the KNOWN-clean dt=60s (not
tparams.py's default dt=600/24=25s or 200/12=16.7s ramp/hold schedule,
which sit BELOW the validated dt floor even before the circuit closure
is added) -- the question here is whether the closure itself converges
under the alpha fix, not whether it also survives the separate short-dt
problem. That combination is a follow-up, not this test.

Output: a single .npz under transient/full_validation_plots/data/.
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

# modest, clean-dt schedule -- same low-current baseline as
# dt_crossing_ramp.py's steps 0-2, for direct comparability against the
# insulated-limit reference.
SCHEDULE = [
    (60.0, 19.6),
    (60.0, 39.2),
    (60.0, 58.8),
]


def main():
    alpha_high = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
    alpha_low = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    max_iters_per_phase = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    out_name = sys.argv[4] if len(sys.argv) > 4 else "ni_closure_smoke.npz"

    import numpy as np
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio
    import dolfinx.fem as fem

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    import ta_transient as tt
    from ta_transient import _picard_phase, _seed_cold
    from ic_model import IcModel, NValueModel

    params.ta_picard_alpha = alpha_high
    params.ta_picard_alpha_fine = alpha_low

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_nismoke_{os.getpid()}{ext}"
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    ta, circuit, bins = tt.build(domain, cell_tags, facet_tags, uniform)
    delta_SC = ta["delta_SC"]
    eps = float(getattr(params, "ta_eps_reg", 1.0))
    coil = ta["coil_cells"]

    layer_dofs = []
    for i in range(params.n_layers):
        idx = ta["layer_cell_idx"][i]
        cells_i = ta["coil_cells"][idx]
        layer_dofs.append(fem.locate_dofs_topological(
            ta["V_T"], domain.topology.dim, cells_i))

    trace = dict(cum_iter=[], step=[], phase=[], scif_mT=[], dB_rel=[],
                T_max_amp=[[] for _ in range(params.n_layers)],
                T_min_amp=[[] for _ in range(params.n_layers)])
    step_summaries = []
    _cum = {"k": 0}

    def make_closure(circuit_fn, step_idx, phase, T_amp, J_unif_local):
        _prev_B = {"arr": None}

        def closure():
            circuit_fn()
            ta["B_fn"].interpolate(ta["curl_expr"])
            B_now = ta["B_fn"].x.array.reshape(-1, 3)[coil]
            Bn = float(np.linalg.norm(B_now.ravel()))
            dB_rel = (float(np.linalg.norm((B_now - _prev_B["arr"]).ravel())) / Bn
                      if _prev_B["arr"] is not None and Bn > 0 else float("nan"))
            _prev_B["arr"] = B_now.copy()

            J_now_ = ta_solve._J_from_T(ta, domain)
            dJs = (J_now_ - J_unif_local) * (delta_SC / ta["Lambda"])
            scif = float(ta_solve.dB_bore_from_dJ(
                ta["coil_centroids"], dJs, ta["coil_vols"])[2] * 1e3)

            trace["cum_iter"].append(_cum["k"])
            trace["step"].append(step_idx)
            trace["phase"].append(phase)
            trace["scif_mT"].append(scif)
            trace["dB_rel"].append(dB_rel)
            for i, T_i in enumerate(ta["layer_T_fns"]):
                vals = T_i.x.array[layer_dofs[i]] / T_amp
                trace["T_max_amp"][i].append(float(vals.max()))
                trace["T_min_amp"][i].append(float(vals.min()))
            _cum["k"] += 1

        return closure

    for step_idx, (dt, I_now) in enumerate(SCHEDULE):
        first = (step_idx == 0)
        T_amp = I_now / (2.0 * delta_SC)
        ta["dt_const"].value = float(dt)
        J_unif_local = ta["t_hat_coil"] * (I_now / (delta_SC * params.w))

        if first:
            _seed_cold(ta, uniform, max(I_now, 1e-6))
            ta["B_fn"].interpolate(ta["curl_expr"])
            B_coil = ta["B_fn"].x.array.reshape(-1, 3)[coil]
            J_coil = ta["t_hat_coil"] * (I_now / (delta_SC * params.w))
            ta["_rho_prev"] = None
            ta_solve._update_rho(ta, J_coil, B_coil, ic, nm, eps)
        else:
            J_coil = ta_solve._J_from_T(ta, domain)

        J_coil, n1, conv1 = _picard_phase(
            ta, domain, ic, nm, I_now, dt, J_coil,
            closure=make_closure(lambda: circuit.freeze(I_now),
                                 step_idx, "warmup", T_amp, J_unif_local),
            max_iters=max_iters_per_phase, min_iters=max_iters_per_phase,
            scif_tol=0.5, label=f"step{step_idx}_warmup", verbose=False)

        J_coil, n2, conv2 = _picard_phase(
            ta, domain, ic, nm, I_now, dt, J_coil,
            closure=make_closure(lambda: circuit.update(I_now, dt, J_coil),
                                 step_idx, "closure", T_amp, J_unif_local),
            max_iters=max_iters_per_phase, min_iters=max_iters_per_phase,
            scif_tol=0.5, label=f"step{step_idx}_closure", verbose=False)

        finite = (np.all(np.isfinite(ta["A_h"].x.array))
                 and all(np.all(np.isfinite(T_i.x.array))
                        for T_i in ta["layer_T_fns"]))

        final_scif = trace["scif_mT"][-1]
        final_T_max = max(trace["T_max_amp"][i][-1] for i in range(params.n_layers))
        final_T_min = min(trace["T_min_amp"][i][-1] for i in range(params.n_layers))
        cdiag = circuit.diagnostics(I_now)
        step_summaries.append(dict(step=step_idx, dt=dt, I_now=I_now,
                                   n_iters_warmup=n1, n_iters_closure=n2,
                                   finite=bool(finite), scif_mT=final_scif,
                                   T_max_amp=final_T_max, T_min_amp=final_T_min,
                                   **cdiag))
        print(f"  step={step_idx} I={I_now:.1f}A  "
              f"n_iters=warmup{n1}+closure{n2}  finite={finite}  "
              f"scif={final_scif:+.3f}mT  T_max/amp={final_T_max:.3f}  "
              f"T_min/amp={final_T_min:.3f}  "
              f"I_z=[{cdiag['I_z_min']:.2f},{cdiag['I_z_max']:.2f}]A  "
              f"I_r_mean={cdiag['I_r_mean']:.3f}A  "
              f"I_r_frac_max={cdiag['I_r_frac_max']*100:.1f}%  "
              f"n_clipped={cdiag['n_clipped']}",
              flush=True)

        if not finite:
            print(f"  ABORTING: non-finite state at step {step_idx}", flush=True)
            break

        ta["A_prev"].x.array[:] = ta["A_h"].x.array
        ta["A_prev"].x.scatter_forward()
        circuit.end_step()

    os.makedirs(OUT_DATA_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DATA_DIR, out_name)

    save_dict = dict(
        schedule=SCHEDULE, alpha_high=alpha_high, alpha_low=alpha_low,
        max_iters_per_phase=max_iters_per_phase,
        cum_iter=np.array(trace["cum_iter"]),
        step_of_iter=np.array(trace["step"]),
        phase_of_iter=np.array(trace["phase"]),
        scif_mT=np.array(trace["scif_mT"]),
        dB_rel=np.array(trace["dB_rel"]),
        n_layers=params.n_layers,
        n_turns=np.array(params.n_turns),
        step_summaries=step_summaries,
    )
    for i in range(params.n_layers):
        save_dict[f"T_max_amp_layer{i}"] = np.array(trace["T_max_amp"][i])
        save_dict[f"T_min_amp_layer{i}"] = np.array(trace["T_min_amp"][i])

    np.savez(out_path, **save_dict)
    print(f"\nSaved NI closure smoke test data to {out_path}", flush=True)
    print(f"completed_all_steps={len(step_summaries) == len(SCHEDULE)}", flush=True)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
