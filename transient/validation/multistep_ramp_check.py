"""multistep_ramp_check.py -- 2026-08-06, Phase 2 of the "thorough test"
directive: does alpha=(0.03, 0.01) work across a GENUINE multi-step
ramp, not just a single first step from cold start? Everything in this
project's nondeterminism/monolithic investigations, and everything
tested so far with the alpha fix, has only ever been ONE step from ZFC
(T=0, A=0). This is a real gap -- the project's original motivating
problem was genuine multi-step time-marching.

Deliberately INSULATED (no NI circuit closure) -- that is a separate,
additional layer of uncertainty this session has not touched (CLAUDE.md
flags `transient/`'s NI-coupled step()/march() as its own exploratory,
not-validated-for-general-use territory). This reuses `_picard_phase`
unmodified, `closure=lambda: None`, exactly matching the insulated-limit
scope `first_step_diagnostic.py` and every alpha-fix test so far has
used -- just called REPEATEDLY, with `ta["A_prev"]` carried forward
between steps (mirroring `ta_transient.step()`'s own bookkeeping:
`ta["A_prev"].x.array[:] = ta["A_h"].x.array` after each step) instead
of staying at zero, which is what makes this a genuine multi-step test
instead of N independent first-steps.

Schedule: 5 steps, all at dt=60s (the established hard regime), current
stepping I=19.6, 39.2, 58.8, 78.4, 98.0 A (5 equal +19.6A increments,
reaching half the champion design current by the end) -- a genuinely
fast/short-dt ramp, the real-world scenario this project has never
gotten working before.

Usage: <env python> multistep_ramp_check.py <alpha_high> <alpha_low>
                    <max_iters_per_step> <out_json_path>
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

SCHEDULE = [(60.0, 19.6), (60.0, 39.2), (60.0, 58.8), (60.0, 78.4), (60.0, 98.0)]


def main():
    alpha_high = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
    alpha_low = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    max_iters_per_step = int(sys.argv[3]) if len(sys.argv) > 3 else 800
    out_path = sys.argv[4] if len(sys.argv) > 4 else "/tmp/multistep_ramp_result.json"

    import numpy as np
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    from ta_transient import _picard_phase, _seed_cold

    params.ta_picard_alpha = alpha_high
    params.ta_picard_alpha_fine = alpha_low

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_ramp_{os.getpid()}{ext}"
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
    coil = ta["coil_cells"]

    # first step: cold (ZFC) seed, exactly as every other script this session
    J_coil = _seed_cold(ta, uniform, max(SCHEDULE[0][1], 1e-6))
    ta["B_fn"].interpolate(ta["curl_expr"])
    B_seed = ta["B_fn"].x.array.reshape(-1, 3)[coil]
    ta["_rho_prev"] = None
    ta_solve._update_rho(ta, J_coil, B_seed, ic, nm, eps)

    steps_out = []
    for step_idx, (dt, I_now) in enumerate(SCHEDULE):
        T_amp = I_now / (2.0 * delta_SC)
        ta["T_bot_val"].value = +T_amp
        ta["T_top_val"].value = -T_amp
        ta["dt_const"].value = float(dt)

        # J_coil carried forward from the end of the previous step (or the
        # cold seed, for step 0) via the T state -- NOT reset.
        J_coil = ta_solve._J_from_T(ta, domain)

        J_unif = ta["t_hat_coil"] * (I_now / (delta_SC * params.w))

        J_coil, n_iters, converged = _picard_phase(
            ta, domain, ic, nm, I_now, dt, J_coil,
            closure=lambda: None, max_iters=max_iters_per_step, min_iters=6,
            scif_tol=0.5, label=f"step{step_idx}", verbose=False)

        finite = (np.all(np.isfinite(ta["A_h"].x.array))
                 and all(np.all(np.isfinite(T_i.x.array))
                        for T_i in ta["layer_T_fns"]))

        ta["B_fn"].interpolate(ta["curl_expr"])
        B_coil = ta["B_fn"].x.array.reshape(-1, 3)[coil]
        dJs = (J_coil - J_unif) * (delta_SC / ta["Lambda"])
        scif = float(ta_solve.dB_bore_from_dJ(
            ta["coil_centroids"], dJs, ta["coil_vols"])[2] * 1e3)
        T_max = float(max(T_i.x.array.max() for T_i in ta["layer_T_fns"]))
        T_min = float(min(T_i.x.array.min() for T_i in ta["layer_T_fns"]))

        step_result = dict(step=step_idx, dt=dt, I_now=I_now,
                           converged=bool(converged), n_iters=int(n_iters),
                           finite=bool(finite), scif_mT=scif,
                           T_max_over_amp=T_max / T_amp,
                           T_min_over_amp=T_min / T_amp)
        steps_out.append(step_result)
        print(f"  step={step_idx} dt={dt} I={I_now:.1f}A  converged={converged}  "
              f"n_iters={n_iters}  finite={finite}  scif={scif:+.3f}mT  "
              f"T_max/amp={T_max/T_amp:.3f}  T_min/amp={T_min/T_amp:.3f}",
              flush=True)

        if not finite:
            print(f"  ABORTING ramp: non-finite state at step {step_idx}",
                  flush=True)
            break

        # advance history for the NEXT step's BDF1 term -- the actual
        # mechanic that makes this a multi-step test, not N independent
        # first-steps (mirrors ta_transient.step()'s own bookkeeping).
        ta["A_prev"].x.array[:] = ta["A_h"].x.array
        ta["A_prev"].x.scatter_forward()

    all_finite = all(s["finite"] for s in steps_out)
    all_converged = all(s["converged"] for s in steps_out)
    completed_all_steps = len(steps_out) == len(SCHEDULE)

    print(f"FINAL: completed_all_steps={completed_all_steps}  "
          f"all_finite={all_finite}  all_converged={all_converged}", flush=True)

    with open(out_path, "w") as fh:
        json.dump(dict(alpha_high=alpha_high, alpha_low=alpha_low,
                       max_iters_per_step=max_iters_per_step,
                       schedule=SCHEDULE, steps=steps_out,
                       completed_all_steps=completed_all_steps,
                       all_finite=all_finite, all_converged=all_converged),
                 fh, indent=2)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
