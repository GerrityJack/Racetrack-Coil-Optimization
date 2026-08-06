"""jitter_retry_check.py -- 2026-08-06, prototype of the "controlled
perturbation as a designed escape mechanism" idea.

Background: nondeterminism_investigation_2026-08-05.md's canonical repro
case (dt=60s, I=19.6A, cold start, run via first_step_diagnostic.py)
succeeds ~40% of the time under normal (multi-threaded, ambient
floating-point noise) execution, but genuinely single-threaded execution
makes it PERFECTLY deterministic -- and deterministically non-convergent
(0/18). The uncontrolled thread-scheduling noise was, by accident, the
only thing occasionally kicking the trajectory off its one doomed
deterministic path and onto a different (sometimes converging) one.

This script replaces "gamble on whichever thread interleaving happens to
occur" with a DESIGNED, reproducible-by-seed escape mechanism: attempt 0
is the unmodified baseline (identical to first_step_diagnostic.py, for
direct comparability with the historical ~40% figure); if it fails to
converge, retry from a freshly cold-reset state with a small EXPLICIT
random perturbation added to the T seed, drawn from a logged seed, up to
`max_retries` times. Does NOT force single-threaded execution -- this
tests the practical question (does wrapping the existing, already
somewhat-noisy execution in a retry-with-jitter loop reliably produce a
converged result within ONE process launch), not an isolated test of
jitter alone.

Jitter scale: 1e-3 (0.1%) of T_amp by default -- roughly 11 orders of
magnitude larger than the ~1e-14 ambient floating-point noise floor this
project has characterised elsewhere, chosen so the injected perturbation
clearly dominates over (and is not confounded with) whatever ambient
thread noise exists, while remaining a genuinely small fraction of the
physical BC scale.

Usage: <env python> jitter_retry_check.py <dt> <I_target> <max_retries>
                                          <jitter_scale> <out_json_path>
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
    max_retries = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    jitter_scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
    out_path = sys.argv[5] if len(sys.argv) > 5 else "/tmp/jitter_retry_result.json"

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
    params.mesh_filename = f"{root}_jitter_{os.getpid()}{ext}"
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

    # one base seed per PROCESS LAUNCH, logged so any run is replayable --
    # os.urandom, not a fixed default, so repeated launches of this script
    # (the realistic "each production attempt tries its luck" scenario)
    # draw independent jitter sequences.
    base_seed = int.from_bytes(os.urandom(4), "little")

    results = []
    success = False
    for attempt in range(max_retries + 1):  # attempt 0 = unmodified baseline
        J_coil = _seed_cold(ta, uniform, max(I_target, 1e-6))
        if attempt > 0:
            rng = np.random.default_rng(base_seed * 1000 + attempt)
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

        J_coil, n_iters, converged = _picard_phase(
            ta, domain, ic, nm, I_target, dt, J_coil,
            closure=lambda: None, max_iters=150, min_iters=6, scif_tol=0.5,
            label=f"attempt{attempt}", verbose=False)

        finite = (np.all(np.isfinite(ta["A_h"].x.array))
                 and all(np.all(np.isfinite(T_i.x.array))
                        for T_i in ta["layer_T_fns"]))

        # Record the final SCIF regardless of convergence status -- the
        # question this exists to answer (does a jitter-forced "converged"
        # run land on the SAME physical answer as another one, or just on
        # SOME state that happens to satisfy the stall criterion) needs
        # this compared ACROSS attempts/launches, not just the pass/fail
        # flag.
        J_unif = ta["t_hat_coil"] * (I_target / (delta_SC * params.w))
        dJs = (J_coil - J_unif) * (delta_SC / ta["Lambda"])
        scif_mT = float(ta_solve.dB_bore_from_dJ(
            ta["coil_centroids"], dJs, ta["coil_vols"])[2] * 1e3)

        ok = bool(converged) and bool(finite)
        results.append(dict(attempt=attempt, converged=bool(converged),
                            n_iters=int(n_iters), finite=bool(finite),
                            scif_mT=scif_mT))
        print(f"  attempt={attempt}  converged={converged}  n_iters={n_iters}  "
              f"finite={finite}  scif={scif_mT:+.3f}mT", flush=True)
        if ok:
            success = True
            break

    with open(out_path, "w") as fh:
        json.dump(dict(dt=dt, I_target=I_target, max_retries=max_retries,
                       jitter_scale=jitter_scale, base_seed=base_seed,
                       success=success, n_attempts_used=len(results),
                       attempts=results), fh, indent=2)

    print(f"FINAL: success={success} after {len(results)} attempt(s) "
          f"(base_seed={base_seed})", flush=True)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
