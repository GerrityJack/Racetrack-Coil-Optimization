"""
adaptive_march.py -- adaptive-dt multi-step time marching for the base
(INSULATED, no NI circuit closure) T-A Picard solver.

WHY THIS EXISTS
---------------
Every multi-step attempt in this project's transient/ history so far
(newton_ta.march()/hybrid_march(), the overnight Newton validation) used a
FIXED, hand-picked dt for every step of a schedule. The one thing never
actually tried is what published T-A/H-formulation transient simulations
normally do: march with MANY steps whose SIZE is controlled automatically
by how hard the solver found the previous step -- shrink dt when Newton/
Picard struggles, grow it back when it doesn't. This is the standard
"Newton-iteration-count-based" adaptive step-size heuristic used throughout
nonlinear implicit PDE time-stepping (see e.g. any adaptive backward-Euler
solver for stiff ODEs/PDEs).

This module deliberately reuses the EXISTING, validated per-step Picard
machinery (`ta_transient._picard_phase`, unmodified) rather than any of the
newer Newton-based schemes -- the point of this experiment is to isolate
whether SMALL ENOUGH STEPS ALONE stabilize the ALREADY-VALIDATED solver,
independent of any of this week's more exotic coupling changes. It also
deliberately does NOT wire in the NI circuit closure (`per_turn_bc=False`,
plain Constant BCs) -- isolating "does adaptive stepping fix convergence"
from "does the NI closure introduce its OWN instability on top of it" (a
separate, already-partially-investigated question).

ALGORITHM
---------
Classic step-doubling/halving control on the per-step Picard iteration
count:
  - Propose a step of size `dt` toward the ramp target.
  - Run `_picard_phase` with a bounded iteration budget.
  - If it converges in `<= iters_low` iterations: ACCEPT, grow dt for the
    NEXT step (dt *= grow, capped at dt_max).
  - If it converges but needed MORE than `iters_low`: ACCEPT, leave dt
    unchanged.
  - If it does NOT converge within the budget: REJECT -- revert every
    piece of state the step touched (T per layer, A, the Picard relaxation
    history array `ta["_rho_prev"]`), shrink dt (dt *= shrink, floored at
    dt_min), and retry the SAME point in the ramp with the smaller step.
  - Gives up (raises) only if dt would need to shrink below dt_min without
    converging -- a genuine, informative failure, not silently swallowed.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params                                  # noqa: E402


class AdaptiveMarchFailure(RuntimeError):
    """Raised when dt shrinks below dt_min without a step converging."""


def adaptive_march(ta, domain, uniform_setup, ic_model, n_model,
                   I_target, t_ramp, t_hold=0.0,
                   dt_init=30.0, dt_min=2.0, dt_max=600.0,
                   grow=1.5, shrink=0.5, iters_low=15,
                   max_iters_per_step=150, min_iters_per_step=6,
                   scif_tol=0.5, max_rejects_per_step=10, verbose=True):
    """Adaptive-dt march of the insulated base Picard solver from ZFC to
    I_target over t_ramp seconds, then (optionally) hold at I_target for
    t_hold more seconds. Returns (history, final_scif_mT) where history is
    a list of per-ACCEPTED-step diagnostics dicts.
    """
    import ta_solve
    from ta_transient import _picard_phase, _seed_cold

    B_h = ta["B_fn"]
    coil = ta["coil_cells"]
    delta_SC = ta["delta_SC"]
    eps = float(getattr(params, "ta_eps_reg", 1.0))

    t = 0.0
    I_cur = 0.0
    dt = float(dt_init)
    first = True
    history = []
    n_rejects_total = 0

    t_end = t_ramp + t_hold

    def _I_of_t(tt):
        if tt <= t_ramp:
            return I_target * tt / t_ramp
        return I_target

    while t < t_end - 1e-9:
        dt_try = min(dt, t_end - t)
        I_next = _I_of_t(t + dt_try)
        n_reject_this_step = 0

        while True:
            # ── snapshot everything this trial step will touch ──────────
            T_snap = [T_i.x.array.copy() for T_i in ta["layer_T_fns"]]
            A_snap = ta["A_h"].x.array.copy()
            Aprev_snap = ta["A_prev"].x.array.copy()
            rho_prev_snap = (ta["_rho_prev"].copy()
                             if ta.get("_rho_prev") is not None else None)

            if first:
                J_coil = _seed_cold(ta, uniform_setup, max(I_next, 1e-6))
                B_h.interpolate(ta["curl_expr"])
                B_seed = B_h.x.array.reshape(-1, 3)[coil]
                ta["_rho_prev"] = None
                ta_solve._update_rho(ta, J_coil, B_seed, ic_model, n_model, eps)
            else:
                J_coil = ta_solve._J_from_T(ta, domain)

            T_amp = I_next / (2.0 * delta_SC)
            ta["T_bot_val"].value = +T_amp
            ta["T_top_val"].value = -T_amp
            # _picard_phase takes dt only to pass through to `closure`
            # (the NI circuit update) -- it does NOT set the FEM dt_const
            # itself (ta_transient.step() does that one line before calling
            # it). Omitting this line means every step silently reuses
            # whatever dt_const was last set to instead of dt_try -- caught
            # by a suspicious "iteration count barely depends on dt" pattern
            # across the first two runs of this script.
            ta["dt_const"].value = float(dt_try)

            J_coil, n_iters, converged = _picard_phase(
                ta, domain, ic_model, n_model, I_next, dt_try, J_coil,
                closure=lambda: None, max_iters=max_iters_per_step,
                min_iters=min_iters_per_step, scif_tol=scif_tol,
                label=f"t={t+dt_try:7.1f}s dt={dt_try:6.1f}s I={I_next:6.2f}A",
                verbose=verbose)

            finite = (np.all(np.isfinite(ta["A_h"].x.array))
                      and all(np.all(np.isfinite(T_i.x.array))
                             for T_i in ta["layer_T_fns"]))
            ok = converged and finite

            if ok:
                break

            # ── reject: revert, shrink, retry the SAME point in the ramp ─
            for T_i, snap in zip(ta["layer_T_fns"], T_snap):
                T_i.x.array[:] = snap
                T_i.x.scatter_forward()
            ta["A_h"].x.array[:] = A_snap
            ta["A_h"].x.scatter_forward()
            ta["A_prev"].x.array[:] = Aprev_snap
            ta["A_prev"].x.scatter_forward()
            ta["_rho_prev"] = rho_prev_snap

            n_reject_this_step += 1
            n_rejects_total += 1
            if verbose:
                print(f"    [adaptive] REJECT at t={t:.1f}s dt={dt_try:.1f}s "
                      f"(converged={converged} finite={finite}, "
                      f"n_iters={n_iters}) -- shrinking", flush=True)

            if dt_try <= dt_min + 1e-9:
                raise AdaptiveMarchFailure(
                    f"step at t={t:.1f}s did not converge even at the dt "
                    f"floor ({dt_min}s) after {n_reject_this_step} rejects "
                    f"this step ({n_rejects_total} total)")
            if n_reject_this_step >= max_rejects_per_step:
                raise AdaptiveMarchFailure(
                    f"step at t={t:.1f}s failed {max_rejects_per_step} "
                    f"times without converging (dt down to {dt_try:.2f}s)")

            dt_try = max(dt_try * shrink, dt_min)
            dt_try = min(dt_try, t_end - t)
            I_next = _I_of_t(t + dt_try)

        # ── accept ────────────────────────────────────────────────────────
        ta["A_prev"].x.array[:] = ta["A_h"].x.array
        ta["A_prev"].x.scatter_forward()

        J_coil = ta_solve._J_from_T(ta, domain)
        J_unif = ta["t_hat_coil"] * (I_next / (delta_SC * params.w))
        dJs = (J_coil - J_unif) * (delta_SC / ta["Lambda"])
        scif = ta_solve.dB_bore_from_dJ(ta["coil_centroids"], dJs,
                                        ta["coil_vols"])[2] * 1e3

        t += dt_try
        I_cur = I_next
        first = False

        if verbose:
            print(f"  [adaptive] ACCEPT t={t:7.1f}s  I={I_cur:6.2f}A  "
                  f"dt_used={dt_try:6.1f}s  n_iters={n_iters}  "
                  f"SCIF={scif:+8.2f} mT  next_dt={min(dt if n_iters > iters_low else dt*grow, dt_max):6.1f}s",
                  flush=True)

        history.append(dict(t=t, I=I_cur, dt=dt_try, n_iters=n_iters,
                            n_rejects_this_step=n_reject_this_step,
                            scif_mT=float(scif)))

        if n_iters <= iters_low:
            dt = min(dt * grow, dt_max)
        # else: leave dt unchanged for the next step

    return history, history[-1]["scif_mT"] if history else float("nan")
