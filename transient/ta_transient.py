"""
ta_transient.py — time-stepped T-A with the no-insulation circuit closure.

This is the first genuinely transient model in the repo.  ta_solve.py takes a
SINGLE implicit-Euler step from the zero-field-cooled state to the end of the
ramp; here that same machinery is marched in time by advancing ta["A_prev"]
after each step, which turns ta_solve's history term

    L_T = -(1/dt) * coil_ind * inner(curl(A_h - A_prev), phi*n_hat) * dx

from "one step from B = 0" into a real BDF1 sequence.  Nothing else in
ta_solve changes; the solver objects, the cached MUMPS factorisation of the
constant A-matrix, _J_from_T, _update_rho, _update_Js and _solve_A are all
reused exactly as the steady-state path uses them.

Per step the Picard loop gains ONE extra update, placed first because it
changes the T Dirichlet data:

    ni_circuit.update()   ->  I_z per radial bin  ->  tape-edge BCs
    solve T (per layer)
    J = _J_from_T
    _update_Js ; _solve_A
    B_h.interpolate(curl_expr)
    _update_rho

Warm continuation is what makes this affordable: T, rho and A are NOT reset
between steps, so a step converges in a handful of iterations instead of the
~80 a cold solve needs.
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve"),
           os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params                      # noqa: E402
import tparams as tp               # noqa: E402
import ta_solve                    # noqa: E402
from turn_map import RadialBins, build_cell_map   # noqa: E402
from ni_circuit import NICircuit                  # noqa: E402


def build(domain, cell_tags, facet_tags, uniform_setup,
          rho_ct_uohm_cm2=None, n_bins=None, ep_factor_mode=None):
    """Set up the transient problem.  Returns (ta, circuit, bins)."""
    rho_ct = (rho_ct_uohm_cm2 if rho_ct_uohm_cm2 is not None
              else tp.RHO_CT_UOHM_CM2)
    n_bins = n_bins or tp.N_RADIAL_BINS

    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags,
                                   uniform_setup, per_layer=True,
                                   per_turn_bc=True)
    bins = RadialBins(n_bins)
    _, groups = build_cell_map(bins, ta["coil_centroids"])
    empty = sum(1 for g in groups if g.size == 0)
    if empty:
        print(f"  [transient] NOTE {empty}/{bins.n_flat} radial bins contain "
              f"no coil cells (mesh coarser than the binning) — they carry "
              f"no E_p/E_i average and simply follow the transport current")
    circuit = NICircuit(ta, bins, groups, rho_ct, ep_factor_mode)
    return ta, circuit, bins


def _seed_cold(ta, uniform_setup, I0):
    """Zero-field-cooled start: A from the uniform-J solve, T = 0."""
    J_mag = I0 / (params.t * params.w)
    uniform_setup["J"].x.array[:] = uniform_setup["J_unit_array"] * J_mag
    ta_solve._solve_A(ta, ta["L_seed_form"])
    ta["T_h"].x.array[:] = 0.0
    for T_i in ta["layer_T_fns"]:
        T_i.x.array[:] = 0.0
        T_i.x.scatter_forward()
    J_seed = ta["t_hat_coil"] * (I0 / (ta["delta_SC"] * params.w))
    ta_solve._update_Js(ta, J_seed)
    return J_seed


def _picard_phase(ta, domain, ic_model, n_model, I_now, dt, J_coil,
                  closure, max_iters, min_iters, scif_tol, label,
                  verbose=True):
    """Run relaxed Picard iterations to convergence, with the SAME two-phase
    relaxation and observable-stall criterion as ta_solve.solve_ta_at_current
    (phase 1: alpha_high=0.30 fast ramp-up until |dB| stops decreasing;
    phase 2: alpha_low=0.15 fixed; convergence = EMA-smoothed bore SCIF
    stalls < scif_tol over a 6-iteration window, floor min_iters).

    2026-08-04: this function EXISTS because the original transient step()
    used only alpha_low throughout, including the warmup phase meant to let
    T/A/rho settle before the NI closure switches on. Without the fast
    ramp-up, a fixed 30-iteration warmup was nowhere near enough -- the
    "frozen" (insulated-equivalent) state was still in its own early,
    chaotically-swinging transient (SCIF swinging thousands of mT between
    iterations, the exact pattern ta_solve's own k=1-10 shows before its
    phase-2 switch) when the closure was switched on. That fed a genuinely
    unconverged, wildly fluctuating E_p into the closure, which is what
    validation/ni_closure_stability_check.py's persistent oscillation
    actually was -- not a defect in the closure's own math.

    `closure` is a 0-argument callable invoked first each iteration: either
    ``lambda: circuit.freeze(I_now)`` (warmup) or
    ``lambda: circuit.update(I_now, dt, J_coil)`` (closure-coupled). J_coil
    is read by that closure, so it must be kept in sync with the outer
    variable via the returned value -- see step() below.

    Returns (J_coil, n_iters, converged).
    """
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
        closure()

        for i, (T_i, prob) in enumerate(zip(ta["layer_T_fns"],
                                            ta["prob_T_layers"])):
            sol = prob.solve()
            if np.any(np.isnan(sol.x.array)):
                raise RuntimeError(
                    f"NaN in T solve ({label}), layer {i}, iter {k}")

            # DIAGNOSTIC HOOK (2026-08-05, nondeterminism investigation) --
            # inert unless DUMP_ITER1_MATRIX_PATH is set. Dumps the FIRST
            # Picard iteration's assembled T-equation matrix (CSR) and RHS
            # for every layer, so two separate process launches of the
            # identical nominal configuration can be bit-diffed at the
            # earliest possible point, before any Picard nonlinear
            # amplification has a chance to act.
            if k == 0:
                import os as _os
                _dump = _os.environ.get("DUMP_ITER1_MATRIX_PATH")
                if _dump:
                    import numpy as _np
                    ai, aj, av = prob.A.getValuesCSR()
                    bv = prob.b.getArray(readonly=True).copy()
                    _np.savez(f"{_dump}_layer{i}.npz",
                             indptr=ai, indices=aj, data=av, rhs=bv,
                             sol=sol.x.array.copy())

            T_i.x.array[:] = (1.0 - alpha) * T_i.x.array + alpha * sol.x.array
            T_i.x.scatter_forward()

        J_coil = ta_solve._J_from_T(ta, domain)
        ta_solve._update_Js(ta, J_coil)
        ta_solve._solve_A(ta, ta["L_A_form"])
        if np.any(np.isnan(ta["A_h"].x.array)):
            raise RuntimeError(f"NaN in A solve ({label}), iter {k}")

        B_h.interpolate(ta["curl_expr"])
        B_coil = B_h.x.array.reshape(-1, 3)[coil]
        ta_solve._update_rho(ta, J_coil, B_coil, ic_model, n_model, eps,
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
        if verbose and os.environ.get("NI_TRACE") == "1":
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


def step(ta, circuit, domain, ic_model, n_model, I_now, dt,
         n_picard, first=False, verbose=True):
    """Advance one time step.  Returns a diagnostics dict.

    Two phases, each run to genuine convergence (not a fixed iteration
    count) via _picard_phase(): first with the circuit FROZEN at the
    insulated state (so T/A/rho settle under the new BCs/dt before the NI
    closure is switched on), then with the closure active. See
    _picard_phase's docstring for why the frozen phase must itself be run
    with the base solver's proper two-phase relaxation, not a shortcut.
    """
    ta["dt_const"].value = float(dt)
    B_h = ta["B_fn"]
    coil = ta["coil_cells"]
    eps = float(getattr(params, "ta_eps_reg", 1.0))

    B_h.interpolate(ta["curl_expr"])
    B_coil = B_h.x.array.reshape(-1, 3)[coil]
    J_coil = ta_solve._J_from_T(ta, domain)
    if first:
        # nothing meaningful in T yet — seed rho from the uniform current
        J_coil = ta["t_hat_coil"] * (I_now / (ta["delta_SC"] * params.w))
        ta["_rho_prev"] = None
        ta_solve._update_rho(ta, J_coil, B_coil, ic_model, n_model, eps)

    t0 = time.time()
    warmup_cap = tp.NI_WARMUP_ITERS_FIRST if first else tp.NI_WARMUP_ITERS
    warmup_min = min(warmup_cap, tp.STEP_MIN_ITERS)

    J_coil, n1, conv1 = _picard_phase(
        ta, domain, ic_model, n_model, I_now, dt, J_coil,
        closure=lambda: circuit.freeze(I_now),
        max_iters=warmup_cap, min_iters=warmup_min,
        scif_tol=tp.STEP_STALL_MT, label="warmup", verbose=verbose)

    remaining = max(tp.STEP_MIN_ITERS, n_picard - n1)
    J_coil, n2, conv2 = _picard_phase(
        ta, domain, ic_model, n_model, I_now, dt, J_coil,
        closure=lambda: circuit.update(I_now, dt, J_coil),
        max_iters=remaining, min_iters=tp.STEP_MIN_ITERS,
        scif_tol=tp.STEP_STALL_MT, label="closure", verbose=verbose)

    info = dict(converged=conv2, n_iters=n1 + n2, n_iters_warmup=n1,
               n_iters_closure=n2, warmup_converged=conv1)
    info["scif_mT"] = float(ta_solve.dB_bore_from_dJ(
        ta["coil_centroids"],
        (J_coil - ta["t_hat_coil"] * (I_now / (ta["delta_SC"] * params.w)))
        * (ta["delta_SC"] / ta["Lambda"]), ta["coil_vols"])[2] * 1e3)
    info["wall_s"] = time.time() - t0
    info.update(circuit.diagnostics(I_now))

    # advance history: A_prev for the T equation's BDF1 term, and the
    # circuit's own I_z latch for its dI/dt
    ta["A_prev"].x.array[:] = ta["A_h"].x.array
    ta["A_prev"].x.scatter_forward()
    circuit.end_step()

    if verbose:
        print(f"    k={info['n_iters']:3d} "
              f"(warmup {n1}{'*' if not conv1 else ''} + "
              f"closure {n2}{'*' if not conv2 else ''})  "
              f"{'conv' if info['converged'] else 'CAP '} "
              f"SCIF={info['scif_mT']:+8.2f} mT  "
              f"I_z=[{info['I_z_min']:7.2f},{info['I_z_max']:7.2f}] A  "
              f"I_r_mean={info['I_r_mean']:7.3f} A  "
              f"clip={info['n_clipped']:2d}  "
              f"({info['wall_s']:.1f} s)", flush=True)
    return info


def march(ta, circuit, domain, uniform_setup, ic_model, n_model,
          schedule, verbose=True):
    """Run a full schedule.  `schedule` is a list of (t, I, dt) triples."""
    hist = []
    for n, (t, I_now, dt) in enumerate(schedule):
        if verbose:
            print(f"  step {n+1}/{len(schedule)}  t={t:7.1f} s  "
                  f"I={I_now:7.2f} A  dt={dt:6.1f} s", flush=True)
        if n == 0:
            _seed_cold(ta, uniform_setup, max(I_now, 1e-6))
        info = step(ta, circuit, domain, ic_model, n_model, I_now, dt,
                    n_picard=(tp.N_PICARD_FIRST if n == 0
                              else tp.N_PICARD_STEP),
                    first=(n == 0), verbose=verbose)
        info.update(t=t, I=I_now, dt=dt,
                    I_z=circuit.I_z.copy(), I_r=circuit.I_r.copy())
        hist.append(info)
    return hist


def ramp_schedule(I_op, t_ramp=None, t_hold=None, n_ramp=None, n_hold=None):
    t_ramp = t_ramp if t_ramp is not None else tp.RAMP_S
    t_hold = t_hold if t_hold is not None else tp.HOLD_S
    # "x or default" is WRONG for these -- 0 is a legitimate, meaningful
    # value (an explicit "no hold phase"/"no ramp phase" request) and is
    # falsy, so it was silently replaced by the tparams default. Caught by
    # a real crash: n_hold=0 (requesting a ramp-only schedule) fell back to
    # tp.N_STEPS_HOLD extra steps, each with dt = t_hold/n_hold = 0.0/N = 0,
    # which then hit a 1/dt term in the T-equation and produced NaN.
    n_ramp = n_ramp if n_ramp is not None else tp.N_STEPS_RAMP
    n_hold = n_hold if n_hold is not None else tp.N_STEPS_HOLD
    out = []
    dt_r = t_ramp / n_ramp
    for i in range(1, n_ramp + 1):
        out.append((i * dt_r, I_op * i / n_ramp, dt_r))
    dt_h = t_hold / max(n_hold, 1)
    for i in range(1, n_hold + 1):
        out.append((t_ramp + i * dt_h, I_op, dt_h))
    return out
