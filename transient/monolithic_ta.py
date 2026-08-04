"""
monolithic_ta.py -- ONE coupled nonlinear residual over every T-layer AND A
simultaneously, as a further step past newton_ta.py.

DOES NOT MODIFY solve/ta_solve.py or transient/newton_ta.py.  Purely
additive: consumes the SAME `ta` dict setup_ta_problem() builds and adds a
new block system alongside whatever newton_ta.py / ta_solve.py already put
there.

WHY THIS EXISTS
---------------
newton_ta.py's per-layer Newton solve fixed the DOMINANT nonlinearity (the
power-law rho(J), n~13-27) decisively -- 11 SNES iterations from a cold
start on the hardest case this project has tried.  But its outer loop is
still a block GAUSS-SEIDEL scheme, not a monolithic one: each layer's T is
solved to ITS OWN tight convergence with A held FROZEN, then A is resolved
with every T held frozen, and this repeats.  The T<->A coupling and the
layer<->layer coupling (which only interact THROUGH the shared A) are both
handled by outer-loop iteration, not by a single linear solve.  CLAUDE.md's
transient/ investigation history traces the project's worst instabilities to
exactly this class of outer-loop staleness -- most sharply, the finding that
a scheme which updates Jc(B)/n(B) once per OUTER sweep can drift to a
plausible-looking but wrong fixed point (t_relax: converged cleanly to
-24 mT against a true answer of +641.26 mT) while the SAME physics, given
the SAME state, corrects itself in 44 iterations once handed back to a
scheme with less staleness between the field state and the coefficients
read from it (see the "picard_from_newton_state.py" entry in docs/HISTORY.md
for the transplant experiment that established this).

This module removes BOTH sources of staleness in one step:
  1. T-A coupling: T_0..T_{L-1} and A are unknowns in ONE PETSc SNES block
     system (`kind="mpi"`), so the Jacobian includes the true cross-terms
     dF_A/dT_i and dF_Ti/dA every single Newton step, instead of alternating
     frozen sub-solves.
  2. Jc(B)/n(B) staleness: refreshed from the CURRENT block iterate's B
     field before EVERY Newton step (see "single-step-then-refresh" below),
     not once per outer Gauss-Seidel sweep.

STILL QUASI-NEWTON, SAME REASON AS newton_ta.py
------------------------------------------------
Ic(B,theta)/n(B,theta) come from measured-CSV scipy splines
(physics/ic_model.py), not a UFL-differentiable expression -- ufl.derivative
cannot form d(Jc)/dB or d(n)/dB symbolically.  Jc/n are still FROZEN DG0
coefficients within any single Newton step, exactly as in newton_ta.py.
What's new here is WHEN they get refreshed (every step, not every outer
Gauss-Seidel sweep) and HOW MUCH of the system a single linear solve covers
(everything, not one layer at a time).

SINGLE-STEP-THEN-REFRESH -- HOW THE COEFFICIENT REFRESH IS WIRED
-------------------------------------------------------------------
PETSc's SNESSetUpdate() callback ordering relative to dolfinx's internal
Function<->PETSc-vector scatter is not something to rely on blindly.
Instead: `snes.setTolerances(max_it=1)` forces exactly ONE Newton step per
`problem.solve()` call (confirmed directly on a toy 2-unknown coupled system
before writing this: repeated max_it=1 solves warm-start correctly from the
Function values left by the previous call and converge to the analytically
correct fixed point in ~8 calls). Between calls, ordinary Python code reads
the just-updated T_i/A_h Functions, recomputes B = curl(A_h), and rewrites
Jc_fn/n_fn from the measured Ic(B,theta)/n(B,theta) models -- no reliance on
any PETSc callback-ordering assumption.  `getConvergedReason()` reports -5
(SNES_DIVERGED_MAX_IT) on every call that hasn't yet met its own rtol/atol
within that one step -- THIS IS EXPECTED, not a failure, and must not be
treated as one (confirmed on the toy system: -5 on 8 consecutive calls,
then reason=2, exactly matching the analytically known fixed point).  Only
a genuinely different negative reason (linear solve failure, line-search
divergence, NaN) is treated as a real per-step failure.

Convergence of the OUTER (Python-level) loop is judged the same way every
other solver in this project judges it: EMA-smoothed bore SCIF stall, not
the raw B-field residual and not SNES's own per-step status -- consistent
with the standing project lesson that a solver's own "converged" flag is
not the same claim as "this is the right answer."
"""

import os
import sys

import numpy as np
import ufl

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params                                          # noqa: E402
from ic_model import angle_with_normal_deg              # noqa: E402

mu0 = 4.0 * np.pi * 1e-7

# Same fix as newton_ta.py -- FFCx cannot estimate a sane quadrature degree
# for the transcendental exp(log(...)) power-law form.
MONO_QUADRATURE_DEGREE = 2

# Safe nonzero placeholders outside a layer's own cells (T is Dirichlet-
# pinned to zero there, so the physical contribution is exactly zero
# regardless of what these evaluate to -- see newton_ta.py gotcha #2).
_JC_SAFE_DEFAULT = 1.0
_N_SAFE_DEFAULT = 2.0

DEFAULT_SNES_OPTIONS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_linesearch_max_it": 40,
    # rtol/atol are close to irrelevant with max_it forced to 1 by the
    # caller (see module docstring) -- the outer Python loop's SCIF-stall
    # criterion is the real convergence check. Kept loose for the same
    # reason newton_ta.py loosened them: a tight inner tolerance chasing
    # noise once already close to the fixed point risks spurious
    # backtracking-linesearch failures for no accuracy benefit.
    "snes_rtol": 1e-6,
    "snes_atol": 1e-8,
    "snes_stol": 1e-10,
    "snes_max_it": 1,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def build_monolithic_problem(ta, snes_options=None, verbose=False):
    """Build ONE block NonlinearProblem over [T_layer_0, ..., T_layer_{L-1}, A].

    Adds new keys to `ta` (does not touch anything setup_ta_problem(),
    ta_solve.py's Picard path, or newton_ta.py's per-layer Newton path
    already put there):
        ta["mono_Jc_fns"] : list of DG0 Functions, one per layer
        ta["mono_n_fns"]  : list of DG0 Functions, one per layer
        ta["mono_problem"]: the block NonlinearProblem
        ta["mono_snes"]   : ta["mono_problem"].solver, tolerances pre-set
                            to max_it=1 (see module docstring)

    Requires ta built with per_layer=True.
    """
    from dolfinx import fem
    from dolfinx.fem.petsc import NonlinearProblem

    domain = ta["T_h"].function_space.mesh
    dx = ufl.Measure("dx", domain=domain,
                     metadata={"quadrature_degree": MONO_QUADRATURE_DEGREE})
    n_hat_ufl = ta["n_hat_ufl"]
    coil_ind = ta["coil_ind"]
    dt_const = ta["dt_const"]
    Vdg0 = ta["Vdg0"]
    delta_SC = ta["delta_SC"]
    Lambda = ta["Lambda"]
    eps_reg = float(getattr(params, "ta_eps_reg", 1.0))
    p_floor = float(getattr(params, "ta_floor_smooth_p", 16.0))
    E_c = 1.0e-4

    layer_T_fns = ta["layer_T_fns"]
    n_layers = len(layer_T_fns)

    Jc_fns, n_fns, F_list = [], [], []
    for i, T_i in enumerate(layer_T_fns):
        Jc_fn = fem.Function(Vdg0, name=f"mono_Jc_{i}")
        n_fn = fem.Function(Vdg0, name=f"mono_n_{i}")
        Jc_fn.x.array[:] = _JC_SAFE_DEFAULT
        n_fn.x.array[:] = _N_SAFE_DEFAULT
        Jc_fns.append(Jc_fn)
        n_fns.append(n_fn)

        phi = ufl.TestFunction(ta["V_T"])
        J_ufl = ufl.cross(ufl.grad(T_i), n_hat_ufl)
        Jmag = ufl.sqrt(ufl.inner(J_ufl, J_ufl) + 1e-30)
        j_norm_raw = Jmag / Jc_fn
        j_norm = (eps_reg ** p_floor + j_norm_raw ** p_floor) ** (1.0 / p_floor)
        rho_SC = (E_c / Jc_fn) * ufl.exp((n_fn - 1.0) * ufl.ln(j_norm))
        rho_expr = rho_SC * (delta_SC / Lambda)

        J_phi = ufl.cross(ufl.grad(phi), n_hat_ufl)
        F_i = (rho_expr * ufl.inner(J_ufl, J_phi) * dx
               + (1.0 / dt_const) * coil_ind
                 * ufl.inner(ufl.curl(ta["A_h"] - ta["A_prev"]),
                             phi * n_hat_ufl) * dx)
        F_list.append(F_i)

    # ── Shared A-equation, source term FULLY SYMBOLIC in every T_i ────────
    # Each T_i is Dirichlet-pinned to zero outside its own layer's cells
    # (ta["layer_bcs"][i]), so cross(grad(T_i), n_hat) is automatically
    # confined to layer i's own cells -- no extra per-layer indicator
    # needed, exactly mirroring how ta_solve._J_from_T's per-layer branch
    # already assumes this. Writing the sum here (rather than going through
    # an intermediate numpy DG0 interpolation, as ta_solve.py's Picard path
    # and newton_ta.py's Gauss-Seidel path both do) is what lets
    # ufl.derivative see dF_A/dT_i for every layer -- the coupling term the
    # outer-loop schemes could only ever approximate one sweep at a time.
    v_A = ufl.TestFunction(ta["V_A"])
    J_dir = ufl.cross(ufl.grad(layer_T_fns[0]), n_hat_ufl)
    for T_i in layer_T_fns[1:]:
        J_dir = J_dir + ufl.cross(ufl.grad(T_i), n_hat_ufl)
    J_s_symbolic = J_dir * (delta_SC / Lambda)

    F_A = ((1.0 / mu0) * ufl.inner(ufl.curl(ta["A_h"]), ufl.curl(v_A)) * dx
           + params.gauge_regularization
             * ufl.inner(ta["A_h"], v_A) * dx
           - ufl.inner(J_s_symbolic, v_A) * dx)
    F_list.append(F_A)

    u_list = list(layer_T_fns) + [ta["A_h"]]
    bcs = [bc for layer_bcs_i in ta["layer_bcs"] for bc in layer_bcs_i]
    bcs.append(ta["bc_A"])

    opts = dict(DEFAULT_SNES_OPTIONS)
    if verbose:
        opts["snes_monitor"] = None
    if snes_options:
        opts.update(snes_options)

    problem = NonlinearProblem(F_list, u_list, bcs=bcs, kind="mpi",
                               petsc_options_prefix="mono_ta_",
                               petsc_options=opts)
    snes = problem.solver
    snes.setTolerances(max_it=1)   # see module docstring

    ta["mono_Jc_fns"] = Jc_fns
    ta["mono_n_fns"] = n_fns
    ta["mono_problem"] = problem
    ta["mono_snes"] = snes
    return ta


def _update_mono_coefficients(ta, ic_model, n_model, layer, B_coil_all,
                              relax=None):
    """Refresh layer `layer`'s frozen Jc/n from the CURRENT B field.

    Same math as newton_ta.update_frozen_coefficients (same models, same
    angle convention) but writes into ta["mono_Jc_fns"]/ta["mono_n_fns"] so
    the two solvers' frozen-coefficient state never aliases if both are
    built on the same ta dict in one process (e.g. a side-by-side check).

    relax under-relaxes exactly as the working Picard/newton_ta code does,
    for the identical reason: a first cut at this whole approach that
    refreshed Jc/n with NO damping at all (relax=1.0 every step) let the
    per-step update swing too far once close to the fixed point and
    triggered spurious line-search failures. relax=None reuses
    params.ta_rho_relax, the same constant everything else in this project
    damps rho-adjacent updates with.
    """
    idx = ta["layer_cell_idx"][layer]
    coil_cells = ta["coil_cells"]
    cells_layer = coil_cells[idx]
    n_hat_layer = ta["n_hat_coil"][idx]

    B_layer = B_coil_all[idx]
    Bmag = np.linalg.norm(B_layer, axis=1)
    theta = angle_with_normal_deg(B_layer, n_hat_layer)
    Ic_arr, clip_frac = ic_model.critical_current(Bmag, theta)
    Jc_vol = Ic_arr / (ta["delta_SC"] * ic_model.tape_width)
    n_arr, _ = n_model.n_value(Bmag, theta)

    relax = float(getattr(params, "ta_rho_relax", 0.5)) if relax is None else relax
    Jc_fn = ta["mono_Jc_fns"][layer]
    n_fn = ta["mono_n_fns"][layer]
    Jc_fn.x.array[cells_layer] = ((1.0 - relax) * Jc_fn.x.array[cells_layer]
                                  + relax * Jc_vol)
    n_fn.x.array[cells_layer] = ((1.0 - relax) * n_fn.x.array[cells_layer]
                                 + relax * n_arr)
    return float(np.mean(clip_frac))


# Real per-step failure reasons (excludes -5 = SNES_DIVERGED_MAX_IT, which
# is EXPECTED every call under the max_it=1 scheme -- see module docstring).
_REAL_FAILURE_REASONS = {-3, -4, -6, -7, -8, -9, -10, -11}


def monolithic_step(ta, domain, ic_model, n_model, I_now, dt, uniform_setup,
                    max_outer=150, min_outer=6, stall_tol=0.05, first=False,
                    bootstrap_iters=30, verbose=True, jc_n_relax=None,
                    step_relax=1.0, debug=False):
    """Advance one time step with the fully-coupled block Newton system.

    Bootstraps a cold start with newton_ta.py's already-validated Picard
    bootstrap (a cold T=0 start is too far from the solution for
    line-search Newton in one shot -- same reasoning as newton_ta.py, reuse
    rather than re-derive). Each outer Python iteration = one block Newton
    step (max_it=1) + a fresh Jc/n refresh from that step's resulting B.

    step_relax (default 1.0 = the raw Newton step, unmodified): blends the
    accepted block-Newton step with the pre-step state,
    x <- (1-step_relax)*x_old + step_relax*x_new, applied to T AND A
    TOGETHER since they are solved simultaneously in one linear system here
    (unlike newton_ta.py's Gauss-Seidel t_relax, which only had T to damp
    and was proven to fix nothing -- see that module's docstring; this is a
    structurally different situation because here T and A never get to
    drift out of sync with each other in the first place, so damping the
    JOINT step is testing a different hypothesis: that a single Newton step
    across this stiff a coupled system, even backtracking-line-search
    globalized on the raw residual norm, can still be too large in the
    SCIF-relevant sense (a near-cancelling quantity the residual norm does
    not see) for the frozen-coefficient linearization to stay valid).

    debug=True prints per-layer j/jc mean/max after every accepted step --
    added specifically to see WHERE a step is blowing up rather than
    guessing from the aggregate SCIF alone.
    """
    import ta_solve
    from newton_ta import ta_transient_seed_cold, _picard_bootstrap

    ta["dt_const"].value = float(dt)
    B_h = ta["B_fn"]
    coil = ta["coil_cells"]
    n_layers = len(ta["layer_T_fns"])
    eps = float(getattr(params, "ta_eps_reg", 1.0))

    T_amp = I_now / (2.0 * ta["delta_SC"])
    ta["T_bot_val"].value = +T_amp
    ta["T_top_val"].value = -T_amp

    if first:
        J_seed = ta_transient_seed_cold(ta, uniform_setup, ic_model, n_model,
                                        I_now)
        B_h.interpolate(ta["curl_expr"])
        B_seed = B_h.x.array.reshape(-1, 3)[coil]
        ta["_rho_prev"] = None
        ta_solve._update_rho(ta, J_seed, B_seed, ic_model, n_model, eps)
        _picard_bootstrap(ta, domain, ic_model, n_model, I_now, dt,
                          n_iters=bootstrap_iters, verbose=verbose)
        B_h.interpolate(ta["curl_expr"])
        B_now = B_h.x.array.reshape(-1, 3)[coil]
        for layer in range(n_layers):
            _update_mono_coefficients(ta, ic_model, n_model, layer, B_now,
                                      relax=1.0)

    problem = ta["mono_problem"]

    B_h.interpolate(ta["curl_expr"])
    B_coil = B_h.x.array.reshape(-1, 3)[coil]
    J_unif = ta["t_hat_coil"] * (I_now / (ta["delta_SC"] * params.w))

    scif_ema = None
    scif_hist = []
    converged = False
    n_outer = max_outer
    stop_reason = "max_outer"
    total_snes_iters = 0

    for k in range(max_outer):
        relax_k = 1.0 if (first and k == 0) else jc_n_relax
        for layer in range(n_layers):
            _update_mono_coefficients(ta, ic_model, n_model, layer, B_coil,
                                      relax=relax_k)

        T_snap = [T_i.x.array.copy() for T_i in ta["layer_T_fns"]]
        A_snap = ta["A_h"].x.array.copy()

        problem.solve()
        reason = ta["mono_snes"].getConvergedReason()
        its = ta["mono_snes"].getIterationNumber()
        total_snes_iters += its

        ok = (reason not in _REAL_FAILURE_REASONS
              and np.all(np.isfinite(ta["A_h"].x.array))
              and all(np.all(np.isfinite(T_i.x.array))
                     for T_i in ta["layer_T_fns"]))

        if not ok:
            for T_i, snap in zip(ta["layer_T_fns"], T_snap):
                T_i.x.array[:] = snap
                T_i.x.scatter_forward()
            ta["A_h"].x.array[:] = A_snap
            ta["A_h"].x.scatter_forward()
            stop_reason = "block_newton_failure_reverted"
            n_outer = k
            converged = False
            if verbose:
                print(f"      [mono] block Newton step failed "
                      f"(reason={reason}) -- reverting and stopping at "
                      f"iteration {k}", flush=True)
            break

        if step_relax != 1.0:
            for T_i, snap in zip(ta["layer_T_fns"], T_snap):
                T_i.x.array[:] = (1.0 - step_relax) * snap + step_relax * T_i.x.array
                T_i.x.scatter_forward()
            ta["A_h"].x.array[:] = (
                (1.0 - step_relax) * A_snap + step_relax * ta["A_h"].x.array)
            ta["A_h"].x.scatter_forward()

        J_coil = ta_solve._J_from_T(ta, domain)
        ta_solve._update_Js(ta, J_coil)
        B_h.interpolate(ta["curl_expr"])
        B_coil = B_h.x.array.reshape(-1, 3)[coil]

        dJs = (J_coil - J_unif) * (ta["delta_SC"] / ta["Lambda"])
        scif = ta_solve.dB_bore_from_dJ(ta["coil_centroids"], dJs,
                                        ta["coil_vols"])[2] * 1e3
        scif_ema = scif if scif_ema is None else 0.8 * scif_ema + 0.2 * scif
        scif_hist.append(scif_ema)
        if len(scif_hist) > 6:
            scif_hist.pop(0)

        if verbose:
            dbg = ""
            if debug:
                jjc = []
                for layer in range(n_layers):
                    idx = ta["layer_cell_idx"][layer]
                    cells_layer = ta["coil_cells"][idx]
                    Jc_l = ta["mono_Jc_fns"][layer].x.array[cells_layer]
                    Jmag_l = np.linalg.norm(J_coil[idx], axis=-1)
                    jjc.append(float(np.mean(Jmag_l / np.maximum(Jc_l, 1e-30))))
                dbg = "  j/jc_per_layer=" + ",".join(f"{v:.2f}" for v in jjc)
            print(f"    [mono k={k+1:3d}] SCIF={scif_ema:+9.2f} mT  "
                  f"reason={reason}  its={its}{dbg}", flush=True)

        if (k + 1) >= min_outer and len(scif_hist) == 6:
            if abs(scif_hist[-1] - scif_hist[0]) < stall_tol:
                converged = True
                n_outer = k + 1
                stop_reason = "stall"
                break

    ta["A_prev"].x.array[:] = ta["A_h"].x.array
    ta["A_prev"].x.scatter_forward()

    if verbose:
        scif_str = f"{scif_ema:+.2f} mT" if scif_ema is not None else "N/A"
        print(f"    [mono] step done: stop_reason={stop_reason}  "
              f"n_outer={n_outer}  SCIF={scif_str}", flush=True)

    return dict(converged=converged, n_outer=n_outer,
               total_snes_iters=total_snes_iters, stop_reason=stop_reason,
               scif_mT=float(scif_ema) if scif_ema is not None else float("nan"),
               scif_hist_tail=[float(s) for s in scif_hist])


def monolithic_march(ta, domain, uniform_setup, ic_model, n_model, schedule,
                     max_outer=150, min_outer=6, stall_tol=0.05,
                     bootstrap_iters=30, verbose=True, jc_n_relax=None):
    """Run a full multi-step schedule. `schedule` is a list of (t, I, dt)
    triples (ta_transient.ramp_schedule() builds one)."""
    hist = []
    for n, (t, I_now, dt) in enumerate(schedule):
        if verbose:
            print(f"  step {n+1}/{len(schedule)}  t={t:7.1f} s  "
                  f"I={I_now:7.2f} A  dt={dt:6.1f} s", flush=True)
        info = monolithic_step(ta, domain, ic_model, n_model, I_now, dt,
                               uniform_setup, max_outer=max_outer,
                               min_outer=min_outer, stall_tol=stall_tol,
                               first=(n == 0), bootstrap_iters=bootstrap_iters,
                               verbose=verbose, jc_n_relax=jc_n_relax)
        info.update(t=t, I=I_now, dt=dt, step_index=n)
        hist.append(info)
        if verbose:
            print(f"  step {n+1}/{len(schedule)} SUMMARY: "
                  f"converged={info['converged']}  "
                  f"stop_reason={info['stop_reason']}  "
                  f"SCIF={info['scif_mT']:+.2f} mT", flush=True)
    return hist
