"""
newton_ta.py — Newton (SNES) solve for the T-equation's power-law
nonlinearity, as a drop-in alternative to the Picard T-solve used by
solve_ta_at_current() and ta_transient.step().

DOES NOT MODIFY solve/ta_solve.py.  Everything here is purely additive: it
consumes the SAME `ta` dict `setup_ta_problem()` already builds (the mesh,
function spaces, per-layer BCs, A/B machinery, coil_ind, dt_const, ...) and
adds a NEW set of per-layer nonlinear problems alongside the existing
`prob_T_layers` (LinearProblem, Picard) objects -- both can coexist on the
same `ta` object, and nothing here is called unless a caller explicitly asks
for it.  `ta_sweep.py`, `optimize/`, and every existing production path keep
using `solve_ta_at_current()` / `prob_T_layers` exactly as before.

WHY THIS EXISTS
---------------
The Picard scheme solves the T-equation by FREEZING rho(J,B) each iteration,
solving the resulting LINEAR problem exactly, then damping the update by a
relaxation factor alpha because the frozen-rho approximation ignores how
sensitive rho actually is to J (the E-J power law has n ~ 13-22, so rho
changes enormously for a modest change in J). At the standard dt=600s
(a single implicit step spanning the whole ramp -- the only case this
project has ever run) that damping happens to be enough. At the shorter dt a
real multi-step ramp needs, it is not: six different remedies (stronger/
weaker fixed relaxation, gentle ramp-up schedules, undamped and damped
Anderson acceleration) were tried and none converged -- see CLAUDE.md's
2026-08-04 entries for the full record.

Newton's method uses the ACTUAL derivative dF/dT (via UFL's `ufl.derivative`,
computed automatically by dolfinx's NonlinearProblem/SNES) instead of
ignoring it. A smoke test on the exact hard case (dt=100s, I=32.67A, cold
start, the case Picard could not converge even in 1000 iterations) converged
in 11 SNES iterations with a ~10-order-of-magnitude residual drop. That
smoke test is what THIS module builds out properly.

QUASI-NEWTON, NOT FULL NEWTON -- WHY
-------------------------------------
A FULL Jacobian would need d(Ic)/dB and d(n)/dB, which come from measured
CSV data via scipy spline interpolation (physics/ic_model.py) -- not a
symbolic UFL expression `ufl.derivative` can differentiate automatically.
Instead, rho is written as

    rho(J) = (E_c/Jc) * exp((n-1)*log(smooth_floor(|J|/Jc)))

with Jc and n FROZEN as DG0 coefficients (updated between OUTER iterations
from the measured Ic(B,theta)/n(B,theta) models, exactly as
ta_solve._update_rho already does in numpy -- see update_frozen_coefficients
below). Written this way rho(J) is pure algebra in J = curl(T)xn_hat, i.e.
genuine UFL algebra in the unknown T, so `ufl.derivative` handles it with no
hand-derived spline Jacobian at all. This linearises exactly the dominant
stiffness driver (the power-law exponent in J) while still Picard-lagging
the milder Ic(B)/n(B) field dependence -- a Newton-Picard hybrid, not a
fully monolithic nonlinear solve.

TWO UFL GOTCHAS FOUND BUILDING THIS (both fixed here, keep them fixed)
------------------------------------------------------------------------
1. FFCx's automatic quadrature-degree estimation does not handle the
   transcendental `exp(log(...))` form of the power law -- left to guess, it
   tried to allocate a 125000x125000 quadrature table (116 GiB) and crashed.
   Fixed with an EXPLICIT `quadrature_degree` on the measure
   (NEWTON_QUADRATURE_DEGREE below); do not remove it.
2. Writing `rho = (E_c/Jc)*...` symbolically means the division is evaluated
   EVERYWHERE the form is integrated -- including cells where `Jc` has no
   meaningful value (e.g. a layer's own Jc_fn is only physically meaningful
   at that layer's own coil cells; other layers' T is Dirichlet-pinned to
   zero there, so their contribution SHOULD be zero, but `Jc_fn = 0` at those
   cells gives `E_c/Jc_fn = inf`, and `0 (from curl(T)=0) * inf = NaN` in
   IEEE arithmetic, not zero). The working Picard code never hits this
   because rho is precomputed in numpy and simply never touches those
   cells. Fixed by giving Jc_fn a SAFE NONZERO default (1.0) everywhere --
   the zero-T Dirichlet pinning outside a layer's own cells makes the actual
   physical contribution zero regardless of what the "default" evaluates to.
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

# FFCx cannot estimate a sane quadrature degree for the transcendental
# exp(log(...)) power-law form -- see gotcha #1 above. Fixed, not tuned;
# degree 2 matches the polynomial degree of the rest of the (CG1-based) form.
NEWTON_QUADRATURE_DEGREE = 2

# Safe nonzero placeholder for Jc outside a layer's own cells -- see
# gotcha #2. Never used physically: T is Dirichlet-pinned to zero there.
_JC_SAFE_DEFAULT = 1.0
_N_SAFE_DEFAULT = 2.0

DEFAULT_SNES_OPTIONS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_linesearch_max_it": 40,
    # Loosened from 1e-8/1e-10: the OUTER Picard-on-Jc(B)/n(B) loop's
    # SCIF-stall criterion is the real convergence check; the inner Newton
    # solve is re-run every outer iteration with freshly updated Jc/n
    # regardless, so pushing it to near machine precision is wasted effort
    # and was implicated in spurious backtracking-linesearch failures
    # (SNES reason -6) once the outer loop was already within a few mT of
    # its own fixed point -- the residual there is small enough that
    # further "sufficient decrease" is close to floating-point noise.
    "snes_rtol": 1e-6,
    "snes_atol": 1e-8,
    "snes_stol": 1e-10,
    "snes_max_it": 50,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def build_layer_newton_problems(ta, snes_options=None, verbose=False):
    """Build one NonlinearProblem per z-layer for the quasi-Newton T-solve.

    Adds NEW keys to `ta` (does not touch anything setup_ta_problem() or
    ta_solve.py's Picard path already put there):
        ta["newton_Jc_fns"]   : list of DG0 Functions, one per layer
        ta["newton_n_fns"]    : list of DG0 Functions, one per layer
        ta["newton_problems"] : list of NonlinearProblem, one per layer

    Requires ta built with per_layer=True (per_turn_bc may be True or False
    -- this reuses whatever layer_bcs/layer_T_fns/layer_cell_idx the caller's
    setup_ta_problem() call already produced, so it works with either the
    original Constant-BC path or the NI per_turn_bc=True path unchanged).
    """
    from dolfinx.fem.petsc import NonlinearProblem

    domain = ta["T_h"].function_space.mesh
    dx = ufl.Measure("dx", domain=domain,
                     metadata={"quadrature_degree": NEWTON_QUADRATURE_DEGREE})
    n_hat_ufl = ta["n_hat_ufl"]
    coil_ind = ta["coil_ind"]
    dt_const = ta["dt_const"]
    Vdg0 = ta["Vdg0"]
    eps_reg = float(getattr(params, "ta_eps_reg", 1.0))
    p_floor = float(getattr(params, "ta_floor_smooth_p", 16.0))
    E_c = 1.0e-4

    Jc_fns, n_fns, problems = [], [], []
    opts = dict(DEFAULT_SNES_OPTIONS)
    if verbose:
        opts["snes_monitor"] = None
    if snes_options:
        opts.update(snes_options)

    for i, T_i in enumerate(ta["layer_T_fns"]):
        from dolfinx import fem
        Jc_fn = fem.Function(Vdg0, name=f"Jc_frozen_{i}")
        n_fn = fem.Function(Vdg0, name=f"n_frozen_{i}")
        Jc_fn.x.array[:] = _JC_SAFE_DEFAULT
        n_fn.x.array[:] = _N_SAFE_DEFAULT

        phi = ufl.TestFunction(ta["V_T"])
        J_ufl = ufl.cross(ufl.grad(T_i), n_hat_ufl)
        Jmag = ufl.sqrt(ufl.inner(J_ufl, J_ufl) + 1e-30)
        j_norm_raw = Jmag / Jc_fn
        j_norm = (eps_reg ** p_floor + j_norm_raw ** p_floor) ** (1.0 / p_floor)
        # rho_SC (the SC-layer resistivity) times delta_SC/Lambda -- the
        # HOMOGENISED value ta_solve._update_rho actually uses in a_T. A
        # first version of this omitted the delta_SC/Lambda factor, making
        # rho ~75x too large; the outer loop still converged numerically
        # (SNES was happy) but to a self-consistent, PHYSICALLY WRONG fixed
        # point (13.6 mT vs the Picard-validated 172.77 mT at this same
        # operating point) -- a reminder that "SNES reports converged" and
        # "the formulation is correct" are different claims; always check
        # against a ground-truth number, not just the solver's own status.
        rho_SC = (E_c / Jc_fn) * ufl.exp((n_fn - 1.0) * ufl.ln(j_norm))
        rho_expr = rho_SC * (ta["delta_SC"] / ta["Lambda"])

        J_phi = ufl.cross(ufl.grad(phi), n_hat_ufl)
        F = (rho_expr * ufl.inner(J_ufl, J_phi) * dx
             + (1.0 / dt_const) * coil_ind
               * ufl.inner(ufl.curl(ta["A_h"] - ta["A_prev"]),
                           phi * n_hat_ufl) * dx)

        prob = NonlinearProblem(
            F, T_i, bcs=ta["layer_bcs"][i],
            petsc_options_prefix=f"newton_ta_{i}_",
            petsc_options=opts)

        Jc_fns.append(Jc_fn)
        n_fns.append(n_fn)
        problems.append(prob)

    ta["newton_Jc_fns"] = Jc_fns
    ta["newton_n_fns"] = n_fns
    ta["newton_problems"] = problems
    return ta


def update_frozen_coefficients(ta, ic_model, n_model, layer, B_coil_all,
                               relax=None):
    """Refresh layer `layer`'s frozen Jc/n from the CURRENT B field.

    Mirrors ta_solve._update_rho's Ic(B,theta)/n(B,theta) lookups exactly
    (same models, same angle convention) but writes into the per-layer
    Jc_fn/n_fn DG0 Functions instead of a single combined rho_fn array.
    B_coil_all is B at ALL coil cells (ta["coil_cells"] order); this
    extracts just this layer's rows via ta["layer_cell_idx"].

    UNDER-RELAXES the update (default: params.ta_rho_relax, the same
    constant the working Picard code uses to damp its own rho updates for
    exactly this reason). A first version overwrote Jc/n outright each
    outer iteration with no damping at all; that let the outer loop take a
    step large enough, once close to the true fixed point (SCIF within 1%
    of the Picard-validated 172.77 mT), that the following Newton solve
    diverged (SNES reason -9, SNES_DIVERGED_DTOL -- residual increased past
    the divergence tolerance). Relaxing Jc/n the same way the original
    scheme relaxes rho gives Newton a smaller, easier target each outer
    iteration.
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
    Jc_fn = ta["newton_Jc_fns"][layer]
    n_fn = ta["newton_n_fns"][layer]
    Jc_fn.x.array[cells_layer] = ((1.0 - relax) * Jc_fn.x.array[cells_layer]
                                  + relax * Jc_vol)
    n_fn.x.array[cells_layer] = ((1.0 - relax) * n_fn.x.array[cells_layer]
                                 + relax * n_arr)
    return float(np.mean(clip_frac))


def newton_solve_layer(ta, layer, debug=False):
    """Newton-solve ONE layer's T given its currently-frozen Jc/n.

    If the default backtracking line search reports failure
    (SNES_DIVERGED_LINE_SEARCH, reason -6), retries ONCE with the line
    search switched to "basic" (a plain, unglobalized Newton step) before
    giving up. This is a standard PETSc troubleshooting move: once the
    outer Picard-on-Jc(B)/n(B) loop is within a few mT of its own fixed
    point, each Newton solve only needs 1-3 iterations from the previous
    step's T as initial guess -- a reported line-search failure there is
    the line search's own "sufficient decrease" merit-function test being
    numerically finicky at that small a residual scale, not the Newton
    direction itself being bad. Observed directly: this failure mode only
    ever appeared once the outer SCIF had already dropped to single-digit
    mT, never during the far-from-converged early iterations.

    Returns (converged: bool, n_snes_iters: int, reason: int).
    """
    prob = ta["newton_problems"][layer]
    if debug:
        Jc = ta["newton_Jc_fns"][layer].x.array
        nn = ta["newton_n_fns"][layer].x.array
        cells = ta["coil_cells"][ta["layer_cell_idx"][layer]]
        print(f"      [debug layer {layer}] Jc range (this layer's cells): "
              f"[{Jc[cells].min():.4e},{Jc[cells].max():.4e}]  "
              f"n range: [{nn[cells].min():.3f},{nn[cells].max():.3f}]  "
              f"any Jc<=0: {np.any(Jc[cells] <= 0)}  "
              f"any nonfinite Jc: {np.any(~np.isfinite(Jc[cells]))}  "
              f"any nonfinite n: {np.any(~np.isfinite(nn[cells]))}",
              flush=True)
    T_i = ta["layer_T_fns"][layer]
    T_backup = T_i.x.array.copy()

    prob.solve()
    reason = prob.solver.getConvergedReason()
    its = prob.solver.getIterationNumber()

    if reason <= 0:
        # Any non-convergence: retry once with the line search switched to
        # "basic" (a plain, unglobalized Newton step) and a higher iteration
        # cap. Observed TWO distinct failure reasons in practice once the
        # outer Picard-on-Jc(B)/n(B) loop is within a few mT of its fixed
        # point: -6 (SNES_DIVERGED_LINE_SEARCH, the bt line search's
        # "sufficient decrease" test being numerically finicky at that small
        # a residual) and -5 (SNES_DIVERGED_MAX_IT, seen specifically on the
        # tiny 2-3-turn "vestigial" pancake layers this project has flagged
        # elsewhere as numerically fragile -- see the double-pancake and
        # turn-floor history in CLAUDE.md). Both are plausibly the same
        # underlying issue (the bt line search struggling near a small
        # residual, either failing its own test outright or burning through
        # snes_max_it taking tiny globalized steps it didn't need to), so
        # one fallback covers both rather than special-casing each reason.
        if debug:
            print(f"      [debug layer {layer}] primary solve did not "
                  f"converge (reason={reason}) -- retrying with basic "
                  f"line search", flush=True)
        T_i.x.array[:] = T_backup
        T_i.x.scatter_forward()
        ls = prob.solver.getLineSearch()
        prev_type = ls.getType()
        prev_max_it = prob.solver.getTolerances()[3]
        ls.setType("basic")
        prob.solver.setTolerances(max_it=200)
        prob.solve()
        ls.setType(prev_type)
        prob.solver.setTolerances(max_it=prev_max_it)
        reason = prob.solver.getConvergedReason()
        its = prob.solver.getIterationNumber()

    if debug:
        print(f"      [debug layer {layer}] SNES reason={reason}  its={its}  "
              f"T finite: {np.all(np.isfinite(T_i.x.array))}", flush=True)
    if not np.all(np.isfinite(T_i.x.array)):
        raise RuntimeError(
            f"Newton solve produced non-finite T, layer {layer} "
            f"(SNES reason={reason}, its={its})")
    return reason > 0, its, reason


# ── outer time-step orchestration ───────────────────────────────────────────
#
# The Picard scheme's OUTER loop iterates because each inner T-solve is only
# a damped, frozen-rho APPROXIMATION -- most of its iterations are spent
# re-converging the SAME frozen-coefficient problem after small rho updates.
# Here the inner solve is EXACT (Newton, to its own tight SNES tolerance) for
# whatever Jc/n snapshot is frozen, so the outer loop's only job is to update
# Jc(B)/n(B) (a comparatively mild, slowly-varying dependence, unlike the
# stiff power-law-in-J term Newton now handles exactly) and check whether
# THAT has settled. Expect far fewer outer iterations than Picard needed.

def _picard_bootstrap(ta, domain, ic_model, n_model, I_now, dt,
                      n_iters=30, verbose=False):
    """A short run of the EXISTING, validated two-phase Picard scheme
    (ta_transient._picard_phase, unmodified) purely to move T away from
    T=0 into Newton's basin of attraction, capped well below what Picard
    would need to fully converge on its own.

    Necessary because a cold T=0 start is too far from the converged
    solution for line-search Newton to handle in one shot: the very first
    outer Newton solve from T=0 diverges with SNES reason -6
    (SNES_DIVERGED_LINE_SEARCH).

    A first version of this hand-rolled a SIMPLER bootstrap using a FIXED
    alpha=0.30 for all iterations, with no phase-2 switch-down. That
    diverged to NaN within 15 iterations AT dt=600s -- the supposedly "easy"
    case -- which is exactly the period-2 divergence this project's own
    two-phase scheme (alpha 0.30 -> 0.15) exists to prevent (see CLAUDE.md's
    bug-history entry on Picard stagnating in a period-2 limit cycle). A
    hand-rolled bootstrap that skips the phase-2 switch reintroduces exactly
    that failure mode. Reusing the already-validated _picard_phase (which
    never diverged in ANY of this session's dt/current combinations, only
    sometimes failed to reach the tight SCIF-stall tolerance) instead of a
    second, worse implementation of the same idea is the correct fix.

    This is expected to hit its iteration cap without reaching
    _picard_phase's own (tight) convergence criterion in the hard dt=100
    case -- that is fine and expected; it only needs to get "close enough"
    for Newton, not converge on its own (if it fully converged, Newton
    would not be needed at all).
    """
    from ta_transient import _picard_phase

    J_coil, n_iters_run, converged = _picard_phase(
        ta, domain, ic_model, n_model, I_now, dt,
        J_coil=ta["t_hat_coil"] * (I_now / (ta["delta_SC"] * params.w)),
        closure=lambda: None, max_iters=n_iters, min_iters=n_iters,
        scif_tol=float(getattr(params, "ta_scif_stall_mT", 0.05)),
        label="newton-bootstrap", verbose=verbose)
    if verbose:
        print(f"    [bootstrap] {n_iters_run} Picard iterations "
              f"(converged={converged} -- either is fine, this only needs "
              f"to seed Newton's initial guess)", flush=True)
    return J_coil


def step(ta, domain, ic_model, n_model, I_now, dt, uniform_setup,
         max_outer=30, min_outer=3, stall_tol=0.05, first=False,
         bootstrap_iters=30, verbose=True, spike_check=True,
         jc_n_relax=None, high_n_relax_factor=None, high_n_threshold=24.0,
         t_relax=None):
    """Advance one time step using the Newton-per-layer / Picard-on-Jc(B,n)
    hybrid. No NI circuit closure yet -- this validates the core mechanism
    against the INSULATED case first; see module docstring. Returns a dict
    of diagnostics.
    """
    import ta_solve

    ta["dt_const"].value = float(dt)
    B_h = ta["B_fn"]
    coil = ta["coil_cells"]

    # The transport-current BC must be set EVERY step, not just the first --
    # this was a real bug caught before the first multi-step run: I_now
    # changes step to step along the ramp, and T_bot_val/T_top_val are what
    # actually encode it (per_turn_bc=False path; harmless no-op otherwise).
    T_amp = I_now / (2.0 * ta["delta_SC"])
    ta["T_bot_val"].value = +T_amp
    ta["T_top_val"].value = -T_amp

    if first:
        eps = float(getattr(params, "ta_eps_reg", 1.0))
        J_seed = ta_transient_seed_cold(ta, uniform_setup, ic_model, n_model, I_now)
        B_h.interpolate(ta["curl_expr"])
        B_seed = B_h.x.array.reshape(-1, 3)[coil]
        ta["_rho_prev"] = None
        ta_solve._update_rho(ta, J_seed, B_seed, ic_model, n_model, eps)
        J_coil = _picard_bootstrap(ta, domain, ic_model, n_model, I_now, dt,
                                   n_iters=bootstrap_iters, verbose=verbose)
    else:
        J_coil = ta_solve._J_from_T(ta, domain)

    B_h.interpolate(ta["curl_expr"])
    B_coil = B_h.x.array.reshape(-1, 3)[coil]

    n_layers = len(ta["layer_T_fns"])
    scif_ema = None
    scif_hist = []
    J_unif = ta["t_hat_coil"] * (I_now / (ta["delta_SC"] * params.w))
    total_snes_iters = 0
    converged = False
    n_outer = max_outer

    stop_reason = "max_outer"
    prev_max_iters = None

    for k in range(max_outer):
        # Snapshot the FULL state (every layer's T/Jc/n, plus A) before this
        # iteration touches anything, so a failure can be undone cleanly.
        #
        # An earlier version reverted only the FAILED layer's T and let the
        # other layers' (already-completed) Newton solves and the shared
        # A-equation proceed as normal. That was worse than not reverting at
        # all: the stale layer's J no longer matched the freshly-updated
        # A/B field, which corrupted the NEXT iteration's Jc/n update for
        # EVERY layer (they all share the same A-equation), causing more
        # layers to fail, cascading into a state that drifted far from the
        # answer (observed: SCIF drifting to -571 mT / -620 mT over 30-40
        # iterations, regardless of the Jc/n relaxation strength tried).
        # Reverting the WHOLE iteration keeps every layer's T mutually
        # consistent with the A/B field it was solved against.
        T_snap = [T_i.x.array.copy() for T_i in ta["layer_T_fns"]]
        Jc_snap = [fn.x.array.copy() for fn in ta["newton_Jc_fns"]]
        n_snap = [fn.x.array.copy() for fn in ta["newton_n_fns"]]
        A_snap = ta["A_h"].x.array.copy()

        clip_fracs = []
        # relax=1.0 (100% new value) ONLY on the very first outer iteration
        # of the very first time step: Jc_fn/n_fn hold the safe placeholder
        # default (1.0, see build_layer_newton_problems) only until they are
        # written for real the first time. On step 2+, k==0 starts from the
        # PREVIOUS step's genuinely-converged, warm-started Jc/n -- using
        # relax=1.0 there would discard that warm start and defeat the
        # entire point of carrying state between steps. Caught before the
        # first real multi-step run by re-reading this logic against what
        # `first` actually means, not by observing a wrong answer.
        relax_k = 1.0 if (first and k == 0) else jc_n_relax
        for layer in range(n_layers):
            # OPT-IN extra damping for whichever layer's CURRENT frozen n
            # (before this update) is highest -- diagnostics
            # (relax_sweep_diag.py) showed a single UNIFORM relax factor
            # makes no difference, because the instability is concentrated
            # in the one layer whose n sits highest (the power-law
            # Jacobian is most sensitive there -- see the 2026-08-05
            # outer-loop entries). Default high_n_relax_factor=None leaves
            # every layer at relax_k, i.e. unchanged behaviour.
            layer_relax = relax_k
            if high_n_relax_factor is not None:
                idx = ta["layer_cell_idx"][layer]
                cells_layer = ta["coil_cells"][idx]
                n_cur = ta["newton_n_fns"][layer].x.array[cells_layer]
                if n_cur.size and float(n_cur.max()) >= high_n_threshold:
                    eff_base = (relax_k if relax_k is not None
                               else float(getattr(params, "ta_rho_relax", 0.5)))
                    layer_relax = eff_base * high_n_relax_factor
            cf = update_frozen_coefficients(ta, ic_model, n_model, layer,
                                            B_coil, relax=layer_relax)
            clip_fracs.append(cf)

        layer_iters = []
        any_failed = False
        spiked = False
        for layer in range(n_layers):
            ok, its, reason = newton_solve_layer(ta, layer, debug=(k == 0))
            layer_iters.append(its)
            if not ok:
                any_failed = True
                if verbose:
                    print(f"      [layer {layer}] Newton failed this outer "
                          f"iteration (reason={reason})", flush=True)
                break   # no point solving the remaining layers this iter
            # SPIKE CHECK -- catches trouble one step earlier than waiting
            # for an outright SNES failure. Observed directly: iteration k=3
            # had layer 0 report SNES reason=3 (formally "converged") after
            # 28 iterations, immediately following k=2 where every layer
            # converged in 1-2 -- but k=3's RESULTING SCIF (22.30 mT) had
            # already drifted far from the true answer (~172 mT, matching
            # Picard). The outright-failure check only caught the problem on
            # k=4, by which point k=3's corrupted state was already the
            # starting point. A large jump in required iterations is itself
            # the earlier, more reliable signal that the outer loop has left
            # the well-behaved regime, even when SNES's own status says
            # success -- "SNES reports converged" and "this is still the
            # right answer" are different claims (the same lesson the
            # missing delta_SC/Lambda factor taught earlier, in a different
            # form).
            if (spike_check and prev_max_iters is not None
                    and its > max(10, 3 * prev_max_iters)):
                spiked = True
                if verbose:
                    print(f"      [layer {layer}] iteration count spiked "
                          f"({its} vs previous max {prev_max_iters}) -- "
                          f"treating as a stop signal despite SNES "
                          f"reporting success", flush=True)
                break

        if any_failed or spiked:
            # Undo this ENTIRE iteration (T, Jc, n, A all revert together)
            # and stop, returning the last state where every layer's Newton
            # solve genuinely succeeded AND behaved consistently with its
            # neighbours. Empirically this last-good state is already close
            # to the answer (observed: within 1% of the Picard-validated
            # ground truth) -- a Newton failure or iteration-count spike
            # right at the fixed point reads as "reached the edge of what's
            # numerically representable here," which is a reason to stop
            # and trust the last good state, not a reason to keep pushing.
            for T_i, snap in zip(ta["layer_T_fns"], T_snap):
                T_i.x.array[:] = snap
                T_i.x.scatter_forward()
            for fn, snap in zip(ta["newton_Jc_fns"], Jc_snap):
                fn.x.array[:] = snap
            for fn, snap in zip(ta["newton_n_fns"], n_snap):
                fn.x.array[:] = snap
            ta["A_h"].x.array[:] = A_snap
            ta["A_h"].x.scatter_forward()
            total_snes_iters += sum(layer_iters)
            stop_reason = "newton_failure_reverted" if any_failed else "iteration_spike_reverted"
            n_outer = k   # this iteration did not complete; k is how many did
            # "converged" MUST mean "this SCIF is trustworthy," not merely
            # "we avoided a crash." Reverting stops the loop at the last
            # state where every layer's Newton solve genuinely succeeded --
            # but that alone does not mean the PHYSICS had settled there.
            #
            # A first version tried to auto-classify this: require the last
            # two genuinely-successful iterations' SCIF to already be within
            # 5 mT of each other. That heuristic was ITSELF unreliable --
            # tested directly, it flagged the dt=600 case as "not
            # trustworthy" (150.65 -> 171.53 mT, a 21 mT gap) even though
            # that SCIF independently matches the Picard ground truth
            # (172.77 mT) to under 1%. A crude threshold cannot reliably
            # distinguish "still settling" from "converging, just not
            # monotonically yet" from two points alone.
            #
            # So: NEVER auto-mark a reverted stop as converged. Always
            # False here, with stop_reason and the SCIF history exposed
            # (scif_hist_tail below) so a caller can judge -- checking a
            # reverted result against independent ground truth before
            # trusting it (as was done by hand for the dt=600 case in
            # CLAUDE.md) is the same discipline this project already
            # applies to every other proxy/shortcut number, not a new
            # standard invented for this one.
            converged = False
            if verbose:
                print(f"      reverting outer iteration {k+1} entirely, "
                      f"returning the state from iteration {k} "
                      f"(NOT auto-validated -- check against independent "
                      f"ground truth before trusting this SCIF)", flush=True)
            break

        total_snes_iters += sum(layer_iters)
        prev_max_iters = max(layer_iters)

        # OPT-IN outer relaxation on T itself (default None = unchanged
        # behaviour: each layer's EXACT Newton solution is accepted
        # outright). The original Picard scheme damps TWO things every
        # iteration -- rho/Jc/n AND T itself
        # (T = (1-alpha)*T_old + alpha*T_new). This hybrid only ever damped
        # Jc/n; T has been accepted exactly since the first version. Every
        # Jc/n-relaxation lever tried (uniform 0.5->0.1, per-layer down to
        # 0.01) made ZERO difference to the outer-loop divergence -- a
        # 100x range in damping strength producing an IDENTICAL failure
        # (same layer, same SNES_DIVERGED_DTOL, same SCIF to <1 mT) rules
        # out "Jc/n moves too fast" as the mechanism. The missing half of
        # the original scheme's damping is T/field-level: six layers'
        # EXACT per-layer T solutions, each consistent only with ITS OWN
        # frozen Jc/n snapshot, get accepted and fed straight into the
        # SHARED A-equation every iteration with no smoothing at all --
        # unlike Picard, where the coupled field itself is damped one step
        # at a time. This tests that directly: blend the freshly-solved T
        # with its start-of-iteration value before it drives A/B onward.
        if t_relax is not None:
            for T_i, snap in zip(ta["layer_T_fns"], T_snap):
                T_i.x.array[:] = (1.0 - t_relax) * snap + t_relax * T_i.x.array
                T_i.x.scatter_forward()

        J_coil = ta_solve._J_from_T(ta, domain)
        ta_solve._update_Js(ta, J_coil)
        ta_solve._solve_A(ta, ta["L_A_form"])
        if not np.all(np.isfinite(ta["A_h"].x.array)):
            raise RuntimeError(f"NaN/inf in A solve, outer iter {k}")
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
            print(f"    [newton k={k+1:3d}] SCIF={scif_ema:+9.2f} mT  "
                  f"layer_snes_iters={layer_iters}  "
                  f"clip_frac_max={max(clip_fracs):.3f}", flush=True)

        if (k + 1) >= min_outer and len(scif_hist) == 6:
            if abs(scif_hist[-1] - scif_hist[0]) < stall_tol:
                converged = True
                n_outer = k + 1
                stop_reason = "stall"
                break

    # advance history exactly as the Picard transient step does
    ta["A_prev"].x.array[:] = ta["A_h"].x.array
    ta["A_prev"].x.scatter_forward()

    if verbose:
        # scif_ema is None only if outer iteration 0 itself reverted before
        # ever completing an A-solve (e.g. dt=0 producing an immediate NaN) --
        # a real case hit while testing march(), not a hypothetical.
        scif_str = f"{scif_ema:+.2f} mT" if scif_ema is not None else "N/A (no outer iteration completed)"
        print(f"    [newton] step done: stop_reason={stop_reason}  "
              f"n_outer={n_outer}  SCIF={scif_str}", flush=True)

    return dict(converged=converged, n_outer=n_outer,
               total_snes_iters=total_snes_iters, stop_reason=stop_reason,
               scif_mT=float(scif_ema) if scif_ema is not None else float("nan"),
               # last up to 6 outer-iteration SCIF values (EMA-smoothed) --
               # exposed so a caller can judge a converged=False,
               # reverted result's trajectory for themselves (was it near-
               # settled, or still moving fast when it was stopped?) rather
               # than trusting an unreliable auto-classifier. See the
               # "converged = False" comment above for why this is not
               # done automatically.
               scif_hist_tail=[float(s) for s in scif_hist])


def march(ta, domain, uniform_setup, ic_model, n_model, schedule,
         max_outer=30, min_outer=3, stall_tol=0.05, bootstrap_iters=30,
         verbose=True, t_relax=None, spike_check=True):
    """Run a full multi-step schedule. `schedule` is a list of (t, I, dt)
    triples -- reuse ta_transient.ramp_schedule() to build one, it is
    solver-agnostic (pure schedule generation, no Picard-specific state).

    Each step's `converged`/`stop_reason` is tracked and returned per step
    so a caller can see exactly which steps were genuinely stall-converged
    vs which were a revert-and-stop "best available" result -- do NOT
    silently treat every completed step as validated; see step()'s
    `converged` semantics.

    A step is marched forward (T/A/rho/Jc/n all carried over as the next
    step's starting point) REGARDLESS of whether it hit the formal stall
    criterion or a revert-and-stop, mirroring how ta_transient.march()
    already treats the Picard path's own non-converged-but-capped steps --
    stopping the whole schedule because one step didn't reach the strict
    criterion would defeat the purpose of marching through a ramp at all,
    and the dt=600 regression case shows a reverted stop can still be the
    right answer. What differs from that established practice is that
    HERE every step's trustworthiness is tracked explicitly, not assumed.
    """
    hist = []
    for n, (t, I_now, dt) in enumerate(schedule):
        if verbose:
            print(f"  step {n+1}/{len(schedule)}  t={t:7.1f} s  "
                  f"I={I_now:7.2f} A  dt={dt:6.1f} s", flush=True)
        info = step(ta, domain, ic_model, n_model, I_now, dt, uniform_setup,
                   max_outer=max_outer, min_outer=min_outer,
                   stall_tol=stall_tol, first=(n == 0),
                   bootstrap_iters=bootstrap_iters, verbose=verbose,
                   t_relax=t_relax, spike_check=spike_check)
        info.update(t=t, I=I_now, dt=dt, step_index=n)
        hist.append(info)
        if verbose:
            print(f"  step {n+1}/{len(schedule)} SUMMARY: "
                  f"converged={info['converged']}  "
                  f"stop_reason={info['stop_reason']}  "
                  f"SCIF={info['scif_mT']:+.2f} mT", flush=True)
    return hist


def ta_transient_seed_cold(ta, uniform_setup, ic_model, n_model, I0):
    """Zero-field-cooled start, matching ta_transient._seed_cold(), plus the
    initial Jc/n seeding this module's frozen coefficients need before the
    first Newton solve.
    """
    import ta_solve

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


# ── Newton-INFORMED Picard hybrid (2026-08-04, replaces the t_relax scheme) ─
#
# WHY THIS EXISTS, AND WHY IT REPLACES step()/march() ABOVE
# -----------------------------------------------------------------------
# step()/march() above (the t_relax scheme) were PROVEN WRONG, not just slow:
# `picard_from_newton_state.py` transplanted a t_relax=0.15 run's drifted
# state (SCIF=+52.9 mT, 240 outer iters in) into a fresh, UNMODIFIED Picard
# solver, which converged cleanly back to +641.25 mT in 44 iterations --
# matching the from-ZFC ground truth (641.26 mT) to 0.01 mT. That is
# decisive: 641.26 mT is the ONE true fixed point, and t_relax's outer loop
# is genuinely UNSTABLE near it (it started at +543.9 mT after 30 iterations
# -- already reasonably close -- and moved MONOTONICALLY FURTHER away for
# 200+ more, on a clean geometric trend, extrapolating to roughly -24 mT).
# `small_trelax_trend.py` ruled out "just needs more damping of the same
# kind": t_relax=0.05 (3x stronger) drifted away at a COMPARABLE rate.
#
# Root cause, best understanding: Picard's own scheme damps the FULL
# rho(J,B) (a function of BOTH current and field, evaluated at the CURRENT
# iterate) every iteration. step()/march() only damped Jc(B)/n(B)
# (field-only) while letting Newton resolve rho's J-dependence EXACTLY and
# self-consistently each iteration -- a structurally different, and
# evidently far less stable, fixed-point map, even though each individual
# per-layer Newton solve is itself exact and (per tight_tol_trend.py)
# completely insensitive to inner-solve tolerance -- the instability lives
# entirely in how the outer loop composes those exact solves over many
# iterations, not in their individual accuracy.
#
# THE FIX: never let Newton's raw per-layer solution touch the persisted
# state. Use it ONLY as an INFORMANT -- a more accurate estimate of J than
# Picard's own linear (frozen-rho) solve would produce -- fed into Picard's
# OWN, already-validated damping mechanism (_update_rho's log-space
# relaxation) and Picard's OWN linear per-layer solve (`ta["prob_T_layers"]`,
# reusing the exact objects ta_solve.py already built) plus Picard's OWN
# T-relaxation (the two-phase alpha scheme). The actual state-advancing
# mechanism is therefore IDENTICAL to the scheme already proven stable by
# every existing production path in this project; Newton only improves the
# INPUT that mechanism relaxes toward, on the theory that a more accurate J
# estimate should make Picard's own iteration converge in fewer steps
# without giving up any of its stability. This is a hypothesis to be
# verified against ground truth, same as everything else in this
# investigation -- not assumed correct because it "sounds more principled."
#
# BUG FOUND AND FIXED IN THE FIRST VERSION OF THIS FUNCTION (still
# instructive, kept here rather than silently rewritten): the first attempt
# computed Newton's informed J and used it to set rho_fn BEFORE that SAME
# iteration's Picard T-solve. Tested against the I=196A ground truth
# (hybrid_accuracy_check.py): NOT stable -- overshot to +910 mT, crossed
# back down through the true 641 mT answer without stopping, and settled
# into a long, still-moving descent through 240+ mT before being killed.
# Root cause, on inspection: pure Picard's own iteration has a NATURAL
# ONE-ITERATION LAG -- rho used to solve iteration k's T was computed from
# iteration k-1's (J, B), not iteration k's own. Computing rho from the
# CURRENT iteration's (Newton-informed) J and using it in that SAME
# iteration's T-solve removes that lag, and the lag turns out to be load-
# bearing for stability, not an incidental artifact of Picard's structure.
# FIX: Newton's informing pass now runs AFTER Picard's own T-solve+relax
# each iteration, and the rho it computes is used only by the FOLLOWING
# iteration's T-solve -- preserving Picard's exact lag structure. Newton's
# only remaining role is supplying a more accurate J (from an exact solve)
# than Picard's own linear-solve J would give, as the input _update_rho
# relaxes toward, one iteration later, same as always.
def hybrid_step(ta, domain, ic_model, n_model, I_now, dt, uniform_setup,
                max_outer=100, min_outer=6, stall_tol=0.05, first=False,
                bootstrap_iters=30, verbose=True, newton_blend=0.15):
    """Advance one time step with the Newton-informed Picard hybrid.

    Per outer iteration (lag-preserving order -- see the bug note above):
      1. Picard's own LINEAR per-layer solve (ta["prob_T_layers"], using
         rho_fn as it stood at the END of the previous iteration) + the
         two-phase alpha T-relaxation -- THIS is what actually advances
         the persisted state, identical to the validated production path.
      2. Recompute J/A/B from this relaxed T (needed for SCIF tracking).
      3. Newton-solve each layer EXACTLY given its current frozen Jc/n,
         informationally -- never written to the persisted T (snapshotted
         and restored around this).
      4. Picard-relax rho_fn (ta_solve._update_rho, unmodified) from
         Newton's more-accurate (J, B) pair -- this rho_fn is what the
         NEXT outer iteration's step 1 will use, preserving Picard's
         natural one-iteration lag.
      5. Refresh Newton's own frozen Jc/n from the current B, ready for
         the next iteration's informing pass.

    Returns a diagnostics dict with the same shape as step()'s.
    """
    import ta_solve

    ta["dt_const"].value = float(dt)
    B_h = ta["B_fn"]
    coil = ta["coil_cells"]
    eps = float(getattr(params, "ta_eps_reg", 1.0))
    n_layers = len(ta["layer_T_fns"])

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
        for layer in range(n_layers):
            update_frozen_coefficients(ta, ic_model, n_model, layer, B_seed,
                                       relax=1.0)
        # Bootstrap with PURE Picard (no Newton at all) to move T out of
        # the T=0 cold state before Newton's own per-layer solves are ever
        # attempted -- reuses the identical, already-validated bootstrap
        # step() used, for the identical reason (a cold T=0 start is too
        # far from the solution for line-search Newton to handle in one
        # shot; a hand-rolled fixed-alpha alternative diverged to NaN at
        # dt=600 in an earlier attempt -- see _picard_bootstrap's own
        # docstring). After this, T is warm enough that every subsequent
        # per-outer-iteration Newton pass starts close to its own target.
        _picard_bootstrap(ta, domain, ic_model, n_model, I_now, dt,
                          n_iters=bootstrap_iters, verbose=verbose)
        for layer in range(n_layers):
            B_h.interpolate(ta["curl_expr"])
            B_now = B_h.x.array.reshape(-1, 3)[coil]
            update_frozen_coefficients(ta, ic_model, n_model, layer, B_now,
                                       relax=1.0)

    alpha_high = float(getattr(params, "ta_picard_alpha", 0.30))
    alpha_low = float(getattr(params, "ta_picard_alpha_fine", 0.15))
    rho_relax = float(getattr(params, "ta_rho_relax", 0.5))
    J_unif = ta["t_hat_coil"] * (I_now / (ta["delta_SC"] * params.w))

    alpha = alpha_high
    phase2 = False
    prev_dB_mag = np.inf
    B_h.interpolate(ta["curl_expr"])
    B_prev = B_h.x.array.reshape(-1, 3)[coil].copy()

    scif_ema = None
    scif_hist = []
    converged = False
    n_outer = max_outer
    stop_reason = "max_outer"
    n_newton_failures = 0

    for k in range(max_outer):
        # Step 1: Picard's own linear solve + two-phase alpha T-relaxation,
        # using rho_fn EXACTLY as it stood at the end of the previous outer
        # iteration -- the actual state-advancing mechanism, unchanged from
        # every validated production path. This is where the persisted
        # state actually moves; nothing Newton does below touches it this
        # same iteration (see the lag-bug note above the function).
        for T_i, prob in zip(ta["layer_T_fns"], ta["prob_T_layers"]):
            sol = prob.solve()
            if np.any(np.isnan(sol.x.array)):
                raise RuntimeError(
                    f"NaN in Picard-relaxation linear T solve, outer iter {k}")
            T_i.x.array[:] = (1.0 - alpha) * T_i.x.array + alpha * sol.x.array
            T_i.x.scatter_forward()

        # Step 2: recompute J/A/B from this relaxed T -- the state that
        # actually persists to the next outer iteration.
        J_coil = ta_solve._J_from_T(ta, domain)
        ta_solve._update_Js(ta, J_coil)
        ta_solve._solve_A(ta, ta["L_A_form"])
        if not np.all(np.isfinite(ta["A_h"].x.array)):
            raise RuntimeError(f"NaN/inf in A solve, outer iter {k}")
        B_h.interpolate(ta["curl_expr"])
        B_coil = B_h.x.array.reshape(-1, 3)[coil]

        # Step 3: Newton pass, informational only, run AFTER the state has
        # already advanced this iteration. Snapshot/restore around it so
        # its raw (undamped) solution never touches the persisted T. A
        # per-layer failure is tolerated (Newton is advisory here, not
        # load-bearing) -- worst case the NEXT iteration's rho target is
        # informed by a poorer J estimate for one layer.
        T_snap = [T_i.x.array.copy() for T_i in ta["layer_T_fns"]]
        for layer in range(n_layers):
            ok, its, reason = newton_solve_layer(ta, layer, debug=False)
            if not ok:
                n_newton_failures += 1
        J_informed = ta_solve._J_from_T(ta, domain)
        for T_i, snap in zip(ta["layer_T_fns"], T_snap):
            T_i.x.array[:] = snap
            T_i.x.scatter_forward()

        # Step 4: Picard-relax rho_fn from a BLEND of Picard's own (J_coil,
        # already inherits the alpha T-relaxation) and Newton's exact,
        # UNDAMPED J_informed -- ta_solve._update_rho, unmodified. This
        # rho_fn is what the NEXT outer iteration's step 1 uses, preserving
        # Picard's own one-iteration lag exactly.
        #
        # newton_blend=1.0 (pure J_informed, tested first): the lag fix and
        # the memoryless-Jc/n fix each helped (stayed close to the I=196A
        # ground truth, 641.26 mT, for longer before drifting) but neither
        # eliminated a persistent bias-then-drift down to ~530-550 mT.
        # Root cause: Picard's OWN rho-informing J is computed from its
        # DAMPED T (inherits alpha's relaxation), so it is itself an
        # already-damped quantity -- Newton's fully undamped, exact
        # J_informed is a stronger perturbation per iteration than Picard's
        # careful T/rho co-damping balance assumes, even with the lag and
        # Jc/n-memory issues fixed. Blending toward J_coil (Picard's own,
        # already-damped estimate) restores that balance while still
        # letting Newton's more-accurate resolve pull rho_fn part of the
        # way toward a better target each iteration.
        J_for_rho = (1.0 - newton_blend) * J_coil + newton_blend * J_informed
        ta_solve._update_rho(ta, J_for_rho, B_coil, ic_model, n_model,
                             eps, relax=rho_relax)

        # Step 5: refresh Newton's own frozen Jc/n from the current B,
        # ready for the next iteration's informing pass. relax=1.0 (full
        # overwrite, memoryless) -- Jc/n are ONLY an input to Newton's
        # advisory solve; giving them their OWN separate relaxation memory
        # (on top of rho_fn's own log-space relaxation, which is the
        # single damping point Picard actually relies on) let two
        # independently-lagged coefficient tracks drift apart from each
        # other. A first version used relax=None (the ta_rho_relax
        # default, 0.5) here: tested against the I=196A ground truth, it
        # stabilised (no longer diverged) but plateaued at a WRONG value
        # (~545 mT vs 641.26 mT, a systematic ~15% low bias) -- a
        # different failure mode than instability, and this is the fix
        # being tested for it.
        for layer in range(n_layers):
            update_frozen_coefficients(ta, ic_model, n_model, layer, B_coil,
                                       relax=1.0)

        dB = np.linalg.norm((B_coil - B_prev).ravel())
        if not phase2 and k >= 4 and dB >= 0.95 * prev_dB_mag:
            phase2 = True
            alpha = alpha_low
            if verbose:
                print(f"      [hybrid] ramp-up done -> alpha={alpha_low}",
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

        if verbose:
            print(f"    [hybrid k={k+1:3d}] SCIF={scif_ema:+9.2f} mT  "
                  f"alpha={alpha:.2f}  |dB|={dB:.3e}  "
                  f"newton_failures_so_far={n_newton_failures}", flush=True)

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
        print(f"    [hybrid] step done: stop_reason={stop_reason}  "
              f"n_outer={n_outer}  SCIF={scif_str}  "
              f"newton_failures={n_newton_failures}", flush=True)

    return dict(converged=converged, n_outer=n_outer,
               total_snes_iters=0, stop_reason=stop_reason,
               scif_mT=float(scif_ema) if scif_ema is not None else float("nan"),
               scif_hist_tail=[float(s) for s in scif_hist],
               n_newton_failures=n_newton_failures)


def hybrid_march(ta, domain, uniform_setup, ic_model, n_model, schedule,
                 max_outer=100, min_outer=6, stall_tol=0.05,
                 bootstrap_iters=30, verbose=True, newton_blend=0.15):
    """Run a full multi-step schedule with hybrid_step. Same semantics as
    march() above -- every step's converged/stop_reason is tracked
    explicitly, a step is marched forward regardless of whether it hit the
    formal stall criterion, per-step trustworthiness is never assumed.
    """
    hist = []
    for n, (t, I_now, dt) in enumerate(schedule):
        if verbose:
            print(f"  step {n+1}/{len(schedule)}  t={t:7.1f} s  "
                  f"I={I_now:7.2f} A  dt={dt:6.1f} s", flush=True)
        info = hybrid_step(ta, domain, ic_model, n_model, I_now, dt,
                           uniform_setup, max_outer=max_outer,
                           min_outer=min_outer, stall_tol=stall_tol,
                           first=(n == 0), bootstrap_iters=bootstrap_iters,
                           verbose=verbose, newton_blend=newton_blend)
        info.update(t=t, I=I_now, dt=dt, step_index=n)
        hist.append(info)
        if verbose:
            print(f"  step {n+1}/{len(schedule)} SUMMARY: "
                  f"converged={info['converged']}  "
                  f"stop_reason={info['stop_reason']}  "
                  f"SCIF={info['scif_mT']:+.2f} mT", flush=True)
    return hist
