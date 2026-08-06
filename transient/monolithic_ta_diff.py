"""
monolithic_ta_diff.py -- monolithic T-A Newton with a GENUINELY
differentiable Jc(B)/n(B), closing the one gap newton_ta.py and
monolithic_ta.py both explicitly flagged and left open.

DOES NOT MODIFY solve/ta_solve.py, transient/newton_ta.py, or
transient/monolithic_ta.py. Purely additive, same pattern as those files.

WHY THIS EXISTS -- AND WHY NEITHER EARLIER FILE NEEDED IT
-----------------------------------------------------------
newton_ta.py's per-layer Newton solve and monolithic_ta.py's block Newton
solve are both "quasi-Newton": Jc(B,theta) and n(B,theta) are frozen DG0
coefficients within any single Newton solve, refreshed only between outer
iterations from measured-CSV scipy splines (not UFL-differentiable).

In newton_ta.py's Gauss-Seidel scheme (solve T with A/B held fixed, then
solve A with T held fixed, repeat), this limitation is actually INERT:
B is not a function of the unknown T being solved for AT ALL during a
single T-solve, so d(Jc)/dB contributes exactly zero to that Jacobian no
matter how Jc(B) is represented -- differentiability of Jc(B) w.r.t. B
literally cannot matter there. (This was confirmed empirically the same
day this file was written: transient/validation/half_domain_investigation
_2026-08-05.md's "frozen Jc/n for the whole step" follow-up found freezing
Jc/n changed nothing on the Gauss-Seidel path -- consistent with it having
already been inert.)

monolithic_ta.py DOES solve T and A simultaneously, so B = curl(A) is a
genuine live unknown alongside every T_i in the SAME Newton step -- but it
STILL freezes Jc_fn/n_fn as plain DG0 Functions (see its own docstring:
"STILL QUASI-NEWTON, SAME REASON AS newton_ta.py ... Ic(B,theta)/n(B,theta)
come from measured-CSV scipy splines ... ufl.derivative cannot form
d(Jc)/dB or d(n)/dB symbolically"). This is the ONE place in the project's
entire history where making Jc(B)/n(B) genuinely differentiable could
change anything -- and it has never actually been tried.

WHAT'S DIFFERENT HERE
----------------------
physics/entropy_ic_model.py fits Jc(B,theta) to the Long (2013) maximum-
entropy Beta model (regularized with a small field offset so it stays
finite at B=0 -- see that module's docstring) and n(B,theta) to a smooth
empirical Hill-type saturating decay, BOTH validated to <=2.3% MAPE over
the full measured grid and BOTH pure algebra in B. Only the ANGLE-
dependent parameters (5 for Jc, 4 for n, per layer) are frozen DG0
coefficients, refreshed between outer iterations exactly as
monolithic_ta.py refreshes Jc_fn/n_fn today -- but the B-MAGNITUDE
dependence is written as LIVE UFL algebra in ufl.curl(A_h), so
ufl.derivative(F, [T_0..T_{L-1}, A]) now includes the true dF/dA
contribution routed through Jc(|curl(A)|) and n(|curl(A)|), for the
first time in this project.

This mirrors newton_ta.py's own quasi-Newton philosophy one layer deeper:
that file made rho(J) exact in J while freezing Jc/n's field dependence
entirely; this file additionally makes Jc/n exact in |B| while freezing
only their (milder, slower-varying) ANGLE dependence.

HONEST EXPECTATION, STATED BEFORE RUNNING ANYTHING
----------------------------------------------------
Three structurally different schemes -- Picard, a Gauss-Seidel Newton
hybrid, and a fully monolithic block Newton with frozen Jc/n -- have all
hit the IDENTICAL short-dt chaotic-wandering failure signature (see
CLAUDE.md's "NI transient work" and transient/validation/half_domain_
investigation_2026-08-05.md). The frozen-Jc/n follow-up test in that same
file found that REMOVING Jc/n's per-iteration B-dependence entirely
(freezing it for a whole step) changed NOTHING on the Gauss-Seidel path --
but as argued above, that path was never sensitive to this in the first
place, so that null result does not predict what happens here. This is a
genuinely new, previously-untried configuration, not a retest of
something already shown not to matter. It may fix nothing -- the bit-
level diff in transient/validation/nondeterminism_investigation_2026-08-05
.md found ~1e17-1e19-fold amplification in the LINEAR T-sub-problem alone,
a form of ill-conditioning this change does not address. Report whatever
actually happens, including a null result, with the same repeat-count
rigor as every other experiment in this project's history.
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
from entropy_ic_model import EntropyBetaIcModel, HillNModel  # noqa: E402
from monolithic_ta import (DEFAULT_SNES_OPTIONS,        # noqa: E402
                           _REAL_FAILURE_REASONS,
                           MONO_QUADRATURE_DEGREE)

mu0 = 4.0 * np.pi * 1e-7

# Safe nonzero defaults for cells never touched by a layer's own update
# (outside that layer -- T is Dirichlet-pinned to zero there, so the
# physical contribution is exactly zero regardless of these values; see
# newton_ta.py gotcha #2). Birr/Bn MUST be nonzero (denominators).
_SAFE_DEFAULTS_JC = dict(A=1000.0, alpha=1.0, beta=3.0, Birr=100.0, B0=0.1)
_SAFE_DEFAULTS_N = dict(n0=25.0, ninf=20.0, Bn=1.0, p=1.5)


def build_monolithic_problem_diff(ta, ic_beta_model=None, n_hill_model=None,
                                  snes_options=None, verbose=False,
                                  lm_lambda=0.0):
    """Build the block Newton system with Jc(|curl(A)|)/n(|curl(A)|)
    genuinely live in the UFL residual.

    Adds to `ta` (does not touch anything monolithic_ta.py/newton_ta.py/
    ta_solve.py already put there -- all keys are 'monodiff_'-prefixed):
        ta["monodiff_ic_model"], ta["monodiff_n_model"] : the fitted models
        ta["monodiff_param_fns"] : list (per layer) of dicts of the 9 DG0
            frozen angle-parameter Functions
        ta["monodiff_problem"], ta["monodiff_snes"]
        ta["monodiff_T_anchor"] : list of per-layer T anchor Functions
            (only meaningful if lm_lambda != 0 -- see below)

    Requires ta built with per_layer=True (same requirement as
    monolithic_ta.py).

    lm_lambda: 2026-08-05 addition. 0.0 (default) is IDENTICAL to the
    original formulation -- inert, changes nothing. When nonzero, adds a
    Levenberg-Marquardt-style proximal term
        + lm_lambda * inner(T_i - T_anchor_i, phi) * dx
    to each layer's residual, where T_anchor_i is a plain DG-- no, CG1 --
    fem.Function snapshot the CALLER must refresh to the CURRENT T_i value
    immediately before each problem.solve() (see monolithic_diff_step's
    T_snap/A_snap pattern -- the anchor IS that snapshot, just also fed
    into the residual instead of only used for post-hoc reverting).

    WHY THIS FORM SPECIFICALLY: this term is IDENTICALLY ZERO whenever
    T_i == T_anchor_i, i.e. at any point where the anchor has already been
    refreshed to the current state -- so it does not bias the fixed point
    a converged solve settles on. Its ONLY effect is on the JACOBIAN: it
    adds lm_lambda to the diagonal of each T_i block (a mass matrix, mildly
    positive-definite), lifting singular values away from zero exactly the
    way classical Levenberg-Marquardt damping does, without needing to
    intercept dolfinx's assembled PETSc Mat directly. This targets the
    diagnosed cause of the block-Newton divergence found the same day
    (mesh-2026-08-05 live debugging session): `snes_linesearch_monitor`
    showed ynorm ~1e16 on the failing step (a near-singular Jacobian, not a
    sign error), and `newtontr` confirmed it by correctly refusing to move
    (residual unchanged to 9 significant figures) rather than either
    masking it (`basic`, which silently diverged behind a misleading SCIF
    plateau) or just giving up (`bt`, reason=-6).

    Only applied to the T-equations for now, not the A-equation, since A
    already has its own Tikhonov-style regularizer
    (params.gauge_regularization * inner(A, v) * dx) built into
    ta_solve.py's original A-form; T has no analogous self-damping term at
    all. lm_lambda's correct magnitude is NOT derived analytically here --
    like ta_eps_reg/ta_rho_relax/ta_floor_smooth_p elsewhere in this
    project, it is found empirically; see
    transient/validation/monolithic_diff_investigation_2026-08-05.md for
    the sweep that determined a working value, if one was found.
    """
    from dolfinx import fem
    from dolfinx.fem.petsc import NonlinearProblem

    ic_beta_model = ic_beta_model or EntropyBetaIcModel()
    n_hill_model = n_hill_model or HillNModel()

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

    # ── B = curl(A_h), LIVE UFL, shared by every layer's Jc/n expression ──
    B_ufl = ufl.curl(ta["A_h"])
    Bmag_ufl = ufl.sqrt(ufl.inner(B_ufl, B_ufl) + 1e-30)

    T_anchor_fns = []
    param_fns = []
    F_list = []
    for i, T_i in enumerate(layer_T_fns):
        # 5 frozen angle-parameters for Jc, 4 for n -- ONLY these are
        # Picard-lagged; Bmag_ufl above is live in every one of them.
        fns = {}
        for name, default in _SAFE_DEFAULTS_JC.items():
            fn = fem.Function(Vdg0, name=f"jc_{name}_{i}")
            fn.x.array[:] = default
            fns[f"jc_{name}"] = fn
        for name, default in _SAFE_DEFAULTS_N.items():
            fn = fem.Function(Vdg0, name=f"n_{name}_{i}")
            fn.x.array[:] = default
            fns[f"n_{name}"] = fn
        param_fns.append(fns)

        Jc_expr = EntropyBetaIcModel.jc_ufl_expr(
            Bmag_ufl, fns["jc_A"], fns["jc_alpha"], fns["jc_beta"],
            fns["jc_Birr"], fns["jc_B0"])
        n_expr = HillNModel.n_ufl_expr(
            Bmag_ufl, fns["n_n0"], fns["n_ninf"], fns["n_Bn"], fns["n_p"])

        phi = ufl.TestFunction(ta["V_T"])
        J_ufl = ufl.cross(ufl.grad(T_i), n_hat_ufl)
        Jmag = ufl.sqrt(ufl.inner(J_ufl, J_ufl) + 1e-30)
        j_norm_raw = Jmag / Jc_expr
        j_norm = (eps_reg ** p_floor + j_norm_raw ** p_floor) ** (1.0 / p_floor)
        rho_SC = (E_c / Jc_expr) * ufl.exp((n_expr - 1.0) * ufl.ln(j_norm))
        rho_expr = rho_SC * (delta_SC / Lambda)

        J_phi = ufl.cross(ufl.grad(phi), n_hat_ufl)
        F_i = (rho_expr * ufl.inner(J_ufl, J_phi) * dx
               + (1.0 / dt_const) * coil_ind
                 * ufl.inner(ufl.curl(ta["A_h"] - ta["A_prev"]),
                             phi * n_hat_ufl) * dx)

        T_anchor_i = fem.Function(ta["V_T"], name=f"T_anchor_{i}")
        T_anchor_i.x.array[:] = T_i.x.array
        T_anchor_fns.append(T_anchor_i)
        if lm_lambda != 0.0:
            F_i = F_i + lm_lambda * ufl.inner(T_i - T_anchor_i, phi) * dx

        F_list.append(F_i)

    # ── Shared A-equation -- identical structure to monolithic_ta.py ──────
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
                               petsc_options_prefix="monodiff_ta_",
                               petsc_options=opts)
    snes = problem.solver
    snes.setTolerances(max_it=1)   # same single-step-then-refresh scheme
                                   # as monolithic_ta.py -- see that
                                   # module's docstring for why this is
                                   # the correct way to wire the outer
                                   # coefficient refresh.

    ta["monodiff_ic_model"] = ic_beta_model
    ta["monodiff_n_model"] = n_hill_model
    ta["monodiff_param_fns"] = param_fns
    ta["monodiff_problem"] = problem
    ta["monodiff_snes"] = snes
    ta["monodiff_T_anchor"] = T_anchor_fns
    ta["monodiff_lm_lambda"] = float(lm_lambda)
    return ta


def _refresh_lm_anchors(ta):
    """Snapshot the CURRENT T_i values into the LM proximal anchors. Call
    immediately before every problem.solve() when lm_lambda != 0 -- if the
    anchor is stale (left at an old T), the proximal term biases the
    residual toward that stale value instead of merely regularizing the
    Jacobian, which is not what this is for. No-op (but harmless) if
    lm_lambda == 0.0, since the anchor term is never added to F in that
    case."""
    for T_i, anchor in zip(ta["layer_T_fns"], ta["monodiff_T_anchor"]):
        anchor.x.array[:] = T_i.x.array
        anchor.x.scatter_forward()


def _update_monodiff_coefficients(ta, layer, B_coil_all, relax=None):
    """Refresh layer `layer`'s frozen ANGLE parameters (5 for Jc, 4 for n)
    from the CURRENT B field's direction. Same relaxation philosophy as
    monolithic_ta.py's _update_mono_coefficients -- undamped updates once
    close to the fixed point were found to trigger spurious line-search
    failures there, and there is no reason to expect this scheme is less
    sensitive to that.
    """
    idx = ta["layer_cell_idx"][layer]
    coil_cells = ta["coil_cells"]
    cells_layer = coil_cells[idx]
    n_hat_layer = ta["n_hat_coil"][idx]

    B_layer = B_coil_all[idx]
    Bmag = np.linalg.norm(B_layer, axis=1)
    theta = angle_with_normal_deg(B_layer, n_hat_layer)

    ic_model = ta["monodiff_ic_model"]
    n_model = ta["monodiff_n_model"]
    A, alpha, beta, Birr, B0 = ic_model._params_at(theta)
    n0, ninf, Bn, p = n_model._params_at(theta)

    # EntropyBetaIcModel is fit directly against IcModel.critical_current(),
    # i.e. Ic in AMPS (matching the CSV's native convention -- see that
    # module's docstring). jc_ufl_expr's "A" prefactor is therefore in amps
    # too, but rho_expr in build_monolithic_problem_diff() divides Jmag
    # (physical SC-layer current density, A/m^2, O(1e11)) by Jc_expr -- it
    # needs the VOLUMETRIC critical current density, exactly as
    # ta_solve._update_rho computes Jc_vol = Ic_arr/(delta_SC*tape_width)
    # for the production Picard/Newton paths. A is linear in the Beta
    # formula, so the unit conversion is a single scalar rescale here
    # rather than needing to touch EntropyBetaIcModel's own (correct, amps-
    # native) fit at all.
    A = A / (ta["delta_SC"] * ic_model.base.tape_width)

    relax = float(getattr(params, "ta_rho_relax", 0.5)) if relax is None else relax
    fns = ta["monodiff_param_fns"][layer]
    for key, arr in [("jc_A", A), ("jc_alpha", alpha), ("jc_beta", beta),
                     ("jc_Birr", Birr), ("jc_B0", B0),
                     ("n_n0", n0), ("n_ninf", ninf), ("n_Bn", Bn), ("n_p", p)]:
        fn = fns[key]
        fn.x.array[cells_layer] = ((1.0 - relax) * fn.x.array[cells_layer]
                                   + relax * arr)
        fn.x.scatter_forward()

    # for diagnostics: return the mean j/jc-relevant Jc at this layer's
    # cells (evaluated with the JUST-updated angle params, at the CURRENT
    # B magnitude) -- purely informational, matches monolithic_ta.py's
    # clip_frac return value in spirit (a health signal, not used by the
    # solve itself)
    Ic_check, _ = ic_model.critical_current(Bmag, theta)
    return float(np.mean(Ic_check))


def monolithic_diff_step(ta, domain, ic_model_spline, n_model_spline,
                         I_now, dt, uniform_setup, max_outer=150,
                         min_outer=6, stall_tol=0.05, first=False,
                         bootstrap_iters=30, verbose=True, jc_n_relax=None,
                         step_relax=1.0, debug=False):
    """Advance one time step with the differentiable-Jc/n block Newton
    system. Structurally identical outer loop to
    monolithic_ta.monolithic_step() -- see that function's docstring for
    the bootstrap/step_relax/debug rationale, all unchanged here. The
    ONLY difference is which coefficient-update function is called
    (_update_monodiff_coefficients, refreshing 9 angle-parameters instead
    of 2 frozen values) and which problem/snes objects are used.

    ic_model_spline/n_model_spline: the ORIGINAL measured-CSV models
    (physics/ic_model.IcModel/NValueModel or an extrapolation wrapper),
    needed ONLY for the cold-start Picard bootstrap (newton_ta.py's
    _picard_bootstrap uses these directly, exactly as monolithic_ta.py's
    bootstrap does) -- the block Newton solve itself uses
    ta["monodiff_ic_model"]/ta["monodiff_n_model"] (the fitted Beta/Hill
    models) via _update_monodiff_coefficients.
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
        J_seed = ta_transient_seed_cold(ta, uniform_setup, ic_model_spline,
                                        n_model_spline, I_now)
        B_h.interpolate(ta["curl_expr"])
        B_seed = B_h.x.array.reshape(-1, 3)[coil]
        ta["_rho_prev"] = None
        ta_solve._update_rho(ta, J_seed, B_seed, ic_model_spline,
                             n_model_spline, eps)
        _picard_bootstrap(ta, domain, ic_model_spline, n_model_spline,
                          I_now, dt, n_iters=bootstrap_iters, verbose=verbose)
        B_h.interpolate(ta["curl_expr"])
        B_now = B_h.x.array.reshape(-1, 3)[coil]
        for layer in range(n_layers):
            _update_monodiff_coefficients(ta, layer, B_now, relax=1.0)

    problem = ta["monodiff_problem"]

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
            _update_monodiff_coefficients(ta, layer, B_coil, relax=relax_k)

        T_snap = [T_i.x.array.copy() for T_i in ta["layer_T_fns"]]
        A_snap = ta["A_h"].x.array.copy()

        if ta.get("monodiff_lm_lambda", 0.0) != 0.0:
            _refresh_lm_anchors(ta)

        problem.solve()
        reason = ta["monodiff_snes"].getConvergedReason()
        its = ta["monodiff_snes"].getIterationNumber()
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
                print(f"      [monodiff] block Newton step failed "
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
            print(f"    [monodiff k={k+1:3d}] SCIF={scif_ema:+9.2f} mT  "
                  f"reason={reason}  its={its}", flush=True)

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
        print(f"    [monodiff] step done: stop_reason={stop_reason}  "
              f"n_outer={n_outer}  SCIF={scif_str}", flush=True)

    return dict(converged=converged, n_outer=n_outer,
               total_snes_iters=total_snes_iters, stop_reason=stop_reason,
               scif_mT=float(scif_ema) if scif_ema is not None else float("nan"),
               scif_hist_tail=[float(s) for s in scif_hist])
