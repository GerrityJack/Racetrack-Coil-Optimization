"""
ta_implicit.py — T-A solve with the inductive T-A coupling made IMPLICIT.
=========================================================================
2026-09-26. Additive; does not change ta_solve.solve_ta_at_current().

Why. solve_ta_at_current() is a staggered Picard scheme: the T-equation's
right-hand side -(1/dt) curl(A - A_prev).n uses the A (self-field) of the
PREVIOUS iterate, and A is then recomputed from the new T. That is an
explicit treatment of the tape's self-inductance, with loop gain
    G ~ mu0 * w * D / (rho_SC * dt)            (w tape width, D winding build)
~30 at the production rho-floor (eps_reg=1, dt=600 s) and growing as
eps^-(n-1) when the floor is lowered and as 1/dt for short steps. Stable
only for relaxation alpha < ~1/G -- which is why dt=60 s needed a 10x
smaller alpha, and why eps < ~0.8 has never converged in this project.

What this does. For FROZEN rho the coupled T-A problem is LINEAR:
    T = F(T),  F(T) = M(rho)^-1 [ b_bc - (1/dt) C (A(T) - A_prev) ]
(F = one T+A sweep: J from T, A back-substitution, per-layer T solve).
Each outer iteration solves this fixed point exactly with matrix-free
GMRES on (I - dF) dT = F(T_k) - T_k, so the self-inductance is implicit
and no T relaxation is needed. The T matrices are assembled/factorised
once per outer iteration; every GMRES matvec is back-substitutions only.
The outer loop then iterates rho(J, B) (Jc(B,theta), n(B,theta), floor)
with a per-cell secant-Newton update in log(rho) (rho_update="newton"),
or the legacy J-based / E-based updates for comparison.

Returns the same (A_h, B_h, T_h, info) as solve_ta_at_current and writes
the same diag CSV columns (plus gmres iterations), so the study drivers
can use either solver.
"""
import os
import time
import csv

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from scipy.sparse.linalg import LinearOperator, gmres

import dolfinx
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting

import params
from ic_model import angle_with_normal_deg
import ta_solve
from ta_solve import (_J_from_T, _update_Js, _solve_A, dB_bore_from_dJ,
                      _update_rho)

E_C = 1e-4   # V/m


# ── split LinearProblem into (assemble+factor) and (rhs+back-substitute) ──

def _assemble_factor(prob):
    prob.A.zeroEntries()
    assemble_matrix(prob.A, prob.a, bcs=prob.bcs)
    prob.A.assemble()
    prob.solver.setOperators(prob.A)
    prob.solver.setUp()          # MUMPS factorisation happens here


def _rhs_solve(prob, out):
    b = prob._b
    dolfinx.la.petsc._zero_vector(b)
    assemble_vector(b, prob.L)
    apply_lifting(b, [prob.a], bcs=[prob.bcs])
    dolfinx.la.petsc._ghost_update(b, PETSc.InsertMode.ADD,
                                   PETSc.ScatterMode.REVERSE)
    for bc in prob.bcs:
        bc.set(b.array_w)
    prob.solver.solve(b, prob._x)
    out[:] = prob._x.array_r[:len(out)]


# ── rho(J, B) with the production soft floor ──────────────────────────────

def _rho_parts(ta, J_coil, B_coil, ic_model, n_model):
    n_hat = ta["n_hat_coil"]
    B_mag = np.linalg.norm(B_coil, axis=-1)
    theta = angle_with_normal_deg(B_coil, n_hat)
    Ic_arr, _ = ic_model.critical_current(B_mag, theta)
    Jc = Ic_arr / (ta["delta_SC"] * ic_model.tape_width)
    n_arr, _ = n_model.n_value(B_mag, theta)
    Jd = np.einsum("ij,ij->i", J_coil, n_hat)
    Jm = np.linalg.norm(J_coil - Jd[:, None] * n_hat, axis=-1)
    return Jm, Jc, n_arr


def _log_rho_from_jr(jr, Jc, n_arr, eps, p):
    """log of homogenised rho for normalised current jr = J/Jc."""
    jn = (eps**p + jr**p) ** (1.0 / p) if p > 0 else np.maximum(jr, eps)
    log_rho_sc = np.log(E_C / Jc) + (n_arr - 1.0) * np.log(np.maximum(jn, 1e-300))
    return log_rho_sc, jn


def _set_rho(ta, log_rho_homog, scale=1.0):
    """scale multiplies rho in the T bilinear form; the caller divides dt by
    the same factor so the T solution is unchanged (see solve_ta_implicit)."""
    ta["rho_fn"].x.array[:] = 0.0
    ta["rho_fn"].x.array[ta["coil_cells"]] = np.exp(log_rho_homog) * scale
    ta["rho_fn"].x.scatter_forward()


# ── the solver ───────────────────────────────────────────────────────────

def solve_ta_implicit(domain, ta, uniform_setup, I_amps, ic_model, n_model,
                      n_outer=60, warm_start=True, rho_update="newton",
                      rho_relax=1.0, max_dlog=2.0, gmres_rtol=1e-6,
                      gmres_maxit=60, gmres_restart=30, diag_log_path=None,
                      verbose=False):
    comm = domain.comm
    assert ta["per_layer"], "implicit solver implemented for per-layer T only"
    delta_SC, Lambda = float(ta["delta_SC"]), float(ta["Lambda"])
    eps = float(getattr(params, "ta_eps_reg", 1.0))
    p = float(getattr(params, "ta_floor_smooth_p", 16.0) or 0.0)
    to_homog = np.log(delta_SC / Lambda)
    B_h = ta["B_fn"]
    probs = ta["prob_T_layers"]
    Tfns = ta["layer_T_fns"]
    nL = len(Tfns)
    N = len(Tfns[0].x.array)

    T_amp = I_amps / (2.0 * delta_SC)
    ta["T_bot_val"].value = +T_amp
    ta["T_top_val"].value = -T_amp

    # ── initial state (same as solve_ta_at_current) ──────────────────────
    warm = (warm_start and ta.get("_I_prev") is not None
            and ta.get("_T_prev_layers") is not None)
    if warm:
        r = I_amps / ta["_I_prev"]
        for T_i, prev in zip(Tfns, ta["_T_prev_layers"]):
            T_i.x.array[:] = prev * r
            T_i.x.scatter_forward()
        J_coil = _J_from_T(ta, domain)
        _update_Js(ta, J_coil)
        _solve_A(ta, ta["L_A_form"])
    else:
        uniform_setup["J"].x.array[:] = (uniform_setup["J_unit_array"]
                                         * I_amps / (params.t * params.w))
        _solve_A(ta, ta["L_seed_form"])
        for T_i in Tfns:
            T_i.x.array[:] = 0.0
            T_i.x.scatter_forward()
        J_coil = ta["t_hat_coil"] * (I_amps / (delta_SC * params.w))
        _update_Js(ta, J_coil)
    B_h.interpolate(ta["curl_expr"])
    B_coil = B_h.x.array.reshape(-1, 3)[ta["coil_cells"]].copy()

    # initial rho: keep a warm state's rho if we have it (it is the rho the
    # previous converged solve ended on), else the J-based seed
    if warm and ta.get("_impl_log_rho") is not None and \
            ta.get("_impl_eps") == eps:
        log_rho = ta["_impl_log_rho"].copy()
    else:
        Jm, Jc, n_arr = _rho_parts(ta, J_coil, B_coil, ic_model, n_model)
        lrs, _ = _log_rho_from_jr(Jm / Jc, Jc, n_arr, eps, p)
        log_rho = lrs + to_homog
    # Conditioning: T-matrix rows scale as rho*h ~ 1e-21 while Dirichlet /
    # pinned rows have diagonal 1. Multiplying rho (bilinear form) and 1/dt
    # (the only other term, in the RHS) by the same factor c leaves T
    # unchanged but brings the physics rows to O(1).
    dt_true = float(ta["dt_const"].value)
    h_typ = float(params.w) * 0.05
    c_scale = 1.0 / (float(np.exp(np.median(log_rho))) * h_typ)
    ta["dt_const"].value = dt_true / c_scale
    _set_rho(ta, log_rho, c_scale)

    # ── F(T): one T+A sweep at frozen rho (factorised T matrices) ────────
    def set_T(vec):
        for i, T_i in enumerate(Tfns):
            T_i.x.array[:] = vec[i * N:(i + 1) * N]
            T_i.x.scatter_forward()

    def get_T():
        return np.concatenate([T_i.x.array.copy() for T_i in Tfns])

    n_matvec = [0]

    def F(vec):
        n_matvec[0] += 1
        set_T(vec)
        Jc_ = _J_from_T(ta, domain)
        _update_Js(ta, Jc_)
        _solve_A(ta, ta["L_A_form"])
        out = np.empty(nL * N)
        for i, prob in enumerate(probs):
            _rhs_solve(prob, out[i * N:(i + 1) * N])
        return out

    J_unif_vec = ta["t_hat_coil"] * (I_amps / (delta_SC * params.w))
    prev_logJ = prev_logrho = None
    converged = False
    rel_err = np.nan
    scif = np.nan
    t0 = time.time()
    for k in range(n_outer):
        for prob in probs:
            _assemble_factor(prob)
        Tk = get_T()
        Fk = F(Tk)
        r = Fk - Tk
        rnorm = np.linalg.norm(r)
        tnorm = np.linalg.norm(Tk) + 1e-300

        # F is affine, so dF(v) = (F(Tk + c v) - F(Tk))/c for any c. GMRES
        # calls this with unit-norm v while |Tk| ~ 1e10: c scales the probe
        # to |Tk| so the difference keeps full precision.
        Tscale = max(np.linalg.norm(Tk), np.linalg.norm(r), 1.0)

        def mv(v):
            vn = np.linalg.norm(v)
            if vn == 0.0:
                return v.copy()
            c = Tscale / vn
            return v - (F(Tk + c * v) - Fk) / c
        op = LinearOperator((nL * N, nL * N), matvec=mv, dtype=np.float64)
        m0 = n_matvec[0]
        tg = time.time()
        hist = []

        def cb(res):
            hist.append(res)
            if verbose and comm.rank == 0 and len(hist) % 10 == 0:
                print(f"      gmres it {len(hist):4d}  rel.res {res:.2e}  "
                      f"({time.time()-tg:.1f}s)", flush=True)
        dT, info_g = gmres(op, r, rtol=gmres_rtol, atol=0.0,
                           restart=gmres_restart,
                           maxiter=max(1, gmres_maxit // gmres_restart),
                           callback=cb, callback_type="pr_norm")
        n_gm = n_matvec[0] - m0
        # true linear residual of the returned correction
        lin_res = np.linalg.norm(r - mv(dT)) / (rnorm + 1e-300)
        T_new = Tk + dT
        set_T(T_new)
        J_coil = _J_from_T(ta, domain)
        _update_Js(ta, J_coil)
        _solve_A(ta, ta["L_A_form"])
        if np.any(~np.isfinite(ta["A_h"].x.array)):
            raise RuntimeError(f"[implicit k={k+1}] non-finite A")
        B_h.interpolate(ta["curl_expr"])
        B_new = B_h.x.array.reshape(-1, 3)[ta["coil_cells"]].copy()
        rel_err = np.linalg.norm(B_new - B_coil) / (np.linalg.norm(B_new) + 1e-30)
        B_coil = B_new

        scif = dB_bore_from_dJ(ta["coil_centroids"],
                               (J_coil - J_unif_vec) * (delta_SC / Lambda),
                               ta["coil_vols"])[2] * 1e3

        # ── rho update ────────────────────────────────────────────────────
        Jm, Jc, n_arr = _rho_parts(ta, J_coil, B_coil, ic_model, n_model)
        jr = Jm / Jc
        lrs_J, _ = _log_rho_from_jr(jr, Jc, n_arr, eps, p)
        target_J = lrs_J + to_homog
        logJ = np.log(np.maximum(Jm, 1e-300))
        if rho_update == "J":
            new = target_J
        elif rho_update == "E":
            # E = rho_used*J (exact: J was solved with rho_used, no relaxation)
            lE = (log_rho - to_homog) + logJ
            jrE = np.exp((lE - np.log(E_C)) / n_arr)
            lrs_E, _ = _log_rho_from_jr(jrE, Jc, n_arr, eps, p)
            new = lrs_E + to_homog
        elif rho_update == "newton":
            # local model log J = c - s*log rho; secant estimate of s
            if prev_logrho is None:
                s = np.ones_like(logJ)
            else:
                dl = log_rho - prev_logrho
                ok = np.abs(dl) > 1e-4
                s = np.ones_like(logJ)
                s[ok] = -(logJ[ok] - prev_logJ[ok]) / dl[ok]
                s = np.clip(s, 0.0, 1.0)
            phi = jr**p / (eps**p + jr**p) if p > 0 else (jr > eps).astype(float)
            new = log_rho + (target_J - log_rho) / (1.0 + (n_arr - 1.0) * phi * s)
        else:
            raise ValueError(rho_update)
        step = np.clip(rho_relax * (new - log_rho), -max_dlog, max_dlog)
        prev_logrho, prev_logJ = log_rho.copy(), logJ
        log_rho = log_rho + step
        _set_rho(ta, log_rho, c_scale)
        # consistency residual: how far rho is from rho(J) at this state
        rho_res = float(np.sqrt(np.mean((target_J - prev_logrho) ** 2)))

        if verbose and comm.rank == 0:
            print(f"  [impl k={k+1:02d}] SCIF={scif:+9.2f} mT |dB|/|B|={rel_err:.2e} "
                  f"gmres={n_gm} (info {info_g}, true {lin_res:.1e}) |r|/|T|={rnorm/tnorm:.1e} "
                  f"rho_res={rho_res:.2e} max|dlog rho|={np.abs(step).max():.2f} "
                  f"({time.time()-t0:.0f}s)")
        if diag_log_path is not None and comm.rank == 0:
            new_f = not os.path.exists(diag_log_path)
            with open(diag_log_path, "a", newline="") as fh:
                w = csv.writer(fh)
                if new_f:
                    w.writerow(["k", "rel_err", "scif_mT", "scif_stall_mT",
                                "T_min", "T_max", "j_over_jc_central", "corr",
                                "gmres_it", "gmres_info", "rho_res",
                                "max_dlogrho", "fp_res", "gmres_true_res"])
                w.writerow([k + 1, rel_err, scif, np.nan, T_new.min(),
                            T_new.max(), float(np.mean(jr)), np.nan, n_gm,
                            info_g, rho_res, float(np.abs(step).max()),
                            rnorm / tnorm, lin_res])

    # final state: T_new with the LAST rho used for it (prev_logrho);
    # store the rho that goes with the returned state for warm restarts
    ta["_I_prev"] = I_amps
    ta["_T_prev_layers"] = [T_i.x.array.copy() for T_i in Tfns]
    _set_rho(ta, prev_logrho)          # rho actually used for the final T
    ta["dt_const"].value = dt_true    # unscaled again for other callers
    ta["_impl_log_rho"] = prev_logrho.copy()
    ta["_impl_eps"] = eps
    ta["_rho_prev"] = None
    info = dict(n_iters=n_outer, converged=converged, rel_err=rel_err,
                scif_mT=scif, rho_res=rho_res)
    return ta["A_h"], B_h, None, info
