"""
entropy_ic_model.py -- UFL-differentiable Jc(B,theta) and n(B,theta)
=====================================================================
2026-08-05. Built for the T-A transient solver's "full Newton" experiment:
a genuinely differentiable Jacobian for Jc(B)/n(B) only matters for a
MONOLITHIC T-A solve (T and A solved simultaneously, so B = curl(A) is a
live unknown) -- in the project's sequential Gauss-Seidel scheme (the
production path and newton_ta.py's quasi-Newton hybrid), B is frozen for
the entire duration of any one T-solve, so d(Jc)/dB contributes exactly
zero to that Jacobian no matter how Jc(B) is represented. See
transient/monolithic_ta_diff.py for the solver that actually uses this.

Both physics/ic_model.py's IcModel/NValueModel (grid interpolation /
scipy spline) and optimize/ic_extrapolation.py's Beta*IcModel (fit only to
a 7-point >8T extrapolation tail) are unsuitable here: neither is
symbolic UFL algebra ufl.derivative can differentiate automatically, and
the existing Beta fit actively diverges at B=0 for 41/43 angles (alpha<1)
-- fatal for a zero-field-cooled solver. This module fits smooth, fully
algebraic closed forms to the FULL measured 0-8T grid instead.

Jc(B, theta): Long (2013) maximum-entropy Beta model, Eq. 2
--------------------------------------------------------------
    N. J. Long, "Maximum Entropy Distributions Describing Critical
    Currents in Superconductors", Entropy 2013, 15(7), 2585-2605.

    Jc(b) ~ b^(alpha-1) * (1-b)^(beta-1),   b = B / B_irr

deriving from a max-entropy argument on the pinning-site field
distribution (see optimize/ic_extrapolation.py's longer docstring for the
full derivation). Fit here PER ANGLE (A, alpha, beta, Birr) to ALL 20
measured field points (not just the 7-point tail optimize/ic_extrapolation
uses for extrapolation only), which is a materially different fit
target and gives a materially different, much better-conditioned result.

DEVIATION FROM THE PAPER'S LITERAL EQUATION, STATED PLAINLY: fitting this
tape's data with the constraint alpha>=1 (needed so Jc(0) stays finite --
the raw Eq. 2 diverges at b=0 for alpha<1) forces poor fit quality
(MAPE ~8%, and alpha pinned to its 1.0 floor at every angle -- the
unconstrained optimum genuinely wants alpha<1). Adding a small field
offset B0, i.e. b = (B+B0)/Birr, removes the b->0 singularity while
letting alpha take its unconstrained best-fit value -- the SAME
regularization philosophy as ta_solve.py's eps_reg smooth floor on the
power-law resistivity, applied here to keep this equation's own bounded
low-field behaviour physical. This is a deliberate, minimal, disclosed
modification, not a silent deviation:  MAPE improves from ~8% to ~2.3%
over the full grid, and Jc(0) becomes accurate to 1-3% at every angle
(vs. literally divergent before).

VALIDATED RANGE: this fit is validated against the MEASURED 0-8T grid
only (MAPE 2.3%, worst single point 10.2%). Behaviour above 8T (needed
for the champion's full I_design ~10.5T operating point) has NOT been
validated here -- optimize/ic_extrapolation.py's KimIcModel remains the
project's validated choice for that regime. Do not use this module for
quench/B_target design numbers; it exists for the transient solver's
Jacobian experiment, at the much lower currents (and therefore fields)
those tests use.

n(B, theta): empirical Hill-type saturating decay -- NOT from the paper
-------------------------------------------------------------------------
    n(B) = n_inf + (n0 - n_inf) / (1 + (B/Bn)^p)

Long (2013) is a Jc-specific derivation; there is no comparable
entropy-maximization result for the E-J power-law exponent n used here.
This form is chosen purely because it fits extremely well (MAPE 0.47%,
worst single point 2.9%) and is smooth, bounded, and fully algebraic in
B for every theta -- suitable for exact UFL differentiation for the same
structural reason as the Jc fit above, with no physical derivation
claimed.

Both fits are per-angle; angle interpolation between the two nearest
fitted angles is LINEAR IN THE PARAMETERS (matching
optimize/ic_extrapolation.py's _extrap()), and is Picard-lagged in the
monolithic solver (frozen DG0 coefficient fields, updated between outer
iterations from the current field direction) -- only the B-MAGNITUDE
dependence is genuinely live/differentiable within a Newton solve. This
mirrors newton_ta.py's existing quasi-Newton design (exact Newton on the
dominant nonlinearity, Picard-lag on the milder one), extended one layer
deeper: there, rho(J) was made exact in J with Jc/n frozen constants;
here, Jc(B)/n(B) are additionally made exact in |B| with only their
ANGLE dependence frozen.
"""
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params  # noqa: E402
from ic_model import IcModel, NValueModel  # noqa: E402


# ── Jc(B, theta): regularized Long (2013) Beta model ────────────────────────

def _law_beta_reg(B, A, alpha, beta, Birr, B0):
    b = np.clip((B + B0) / Birr, 1e-12, 1.0 - 1e-12)
    return A * b ** (alpha - 1.0) * (1.0 - b) ** (beta - 1.0)


class EntropyBetaIcModel:
    """Per-angle (A, alpha, beta, Birr, B0) fit of Long (2013) Eq. 2 (with
    the B0 field-offset regularization -- see module docstring) to the
    FULL measured Jc(B,theta) grid. Fully algebraic in B; meant to be
    written directly as a UFL expression (see jc_ufl_expr) so
    ufl.derivative differentiates it exactly."""

    def __init__(self, csv_path=None, base_ic_model=None):
        base = base_ic_model or IcModel(csv_path or params.csv_filename)
        self.base = base
        self.B_vals = base.field_vals
        self.angles = base.angle_vals
        self.Ic_min, self.Ic_max = base.Ic_min, base.Ic_max

        self.params = {}       # angle -> (A, alpha, beta, Birr, B0)
        self.fit_rows = []
        for ang in self.angles:
            Ic, _ = base.critical_current(
                self.B_vals, np.full_like(self.B_vals, ang), clip_B=False)
            popt, _ = curve_fit(
                _law_beta_reg, self.B_vals, Ic,
                p0=[Ic[0], 0.8, 3.0, 30.0, 0.05],
                bounds=([1e-3, 0.3, 1.0, 15.0, 1e-4],
                        [1e6, 4.0, 40.0, 400.0, 2.0]),
                maxfev=200000)
            self.params[float(ang)] = tuple(popt)
            pred = _law_beta_reg(self.B_vals, *popt)
            err = np.abs(pred - Ic) / np.maximum(Ic, 1e-9) * 100
            self.fit_rows.append(dict(angle_deg=float(ang),
                                      mape_pct=float(err.mean()),
                                      max_err_pct=float(err.max())))

    # -- numpy evaluation (drop-in-ish with IcModel.critical_current) -------

    def _params_at(self, theta_deg):
        """Per-point (A, alpha, beta, Birr, B0), linearly interpolated
        between the two nearest fitted angles -- same scheme as
        optimize/ic_extrapolation.py's _ExtrapolatedIcModel._extrap()."""
        theta = np.clip(np.asarray(theta_deg, dtype=np.float64),
                        self.angles[0], self.angles[-1])
        idx = np.clip(np.searchsorted(self.angles, theta), 1,
                      len(self.angles) - 1)
        a_lo, a_hi = self.angles[idx - 1], self.angles[idx]
        wt = np.where(a_hi > a_lo,
                      (theta - a_lo) / np.maximum(a_hi - a_lo, 1e-12), 0.0)
        out = np.zeros((5,) + theta.shape)
        for i in range(len(theta.ravel())):
            flat = np.unravel_index(i, theta.shape)
            p_lo = np.array(self.params[float(a_lo[flat])])
            p_hi = np.array(self.params[float(a_hi[flat])])
            out[(slice(None),) + flat] = (1.0 - wt[flat]) * p_lo + wt[flat] * p_hi
        return out   # (5, ...) = A, alpha, beta, Birr, B0

    def critical_current(self, B_tesla, theta_deg, clip_B=True):
        B = np.atleast_1d(np.asarray(B_tesla, dtype=np.float64))
        theta = np.atleast_1d(np.asarray(theta_deg, dtype=np.float64))
        B, theta = np.broadcast_arrays(B, theta)
        if clip_B:
            B = np.clip(B, self.B_vals[0], self.B_vals[-1])
        A, alpha, beta, Birr, B0 = self._params_at(theta)
        Ic = _law_beta_reg(B, A, alpha, beta, Birr, B0)
        Ic = np.clip(Ic, 0.02 * self.Ic_max, self.Ic_max)
        return Ic, 0.0

    def dIc_dB(self, B_tesla, theta_deg, clip_B=True):
        """Analytic derivative, for validating UFL's automatic
        differentiation against a hand-derived closed form (NOT used by
        the solver itself -- ufl.derivative differentiates jc_ufl_expr
        directly and exactly)."""
        B = np.atleast_1d(np.asarray(B_tesla, dtype=np.float64))
        theta = np.atleast_1d(np.asarray(theta_deg, dtype=np.float64))
        B, theta = np.broadcast_arrays(B, theta)
        if clip_B:
            B = np.clip(B, self.B_vals[0], self.B_vals[-1])
        A, alpha, beta, Birr, B0 = self._params_at(theta)
        b = np.clip((B + B0) / Birr, 1e-12, 1.0 - 1e-12)
        df_db = (b ** (alpha - 2.0) * (1.0 - b) ** (beta - 2.0)
                *((alpha - 1.0) * (1.0 - b) - (beta - 1.0) * b))
        return A / Birr * df_db

    # -- UFL expression builder ----------------------------------------------

    @staticmethod
    def jc_ufl_expr(B_mag, A, alpha, beta, Birr, B0):
        """Build the UFL expression for Jc(|B|), given:
          B_mag : UFL scalar expression for |B| (e.g. sqrt(dot(B,B)+eps)) --
                  the live, differentiable quantity.
          A, alpha, beta, Birr, B0 : DG0 fem.Function coefficients, holding
                  the per-cell angle-interpolated parameter values, FROZEN
                  for the duration of one Newton solve (Picard-lagged
                  between outer iterations -- see update_entropy_params()).

        Pure UFL algebra (powers, a sum, a ratio) -- ufl.derivative(F, ...)
        differentiates this exactly w.r.t. whatever B_mag ultimately
        depends on (i.e. the unknown A in a monolithic solve).
        """
        import ufl
        b = (B_mag + B0) / Birr
        b = ufl.max_value(ufl.min_value(b, 1.0 - 1e-9), 1e-9)
        return A * b ** (alpha - 1.0) * (1.0 - b) ** (beta - 1.0)

    def update_dg0_params(self, A_fn, alpha_fn, beta_fn, Birr_fn, B0_fn,
                          theta_cell):
        """Write the current per-cell angle-interpolated parameters into
        the five DG0 coefficient Functions. Call once per OUTER iteration
        (theta changes as B's direction evolves); the B-magnitude
        dependence inside jc_ufl_expr stays live within the Newton solve
        that follows."""
        A, alpha, beta, Birr, B0 = self._params_at(theta_cell)
        for fn, arr in zip((A_fn, alpha_fn, beta_fn, Birr_fn, B0_fn),
                           (A, alpha, beta, Birr, B0)):
            fn.x.array[:len(arr)] = arr
            fn.x.scatter_forward()


# ── n(B, theta): empirical Hill-type saturating decay ───────────────────────

def _law_hill(B, n0, ninf, Bn, p):
    return ninf + (n0 - ninf) / (1.0 + np.clip(B / Bn, 0, None) ** p)


class HillNModel:
    """Per-angle (n0, ninf, Bn, p) empirical Hill/logistic fit to the full
    measured n(B,theta) grid. NOT derived from the entropy-maximization
    paper (see module docstring) -- chosen for smoothness and fit quality
    (MAPE 0.47%) so it, too, is exact UFL algebra."""

    def __init__(self, csv_path=None, base_n_model=None):
        base = base_n_model or NValueModel(csv_path or params.n_value_csv_filename)
        self.base = base
        self.B_vals = base.B_vals
        self.angles = base.angles
        self.n_min, self.n_max = base.n_min, base.n_max

        self.params = {}
        self.fit_rows = []
        for ang in self.angles:
            nvals, _ = base.n_value(
                self.B_vals, np.full_like(self.B_vals, ang), clip_B=False)
            popt, _ = curve_fit(
                _law_hill, self.B_vals, nvals,
                p0=[nvals[0], nvals[-1], 0.5, 1.5],
                bounds=([10, 5, 0.01, 0.3], [35, 30, 10.0, 6.0]),
                maxfev=200000)
            self.params[float(ang)] = tuple(popt)
            pred = _law_hill(self.B_vals, *popt)
            err = np.abs(pred - nvals) / np.maximum(nvals, 1e-9) * 100
            self.fit_rows.append(dict(angle_deg=float(ang),
                                      mape_pct=float(err.mean()),
                                      max_err_pct=float(err.max())))

    def _params_at(self, theta_deg):
        theta = np.clip(np.asarray(theta_deg, dtype=np.float64),
                        self.angles[0], self.angles[-1])
        idx = np.clip(np.searchsorted(self.angles, theta), 1,
                      len(self.angles) - 1)
        a_lo, a_hi = self.angles[idx - 1], self.angles[idx]
        wt = np.where(a_hi > a_lo,
                      (theta - a_lo) / np.maximum(a_hi - a_lo, 1e-12), 0.0)
        out = np.zeros((4,) + theta.shape)
        for i in range(len(theta.ravel())):
            flat = np.unravel_index(i, theta.shape)
            p_lo = np.array(self.params[float(a_lo[flat])])
            p_hi = np.array(self.params[float(a_hi[flat])])
            out[(slice(None),) + flat] = (1.0 - wt[flat]) * p_lo + wt[flat] * p_hi
        return out   # (4, ...) = n0, ninf, Bn, p

    def n_value(self, B_tesla, theta_deg, clip_B=True):
        B = np.atleast_1d(np.asarray(B_tesla, dtype=np.float64))
        theta = np.atleast_1d(np.asarray(theta_deg, dtype=np.float64))
        B, theta = np.broadcast_arrays(B, theta)
        if clip_B:
            B = np.clip(B, self.B_vals[0], self.B_vals[-1])
        n0, ninf, Bn, p = self._params_at(theta)
        n = _law_hill(B, n0, ninf, Bn, p)
        return np.clip(n, self.n_min, self.n_max), 0.0

    def dn_dB(self, B_tesla, theta_deg, clip_B=True):
        """Analytic derivative -- for validation only, see EntropyBetaIcModel.dIc_dB."""
        B = np.atleast_1d(np.asarray(B_tesla, dtype=np.float64))
        theta = np.atleast_1d(np.asarray(theta_deg, dtype=np.float64))
        B, theta = np.broadcast_arrays(B, theta)
        if clip_B:
            B = np.clip(B, self.B_vals[0], self.B_vals[-1])
        n0, ninf, Bn, p = self._params_at(theta)
        x = np.clip(B / Bn, 1e-300, None) ** p
        # d/dB [ninf + (n0-ninf)/(1+x)],  x = (B/Bn)^p
        dx_dB = p * x / np.maximum(B, 1e-12)
        return -(n0 - ninf) / (1.0 + x) ** 2 * dx_dB

    @staticmethod
    def n_ufl_expr(B_mag, n0, ninf, Bn, p):
        """UFL expression for n(|B|); see EntropyBetaIcModel.jc_ufl_expr
        for the calling convention (n0/ninf/Bn/p are frozen DG0
        coefficients, B_mag is the live UFL expression)."""
        import ufl
        x = (B_mag / Bn) ** p
        return ninf + (n0 - ninf) / (1.0 + x)

    def update_dg0_params(self, n0_fn, ninf_fn, Bn_fn, p_fn, theta_cell):
        n0, ninf, Bn, p = self._params_at(theta_cell)
        for fn, arr in zip((n0_fn, ninf_fn, Bn_fn, p_fn), (n0, ninf, Bn, p)):
            fn.x.array[:len(arr)] = arr
            fn.x.scatter_forward()


if __name__ == "__main__":
    ic = EntropyBetaIcModel()
    nm = HillNModel()

    def _report(name, model, rows):
        mapes = np.array([r["mape_pct"] for r in rows])
        maxes = np.array([r["max_err_pct"] for r in rows])
        print(f"{name}: overall MAPE={mapes.mean():.2f}%  "
              f"worst-angle MAPE={mapes.max():.2f}%  "
              f"worst single point={maxes.max():.2f}%")

    _report("EntropyBetaIcModel (Jc)", ic, ic.fit_rows)
    _report("HillNModel (n)", nm, nm.fit_rows)

    # Jc(0) sanity — must be finite and close to measured
    print("\nJc(B=0) spot-check:")
    for ang in [0.0, 45.0, 88.0, 135.0, 180.0]:
        pred, _ = ic.critical_current(np.array([0.0]), np.array([ang]))
        meas, _ = ic.base.critical_current(np.array([0.0]), np.array([ang]))
        print(f"  angle={ang:5.0f}: fit={pred[0]:8.1f} A  measured={meas[0]:8.1f} A")

    # Analytic-derivative sanity check: dIc_dB via central finite difference
    print("\ndIc/dB analytic vs. finite-difference (angle=45 deg):")
    for B0 in [0.01, 0.5, 2.0, 5.0]:
        h = 1e-4
        fd = (ic.critical_current(np.array([B0 + h]), np.array([45.0]))[0][0]
             - ic.critical_current(np.array([B0 - h]), np.array([45.0]))[0][0]) / (2 * h)
        an = ic.dIc_dB(np.array([B0]), np.array([45.0]))[0]
        print(f"  B={B0:.2f} T: analytic={an:10.2f}  finite-diff={fd:10.2f}  "
              f"rel_diff={abs(an - fd) / max(abs(fd), 1e-9) * 100:.3f}%")
