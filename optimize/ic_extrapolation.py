"""
ic_extrapolation.py -- physically-validated Ic(B) extrapolation above 8 T
=========================================================================
2026-07-31. The measured Shanghai tape data stops at 8 T, but the coil's
peak conductor field is ~9-13 T, so a meaningful fraction of every quench
evaluation depends on what happens ABOVE the data. The project default
(`clip_B=True`) flat-clamps Ic at its 8 T value -- and because Ic DECREASES
with B, that is optimistic, not conservative.

**These wrappers are validated, not assumed.**
`optimize/studies/ic_extrapolation_validation.py` performs hold-out tests
on the measured data itself: fit each form on a low-field subset, then
score it against measured high-field points it never saw. At the split
closest to the real use (fit <=5 T, predict 7-8 T, a 1.6x extrapolation --
the same factor as 8 T -> 13 T), over all 43 angles:

    form         MAPE     bias      (bias > 0 = over-predicts Ic = unsafe)
    flat clamp   26.7 %   +26.7 %   badly optimistic
    power law     6.9 %    +6.8 %   slightly optimistic
    kim           4.1 %    -3.3 %   BEST, mildly conservative
    scaling(45T)  6.1 %    -5.8 %   good, mildly conservative

At a harsher 2.7x split the flat clamp over-predicts by +54 % (up to +88 %
on individual angles), while kim/scaling stay within ~5-15 %.

So: `kim` is the best-supported model for design work, and `scaling*` is a
slightly conservative alternative useful as a lower bound. Both are far
better than the flat clamp, which should not be used for any load-bearing
number.

Both wrappers:
  - use the MEASURED interpolation unchanged below B_max (8 T); the fitted
    form is an EXTRAPOLATOR only,
  - fit PER ANGLE, preserving the tape's Ic(B,theta) anisotropy,
  - are continuity-rescaled at B_max so there is no step at the handover,
  - expose IcModel's `critical_current(B, theta, clip_B=...)` signature, so
    they drop straight into `optimize_geometry.evaluate(cand, ic_model, comm)`.

**Bc2 is FIXED, never free-fit** (scaling law only). Over a 1-8 T window
with Bc2 >> 8 T the (1 - B/Bc2)^q factor barely varies, so q and Bc2 are
degenerate; an unconstrained fit exploits this and can pick physically
meaningless parameters (it once chose Bc2 = 10.23 T at angle 88 deg, which
sent Ic to zero just above the data and corrupted the whole quench
calculation). See ic_scaling_law_test.py's docstring for that incident.
"""
import numpy as np
from scipy.optimize import curve_fit

B_FIT = np.array([1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 8.0])


def _law_scaling(B, C, p, q, Bc2):
    return C * B ** (p - 1.0) * np.clip(1.0 - B / Bc2, 1e-9, None) ** q


def _law_kim(B, Jc0, B0):
    return Jc0 / (1.0 + B / B0)


class _ExtrapolatedIcModel:
    """Shared machinery: measured below B_max, per-angle fitted form above."""

    kind = "base"

    def __init__(self, base_ic_model):
        self.base = base_ic_model
        self.B_max = base_ic_model.B_max
        self.B_min = base_ic_model.B_min
        self.Ic_min = base_ic_model.Ic_min
        self.Ic_max = base_ic_model.Ic_max
        self.angles = np.asarray(base_ic_model.angle_vals, dtype=np.float64)
        self.params = {}
        self.fit_rows = []
        self._fit_all()

    # -- subclasses provide these -------------------------------------------
    def _fit_one(self, Bfit, Ic):
        raise NotImplementedError

    def _eval_one(self, B, par):
        raise NotImplementedError

    # -----------------------------------------------------------------------
    def _fit_all(self):
        for ang in self.angles:
            Ic = np.array([self.base.critical_current(
                np.array([b]), np.array([ang]), clip_B=False)[0][0]
                for b in B_FIT])
            par = self._fit_one(B_FIT, Ic)
            # continuity: reproduce the measured Ic exactly at B_max
            par = self._rescale(par, Ic[-1] / self._eval_one(self.B_max, par))
            pred = np.array([self._eval_one(b, par) for b in B_FIT])
            self.params[float(ang)] = par
            self.fit_rows.append(dict(
                angle_deg=float(ang), Ic_8T=float(Ic[-1]),
                rel_rms_pct=float(np.sqrt(np.mean(((pred - Ic) / Ic) ** 2))
                                  * 100)))

    @staticmethod
    def _rescale(par, factor):
        return (par[0] * factor,) + tuple(par[1:])

    def _extrap(self, B, theta):
        """Per-point evaluation, linearly interpolating the fitted parameters
        between the two nearest measured angles."""
        out = np.empty_like(B)
        idx = np.clip(np.searchsorted(self.angles, theta), 1,
                      len(self.angles) - 1)
        a_lo, a_hi = self.angles[idx - 1], self.angles[idx]
        wt = np.where(a_hi > a_lo,
                      (theta - a_lo) / np.maximum(a_hi - a_lo, 1e-12), 0.0)
        for i in range(len(B)):
            v_lo = self._eval_one(B[i], self.params[float(a_lo[i])])
            v_hi = self._eval_one(B[i], self.params[float(a_hi[i])])
            out[i] = (1.0 - wt[i]) * v_lo + wt[i] * v_hi
        return out

    def critical_current(self, B_tesla, theta_deg, clip_B=True):
        B = np.atleast_1d(np.asarray(B_tesla, dtype=np.float64))
        theta = np.atleast_1d(np.asarray(theta_deg, dtype=np.float64))
        theta = np.clip(theta, self.angles[0], self.angles[-1])
        over = B > self.B_max
        Ic, _ = self.base.critical_current(
            np.clip(B, self.B_min, self.B_max), theta, clip_B=False)
        Ic = Ic.copy()
        if np.any(over):
            Ic[over] = self._extrap(B[over], theta[over])
        Ic = np.clip(Ic, 0.02 * self.Ic_max, self.Ic_max)
        # fraction that relied on EXTRAPOLATION (not comparable to IcModel's
        # own frac, which measures something else -- compare within a model)
        return Ic, float(np.mean(over))


class KimIcModel(_ExtrapolatedIcModel):
    """Jc(B) = Jc0 / (1 + B/B0). Best hold-out extrapolation skill of the
    forms tested (MAPE 4.1 %, bias -3.3 % at a 1.6x extrapolation)."""

    kind = "kim"

    def _fit_one(self, Bfit, Ic):
        popt, _ = curve_fit(_law_kim, Bfit, Ic, p0=[Ic[0] * 2.0, 2.0],
                            bounds=([1e-3, 1e-3], [1e8, 1e4]), maxfev=400000)
        return tuple(popt)

    def _eval_one(self, B, par):
        return _law_kim(B, *par)


class ScalingLawIcModel(_ExtrapolatedIcModel):
    """Jc(B) = C*B^(p-1)*(1-B/Bc2)^q, Bc2 FIXED (see module docstring).
    Mildly conservative (bias -5.8 % at Bc2=45 T); useful as a lower bound."""

    kind = "scaling"

    def __init__(self, base_ic_model, bc2=45.0):
        if bc2 is None:
            raise ValueError("Bc2 must be fixed -- a free fit is degenerate "
                             "over the 1-8T window and can select physically "
                             "meaningless parameters.")
        self.bc2 = float(bc2)
        self.kind = f"scaling{self.bc2:.0f}T"
        super().__init__(base_ic_model)

    def _fit_one(self, Bfit, Ic):
        def fn(B, C, p, q):
            return _law_scaling(B, C, p, q, self.bc2)
        popt, _ = curve_fit(fn, Bfit, Ic, p0=[Ic[0], 0.6, 2.0],
                            bounds=([1e-3, 0.01, 0.1], [1e8, 3.0, 12.0]),
                            maxfev=400000)
        return tuple(popt)

    def _eval_one(self, B, par):
        return _law_scaling(B, *par, self.bc2)




# ── Long (2013) maximum-entropy Beta model ──────────────────────────────────
# N. J. Long, "Maximum Entropy Distributions Describing Critical Currents in
# Superconductors", Entropy 2013, 15(7), 2585-2605, doi:10.3390/e15072585.
#
# The paper derives, from maximum-entropy inference with logarithmic
# constraints <ln b> and <ln(1-b)>, that the field dependence of Jc is a
# BETA distribution (their Equation 2):
#
#     Jc(b)  ~  b^(alpha-1) * (1 - b)^(beta-1),      b = B / B_irr
#
# with the max-entropy constraints  <ln b>     = psi(alpha) - psi(alpha+beta)
#                                   <ln(1-b)>  = psi(beta)  - psi(alpha+beta)
# (psi = digamma). Multiplying by B recovers the long-established pinning
# force form F_p = Jc*B ~ b^m (1-b)^n, which the paper notes has described
# both LTS and HTS over 40 years.
#
# Algebraically this is the same family as ScalingLawIcModel, but with two
# improvements that matter for EXTRAPOLATION:
#
#  1. The normalizing field is the IRREVERSIBILITY field B_irr -- the field
#     at which Jc actually reaches zero -- not Bc2. B_irr is far lower than
#     Bc2, so the (1-b) roll-off is a real, determinable feature rather than
#     the near-1 nuisance factor that made q and Bc2 degenerate in
#     ScalingLawIcModel (where Bc2 had to be FIXED by hand).
#  2. The paper shows Jc data at DIFFERENT FIELD ANGLES collapses onto a
#     COMMON curve -- one (alpha, beta) pair with an angle-dependent B_irr
#     (their Bi-2223 example: alpha=1.8, beta=9.6, B_irr from 1.73 T at 0 deg
#     to 11.1 T at 90 deg). BetaSharedIcModel exploits this: it fits ONE
#     (alpha, beta) globally and only (A, B_irr) per angle, which is far
#     better conditioned than 4 free parameters per angle.
#
# HONEST LIMITATION for this tape: the pinning force F_p = Ic*B is still
# RISING at 8 T at every angle, so its peak (at b = alpha/(alpha+beta-1))
# lies above the measured data and B_irr is inferred from curvature rather
# than pinned by a visible maximum. Whether that inference extrapolates
# better than the Kim model is decided empirically by
# optimize/studies/ic_extrapolation_validation.py, not by assumption.

# 2026-08-03: the lower bound MUST sit well above the 8 T data ceiling.
# With it at 9 T, the per-angle free fit chose B_irr = 10.23 T at angle
# 88 deg -- collapsing Ic from 894 A at 10 T to 39.6 A at 12 T, at the angle
# where Ic is HIGHEST -- which dominated the quench bisection and produced a
# spurious B_target of 5.59 T for the champion. This is the SAME angle and
# the SAME 10.23 T value that the free-Bc2 scaling-law fit picked earlier
# (see ic_scaling_law_test.py); angle 88 deg in this dataset reliably
# induces it. REBCO at 20 K has an irreversibility field of roughly
# 30-45 T, so 20 T is a safe physical floor that still leaves the fit free.
BIRR_BOUNDS = (20.0, 400.0)    # T; must sit well above the 8 T data ceiling


def _law_beta(B, A, alpha, beta, Birr):
    b = np.clip(B / Birr, 1e-12, 1.0 - 1e-12)
    return A * b ** (alpha - 1.0) * (1.0 - b) ** (beta - 1.0)


class BetaIcModel(_ExtrapolatedIcModel):
    """Long (2013) Eq. 2, fitted independently per angle (A, alpha, beta,
    B_irr). Four parameters on seven points -- see BetaSharedIcModel for the
    better-conditioned variant the paper's angular scaling motivates."""

    kind = "beta"

    def _fit_one(self, Bfit, Ic):
        popt, _ = curve_fit(
            _law_beta, Bfit, Ic, p0=[Ic[0], 0.7, 3.0, 60.0],
            bounds=([1e-6, 0.05, 1.0, BIRR_BOUNDS[0]],
                    [1e9, 4.0, 40.0, BIRR_BOUNDS[1]]), maxfev=800000)
        return tuple(popt)

    def _eval_one(self, B, par):
        return _law_beta(B, *par)


class BetaSharedIcModel(_ExtrapolatedIcModel):
    """Long (2013) Eq. 2 with (alpha, beta) SHARED across all angles and only
    (A, B_irr) fitted per angle -- the paper's observation that angular data
    scales to a common curve. Outer Nelder-Mead over (alpha, beta); inner
    per-angle 2-parameter fit."""

    kind = "beta_shared"

    def __init__(self, base_ic_model):
        self._shared = None
        super().__init__(base_ic_model)

    def _fit_all(self):
        from scipy.optimize import minimize
        angles = self.angles
        curves = []
        for ang in angles:
            Ic = np.array([self.base.critical_current(
                np.array([b]), np.array([ang]), clip_B=False)[0][0]
                for b in B_FIT])
            curves.append(Ic)

        def per_angle(alpha, beta, Ic):
            def fn(B, A, Birr):
                return _law_beta(B, A, alpha, beta, Birr)
            popt, _ = curve_fit(fn, B_FIT, Ic, p0=[Ic[0], 60.0],
                                bounds=([1e-6, BIRR_BOUNDS[0]],
                                        [1e9, BIRR_BOUNDS[1]]), maxfev=400000)
            pred = fn(B_FIT, *popt)
            return popt, float(np.mean(((pred - Ic) / Ic) ** 2))

        def cost(ab):
            alpha, beta = ab
            if not (0.05 < alpha < 4.0 and 1.0 < beta < 40.0):
                return 1e6
            tot = 0.0
            for Ic in curves:
                try:
                    _, mse = per_angle(alpha, beta, Ic)
                except Exception:
                    return 1e6
                tot += mse
            return tot

        best = minimize(cost, x0=[0.7, 3.0], method="Nelder-Mead",
                        options=dict(xatol=1e-3, fatol=1e-9, maxiter=400))
        alpha, beta = float(best.x[0]), float(best.x[1])
        self._shared = (alpha, beta)
        for ang, Ic in zip(angles, curves):
            (A, Birr), _ = per_angle(alpha, beta, Ic)
            par = (A, alpha, beta, Birr)
            par = self._rescale(par, Ic[-1] / self._eval_one(self.B_max, par))
            pred = np.array([self._eval_one(b, par) for b in B_FIT])
            self.params[float(ang)] = par
            self.fit_rows.append(dict(
                angle_deg=float(ang), Ic_8T=float(Ic[-1]),
                rel_rms_pct=float(np.sqrt(np.mean(((pred - Ic) / Ic) ** 2)) * 100)))

    def _fit_one(self, Bfit, Ic):
        raise NotImplementedError

    def _eval_one(self, B, par):
        return _law_beta(B, *par)


def make_ic_model(kind="kim", base=None):
    """Factory. kind: 'flat' (IcModel as-is, OPTIMISTIC -- not recommended),
    'kim', 'beta' / 'beta_shared' (Long 2013 max-entropy Beta), or
    'scaling[:Bc2]' e.g. 'scaling:25'."""
    if base is None:
        from ic_model import IcModel
        base = IcModel()
    k = (kind or "kim").strip().lower()
    if k in ("flat", "clip", "none", "default"):
        return base
    if k == "kim":
        return KimIcModel(base)
    if k in ("beta", "long", "maxent"):
        return BetaIcModel(base)
    if k in ("beta_shared", "long_shared"):
        return BetaSharedIcModel(base)
    if k.startswith("scaling"):
        bc2 = 45.0
        if ":" in k:
            bc2 = float(k.split(":", 1)[1])
        return ScalingLawIcModel(base, bc2=bc2)
    raise ValueError(f"unknown Ic extrapolation kind: {kind!r}")
