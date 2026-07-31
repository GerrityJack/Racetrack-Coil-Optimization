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


def make_ic_model(kind="kim", base=None):
    """Factory. kind: 'flat' (IcModel as-is, OPTIMISTIC -- not recommended),
    'kim' (default, best validated), or 'scaling[:Bc2]' e.g. 'scaling:25'."""
    if base is None:
        from ic_model import IcModel
        base = IcModel()
    k = (kind or "kim").strip().lower()
    if k in ("flat", "clip", "none", "default"):
        return base
    if k == "kim":
        return KimIcModel(base)
    if k.startswith("scaling"):
        bc2 = 45.0
        if ":" in k:
            bc2 = float(k.split(":", 1)[1])
        return ScalingLawIcModel(base, bc2=bc2)
    raise ValueError(f"unknown Ic extrapolation kind: {kind!r}")
