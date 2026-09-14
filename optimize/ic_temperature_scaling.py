"""
ic_temperature_scaling.py -- 4.2K-scaled Ic(B,theta) model, built by borrowing
a DIFFERENT tape's measured temperature-scaling RATIO
=============================================================================
2026-09-12. This project's own Ic(B,theta) dataset (physics/*.csv) is
measured at only 15K and 20K -- there is no measured Shanghai Superconductor
data at helium-range temperatures. This module does NOT invent one. Instead
it borrows the ratio Ic(4.2K)/Ic(20K) from a different, independently
published REBCO tape (Fujikura FESC-SCH04(40), digitized by hand from a
multi-temperature Ic(B,theta) characterization figure into
physics/digitized_IC_data/{5,10,15,20,24}T_{4p2K,20K}.csv, each file a plain
"theta_deg, Ic_A_per_4mm" point list) and applies that RATIO -- never the
Fujikura tape's own absolute Ic -- on top of this project's own measured 20K
Shanghai grid (itself already extended above 8T by ic_extrapolation.py's
validated KimIcModel).

Why borrowing the RATIO (not the absolute values) is defensible here: a
direct check at a field/angle both datasets cover (20K, 5T, theta=90) gives
Fujikura ~1827 A/4mm vs Shanghai's own MEASURED 1728 A/4mm -- 6% apart, i.e.
comparable-class modern 4mm 2G HTS tape. An earlier candidate reference
sample (SP06-M3-609-1-MS) failed this exact check (its absolute Ic was
LOWER than our own tape's at a warmer temperature -- a physical
impossibility if treated as our own tape's behavior) and was rejected for
this reason. Fujikura passes; SP06 did not.

**This is a bridging estimate, not a validated model.** There is no way to
hold-out validate it the way ic_extrapolation.py's B-extrapolation forms
were validated (that required measured Shanghai data ABOVE the fitted
range, which exists at 8T; there is no measured Shanghai data below 15K to
check this ratio against). It assumes Fujikura's temperature-scaling SHAPE
transfers to the Shanghai tape -- physically plausible (same REBCO pinning
physics class), not independently confirmed for this specific tape. Treat
any number from this model as "what a comparable tape's own cooling curve
implies," not ground truth -- exactly the caveat given verbally before this
file was written.

Ratio coverage and extrapolation policy (bounded both directions, same
"never let it run away past the data" discipline as ic_extrapolation.py):
  - B in [5, 24] T: bilinear interpolation over the 5 digitized field
    slices and a shared theta axis (built as the intersection of all 5
    slices' own digitized theta ranges, so no extrapolation is needed to
    BUILD the grid).
  - B < 5 T (the operationally relevant low-field regime for this
    project's actual T-A-safe operating currents) is held FLAT at the B=5T
    ratio curve. The measured ratio increases mildly with B (see
    `fit_rows`), so flat-holding the lowest available value likely
    UNDERSTATES, not overstates, the true low-field benefit -- the
    conservative direction.
  - B > 24 T is held FLAT at the B=24T ratio curve.
  - theta outside the shared digitized range is held flat at that range's
    nearest edge.
All of the above is implemented by clamping the QUERY point into the grid
before interpolating (never by extrapolating the interpolator itself), so
the flat-outside-data behavior is structural, not a fitted assumption.

Usage:
    from ic_model import IcModel
    from ic_extrapolation import KimIcModel
    from ic_temperature_scaling import Fujikura4p2KScaledIcModel

    base_20K = KimIcModel(IcModel())          # our own tape, Kim-extrapolated
    ic_4p2K = Fujikura4p2KScaledIcModel(base_20K)
    Ic_A, frac_outside_ratio_data = ic_4p2K.critical_current(B_tesla, theta_deg)

Drop-in for ta_safe_current.evaluate() / optimize_geometry.evaluate(), same
`critical_current(B, theta, clip_B=True) -> (Ic_A, frac)` interface as
IcModel/KimIcModel.
"""
import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIGITIZED_DIR = os.path.join(os.path.dirname(_HERE), "physics",
                               "digitized_IC_data")
_COMMON_FIELDS = [5, 10, 15, 20, 24]   # T -- present in both 4p2K and 20K
_N_THETA = 90                          # shared-axis resolution


def _load_csv(path):
    theta, ic = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            a, b = line.split(",")
            theta.append(float(a))
            ic.append(float(b))
    theta = np.asarray(theta, dtype=np.float64)
    ic = np.asarray(ic, dtype=np.float64)
    order = np.argsort(theta)
    return theta[order], ic[order]


# 2026-09-13: p-q generalized scaling-law fit, Ic(B) = Ic0(1+B/B0)^-p
# (1-B/Birr)^q, from Tsuchiya et al. 2026 (the same IEEE paper the 5-24T
# angular ratio grid above was digitized from), Table II, at theta=0
# (B-perpendicular-to-tape) -- fit to the paper's OWN measured field-
# dependence curve (Fig. 3) from near self-field through >24T, not just
# the 5 angular slices digitized for the ratio grid. Used ONLY to correct
# the SHAPE of the ratio below B_ratio_lo=5T (see
# Fujikura4p2KScaledIcModel.ratio() below) -- never as an absolute Ic
# source (see the module docstring on why only the RATIO is borrowed).
#
# Why this matters: the champion's own T-A-safe operating point sits at
# only ~1.1-1.2T peak coil field (verified 2026-09-13 directly, T-A-solve
# fixed-current sweep), entirely inside the region the flat-hold-below-5T
# policy used to cover with zero real data. Plugging these params in:
# ratio(0T)=1.694, ratio(1.1T)=1.71-1.72, ratio(5T)=1.756, ratio(24T)=2.056
# -- the ratio INCREASES with B, so the flat-hold-at-5T value was already
# a fair (very slightly generous, not conservative-against) approximation
# at the champion's actual operating field, not an underestimate. Kept as
# a genuine correction (not just a sanity check) since it removes the
# flat-hold assumption entirely and is more defensible than holding it.
_TABLE2_PARAMS_4P2K = dict(Ic0=3050.0, B0=1.6, Birr=117.0, p=0.6, q=2.0)
_TABLE2_PARAMS_20K = dict(Ic0=1800.0, B0=1.5, Birr=84.0, p=0.58, q=2.0)


def _table2_Ic(B, Ic0, B0, Birr, p, q):
    B = np.asarray(B, dtype=np.float64)
    return Ic0 * (1.0 + B / B0) ** (-p) * np.clip(1.0 - B / Birr, 0.0, None) ** q


def _table2_ratio(B):
    """Ic(4.2K)/Ic(20K) at theta=0, from Table II's own analytic fit --
    see the module-level comment above for provenance and why it's used
    only below B_ratio_lo."""
    return (_table2_Ic(np.clip(B, 0.0, None), **_TABLE2_PARAMS_4P2K)
            / _table2_Ic(np.clip(B, 0.0, None), **_TABLE2_PARAMS_20K))


class Fujikura4p2KScaledIcModel:
    kind = "fujikura_4p2K_scaled"

    def __init__(self, base_ic_model, digitized_dir=_DIGITIZED_DIR):
        self.base = base_ic_model
        self.tape_width = base_ic_model.tape_width
        self.B_min = base_ic_model.B_min
        # NOTE: deliberately NOT reusing base.Ic_min/Ic_max as an output
        # clamp -- those bounds were fit to the 20K measured grid's own
        # range and would silently discard the entire temperature benefit
        # this model exists to add.

        per_field = {}
        th_lo, th_hi = -1e9, 1e9
        for B in _COMMON_FIELDS:
            th4, ic4 = _load_csv(os.path.join(digitized_dir, f"{B}T_4p2K.csv"))
            th20, ic20 = _load_csv(os.path.join(digitized_dir, f"{B}T_20K.csv"))
            lo = max(th4.min(), th20.min())
            hi = min(th4.max(), th20.max())
            th_lo, th_hi = max(th_lo, lo), min(th_hi, hi)
            per_field[B] = (th4, ic4, th20, ic20)

        self.theta_axis = np.linspace(th_lo, th_hi, _N_THETA)
        ratio_grid = np.empty((len(_COMMON_FIELDS), _N_THETA))
        self.fit_rows = []
        for i, B in enumerate(_COMMON_FIELDS):
            th4, ic4, th20, ic20 = per_field[B]
            ic4_c = np.interp(self.theta_axis, th4, ic4)
            ic20_c = np.interp(self.theta_axis, th20, ic20)
            ratio = ic4_c / ic20_c
            ratio_grid[i] = ratio
            self.fit_rows.append(dict(
                B_T=B, ratio_min=float(ratio.min()),
                ratio_max=float(ratio.max()),
                ratio_median=float(np.median(ratio))))

        self._B_grid = np.array(_COMMON_FIELDS, dtype=np.float64)
        self.B_ratio_lo = float(self._B_grid.min())
        self.B_ratio_hi = float(self._B_grid.max())
        self._interp = RegularGridInterpolator(
            (self._B_grid, self.theta_axis), ratio_grid, method="linear")
        self._table2_ratio_5T = float(_table2_ratio(self.B_ratio_lo))

    def ratio(self, B_tesla, theta_deg):
        """Ic(4.2K)/Ic(20K) at (B,theta). B >= B_ratio_lo (5T) uses the
        digitized angular grid directly. B < B_ratio_lo is no longer a
        flat hold (2026-09-13): the angular value AT 5T is corrected by
        Table II's own analytic low-field ratio trend (`_table2_ratio`),
        continuous at B=5T (correction factor = 1 there) -- see the
        module-level comment above `_TABLE2_PARAMS_4P2K` for why this
        replaces the flat-hold rather than just checking it."""
        B_raw = np.atleast_1d(np.asarray(B_tesla, dtype=np.float64))
        theta = np.clip(np.atleast_1d(np.asarray(theta_deg, dtype=np.float64)),
                        self.theta_axis.min(), self.theta_axis.max())
        B_grid = np.clip(B_raw, self.B_ratio_lo, self.B_ratio_hi)
        pts = np.stack([B_grid, theta], axis=-1)
        base = self._interp(pts)
        low = B_raw < self.B_ratio_lo
        if np.any(low):
            base = base.copy()
            corr = _table2_ratio(B_raw[low]) / self._table2_ratio_5T
            base[low] = base[low] * corr
        return base

    def critical_current(self, B_tesla, theta_deg, clip_B=True):
        B = np.atleast_1d(np.asarray(B_tesla, dtype=np.float64))
        theta = np.atleast_1d(np.asarray(theta_deg, dtype=np.float64))
        Ic_20K, frac_base = self.base.critical_current(B, theta, clip_B=clip_B)
        Ic = Ic_20K * self.ratio(B, theta)
        Ic = np.clip(Ic, 0.0, None)
        frac_ratio_extrap = float(np.mean(
            (B < self.B_ratio_lo) | (B > self.B_ratio_hi)))
        # combined diagnostic: fraction of points relying on EITHER the
        # base model's own B>8T extrapolation OR this wrapper's B<5T/>24T
        # flat-hold -- not additive, just "how much of this evaluation
        # touched unmeasured territory of one kind or another"
        return Ic, max(frac_base, frac_ratio_extrap)


def _load_nvalue_csv(path):
    data = np.loadtxt(path, delimiter=",")
    B, n = data[:, 0], data[:, 1]
    order = np.argsort(B)
    return B[order], n[order]


class Fujikura4p2KScaledNValueModel:
    """
    n(B,theta) at 4.2K -- the missing half of the temperature-scaled model.

    2026-09-12: the first attempt at a 4.2K T-A check (Ic scaled, n left at
    its 20K measured value) gave a physically inconsistent, worse-than-20K
    local margin -- traced to combining 4.2K's higher Jc with 20K's lower
    (softer) n, a mismatch the coupled ρ=(E_c/Jc)(J/Jc)^(n-1) power law is
    sensitive to (see docs/HISTORY.md-equivalent discussion in this
    session). This class fixes that by scaling n the SAME way
    Fujikura4p2KScaledIcModel scales Ic: borrow the n(4.2K)/n(20K) RATIO
    from the Fujikura AP (FESC) datasheet's own page-14 n-value-B-T chart
    (digitized into physics/digitized_IC_data/nvalue_Bc_{4p2K,20K}.csv --
    the properly tape-matched AP/FESC chart, not the Non-AP/FYSC one first
    shown), and apply it to this project's own base (20K) NValueModel.

    Orientation convention matches NValueModel exactly: theta=0 -> B ⊥
    tape face -- i.e. B//c, exactly what nvalue_Bc_*.csv measures. No
    conversion needed for that axis.

    **Applied ISOTROPICALLY (same ratio at every theta).** The Fujikura
    datasheet has no AP-type B//ab n-value chart -- only the B//c one used
    here -- so this is a real, disclosed simplification, not a validated
    angular n(T) model. Cells at large theta (near the tape's ab-plane)
    may be more or less accurate than this ratio implies; there is
    currently no data to do better.

    Bounded flat extrapolation outside the digitized B range (~0.75-17.5T,
    itself already trimmed to exclude a label-text digitization artifact
    at the chart's edges -- see the session that produced these CSVs),
    same discipline as Fujikura4p2KScaledIcModel: never extrapolate the
    ratio itself, only clamp the query point into range before looking it
    up.
    """
    def __init__(self, base_n_model, digitized_dir=_DIGITIZED_DIR):
        self.base = base_n_model
        B4, n4 = _load_nvalue_csv(
            os.path.join(digitized_dir, "nvalue_Bc_4p2K.csv"))
        B20, n20 = _load_nvalue_csv(
            os.path.join(digitized_dir, "nvalue_Bc_20K.csv"))
        lo = max(float(B4.min()), float(B20.min()))
        hi = min(float(B4.max()), float(B20.max()))
        self.B_ratio_lo, self.B_ratio_hi = lo, hi
        self._B_grid = np.linspace(lo, hi, 200)
        n4_c = np.interp(self._B_grid, B4, n4)
        n20_c = np.interp(self._B_grid, B20, n20)
        self._ratio_grid = n4_c / n20_c
        self.ratio_min = float(self._ratio_grid.min())
        self.ratio_max = float(self._ratio_grid.max())

    def ratio(self, B_tesla):
        """n(4.2K)/n(20K) at B, held flat outside the digitized range."""
        B = np.clip(np.atleast_1d(np.asarray(B_tesla, dtype=np.float64)),
                    self.B_ratio_lo, self.B_ratio_hi)
        return np.interp(B, self._B_grid, self._ratio_grid)

    def n_value(self, B_tesla, theta_deg, clip_B=True):
        B = np.atleast_1d(np.asarray(B_tesla, dtype=np.float64))
        theta = np.atleast_1d(np.asarray(theta_deg, dtype=np.float64))
        n20, frac = self.base.n_value(B, theta, clip_B=clip_B)
        n = n20 * self.ratio(B)
        return n, frac
