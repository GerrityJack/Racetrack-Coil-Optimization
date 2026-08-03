"""
ic_beta_shared_validation.py -- Long (2013) Beta model, PARAMETERS SHARED
=========================================================================
2026-08-03. `ic_extrapolation_validation.py` added Long's max-entropy Beta
form (Entropy 15(7) 2585, Eq. 2) fitted INDEPENDENTLY per angle, and it did
not beat the Kim model:

    split          kim MAPE   beta(free, per-angle) MAPE
    fit<=5T (1.6x)   4.14 %     5.46 %
    fit<=3T (2.7x)   6.98 %    22.72 %

with scipy warning that the covariance could not be estimated. That is an
under-determined 4-parameter fit on 4-5 points: B_irr drifts to large
values and the model collapses onto the power law (its 5.46 % is
indistinguishable from scaling100's 5.47 %).

**But that is not the paper's method.** Long's central empirical claim is
that Jc data at DIFFERENT FIELD ANGLES scales onto a COMMON curve -- a
single (alpha, beta) pair with only the irreversibility field B_irr varying
with angle (their Bi-2223 example: alpha=1.8, beta=9.6, B_irr from 1.73 T
at 0 deg to 11.1 T at 90 deg). Sharing (alpha, beta) across all 43 angles
turns 4 free parameters per angle into 2 global + 2 per angle, which is far
better conditioned and is the version worth testing.

This script hold-out validates that shared-parameter fit on exactly the
same splits, so the comparison against kim/power/scaling is apples to
apples: fit on low-field points only, score against measured high-field
points never seen.

Also reports the fitted B_irr distribution, because that is physically
checkable -- REBCO at 20 K should have B_irr well above 8 T, and an
implausible value (say < 10 T, or > 300 T) is evidence the shape is not
constrained by data in this range rather than evidence about the tape.

Run:
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \\
        optimize/studies/ic_beta_shared_validation.py
"""
import os, sys, csv, time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "optimize"), os.path.join(_ROOT, "physics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
from scipy.optimize import curve_fit, minimize

RUN_DIR = os.path.join(_ROOT, "optimize", "runs", "ic_beta_shared")
os.makedirs(RUN_DIR, exist_ok=True)
LOG_PATH = os.path.join(RUN_DIR, "log.txt")
CSV_PATH = os.path.join(RUN_DIR, "scores.csv")

SPLITS = [
    ("fit<=3T", [1.0, 1.5, 2.0, 3.0], [5.0, 7.0, 8.0]),
    ("fit<=5T", [1.0, 1.5, 2.0, 3.0, 5.0], [7.0, 8.0]),
]
BIRR_LO, BIRR_HI = 9.0, 400.0


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def beta_law(B, A, alpha, beta, Birr):
    b = np.clip(B / Birr, 1e-12, 1.0 - 1e-12)
    return A * b ** (alpha - 1.0) * (1.0 - b) ** (beta - 1.0)


def fit_shared(Bfit, curves):
    """One (alpha,beta) for all angles; (A, B_irr) per angle."""
    def per_angle(alpha, beta, Ic):
        def fn(B, A, Birr):
            return beta_law(B, A, alpha, beta, Birr)
        popt, _ = curve_fit(fn, Bfit, Ic, p0=[Ic[0], 60.0],
                            bounds=([1e-9, BIRR_LO], [1e10, BIRR_HI]),
                            maxfev=400000)
        pred = fn(Bfit, *popt)
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

    best = None
    for a0, b0 in ((0.7, 3.0), (1.5, 6.0), (1.8, 9.6), (0.4, 2.0)):
        r = minimize(cost, x0=[a0, b0], method="Nelder-Mead",
                     options=dict(xatol=1e-3, fatol=1e-10, maxiter=600))
        if best is None or r.fun < best.fun:
            best = r
    alpha, beta = float(best.x[0]), float(best.x[1])
    pars = []
    for Ic in curves:
        (A, Birr), _ = per_angle(alpha, beta, Ic)
        pars.append((A, alpha, beta, Birr))
    return alpha, beta, pars


def main():
    open(LOG_PATH, "w").close()
    from ic_model import IcModel
    m = IcModel()
    angles = np.asarray(m.angle_vals, dtype=np.float64)

    def Ic_at(B, ang):
        return float(m.critical_current(np.array([B]), np.array([ang]),
                                        clip_B=False)[0][0])

    _log("Long (2013) Beta model with SHARED (alpha, beta) across angles")
    _log("  -- the paper's angular-scaling claim, hold-out validated")
    _log(f"  {len(angles)} angles")

    rows = []
    for split_name, Bfit, Bhold in SPLITS:
        Bfit = np.array(Bfit)
        curves = [np.array([Ic_at(b, a) for b in Bfit]) for a in angles]
        alpha, beta, pars = fit_shared(Bfit, curves)

        errs, per_field = [], {b: [] for b in Bhold}
        for ang, par in zip(angles, pars):
            scale = Ic_at(Bfit[-1], ang) / beta_law(Bfit[-1], *par)
            for b in Bhold:
                pred = beta_law(b, *par) * scale
                meas = Ic_at(b, ang)
                e = (pred - meas) / meas * 100.0
                errs.append(e); per_field[b].append(e)
        errs = np.array(errs)
        birrs = np.array([p[3] for p in pars])
        mape = float(np.mean(np.abs(errs)))
        bias = float(np.mean(errs))
        worst = float(errs[np.argmax(np.abs(errs))])
        _log("")
        _log(f"=== {split_name}: fit {list(Bfit)} -> predict {Bhold} ===")
        _log(f"  shared alpha={alpha:.3f}  beta={beta:.3f}")
        _log(f"  B_irr fitted: min={birrs.min():.1f} T  median="
             f"{np.median(birrs):.1f} T  max={birrs.max():.1f} T")
        n_rail = int(np.sum((birrs <= BIRR_LO * 1.01)
                            | (birrs >= BIRR_HI * 0.99)))
        _log(f"  B_irr on a bound: {n_rail}/{len(birrs)} angles"
             + ("  <-- shape NOT constrained by the data" if n_rail else ""))
        _log(f"  MAPE={mape:.2f}%  bias={bias:+.2f}%  worst={worst:+.1f}%")
        _log("  per-hold-field mean signed error: "
             + "  ".join(f"{b:.0f}T:{np.mean(per_field[b]):+6.1f}"
                         for b in Bhold))
        rows.append(dict(split=split_name, form="beta_shared", alpha=alpha,
                         beta=beta, Birr_min=birrs.min(),
                         Birr_med=float(np.median(birrs)),
                         Birr_max=birrs.max(), n_on_bound=n_rail,
                         MAPE_pct=mape, bias_pct=bias, worst_pct=worst))

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    _log("")
    _log("Compare against the per-angle results in "
         "optimize/runs/ic_extrap_validation/scores.csv")
    _log("  (kim: MAPE 4.14% @1.6x, 6.98% @2.7x -- the incumbent to beat)")
    _log(f"scores -> {CSV_PATH}")


if __name__ == "__main__":
    main()
