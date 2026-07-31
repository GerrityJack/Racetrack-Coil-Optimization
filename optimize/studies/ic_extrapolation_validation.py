"""
ic_extrapolation_validation.py -- is the Jc(B) extrapolation trustworthy?
=========================================================================
2026-07-31. Before re-optimizing the whole design against the scaling-law
Ic model, check whether that model actually EXTRAPOLATES well -- otherwise
we would be redesigning against a phantom constraint.

**Method: hold-out validation on the measured data itself.** Fit each
candidate Jc(B) form on a LOW-field subset only, then score it against the
measured high-field points it never saw. This measures extrapolation skill
directly, in this tape's own regime, instead of arguing from theory.

Two splits, both extrapolating by a similar factor to what the real
application needs (8 T -> ~9-13 T peak conductor field, i.e. 1.1-1.6x):

    fit B in [1, 3] T  -> predict 5, 7, 8 T     (up to 2.7x)
    fit B in [1, 5] T  -> predict 7, 8 T        (up to 1.6x, closest
                                                 analogue to the real use)

**Candidate forms** (all fitted per angle, so anisotropy is preserved):

  flat        Jc(B) = Jc(B_fit_max)                 -- the current default
  power       Jc(B) = C * B^(-alpha)                -- pure power law
  kim         Jc(B) = Jc0 / (1 + B/B0)              -- Kim model
  scaling25   Jc(B) = C*B^(p-1)*(1-B/25)^q          -- pinning-force law
  scaling45   ... Bc2 = 45 T
  scaling100  ... Bc2 = 100 T

**Why this matters physically.** At 20 K, REBCO's irreversibility field is
far above our ~9-13 T peak field, so B/B_irr is small and the
(1 - B/Bc2)^q roll-off term should be nearly inactive -- the physics in
THIS regime is essentially a power law. If that is right, the scaling law
and the pure power law should score similarly, and a small fixed Bc2 (25 T)
should be mildly over-pessimistic. The hold-out test checks that claim
instead of assuming it.

Run:
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \
        optimize/studies/ic_extrapolation_validation.py

Outputs: optimize/runs/ic_extrap_validation/{log.txt, scores.csv} and
visualization/ic_extrapolation_validation.png
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
from scipy.optimize import curve_fit

RUN_DIR = os.path.join(_ROOT, "optimize", "runs", "ic_extrap_validation")
os.makedirs(RUN_DIR, exist_ok=True)
LOG_PATH = os.path.join(RUN_DIR, "log.txt")
SCORES_CSV = os.path.join(RUN_DIR, "scores.csv")
FIG_PATH = os.path.join(_ROOT, "visualization",
                        "ic_extrapolation_validation.png")

SPLITS = [
    ("fit<=3T", [1.0, 1.5, 2.0, 3.0], [5.0, 7.0, 8.0]),
    ("fit<=5T", [1.0, 1.5, 2.0, 3.0, 5.0], [7.0, 8.0]),
]


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ── candidate Jc(B) forms ───────────────────────────────────────────────────

def f_power(B, C, alpha):
    return C * B ** (-alpha)


def f_kim(B, Jc0, B0):
    return Jc0 / (1.0 + B / B0)


def _f_scaling(Bc2):
    def f(B, C, p, q):
        return C * B ** (p - 1.0) * np.clip(1.0 - B / Bc2, 1e-9, None) ** q
    return f


FORMS = [
    ("power", f_power, [1000.0, 0.5], ([1e-3, 0.0], [1e8, 3.0])),
    ("kim", f_kim, [2000.0, 2.0], ([1e-3, 1e-3], [1e8, 1e4])),
    ("scaling25", _f_scaling(25.0), [1000.0, 0.6, 2.0],
     ([1e-3, 0.01, 0.1], [1e8, 3.0, 12.0])),
    ("scaling45", _f_scaling(45.0), [1000.0, 0.6, 2.0],
     ([1e-3, 0.01, 0.1], [1e8, 3.0, 12.0])),
    ("scaling100", _f_scaling(100.0), [1000.0, 0.6, 2.0],
     ([1e-3, 0.01, 0.1], [1e8, 3.0, 12.0])),
]


def main():
    open(LOG_PATH, "w").close()
    from ic_model import IcModel
    m = IcModel()
    angles = np.asarray(m.angle_vals, dtype=np.float64)

    def Ic_at(B, ang):
        return float(m.critical_current(np.array([B]), np.array([ang]),
                                        clip_B=False)[0][0])

    _log("Hold-out validation of Ic(B) extrapolation forms")
    _log("  fit on low-field points only, score on unseen high-field points")
    _log(f"  {len(angles)} angles, signed error = (pred-meas)/meas "
         f"(positive = OPTIMISTIC)")

    rows = []
    for split_name, Bfit, Bhold in SPLITS:
        Bfit = np.array(Bfit)
        _log("")
        _log(f"=== {split_name}: fit {list(Bfit)} -> predict {Bhold} ===")
        _log(f"  {'form':<12}{'MAPE%':>8}{'bias%':>8}{'worst%':>9}"
             f"{'  per-hold-field mean signed error %'}")
        for form_name, fn, p0, bounds in [("flat", None, None, None)] + FORMS:
            errs, per_field = [], {b: [] for b in Bhold}
            for ang in angles:
                y = np.array([Ic_at(b, ang) for b in Bfit])
                if form_name == "flat":
                    pred = {b: y[-1] for b in Bhold}
                else:
                    try:
                        popt, _ = curve_fit(fn, Bfit, y, p0=p0, bounds=bounds,
                                            maxfev=400000)
                    except Exception:
                        continue
                    # continuity-rescale at the last fit point, exactly as the
                    # production extrapolator does
                    scale = y[-1] / fn(Bfit[-1], *popt)
                    pred = {b: fn(b, *popt) * scale for b in Bhold}
                for b in Bhold:
                    meas = Ic_at(b, ang)
                    e = (pred[b] - meas) / meas * 100.0
                    errs.append(e)
                    per_field[b].append(e)
            errs = np.array(errs)
            mape = float(np.mean(np.abs(errs)))
            bias = float(np.mean(errs))
            worst = float(errs[np.argmax(np.abs(errs))])
            pf = "  ".join(f"{b:.0f}T:{np.mean(per_field[b]):+6.1f}"
                           for b in Bhold)
            _log(f"  {form_name:<12}{mape:>8.2f}{bias:>+8.2f}{worst:>+9.1f}  {pf}")
            rows.append(dict(split=split_name, form=form_name, MAPE_pct=mape,
                             bias_pct=bias, worst_pct=worst))

    with open(SCORES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "form", "MAPE_pct",
                                          "bias_pct", "worst_pct"])
        w.writeheader(); w.writerows(rows)
    _log("")
    _log(f"scores -> {SCORES_CSV}")
    _log("Reading: bias > 0 means the form OVER-predicts Ic (optimistic, "
         "unsafe); bias < 0 means it under-predicts (conservative).")
    _make_fig(m, angles)


def _make_fig(m, angles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_BG, AX_BG, SPINE = "#111", "#0d0d1a", "#444"

    def Ic_at(B, ang):
        return float(m.critical_current(np.array([B]), np.array([ang]),
                                        clip_B=False)[0][0])

    Bfit = np.array([1.0, 1.5, 2.0, 3.0, 5.0])
    Ball = np.array([1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 8.0])
    Bsm = np.linspace(1.0, 8.0, 100)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(FIG_BG)
    for ax, ang in zip(axes, (0.0, 90.0)):
        ax.set_facecolor(AX_BG)
        y_fit = np.array([Ic_at(b, ang) for b in Bfit])
        y_all = np.array([Ic_at(b, ang) for b in Ball])
        ax.plot(Ball, y_all, "o", color="white", ms=7, label="measured")
        ax.plot(Bfit, y_fit, "o", color="#00e5ff", ms=7,
                label="used for fit (≤5 T)")
        ax.axvline(5.0, color="#888", ls=":", lw=1.2)
        for name, fn, p0, bounds, col in (
                ("flat", None, None, None, "#ff5555"),
                ("power", f_power, [1000.0, 0.5], ([1e-3, 0.0], [1e8, 3.0]),
                 "#7cff6e"),
                ("scaling45", _f_scaling(45.0), [1000.0, 0.6, 2.0],
                 ([1e-3, 0.01, 0.1], [1e8, 3.0, 12.0]), "#ffb000")):
            if name == "flat":
                ax.plot(Bsm[Bsm >= 5.0], np.full((Bsm >= 5.0).sum(), y_fit[-1]),
                        ls="--", color=col, lw=1.8, label="flat clamp")
                continue
            popt, _ = curve_fit(fn, Bfit, y_fit, p0=p0, bounds=bounds,
                                maxfev=400000)
            scale = y_fit[-1] / fn(Bfit[-1], *popt)
            ax.plot(Bsm, fn(Bsm, *popt) * scale, color=col, lw=2.0, label=name)
        ax.set_title(f"extrapolation test at {ang:.0f}°  "
                     f"(fit ≤5 T, predict 7–8 T)", color="white", fontsize=11)
        ax.set_xlabel("B [T]", color="white")
        ax.set_ylabel("Ic [A]", color="white")
        ax.tick_params(colors="white", labelsize=8)
        for s in ax.spines.values():
            s.set_color(SPINE)
        ax.grid(alpha=0.15, color="white", lw=0.5)
        ax.legend(facecolor=AX_BG, edgecolor=SPINE, labelcolor="white",
                  fontsize=8)
    fig.suptitle("Does the Ic(B) extrapolation actually work? "
                 "Hold-out validation on measured data",
                 color="white", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_PATH, dpi=160, facecolor=FIG_BG)
    _log(f"figure -> {FIG_PATH}")


if __name__ == "__main__":
    main()
