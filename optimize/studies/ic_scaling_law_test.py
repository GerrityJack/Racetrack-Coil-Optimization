"""
ic_scaling_law_test.py -- physical Jc(B) extrapolation above the 8 T data
=========================================================================
2026-07-31, small standalone test.

**The problem it addresses.** Every Ic lookup in this project defaults to
`clip_B=True`, which flat-clamps Ic to its measured B = 8 T value for any
cell above that field. Since Ic DECREASES with B in the measured range,
that clamp is OPTIMISTIC, not conservative -- and ~12% of the champion's
quench-point Ic evaluations sit above 8 T. The only alternative tried so
far (`ConservativeIcModel`, a linear continuation of the 8 T slope) is
deliberately pessimistic and drops B_target from 10.0 T to 6.51 T. Neither
is a physical model; the truth is in between.

**This test.** Fit the standard pinning-force scaling law

    Jc(B) = C * B^(p-1) * (1 - B/Bc2)^q          [F_p = Jc*B ~ b^p (1-b)^q]

to the measured data and use it to extrapolate above 8 T, then re-evaluate
the champion at operating points of 55 / 60 / 65 % of local Ic.

**Fitting choices, and why:**
- Fit **per angle**, so the tape's Ic(B,theta) anisotropy is preserved
  rather than collapsed to an isotropic curve.
- Fit over **B in [1, 8] T only.** The law diverges as B -> 0 for p < 1
  (measured Ic(0T) is finite, 1979 A), so it is neither fit nor used in
  the low-field regime -- below 8 T the measured interpolation is used
  unchanged. The law is an EXTRAPOLATOR, not a replacement model.
- **Continuity-rescaled at 8 T**: C is adjusted so the law reproduces the
  measured Ic exactly at B = 8 T, so there is no step discontinuity at the
  handover.

**The honest caveat, and why Bc2 is a fixed sensitivity parameter.** Over
the 1-8 T fit window with Bc2 >> 8 T, the factor (1 - B/Bc2)^q stays
between ~0.93 and ~1.0 -- it barely varies, so **q and Bc2 are strongly
degenerate and the data cannot constrain them.** Fit quality is excellent
regardless of Bc2 (sub-1 % RMS at every value tried), which is a statement
about the B^(p-1) power law, not evidence for any particular Bc2. Bc2 is
therefore FIXED at three physically motivated values spanning REBCO's
plausible 20 K behaviour, and the spread between them IS the honest
uncertainty band:

  - `bc2_25T`  : near the low end of the irreversibility field at 20 K
                 (pessimistic -- fastest roll-off above 8 T)
  - `bc2_45T`  : mid-range
  - `bc2_100T` : weak roll-off (optimistic end)

Quote the band, not any single number. Closing this gap needs measured Ic
data above 8 T -- no amount of fitting can substitute.

**2026-07-31 -- a FREE Bc2 fit was tried first and is unusable.** Left
unconstrained, the fit at angle 88 deg landed on Bc2 = 10.23 T with
q = 0.18: an excellent fit INSIDE the 1-8 T window (0.51 % RMS) that is
physical nonsense as an extrapolator, because above 10.23 T the
(1 - B/Bc2) factor goes negative, clips, and sends Ic to the floor
(~40 A). Angle 88 deg is near the ab-plane peak where Ic is HIGHEST, so
that spurious collapse dominated the quench bisection and produced the
absurd result that the OPTIMISTIC variant gave a LOWER operating current
(119.6 A) than the pessimistic one (167.4 A). The degeneracy is not a
cosmetic caveat -- an unconstrained fit will happily pick a meaningless
(Bc2, q) pair. Bc2 is therefore FIXED at physically motivated values and
only C, p, q are fitted.

Run (after any mesh study finishes -- it is CPU-light but the FEM
evaluations are not free):
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \
        optimize/studies/ic_scaling_law_test.py

Outputs: optimize/runs/ic_scaling_law/{log.txt, fit_params.csv,
results.csv} and visualization/ic_scaling_law.png
"""
import os, sys, csv, time, json

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

RUN_DIR = os.path.join(_ROOT, "optimize", "runs", "ic_scaling_law")
os.makedirs(RUN_DIR, exist_ok=True)
LOG_PATH = os.path.join(RUN_DIR, "log.txt")
FIT_CSV = os.path.join(RUN_DIR, "fit_params.csv")
RES_CSV = os.path.join(RUN_DIR, "results.csv")
FIG_PATH = os.path.join(_ROOT, "visualization", "ic_scaling_law.png")

B_FIT = np.array([1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 8.0])
FRACTIONS = [0.55, 0.60, 0.65]          # operating point as fraction of Ic
# Bc2 is FIXED (never free-fit -- see the docstring). Values bracket REBCO's
# plausible 20 K behaviour: 25 T is near the low end of the irreversibility
# field (pessimistic, fast roll-off); 100 T is a weak-roll-off optimistic end.
BC2_VALUES = [25.0, 45.0, 100.0]

CHAMPION = dict(a=0.022227029065529628, b=0.02726822715975084,
                coil_half_gap=0.013500289306395013,
                n_turns=[295, 295, 369, 369, 2, 2])


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def _law(B, C, p, q, Bc2):
    return C * B ** (p - 1.0) * np.clip(1.0 - B / Bc2, 1e-9, None) ** q


# ── the model wrapper ───────────────────────────────────────────────────────

class ScalingLawIcModel:
    """Measured interpolation below B_max; pinning-force scaling law above,
    fitted per angle and continuity-matched at B_max. Mirrors IcModel's
    critical_current() signature so optimize_geometry.evaluate() can take it
    as a drop-in (same approach as day_search.py's ConservativeIcModel)."""

    def __init__(self, base_ic_model, bc2_fixed=None, verbose=False):
        self.base = base_ic_model
        self.B_max = base_ic_model.B_max
        self.B_min = base_ic_model.B_min
        self.Ic_min = base_ic_model.Ic_min
        self.Ic_max = base_ic_model.Ic_max
        self.bc2_fixed = bc2_fixed
        self.angles = np.asarray(base_ic_model.angle_vals, dtype=np.float64)
        self.params = {}          # angle -> (C, p, q, Bc2)
        self._fit(verbose)

    def _fit(self, verbose):
        rows = []
        if self.bc2_fixed is None:
            raise ValueError(
                "Bc2 must be fixed. A free fit is degenerate over the 1-8T "
                "window and can select a physically meaningless (Bc2, q) -- "
                "see the module docstring's 2026-07-31 note.")
        for ang in self.angles:
            Ic = np.array([self.base.critical_current(
                np.array([b]), np.array([ang]), clip_B=False)[0][0]
                for b in B_FIT])
            bc2 = float(self.bc2_fixed)
            p0 = [Ic[0], 0.6, 2.0]
            lo = [1e-3, 0.01, 0.1]
            hi = [1e7, 3.0, 12.0]

            def fn(B, C, p, q, _b=bc2):
                return _law(B, C, p, q, _b)

            popt, _ = curve_fit(fn, B_FIT, Ic, p0=p0, bounds=(lo, hi),
                                maxfev=400000)
            C, p, q = popt
            Bc2 = bc2
            # continuity: make the law reproduce the measured Ic at B_max
            pred_max = _law(self.B_max, C, p, q, Bc2)
            C *= Ic[-1] / pred_max
            pred = _law(B_FIT, C, p, q, Bc2)
            rel_rms = float(np.sqrt(np.mean(((pred - Ic) / Ic) ** 2)) * 100)
            self.params[float(ang)] = (C, p, q, Bc2)
            rows.append(dict(angle_deg=float(ang), C=C, p=p, q=q, Bc2=Bc2,
                             Ic_8T=float(Ic[-1]), rel_rms_pct=rel_rms))
        self.fit_rows = rows
        if verbose:
            _log(f"  fitted {len(rows)} angles, "
                 f"rel_rms {min(r['rel_rms_pct'] for r in rows):.2f}"
                 f"-{max(r['rel_rms_pct'] for r in rows):.2f}%")

    def _law_at(self, B, theta):
        """Per-point law evaluation, linearly interpolating the fitted
        parameters between the two nearest measured angles."""
        out = np.empty_like(B)
        idx = np.clip(np.searchsorted(self.angles, theta), 1,
                      len(self.angles) - 1)
        a_lo = self.angles[idx - 1]
        a_hi = self.angles[idx]
        wt = np.where(a_hi > a_lo, (theta - a_lo) / np.maximum(a_hi - a_lo, 1e-12), 0.0)
        for i in range(len(B)):
            plo = self.params[float(a_lo[i])]
            phi = self.params[float(a_hi[i])]
            v_lo = _law(B[i], *plo)
            v_hi = _law(B[i], *phi)
            out[i] = (1.0 - wt[i]) * v_lo + wt[i] * v_hi
        return out

    def critical_current(self, B_tesla, theta_deg, clip_B=True):
        B = np.atleast_1d(np.asarray(B_tesla, dtype=np.float64))
        theta = np.atleast_1d(np.asarray(theta_deg, dtype=np.float64))
        theta = np.clip(theta, self.angles[0], self.angles[-1])
        over = B > self.B_max
        Ic_normal, _ = self.base.critical_current(
            np.clip(B, self.B_min, self.B_max), theta, clip_B=False)
        Ic = Ic_normal.copy()
        if np.any(over):
            Ic[over] = self._law_at(B[over], theta[over])
        Ic = np.clip(Ic, 0.02 * self.Ic_max, self.Ic_max)
        # report the fraction that RELIED ON EXTRAPOLATION (above the measured
        # ceiling) -- that is the quantity of interest here. Note IcModel's own
        # frac is a different measure, so these columns are not comparable
        # across models; compare within a model only.
        frac_extrapolated = float(np.mean(over))
        return Ic, frac_extrapolated


# ── evaluation ──────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    open(LOG_PATH, "w").close()
    from mpi4py import MPI
    import optimize_geometry as og
    import opt_config as cfg
    from ic_model import IcModel
    comm = MPI.COMM_WORLD

    _log("Jc(B) = C*B^(p-1)*(1-B/Bc2)^q extrapolation test on the champion")
    _log(f"  design n_turns={CHAMPION['n_turns']} "
         f"a={CHAMPION['a']*1e3:.3f}mm b={CHAMPION['b']*1e3:.3f}mm")

    base_ic = IcModel()
    variants = []
    for bc2 in BC2_VALUES:
        _log(f"fitting scaling law, Bc2 fixed at {bc2:.0f} T ...")
        variants.append((f"bc2_{bc2:.0f}T",
                         ScalingLawIcModel(base_ic, bc2_fixed=bc2,
                                           verbose=True)))

    with open(FIT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "angle_deg", "C", "p", "q",
                                          "Bc2", "Ic_8T", "rel_rms_pct"])
        w.writeheader()
        for name, mdl in variants:
            for r in mdl.fit_rows:
                w.writerow(dict(variant=name, **r))
    _log(f"fit parameters -> {FIT_CSV}")

    # what the extrapolation actually predicts, at the worst angle (0 deg)
    _log("")
    _log("Ic at 0 deg (worst angle), A:")
    _log("  " + f"{'B[T]':>6}{'flat clamp':>12}"
         + "".join(f"{n:>11}" for n, _ in variants))
    for B in (8.0, 10.0, 12.0, 14.0, 16.0, 20.0):
        flat = base_ic.critical_current(np.array([B]), np.array([0.0]),
                                        clip_B=True)[0][0]
        vals = [m.critical_current(np.array([B]), np.array([0.0]))[0][0]
                for _, m in variants]
        _log(f"  {B:>6.1f}{flat:>12.1f}"
             + "".join(f"{v:>11.1f}" for v in vals))

    # ── re-evaluate the champion under each model / operating fraction ──
    _log("")
    _log("re-evaluating the champion (one FEM solve per row) ...")
    orig_sf = cfg.SAFETY_FACTOR
    rows = []
    models = [("flat_clamp (current default)", base_ic)] + \
             [(f"scaling_law_{n}", m) for n, m in variants]
    try:
        for frac in FRACTIONS:
            cfg.SAFETY_FACTOR = 1.0 / frac
            for name, mdl in models:
                cand = dict(a=CHAMPION["a"], b=CHAMPION["b"],
                            n_turns=CHAMPION["n_turns"],
                            coil_half_gap=CHAMPION["coil_half_gap"])
                r = og.evaluate(cand, mdl, comm)
                if not r.get("feasible"):
                    _log(f"  [{name} @ {frac:.0%}] INFEASIBLE: {r.get('reason')}")
                    continue
                row = dict(model=name, frac_of_Ic=frac,
                           SAFETY_FACTOR=cfg.SAFETY_FACTOR,
                           I_quench_A=r["I_quench_A"], I_op_A=r["I_op_A"],
                           B_target_T=r["B_target_T"], hoop_MPa=r["hoop_MPa"],
                           peak_B_T=r["peak_B_T"], clip_frac=r["clip_frac"],
                           tape_km=r["tape_km"], binding=r["binding"])
                rows.append(row)
                _log(f"  {name:<30s} @{frac:.0%}  I_op={r['I_op_A']:>6.1f}A  "
                     f"B={r['B_target_T']:>6.2f}T  hoop={r['hoop_MPa']:>5.0f}MPa "
                     f"peakB={r['peak_B_T']:>5.2f}T  clip={r['clip_frac']:.3f}")
                with open(RES_CSV, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader(); w.writerows(rows)
    finally:
        cfg.SAFETY_FACTOR = orig_sf

    # ── summary ─────────────────────────────────────────────────────────────
    _log("")
    _log(f"{'model':<30}{'%Ic':>6}{'I_op[A]':>9}{'B_tgt[T]':>10}"
         f"{'hoop':>7}{'>=10T':>7}")
    for r in rows:
        _log(f"{r['model']:<30}{r['frac_of_Ic']:>6.0%}{r['I_op_A']:>9.1f}"
             f"{r['B_target_T']:>10.2f}{r['hoop_MPa']:>7.0f}"
             f"{'YES' if r['B_target_T'] >= 10.0 else 'no':>7}")
    _log("")
    _log("Reminder: the bc2_* rows BOUND the answer -- q and Bc2 are")
    _log("degenerate over the 1-8T fit window, so quote the band, not one row.")
    _log("Any row that changes I_op also changes the screening currents, so")
    _log("box uniformity must be re-validated with ta_validate.py before a")
    _log("new operating point is adopted.")
    _log(f"total {(time.time()-t0)/60:.1f} min -> {RES_CSV}")
    _make_fig(base_ic, variants)


def _make_fig(base_ic, variants):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_BG, AX_BG, SPINE = "#111", "#0d0d1a", "#444"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(FIG_BG)
    Bm = np.linspace(0.5, 8.0, 60)
    Bx = np.linspace(8.0, 20.0, 60)
    for ax, ang in zip(axes, (0.0, 90.0)):
        ax.set_facecolor(AX_BG)
        meas = np.array([base_ic.critical_current(np.array([b]), np.array([ang]),
                                                   clip_B=False)[0][0] for b in Bm])
        ax.plot(Bm, meas, color="white", lw=2.2, label="measured (0–8 T)")
        flat = np.full_like(Bx, meas[-1])
        ax.plot(Bx, flat, color="#ff5555", ls="--", lw=1.8,
                label="flat clamp (current default, optimistic)")
        cols = ["#ffb000", "#7cff6e", "#00e5ff"]
        curves = []
        for (name, mdl), c in zip(variants, cols):
            v = np.array([mdl.critical_current(np.array([b]),
                                               np.array([ang]))[0][0]
                          for b in Bx])
            curves.append(v)
            ax.plot(Bx, v, color=c, lw=2.0,
                    label=f"scaling law, $B_{{c2}}$={name.split('_')[1]}")
        if len(curves) >= 2:
            ax.fill_between(Bx, curves[0], curves[-1], color="#00e5ff",
                            alpha=0.13)
        ax.axvline(8.0, color="#888", ls=":", lw=1.2)
        ax.set_title(f"Ic vs B at {ang:.0f}°", color="white", fontsize=11)
        ax.set_xlabel("B [T]", color="white")
        ax.set_ylabel("Ic [A]", color="white")
        ax.tick_params(colors="white", labelsize=8)
        for s in ax.spines.values():
            s.set_color(SPINE)
        ax.grid(alpha=0.15, color="white", lw=0.5)
        ax.legend(facecolor=AX_BG, edgecolor=SPINE, labelcolor="white",
                  fontsize=7.5)
    fig.suptitle("Critical-current extrapolation above the 8 T measured limit\n"
                 "$J_c(B)=C\\,B^{p-1}(1-B/B_{c2})^q$  — band = $B_{c2}$ sensitivity "
                 "(degenerate over the 1–8 T fit window)", color="white", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIG_PATH, dpi=160, facecolor=FIG_BG)
    _log(f"figure -> {FIG_PATH}")


if __name__ == "__main__":
    main()
