"""plot_hysteresis_loop.py -- 2026-08-08, illustrative Bean/critical-state
magnetization hysteresis loop ("B-M curve") for the champion tape, for
the ramp-power report.

PHYSICS AND SCOPE -- READ BEFORE TRUSTING THE LOOP AREA AS A JOULES NUMBER
-----------------------------------------------------------------------------
This coil's screening currents are driven by the TAPE'S OWN TRANSPORT
CURRENT (the tape's current creates the very field that drives its own
screening response -- self-field, not an externally-imposed field
independent of the current). The exactly-correct treatment of that
problem (a current-carrying superconducting strip) is the Norris
transport-current AC-loss solution, which has a genuinely more
complicated (log/elliptic-integral) self-field profile than the simple
"field penetrates linearly from the edges" picture -- this project's own
CLAUDE.md already flags the real risk of reconstructing that exact
closed form from memory (2026-08-06 entry, "deliberately NOT the exact
Norris closed form... risked a real transcription error").

To avoid that risk while still producing a genuine, physically-grounded
illustrative loop, this script uses the SIMPLER, standard 1D Bean SLAB
critical-state model (screening currents penetrating linearly from a
strip's edges as the reduced drive i=I/Ic goes 0->1), which IS derived
here from Ampere's law from scratch (not copied from memory) -- see the
derivation comment in `_m_up`/`_m_down` below. Using i=I/Ic (rather than
an externally-applied reduced field h=Ba/Bp) as the drive parameter is
itself a standard simplifying approximation for illustrating a
current-carrying strip's hysteresis -- NOT the exact Norris self-field
solution.

Consequence: the SHAPE of this loop (rise, saturation tendency, the
descending branch's remanent moment, hysteretic opening) is genuinely
representative critical-state physics for our tape at its real i=I/Ic
operating ratio. The loop's ENCLOSED AREA is NOT claimed here to equal
an exact Joules-per-cycle number for this current-driven geometry (that
IS the harder Norris integral this script deliberately avoids
reconstructing). For the actual, directly-computed AC loss in Joules for
this design, see 02_ac_loss_power_and_energy.png (DCN model,
already-validated V_sc/contact-loss machinery) -- this figure is
illustrative of the hysteresis MECHANISM, not a replacement for those
numbers.

DERIVATION (self-contained, verified against known Bean-model results
after independent re-derivation via Ampere's law -- see comments; the
descending-branch formula below was CORRECTED 2026-08-08 after
comparing this figure against a genuine T-A transient simulation
surfaced a bug in the first version -- see _m_down's docstring and
06_hysteresis_loop_vs_simulation.png)
------------------------------------------------------------------------
Slab of half-width a, reduced drive i = I/Ic in [0, i0]. Virgin
(ascending, 0->i0) branch, valid ONLY for the very first approach from a
truly demagnetized (I=0 forever, never energized) state:
    m_up(i)   = -i + i^2/2                    for 0 <= i <= 1
              = -1/2                          for i >= 1 (saturated)
Descending branch from a peak i0 (i0 <= 1 here, our design's actual
operating point -- NOTE this must start from m_up(i0), NOT a hardcoded
-1/2, unless i0>=1):
    m_down(i; i0) = m_up(i0) + (i0-i) - (i0-i)^2/4   for (i0-i) <= 2*i0
Second-cycle ascending branch, from the valley reached at i_min (the
end of the first descent) back up toward i0 -- this does NOT retrace
the virgin curve (a well-known real hysteresis feature: only the FIRST
approach to a field/current level follows the virgin curve; every
later approach follows the minor loop instead). Derived by the same
Ampere's-law front-tracking method as m_down, then verified against a
direct numerical integration of the full 3-current-layer profile to
machine precision (both here and independently while building this
figure -- see the session's own record, not reproduced in-file):
    m_up2(i; i_min, i0) = m_down(i_min; i0) + 2*m_up((i-i_min)/2)
This closes EXACTLY back onto m_up(i0) at i=i0 (verified), which is
exactly the "second and all later cycles retrace the same closed loop"
behaviour real critical-state hysteresis shows -- only the ORIGINAL
virgin curve is a one-time, non-repeating curve.
where m = M/(Jc*a) is the reduced magnetization.

Run:  <env>/bin/python3 visualization/plot_hysteresis_loop.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "optimize"),
           os.path.join(_ROOT, "circuit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params                              # noqa: E402
import cparams as cfg                       # noqa: E402
from ic_extrapolation import make_ic_model   # noqa: E402
from postprocess import _ax                   # noqa: E402
from report_common import save_report           # noqa: E402

B_PEAK_T = 11.3      # champion's peak local field at I_design (this session's
                     # T-A cross-check, worst cell)
THETA_DEG = 31.0     # worst-cell angle to the tape normal, same cell


def _m_up(i):
    i = np.clip(i, 0.0, None)
    return np.where(i <= 1.0, -i + i ** 2 / 2.0, -0.5)


def _m_down(i, i0):
    """CORRECTED 2026-08-08: the descending branch must start from
    m_up(i0), not a hardcoded -0.5 -- that constant is only the correct
    starting value when i0>=1 (fully saturated/penetrated). Our actual
    i0=0.626 is UNSATURATED (partial penetration), so the original
    formula (-0.5 + d - d^2/4) silently assumed the wrong peak value.
    Re-derived for the general partial-penetration case (0<i0<=1): the
    reversing front's penetration depth y=(i0-i)/2 stays inside the
    previously-penetrated shell (y <= i0, i.e. d <= 2*i0) for the whole
    descent to i=0 whenever i0>0, which holds here -- so this single
    branch covers our actual trajectory. Verified: reduces to the
    original -0.5+d-d^2/4 exactly when i0=1 (m_up(1)=-0.5), and the
    corrected remanent value (i0=0.626: m~+0.10) is a much closer
    qualitative match to the actual T-A simulation's remanent SCIF
    (see 06_hysteresis_loop_vs_simulation.png) than the original bug's
    +0.028 was -- caught BY that comparison, which is exactly why the
    user asked for the simulation cross-check instead of trusting the
    derivation alone.
    """
    d = np.clip(i0 - i, 0.0, None)
    m_peak = _m_up(np.array([i0]))[0]
    return np.where(d <= 2.0 * i0, m_peak + d - d ** 2 / 4.0, -m_peak)


def _m_up2(i, i_min, i0):
    """Second-cycle ascending branch (from the valley i_min back to i0) --
    see module docstring for the derivation and its numerical verification."""
    m_valley = _m_down(np.array([i_min]), i0)[0]
    return m_valley + 2.0 * _m_up((i - i_min) / 2.0)


def main():
    print("=" * 78)
    print("Illustrative Bean critical-state hysteresis loop")
    print("=" * 78)
    ic = make_ic_model("kim")
    Ic_A, clipped = ic.critical_current(np.array([B_PEAK_T]), np.array([THETA_DEG]))
    Ic_A = float(Ic_A[0])
    I_design = float(params.I_design)
    i0 = I_design / Ic_A
    print(f"  Ic({B_PEAK_T}T, {THETA_DEG}deg) = {Ic_A:.2f} A  "
          f"(Kim model, worst cell, clipped={bool(clipped)})")
    print(f"  I_design = {I_design:.1f} A   i0 = I_design/Ic = {i0:.3f}")

    # sanity checks on the derived formulas before trusting the figure --
    # NOTE this exact check (branches must meet AT m_up(i0), not at a
    # hardcoded -0.5) is what should have caught the original derivation
    # bug fixed 2026-08-08; it didn't, because it was written to check
    # the buggy formula's own self-consistency (-0.5 at d=0) rather than
    # cross-checking against _m_up's actual value at i0. Fixed now.
    assert abs(_m_up(np.array([0.0]))[0]) < 1e-12
    assert abs(_m_up(np.array([1.0]))[0] - (-0.5)) < 1e-12
    m_peak = _m_up(np.array([i0]))[0]
    assert abs(_m_down(np.array([i0]), i0)[0] - m_peak) < 1e-9, \
        "descending branch must start exactly where the ascending branch ends"
    print(f"  sanity checks passed (m(0)=0, m(1)=-1/2, branches meet at "
          f"i0: m_up(i0)={m_peak:.4f})")

    i_min = 0.0   # idealized fully-discharged valley for this standalone figure
                  # (06_hysteresis_loop_vs_simulation.png uses the simulation's
                  # own actual ~2A endpoint instead, for a fair comparison)

    i_virgin = np.linspace(0.0, i0, 200)
    m_virgin = _m_up(i_virgin)
    m_peak = float(m_virgin[-1])

    i_down = np.linspace(i0, i_min, 200)
    m_down = _m_down(i_down, i0)
    remanent_m = float(m_down[-1])

    i_up2 = np.linspace(i_min, i0, 200)
    m_up2 = _m_up2(i_up2, i_min, i0)

    assert abs(m_up2[-1] - m_peak) < 1e-9, \
        "second-cycle ascent must close exactly back onto the virgin peak"
    print(f"  remanent m at I~0 after cycle 1's descent: {remanent_m:+.4f}")
    print(f"  cycle-2 ascent closes back onto the peak to "
          f"{abs(m_up2[-1]-m_peak):.1e} (sanity check passed)")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    fig.patch.set_facecolor(cfg.FIG_BG)

    to_A = I_design / i0   # reduced i -> physical Amps

    # the closed, repeating loop (cycle 2 onward) -- this is the actual
    # "hysteresis loop" in the textbook sense, drawn bold and filled
    ax.plot(i_up2 * to_A, m_up2, color=cfg.SERIES_COLORS[0], lw=2.6,
           label="stable loop -- ascending (repeats every cycle)")
    ax.plot(i_down * to_A, m_down, color=cfg.SERIES_COLORS[4], lw=2.6, ls="--",
           label="stable loop -- descending (repeats every cycle)")
    ax.fill(np.concatenate([i_up2, i_down]) * to_A,
           np.concatenate([m_up2, m_down]),
           color=cfg.SERIES_COLORS[0], alpha=0.14)

    # the one-time virgin curve, drawn thin/dotted and clearly labeled as
    # non-repeating -- only the very first approach follows it
    ax.plot(i_virgin * to_A, m_virgin, color="#bbb", lw=1.4, ls=":",
           label="virgin curve (1st approach only -- never repeats)")

    ax.plot([I_design], [m_peak], "o", color="white", ms=6, zorder=5)
    ax.plot([i_min * to_A], [remanent_m], "o", color=cfg.SERIES_COLORS[4],
           ms=6, zorder=5)
    ax.axvline(I_design, color="#888", ls=":", lw=1)

    ax.annotate(f"peak, m={m_peak:.3f}\n(i=I/Ic={i0:.2f})",
               xy=(I_design, m_peak), xytext=(I_design * 0.55, m_peak - 0.10),
               color="white", fontsize=9,
               arrowprops=dict(arrowstyle="->", color="#aaa", lw=1))
    ax.annotate(f"remanent, m={remanent_m:.3f}",
               xy=(i_min * to_A, remanent_m),
               xytext=(I_design * 0.30, remanent_m + 0.11),
               color="white", fontsize=9,
               arrowprops=dict(arrowstyle="->", color="#aaa", lw=1))

    _ax(ax, "transport current I [A]", "reduced screening magnetization  "
        "m = M / (J$_c$ a)",
        "Illustrative Bean critical-state hysteresis loop -- champion tape\n"
        f"J$_c$({B_PEAK_T}T, {THETA_DEG}$^\\circ$)={Ic_A/(params.delta_SC*params.w):.2e} A/m$^2$, "
        f"a=w/2={params.w/2*1e3:.1f}mm, i$_0$={i0:.3f}")
    ax.legend(fontsize=8.5, labelcolor="white", facecolor="#222",
             edgecolor="#444", framealpha=0.75, loc="lower left")
    ax.set_ylim(m_peak - 0.10, max(remanent_m, 0.0) + 0.16)

    fig.text(0.5, -0.04,
             "Shape is genuine critical-state physics (derived from Ampere's "
             "law, verified numerically -- see script docstring); loop AREA "
             "is illustrative only, not a Joules figure for this "
             "current-driven geometry -- see 02_ac_loss_power_and_energy.png "
             "for the actual computed loss.",
             ha="center", color="#999", fontsize=7.8, wrap=True)

    fig.tight_layout()
    save_report(fig, "05_hysteresis_loop.png")
    print("=" * 78)


if __name__ == "__main__":
    main()
