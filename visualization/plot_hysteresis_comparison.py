"""plot_hysteresis_comparison.py -- 2026-08-08/10, compares the GENUINE
T-A transient simulation's screening-current-induced field (SCIF)
trajectory across THREE full up+down cycles
(transient/validation/full_ramp_up_down_run.py -- the first-ever
ramp-DOWN run, now extended to three cycles, in this project's T-A
solver) against the analytically-derived Bean critical-state hysteresis
loop (visualization/plot_hysteresis_loop.py).

WHY THREE CYCLES: two cycles showed the loop does NOT close after one
pass (cycle 2 sat ~100-110mT below cycle 1 at every matching current) --
but that alone couldn't distinguish "settling toward a stable loop"
(gap should shrink) from "a real ongoing drift" (gap stays ~constant).
A third cycle answers this directly.

RESULT: the gap SHRINKS by a consistent ~0.65x ratio at both the peak
(757.0 -> 647.6 -> 576.1mT, deltas -109.4 then -71.5mT) and the remanent
point (-367.1 -> -464.8 -> -529.8mT, deltas -97.7 then -65.0mT). That
consistency across two independent points on the loop is itself
evidence this is a real, systematic (likely geometrically-decaying)
convergence toward a stable minor loop, NOT constant drift or a runaway
instability -- though the loop still has not closed within 3 cycles,
and a decaying-ratio read from only 2 data points is not a tight
extrapolation. All 30 steps across all 3 cycles stayed numerically clean
(dB_rel 0.003-0.024, every step finite, all 6 direction reversals
handled without incident).

UNITS: the analytical curve (dimensionless m) is scaled by
scif_peak_cycle1/|m_peak| so it plots directly in mT-equivalent units
alongside the real SCIF data -- a peak-matched VISUAL overlay for shape
comparison, not a claim that m and SCIF are the same physical quantity.

Run:  <env>/bin/python3 visualization/plot_hysteresis_comparison.py
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
           os.path.join(_ROOT, "circuit"), os.path.join(_ROOT, "transient", "validation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params                              # noqa: E402
import cparams as cfg                       # noqa: E402
from ic_extrapolation import make_ic_model   # noqa: E402
from postprocess import _ax                   # noqa: E402
from report_common import save_report           # noqa: E402
from plot_hysteresis_loop import _m_up, _m_down, _m_up2, B_PEAK_T, THETA_DEG  # noqa: E402

SIM_NPZ = os.path.join(_ROOT, "transient", "full_validation_plots", "data",
                       "full_ramp_3cycle.npz")

UP_COLORS = [cfg.SERIES_COLORS[0], cfg.SERIES_COLORS[7], cfg.SERIES_COLORS[6]]
DOWN_COLORS = [cfg.SERIES_COLORS[4], cfg.SERIES_COLORS[5], cfg.SERIES_COLORS[1]]
MARKERS = ["o", "s", "^"]


def _branch_xy(I_sim, scif_sim, branch, name, prepend=None):
    mask = branch == name
    x, y = I_sim[mask], scif_sim[mask]
    if prepend is not None:
        x = np.concatenate([[prepend[0]], x])
        y = np.concatenate([[prepend[1]], y])
    return x, y


def main():
    print("=" * 78)
    print("Simulated (T-A, 3 cycles) vs. analytical (Bean) hysteresis comparison")
    print("=" * 78)

    d = np.load(SIM_NPZ, allow_pickle=True)
    steps = list(d["step_summaries"])
    I_sim = np.array([s["I_now"] for s in steps])
    scif_sim = np.array([s["scif_mT"] for s in steps])
    finite_sim = np.array([s["finite"] for s in steps])
    dB_rel_sim = np.array([s["dB_rel"] for s in steps])
    branch = np.array([s["branch"] for s in steps])
    print(f"  loaded {len(steps)} steps from {SIM_NPZ}")
    print(f"  all finite: {bool(finite_sim.all())}   "
          f"dB_rel range: [{dB_rel_sim.min():.4f}, {dB_rel_sim.max():.4f}]")
    if not finite_sim.all() or len(steps) < 30:
        print("  *** WARNING: incomplete or non-finite run -- treat this "
              "comparison as unreliable ***")

    ups, downs = [], []
    prev = (0.0, 0.0)
    for c in range(3):
        Iu, su = _branch_xy(I_sim, scif_sim, branch, f"up{c+1}", prepend=prev)
        ups.append((Iu, su))
        prev = (Iu[-1], su[-1])
        Id, sd = _branch_xy(I_sim, scif_sim, branch, f"down{c+1}", prepend=prev)
        downs.append((Id, sd))
        prev = (Id[-1], sd[-1])

    peaks = [u[1][-1] for u in ups]
    remanents = [dn[1][-1] for dn in downs]
    print()
    for c in range(3):
        dpk = f"  (d={peaks[c]-peaks[c-1]:+.1f}mT)" if c > 0 else ""
        drm = f"  (d={remanents[c]-remanents[c-1]:+.1f}mT)" if c > 0 else ""
        print(f"  cycle {c+1}: peak={peaks[c]:+7.1f}mT{dpk:<16s}"
              f" remanent={remanents[c]:+7.1f}mT{drm}")
    ratio_peak = (peaks[2] - peaks[1]) / (peaks[1] - peaks[0])
    ratio_rem = (remanents[2] - remanents[1]) / (remanents[1] - remanents[0])
    print(f"\n  cycle-to-cycle gap ratio: peak={ratio_peak:.3f}  "
          f"remanent={ratio_rem:.3f}  (both <1 and similar => decaying "
          f"toward a stable loop, not constant drift)")

    # analytical curve (dimensionless m), scaled to mT-equivalent by matching
    # cycle 1's peak -- see docstring "UNITS"
    ic = make_ic_model("kim")
    Ic_A, _ = ic.critical_current(np.array([B_PEAK_T]), np.array([THETA_DEG]))
    Ic_A = float(Ic_A[0])
    I_design = float(params.I_design)
    i0 = I_design / Ic_A
    i_min = float(downs[0][0][-1]) / Ic_A
    i_up = np.linspace(0.0, i0, 200)
    i_down = np.linspace(i0, i_min, 200)
    i_up2a = np.linspace(i_min, i0, 200)
    m_up, m_down = _m_up(i_up), _m_down(i_down, i0)
    m_up2 = _m_up2(i_up2a, i_min, i0)
    m_peak = float(m_up[-1])
    scale = peaks[0] / abs(m_peak)
    ana_up, ana_down, ana_up2 = -m_up * scale, -m_down * scale, -m_up2 * scale

    # ── figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7.2))
    fig.patch.set_facecolor(cfg.FIG_BG)

    ax.plot(i_up * I_design / i0, ana_up, "-", color="#888", lw=1.2, alpha=0.5,
           label="Bean model, virgin curve (scaled to cycle-1 peak)")
    ax.plot(i_down * I_design / i0, ana_down, "-", color="#888", lw=1.2, alpha=0.5)
    ax.plot(i_up2a * I_design / i0, ana_up2, "--", color="#888", lw=1.2, alpha=0.5,
           label="Bean model, stable loop (exact closure)")

    for c in range(3):
        Iu, su = ups[c]
        Id, sd = downs[c]
        a = 1.0 - 0.18 * c
        ax.plot(Iu, su, MARKERS[c] + "-", color=UP_COLORS[c], lw=2.2 - 0.2 * c,
               ms=6 - c, alpha=a, label=f"T-A sim, cycle {c+1} up")
        ax.plot(Id, sd, MARKERS[c] + "--", color=DOWN_COLORS[c], lw=2.2 - 0.2 * c,
               ms=6 - c, alpha=a, label=f"T-A sim, cycle {c+1} down")

    ax.axhline(0, color="#555", lw=0.8)
    ax.axvline(I_design, color="#777", ls=":", lw=1)

    stats = (f"{'':>10s}{'cyc1':>9s}{'cyc2':>9s}{'cyc3':>9s}{'d(2-1)':>9s}{'d(3-2)':>9s}\n"
            f"{'peak':>10s}{peaks[0]:>9.0f}{peaks[1]:>9.0f}{peaks[2]:>9.0f}"
            f"{peaks[1]-peaks[0]:>+9.0f}{peaks[2]-peaks[1]:>+9.0f}\n"
            f"{'remanent':>10s}{remanents[0]:>9.0f}{remanents[1]:>9.0f}{remanents[2]:>9.0f}"
            f"{remanents[1]-remanents[0]:>+9.0f}{remanents[2]-remanents[1]:>+9.0f}\n"
            f"{'(mT)':>10s}{'':>9s}{'':>9s}{'':>9s}"
            f"{'ratio:':>9s}{ratio_peak:>8.2f}x")
    ax.text(0.985, 0.03, stats, transform=ax.transAxes, ha="right", va="bottom",
           fontsize=8, family="monospace", color="white",
           bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                     edgecolor="#555", alpha=0.95))

    _ax(ax, "transport current I [A]", "SCIF [mT]  (T-A sim); "
        "scaled reduced magnetization [mT-equiv.]  (Bean model)",
        "Simulated T-A SCIF over THREE full ramp cycles vs. the analytical "
        "Bean critical-state loop\n"
        "-- gap shrinks each cycle (ratio~"
        f"{ratio_peak:.2f}x, decaying toward closure, not constant drift); "
        f"clean throughout (dB_rel<={dB_rel_sim.max():.3f}, all steps finite)")
    ax.legend(fontsize=7.8, labelcolor="white", facecolor="#222",
             edgecolor="#444", framealpha=0.75, loc="upper left", ncol=1)

    fig.text(0.5, -0.03,
             "Analytical curve scaled to cycle-1's peak for shape comparison "
             "only (not the same physical quantity as SCIF). The loop has "
             "not fully closed within 3 cycles -- the shrinking (not "
             "constant) gap is evidence of genuine convergence, not proof "
             "of where it settles; see 00_summary_dashboard.png.",
             ha="center", color="#999", fontsize=7.6, wrap=True)

    fig.tight_layout()
    save_report(fig, "06_hysteresis_loop_vs_simulation.png")
    print("=" * 78)


if __name__ == "__main__":
    main()
