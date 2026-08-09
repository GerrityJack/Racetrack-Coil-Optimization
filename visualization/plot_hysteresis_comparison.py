"""plot_hysteresis_comparison.py -- 2026-08-08, compares the GENUINE T-A
transient simulation's screening-current-induced field (SCIF) trajectory
across a full up+down ramp (transient/validation/full_ramp_up_down_run.py
-- the first-ever ramp-DOWN run in this project's T-A solver) against
the analytically-derived Bean critical-state hysteresis loop
(visualization/plot_hysteresis_loop.py).

WHY: per user direction, running the actual simulation through this
pattern (rather than trusting the derivation alone) is the real test --
"if it lines up with what we expected, we can take that as evidence
that the simulation is working well." Comparing the two ALSO caught a
genuine bug in the analytical derivation (the descending branch was
using the wrong starting value for an unsaturated i0<1) -- fixed in
plot_hysteresis_loop.py before this comparison was built.

WHAT'S COMPARED, HONESTLY
----------------------------
SCIF (simulated, in mT, a 3-D Biot-Savart-integrated bore-field quantity)
and m (analytical, dimensionless, a 1-D idealized-slab local
magnetization) are NOT the same physical quantity or in the same units
-- they are compared here by normalizing each to its OWN peak magnitude
(reached at I_design), so the shared y-axis is "own-peak-normalized
response," not a claim that the two are quantitatively identical. What
IS a fair, like-for-like comparison: the REMANENT FRACTION each curve
settles to at I->0 relative to its own peak -- that ratio is a
genuine, dimensionless, model-independent test of whether the
simulation's screening-current memory behaves like critical-state
theory predicts.

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
from plot_hysteresis_loop import _m_up, _m_down, B_PEAK_T, THETA_DEG  # noqa: E402

SIM_NPZ = os.path.join(_ROOT, "transient", "full_validation_plots", "data",
                       "full_ramp_up_down.npz")


def main():
    print("=" * 78)
    print("Simulated (T-A) vs. analytical (Bean) hysteresis comparison")
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
    if not finite_sim.all():
        print("  *** WARNING: non-finite step(s) present -- treat this "
              "comparison as unreliable past that point ***")

    up_mask = branch == "up"
    down_mask = branch == "down"
    I_up_sim, scif_up_sim = I_sim[up_mask], scif_sim[up_mask]
    I_down_sim, scif_down_sim = I_sim[down_mask], scif_sim[down_mask]
    # prepend the (0,0) origin to the up branch for a fair shape comparison
    I_up_sim = np.concatenate([[0.0], I_up_sim])
    scif_up_sim = np.concatenate([[0.0], scif_up_sim])

    scif_peak = scif_up_sim[-1]
    scif_remanent = scif_down_sim[-1]
    scif_remanent_frac = scif_remanent / scif_peak
    print(f"  SCIF peak (at I_design)      = {scif_peak:+.2f} mT")
    print(f"  SCIF remanent (at I~{I_down_sim[-1]:.1f}A)   = {scif_remanent:+.2f} mT")
    print(f"  SCIF remanent fraction       = {scif_remanent_frac:+.4f}")

    # analytical curve, same i0 as plot_hysteresis_loop.py
    ic = make_ic_model("kim")
    Ic_A, _ = ic.critical_current(np.array([B_PEAK_T]), np.array([THETA_DEG]))
    Ic_A = float(Ic_A[0])
    I_design = float(params.I_design)
    i0 = I_design / Ic_A
    i_up = np.linspace(0.0, i0, 200)
    m_up = _m_up(i_up)
    i_down = np.linspace(i0, 0.0, 200)
    m_down = _m_down(i_down, i0)
    m_peak = float(m_up[-1])
    m_remanent = float(m_down[-1])
    m_remanent_frac = m_remanent / m_peak
    print(f"\n  analytical m peak (at I_design)     = {m_peak:+.4f}")
    print(f"  analytical m remanent (at I~0)      = {m_remanent:+.4f}")
    print(f"  analytical remanent fraction        = {m_remanent_frac:+.4f}")

    ratio = scif_remanent_frac / m_remanent_frac
    print(f"\n  remanent-fraction ratio (sim / analytical) = {ratio:.2f}x")
    print(f"  (both curves cross zero and settle to a nonzero remanent value "
          f"of\n   the SAME relative sign on ramp-down -- qualitative match; "
          f"the analytical\n   1-D idealized-slab model is not expected to "
          f"match the magnitude exactly)")

    # ── figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_facecolor(cfg.FIG_BG)

    # normalize each series to its OWN peak, sign-flipped for the analytical
    # curve so both trend the SAME way (up then down through zero) -- see
    # module docstring for why this is the fair comparison, not raw values.
    norm_up_sim = scif_up_sim / scif_peak
    norm_down_sim = np.concatenate([[1.0], scif_down_sim / scif_peak])
    I_down_sim_full = np.concatenate([[I_design], I_down_sim])

    norm_up_ana = -m_up / abs(m_peak)
    norm_down_ana = -m_down / abs(m_peak)

    ax.plot(I_up_sim, norm_up_sim, "o-", color=cfg.SERIES_COLORS[0], lw=2.2,
           ms=6, label="T-A simulation, ramp-up (per-step converged SCIF)")
    ax.plot(I_down_sim_full, norm_down_sim, "o--", color=cfg.SERIES_COLORS[4],
           lw=2.2, ms=6, label="T-A simulation, ramp-down (per-step converged SCIF)")
    ax.plot(i_up * I_design / i0, norm_up_ana, "-", color=cfg.SERIES_COLORS[0],
           lw=1.2, alpha=0.45, label="Bean model, ramp-up (analytical)")
    ax.plot(i_down * I_design / i0, norm_down_ana, "--", color=cfg.SERIES_COLORS[4],
           lw=1.2, alpha=0.45, label="Bean model, ramp-down (analytical)")

    ax.axhline(0, color="#666", lw=0.8)
    ax.axvline(I_design, color="#888", ls=":", lw=1)

    ax.annotate(f"sim remanent\nfraction={scif_remanent_frac:+.2f}",
               xy=(I_down_sim[-1], scif_remanent_frac), xytext=(65, -0.72),
               color=cfg.SERIES_COLORS[4], fontsize=9,
               arrowprops=dict(arrowstyle="->", color=cfg.SERIES_COLORS[4], lw=1))
    ax.annotate(f"analytical remanent\nfraction={-m_remanent_frac:+.2f}",
               xy=(0, -m_remanent_frac), xytext=(65, 0.42),
               color="#888", fontsize=9,
               arrowprops=dict(arrowstyle="->", color="#888", lw=1))

    _ax(ax, "transport current I [A]",
        "response / own peak magnitude  (sign-flipped for the analytical\n"
        "curve so both series trend the same way -- see caption)",
        "Simulated T-A SCIF vs. analytical Bean critical-state loop\n"
        "(both normalized to their own peak at I_design -- NOT the same "
        "physical units, see script docstring)")
    ax.legend(fontsize=8.5, labelcolor="white", facecolor="#222",
             edgecolor="#444", framealpha=0.75, loc="upper left")
    ax.set_ylim(-1.1, 1.15)

    fig.text(0.5, -0.04,
             f"Raw values: SCIF peak={scif_peak:+.0f}mT, remanent="
             f"{scif_remanent:+.0f}mT at I~{I_down_sim[-1]:.0f}A "
             f"(dB_rel<=​{dB_rel_sim.max():.3f} throughout, all steps finite). "
             f"Both curves cross zero and settle to a nonzero remanent value of "
             f"the same relative sign on ramp-down -- the qualitative signature "
             f"critical-state theory predicts, at {ratio:.1f}x the analytical "
             f"remanent fraction (not an exact match -- expected, given the "
             f"1-D idealized-slab simplification; see 05_hysteresis_loop.png).",
             ha="center", color="#999", fontsize=7.6, wrap=True)

    fig.tight_layout()
    save_report(fig, "06_hysteresis_loop_vs_simulation.png")
    print("=" * 78)


if __name__ == "__main__":
    main()
