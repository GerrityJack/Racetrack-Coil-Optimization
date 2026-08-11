"""plot_ramp_summary_dashboard.py -- 2026-08-08, one-page summary of the
constant-power ramp-up analysis, for the ramp-power report. Ties together
numbers already computed and verified by the other scripts in this
folder (does not recompute physics itself) -- see the source scripts
named in each section for how each number was derived.

Run:  <env>/bin/python3 visualization/plot_ramp_summary_dashboard.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "circuit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params                          # noqa: E402
import cparams as cfg                   # noqa: E402
from report_common import save_report    # noqa: E402

# Numbers below come from this session's actual script runs -- see:
#   circuit/power_ramp.py             (P recommendation, DCN margin)
#   circuit/power_ramp_loss_plots.py  (AC loss)
#   transient/validation/ta_quench_margin_check.py  (T-A margin)
I_DESIGN = 196.0
E_STORED_J = 8066.0
RAMP_S = 600.0
P_BY_RHO = {30: 14.59, 100: 13.96, 400: 13.62}
AC_LOSS_PCT_BY_RHO = {30: 16.16, 100: 6.03, 400: 1.92}
MARGIN_UNIFORM = 1.600
MARGIN_TA_SLOW = 0.709
MARGIN_TA_FAST = 0.687
MARGIN_THRESHOLD = 1.0 / 0.65

FONT_PT = 10.0
HEADER_PT = 13.0
LINE_IN = (FONT_PT * 1.65) / 72.0     # inches per body line, incl. leading
HEADER_IN = (HEADER_PT * 2.0) / 72.0  # inches consumed by a panel header
PAD_IN = 0.14                         # top+bottom padding inside a panel
GAP_IN = 0.10                         # gap between panels
MARGIN_IN = 0.35                      # outer left/right margin driver (unused directly)


def panel_height(n_lines):
    return HEADER_IN + n_lines * LINE_IN + 2 * PAD_IN


class Cursor:
    def __init__(self, fig_h_in, top_in):
        self.fig_h_in = fig_h_in
        self.y_in = fig_h_in - top_in   # current TOP edge, in inches from bottom

    def panel(self, ax, header, lines, header_color="#4fc3f7"):
        n = len(lines)
        h_in = panel_height(n)
        y1_in = self.y_in
        y0_in = y1_in - h_in
        y1 = y1_in / self.fig_h_in
        y0 = y0_in / self.fig_h_in

        ax.add_patch(plt.Rectangle((0.045, y0), 0.91, y1 - y0, transform=ax.transAxes,
                                   facecolor="#1a1a2e", edgecolor="#555",
                                   lw=0.8))
        ax.text(0.075, y1 - PAD_IN / self.fig_h_in, header, transform=ax.transAxes,
               ha="left", va="top", fontsize=HEADER_PT, color=header_color,
               fontweight="bold")
        body = "\n".join(lines)
        ax.text(0.075, y1 - PAD_IN / self.fig_h_in - HEADER_IN / self.fig_h_in,
               body, transform=ax.transAxes, ha="left", va="top", fontsize=FONT_PT,
               color="white", family="monospace", linespacing=1.65)

        self.y_in = y0_in - GAP_IN


def main():
    rec_lines = [
        f"Ramp to I_design over ~{RAMP_S:.0f} s (matches the fastest ramp",
        "this project has direct T-A evidence for -- see caveat below)",
        "",
        "  rho_c              P (const.)   Ramp time",
        f"  30  uOhm.cm^2 (low)      {P_BY_RHO[30]:>6.2f} W    {RAMP_S:.0f} s",
        f"  100 uOhm.cm^2 (nominal)  {P_BY_RHO[100]:>6.2f} W    {RAMP_S:.0f} s   <== headline",
        f"  400 uOhm.cm^2 (high)     {P_BY_RHO[400]:>6.2f} W    {RAMP_S:.0f} s",
    ]

    energy_lines = [
        f"Stored magnetic energy               {E_STORED_J:>8.0f} J",
        f"  (comparable to a 60W bulb for ~2 minutes)",
        f"AC loss (contact+SC) at nominal rho_c=100    {AC_LOSS_PCT_BY_RHO[100]:>5.1f}%  of E",
        f"AC loss range across rho_c uncertainty  {AC_LOSS_PCT_BY_RHO[400]:.1f}% -- {AC_LOSS_PCT_BY_RHO[30]:.1f}%  of E",
        "Dominant mechanism: NI turn-to-turn contact loss",
        "  (SC hysteresis loss is ~1e4-1e5x smaller)",
        "NOT modeled: eddy current loss in any metal structure",
        "  (none is represented in this project's models -- a real",
        "   gap if the build has metal former/stabilizer nearby)",
    ]

    margin_lines = [
        "Worst-cell margin = Jc(B,theta)/|J|, at I_design:",
        "",
        f"  uniform-J (used by every other quench check)   {MARGIN_UNIFORM:.3f}",
        f"  T-A, dt=600s single-step (production)           {MARGIN_TA_SLOW:.3f}  *",
        f"  T-A, dt=60s x10 genuine multi-step               {MARGIN_TA_FAST:.3f}  *",
        f"  design threshold (1/0.65)                        {MARGIN_THRESHOLD:.3f}",
        "",
        "* below 1.0 (E_c-defined Ic) at 27-34% of coil cells --",
        "  present at ANY ramp speed tested, not fast-ramp-specific.",
        "Open question (not resolved): real safety gap, or expected",
        "Bean critical-state behaviour the uniform-J margin was never",
        "built to catch? See CLAUDE.md \"Ramp-up power analysis\".",
    ]

    basis_lines = [
        "DCN (Phase A, circuit/) -- validated: energy balance closes",
        "to 0.00-0.08%, filament sum agrees with production",
        "Biot-Savart to 0.18%/0.44% median/max, He et al. 2025",
        "benchmark to 4.2%.",
        "",
        "T-A cross-check -- uses the alpha=(0.03,0.01) relaxation fix,",
        "validated across dt in [60,600]s, I in [19.6,196]A, including",
        "a genuine 10-step multi-step ramp. Single-threaded only fully",
        "validated; ~2% single-vs-multi-threaded gap still open.",
        "",
        "NEW 2026-08-08/10: first-ever ramp-DOWN run of the T-A solver,",
        "extended to THREE full cycles (196A<->~2A, x3). Clean",
        "throughout (dB_rel 0.003-0.024, all 30 steps finite, all 6",
        "reversals). Cycle 1 alone qualitatively matches Bean theory",
        "(SCIF crosses zero, settles at a remanent value of the right",
        "sign). This comparison also caught and fixed a real bug in the",
        "analytical derivation (wrong peak assumed for i0<1).",
        "",
        "FINDING: the loop does NOT close after 1 cycle -- but the gap",
        "SHRINKS each successive cycle by a consistent ~0.65x ratio at",
        "both the peak (757.0->647.6->576.1mT, d=-109.4 then -71.5mT)",
        "and the remanent point (-367.1->-464.8->-529.8mT, d=-97.7 then",
        "-65.0mT). That consistency across two independent points is",
        "real evidence of decaying convergence toward a stable loop, NOT",
        "constant drift or a runaway instability -- though where it",
        "settles, and how many more cycles it needs, is not established",
        "(a 2-point ratio is not a tight extrapolation).",
    ]

    figs_lines = [
        "01  Current / voltage / power vs time",
        "02  AC loss power + cumulative energy vs time",
        "03  AC loss fraction vs contact resistivity",
        "04  Quench margin comparison (uniform-J vs T-A)",
        "05  Illustrative Bean hysteresis loop (analytical)",
        "06  Simulated T-A 3-cycle loop vs. analytical (gap shrinks ~0.65x/cycle)",
        "field_animation.gif / field_frames/   3D field build-up",
    ]

    sections = [
        ("RECOMMENDATION", rec_lines, "#81c784"),
        ("ENERGY BUDGET", energy_lines, "#4fc3f7"),
        ("QUENCH MARGIN -- KEY OPEN CAVEAT", margin_lines, "#ffb74d"),
        ("WHAT THIS RESTS ON", basis_lines, "#4fc3f7"),
        ("FIGURES IN THIS FOLDER", figs_lines, "#4fc3f7"),
    ]

    title_block_in = 1.0
    total_in = title_block_in
    for _, lines, _ in sections:
        total_in += panel_height(len(lines)) + GAP_IN
    total_in += 0.25

    fig_w = 9.5
    fig = plt.figure(figsize=(fig_w, total_in))
    fig.patch.set_facecolor(cfg.FIG_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.5, 1.0 - 0.30 / total_in, "Constant-Power Ramp-Up to I_design -- Summary",
           transform=ax.transAxes, ha="center", va="top", fontsize=19,
           color="white", fontweight="bold")
    ax.text(0.5, 1.0 - 0.62 / total_in,
           f"Champion coil ({params.n_turns_total} turns, I_design={I_DESIGN:.0f} A)  --  2026-08-08",
           transform=ax.transAxes, ha="center", va="top", fontsize=11, color="#aaa")

    cur = Cursor(total_in, title_block_in)
    for header, lines, color in sections:
        cur.panel(ax, header, lines, header_color=color)

    out = save_report(fig, "00_summary_dashboard.png")


if __name__ == "__main__":
    main()
