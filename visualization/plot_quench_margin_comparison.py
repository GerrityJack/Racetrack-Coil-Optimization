"""plot_quench_margin_comparison.py -- 2026-08-08, summary bar chart of
the T-A-vs-uniform-J quench margin finding (transient/validation/
ta_quench_margin_check.py), for the ramp-power report.

Reuses that script's own evaluate() function directly (not a
re-implementation) so the numbers here are guaranteed to match the
underlying investigation exactly.

Run:  <env>/bin/python3 visualization/plot_quench_margin_comparison.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "solve"),
           os.path.join(_ROOT, "optimize"), os.path.join(_ROOT, "circuit"),
           os.path.join(_ROOT, "transient", "validation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params                          # noqa: E402
import cparams as cfg                   # noqa: E402
from ic_extrapolation import make_ic_model   # noqa: E402
from postprocess import _ax                   # noqa: E402
from report_common import save_report           # noqa: E402
import ta_quench_margin_check as tqmc            # noqa: E402


def main():
    print("=" * 78)
    print("Quench-margin comparison chart (T-A vs uniform-J)")
    print("=" * 78)
    ic = make_ic_model("kim")

    d1 = np.load(tqmc.FIELDS_NPZ)
    r_unif = tqmc.evaluate("uniform-J\n(naive)", d1["coil_centroids"], d1["coil_B"],
                          d1["J_unif_coil"], float(d1["delta_SC"]), params.w, ic)
    r_slow = tqmc.evaluate("T-A, dt=600s\n(1 step, production)", d1["coil_centroids"],
                          d1["coil_B"], d1["J_TA_coil"], float(d1["delta_SC"]),
                          params.w, ic)

    rows = [r_unif, r_slow]
    if os.path.exists(tqmc.FULL_RAMP_NPZ):
        d2 = np.load(tqmc.FULL_RAMP_NPZ, allow_pickle=True)
        r_fast = tqmc.evaluate("T-A, dt=60s x10\n(genuine multi-step)",
                              d2["step9__coil_centroids"], d2["step9__B_coil"],
                              d2["step9__J_coil"], float(d1["delta_SC"]), params.w, ic)
        rows.append(r_fast)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(cfg.FIG_BG)

    labels = [r["label"] for r in rows]
    margins = [r["min_margin"] for r in rows]
    colors = [cfg.SERIES_COLORS[k] for k in range(len(rows))]

    bars = ax.bar(labels, margins, color=colors, edgecolor="white",
                 linewidth=0.6, width=0.55)
    for b, m, r in zip(bars, margins, rows):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.04, f"{m:.3f}",
               ha="center", va="bottom", color="white", fontsize=11)
        pct = 100 * r["n_below_1"] / r["n_cells"]
        if r["n_below_1"] > 0:
            ax.text(b.get_x() + b.get_width() / 2, m / 2,
                   f"{pct:.0f}% of cells\nbelow Ic", ha="center", va="center",
                   color="#111", fontsize=8.5, fontweight="bold")

    ax.axhline(1.0, color="#e57373", ls="--", lw=1.6,
              label="margin = 1.0 (E$_c$-defined Ic)")
    ax.axhline(1.0 / 0.65, color="#90caf9", ls=":", lw=1.6,
              label="1.538 (champion's steady-state design threshold)")

    _ax(ax, "", "worst-cell margin  =  Jc(B,$\\theta$) / |J|",
        "Worst-cell local quench margin: uniform-J assumption vs. T-A "
        "(screening-current-resolved)\nsame I_design=196A operating point, "
        "at two different ramp speeds")
    ax.legend(fontsize=8, labelcolor="white", facecolor="#222",
             edgecolor="#444", framealpha=0.7, loc="upper right")
    ax.set_ylim(0, max(margins) * 1.3)
    ax.tick_params(axis="x", labelsize=8.5)

    fig.text(0.5, -0.02,
             "Interpretation caveat (deliberately not resolved by this chart): "
             "some cells at/above the E$_c$-defined Ic near a flux-penetration "
             "front is normal Bean/critical-state behaviour, not necessarily "
             "catastrophic quench -- see CLAUDE.md's \"Ramp-up power analysis\".",
             ha="center", color="#999", fontsize=7.5, wrap=True)

    fig.tight_layout()
    save_report(fig, "04_quench_margin_comparison.png")
    print("=" * 78)


if __name__ == "__main__":
    main()
