"""plot_dt_sweep_summary.py -- 2026-08-06, plots for the dt-boundary
sweep run (dt_boundary_sweep.py's output). Dark theme, matching
plot_ramp_diagnostics.py / solve/ta_sweep.py's style. Output:
transient/full_validation_plots/.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRANS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_TRANS)
for _p in (_TRANS, _ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve"),
           os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(_TRANS, "full_validation_plots")


def _dark_fig(figsize):
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("#111")
    return fig


def _dark_ax(ax):
    ax.set_facecolor("#0d0d1a")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    return ax


def main():
    npz_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _TRANS, "full_validation_plots", "data", "dt_boundary_sweep.npz")
    d = np.load(npz_path, allow_pickle=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    dt_points = d["dt_points"]
    final_scif, final_T_max, final_T_min, final_dB_rel, finite_flags = [], [], [], [], []
    for dt in dt_points:
        key = f"dt{dt:.0f}"
        final_scif.append(d[f"{key}__trace_scif_mT"][-1])
        final_T_max.append(d[f"{key}__trace_T_max_amp"][-1])
        final_T_min.append(d[f"{key}__trace_T_min_amp"][-1])
        final_dB_rel.append(d[f"{key}__trace_dB_rel"][-1])
        finite_flags.append(bool(d[f"{key}__finite"]))

    genuine = [(tmax < 2.0 and tmin > -100 and abs(dbr) < 0.15)
              for tmax, tmin, dbr in zip(final_T_max, final_T_min, final_dB_rel)]

    fig = _dark_fig((11, 9))
    ax1 = _dark_ax(fig.add_subplot(311))
    ax2 = _dark_ax(fig.add_subplot(312, sharex=ax1))
    ax3 = _dark_ax(fig.add_subplot(313, sharex=ax1))

    colors = ["lime" if g else "red" for g in genuine]
    ax1.scatter(dt_points, final_T_max, c=colors, s=60, zorder=3)
    ax1.plot(dt_points, final_T_max, color="#666", lw=1.0, zorder=2)
    ax1.axhline(1.0, color="#888", ls=":", lw=1.0, label="T_amp (ideal)")
    ax1.set_ylabel("Worst $T_{max}/T_{amp}$")
    ax1.set_title("dt-boundary sweep: alpha=(0.03,0.01), I=19.6A, forced full-length\n"
                 "green = genuine convergence, red = still chaotic")
    ax1.legend(fontsize=8, labelcolor="white", facecolor="#222", framealpha=0.6)
    ax1.grid(True, alpha=0.25, color="#555")

    ax2.scatter(dt_points, final_dB_rel, c=colors, s=60, zorder=3)
    ax2.plot(dt_points, final_dB_rel, color="#666", lw=1.0, zorder=2)
    ax2.set_ylabel("Final $|\\Delta B|/|B|$")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.25, color="#555", which="both")

    ax3.scatter(dt_points, final_scif, c=colors, s=60, zorder=3)
    ax3.plot(dt_points, final_scif, color="#666", lw=1.0, zorder=2)
    ax3.set_xlabel("dt  [s]")
    ax3.set_ylabel("Final SCIF  [mT]\n(not physically meaningful\nwhere not converged)")
    ax3.invert_xaxis()
    ax3.grid(True, alpha=0.25, color="#555")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "dt_boundary_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Wrote {out}")

    # per-dt trend overlay (T_max/amp vs iteration, one line per dt)
    fig2 = _dark_fig((11, 6))
    ax = _dark_ax(fig2.add_subplot(111))
    cmap = matplotlib.colormaps.get_cmap("viridis")
    for i, dt in enumerate(dt_points):
        key = f"dt{dt:.0f}"
        trace = d[f"{key}__trace_T_max_amp"]
        color = cmap(i / max(1, len(dt_points) - 1))
        ax.plot(np.arange(len(trace)), trace, color=color, lw=1.0,
               label=f"dt={dt:.0f}s")
    ax.axhline(1.0, color="lime", lw=0.8, ls=":")
    ax.set_xlabel("Picard iteration")
    ax.set_ylabel("Worst $T_{max}/T_{amp}$")
    ax.set_title("T overshoot trend by dt -- shows the transition into chaos directly")
    ax.legend(fontsize=8, labelcolor="white", facecolor="#222", framealpha=0.6,
             ncol=2)
    ax.grid(True, alpha=0.25, color="#555")
    fig2.tight_layout()
    out2 = os.path.join(OUT_DIR, "dt_boundary_T_trend_overlay.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"  Wrote {out2}")

    print("Done.")


if __name__ == "__main__":
    main()
