"""plot_ramp_diagnostics.py -- 2026-08-06, plots for the full 0->196A
ramp validation run (full_ramp_run.py's output). Dark theme, following
solve/ta_sweep.py's style helpers. Output: transient/full_validation_plots/.
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


def _dark_cbar(fig, mappable, ax, label):
    cb = fig.colorbar(mappable, ax=ax, pad=0.02)
    cb.set_label(label, color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    return cb


def _step_boundaries(d):
    step_of_iter = d["step_of_iter"]
    cum_iter = d["cum_iter"]
    boundaries = []
    for s in np.unique(step_of_iter):
        idx = np.where(step_of_iter == s)[0]
        boundaries.append((cum_iter[idx[0]], s))
    return boundaries


def plot_scif_trend(d):
    fig = _dark_fig((13, 5))
    ax = _dark_ax(fig.add_subplot(111))
    ax.plot(d["cum_iter"], d["scif_mT"], color="tomato", lw=1.0)
    schedule = d["schedule"]
    ymin, ymax = min(d["scif_mT"]), max(d["scif_mT"])
    for cum0, s in _step_boundaries(d):
        I_now = schedule[int(s)][1]
        ax.axvline(cum0, color="#555", lw=0.7, ls="--")
        ax.text(cum0, ymax, f"I={I_now:.0f}A", rotation=90, fontsize=7,
                color="#999", va="top", ha="right")
    ax.set_xlabel("Cumulative Picard iteration")
    ax.set_ylabel("SCIF  [mT]")
    ax.set_title("Full ramp (0→196A, dt=60s, alpha=(0.03,0.01)): "
                 "SCIF vs cumulative iteration")
    ax.grid(True, alpha=0.25, color="#555")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "ramp_scif_trend.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Wrote {out}")


def plot_T_extrema_trend(d):
    n_layers = int(d["n_layers"])
    n_turns = d["n_turns"]
    cmap = matplotlib.colormaps.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n_layers)]

    fig = _dark_fig((13, 8))
    ax1 = _dark_ax(fig.add_subplot(211))
    ax2 = _dark_ax(fig.add_subplot(212, sharex=ax1))

    for i in range(n_layers):
        lbl = f"layer{i} (n={n_turns[i]})"
        ax1.plot(d["cum_iter"], d[f"T_max_amp_layer{i}"], color=colors[i],
                 lw=1.0, label=lbl)
        ax2.plot(d["cum_iter"], d[f"T_min_amp_layer{i}"], color=colors[i],
                 lw=1.0, label=lbl)

    for cum0, s in _step_boundaries(d):
        ax1.axvline(cum0, color="#555", lw=0.7, ls="--")
        ax2.axvline(cum0, color="#555", lw=0.7, ls="--")

    ax1.axhline(1.0, color="lime", lw=0.8, ls=":")
    ax1.set_ylabel("$T_{max}$ / $T_{amp}$")
    ax1.set_title("Per-layer T extrema vs cumulative iteration "
                 "(genuine convergence signature: $T_{max}/T_{amp} \\to 1$)")
    ax1.legend(fontsize=7, labelcolor="white", facecolor="#222", framealpha=0.6,
              ncol=3, loc="upper right")
    ax1.grid(True, alpha=0.25, color="#555")

    ax2.set_xlabel("Cumulative Picard iteration")
    ax2.set_ylabel("$T_{min}$ / $T_{amp}$")
    ax2.grid(True, alpha=0.25, color="#555")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "ramp_T_extrema_trend.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Wrote {out}")


def plot_dB_rel_trend(d):
    fig = _dark_fig((13, 5))
    ax = _dark_ax(fig.add_subplot(111))
    valid = np.isfinite(d["dB_rel"]) & (d["dB_rel"] > 0)
    ax.semilogy(d["cum_iter"][valid], d["dB_rel"][valid], color="orchid", lw=0.8)
    for cum0, s in _step_boundaries(d):
        ax.axvline(cum0, color="#555", lw=0.7, ls="--")
    ax.set_xlabel("Cumulative Picard iteration")
    ax.set_ylabel("$|\\Delta B|/|B|$ per iteration")
    ax.set_title("Relative field change per iteration (log scale) -- "
                 "genuine convergence signature: stays small and flat within each step")
    ax.grid(True, alpha=0.25, color="#555", which="both")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "ramp_dB_rel_trend.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Wrote {out}")


def plot_step_summary_bars(d):
    summaries = d["step_summaries"]
    I_arr = [s["I_now"] for s in summaries]
    scif_arr = [s["scif_mT"] for s in summaries]
    n_iters_arr = [s["n_iters"] for s in summaries]

    fig = _dark_fig((10, 6))
    ax1 = _dark_ax(fig.add_subplot(211))
    ax2 = _dark_ax(fig.add_subplot(212, sharex=ax1))

    ax1.plot(I_arr, scif_arr, "o-", color="tomato", lw=1.8, ms=6)
    ax1.set_ylabel("Converged SCIF  [mT]")
    ax1.set_title("Full ramp step summary: SCIF and iteration count vs current")
    ax1.grid(True, alpha=0.25, color="#555")

    ax2.bar(I_arr, n_iters_arr, width=12, color="steelblue", alpha=0.85)
    ax2.set_xlabel("Transport current I  [A]")
    ax2.set_ylabel("Picard iterations\n(forced full-length)")
    ax2.grid(True, alpha=0.25, color="#555")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "ramp_step_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Wrote {out}")


def plot_snapshot(d, step):
    key = f"step{step}"
    I_now = float(d[f"{key}__I_now"])
    cents = d[f"{key}__coil_centroids"]
    B_coil = d[f"{key}__B_coil"]
    J_coil = d[f"{key}__J_coil"]
    Bmag = np.linalg.norm(B_coil, axis=-1)
    Jmag = np.linalg.norm(J_coil, axis=-1)

    fig = _dark_fig((11, 5))
    ax1 = _dark_ax(fig.add_subplot(121))
    sc1 = ax1.scatter(cents[:, 0] * 1e3, cents[:, 1] * 1e3, c=Bmag, cmap="magma",
                      s=4, linewidths=0)
    ax1.set_xlabel("x [mm]"); ax1.set_ylabel("y [mm]")
    ax1.set_title(f"|B| at coil cells, I={I_now:.0f}A")
    ax1.set_aspect("equal")
    _dark_cbar(fig, sc1, ax1, "|B| [T]")

    ax2 = _dark_ax(fig.add_subplot(122))
    sc2 = ax2.scatter(cents[:, 0] * 1e3, cents[:, 1] * 1e3, c=Jmag, cmap="plasma",
                      s=4, linewidths=0)
    ax2.set_xlabel("x [mm]"); ax2.set_ylabel("y [mm]")
    ax2.set_title(f"|J| (SC layer) at coil cells, I={I_now:.0f}A")
    ax2.set_aspect("equal")
    _dark_cbar(fig, sc2, ax2, "|J| [A/m$^2$]")

    fig.suptitle(f"Field snapshot, ramp step {step} (I={I_now:.0f}A)", color="white")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"ramp_snapshot_step{step}_BJ.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Wrote {out}")

    n_layers = int(d["n_layers"])
    n_turns = d["n_turns"]
    fig2 = _dark_fig((13, 8))
    for i in range(n_layers):
        ax = _dark_ax(fig2.add_subplot(2, 3, i + 1))
        coords = d[f"{key}__T_layer{i}_coords"]
        T_vals = d[f"{key}__T_layer{i}"]
        sc = ax.scatter(coords[:, 0] * 1e3, coords[:, 2] * 1e3, c=T_vals,
                        cmap="coolwarm", s=3, linewidths=0)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
        ax.set_title(f"layer{i} (n_turns={n_turns[i]})", fontsize=9)
        _dark_cbar(fig2, sc, ax, "T [A/m]")
    fig2.suptitle(f"Per-layer T field, ramp step {step} (I={I_now:.0f}A)",
                 color="white")
    fig2.tight_layout()
    out2 = os.path.join(OUT_DIR, f"ramp_snapshot_step{step}_T.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"  Wrote {out2}")


def main():
    npz_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _TRANS, "full_validation_plots", "data", "full_ramp_0to196A.npz")
    d = np.load(npz_path, allow_pickle=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    plot_scif_trend(d)
    plot_T_extrema_trend(d)
    plot_dB_rel_trend(d)
    plot_step_summary_bars(d)
    for step in d["snapshot_steps"]:
        plot_snapshot(d, int(step))

    print("Done.")


if __name__ == "__main__":
    main()
