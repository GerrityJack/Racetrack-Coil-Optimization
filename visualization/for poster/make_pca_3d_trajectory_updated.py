"""
make_pca_3d_trajectory_updated.py
===================================
Poster figure (updated design, 2026-10-04): make_pca_3d_trajectory.py's
figure for the layer-count search in optimize/runs/updated_10_4/ --
the top 3 principal-component scores as (x, y, z), each evaluation's
point connected to the next in evaluation order, coloured along a
colorbar by evaluation index so the search's path through this reduced
3-mode space (scatter early -> converge onto a point late) is visible
directly, instead of as three separate 1D traces over time.

Difference from the original: this search runs one CMA-ES island per
layer count (12-32 layers), so designs have different numbers of turn
variables and the original (a, b, gap, turns) vector is not one fixed
length. Every design is put into the same 10-dimensional space instead:
a, b, gap, n_layers, and its per-pair turn profile resampled (linear
interpolation) onto 6 fixed positions across the stack. Evaluations are
in evaluation order, which interleaves the islands, so the smoothed path
is the population's mean path.

Inputs: optimize/runs/updated_10_4/history.csv
Output: visualization/for poster/pca_3d_trajectory_updated.png
"""
import ast
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

_HERE = os.path.dirname(os.path.abspath(__file__))
_VIZ = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_VIZ)

CSV_PATH = os.path.join(_ROOT, "optimize", "runs", "updated_10_4", "history.csv")
N_PC = 3
N_PROFILE = 6      # turn-profile resampling points (the original had 6 turn vars)
ROLL_WINDOW = 12   # ~4% of 286 evaluations (the original: 45 of ~4000)
CMAP = "viridis"
BLACK = "#111111"


def _rolling_mean(y, window):
    if window < 2:
        return y
    kernel = np.ones(window) / window
    pad = window // 2
    y_pad = np.pad(y, (pad, window - 1 - pad), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def _profile(turns):
    """Per-pair turn counts resampled onto N_PROFILE fixed stack positions."""
    pairs = np.asarray(turns, float)[::2]
    src = np.linspace(0.0, 1.0, len(pairs))
    return np.interp(np.linspace(0.0, 1.0, N_PROFILE), src, pairs)


def _load_design_matrix():
    sub = pd.read_csv(CSV_PATH).sort_values("eval").reset_index(drop=True)
    sub = sub.dropna(subset=["a_mm", "b_mm", "gap_mm", "n_turns"])
    prof = np.array([_profile(ast.literal_eval(s)) for s in sub["n_turns"]])
    X = np.column_stack([sub["a_mm"].values, sub["b_mm"].values,
                          sub["gap_mm"].values, sub["n_layers"].values, prof])
    return X, sub["eval"].values


def _pca(X, n_pc):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    Xs = (X - mu) / sd
    Xs -= Xs.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    evr = (S ** 2) / (S ** 2).sum()
    scores = Xs @ Vt.T
    return scores[:, :n_pc], evr[:n_pc]


def main():
    X, evals = _load_design_matrix()
    scores, evr = _pca(X, N_PC)

    # Raw per-evaluation scores are extremely jumpy early on (CMA-ES's
    # step size starts large) -- connecting every consecutive raw point
    # produces an unreadable 3D hairball even after clipping the axis
    # range (mpl's 3D line collections aren't clipped to the view box, so
    # a jump between two otherwise-in-range points still draws as a long
    # spike straight through the plot). Smooth each PC axis with the same
    # rolling window make_pca_convergence.py's 1D trend lines use, and
    # draw THAT as the connected trajectory -- it tells the same
    # scattered-then-converged story without the raw noise obscuring it.
    smoothed = np.column_stack([_rolling_mean(scores[:, k], ROLL_WINDOW)
                                 for k in range(N_PC)])

    lo = np.percentile(smoothed, 0.5, axis=0)
    hi = np.percentile(smoothed, 99.5, axis=0)
    pad = 0.08 * (hi - lo)
    lims = np.column_stack([lo - pad, hi + pad])

    pts = smoothed
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    seg_t = 0.5 * (evals[:-1] + evals[1:])

    fig = plt.figure(figsize=(9, 8))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    lc = Line3DCollection(segs, cmap=CMAP, array=seg_t, linewidths=2.0,
                          norm=plt.Normalize(evals.min(), evals.max()))
    ax.add_collection3d(lc)

    # Faint raw-score scatter underneath the smoothed line for context --
    # shows the true scatter/density the smoothed path is summarising.
    raw_inside = np.all((scores >= lims[:, 0]) & (scores <= lims[:, 1]), axis=1)
    ax.scatter(scores[raw_inside, 0], scores[raw_inside, 1], scores[raw_inside, 2],
               c=evals[raw_inside], cmap=CMAP, s=3, alpha=0.12, linewidths=0,
               vmin=evals.min(), vmax=evals.max())

    # Mark start and end distinctly.
    ax.scatter(*pts[0], color="#2e7d32", s=70, marker="o",
               edgecolor=BLACK, linewidth=0.8, depthshade=False,
               label="First evaluation", zorder=5)
    ax.scatter(*pts[-1], color="#c0392b", s=90, marker="*",
               edgecolor=BLACK, linewidth=0.8, depthshade=False,
               label="Final evaluation", zorder=5)

    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.set_zlim(*lims[2])
    ax.set_xlabel(f"Principal Component 1  ({evr[0]*100:.0f}% var)", fontsize=15, labelpad=16)
    ax.set_ylabel(f"Principal Component 2  ({evr[1]*100:.0f}% var)", fontsize=15, labelpad=16)
    ax.set_zlabel(f"Principal Component 3  ({evr[2]*100:.0f}% var)", fontsize=15, labelpad=16)
    ax.tick_params(colors=BLACK, labelsize=12)
    ax.view_init(elev=22, azim=-60)

    cbar = fig.colorbar(lc, ax=ax, pad=0.08, fraction=0.035, shrink=0.6)
    cbar.set_label("Evaluation index", fontsize=15)
    cbar.ax.tick_params(labelsize=12)

    ax.legend(fontsize=15, markerscale=1.6, loc="upper left", framealpha=0.9, edgecolor="#ccc", borderpad=0.8, labelspacing=0.7)

    fig.tight_layout(pad=0.4)
    out = os.path.join(_HERE, "pca_3d_trajectory_updated.png")
    # the tight bbox under-measures 3D axis labels at these font sizes, so
    # list them explicitly and pad generously or the PC1 label is clipped
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.35,
                bbox_extra_artists=[ax.xaxis.label, ax.yaxis.label,
                                    ax.zaxis.label],
                facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}  ({len(evals)} evaluations)")
    print(f"  explained variance: {', '.join(f'PC{k+1}={v*100:.1f}%' for k, v in enumerate(evr))}  "
          f"(cumulative {evr.sum()*100:.1f}%)")


if __name__ == "__main__":
    main()
