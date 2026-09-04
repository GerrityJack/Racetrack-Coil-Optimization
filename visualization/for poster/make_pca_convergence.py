"""
make_pca_convergence.py
=========================
Poster figure: convergence of the CMA-ES design search, visualised via
PCA on the design-variable trajectory.

Each evaluation is a point in design-variable space (a, b, coil_half_gap,
per-layer turn counts) pulled straight from `optimize/runs/
cmaes_all_evaluations.csv` for one representative full run. Standardising
and taking the PCA of that trajectory collapses the 6-dimensional search
into its dominant modes of variation -- CMA-ES's adaptive covariance
naturally shrinks the search distribution as it converges, so the top
principal-component scores start scattered (early, exploratory
generations) and settle onto a flat plateau (late generations, the
search has converged around a single design).

RUN_TAG is a single fixed run picked for having the most evaluations
(4005) and a clean, fully-converged tail (population's a/b/gap/n_turns
all settle to a single point by the end) -- not the literal search that
produced the current champion (that came from a later, smaller
margin-aware grid scan, see CLAUDE.md's "Current design"). This figure
demonstrates the CMA-ES search's own convergence behaviour, which is the
project's primary search method throughout `optimize/`.

Inputs: optimize/runs/cmaes_all_evaluations.csv
Output: visualization/for poster/pca_convergence.png
"""
import ast
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_VIZ = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_VIZ)

CSV_PATH = os.path.join(_ROOT, "optimize", "runs", "cmaes_all_evaluations.csv")
RUN_TAG = "run_20260722_221136"
N_PC = 3
ROLL_WINDOW = 45   # ~5 generations at this run's ~9-per-generation popsize
COLORS = ["#1f6feb", "#c0392b", "#2e7d32"]
BLACK = "#111111"


def _load_design_matrix():
    df = pd.read_csv(CSV_PATH)
    sub = df[df["run_tag"] == RUN_TAG].reset_index(drop=True)
    if sub.empty:
        raise SystemExit(f"No rows for run_tag={RUN_TAG!r} in {CSV_PATH}")

    turns = np.array([ast.literal_eval(s) for s in sub["n_turns"]], dtype=float)
    X = np.column_stack([sub["a_mm"].values, sub["b_mm"].values,
                          sub["gap_mm"].values, turns])
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


def _rolling_mean(y, window):
    if window < 2:
        return y
    kernel = np.ones(window) / window
    pad = window // 2
    y_pad = np.pad(y, (pad, window - 1 - pad), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def main():
    X, evals = _load_design_matrix()
    scores, evr = _pca(X, N_PC)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for k in range(N_PC):
        y = scores[:, k]
        ax.scatter(evals, y, s=5, color=COLORS[k], alpha=0.18,
                   linewidths=0)
        ax.plot(evals, _rolling_mean(y, ROLL_WINDOW), color=COLORS[k],
                lw=2.2, label=f"Principal Component {k+1}  ({evr[k]*100:.0f}% var)")

    ax.axhline(0.0, color="#aaa", lw=0.8, ls=":")

    ax.set_xlabel("Evaluation", fontsize=12)
    ax.set_ylabel("Principal component score", fontsize=12)
    ax.set_xlim(evals.min(), evals.max())
    # A handful of first-generation outliers span a much wider range than
    # everything after the search actually starts converging -- clip to
    # the robust (0.5-99.5 percentile) range across all components so the
    # plateau (the actual convergence story) fills the plot instead of
    # being squashed into a thin band by a few early spikes.
    y_all = scores.ravel()
    y_lo, y_hi = np.percentile(y_all, [0.5, 99.5])
    y_pad = 0.08 * (y_hi - y_lo)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
    ax.tick_params(colors=BLACK)
    for sp in ax.spines.values():
        sp.set_edgecolor("#888")
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9,
              edgecolor="#ccc")

    fig.tight_layout(pad=0.4)
    out = os.path.join(_HERE, "pca_convergence.png")
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.03,
                facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}  ({len(evals)} evaluations, run_tag={RUN_TAG})")
    print(f"  explained variance: {', '.join(f'PC{k+1}={v*100:.1f}%' for k, v in enumerate(evr))}")
    print(f"  y-limits clipped to {ax.get_ylim()}")


if __name__ == "__main__":
    main()
