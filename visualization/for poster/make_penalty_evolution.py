"""
make_penalty_evolution.py
============================
Poster figure: how the CMA-ES search's constraint-violation penalty
shrinks to zero over the course of a run.

optimize/cmaes_search.py's fitness is `f = tape_km + penalty`, where
penalty sums the squared, weighted constraint violations (field, hoop,
uniformity -- each 0 when satisfied). optimize/runs/
cmaes_all_evaluations.csv only logs the combined fitness and tape_km
directly, so penalty = fitness - tape_km recovers it exactly (an
infeasible-geometry row has no tape_km; treated as 0 tape_km, so its
fitness IS its penalty). Same RUN_TAG as make_pca_convergence.py /
make_pca_3d_trajectory.py for a consistent, cross-referenceable set of
poster figures describing the same run.

The penalty spans ~7000 down to ~0 over the run and legitimately hits
exactly 0 for many converged-feasible evaluations, so it's plotted on a
symlog y-axis (linear near 0, log above a small threshold) rather than a
plain log axis, which can't represent 0 at all.

Inputs: optimize/runs/cmaes_all_evaluations.csv
Output: visualization/for poster/penalty_evolution.png
"""
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
ROLL_WINDOW = 45   # same window as make_pca_convergence.py
LINTHRESH = 1.0    # symlog linear-region half-width
BLACK = "#111111"
POINT_COLOR = "#c0392b"
TREND_COLOR = "#1f6feb"


def _load():
    df = pd.read_csv(CSV_PATH)
    sub = df[df["run_tag"] == RUN_TAG].reset_index(drop=True)
    if sub.empty:
        raise SystemExit(f"No rows for run_tag={RUN_TAG!r} in {CSV_PATH}")
    penalty = (sub["fitness"] - sub["tape_km"].fillna(0.0)).clip(lower=0.0)
    return sub["eval"].values, penalty.values


def _rolling_mean(y, window):
    if window < 2:
        return y
    kernel = np.ones(window) / window
    pad = window // 2
    y_pad = np.pad(y, (pad, window - 1 - pad), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def main():
    evals, penalty = _load()
    trend = _rolling_mean(penalty, ROLL_WINDOW)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.scatter(evals, penalty, s=6, color=POINT_COLOR, alpha=0.22,
               linewidths=0, label="Per-evaluation penalty")
    ax.plot(evals, trend, color=TREND_COLOR, lw=2.2,
            label="Rolling mean")

    ax.set_yscale("symlog", linthresh=LINTHRESH)
    ax.set_xlabel("Evaluation", fontsize=12)
    ax.set_ylabel("Constraint-violation penalty", fontsize=12)
    ax.set_xlim(evals.min(), evals.max())
    ax.set_ylim(0, penalty.max() * 1.15)
    ax.tick_params(colors=BLACK)
    for sp in ax.spines.values():
        sp.set_edgecolor("#888")
    ax.grid(True, which="both", axis="y", color="#ddd", lw=0.6)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9,
              edgecolor="#ccc")

    fig.tight_layout(pad=0.4)
    out = os.path.join(_HERE, "penalty_evolution.png")
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.03,
                facecolor="white")
    plt.close(fig)
    n_zero = int((penalty == 0).sum())
    print(f"Wrote {out}  ({len(evals)} evaluations, run_tag={RUN_TAG})")
    print(f"  penalty range: [{penalty.min():.4g}, {penalty.max():.4g}], "
          f"{n_zero} evals ({n_zero/len(evals)*100:.1f}%) at exactly 0")


if __name__ == "__main__":
    main()
