"""
make_penalty_evolution.py
============================
Poster figure: how the CMA-ES search's combined fitness shrinks over the
full_config_search run.  fitness = tape_km + constraint penalties, so a
passing design has fitness = tape_km.

Inputs: optimize/runs/full_config_search/history.csv
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

CSV_PATH = os.path.join(_ROOT, "optimize", "runs",
                        "full_config_search", "history.csv")
ROLL_WINDOW = 10
LINTHRESH = 1.0
BLACK = "#111111"
POINT_COLOR = "#c0392b"
TREND_COLOR = "#1f6feb"
PASS_COLOR  = "#1e7d34"


def _load():
    df = pd.read_csv(CSV_PATH)
    # Keep all evaluated rows (skip rejected_geometry with no tape_km)
    df = df[df["status"] == "ok"].reset_index(drop=True)
    evals   = df["eval"].values
    fitness = df["fitness"].values
    passed  = df["all_constraints_ok"].astype(bool).values
    return evals, fitness, passed


def _rolling_min(y, window):
    """Rolling minimum — shows the best-so-far trend."""
    out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = y[max(0, i - window + 1):i + 1].min()
    return out


def main():
    evals, fitness, passed = _load()
    best_so_far = np.minimum.accumulate(fitness)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Failed evals
    fail_mask = ~passed
    ax.scatter(evals[fail_mask], fitness[fail_mask], s=10,
               color=POINT_COLOR, alpha=0.35, linewidths=0,
               label="Evaluated (constraint violation)")
    # Passing evals
    pass_mask = passed
    ax.scatter(evals[pass_mask], fitness[pass_mask], s=30,
               color=PASS_COLOR, alpha=0.8, linewidths=0.5,
               edgecolors="black", zorder=4,
               label=f"Passing (all constraints OK, {pass_mask.sum()} designs)")
    # Best-so-far line
    ax.plot(evals, best_so_far, color=TREND_COLOR, lw=2.2,
            label="Best fitness so far")

    ax.set_yscale("symlog", linthresh=LINTHRESH)
    ax.set_xlabel("Evaluation index", fontsize=12)
    ax.set_ylabel("Fitness  (≈ tape km when passing)", fontsize=12)
    ax.set_xlim(evals.min(), evals.max())
    ax.tick_params(colors=BLACK)
    for sp in ax.spines.values():
        sp.set_edgecolor("#888")
    ax.grid(True, which="both", axis="y", color="#ddd", lw=0.6)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9,
              edgecolor="#ccc")
    ax.set_title("full_config_search: fitness convergence", fontsize=13)

    fig.tight_layout(pad=0.4)
    out = os.path.join(_HERE, "penalty_evolution.png")
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.03,
                facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}  ({len(evals)} evaluations, {pass_mask.sum()} passing)")


if __name__ == "__main__":
    main()
