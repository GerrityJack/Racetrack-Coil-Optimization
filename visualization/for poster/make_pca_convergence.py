"""
make_pca_convergence.py
=========================
Poster figure: convergence of the multi-layer CMA-ES search in
full_config_search, visualised as the three geometric variables
(a, b, coil_half_gap) converging over evaluation index.

Since different islands have different n_layers (and thus different-length
turn vectors), PCA on the full variable vector is not meaningful across
islands.  Instead: show the three continuous geometric variables directly
(not PCA) as parallel traces -- each starts scattered and narrows toward
the winning design.  Colour by layer count.

Inputs: optimize/runs/full_config_search/history.csv
Output: visualization/for poster/pca_convergence.png
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

_HERE = os.path.dirname(os.path.abspath(__file__))
_VIZ = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_VIZ)

CSV_PATH = os.path.join(_ROOT, "optimize", "runs",
                        "full_config_search", "history.csv")
COLORS = ["#1f6feb", "#c0392b", "#2e7d32"]
BLACK = "#111111"
VAR_LABELS = ["a  (mm)", "b  (mm)", "coil gap  (mm)"]


def _load():
    df = pd.read_csv(CSV_PATH)
    df = df[df["status"] == "ok"].reset_index(drop=True)
    return df


def main():
    df = _load()
    evals   = df["eval"].values
    a_mm    = df["a_mm"].values
    b_mm    = df["b_mm"].values
    gap_mm  = df["gap_mm"].values
    layers  = df["n_layers"].values
    passed  = df["all_constraints_ok"].astype(bool).values
    B       = df["B_target_T"].values

    unique_layers = sorted(df["n_layers"].unique())
    layer_cmap = plt.colormaps["tab10"].resampled(len(unique_layers))
    lcolor = {nl: layer_cmap(i) for i, nl in enumerate(unique_layers)}

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    fig.patch.set_facecolor("white")

    for ax, vals, label in zip(axes, [a_mm, b_mm, gap_mm], VAR_LABELS):
        ax.set_facecolor("white")
        for nl in unique_layers:
            mask = layers == nl
            ax.scatter(evals[mask & ~passed], vals[mask & ~passed],
                       s=14, color=lcolor[nl], alpha=0.45, linewidths=0)
            ax.scatter(evals[mask & passed], vals[mask & passed],
                       s=40, color=lcolor[nl], alpha=0.9, linewidths=0.6,
                       edgecolors="black", zorder=5,
                       label=f"{nl}L" if label == VAR_LABELS[0] else "")
        # Mark winner (eval 92)
        win_mask = df["eval"].values == 92
        if win_mask.any():
            win_val = vals[df["eval"].values == 92][0]
            win_eval = 92
            ax.scatter([win_eval], [win_val], s=200, marker="*",
                       color="#1e7d34", edgecolor="black", linewidth=1,
                       zorder=8, label="Winner (eval 92)" if label == VAR_LABELS[0] else "")
        ax.set_ylabel(label, fontsize=11)
        ax.tick_params(colors=BLACK, labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor("#888")
        ax.grid(True, alpha=0.15)

    axes[0].legend(fontsize=8, loc="upper right", ncol=4,
                   framealpha=0.9, edgecolor="#ccc",
                   title="Layer count (filled = passing)")
    axes[2].set_xlabel("Evaluation index", fontsize=12)
    fig.suptitle("Geometric variables over the full_config_search run",
                 fontsize=13, y=1.01)

    fig.tight_layout(pad=0.4)
    out = os.path.join(_HERE, "pca_convergence.png")
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.03,
                facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}  ({len(evals)} evaluated designs)")


if __name__ == "__main__":
    main()
