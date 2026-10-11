"""
make_pca_3d_trajectory.py
===========================
Poster figure: 3D scatter of the full_config_search designs in
(a, b, coil_half_gap) space, coloured by B_target, with passing designs
larger and layer count indicated by marker shape.

Since different islands have different n_layers, PCA on the full variable
vector is not well-defined across islands.  Using the three continuous
geometric variables (a, b, gap) directly gives an interpretable 3D view.

Inputs: optimize/runs/full_config_search/history.csv
Output: visualization/for poster/pca_3d_trajectory.png
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))
_VIZ = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_VIZ)

CSV_PATH = os.path.join(_ROOT, "optimize", "runs",
                        "full_config_search", "history.csv")
CMAP = "plasma"
BLACK = "#111111"

# Layer-count → marker shape for visual distinction
LAYER_MARKERS = {12: "o", 16: "s", 20: "^", 24: "D", 28: "P", 32: "X"}


def main():
    df = pd.read_csv(CSV_PATH)
    df = df[df["status"] == "ok"].reset_index(drop=True)

    a   = df["a_mm"].values
    b   = df["b_mm"].values
    gap = df["gap_mm"].values
    B   = df["B_target_T"].values
    layers = df["n_layers"].values
    passed = df["all_constraints_ok"].astype(bool).values

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    vmin, vmax = B.min(), B.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm   = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])

    unique_layers = sorted(set(layers))
    for nl in unique_layers:
        mk = LAYER_MARKERS.get(nl, "o")
        mask_fail = (layers == nl) & ~passed
        mask_pass = (layers == nl) & passed
        if mask_fail.any():
            ax.scatter(a[mask_fail], b[mask_fail], gap[mask_fail],
                       c=B[mask_fail], cmap=CMAP, vmin=vmin, vmax=vmax,
                       s=20, alpha=0.4, marker=mk, linewidths=0,
                       depthshade=True, label=f"{nl}L")
        if mask_pass.any():
            ax.scatter(a[mask_pass], b[mask_pass], gap[mask_pass],
                       c=B[mask_pass], cmap=CMAP, vmin=vmin, vmax=vmax,
                       s=80, alpha=0.9, marker=mk, linewidths=0.7,
                       edgecolors="black", depthshade=True,
                       label=f"{nl}L (pass)" if not mask_fail.any() else "")

    # Mark winner
    win = df[df["eval"] == 92]
    if not win.empty:
        ax.scatter(win["a_mm"], win["b_mm"], win["gap_mm"],
                   c=win["B_target_T"], cmap=CMAP, vmin=vmin, vmax=vmax,
                   s=300, marker="*", edgecolors="black", linewidth=1.5,
                   depthshade=False, zorder=10, label="Winner (eval 92)")

    cbar = fig.colorbar(sm, ax=ax, pad=0.12, fraction=0.03, shrink=0.6)
    cbar.set_label("B_target  (T)", fontsize=11)

    ax.set_xlabel("a  (mm)", fontsize=11, labelpad=8)
    ax.set_ylabel("b  (mm)", fontsize=11, labelpad=8)
    ax.set_zlabel("coil gap  (mm)", fontsize=11, labelpad=8)
    ax.tick_params(colors=BLACK, labelsize=8)
    ax.view_init(elev=18, azim=-55)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85,
              edgecolor="#ccc", ncol=2)
    ax.set_title("Geometric search space: (a, b, gap) coloured by B_target",
                 fontsize=12)

    fig.tight_layout(pad=0.4)
    out = os.path.join(_HERE, "pca_3d_trajectory.png")
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}  ({len(df)} evaluated designs, "
          f"{passed.sum()} passing)")


if __name__ == "__main__":
    main()
