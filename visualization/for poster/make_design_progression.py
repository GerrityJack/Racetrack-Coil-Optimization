"""
make_design_progression.py
============================
Poster figure: B_target vs tape length for the full_config_search run,
coloured by layer count.  Background = all 77 evaluated (status=ok)
designs from optimize/runs/full_config_search/history.csv.
Foreground = the three key milestones hardcoded with validated numbers:
  1. Legacy 6-layer champion  (ramp+hold, 10.38 T, 0.337 km)
  2. 16-layer starting design (end-of-ramp, 7.11 T, 1.960 km)
  3. 24-layer winner eval 92  (end-of-ramp, 10.24 T, 1.828 km)

Output: visualization/for poster/design_progression.png
"""
import csv
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

_HERE = os.path.dirname(os.path.abspath(__file__))
_VIZ  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_VIZ)
sys.path.insert(0, _ROOT)
import params

HISTORY_CSV = os.path.join(_ROOT, "optimize", "runs",
                            "full_config_search", "history.csv")
TARGET_B = 10.0

# label, tape_km, B_target_T, n_layers, is_champion
MILESTONES = [
    ("Legacy 6-layer champion\n(1h ramp + hold, 10.38 T)\n[382,382,478,478,3,3]",
     0.3372, 10.38, 6, False),
    ("16-layer start\n(end-of-2h-ramp, 7.11 T)",
     1.960, 7.11, 16, False),
    ("24-layer winner\n(eval 92, end-of-ramp)\n10.24 T, 0.656% unif",
     1.8278, 10.235, 24, True),
]


def _load_background():
    tapes, Bs, layers = [], [], []
    with open(HISTORY_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status", "").strip() != "ok":
                continue
            try:
                t = float(r["tape_km"])
                b = float(r["B_target_T"])
                nl = int(r["n_layers"])
            except (TypeError, ValueError, KeyError):
                continue
            tapes.append(t); Bs.append(b); layers.append(nl)
    return np.array(tapes), np.array(Bs), np.array(layers)


def main():
    bg_tape, bg_B, bg_layers = _load_background()

    # Color map: discrete colors per layer count
    unique_layers = sorted(set(bg_layers))
    cmap = plt.colormaps["tab10"].resampled(len(unique_layers))
    layer_color = {nl: cmap(i) for i, nl in enumerate(unique_layers)}

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#888")

    # Background scatter, coloured by layer count
    for nl in unique_layers:
        mask = bg_layers == nl
        ax.scatter(bg_tape[mask], bg_B[mask], s=28, color=layer_color[nl],
                   alpha=0.55, linewidths=0.4, edgecolors="white", zorder=2,
                   label=f"{nl}-layer ({mask.sum()} evals)")

    # 10T target line
    ax.axhline(TARGET_B, color="#c0392b", lw=1.5, ls="--", alpha=0.8, zorder=3)
    ax.text(0.02, TARGET_B + 0.05, "10 T target", color="#c0392b",
            fontsize=10, va="bottom", transform=ax.get_yaxis_transform())

    # Milestone markers
    milestone_colors = ["#7a7a7a", "#5a7fb5", "#1e7d34"]
    for (label, tape, B, nl, is_champ), col in zip(MILESTONES, milestone_colors):
        ax.scatter([tape], [B], s=500 if is_champ else 300,
                   marker="*" if is_champ else "D",
                   color=col, edgecolor="black", linewidths=1.2, zorder=6)
        short = label.split("\n")[0]
        ax.annotate(short, (tape, B), xytext=(8, 5),
                    textcoords="offset points", fontsize=9, color=col,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=col, alpha=0.85))

    ax.set_xlabel("Tape length  (km)", fontsize=13)
    ax.set_ylabel("B_target  (T, end of 2 h constant-power ramp)", fontsize=13)
    ax.set_title(
        "Full-config search: B_target vs tape by layer count\n"
        "(77 T-A-evaluated designs; milestones = validated results)",
        fontsize=14)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.15)

    out = os.path.join(_HERE, "design_progression.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}  ({len(bg_tape)} background evals)")


if __name__ == "__main__":
    main()
