"""
plot_optimization_poster.py
=============================
Poster figure for the optimization section: tape length vs. field
uniformity, showing BOTH the true scale of the search (every feasible
evaluation across the whole project's cumulative CMA-ES history, ~88k
points, plotted faint/small as backdrop) AND the handful of designs that
were actually independently validated with the real per-layer T-A solve
(bold, labeled, on top). The champion is the only validated design that
passes the 1% target.

2026-07-27, second version: the first cut only plotted the 7 validated
points, which correctly told the validation story but visually looked
like the whole search was just 7 designs -- this version adds the dense
background cloud specifically to show the real scale of exploration
(~88,000 feasible evaluations) while keeping the honesty of the first
version intact: the background cloud is explicitly labeled as the coarse
screen (NOT independently verified -- see CLAUDE.md's "Coarse-screen SCIF
proxy found unreliable" section for why its own uniformity_pct axis can't
be trusted on its own), visually de-emphasized (small, faint, gray) so it
reads as context/scale, not as a claim about which designs are good.

Foreground data hardcoded from optimize/day_search_report.md's Phase B
table (the only points in this whole figure with a real, trustworthy
uniformity value) -- fixed, already-concluded results for the poster, not
live-updating.

Output: visualization/optimization_summary_poster.png
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import params
import opt_config as cfg

TARGET_PCT = 1.0

# label, n_layers, tape_km, box_ptp_pct (mean of the two Phase B repeats),
# is_champion
CANDIDATES = [
    ("6 layers\n(champion)", 6, 0.2259, (0.828 + 0.828) / 2, True),
    ("6 layers", 6, 0.1781, (4.462 + 4.487) / 2, False),
    ("8 layers", 8, 0.1877, (3.409 + 3.410) / 2, False),
    ("10 layers", 10, 0.2121, (3.475 + 3.471) / 2, False),
    ("12 layers", 12, 0.2315, (4.096 + 4.097) / 2, False),
    ("14 layers", 14, 0.2254, (3.091 + 3.097) / 2, False),
    ("16 layers", 16, 0.2214, (3.845 + 3.845) / 2, False),
]

ORANGE = "#e8792a"
GRAY = "#8a8a8a"
BG_GRAY = "#b7b7b7"
PASS_GREEN = "#3a9d5c"
FAIL_RED = "#c0392b"

X_LO, X_HI = 0.015, 3.0
Y_LO, Y_HI = 0.05, 30.0


def _load_background():
    path = os.path.join(_ROOT, cfg.CMAES_MASTER_LOG)
    tapes, unifs = [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("feasible", "")).strip().lower() != "true":
                continue
            try:
                t = float(r["tape_km"]); u = float(r["uniformity_pct"])
            except (TypeError, ValueError):
                continue
            if X_LO <= t <= X_HI and Y_LO <= u <= Y_HI:
                tapes.append(t); unifs.append(u)
    return np.array(tapes), np.array(unifs)


def main():
    bg_tape, bg_unif = _load_background()

    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#888")
    ax.tick_params(colors="black", labelsize=12)

    # PASS / FAIL background shading at the 1% target line
    ax.axhspan(Y_LO, TARGET_PCT, color=PASS_GREEN, alpha=0.07, zorder=0)
    ax.axhspan(TARGET_PCT, Y_HI, color=FAIL_RED, alpha=0.05, zorder=0)
    ax.axhline(TARGET_PCT, color="#444", ls="--", lw=1.6, zorder=2,
              label=f"{TARGET_PCT:.0f}% uniformity target")

    ax.scatter(bg_tape, bg_unif, s=2.5, color=BG_GRAY, alpha=0.18,
              linewidths=0, zorder=1, rasterized=True,
              label=f"coarse screen, every evaluation "
                    f"(n={len(bg_tape):,}) -- NOT independently verified")

    for label, n_layers, tape, unif, is_champ in CANDIDATES:
        if is_champ:
            ax.scatter([tape], [unif], marker="*", s=800, color=ORANGE,
                      edgecolor="black", linewidth=1.3, zorder=5)
            ax.annotate(label, (tape, unif), xytext=(10, 16),
                       textcoords="offset points", fontsize=13,
                       fontweight="bold", color="black", ha="left")
        else:
            ax.scatter([tape], [unif], marker="o", s=170, color=GRAY,
                      edgecolor="black", linewidth=1.0, zorder=4, alpha=0.95)
            ax.annotate(label, (tape, unif), xytext=(9, -4),
                       textcoords="offset points", fontsize=10.5,
                       color="#333", ha="left")
    # dummy handle for the legend (the 7 real points use 2 different
    # markers above, but should read as one "validated" category)
    ax.scatter([], [], marker="o", s=170, color=GRAY, edgecolor="black",
              linewidth=1.0, label="T-A validated (real per-layer physics)")
    ax.scatter([], [], marker="*", s=500, color=ORANGE, edgecolor="black",
              linewidth=1.3, label="T-A validated -- champion (only PASS)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xlabel("Tape length  (km, log scale)", fontsize=15, color="black")
    ax.set_ylabel("Peak-to-peak field uniformity  (%, log scale)",
                 fontsize=15, color="black")
    ax.legend(fontsize=10.5, loc="lower right", frameon=True,
             facecolor="white", edgecolor="#888", labelcolor="black")

    fig.tight_layout()
    out = os.path.join(params.VIZ_DIR, "optimization_summary_poster.png")
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}  ({len(bg_tape)} background points)")


if __name__ == "__main__":
    main()
