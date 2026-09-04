"""
make_design_progression.py
============================
Poster figure: tape length vs. real (T-A validated) box uniformity,
showing the true scale of the CMA-ES search (the whole project's
cumulative cmaes_all_evaluations.csv, ~100k rows, plotted faint/small as
a coarse-screen backdrop -- NOT independently verified, see CLAUDE.md's
"Coarse-screen SCIF proxy found unreliable") behind the actual n_layers=6
design LINEAGE that led to the current champion, each point independently
T-A-validated:

  1. 2026-07-24 champion            0.2258 km  [285,285,379,379,2,2]
  2. 2026-07-30 turn-split refined  0.2235 km  [295,295,369,369,2,2]
  3. 2026-07-31 Kim Ic correction   0.2596 km  [329,329,411,411,2,2]
     -- reached 10.03T nominal but FAILED build tolerance (0/14 jitter
     builds reached 10T): converged exactly onto the B>=10T floor with
     no margin.
  4. 2026-07-31 margin-aware CHAMPION 0.3372 km [382,382,478,478,3,3]
     -- 15/15 jitter builds pass.

This is a more direct "how we actually got here" story than a same-day
n_layers comparison sweep would be -- reuses the existing
optimization_summary_poster.py's background-cloud concept but swaps its
frozen n_layers-comparison foreground for the real chronological lineage
of the champion itself. Values are hardcoded from CLAUDE.md/params.py
history (see the module docstring above) -- not live-recomputed.

Output: visualization/for poster/design_progression.png
"""
import csv
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_VIZ  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_VIZ)
for _p in (_ROOT, os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import params
import opt_config as cfg

TARGET_PCT = 1.0

# label, tape_km, box_ptp_pct (T-A validated), B_target_T, jitter_result
LINEAGE = [
    ("2026-07-24\nn_layers=6 champion\n[285,285,379,379,2,2]",
     0.2258, 0.828, None, None),
    ("2026-07-30\nturn-split refined\n[295,295,369,369,2,2]",
     0.2235, 0.687, 10.215, None),
    ("2026-07-31\nKim Ic model applied\n[329,329,411,411,2,2]",
     0.2596, 0.442, 10.03, "FAILED build tol.\n0/14 jitter builds"),
    ("CHAMPION (current)\nmargin-aware\n[382,382,478,478,3,3]",
     0.3372, 0.495, 10.49, "PASS\n15/15 jitter builds"),
]

X_LO, X_HI = 0.15, 3.0
Y_LO, Y_HI = 0.1, 30.0


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

    fig, ax = plt.subplots(figsize=(12, 8.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#888")

    ax.scatter(bg_tape, bg_unif, s=4, c="#c8c8c8", alpha=0.35,
               linewidths=0, zorder=1,
               label=f"coarse-screen search history\n"
                     f"({len(bg_tape):,} feasible evals, NOT independently "
                     f"verified)")

    xs = [row[1] for row in LINEAGE]
    ys = [row[2] for row in LINEAGE]
    ax.plot(xs, ys, "-", color="#888", lw=1.6, zorder=2)

    # The four points sit within ~2x of each other in both axes -- too
    # close for inline text boxes without collisions. Use numbered
    # markers instead and put the details in a separate legend panel.
    colors = ["#7a7a7a", "#5a7fb5", "#c0392b", "#1e7d34"]
    markers = ["1", "2", "3", "4"]
    for (label, tape, unif, B, jitter), col, num in zip(LINEAGE, colors, markers):
        is_champion = "CHAMPION" in label
        size = 480 if is_champion else 340
        ax.scatter([tape], [unif], s=size,
                   marker="*" if is_champion else "o",
                   color=col, edgecolor="black", linewidths=1.2, zorder=5)
        ax.annotate(("★" if is_champion else num), (tape, unif),
                    xytext=(0, 0), textcoords="offset points",
                    fontsize=9 if not is_champion else 7, color="white",
                    ha="center", va="center", fontweight="bold", zorder=6)

    ax.axhline(TARGET_PCT, color="#c0392b", lw=1.4, ls="--", alpha=0.8,
               zorder=3)
    ax.text(X_HI * 0.97, TARGET_PCT * 1.06, "1% uniformity target",
            color="#c0392b", fontsize=10, ha="right", va="bottom")

    # Legend panel with the full details, placed in the sparse
    # low-uniformity / high-tape corner so it doesn't sit on the cloud.
    lines = []
    for (label, tape, unif, B, jitter), num in zip(LINEAGE, ["1", "2", "3", "★"]):
        short = label.split("\n")[0]
        detail = f"{num}  {short} -- {tape:.4f} km, {unif:.3f}%"
        if B is not None:
            detail += f", B={B:.2f}T"
        lines.append(detail)
        if jitter is not None:
            lines.append("     " + jitter.replace("\n", " "))
    legend_txt = "\n".join(lines)
    ax.text(0.985, 0.03, legend_txt, transform=ax.transAxes,
            fontsize=9.3, ha="right", va="bottom", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#888", alpha=0.95))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(X_LO, X_HI); ax.set_ylim(Y_LO, Y_HI)
    ax.set_xlabel("Tape length  (km)", fontsize=13)
    ax.set_ylabel("Box peak-to-peak uniformity  (%)  --  T-A validated where labeled",
                  fontsize=13)
    ax.set_title(
        "From coarse uniform-J screen to the validated T-A champion\n"
        "(n_layers=6 lineage; background = full CMA-ES search history, scale only)",
        fontsize=14)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.15)

    out = os.path.join(_HERE, "design_progression.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
