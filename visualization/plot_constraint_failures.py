"""
plot_constraint_failures.py
=============================
Every evaluation across the whole project's cumulative CMA-ES history
(optimize/runs/cmaes_all_evaluations.csv), tape length vs. chronological
evaluation order, colored by WHICH constraint it failed.

Categories (priority order -- a row can fail more than one constraint;
it's colored by the first one below that applies):
  1. B_target < 10T
  2. hoop > 400 MPa
  3. uniformity > 0.8% -- the COARSE-SCREEN PROXY (uniformity_pct), which
     CLAUDE.md documents as unreliable by up to ~10x and no longer part
     of the actual CMA-ES fitness/constraint check (removed once found
     anti-correlated with the real T-A-validated result on this project's
     own data). Shown here for historical/diagnostic completeness (it WAS
     the enforced constraint in every run before 2026-07-24) and clearly
     labeled as such -- do not read it as "these designs are actually
     non-uniform," only "the coarse screen of the day would have rejected
     them."
  4. pass (all three satisfied, by current thresholds)
  5. infeasible geometry (failed generate/bend-radius/face-gap pre-check
     -- see optimize/cmaes_search.py's geometry_violation())

2026-07-27, second version: infeasible-geometry rows are now INCLUDED,
not excluded. tape_length_m (params.py) is a pure closed-form function of
(a, b, n_turns, t) -- it was NEVER derived from the FEM solve for ANY
row, feasible or not (only B_target/uniformity/hoop need the solve) -- so
the exact same formula used for feasible rows' tape_km is applied here
directly from each infeasible row's stored a_mm/b_mm/n_turns fields
(always populated regardless of feasibility). Not an estimate/prediction;
identical precision to every other point in this figure.

Output: visualization/constraint_failures_poster.png
"""
import ast
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import params
import opt_config as cfg

B_MIN = cfg.B_TARGET_MIN_T
HOOP_MAX = cfg.SIGMA_HOOP_MAX_PA / 1e6
# 1% -- the project's headline uniformity target (see CLAUDE.md's design
# targets), not cfg.UNIFORMITY_MAX_PCT (0.8%, a later-tightened internal
# research value covering measured mesh-realization noise on top of it)
UNIF_MAX = 1.0
_T = params.t   # tape pitch -- fixed constant across every run to date

# Okabe-Ito colorblind-safe palette. Yellow (the original
# uniformity_high color) washes out badly on printed poster paper and is
# one of the two Okabe-Ito hues deliberately NOT used here for that
# reason -- replaced with reddish purple, clearly distinct from both the
# blue (B_target_low) and vermillion (hoop_high) already in the palette.
COLORS = {
    "pass":            "#009E73",   # bluish green
    "B_target_low":    "#0072B2",   # blue
    "hoop_high":       "#D55E00",   # vermillion
    "uniformity_high": "#CC79A7",   # reddish purple (was yellow)
    "infeasible":      "#7F7F7F",   # neutral grey, darkened for contrast
}
LABELS = {
    "pass":         f"Pass (All Constraints Met)",
    "B_target_low": f"Failed: Target Field < {B_MIN:.0f} T",
    "hoop_high":    f"Failed: Hoop Stress > {HOOP_MAX:.0f} MPa",
    "uniformity_high": f"Failed: Uniformity Estimate > {UNIF_MAX:.0f}%",
    "infeasible":   "Infeasible Geometry",
}
# Draw order, back to front. "pass" is the main object of interest (where
# valid designs cluster) so it goes on TOP of everything else, not
# beneath the much larger infeasible/failed clouds as before.
ZORDER = {"infeasible": 1, "B_target_low": 2, "uniformity_high": 3,
         "hoop_high": 4, "pass": 5}

Y_CAP_M = 1000.0   # linear axis cap -- see main()'s caption for the
                   # excluded fraction; keeps the interesting low range
                   # (all real candidates sit under ~250m) legible
                   # instead of being crushed by the early-search outliers

# The true current champion is NOT a row in this log -- it came from a
# separate, later margin-aware search (margin_design_search.py /
# jitter_margin_design.py, see CLAUDE.md's "2026-07-31 (later)" entry),
# not the CMA-ES loop that produced cmaes_all_evaluations.csv (whose last
# logged run, run_20260731_103138, tops out around 0.23km). The previous
# hardcoded reference here (run_20260723_124414 eval 1759, row 67969,
# 0.2259km) was a design from BEFORE the turn-split refinement, the Kim
# Ic-model correction, AND the margin-aware redesign -- stale on three
# counts, not just one. Read live from params.py instead, which is always
# in sync with the actual champion; CHAMPION_IDX is set in main() once the
# log's real max evaluation index is known, since the champion has no
# genuine chronological position in this file.
CHAMPION_TAPE_M = params.tape_length_m   # already in metres, not km


def _predicted_tape_m(a_m, b_m, n_turns):
    """Exact same formula as params.py's tape_length_m -- pure geometry,
    no FEM involved for ANY row (see module docstring)."""
    L = b_m - a_m
    pack_thickness = max(n * _T for n in n_turns)
    a_out = a_m + pack_thickness / 2
    tape_m = sum(n * (4 * L + 2 * math.pi * (a_out - n * _T / 2))
                for n in n_turns)
    return tape_m


def _load():
    path = os.path.join(_ROOT, cfg.CMAES_MASTER_LOG)
    idx, tape, cat = [], [], []
    with open(path, newline="") as f:
        for i, r in enumerate(csv.DictReader(f), start=1):
            feas = str(r.get("feasible", "")).strip().lower() == "true"
            if not feas:
                try:
                    a_m = float(r["a_mm"]) / 1e3
                    b_m = float(r["b_mm"]) / 1e3
                    nt = ast.literal_eval(r["n_turns"])
                except (TypeError, ValueError, SyntaxError):
                    continue
                idx.append(i)
                tape.append(_predicted_tape_m(a_m, b_m, nt))
                cat.append("infeasible")
                continue
            try:
                t = float(r["tape_km"]) * 1000
                bt = float(r["B_target_T"])
                hp = float(r["hoop_MPa"])
                uf = float(r["uniformity_pct"])
            except (TypeError, ValueError):
                continue
            if bt < B_MIN:
                c = "B_target_low"
            elif hp > HOOP_MAX:
                c = "hoop_high"
            elif uf > UNIF_MAX:
                c = "uniformity_high"
            else:
                c = "pass"
            idx.append(i); tape.append(t); cat.append(c)
    return np.array(idx), np.array(tape), np.array(cat)


def main():
    idx, tape, cat = _load()
    # Placed just past the last real evaluation, not AT a real index --
    # see CHAMPION_TAPE_M's comment above for why this design has no
    # genuine position in this log's chronology.
    CHAMPION_IDX = int(idx.max()) + max(1000, int(idx.max() * 0.01))

    # Wide figure -- a 3-column legend at this font size is wider than a
    # normal (14-16in) figure, and bbox_inches='tight' at save time crops
    # to the UNION of the axes and that externally-anchored legend, which
    # otherwise leaves the actual data axes looking squeezed into a
    # narrow central column of a much wider final canvas.
    fig, ax = plt.subplots(figsize=(24, 13))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("black")
        sp.set_linewidth(2.0)
    ax.tick_params(colors="black", labelsize=26, width=2.0, length=10)

    n_above_cap = 0
    legend_handles = []
    # Plot order follows ZORDER (back to front) rather than the fixed
    # tuple used before, so "pass" (the main object of interest) is drawn
    # last/on top regardless of category size.
    for c in sorted(ZORDER, key=ZORDER.get):
        m = cat == c
        if not m.any():
            continue
        n_above_cap += int((tape[m] > Y_CAP_M).sum())
        # s and alpha both raised from the original (s=3, alpha=0.35) --
        # at 100k+ points the old settings read as faded background noise
        # rather than a real density signal.
        ax.scatter(idx[m], tape[m], s=7, color=COLORS[c], alpha=0.55,
                  linewidths=0, rasterized=True, zorder=ZORDER[c])
        legend_handles.append(Line2D([0], [0], marker="o", color="w",
                                     markerfacecolor=COLORS[c],
                                     markersize=20,
                                     label=f"{LABELS[c]}  (n={m.sum():,})"))

    ax.scatter([CHAMPION_IDX], [CHAMPION_TAPE_M], marker="*", s=2200,
              color="#E69F00", edgecolor="black", linewidth=3.0, zorder=10)
    legend_handles.append(Line2D([0], [0], marker="*", color="w",
                                 markerfacecolor="#E69F00",
                                 markeredgecolor="black", markersize=32,
                                 label=f"Current Champion "
                                       f"({CHAMPION_TAPE_M:.0f} m)"))

    # Callout arrow -- the star alone, however large, can still get lost
    # in a scatter this dense; an explicit pointer removes any doubt.
    ax.annotate("Current Champion\n(optimal design)",
               xy=(CHAMPION_IDX, CHAMPION_TAPE_M),
               xytext=(CHAMPION_IDX - 22000, CHAMPION_TAPE_M + 220),
               fontsize=22, fontweight="bold", color="black", ha="center",
               arrowprops=dict(arrowstyle="-|>", color="black", lw=2.5,
                               shrinkA=0, shrinkB=18),
               bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3D6",
                         edgecolor="black", linewidth=1.5), zorder=11)

    ax.set_ylim(0, Y_CAP_M)
    ax.set_xlabel("Evaluation Number  (Chronological, All Runs)",
                 fontsize=40, color="black")
    ax.set_ylabel("Tape Length  (m)", fontsize=40, color="black")

    # Legend moved OUTSIDE the plot (below, horizontal) instead of
    # floating over the top-right data area, with a stronger opaque
    # border and much larger markers/text.
    legend = ax.legend(handles=legend_handles, fontsize=28,
                       loc="upper center", bbox_to_anchor=(0.5, -0.13),
                       ncol=3, frameon=True, facecolor="white",
                       framealpha=0.97, edgecolor="black",
                       labelcolor="black", borderpad=1.0,
                       labelspacing=0.9, handletextpad=0.8,
                       columnspacing=1.8)
    legend.get_frame().set_linewidth(2.0)

    # subplots_adjust, not tight_layout -- tight_layout has no knowledge
    # of the externally-anchored legend below and (empirically) left the
    # axes far narrower than the legend needs, which bbox_inches='tight'
    # then crops AROUND rather than fixing, producing a squeezed-looking
    # plot in a wide canvas. This gives the axes nearly the full figure
    # width directly.
    fig.subplots_adjust(left=0.06, right=0.985, top=0.97, bottom=0.08)
    out = os.path.join(params.VIZ_DIR, "constraint_failures_poster.png")
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}  ({len(idx)} total evaluations plotted, "
         f"{n_above_cap} above the {Y_CAP_M:.0f}m cap)")


if __name__ == "__main__":
    main()
