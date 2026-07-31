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

COLORS = {
    "pass":          "#3a9d5c",
    "B_target_low":  "#3572c6",
    "hoop_high":     "#c0392b",
    "uniformity_high": "#e0a300",
    "infeasible":    "#999999",
}
LABELS = {
    "pass":         f"Pass (All Constraints Met)",
    "B_target_low": f"Failed: Target Field < {B_MIN:.0f} T",
    "hoop_high":    f"Failed: Hoop Stress > {HOOP_MAX:.0f} MPa",
    "uniformity_high": f"Failed: Uniformity Estimate > {UNIF_MAX:.0f}%",
    "infeasible":   "Infeasible Geometry",
}
ZORDER = {"pass": 1, "infeasible": 1, "B_target_low": 2,
         "uniformity_high": 3, "hoop_high": 4}

Y_CAP_M = 1000.0   # linear axis cap -- see main()'s caption for the
                   # excluded fraction; keeps the interesting low range
                   # (all real candidates sit under ~250m) legible
                   # instead of being crushed by the early-search outliers

# current champion: run_20260723_124414 eval 1759, verified as row 67969
# in the cumulative master log (chronological position plotted below)
CHAMPION_IDX = 67969
CHAMPION_TAPE_M = 0.22586182219270562 * 1000


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

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#888")
    ax.tick_params(colors="black", labelsize=15)

    n_above_cap = 0
    legend_handles = []
    for c in ("infeasible", "pass", "B_target_low", "uniformity_high",
             "hoop_high"):
        m = cat == c
        if not m.any():
            continue
        n_above_cap += int((tape[m] > Y_CAP_M).sum())
        ax.scatter(idx[m], tape[m], s=3, color=COLORS[c], alpha=0.35,
                  linewidths=0, rasterized=True, zorder=ZORDER[c])
        legend_handles.append(Line2D([0], [0], marker="o", color="w",
                                     markerfacecolor=COLORS[c],
                                     markersize=10,
                                     label=f"{LABELS[c]}  (n={m.sum():,})"))

    ax.scatter([CHAMPION_IDX], [CHAMPION_TAPE_M], marker="*", s=700,
              color="#e8792a", edgecolor="black", linewidth=1.3, zorder=10)
    legend_handles.append(Line2D([0], [0], marker="*", color="w",
                                 markerfacecolor="#e8792a",
                                 markeredgecolor="black", markersize=18,
                                 label=f"Current Champion "
                                       f"({CHAMPION_TAPE_M:.0f} m)"))

    ax.set_ylim(0, Y_CAP_M)
    ax.set_xlabel("Evaluation Number  (Chronological, All Runs)",
                 fontsize=24, color="black")
    ax.set_ylabel("Tape Length  (m)", fontsize=24, color="black")
    legend = ax.legend(handles=legend_handles, fontsize=18,
                       loc="upper right", frameon=True, facecolor="white",
                       edgecolor="#888", labelcolor="black", borderpad=0.8,
                       labelspacing=0.6, handletextpad=0.6)
    legend.get_frame().set_linewidth(1.0)

    fig.tight_layout()
    out = os.path.join(params.VIZ_DIR, "constraint_failures_poster.png")
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}  ({len(idx)} total evaluations plotted, "
         f"{n_above_cap} above the {Y_CAP_M:.0f}m cap)")


if __name__ == "__main__":
    main()
