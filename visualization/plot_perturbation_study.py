"""
plot_perturbation_study.py -- figure for optimize/perturbation_study.py
=======================================================================
Four panels, project dark theme:
  1. box uniformity vs radial translation of the whole coil (a and b moved
     together, everything else fixed) -- the fine-spacing refill of the
     earlier coarse +-3mm a-isolation sweep
  2. box uniformity vs each single-axis perturbation (b alone, gap alone)
  3. box uniformity for the turn-distribution variants
  4. all-axes-at-once jitter samples vs the champion, with the T-A
     repeat-to-repeat spread drawn as an error bar so mesh noise is
     visually separable from real sensitivity

Run:
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \
        visualization/plot_perturbation_study.py
"""
import os, sys, csv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import params
import optimize.opt_config as cfg  # noqa: F401  (kept for target constants)

CSV_PATH = os.path.join(_ROOT, "optimize", "runs", "perturbation", "perturbation_results.csv")
OUT_PATH = os.path.join(_ROOT, "visualization", "perturbation_study.png")
UNIF_LIMIT = 1.0     # the design target this study is judged against [%]

FIG_BG, AX_BG, SPINE = "#111", "#0d0d1a", "#444"


def load():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if not r.get("box_ptp_mean") or r["box_ptp_mean"] in ("", "nan"):
                continue
            for k in ("a_mm", "b_mm", "gap_mm", "tape_km", "B_target_T",
                      "hoop_MPa", "box_ptp_0", "box_ptp_1", "box_ptp_mean",
                      "box_ptp_spread"):
                try:
                    r[k] = float(r[k])
                except (TypeError, ValueError):
                    r[k] = float("nan")
            rows.append(r)
    return rows


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(AX_BG)
    ax.set_title(title, color="white", fontsize=11, pad=8)
    ax.set_xlabel(xlabel, color="white", fontsize=9)
    ax.set_ylabel(ylabel, color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    for s in ax.spines.values():
        s.set_color(SPINE)
    ax.grid(alpha=0.15, color="white", lw=0.5)


def _limit_line(ax):
    ax.axhline(UNIF_LIMIT, color="#ff5555", ls="--", lw=1.2,
               label=f"{UNIF_LIMIT}% design target")


def main():
    rows = load()
    base = next(r for r in rows if r["label"] == "champion")
    b0 = base["box_ptp_mean"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor(FIG_BG)

    # ── panel 1: rigid radial translation ───────────────────────────────
    ax = axes[0, 0]
    sel = [r for r in rows if r["group"] in ("a_translate", "baseline")]
    sel.sort(key=lambda r: r["a_mm"])
    x = [r["a_mm"] for r in sel]
    y = [r["box_ptp_mean"] for r in sel]
    e = [r["box_ptp_spread"] / 2 if np.isfinite(r["box_ptp_spread"]) else 0
         for r in sel]
    ax.errorbar(x, y, yerr=e, color="#ffb000", marker="o", ms=5, lw=1.5,
                capsize=3, label="perturbed")
    ax.plot([base["a_mm"]], [b0], marker="*", ms=18, color="#00e5ff", ls="",
            label="champion", zorder=5)
    _limit_line(ax)
    _style(ax, "Rigid radial translation (a and b moved together)",
           "coil inner radius a [mm]", "T-A box peak-to-peak [%]")
    ax.legend(facecolor=AX_BG, edgecolor=SPINE, labelcolor="white", fontsize=8)

    # ── panel 2: single-axis (b alone, gap alone) ───────────────────────
    ax = axes[0, 1]
    for grp, key, col, lbl in (("b_only", "b_mm", "#ff6ec7",
                                "b alone (straight length)"),
                               ("gap_only", "gap_mm", "#7cff6e",
                                "coil_half_gap alone")):
        sel = [r for r in rows if r["group"] == grp]
        sel.sort(key=lambda r: r[key])
        dx = [r[key] - base[key] for r in sel]
        y = [r["box_ptp_mean"] for r in sel]
        e = [r["box_ptp_spread"] / 2 if np.isfinite(r["box_ptp_spread"]) else 0
             for r in sel]
        ax.errorbar([0] + dx, [b0] + y, yerr=[0] + e, color=col, marker="o",
                    ms=5, lw=1.5, capsize=3, label=lbl)
    ax.plot([0], [b0], marker="*", ms=18, color="#00e5ff", ls="",
            label="champion", zorder=5)
    _limit_line(ax)
    _style(ax, "Single-axis perturbations (one variable at a time)",
           "offset from champion [mm]", "T-A box peak-to-peak [%]")
    ax.legend(facecolor=AX_BG, edgecolor=SPINE, labelcolor="white", fontsize=8)

    # ── panel 3: turn distribution ──────────────────────────────────────
    ax = axes[1, 0]
    sel = [base] + [r for r in rows if r["group"] == "turns"]
    lbls = [r["label"].replace("turns_", "") for r in sel]
    y = [r["box_ptp_mean"] for r in sel]
    e = [r["box_ptp_spread"] / 2 if np.isfinite(r["box_ptp_spread"]) else 0
         for r in sel]
    cols = ["#00e5ff"] + ["#ffb000"] * (len(sel) - 1)
    ax.bar(range(len(sel)), y, yerr=e, color=cols, capsize=3,
           error_kw=dict(ecolor="white", lw=1))
    ax.set_xticks(range(len(sel)))
    ax.set_xticklabels(lbls, rotation=30, ha="right", color="white", fontsize=8)
    _limit_line(ax)
    _style(ax, "Turn redistribution (double-pancake pairing preserved)",
           "", "T-A box peak-to-peak [%]")
    ax.legend(facecolor=AX_BG, edgecolor=SPINE, labelcolor="white", fontsize=8)

    # ── panel 4: all-axes jitter ────────────────────────────────────────
    ax = axes[1, 1]
    sel = [base] + [r for r in rows if r["group"] == "jitter"]
    lbls = [r["label"] for r in sel]
    y = [r["box_ptp_mean"] for r in sel]
    e = [r["box_ptp_spread"] / 2 if np.isfinite(r["box_ptp_spread"]) else 0
         for r in sel]
    cols = ["#00e5ff"] + ["#b06eff"] * (len(sel) - 1)
    ax.bar(range(len(sel)), y, yerr=e, color=cols, capsize=3,
           error_kw=dict(ecolor="white", lw=1))
    ax.set_xticks(range(len(sel)))
    ax.set_xticklabels(lbls, rotation=30, ha="right", color="white", fontsize=8)
    _limit_line(ax)
    _style(ax, "Simultaneous jitter on all axes (a, b, gap, turns)\n"
               "error bar = T-A independent-mesh repeat spread",
           "", "T-A box peak-to-peak [%]")
    ax.legend(facecolor=AX_BG, edgecolor=SPINE, labelcolor="white", fontsize=8)

    fig.suptitle("Robustness of the 6-layer champion: T-A box uniformity "
                 "under small, buildable perturbations",
                 color="white", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PATH, dpi=160, facecolor=FIG_BG)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
