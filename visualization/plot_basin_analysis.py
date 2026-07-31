"""
plot_basin_analysis.py
========================
Answers "how many local minima did the CMA-ES search find, and how
similar are they?" by taking the best (all-constraints-passing) result
from EVERY independent cmaes_search.py run in the cumulative log
(optimize/runs/cmaes_all_evaluations.csv -- 84 runs as of 2026-07-22: manual
runs 1-3, the 4 diverse restarts, and 76 from the overnight
sweep_restarts.py sweep) and analyzing that population of ~82 "run
optima" directly, rather than the raw 26k-evaluation cloud.

Finding worth visualizing explicitly: hierarchical clustering of these
run-optima in (a, b, gap) is UNSTABLE across any reasonable distance
threshold (35-64 "clusters" for fine thresholds, collapsing to 3 for
coarse ones) -- the signature of a continuous ridge, not discrete,
well-separated basins. Spearman correlation confirms it: tape_km vs b is
strongly correlated (rho ~0.7, p<1e-13) while tape_km vs gap is not
significant at all (rho~0.14, p~0.19). Mechanistic reading: nearly every
restart found the SAME attractor (small-to-moderate a, gap pinned near
its 3mm-face-gap floor) -- the spread in outcomes is mostly how far each
restart's fixed 300-eval budget let it shrink b (and the turns that go
with it) toward run 3's true optimum, not evidence of many competing
local minima.

Produces two figures in visualization/:
  cmaes_basin_structure.png  -- continuous view: a-vs-b colored by
                                tape_km, plus tape_km vs each of
                                a/b/gap/n_total with Spearman rho, showing
                                WHY this reads as one ridge, not many
                                basins.
  cmaes_basin_clusters.png   -- discrete view: hierarchical clustering at
                                a threshold chosen where the cluster count
                                is locally STABLE (a genuine plateau, not
                                an arbitrary cutoff), for readers who want
                                a concrete cluster count despite the above
                                caveat.

Run from Racetrack_v4 root:
    conda run -n fenicsx-env python3 visualization/plot_basin_analysis.py
"""
import os, sys, csv
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
import opt_config as cfg

# same physically-motivated per-axis tolerances used to judge "same basin"
SCALE_A, SCALE_B, SCALE_GAP = 5.0, 10.0, 3.0   # mm


def _style_axes(ax):
    ax.set_facecolor("#0d0d1a")
    ax.tick_params(colors="white", labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.grid(True, alpha=0.2, color="#555")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")


def load_run_optima():
    path = os.path.join(_ROOT, cfg.CMAES_MASTER_LOG)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    by_run = {}
    for r in rows:
        by_run.setdefault(r["run_tag"], []).append(r)

    optima = []
    for tag, rs in by_run.items():
        ok_rows = [r for r in rs
                  if str(r["all_constraints_ok"]).strip().lower() == "true"]
        if not ok_rows:
            continue
        best = min(ok_rows, key=lambda r: float(r["tape_km"]))
        optima.append(dict(
            run_tag=tag, a=float(best["a_mm"]), b=float(best["b_mm"]),
            gap=float(best["gap_mm"]), tape=float(best["tape_km"]),
            B=float(best["B_target_T"]), unif=float(best["uniformity_pct"]),
            hoop=float(best["hoop_MPa"]), n_total=float(best["n_total"])))
    return optima


def cluster_optima(optima, threshold=3.5):
    X = np.array([[o["a"] / SCALE_A, o["b"] / SCALE_B, o["gap"] / SCALE_GAP]
                 for o in optima])
    Z = linkage(X, method="average", metric="euclidean")
    labels = fcluster(Z, t=threshold, criterion="distance")
    for o, l in zip(optima, labels):
        o["cluster"] = int(l)
    return Z, labels


def plot_structure(optima):
    tape = np.array([o["tape"] for o in optima])
    a = np.array([o["a"] for o in optima])
    b = np.array([o["b"] for o in optima])
    gap = np.array([o["gap"] for o in optima])
    n_tot = np.array([o["n_total"] for o in optima])
    best_i = int(np.argmin(tape))

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#111")
    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[0, :])
    _style_axes(ax0)
    sc = ax0.scatter(a, b, c=tape, cmap="magma_r", s=45, edgecolor="white",
                     linewidth=0.3)
    ax0.scatter([a[best_i]], [b[best_i]], marker="*", s=500, color="white",
               edgecolor="black", linewidth=1.2, zorder=5,
               label=f"run 3 (best): {tape[best_i]:.3f} km")
    cb = fig.colorbar(sc, ax=ax0)
    cb.set_label("best tape found by that run  [km]", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    ax0.set_xlabel("a  [mm]"); ax0.set_ylabel("b  [mm]")
    ax0.set_title("Every independent run's best design -- smooth gradient, "
                 "not separated blobs", color="white", fontsize=11)
    ax0.legend(fontsize=9, labelcolor="white", facecolor="#222")

    gs_bottom = gs[1, :].subgridspec(1, 3)
    for i, (name, x) in enumerate([("a", a), ("b", b), ("gap", gap)]):
        ax = fig.add_subplot(gs_bottom[0, i])
        _style_axes(ax)
        rho, p = spearmanr(x, tape)
        ax.scatter(x, tape, c=tape, cmap="magma_r", s=22, edgecolor="none")
        ax.scatter([x[best_i]], [tape[best_i]], marker="*", s=260,
                  color="white", edgecolor="black", linewidth=1, zorder=5)
        ax.set_xlabel(f"{name}  [mm]")
        ax.set_ylabel("tape  [km]" if i == 0 else "")
        sig = "***" if p < 1e-6 else ("*" if p < 0.05 else "n.s.")
        ax.set_title(f"tape vs {name}: ρ={rho:.2f} {sig}",
                    color="white", fontsize=10)

    fig.suptitle("Basin structure across 82 independent run-optima -- "
                 "tape cost tracks b (ρ=0.72) far more than gap "
                 "(ρ=0.15, not significant): one ridge, not many basins",
                 color="white", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(params.VIZ_DIR, "cmaes_basin_structure.png")
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor(),
               bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_clusters(optima, threshold=3.5):
    Z, labels = cluster_optima(optima, threshold)
    tape = np.array([o["tape"] for o in optima])
    a = np.array([o["a"] for o in optima])
    b = np.array([o["b"] for o in optima])
    best_i = int(np.argmin(tape))

    unique = sorted(set(labels), key=lambda c: -np.sum(labels == c))
    palette = ["#ff8f00", "#7d1ea8", "#1e88e5", "#43a047", "#e53935",
              "#00acc1", "#fdd835", "#8e24aa"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor("#111")
    for ax in axes:
        _style_axes(ax)

    for i, cl in enumerate(unique):
        m = labels == cl
        n = int(m.sum())
        best_tape = tape[m].min()
        axes[0].scatter(a[m], b[m], color=palette[i % len(palette)],
                       s=60, edgecolor="white", linewidth=0.4,
                       label=f"region {i+1}: n={n}, best={best_tape:.3f}km")
    axes[0].scatter([a[best_i]], [b[best_i]], marker="*", s=500,
                   color="white", edgecolor="black", linewidth=1.2, zorder=5)
    axes[0].set_xlabel("a  [mm]"); axes[0].set_ylabel("b  [mm]")
    axes[0].set_title(f"{len(unique)} regions at a distance threshold "
                      "chosen where the count is locally stable",
                      color="white", fontsize=10)
    axes[0].legend(fontsize=8, labelcolor="white", facecolor="#222",
                  loc="upper left")

    # cluster-count sensitivity to threshold -- shows WHY 5 isn't "the"
    # answer, just a defensible plateau
    X = np.array([[o["a"] / SCALE_A, o["b"] / SCALE_B, o["gap"] / SCALE_GAP]
                 for o in optima])
    Zfull = linkage(X, method="average", metric="euclidean")
    threshes = np.linspace(0.3, 6.0, 60)
    counts = [fcluster(Zfull, t=t, criterion="distance").max()
             for t in threshes]
    axes[1].plot(threshes, counts, color="#ff8f00", linewidth=2.5)
    axes[1].axvline(threshold, color="white", ls="--", lw=1.5,
                    label=f"chosen threshold ({threshold})")
    axes[1].set_xlabel("clustering distance threshold (a.u.)")
    axes[1].set_ylabel("number of clusters")
    axes[1].set_title("Cluster count keeps falling smoothly with looser "
                      "thresholds -- no sharp elbow -> continuum, not "
                      "discrete basins", color="white", fontsize=10)
    axes[1].legend(fontsize=9, labelcolor="white", facecolor="#222")

    fig.suptitle("Discrete-cluster view (use with caution -- see right "
                 "panel)", color="white", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(params.VIZ_DIR, "cmaes_basin_clusters.png")
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor(),
               bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    print(f"\n{len(unique)} regions at threshold={threshold}:")
    for i, cl in enumerate(unique):
        m = labels == cl
        print(f"  region {i+1}: n={int(m.sum())}  "
             f"a=[{a[m].min():.1f},{a[m].max():.1f}]  "
             f"b=[{b[m].min():.1f},{b[m].max():.1f}]  "
             f"best_tape={tape[m].min():.3f}km")


def main():
    optima = load_run_optima()
    print(f"{len(optima)} runs with an all-pass optimum")
    plot_structure(optima)
    plot_clusters(optima)


if __name__ == "__main__":
    main()
