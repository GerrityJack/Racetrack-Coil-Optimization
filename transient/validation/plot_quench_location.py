"""
plot_quench_location.py -- 2026-09-07, spatial visualization of the
champion design's local quench margin, to directly SEE where CLAUDE.md's
"Ramp-up power analysis" finding lives: the naive uniform-J assumption
shows no cell anywhere near quench at I_design, but the T-A (screening-
current-resolved) solve shows a meaningfully tighter margin with real
locally-quenched cells (margin < 1, i.e. Jc(B,theta) < |J_inplane|)
concentrated in a specific spatial region -- not spread uniformly.

Reuses ta_quench_margin_check.py's evaluate() (the exact Jc/Jmag formula
ta_solve.py's own Picard solver uses internally for rho(B,J)) -- no
physics duplicated here, this is pure post-processing/plotting of the
project's standing racetrack_ta_fields.npz.

Output:
  visualization/quench_location_margin_map.png -- top/side view margin
    maps, uniform-J vs T-A, same color scale (RdYlGn, centered at the
    margin=1.0 quench boundary) so the two methods are directly comparable.
  visualization/quench_location_3d.png -- 3-D scatter isolating just the
    T-A locally-quenched cells (margin < 1), colored by how far below 1.

Run:  <env>/bin/python3 transient/validation/plot_quench_location.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D   # noqa: registers projection

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRANS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_TRANS)
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "solve"),
           os.path.join(_ROOT, "optimize"), os.path.join(_ROOT, "visualization"),
           _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params                                            # noqa: E402
from ta_quench_margin_check import evaluate, MARGIN_REQUIRED, FIELDS_NPZ  # noqa: E402
from ic_extrapolation import make_ic_model                # noqa: E402
from plot_3d import _expand_to_full_system                # noqa: E402

VMAX = 3.0   # fixed color-scale cap (in units of margin) shared by both
             # panels, so uniform-J's much larger raw margins don't wash
             # out the T-A panel's contrast right around the quench boundary


def _expand(centroids, scalar):
    c_full, s_full = _expand_to_full_system(centroids, scalar)
    return c_full, s_full


def _panel(ax, x, y, margin, title, xlabel, ylabel):
    """Two-layer render: safe cells (margin >= 1) as a muted blue
    sequential background (context only -- how much margin to spare),
    quenched cells (margin < 1) punched out on top in solid red/orange.
    A single continuous diverging colormap centered at 1.0 was tried
    first and made the quench boundary nearly invisible -- most quenched
    cells sit in 0.59-1.0, a narrow slice of a continuous scale that
    blends into the "just barely safe" cells right above 1.0. This
    two-layer scheme makes "quenched or not" (the actual physically
    meaningful distinction) pop regardless of exactly how far below 1.0
    a cell sits."""
    safe = margin >= 1.0
    ax.scatter(x[safe], y[safe], c=np.clip(margin[safe], 1.0, VMAX),
              cmap="Blues", vmin=1.0, vmax=VMAX, s=3, linewidths=0,
              alpha=0.5)
    sc = ax.scatter(x[~safe], y[~safe], c=margin[~safe],
                    cmap="autumn", vmin=margin[~safe].min() if (~safe).any() else 0.0,
                    vmax=1.0, s=4, linewidths=0)
    ax.set_facecolor("#0d0d1a")
    ax.set_title(title, color="white", fontsize=11)
    ax.set_xlabel(xlabel, color="white"); ax.set_ylabel(ylabel, color="white")
    ax.set_aspect("equal")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    return sc


def main(npz_path=None, out_prefix="quench_location", design_label=None,
         ic_model=None):
    """npz_path: defaults to the champion's solve/racetrack_ta_fields.npz.
    For any OTHER design's npz (e.g. one written by
    optimize/ta_safe_current.evaluate()'s save_fields_path hook), this
    applies that npz's own saved a/b/coil_half_gap/n_turns to params
    in-memory FIRST -- evaluate()'s tangent/normal-direction formulas
    read params.a/params.b directly (L = params.b - params.a), so
    plotting a non-champion design against the champion's live params
    geometry would silently compute the wrong tangent/normal frame.

    ic_model: 2026-09-13 addition -- the Ic(B,theta) model used to GRADE
    the margin shown here. Defaults to the plain 20K KimIcModel (every
    existing caller's behaviour, unchanged). Pass a temperature-scaled
    model (e.g. ic_temperature_scaling.Fujikura4p2KScaledIcModel) when the
    npz's own J_TA_coil was solved under that SAME model -- grading a
    4.2K-solved current distribution against the 20K Ic model would be
    self-inconsistent (the fields would reflect one model's screening
    behaviour while the margin map reflects a different model's Jc)."""
    npz_path = npz_path or FIELDS_NPZ
    ic = ic_model if ic_model is not None else make_ic_model("kim")
    d = np.load(npz_path)
    I_solved = float(d["I_solved"])
    delta_SC = float(d["delta_SC"])
    centroids = d["coil_centroids"]

    if "a" in d.files and "b" in d.files:
        params.a = float(d["a"]); params.b = float(d["b"])
        params.coil_half_gap = float(d["coil_half_gap"])
        params.n_turns = list(d["n_turns"])
        params.recompute_derived()
        print(f"Applied geometry from {npz_path}: a={params.a*1e3:.2f}mm "
              f"b={params.b*1e3:.2f}mm gap={params.coil_half_gap*1e3:.2f}mm "
              f"n_turns={params.n_turns}")
    w = params.w

    print(f"Loaded {npz_path}")
    print(f"  I_solved={I_solved:.1f} A")
    if "box_ptp_pct" in d.files:
        print(f"  box_ptp_pct={float(d['box_ptp_pct']):.4f}%")
    if design_label is None and abs(I_solved - params.I_design) > 0.5:
        print(f"  WARNING: solved at {I_solved:.1f}A, not current I_design "
              f"{params.I_design:.1f}A -- re-run solve/ta_solve.py first.")
    design_label = design_label or f"champion (n_turns={params.n_turns})"

    r_unif = evaluate("uniform-J", centroids, d["coil_B"], d["J_unif_coil"],
                      delta_SC, w, ic)
    r_ta = evaluate("T-A dt=600s", centroids, d["coil_B"], d["J_TA_coil"],
                    delta_SC, w, ic)

    print(f"\nuniform-J : min margin {r_unif['min_margin']:.3f}   "
          f"{r_unif['n_below_1']}/{r_unif['n_cells']} cells < 1.0")
    print(f"T-A       : min margin {r_ta['min_margin']:.3f}   "
          f"{r_ta['n_below_1']}/{r_ta['n_cells']} cells < 1.0   "
          f"({100*r_ta['n_below_1']/r_ta['n_cells']:.1f}% of coil)")

    c_full, m_unif_full = _expand(centroids, r_unif["margin"])
    _, m_ta_full = _expand(centroids, r_ta["margin"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor("#111")

    _panel(axes[0, 0], c_full[:, 0]*1e3, c_full[:, 1]*1e3, m_unif_full,
          f"uniform-J margin, top view  (min={r_unif['min_margin']:.2f}, "
          f"0 cells < 1.0)", "x [mm]", "y [mm]")
    sc = _panel(axes[0, 1], c_full[:, 0]*1e3, c_full[:, 1]*1e3, m_ta_full,
               f"T-A margin, top view  (min={r_ta['min_margin']:.2f}, "
               f"{100*r_ta['n_below_1']/r_ta['n_cells']:.0f}% of coil < 1.0)",
               "x [mm]", "y [mm]")
    _panel(axes[1, 0], c_full[:, 0]*1e3, c_full[:, 2]*1e3, m_unif_full,
          "uniform-J margin, side view", "x [mm]", "z [mm]")
    _panel(axes[1, 1], c_full[:, 0]*1e3, c_full[:, 2]*1e3, m_ta_full,
          "T-A margin, side view", "x [mm]", "z [mm]")

    cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02, aspect=30)
    cb.set_label("locally quenched cells only (margin < 1.0)\n"
                "darker red = further below E_c-defined Ic\n"
                "(blue background: safe cells, margin ≥ 1.0, for context)",
                color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.get_yticklabels(), color="white")

    fig.suptitle(
        f"{design_label}, I={I_solved:.0f} A -- "
        f"local quench margin, uniform-J assumption vs T-A ground truth",
        color="white", fontsize=13)
    out = os.path.join(params.VIZ_DIR, f"{out_prefix}_margin_map.png")
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(),
               bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # ── isolate just the T-A locally-quenched cells in 3-D ────────────────
    quenched = m_ta_full < 1.0
    fig2 = plt.figure(figsize=(9, 8))
    fig2.patch.set_facecolor("#111")
    ax = fig2.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d0d1a")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("#0d0d1a")
        pane.set_edgecolor("#333")
    ax.grid(True, color="#333", alpha=0.4)
    ax.scatter(c_full[~quenched, 0]*1e3, c_full[~quenched, 1]*1e3,
              c_full[~quenched, 2]*1e3, c="#5577aa", s=2, alpha=0.35,
              linewidths=0, depthshade=False, label="safe (margin ≥ 1)")
    if quenched.any():
        pop = quenched.sum() < 200   # sparse case: make the few red dots pop
        sc2 = ax.scatter(c_full[quenched, 0]*1e3, c_full[quenched, 1]*1e3,
                        c_full[quenched, 2]*1e3, c=m_ta_full[quenched],
                        cmap="autumn", vmin=m_ta_full[quenched].min(), vmax=1.0,
                        s=40 if pop else 6, linewidths=0.6 if pop else 0,
                        edgecolors="white" if pop else None, depthshade=False,
                        label="locally quenched (margin < 1)")
        cb2 = fig2.colorbar(sc2, ax=ax, shrink=0.6, pad=0.1)
        cb2.set_label("margin (quenched cells only)", color="white")
        cb2.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb2.ax.get_yticklabels(), color="white")
    ax.set_xlabel("x [mm]", color="white"); ax.set_ylabel("y [mm]", color="white")
    ax.set_zlabel("z [mm]", color="white")
    ax.tick_params(colors="white")
    ax.legend(loc="upper left", fontsize=8, labelcolor="white",
             facecolor="#222", framealpha=0.6)
    frac = 100.0 * quenched.sum() / len(quenched)
    ax.set_title(f"{design_label}\nT-A locally-quenched cells only  "
                f"({frac:.1f}% of coil), I={I_solved:.0f} A",
                color="white", fontsize=12)
    out2 = os.path.join(params.VIZ_DIR, f"{out_prefix}_3d.png")
    fig2.savefig(out2, dpi=160, facecolor=fig2.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
