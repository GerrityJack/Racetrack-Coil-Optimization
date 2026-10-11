"""
make_field_comparison_updated.py
==================================
Updated-design version (2026-10-04) of make_field_comparison.py, for the
28-layer best design of optimize/runs/updated_10_2/ (eval 166), at its
I_op = 131.87 A, end of the 2 h constant-power ramp (4 h single-step
surrogate), rho-floor 0.6, solved with the 2026-10-04 corrected T-A
resistivity by optimize/studies/ramp_hold_test.py (RH_TAG
updated_10_2_eval166_131p9A_4h, snapshot step 4).  Original description:

Poster figure: genuine uniform-J vs. T-A midplane |B| comparison, on a
SHARED color scale, plus a third panel showing the spatial SCIF
percentage difference directly.

Important distinction from the existing ta_field_top.png: that figure's
full-domain field map is actually just the uniform-J Biot-Savart map
(compute_both_coils_field_multilayer alone) with the T-A on-axis SCIF
number only in the TITLE TEXT -- it is not spatially SCIF-corrected. Only
ta_uniformity.png's 30x6mm box is a true per-point T-A correction. This
script extends that same per-cell dB_bore_from_dJ() machinery (used by
ta_solve.py for its validated box_ptp_pct ground truth) across a much
wider grid, to get a genuinely T-A-corrected FULL field map -- not just
one box.

Reuses per-cell screening data already saved in racetrack_ta_fields.npz
by solve/ta_solve.py's main() (J_TA_coil, J_unif_coil, dV, at I_design) --
no new T-A solve required.

Inputs:
  solve/racetrack_ta_fields.npz  (run solve/ta_solve.py if missing/stale)
Output: visualization/for poster/field_comparison_uniform_vs_TA_updated.png
"""
import os, sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

_HERE = os.path.dirname(os.path.abspath(__file__))
_VIZ  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_VIZ)
for _p in (_ROOT, _VIZ, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "solve"), os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
from coil2_field import compute_both_coils_field_multilayer
from ta_solve import dB_bore_from_dJ
import opt_config as cfg

NX, NY = 130, 65
_DEFAULT_FIELDS_NPZ = os.path.join(
    _ROOT, "optimize", "runs", "ramp_hold",
    "updated_10_2_eval166_131p9A_4h_s4_fields.npz")
_FIELDS_NPZ = os.environ.get("FIELDS_NPZ", _DEFAULT_FIELDS_NPZ)
_cache_tag = os.path.splitext(os.path.basename(_FIELDS_NPZ))[0]
_CACHE = os.path.join(_HERE, f"_field_comparison_cache_{_cache_tag}.npz")


def _override_params_from_npz(ta):
    if "a" in ta and "b" in ta and "n_turns" in ta and "coil_half_gap" in ta:
        params.a             = float(ta["a"])
        params.b             = float(ta["b"])
        params.coil_half_gap = float(ta["coil_half_gap"])
        params.n_turns       = list(map(int, ta["n_turns"]))
        params.recompute_derived()
        print(f"  [geometry from npz] a={params.a*1e3:.2f}mm "
              f"b={params.b*1e3:.2f}mm gap={params.coil_half_gap*1e3:.2f}mm "
              f"n_layers={params.n_layers}")


def _compute():
    """Runs the two Biot-Savart passes (~45s combined) and caches the
    result to disk -- re-running this script for a pure styling tweak
    should not re-pay that cost."""
    ta = np.load(_FIELDS_NPZ)
    _override_params_from_npz(ta)
    I_solved = float(ta["I_solved"])
    print(f"  I_solved = {I_solved:.1f} A  (params.I_design = {params.I_design:.1f} A)")

    centroids = ta["coil_centroids"]
    J_TA      = ta["J_TA_coil"]
    J_unif    = ta["J_unif_coil"]
    dV        = ta["dV"]
    delta_SC  = float(ta["delta_SC"])
    Lambda    = float(params.t)

    dJ_arr = J_TA - J_unif
    dJ_s   = dJ_arr * (delta_SC / Lambda)

    b = params.b
    g = float(params.coil_half_gap)
    xs = np.linspace(-b * 1.18, b * 1.18, NX)
    ys = np.linspace(-b * 0.68, b * 0.68, NY)
    Xg, Yg = np.meshgrid(xs, ys)
    fp = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, g)])

    print(f"  Computing uniform-J field on {NX}x{NY} grid …")
    t0 = time.time()
    B_uniform = compute_both_coils_field_multilayer(
        fp, I_per_turn=I_solved, n_straight=400, n_cap=300)
    print(f"    done in {time.time()-t0:.1f}s")

    print(f"  Computing T-A screening correction on {NX}x{NY} grid …")
    t0 = time.time()
    dB = np.array([dB_bore_from_dJ(centroids, dJ_s, dV, bore_pt=p) for p in fp])
    print(f"    done in {time.time()-t0:.1f}s")

    Bmag_u  = np.linalg.norm(B_uniform, axis=1).reshape(Xg.shape)
    Bmag_ta = np.linalg.norm(B_uniform + dB, axis=1).reshape(Xg.shape)

    np.savez(_CACHE, Xg=Xg, Yg=Yg, Bmag_u=Bmag_u, Bmag_ta=Bmag_ta,
             box_ptp=float(ta["box_ptp_pct"]), I_solved=I_solved)
    return Xg, Yg, Bmag_u, Bmag_ta, float(ta["box_ptp_pct"]), I_solved


def _load_or_compute():
    if os.path.exists(_CACHE):
        # geometry must match the data even when the fields come from cache
        # (main() draws the coil outline from params)
        _override_params_from_npz(np.load(_FIELDS_NPZ))
        d = np.load(_CACHE)
        print(f"  Using cached field data ({_CACHE})")
        return (d["Xg"], d["Yg"], d["Bmag_u"], d["Bmag_ta"],
                float(d["box_ptp"]), float(d["I_solved"]))
    return _compute()


def main():
    Xg, Yg, Bmag_u, Bmag_ta, box_ptp, I_solved = _load_or_compute()
    a, b = params.a, params.b

    # vmin=10T (not 0) deliberately clips/saturates everything below it to
    # the colormap's bottom color -- the field outside the coil (0-10T) is
    # not the story here, and burying it stretches the ENTIRE colormap
    # across just the 10T-vmax plateau where the uniformity box sits and
    # the uniform-J vs. T-A difference actually lives, instead of that
    # difference being compressed into a sliver near the top of a 0-vmax
    # scale.
    # Updated design: its field tops out at ~10.5 T, so a 10 T floor would
    # colour only a thin plateau; 9 T also shows the fall-off around the
    # box (env COLOR_VMIN overrides).
    # Broken (two-segment, piecewise-linear) colour scale: COLOR_VMIN ->
    # COLOR_BREAK gets the bottom BREAK_FRAC of the colormap, COLOR_BREAK ->
    # vmax the rest, so the ~0.1 T uniform-J vs T-A difference above 10 T
    # is resolved while the low-field fall-off stays visible. The colorbar
    # uses the same piecewise scale and marks the break with a white line.
    COLOR_VMIN = float(os.environ.get("COLOR_VMIN", 9.0))
    COLOR_BREAK = float(os.environ.get("COLOR_BREAK", 10.0))
    BREAK_FRAC = float(os.environ.get("BREAK_FRAC", 0.25))
    vmax = max(Bmag_u.max(), Bmag_ta.max())
    xp = [COLOR_VMIN, COLOR_BREAK, vmax]
    fp = [0.0, BREAK_FRAC, 1.0]

    def _fwd(v):
        return np.interp(v, xp, fp)

    def _inv(u):
        return np.interp(u, fp, xp)

    norm = mcolors.FuncNorm((_fwd, _inv), vmin=COLOR_VMIN, vmax=vmax)
    lo_step = 0.5 if COLOR_BREAK - COLOR_VMIN <= 2.0 else 1.0
    lo_ticks = np.arange(np.ceil(COLOR_VMIN / lo_step) * lo_step,
                         COLOR_BREAK - 1e-9, lo_step)
    hi_ticks = np.arange(COLOR_BREAK, vmax + 1e-9, 0.1)
    cb_ticks = np.concatenate([lo_ticks, hi_ticks])

    # ax.set_aspect("equal") shrinks an axes' BOX within its allocated
    # cell whenever the cell's own aspect ratio doesn't already match the
    # data's -- that box-shrink (not hspace, not subplots_adjust ordering)
    # was the actual source of the white gap between the two panels seen
    # in earlier iterations. Fixed here by manually sizing/positioning
    # every axes (main + its own colorbar, via add_axes, not the
    # automatic ax-shrinking fig.colorbar(ax=...) path) so each main
    # panel's box is already exactly data_aspect -- no shrink needed.
    data_aspect = (Xg.max() - Xg.min()) / (Yg.max() - Yg.min())
    FIG_W = 10.0
    PANEL_W = 0.80     # fraction of figure width per panel
    CBAR_W = 0.05
    CBAR_GAP = 0.015
    FIG_H = 2 * PANEL_W * FIG_W / data_aspect

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    axes, caxes = [], []
    for i in range(2):
        y0 = 1.0 - (i + 1) / 2.0
        h = 1.0 / 2.0
        ax = fig.add_axes([0.0, y0, PANEL_W, h])
        cax = fig.add_axes([PANEL_W + CBAR_GAP, y0 + h * 0.04,
                            CBAR_W, h * 0.92])
        axes.append(ax); caxes.append(cax)

    def _outline(ax):
        L = b - a
        th = np.linspace(np.pi / 2, -np.pi / 2, 100)
        radius = a
        xo = np.concatenate([np.linspace(-L, L, 60),
                              L + radius * np.cos(th),
                              np.linspace(L, -L, 60),
                              -L + radius * np.cos(th + np.pi)])
        yo = np.concatenate([np.full(60, radius), radius * np.sin(th),
                              np.full(60, -radius), radius * np.sin(th + np.pi)])
        ax.plot(xo * 1e3, yo * 1e3, "k-", lw=1.0, alpha=0.55)

    panels = [(axes[0], caxes[0], Bmag_u, "Uniform-J  |B|"),
              (axes[1], caxes[1], Bmag_ta, "T-A (screening-corrected)  |B|")]
    for ax, cax, data, label in panels:
        ax.set_facecolor("white")
        cf = ax.pcolormesh(Xg * 1e3, Yg * 1e3, data, shading="auto",
                            cmap="magma", norm=norm)
        cb = fig.colorbar(cf, cax=cax)
        cb.set_ticks(cb_ticks)
        cb.set_ticklabels([f"{t:g}" if t < COLOR_BREAK else f"{t:.1f}"
                           for t in cb_ticks])
        cb.set_label("|B|  (T)")
        # mark the scale change: a white gap across the bar
        cax.axhline(COLOR_BREAK, color="white", lw=2.5)

        _outline(ax)
        rx, ry = cfg.TARGET_X_M * 1e3, cfg.TARGET_Y_M * 1e3
        box_patch = mpatches.Rectangle(
            (-rx / 2, -ry / 2), rx, ry, lw=1.6, edgecolor="lime",
            facecolor="none", zorder=5,
            label=f"{cfg.TARGET_X_M*1e3:.0f}x{cfg.TARGET_Y_M*1e3:.0f}mm "
                  f"uniformity target box")
        ax.add_patch(box_patch)
        ax.set_aspect("equal")
        # Clamp to the data grid's exact extent -- the outline circle
        # (radius = a) is taller than the grid's y half-range, so without
        # this matplotlib auto-expands the axes to fit it, leaving blank
        # canvas above/below the colored data instead of clipping it.
        ax.set_xlim(Xg.min() * 1e3, Xg.max() * 1e3)
        ax.set_ylim(Yg.min() * 1e3, Yg.max() * 1e3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        ax.text(0.015, 0.02, label, transform=ax.transAxes, fontsize=13,
                color="white", fontweight="bold", ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black",
                          alpha=0.55, edgecolor="none"))

    # thin white separator between the two panels
    fig.add_artist(plt.Line2D([0.0, PANEL_W], [0.5, 0.5], color="white",
                              lw=3.0, transform=fig.transFigure, zorder=10))
    axes[0].legend(handles=[box_patch], loc="upper right",
                   fontsize=16, framealpha=0.9, handlelength=2.2,
                   handleheight=1.2, borderpad=0.6)

    # OUT_TAG distinguishes variants (e.g. _searchmethod)
    out = os.path.join(_HERE, "field_comparison_uniform_vs_TA_updated"
                       + os.environ.get("OUT_TAG", "") + ".png")
    # write to a temp file and rename atomically: overwriting in place
    # while the image is open in a viewer (e.g. from Windows) produced a
    # corrupted PNG on 2026-10-04
    tmp = out + ".tmp.png"
    fig.savefig(tmp, dpi=180, bbox_inches="tight", pad_inches=0.02,
               facecolor="white")
    os.replace(tmp, out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
