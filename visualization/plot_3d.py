"""
plot_3d.py
===========
Produces exactly two output images:

  geometry.png   — square 3-D wire-frame of the winding pack + y-z
                   cross-section diagram (aligned outer edge, varying inner
                   edges); works for any n_turns list

  field_3d.png   — 3-D scatter of coil cells coloured by |B| for the full
                   two-coil system (FEM 1/8-domain expanded by x/y mirror
                   symmetry and coil-2 z-offset)

Inputs: solve/racetrack_fields.npz
Run after solve.py.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D   # noqa: registers projection
import pandas as pd

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params


# ── dynamic helpers (n_layers-agnostic) ─────────────────────────────────────

def _layer_colors():
    """Return one distinct color per layer; works for any n_layers."""
    cmap = matplotlib.colormaps.get_cmap("tab10")
    return [cmap(i % 10) for i in range(params.n_layers)]


def _racetrack_xy(radius, n=300):
    a, b, L = params.a, params.b, params.b - params.a
    th = np.linspace(np.pi/2, -np.pi/2, n//4, endpoint=False)
    x = np.concatenate([np.linspace(-L, L, n//4, endpoint=False),
                        L + radius*np.cos(th),
                        np.linspace(L, -L, n//4, endpoint=False),
                        -L + radius*np.cos(th + np.pi)])
    y = np.concatenate([np.full(n//4, radius), radius*np.sin(th),
                        np.full(n//4, -radius), radius*np.sin(th + np.pi)])
    return np.append(x, x[0]), np.append(y, y[0])


def _ax3d_style(ax):
    """Apply consistent dark styling to a 3-D axis."""
    ax.set_facecolor("#0d0d1a")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False; pane.set_edgecolor("#2a2a3a")
    ax.tick_params(colors="white", labelsize=7)
    ax.xaxis.label.set_color("white"); ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_color("white"); ax.yaxis.label.set_fontsize(8)
    ax.zaxis.label.set_color("white"); ax.zaxis.label.set_fontsize(8)


def _colorbar_dark(fig, mappable, ax, label):
    cb = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.03)
    cb.set_label(label, color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    return cb


# ═══════════════════════════════════════════════════════════════════════════
# geometry.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_geometry():
    """
    Left panel  — 3-D isometric wire-frame.
    Right panel — y-z cross-section at x ≈ 0 (clearly shows the
                  staircase inner-edge profile and aligned outer edge).
    Works for any params.n_turns list.
    """
    colors = _layer_colors()
    a_out  = params.a_out
    fig = plt.figure(figsize=(12, 12))
    fig.patch.set_facecolor("#111")

    # ── left: 3-D isometric ──────────────────────────────────────────────
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    _ax3d_style(ax3)

    n_connectors = 8   # vertical connectors around the racetrack per edge

    for i in range(params.n_layers):
        a_in  = params.a_inner_list[i]
        z_bot = params.layer_z_bottoms[i]
        z_top = params.layer_z_tops[i]
        col   = colors[i]
        n_turns_i = params.n_turns[i]

        # Top and bottom face: outer edge (dashed white) + inner edge (colour)
        for z_face in (z_bot, z_top):
            xo, yo = _racetrack_xy(a_out)
            xi, yi = _racetrack_xy(a_in)
            zo = np.full_like(xo, z_face)
            zi = np.full_like(xi, z_face)
            ax3.plot(xo * 1e3, yo * 1e3, zo * 1e3,
                     "w--", lw=0.7, alpha=0.35)
            ax3.plot(xi * 1e3, yi * 1e3, zi * 1e3,
                     color=col, lw=1.6, alpha=0.9,
                     label=(f"Layer {i}  ({n_turns_i} turns)"
                            if z_face == z_bot else None))

        # Vertical connectors on outer and inner edges
        step = max(1, len(_racetrack_xy(a_out)[0]) // (n_connectors * 2))
        for radius, ls in [(a_out, "--"), (a_in, "-")]:
            xr, yr = _racetrack_xy(radius, n=n_connectors * 4)
            for k in range(0, len(xr) - 1, max(1, len(xr) // n_connectors)):
                lc = "white" if ls == "--" else col
                ax3.plot([xr[k]*1e3, xr[k]*1e3],
                         [yr[k]*1e3, yr[k]*1e3],
                         [z_bot*1e3, z_top*1e3],
                         color=lc, lw=0.8 if ls == "--" else 1.0,
                         alpha=0.3 if ls == "--" else 0.6, ls=ls)

    ax3.set_xlabel("x (mm)"); ax3.set_ylabel("y (mm)"); ax3.set_zlabel("z (mm)")
    ax3.view_init(elev=28, azim=-55)
    ax3.set_title(f"{params.n_layers}-layer winding pack\n"
                  f"n_turns = {params.n_turns}",
                  color="white", fontsize=10, pad=6)
    ax3.legend(fontsize=8, labelcolor="white", facecolor="#222",
               framealpha=0.4, loc="upper left")

    # ── right: y-z cross-section ──────────────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor("#0d0d1a")
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values(): sp.set_edgecolor("#444")

    for i in range(params.n_layers):
        a_in  = params.a_inner_list[i]
        z_bot = params.layer_z_bottoms[i] * 1e3
        z_top = params.layer_z_tops[i]    * 1e3
        col   = colors[i]
        h     = z_top - z_bot

        # Positive-y side
        ax2.add_patch(mpatches.Rectangle(
            (a_in * 1e3, z_bot), (a_out - a_in) * 1e3, h,
            facecolor=col, alpha=0.55, edgecolor="white", linewidth=0.7))
        # Negative-y side (mirror)
        ax2.add_patch(mpatches.Rectangle(
            (-a_out * 1e3, z_bot), (a_out - a_in) * 1e3, h,
            facecolor=col, alpha=0.55, edgecolor="white", linewidth=0.7))
        # Turn count annotation
        ax2.text((a_in + a_out) / 2 * 1e3, (z_bot + z_top) / 2,
                 f"{params.n_turns[i]}t", ha="center", va="center",
                 color="white", fontsize=8, fontweight="bold")

    # Outer edge lines
    ax2.axvline( a_out * 1e3, color="white", lw=1.2, ls="--", alpha=0.6,
                 label="outer edge (aligned)")
    ax2.axvline(-a_out * 1e3, color="white", lw=1.2, ls="--", alpha=0.6)

    # Bore region
    a_in_min = min(params.a_inner_list) * 1e3
    ax2.axvspan(-a_in_min, a_in_min, color="#222255", alpha=0.4,
                label="bore (min inner edge)")

    ax2.set_xlabel("y  (mm,  radial)", color="white")
    ax2.set_ylabel("z  (mm,  axial)", color="white")
    ax2.set_title("Winding-pack cross-section at x ≈ 0\n"
                  "(dashed white = aligned outer edge)",
                  color="white", fontsize=10)
    ax2.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e",
               framealpha=0.6, loc="upper right")

    # Legend patches for layers
    handles = [mpatches.Patch(facecolor=colors[i], alpha=0.7,
                               label=f"Layer {i} ({params.n_turns[i]} turns,  "
                                     f"a_in={params.a_inner_list[i]*1e3:.1f} mm)")
               for i in range(params.n_layers)]
    ax2.legend(handles=handles, fontsize=8, labelcolor="white",
               facecolor="#1a1a2e", framealpha=0.7, loc="lower right")

    fig.suptitle(
        f"Racetrack coil winding geometry  —  "
        f"outer edge = {params.a_out*1e3:.2f} mm  (all layers aligned)\n"
        f"Total stack height = {params.n_layers * params.w * 1e3:.1f} mm  "
        f"|  Total turns = {params.n_turns_total}",
        color="white", fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = os.path.join(params.VIZ_DIR, "geometry.png")
    # Do NOT use bbox_inches='tight' — 3-D axes inflate the bounding box
    # arbitrarily and produce a non-square output even with figsize=(12,12).
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Wrote {out}")


# ═══════════════════════════════════════════════════════════════════════════
# field_3d.png
# ═══════════════════════════════════════════════════════════════════════════

def _mirror_z(z, g):
    """z-coordinate of coil 2's image of a point at z in coil 1's own local
    frame, MIRRORED about the midplane (z=g) -- NOT translated by +2g.

    2026-07-27 fix: this project's FEM solve uses a PMC (n×H=0) boundary
    at z=coil_half_gap specifically because the true two-coil system is
    mirror-symmetric about that plane (same physical winding sense on
    both sides, same-direction "image" current -- the standard PMC image
    principle) -- confirmed by physics/coil2_field.py's
    compute_both_coils_field_multilayer(), the function actually used for
    every real design number (B_target, uniformity, tape optimization,
    T-A SCIF), which places layer z-centers at `2*g - z_c` (a true
    mirror), not `z_c + 2*g`. This module's plotting code used the wrong
    (translation) convention throughout -- harmless for a palindromic
    layer stack, but for an asymmetric one like the champion's
    [285,285,379,379,2,2] it silently drew coil 2's layers in the wrong
    relative order (e.g. the thin 2-turn layer ending up near the
    midplane instead of at the far end, on the OPPOSITE side from where
    it correctly sits on coil 1) -- a picture bug only, since none of the
    actual physics calculations went through this code path."""
    return 2.0 * g - z


def _expand_to_full_system(centroids, Bmag):
    """
    Expand FEM coil cells from the simulated 1/8 domain (x≥0, y≥0, full z of
    coil 1) to the complete two-coil system by reflecting x and y, then
    mirroring coil 1 about the midplane (z=coil_half_gap) to get coil 2 --
    see _mirror_z's docstring for why this must be a mirror, not a shift.
    Returns (centroids_full, Bmag_full).
    """
    g = params.coil_half_gap
    c_parts, B_parts = [], []
    # Four quadrants (x, y reflections) → full coil 1
    for sx in (1, -1):
        for sy in (1, -1):
            c = centroids.copy()
            c[:, 0] *= sx; c[:, 1] *= sy
            c_parts.append(c); B_parts.append(Bmag)
    coil1_c = np.vstack(c_parts)
    coil1_B = np.concatenate(B_parts)
    # Coil 2: identical geometry/current, MIRRORED about the midplane
    coil2_c = coil1_c.copy(); coil2_c[:, 2] = _mirror_z(coil2_c[:, 2], g)
    return np.vstack([coil1_c, coil2_c]), np.concatenate([coil1_B, coil1_B])


def plot_field_3d(npz_data):
    """Coil cells coloured by |B|, full two-coil system, isometric + top-down."""
    centroids_raw = npz_data["coil_centroids"]
    Bmag_raw      = np.linalg.norm(npz_data["coil_B"], axis=1)
    I_pt          = float(npz_data["I_solved"])
    colors        = _layer_colors()
    g             = params.coil_half_gap

    # Expand from 1/8 domain to full two-coil system
    if getattr(params, "use_eighth_symmetry", False):
        centroids, Bmag = _expand_to_full_system(centroids_raw, Bmag_raw)
    else:
        centroids, Bmag = centroids_raw, Bmag_raw
        if getattr(params, "two_coil_mode", False):
            c2 = centroids_raw.copy(); c2[:, 2] = _mirror_z(c2[:, 2], g)
            centroids = np.vstack([centroids_raw, c2])
            Bmag      = np.tile(Bmag_raw, 2)

    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor("#0d0d1a")

    for col_idx, (elev, azim, vtitle) in enumerate([
            (28, -55, "Isometric"), (88, -90, "Top-down  (x-y)")]):
        ax = fig.add_subplot(1, 2, col_idx + 1, projection="3d")
        _ax3d_style(ax)
        sc = ax.scatter(centroids[:, 0] * 1e3,
                        centroids[:, 1] * 1e3,
                        centroids[:, 2] * 1e3,
                        c=Bmag, cmap="magma", s=3, alpha=0.5, rasterized=True)
        # Layer boundary rings for both coils. Coil 2's z is MIRRORED about
        # the midplane (z=g), not shifted -- see _mirror_z's docstring.
        for coil2 in (False, True):
            for i in range(params.n_layers):
                xo, yo = _racetrack_xy(params.a_out)
                for z_face in (params.layer_z_tops[i], params.layer_z_bottoms[i]):
                    z_plot = _mirror_z(z_face, g) if coil2 else z_face
                    zo = np.full_like(xo, z_plot * 1e3)
                    ax.plot(xo * 1e3, yo * 1e3, zo,
                            color=colors[i], lw=0.5, alpha=0.45)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
        ax.set_title(vtitle, color="white", fontsize=10)

    _colorbar_dark(fig, sc, fig.axes, "|B| (T)")
    # Layer legend
    handles = [plt.Line2D([0], [0], color=colors[i], lw=2,
                           label=f"Layer {i} ({params.n_turns[i]}t)")
               for i in range(params.n_layers)]
    fig.legend(handles=handles, loc="lower center", ncol=params.n_layers,
               fontsize=8, labelcolor="white", facecolor="#111",
               framealpha=0.4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"|B| field distribution  —  full two-coil system,  "
        f"{params.n_layers} layers,  I = {I_pt:.0f} A/turn,  "
        f"{params.n_turns_total} total turns",
        color="white", fontsize=11, y=1.01)

    out = os.path.join(params.VIZ_DIR, "field_3d.png")
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Wrote {out}")


# ═══════════════════════════════════════════════════════════════════════════
# quench_3d.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_quench_3d(df):
    """4-panel 3-D quench scatter; weakest point marked ★."""
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("  No 'ok' cells — skipping quench_3d.png"); return

    colors = _layer_colors()
    x, y, z = ok["x"].values * 1e3, ok["y"].values * 1e3, ok["z"].values * 1e3
    Iq = ok["quench_current_A"].values
    vmin, vmax = np.percentile(Iq, 2), np.percentile(Iq, 98)
    norm  = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap  = cm.plasma_r

    weak_idx = int(np.argmin(Iq))
    wx, wy, wz, wq = x[weak_idx], y[weak_idx], z[weak_idx], Iq[weak_idx]

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0d0d1a")
    axs = []
    for idx, (elev, azim, vtitle) in enumerate([
            (28, -55, "Isometric"),
            (88, -90, "Top-down"),
            (0,  -90, "Side  (x-z)"),
            (0,    0, "End  (y-z)")]):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        _ax3d_style(ax)
        axs.append(ax)
        ax.scatter(x, y, z, c=Iq, cmap=cmap, norm=norm,
                   s=4, alpha=0.45, rasterized=True)
        ax.scatter([wx], [wy], [wz], c="red", s=350, marker="*",
                   zorder=10, edgecolors="white", linewidths=0.8)
        if idx == 0:
            ax.text(wx, wy, wz + params._z_top * 250,
                    f"  ★ {wq:.0f} A\n  L{int(ok['layer_index'].values[weak_idx]) if 'layer_index' in ok.columns else '?'}",
                    color="red", fontsize=8)
        for i in range(params.n_layers):
            xo, yo = _racetrack_xy(params.a_out)
            for z_f in (params.layer_z_tops[i]*1e3, params.layer_z_bottoms[i]*1e3):
                ax.plot(xo*1e3, yo*1e3, np.full_like(xo, z_f),
                        color=colors[i], lw=0.7, alpha=0.5)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
        ax.set_title(vtitle, color="white", fontsize=9, pad=4)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axs, shrink=0.55, pad=0.04, aspect=30)
    cb.set_label("Quench current  (A/turn)", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    handles = ([plt.Line2D([0], [0], color=colors[i], lw=2,
                            label=f"Layer {i} ({params.n_turns[i]}t)")
                for i in range(params.n_layers)] +
               [plt.Line2D([0], [0], marker="*", color="red", markersize=12,
                           linestyle="none",
                           label=f"Weakest  {wq:.0f} A  "
                                 f"({wx:.0f},{wy:.0f},{wz:.0f}) mm")])
    fig.legend(handles=handles, loc="lower center", ncol=params.n_layers + 1,
               fontsize=8, labelcolor="white", facecolor="#111",
               framealpha=0.4, bbox_to_anchor=(0.5, -0.01))

    li = int(ok["layer_index"].values[weak_idx]) if "layer_index" in ok.columns else "?"
    fig.suptitle(
        f"Quench current distribution  —  n_turns = {params.n_turns}  "
        f"({params.n_turns_total} total)\n"
        f"{len(ok)} cells  |  "
        f"median {np.median(Iq):.0f} A  |  min {wq:.0f} A ★  layer {li}",
        color="white", fontsize=11, y=1.01)

    out = os.path.join(params.VIZ_DIR, "quench_3d.png")
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Wrote {out}")

    print(f"\n  ★ Weakest point:")
    print(f"    Quench current : {wq:.1f} A/turn")
    print(f"    Position       : ({wx:.1f}, {wy:.1f}, {wz:.1f}) mm")
    print(f"    Layer          : {li}  ({params.n_turns[li] if isinstance(li,int) else '?'} turns)")
    if "turn_index" in ok.columns:
        print(f"    Turn in layer  : {int(ok['turn_index'].values[weak_idx])}")


# ═══════════════════════════════════════════════════════════════════════════
# quench_2d.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_quench_2d(df):
    """
    Two-panel 2-D quench map.
    Left:  xy top-down view
    Right: radial-offset vs z cross-section
    """
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("  No 'ok' cells — skipping quench_2d.png"); return

    colors = _layer_colors()
    Iq = ok["quench_current_A"].values
    vmin, vmax = np.percentile(Iq, 2), np.percentile(Iq, 98)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.plasma_r

    weak_idx = int(np.argmin(Iq))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor("#111")

    # ── left: xy top-down ────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0d0d1a")
    sc = ax.scatter(ok["x"] * 1e3, ok["y"] * 1e3,
                    c=Iq, cmap=cmap, norm=norm,
                    s=6, alpha=0.6, rasterized=True)
    ax.scatter(ok["x"].values[weak_idx]*1e3, ok["y"].values[weak_idx]*1e3,
               c="red", s=300, marker="*", zorder=10,
               edgecolors="white", linewidths=0.7)
    xo, yo = _racetrack_xy(params.a_out)
    ax.plot(xo*1e3, yo*1e3, "w--", lw=0.8, alpha=0.4, label="outer edge")
    for i in range(params.n_layers):
        xi, yi = _racetrack_xy(params.a_inner_list[i])
        ax.plot(xi*1e3, yi*1e3, color=colors[i], lw=0.8, alpha=0.5,
                label=f"Layer {i} inner edge ({params.n_turns[i]}t)")

    ax.set_aspect("equal")
    ax.set_xlabel("x  (mm)", color="white"); ax.set_ylabel("y  (mm)", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.set_title("Top-down quench map  (all cells)", color="white", fontsize=11)
    ax.legend(fontsize=7, labelcolor="white", facecolor="#222",
              framealpha=0.5, loc="upper right")

    # ── right: radial-offset vs z ────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#0d0d1a")

    x_c, y_c, z_c = ok["x"].values, ok["y"].values, ok["z"].values
    L = params.b - params.a
    if "layer_index" in ok.columns:
        layer_idx = ok["layer_index"].values.astype(int)
        a_centers = np.array(params.a_center_list)[layer_idx]
        straight  = np.abs(x_c) <= L
        r = np.where(straight, np.abs(y_c),
                     np.hypot(x_c - np.where(x_c > 0, L, -L), y_c))
        radial_off = r - a_centers
    else:
        radial_off = np.zeros_like(x_c)

    sc2 = ax.scatter(radial_off * 1e3, z_c * 1e3,
                     c=Iq, cmap=cmap, norm=norm,
                     s=6, alpha=0.6, rasterized=True)
    ax.scatter(radial_off[weak_idx]*1e3, z_c[weak_idx]*1e3,
               c="red", s=300, marker="*", zorder=10,
               edgecolors="white", linewidths=0.7)

    for i in range(params.n_layers):
        half_pt = params.pack_thickness_list[i] / 2 * 1e3
        z_bot_i = params.layer_z_bottoms[i] * 1e3
        z_top_i = params.layer_z_tops[i]    * 1e3
        ax.add_patch(mpatches.Rectangle(
            (-half_pt, z_bot_i), 2*half_pt, z_top_i - z_bot_i,
            facecolor="none", edgecolor=colors[i], lw=1.2, alpha=0.7,
            label=f"Layer {i} ({params.n_turns[i]}t)"))

    ax.set_xlabel("radial offset from layer centreline  (mm)", color="white")
    ax.set_ylabel("z  (mm)", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.set_title("Radial × axial quench map\n"
                 "(shows which layer and which radial position is weakest)",
                 color="white", fontsize=11)
    ax.legend(fontsize=7, labelcolor="white", facecolor="#222",
              framealpha=0.5, loc="upper right")

    cb = fig.colorbar(sc, ax=axes, shrink=0.7, pad=0.02)
    cb.set_label("Quench current  (A/turn)", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    fig.suptitle(
        f"2-D quench current map  —  n_turns = {params.n_turns}  "
        f"({params.n_turns_total} total turns)\n"
        f"★ = weakest point  ({Iq.min():.0f} A/turn)",
        color="white", fontsize=11, y=1.01)
    fig.tight_layout()

    out = os.path.join(params.VIZ_DIR, "quench_2d.png")
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Wrote {out}")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("Plotting 3-D geometry …")
    plot_geometry()

    npz_path = params.fields_npz_filename
    if os.path.exists(npz_path):
        print("Plotting 3-D field distribution …")
        plot_field_3d(np.load(npz_path))
    else:
        print(f"  Skipping field_3d.png — run solve.py first")

    csv_path = os.path.join(params.SWEEP_DIR, "quench_results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print("Plotting 3-D quench map …")
        plot_quench_3d(df)
        print("Plotting 2-D quench map …")
        plot_quench_2d(df)
    else:
        print(f"  Skipping quench plots — run quench_sweep.py first")

    print(f"\nDone. Wrote geometry.png, field_3d.png, quench_3d.png, "
          f"quench_2d.png  →  {params.VIZ_DIR}")


if __name__ == "__main__":
    main()
