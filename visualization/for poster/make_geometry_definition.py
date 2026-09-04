"""
make_geometry_definition.py
=============================
Poster figure: labeled dimensions of the champion geometry.

Left panel  -- top-down single-layer footprint: outer edge (a_out), bore
               edge (min inner edge), straight-section length L = b - a.
Right panel -- axial cross-section at x=0 showing BOTH coils (mirrored
               about the midplane), the coil-to-coil half-gap, the
               face-to-face clearance, and the winding-pack stack height.

a and b are the two underlying design parameters (a = nominal inner
radius, b = nominal outer radius; L = b - a is the physical straight
length) -- they are not literal edges of the pack, so they're called out
in a formula box rather than drawn as arrows on top of a and b that don't
exist as boundaries. t (turn pitch, 75 micron) is far too small to draw
to scale next to a 24mm stack, so it's given as a text callout instead.

Output: visualization/for poster/geometry_definition.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_HERE = os.path.dirname(os.path.abspath(__file__))
_VIZ  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_VIZ)
for _p in (_ROOT, _VIZ, os.path.join(_ROOT, "physics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
from plot_3d import _racetrack_xy, _layer_colors

BLACK = "#111111"
DIM_COLOR = "#c0392b"


_LABEL_BBOX = dict(boxstyle="round,pad=0.2", facecolor="white",
                    edgecolor="none", alpha=0.85)


def _dim_line(ax, p0, p1, text, offset=0.0, va="bottom", ha="center",
              color=DIM_COLOR, fontsize=13):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.6))
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    ax.text(mx, my + offset, text, color=color, fontsize=fontsize,
            ha=ha, va=va, fontweight="bold", bbox=_LABEL_BBOX)


def make_top_view(ax):
    a_out = params.a_out * 1e3
    a_in  = min(params.a_inner_list) * 1e3
    L     = (params.b - params.a) * 1e3

    xo, yo = _racetrack_xy(params.a_out)
    xi, yi = _racetrack_xy(min(params.a_inner_list))
    ax.fill(xo * 1e3, yo * 1e3, color="#dfe6f0", zorder=1,
            label="winding pack (all layers, outer edge aligned)")
    ax.fill(xi * 1e3, yi * 1e3, color="white", zorder=2)
    ax.plot(xo * 1e3, yo * 1e3, color=BLACK, lw=1.8, zorder=3)
    ax.plot(xi * 1e3, yi * 1e3, color=BLACK, lw=1.2, ls="--", zorder=3,
            label="bore (min. inner edge)")

    # Straight length L = b - a, anchored to the flat with guide lines
    # (at these turn counts the pack is radially much thicker than L is
    # long, so the flat itself is barely visible at this scale -- the
    # dashed guides tie the dimension to where it actually is).
    ax.plot([-L, -L], [a_out, a_out + 8], color=DIM_COLOR, lw=0.8, ls=":")
    ax.plot([L, L], [a_out, a_out + 8], color=DIM_COLOR, lw=0.8, ls=":")
    _dim_line(ax, (-L, a_out + 6), (L, a_out + 6), f"L = b − a = {L:.1f} mm",
               offset=1.5)
    ax.text(0, a_out + 12,
            "(flat sides barely visible here -- pack is\n"
            "radially much thicker than L is long)",
            color=DIM_COLOR, fontsize=8.5, ha="center", va="bottom",
            style="italic")

    # Outer edge radius (from centre of the +x end cap)
    cx = L
    ax.annotate("", xy=(cx, a_out), xytext=(cx, 0),
                arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, lw=1.6))
    ax.text(cx + 3, a_out / 2, f"outer radius\n(a_out) = {a_out:.1f} mm",
            color=DIM_COLOR, fontsize=12, ha="left", va="center",
            fontweight="bold", bbox=_LABEL_BBOX)

    # Bore radius
    ax.annotate("", xy=(0, a_in), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", color="#1f6feb", lw=1.6))
    ax.text(2, a_in / 2, f"bore radius\n= {a_in:.1f} mm", color="#1f6feb",
            fontsize=12, ha="left", va="center", fontweight="bold",
            bbox=_LABEL_BBOX)

    ax.set_aspect("equal")
    ax.set_xlim(-L - a_out - 10, L + a_out + 15)
    ax.set_ylim(-a_out - 8, a_out + 28)
    ax.set_xlabel("x  (mm)"); ax.set_ylabel("y  (mm)")
    ax.set_title("Top view -- single-layer footprint", fontsize=14)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)


def make_side_view(ax):
    g = params.coil_half_gap * 1e3
    stack_h = params.n_layers * params.w * 1e3
    a_out = params.a_out * 1e3
    colors = _layer_colors()

    # Coil 1: centred on z=0. Coil 2: mirror image about z=g.
    for coil, z_c in [(1, 0.0), (2, 2 * g)]:
        for i in range(params.n_layers):
            z_bot = params.layer_z_bottoms[i] * 1e3
            z_top = params.layer_z_tops[i] * 1e3
            if coil == 2:
                z_bot, z_top = 2 * g - z_top, 2 * g - z_bot
            ax.add_patch(mpatches.Rectangle(
                (-a_out, z_bot), 2 * a_out, z_top - z_bot,
                facecolor=colors[i], alpha=0.7, edgecolor=BLACK, lw=0.6))

    # Coil half-gap (centre 1 -> midplane), drawn in the RIGHT margin,
    # clear of the stack rectangles (which only span |y| <= a_out).
    right_x = a_out + 14
    _dim_line(ax, (right_x, 0), (right_x, g), f"coil_half_gap\n= {g:.2f} mm",
              offset=0, ha="left", fontsize=11)
    ax.axhline(g, color="#555", lw=1.0, ls=":")
    ax.text(0, g + 0.4, "midplane", color="#555", fontsize=8.5,
            va="bottom", ha="center", bbox=_LABEL_BBOX)

    # Face-to-face clearance, drawn in the LEFT margin (also clear of the
    # rectangles) at a different x than coil_half_gap so the two labels
    # can't collide.
    left_x = -a_out - 12
    face_gap = 2 * (g - stack_h / 2)
    z1_top = stack_h / 2
    z2_bot = 2 * g - stack_h / 2
    ax.annotate("", xy=(left_x, z2_bot), xytext=(left_x, z1_top),
                arrowprops=dict(arrowstyle="<->", color="#1f6feb", lw=1.8))
    ax.text(left_x - 2, (z1_top + z2_bot) / 2, f"face gap = {face_gap:.2f} mm",
            color="#1f6feb", fontsize=11, ha="right", va="center",
            fontweight="bold", bbox=_LABEL_BBOX)

    # Stack height callout (coil 1)
    ax.text(0, -stack_h / 2 - 3,
            f"stack height = {stack_h:.1f} mm  ({params.n_layers} layers x w)",
            color=BLACK, fontsize=9, ha="center", va="top", bbox=_LABEL_BBOX)

    ax.set_xlim(-a_out - 40, a_out + 32)
    ax.set_ylim(-stack_h - 6, 2 * g + stack_h)
    ax.set_aspect("equal")
    ax.set_xlabel("y  (mm, radial)")
    ax.set_ylabel("z  (mm, axial)", labelpad=12)
    ax.set_title("Side view -- both coils (x = 0 cut)", fontsize=14)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#888")

    make_top_view(ax1)
    make_side_view(ax2)

    info = (
        f"a = {params.a*1e3:.2f} mm  (nominal inner radius)\n"
        f"b = {params.b*1e3:.2f} mm  (nominal outer radius)\n"
        f"t = {params.t*1e6:.0f} um  (tape pitch, radial -- not to scale)\n"
        f"w = {params.w*1e3:.1f} mm  (tape width, axial)\n"
        f"n_layers = {params.n_layers}   n_turns = {params.n_turns}\n"
        f"n_turns_total = {params.n_turns_total}   "
        f"tape length = {params.tape_length_m/1e3:.4f} km\n"
        f"I_design = {params.I_design:.1f} A/turn"
    )
    fig.text(0.5, -0.02, info, ha="center", va="top", fontsize=11,
              family="monospace", color=BLACK,
              bbox=dict(boxstyle="round", facecolor="#f2f2f2",
                        edgecolor="#888"))

    fig.suptitle("Champion racetrack coil -- geometry definition",
                  fontsize=17, y=1.02, color=BLACK)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.32)

    out = os.path.join(_HERE, "geometry_definition.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
