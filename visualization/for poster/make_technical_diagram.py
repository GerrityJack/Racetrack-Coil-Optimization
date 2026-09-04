"""
make_technical_diagram.py
===========================
Technical line-diagram of the racetrack coil-pair SHAPE, styled to match
a specific reference sketch: thin black multi-loop coil outlines (no
fill), tick-marked (not arrowhead) dimension lines, and color-coded
labels -- blue for a/b/Midplane, green for Coil Gap, black for turn
count and the target-region box dimensions. No panel titles, no
crosshairs, no center-guide lines -- matches the reference's minimal
look. Meant to sit next to a constraints table (a & b constant across
layers, turn-pairing, target-region field/uniformity, hoop stress, bend
radius, critical-current fraction, coil-to-coil gap) -- only the
GEOMETRIC variables from that table have a simple spatial location to
draw; hoop stress / bend radius / critical-current are distributed
physics constraints, not a single point on the shape, so they are
deliberately NOT depicted here.

Three vertically stacked panels, EACH GIVEN ITS OWN ROW HEIGHT sized to
match its own content's aspect ratio exactly (see _row_heights() /
main()) -- avoids the letterbox blank-margin waste of forcing every
panel through one shared aspect ratio.
  1. TOP VIEW    -- several concentric loops (many turns, viewed down
                     the coil axis), b horizontal, a vertical, turn
                     count leader.
  2. SIDE VIEW    -- two coils on a common center axis (foreshortened,
                     not isometric -- see panel_side's docstring), each
                     drawn as just 2 loops (a thin tape edge-on, not
                     many turns), gap dimensioned between their facing
                     inner edges using the TRUE face-to-face value
                     (3.4mm) -- what the "Gap between coils must exceed
                     3mm" constraint actually refers to.
  3. MIDPLANE VIEW -- dashed ghost footprint, unfilled target-region box
                     labeled directly beneath it.

Output: visualization/for poster/technical_diagram.png (+ .svg)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_VIZ = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_VIZ)
for _p in (_ROOT, os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import params
import opt_config as cfg

plt.rcParams["font.family"] = "DejaVu Sans"

# ── palette ──────────────────────────────────────────────────────────────
BLACK = "#000000"         # coil/box geometry
GRAY_GHOST = "#8A8A8A"    # dashed reference outline
LABEL_COLOR = "#1B4FA0"   # every text label uses this one color

LABEL_FONTSIZE = 12
LABEL_FONT = "DejaVu Sans"

# The true straight length (b - a = 5.4mm) is tiny next to the radius
# (a=26mm) -- drawn to true scale the "racetrack" reads as a plain
# circle. This figure is a SHAPE schematic, not a to-scale drawing, so
# the straight section is exaggerated for legibility (matched to the
# reference sketch's own elongated proportions, L/b ~ 1.4) -- every
# LABELED value is still the real number.
STYLIZE_STRAIGHT_FRAC = 1.40
FORESHORTEN = 0.42   # side-view vertical squash factor, see panel_side()

FIG_W = 8.0   # inches; each row's height is DERIVED from this (see
              # _row_heights()), not fixed independently


def _racetrack_xy(radius, L, n=240):
    th = np.linspace(np.pi / 2, -np.pi / 2, n // 4, endpoint=False)
    x = np.concatenate([np.linspace(-L, L, n // 4, endpoint=False),
                        L + radius * np.cos(th),
                        np.linspace(L, -L, n // 4, endpoint=False),
                        -L + radius * np.cos(th + np.pi)])
    y = np.concatenate([np.full(n // 4, radius), radius * np.sin(th),
                        np.full(n // 4, -radius), radius * np.sin(th + np.pi)])
    return np.column_stack([np.append(x, x[0]), np.append(y, y[0])])


# ── content bounds, computed ONCE and shared between the row-height
# sizing pass (main()) and each panel's own axis limits, so the two can
# never disagree ─────────────────────────────────────────────────────────

def _bounds_top():
    b_mm = params.b * 1e3
    L = STYLIZE_STRAIGHT_FRAC * b_mm
    # "Major Radius (b) = .. mm" is long -- placed OUTSIDE the coil to
    # the right of its tip, so x_half must clear the full label width,
    # not just the tip itself.
    x_half = L + b_mm + 115
    y_hi = b_mm * 1.45             # room for "Minor Radius (a)" above the coil
    y_lo = -b_mm * 1.15
    return (-x_half, x_half, y_lo, y_hi)


# Each physical coil is itself a 6-layer, 3-double-pancake-pair stack
# (params.n_layers=6, n_turns paired [382,382,478,478,3,3]) -- NOT a
# single thin tape, which the previous version of this panel drew.
# _layer_offsets() gives each layer's small local z-offset (in the same
# display units as everything else) within its own coil assembly,
# grouped so the pairing is visible: small gap within a pair, bigger gap
# between pairs.
N_PAIRS = 3
LAYER_STACK_FRAC = 0.30   # stack half-height as a fraction of ring_half_h


def _layer_offsets(ring_half_h):
    half_h = ring_half_h * LAYER_STACK_FRAC
    pair_centers = np.linspace(-half_h, half_h, N_PAIRS)
    intra_gap = half_h * 0.32
    offsets = []
    for pc in pair_centers:
        offsets += [pc - intra_gap / 2, pc + intra_gap / 2]
    return offsets   # 2*N_PAIRS values, symmetric about 0


def _bounds_side():
    b_mm = params.b * 1e3
    L = STYLIZE_STRAIGHT_FRAC * b_mm
    gap_draw_mm = 0.55 * b_mm
    ring_half_h = b_mm * FORESHORTEN
    stack_half = max(np.abs(_layer_offsets(ring_half_h)))
    y_bot = -(gap_draw_mm / 2 + ring_half_h + stack_half)
    y_top = -y_bot
    dim_x = L + b_mm * 1.08
    # x_half is symmetric, so it must cover BOTH the "Coil Gap" label on
    # the right and the two-line layer-count label on the left.
    x_half = dim_x + 105
    y_lo = y_bot - ring_half_h - stack_half - 14   # room for the layer label
    y_hi = y_top + ring_half_h + stack_half + 8
    return (-x_half, x_half, y_lo, y_hi)


def _bounds_midplane():
    b_mm = params.b * 1e3
    L = STYLIZE_STRAIGHT_FRAC * b_mm
    x_half = L + b_mm * 1.15
    y_hi = b_mm * 1.15
    y_lo = -b_mm * 1.55           # room for the "Midplane" label below
    return (-x_half, x_half, y_lo, y_hi)


def _aspect(bounds):
    x_lo, x_hi, y_lo, y_hi = bounds
    return (x_hi - x_lo) / (y_hi - y_lo)


def _row_heights():
    """Each row's height in inches, chosen so that FIG_W at that
    panel's own content aspect ratio fills the row with ZERO letterbox
    waste."""
    return [FIG_W / _aspect(b) for b in
           (_bounds_top(), _bounds_side(), _bounds_midplane())]


def _outline(ax, verts, lw=1.4, color=BLACK, zorder=5, ls="-"):
    ax.plot(verts[:, 0], verts[:, 1], color=color, lw=lw, zorder=zorder,
            ls=ls, solid_joinstyle="round")


def _draw_loops(ax, a_mm, b_mm, L, n_loops, zorder=3, lw=1.3):
    """n_loops concentric racetrack outlines between radius a and b, no
    fill -- reads as actual wound turns (top view, several loops) or a
    thin tape's own two edges (side view, n_loops=2), matching the
    reference sketch's literal line-art style instead of a filled band."""
    for r in np.linspace(a_mm, b_mm, max(n_loops, 2)):
        pts = _racetrack_xy(r, L)
        ax.plot(pts[:, 0], pts[:, 1], color=BLACK, lw=lw, zorder=zorder)


def _dim_line(ax, p0, p1, text, color=LABEL_COLOR, offset=(0, 0), ha="center",
              va="center", tick=4.5, zorder=8):
    """Tick-marked dimension line (I-beam ends, no arrowheads) -- matches
    the reference sketch's style. Bare colored text, no box background."""
    p0 = np.array(p0, dtype=float); p1 = np.array(p1, dtype=float)
    d = p1 - p0
    length = np.linalg.norm(d)
    perp = np.array([-d[1], d[0]]) / length if length > 1e-9 else np.array([0, 1])
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=1.1, zorder=zorder)
    for p in (p0, p1):
        t0, t1 = p - perp * tick, p + perp * tick
        ax.plot([t0[0], t1[0]], [t0[1], t1[1]], color=color, lw=1.1,
               zorder=zorder)
    mx, my = (p0[0] + p1[0]) / 2 + offset[0], (p0[1] + p1[1]) / 2 + offset[1]
    ax.text(mx, my, text, color=color, fontsize=LABEL_FONTSIZE, ha=ha,
           va=va, zorder=zorder + 1, family=LABEL_FONT)


def _clean_axes(ax, bounds):
    x_lo, x_hi, y_lo, y_hi = bounds
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_facecolor("white")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_aspect("equal")


# ═══════════════════════════════════════════════════════════════════════
# Panel 1 -- TOP VIEW
# ═══════════════════════════════════════════════════════════════════════

def panel_top(ax):
    a_mm, b_mm = params.a * 1e3, params.b * 1e3
    L = STYLIZE_STRAIGHT_FRAC * b_mm
    bounds = _bounds_top()

    _draw_loops(ax, a_mm, b_mm, L, n_loops=5, zorder=3)

    # b: horizontal, from the end-cap's own local center (L,0) to its
    # tip (L+b,0) -- length is exactly b. Label placed OUTSIDE the coil,
    # to the right of the tip (in open space), rather than centered over
    # its own arrow -- that arrow runs under the CURVED cap, and text
    # long enough to say "Major Radius (b) = .. mm" doesn't fit under a
    # curve without clipping it.
    mid_b = (L + b_mm / 2) - (L + b_mm)   # arrow midpoint, tip-relative
    _dim_line(ax, (L, 0), (L + b_mm, 0), f"Major Radius (b) = {b_mm:.1f} mm",
             offset=(-mid_b + 10, 0), ha="left")
    # a: vertical, center to bore edge. Label's BOTTOM edge (va="bottom")
    # placed a clear 6mm above the coil's own flat top edge (y=b_mm).
    a_mid_y = a_mm / 2
    _dim_line(ax, (0, 0), (0, a_mm), f"Minor Radius (a) = {a_mm:.1f} mm",
             offset=(0, (b_mm + 6) - a_mid_y), ha="center", va="bottom")

    _clean_axes(ax, bounds)


# ═══════════════════════════════════════════════════════════════════════
# Panel 2 -- SIDE VIEW
# ═══════════════════════════════════════════════════════════════════════

# A true isometric projection was tried and rejected: mixing x/y tilts
# each ring into a slanted oval, and stacking two tilted ovals along z
# reads as a diagonal offset, not a shared vertical axis. A foreshortened
# (squashed) front-on view -- the standard convention for drawing
# coaxial rings -- fixes this: both rings keep IDENTICAL x-coordinates,
# so they are provably centered on the same vertical axis.

def _squash(xy, y_center):
    out = xy.copy()
    out[:, 1] = out[:, 1] * FORESHORTEN + y_center
    return out


def panel_side(ax):
    b_mm = params.b * 1e3
    L = STYLIZE_STRAIGHT_FRAC * b_mm
    bounds = _bounds_side()
    # TRUE face-to-face gap -- the number the constraint table's "Gap
    # between coils must exceed 3 mm" actually refers to (matches
    # opt_config.py's MIN_COIL_GAP_M=3mm floor), not a center-to-center
    # distance.
    face_gap_mm = 2 * (params.coil_half_gap
                       - params.n_layers * params.w / 2) * 1e3
    # Drawn separation is still exaggerated for legibility -- only the
    # LABEL states the true value.
    gap_draw_mm = 0.55 * b_mm

    outer_flat = _racetrack_xy(b_mm, L)
    ring_half_h = b_mm * FORESHORTEN
    offsets = _layer_offsets(ring_half_h)   # 6 layers, 3 pairs
    stack_half = max(np.abs(offsets))

    y_bot = -(gap_draw_mm / 2 + ring_half_h + stack_half)
    y_top = -y_bot

    # Each coil assembly drawn as its own 6-layer, 3-double-pancake-pair
    # stack -- "All layers have the same outer shape (a & b constant)"
    # is exactly why every layer here is the SAME radius-b loop, just at
    # a different small z-offset within its own coil.
    for y_c in (y_bot, y_top):
        for off in offsets:
            pts = _squash(outer_flat, y_c + off)
            ax.plot(pts[:, 0], pts[:, 1], color=BLACK, lw=1.1, zorder=3)

    face_bot = y_bot + ring_half_h + stack_half   # top edge of lower coil
    face_top = y_top - ring_half_h - stack_half   # bottom edge of upper coil

    dim_x = L + b_mm * 1.08
    _dim_line(ax, (dim_x, face_bot), (dim_x, face_top),
             f"Coil Gap\n{face_gap_mm:.1f} mm", offset=(40, 0), ha="left")

    # Layer-count label, with a small bracket spanning one coil's stack.
    bracket_x = -(L + b_mm * 1.02)
    lo, hi = y_bot - ring_half_h - stack_half, y_bot + ring_half_h + stack_half
    _dim_line(ax, (bracket_x, lo), (bracket_x, hi),
             f"{params.n_layers} layers\n({N_PAIRS} double-pancake pairs)",
             offset=(-8, 0), ha="right")

    _clean_axes(ax, bounds)


# ═══════════════════════════════════════════════════════════════════════
# Panel 3 -- MIDPLANE VIEW
# ═══════════════════════════════════════════════════════════════════════

def panel_midplane(ax):
    a_mm, b_mm = params.a * 1e3, params.b * 1e3
    L = STYLIZE_STRAIGHT_FRAC * b_mm
    bounds = _bounds_midplane()
    outer = _racetrack_xy(b_mm, L)

    _outline(ax, outer, lw=1.3, color=GRAY_GHOST, zorder=2, ls=(0, (6, 4)))

    # Unfilled target-region box, labeled directly beneath -- no arrows,
    # matching the reference sketch.
    rx, ry = cfg.TARGET_X_M * 1e3, cfg.TARGET_Y_M * 1e3
    box = np.array([[-rx / 2, -ry / 2], [rx / 2, -ry / 2],
                    [rx / 2, ry / 2], [-rx / 2, ry / 2], [-rx / 2, -ry / 2]])
    _outline(ax, box, lw=1.4, color=BLACK, zorder=5)
    ax.text(0, ry / 2 + 5, "Target Region", color=LABEL_COLOR,
           fontsize=LABEL_FONTSIZE, ha="center", va="bottom",
           family=LABEL_FONT, zorder=9)
    ax.text(0, -ry / 2 - 6, f"{rx:.0f} mm x {ry:.0f} mm", color=LABEL_COLOR,
           fontsize=LABEL_FONTSIZE, ha="center", va="top", family=LABEL_FONT,
           zorder=9)

    ax.text(L * 0.55, -b_mm * 1.35, "Midplane", color=LABEL_COLOR,
           fontsize=LABEL_FONTSIZE, ha="center", va="top", family=LABEL_FONT,
           zorder=9)

    _clean_axes(ax, bounds)


def main():
    # hspace=0 deliberately: gridspec's hspace subtracts a roughly EQUAL
    # absolute gap from every row regardless of that row's own height,
    # which would distort the exact height_ratios computed above. Each
    # panel's own bounds already reserve enough margin around its
    # content for visual separation between panels.
    row_h = _row_heights()
    fig = plt.figure(figsize=(FIG_W, sum(row_h) + 0.3))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 1, height_ratios=row_h, hspace=0.0,
                          left=0.01, right=0.99, top=0.99, bottom=0.01)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]

    panel_top(axes[0])
    panel_side(axes[1])
    panel_midplane(axes[2])

    out_png = os.path.join(_HERE, "technical_diagram.png")
    out_svg = os.path.join(_HERE, "technical_diagram.svg")
    fig.savefig(out_png, dpi=220, facecolor="white", bbox_inches="tight")
    fig.savefig(out_svg, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_svg}")


if __name__ == "__main__":
    main()
