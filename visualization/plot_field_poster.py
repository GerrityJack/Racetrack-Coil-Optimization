"""
plot_field_poster.py
=====================
Poster-quality figure: |B| field distribution, isometric view only.
No title, no legend, minimal text (colorbar label/ticks only, large font).

2026-07-27, third pass -- REVERTED from filled solid surfaces back to a
scattered point cloud (one point per real FEM cell). The filled-surface
approach (kept in git history, not here) went through several rounds
trying to fix mplot3d rendering problems -- an interior cutaway-wedge
face blending into the curved outer wall with no shading cue between
them, and separately a "shell of the outer layer cutting into the
center" artifact from mplot3d's approximate (not true per-pixel) depth
sorting of intersecting/adjacent Poly3DCollections -- and never fully
resolved cleanly. Points sidestep the whole class of problem: there is
no opaque shell that can incorrectly occlude anything, so the wedge cut
(simply removing points whose position falls in one angular sector)
directly reveals interior structure with no special surface-ordering
logic needed at all.

Kept from the surface-era work:
  - White background, poster-styled colorbar.
  - Inferno colormap truncated below its pale-yellow tail so the
    brightest end stays vivid orange.
  - PowerNorm(gamma<1) color scale pinned to a true 0 T floor (not the
    data's sampled minimum) -- stretches the low end of the range so
    the field variation away from the bore doesn't collapse into a
    uniform-looking purple, and gives the colorbar a real 0 T tick.
  - GAP_VISUAL_STRETCH_MM: coil 2's points are shifted an extra fixed
    amount in z purely for display. Unlike the surface version, points
    need no "draw vs. query" offset split -- each point already carries
    its own correct field value directly, no spatial interpolation
    involved, so shifting its plotted position doesn't affect its color.
  - WEDGE_HALF_ANGLE_DEG: points within this angular half-width of the
    +x end-cap's outermost tip (local cap center (b-a, 0), matching
    plot_3d.py's _racetrack_xy convention) are dropped entirely, opening
    a wedge-shaped viewing corridor into the interior.
  - Lower, more side-on camera facing that same tip, so the wedge is
    actually visible instead of edge-on.

Reuses the same npz + expansion-to-full-system logic as plot_3d.py's
field_3d.png (imports its helper directly, no duplication).

Input: solve/racetrack_fields.npz (run solve/solve.py first if missing).
Output: visualization/field_3d_poster.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D   # noqa: registers projection
from scipy.spatial import cKDTree
from PIL import Image, ImageChops

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "visualization")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
from plot_3d import _expand_to_full_system, _mirror_z

# black -> deep purple -> red-orange -> vivid orange (inferno's own ramp,
# truncated before its pale-yellow tail so orange stays the brightest hue)
_POSTER_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "inferno_poster", plt.cm.inferno(np.linspace(0.0, 0.88, 256)))

GAP_VISUAL_STRETCH_MM = 12.0   # extra z-separation between the two coils,
                               # DISPLAY-ONLY (see plot_field_poster) --
                               # was 20.0, slightly reduced per feedback
WEDGE_HALF_ANGLE_DEG = 55.0    # half-width of the removed viewing wedge,
                               # centered on the right end-cap's outermost
                               # tip (local cap center (b-a, 0))
_COLOR_GAMMA = 0.5   # PowerNorm exponent < 1 stretches the LOW end of the
                     # colormap over more visual range (and compresses the
                     # high end) -- most of this design's field variation
                     # away from the bore is a comparatively small swing
                     # near the low end of the 0-10T range, which a plain
                     # linear norm renders as indistinguishable dark purple

OUTLINE_COLOR = "#333333"
OUTLINE_LW = 1.3
OUTLINE_ALPHA = 0.85

# ── translucent color fill (2026-07-27, fourth pass) ─────────────────────
# Adds a semi-transparent solid fill back underneath the point cloud and
# wireframe (both kept as-is) -- unlike the earlier, fully-opaque filled-
# surface attempt (abandoned above), a low FILL_ALPHA means any residual
# mplot3d depth-sort imprecision blends rather than produces a hard wrong-
# object-in-front artifact, and the points/wireframe stay legible on top.
FILL_ALPHA = 0.25
N_PATH = 160     # path samples around the racetrack loop (must be %4 == 0)
N_R = 6          # radial samples across a cap/shelf face
N_Z_WALL = 3     # vertical samples across one layer's inner wall
N_Z_OUTER = 60   # vertical samples across the whole stack's outer wall
N_Z_CUT = 10     # vertical samples across one layer's interior cut face
_IDW_K = 4       # neighbours averaged per color-lookup query point
_IDW_EPS = 1e-4  # regularizer [m] -- avoids a divide-by-zero at a source point


def _make_color_lookup(centroids, Bmag, norm):
    """KDTree + inverse-distance-weighted color lookup for the fill
    surfaces (points don't need this -- each carries its own value
    directly). Built from the FULL (non-wedge-filtered) cell data; the
    fill surfaces already exclude the wedge geometrically via
    _racetrack_xy_open, so the lookup itself doesn't need filtering."""
    tree = cKDTree(centroids)

    def facecolors(X_mm, Y_mm, Z_mm):
        pts_m = np.column_stack([X_mm.ravel(), Y_mm.ravel(),
                                 Z_mm.ravel()]) / 1e3
        dist, idx = tree.query(pts_m, k=_IDW_K)
        w = 1.0 / (dist + _IDW_EPS)
        vals = np.sum(w * Bmag[idx], axis=1) / np.sum(w, axis=1)
        vals = vals.reshape(X_mm.shape)
        face_vals = 0.25 * (vals[:-1, :-1] + vals[1:, :-1]
                           + vals[:-1, 1:] + vals[1:, 1:])
        return _POSTER_CMAP(norm(face_vals))

    return facecolors


def _draw_filled_stack(ax, facecolors, g, mirror, stretch_m=0.0):
    """One coil's solid geometry (outer wall, top/bottom caps, per-layer
    inner walls + shelf steps, interior cutaway-wedge faces), filled at
    FILL_ALPHA -- a translucent backdrop under the point cloud/wireframe,
    not a replacement for either.

    mirror: False for coil 1 (drawn in its own local frame, unchanged).
    True for coil 2 -- MIRRORED about the midplane (z=g), matching the
    real physics convention (physics/coil2_field.py's
    compute_both_coils_field_multilayer, and plot_3d.py's _mirror_z) --
    NOT translated by +2g, which would silently draw an asymmetric layer
    stack's layers in the wrong relative order on coil 2's side (see
    _mirror_z's docstring for the full explanation).
    stretch_m: extra DISPLAY-ONLY offset added after mirroring, pushing
    coil 2 further from the midplane (GAP_VISUAL_STRETCH_MM). Since every
    one of coil 2's mirrored points already has drawn-z > g, adding a
    constant here just pushes them further out uniformly -- the shift
    between a point's DRAWN position and its TRUE (for color-lookup)
    mirrored position is this same constant everywhere, regardless of
    which local z it came from."""
    z_shift_mm = stretch_m * 1e3   # draw - query, constant regardless of z

    def zc(z_local):
        return _mirror_z(z_local, g) + stretch_m if mirror else z_local

    a_out = params.a_out
    L = params.b - params.a
    z_top0 = zc(params.layer_z_tops[0])
    z_bot_last = zc(params.layer_z_bottoms[-1])

    def surf(X, Y, Z):
        ax.plot_surface(X, Y, Z, facecolors=facecolors(X, Y, Z - z_shift_mm),
                        rstride=1, cstride=1, linewidth=0,
                        antialiased=False, shade=False, alpha=FILL_ALPHA)

    xo, yo = _racetrack_xy_open(a_out, WEDGE_HALF_ANGLE_DEG, n=N_PATH)
    zs = np.linspace(z_top0, z_bot_last, N_Z_OUTER)
    X = np.tile(xo * 1e3, (N_Z_OUTER, 1))
    Y = np.tile(yo * 1e3, (N_Z_OUTER, 1))
    Z = np.tile((zs * 1e3).reshape(-1, 1), (1, len(xo)))
    surf(X, Y, Z)

    for i, z_face in ((0, z_top0), (params.n_layers - 1, z_bot_last)):
        a_in = params.a_inner_list[i]
        radii = np.linspace(a_in, a_out, N_R)
        path = [_racetrack_xy_open(r, WEDGE_HALF_ANGLE_DEG, n=N_PATH)
               for r in radii]
        X = np.array([p[0] for p in path]) * 1e3
        Y = np.array([p[1] for p in path]) * 1e3
        Z = np.full_like(X, z_face * 1e3)
        surf(X, Y, Z)

    for i in range(params.n_layers):
        a_in = params.a_inner_list[i]
        z_bot = zc(params.layer_z_bottoms[i])
        z_top = zc(params.layer_z_tops[i])
        xi, yi = _racetrack_xy_open(a_in, WEDGE_HALF_ANGLE_DEG, n=N_PATH)
        zs_i = np.linspace(z_bot, z_top, N_Z_WALL)
        X = np.tile(xi * 1e3, (N_Z_WALL, 1))
        Y = np.tile(yi * 1e3, (N_Z_WALL, 1))
        Z = np.tile((zs_i * 1e3).reshape(-1, 1), (1, len(xi)))
        surf(X, Y, Z)

        r_samples = np.linspace(a_in, a_out, N_R)
        z_samples = np.linspace(z_bot, z_top, N_Z_CUT)
        Rg, Zg = np.meshgrid(r_samples, z_samples)
        for theta in (np.deg2rad(WEDGE_HALF_ANGLE_DEG),
                     -np.deg2rad(WEDGE_HALF_ANGLE_DEG)):
            Xc = (L + Rg * np.cos(theta)) * 1e3
            Yc = (Rg * np.sin(theta)) * 1e3
            Zc = Zg * 1e3
            surf(Xc, Yc, Zc)

        if i < params.n_layers - 1:
            a_in_next = params.a_inner_list[i + 1]
            if abs(a_in_next - a_in) > 1e-9:
                r_lo, r_hi = sorted((a_in, a_in_next))
                radii = np.linspace(r_lo, r_hi, N_R)
                path = [_racetrack_xy_open(r, WEDGE_HALF_ANGLE_DEG, n=N_PATH)
                       for r in radii]
                X = np.array([p[0] for p in path]) * 1e3
                Y = np.array([p[1] for p in path]) * 1e3
                Z = np.full_like(X, z_bot * 1e3)
                surf(X, Y, Z)


def _wedge_keep_mask(centroids):
    """True for points to KEEP. Drops points within WEDGE_HALF_ANGLE_DEG
    of the right end-cap's outermost tip (x > L, local angle from the cap
    center (L, 0) within +-half), opening a viewing corridor into the
    interior -- same angular convention plot_3d.py's _racetrack_xy() uses
    for that cap (th=0 at the tip)."""
    L = params.b - params.a
    x, y = centroids[:, 0], centroids[:, 1]
    in_cap_b = x > L
    theta_deg = np.degrees(np.arctan2(y, x - L))
    in_wedge = in_cap_b & (np.abs(theta_deg) < WEDGE_HALF_ANGLE_DEG)
    return ~in_wedge


def _racetrack_xy_open(radius, wedge_half_angle_deg, n=300):
    """Racetrack outline (same shape as plot_3d.py's _racetrack_xy) with
    a wedge of half-angle `wedge_half_angle_deg` removed from the right
    end-cap's outermost tip (local center (L, 0), th=0) -- matches
    _wedge_keep_mask's cut exactly, so these outline curves trace the
    boundary of what's actually left in the point cloud instead of
    drawing a closed ring across the open wedge. Open ends sit at the
    exact analytic +-half angle (linspace(..., endpoint=True) on the
    boundary-adjacent piece), not rounded to the nearest fixed sample."""
    assert 0.0 < wedge_half_angle_deg < 90.0
    L = params.b - params.a
    n4 = n // 4
    half = np.deg2rad(wedge_half_angle_deg)
    step = np.pi / n4
    m = max(2, int(round((np.pi / 2 - half) / step)))

    th_upper = np.linspace(np.pi / 2, half, m + 1, endpoint=True)
    th_lower = np.linspace(-half, -np.pi / 2, m, endpoint=False)
    xBl, yBl = L + radius * np.cos(th_lower), radius * np.sin(th_lower)
    xBu, yBu = L + radius * np.cos(th_upper), radius * np.sin(th_upper)

    xA = np.linspace(-L, L, n4, endpoint=False); yA = np.full(n4, radius)
    xC = np.linspace(L, -L, n4, endpoint=False); yC = np.full(n4, -radius)
    thD = np.linspace(np.pi / 2, -np.pi / 2, n4, endpoint=False)
    xD, yD = -L - radius * np.cos(thD), -radius * np.sin(thD)

    x_open = np.concatenate([xBl, xC, xD, xA, xBu])
    y_open = np.concatenate([yBl, yC, yD, yA, yBu])
    return x_open, y_open   # intentionally NOT closed


def _draw_layer_outlines(ax, g, mirror, stretch_m=0.0):
    """Thin outline at each layer's top AND bottom z, tracing both its
    outer edge (a_out, the same for every layer -- "aligned outer edge")
    and its inner/bore edge (a_inner_list[i], which DOES vary layer to
    layer with turn count) -- so the stack reads as N distinct annular
    rings of different bore sizes instead of one undifferentiated point
    cloud. Same visual language as geometry.png's existing 3-D wireframe
    panel, adapted for the wedge cut and this poster's styling.

    mirror/stretch_m: see _draw_filled_stack's docstring -- same
    mirror-about-the-midplane convention for coil 2, not a translation."""
    a_out = params.a_out

    def zc(z_local):
        return _mirror_z(z_local, g) + stretch_m if mirror else z_local

    for i in range(params.n_layers):
        a_in = params.a_inner_list[i]
        for z_local in (params.layer_z_tops[i], params.layer_z_bottoms[i]):
            z_mm = zc(z_local) * 1e3
            for r in (a_out, a_in):
                xr, yr = _racetrack_xy_open(r, WEDGE_HALF_ANGLE_DEG)
                ax.plot(xr * 1e3, yr * 1e3, np.full_like(xr, z_mm),
                        color=OUTLINE_COLOR, lw=OUTLINE_LW,
                        alpha=OUTLINE_ALPHA)


def _draw_cut_cross_section(ax, g, mirror, stretch_m=0.0):
    """At each of the two wedge cut-boundary angles (where
    _draw_layer_outlines' ring outlines terminate -- _racetrack_xy_open's
    open ends sit exactly there), outline each layer's exposed rectangular
    cross-section: inner edge, outer edge, top, and bottom. The ring
    outlines alone stop at the cut with two disconnected loose ends (the
    outer ring's end, the inner ring's end); this connects them into a
    closed rectangle per layer, so the layer-to-layer separation is
    visible looking INTO the cut -- a real cross-section, not just the
    outer curved surface's banding.

    mirror/stretch_m: see _draw_filled_stack's docstring."""
    a_out = params.a_out
    L = params.b - params.a

    def zc(z_local):
        return _mirror_z(z_local, g) + stretch_m if mirror else z_local

    for theta in (np.deg2rad(WEDGE_HALF_ANGLE_DEG),
                 -np.deg2rad(WEDGE_HALF_ANGLE_DEG)):
        cth, sth = np.cos(theta), np.sin(theta)
        for i in range(params.n_layers):
            a_in = params.a_inner_list[i]
            z_top = zc(params.layer_z_tops[i]) * 1e3
            z_bot = zc(params.layer_z_bottoms[i]) * 1e3
            x_in, y_in = (L + a_in * cth) * 1e3, (a_in * sth) * 1e3
            x_out, y_out = (L + a_out * cth) * 1e3, (a_out * sth) * 1e3
            # inner-edge vertical, outer-edge vertical, top radial, bottom radial
            ax.plot([x_in, x_in], [y_in, y_in], [z_bot, z_top],
                    color=OUTLINE_COLOR, lw=OUTLINE_LW, alpha=OUTLINE_ALPHA)
            ax.plot([x_out, x_out], [y_out, y_out], [z_bot, z_top],
                    color=OUTLINE_COLOR, lw=OUTLINE_LW, alpha=OUTLINE_ALPHA)
            ax.plot([x_in, x_out], [y_in, y_out], [z_top, z_top],
                    color=OUTLINE_COLOR, lw=OUTLINE_LW, alpha=OUTLINE_ALPHA)
            ax.plot([x_in, x_out], [y_in, y_out], [z_bot, z_bot],
                    color=OUTLINE_COLOR, lw=OUTLINE_LW, alpha=OUTLINE_ALPHA)


def plot_field_poster(npz_data, out_name="field_3d_poster.png"):
    centroids_raw = npz_data["coil_centroids"]
    Bmag_raw = np.linalg.norm(npz_data["coil_B"], axis=1)
    g = params.coil_half_gap

    if getattr(params, "use_eighth_symmetry", False):
        centroids, Bmag = _expand_to_full_system(centroids_raw, Bmag_raw)
    else:
        centroids, Bmag = centroids_raw, Bmag_raw
        if getattr(params, "two_coil_mode", False):
            c2 = centroids_raw.copy(); c2[:, 2] = _mirror_z(c2[:, 2], g)
            centroids = np.vstack([centroids_raw, c2])
            Bmag = np.tile(Bmag_raw, 2)

    vmin, vmax = 0.0, float(Bmag.max())
    norm = mcolors.PowerNorm(gamma=_COLOR_GAMMA, vmin=vmin, vmax=vmax)
    # fill surfaces query the FULL (non-wedge-filtered) cell data -- the
    # wedge is already excluded geometrically via _racetrack_xy_open, so
    # the lookup itself doesn't need filtering, and keeping the full set
    # gives it more neighbours to interpolate from right at the cut edge.
    fill_facecolors = _make_color_lookup(centroids, Bmag, norm)

    keep = _wedge_keep_mask(centroids)
    centroids, Bmag = centroids[keep], Bmag[keep]

    # exaggerated gap: shift only coil-2's points (z > g, the midplane)
    # further out for display -- no color-lookup involved for points, so
    # (unlike the fill surfaces) there's no need to track a separate
    # "true" z for anything; each point already carries its own correct
    # field value regardless of where it's drawn.
    gap_stretch_m = GAP_VISUAL_STRETCH_MM / 1e3
    z_draw = centroids[:, 2].copy()
    z_draw[centroids[:, 2] > g] += gap_stretch_m

    fig = plt.figure(figsize=(11, 10))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color((0, 0, 0, 0))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)

    _draw_filled_stack(ax, fill_facecolors, g, mirror=False)
    _draw_filled_stack(ax, fill_facecolors, g, mirror=True,
                       stretch_m=gap_stretch_m)

    sc = ax.scatter(centroids[:, 0] * 1e3, centroids[:, 1] * 1e3,
                    z_draw * 1e3, c=Bmag, cmap=_POSTER_CMAP, norm=norm,
                    s=6, alpha=0.8, linewidths=0, rasterized=True)

    _draw_layer_outlines(ax, g, mirror=False)
    _draw_layer_outlines(ax, g, mirror=True, stretch_m=gap_stretch_m)
    _draw_cut_cross_section(ax, g, mirror=False)
    _draw_cut_cross_section(ax, g, mirror=True, stretch_m=gap_stretch_m)

    # Camera faces the wedge (the +x tip) roughly head-on: matplotlib
    # positions the camera at direction
    # (cos(elev)cos(azim), cos(elev)sin(azim), sin(elev)) from the
    # subject, so azim=0 looks straight down the -x axis at that tip. A
    # small azim offset keeps some 3-D depth instead of a perfectly flat
    # symmetric view while still looking into the opening.
    ax.view_init(elev=15, azim=-20)
    true_total_z = 2.0 * g + params.n_layers * params.w
    drawn_total_z = true_total_z + gap_stretch_m
    ax.set_box_aspect((1, 1, 0.7 * drawn_total_z / true_total_z))

    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, aspect=18)
    cb.set_label("|B|  (T)", color="black", fontsize=26, labelpad=14)
    cb.ax.tick_params(color="black", labelsize=20, length=6, width=1.2)
    plt.setp(cb.ax.get_yticklabels(), color="black")
    cb.outline.set_edgecolor("black")
    cb.outline.set_linewidth(1.0)

    out = os.path.join(params.VIZ_DIR, out_name)
    # pad_inches keeps the colorbar's bottom "0" tick label from being
    # clipped by the tight bbox -- bbox_inches='tight' alone left it
    # sitting right at (and partly past) the saved canvas edge.
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.3,
               facecolor="white")
    plt.close(fig)
    _autocrop(out, pad=50, bg_color=(255, 255, 255))
    print(f"  Wrote {out}")
    return out


def _autocrop(path, pad=40, bg_color=(255, 255, 255)):
    """3-D axes reserve a lot of dead background margin regardless of
    framing -- crop tight to the actual content (subject + colorbar) with a
    small padding, since matplotlib's bbox_inches='tight' doesn't help
    here. bg_color must match the figure's actual facecolor or the diff
    picks up the whole canvas instead of just the empty margin."""
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, bg_color)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return
    l, t, r, b = bbox
    l = max(0, l - pad); t = max(0, t - pad)
    r = min(im.width, r + pad); b = min(im.height, b + pad)
    im.crop((l, t, r, b)).save(path)


def main():
    npz_path = os.path.join(_ROOT, "solve", "racetrack_fields.npz")
    plot_field_poster(np.load(npz_path))


if __name__ == "__main__":
    main()
