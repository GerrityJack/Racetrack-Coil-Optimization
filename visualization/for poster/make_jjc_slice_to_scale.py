"""
make_jjc_slice_to_scale.py
============================
Poster/paper figure (2026-10-04): true-to-scale cross-sections of ONE coil
(coil 1; coil 2 is its mirror image about the midplane), side by side:

  left  -- straight section, the x = 0 plane (radius = y)
  right -- end-cap tip, the y = 0 plane (radius = x - L from the cap centre)

Styled like the hoop-stress render (make_optimal_configuration_updated.py):
  * light background shading: every finite-element face lying in the plane,
    filled translucently with its cell's J / Jc(B, theta);
  * points: one per element touching the plane, at its centroid projected
    onto the plane, in the same colour (opaque);
  * faint layer outlines.
J/Jc is the per-cell load used by the optimizer criterion (in-plane T-A
current over Jc(B, theta), Kim Ic extrapolation). Values are shown in %.

Minimal text by design: two panel labels, colorbar label, scale bar.

Input:  env FIELDS_NPZ (a ramp_hold_test.py snapshot with
        coil_cell_vertices and load), default the eval-166 search-method
        reproduction.
Env:    OUT_TAG, CMAP (plasma), FILL_ALPHA (0.25), POINT_ALPHA (0.9),
        POINT_SIZE (9), OUTLINE_ALPHA (0.25)
Output: visualization/for poster/jjc_slice_to_scale<OUT_TAG>.png (+ .pdf)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

_HERE = os.path.dirname(os.path.abspath(__file__))
_VIZ = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_VIZ)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "physics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
from make_jjc_cross_section_updated import _plane_faces

FIELDS_NPZ = os.environ.get("FIELDS_NPZ", os.path.join(
    _ROOT, "optimize", "runs", "ramp_hold",
    "updated_10_2_eval166_131p9A_searchmethod_s1_fields.npz"))
OUT = os.path.join(_HERE, "jjc_slice_to_scale" + os.environ.get("OUT_TAG", ""))
# plasma: its dark-blue low end shades to light lavender at low alpha,
# where magma's near-black end turns the background grey
CMAP = os.environ.get("CMAP", "plasma")
FILL_ALPHA = float(os.environ.get("FILL_ALPHA", 0.25))
POINT_ALPHA = float(os.environ.get("POINT_ALPHA", 0.9))
POINT_SIZE = float(os.environ.get("POINT_SIZE", 6))
OUTLINE_ALPHA = float(os.environ.get("OUTLINE_ALPHA", 0.25))
SCALE_BAR_MM = 10.0
# half-thickness of the slab whose elements are drawn as points (mm); the
# straight-section slab is capped at the straight half-length L
SLAB_MM = float(os.environ.get("SLAB_MM", 7.5))


def _section(verts, load, axis, r_shift):
    """Background faces in the plane coord[axis]=0 (as (radius, z) mm, with
    their cells' load %), plus points: every cell whose centroid lies in a
    slab of half-thickness SLAB_MM around the plane, projected onto it.
      straight (axis 0): |x| < min(SLAB, L) -- the straight section is
          only +-L long, radius = y;
      end cap  (axis 1): x > L and within the wedge of arc SLAB around the
          tip, radius = distance from the cap centre (L, 0)."""
    cells, faces = _plane_faces(verts, axis, 1e-9 + 1e-7 * params.b)
    r_axis = 1 - axis
    poly = np.stack([faces[:, :, r_axis] - r_shift, faces[:, :, 2]],
                    axis=-1) * 1e3
    face_vals = load[cells] * 100.0
    cen = verts.mean(axis=1)
    slab = SLAB_MM * 1e-3
    L = params.L
    if axis == 0:
        sel = cen[:, 0] < min(slab, L)
        r = cen[sel, 1]
    else:
        dx = cen[:, 0] - L
        rr = np.hypot(dx, cen[:, 1])
        ang = np.arctan2(cen[:, 1], dx)
        sel = (dx > 0) & (ang * rr < slab)
        r = rr[sel]
    pts = np.column_stack([r, cen[sel, 2]]) * 1e3
    return poly, face_vals, pts, load[sel] * 100.0


def _outlines(ax):
    """Faint per-layer outlines (radial extent a_inner_i .. a_out)."""
    ro = params.a_out * 1e3
    for i in range(params.n_layers):
        ri = params.a_inner_list[i] * 1e3
        zb, zt = params.layer_z_bottoms[i] * 1e3, params.layer_z_tops[i] * 1e3
        ax.plot([ri, ro, ro, ri, ri], [zb, zb, zt, zt, zb], color="#333333",
                lw=0.5, alpha=OUTLINE_ALPHA, zorder=3)


def main():
    d = np.load(FIELDS_NPZ)
    params.a = float(d["a"])
    params.b = float(d["b"])
    params.coil_half_gap = float(d["coil_half_gap"])
    params.n_turns = list(map(int, d["n_turns"]))
    params.recompute_derived()
    load = np.asarray(d["load"], float)
    verts = np.asarray(d["coil_cell_vertices"], float)

    sections = [_section(verts, load, 0, 0.0),          # straight, x = 0
                _section(verts, load, 1, params.L)]      # end cap, y = 0
    labels = ["straight section", "end cap"]
    norm = Normalize(0.0, 100.0)

    rlo = min(params.a_inner_list) * 1e3
    rhi = params.a_out * 1e3
    zlo = params.layer_z_bottoms[-1] * 1e3
    zhi = params.layer_z_tops[0] * 1e3
    pad = 2.0
    pw, ph = (rhi - rlo) + 2 * pad, (zhi - zlo) + 2 * pad   # mm per panel
    gap = 10.0                                               # mm between
    H_IN = 9.0
    s = H_IN / (ph + 14.0)                                   # in per mm
    fig_w = (2 * pw + gap) * s + 1.9
    fig = plt.figure(figsize=(fig_w, H_IN), facecolor="white")
    bottom = 8.0 * s / H_IN
    axes = []
    for k in range(2):
        left = (0.25 + k * (pw + gap) * s) / fig_w
        axes.append(fig.add_axes([left, bottom, pw * s / fig_w, ph * s / H_IN]))

    for ax, (poly, fvals, pts, pvals), lab in zip(axes, sections, labels):
        ax.add_collection(PolyCollection(
            poly, array=fvals, cmap=CMAP, norm=norm, alpha=FILL_ALPHA,
            edgecolors="none", zorder=1))
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=pvals, cmap=CMAP, norm=norm,
                        s=POINT_SIZE, alpha=POINT_ALPHA, linewidths=0,
                        zorder=4)
        _outlines(ax)
        ax.set_xlim(rlo - pad, rhi + pad)
        ax.set_ylim(zlo - pad, zhi + pad)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0.5, 1.0, lab, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=13)

    # scale bar under the left panel
    ax = axes[0]
    x0 = rlo
    yb = zlo - pad - 3.0
    ax.plot([x0, x0 + SCALE_BAR_MM], [yb, yb], color="black", lw=2.2,
            solid_capstyle="butt", clip_on=False)
    ax.text(x0 + SCALE_BAR_MM / 2, yb - 1.5, f"{SCALE_BAR_MM:.0f} mm",
            ha="center", va="top", fontsize=11, clip_on=False)

    l, b0, w, h = axes[1].get_position().bounds
    cax = fig.add_axes([l + w + 0.35 / fig_w, b0 + 0.15 * h,
                        0.22 / fig_w, 0.7 * h])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"$J\,/\,J_c$  (%)", fontsize=14)
    cb.set_ticks(np.arange(0, 101, 20))
    cb.ax.tick_params(labelsize=11)

    for ext in ("png", "pdf"):
        # temp file + atomic rename: an in-place overwrite while the image
        # was open in a viewer corrupted a PNG on 2026-10-04
        tmp = f"{OUT}.tmp.{ext}"
        fig.savefig(tmp, dpi=300, facecolor="white",
                    bbox_inches="tight", pad_inches=0.15)
        os.replace(tmp, f"{OUT}.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}.png/.pdf  (points: straight {len(sections[0][2])}, "
          f"end cap {len(sections[1][2])}; max {sections[0][3].max():.1f} / "
          f"{sections[1][3].max():.1f} %)")


if __name__ == "__main__":
    main()
