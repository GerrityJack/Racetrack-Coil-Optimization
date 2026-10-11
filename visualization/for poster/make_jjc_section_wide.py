"""
make_jjc_section_wide.py
==========================
Poster/paper figure (2026-10-04): one winding cross-section of one coil,
laid out horizontally (axial position across, radius up), shaded by
J / Jc(B, theta) on a LOG colour scale, with the radial direction
exaggerated (STRETCH, default 2x) so the thin winding's area reads clearly.
Both axes keep mm tick labels, so the stretch is explicit.

Shading: the finite-element faces lying in the cut plane carry their
cells' per-cell load (the optimizer criterion's |J_inplane| / Jc, Kim Ic
extrapolation). For a smooth look (SMOOTH=1, default) each node gets the
mean of log(load) over its faces and colours are Gouraud-interpolated
across each triangle; this is done per pancake, so nothing blends across
pancake boundaries. SMOOTH=0 shows the raw per-element values. Faint
lines mark the pancakes. The radius axis is inverted so the flat outer
edge of the winding (all pancakes end at a_out) is at the bottom.

  SECTION=straight (default): the x = 0 plane, radius = y
  SECTION=endcap:             the y = 0 plane, radius = x - L

Input:  env FIELDS_NPZ (ramp_hold_test.py snapshot with coil_cell_vertices
        and load); default the eval-166 search-method reproduction.
Env:    OUT_TAG, SECTION, STRETCH (2.0), CMAP (plasma),
        LOG_MIN_PCT (1.0) -- lower end of the log scale
Output: visualization/for poster/jjc_section_wide<OUT_TAG>.png (+ .pdf)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, FixedFormatter

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
SECTION = os.environ.get("SECTION", "straight")
OUT = os.path.join(_HERE, "jjc_section_wide" + os.environ.get("OUT_TAG", "")
                   + ("" if SECTION == "straight" else "_" + SECTION))
STRETCH = float(os.environ.get("STRETCH", 2.0))
CMAP = os.environ.get("CMAP", "plasma")
LOG_MIN_PCT = float(os.environ.get("LOG_MIN_PCT", 1.0))
SMOOTH = os.environ.get("SMOOTH", "1") != "0"


def _smooth_layers(ax, poly, vals, norm):
    """Gouraud-shade each pancake separately from node values = mean of
    log10(face value) over the faces sharing the node (within that
    pancake). Returns the last mappable (for the colorbar)."""
    import matplotlib.tri as mtri
    z0 = params.layer_z_bottoms[-1] * 1e3
    edges = [(zb * 1e3 - z0, zt * 1e3 - z0) for zb, zt in
             zip(params.layer_z_bottoms, params.layer_z_tops)]
    cz = poly[:, :, 0].mean(axis=1)
    m = None
    for lo, hi in edges:
        sel = (cz >= lo) & (cz < hi)
        if not sel.any():
            continue
        P = poly[sel].reshape(-1, 2)
        key = np.round(P, 6)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        tri = inv.reshape(-1, 3)
        lv = np.repeat(np.log10(vals[sel]), 3)
        acc = np.bincount(inv, weights=lv, minlength=len(uniq))
        cnt = np.bincount(inv, minlength=len(uniq))
        node = 10.0 ** (acc / np.maximum(cnt, 1))
        T = mtri.Triangulation(uniq[:, 0], uniq[:, 1], tri)
        m = ax.tripcolor(T, node, shading="gouraud", cmap=CMAP, norm=norm,
                         rasterized=True)
    return m


def main():
    d = np.load(FIELDS_NPZ)
    params.a = float(d["a"])
    params.b = float(d["b"])
    params.coil_half_gap = float(d["coil_half_gap"])
    params.n_turns = list(map(int, d["n_turns"]))
    params.recompute_derived()
    load = np.asarray(d["load"], float)
    verts = np.asarray(d["coil_cell_vertices"], float)

    axis, r_shift = (0, 0.0) if SECTION == "straight" else (1, params.L)
    cells, faces = _plane_faces(verts, axis, 1e-9 + 1e-7 * params.b)
    z0 = params.layer_z_bottoms[-1]
    # horizontal: axial position from the coil's outer end; vertical: radius
    poly = np.stack([(faces[:, :, 2] - z0) * 1e3,
                     (faces[:, :, 1 - axis] - r_shift) * 1e3], axis=-1)
    vals = np.clip(load[cells] * 100.0, LOG_MIN_PCT, None)
    norm = LogNorm(vmin=LOG_MIN_PCT, vmax=100.0)

    zlen = params.n_layers * params.w * 1e3
    rlo = min(params.a_inner_list) * 1e3
    rhi = params.a_out * 1e3
    W_IN = 13.0
    ax_w = 0.84
    ax_h_in = W_IN * ax_w * (rhi - rlo) * STRETCH / zlen
    H_IN = ax_h_in + 1.3
    fig = plt.figure(figsize=(W_IN, H_IN), facecolor="white")
    ax = fig.add_axes([0.07, 0.95 / H_IN, ax_w, ax_h_in / H_IN])

    if SMOOTH:
        pc = _smooth_layers(ax, poly, vals, norm)
    else:
        pc = PolyCollection(poly, array=vals, cmap=CMAP, norm=norm,
                            edgecolors="face", linewidths=0.2)
        ax.add_collection(pc)
    # faint pancake boundaries, heavier between double pancakes
    for i, zt in enumerate(params.layer_z_tops[1:], start=1):
        ax.axvline((zt - z0) * 1e3, color="white",
                   lw=1.0 if i % 2 == 0 else 0.35,
                   alpha=0.8 if i % 2 == 0 else 0.45)
    ax.set_xlim(0.0, zlen)
    ax.set_ylim(rhi, rlo)           # inverted: flat outer edge at the bottom
    ax.set_aspect(STRETCH)          # 1 mm radial drawn STRETCH x 1 mm axial
    ax.set_facecolor("white")
    ax.set_xlabel("axial position (mm)", fontsize=20)
    ax.set_ylabel("radius (mm)", fontsize=20)
    ax.tick_params(labelsize=17)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    cax = fig.add_axes([0.07 + ax_w + 0.012, 0.95 / H_IN, 0.012,
                        ax_h_in / H_IN])
    cb = fig.colorbar(pc, cax=cax)
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100) if t >= LOG_MIN_PCT]
    cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
    cb.ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    cb.ax.yaxis.set_minor_locator(FixedLocator([]))
    cb.set_label(r"$J\,/\,J_c$  (%)", fontsize=20)
    cb.ax.tick_params(labelsize=17)

    for ext in ("png", "pdf"):
        # temp file + atomic rename (an in-place overwrite while the image
        # was open in a viewer corrupted a PNG on 2026-10-04)
        tmp = f"{OUT}.tmp.{ext}"
        fig.savefig(tmp, dpi=300, facecolor="white", bbox_inches="tight",
                    pad_inches=0.1)
        os.replace(tmp, f"{OUT}.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT}.png/.pdf  ({SECTION}, {len(cells)} faces, "
          f"load {load[cells].min()*100:.2f}-{load[cells].max()*100:.1f} %)")


if __name__ == "__main__":
    main()
