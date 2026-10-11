"""
make_jjc_cross_section_updated.py
===================================
Poster figure (2026-10-04): the fraction of the local critical current
density, J / Jc(B, theta), at every point of the winding cross-section of
the updated 28-layer design (optimize/runs/updated_10_2/, eval 166) at
its I_op = 131.87 A, at the moment a 2 h constant-power ramp ends.

What is plotted -- exactly the FEM solution, no interpolation:
  The two natural cuts of a racetrack are symmetry planes of the FEM
  domain: x = 0 (through the middle of the straight legs) and y = 0
  (through the end-cap tips).  Every tetrahedron face lying in such a
  plane is drawn as a filled triangle in the colour of its cell's DG0
  value of |J_inplane| / Jc(B, theta) -- the same per-cell "load" the
  optimizer's criterion uses (optimize/ta_safe_current._local_margin,
  Kim Ic(B) extrapolation, in-plane projected T-A current).  So the
  picture shows the solution at the resolution it was computed at.

  Coil 1 only: coil 2 is its mirror image about the midplane z = g,
  which the FEM imposes exactly (PMC plane), so it carries no extra
  information.  Axial position is horizontal and radial depth vertical
  so the 28-pancake stack reads as a wide strip.

Solution: T-A, corrected resistivity (2026-10-04), single 4 h
backward-Euler step from the virgin state (the constant-power-ramp
surrogate), rho-floor continued 1.0 -> 0.6, medium mesh -- from
optimize/studies/ramp_hold_test.py, RH_TAG updated_10_2_eval166_131p9A_4h.
Caveat printed on the figure: in-plane cells are ~7 mm (0.25 x the
deepest pancake), so the radial structure is resolved by ~4 cells.

Input:  optimize/runs/ramp_hold/updated_10_2_eval166_131p9A_4h_s4_fields.npz
        (or env FIELDS_NPZ; must contain coil_cell_vertices and load)
Output: visualization/for poster/jjc_cross_section_updated.png (+ .pdf)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe

_HERE = os.path.dirname(os.path.abspath(__file__))
_VIZ = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_VIZ)
for _p in (_ROOT, os.path.join(_ROOT, "physics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params

FIELDS_NPZ = os.environ.get("FIELDS_NPZ", os.path.join(
    _ROOT, "optimize", "runs", "ramp_hold",
    "updated_10_2_eval166_131p9A_4h_s4_fields.npz"))
# OUT_TAG distinguishes variants, e.g. OUT_TAG=_searchmethod for the
# pre-2026-10-04 (search-time) solver; FOOTNOTE=0 drops the methods note
# (e.g. when the paper caption carries it)
OUT = os.path.join(_HERE, "jjc_cross_section_updated"
                   + os.environ.get("OUT_TAG", ""))
FOOTNOTE = os.environ.get("FOOTNOTE", "1") != "0"
# rho-floor of the snapshot (ramp_hold_test step 3 = 0.7, step 4 = 0.6)
RHO_FLOOR = os.environ.get("RHO_FLOOR", "0.6")

CMAP = "magma"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def _plane_faces(verts, axis, tol):
    """(cell index, 3x3 face vertices) for every tet face lying in the
    plane coord[axis] = 0."""
    on = np.abs(verts[:, :, axis]) < tol                    # (N, 4)
    out_c, out_f = [], []
    for skip in range(4):                                   # face = 3 of 4 verts
        idx = [k for k in range(4) if k != skip]
        hit = np.all(on[:, idx], axis=1)
        if hit.any():
            out_c.append(np.nonzero(hit)[0])
            out_f.append(verts[hit][:, idx, :])
    return np.concatenate(out_c), np.concatenate(out_f)


QUAL_MIN = float(os.environ.get("QUAL_MIN", 1e-3))


def _tet_quality(verts):
    """Normalised tet shape quality 6*sqrt(2)*V / l_max^3 (1 = regular)."""
    e = verts[:, :, None, :] - verts[:, None, :, :]
    lmax = np.sqrt((e ** 2).sum(-1)).reshape(len(verts), -1).max(1)
    vol = np.abs(np.linalg.det(verts[:, 1:] - verts[:, :1])) / 6.0
    return 6.0 * np.sqrt(2.0) * vol / lmax ** 3


def _load():
    d = np.load(FIELDS_NPZ)
    params.a = float(d["a"])
    params.b = float(d["b"])
    params.coil_half_gap = float(d["coil_half_gap"])
    params.n_turns = list(map(int, d["n_turns"]))
    params.recompute_derived()
    return d


def _layer_lines(ax, z0):
    """Pancake boundaries: thin within a double pancake, heavier between."""
    for i, zt in enumerate(params.layer_z_tops[1:], start=1):
        x = (zt - z0) * 1e3
        joint = (i % 2 == 0)
        ax.axvline(x, color="white", lw=1.1 if joint else 0.35,
                   alpha=0.95 if joint else 0.55, zorder=3)


def _panel(ax, faces_uz, vals, norm, title, r_label, z0):
    polys = faces_uz * 1e3
    pc = PolyCollection(polys, array=vals, cmap=CMAP, norm=norm,
                        edgecolors="face", linewidths=0.15, zorder=2)
    ax.add_collection(pc)
    _layer_lines(ax, z0)
    ax.set_xlim(0.0, params.n_layers * params.w * 1e3)
    rlo = min(params.a_inner_list) * 1e3
    rhi = params.a_out * 1e3
    pad = 0.04 * (rhi - rlo)
    ax.set_ylim(rlo - pad, rhi + pad)
    ax.set_facecolor("#f4f4f4")
    ax.set_ylabel(r_label)
    ax.set_title(title, loc="left", fontsize=12.5, color=INK, pad=18,
                 fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(direction="out", length=3.5)
    return pc


def _inset_topview(fig, rect):
    """Small racetrack top view showing where the two cuts are."""
    ax = fig.add_axes(rect)
    L, ro, ri = params.L, params.a_out, min(params.a_inner_list)

    def stadium(r):
        th_r = np.linspace(-np.pi / 2, np.pi / 2, 100)       # +x cap
        th_l = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)    # -x cap
        xs = np.concatenate([L + r * np.cos(th_r), -L + r * np.cos(th_l)])
        ys = np.concatenate([r * np.sin(th_r), r * np.sin(th_l)])
        return np.append(xs, xs[0]) * 1e3, np.append(ys, ys[0]) * 1e3

    xo, yo = stadium(ro)
    xi, yi = stadium(ri)
    ax.fill(xo, yo, color="#cfcfcf", zorder=1)
    ax.fill(xi, yi, color="white", zorder=2)
    ax.plot(xo, yo, color=INK, lw=0.9, zorder=3)
    ax.plot(xi, yi, color=INK, lw=0.9, zorder=3)
    e = (L + ro) * 1e3
    ax.plot([0, 0], [-ro * 1e3 * 1.2, ro * 1e3 * 1.2], color="#2b8cbe",
            lw=1.8, zorder=4)
    ax.plot([-e * 1.12, e * 1.12], [0, 0], color="#e6550d", lw=1.8,
            ls=(0, (4, 2)), zorder=4)
    ax.text(0, ro * 1e3 * 1.27, "x = 0", color="#2b8cbe", ha="center",
            va="bottom", fontsize=8, fontweight="bold")
    ax.text(e * 1.16, 0, "y = 0", color="#e6550d", ha="left", va="center",
            fontsize=8, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-e * 1.15, e * 1.6)
    ax.set_ylim(-ro * 1e3 * 1.3, ro * 1e3 * 1.6)
    ax.axis("off")
    ax.text(-0.04, 0.5, "cut planes\n(top view)", transform=ax.transAxes,
            ha="right", va="center", fontsize=8, color=MUTED)


def main():
    d = _load()
    load = np.asarray(d["load"], float)
    verts = np.asarray(d["coil_cell_vertices"], float)
    I = float(d["I_solved"])
    tol = 1e-9 + 1e-7 * params.b

    # x = 0 plane: straight-leg section (radial coordinate = y)
    c_a, f_a = _plane_faces(verts, 0, tol)
    # y = 0 plane: end-cap tip section (radial coordinate = x - L)
    c_b, f_b = _plane_faces(verts, 1, tol)
    z0 = params.layer_z_bottoms[-1]                  # bottom of the stack

    def uz(faces, r_axis, r_shift):
        u = faces[:, :, r_axis] - r_shift
        z = faces[:, :, 2] - z0
        return np.stack([z, u], axis=-1)            # horizontal z, vertical r

    faces_a = uz(f_a, 1, 0.0)
    faces_b = uz(f_b, 0, params.L)
    va, vb = load[c_a], load[c_b]

    norm = Normalize(vmin=0.0, vmax=1.0)
    fig = plt.figure(figsize=(13.0, 8.0), facecolor="white")
    ax_a = fig.add_axes([0.075, 0.52, 0.80, 0.285])
    ax_b = fig.add_axes([0.075, 0.15, 0.80, 0.285], sharex=ax_a)
    cax = fig.add_axes([0.895, 0.15, 0.016, 0.655])

    pc = _panel(ax_a, faces_a, va, norm,
                "Straight-leg cross-section  (x = 0)",
                "radius from axis (mm)", z0)
    _panel(ax_b, faces_b, vb, norm,
           "End-cap cross-section  (y = 0, through the cap tip)",
           "radius from cap centre (mm)", z0)
    ax_b.set_xlabel("axial position through the stack of coil 1 (mm)"
                    "   —   midplane (coil 2 mirrored) at the right edge")
    plt.setp(ax_a.get_xticklabels(), visible=False)

    # double-pancake numbering along the top of both panels, counted
    # from the axially outermost pair (left) to the one at the midplane
    n_pairs = params.n_layers // 2
    for ax in (ax_a, ax_b):
        for i in range(0, params.n_layers, 2):
            zc = (params.layer_z_bottoms[i + 1] - z0 + params.w) * 1e3
            ax.text(zc, 1.01, f"DP{n_pairs - i // 2}",
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=7.2, color=MUTED)

    # worst cell in each section, and overall
    for ax, faces, vals in ((ax_a, faces_a, va), (ax_b, faces_b, vb)):
        k = int(np.argmax(vals))
        cz, cr = faces[k].mean(axis=0) * 1e3
        ax.plot(cz, cr, marker="o", ms=9, mfc="none", mec="#00e5ff", mew=1.8,
                zorder=5)
        right = cz > 0.5 * params.n_layers * params.w * 1e3
        ax.annotate(f"max {vals[k]*100:.0f} % of $J_c$", (cz, cr),
                    xytext=(-14 if right else 14, 12),
                    textcoords="offset points",
                    ha="right" if right else "left",
                    color="white", fontsize=10.5, fontweight="bold", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.6,
                                                foreground="black")])

    cb = fig.colorbar(pc, cax=cax)
    cb.set_label(r"$J\,/\,J_c(B,\theta)$   (share of the local critical "
                 r"current density)", fontsize=11)
    cb.set_ticks(np.linspace(0, 1, 6))
    cb.set_ticklabels([f"{t:.0%}" for t in np.linspace(0, 1, 6)])
    cb.outline.set_edgecolor(INK)

    # Summary statistics skip degenerate (sliver) elements: their DG0 value
    # is unreliable, e.g. a flat tet (quality 1e-4) at 1.10 among neighbours
    # at 0.75-0.88 in the 2026-10-04 eval-166 reproduction.
    qual = _tet_quality(verts)
    good = qual >= QUAL_MIN
    n_bad = int((~good).sum())
    if n_bad:
        print(f"  excluding {n_bad} degenerate element(s) from the summary "
              f"(quality < {QUAL_MIN:g}; their loads "
              f"{np.round(load[~good], 3).tolist()})")
    peak = float(load[good].max())
    n_over = int(np.sum(load[good] > 1.0))
    fig.text(0.075, 0.955, "Local critical-current loading of the optimised "
             f"{params.n_layers}-layer REBCO racetrack coil",
             fontsize=16, fontweight="bold", color=INK, ha="left")
    fig.text(0.075, 0.915,
             f"{I:.1f} A per turn, at the end of a 2 h constant-power ramp  ·  "
             f"peak {peak*100:.0f} % of $J_c$  ·  "
             f"{n_over} of {int(good.sum()):,} coil cells above $J_c$",
             fontsize=11.5, color=MUTED, ha="left")
    if FOOTNOTE:
      fig.text(0.075, 0.018,
             "T-A homogenised FEM (Vargas-Llanos et al. 2022), measured "
             "Ic(B,θ) and n(B,θ) at 20 K with Kim extrapolation above 8 T; "
             "single 4 h implicit step (constant-power ramp surrogate), "
             f"ρ-floor {RHO_FLOOR}.\nEach triangle is one finite-element face "
             "on the symmetry plane, coloured by its cell value (no "
             "interpolation); in-plane element size ≈ 7 mm."
             + (f"\n{n_bad} degenerate (sliver) element(s), shape quality "
                f"< {QUAL_MIN:g}, excluded from the peak and the count."
                if n_bad else "")
             + "\nDP = double pancake, numbered from the axial end of the "
             "coil to the midplane.",
             fontsize=8.3, color=MUTED, ha="left", va="bottom")
    _inset_topview(fig, [0.875, 0.86, 0.11, 0.115])

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}.png/.pdf  (x=0 faces {len(va)}, y=0 faces {len(vb)}; "
          f"peak load {peak:.3f}, {n_over} cells > Jc, {n_bad} excluded)")


if __name__ == "__main__":
    main()
