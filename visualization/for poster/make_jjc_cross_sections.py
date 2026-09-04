"""
make_jjc_cross_sections.py
============================
Poster figures: J/Jc(B,theta) at I_design, plotted on the two natural
winding cross-sections of the racetrack coil --

  major-axis cross section  (y = 0 cut, x vs z)  -- through the two
                              end-cap tips
  minor-axis cross section  (x = 0 cut, y vs z)  -- through the two
                              straight-section legs

Both coils are shown (coil 2 mirrored about the midplane, z = g -- see
plot_3d.py's _mirror_z). Only the u>=0 half of each cross section is
drawn (u = x for the major-axis/cap cut, u = y for the minor-axis/
straight-leg cut) -- the u<0 half is the mirror image of what's shown,
so it carries no new information; omitting it halves the figure width.

Per-cell ratio = |J_TA_inplane| / Jc(B,theta), same normalisation
ta_solve.py's own Picard solver and transient/validation/
ta_quench_margin_check.py both use, with the project's default (Kim)
Ic(B) extrapolation model.

Each layer's homogenised cross-section is rendered as its own
triangulated contour (matplotlib tri), so a thin closure layer does not
get bridged into its wide neighbour and each layer's true a_inner..a_out
radial extent is respected.

Inputs: solve/racetrack_ta_fields.npz (run solve/ta_solve.py if missing/stale)
Output: visualization/for poster/jjc_major_axis_cross_section.png
        visualization/for poster/jjc_minor_axis_cross_section.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

_HERE = os.path.dirname(os.path.abspath(__file__))
_VIZ = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_VIZ)
for _p in (_ROOT, _VIZ, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "solve"), os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params                                          # noqa: E402
from current_source import normal_xy                     # noqa: E402
from ic_model import angle_with_normal_deg                # noqa: E402
from ic_extrapolation import make_ic_model                 # noqa: E402
from plot_3d import _mirror_z, _layer_colors                # noqa: E402

FIELDS_NPZ = os.path.join(params.SOLVE_DIR, "racetrack_ta_fields.npz")
SLAB_EPS = 3.0e-3    # m -- thin-slab half-width around the cut plane
CMAP = "plasma"
BLACK = "#111111"


def _load_ratio():
    d = np.load(FIELDS_NPZ)
    I_solved = float(d["I_solved"])
    if abs(I_solved - params.I_design) > 0.5:
        print(f"WARNING: {FIELDS_NPZ} is at I={I_solved:.1f} A, "
              f"not I_design={params.I_design:.1f} A.")

    c = d["coil_centroids"]
    B = d["coil_B"]
    J = d["J_TA_coil"]
    delta_SC = float(d["delta_SC"])

    L = params.L
    nx, ny = normal_xy(c[:, 0], c[:, 1], L)
    n_hat = np.column_stack([nx, ny, np.zeros_like(nx)])

    Bmag = np.linalg.norm(B, axis=1)
    theta = angle_with_normal_deg(B, n_hat)
    ic = make_ic_model("kim")
    Ic_A, _ = ic.critical_current(Bmag, theta)
    Jc = Ic_A / (delta_SC * params.w)

    J_dot_n = np.einsum("ij,ij->i", J, n_hat)
    J_inplane = J - J_dot_n[:, None] * n_hat
    Jmag = np.linalg.norm(J_inplane, axis=-1)

    return c, Jmag / Jc, I_solved


def _layer_patches(u_all, z_all, r_all, u_lo, u_hi, z_lo, z_hi):
    """Cells belonging to one (layer, coil, sign) patch."""
    m = (u_all >= u_lo - 1e-9) & (u_all <= u_hi + 1e-9) & \
        (z_all >= z_lo - 1e-9) & (z_all <= z_hi + 1e-9)
    return u_all[m], z_all[m], r_all[m]


def _draw_cross_section(u_axis, out_name, u_label):
    """u_axis: 0 -> plotted coordinate is x, slab filter is |y|<eps
                (major-axis / cap cut)
               1 -> plotted coordinate is y, slab filter is |x|<eps
                (minor-axis / straight-leg cut)
    """
    c, ratio, I_solved = _load_ratio()
    slab_axis = 1 - u_axis
    slab_mask = np.abs(c[:, slab_axis]) < SLAB_EPS
    u0 = c[slab_mask, u_axis]
    z0 = c[slab_mask, 2]
    r0 = ratio[slab_mask]

    g = params.coil_half_gap
    vmax = 0.65
    layer_colors = _layer_colors()

    fig, ax = plt.subplots(figsize=(9, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    u_bounds = [np.inf, -np.inf]
    z_bounds = [np.inf, -np.inf]

    # Major-axis (cap) cut: turn positions are offset by L from the
    # origin (cap centred at x=+-L), so a_inner_list/a_out (radii
    # measured FROM the cap centre) must be shifted by L to land on the
    # real x range of the data. Minor-axis (straight) cut: the straight
    # centreline already sits at y=0, so a_inner_list/a_out apply as-is.
    u_offset = params.L if u_axis == 0 else 0.0

    cf_ref = None
    for coil in (1, 2):
        # Coil 2's cells are coil 1's own eighth-symmetry solution
        # MIRRORED about the midplane z=g (physically identical J/Jc,
        # see plot_3d.py's _mirror_z) -- transform the data's z into
        # coil 2's frame before filtering, not just the target z-band.
        z_c = _mirror_z(z0, g) if coil == 2 else z0
        for i in range(params.n_layers):
            zb, zt = params.layer_z_bottoms[i], params.layer_z_tops[i]
            if coil == 2:
                zb, zt = sorted((_mirror_z(zb, g), _mirror_z(zt, g)))
            u_in = u_offset + params.a_inner_list[i]
            u_out = u_offset + params.a_out
            for sign in (+1,):
                lo, hi = u_in, u_out
                uu, zz, rr = _layer_patches(u0, z_c, r0, lo, hi, zb, zt)
                if len(uu) < 3:
                    continue
                # Interpolate onto a fine regular grid spanning the
                # layer's TRUE physical rectangle (not just the convex
                # hull of the sampled cells) so the fill reaches every
                # corner instead of leaving the ragged/blocky triangle
                # edges a raw scattered triangulation gives. Linear
                # interpolation in the interior, nearest-neighbour
                # fallback wherever linear can't extrapolate (rectangle
                # corners, or a near-degenerate thin closure layer where
                # a linear fit isn't even well posed).
                pts = np.column_stack([uu * 1e3, zz * 1e3])
                gu = np.linspace(lo * 1e3, hi * 1e3, 60)
                gz = np.linspace(zb * 1e3, zt * 1e3, 60)
                GU, GZ = np.meshgrid(gu, gz)
                try:
                    grid_vals = griddata(pts, rr, (GU, GZ), method="linear")
                except Exception:
                    grid_vals = np.full(GU.shape, np.nan)
                nan_mask = np.isnan(grid_vals)
                if nan_mask.any():
                    grid_vals[nan_mask] = griddata(
                        pts, rr, (GU[nan_mask], GZ[nan_mask]), method="nearest")
                cf = ax.pcolormesh(GU, GZ, grid_vals, shading="gouraud",
                                    cmap=CMAP, vmin=0, vmax=vmax)
                cf_ref = cf
                u_bounds[0] = min(u_bounds[0], lo * 1e3)
                u_bounds[1] = max(u_bounds[1], hi * 1e3)
                z_bounds[0] = min(z_bounds[0], zb * 1e3)
                z_bounds[1] = max(z_bounds[1], zt * 1e3)
            # thin layer-boundary outline
            lo, hi = u_in, u_out
            ax.plot([lo * 1e3, hi * 1e3, hi * 1e3, lo * 1e3, lo * 1e3],
                    [zb * 1e3, zb * 1e3, zt * 1e3, zt * 1e3, zb * 1e3],
                    color=BLACK, lw=0.4, alpha=0.5)

    cbar = fig.colorbar(cf_ref, ax=ax, pad=0.015, fraction=0.035, shrink=0.5,
                         extend="max")
    cbar.set_label("J / J$_c$", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_aspect("equal")
    ax.set_xlabel(u_label, fontsize=12)
    ax.set_ylabel("Axial position  (mm)", fontsize=12)
    for sp in ax.spines.values():
        sp.set_edgecolor("#888")
    ax.tick_params(colors=BLACK)

    pad = 1.5
    ax.set_xlim(u_bounds[0] - pad, u_bounds[1] + pad)
    ax.set_ylim(z_bounds[0] - pad, z_bounds[1] + pad)

    fig.tight_layout(pad=0.4)
    out = os.path.join(_HERE, out_name)
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.03,
                facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}  (I={I_solved:.1f} A)")


def main():
    if not os.path.exists(FIELDS_NPZ):
        print(f"ERROR: {FIELDS_NPZ} not found — run solve/ta_solve.py first")
        return
    _draw_cross_section(0, "jjc_major_axis_cross_section.png",
                         "Distance along major axis  (mm)")
    _draw_cross_section(1, "jjc_minor_axis_cross_section.png",
                         "Distance along minor axis  (mm)")


if __name__ == "__main__":
    main()
