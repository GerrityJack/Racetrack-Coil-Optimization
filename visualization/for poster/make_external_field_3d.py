"""
make_external_field_3d.py
============================
Poster figure: minimal 3D view of the stray field OUTSIDE the coil
assembly -- direction (arrow orientation) and magnitude (arrow color,
log scale) sampled on two concentric spherical shells centered on the
assembly, both comfortably clear of the winding pack by construction
(see radius argument below).

Field evaluated with the same multi-filament Biot-Savart path every
other near-coil field number in this project uses
(physics/coil2_field.compute_both_coils_field_multilayer -- required,
not the single-filament version, at this coil's compact scale).

Geometry note: the assembly's largest in-plane half-extent is
R_eff = a_out + L (the major-axis half-length, i.e. the farthest any
winding cell can be from the assembly's centroid in x). A sphere of
radius r centered on the assembly's centroid (0, 0, coil_half_gap)
clears the ENTIRE padded winding-pack bounding box whenever r > R_eff,
regardless of direction, because R_eff is by definition the box's
farthest corner-distance-bound along its longest axis -- so shells at
1.6x/2.6x R_eff need no per-point exclusion test.

Inputs: params.py (champion geometry, I_design)
Output: visualization/for poster/external_field_3d.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_VIZ = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_VIZ)
for _p in (_ROOT, _VIZ, os.path.join(_ROOT, "physics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params                                              # noqa: E402
from coil2_field import compute_both_coils_field_multilayer  # noqa: E402

BLACK = "#111111"
CMAP = "magma"
N_NEAR, N_FAR = 40, 20
SHELL_FACTORS = (1.6, 2.6)
ARROW_LEN_FRAC = 0.10   # of the near-shell radius


def _racetrack_loop(radius, L, n=48):
    th = np.linspace(np.pi / 2, -np.pi / 2, n // 4, endpoint=False)
    x = np.concatenate([np.linspace(-L, L, n // 4, endpoint=False),
                        L + radius * np.cos(th),
                        np.linspace(L, -L, n // 4, endpoint=False),
                        -L + radius * np.cos(th + np.pi)])
    y = np.concatenate([np.full(n // 4, radius), radius * np.sin(th),
                        np.full(n // 4, -radius), radius * np.sin(th + np.pi)])
    return np.append(x, x[0]), np.append(y, y[0])


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta),
                            np.cos(phi)])


def main():
    L, a_out, g = params.L, params.a_out, params.coil_half_gap
    centroid = np.array([0.0, 0.0, g])
    R_eff = a_out + L

    dirs = np.vstack([_fibonacci_sphere(N_NEAR), _fibonacci_sphere(N_FAR)])
    radii = np.concatenate([np.full(N_NEAR, SHELL_FACTORS[0] * R_eff),
                            np.full(N_FAR, SHELL_FACTORS[1] * R_eff)])
    pts = centroid + dirs * radii[:, None]

    B = compute_both_coils_field_multilayer(pts, I_per_turn=params.I_design)
    Bmag = np.linalg.norm(B, axis=1)
    u = B / Bmag[:, None]

    arrow_len = ARROW_LEN_FRAC * SHELL_FACTORS[0] * R_eff

    fig = plt.figure(figsize=(8, 7))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # Coil outline -- outer edge of the winding pack only, one loop per
    # coil at its own central z, kept deliberately plain (no per-layer
    # detail) so it reads as context, not the subject of the figure.
    for z_c in (0.0, 2 * g):
        xo, yo = _racetrack_loop(a_out, L)
        ax.plot(xo, yo, z_c, color=BLACK, lw=1.3, zorder=6)

    norm = matplotlib.colors.LogNorm(vmin=Bmag.min(), vmax=Bmag.max())
    cmap = matplotlib.colormaps[CMAP]
    colors = cmap(norm(Bmag))
    # Fade the far shell so the near shell reads as visually "in front" --
    # the only depth cue this minimal a figure has, since axes are off.
    colors[N_NEAR:, 3] = 0.45

    ax.quiver(pts[:, 0], pts[:, 1], pts[:, 2],
              u[:, 0], u[:, 1], u[:, 2],
              length=arrow_len, normalize=True,
              colors=colors, linewidth=1.5, arrow_length_ratio=0.35)

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.035, shrink=0.55)
    cbar.set_label("|B|  (T)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    span = SHELL_FACTORS[1] * R_eff * 1.05
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_zlim(g - span, g + span)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=22, azim=25)

    fig.tight_layout(pad=0.2)
    out = os.path.join(_HERE, "external_field_3d.png")
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"  |B| range on shells: {Bmag.min()*1e3:.3f}-{Bmag.max()*1e3:.3f} mT")


if __name__ == "__main__":
    main()
