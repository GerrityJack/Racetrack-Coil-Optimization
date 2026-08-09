"""plot_ramp_field_animation.py -- 2026-08-08, 3-D field-evolution frames
+ animated GIF for the constant-power ramp-up recommendation
(circuit/power_ramp.py).

APPROACH
----------
This is vacuum magnetostatics (mu0 everywhere, no iron) -- the field is
EXACTLY linear in current, so the multi-filament Biot-Savart field
(physics/coil2_field.compute_both_coils_field_multilayer, the same
production path used for every other near-coil field number in this
project) is computed ONCE at full I_design on two orthogonal slice
planes (horizontal, through the bore midplane; vertical, through the
coil axis), then scaled by i_spiral_mean(t)/I_design at each sampled
frame. Using i_spiral_mean (the actual, DCN-computed mean spiral/turn
current) rather than the raw supply current I(t) is deliberate: it is
what actually produces the field, and during a fast ramp it LAGS I(t)
because of NI radial leakage -- this animation shows that lag, not just
I(t) replayed as a scale factor.

Rendered as filled-contour heatmaps projected onto their planes in a 3-D
axis (matplotlib's contourf(..., zdir=...) trick), with the coil
geometry rings drawn for context -- a plain 3-D scatter of a sparse grid
was tried first and rejected: at this coil's actual field range (peak
~11-12T, most of the volume much lower) a coarse point cloud mostly
shows near-zero-field points and hides the smooth gradient a filled
slice makes obvious.

This is a linear-scaling APPROXIMATION for visualization: it does not
resolve per-turn current imbalance (every turn group scaled by the same
mean fraction) or T-A screening-current redistribution -- see
CLAUDE.md's "Ramp-up power analysis" for what those effects actually do
to the local current distribution. Good enough to SHOW the field
building up and lagging; not a source of new physics numbers.

Run:  <env>/bin/python3 visualization/plot_ramp_field_animation.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: registers projection
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "circuit"),
           os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params                                              # noqa: E402
import cparams as cfg                                       # noqa: E402
import dcn as dcn_mod                                        # noqa: E402
import power_ramp as pr                                      # noqa: E402
from geometry import CoilGeometry                             # noqa: E402
from coil2_field import compute_both_coils_field_multilayer   # noqa: E402
from plot_3d import _racetrack_xy, _layer_colors, _ax3d_style  # noqa: E402
from report_common import REPORT_DIR                            # noqa: E402

OUT_DIR = os.path.join(REPORT_DIR, "field_frames")
SLICE_N = 70              # points per axis per slice -- one-time cost only
N_FRAMES = 28
TARGET_T_RAMP = 600.0
VMAX_T = 13.0             # colour scale ceiling -- just above the champion's ~11-12T
                          # bore peak, so the interesting range uses the full colormap


def build_slices():
    half = params.b * 1.9
    g = params.coil_half_gap
    zc = g   # bore midplane

    xh = np.linspace(-half, half, SLICE_N)
    yh = np.linspace(-half, half, SLICE_N)
    Xh, Yh = np.meshgrid(xh, yh, indexing="ij")
    pts_h = np.column_stack([Xh.ravel(), Yh.ravel(), np.full(Xh.size, zc)])

    xv = np.linspace(-half, half, SLICE_N)
    zv = np.linspace(-0.005, 2 * g + 0.005, SLICE_N)
    Xv, Zv = np.meshgrid(xv, zv, indexing="ij")
    pts_v = np.column_stack([Xv.ravel(), np.zeros(Xv.size), Zv.ravel()])

    return (Xh, Yh, zc, pts_h), (Xv, Zv, pts_v)


def draw_coil_rings(ax, colors):
    g = params.coil_half_gap
    for coil2 in (False, True):
        for i in range(params.n_layers):
            xo, yo = _racetrack_xy(params.a_out)
            for z_face in (params.layer_z_tops[i], params.layer_z_bottoms[i]):
                z_plot = (2.0 * g - z_face) if coil2 else z_face
                zo = np.full_like(xo, z_plot * 1e3)
                ax.plot(xo * 1e3, yo * 1e3, zo, color=colors[i], lw=1.3, alpha=0.9)


def render_frame(slice_h, slice_v, Bh_scaled, Bv_scaled, t, I_now, i_spiral,
                 frame_idx, out_path):
    Xh, Yh, zc, _ = slice_h
    Xv, Zv, _ = slice_v
    colors = _layer_colors()

    fig = plt.figure(figsize=(7.5, 7))
    fig.patch.set_facecolor("#0d0d1a")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _ax3d_style(ax)

    levels = np.linspace(0, VMAX_T, 25)
    cs = ax.contourf(Xh * 1e3, Yh * 1e3, np.clip(Bh_scaled, 0, VMAX_T),
                     zdir="z", offset=zc * 1e3, levels=levels, cmap="magma",
                     alpha=0.9)
    ax.contourf(Xv * 1e3, np.clip(Bv_scaled, 0, VMAX_T), Zv * 1e3,
               zdir="y", offset=0.0, levels=levels, cmap="magma", alpha=0.9)

    draw_coil_rings(ax, colors)

    ax.view_init(elev=22, azim=-50 + frame_idx * 1.2)   # slow orbit
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
    half = params.b * 1.9 * 1e3
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_zlim(-5, 2 * params.coil_half_gap * 1e3 + 5)

    cb = fig.colorbar(cs, ax=ax, shrink=0.55, pad=0.05, ticks=np.linspace(0, VMAX_T, 6))
    cb.set_label("|B| (T)", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax, "yticklabels"), color="white")

    ax.set_title(
        f"t = {t:6.1f} s   I(t) = {I_now:5.1f} A   "
        f"i_spiral,mean = {i_spiral:5.1f} A   "
        f"({100*i_spiral/params.I_design:4.1f}% of I_design)",
        color="white", fontsize=10)
    fig.suptitle("Field build-up during constant-power ramp -- champion coil, "
                 "both coils\n(horizontal slice: bore midplane; vertical slice: "
                 "through the coil axis)",
                 color="white", fontsize=11)

    fig.savefig(out_path, dpi=110, bbox_inches="tight",
               facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    print("=" * 78)
    print("3-D field-evolution animation for the constant-power ramp")
    print("=" * 78)
    os.makedirs(OUT_DIR, exist_ok=True)

    geom = CoilGeometry.from_params()
    I_design = float(params.I_design)
    d = dcn_mod.build(geom, rho_ct_uohm_cm2=cfg.RHO_CT_UOHM_CM2, verbose=True)

    print("Finding constant power for a ~600s ramp (nominal rho_c)...")
    lo, hi = 1.0, 5000.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        r = pr.run_power_ramp_auto_span(d, mid, I_design, 1200.0, verbose=False)
        if r["t_ramp_end"] > TARGET_T_RAMP:
            lo = mid
        else:
            hi = mid
    P = hi
    r = pr.run_power_ramp_auto_span(d, P, I_design, 1200.0, verbose=False)
    print(f"  P={P:.2f} W  t_ramp={r['t_ramp_end']:.1f} s")

    t_end_show = r["t_ramp_end"] * 1.15   # a bit into the hold, to show the lag settle
    frame_times = np.linspace(0.0, t_end_show, N_FRAMES)
    I_frames = np.interp(frame_times, r["t"], r["I"])
    ispiral_frames = np.interp(frame_times, r["t"], r["i_spiral_mean"])

    print(f"Computing Biot-Savart field on two {SLICE_N}x{SLICE_N} slices at "
          f"I_design (one-time cost)...")
    slice_h, slice_v = build_slices()
    _, _, _, pts_h = slice_h
    _, _, pts_v = slice_v
    Bh_full = np.linalg.norm(
        compute_both_coils_field_multilayer(pts_h, I_per_turn=I_design,
                                            turns_per_filament=150), axis=1
    ).reshape(slice_h[0].shape)
    Bv_full = np.linalg.norm(
        compute_both_coils_field_multilayer(pts_v, I_per_turn=I_design,
                                            turns_per_filament=150), axis=1
    ).reshape(slice_v[0].shape)
    print(f"  bore-midplane slice peak |B| = {Bh_full.max():.2f} T   "
          f"axial slice peak |B| = {Bv_full.max():.2f} T")

    frame_paths = []
    for k in range(N_FRAMES):
        frac = ispiral_frames[k] / I_design
        out_path = os.path.join(OUT_DIR, f"frame_{k:03d}.png")
        render_frame(slice_h, slice_v, Bh_full * frac, Bv_full * frac,
                    frame_times[k], I_frames[k], ispiral_frames[k], k, out_path)
        frame_paths.append(out_path)
        print(f"  frame {k+1}/{N_FRAMES}  t={frame_times[k]:.1f}s  "
              f"i_spiral={ispiral_frames[k]:.1f}A", flush=True)

    gif_path = os.path.join(REPORT_DIR, "field_animation.gif")
    imgs = [Image.open(p) for p in frame_paths]
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                duration=180, loop=0)
    print(f"\nWrote {len(frame_paths)} frames to {OUT_DIR}/")
    print(f"Wrote animated GIF: {gif_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
