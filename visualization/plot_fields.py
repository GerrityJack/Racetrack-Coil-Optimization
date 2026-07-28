"""
plot_fields.py
==============
Produces exactly two output images:

  field_top.png   — top-down view of the field at the most informative
                    z-plane, with in-plane arrow vectors
  field_side.png  — y=0 side view with Bx/Bz arrow vectors and
                    winding-layer extent bands

If two_coil_mode = True:
  • field_top.png  shows the midplane z = coil_half_gap (where both coils
    contribute and in-plane Bx/By arrows are meaningful)
  • field_side.png shows an extended z-range covering both coil stacks

If two_coil_mode = False:
  • field_top.png  shows z = 0 (the coil plane) with signed Bz
  • field_side.png shows the single-coil z range

Inputs: solve/racetrack_fields.npz  (must exist — run solve.py first)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "visualization")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
from coil2_field import (compute_both_coils_field_multilayer,
                         compute_field_from_coil_at_z)
from plot_3d import _mirror_z


# ── dynamic layer colours (works for any n_layers) ──────────────────────────
def _layer_colors():
    cmap = matplotlib.colormaps.get_cmap("tab10")
    return [cmap(i % 10) for i in range(params.n_layers)]


# ── racetrack centreline outline ─────────────────────────────────────────────
def _outline(a=None, b=None, n=300):
    a = a or params.a; b = b or params.b; L = b - a
    th = np.linspace(np.pi/2, -np.pi/2, n//4, endpoint=False)
    x = np.concatenate([np.linspace(-L, L, n//4, endpoint=False),
                        L + a*np.cos(th),
                        np.linspace(L, -L, n//4, endpoint=False),
                        -L + a*np.cos(th + np.pi)])
    y = np.concatenate([np.full(n//4, a), a*np.sin(th),
                        np.full(n//4, -a), a*np.sin(th + np.pi)])
    return np.append(x, x[0]), np.append(y, y[0])


# ═══════════════════════════════════════════════════════════════════════════
# field_top.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_top(npz_data):
    a, b   = float(npz_data["a"]), float(npz_data["b"])
    I_pt   = float(npz_data["I_solved"])
    two    = getattr(params, "two_coil_mode", False)
    g      = getattr(params, "coil_half_gap", 0.0)

    nx, ny = 200, 100
    xs = np.linspace(-b * 1.18, b * 1.18, nx)
    ys = np.linspace(-b * 0.68, b * 0.68, ny)
    Xg, Yg = np.meshgrid(xs, ys)

    if two:
        z_plane = g
        fp = np.column_stack([Xg.ravel(), Yg.ravel(),
                               np.full(Xg.size, z_plane)])
        print(f"  field_top: computing at midplane z = {z_plane*1e3:.0f} mm "
             f"(multi-filament, per-layer) …")
        B_tot = compute_both_coils_field_multilayer(
            fp, I_per_turn=I_pt, n_straight=400, n_cap=300)
        Bx = B_tot[:, 0].reshape(Xg.shape)
        By = B_tot[:, 1].reshape(Xg.shape)
        Bz = B_tot[:, 2].reshape(Xg.shape)
        Bm = np.linalg.norm(B_tot, axis=1).reshape(Xg.shape)
        title = (f"Top-down field  (z = {z_plane*1e3:.0f} mm midplane)"
                 f"\nI = {I_pt:.0f} A/turn,  {params.n_turns_total} total turns"
                 f"  |  coil-half-gap = {g*1e3:.0f} mm")
        z_label = f"z = {z_plane*1e3:.0f} mm (midplane)"
    else:
        z_plane = 0.0
        # Use FEM grid data at z=0
        top_B = npz_data["top_B"]                         # (ny, nx, 3)
        top_mask = npz_data["top_mask"]
        Bx = np.where(top_mask, top_B[..., 0], np.nan)
        By = np.where(top_mask, top_B[..., 1], np.nan)
        Bz = np.where(top_mask, top_B[..., 2], np.nan)
        Bm = np.where(top_mask, np.linalg.norm(top_B, axis=2), np.nan)
        Xg, Yg = npz_data["top_X"], npz_data["top_Y"]
        title = (f"Top-down field  (z = 0,  coil plane)"
                 f"\nI = {I_pt:.0f} A/turn,  {params.n_turns_total} total turns")
        z_label = "z = 0 (coil plane)"

    ox, oy = _outline(a, b)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#111")

    # |B| background
    cf = ax.pcolormesh(Xg * 1e3, Yg * 1e3, Bm,
                       shading="auto", cmap="magma")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("|B| (T)", color="white"); cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Arrow vectors — subsample the grid
    stride = max(1, nx // 22)
    sl = (slice(None, None, stride), slice(None, None, stride))
    Bxs, Bys = (Bx if two else np.zeros_like(Bz))[sl], (By if two else np.zeros_like(Bz))[sl]
    Bzs = Bz[sl]
    if two:
        # In-plane arrows (Bx, By)
        nrm = np.where(np.hypot(Bxs, Bys) > 0, np.hypot(Bxs, Bys), 1.0)
        ax.quiver(Xg[sl] * 1e3, Yg[sl] * 1e3, Bxs/nrm, Bys/nrm,
                  np.hypot(Bxs, Bys), cmap="cool", pivot="mid",
                  scale=40, width=0.003, alpha=0.80)
        ax.text(0.01, 0.01, "arrows: in-plane (Bx, By) direction",
                transform=ax.transAxes, color="white", fontsize=8, alpha=0.7)
    else:
        # Bz contours (in-plane field is zero in the coil plane)
        lvls = np.linspace(np.nanpercentile(Bz, 5), np.nanpercentile(Bz, 95), 10)
        ax.contour(Xg * 1e3, Yg * 1e3, Bz, levels=lvls,
                   colors="white", linewidths=0.6, alpha=0.5)
        ax.text(0.01, 0.01,
                "Field is purely Bz in the coil plane — contours show Bz",
                transform=ax.transAxes, color="white", fontsize=8, alpha=0.7)

    # Coil outline
    ax.plot(ox * 1e3, oy * 1e3, "w-", lw=1.2, alpha=0.6, label="coil centreline")

    ax.set_aspect("equal")
    ax.set_xlabel("x  (mm)", color="white"); ax.set_ylabel("y  (mm)", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.set_title(title, color="white", fontsize=11)
    ax.legend(fontsize=8, labelcolor="white", facecolor="#222", framealpha=0.5)

    out = os.path.join(params.VIZ_DIR, "field_top.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Wrote {out}")


# ═══════════════════════════════════════════════════════════════════════════
# field_side.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_side(npz_data):
    a, b   = float(npz_data["a"]), float(npz_data["b"])
    w      = float(npz_data["w"])
    I_pt   = float(npz_data["I_solved"])
    two    = getattr(params, "two_coil_mode", False)
    g      = getattr(params, "coil_half_gap", 0.0)
    colors = _layer_colors()

    if two:
        # Extended z range covering both stacks
        z_c2   = 2.0 * g
        margin = max(b * 0.20, g * 0.25)
        nx, nz = params.sideview_nx, params.sideview_nz * 2
        xs = np.linspace(-b * 1.15, b * 1.15, nx)
        zs = np.linspace(-params._z_top - margin, z_c2 + params._z_top + margin, nz)
        Xs, Zs = np.meshgrid(xs, zs)
        fp = np.column_stack([Xs.ravel(), np.zeros(Xs.size), Zs.ravel()])
        print(f"  field_side: two-coil multi-filament Biot-Savart "
             f"({fp.shape[0]} points) …")
        B_tot = compute_both_coils_field_multilayer(
            fp, I_per_turn=I_pt, n_straight=400, n_cap=300)
        Bmag = np.linalg.norm(B_tot, axis=1).reshape(Xs.shape)
        Bx_g = B_tot[:, 0].reshape(Xs.shape)
        Bz_g = B_tot[:, 2].reshape(Xs.shape)
        title = (f"Side view  (y = 0)  —  two-coil field\n"
                 f"I = {I_pt:.0f} A/turn,  separation = {z_c2*1e3:.0f} mm")
    else:
        sX, sZ = npz_data["side_X"], npz_data["side_Z"]
        sB = npz_data["side_B"]; sm = npz_data["side_mask"]
        Bmag = np.where(sm, np.linalg.norm(sB, axis=2), np.nan)
        Bx_g = np.where(sm, sB[..., 0], np.nan)
        Bz_g = np.where(sm, sB[..., 2], np.nan)
        Xs, Zs = sX, sZ
        title = (f"Side view  (y = 0)  —  single-coil FEM\n"
                 f"I = {I_pt:.0f} A/turn,  {params.n_turns_total} total turns")

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#111")

    # Clamp colorbar: analytic Biot-Savart diverges near conductors; use
    # the 98th-percentile as vmax so the bore/gap region is well-resolved.
    _vmax = float(np.nanpercentile(Bmag, 98))
    cf = ax.pcolormesh(Xs * 1e3, Zs * 1e3, Bmag,
                       shading="auto", cmap="magma", vmin=0, vmax=_vmax)
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("|B| (T)", color="white"); cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Arrow vectors
    stride = max(1, Xs.shape[1] // 22)
    sl = (slice(None, None, stride), slice(None, None, stride))
    Bxs, Bzs = Bx_g[sl], Bz_g[sl]
    nrm = np.where(np.hypot(Bxs, Bzs) > 0, np.hypot(Bxs, Bzs), 1.0)
    ax.quiver(Xs[sl] * 1e3, Zs[sl] * 1e3, Bxs/nrm, Bzs/nrm,
              np.hypot(Bxs, Bzs), cmap="cool", pivot="mid",
              scale=40, width=0.003, alpha=0.80)

    # Shade winding-pack layers (both coils if two_coil_mode). Coil 2's z
    # is MIRRORED about the midplane (z=g), not shifted by +2g -- see
    # plot_3d.py's _mirror_z docstring for why (matches the real physics
    # code's convention, compute_both_coils_field_multilayer).
    for coil2 in ((False, True) if two else (False,)):
        for i in range(params.n_layers):
            zb_local = params.layer_z_bottoms[i]
            zt_local = params.layer_z_tops[i]
            if coil2:
                zb, zt = sorted((_mirror_z(zb_local, g) * 1e3,
                                _mirror_z(zt_local, g) * 1e3))
            else:
                zb, zt = zb_local * 1e3, zt_local * 1e3
            ax.axhspan(zb, zt, alpha=0.12, color=colors[i],
                       label=(f"Layer {i} ({params.n_turns[i]}t)"
                              if not coil2 else None))

    if two:
        ax.axhline(g * 1e3, color="lime", lw=1.5, ls="--",
                   label=f"midplane (z = {g*1e3:.0f} mm)")

    # Cap-tip vertical markers
    for xc in (-b, b):
        ax.axvspan(xc*1e3 - 3, xc*1e3 + 3, color="cyan", alpha=0.25,
                   label="cap tips" if xc == -b else None)

    ax.set_aspect("equal")
    ax.set_xlabel("x  (mm)", color="white"); ax.set_ylabel("z  (mm)", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.set_title(title, color="white", fontsize=11)
    ax.legend(fontsize=8, labelcolor="white", facecolor="#222",
              framealpha=0.5, loc="upper right")

    out = os.path.join(params.VIZ_DIR, "field_side.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Wrote {out}")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    npz_path = params.fields_npz_filename
    if not os.path.exists(npz_path):
        print(f"ERROR: {npz_path} not found — run solve.py first"); return
    data = np.load(npz_path)
    print("Plotting field distribution …")
    plot_top(data)
    plot_side(data)
    print(f"\nDone. Wrote field_top.png, field_side.png  →  {params.VIZ_DIR}")


if __name__ == "__main__":
    main()
