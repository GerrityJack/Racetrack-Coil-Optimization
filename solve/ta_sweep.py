"""
ta_sweep.py
===========
T-A screening-current sweep, producing figures that match the dark-theme
style of plot_fields.py / field_uniformity.py / plot_3d.py.

Outputs → params.VIZ_DIR:
  ta_bore_vs_current.png   — |Bz| vs I: uniform-J vs T-A  (sweep summary)
  ta_scif_vs_current.png   — SCIF ΔBz [mT] and [%] vs I   (sweep summary)
  ta_field_top.png         — top-down |B| map at max safe current
  ta_field_side.png        — y=0 side |B| map at max safe current
  ta_uniformity.png        — uniformity box at max safe current (SCIF corrected)
  ta_field_3d.png          — 3-D coil scatter coloured by T-A |B|
  ta_scif_3d.png           — 3-D coil scatter coloured by |ΔJ| (screening J)

  ta_sweep_results.csv     — full numerical table
"""

import os, sys, csv, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D   # noqa
from mpi4py import MPI

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "solve"),
           os.path.join(_ROOT, "visualization")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
from dolfinx.io import gmsh as gmshio
import dolfinx.mesh as dmesh
import solve as base_solve
import build_mesh

from physics.ic_model import IcModel, NValueModel
from ta_solve import (setup_ta_problem, solve_ta_at_current, _J_from_T,
                      dB_bore_from_dJ)
from coil2_field import compute_both_coils_field_multilayer

# ── Sweep settings ─────────────────────────────────────────────────────────
SWEEP_CURRENTS = list(range(150, 425, 25))   # 150 … 400 A

# ── Uniformity box (matches field_uniformity.py) ───────────────────────────
REGION_X_M   = 0.015;  REGION_Y_M  = 0.006
GRID_NX      = 150;    GRID_NY     = 60
SURROUND_X_M = 0.040;  SURROUND_Y_M = 0.024
SURROUND_NX  = 200;    SURROUND_NY  = 120
TARGET_PCT   = 1.0

# ── Shared dark-theme helpers ───────────────────────────────────────────────

def _dark_fig(figsize):
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("#111")
    return fig

def _dark_ax(ax):
    ax.set_facecolor("#0d0d1a")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    return ax

def _dark_cbar(fig, mappable, ax, label):
    cb = fig.colorbar(mappable, ax=ax, pad=0.02)
    cb.set_label(label, color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    return cb

def _dark_cbar3d(fig, mappable, ax, label):
    cb = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.03)
    cb.set_label(label, color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    return cb

def _ax3d_style(ax):
    ax.set_facecolor("#0d0d1a")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False; pane.set_edgecolor("#2a2a3a")
    ax.tick_params(colors="white", labelsize=7)
    for attr in ("xaxis", "yaxis", "zaxis"):
        getattr(ax, attr).label.set_color("white")
        getattr(ax, attr).label.set_fontsize(8)

def _layer_colors():
    cmap = matplotlib.colormaps.get_cmap("tab10")
    return [cmap(i % 10) for i in range(params.n_layers)]

def _outline(n=300):
    a, b, L = params.a, params.b, params.b - params.a
    th = np.linspace(np.pi/2, -np.pi/2, n//4, endpoint=False)
    x = np.concatenate([np.linspace(-L, L, n//4, endpoint=False),
                        L + a*np.cos(th),
                        np.linspace(L, -L, n//4, endpoint=False),
                        -L + a*np.cos(th + np.pi)])
    y = np.concatenate([np.full(n//4, a), a*np.sin(th),
                        np.full(n//4, -a), a*np.sin(th + np.pi)])
    return np.append(x, x[0]), np.append(y, y[0])

def _racetrack_xy(radius, n=300):
    a, b, L = params.a, params.b, params.b - params.a
    th = np.linspace(np.pi/2, -np.pi/2, n//4, endpoint=False)
    x = np.concatenate([np.linspace(-L, L, n//4, endpoint=False),
                        L + radius*np.cos(th),
                        np.linspace(L, -L, n//4, endpoint=False),
                        -L + radius*np.cos(th + np.pi)])
    y = np.concatenate([np.full(n//4, radius), radius*np.sin(th),
                        np.full(n//4, -radius), radius*np.sin(th + np.pi)])
    return np.append(x, x[0]), np.append(y, y[0])

# ── Quench limit helper ─────────────────────────────────────────────────────

def _max_safe_current(sweep_currents):
    csv_q = os.path.join(params.SWEEP_DIR, "quench_results.csv")
    if os.path.exists(csv_q):
        with open(csv_q) as f:
            vals = [float(v) for r in csv.DictReader(f)
                    if (v := (r.get("quench_current_A") or "").strip())]
        if vals:
            I_ql = min(vals)
            safe = [I for I in sweep_currents if I < I_ql]
            if safe:
                print(f"  Quench limit: {I_ql:.1f} A  → max safe: {max(safe):.0f} A")
                return max(safe), I_ql
    print(f"  No quench data; using I_design = {params.I_design:.0f} A")
    return params.I_design, None

# ── SCIF at bore ────────────────────────────────────────────────────────────

def _scif_at_bore(ta, domain, I_amps):
    delta_SC = ta["delta_SC"]; Lambda = ta["Lambda"]; w = float(params.w)
    cents = ta["coil_centroids"]

    J_TA  = _J_from_T(ta, domain)
    dJ    = J_TA - ta["t_hat_coil"] * (I_amps / (delta_SC * w))
    dJ_s  = dJ * (delta_SC / Lambda)

    # Full two-coil system (4 quadrants × 2 coils) with exact cell volumes
    dB = dB_bore_from_dJ(cents, dJ_s, ta["coil_vols"])

    bore = np.array([[0., 0., float(params.coil_half_gap)]])
    B_u = compute_both_coils_field_multilayer(bore, I_per_turn=I_amps)
    Bz_u = float(B_u[0, 2])
    return dB, Bz_u, Bz_u + float(dB[2]), cents

# ══════════════════════════════════════════════════════════════════════════════
# SWEEP SUMMARY PLOTS  (dark theme, across all currents)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_bore_vs_current(rows, I_quench):
    I_arr    = np.array([r["I_A"] for r in rows])
    Bzu_arr  = np.abs([r["Bz_bore_uniform_T"] for r in rows])
    BzTA_arr = np.abs([r["Bz_bore_TA_T"] for r in rows])
    B_target = getattr(params, "B_target", 10.0)

    fig = _dark_fig((9, 5)); ax = _dark_ax(fig.add_subplot(111))
    ax.plot(I_arr, Bzu_arr,  "b-o",  lw=1.8, ms=5, label="Uniform J (Biot-Savart)")
    ax.plot(I_arr, BzTA_arr, color="tomato", ls="--", marker="s",
            lw=1.8, ms=5, label="T-A (screening corrected)")
    ax.fill_between(I_arr, Bzu_arr, BzTA_arr,
                    alpha=0.18, color="tomato", label="SCIF offset")
    ax.axhline(B_target, color="lime", ls=":", lw=1.2,
               label=f"Target {B_target:.0f} T")
    if I_quench:
        ax.axvline(I_quench, color="yellow", ls="--", lw=1.0,
                   label=f"Quench limit {I_quench:.0f} A")
    ax.set_xlabel("Transport current I  [A]", color="white")
    ax.set_ylabel("|$B_z$| at bore midplane  [T]", color="white")
    ax.set_title(f"Bore field vs current — T-A screening currents\n"
                 f"(Δt = {params.ramp_duration:.0f} s  |  "
                 f"δ_SC = {params.delta_SC*1e6:.1f} µm)",
                 color="white")
    ax.legend(fontsize=9, labelcolor="white", facecolor="#222", framealpha=0.7)
    ax.grid(True, alpha=0.25, color="#555")
    fig.tight_layout()
    out = os.path.join(params.VIZ_DIR, "ta_bore_vs_current.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig); print(f"  Wrote {out}")


def _plot_scif_vs_current(rows, I_quench):
    I_arr   = np.array([r["I_A"] for r in rows])
    dBz_arr = np.array([r["SCIF_dBz_mT"] for r in rows])
    pct_arr = np.array([r["SCIF_pct"] for r in rows])

    fig = _dark_fig((9, 7))
    ax1 = _dark_ax(fig.add_subplot(211))
    ax2 = _dark_ax(fig.add_subplot(212, sharex=ax1))

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.25, color="#555")
        if I_quench:
            ax.axvline(I_quench, color="yellow", ls="--", lw=1.0,
                       label=f"Quench {I_quench:.0f} A")

    ax1.plot(I_arr, dBz_arr, color="tomato", marker="o", lw=1.8, ms=5)
    ax1.axhline(0, color="#888", lw=0.7)
    ax1.set_ylabel("SCIF  ΔBz  [mT]", color="white")
    ax1.set_title(f"Screening-Current-Induced Field vs transport current\n"
                  f"(Δt = {params.ramp_duration:.0f} s  |  "
                  f"δ_SC = {params.delta_SC*1e6:.1f} µm)",
                  color="white")
    if I_quench:
        ax1.legend(fontsize=8, labelcolor="white", facecolor="#222", framealpha=0.6)

    ax2.plot(I_arr, pct_arr, color="orchid", marker="^", lw=1.8, ms=5)
    ax2.axhline(1.0, color="#888", ls="--", lw=0.9, label="1% reference")
    ax2.set_xlabel("Transport current I  [A]", color="white")
    ax2.set_ylabel("Relative SCIF  [%]", color="white")
    ax2.legend(fontsize=8, labelcolor="white", facecolor="#222", framealpha=0.6)

    fig.tight_layout()
    out = os.path.join(params.VIZ_DIR, "ta_scif_vs_current.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig); print(f"  Wrote {out}")


# ══════════════════════════════════════════════════════════════════════════════
# DETAIL PLOTS  (at max-safe current; match style of uploaded scripts exactly)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_field_top(I_amps, Bz_TA, Bz_u):
    """Dark-theme top-down |B| — matches plot_fields.py plot_top() style."""
    a, b  = params.a, params.b
    g     = float(params.coil_half_gap)
    nx, ny = 200, 100
    xs = np.linspace(-b*1.18, b*1.18, nx)
    ys = np.linspace(-b*0.68, b*0.68, ny)
    Xg, Yg = np.meshgrid(xs, ys)
    fp = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, g)])

    print(f"  ta_field_top: multi-filament Biot-Savart at z={g*1e3:.0f} mm …")
    B_tot = compute_both_coils_field_multilayer(fp, I_per_turn=I_amps,
                                                n_straight=400, n_cap=300)
    Bm  = np.linalg.norm(B_tot, axis=1).reshape(Xg.shape)
    Bxg = B_tot[:, 0].reshape(Xg.shape)
    Byg = B_tot[:, 1].reshape(Xg.shape)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#111"); ax.set_facecolor("#111")

    cf = ax.pcolormesh(Xg*1e3, Yg*1e3, Bm, shading="auto", cmap="magma")
    _dark_cbar(fig, cf, ax, "|B|  (T)")

    stride = max(1, nx // 22)
    sl = (slice(None, None, stride), slice(None, None, stride))
    Bxs, Bys = Bxg[sl], Byg[sl]
    nrm = np.where(np.hypot(Bxs, Bys) > 0, np.hypot(Bxs, Bys), 1.0)
    ax.quiver(Xg[sl]*1e3, Yg[sl]*1e3, Bxs/nrm, Bys/nrm,
              np.hypot(Bxs, Bys), cmap="cool", pivot="mid",
              scale=40, width=0.003, alpha=0.80)
    ax.text(0.01, 0.01, "arrows: in-plane (Bx, By) direction",
            transform=ax.transAxes, color="white", fontsize=8, alpha=0.7)

    ox, oy = _outline()
    ax.plot(ox*1e3, oy*1e3, "w-", lw=1.2, alpha=0.6, label="coil centreline")

    rx, ry = REGION_X_M*1e3, REGION_Y_M*1e3
    ax.add_patch(mpatches.Rectangle((-rx/2, -ry/2), rx, ry,
                                     lw=2, edgecolor="lime", facecolor="none",
                                     zorder=5, label="Uniformity box"))

    ax.set_aspect("equal")
    ax.set_xlabel("x  (mm)", color="white"); ax.set_ylabel("y  (mm)", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.set_title(
        f"Top-down field  (z = {g*1e3:.0f} mm midplane)  [T-A corrected]\n"
        f"I = {I_amps:.0f} A/turn,  {params.n_turns_total} total turns  "
        f"|  bore Bz = {Bz_TA:.4f} T  (SCIF = {(Bz_TA - Bz_u)*1e3:+.1f} mT)",
        color="white", fontsize=10)
    ax.legend(fontsize=8, labelcolor="white", facecolor="#222", framealpha=0.5)

    out = os.path.join(params.VIZ_DIR, "ta_field_top.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig); print(f"  Wrote {out}")


def _plot_field_side(I_amps, Bz_TA):
    """Dark-theme y=0 side view — matches plot_fields.py plot_side() style."""
    a, b  = params.a, params.b
    g     = float(params.coil_half_gap)
    z_c2  = 2.0 * g
    margin = max(b*0.20, g*0.25)
    colors = _layer_colors()

    nx = getattr(params, "sideview_nx", 200)
    nz = getattr(params, "sideview_nz", 150) * 2
    xs = np.linspace(-b*1.15, b*1.15, nx)
    zs = np.linspace(-params._z_top - margin, z_c2 + params._z_top + margin, nz)
    Xs, Zs = np.meshgrid(xs, zs)
    fp = np.column_stack([Xs.ravel(), np.zeros(Xs.size), Zs.ravel()])

    print(f"  ta_field_side: multi-filament Biot-Savart ({fp.shape[0]} pts) …")
    B_tot = compute_both_coils_field_multilayer(fp, I_per_turn=I_amps,
                                                n_straight=400, n_cap=300)
    Bmag = np.linalg.norm(B_tot, axis=1).reshape(Xs.shape)
    Bxg  = B_tot[:, 0].reshape(Xs.shape)
    Bzg  = B_tot[:, 2].reshape(Xs.shape)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#111"); ax.set_facecolor("#111")

    _vmax = float(np.nanpercentile(Bmag, 98))
    cf = ax.pcolormesh(Xs*1e3, Zs*1e3, Bmag, shading="auto",
                       cmap="magma", vmin=0, vmax=_vmax)
    _dark_cbar(fig, cf, ax, "|B|  (T)")

    stride = max(1, Xs.shape[1] // 22)
    sl = (slice(None, None, stride), slice(None, None, stride))
    Bxs, Bzs = Bxg[sl], Bzg[sl]
    nrm = np.where(np.hypot(Bxs, Bzs) > 0, np.hypot(Bxs, Bzs), 1.0)
    ax.quiver(Xs[sl]*1e3, Zs[sl]*1e3, Bxs/nrm, Bzs/nrm,
              np.hypot(Bxs, Bzs), cmap="cool", pivot="mid",
              scale=40, width=0.003, alpha=0.80)

    for z_off in (0.0, z_c2):
        for i in range(params.n_layers):
            zb = (params.layer_z_bottoms[i] + z_off)*1e3
            zt = (params.layer_z_tops[i]    + z_off)*1e3
            ax.axhspan(zb, zt, alpha=0.12, color=colors[i],
                       label=(f"Layer {i} ({params.n_turns[i]}t)"
                              if z_off == 0 else None))

    ax.axhline(g*1e3, color="lime", lw=1.5, ls="--",
               label=f"midplane z={g*1e3:.0f} mm")
    for xc in (-b, b):
        ax.axvspan(xc*1e3-3, xc*1e3+3, color="cyan", alpha=0.25,
                   label="cap tips" if xc == -b else None)

    ax.set_aspect("equal")
    ax.set_xlabel("x  (mm)", color="white"); ax.set_ylabel("z  (mm)", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.set_title(f"Side view  (y = 0)  —  two-coil field  [T-A corrected]\n"
                 f"I = {I_amps:.0f} A/turn,  separation = {z_c2*1e3:.0f} mm  "
                 f"|  bore Bz = {Bz_TA:.4f} T",
                 color="white", fontsize=11)
    ax.legend(fontsize=8, labelcolor="white", facecolor="#222",
              framealpha=0.5, loc="upper right")

    out = os.path.join(params.VIZ_DIR, "ta_field_side.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig); print(f"  Wrote {out}")


def _plot_uniformity(I_amps, Bz_TA, Bz_u):
    """Dark-theme uniformity box — matches field_uniformity.py style exactly."""
    g = float(params.coil_half_gap)
    a, b = params.a, params.b

    def _grid(rx, ry, nx, ny):
        xs = np.linspace(-rx, rx, nx); ys = np.linspace(-ry, ry, ny)
        Xg, Yg = np.meshgrid(xs, ys)
        fp = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, g)])
        return Xg, Yg, fp

    Xb, Yb, fp_b = _grid(REGION_X_M/2, REGION_Y_M/2, GRID_NX, GRID_NY)
    Xs, Ys, fp_s = _grid(SURROUND_X_M/2, SURROUND_Y_M/2, SURROUND_NX, SURROUND_NY)

    print("  ta_uniformity: multi-filament Biot-Savart on grids …")
    B_b = compute_both_coils_field_multilayer(fp_b, I_per_turn=I_amps,
                                              n_straight=400, n_cap=300)
    B_s = compute_both_coils_field_multilayer(fp_s, I_per_turn=I_amps,
                                              n_straight=400, n_cap=300)

    # Apply SCIF correction: scale field magnitudes by |Bz_TA/Bz_u|
    # (the SCIF is approximately spatially uniform over the small bore box)
    corr  = Bz_TA / Bz_u if abs(Bz_u) > 1e-3 else 1.0
    Bmag  = np.linalg.norm(B_b, axis=1).reshape(Xb.shape) * abs(corr)
    Bmag_s = np.linalg.norm(B_s, axis=1).reshape(Xs.shape) * abs(corr)

    mean_B = float(Bmag.mean()); std_B = float(Bmag.std())
    ptp = (Bmag.max() - Bmag.min()) / mean_B * 100
    rms = std_B / mean_B * 100
    passed = ptp < TARGET_PCT

    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor("#111"); ax.set_facecolor("#0d0d1a")

    cf = ax.pcolormesh(Xs*1e3, Ys*1e3, Bmag_s, shading="auto", cmap="magma")
    _dark_cbar(fig, cf, ax, "|B|  (T)")

    rx, ry = REGION_X_M/2*1e3, REGION_Y_M/2*1e3
    ax.add_patch(mpatches.Rectangle((-rx, -ry), 2*rx, 2*ry,
                                     lw=2.5, edgecolor="lime",
                                     facecolor="none", zorder=5))
    ax.annotate(f"{REGION_X_M*1e3:.0f} mm × {REGION_Y_M*1e3:.0f} mm",
                xy=(0, ry), xytext=(0, ry + SURROUND_Y_M/2*1e3*0.16),
                ha="center", va="bottom", fontsize=9, color="lime",
                arrowprops=dict(arrowstyle="-", color="lime", lw=1.2))

    # Iso-contours inside box
    lvls = np.linspace(Bmag.min(), Bmag.max(), 8)
    ax.contour(Xb*1e3, Yb*1e3, Bmag, levels=lvls,
               colors="lime", linewidths=0.6, alpha=0.5)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.set_xlabel("x  (mm)", color="white", fontsize=11)
    ax.set_ylabel("y  (mm)", color="white", fontsize=11)
    ax.set_xlim(Xs.min()*1e3, Xs.max()*1e3)
    ax.set_ylim(Ys.min()*1e3, Ys.max()*1e3)
    ax.set_aspect("equal")

    ok  = "✓" if passed else "✗"
    ok2 = "✓" if rms < TARGET_PCT/2 else "✗"
    SCIF_mT = (Bz_TA - Bz_u)*1e3
    info = (f"Box: {REGION_X_M*1e3:.0f} mm × {REGION_Y_M*1e3:.0f} mm\n"
            f"z = {g*1e3:.0f} mm  (midplane)\n"
            f"I = {I_amps:.0f} A/turn  |  {params.n_turns_total} turns\n\n"
            f"Mean |B| = {mean_B:.5f} T\n"
            f"Std      = ±{std_B*1e6:.2f} μT\n"
            f"Range    = {(Bmag.max()-Bmag.min())*1e6:.1f} μT\n\n"
            f"P-t-p:  {ptp:.3f}%  {ok}\n"
            f"RMS:    {rms:.3f}%  {ok2}\n"
            f"Target < {TARGET_PCT:.0f}%\n\n"
            f"T-A SCIF correction:\n"
            f"  ΔBz = {SCIF_mT:+.2f} mT\n"
            f"  ({abs(SCIF_mT)/abs(Bz_u*1e3)*100:.2f}% of bore field)")
    ax.text(1.02, 0.98, info, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace", color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                      edgecolor="#555", alpha=0.95))

    colour = "limegreen" if passed else "tomato"
    status = "PASS" if passed else f"FAIL  ({ptp:.2f}%)"
    ax.set_title(
        f"Field uniformity — {REGION_X_M*1e3:.0f} mm × {REGION_Y_M*1e3:.0f} mm "
        f"bore region  [{status}]  [T-A corrected]\n"
        f"Background: {SURROUND_X_M*1e3:.0f} mm × {SURROUND_Y_M*1e3:.0f} mm  "
        f"|  lime contours = field inside box",
        color=colour, fontsize=10)

    out = os.path.join(params.VIZ_DIR, "ta_uniformity.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig); print(f"  Wrote {out}")
    return ptp, mean_B


def _expand_full_system(cents, vals):
    """Expand 1/8-domain cells to full two-coil system (4 quadrants × 2 coils)."""
    g = params.coil_half_gap
    c_p, v_p = [], []
    for sx in (1, -1):
        for sy in (1, -1):
            c = cents.copy(); c[:, 0] *= sx; c[:, 1] *= sy
            c_p.append(c); v_p.append(vals)
    coil1_c = np.vstack(c_p); coil1_v = np.concatenate(v_p)
    coil2_c = coil1_c.copy(); coil2_c[:, 2] += 2.0 * g
    return np.vstack([coil1_c, coil2_c]), np.concatenate([coil1_v, coil1_v])


def _plot_field_3d(I_amps, Bz_TA, cents, Bmag_coil):
    """Coil cells coloured by |B| (T-A), full two-coil system — matches plot_3d.py."""
    colors = _layer_colors()
    g = params.coil_half_gap

    if getattr(params, "use_eighth_symmetry", False):
        cents_f, Bmag_f = _expand_full_system(cents, Bmag_coil)
    else:
        c2 = cents.copy(); c2[:, 2] += 2.0*g
        cents_f = np.vstack([cents, c2])
        Bmag_f  = np.tile(Bmag_coil, 2)

    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor("#0d0d1a")

    for col_idx, (elev, azim, vtitle) in enumerate([
            (28, -55, "Isometric"), (88, -90, "Top-down  (x-y)")]):
        ax = fig.add_subplot(1, 2, col_idx+1, projection="3d")
        _ax3d_style(ax)
        sc = ax.scatter(cents_f[:, 0]*1e3, cents_f[:, 1]*1e3, cents_f[:, 2]*1e3,
                        c=Bmag_f, cmap="magma", s=3, alpha=0.5, rasterized=True)
        for z_off in (0.0, 2.0*g):
            for i in range(params.n_layers):
                xo, yo = _racetrack_xy(params.a_out)
                for z_face in (params.layer_z_tops[i], params.layer_z_bottoms[i]):
                    zo = np.full_like(xo, (z_face + z_off)*1e3)
                    ax.plot(xo*1e3, yo*1e3, zo, color=colors[i], lw=0.5, alpha=0.45)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
        ax.set_title(vtitle, color="white", fontsize=10)

    _dark_cbar3d(fig, sc, fig.axes, "|B|  (T)  [T-A]")
    handles = [plt.Line2D([0], [0], color=colors[i], lw=2,
                           label=f"Layer {i} ({params.n_turns[i]}t)")
               for i in range(params.n_layers)]
    fig.legend(handles=handles, loc="lower center", ncol=params.n_layers,
               fontsize=8, labelcolor="white", facecolor="#111",
               framealpha=0.4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"|B| field distribution  [T-A corrected]  —  full two-coil system,  "
        f"{params.n_layers} layers,  I = {I_amps:.0f} A/turn,  "
        f"{params.n_turns_total} total turns\n"
        f"Bore Bz = {Bz_TA:.4f} T",
        color="white", fontsize=11, y=1.01)

    out = os.path.join(params.VIZ_DIR, "ta_field_3d.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig); print(f"  Wrote {out}")


def _plot_scif_3d(I_amps, cents, dJmag_coil):
    """Coil cells coloured by |ΔJ| (screening current magnitude)."""
    colors = _layer_colors()
    g = params.coil_half_gap

    if getattr(params, "use_eighth_symmetry", False):
        cents_f, dJ_f = _expand_full_system(cents, dJmag_coil)
    else:
        c2 = cents.copy(); c2[:, 2] += 2.0*g
        cents_f = np.vstack([cents, c2])
        dJ_f    = np.tile(dJmag_coil, 2)

    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor("#0d0d1a")

    for col_idx, (elev, azim, vtitle) in enumerate([
            (28, -55, "Isometric"), (88, -90, "Top-down  (x-y)")]):
        ax = fig.add_subplot(1, 2, col_idx+1, projection="3d")
        _ax3d_style(ax)
        sc = ax.scatter(cents_f[:, 0]*1e3, cents_f[:, 1]*1e3, cents_f[:, 2]*1e3,
                        c=dJ_f/1e9, cmap="plasma", s=3, alpha=0.5, rasterized=True)
        for z_off in (0.0, 2.0*g):
            for i in range(params.n_layers):
                xo, yo = _racetrack_xy(params.a_out)
                for z_face in (params.layer_z_tops[i], params.layer_z_bottoms[i]):
                    zo = np.full_like(xo, (z_face + z_off)*1e3)
                    ax.plot(xo*1e3, yo*1e3, zo, color=colors[i], lw=0.5, alpha=0.45)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
        ax.set_title(vtitle, color="white", fontsize=10)

    _dark_cbar3d(fig, sc, fig.axes, "|ΔJ|  (GA/m²)  screening current")
    handles = [plt.Line2D([0], [0], color=colors[i], lw=2,
                           label=f"Layer {i} ({params.n_turns[i]}t)")
               for i in range(params.n_layers)]
    fig.legend(handles=handles, loc="lower center", ncol=params.n_layers,
               fontsize=8, labelcolor="white", facecolor="#111",
               framealpha=0.4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Screening-current magnitude |ΔJ| = |J_TA − J_uniform|  —  "
        f"I = {I_amps:.0f} A/turn,  Δt = {params.ramp_duration:.0f} s\n"
        f"Full two-coil system  |  {params.n_turns_total} total turns",
        color="white", fontsize=11, y=1.01)

    out = os.path.join(params.VIZ_DIR, "ta_scif_3d.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig); print(f"  Wrote {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        build_mesh.build(write_path=params.mesh_filename)
    comm.barrier()

    mesh_data  = gmshio.read_from_msh(params.mesh_filename, comm, rank=0, gdim=3)
    domain, cell_tags, facet_tags = (mesh_data.mesh,
                                      mesh_data.cell_tags, mesh_data.facet_tags)

    ic_mod = IcModel(csv_path=params.shanghai_csv_filename)
    n_mod  = NValueModel(csv_path=params.n_value_csv_filename)

    uniform_setup = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ta = setup_ta_problem(domain, cell_tags, facet_tags, uniform_setup)
    coil_cells = ta["coil_cells"]

    I_max_safe, I_quench = (
        _max_safe_current(SWEEP_CURRENTS) if comm.rank == 0 else (None, None))
    I_max_safe = comm.bcast(I_max_safe, root=0)
    I_quench   = comm.bcast(I_quench,   root=0)

    print(f"\n{'='*62}")
    print(f"  T-A SWEEP  |  {SWEEP_CURRENTS[0]}–{SWEEP_CURRENTS[-1]} A  "
          f"|  Δt = {params.ramp_duration:.0f} s")
    print(f"  Max safe: {I_max_safe:.0f} A  "
          + (f" (quench at {I_quench:.1f} A)" if I_quench else ""))
    print(f"{'='*62}\n")

    rows = []

    for I_amps in SWEEP_CURRENTS:
        t0 = time.time()
        print(f"── I = {I_amps:.0f} A ─────────────────────────────────")

        *_, info = solve_ta_at_current(domain, ta, uniform_setup,
                                        I_amps=float(I_amps),
                                        ic_model=ic_mod, n_model=n_mod,
                                        verbose=True)

        dB_bore, Bz_u, Bz_TA, cents = _scif_at_bore(ta, domain, float(I_amps))
        dBz    = float(dB_bore[2])
        dBz_mT = dBz * 1e3
        rel_pct = abs(dBz) / abs(Bz_u) * 100 if abs(Bz_u) > 1e-3 else 0.0

        elapsed = time.time() - t0
        print(f"  Bz_unif = {Bz_u:+.4f} T  |  "
              f"SCIF = {dBz_mT:+.2f} mT ({rel_pct:.3f}%)  |  "
              f"Bz_TA = {Bz_TA:+.4f} T  |  {elapsed:.0f} s")

        # Detail figures only at max safe current
        if comm.rank == 0 and abs(I_amps - I_max_safe) < 1:
            print(f"\n  ── Detail figures at {I_amps:.0f} A ──")
            J_TA   = _J_from_T(ta, domain)
            J_unif = ta["t_hat_coil"] * (float(I_amps) / (ta["delta_SC"] * float(params.w)))
            dJ_arr = J_TA - J_unif

            # B at coil cells from the converged solve (ta["B_fn"] was
            # interpolated from curl(A_h) on the final Picard iteration)
            Bmag_coil = np.linalg.norm(
                ta["B_fn"].x.array.reshape(-1, 3)[coil_cells], axis=1)
            dJmag_coil = np.linalg.norm(dJ_arr, axis=1)

            _plot_field_top(float(I_amps), Bz_TA, Bz_u)
            _plot_field_side(float(I_amps), Bz_TA)
            _plot_uniformity(float(I_amps), Bz_TA, Bz_u)
            _plot_field_3d(float(I_amps), Bz_TA, cents, Bmag_coil)
            _plot_scif_3d(float(I_amps), cents, dJmag_coil)
            print()

        rows.append(dict(
            I_A               = float(I_amps),
            Bz_bore_uniform_T = round(Bz_u,   6),
            Bz_bore_TA_T      = round(Bz_TA,  6),
            SCIF_dBz_mT       = round(dBz_mT, 4),
            SCIF_pct          = round(rel_pct, 4),
            n_iters           = info["n_iters"] if info["converged"]
                                else -info["n_iters"],
            time_s            = round(elapsed, 1),
        ))

    if comm.rank == 0:
        csv_out = os.path.join(params.VIZ_DIR, "ta_sweep_results.csv")
        with open(csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"Saved CSV → {csv_out}")

        print("\nGenerating sweep summary plots …")
        _plot_bore_vs_current(rows, I_quench)
        _plot_scif_vs_current(rows, I_quench)

        print(f"\n{'='*62}")
        print(f"  T-A SWEEP SUMMARY  (Δt = {params.ramp_duration:.0f} s)")
        print(f"{'='*62}")
        print(f"  {'I [A]':>7}  {'Bz_unif [T]':>12}  {'Bz_TA [T]':>11}  "
              f"{'SCIF [mT]':>10}  {'SCIF %':>7}")
        print(f"  {'-'*58}")
        for r in rows:
            mk = " ← max safe" if abs(r["I_A"] - I_max_safe) < 1 else ""
            print(f"  {r['I_A']:>7.0f}  {r['Bz_bore_uniform_T']:>+12.4f}  "
                  f"{r['Bz_bore_TA_T']:>+11.4f}  "
                  f"{r['SCIF_dBz_mT']:>+10.2f}  "
                  f"{r['SCIF_pct']:>7.3f}%{mk}")
        print(f"{'='*62}")
        print(f"\nAll figures → {params.VIZ_DIR}/")


if __name__ == "__main__":
    main()
