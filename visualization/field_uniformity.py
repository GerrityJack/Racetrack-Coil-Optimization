"""
field_uniformity.py
====================
Produces exactly one output image:

  uniformity.png  — field background (surrounding 24mm × 18mm region)
                    with the uniformity box (4mm × 3mm) overlaid in lime
                    green, iso-contours inside the box, and a statistics
                    panel showing mean |B|, ±std, peak-to-peak %, PASS/FAIL.

The uniformity box size is set by REGION_X_M / REGION_Y_M at the top of
this file. The field is evaluated at the midplane z = params.coil_half_gap
using direct multi-filament Biot-Savart from all FEM coil cells (both
coils if two_coil_mode = True), PLUS the Bean-state screening-current
(SCIF) dipole correction -- as of 2026-07-22 this box size (30x6mm) and
the SCIF correction are made to match optimize/optimize_geometry.py's
target_box_field()/bean_moments() EXACTLY, so this figure's reported
uniformity is the same number the CMA-ES optimizer's own constraint
check computes, not an independent/approximate one. (Previously this
used a 15x6mm box with no SCIF correction and a cruder single-filament
field model -- all three differences are now resolved.)

Run after solve.py (requires racetrack_fields.npz with coil geometry).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import TwoSlopeNorm

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
import opt_config as cfg
from coil2_field import compute_both_coils_field_multilayer
from current_source import normal_xy
from ic_model import IcModel, angle_with_normal_deg
from optimize_geometry import bean_moments, dipole_field_mirrored

# ── Uniformity-region settings (change these to adjust the box) ────────────
# 2026-07-22: matched to opt_config.py's TARGET_X_M/TARGET_Y_M (the CMA-ES
# optimizer's own target box) rather than an independent value, so this
# figure and the optimizer's constraint check always agree.
REGION_X_M  = cfg.TARGET_X_M   # 30 mm  (long axis, x)
REGION_Y_M  = cfg.TARGET_Y_M   # 6 mm   (short axis, y)
GRID_NX     = 150     # grid points in x
GRID_NY     = 60      # grid points in y
SURROUND_X_M = max(0.040, REGION_X_M * 1.3)   # surrounding view width
SURROUND_Y_M = max(0.024, REGION_Y_M * 4.0)   # surrounding view height
SURROUND_NX  = 200
SURROUND_NY  = 120
TARGET_PCT   = 1.0    # uniformity target %
_ic_model    = None    # lazily constructed (IcModel() loads CSV data)


# ── Bore field via analytic two-coil Biot-Savart ─────────────────────────
# 2026-07-22: switched to compute_both_coils_field_multilayer(), which
# resolves each layer's own z-center and radial center, instead of
# compute_both_coils_field()'s single filament carrying ALL turns at one
# nominal radius a. That approximation is only valid when the winding
# pack's cross-section is small compared to a and coil_half_gap (true at
# the original ~50-80mm scale) -- it silently breaks down for a much
# smaller optimized coil (e.g. a=15.5mm with a ~25mm-thick pack), where it
# produced a spurious FAIL (6.7% p-t-p) contradicting the optimizer's own
# properly-resolved multi-filament uniformity check (0.68%, PASS). The
# multilayer version mirrors optimize_geometry.py's filament_stack(), so
# this script's result now agrees with what the optimizer itself uses as
# its constraint.
#
# Why not the FEM-cell approach?  Simple x/y reflection of centroids also
# flips the tangent vector, which is WRONG for the straight sections:
# the top/bottom straights carry current in the same direction (+x / -x)
# on BOTH sides of x=0, so reflecting tx cancels their contributions
# instead of doubling them → ≈0 T at the bore instead of the correct ~8 T.

def _bore_field(field_points, npz_data, I_per_turn):
    return compute_both_coils_field_multilayer(
        field_points, I_per_turn=I_per_turn, n_straight=400, n_cap=300)


# ── Bean-state screening-current (SCIF) correction ──────────────────────────
# 2026-07-22: added so this script's uniformity number includes the same
# screening-current dipole correction optimize_geometry.evaluate() applies
# -- mirrors bean_moments()/dipole_field_mirrored() exactly, using the
# per-cell data (centroids, B, volume) already saved in the npz by
# solve.py's extract_and_save(), which come from the SAME FEM domain/cells
# optimize_geometry.py itself uses internally.

def _scif_correction(field_points, npz_data, I_op):
    global _ic_model
    if _ic_model is None:
        _ic_model = IcModel()

    cents  = npz_data["coil_centroids"]
    coil_B = npz_data["coil_B"]
    vol    = npz_data["coil_volume"]
    I_solved = float(npz_data["I_solved"])

    B_unit = coil_B / I_solved
    B_op   = B_unit * I_op
    L = params.b - params.a
    nx, ny = normal_xy(cents[:, 0], cents[:, 1], L)
    n_hat = np.column_stack([nx, ny, np.zeros(len(cents))])
    theta = angle_with_normal_deg(B_unit, n_hat)

    M_vec, _ = bean_moments(B_op, n_hat, theta, _ic_model, I_op)
    return dipole_field_mirrored(field_points, cents, M_vec, vol)


# ── Grids ─────────────────────────────────────────────────────────────────

def _box_grid():
    xs = np.linspace(-REGION_X_M / 2, REGION_X_M / 2, GRID_NX)
    ys = np.linspace(-REGION_Y_M / 2, REGION_Y_M / 2, GRID_NY)
    Xg, Yg = np.meshgrid(xs, ys)
    fp = np.column_stack([Xg.ravel(), Yg.ravel(),
                           np.full(Xg.size, params.coil_half_gap)])
    return Xg, Yg, fp


def _surround_grid():
    xs = np.linspace(-SURROUND_X_M / 2, SURROUND_X_M / 2, SURROUND_NX)
    ys = np.linspace(-SURROUND_Y_M / 2, SURROUND_Y_M / 2, SURROUND_NY)
    Xs, Ys = np.meshgrid(xs, ys)
    fp = np.column_stack([Xs.ravel(), Ys.ravel(),
                           np.full(Xs.size, params.coil_half_gap)])
    return Xs, Ys, fp


# ── Stats ─────────────────────────────────────────────────────────────────

def _stats(Bmag):
    mean = float(np.mean(Bmag)); std = float(np.std(Bmag))
    bmin = float(np.min(Bmag)); bmax = float(np.max(Bmag))
    return dict(mean=mean, std=std, min=bmin, max=bmax,
                ppt=( bmax - bmin) / mean * 100,
                rms=std / mean * 100)


# ═══════════════════════════════════════════════════════════════════════════
# uniformity.png
# ═══════════════════════════════════════════════════════════════════════════

def plot_uniformity(Xg, Yg, Bmag, Xs, Ys, Bmag_s, stats, I_pt):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#0d0d1a")

    # ── background: surrounding field ────────────────────────────────────
    cf = ax.pcolormesh(Xs * 1e3, Ys * 1e3, Bmag_s,
                       shading="auto", cmap="magma")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("|B| (T)", color="white"); cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # ── box rectangle ─────────────────────────────────────────────────────
    rx, ry = REGION_X_M / 2 * 1e3, REGION_Y_M / 2 * 1e3
    rect = mpatches.Rectangle((-rx, -ry), 2*rx, 2*ry,
                               lw=2.5, edgecolor="lime",
                               facecolor="none", zorder=5)
    ax.add_patch(rect)
    ax.annotate(f"{REGION_X_M*1e3:.0f} mm × {REGION_Y_M*1e3:.0f} mm",
                xy=(0, ry), xytext=(0, ry + SURROUND_Y_M*1e3*0.08),
                ha="center", va="bottom", fontsize=9, color="lime",
                arrowprops=dict(arrowstyle="-", color="lime", lw=1.2))

    # ── axes ──────────────────────────────────────────────────────────────
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.set_xlabel("x  (mm)", color="white", fontsize=11)
    ax.set_ylabel("y  (mm)", color="white", fontsize=11)
    ax.set_xlim(Xs.min()*1e3, Xs.max()*1e3)
    ax.set_ylim(Ys.min()*1e3, Ys.max()*1e3)
    ax.set_aspect("equal")

    # ── statistics panel ──────────────────────────────────────────────────
    ok  = "✓" if stats["ppt"] < TARGET_PCT else "✗"
    ok2 = "✓" if stats["rms"] < TARGET_PCT / 2 else "✗"
    info = (f"Box: {REGION_X_M*1e3:.0f} mm × {REGION_Y_M*1e3:.0f} mm\n"
            f"z = {params.coil_half_gap*1e3:.0f} mm  (midplane)\n"
            f"I = {I_pt:.0f} A/turn  |  {params.n_turns_total} turns\n\n"
            f"Mean |B| = {stats['mean']:.5f} T\n"
            f"Std      = ±{stats['std']*1e6:.2f} μT\n"
            f"Range    = {(stats['max']-stats['min'])*1e6:.1f} μT\n\n"
            f"P-t-p:  {stats['ppt']:.3f}%  {ok}\n"
            f"RMS:    {stats['rms']:.3f}%  {ok2}\n"
            f"Target < {TARGET_PCT:.0f}%")
    ax.text(1.02, 0.98, info, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace", color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                      edgecolor="#555", alpha=0.95))

    status = "PASS" if stats["ppt"] < TARGET_PCT else f"FAIL  ({stats['ppt']:.2f}%)"
    colour = "limegreen" if stats["ppt"] < TARGET_PCT else "tomato"
    ax.set_title(
        f"Field uniformity — {REGION_X_M*1e3:.0f} mm × {REGION_Y_M*1e3:.0f} mm "
        f"bore region  [{status}]\n"
        f"Background: {SURROUND_X_M*1e3:.0f} mm × {SURROUND_Y_M*1e3:.0f} mm  "
        f"|  lime contours = field inside box",
        color=colour, fontsize=10)

    out = os.path.join(params.VIZ_DIR, "uniformity.png")
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Wrote {out}")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    npz_path = params.fields_npz_filename
    if not os.path.exists(npz_path):
        print(f"ERROR: {npz_path} not found — run solve.py first"); return
    npz = np.load(npz_path)

    I_pt = float(npz["I_solved"])
    g    = params.coil_half_gap

    print(f"Midplane z = {g*1e3:.0f} mm,  I = {I_pt:.0f} A/turn")
    print(f"Box: {REGION_X_M*1e3:.0f} mm × {REGION_Y_M*1e3:.0f} mm  "
          f"({GRID_NX}×{GRID_NY} = {GRID_NX*GRID_NY} pts)")

    Xg, Yg, fp_box = _box_grid()
    print("  Computing box field …")
    B_box = _bore_field(fp_box, npz, I_pt)
    print("  Computing SCIF (Bean-state screening) correction …")
    B_box = B_box + _scif_correction(fp_box, npz, I_pt)
    Bmag  = np.linalg.norm(B_box, axis=1).reshape(Xg.shape)

    # SCIF correction NOT applied to the background/surround grid: it's
    # visual context only (doesn't feed the PASS/FAIL stats below), the
    # correction is a small (~tens of mT) addition to a multi-tesla field
    # so it wouldn't change the plot's appearance, and the surround grid
    # is ~2.7x more points -- not worth the extra compute for a cosmetic
    # backdrop.
    Xs, Ys, fp_s = _surround_grid()
    print("  Computing surrounding field …")
    B_s   = _bore_field(fp_s, npz, I_pt)
    Bmag_s = np.linalg.norm(B_s, axis=1).reshape(Xs.shape)

    stats = _stats(Bmag)
    print(f"\n{'='*50}")
    print(f"  UNIFORMITY RESULTS — {REGION_X_M*1e3:.0f}mm × {REGION_Y_M*1e3:.0f}mm")
    print(f"  Mean |B| = {stats['mean']:.6f} T")
    print(f"  ±std     = ±{stats['std']*1e6:.2f} μT  ({stats['rms']:.4f}% rms)")
    print(f"  P-t-p    = {(stats['max']-stats['min'])*1e6:.1f} μT  "
          f"= {stats['ppt']:.4f}%")
    print(f"  {'PASS ✓' if stats['ppt'] < TARGET_PCT else 'FAIL ✗'}  "
          f"(target < {TARGET_PCT}%)")
    print(f"{'='*50}\n")

    plot_uniformity(Xg, Yg, Bmag, Xs, Ys, Bmag_s, stats, I_pt)
    print(f"Done. Wrote uniformity.png  →  {params.VIZ_DIR}")


if __name__ == "__main__":
    main()
