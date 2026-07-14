"""
mechanical_stress_check.py
==========================
First-order Lorentz-force / stress screen for the racetrack pair.

Computes the J×B body force on every coil cell from the validated
uniform-J FEM field solution (eighth-symmetry domain — coil 2 enters via
the PMC image) and derives analytic stress estimates:

  1. CAP HOOP STRESS  σ_θ = f_r · r  (local thin-ring equilibrium, each
     turn assumed to carry its own load — conservative for tension; a
     bonded winding pack shares load and lowers the peak).
  2. TRANSVERSE (through-winding) STRESS in the straight legs:
     σ_n(y) = ∫ f_n dy integrated across the winding depth per layer.
     This acts along the tape stacking direction (the REBCO c-axis /
     delamination direction — the weakest axis of the conductor).
  3. STRAIGHT-LEG LINE LOAD  w [kN/m]: outward magnetic load the straight
     sections must carry by tension + bending / external structure
     (straights have no hoop curvature to react the load).
  4. AXIAL: net attraction of coil 1 toward coil 2 and the axial
     compression it implies on the support structure.

Allowables (typical 4 mm REBCO tape, adjust to vendor data):
  axial (lengthwise) tension  ~ 500 MPa  (irreversible-degradation limit)
  transverse tension (delam.) ~  30 MPa  (10–100 MPa scatter — WEAK axis)
  transverse compression      ~ 150 MPa

NOT modelled: winding-pack modulus / load sharing, cooldown prestress,
impregnation, friction, support structure.  This is a screen for the
configuration optimizer, not a structural analysis.

Run from Racetrack_v4 root:
    conda run -n fenicsx-env python3 validation/mechanical_stress_check.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
from mpi4py import MPI
import dolfinx.mesh as dmesh
import solve as base_solve
from current_source import normal_xy, tangent_xy

# ── Allowables [Pa] ──────────────────────────────────────────────────────────
SIGMA_AXIAL_ALLOW   = 500e6   # tape lengthwise tension
SIGMA_DELAM_ALLOW   =  30e6   # transverse tension (delamination)
SIGMA_TCOMP_ALLOW   = 150e6   # transverse compression

I_OP = float(params.I_design)


def main():
    comm = MPI.COMM_WORLD

    # ── FEM field at operating current ───────────────────────────────────
    domain, cell_tags, facet_tags = base_solve.setup_mesh(comm)
    setup = base_solve.setup_problem(domain, cell_tags, facet_tags)
    A_h, B_h = base_solve.solve_at_current(domain, setup, I_OP, comm,
                                           verbose_label="stress check")

    cells = setup["coil_cells"]
    cents = dmesh.compute_midpoints(domain, domain.topology.dim, cells)
    B     = B_h.x.array.reshape(-1, 3)[cells]

    # exact cell volumes
    import ufl
    from dolfinx import fem
    Vv = fem.functionspace(domain, ("DG", 0))
    vf = fem.Function(Vv)
    vf.interpolate(fem.Expression(ufl.CellVolume(domain),
                                  Vv.element.interpolation_points))
    vol = vf.x.array[cells]

    # ── J×B body force density ───────────────────────────────────────────
    L  = params.b - params.a
    Je = I_OP / (params.t * params.w)              # engineering J [A/m²]
    tx, ty = tangent_xy(cents[:, 0], cents[:, 1], L)
    nx, ny = normal_xy(cents[:, 0], cents[:, 1], L)
    t_hat = np.column_stack([tx, ty, np.zeros(len(cells))])
    n_hat = np.column_stack([nx, ny, np.zeros(len(cells))])

    f = np.cross(Je * t_hat, B)                     # [N/m³]
    f_n = np.einsum("ij,ij->i", f, n_hat)           # transverse (stack dir)
    f_z = f[:, 2]                                    # axial

    print(f"\n{'='*66}")
    print(f"  MECHANICAL STRESS SCREEN   I = {I_OP:.0f} A/turn   "
          f"Je = {Je/1e8:.2f}e8 A/m²")
    print(f"  |B| at conductor: mean {np.linalg.norm(B,axis=1).mean():.2f} T, "
          f"max {np.linalg.norm(B,axis=1).max():.2f} T")
    print(f"{'='*66}")

    # ── 1. Cap hoop stress (|x| > L) ─────────────────────────────────────
    cap = np.abs(cents[:, 0]) > L
    cx  = np.where(cents[cap, 0] > 0, L, -L)
    r_c = np.hypot(cents[cap, 0] - cx, cents[cap, 1])
    sigma_hoop = f_n[cap] * r_c                     # [Pa], + = tension
    print(f"\n  1. CAP HOOP STRESS (self-supported-turn estimate)")
    print(f"     max  : {sigma_hoop.max()/1e6:8.1f} MPa"
          f"   (allowable {SIGMA_AXIAL_ALLOW/1e6:.0f} MPa)"
          f"   {'OK' if sigma_hoop.max() < SIGMA_AXIAL_ALLOW else '** EXCEEDED **'}")
    print(f"     mean : {sigma_hoop.mean()/1e6:8.1f} MPa   "
          f"95th pct: {np.percentile(sigma_hoop,95)/1e6:.1f} MPa")

    # ── 2. Transverse stress across the winding (straight legs) ─────────
    straight = np.abs(cents[:, 0]) < 0.6 * L
    zc = np.array([(t_ + b_) / 2 for t_, b_ in
                   zip(params.layer_z_tops, params.layer_z_bottoms)])
    assign = np.argmin(np.abs(cents[:, 2][:, None] - zc[None, :]), axis=1)

    print(f"\n  2. TRANSVERSE (delamination-direction) STRESS, straight legs")
    print(f"     σ_n(y) = ∫ f_n dy from the outer edge inward, per layer")
    profiles = {}
    worst_t, worst_c = 0.0, 0.0
    for li in range(params.n_layers):
        m = straight & (assign == li)
        if m.sum() < 5:
            continue
        y  = cents[m, 1]
        fy = f_n[m]
        bins = np.linspace(params.a_inner_list[li], params.a_out, 25)
        ib = np.digitize(y, bins)
        fb = np.array([fy[ib == k].mean() if (ib == k).any() else 0.0
                       for k in range(1, len(bins))])
        dy = np.diff(bins)
        # integrate outward-acting force from outer surface inward:
        # tension between turns where inner turns push outward
        sig = -np.cumsum((fb * dy)[::-1])[::-1]     # [Pa] at bin inner edges
        yc  = 0.5 * (bins[:-1] + bins[1:])
        profiles[li] = (yc, sig)
        worst_t = max(worst_t, sig.max())
        worst_c = min(worst_c, sig.min())
    print(f"     peak transverse tension    : {worst_t/1e6:7.2f} MPa"
          f"   (allowable {SIGMA_DELAM_ALLOW/1e6:.0f} MPa)"
          f"   {'OK' if worst_t < SIGMA_DELAM_ALLOW else '** EXCEEDED **'}")
    print(f"     peak transverse compression: {abs(worst_c)/1e6:7.2f} MPa"
          f"   (allowable {SIGMA_TCOMP_ALLOW/1e6:.0f} MPa)"
          f"   {'OK' if abs(worst_c) < SIGMA_TCOMP_ALLOW else '** EXCEEDED **'}")

    # ── 3. Straight-leg line load ─────────────────────────────────────────
    # quarter domain holds half of one leg (x ≥ 0, y > 0 leg)
    leg_len_half = 0.6 * L
    w_line = np.sum(f_n[straight] * vol[straight]) / leg_len_half  # [N/m]
    print(f"\n  3. STRAIGHT-LEG OUTWARD LINE LOAD")
    print(f"     w ≈ {w_line/1e3:.1f} kN/m per leg — must be reacted by "
          f"leg tension/bending or external structure")

    # ── 4. Axial forces ───────────────────────────────────────────────────
    Fz_quarter = np.sum(f_z * vol)
    Fz_coil    = 4.0 * Fz_quarter                   # full coil 1
    plan_area  = 2 * (2*L*(params.a_out - min(params.a_inner_list))
                      + np.pi*(params.a_out**2 - min(params.a_inner_list)**2)/2)
    print(f"\n  4. AXIAL (coil-coil) FORCE")
    print(f"     net Fz on coil 1: {Fz_coil/1e3:+.1f} kN "
          f"({'toward' if Fz_coil > 0 else 'away from'} coil 2)")
    print(f"     mean axial pressure on support ≈ "
          f"{abs(Fz_coil)/plan_area/1e6:.1f} MPa over the winding footprint")

    print(f"\n  NOTE: single-turn-self-support hoop estimate is conservative;")
    print(f"  bonded winding shares hoop load.  No cooldown prestress, no")
    print(f"  structure.  Use as an optimizer screen, not a design proof.")
    print(f"{'='*66}")

    # ── figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("#111")
    for ax in axes:
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.grid(True, alpha=0.2, color="#555")

    # (a) hoop stress map on cap cells
    sc0 = axes[0].scatter(cents[cap, 0]*1e3, cents[cap, 1]*1e3,
                          c=sigma_hoop/1e6, cmap="magma", s=8)
    cb0 = fig.colorbar(sc0, ax=axes[0]); cb0.set_label("σ_hoop [MPa]", color="white")
    cb0.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb0.ax.yaxis.get_ticklabels(), color="white")
    axes[0].set_title(f"Cap hoop stress (max {sigma_hoop.max()/1e6:.0f} MPa)",
                      color="white", fontsize=10)
    axes[0].set_xlabel("x [mm]", color="white"); axes[0].set_ylabel("y [mm]", color="white")
    axes[0].set_aspect("equal")

    # (b) transverse stress profiles
    cmap = matplotlib.colormaps.get_cmap("tab10")
    for li, (yc, sig) in profiles.items():
        axes[1].plot(yc*1e3, sig/1e6, "-o", ms=3, lw=1.4,
                     color=cmap(li % 10), label=f"layer {li}")
    axes[1].axhline(SIGMA_DELAM_ALLOW/1e6, color="tomato", ls="--", lw=1,
                    label="delam. allowable")
    axes[1].axhline(0, color="#888", lw=0.7)
    axes[1].set_title("Transverse stress across winding (straight legs)",
                      color="white", fontsize=10)
    axes[1].set_xlabel("y [mm]", color="white")
    axes[1].set_ylabel("σ_n [MPa]  (+ tension)", color="white")
    axes[1].legend(fontsize=7, labelcolor="white", facecolor="#222",
                   framealpha=0.6, ncol=2)

    # (c) |f| force-density map, top view
    fmag = np.linalg.norm(f, axis=1)
    sc2 = axes[2].scatter(cents[:, 0]*1e3, cents[:, 1]*1e3,
                          c=fmag/1e9, cmap="plasma", s=6)
    cb2 = fig.colorbar(sc2, ax=axes[2]); cb2.set_label("|J×B| [GN/m³]", color="white")
    cb2.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color="white")
    axes[2].set_title("Lorentz force density (top view)", color="white", fontsize=10)
    axes[2].set_xlabel("x [mm]", color="white"); axes[2].set_ylabel("y [mm]", color="white")
    axes[2].set_aspect("equal")

    fig.suptitle(f"Mechanical stress screen — I = {I_OP:.0f} A/turn, "
                 f"two-coil racetrack, {params.n_turns_total} turns",
                 color="white", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(params.VIZ_DIR, "mechanical_stress_check.png")
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved figure → {out}")

    _force_direction_figure(cents, f, vol, L)


def _force_direction_figure(cents, f, vol, L):
    """
    Quiver maps of the J×B force DIRECTION on the winding:
      (a) top view (x-y): in-plane force vectors — the outward 'bursting'
          load; arrows point where the conductor is pushed.
      (b) straight-leg cross-section (y-z at |x| < 0.2L): transverse +
          axial force vectors on the winding pack — the components that
          load the tape stack (delamination direction) and squeeze the
          pancakes toward coil 2.
      (c) cap cross-section (r-z along the 45° cap diagonal): radial +
          axial force pattern in the end caps.
    """
    fig, axes = plt.subplots(1, 3, figsize=(19, 6),
                             gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    fig.patch.set_facecolor("#111")
    for ax in axes:
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.grid(True, alpha=0.15, color="#555")

    fmag = np.linalg.norm(f, axis=1)

    def _quiver(ax, X, Y, U, V, C, n_max=420):
        idx = np.arange(len(X))
        if len(idx) > n_max:
            idx = np.random.default_rng(0).choice(idx, n_max, replace=False)
        nrm = np.hypot(U[idx], V[idx]) + 1e-30
        q = ax.quiver(X[idx], Y[idx], U[idx]/nrm, V[idx]/nrm, C[idx],
                      cmap="plasma", pivot="tail", scale=32,
                      width=0.004, headwidth=3.2, alpha=0.95)
        return q

    # (a) top view — in-plane force direction
    q0 = _quiver(axes[0], cents[:, 0]*1e3, cents[:, 1]*1e3,
                 f[:, 0], f[:, 1], fmag/1e9)
    cb0 = fig.colorbar(q0, ax=axes[0], pad=0.02)
    cb0.set_label("|f| [GN/m³]", color="white")
    cb0.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb0.ax.yaxis.get_ticklabels(), color="white")
    axes[0].set_title("Top view — in-plane force direction\n"
                      "(winding is pushed OUTWARD: bursting load)",
                      color="white", fontsize=10)
    axes[0].set_xlabel("x [mm]", color="white")
    axes[0].set_ylabel("y [mm]", color="white")
    axes[0].set_aspect("equal")

    # (b) straight-leg cross-section — (f_y, f_z)
    m = np.abs(cents[:, 0]) < 0.2 * L
    q1 = _quiver(axes[1], cents[m, 1]*1e3, cents[m, 2]*1e3,
                 f[m, 1], f[m, 2], fmag[m]/1e9)
    cb1 = fig.colorbar(q1, ax=axes[1], pad=0.02)
    cb1.set_label("|f| [GN/m³]", color="white")
    cb1.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb1.ax.yaxis.get_ticklabels(), color="white")
    for zb in [t*1e3 for t in params.layer_z_tops] + \
              [params.layer_z_bottoms[-1]*1e3]:
        axes[1].axhline(zb, color="#333", lw=0.5)
    axes[1].annotate("to coil 2 / bore midplane ↑", xy=(0.5, 1.005),
                     xycoords="axes fraction", ha="center",
                     color="lime", fontsize=8)
    axes[1].set_title("Straight-leg cross-section (|x|<0.2L)\n"
                      "transverse (y) + axial (z) force on the stack",
                      color="white", fontsize=10)
    axes[1].set_xlabel("y — tape stacking direction [mm]", color="white")
    axes[1].set_ylabel("z — tape width direction [mm]", color="white")

    # (c) cap cross-section — force along the cap diagonal (r, z)
    cap = cents[:, 0] > L
    dx = cents[cap, 0] - L
    r  = np.hypot(dx, cents[cap, 1])
    phi = np.arctan2(cents[cap, 1], dx)
    band = np.abs(phi - np.pi/4) < np.pi/10       # ±18° around 45°
    r_hat = np.column_stack([np.cos(phi[band]), np.sin(phi[band])])
    f_r = (f[cap][band, 0]*r_hat[:, 0] + f[cap][band, 1]*r_hat[:, 1])
    q2 = _quiver(axes[2], r[band]*1e3, cents[cap][band, 2]*1e3,
                 f_r, f[cap][band, 2], fmag[cap][band]/1e9)
    cb2 = fig.colorbar(q2, ax=axes[2], pad=0.02)
    cb2.set_label("|f| [GN/m³]", color="white")
    cb2.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color="white")
    axes[2].set_title("End-cap cross-section (45° diagonal)\n"
                      "radial (hoop-loading) + axial force",
                      color="white", fontsize=10)
    axes[2].set_xlabel("r from cap centre [mm]", color="white")
    axes[2].set_ylabel("z [mm]", color="white")

    fig.suptitle(
        f"Lorentz force DIRECTION on the winding — I = {I_OP:.0f} A/turn "
        f"(arrows: unit force direction, colour: magnitude)",
        color="white", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(params.VIZ_DIR, "mechanical_force_directions.png")
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved force-direction figure → {out}")


if __name__ == "__main__":
    main()
