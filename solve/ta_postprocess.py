"""
ta_postprocess.py
=================
Post-processing for the T-A screening-current solve.

Loads the .npz written by ta_solve.py and produces:
  1. SCIF summary (reads pre-computed dB_bore from npz — no approximation)
  2. Bore-axis Bz profile (uniform-J Biot-Savart, both coils)
  3. ΔJ_x colour map in the central z-layer of the winding pack

Run from Racetrack_v4 root:
    conda run -n fenicsx-env python3 solve/ta_postprocess.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
from coil2_field import compute_both_coils_field

# ── 1. Load saved fields ──────────────────────────────────────────────────────
npz_path = os.path.join(params.SOLVE_DIR, "racetrack_ta_fields.npz")
if not os.path.exists(npz_path):
    raise FileNotFoundError(f"No T-A results at {npz_path}. Run ta_solve.py first.")

data = np.load(npz_path, allow_pickle=False)

I_solved        = float(data["I_solved"])
centroids       = data["coil_centroids"]    # (N, 3)
B_coil          = data["coil_B"]            # (N, 3)  FEM coil-1 field
J_TA_coil       = data["J_TA_coil"]        # (N, 3)  T-A SC-layer J
J_unif_coil     = data["J_unif_coil"]      # (N, 3)  uniform SC-layer J
dB_bore         = data["dB_bore"]           # (3,)    bore SCIF vector
Bz_bore_uniform = float(data["Bz_bore_uniform"])
Bz_bore_TA      = float(data["Bz_bore_TA"])

print(f"Loaded T-A fields:  {len(centroids)} coil cells  "
      f"I = {I_solved:.1f} A  |B|_mean = {np.linalg.norm(B_coil,axis=1).mean():.3f} T")

# ── 2. SCIF summary from saved values (correct) ───────────────────────────────
dBz  = float(dB_bore[2])
dBmag = float(np.linalg.norm(dB_bore))

print(f"\n{'='*55}")
print(f"  SCREENING CURRENT SUMMARY  (from ta_solve.py)")
print(f"{'='*55}")
print(f"  Transport current I     = {I_solved:.1f} A")
print(f"  Ramp duration Δt        = {float(data.get('ramp_duration', 600)):.0f} s")
print(f"  Uniform-J bore Bz       = {Bz_bore_uniform:+.4f} T")
print(f"  SCIF ΔBz at bore        = {dBz*1e3:+.2f} mT")
print(f"  SCIF |ΔB| at bore       = {dBmag*1e3:.2f} mT")
print(f"  T-A bore Bz             = {Bz_bore_TA:+.4f} T")
print(f"  Relative SCIF           = {abs(dBz)/abs(Bz_bore_uniform)*100:.3f}%")
print(f"{'='*55}")

# ── 3. Bore-axis profile (uniform-J Biot-Savart, both coils) ─────────────────
print("\nComputing bore-axis profile …")
z_bore = np.linspace(0.0, params.coil_half_gap * 1.6, 250)
bore_pts = np.column_stack([np.zeros_like(z_bore),
                             np.zeros_like(z_bore), z_bore])
B_axis, _, _ = compute_both_coils_field(bore_pts, I_per_turn=I_solved)
Bz_axis = B_axis[:, 2]
mid_idx = np.argmin(np.abs(z_bore - params.coil_half_gap))
print(f"  Bz at bore midplane z={params.coil_half_gap*1e3:.0f} mm : {Bz_axis[mid_idx]:.4f} T")
print(f"  Design target : {getattr(params,'B_target',10.0):.1f} T  "
      f"({abs(Bz_axis[mid_idx])/getattr(params,'B_target',10.0)*100:.1f}% achieved)")

# ── 4. ΔJ_x map in the central z-layer ───────────────────────────────────────
w_tape = float(params.w)
tol_z  = w_tape / 8.0
mask_cl = np.abs(centroids[:, 2]) <= w_tape / 2.0 + tol_z
n_cl = mask_cl.sum()

if n_cl < 3:
    print("WARNING: no central-layer cells found — check params.w")
else:
    xy   = centroids[mask_cl, :2] * 1e3
    Jx   = J_TA_coil[mask_cl, 0]
    Jx_u = J_unif_coil[mask_cl, 0]
    dJx  = Jx - Jx_u
    print(f"\nCentral z-layer ({n_cl} cells):")
    print(f"  J_x range (T-A)   : [{Jx.min()/1e9:.3f}, {Jx.max()/1e9:.3f}] GA/m²")
    print(f"  J_x range (uniform): [{Jx_u.min()/1e9:.3f}, {Jx_u.max()/1e9:.3f}] GA/m²")
    print(f"  ΔJ_x range         : [{dJx.min()/1e6:.1f}, {dJx.max()/1e6:.1f}] MA/m²")

    # ── Figure 1: bore-axis profile + SCIF annotation ─────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(z_bore * 1e3, Bz_axis, "b-", lw=2, label="Uniform J (Biot-Savart)")
    ax1.axvline(params.coil_half_gap * 1e3, color="gray", ls="--", lw=0.9,
                label=f"Midplane  z={params.coil_half_gap*1e3:.0f} mm")
    ax1.axhline(Bz_bore_TA, color="r", ls=":", lw=1.2,
                label=f"T-A bore Bz = {Bz_bore_TA:.3f} T")
    ax1.set_xlabel("z  [mm]"); ax1.set_ylabel("$B_z$  [T]")
    ax1.set_title("Bore-axis $B_z$ — uniform J (both coils)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # ── Figure 2: ΔJ_x screening current map ──────────────────────────────
    vmax = np.abs(dJx).max() / 1e6 or 1.0
    sc = ax2.scatter(xy[:, 0], xy[:, 1], c=dJx / 1e6,
                     cmap="RdBu_r", s=8, linewidths=0,
                     vmin=-vmax, vmax=vmax)
    fig.colorbar(sc, ax=ax2, label="$\\Delta J_x = J_x^{TA} - J_x^{\\rm unif}$  [MA/m²]")
    ax2.set_xlabel("x [mm]"); ax2.set_ylabel("y [mm]")
    ax2.set_title(f"Screening-current ΔJ_x — central z-layer\n"
                  f"I = {I_solved:.0f} A  |  Δt = {float(data.get('ramp_duration',600)):.0f} s  "
                  f"|  SCIF = {abs(dBz)*1e3:.1f} mT ({abs(dBz)/abs(Bz_bore_uniform)*100:.2f}%)")
    ax2.set_aspect("equal"); ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    out_fig = os.path.join(params.SOLVE_DIR, "ta_scif_overview.png")
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"\nSaved overview figure → {out_fig}")

    # ── Figure 2: three-panel J comparison ────────────────────────────────
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax_i, (vals, lbl, unit) in zip(axes, [
        (Jx,   "J_x (T-A)",         "GA/m²"),
        (Jx_u, "J_x (uniform)",      "GA/m²"),
        (dJx,  "ΔJ_x = TA − unif",  "MA/m²"),
    ]):
        scale = 1e9 if "GA" in unit else 1e6
        v     = vals / scale
        vm    = np.abs(v).max() or 1.0
        sc2 = ax_i.scatter(xy[:, 0], xy[:, 1], c=v,
                           cmap="RdBu_r", s=8, linewidths=0,
                           vmin=-vm, vmax=vm)
        fig2.colorbar(sc2, ax=ax_i, label=f"{lbl}  [{unit}]")
        ax_i.set_xlabel("x [mm]"); ax_i.set_ylabel("y [mm]")
        ax_i.set_title(lbl); ax_i.set_aspect("equal"); ax_i.grid(True, alpha=0.2)

    fig2.suptitle(f"Central z-layer current distribution  "
                  f"(I={I_solved:.0f}A, Δt={float(data.get('ramp_duration',600)):.0f}s)", fontsize=11)
    plt.tight_layout()
    out_fig2 = os.path.join(params.SOLVE_DIR, "ta_J_central_layer.png")
    fig2.savefig(out_fig2, dpi=150)
    plt.close(fig2)
    print(f"Saved J_x panel figure → {out_fig2}")

