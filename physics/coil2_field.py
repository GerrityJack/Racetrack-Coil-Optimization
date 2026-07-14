"""
coil2_field.py
==============
Biot-Savart field contribution from the second (virtual) coil in the
symmetric two-coil configuration, added via superposition.

Geometry
--------
  Coil 1  (FEM mesh)  :  centred at  z = 0
  Coil 2  (virtual)   :  centred at  z = 2 * params.coil_half_gap
  Midplane            :  z = params.coil_half_gap

"coil_half_gap" is the distance from each coil's geometric centre to the
midplane between them, i.e. half the centre-to-centre separation.
The face-to-face gap between the two winding packs is:
    2 * coil_half_gap  −  params.w

Both coils carry the same current in the same circulation sense (both
clockwise when viewed from +z in our parameterisation), so their fields
add constructively at the midplane and inside each coil.

Why the filament approximation is valid here
--------------------------------------------
The Biot-Savart integral uses the coil 2 centreline as a single filament
carrying I_total = n_turns × I_per_turn.  The minimum distance from any
coil 1 cell (near z = 0) to the coil 2 centreline (at z = 2g) is:
    r_min ≥ 2 * coil_half_gap

For a reasonable coil_half_gap (≥ 20 mm), this is always much larger
than the winding-pack cross-section (~7.5 mm × 4 mm), so the filament
approximation introduces < 1 % error in the far-field regime.  No
softening distance is needed.

Superposition
-------------
Since linear magnetostatics with no magnetic materials is exactly linear
in the source current:
    B_total(cell, I) = B_FEM(cell, I) + B_coil2(cell, I)
                     = B_FEM(cell, I) + B_coil2_unit(cell) × I

where B_coil2_unit is computed once at I_per_turn = 1 A and then scaled
to every sweep current.  This is how quench_sweep.py uses this module.
"""
import numpy as np

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params

mu0 = 4.0 * np.pi * 1e-7


# ---------------------------------------------------------------------------
# Centreline discretisation
# ---------------------------------------------------------------------------

def coil2_centreline(a=None, b=None, coil_half_gap=None,
                     n_straight=600, n_cap=400):
    """
    Discretised closed centreline of coil 2, at z = 2 * coil_half_gap.
    Returns (seg_mid, dl) where:
      seg_mid : (M, 3) midpoints of each segment
      dl      : (M, 3) directed segment vectors  (dl = t_hat * ds)
                pointing in the direction of current flow
    """
    a   = a   if a   is not None else params.a
    b   = b   if b   is not None else params.b
    coil_half_gap = (coil_half_gap if coil_half_gap is not None
                     else params.coil_half_gap)

    L    = b - a
    z_c2 = 2.0 * coil_half_gap   # z-coordinate of coil 2 centre

    pts = []
    # top straight: left to right
    pts += [(-L + 2*L*i/n_straight, a,  z_c2) for i in range(n_straight)]
    # right cap
    th_r = np.linspace(np.pi/2, -np.pi/2, n_cap, endpoint=False)
    pts += [(L + a*np.cos(th), a*np.sin(th), z_c2) for th in th_r]
    # bottom straight: right to left
    pts += [(L - 2*L*i/n_straight, -a, z_c2) for i in range(n_straight)]
    # left cap
    th_l = np.linspace(-np.pi/2, -3*np.pi/2, n_cap, endpoint=False)
    pts += [(-L + a*np.cos(th), a*np.sin(th), z_c2) for th in th_l]

    pts = np.array(pts, dtype=np.float64)
    seg_mid = 0.5 * (pts + np.roll(pts, -1, axis=0))
    dl      = np.roll(pts, -1, axis=0) - pts     # directed segment vectors
    return seg_mid, dl


# ---------------------------------------------------------------------------
# Biot-Savart computation
# ---------------------------------------------------------------------------

def compute_coil2_field(field_points, I_per_turn=1.0, n_turns=None,
                         a=None, b=None, coil_half_gap=None,
                         n_straight=600, n_cap=400):
    """
    Biot-Savart field from coil 2 at arbitrary field_points.

    Parameters
    ----------
    field_points  : (N, 3) array of evaluation positions [m]
    I_per_turn    : transport current per turn [A]  (default 1 A for unit field)
    n_turns       : number of turns  (default params.n_turns_total)
    a, b          : coil geometry [m]  (default params.a, params.b)
    coil_half_gap : [m]  (default params.coil_half_gap)

    Returns
    -------
    B  : (N, 3) magnetic field vectors [T]
    """
    n_turns = n_turns if n_turns is not None else params.n_turns_total
    a   = a   if a   is not None else params.a
    b   = b   if b   is not None else params.b
    coil_half_gap = (coil_half_gap if coil_half_gap is not None
                     else params.coil_half_gap)

    I_total = float(n_turns) * float(I_per_turn)
    seg_mid, dl = coil2_centreline(a, b, coil_half_gap, n_straight, n_cap)

    field_points = np.atleast_2d(np.asarray(field_points, dtype=np.float64))
    B = np.zeros_like(field_points)
    prefac = mu0 * I_total / (4.0 * np.pi)

    for i in range(len(field_points)):
        r    = field_points[i] - seg_mid           # (M, 3)
        r2   = np.sum(r * r, axis=1)              # (M,)
        denom = (r2 * np.sqrt(r2))                 # |r|^3
        B[i] = prefac * np.sum(
            np.cross(dl, r) / denom[:, None], axis=0)
    return B


# ---------------------------------------------------------------------------
# Superposition helper for quench_sweep.py
# ---------------------------------------------------------------------------

def superpose_coil2(B_all, I_solved, centroids,
                    n_turns=None, a=None, b=None, coil_half_gap=None):
    """
    Add the coil 2 Biot-Savart field to an existing (K, N, 3) field array.

    B_all     : (K, N, 3) — FEM field at N cells for K sweep currents
    I_solved  : (K,)      — per-turn current for each sweep
    centroids : (N, 3)    — coil cell positions

    Returns a modified copy of B_all (does not mutate the input).
    The coil 2 field is computed once at I=1 A and scaled to each I_solved.
    """
    n_turns = n_turns if n_turns is not None else params.n_turns_total
    a   = a   if a   is not None else params.a
    b   = b   if b   is not None else params.b
    coil_half_gap = (coil_half_gap if coil_half_gap is not None
                     else params.coil_half_gap)

    z_c2 = 2.0 * coil_half_gap
    r_min = np.sqrt(np.min(
        np.sum((centroids - np.array([0, 0, z_c2]))**2, axis=1)))
    pack_diag = np.hypot(params.pack_thickness, params.w)

    print(f"  [coil2] centred at z = {z_c2*1e3:.1f} mm  "
          f"(coil_half_gap = {coil_half_gap*1e3:.1f} mm)")
    print(f"  [coil2] nearest coil-1 cell to coil-2 centreline: "
          f"{r_min*1e3:.1f} mm  (pack diagonal = {pack_diag*1e3:.1f} mm — "
          f"filament error < {pack_diag**2/(2*r_min**2)*100:.1f}%)")

    # Compute unit field once; scale to each sweep current
    print(f"  [coil2] computing Biot-Savart at {len(centroids)} cells …")
    B_unit = compute_coil2_field(centroids, I_per_turn=1.0, n_turns=n_turns,
                                  a=a, b=b, coil_half_gap=coil_half_gap)

    B_out = B_all.copy().astype(np.float64)
    for k, I_k in enumerate(I_solved):
        B_out[k] += B_unit * float(I_k)

    # sanity: report how much coil 2 adds at the highest current
    dB = np.linalg.norm(B_unit * I_solved[-1], axis=1)
    dBfem = np.linalg.norm(B_all[-1], axis=1)
    print(f"  [coil2] at I = {I_solved[-1]:.0f} A/turn:  "
          f"coil-2 adds median {np.median(dB):.4g} T  "
          f"({np.median(dB/np.maximum(dBfem,1e-12))*100:.1f}% of FEM coil-1 field)")
    return B_out


# ---------------------------------------------------------------------------
# Generalised helper: field from a single coil centred at any z
# ---------------------------------------------------------------------------

def compute_field_from_coil_at_z(field_points, z_center, I_per_turn=1.0,
                                   n_turns=None, a=None, b=None,
                                   n_straight=600, n_cap=400):
    """
    Biot-Savart field from a single racetrack coil centred at z = z_center.
    Interface is identical to compute_coil2_field but the z position is
    explicit.  Used internally by plot_fields.py to compute the midplane
    field from both coils.
    """
    n_turns = n_turns if n_turns is not None else params.n_turns_total
    a = a if a is not None else params.a
    b = b if b is not None else params.b
    L = b - a
    I_total = float(n_turns) * float(I_per_turn)

    # Build centreline at z_center
    pts = []
    pts += [(-L + 2*L*i/n_straight, a,  z_center) for i in range(n_straight)]
    th_r = np.linspace(np.pi/2, -np.pi/2, n_cap, endpoint=False)
    pts += [(L + a*np.cos(th), a*np.sin(th), z_center) for th in th_r]
    pts += [(L - 2*L*i/n_straight, -a, z_center) for i in range(n_straight)]
    th_l = np.linspace(-np.pi/2, -3*np.pi/2, n_cap, endpoint=False)
    pts += [(-L + a*np.cos(th), a*np.sin(th), z_center) for th in th_l]
    pts = np.array(pts, dtype=np.float64)
    seg_mid = 0.5 * (pts + np.roll(pts, -1, axis=0))
    dl      = np.roll(pts, -1, axis=0) - pts

    field_points = np.atleast_2d(np.asarray(field_points, dtype=np.float64))
    B = np.zeros_like(field_points)
    prefac = mu0 * I_total / (4.0 * np.pi)
    for i in range(len(field_points)):
        r    = field_points[i] - seg_mid
        r2   = np.sum(r * r, axis=1)
        denom = r2 * np.sqrt(r2)
        B[i] = prefac * np.sum(np.cross(dl, r) / denom[:, None], axis=0)
    return B


def compute_both_coils_field(field_points, I_per_turn=None,
                              n_turns=None, a=None, b=None,
                              coil_half_gap=None, n_straight=600, n_cap=400):
    """
    Combined Biot-Savart field from BOTH coils at arbitrary field_points.
    Coil 1 at z = 0,  Coil 2 at z = 2 * coil_half_gap.
    Returns (B_total, B_coil1, B_coil2) each of shape (N, 3).
    """
    I_per_turn    = I_per_turn    if I_per_turn    is not None else params.I_design
    n_turns       = n_turns       if n_turns        is not None else params.n_turns_total
    a             = a             if a              is not None else params.a
    b             = b             if b              is not None else params.b
    coil_half_gap = coil_half_gap if coil_half_gap is not None else params.coil_half_gap

    z_c2 = 2.0 * coil_half_gap
    kw   = dict(I_per_turn=I_per_turn, n_turns=n_turns, a=a, b=b,
                n_straight=n_straight, n_cap=n_cap)
    B1 = compute_field_from_coil_at_z(field_points, z_center=0.0,   **kw)
    B2 = compute_field_from_coil_at_z(field_points, z_center=z_c2,  **kw)
    return B1 + B2, B1, B2


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sanity check: field at coil 1 cell midplane and at the midplane bore
    print(f"coil_half_gap = {params.coil_half_gap*1e3:.1f} mm")
    print(f"Coil 2 centre: z = {2*params.coil_half_gap*1e3:.1f} mm")
    print(f"Midplane:       z = {params.coil_half_gap*1e3:.1f} mm")
    print()

    # Field at bore centre (x=0, y=0, z=coil_half_gap)
    bore = np.array([[0.0, 0.0, params.coil_half_gap]])
    B_bore = compute_coil2_field(bore, I_per_turn=params.I_design)
    print(f"Coil-2 field at bore midplane (x=y=0): |B| = "
          f"{np.linalg.norm(B_bore)*1000:.3f} mT")

    # Field at a representative coil 1 cell location
    cell = np.array([[0.0, params.a, 0.0]])
    B_cell = compute_coil2_field(cell, I_per_turn=params.I_design)
    print(f"Coil-2 field at coil-1 centreline (x=0, y=a): |B| = "
          f"{np.linalg.norm(B_cell)*1000:.3f} mT")
