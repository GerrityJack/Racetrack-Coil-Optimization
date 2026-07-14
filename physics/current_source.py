"""
current_source.py
==================
Geometric helpers for the racetrack coil: tangent/normal vectors,
arc-length, radial offset, turn indexing, and the 8-fold symmetry
expansion used in eighth-symmetry mode.

All functions work for any n_turns list (multi-layer) or scalar
(single layer) without hardcoding a specific number of layers.
"""
import numpy as np

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import params

mu0 = 4.0 * np.pi * 1e-7


# ── Basic geometry (z-independent, all functions work in the xy plane) ────────

def tangent_xy(x, y, L):
    """Unit tangent of the racetrack centreline at (x,y), in the xy-plane."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    tx = np.empty_like(x); ty = np.empty_like(y)
    straight = np.abs(x) <= L
    tx[straight] = 1.0
    ty[straight] = 0.0
    cap = ~straight
    cx = np.where(x[cap] > 0, L, -L)
    angle = np.arctan2(y[cap], x[cap] - cx)
    tx[cap] = np.sin(angle)
    ty[cap] = -np.cos(angle)
    return tx, ty


def normal_xy(x, y, L):
    """Outward unit normal (tape's out-of-plane broad-face normal) at (x,y)."""
    tx, ty = tangent_xy(x, y, L)
    return -ty, tx


def arc_length_xy(x, y, L, a):
    """Arc-length position along the coil centreline (0 → perimeter)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    P = 4.0 * L + 2.0 * np.pi * a
    s = np.empty_like(x)
    # top straight (left→right): s ∈ [0, 2L]
    top = (np.abs(x) <= L) & (y >= 0)
    s[top] = x[top] + L
    # right cap: s ∈ [2L, 2L + πa]
    rcap = x > L
    theta_r = np.arctan2(y[rcap], x[rcap] - L)
    theta_r = np.where(theta_r < 0, theta_r + 2*np.pi, theta_r)
    s[rcap] = 2*L + a * (np.pi/2 - theta_r)
    # bottom straight (right→left): s ∈ [2L+πa, 4L+πa]
    bot = (np.abs(x) <= L) & (y < 0)
    s[bot] = 2*L + np.pi*a + (L - x[bot])
    # left cap: s ∈ [4L+πa, 4L+2πa]
    lcap = x < -L
    theta_l = np.arctan2(y[lcap], x[lcap] + L)
    theta_l = np.where(theta_l > 0, theta_l - 2*np.pi, theta_l)
    s[lcap] = 4*L + np.pi*a + a * (-np.pi/2 - theta_l)
    return s % P


def radial_offset_xy(x, y, L, a):
    """Signed radial distance from the centreline at radius a."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = np.empty_like(x)
    straight = np.abs(x) <= L
    r[straight] = np.abs(y[straight])
    cap = ~straight
    cx = np.where(x[cap] > 0, L, -L)
    r[cap] = np.hypot(x[cap] - cx, y[cap])
    return r - a


def turn_index_xy(x, y, L, a, t, n_turns_scalar):
    """Turn index for a single-layer coil (0=innermost, n_turns-1=outermost)."""
    off = radial_offset_xy(x, y, L, a)
    pack = n_turns_scalar * t
    idx = np.floor((off + pack/2) / t).astype(int)
    return np.clip(idx, 0, n_turns_scalar - 1)


# ── Multi-layer helpers ───────────────────────────────────────────────────────

def layer_index_from_z(z):
    """
    Return which layer (0=top) a point at height z belongs to.
    Clips to [0, n_layers-1] for points slightly outside the stack.
    Works for any params.n_turns list.
    """
    z = np.asarray(z, dtype=np.float64)
    idx = np.floor((params._z_top - z) / params.w).astype(int)
    return np.clip(idx, 0, params.n_layers - 1)


def radial_offset_xy_multilayer(x, y, z, L):
    """
    Signed radial offset from the LAYER'S centreline radius.
    Layer is determined from z-coordinate via layer_index_from_z.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    idx = layer_index_from_z(z)
    a_c = np.array(params.a_center_list)[idx]

    r = np.empty_like(x)
    straight = np.abs(x) <= L
    r[straight] = np.abs(y[straight])
    cap = ~straight
    cx = np.where(x[cap] > 0, L, -L)
    r[cap] = np.hypot(x[cap] - cx, y[cap])
    return r - a_c


def turn_index_xy_multilayer(x, y, z, L):
    """
    Returns (layer_idx, turn_within_layer) for multi-layer coils.
    layer_idx: 0=top, n_layers-1=bottom.
    turn_within_layer: 0=innermost, n_i-1=outermost.
    Works for any params.n_turns list.
    """
    x  = np.asarray(x, dtype=np.float64)
    z  = np.asarray(z, dtype=np.float64)
    li = layer_index_from_z(z)
    pt = np.array(params.pack_thickness_list)[li]
    off = radial_offset_xy_multilayer(x, y, z, L)
    n_arr = np.array(params.n_turns)[li]
    turn  = np.floor((off + pt/2) / params.t).astype(int)
    return li, np.clip(turn, 0, n_arr - 1)


# ── 8-fold symmetry expansion ─────────────────────────────────────────────────

def expand_to_full_domain(centroids, B_vecs, coil_half_gap=None):
    """
    Mirror the FEM result from the fundamental octant (x≥0, y≥0, z≤g)
    to all 8 octants of the full two-coil system.

    Symmetry relations:
      x→-x : Bx→-Bx,  By→+By,  Bz→+Bz
      y→-y : Bx→+Bx,  By→-By,  Bz→+Bz
      z→2g-z: Bx→-Bx, By→-By,  Bz→+Bz  (same-sense Helmholtz pair)

    Bz is always even (same sign everywhere) — it is the main useful component.
    The field angle |B·n̂|/|B| used for quench analysis is preserved correctly.

    Returns (centroids_full, B_full) with 8× the input rows.
    """
    if coil_half_gap is None:
        coil_half_gap = getattr(params, "coil_half_gap", 0.0)
    g  = coil_half_gap
    z2 = 2.0 * g

    x, y, z  = centroids[:, 0], centroids[:, 1], centroids[:, 2]
    Bx, By, Bz = B_vecs[:, 0], B_vecs[:, 1], B_vecs[:, 2]

    all_c, all_B = [], []
    for sx in (+1, -1):
        for sy in (+1, -1):
            # Coil-1 side (z ≤ g): position unchanged in z
            all_c.append(np.column_stack([sx*x,  sy*y,  z    ]))
            all_B.append(np.column_stack([sx*Bx, sy*By, Bz   ]))
            # Coil-2 side (z ≥ g): z → 2g-z; Bx and By flip
            all_c.append(np.column_stack([sx*x,  sy*y,  z2-z ]))
            all_B.append(np.column_stack([-sx*Bx, -sy*By, Bz ]))

    return np.vstack(all_c), np.vstack(all_B)


# ── dolfinx-dependent functions (require FEniCSx) ────────────────────────────

def build_unit_current_direction(domain, cell_tags, coil_marker, a, b):
    """
    Returns (J_unit, coil_cells):
      J_unit    — dolfinx DG0 vector Function with unit tangent t_hat in
                  coil cells, 0 elsewhere. Multiply .x.array by J_mag to
                  get the actual current density without recomputing geometry.
      coil_cells — array of coil cell indices (from cell_tags.find).
    """
    from dolfinx import fem, mesh as dmesh

    L = b - a
    Vdg = fem.functionspace(domain, ("DG", 0, (3,)))
    J_unit = fem.Function(Vdg, name="J_unit")
    J_unit.x.array[:] = 0.0

    coil_cells = cell_tags.find(coil_marker)
    if len(coil_cells) == 0:
        raise RuntimeError(
            "No cells found with coil_marker — check physical group "
            "tagging in build_mesh.py")

    midpoints = dmesh.compute_midpoints(domain, domain.topology.dim, coil_cells)
    tx, ty = tangent_xy(midpoints[:, 0], midpoints[:, 1], L)

    # DG0 with block size 3: x.array is [c0x,c0y,c0z, c1x,c1y,c1z, ...]
    J_block = J_unit.x.array.reshape(-1, 3)
    J_block[coil_cells, 0] = tx
    J_block[coil_cells, 1] = ty
    J_block[coil_cells, 2] = 0.0

    return J_unit, coil_cells


def build_current_density(domain, cell_tags, coil_marker, J_mag, a, b):
    """DG0 current density J = J_mag * t_hat in coil cells, 0 elsewhere."""
    J, coil_cells = build_unit_current_direction(
        domain, cell_tags, coil_marker, a, b)
    J.x.array[:] *= J_mag
    J.name = "J"
    return J, coil_cells


if __name__ == "__main__":
    print("Testing current_source.py ...")
    x = np.array([0.0, 0.03, 0.08])
    y = np.array([0.05, 0.05, 0.0])
    L = params.b - params.a
    tx, ty = tangent_xy(x, y, L)
    print(f"tangents: {list(zip(tx.round(3), ty.round(3)))}")
    # Test expand
    c = np.array([[0.01, 0.01, 0.001]])
    B = np.array([[0.5, 0.3, 2.0]])
    ce, Be = expand_to_full_domain(c, B)
    print(f"Expanded: {len(ce)} cells, Bz all positive: {np.all(Be[:,2]>0)}")
    print("OK")
