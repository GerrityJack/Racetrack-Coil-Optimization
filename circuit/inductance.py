"""
inductance.py — per-turn mutual inductance matrix for the DCN.

Builds M[i, j] = mutual inductance between ONE turn of group i and ONE turn
of group j, via the Neumann double line integral

    M_ij = (mu0/4pi) * sum_a sum_b (dl_a . dl_b) / d_ab

over the discretised racetrack centrelines, with the self/near-neighbour
singularity regularised by the geometric mean distance (GMD) of the tape
cross-section:

    d_ab = sqrt(|r_a - r_b|^2 + gmd^2),   gmd = 0.2235 * (t + w)

That regularisation is not a numerical fudge -- it is the standard finite-
cross-section correction, and it carries real physics here: at a 75 um pitch
with a 4 mm wide tape the GMD (~0.91 mm) is an order of magnitude LARGER than
the turn-to-turn spacing, so adjacent turns come out almost as strongly
coupled as a turn is to itself, which is the physical truth for a tightly
stacked winding.

Coil 2 is included by MIRRORING each loop about the midplane (z -> 2g - z),
never by translating it -- matching physics/coil2_field's
compute_both_coils_field_multilayer() and the PMC boundary condition the FEM
uses.  (A translate-instead-of-mirror bug of exactly this kind was found in
the visualization code on 2026-07-27; it is a real trap for non-palindromic
layer stacks like [382,382,478,478,3,3].)

Sign/units check available in validation/lumped.py: with every group carrying
the same current, the total system inductance is

    L_total = 2 * sum_ij  n_i * n_j * M_eff[i, j]

(factor 2 = two coils each linking flux) and must reproduce the ~54 mH
obtained independently by flux linkage through the validated multi-filament
Biot-Savart path.
"""

import hashlib
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "physics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cparams as cfg        # noqa: E402
from geometry import racetrack_loop   # noqa: E402

MU0 = 4.0e-7 * np.pi


def gmd_length(geom):
    """Geometric mean distance of one tape's (t x w) cross-section [m]."""
    return cfg.GMD_FACTOR * (geom.t + geom.w)


def _loops(groups, geom, n_straight, n_cap):
    """Discretised loops for every group, coil 1 and its coil-2 mirror.

    Returns (pos, dl, pos_mirror) each a list of (S,3) arrays.  dl is shared:
    the mirror is a pure z-reflection of a planar loop, so the in-plane
    segment vectors are unchanged and the current sense is preserved (both
    coils are series-connected in the same rotational sense).
    """
    pos, dl, pos_m = [], [], []
    g2 = 2.0 * geom.coil_half_gap
    for k in range(groups.N):
        p, d, _ = racetrack_loop(groups.r[k], groups.z[k], geom.L,
                                 n_straight, n_cap)
        pos.append(p)
        dl.append(d)
        if geom.two_coil:
            pm = p.copy()
            pm[:, 2] = g2 - p[:, 2]
            pos_m.append(pm)
        else:
            pos_m.append(None)
    return pos, dl, pos_m


def _neumann_pair(p_a, d_a, p_b, d_b, gmd2):
    """(mu0/4pi) * sum_a sum_b (dl_a.dl_b)/sqrt(|r_a-r_b|^2 + gmd^2)."""
    # (Sa, Sb) squared distances
    diff = p_a[:, None, :] - p_b[None, :, :]
    d2 = np.einsum("abk,abk->ab", diff, diff) + gmd2
    dot = d_a @ d_b.T
    return (MU0 / (4.0 * np.pi)) * float(np.sum(dot / np.sqrt(d2)))


def build_M(groups, geom, n_straight=None, n_cap=None, verbose=True):
    """Per-turn mutual inductance matrix including the coil-2 mirror.

    Returns
    -------
    M_eff : (N, N) symmetric [H].  M_eff[i, j] is the flux linked by one turn
            of group i per amp in one turn of group j, summed over both coils.
    """
    n_straight = n_straight if n_straight is not None else cfg.N_STRAIGHT
    n_cap = n_cap if n_cap is not None else cfg.N_CAP

    gmd = gmd_length(geom)
    gmd2 = gmd * gmd
    pos, dl, pos_m = _loops(groups, geom, n_straight, n_cap)

    N = groups.N
    M = np.zeros((N, N))
    t0 = time.time()
    for i in range(N):
        for j in range(i, N):
            v = _neumann_pair(pos[i], dl[i], pos[j], dl[j], gmd2)
            if geom.two_coil:
                # mutual with coil 2's image of group j.  Symmetric in (i, j):
                # reflection is an isometry, so M(i, mirror(j)) = M(j, mirror(i)).
                v += _neumann_pair(pos[i], dl[i], pos_m[j], dl[j], gmd2)
            M[i, j] = M[j, i] = v
        if verbose and (i % 20 == 0 or i == N - 1):
            print(f"  [M] row {i+1}/{N}  ({time.time()-t0:.1f} s)", flush=True)

    if verbose:
        print(f"  [M] GMD = {gmd*1e3:.3f} mm, "
              f"{n_straight*2 + n_cap*2} segments/loop, "
              f"built in {time.time()-t0:.1f} s", flush=True)
    return M


def total_inductance(M, groups, two_coil=True):
    """System self-inductance with every turn in series [H].

    L = (2 if two_coil else 1) * sum_ij n_i n_j M_eff[i,j]
    """
    n = groups.n
    L = float(n @ M @ n)
    return (2.0 if two_coil else 1.0) * L


# ── caching ─────────────────────────────────────────────────────────────────

def _key(groups, geom, n_straight, n_cap):
    sig = (f"{geom.a_out:.9e}|{geom.t:.9e}|{geom.w:.9e}|{geom.L:.9e}|"
           f"{geom.coil_half_gap:.9e}|{int(geom.two_coil)}|"
           f"{list(geom.n_turns)}|{list(np.round(geom.layer_z, 12))}|"
           f"{groups.turns_per_group}|{n_straight}|{n_cap}|"
           f"{cfg.GMD_FACTOR}")
    return hashlib.sha1(sig.encode()).hexdigest()[:16]


def build_M_cached(groups, geom, n_straight=None, n_cap=None, verbose=True):
    """build_M() with an on-disk cache (the matrix is geometry-only, so it is
    reused across every rho_c / ramp-rate / current-schedule variation)."""
    n_straight = n_straight if n_straight is not None else cfg.N_STRAIGHT
    n_cap = n_cap if n_cap is not None else cfg.N_CAP
    path = os.path.join(cfg.CACHE_DIR,
                        f"M_{_key(groups, geom, n_straight, n_cap)}.npz")
    if os.path.exists(path):
        if verbose:
            print(f"  [M] cache hit: {os.path.basename(path)}", flush=True)
        return np.load(path)["M"]
    M = build_M(groups, geom, n_straight, n_cap, verbose=verbose)
    np.savez_compressed(path, M=M)
    if verbose:
        print(f"  [M] cached -> {os.path.basename(path)}", flush=True)
    return M
