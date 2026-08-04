"""
fieldmatrix.py — per-unit-current field matrix for the DCN.

F[i, s, j, :] = magnetic field at sample point s of turn-group i, per amp
flowing in ONE turn of group j, summed over both coils.

This exists because the resistive part of the circuit needs the LOCAL field
at each group to evaluate Ic(B, theta) and n(B, theta), and the field is
exactly linear in the currents (no iron anywhere in this design).  That
linearity is the same assumption optimize_geometry.quench_current() relies
on, and it is what lets the matrix be built once and merely rescaled at every
ODE step instead of recomputed.

Sampling several points around each turn is deliberate: Ic varies strongly
between the straight sections and the end caps, and the resistive transition
is driven by the worst point, not the average.  A single-point evaluation
per turn understates the resistance.
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

import cparams as cfg                                  # noqa: E402
from geometry import racetrack_loop, turn_sample_points  # noqa: E402

MU0 = 4.0e-7 * np.pi


def _biot_savart(points, seg_mid, dl, chunk=256, reg=0.0):
    """B [T] at `points` from a closed filament carrying 1 A.

    Vectorised over field points, unlike coil2_field.compute_coil2_field()
    which loops one point at a time -- that per-point loop is what makes the
    O(N^2) matrix build here infeasible through the existing helper.

    ``reg`` softens the kernel to |r| -> sqrt(|r|^2 + reg^2).  This is NOT
    cosmetic: the field-matrix sample points lie ON their own group's
    centreline, so with reg=0 a group sees its own divergent self-field and
    the local |B| used for Ic(B, theta) is meaningless (it came out at 24.9 T
    against a true peak of ~10.7 T).  Setting reg to the conductor's
    geometric mean distance makes the self-term the field averaged over the
    conductor cross-section, which is what a homogenised FEM cell reports and
    what Ic should be evaluated at.  Use reg=0 for points far from any
    conductor (e.g. the bore centre), where it makes no difference anyway.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    out = np.zeros_like(points)
    pre = MU0 / (4.0 * np.pi)
    reg2 = float(reg) ** 2
    for i0 in range(0, len(points), chunk):
        p = points[i0:i0 + chunk]
        r = p[:, None, :] - seg_mid[None, :, :]              # (P, M, 3)
        r2 = np.einsum("pmk,pmk->pm", r, r) + reg2
        den = r2 * np.sqrt(r2)
        cross = np.cross(np.broadcast_to(dl, r.shape), r)    # (P, M, 3)
        out[i0:i0 + chunk] = pre * np.einsum("pmk->pk", cross / den[:, :, None])
    return out


def sample_points(groups, geom, n_samples=None):
    """Evaluation points, arc weights and tape normals for every group.

    Returns (pts (N,S,3), ds (N,S), n_hat (N,S,3)).
    """
    n_samples = n_samples if n_samples is not None else cfg.N_TURN_SAMPLES
    P, W, NH = [], [], []
    for k in range(groups.N):
        p, ds, nh = turn_sample_points(groups.r[k], groups.z[k], geom.L,
                                       n_samples)
        P.append(p)
        W.append(ds)
        NH.append(nh)
    return np.asarray(P), np.asarray(W), np.asarray(NH)


def build_F(groups, geom, n_samples=None, n_straight=None, n_cap=None,
            verbose=True):
    """Field matrix F (N, S, N, 3) [T per amp per turn].

    The kernel is GMD-regularised (see _biot_savart's ``reg``): sample points
    lie on their own group's centreline, so the unregularised self-term
    diverges and poisons every Ic(B, theta) lookup.
    """
    n_samples = n_samples if n_samples is not None else cfg.N_TURN_SAMPLES
    n_straight = n_straight if n_straight is not None else cfg.N_STRAIGHT
    n_cap = n_cap if n_cap is not None else cfg.N_CAP

    from inductance import gmd_length
    reg = gmd_length(geom)

    pts, _, _ = sample_points(groups, geom, n_samples)
    flat = pts.reshape(-1, 3)
    N = groups.N
    F = np.zeros((N * n_samples, N, 3))
    g2 = 2.0 * geom.coil_half_gap

    t0 = time.time()
    for j in range(N):
        p, d, _ = racetrack_loop(groups.r[j], groups.z[j], geom.L,
                                 n_straight, n_cap)
        B = _biot_savart(flat, p, d, reg=reg)
        if geom.two_coil:
            pm = p.copy()
            pm[:, 2] = g2 - p[:, 2]
            B = B + _biot_savart(flat, pm, d, reg=reg)
        F[:, j, :] = B
        if verbose and (j % 20 == 0 or j == N - 1):
            print(f"  [F] source {j+1}/{N}  ({time.time()-t0:.1f} s)",
                  flush=True)
    if verbose:
        print(f"  [F] GMD regularisation = {reg*1e3:.3f} mm", flush=True)
    return F.reshape(N, n_samples, N, 3)


def field_at(F, i_turn, n_turns_per_group):
    """B (N, S, 3) [T] for per-turn currents `i_turn` (N,).

    Each group carries n_j turns at i_j amps, so the source strength is
    n_j * i_j.
    """
    return np.einsum("isjk,j->isk", F, n_turns_per_group * i_turn)


def angle_with_normal_deg(B, n_hat):
    """Angle between B and the tape broad-face normal, in degrees [0, 180].

    Matches physics/ic_model.angle_with_normal_deg()'s convention:
    0 deg = B perpendicular to the tape face.
    """
    bn = np.einsum("...k,...k->...", B, n_hat)
    bmag = np.linalg.norm(B, axis=-1)
    c = np.clip(bn / np.maximum(bmag, 1e-30), -1.0, 1.0)
    return np.degrees(np.arccos(c))


# ── caching ─────────────────────────────────────────────────────────────────

def _key(groups, geom, n_samples, n_straight, n_cap):
    sig = (f"{geom.a_out:.9e}|{geom.t:.9e}|{geom.w:.9e}|{geom.L:.9e}|"
           f"{geom.coil_half_gap:.9e}|{int(geom.two_coil)}|"
           f"{list(geom.n_turns)}|{list(np.round(geom.layer_z, 12))}|"
           f"{groups.turns_per_group}|{n_samples}|{n_straight}|{n_cap}|"
           f"gmd{cfg.GMD_FACTOR}")
    return hashlib.sha1(sig.encode()).hexdigest()[:16]


def build_F_cached(groups, geom, n_samples=None, n_straight=None, n_cap=None,
                   verbose=True):
    n_samples = n_samples if n_samples is not None else cfg.N_TURN_SAMPLES
    n_straight = n_straight if n_straight is not None else cfg.N_STRAIGHT
    n_cap = n_cap if n_cap is not None else cfg.N_CAP
    path = os.path.join(
        cfg.CACHE_DIR,
        f"F_{_key(groups, geom, n_samples, n_straight, n_cap)}.npz")
    if os.path.exists(path):
        if verbose:
            print(f"  [F] cache hit: {os.path.basename(path)}", flush=True)
        return np.load(path)["F"]
    F = build_F(groups, geom, n_samples, n_straight, n_cap, verbose=verbose)
    np.savez_compressed(path, F=F)
    if verbose:
        print(f"  [F] cached -> {os.path.basename(path)}", flush=True)
    return F
