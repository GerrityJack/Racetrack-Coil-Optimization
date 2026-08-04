"""
geometry.py — turn grouping and racetrack loop discretisation for the DCN.

Pure numpy.  Imports params only to read the default coil; every function
takes an explicit CoilGeometry so the He et al. (2025) benchmark coil can be
built without touching params.py.

Conventions inherited from the rest of the repo (do not change these
independently -- physics/coil2_field.py and params.tape_length_m both rely on
them):

  * ``L`` is the racetrack HALF straight-leg length (= params.b - params.a).
    A straight section runs from x = -L to x = +L, and there are two of them,
    so one turn of radius r has length ``4*L + 2*pi*r``.  This is exactly the
    formula embedded in params.recompute_derived()'s tape_length_m.
  * Turn radii eat INWARD from a shared outer edge ``a_out``: turn k of a
    layer with n turns sits at r = a_out - (k + 0.5)*t.  Mirrors
    optimize_geometry.filament_stack() and coil2_field's inlined copy.
  * The tape broad-face normal is the in-plane outward normal
    ``n_hat = (-t_y, t_x, 0)``, matching current_source.normal_xy() and
    ta_solve.build_n_hat_ufl().

Note on naming: ``L`` in this codebase ALWAYS means the straight-leg length,
never inductance.  Circuit inductances are ``M`` / ``L_ind``.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "physics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params  # noqa: E402


class CoilGeometry:
    """A racetrack winding pack, independent of params.py's global state.

    Parameters
    ----------
    a_out         : outer radial edge shared by every layer [m]
    n_turns       : turns per layer (list of int)
    t             : radial pitch = single tape thickness [m]
    w             : tape width = axial extent of one layer [m]
    L             : HALF straight-leg length [m]
    coil_half_gap : half the centre-to-centre distance to coil 2 [m].
                    Ignored when two_coil is False.
    two_coil      : whether a mirrored second coil is present.
    layer_z       : optional explicit per-layer z-centres [m].  Defaults to
                    the params.py stacking (layer 0 nearest the midplane).
    """

    def __init__(self, a_out, n_turns, t, w, L, coil_half_gap=0.0,
                 two_coil=True, layer_z=None, label="coil"):
        self.a_out = float(a_out)
        self.n_turns = [int(n) for n in n_turns]
        self.t = float(t)
        self.w = float(w)
        self.L = float(L)
        self.coil_half_gap = float(coil_half_gap)
        self.two_coil = bool(two_coil)
        self.label = label

        self.n_layers = len(self.n_turns)
        if layer_z is not None:
            self.layer_z = [float(z) for z in layer_z]
        else:
            # params.py stacking: coil 1 centred on z = 0, layer 0 on top
            z_top = self.n_layers * self.w / 2.0
            self.layer_z = [z_top - (i + 0.5) * self.w
                            for i in range(self.n_layers)]

        self.n_turns_total = sum(self.n_turns)
        assert min(self.n_turns) >= 1, "every layer needs >= 1 turn"
        assert self.a_out - max(self.n_turns) * self.t > 0, \
            "innermost turn radius is non-positive"

    # ── constructors ────────────────────────────────────────────────────────

    @classmethod
    def from_params(cls, label="champion"):
        """The current design, read from params.py at call time."""
        return cls(a_out=params.a_out, n_turns=params.n_turns,
                   t=params.t, w=params.w, L=params.L,
                   coil_half_gap=params.coil_half_gap,
                   two_coil=bool(getattr(params, "two_coil_mode", True)),
                   layer_z=[0.5 * (zt + zb) for zt, zb in
                            zip(params.layer_z_tops, params.layer_z_bottoms)],
                   label=label)

    def __repr__(self):
        return (f"<CoilGeometry {self.label}: {self.n_layers} layers, "
                f"{self.n_turns_total} turns, a_out={self.a_out*1e3:.2f} mm, "
                f"L={self.L*1e3:.2f} mm, two_coil={self.two_coil}>")

    # ── derived scalars ─────────────────────────────────────────────────────

    def turn_radius(self, layer, k):
        """Cap radius of turn k (0-based, outermost first) of a layer."""
        return self.a_out - (k + 0.5) * self.t

    def turn_length(self, r):
        """Conductor length of one turn at cap radius r [m]."""
        return 4.0 * self.L + 2.0 * np.pi * np.asarray(r, dtype=np.float64)

    def tape_length_m(self):
        """Total conductor length -- must match params.tape_length_m for the
        params-derived geometry (used as a self-check)."""
        tot = 0.0
        for i, n in enumerate(self.n_turns):
            for k in range(n):
                tot += self.turn_length(self.turn_radius(i, k))
        return float(tot)


# ── turn grouping ───────────────────────────────────────────────────────────

class TurnGroups:
    """Radial lumping of turns into groups that share one current.

    Attributes (all length N_group unless noted)
    -------------------------------------------
    r        : group cap radius [m]
    z        : group z-centre [m]
    n        : turns in the group (float -- groups need not divide evenly)
    layer    : layer index
    l_turn   : conductor length of ONE turn in the group [m]
    r_inner_edge / r_outer_edge : the group's radial extent [m]
    """

    def __init__(self, geom, turns_per_group):
        self.geom = geom
        self.turns_per_group = int(turns_per_group)

        r, z, n, layer, lo, hi = [], [], [], [], [], []
        for i, n_i in enumerate(geom.n_turns):
            n_sub = max(1, int(round(n_i / self.turns_per_group)))
            # radial edges of this layer's pack, outermost edge shared
            edges = np.linspace(geom.a_out - n_i * geom.t, geom.a_out,
                                n_sub + 1)
            for j in range(n_sub):
                r.append(0.5 * (edges[j] + edges[j + 1]))
                z.append(geom.layer_z[i])
                n.append(n_i / n_sub)
                layer.append(i)
                lo.append(edges[j])
                hi.append(edges[j + 1])

        self.r = np.array(r)
        self.z = np.array(z)
        self.n = np.array(n)
        self.layer = np.array(layer, dtype=int)
        self.r_inner_edge = np.array(lo)
        self.r_outer_edge = np.array(hi)
        self.N = len(self.r)
        self.l_turn = geom.turn_length(self.r)

    def __repr__(self):
        return (f"<TurnGroups N={self.N} "
                f"(~{self.turns_per_group} turns/group), "
                f"{self.n.sum():.0f} turns total>")

    def contact_resistance(self, rho_ct_ohm_m2):
        """Turn-to-turn contact resistance seen by ONE turn of each group.

        In the NI ladder each turn's radial shortcut spans one turn pitch, so
        the resistance across it is rho_c / (contact area of one turn), with
        contact area = (turn length) x (tape width).
        """
        return rho_ct_ohm_m2 / (self.l_turn * self.geom.w)

    def group_contact_resistance(self, rho_ct_ohm_m2):
        """Series contact resistance across a whole group (n turns)."""
        return self.n * self.contact_resistance(rho_ct_ohm_m2)


# ── racetrack loop discretisation ───────────────────────────────────────────

def racetrack_loop(r, z, L, n_straight, n_cap):
    """Closed racetrack centreline at cap radius r, in the plane z.

    Returns
    -------
    seg_mid : (M, 3) segment midpoints
    dl      : (M, 3) directed segment vectors (current-flow direction)
    n_hat   : (M, 3) in-plane outward tape-face normal at each midpoint

    Same construction and traversal order as
    physics/coil2_field.coil2_centreline(), generalised to arbitrary (r, z).
    """
    pts = []
    # top straight, -x -> +x at y = +r
    pts += [(-L + 2 * L * i / n_straight, r, z) for i in range(n_straight)]
    # right cap, centred at x = +L
    th_r = np.linspace(np.pi / 2, -np.pi / 2, n_cap, endpoint=False)
    pts += [(L + r * np.cos(th), r * np.sin(th), z) for th in th_r]
    # bottom straight, +x -> -x at y = -r
    pts += [(L - 2 * L * i / n_straight, -r, z) for i in range(n_straight)]
    # left cap, centred at x = -L
    th_l = np.linspace(-np.pi / 2, -3 * np.pi / 2, n_cap, endpoint=False)
    pts += [(-L + r * np.cos(th), r * np.sin(th), z) for th in th_l]

    pts = np.asarray(pts, dtype=np.float64)
    seg_mid = 0.5 * (pts + np.roll(pts, -1, axis=0))
    dl = np.roll(pts, -1, axis=0) - pts

    # in-plane outward normal: n_hat = (-t_y, t_x, 0), matching
    # current_source.normal_xy() and ta_solve.build_n_hat_ufl()
    tang = dl / np.maximum(np.linalg.norm(dl, axis=1, keepdims=True), 1e-30)
    n_hat = np.zeros_like(tang)
    n_hat[:, 0] = -tang[:, 1]
    n_hat[:, 1] = tang[:, 0]
    return seg_mid, dl, n_hat


def turn_sample_points(r, z, L, n_samples):
    """``n_samples`` points spread around one turn, with the arc length each
    represents and the tape normal there.

    Used to resolve Ic(B, theta) variation around a turn -- the peak-field
    point dominates the resistive transition, so a single-point evaluation is
    not good enough.

    Returns (pts (S,3), ds (S,), n_hat (S,3)).
    """
    # Reuse the loop discretisation, then decimate to n_samples buckets so the
    # sample points and their arc weights stay exactly consistent with the
    # geometry used for the field/inductance integrals.
    seg_mid, dl, n_hat = racetrack_loop(r, z, L,
                                        n_straight=max(4, n_samples),
                                        n_cap=max(6, n_samples))
    ds = np.linalg.norm(dl, axis=1)
    M = len(seg_mid)
    edges = np.linspace(0, M, n_samples + 1).astype(int)
    pts_o, ds_o, nh_o = [], [], []
    for j in range(n_samples):
        sl = slice(edges[j], max(edges[j] + 1, edges[j + 1]))
        wgt = ds[sl]
        pts_o.append(np.average(seg_mid[sl], axis=0, weights=wgt))
        nh_o.append(np.average(n_hat[sl], axis=0, weights=wgt))
        ds_o.append(wgt.sum())
    nh_o = np.asarray(nh_o)
    nh_o /= np.maximum(np.linalg.norm(nh_o, axis=1, keepdims=True), 1e-30)
    return np.asarray(pts_o), np.asarray(ds_o), nh_o
