"""
dcn.py — the distributed circuit network for a no-insulation racetrack coil.

TOPOLOGY
--------
An NI winding is the classic parallel ladder.  Turn m runs from node m-1 to
node m carrying the spiral current i_m; the turn-to-turn contact resistance
R_ct,m bridges the SAME two nodes, because travelling one turn azimuthally
brings the tape radially adjacent to itself, so the radial shortcut spans
exactly one turn.  KCL then gives, at every rung,

    i_m + j_m = I(t)                                            (exactly)

with j_m the radial (leakage) current.  Node 0 sees I injected and splits it
into i_1 and j_1; node 1 receives i_1 + j_1 = I and splits it again, and so
on.  This is the same statement the He et al. (2025) T-A paper writes as
I_z = I - I_r, but here it is a derived consequence of KCL rather than an
imposed closure.

Eliminating j_m leaves one ODE per unknown spiral current.  Lumping turns
into groups that share a current (all n_k turns of group k carry i_k):

    sum_j  n_j M[k,j] di_j/dt  +  V_sc,k(i_k)  =  R_ct,k * (I(t) - i_k)

i.e.   A di/dt = R_ct * (I(t) - i) - V_sc(i),      A[k,j] = n_j M[k,j]

M is the per-turn mutual inductance matrix (both coils, see inductance.py),
R_ct,k is the contact resistance seen by ONE turn of group k, and V_sc is the
power-law voltage across one turn.

WHY V_sc AND NOT R_sc
---------------------
The power law is written as a VOLTAGE

    V_sc,k = sign(i_k) * sum_s  E_c * ds_s * (|i_k| / Ic_s)^n_s

summed over sample points s around the turn, rather than as a resistance
E_c*l/Ic*(i/Ic)^(n-1).  The two are algebraically identical for i != 0, but
the voltage form is finite and smooth at i = 0, which matters because every
charge run starts there.

Ic_s and n_s are evaluated at each sample point's OWN local field and field
angle -- Ic varies strongly between the straight sections and the end caps,
and the resistive transition is set by the worst point, so a single
evaluation per turn understates the voltage.

Coil 2 is not carried as separate unknowns: it is the mirror image of coil 1,
series-connected, so by symmetry its turns carry identical currents.  Its
flux contribution is already inside M; its dissipation is added as a factor 2.
"""

import os
import sys

import numpy as np
from scipy.linalg import lu_factor, lu_solve

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cparams as cfg                              # noqa: E402
import fieldmatrix as fm                           # noqa: E402

E_C = 1.0e-4        # V/m, matches physics/ic_model.E_C


class DCN:
    """Assembled circuit for one (geometry, grouping, rho_c) combination."""

    def __init__(self, geom, groups, M, F, ic_model, n_model,
                 rho_ct_uohm_cm2, n_max=None):
        self.geom = geom
        self.groups = groups
        self.M = M
        self.F = F
        self.ic = ic_model
        self.nm = n_model
        self.rho_ct = rho_ct_uohm_cm2 * cfg.UOHM_CM2_TO_OHM_M2

        self.n = groups.n                                    # turns per group
        self.N = groups.N
        _, self.ds, self.n_hat = fm.sample_points(groups, geom)
        self.S = self.ds.shape[1]

        # A[k,j] = n_j * M[k,j]
        self.A = M * self.n[None, :]
        self._lu = lu_factor(self.A)

        # contact resistance seen by one turn of each group
        self.R_ct = groups.contact_resistance(self.rho_ct)

        # bore-field-per-amp row, for the observable
        self._bore_row = self._bore_field_row()

        # cap on the power-law exponent used in the exponentiation, purely to
        # keep overflow away in the far-overcritical regime (same spirit as
        # ta_solve's exp((n-1)*log(j)) reformulation)
        self.n_max = n_max

    # ── observables ─────────────────────────────────────────────────────────

    def _bore_field_row(self):
        """Bz at the bore centre per amp per turn, for each group."""
        from geometry import racetrack_loop
        pt = np.array([[0.0, 0.0, self.geom.coil_half_gap
                        if self.geom.two_coil else 0.0]])
        row = np.zeros(self.N)
        g2 = 2.0 * self.geom.coil_half_gap
        for j in range(self.N):
            p, d, _ = racetrack_loop(self.groups.r[j], self.groups.z[j],
                                     self.geom.L, cfg.N_STRAIGHT, cfg.N_CAP)
            B = fm._biot_savart(pt, p, d)
            if self.geom.two_coil:
                pm = p.copy()
                pm[:, 2] = g2 - p[:, 2]
                B = B + fm._biot_savart(pt, pm, d)
            row[j] = B[0, 2]
        return row

    def bore_Bz(self, i_turn):
        """Bore-centre Bz [T] for per-turn group currents i_turn."""
        return float(self._bore_row @ (self.n * np.atleast_1d(i_turn)))

    def spiral_current(self, i_turn):
        """Turn-weighted mean spiral current [A] (the paper's metric)."""
        return float(self.n @ i_turn / self.n.sum())

    # ── constitutive ────────────────────────────────────────────────────────

    def local_Ic_n(self, i_turn):
        """(Ic (N,S), n (N,S), |B| (N,S), theta (N,S)) at the sample points."""
        B = fm.field_at(self.F, i_turn, self.n)          # (N, S, 3)
        Bmag = np.linalg.norm(B, axis=-1)
        theta = fm.angle_with_normal_deg(B, self.n_hat)
        Ic, _ = self.ic.critical_current(Bmag, theta)
        nval, _ = self.nm.n_value(Bmag, theta)
        Ic = np.reshape(Ic, Bmag.shape)
        nval = np.reshape(nval, Bmag.shape)
        if self.n_max is not None:
            nval = np.minimum(nval, self.n_max)
        return Ic, nval, Bmag, theta

    def V_sc(self, i_turn, Ic=None, nval=None):
        """Power-law voltage across ONE turn of each group [V] (N,)."""
        if Ic is None or nval is None:
            Ic, nval, _, _ = self.local_Ic_n(i_turn)
        x = np.abs(i_turn)[:, None] / np.maximum(Ic, 1e-12)
        # exp(n*log x) instead of x**n -- overflow-safe for large n*x
        with np.errstate(divide="ignore", invalid="ignore"):
            logx = np.log(np.maximum(x, 1e-300))
            term = np.exp(np.clip(nval * logx, -700.0, 700.0))
        V = E_C * np.sum(self.ds * term, axis=1)
        return np.sign(i_turn) * V

    # ── the ODE ─────────────────────────────────────────────────────────────

    def rhs(self, t, i_turn, I_of_t):
        I = I_of_t(t)
        b = self.R_ct * (I - i_turn) - self.V_sc(i_turn)
        return lu_solve(self._lu, b)

    # ── power ───────────────────────────────────────────────────────────────

    def power(self, i_turn, I):
        """(P_contact, P_sc) [W] for the FULL two-coil system."""
        j_r = I - i_turn
        P_c = float(np.sum(self.n * j_r ** 2 * self.R_ct))
        P_s = float(np.sum(self.n * i_turn * self.V_sc(i_turn)))
        k = 2.0 if self.geom.two_coil else 1.0
        return k * P_c, k * P_s

    def terminal_voltage(self, i_turn, I):
        k = 2.0 if self.geom.two_coil else 1.0
        return k * float(np.sum(self.n * self.R_ct * (I - i_turn)))


# ── assembly helper ─────────────────────────────────────────────────────────

def build(geom, turns_per_group=None, rho_ct_uohm_cm2=None,
          ic_extrapolation=None, verbose=True):
    """Assemble a DCN for `geom`, building/loading M and F from cache."""
    import inductance as ind
    from geometry import TurnGroups
    from ic_model import NValueModel
    from ic_extrapolation import make_ic_model
    import params

    turns_per_group = turns_per_group or cfg.TURNS_PER_GROUP
    rho_ct = (rho_ct_uohm_cm2 if rho_ct_uohm_cm2 is not None
              else cfg.RHO_CT_UOHM_CM2)
    kind = ic_extrapolation or cfg.IC_EXTRAPOLATION

    groups = TurnGroups(geom, turns_per_group)
    if verbose:
        print(f"  {groups}", flush=True)
    M = ind.build_M_cached(groups, geom, verbose=verbose)
    F = fm.build_F_cached(groups, geom, verbose=verbose)

    ic = make_ic_model(kind)
    nm = NValueModel(params.n_value_csv_filename)
    dcn = DCN(geom, groups, M, F, ic, nm, rho_ct)
    if verbose:
        L = ind.total_inductance(M, groups, geom.two_coil)
        print(f"  L_total = {L*1e3:.2f} mH   "
              f"rho_c = {rho_ct:.0f} uOhm.cm^2   Ic model = {kind}",
              flush=True)
    return dcn
