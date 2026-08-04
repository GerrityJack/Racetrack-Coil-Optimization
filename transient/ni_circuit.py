"""
ni_circuit.py — no-insulation radial-current closure for the T-A solve.

Implements the coupling of He et al., Energies 2025, 18, 3669 (Eqs. 11-13),
stated in the unambiguous ladder form that Phase A validated.

THE RELATION
------------
Per radial bin k, comparing the resistive field along the tape against the
induced field:

    E_i,k = -(A - A_prev).t_hat / dt          (Eq. 11, induced)
    E_p,k =  rho_fn * (J_sc . t_hat) * F      (Eq. 12, resistive)
    I_r,k = (E_p,k - E_i,k) * l_turn,k / R_ct,k
    I_z,k = I(t) - I_r,k                      -> the T Dirichlet data

with R_ct,k = rho_c / (l_turn,k * w), i.e. the contact area of one turn.

WHY THIS FORM AND NOT THE PAPER'S LITERAL ONE
---------------------------------------------
The paper's Eq. (13) is J_r = (E_p - E_i)*l_turn/(Lambda*rho_c) and then says
"integrating J_r over the cross-sectional area" gives I_r -- but it does not
define that area, and the obvious candidates differ by a factor of l_turn.
The ladder form above is derived instead from KCL on the NI ladder, where
the contact bridges the same two nodes as the turn it parallels, so
I_r = V_turn / R_ct with V_turn = E * l_turn.  It is dimensionally
unambiguous and it is exactly the relation Phase A's DCN solves.

So the T-A model here differs from the Phase A DCN in ONE respect only:
E_p and E_i come from the field solve (and therefore carry the screening-
current distribution across the tape width) instead of from a lumped
inductance matrix and a power law.

THE F FACTOR
------------
ta_solve's rho_fn is rho_sc*(delta_SC/Lambda) and _J_from_T returns SC-layer
J, so the true field along the tape is rho_fn*(Lambda/delta_SC)*J_sc.  See
tparams.EP_FACTOR_MODE.

═══════════════════════════════════════════════════════════════════════════
2026-08-04: THE CLOSURE WAS RESTRUCTURED -- E_i IS NOW SOLVED IMPLICITLY
═══════════════════════════════════════════════════════════════════════════

The first implementation computed E_i explicitly each Picard iteration from
a finite difference dI/dt = (I_z_trial - I_z_prev_step)/dt, where I_z_trial
was the PREVIOUS Picard iteration's (not-yet-converged) estimate of I_z, and
combined it with E_p via elementwise per-bin arithmetic. That diverged: a
regression probe (validation/ni_closure_stability_check.py) showed a
persistent, non-decaying oscillation (20-30 of 48 bins in the clip band,
every iteration, no trend toward convergence over 250 iterations) that did
NOT respond to 5x stronger relaxation, reordering the update, or 20x weaker
contact coupling.

Root cause: the per-turn mutual inductance matrix (induction.BinInductance.A)
is DENSE and far from diagonally dominant -- the GMD regularisation
(~0.91 mm) is an order of magnitude larger than the 75 um turn pitch, so
essentially every turn in a layer is coupled to every other turn about as
strongly as it is "coupled to itself". Treating that matrix with an
ELEMENTWISE (Jacobi-style) fixed-point update -- which is what computing E_i
from a stale per-bin trial current amounts to -- is exactly the case
classical iterative-methods theory says will not converge: Jacobi iteration
needs diagonal dominance, and this matrix has the opposite property. No
amount of scalar relaxation fixes a matrix-level conditioning problem.

FIX: since E_i is LINEAR in I_z (E_i = -(A_ind @ (I_z - I_z_prev)/dt)/l),
solve for I_z EXACTLY against that linear relationship, treating only the
nonlinear part (E_p, which comes from the T-solve and can only be updated by
Picard lagging) as frozen for this iteration. Substituting the definition of
E_i into I_z = I - (E_p - E_i)*l/R gives a linear system in I_z:

    (I_48 + dt^-1 * D @ A_ind) @ I_z =
        I*1 - D*(l*E_p) + dt^-1 * D @ A_ind @ I_z_prev_step

    D = diag(1/R_ct)   (48x48 diagonal)
    A_ind[k,j] = n_j * M[k,j]   (induction.BinInductance.A)

This is a 48x48 dense linear solve, factorised once per distinct dt (cheap:
microseconds) and reused every Picard iteration of a step. It removes BOTH
the differentiate-a-noisy-iterate problem AND the elementwise-Jacobi problem
at once, since the whole inductively-coupled system is solved together
rather than one bin at a time. Only E_p remains explicit/Picard-lagged --
which is unavoidable, since it depends on J from the T-solve, and the
T-solve needs I_z as its boundary data.
"""

import os
import sys

import numpy as np
from scipy.linalg import lu_factor, lu_solve

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params        # noqa: E402
import tparams as tp  # noqa: E402


class NICircuit:
    """Radial-current closure bound to one ta setup."""

    def __init__(self, ta, bins, cell_groups, rho_ct_uohm_cm2,
                 ep_factor_mode=None, induction=None):
        self.ta = ta
        self.bins = bins
        self.groups = cell_groups          # list of row-index arrays
        self.rho_ct = rho_ct_uohm_cm2 * tp.UOHM_CM2_TO_OHM_M2
        self.R_ct = bins.contact_resistance(self.rho_ct)      # (n_flat,)
        self.l_turn = bins.flat(bins.l_turn)                  # (n_flat,)
        self.Dvec = 1.0 / self.R_ct                           # (n_flat,)
        self._n = bins.n_flat

        mode = ep_factor_mode or tp.EP_FACTOR_MODE
        if mode == "derived":
            self.ep_factor = params.t / params.delta_SC       # Lambda/delta
        elif mode == "paper":
            self.ep_factor = 1.0
        else:
            raise ValueError(f"unknown EP_FACTOR_MODE {mode!r}")
        self.ep_mode = mode

        # E_i is solved IMPLICITLY against this mutual-inductance matrix --
        # NOT taken from A.t_hat (this A-form's gauge component makes
        # pointwise A meaningless; see induction.py) and NOT from an
        # elementwise finite difference (see the module docstring above for
        # why that diverged).
        from induction import BinInductance
        self.induction = induction or BinInductance(bins)
        self.I_z_prev_step = None   # I_z at the end of the previous step

        self.I_z = None            # per-bin transport current [A]
        self.I_r = None
        self.last_E_p = None
        self.n_clipped = 0
        self._lu_cache = {}         # dt -> (lu, piv)

    # ── field averages per bin ──────────────────────────────────────────

    def _bin_mean(self, per_cell):
        """VOLUME-weighted bin average of a per-cell quantity.

        Not a refinement -- an arithmetic cell mean is wrong here, since the
        wanted quantity is the average of rho*J over the tape cross-section
        (a volume integral) and the mesh cells are not equal-sized.
        """
        out = np.zeros(self.bins.n_flat)
        vol = self.ta["coil_vols"]
        for k, idx in enumerate(self.groups):
            if idx.size:
                w = vol[idx]
                out[k] = float(np.sum(per_cell[idx] * w) / np.sum(w))
        return out

    def E_p_bins(self, J_coil):
        """Volume-weighted per-bin E_p [V/m] from the current T-A state."""
        ta = self.ta
        coil = ta["coil_cells"]
        t_hat = ta["t_hat_coil"]
        rho = ta["rho_fn"].x.array[coil]
        J_t = np.einsum("ij,ij->i", J_coil, t_hat)
        return self._bin_mean(rho * J_t * self.ep_factor)

    # ── the implicit linear solve for I_z ────────────────────────────────

    def _lu(self, dt):
        key = round(float(dt), 9)
        cached = self._lu_cache.get(key)
        if cached is not None:
            return cached
        A_ind = self.induction.A
        M_sys = np.eye(self._n) + (1.0 / dt) * (self.Dvec[:, None] * A_ind)
        cached = lu_factor(M_sys)
        self._lu_cache[key] = cached
        return cached

    def _solve_I_z(self, I_transport, dt, E_p):
        """Exact solve of the linear (inductive) part of the closure,
        given E_p frozen at its current (Picard-lagged) value."""
        prev = (self.I_z_prev_step if self.I_z_prev_step is not None
                else np.zeros(self._n))
        A_ind = self.induction.A
        rhs = (I_transport * np.ones(self._n)
               - self.Dvec * (self.l_turn * E_p)
               + (1.0 / dt) * (self.Dvec[:, None] * A_ind) @ prev)
        return lu_solve(self._lu(dt), rhs)

    # ── the closure ─────────────────────────────────────────────────────

    def update(self, I_transport, dt, J_coil, relax=None):
        """Recompute I_z per bin and write it into the T Dirichlet data."""
        relax = tp.NI_RELAX if relax is None else relax
        E_p = self.E_p_bins(J_coil)
        I_z_new = self._solve_I_z(I_transport, dt, E_p)

        # Physical band on the imposed current.  |I_r| > |I| would mean the
        # contacts carry more than the whole transport current, which the
        # ladder forbids. Excursions past it are clipped AND counted -- a
        # silently clipped solution would look converged while being wrong.
        lim_lo = -abs(I_transport) * tp.I_Z_BAND
        lim_hi = abs(I_transport) * (1.0 + tp.I_Z_BAND)
        self.n_clipped = int(np.sum((I_z_new < lim_lo) | (I_z_new > lim_hi)))
        I_z_new = np.clip(I_z_new, lim_lo, lim_hi)

        if self.I_z is None:
            self.I_z = I_z_new.copy()
        else:
            self.I_z = (1.0 - relax) * self.I_z + relax * I_z_new
        self.I_r = I_transport - self.I_z
        self.last_E_p = E_p
        self.write_bcs()
        return self.I_z

    def freeze(self, I_transport):
        """Hold every bin at the full transport current (insulated state).

        Used to warm up a time step: the T/A/rho fields are let settle at
        fixed BCs for a few Picard iterations before the circuit closure is
        switched on.
        """
        self.I_z = np.full(self.bins.n_flat, float(I_transport))
        self.I_r = np.zeros(self.bins.n_flat)
        self.n_clipped = 0
        self.write_bcs()
        return self.I_z

    def end_step(self):
        """Latch I_z as the previous step's value (call after each step)."""
        self.I_z_prev_step = (None if self.I_z is None else self.I_z.copy())

    def write_bcs(self):
        """Push per-bin I_z into the per-layer tape-edge BC Functions."""
        ta = self.ta
        delta = ta["delta_SC"]
        for i, (top_fn, bot_fn) in enumerate(ta["layer_bc_fns"]):
            for dofs, fn, sign in ((ta["layer_top_dofs"][i], top_fn, -1.0),
                                   (ta["layer_bot_dofs"][i], bot_fn, +1.0)):
                if dofs.size == 0:
                    continue
                flat = self._dof_bins(i, dofs)
                fn.x.array[dofs] = sign * self.I_z[flat] / (2.0 * delta)
                fn.x.scatter_forward()

    def _dof_bins(self, layer, dofs):
        key = ("_dofbins", layer, id(dofs))
        cache = getattr(self, "_dofbin_cache", None)
        if cache is None:
            cache = self._dofbin_cache = {}
        if key not in cache:
            from turn_map import build_dof_map
            cache[key] = build_dof_map(self.bins, self.ta["T_dof_coords"],
                                       dofs)
        return cache[key]

    # ── diagnostics ─────────────────────────────────────────────────────

    def diagnostics(self, I_transport):
        """Monitored quantities -- assumptions, not assertions."""
        w = self.bins.flat(self.bins.turns)
        tot = w.sum()
        return dict(
            I_z_mean=float((self.I_z * w).sum() / tot),
            I_z_min=float(self.I_z.min()), I_z_max=float(self.I_z.max()),
            I_r_mean=float((self.I_r * w).sum() / tot),
            I_r_frac_max=float(np.max(np.abs(self.I_r))
                               / max(abs(I_transport), 1e-30)),
            n_out_of_range=int(np.sum((self.I_z < -1e-9)
                                      | (self.I_z > abs(I_transport) * 1.5))),
            n_clipped=int(getattr(self, "n_clipped", 0)),
            E_p_mean=(float(np.mean(self.last_E_p))
                     if self.last_E_p is not None else 0.0),
            E_p_min=(float(np.min(self.last_E_p))
                    if self.last_E_p is not None else 0.0),
            E_p_max=(float(np.max(self.last_E_p))
                    if self.last_E_p is not None else 0.0),
        )
