"""
induction.py — gauge-safe induced field E_i for the NI closure.

WHY THIS FILE EXISTS
--------------------
He et al. Eq. (11) writes the induced field as E_i = -dA_z/dt, evaluated
pointwise.  That CANNOT be done in this repo's A-formulation.

solve.py's A-form is

    (1/mu0)(curl A, curl v) + gauge_regularization * (A, v) = (J, v)

with params.gauge_regularization = 1e-3 against 1/mu0 ~ 8e5.  The curl-curl
operator has the gradients as its null space, and that null space is
penalised only by the 1e-3 term, so any divergence in the source is amplified
by ~1/1e-3 into a gradient component of A.  The eighth-symmetry domain's
current enters and leaves through the cut faces, so the source is NOT
divergence-free, and the result is severe:

    measured |A| over the coil cells : ~6.5e9      (mean)
    physical Phi_turn / l_turn       :  1.1e-1 Wb/m
    ratio                            : ~1e10

A is meaningful in this repo ONLY through curl A.  (The same gauge component
is why integral(J.A) gives an inductance ~1e8 too large -- see
circuit/validation/lumped.py.)  Every existing user of A takes its curl, so
this has never mattered before; the NI closure is the first place A itself is
needed.

Using the raw A.t_hat anyway produced radial currents of 39 A against a
transport current of 65 A (the circuit model says ~2-3 A) and clipped 38 of
48 bins.

WHAT IS DONE INSTEAD
--------------------
E_i is a purely inductive, geometric quantity -- it does not depend on the
current distribution across the tape width, only on the flux linked by each
turn.  So it is taken from a mutual-inductance matrix built with the SAME
validated Neumann + GMD machinery as Phase A (circuit/inductance.py), for
this model's radial bins:

    E_i,k = -( sum_j n_j M[k,j] * dI_z,j/dt ) / l_turn,k

The T-A field solve is still doing the induction correctly INTERNALLY -- its
T right-hand side is -(1/dt) curl(A - A_prev).n_hat, which is gauge-invariant
because it takes the curl.  What is replaced is only the extraction of a
per-bin scalar E_i for the circuit closure.

CONSEQUENCE FOR VALIDATION, STATED PLAINLY: the induced term is now the same
in both models, so validation/dcn_crosscheck.py no longer independently
checks the induction.  What it isolates is the effect of E_p -- i.e. whether
the screening-current distribution the T-A resolves changes the radial
redistribution relative to the lumped power law.  That is still worth
measuring, but it is a narrower claim than "two independent models agree".
"""

import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_CIRC = os.path.join(_ROOT, "circuit")
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "physics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params      # noqa: E402


def _load_circuit(name):
    """Load a circuit/ module by path (bare imports collide -- see tparams)."""
    spec = importlib.util.spec_from_file_location(
        f"_circuit_{name}", os.path.join(_CIRC, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class BinInductance:
    """Per-turn mutual inductance between radial bins, both coils."""

    def __init__(self, bins, n_straight=40, n_cap=60, verbose=True):
        cgeom = _load_circuit("geometry")
        cind = _load_circuit("inductance")
        ccfg = _load_circuit("cparams")

        self.bins = bins
        r = bins.flat(bins.r_center)
        turns = bins.flat(bins.turns)
        # z of each bin = its layer's z-centre
        z_layer = np.array([0.5 * (zt + zb) for zt, zb in
                            zip(params.layer_z_tops, params.layer_z_bottoms)])
        z = np.repeat(z_layer, bins.n_bins)

        gmd = ccfg.GMD_FACTOR * (params.t + params.w)
        gmd2 = gmd * gmd
        two_coil = bool(getattr(params, "two_coil_mode", True))
        g2 = 2.0 * params.coil_half_gap

        N = len(r)
        loops = []
        for k in range(N):
            p, d, _ = cgeom.racetrack_loop(r[k], z[k], params.L,
                                           n_straight, n_cap)
            pm = None
            if two_coil:
                pm = p.copy()
                pm[:, 2] = g2 - p[:, 2]
            loops.append((p, d, pm))

        M = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                v = cind._neumann_pair(loops[i][0], loops[i][1],
                                       loops[j][0], loops[j][1], gmd2)
                if two_coil:
                    v += cind._neumann_pair(loops[i][0], loops[i][1],
                                            loops[j][2], loops[j][1], gmd2)
                M[i, j] = M[j, i] = v

        self.M = M
        self.turns = turns
        self.l_turn = bins.flat(bins.l_turn)
        # A[k,j] = n_j M[k,j]: flux linked by ONE turn of bin k per amp in
        # each turn of bin j, summed over that bin's n_j turns
        self.A = M * turns[None, :]
        self.L_total = float(turns @ M @ turns) * (2.0 if two_coil else 1.0)
        if verbose:
            print(f"  [induction] bin inductance built: "
                  f"L_total = {self.L_total*1e3:.2f} mH "
                  f"(circuit/ reference 419.7 mH — the coarse "
                  f"{bins.n_bins}-bin/layer grouping will read a few % high)",
                  flush=True)

    def E_induced(self, I_z, I_z_prev, dt):
        """E_i per bin [V/m] from dI/dt.  Sign matches -dPhi/dt / l_turn."""
        dIdt = (np.asarray(I_z) - np.asarray(I_z_prev)) / dt
        return -(self.A @ dIdt) / self.l_turn
