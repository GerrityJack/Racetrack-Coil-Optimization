"""
loss.py — dissipation from the transient T-A + NI state.

Two channels, and the difference between them is the whole reason Phase B
exists on top of Phase A's circuit model:

  HYSTERETIC (superconducting).  P_sc = sum_cells rho_fn * |J_sc|^2 * dV.
  This is exact rather than a bolt-on: the SC occupies a fraction
  delta_SC/Lambda of each homogenised cell and ta_solve's rho_fn is already
  rho_sc*(delta_SC/Lambda), so rho_fn*|J_sc|^2*dV_bulk IS
  integral(rho_sc |J_sc|^2 dV_SC).  It carries the screening-current
  distribution across the tape width, which a lumped circuit model cannot
  represent at all -- the Phase A DCN reports ~0.02 J here and that number is
  meaningless by construction.

  CONTACT.  P_c = sum_bins n_turns * I_r^2 * R_ct.  Same quantity the DCN
  computes, and the cross-check between them is the point of B2.

SYMMETRY FACTORS -- these differ between the two channels, which is easy to
get wrong:
  * The FEM domain is a QUARTER of coil 1's ring (x>=0, y>=0; it spans coil
    1's full z extent).  So cell sums need x4 for the quadrants and x2 for
    coil 2 = x8.  Same 8 pieces dB_bore_from_dJ expands to.
  * The radial bins are built from params.n_turns, which is already the FULL
    turn count of one coil.  So bin sums need x2 only.

NOT INCLUDED, in either channel: eddy and coupling loss in the copper
stabiliser and substrate.  There is no normal-metal conductivity anywhere in
this model, so every number here is a LOWER BOUND on the real heat load.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CELL_SYMMETRY = 8.0     # quarter ring x 2 coils
BIN_SYMMETRY = 2.0      # bins already count a whole coil's turns


def hysteretic_power(ta, J_coil):
    """P_sc [W] for the full two-coil system."""
    rho = ta["rho_fn"].x.array[ta["coil_cells"]]
    vol = ta["coil_vols"]
    j2 = np.einsum("ij,ij->i", J_coil, J_coil)
    return CELL_SYMMETRY * float(np.sum(rho * j2 * vol))


def contact_power(circuit):
    """P_contact [W] for the full two-coil system."""
    turns = circuit.bins.flat(circuit.bins.turns)
    return BIN_SYMMETRY * float(np.sum(turns * circuit.I_r ** 2
                                       * circuit.R_ct))


def integrate(times, powers):
    """Trapezoid cumulative energy [J]."""
    t = np.asarray(times, dtype=float)
    p = np.asarray(powers, dtype=float)
    if t.size < 2:
        return np.zeros_like(p)
    return np.concatenate([[0.0],
                           np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(t))])
