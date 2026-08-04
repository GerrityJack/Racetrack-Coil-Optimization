"""
turn_map.py — map FEM entities onto radial bins of the winding.

Two mappings are needed by the NI coupling:

  * coil CELLS -> bin, so E_p and E_i can be averaged over each bin
  * tape-edge BC DOFS -> bin, so the per-bin transport current can be written
    into the Dirichlet data

Both go through physics/current_source.turn_index_xy_multilayer(), which is
already the repo's authority on "which turn is this point in".  Note its
conventions: layer 0 is the TOP of the local stack, and turn 0 is the
INNERMOST turn of a layer.

Binning (rather than per-turn resolution) is forced by the mesh, not chosen
for convenience -- see tparams.N_RADIAL_BINS.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "physics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params                                       # noqa: E402
from current_source import turn_index_xy_multilayer  # noqa: E402


class RadialBins:
    """Per-(layer, radial bin) bookkeeping.

    Attributes
    ----------
    n_bins     : bins per layer
    n_layers   : layers
    key(l, b)  : flat index of bin b of layer l
    turns      : (n_layers, n_bins) turns in each bin
    r_center   : (n_layers, n_bins) mean turn radius [m]
    l_turn     : (n_layers, n_bins) conductor length of one turn [m]
    """

    def __init__(self, n_bins):
        self.n_bins = int(n_bins)
        self.n_layers = params.n_layers
        self.L = params.L

        nb, nl = self.n_bins, self.n_layers
        self.turns = np.zeros((nl, nb))
        self.r_center = np.zeros((nl, nb))
        for i, n_i in enumerate(params.n_turns):
            # turn k (0 = innermost) sits at radius a_inner + (k+0.5)*t
            a_in = params.a_out - n_i * params.t
            k = np.arange(n_i)
            r_k = a_in + (k + 0.5) * params.t
            b = self._bin_of_turn(k, n_i)
            for bb in range(nb):
                m = b == bb
                self.turns[i, bb] = m.sum()
                self.r_center[i, bb] = r_k[m].mean() if m.any() else np.nan
        # empty bins (layers with fewer turns than bins) fall back to the
        # layer's own mean radius so downstream arithmetic stays finite
        for i, n_i in enumerate(params.n_turns):
            bad = ~np.isfinite(self.r_center[i])
            if bad.any():
                self.r_center[i, bad] = params.a_center_list[i]
        self.l_turn = 4.0 * params.L + 2.0 * np.pi * self.r_center

    def _bin_of_turn(self, turn, n_i):
        """Turn index -> bin index, equal-turn-count bins."""
        turn = np.asarray(turn)
        return np.clip((turn * self.n_bins) // max(n_i, 1), 0,
                       self.n_bins - 1).astype(int)

    def key(self, layer, b):
        return layer * self.n_bins + b

    @property
    def n_flat(self):
        return self.n_layers * self.n_bins

    # ── mapping FEM entities in ──────────────────────────────────────────

    def index_points(self, xyz):
        """(N,3) coordinates -> (layer, bin, flat_index) arrays."""
        xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
        li, turn = turn_index_xy_multilayer(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                                            self.L)
        li = np.asarray(li, dtype=int)
        n_arr = np.asarray(params.n_turns)[li]
        b = np.clip((np.asarray(turn) * self.n_bins) // np.maximum(n_arr, 1),
                    0, self.n_bins - 1).astype(int)
        return li, b, li * self.n_bins + b

    def contact_resistance(self, rho_ct_ohm_m2):
        """R_ct seen by ONE turn of each bin [Ohm], flattened.

        Same ladder definition as the Phase A DCN: the radial shortcut spans
        one turn, so the contact area is (turn length) x (tape width).
        """
        return (rho_ct_ohm_m2 / (self.l_turn * params.w)).ravel()

    def flat(self, arr2d):
        return np.asarray(arr2d).ravel()


def build_cell_map(bins, coil_centroids):
    """Rows of coil_cells -> flat bin index, plus a list of row groups."""
    _, _, flat = bins.index_points(coil_centroids)
    groups = [np.nonzero(flat == k)[0] for k in range(bins.n_flat)]
    return flat, groups


def build_dof_map(bins, dof_coords, dofs):
    """Tape-edge DOF indices -> flat bin index (same length as `dofs`)."""
    _, _, flat = bins.index_points(dof_coords[dofs])
    return flat
