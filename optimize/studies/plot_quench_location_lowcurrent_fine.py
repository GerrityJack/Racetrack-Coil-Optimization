"""
plot_quench_location_lowcurrent_fine.py -- quench-location plots for the
low-current T-A-safe design, re-solved at the "fine" mesh tier (see
regenerate_ta_safe_lowcurrent_fields.py, TA_SAFE_MESH_TIER=fine) to check
whether the handful of quenched cells seen at medium resolution
(1/1833 cells, 0.1%) survive mesh refinement or were a coarse-mesh
artifact. Thin wrapper around transient/validation/plot_quench_location.py.

Run:  <env>/bin/python3 optimize/studies/plot_quench_location_lowcurrent_fine.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_OPT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_OPT)
for _p in (os.path.join(_ROOT, "transient", "validation"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plot_quench_location import main

NPZ = os.path.join(_OPT, "runs", "ta_safe_margin",
                   "lowcurrent_design_fields_fine.npz")

if __name__ == "__main__":
    main(npz_path=NPZ, out_prefix="quench_location_lowcurrent_fine",
        design_label="T-A-safe low-current design, FINE mesh "
                     "(a=49.4mm, b=54.6mm, n_turns=[1032,1032,153,153,442,442])")
