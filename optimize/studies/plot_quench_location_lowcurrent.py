"""
plot_quench_location_lowcurrent.py -- quench-location plots for the
2026-09-02/03 ta_safe_margin_search's best (T-A-locally-safe) candidate,
using the field snapshot regenerate_ta_safe_lowcurrent_fields.py wrote.
Thin wrapper around transient/validation/plot_quench_location.py's main()
-- geometry is applied from the npz's own saved a/b/coil_half_gap/n_turns,
so this works regardless of what params.py's live geometry currently is.

Run:  <env>/bin/python3 optimize/studies/plot_quench_location_lowcurrent.py
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

NPZ = os.path.join(_OPT, "runs", "ta_safe_margin", "lowcurrent_design_fields.npz")

if __name__ == "__main__":
    main(npz_path=NPZ, out_prefix="quench_location_lowcurrent",
        design_label="T-A-safe low-current design (a=49.4mm, b=54.6mm, "
                     "n_turns=[1032,1032,153,153,442,442])")
