"""
plot_quench_location_champion_medium.py -- quench-location plots for the
champion, re-solved at the medium mesh tier, to compare against the
standing xdense result (28.6% of cells < margin 1.0, worst-cell margin
0.586) -- does the coarser tier understate the champion's quench
severity, the same pattern found on the low-current design?

Run:  <env>/bin/python3 transient/validation/plot_quench_location_champion_medium.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plot_quench_location import main

_ROOT = os.path.dirname(os.path.dirname(_HERE))
NPZ = os.path.join(_ROOT, "solve", "racetrack_ta_fields_medium.npz")

if __name__ == "__main__":
    main(npz_path=NPZ, out_prefix="quench_location_champion_medium",
        design_label="Champion, MEDIUM mesh (n_turns=[382,382,478,478,3,3])")
