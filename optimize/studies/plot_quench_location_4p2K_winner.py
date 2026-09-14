"""
plot_quench_location_4p2K_winner.py -- quench-location plots for tonight's
4.2K-model overnight search's current best (highest B_target_T) candidate,
using the field snapshot regenerate_4p2K_winner_fields.py wrote.

Thin wrapper around transient/validation/plot_quench_location.py's main()
-- geometry is applied from the npz's own saved a/b/coil_half_gap/n_turns,
and the margin is graded with the SAME 4.2K-scaled Ic model the search
itself used (not the plain 20K default), so the plot is internally
consistent with how this design was actually found/qualified.

Run:  <env>/bin/python3 optimize/studies/plot_quench_location_4p2K_winner.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_OPT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_OPT)
for _p in (_ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "transient", "validation"), _OPT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plot_quench_location import main
from ic_extrapolation import make_ic_model
from ic_temperature_scaling import Fujikura4p2KScaledIcModel

NPZ = os.path.join(_OPT, "runs", "ta_safe_margin_4p2K", "winner_fields.npz")

if __name__ == "__main__":
    ic_model = Fujikura4p2KScaledIcModel(make_ic_model("kim"))
    main(npz_path=NPZ, out_prefix="quench_location_4p2K_winner",
        design_label="4.2K-search winner so far (12 layers, a=32.19mm, "
                     "b=45.25mm, n_turns=[600,600,559,559,37,37,519,519,"
                     "583,583,601,601])",
        ic_model=ic_model)
