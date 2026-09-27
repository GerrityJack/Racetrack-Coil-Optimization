"""
plot_quench_location_ramphold.py -- per-cell J/Jc (load) maps for a state
saved by ramp_hold_test.py (RH_SAVE_STEPS), graded with the same 20 K
KimIcModel the solve used. Thin wrapper around
transient/validation/plot_quench_location.py's main().

Run:  <env>/bin/python3 optimize/studies/plot_quench_location_ramphold.py \
          <npz under optimize/runs/ramp_hold/> <out_prefix> "<label>"
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

if __name__ == "__main__":
    npz, prefix, label = sys.argv[1], sys.argv[2], sys.argv[3]
    if not os.path.isabs(npz):
        npz = os.path.join(_OPT, "runs", "ramp_hold", npz)
    main(npz_path=npz, out_prefix=prefix, design_label=label,
         ic_model=make_ic_model("kim"))
