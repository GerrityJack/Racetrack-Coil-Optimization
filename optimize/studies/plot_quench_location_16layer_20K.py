"""
plot_quench_location_16layer_20K.py -- quench-location plots for the
16-layer design (4.2K-search eval 13) re-solved at 20K, 4mm tape, at
I=88.5A -- the ~10T operating point found by relaxed_margin_test.py
(2026-09-24: 95% of cells <= 100% of local Ic).

Thin wrapper around transient/validation/plot_quench_location.py's main();
graded with the plain 20K KimIcModel, the same model the solve used.
Field snapshot written by:
    RELAX_I_SWEEP=80,88.5 RELAX_SAVE_NPZ=... relaxed_margin_test.py

Run:  <env>/bin/python3 optimize/studies/plot_quench_location_16layer_20K.py
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

NPZ = os.path.join(_OPT, "runs", "relaxed_margin",
                   "16layer_20K_I88p5_fields.npz")

if __name__ == "__main__":
    main(npz_path=NPZ, out_prefix="quench_location_16layer_20K_88p5A",
         design_label="16-layer design @ 20K, 4mm tape (a=31.13mm, "
                      "b=40.02mm, 8024 turns), ~10T operating point",
         ic_model=make_ic_model("kim"))
