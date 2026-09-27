"""
test_ta_ramp_current.py -- run ta_ramp_current.evaluate() on one design
(default: the champion, params.py geometry) and print every trial.

Run:  <env>/bin/python3 optimize/studies/test_ta_ramp_current.py
      TA_RAMP_DESIGN_JSON='{"a":..,"b":..,"coil_half_gap":..,"n_turns":[..]}'
"""
import os
import sys
import json

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_OPT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_OPT)
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "mesh"),
           os.path.join(_ROOT, "solve"), _OPT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
from ta_thermal_validation import apply_mesh_config   # medium tier, in memory
apply_mesh_config("medium", "medium5")

from mpi4py import MPI
from ic_extrapolation import make_ic_model
import ta_ramp_current

design = json.loads(os.environ.get("TA_RAMP_DESIGN_JSON", "null")) or dict(
    a=params.a, b=params.b, coil_half_gap=params.coil_half_gap,
    n_turns=list(params.n_turns))
print(f"design {design}  t_ramp={ta_ramp_current.T_RAMP_S:.0f}s "
      f"dt_eff={ta_ramp_current.DT_EFF_S:.0f}s eps={ta_ramp_current.EPS_FLOOR}")
r = ta_ramp_current.evaluate(
    design, make_ic_model("kim"), MPI.COMM_WORLD, verbose=True,
    save_fields_path=os.environ.get("TA_RAMP_SAVE_NPZ"))
for k, v in r.items():
    print(f"  {k}: {v}")
