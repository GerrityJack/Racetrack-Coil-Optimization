"""
test_12mm_tape_alpha_sweep.py -- 2026-09-14

The naive 12mm-tape-width test (test_12mm_tape_width.py) at DEFAULT alpha
(0.30, 0.15) did not converge (raw |dB|/|B|=0.277, SCIF oscillating
chaotically 857-1458 mT even at k=150) -- same class of problem the whole
4.2K investigation was built around: default alpha provides ~zero damping
outside the regime it was validated for. There, the stiff regime was
n~50-60 (temperature). Here it's a much higher local j/jc from the 3x
Ic-scaled, much-higher-current design (worst_margin=0.627 at the naive
result means a large fraction of cells sit at/above their local Ic).

This sweeps the same reduced-alpha candidates the 4.2K investigation
tried, at the SAME fixed current (I_hi=476.56A, the actual bracket the
real evaluation needs), forced to the full ta_n_picard=150 iterations,
checking the RAW residual (not the EMA stall flag) for genuine
convergence -- same methodology, applied to a new regime.

Run:  <env>/bin/python3 optimize/studies/test_12mm_tape_alpha_sweep.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_OPT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_OPT)
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "mesh"),
           os.path.join(_ROOT, "solve"), _OPT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from mpi4py import MPI
import params
import build_mesh
import solve as base_solve
import ta_solve
from dolfinx.io import gmsh as gmshio
from ic_extrapolation import make_ic_model
from ic_model import NValueModel
from ta_safe_current import _local_margin

params.mesh_size_min_factor = 0.25
params.mesh_size_max_factor = 0.60
params.mesh_dist_min_factor = 1.50
params.mesh_dist_max_factor = 2.00
params.box_scale = 4.0
params.mesh_z_grading = [0.075, 0.15, 0.55, 0.15, 0.075]

W_NEW = 0.012
WIDTH_RATIO = W_NEW / 0.004
N_LAYERS = 16
MIN_COIL_GAP_M = 0.003


class WidthScaledIcModel:
    kind = "width_scaled"

    def __init__(self, base_ic_model, ratio):
        self.base = base_ic_model
        self.ratio = ratio
        self.tape_width = base_ic_model.tape_width * ratio
        self.B_min = base_ic_model.B_min

    def critical_current(self, B_tesla, theta_deg, clip_B=True):
        Ic, frac = self.base.critical_current(B_tesla, theta_deg, clip_B=clip_B)
        return Ic * self.ratio, frac


def gap_floor(n_layers, w):
    return n_layers * w / 2.0 + MIN_COIL_GAP_M / 2.0


if __name__ == "__main__":
    a = 31.130821333791943e-3
    b = 40.01565009808927e-3
    n_turns = [550, 550, 584, 584, 17, 17, 507, 507, 507, 507, 613, 613,
               629, 629, 605, 605]
    orig_gap = 33.97411042487954e-3
    margin_above_floor = orig_gap - gap_floor(N_LAYERS, 0.004)
    new_gap = gap_floor(N_LAYERS, W_NEW) + margin_above_floor

    params.a = a
    params.b = b
    params.coil_half_gap = new_gap
    params.n_turns = list(n_turns)
    params.w = W_NEW
    params.recompute_derived()

    comm = MPI.COMM_WORLD
    mesh_path = f"/tmp/12mm_alpha_sweep_{os.getpid()}.msh"
    print("Building mesh...", flush=True)
    t0 = time.time()
    build_mesh.build(write_path=mesh_path)
    md = gmshio.read_from_msh(mesh_path, comm, rank=0, gdim=3)
    domain = md.mesh
    uniform_setup = base_solve.setup_problem(domain, md.cell_tags, md.facet_tags)
    ta = ta_solve.setup_ta_problem(domain, md.cell_tags, md.facet_tags, uniform_setup)
    print(f"mesh+setup: {time.time()-t0:.1f}s", flush=True)
    try:
        os.remove(mesh_path)
    except OSError:
        pass

    base_ic = make_ic_model("kim")
    ic_model = WidthScaledIcModel(base_ic, WIDTH_RATIO)
    n_model = NValueModel(csv_path=params.n_value_csv_filename)  # UNSCALED

    I_TEST = 476.56  # the actual I_hi bracket the real evaluation needs

    ALPHA_SWEEP = [(0.30, 0.15), (0.15, 0.08), (0.10, 0.05), (0.05, 0.02),
                  (0.03, 0.01), (0.02, 0.008), (0.01, 0.005)]

    print(f"\nSweeping alpha at I={I_TEST}A, forced {params.ta_n_picard} "
         f"iterations, cold start each time\n", flush=True)
    for ah, al in ALPHA_SWEEP:
        params.ta_picard_alpha = ah
        params.ta_picard_alpha_fine = al
        t0 = time.time()
        A_h, B_h, T_h, info = ta_solve.solve_ta_at_current(
            domain, ta, uniform_setup, I_amps=I_TEST, ic_model=ic_model,
            n_model=n_model, verbose=False, warm_start=False,
            min_iters=params.ta_n_picard)
        dt = time.time() - t0
        margin_arr, clip_ta, _, _ = _local_margin(domain, ta, B_h, ic_model)
        worst = float(margin_arr.min())
        p5 = float(np.percentile(margin_arr, 5))
        print(f"alpha=({ah},{al})  wall={dt:.1f}s  n_iters={info['n_iters']}  "
             f"rel_err={info['rel_err']:.3e}  scif_mT={info['scif_mT']:.2f}  "
             f"worst_margin={worst:.4f}  p5_margin={p5:.4f}", flush=True)
