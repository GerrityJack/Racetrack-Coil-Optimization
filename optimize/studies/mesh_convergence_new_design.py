"""
mesh_convergence_new_design.py -- is the new design's uniformity converged?
===========================================================================
2026-07-31. The 2026-07-31 design was promoted on a box uniformity of
0.442% measured by `ta_in_loop_search.py`. The jitter study then measured
the SAME geometry at 0.938/0.942%, and two further independent processes
at 0.9479 / 0.9347%.

The only difference between the two geometries is `coil_half_gap`:
13.500000 mm (0.442%) vs 13.500289 mm (0.94%) -- a 0.289 um change, 0.002%
of the geometry. That flipped the gmsh mesh from 4390 to 4400 coil cells.
gmsh is deterministic, so each value reproduces its own answer reliably --
this is not random noise, it is the box metric being UNDER-RESOLVED: two
physically indistinguishable geometries give 0.44% and 0.94% because the
metric is a near-cancelling dipole sum that the production mesh cannot
pin down for this design.

(The same failure mode invalidated the n_layers=4 design earlier in this
project: 0.79% in-process vs 2.19% on an independent mesh.)

Meanwhile every PERTURBED design in the jitter study -- a+-0.2mm,
b+-0.2mm, gap-0.2mm -- lands at 0.36-0.49%, which suggests the nominal's
0.94% is the unrepresentative mesh rather than the truth. But that is a
hypothesis, and the only way to settle it is refinement.

This runs the same 2-axis refinement ladder that produced a trustworthy
answer for the previous champion (0.62-0.69%, from 15 configurations):
z-resolution across the tape width, and in-plane resolution, plus repeats
at production settings and at BOTH gap values so the discontinuity itself
is characterized.

Run:
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \\
        optimize/studies/mesh_convergence_new_design.py
"""
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import mesh_convergence_champion as base

# the promoted design; I_op from its own FEM evaluation under the kim model
DESIGN_GAP_A = 0.0135              # what ta_in_loop used -> 0.442%
DESIGN_GAP_B = 0.013500289306395013  # what the jitter study used -> 0.94%

PROD = base.PROD_GRADING
GRADED7 = [0.05, 0.09, 0.13, 0.46, 0.13, 0.09, 0.05]
GRADED9 = [0.05, 0.07, 0.09, 0.12, 0.34, 0.12, 0.09, 0.05, 0.07]
GRADED9 = [0.05, 0.07, 0.09, 0.12, 0.34, 0.12, 0.09, 0.07, 0.05]
FINE_INPLANE = dict(mesh_size_min_factor=0.125, mesh_size_max_factor=0.30)

CONFIGS = [
    # -- characterize the discontinuity itself: both gaps, production mesh
    ("gapA_prod", "gap-discontinuity", dict(mesh_z_grading=PROD)),
    ("gapB_prod", "gap-discontinuity", dict(mesh_z_grading=PROD)),
    # -- z refinement (production in-plane), on gap B (the current params.py)
    ("z_nz3", "z", dict(mesh_z_grading=None, mesh_nz_per_layer=3)),
    ("z_graded7", "z", dict(mesh_z_grading=GRADED7, ta_n_picard=400)),
    ("z_graded9", "z", dict(mesh_z_grading=GRADED9, ta_n_picard=400)),
    # -- in-plane refinement (production z)
    ("inplane_fine", "in-plane", dict(mesh_z_grading=PROD, **FINE_INPLANE)),
    ("inplane_finer", "in-plane", dict(
        mesh_z_grading=PROD, mesh_size_min_factor=0.0625,
        mesh_size_max_factor=0.15)),
    # -- both refined: the best available estimate
    ("both_fine", "both", dict(mesh_z_grading=GRADED7, ta_n_picard=400,
                               **FINE_INPLANE)),
]

RUN_DIR = os.path.join(base._ROOT, "optimize", "runs", "mesh_conv_new_design")
os.makedirs(os.path.join(RUN_DIR, "logs"), exist_ok=True)


def main():
    base.RUN_DIR = RUN_DIR
    base.LOGS_DIR = os.path.join(RUN_DIR, "logs")
    base.LOG_PATH = os.path.join(RUN_DIR, "log.txt")
    base.CSV_PATH = os.path.join(RUN_DIR, "results.csv")
    base.TIMEOUT_S = 4500
    base.DESIGN = dict(a=0.023227029065529628, b=0.02826822715975084,
                       coil_half_gap=DESIGN_GAP_B,
                       n_turns=[329, 329, 411, 411, 2, 2],
                       I_design=204.5866)

    # the one config that must use the OTHER gap value
    orig_run = base.run_config

    def run_config(cfg_tuple):
        label, axis, mesh = cfg_tuple
        if label == "gapA_prod":
            saved = base.DESIGN
            base.DESIGN = dict(saved, coil_half_gap=DESIGN_GAP_A,
                               I_design=204.5737)
            try:
                return orig_run(cfg_tuple)
            finally:
                base.DESIGN = saved
        return orig_run(cfg_tuple)

    base.run_config = run_config
    base.CONFIGS = CONFIGS
    base.main()


if __name__ == "__main__":
    main()
