"""
mesh_convergence_followup.py -- pin down the fine-z end (2026-07-31)
=====================================================================
mesh_convergence_champion.py found the champion's box uniformity is NOT
mesh-converged along the z (tape-width) axis:

    uniform nz=2  1921 cells  0.549%
    uniform nz=3  2857 cells  0.547%
    graded 5 PROD 4844 cells  0.686%
    graded 7      6689 cells  0.996%     <- single realization
    graded 9      8539 cells  DIVERGED (NaN, Picard hit the 150 cap)

Monotonically increasing, and graded7 lands exactly on the 1% target, so
the design's PASS/FAIL verdict now hinges on a single unreplicated number
whose next refinement step failed to solve. This run settles three
things:

1. **Is graded7's 0.996% reproducible?** 3 independent meshes. The
   perturbation study measured <=0.003pp repeat noise at production
   resolution for 22 of 23 designs -- but one (jitter4) showed 0.086pp,
   so a single realization at a NEW resolution is not self-evidently
   trustworthy.
2. **A valid 9-slab point.** graded9's edge slabs were 0.035*w = 0.14mm,
   about half graded7's thinnest; combined with the known tendency of
   graded/nz>=5 meshes to plateau just above the raw |dB|/|B| tolerance
   (CLAUDE.md), it hit the iteration cap and produced NaN. Retried with a
   gentler grading (min fraction 0.05, matching graded7's thinnest) and a
   raised iteration cap.
3. **Is the Picard actually converged in the BOX observable?** The
   convergence criterion (params.ta_scif_stall_mT) watches the EMA of the
   ON-AXIS bore SCIF -- a quantity this project has repeatedly shown is
   near-cancelling and anti-correlated with true box uniformity. A solve
   can therefore report converged=True while the box metric is still
   drifting. The high-iteration-cap variants below test whether the
   answer moves when the solver is allowed to run much longer.

Note `ta_n_picard` is itself a params attribute, so it goes through
ta_validate.py's same "mesh" override hook (which sets any params
attribute, not only mesh ones).

Run:
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \
        optimize/studies/mesh_convergence_followup.py
"""
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import mesh_convergence_champion as base   # reuse the runner machinery

GRADED7 = [0.05, 0.09, 0.13, 0.46, 0.13, 0.09, 0.05]
GRADED9_GENTLE = [0.05, 0.07, 0.09, 0.12, 0.34, 0.12, 0.09, 0.07, 0.05]

# (label, axis, params-override dict)
CONFIGS = [
    # 1. reproducibility of the decisive point
    ("g7_rep1", "z-repeat", dict(mesh_z_grading=GRADED7)),
    ("g7_rep2", "z-repeat", dict(mesh_z_grading=GRADED7)),
    ("g7_rep3", "z-repeat", dict(mesh_z_grading=GRADED7)),
    # 2. a valid 9-slab point (gentler grading + higher iteration cap)
    ("g9_gentle_r1", "z-fine", dict(mesh_z_grading=GRADED9_GENTLE,
                                    ta_n_picard=400)),
    ("g9_gentle_r2", "z-fine", dict(mesh_z_grading=GRADED9_GENTLE,
                                    ta_n_picard=400)),
    # 3. does the answer move when the solver runs much longer?
    ("g7_iter400", "solver", dict(mesh_z_grading=GRADED7, ta_n_picard=400)),
    ("g5_iter400", "solver", dict(mesh_z_grading=base.PROD_GRADING,
                                  ta_n_picard=400)),
]

RUN_DIR = os.path.join(base._ROOT, "optimize", "runs", "mesh_convergence_fu")
os.makedirs(os.path.join(RUN_DIR, "logs"), exist_ok=True)


def main():
    # redirect the reused runner's output paths to this study's own dir
    base.RUN_DIR = RUN_DIR
    base.LOGS_DIR = os.path.join(RUN_DIR, "logs")
    base.LOG_PATH = os.path.join(RUN_DIR, "log.txt")
    base.CSV_PATH = os.path.join(RUN_DIR, "results.csv")
    base.CONFIGS = CONFIGS
    base.main()


if __name__ == "__main__":
    main()
