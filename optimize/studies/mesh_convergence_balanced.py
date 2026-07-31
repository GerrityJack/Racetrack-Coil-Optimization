"""
mesh_convergence_balanced.py -- the z-ladder at ADEQUATE in-plane resolution
============================================================================
2026-07-31, third and decisive batch.

What the first two batches established:

  in-plane axis, z held at production graded5 -- CONVERGED:
      4844 cells  0.686%   (production)
     16128 cells  0.661%
     55844 cells  0.642%     steps -0.025, -0.019pp

  z axis at PRODUCTION in-plane -- large, and REPRODUCIBLE:
      graded5  0.686%
      graded7  0.996 / 0.980 / 0.983%   (+-0.008pp repeat noise)

  z axis at FINE in-plane -- much smaller:
      graded5  0.661%
      graded7  0.736%

So z-refinement moves the answer +0.30pp on the production in-plane mesh
but only +0.075pp on the fine one. The interpretation is that the
production in-plane mesh is too coarse to support the finer z-structure:
the ~0.99% is a real number for an UNBALANCED mesh, not the converged
physics. Refining one axis alone is what produces it.

That makes the design's PASS/FAIL verdict rest entirely on a z-ladder
that has only TWO points at adequate in-plane resolution. This batch adds
the third (graded9) and repeats the existing two, so the fine-in-plane
z-trend can be read directly instead of inferred.

Note the previous batch's g9 configs used PRODUCTION in-plane -- exactly
the combination now believed to be unbalanced -- so they cannot answer
this. Everything here holds in-plane at the `fine` setting
(mesh_size_min_factor=0.125, mesh_size_max_factor=0.30) and varies only z.

Run:
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \
        optimize/studies/mesh_convergence_balanced.py
"""
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import mesh_convergence_champion as base

FINE_INPLANE = dict(mesh_size_min_factor=0.125, mesh_size_max_factor=0.30)

GRADED5 = base.PROD_GRADING
GRADED7 = [0.05, 0.09, 0.13, 0.46, 0.13, 0.09, 0.05]
GRADED9 = [0.05, 0.07, 0.09, 0.12, 0.34, 0.12, 0.09, 0.07, 0.05]


def _cfg(grading, **extra):
    d = dict(mesh_z_grading=grading, ta_n_picard=400)
    d.update(FINE_INPLANE)
    d.update(extra)
    return d


# All at FINE in-plane; only the z-grading changes. Two repeats each so the
# trend is separable from realization noise.
CONFIGS = [
    ("fi_g5_r1", "z@fine-inplane", _cfg(GRADED5)),
    ("fi_g5_r2", "z@fine-inplane", _cfg(GRADED5)),
    ("fi_g7_r1", "z@fine-inplane", _cfg(GRADED7)),
    ("fi_g7_r2", "z@fine-inplane", _cfg(GRADED7)),
    ("fi_g9_r1", "z@fine-inplane", _cfg(GRADED9)),
    ("fi_g9_r2", "z@fine-inplane", _cfg(GRADED9)),
]

RUN_DIR = os.path.join(base._ROOT, "optimize", "runs", "mesh_convergence_bal")
os.makedirs(os.path.join(RUN_DIR, "logs"), exist_ok=True)


def main():
    base.RUN_DIR = RUN_DIR
    base.LOGS_DIR = os.path.join(RUN_DIR, "logs")
    base.LOG_PATH = os.path.join(RUN_DIR, "log.txt")
    base.CSV_PATH = os.path.join(RUN_DIR, "results.csv")
    base.CONFIGS = CONFIGS
    base.TIMEOUT_S = 4500          # finest configs are slow (~10-25 min)
    base.main()


if __name__ == "__main__":
    main()
