"""
make_optimal_configuration_updated.py
=======================================
Updated-design version (2026-10-04) of make_optimal_configuration.py: the
isometric |sigma_hoop| render for the 28-layer best design of
optimize/runs/updated_10_2/ (eval 166) at its I_op = 131.87 A.

Same stress formula (make_optimal_configuration.compute_hoop_stress_
everywhere: sigma = f_n * r, r = distance to the nearest end-cap centre,
uniform-J Lorentz force density) and the same renderer
(plot_field_poster). The input is the uniform-J coil field of a single
A-form solve of this design at 131.87 A (medium mesh), saved with its
geometry in optimize/runs/ramp_hold/updated_10_2_eval166_131p9A_uniformJ_
fields.npz; any npz with `coil_B_uniform` works (e.g. a ramp_hold_test.py
T-A snapshot) via FIELDS_NPZ. Display-only settings (coil separation,
point/fill/outline alpha) are env-overridable -- see the constants below.
Hoop stress is uniform-J based, so it is unaffected by the 2026-10-04
T-A resistivity fix.

Note on the formula's caveat (see make_optimal_configuration.py): the
nearest-cap-centre radius stays well-behaved only while the straight
half-length L = b - a is small next to the pack radius. Here L = 7.9 mm
vs a_out ~ 40 mm, so it holds.

Input:  optimize/runs/ramp_hold/updated_10_2_eval166_131p9A_uniformJ_fields.npz
        (or env FIELDS_NPZ)
Output: visualization/for poster/optimal_configuration_updated.png
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_VIZ = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_VIZ)
for _p in (_ROOT, _VIZ, _HERE, os.path.join(_ROOT, "physics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
from make_optimal_configuration import compute_hoop_stress_everywhere
import plot_field_poster as pfp
from plot_field_poster import plot_field_poster

GAP_STRETCH_MM = float(os.environ.get("GAP_STRETCH_MM", 90.0))
POINT_ALPHA = float(os.environ.get("POINT_ALPHA", 0.4))
POINT_SIZE = float(os.environ.get("POINT_SIZE", 3.0))
FILL_ALPHA = float(os.environ.get("FILL_ALPHA", 0.15))
# 28 layers x 2 coils of per-layer outlines swamp the picture at the
# renderer's default weight (alpha 0.85, lw 1.3)
OUTLINE_ALPHA = float(os.environ.get("OUTLINE_ALPHA", 0.18))
OUTLINE_LW = float(os.environ.get("OUTLINE_LW", 0.45))

FIELDS_NPZ = os.environ.get("FIELDS_NPZ", os.path.join(
    _ROOT, "optimize", "runs", "ramp_hold",
    "updated_10_2_eval166_131p9A_uniformJ_fields.npz"))


if __name__ == "__main__":
    d = np.load(FIELDS_NPZ)
    params.a = float(d["a"])
    params.b = float(d["b"])
    params.coil_half_gap = float(d["coil_half_gap"])
    params.n_turns = list(map(int, d["n_turns"]))
    params.recompute_derived()
    data = dict(I_solved=d["I_solved"], coil_centroids=d["coil_centroids"],
                coil_B=d["coil_B_uniform"])
    print(f"  geometry from npz: a={params.a*1e3:.2f}mm b={params.b*1e3:.2f}mm "
          f"gap={params.coil_half_gap*1e3:.2f}mm n_layers={params.n_layers}  "
          f"I={float(d['I_solved']):.2f} A")
    sigma_hoop = compute_hoop_stress_everywhere(data)
    print(f"  hoop stress (extended): max {sigma_hoop.max()/1e6:.1f} MPa, "
          f"mean {sigma_hoop.mean()/1e6:.1f} MPa")
    # Display-only: the 28-layer stacks are ~112 mm tall each and the real
    # face gap is 3.3 mm, so the default 12 mm separation and dense opaque
    # points (~235k of them) hide the structure. Spread the coils apart and
    # thin the points so the interior gradient reads.
    pfp.OUTLINE_ALPHA, pfp.OUTLINE_LW = OUTLINE_ALPHA, OUTLINE_LW
    plot_field_poster(data, out_name="for poster/optimal_configuration_updated.png",
                      value_raw=sigma_hoop / 1e6, cbar_label="|σ_hoop|  (MPa)",
                      gap_stretch_mm=GAP_STRETCH_MM, point_alpha=POINT_ALPHA,
                      point_size=POINT_SIZE, fill_alpha=FILL_ALPHA)
