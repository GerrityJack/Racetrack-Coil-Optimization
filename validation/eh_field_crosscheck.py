"""
eh_field_crosscheck.py -- error hunt 2026-10-04: is B_target computed right?
==============================================================================
Every reported B_target / uniformity comes from Biot-Savart sums, not from
the FEM field:
    B = og.target_box_field(I)            (multi-filament uniform-J BS)
      + dB_bore_from_dJ(cells, dJ_s, ...) (T-A screening correction, BS over
                                           FEM cells with 8 mirror images)
The FEM itself solves for A everywhere (eighth domain, PEC/PMC symmetry),
so its B at the box points is an independent value of the same quantity.
This compares, at the 30x6 mm box points on the midplane:
  1. uniform-J:  FEM B_h(uniform)  vs  og.target_box_field
  2. T-A total:  FEM B_h(T-A)      vs  BS uniform + dB_bore_from_dJ
  3. the T-A - uniform DIFFERENCE both ways (tests the SCIF sum itself,
     with FEM discretisation error largely cancelling)
Legacy 6-layer design, 180 A, 4 h step, floor 1.0 (cheap; the point is the
summation, not the floor physics).
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "mesh"),
           os.path.join(_ROOT, "solve"), os.path.join(_ROOT, "optimize"),
           os.path.join(_ROOT, "optimize", "studies")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
from ta_thermal_validation import apply_mesh_config, build_setup, make_models
import ta_solve
import solve as base_solve
import optimize_geometry as og
import dolfinx.mesh as dmesh
from dolfinx import geometry
from mpi4py import MPI

I_A = float(os.environ.get("XC_I", 180.0))


def fem_eval(domain, fn, pts):
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)
    cells = [coll.links(i)[0] for i in range(len(pts))]
    return fn.eval(pts, np.array(cells, dtype=np.int32))


def main():
    params.a, params.b, params.coil_half_gap = 0.026, 0.0314, 0.0137
    params.n_turns = [382, 382, 478, 478, 3, 3]
    apply_mesh_config("medium", "medium5")
    if os.environ.get("XC_BOX"):          # air-box edge in units of b
        params.box_scale = float(os.environ["XC_BOX"])
        params.recompute_derived()
    print(f"box_scale = {params.box_scale}")
    domain, uni, ta = build_setup("xc")
    ic, nm = make_models("20K")
    g = params.coil_half_gap

    B_bs_u, pts = og.target_box_field(I_A)            # full-box points
    # FEM covers x>=0, y>=0, z<=g: Bz is even in x and y; evaluate at
    # (|x|, |y|, g - eps) and compare Bz and |B|
    q = np.column_stack([np.abs(pts[:, 0]), np.abs(pts[:, 1]),
                         np.full(len(pts), g - 1e-6)])

    _, B_u = base_solve.solve_at_current(domain, uni, I_A, MPI.COMM_WORLD,
                                         verbose_label="xc")
    Bf_u = fem_eval(domain, B_u, q)

    params.ta_eps_reg = 1.0
    params.ta_picard_alpha = params.ta_picard_alpha_fine = 0.10
    params.ta_n_picard = 150
    ta["dt_const"].value = 14400.0
    _, B_t, _, _ = ta_solve.solve_ta_at_current(
        domain, ta, uni, I_amps=I_A, ic_model=ic, n_model=nm, verbose=False,
        warm_start=False, min_iters=150)
    Bf_t = fem_eval(domain, B_t, q)

    J_TA = ta_solve._J_from_T(ta, domain)
    J_un = ta["t_hat_coil"] * (I_A / (params.delta_SC * params.w))
    dJ_s = (J_TA - J_un) * (params.delta_SC / params.t)
    cents = dmesh.compute_midpoints(domain, 3, ta["coil_cells"])
    dB = np.array([ta_solve.dB_bore_from_dJ(cents, dJ_s, ta["coil_vols"],
                                            bore_pt=p) for p in pts])
    B_bs_t = B_bs_u + dB

    def stats(name, a, b):
        da = a[:, 2] - b[:, 2]
        print(f"{name:34s} mean Bz {np.mean(np.abs(a[:,2])):9.4f} vs "
              f"{np.mean(np.abs(b[:,2])):9.4f} T  diff mean {np.mean(da)*1e3:+8.2f} mT"
              f"  max|d| {np.max(np.abs(da))*1e3:7.2f} mT")
    print(f"legacy design, I = {I_A} A, {len(pts)} box points")
    stats("uniform-J: FEM vs Biot-Savart", Bf_u, B_bs_u)
    stats("T-A total: FEM vs BS+dB", Bf_t, B_bs_t)
    d_fem = Bf_t[:, 2] - Bf_u[:, 2]
    d_bs = dB[:, 2]
    print(f"SCIF (T-A - uniform) Bz: FEM mean {np.mean(d_fem)*1e3:+.2f} mT, "
          f"BS mean {np.mean(d_bs)*1e3:+.2f} mT, ratio "
          f"{np.mean(d_bs)/np.mean(d_fem) if np.mean(d_fem) else float('nan'):.3f}")
    print(f"SCIF spread across box: FEM {np.ptp(d_fem)*1e3:.2f} mT, "
          f"BS {np.ptp(d_bs)*1e3:.2f} mT")
    for lab, B in (("FEM T-A", Bf_t), ("BS T-A", B_bs_t)):
        m = np.linalg.norm(B, axis=1)
        print(f"uniformity {lab:8s}: {(m.max()-m.min())/m.mean()*100:.3f} %")


if __name__ == "__main__":
    main()
