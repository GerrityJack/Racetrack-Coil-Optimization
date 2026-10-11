"""
eh_box_size_check.py -- error hunt 2026-10-04: does the FEM air box truncate
the field? Uniform-J A-form solves of the legacy design at 180 A with
increasing params.box_scale (outer PEC box edge = box_scale * b), compared
with the independent multi-filament Biot-Savart value at the bore centre
and with the peak |B| over the coil cells (which sets Jc(B) for the loads).
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
from ta_thermal_validation import apply_mesh_config
import build_mesh
import solve as base_solve
import optimize_geometry as og
from dolfinx.io import gmsh as gmshio
from dolfinx import geometry
from mpi4py import MPI

I_A = 180.0
SCALES = [float(s) for s in os.environ.get("XC_SCALES", "4,8,16").split(",")]


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
    g = params.coil_half_gap
    B_bs, pts = og.target_box_field(I_A)
    i0 = int(np.argmin(np.linalg.norm(pts[:, :2], axis=1)))
    print(f"Biot-Savart bore centre Bz = {B_bs[i0, 2]:+.4f} T, box mean |Bz| "
          f"= {np.mean(np.abs(B_bs[:, 2])):.4f} T")
    q = np.column_stack([np.abs(pts[:, 0]), np.abs(pts[:, 1]),
                         np.full(len(pts), g - 1e-6)])
    for sc in SCALES:
        params.box_scale = sc
        params.recompute_derived()
        path = os.path.join(_ROOT, "mesh", f"eh_box_{sc:g}_{os.getpid()}.msh")
        build_mesh.build(write_path=path)
        md = gmshio.read_from_msh(path, MPI.COMM_WORLD, rank=0, gdim=3)
        os.remove(path)
        dom = md.mesh
        uni = base_solve.setup_problem(dom, md.cell_tags, md.facet_tags)
        _, B = base_solve.solve_at_current(dom, uni, I_A, MPI.COMM_WORLD,
                                           verbose_label=f"box{sc:g}")
        Bq = fem_eval(dom, B, q)
        Bc = B.x.array.reshape(-1, 3)[uni["coil_cells"]]
        print(f"box_scale {sc:5.1f} (edge {sc*params.b*1e3:6.0f} mm, "
              f"{dom.topology.index_map(3).size_global} cells): box mean |Bz| "
              f"{np.mean(np.abs(Bq[:, 2])):.4f} T "
              f"({(np.mean(np.abs(Bq[:,2]))/np.mean(np.abs(B_bs[:,2]))-1)*100:+.2f} % vs BS), "
              f"coil peak |B| {np.linalg.norm(Bc, axis=1).max():.3f} T, "
              f"coil mean |B| {np.linalg.norm(Bc, axis=1).mean():.3f} T", flush=True)


if __name__ == "__main__":
    main()
