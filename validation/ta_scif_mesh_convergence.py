"""
ta_scif_mesh_convergence.py
===========================
Convergence of the bore SCIF with z-resolution across the tape width.

The screening-current profile lives across the tape width w (the z
direction within each layer), so `params.mesh_nz_per_layer` — the number
of mesh sub-slabs per layer — is the resolution that matters for the T-A
solve.  This study rebuilds the mesh at increasing nz and re-solves the
per-layer T-A problem at I_design for BOTH Ic datasets (the Shanghai CSV,
over-critical at 200 A, and the default Low_Field CSV, sub-critical),
tracking the bore SCIF ΔBz.

Run from Racetrack_v4 root:
    conda run -n fenicsx-env python3 validation/ta_scif_mesh_convergence.py
"""
import os, sys, time
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
params.ta_picard_tol = 1e-4
params.ta_n_picard   = 150

from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
import solve as base_solve
import build_mesh
from ic_model import IcModel, NValueModel
from ta_solve import (setup_ta_problem, solve_ta_at_current,
                      _J_from_T, dB_bore_from_dJ)

NZ_VALUES = (1, 2, 3, 4)
I_STUDY   = float(params.I_design)


def main():
    comm  = MPI.COMM_WORLD
    n_mod = NValueModel(csv_path=params.n_value_csv_filename)
    datasets = (("shanghai", IcModel(csv_path=params.shanghai_csv_filename)),
                ("strong",   IcModel()))

    rows = []
    for nz in NZ_VALUES:
        params.mesh_nz_per_layer = nz
        t0 = time.time()
        build_mesh.build(write_path=params.mesh_filename)
        md = gmshio.read_from_msh(params.mesh_filename, comm, rank=0, gdim=3)
        domain = md.mesh
        us = base_solve.setup_problem(domain, md.cell_tags, md.facet_tags)
        ta = setup_ta_problem(domain, md.cell_tags, md.facet_tags, us,
                              per_layer=True)
        n_dofs  = us["V"].dofmap.index_map.size_global
        n_coil  = len(us["coil_cells"])
        t_setup = time.time() - t0
        print(f"\n── nz = {nz}  ({n_dofs} A-dofs, {n_coil} coil cells, "
              f"setup {t_setup:.0f} s) " + "─" * 20)

        row = dict(nz=nz, dofs=n_dofs, coil_cells=n_coil)
        for label, ic_mod in datasets:
            t1 = time.time()
            *_, info = solve_ta_at_current(domain, ta, us, I_STUDY,
                                           ic_mod, n_mod, verbose=False,
                                           warm_start=False)
            J_TA   = _J_from_T(ta, domain)
            J_unif = ta["t_hat_coil"] * (I_STUDY / (ta["delta_SC"] * params.w))
            dJ_s   = (J_TA - J_unif) * (ta["delta_SC"] / ta["Lambda"])
            dB     = dB_bore_from_dJ(ta["coil_centroids"], dJ_s,
                                     ta["coil_vols"])
            row[label] = dB[2] * 1e3
            row[f"{label}_k"] = info["n_iters"]
            row[f"{label}_conv"] = info["converged"]
            print(f"  {label:9s}: SCIF dBz = {dB[2]*1e3:+8.2f} mT   "
                  f"(k={info['n_iters']}, conv={info['converged']}, "
                  f"{time.time()-t1:.0f} s)")
        rows.append(row)

    print(f"\n{'='*70}")
    print(f"  SCIF MESH CONVERGENCE  (per-layer T-A, I = {I_STUDY:.0f} A, "
          f"tol = {params.ta_picard_tol:.0e})")
    print(f"{'='*70}")
    print(f"  {'nz':>3} {'A-dofs':>8} {'coil cells':>11} "
          f"{'shanghai [mT]':>14} {'strong [mT]':>12}")
    for r in rows:
        print(f"  {r['nz']:>3} {r['dofs']:>8} {r['coil_cells']:>11} "
              f"{r['shanghai']:>+14.2f} {r['strong']:>+12.2f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
