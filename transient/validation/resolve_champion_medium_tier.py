"""
resolve_champion_medium_tier.py -- 2026-09-08

The champion's standing solve/racetrack_ta_fields.npz turns out to be at
the XDENSE mesh tier (108199 raw 1/8-domain cells, box_ptp_pct=0.4253%
matching CLAUDE.md's documented xdense investigation) -- the finest tier
ever tested in this project, not a coarse one. There is no finer tier to
re-check it against.

The informative direction instead: does the champion's already-known
severe quench picture (28.6% of cells < margin 1.0, worst 0.586 at
xdense) get UNDERSTATED at the coarser medium tier -- the same pattern
just found on the low-current design (medium made it look almost fine;
fine tier revealed real, spatially coherent quenched clusters at the
straight/cap corners). This re-solves the champion at I_design=196A on
the medium tier, otherwise following solve/ta_solve.py's own main()
pipeline exactly (same physics, only mesh resolution differs).

Mesh tier forced IN-MEMORY only -- never written back to params.py on
disk. Champion geometry (a/b/n_turns/coil_half_gap) is params.py's own
live default, untouched here.

Output: solve/racetrack_ta_fields_medium.npz (does NOT overwrite the
standing xdense racetrack_ta_fields.npz).

Run:  <env>/bin/python3 transient/validation/resolve_champion_medium_tier.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRANS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_TRANS)
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "mesh"),
           os.path.join(_ROOT, "solve"), os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from mpi4py import MPI

import params

# Force medium tier in-memory (params.py's committed default is xdense --
# see CLAUDE.md 2026-09-03/04). Distinct mesh_filename so this doesn't
# clobber anything the standing xdense npz's mesh may still reference.
params.mesh_size_min_factor = 0.25
params.mesh_size_max_factor = 0.60
params.mesh_dist_min_factor = 1.50
params.mesh_dist_max_factor = 2.00
params.box_scale = 4.0
params.mesh_z_grading = [0.075, 0.15, 0.55, 0.15, 0.075]
_root, _ext = os.path.splitext(params.mesh_filename)
params.mesh_filename = f"{_root}_mediumcheck{_ext}"

from dolfinx.io import gmsh as gmshio
import build_mesh
import solve as base_solve
from ta_solve import (setup_ta_problem, solve_ta_at_current, _J_from_T,
                      dB_bore_from_dJ)
import dolfinx.mesh as dmesh
from ic_model import IcModel, NValueModel
from coil2_field import compute_both_coils_field_multilayer

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    print(f"Mesh tier: medium (forced in-memory)  mesh_filename={params.mesh_filename}")

    build_mesh.build(write_path=params.mesh_filename)
    mesh_data = gmshio.read_from_msh(params.mesh_filename, comm, rank=0, gdim=3)
    domain, cell_tags, facet_tags = (mesh_data.mesh, mesh_data.cell_tags,
                                     mesh_data.facet_tags)

    ic_mod = IcModel(csv_path=params.shanghai_csv_filename)
    n_mod = NValueModel(csv_path=params.n_value_csv_filename)

    uniform_setup = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ta = setup_ta_problem(domain, cell_tags, facet_tags, uniform_setup)

    A_h, B_h, T_h, info = solve_ta_at_current(
        domain, ta, uniform_setup, I_amps=params.I_design,
        ic_model=ic_mod, n_model=n_mod, verbose=True)

    coil_cells = ta["coil_cells"]
    centroids = dmesh.compute_midpoints(domain, domain.topology.dim, coil_cells)
    B_coil = B_h.x.array.reshape(-1, 3)[coil_cells]
    print(f"raw (1/8-domain) cell count: {len(centroids)}")

    delta_SC = float(params.delta_SC)
    w_tape = float(params.w)
    J_TA_arr = _J_from_T(ta, domain)
    J_unif_mag = params.I_design / (delta_SC * w_tape)
    J_unif_arr = ta["t_hat_coil"] * J_unif_mag
    dJ_arr = J_TA_arr - J_unif_arr
    Lambda = float(params.t)
    dJ_s = dJ_arr * (delta_SC / Lambda)
    dV = ta["coil_vols"]

    dB_bore = dB_bore_from_dJ(centroids, dJ_s, dV)
    bore_pt = np.array([[0.0, 0.0, float(params.coil_half_gap)]])
    B_bore_uniform = compute_both_coils_field_multilayer(
        bore_pt, I_per_turn=params.I_design)
    Bz_bore_uniform = float(B_bore_uniform[0, 2])
    Bz_bore_TA = Bz_bore_uniform + float(dB_bore[2])

    import opt_config as cfg
    g_half = float(params.coil_half_gap)
    xs = np.linspace(-cfg.TARGET_X_M / 2, cfg.TARGET_X_M / 2, cfg.TARGET_NX)
    ys = np.linspace(-cfg.TARGET_Y_M / 2, cfg.TARGET_Y_M / 2, cfg.TARGET_NY)
    Xg, Yg = np.meshgrid(xs, ys)
    box_pts = np.column_stack([Xg.ravel(), Yg.ravel(),
                               np.full(Xg.size, g_half)])
    B_box_uniform = compute_both_coils_field_multilayer(
        box_pts, I_per_turn=params.I_design)
    dB_box = np.array([dB_bore_from_dJ(centroids, dJ_s, dV, bore_pt=p)
                       for p in box_pts])
    Bmag_box = np.linalg.norm(B_box_uniform + dB_box, axis=1)
    box_ptp_pct = (Bmag_box.max() - Bmag_box.min()) / Bmag_box.mean() * 100.0

    out = os.path.join(params.SOLVE_DIR, "racetrack_ta_fields_medium.npz")
    np.savez(out, I_solved=params.I_design, ramp_duration=params.ramp_duration,
             delta_SC=delta_SC, coil_centroids=centroids, coil_B=B_coil,
             J_TA_coil=J_TA_arr, J_unif_coil=J_unif_arr, dB_bore=dB_bore,
             Bz_bore_uniform=Bz_bore_uniform, Bz_bore_TA=Bz_bore_TA,
             dV=dV, box_ptp_pct=box_ptp_pct)
    print(f"\nSaved {out}")
    print(f"box_ptp_pct={box_ptp_pct:.4f}%  "
         f"(xdense reference: 0.4253% -- champion's standing number)")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass
