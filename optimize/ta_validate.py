"""
ta_validate.py — standalone true-box-uniformity validator (2026-07-26)
=======================================================================
The coarse CMA-ES screen's uniformity_pct (Bean-state on-axis proxy) was
found unreliable by up to ~10x for compact designs (see CLAUDE.md's
"Coarse-screen SCIF proxy found unreliable" / "Box uniformity is the real
target" sections) -- the only trustworthy check is a full per-layer T-A
solve with the box-uniformity extension already built into
solve/ta_solve.py's __main__ (box_ptp_pct). That extension is only wired
into the CLI script's own params.py-driven run; this module factors the
same sequence (mesh build -> uniform-J setup -> T-A setup -> Picard solve
-> box grid) into a standalone, subprocess-friendly call so an orchestrator
can validate an arbitrary candidate WITHOUT ever touching params.py on
disk -- it mutates the in-process `params` module only, exactly like
cmaes_search.py's parallel workers already do. Run each candidate in its
OWN subprocess (fresh Python interpreter) so there is no chance of state
bleeding between candidates.

Also runs the SAME candidate TWICE inside one call (two independent mesh
builds — see MESH_REPEATS) because gmsh mesh generation is NOT perfectly
reproducible across separate builds within a process either when the
mesh_filename differs each time, and the n_layers=4 episode showed this
SCIF/box calculation can be a near-cancelling sum that amplifies normal
mesh variation by >20x. Reports both repeats so the caller can judge
whether the design is comfortably inside/outside the target or borderline.

Input: env var TA_VALIDATE_JSON, a JSON object:
    {"label": "6L champion", "a": 0.0222, "b": 0.0273,
     "coil_half_gap": 0.0135, "n_turns": [285,285,379,379,2,2],
     "I_design": 224.29}
(I_design is optional -- if omitted, params.py's default is used, which
is wrong for a candidate with a different quench-limited I_op, so callers
should always pass it explicitly.)

Output: one line per repeat to stdout, machine-parseable:
    TA_VALIDATE_RESULT label=<label> repeat=<i> box_ptp_pct=<x> \
        onaxis_scif_pct=<x> Bz_bore_uniform=<x> Bz_bore_TA=<x> \
        converged=<bool> n_iters=<n>

Run (per candidate, fresh process):
    TA_VALIDATE_JSON='{...}' \
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \
        optimize/ta_validate.py
"""
import os, sys, json, time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "mesh"),
           os.path.join(_ROOT, "solve"), os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import params
import opt_config as cfg

MESH_REPEATS = 2


def _apply(design):
    params.a = float(design["a"])
    params.b = float(design["b"])
    params.coil_half_gap = float(design["coil_half_gap"])
    params.n_turns = list(design["n_turns"])
    if "I_design" in design and design["I_design"] is not None:
        params.I_design = float(design["I_design"])
    # Deliberately do NOT apply cfg.SCREEN_MESH_OVERRIDES here -- this is
    # the full-fidelity T-A validation, not the coarse screen; keep
    # params.py's production mesh_nz_per_layer / mesh_z_grading.
    #
    # 2026-07-31: optional "mesh" dict sets arbitrary params mesh
    # attributes (mesh_z_grading, mesh_nz_per_layer, mesh_size_min_factor,
    # ...) BEFORE recompute_derived(), so a mesh-convergence study can vary
    # resolution on a FIXED design. Absent (the normal case) nothing
    # changes and production settings are used exactly as before.
    for k, v in (design.get("mesh") or {}).items():
        if not hasattr(params, k):
            raise KeyError(f"unknown params attribute in 'mesh' override: {k}")
        setattr(params, k, v)
    params.recompute_derived()


def validate_once(design, repeat_idx, comm):
    from dolfinx.io import gmsh as gmshio
    import solve as base_solve
    import build_mesh
    import ta_solve
    from ic_model import IcModel, NValueModel
    from coil2_field import compute_both_coils_field_multilayer

    _apply(design)
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_val{os.getpid()}_{repeat_idx}{ext}"

    build_mesh.build(write_path=params.mesh_filename)
    mesh_data = gmshio.read_from_msh(params.mesh_filename, comm, rank=0,
                                      gdim=3)
    domain = mesh_data.mesh

    ic_mod = IcModel(csv_path=params.shanghai_csv_filename)
    n_mod = NValueModel(csv_path=params.n_value_csv_filename)

    uniform_setup = base_solve.setup_problem(domain, mesh_data.cell_tags,
                                              mesh_data.facet_tags)
    ta = ta_solve.setup_ta_problem(domain, mesh_data.cell_tags,
                                    mesh_data.facet_tags, uniform_setup)

    t0 = time.time()
    A_h, B_h, T_h, info = ta_solve.solve_ta_at_current(
        domain, ta, uniform_setup, I_amps=params.I_design,
        ic_model=ic_mod, n_model=n_mod, verbose=False, warm_start=False)
    dt = time.time() - t0

    import dolfinx.mesh as dmesh
    coil_cells = ta["coil_cells"]
    centroids = dmesh.compute_midpoints(domain, domain.topology.dim,
                                        coil_cells)

    delta_SC = float(params.delta_SC)
    w_tape = float(params.w)
    J_TA_arr = ta_solve._J_from_T(ta, domain)
    J_unif_mag = params.I_design / (delta_SC * w_tape)
    J_unif_arr = ta["t_hat_coil"] * J_unif_mag
    dJ_arr = J_TA_arr - J_unif_arr
    Lambda = float(params.t)
    dJ_s = dJ_arr * (delta_SC / Lambda)
    dV = ta["coil_vols"]

    dB_bore = ta_solve.dB_bore_from_dJ(centroids, dJ_s, dV)
    bore_pt = np.array([[0.0, 0.0, float(params.coil_half_gap)]])
    B_bore_uniform = compute_both_coils_field_multilayer(
        bore_pt, I_per_turn=params.I_design)
    Bz_bore_uniform = float(B_bore_uniform[0, 2])
    Bz_bore_TA = Bz_bore_uniform + float(dB_bore[2])
    onaxis_scif_pct = abs(dB_bore[2]) / abs(Bz_bore_uniform) * 100.0

    g_half = float(params.coil_half_gap)
    xs = np.linspace(-cfg.TARGET_X_M / 2, cfg.TARGET_X_M / 2, cfg.TARGET_NX)
    ys = np.linspace(-cfg.TARGET_Y_M / 2, cfg.TARGET_Y_M / 2, cfg.TARGET_NY)
    Xg, Yg = np.meshgrid(xs, ys)
    box_pts = np.column_stack([Xg.ravel(), Yg.ravel(),
                               np.full(Xg.size, g_half)])
    B_box_uniform = compute_both_coils_field_multilayer(
        box_pts, I_per_turn=params.I_design)
    dB_box = np.array([ta_solve.dB_bore_from_dJ(centroids, dJ_s, dV,
                                                bore_pt=p)
                       for p in box_pts])
    Bmag_box = np.linalg.norm(B_box_uniform + dB_box, axis=1)
    box_ptp_pct = float((Bmag_box.max() - Bmag_box.min())
                       / Bmag_box.mean() * 100.0)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass

    n_coil_cells = int(len(coil_cells))
    try:
        n_dofs = int(uniform_setup["V"].dofmap.index_map.size_global
                     * uniform_setup["V"].dofmap.index_map_bs)
    except Exception:
        n_dofs = -1

    return dict(box_ptp_pct=box_ptp_pct, onaxis_scif_pct=onaxis_scif_pct,
               Bz_bore_uniform=Bz_bore_uniform, Bz_bore_TA=Bz_bore_TA,
               converged=bool(info["converged"]), n_iters=int(info["n_iters"]),
               solve_s=dt, n_coil_cells=n_coil_cells, n_dofs=n_dofs)


def main():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    raw = os.environ.get("TA_VALIDATE_JSON")
    if not raw:
        print("TA_VALIDATE_JSON env var not set", file=sys.stderr)
        sys.exit(1)
    design = json.loads(raw)
    label = design.get("label", "candidate")
    n_repeats = int(os.environ.get("TA_VALIDATE_REPEATS", MESH_REPEATS))

    for i in range(n_repeats):
        r = validate_once(design, i, comm)
        if comm.rank == 0:
            print(f"TA_VALIDATE_RESULT label={label!r} repeat={i} "
                  f"box_ptp_pct={r['box_ptp_pct']:.4f} "
                  f"onaxis_scif_pct={r['onaxis_scif_pct']:.4f} "
                  f"Bz_bore_uniform={r['Bz_bore_uniform']:.5f} "
                  f"Bz_bore_TA={r['Bz_bore_TA']:.5f} "
                  f"converged={r['converged']} n_iters={r['n_iters']} "
                  f"solve_s={r['solve_s']:.1f} "
                  f"n_coil_cells={r['n_coil_cells']} n_dofs={r['n_dofs']}",
                  flush=True)


if __name__ == "__main__":
    main()
