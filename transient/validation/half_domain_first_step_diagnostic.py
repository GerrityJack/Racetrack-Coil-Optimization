"""
half_domain_first_step_diagnostic.py -- the half-domain analogue of
transient/validation/first_step_diagnostic.py, for the gauge-pollution /
short-dt-convergence hypothesis test.

Identical physics test as the eighth-domain original (one Picard step from
ZFC to an arbitrary (dt, I_target) pair, insulated limit, per_layer=True,
per_turn_bc=False -- no NI closure at all), on the HALF-domain mesh
(mesh/build_mesh_half.py) instead. Each invocation builds its OWN mesh file
in its OWN process, exactly like the original, so repeats are genuinely
independent OS process launches.

Usage: <env python> half_domain_first_step_diagnostic.py <dt_seconds> <I_target_amps> [coarsen]

coarsen (optional, default 1.0): multiplies params.mesh_size_min/max and
mesh_dist_min/max to shrink the half-domain's ~3.5x cell-count penalty for
this research spike. 1.0 = production resolution (slow); e.g. 1.8 roughly
halves the cell count.
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRANS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_TRANS)
for _p in (_TRANS, _ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve"),
           os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    dt = float(sys.argv[1])
    I_target = float(sys.argv[2])
    coarsen = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh_half
    import solve as base_solve
    import ta_solve
    from ta_transient import _picard_phase, _seed_cold

    comm = MPI.COMM_WORLD
    mesh_path = os.path.join(
        params.MESH_DIR,
        f"racetrack_mesh_half_fsdiag_{int(dt)}_{int(I_target*10)}_{os.getpid()}.msh")
    print(f"[HALF dt={dt}s I={I_target}A coarsen={coarsen}] "
         f"building half-domain mesh (own process) ...", flush=True)
    build_mesh_half.build_half(
        write_path=mesh_path, verbose=False,
        mesh_size_min=params.mesh_size_min * coarsen,
        mesh_size_max=params.mesh_size_max * coarsen,
        mesh_dist_min=params.mesh_dist_min * coarsen,
        mesh_dist_max=params.mesh_dist_max * coarsen)
    md = gmshio.read_from_msh(mesh_path, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    from ic_model import IcModel, NValueModel
    ic = IcModel(params.shanghai_csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)

    delta_SC = ta["delta_SC"]
    eps = float(getattr(params, "ta_eps_reg", 1.0))
    B_h = ta["B_fn"]
    coil = ta["coil_cells"]

    print(f"[HALF] mesh: {len(coil)} coil cells, "
         f"{domain.topology.index_map(domain.topology.dim).size_local} total",
         flush=True)

    J_coil = _seed_cold(ta, uniform, max(I_target, 1e-6))
    B_h.interpolate(ta["curl_expr"])
    B_seed = B_h.x.array.reshape(-1, 3)[coil]
    ta["_rho_prev"] = None
    ta_solve._update_rho(ta, J_coil, B_seed, ic, nm, eps)

    T_amp = I_target / (2.0 * delta_SC)
    ta["T_bot_val"].value = +T_amp
    ta["T_top_val"].value = -T_amp
    ta["dt_const"].value = float(dt)

    print("\n" + "=" * 78)
    print(f"HALF-DOMAIN FIRST STEP FROM ZFC: dt={dt}s  I_target={I_target}A  "
          f"(1/dt forcing coeff = {1.0/dt:.4f})")
    print("=" * 78)

    J_coil, n_iters, converged = _picard_phase(
        ta, domain, ic, nm, I_target, dt, J_coil,
        closure=lambda: None, max_iters=150, min_iters=6, scif_tol=0.5,
        label=f"HALF dt={dt}s I={I_target}A", verbose=True)

    finite = (np.all(np.isfinite(ta["A_h"].x.array))
              and all(np.all(np.isfinite(T_i.x.array))
                     for T_i in ta["layer_T_fns"]))

    print("\n" + "=" * 78)
    print(f"RESULT HALF dt={dt}s I_target={I_target}A: "
         f"converged={converged}  n_iters={n_iters}  finite={finite}")
    print("=" * 78)

    try:
        os.remove(mesh_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
