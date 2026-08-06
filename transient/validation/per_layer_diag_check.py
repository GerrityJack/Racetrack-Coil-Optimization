"""per_layer_diag_check.py -- 2026-08-06, targeted follow-up: the global
T_min/T_amp diagnostic used throughout the alpha-fix validation showed a
large, still-drifting-at-final-iteration negative value (-86x T_amp) that
T_max (~+1x) never revealed. Breaks T_min/T_max down PER LAYER (6 layers,
uneven turn counts 382/382/478/478/3/3) to find which layer(s) this comes
from, instead of the pooled-across-all-layers number used so far.
"""
import os
import sys

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
    dt = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    I_target = float(sys.argv[2]) if len(sys.argv) > 2 else 19.6
    alpha_high = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03
    alpha_low = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
    max_iters = int(sys.argv[5]) if len(sys.argv) > 5 else 500

    import numpy as np
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    from ta_transient import _picard_phase, _seed_cold

    params.ta_picard_alpha = alpha_high
    params.ta_picard_alpha_fine = alpha_low

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_perlayer_{os.getpid()}{ext}"
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    from ic_model import IcModel, NValueModel
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)
    delta_SC = ta["delta_SC"]
    eps = float(getattr(params, "ta_eps_reg", 1.0))
    T_amp = I_target / (2.0 * delta_SC)

    print(f"n_turns per layer: {params.n_turns}", flush=True)

    J_coil = _seed_cold(ta, uniform, max(I_target, 1e-6))
    ta["B_fn"].interpolate(ta["curl_expr"])
    B_seed = ta["B_fn"].x.array.reshape(-1, 3)[ta["coil_cells"]]
    ta["_rho_prev"] = None
    ta_solve._update_rho(ta, J_coil, B_seed, ic, nm, eps)

    ta["T_bot_val"].value = +T_amp
    ta["T_top_val"].value = -T_amp
    ta["dt_const"].value = float(dt)


    # per-layer diagnostic snapshots at a handful of iteration counts, via
    # the closure hook, so we can see the per-layer trend, not just the
    # final snapshot. Dof sets precomputed ONCE (not inside the closure).
    import dolfinx.fem as fem
    layer_dofs = []
    for i in range(params.n_layers):
        idx = ta["layer_cell_idx"][i]
        cells_i = ta["coil_cells"][idx]
        layer_dofs.append(fem.locate_dofs_topological(
            ta["V_T"], domain.topology.dim, cells_i))

    checkpoints = {max(1, int(max_iters * f)) for f in
                  (0.1, 0.25, 0.5, 0.75, 1.0)} | {max_iters - 1}
    # dense early-iteration checkpoints -- the official production path
    # (ta_n_picard=150, ta_scif_stall_mT=0.05) stops somewhere in this
    # window; need fine resolution here specifically to check whether its
    # historical 641.26mT reading was itself a mid-transient snapshot.
    checkpoints |= {n for n in
                    (5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
                     110, 120, 130, 140, 149, 150, 160, 175, 200, 250)
                    if n < max_iters}
    _counter = {"k": 0}
    J_unif = ta["t_hat_coil"] * (I_target / (delta_SC * params.w))

    def diag_closure():
        k = _counter["k"]
        if k in checkpoints:
            J_now = ta_solve._J_from_T(ta, domain)
            dJs = (J_now - J_unif) * (delta_SC / ta["Lambda"])
            scif = float(ta_solve.dB_bore_from_dJ(
                ta["coil_centroids"], dJs, ta["coil_vols"])[2] * 1e3)
            print(f"  --- iter {k}  SCIF={scif:+9.3f}mT  per-layer T/T_amp ---",
                  flush=True)
            for i, T_i in enumerate(ta["layer_T_fns"]):
                vals = T_i.x.array[layer_dofs[i]] / T_amp
                print(f"    layer{i} (n_turns={params.n_turns[i]}): "
                      f"min={vals.min():+9.3f}  max={vals.max():+9.3f}  "
                      f"mean={vals.mean():+9.3f}", flush=True)
        _counter["k"] += 1

    J_coil = ta["t_hat_coil"] * (I_target / (delta_SC * params.w))
    J_coil, n_iters, converged = _picard_phase(
        ta, domain, ic, nm, I_target, dt, J_coil,
        closure=diag_closure, max_iters=max_iters, min_iters=max_iters, scif_tol=0.5,
        label="perlayer", verbose=False)

    print(f"FINAL: converged={converged} n_iters={n_iters}", flush=True)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
