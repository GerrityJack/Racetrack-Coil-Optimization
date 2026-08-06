"""bootstrap_saturation_check.py -- 2026-08-06, follow-up to Part 8 of
monolithic_diff_investigation_2026-08-05.md.

Part 8 found that by the time `_picard_bootstrap`'s 30 Picard iterations
finish, independent process launches of the IDENTICAL nominal
configuration (dt=60s, I=19.6A) have already diverged to O(1) relative
differences in the state fed to the monolithic system -- unlike every
prior bit-level diff in this project, which found a near-machine-epsilon
INPUT difference that only became large after a solve amplified it. Part
8 flagged, but did not do, a cheap follow-up: checkpoint the T/A state
DURING the bootstrap (not just before/after) to see how many iterations
it actually takes to saturate.

This does that using `_picard_phase`'s own existing `closure` extension
point (called once per iteration, before that iteration's T-solve) --
NOT by modifying `_picard_phase`/`_picard_bootstrap` themselves, which
CLAUDE.md flags as validated, do-not-touch code. The closure here is a
strict superset of `_picard_bootstrap`'s own `lambda: None` (still a
no-op for the actual dynamics), it only ADDS a state dump at chosen
checkpoints.
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

CHECKPOINTS = (0, 2, 5, 10, 15, 20, 25)  # state BEFORE this many iterations
                                          # have run this step; final (30)
                                          # state is dumped after the loop


def main():
    out_prefix = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bootstrap_sat"

    import numpy as np
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    from ic_model import IcModel, NValueModel
    from newton_ta import ta_transient_seed_cold
    from ta_transient import _picard_phase

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_bootsat_{os.getpid()}{ext}"
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    dt, I_now = 60.0, 19.6
    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)
    eps = float(getattr(params, "ta_eps_reg", 1.0))
    ta["dt_const"].value = dt
    T_amp = I_now / (2.0 * ta["delta_SC"])
    ta["T_bot_val"].value = +T_amp
    ta["T_top_val"].value = -T_amp
    J_seed = ta_transient_seed_cold(ta, uniform, ic, nm, I_now)
    ta["B_fn"].interpolate(ta["curl_expr"])
    B_seed = ta["B_fn"].x.array.reshape(-1, 3)[ta["coil_cells"]]
    ta["_rho_prev"] = None
    ta_solve._update_rho(ta, J_seed, B_seed, ic, nm, eps)

    def dump_state(tag):
        T_stack = np.concatenate([T_i.x.array.copy()
                                  for T_i in ta["layer_T_fns"]])
        A_arr = ta["A_h"].x.array.copy()
        np.savez(f"{out_prefix}_iter{tag}.npz", T=T_stack, A=A_arr)

    _counter = {"k": 0}

    def checkpoint_closure():
        k = _counter["k"]
        if k in CHECKPOINTS:
            dump_state(k)
        _counter["k"] += 1

    J_coil = ta["t_hat_coil"] * (I_now / (ta["delta_SC"] * params.w))
    _picard_phase(ta, domain, ic, nm, I_now, dt, J_coil=J_coil,
                  closure=checkpoint_closure, max_iters=30, min_iters=30,
                  scif_tol=float(getattr(params, "ta_scif_stall_mT", 0.05)),
                  label="bootstrap-sat-check", verbose=False)
    dump_state(30)  # final state, after all 30 iterations

    print(f"Dumped checkpoints {list(CHECKPOINTS) + [30]} to {out_prefix}_iter*.npz",
          flush=True)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
