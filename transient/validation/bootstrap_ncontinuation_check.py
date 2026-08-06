"""bootstrap_ncontinuation_check.py -- 2026-08-06, prototype of the
n-value continuation (homotopy) fix discussed for the Picard-bootstrap
divergence found in bootstrap_saturation_check.py / Part 8 of
monolithic_diff_investigation_2026-08-05.md.

Idea: the E-J power law's steepness (n(B,theta) ~ 13-34 physically) is
WHY the T-equation's linear sub-problem is so ill-conditioned near
j/jc=1 -- ill-conditioned enough that a ~1e-14 floating-point difference
between independent process launches becomes a ~1e-3 relative state
difference after just 1-2 Picard iterations, then compounds to full O(1)
decorrelation by iteration ~20-25. n-continuation (standard practice in
the H-formulation/T-A superconductor modelling literature) starts the
Picard bootstrap at a mild, well-conditioned exponent and ramps UP to
the true physical n(B,theta) over the first several iterations, each
stage warm-started from the previous -- so the solver never has to face
the sharp, near-singular transition from a cold, far-away guess.

Implementation note: does NOT modify `_picard_phase`/`_picard_bootstrap`
(CLAUDE.md flags these as validated, do-not-touch). Instead wraps the
real NValueModel in `ContinuationNModel`, which exposes the exact same
`.n_value(B_mag, theta)` interface `_update_rho` already calls, and
blends its output toward n_start using a `frac` attribute that this
script's own closure (the SAME per-iteration extension point
bootstrap_saturation_check.py used for checkpointing) advances each
iteration. `_picard_phase` itself is called completely unmodified with
this wrapper passed in place of the real n_model.
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

CHECKPOINTS = (0, 2, 5, 10, 15, 20, 25)  # same as bootstrap_saturation_check.py,
                                          # for a directly-comparable curve


class ContinuationNModel:
    """Wraps a real NValueModel; blends its n(B,theta) output toward
    n_start by `frac` (0 = fully n_start, 1 = fully physical). `frac` is
    mutated externally, once per Picard iteration, by the caller's
    closure -- this class does no scheduling itself."""

    def __init__(self, real_model, n_start):
        self.real_model = real_model
        self.n_start = float(n_start)
        self.frac = 0.0

    def n_value(self, B_mag, theta):
        n_arr, meta = self.real_model.n_value(B_mag, theta)
        n_eff = self.n_start + self.frac * (n_arr - self.n_start)
        return n_eff, meta


def main():
    out_prefix = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bootstrap_ncont"
    ramp_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 10  # 0 = no continuation
    n_start = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0

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
    params.mesh_filename = f"{root}_bootncont_{os.getpid()}{ext}"
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm_real = NValueModel(params.n_value_csv_filename)
    nm = ContinuationNModel(nm_real, n_start=n_start)

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
    nm.frac = 0.0  # iteration-0 coefficients also start at n_start, not physical n
    ta_solve._update_rho(ta, J_seed, B_seed, ic, nm, eps)

    def dump_state(tag):
        T_stack = np.concatenate([T_i.x.array.copy()
                                  for T_i in ta["layer_T_fns"]])
        A_arr = ta["A_h"].x.array.copy()
        np.savez(f"{out_prefix}_iter{tag}.npz", T=T_stack, A=A_arr)

    _counter = {"k": 0}

    def continuation_closure():
        k = _counter["k"]
        if ramp_iters > 0:
            nm.frac = min(1.0, k / float(ramp_iters))
        else:
            nm.frac = 1.0
        if k in CHECKPOINTS:
            dump_state(k)
        _counter["k"] += 1

    J_coil = ta["t_hat_coil"] * (I_now / (ta["delta_SC"] * params.w))
    _picard_phase(ta, domain, ic, nm, I_now, dt, J_coil=J_coil,
                  closure=continuation_closure, max_iters=30, min_iters=30,
                  scif_tol=float(getattr(params, "ta_scif_stall_mT", 0.05)),
                  label="bootstrap-ncont-check", verbose=False)
    dump_state(30)

    print(f"Dumped checkpoints {list(CHECKPOINTS) + [30]} to {out_prefix}_iter*.npz "
          f"(ramp_iters={ramp_iters}, n_start={n_start})", flush=True)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
