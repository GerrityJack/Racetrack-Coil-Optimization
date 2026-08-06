"""bootstrap_beanseed_check.py -- 2026-08-06, prototype of the analytic
Bean/critical-state seed idea discussed alongside n-continuation.

Idea: replace the cold T=0 start (`ta_transient_seed_cold`, unmodified,
still used here for the A-field seed and Js/rho bootstrapping) with an
INITIAL T(z) profile per layer that already looks roughly like the
expected self-consistent critical-state shape -- current concentrated
near the tape edges, increasingly so as the local operating current
approaches the local critical current -- rather than the flat/uniform
implicit start T=0 gives. The hope: fewer Picard iterations are needed
to reach the true nonlinear fixed point, which (per the bootstrap
saturation trace) should mean less exposure to the chaotic amplification
regime before handoff.

IMPORTANT HONESTY NOTE: this is a HEURISTIC, self-derived, BC-anchored
piecewise-linear approximation to Bean critical-state behaviour -- NOT a
transcription of the published Norris (1970) closed-form self-field
strip solution. That exact closed form was deliberately NOT used here:
reconstructing it correctly from memory carries real risk of a sign/
exponent transcription error (a concern raised explicitly while working
out this prototype -- an early attempt to recall the Norris arctan
formula was caught confusing it with the DIFFERENT, antisymmetric
external-field screening problem). The profile used here is instead
built to GUARANTEE the correct T_bot/T_top boundary values by
construction (verified with an explicit assertion, not just asserted in
prose), and to capture only the qualitative behaviour everyone agrees
on: near-uniform gradient at low I/Ic, increasingly edge-concentrated
gradient as I/Ic -> 1. Treat this as "a smarter warm start than cold
T=0", not as validated Bean-model physics.

Construction, per layer i (half-width a = w/2, local coordinate
zeta = z - z_center in [-a, +a], T(-a) = T_bot_val = +T_amp,
T(+a) = T_top_val = -T_amp):
  I_frac  = clip(J_unif / Jc_layer, 0, 0.98)   -- J_unif = I/(delta_SC*w),
            Jc_layer = median Jc_vol over that layer's cells, reusing
            the value the EXISTING seed-time _update_rho call already
            computes (no new Ic/n-model call).
  f       = 1 - I_frac      -- core half-width fraction (unpenetrated)
  ratio   = 1 + 9*I_frac    -- edge/core |dT/dz| ratio (1 = uniform, at
                                I_frac=0; 10 = strongly edge-weighted, at
                                I_frac->1)
  G_core, G_edge solved so that integrating the piecewise-constant
  |dT/dz| = G_core in the core (|zeta|<f*a), G_edge in the two edge
  regions (f*a<|zeta|<a), with the correct overall sign, exactly
  reproduces T(-a)=+T_amp, T(+a)=-T_amp -- algebraically guaranteed by
  construction (checked below), not by trusting the derivation alone.

This does NOT modify `ta_transient_seed_cold`, `_picard_phase`, or
`_picard_bootstrap` -- it runs the exact same `ta_transient_seed_cold`
call, then OVERWRITES `T_i.x.array` for each layer's own cells with this
profile (in place of the all-zero cold start) before `_update_rho` and
`_picard_phase` run, completely unmodified from there.
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

CHECKPOINTS = (0, 2, 5, 10, 15, 20, 25)


def _bean_profile_1d(zeta, a, T_amp, I_frac):
    """Vectorized piecewise-linear T(zeta) for one layer. zeta: (N,) local
    z-coordinates in [-a, a]. Returns T values, sign convention matching
    T(-a)=+T_amp, T(+a)=-T_amp (see module docstring)."""
    import numpy as np
    I_frac = min(max(I_frac, 0.0), 0.98)
    f = 1.0 - I_frac
    ratio = 1.0 + 9.0 * I_frac
    denom = f + ratio * (1.0 - f)
    G_core = T_amp / (a * denom)
    G_edge = ratio * G_core

    z = zeta + a  # shift to [0, 2a], integrate from the bottom (z=0 -> T_amp)
    core_lo, core_hi = (1.0 - f) * a, (1.0 + f) * a  # core region in shifted coords

    T = np.empty_like(zeta)
    left = z <= core_lo
    core = (z > core_lo) & (z < core_hi)
    right = z >= core_hi

    T[left] = T_amp - G_edge * z[left]
    T_at_core_lo = T_amp - G_edge * core_lo
    T[core] = T_at_core_lo - G_core * (z[core] - core_lo)
    T_at_core_hi = T_at_core_lo - G_core * (core_hi - core_lo)
    T[right] = T_at_core_hi - G_edge * (z[right] - core_hi)
    return T


def seed_bean_profile(ta, params, I_now, Jc_per_coil_cell):
    """Overwrite each layer's T_i.x.array (only that layer's own cells'
    DOFs) with the Bean-like profile. Jc_per_coil_cell: array aligned
    with ta['coil_cells'] (i.e. the Jc_vol _update_rho already computed
    for the seed field)."""
    import numpy as np

    V_T = ta["V_T"]
    dof_coords = V_T.tabulate_dof_coordinates()
    delta_SC = ta["delta_SC"]
    w = float(params.w)
    a = w / 2.0
    T_amp = I_now / (2.0 * delta_SC)
    J_unif = I_now / (delta_SC * w)

    tdim = ta["V_T"].mesh.topology.dim
    from dolfinx import fem

    max_bc_err = 0.0
    for i in range(params.n_layers):
        idx_i = ta["layer_cell_idx"][i]          # rows into coil_cells
        cells_i = ta["coil_cells"][idx_i]
        Jc_layer = float(np.median(Jc_per_coil_cell[idx_i]))
        I_frac = J_unif / Jc_layer

        z_lo = float(params.layer_z_bottoms[i])
        z_hi = float(params.layer_z_tops[i])
        z_center = 0.5 * (z_lo + z_hi)

        dofs_layer = fem.locate_dofs_topological(V_T, tdim, cells_i)
        zeta = dof_coords[dofs_layer, 2] - z_center
        zeta = np.clip(zeta, -a, a)  # guard against off-center dof coords
        T_vals = _bean_profile_1d(zeta, a, T_amp, I_frac)

        ta["layer_T_fns"][i].x.array[dofs_layer] = T_vals
        ta["layer_T_fns"][i].x.scatter_forward()

        # correctness guard: the profile MUST reproduce the exact BC
        # values at the tape edges, by construction -- verify, don't
        # just assume the algebra above was transcribed correctly.
        bc_check = _bean_profile_1d(np.array([-a, a]), a, T_amp, I_frac)
        max_bc_err = max(max_bc_err, abs(bc_check[0] - T_amp),
                         abs(bc_check[1] + T_amp))

    assert max_bc_err < 1e-6 * abs(T_amp), (
        f"Bean seed profile failed its own BC self-check: "
        f"max_bc_err={max_bc_err:.3e}, T_amp={T_amp:.3e}")


def main():
    out_prefix = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bootstrap_bean"

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
    params.mesh_filename = f"{root}_bootbean_{os.getpid()}{ext}"
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
    Jmag0, Jc0, n_arr0 = ta_solve._update_rho(ta, J_seed, B_seed, ic, nm, eps)

    # -- the ONLY substantive change vs. bootstrap_saturation_check.py's
    # baseline: overwrite the cold T=0 start with the Bean-like profile,
    # using the Jc0 the seed _update_rho call just computed.
    seed_bean_profile(ta, params, I_now, Jc0)
    # rho_fn was computed from the OLD (all-zero-T-implied) J_seed; recompute
    # from the new T profile's own implied J so rho is self-consistent with
    # the seed we just installed, before the first real Picard iteration.
    J_from_bean = ta_solve._J_from_T(ta, domain)
    ta_solve._update_rho(ta, J_from_bean, B_seed, ic, nm, eps)

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
                  label="bootstrap-bean-check", verbose=False)
    dump_state(30)

    print(f"Dumped checkpoints {list(CHECKPOINTS) + [30]} to {out_prefix}_iter*.npz",
          flush=True)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
