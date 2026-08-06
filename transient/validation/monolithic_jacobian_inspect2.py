"""
monolithic_jacobian_inspect2.py -- 2026-08-05 overnight follow-up, part 2.

Follow-up to monolithic_jacobian_inspect.py's finding that EVERY T-A
off-diagonal coupling block is exactly zero, and every T-block diagonal
entry is exactly 1.0. Distinguishes "free" (non-Dirichlet) dofs from
Dirichlet-pinned ones explicitly, to check whether the free dofs ALSO
show the suspicious identity pattern (a real bug) or whether the
aggregate check in part 1 was just swamped by the Dirichlet-majority
dofs (in which case the free dofs' own diagonal/coupling needs a
separate, targeted look).
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
    import numpy as np
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio
    from dolfinx import fem

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    import monolithic_ta
    from ic_model import IcModel, NValueModel
    from newton_ta import ta_transient_seed_cold, _picard_bootstrap

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_jacinspect2_{os.getpid()}{ext}"
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

    dt, I_now = 60.0, 19.6
    ta = ta_solve.setup_ta_problem(domain, cell_tags, facet_tags, uniform,
                                   per_layer=True, per_turn_bc=False)
    monolithic_ta.build_monolithic_problem(ta, verbose=False)

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
    _picard_bootstrap(ta, domain, ic, nm, I_now, dt, n_iters=30, verbose=False)
    ta["B_fn"].interpolate(ta["curl_expr"])
    B_now = ta["B_fn"].x.array.reshape(-1, 3)[ta["coil_cells"]]
    for layer in range(6):
        monolithic_ta._update_mono_coefficients(ta, ic, nm, layer, B_now, relax=1.0)

    problem = ta["mono_problem"]
    snes = ta["mono_snes"]
    # ONE solve to reach a state comparable to part 1's inspection point
    # (matches "n_outer_before_inspect=3" there approximately -- 3 raw
    # solves without the relax blending, close enough for this check,
    # which only cares about WHICH entries are structurally zero, not the
    # precise numeric trajectory).
    for _ in range(3):
        problem.solve()
    print(f"||F|| after 3 raw solves = {snes.getFunction()[0].norm():.4e}",
          flush=True)

    x = snes.getSolution()
    Fvec = snes.getFunction()[0]
    Jmat, Pmat = snes.getJacobian()[0:2]
    snes.computeFunction(x, Fvec)
    snes.computeJacobian(x, Jmat, Pmat)

    n_T = ta["V_T"].dofmap.index_map.size_local
    n_A = ta["V_A"].dofmap.index_map.size_local

    ai, aj, av = Jmat.getValuesCSR()
    from scipy.sparse import csr_matrix
    J_sp = csr_matrix((av, aj, ai))

    # Identify FREE (non-Dirichlet) dofs for layer 0 directly from the BC
    # list ta_solve.py built, rather than inferring from the matrix.
    bcs_layer0 = ta["layer_bcs"][0]
    pinned_dofs = set()
    for bc in bcs_layer0:
        try:
            dofs = bc._cpp_object.dof_indices()[0]
        except Exception:
            # dolfinx API variant fallback
            dofs = bc.dof_indices()[0] if hasattr(bc, "dof_indices") else []
        pinned_dofs.update(int(d) for d in dofs)
    all_dofs = set(range(n_T))
    free_dofs = sorted(all_dofs - pinned_dofs)
    print(f"\nLayer 0: {len(pinned_dofs)} pinned dofs, {len(free_dofs)} free dofs "
          f"out of {n_T} total")

    if free_dofs:
        free_arr = np.array(free_dofs)
        diag_T0 = J_sp.diagonal()[0:n_T]
        free_diag = diag_T0[free_arr]
        print(f"Layer-0 FREE-dof diagonal: min={free_diag.min():.6e}  "
              f"mean={free_diag.mean():.6e}  max={free_diag.max():.6e}  "
              f"n_exactly_1.0={np.sum(free_diag == 1.0)}/{len(free_diag)}")

        # Check off-diagonal T0-A coupling rows/cols RESTRICTED to free dofs
        # only (part 1's check used the WHOLE block including pinned rows,
        # which are correctly zero for a Dirichlet row -- that could have
        # hidden genuine nonzero coupling on the free rows specifically).
        A_lo = 6 * n_T
        A_hi = A_lo + n_A
        block_AT0_free_rows = J_sp[A_lo:A_hi, free_arr]
        block_T0A_free_rows = J_sp[free_arr, A_lo:A_hi]
        print(f"\ndF_A/dT_0 restricted to layer-0's FREE columns only: "
              f"nnz={block_AT0_free_rows.nnz}  "
              f"max|.|={np.abs(block_AT0_free_rows.data).max() if block_AT0_free_rows.nnz else 0.0:.3e}")
        print(f"dF_T0/dA restricted to layer-0's FREE rows only: "
              f"nnz={block_T0A_free_rows.nnz}  "
              f"max|.|={np.abs(block_T0A_free_rows.data).max() if block_T0A_free_rows.nnz else 0.0:.3e}")

        # Also print a handful of actual free-dof rows in full (T0 self-block
        # only) to see whether they contain the expected rho*J physics
        # (multiple nonzero entries spread over neighboring dofs) or are
        # ALSO trivial/identity-like.
        print("\nSample free-dof rows (T0 self-block, first 3 free dofs):")
        for fd in free_arr[:3]:
            row = J_sp[fd, 0:n_T]
            print(f"  dof {fd}: nnz={row.nnz}  "
                  f"entries(min,max)=({row.data.min():.3e},{row.data.max():.3e})"
                  if row.nnz else f"  dof {fd}: EMPTY ROW")
    else:
        print("Could not identify any free dofs via the BC list -- dof_indices() "
              "API call likely failed; inspect ta['layer_bcs'][0] structure "
              "manually if this happens.")

    # Sanity check: also verify the SAME state's A-block diagonal isn't
    # ALSO suspiciously trivial, and directly check whether ta["A_h"] and
    # ta["layer_T_fns"] actually hold DIFFERENT (non-BC-default) values at
    # this point, i.e. that real nonlinear iteration has actually happened
    # and this isn't somehow inspecting a pristine unsolved state.
    print(f"\nT_0 array stats: min={ta['layer_T_fns'][0].x.array.min():.3e} "
          f"max={ta['layer_T_fns'][0].x.array.max():.3e}")
    print(f"A_h array stats: min={ta['A_h'].x.array.min():.3e} "
          f"max={ta['A_h'].x.array.max():.3e}")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
