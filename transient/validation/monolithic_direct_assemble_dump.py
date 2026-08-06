"""monolithic_direct_assemble_dump.py -- Part 7 continuation (2026-08-06).

Part 7 of monolithic_diff_investigation_2026-08-05.md tried to dump the
pre-solve (iteration-1) assembled system for the block-Jacobi field-split
config via `snes.computeFunction()`/`snes.computeJacobian()` and got
blocked: calling those before any real `problem.solve()` either
segfaults (kind="nest", after pc.setUp()) or raises PETSc error 73 ("Not
for unassembled matrix"). The file's own identified fix, not attempted
there: bypass SNES's introspection entirely and call dolfinx's
module-level `assemble_residual`/`assemble_jacobian` functions directly
-- these are the exact functions SNES itself calls internally via
`setFunction`/`setJacobian`, so calling them ourselves reproduces
precisely the "system that will be fed to the first Newton step" without
going through any SNES-internal lifecycle state.

Reuses `build_nest_additive_problem` from monolithic_fieldsplit_dump.py
unchanged (same confirmed-safe kind="nest" + early sub-solver
configuration construction) -- only the dump mechanism is new.
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
sys.path.insert(0, _HERE)

from monolithic_fieldsplit_dump import build_nest_additive_problem


def main():
    dump_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/monodirect_dump"

    import numpy as np
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio
    from dolfinx.fem.petsc import assemble_residual, assemble_jacobian, assign

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    from ic_model import IcModel, NValueModel, angle_with_normal_deg
    from newton_ta import ta_transient_seed_cold, _picard_bootstrap

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_monodirect_{os.getpid()}{ext}"
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
    _picard_bootstrap(ta, domain, ic, nm, I_now, dt, n_iters=30, verbose=False)

    fx = build_nest_additive_problem(ta, domain, verbose=False)

    def update_coeffs(relax):
        ta["B_fn"].interpolate(ta["curl_expr"])
        B_coil = ta["B_fn"].x.array.reshape(-1, 3)[ta["coil_cells"]]
        for layer in range(6):
            idx = fx["cell_idx_list"][layer]
            cells_layer = ta["coil_cells"][idx]
            n_hat_layer = ta["n_hat_coil"][idx]
            B_layer = B_coil[idx]
            Bmag = np.linalg.norm(B_layer, axis=1)
            theta = angle_with_normal_deg(B_layer, n_hat_layer)
            Ic_arr, _ = ic.critical_current(Bmag, theta)
            Jc_vol = Ic_arr / (ta["delta_SC"] * ic.tape_width)
            n_arr, _ = nm.n_value(Bmag, theta)
            fx["Jc_fns"][layer].x.array[cells_layer] = Jc_vol
            fx["n_fns"][layer].x.array[cells_layer] = n_arr

    update_coeffs(relax=1.0)
    problem = fx["problem"]
    bcs = [bc for bcs_i in fx["bcs_list"] for bc in bcs_i]
    bcs.append(ta["bc_A"])

    # ---- Dump the pre-solve (iteration-1) system via DIRECT assembly,
    # bypassing snes.computeFunction()/computeJacobian() entirely. These
    # are the same module-level functions SNES calls internally via
    # setFunction/setJacobian -- calling them ourselves sidesteps whatever
    # internal SNES lifecycle state made the direct-introspection route
    # crash in Part 7.
    x = problem.x
    u = problem.u
    assign(u, x)  # Function -> Vec: seed x from the current warm-started u

    b = problem.b
    Jmat = problem.A
    Pmat = problem.P_mat  # None here (no separate preconditioner form)

    assemble_residual(None, x, b, u=u, residual=problem.F, jacobian=problem.J,
                       bcs=bcs)
    assemble_jacobian(None, x, Jmat, Pmat, u=u, jacobian=problem.J,
                      preconditioner=problem.preconditioner, bcs=bcs)

    f0 = b.getArray(readonly=True).copy()
    # Mat.convert(..., out=None) converts IN PLACE (petsc4py semantics --
    # confirmed by reading Mat.convert's own docstring after this bit us
    # once: it silently mutated problem.A from nest to aij, which then
    # broke the subsequent real field-split solve with PETSc error 73).
    # Convert a COPY, not problem.A itself.
    Jmat_copy = Jmat.copy()
    Jmat_aij = Jmat_copy.convert("aij")
    ai, aj, av = Jmat_aij.getValuesCSR()
    Jmat_copy.destroy()
    np.savez(f"{dump_path}_input.npz", indptr=ai, indices=aj, data=av, rhs=f0)
    print(f"Dumped pre-solve system (direct assembly): ||F||={np.linalg.norm(f0):.6e}  "
          f"nnz={len(av)}", flush=True)

    # ---- Now run the real solve loop (same as monolithic_fieldsplit_dump.py)
    # to record whether THIS specific pre-solve system goes on to succeed
    # or fail, so the direct-assembly dump can be correlated with outcome.
    snes = fx["snes"]
    prev_fnorm = None
    outcome = "unknown"
    for k in range(3):
        update_coeffs(relax=0.3)
        problem.solve()
        reason = snes.getConvergedReason()
        fnorm = snes.getFunction()[0].norm()
        ksp_its = snes.getKSP().getIterationNumber()
        ratio = fnorm / prev_fnorm if prev_fnorm else float("nan")
        print(f"  k={k+1}  ||F||={fnorm:.6e}  ratio={ratio:.4f}  "
              f"reason={reason}  ksp_its={ksp_its}", flush=True)
        prev_fnorm = fnorm
        if k == 0:
            outcome = "success" if ksp_its > 0 else "failure"
        ok = (reason not in {-3, -4, -6, -7, -8, -9, -10, -11}
              and np.all(np.isfinite(ta["A_h"].x.array))
              and all(np.all(np.isfinite(T_i.x.array)) for T_i in fx["T_fns"]))
        if not ok:
            break

    print(f"OUTCOME: {outcome}", flush=True)
    with open(f"{dump_path}_outcome.txt", "w") as fh:
        fh.write(outcome)

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
