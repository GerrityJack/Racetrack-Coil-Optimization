"""
monolithic_fieldsplit_dump.py -- 2026-08-05 overnight, final bounded round
(coordinator-directed).

Bit-level first-iteration diff for the block-Jacobi (additive) field-split
configuration, applying the SAME technique
nondeterminism_investigation_2026-08-05.md used for the base T-solve.

IMPORTANT METHODOLOGICAL CORRECTION, found while building this: Part 6's
own additive runs are NOT all mutually comparable. `smoke4`/`smoke5`/
`long1`/`repro1` used `kind="nest"` with sub-solvers configured to
preonly+LU+MUMPS BEFORE any solve (confirmed safe/non-crashing for
additive specifically). `repro2` (labelled "another reproducibility
check" in Part 6) ran AFTER `monolithic_fieldsplit_test.py`'s shared
`build_fieldsplit_monolithic_problem` helper was changed to `kind="mpi"`
with DEFERRED sub-solver configuration -- a change made to fix an
unrelated Schur-complement MatMatMult limitation, but which (since
additive and schur share that one helper) silently altered additive's
own iteration-1 behaviour too (iteration 1 now runs under PETSc's
default ILU sub-solvers, not the intended LU, until reconfigured AFTER
that first solve). `repro2`'s "immediate failure" is therefore NOT
proof of the same phenomenon as `long1`'s clean, same-code immediate
failure -- it's confounded by an actual code difference. Part 6's
overall conclusion (genuine run-to-run variance under IDENTICAL code)
still holds on `smoke4`/`smoke5`/`long1`/`repro1` alone (3 of those 4
already form a clean, consistent set showing real variability), but
`repro2` should not have been cited as supporting evidence and is
retracted as such here.

This script is therefore SELF-CONTAINED (does not import the now-`mpi`
shared helper) and reimplements the ORIGINAL working `kind="nest"` +
early-configuration construction directly, so every run using this
script is a genuinely clean, apples-to-apples repeat.
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


def build_nest_additive_problem(ta, domain, verbose=False):
    """Self-contained rebuild of the ORIGINAL (kind="nest", additive,
    early-configured) construction confirmed working in Part 6's
    smoke4/smoke5/long1/repro1 -- deliberately NOT importing
    monolithic_fieldsplit_test.py's shared helper, which was since
    changed to kind="mpi" for Schur's sake and no longer exhibits this
    code path. Same BC fix (distinct per-layer function spaces) as
    every other file in this investigation."""
    import numpy as np
    import ufl
    import basix.ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import NonlinearProblem
    from petsc4py import PETSc
    import params
    import monolithic_ta as mono

    tdim = domain.topology.dim
    elem_T = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    coil_cells = ta["coil_cells"]
    cell_centroids = ta["cell_centroids"]
    n_hat_ufl = ta["n_hat_ufl"]
    coil_ind = ta["coil_ind"]
    dt_const = ta["dt_const"]
    Vdg0 = ta["Vdg0"]
    delta_SC = ta["delta_SC"]
    Lambda = ta["Lambda"]
    eps_reg = float(getattr(params, "ta_eps_reg", 1.0))
    p_floor = float(getattr(params, "ta_floor_smooth_p", 16.0))
    E_c = 1.0e-4

    z_centers = np.array([(t_ + b_) / 2.0 for t_, b_ in
                          zip(params.layer_z_tops, params.layer_z_bottoms)])
    zc_coil = cell_centroids[coil_cells, 2]
    layer_assign = np.argmin(np.abs(zc_coil[:, None] - z_centers[None, :]), axis=1)
    w_tape = float(params.w)
    _grading = getattr(params, "mesh_z_grading", None)
    if _grading:
        _min_slab = w_tape * min(float(f) for f in _grading)
    else:
        _min_slab = w_tape / max(1, int(getattr(params, "mesh_nz_per_layer", 1)))
    tol_z = min(w_tape / 8.0, 0.4 * _min_slab)

    V_T_list, T_fns, bcs_list, cell_idx_list = [], [], [], []
    for i in range(params.n_layers):
        V_T_i = fem.functionspace(domain, elem_T)
        idx_i = np.nonzero(layer_assign == i)[0]
        cells_i = coil_cells[idx_i]
        dofs_layer = fem.locate_dofs_topological(V_T_i, tdim, cells_i)
        z_hi = float(params.layer_z_tops[i])
        z_lo = float(params.layer_z_bottoms[i])
        top_i = np.intersect1d(dofs_layer, fem.locate_dofs_geometrical(
            V_T_i, lambda x, z=z_hi: np.abs(x[2] - z) < tol_z)).astype(np.int32)
        bot_i = np.intersect1d(dofs_layer, fem.locate_dofs_geometrical(
            V_T_i, lambda x, z=z_lo: np.abs(x[2] - z) < tol_z)).astype(np.int32)
        all_dofs_i = np.arange(V_T_i.dofmap.index_map.size_local, dtype=np.int32)
        non_layer = np.setdiff1d(all_dofs_i, dofs_layer).astype(np.int32)

        T_zero_i = fem.Function(V_T_i)
        T_zero_i.x.array[:] = 0.0
        bcs_i = [fem.dirichletbc(T_zero_i, non_layer),
                 fem.dirichletbc(ta["T_top_val"], top_i, V_T_i),
                 fem.dirichletbc(ta["T_bot_val"], bot_i, V_T_i)]

        T_i = fem.Function(V_T_i, name=f"T_layer{i}")
        T_i.x.array[:] = 0.0
        T_i.interpolate(ta["layer_T_fns"][i])

        V_T_list.append(V_T_i)
        T_fns.append(T_i)
        bcs_list.append(bcs_i)
        cell_idx_list.append(idx_i)

    Jc_fns, n_fns, F_list = [], [], []
    dx = ufl.Measure("dx", domain=domain,
                     metadata={"quadrature_degree": mono.MONO_QUADRATURE_DEGREE})
    for i, T_i in enumerate(T_fns):
        Jc_fn = fem.Function(Vdg0, name=f"jc_fsd_{i}")
        n_fn = fem.Function(Vdg0, name=f"n_fsd_{i}")
        Jc_fn.x.array[:] = 1.0
        n_fn.x.array[:] = 2.0
        Jc_fns.append(Jc_fn)
        n_fns.append(n_fn)

        phi = ufl.TestFunction(V_T_list[i])
        J_ufl = ufl.cross(ufl.grad(T_i), n_hat_ufl)
        Jmag = ufl.sqrt(ufl.inner(J_ufl, J_ufl) + 1e-30)
        j_norm_raw = Jmag / Jc_fn
        j_norm = (eps_reg ** p_floor + j_norm_raw ** p_floor) ** (1.0 / p_floor)
        rho_SC = (E_c / Jc_fn) * ufl.exp((n_fn - 1.0) * ufl.ln(j_norm))
        rho_expr = rho_SC * (delta_SC / Lambda)
        J_phi = ufl.cross(ufl.grad(phi), n_hat_ufl)
        F_i = (rho_expr * ufl.inner(J_ufl, J_phi) * dx
               + (1.0 / dt_const) * coil_ind
                 * ufl.inner(ufl.curl(ta["A_h"] - ta["A_prev"]),
                             phi * n_hat_ufl) * dx)
        F_list.append(F_i)

    v_A = ufl.TestFunction(ta["V_A"])
    J_dir = ufl.cross(ufl.grad(T_fns[0]), n_hat_ufl)
    for T_i in T_fns[1:]:
        J_dir = J_dir + ufl.cross(ufl.grad(T_i), n_hat_ufl)
    J_s_symbolic = J_dir * (delta_SC / Lambda)
    F_A = ((1.0 / mono.mu0) * ufl.inner(ufl.curl(ta["A_h"]), ufl.curl(v_A)) * dx
           + params.gauge_regularization * ufl.inner(ta["A_h"], v_A) * dx
           - ufl.inner(J_s_symbolic, v_A) * dx)
    F_list.append(F_A)
    ta["A_h"].name = "A_h"

    u_list = T_fns + [ta["A_h"]]
    bcs = [bc for bcs_i in bcs_list for bc in bcs_i]
    bcs.append(ta["bc_A"])

    opts = dict(
        snes_type="newtonls", snes_linesearch_type="bt",
        snes_linesearch_max_it=40,
        snes_rtol=1e-6, snes_atol=1e-8, snes_stol=1e-10, snes_max_it=1,
        ksp_type="fgmres", ksp_rtol=1e-8, ksp_max_it=200,
        pc_type="fieldsplit", pc_fieldsplit_type="additive",
    )
    if verbose:
        opts["snes_monitor"] = None

    problem = NonlinearProblem(F_list, u_list, bcs=bcs, kind="nest",
                               petsc_options_prefix="monofsd_ta_",
                               petsc_options=opts)
    snes = problem.solver
    snes.setTolerances(max_it=1)

    Amat = problem.A
    assert Amat.getType() == "nest"
    nest_IS = Amat.getNestISs()[0]
    names = [f"T_layer{i}" for i in range(params.n_layers)] + ["A_h"]
    fieldsplit_IS = tuple(zip(names, nest_IS))
    pc = snes.getKSP().getPC()
    pc.setFieldSplitIS(*fieldsplit_IS)
    # Confirmed safe/non-crashing for additive+nest (unlike schur+nest or
    # any variant of kind="mpi"): early configuration, BEFORE any solve.
    pc.setUp()
    for sub_ksp in pc.getFieldSplitSubKSP():
        sub_ksp.setType("preonly")
        sub_ksp.getPC().setType("lu")
        sub_ksp.getPC().setFactorSolverType("mumps")

    return dict(V_T_list=V_T_list, T_fns=T_fns, bcs_list=bcs_list,
               Jc_fns=Jc_fns, n_fns=n_fns, problem=problem, snes=snes,
               cell_idx_list=cell_idx_list, layer_assign=layer_assign)


def main():
    dump_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/monofs_dump"

    import numpy as np
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    from ic_model import IcModel, NValueModel, angle_with_normal_deg
    from newton_ta import ta_transient_seed_cold, _picard_bootstrap

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_monofsdump_{os.getpid()}{ext}"
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
    snes = fx["snes"]
    problem = fx["problem"]

    # Dump the system that WILL be fed to the first problem.solve().
    x = snes.getSolution()
    Fvec = snes.getFunction()[0]
    Jmat, Pmat = snes.getJacobian()[0:2]
    snes.computeFunction(x, Fvec)
    snes.computeJacobian(x, Jmat, Pmat)
    f0 = Fvec.getArray(readonly=True).copy()
    # Jmat is nest -- convert to plain AIJ before CSR extraction, to keep
    # this dump comparable/diffable across runs the same simple way as
    # the rest of this investigation's dumps.
    Jmat_aij = Jmat.convert("aij")
    ai, aj, av = Jmat_aij.getValuesCSR()
    np.savez(f"{dump_path}_input.npz", indptr=ai, indices=aj, data=av, rhs=f0)
    print(f"Dumped pre-solve system: ||F||={np.linalg.norm(f0):.6e}  nnz={len(av)}",
          flush=True)

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
