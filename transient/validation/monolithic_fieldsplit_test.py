"""
monolithic_fieldsplit_test.py -- 2026-08-05 overnight follow-up, part 6
(coordinator-directed continuation after the BC-scoping fix was verified).

Tests real PETSc PCFIELDSPLIT (not the MUMPS-auto-scaling proxy from Part
5) on the BC-bug-fixed monolithic system, targeting the 6-8 order of
magnitude scale mismatch between T-blocks (diagonal mean ~2.5e3-5.7e3)
and the A-block (diagonal mean ~1.19e10) found in Part 3.

KEY API FACT, established by reading dolfinx/fem/petsc.py directly rather
than guessing: dolfinx's NonlinearProblem only auto-wires field-split
index sets when kind="nest" AND a separate preconditioner form P is
passed (creating P_mat as a nest matrix too) -- monolithic_ta.py passes
neither. This script builds with kind="nest" and wires the field-split
ISs MANUALLY via A.getNestISs() + pc.setFieldSplitIS(), not relying on
that automatic path.

Same distinct-per-layer-function-space BC fix as
monolithic_fixed_bc_test.py (verified structurally correct in Part 3) --
this file changes ONLY the matrix kind and PC configuration, reusing that
fix's per-layer space construction directly.
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


def build_fieldsplit_monolithic_problem(ta, domain, pc_fieldsplit_type="additive",
                                        verbose=False):
    """Same physics/BC-fix construction as monolithic_fixed_bc_test.py's
    build_fixed_monolithic_problem, but kind="nest" + a real Krylov method
    (fgmres) wrapped around a fieldsplit PC instead of kind="mpi" +
    preonly+lu. Field names: T_layer{i} (matching each Function's own
    .name, matching ta_solve.py's naming convention) and "A_h".
    """
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

    import basix.ufl as _bu
    elem_dg3 = _bu.element("DG", domain.basix_cell(), 0, shape=(3,))
    Vdg3 = fem.functionspace(domain, elem_dg3)

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
        Jc_fn = fem.Function(Vdg0, name=f"jc_fs_{i}")
        n_fn = fem.Function(Vdg0, name=f"n_fs_{i}")
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
    ta["A_h"].name = "A_h"   # ensure a stable, predictable field-split name

    u_list = T_fns + [ta["A_h"]]
    bcs = [bc for bcs_i in bcs_list for bc in bcs_i]
    bcs.append(ta["bc_A"])

    # ksp_type MUST be a real Krylov method here, not "preonly" -- a
    # block-Jacobi/Schur fieldsplit PC is only an APPROXIMATE inverse of
    # the true (fully coupled) operator; "preonly" would apply it once as
    # if exact, silently discarding exactly the coupling we need Krylov
    # iteration to correct for.
    # Split into two groups: the TOP-LEVEL options (read once by
    # solver.setFromOptions() inside NonlinearProblem.__init__, which is
    # sufficient since it structurally creates the PCFIELDSPLIT object) vs.
    # the FIELDSPLIT SUB-OPTIONS. The sub-options must NOT be passed
    # through NonlinearProblem's own petsc_options dict: dolfinx pushes
    # them onto the global options database, calls setFromOptions() once,
    # then IMMEDIATELY `del`s every key it pushed and pops the prefix --
    # but the per-split sub-KSP/sub-PC objects do not exist yet at that
    # point (they are created lazily, inside PCSetUp, triggered by the
    # FIRST real solve call, which happens well after __init__ returns and
    # already deleted these options). Confirmed directly: passing them
    # through petsc_options produced PETSc's own "there are options you
    # set that were not used" warning naming exactly these three keys.
    # Fix: set the sub-options directly and PERSISTENTLY on the global
    # PETSc.Options() object (fully-prefixed key, never deleted), so they
    # are still present whenever PCSetUp actually reads them.
    top_opts = dict(
        snes_type="newtonls",
        snes_linesearch_type="bt",
        snes_linesearch_max_it=40,
        snes_rtol=1e-6, snes_atol=1e-8, snes_stol=1e-10,
        snes_max_it=1,
        ksp_type="fgmres",
        ksp_rtol=1e-8, ksp_max_it=200,
        pc_type="fieldsplit",
        pc_fieldsplit_type=pc_fieldsplit_type,
    )
    if verbose:
        top_opts["snes_monitor"] = None
        top_opts["ksp_monitor"] = None

    prefix = "monofs_ta_"
    # kind="mpi" (a genuine monolithic AIJ matrix), NOT "nest" -- confirmed
    # directly that SELFP's Schur-complement approximation needs a
    # MatMatMult that PETSc does not support between two MATNEST-typed
    # sub-blocks (extracting a combined multi-block "T" field from a nest
    # matrix keeps it nest-structured internally: "Unspecified symbolic
    # phase for product AB with A nest, B nest. The product is not
    # supported"). A plain AIJ matrix has no such limitation -- field-split
    # over a monolithic AIJ matrix using manually-supplied IS objects is
    # the standard, common way PCFIELDSPLIT is used in practice.
    problem = NonlinearProblem(F_list, u_list, bcs=bcs, kind="mpi",
                               petsc_options_prefix=prefix,
                               petsc_options=top_opts)
    snes = problem.solver
    snes.setTolerances(max_it=1)

    # Build the field-split index sets MANUALLY from known dof offsets
    # (no getNestISs() available for a plain AIJ matrix -- and not needed,
    # since we already know each layer's V_T_i size and V_A's size).
    Amat = problem.A
    assert Amat.getType() != "nest", f"expected a plain AIJ matrix, got {Amat.getType()}"
    n_T = V_T_list[0].dofmap.index_map.size_local
    n_A = ta["V_A"].dofmap.index_map.size_local
    offsets = [i * n_T for i in range(params.n_layers + 1)] + [params.n_layers * n_T + n_A]
    block_IS = [PETSc.IS().createStride(offsets[i + 1] - offsets[i], first=offsets[i], step=1,
                                        comm=Amat.comm)
               for i in range(params.n_layers + 1)]

    if pc_fieldsplit_type == "schur":
        t_idx = np.concatenate([isx.getIndices() for isx in block_IS[:params.n_layers]])
        t_idx = np.sort(t_idx.astype(np.int32))
        is_T = PETSc.IS().createGeneral(t_idx, comm=Amat.comm)
        is_A = block_IS[params.n_layers]
        fieldsplit_IS = (("T", is_T), ("A_h", is_A))
    else:
        names = [f"T_layer{i}" for i in range(params.n_layers)] + ["A_h"]
        fieldsplit_IS = tuple(zip(names, block_IS))

    pc = snes.getKSP().getPC()
    pc.setFieldSplitIS(*fieldsplit_IS)
    if pc_fieldsplit_type == "schur":
        pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)
        # SELFP: build an EXPLICIT approximate Schur complement matrix
        # (using the assembled A11 block and an approximate inverse of
        # A00's diagonal) rather than treating S as a fully implicit
        # matrix-free operator -- the implicit form cannot be handed to a
        # direct LU factorization at all.
        pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.SELFP)
    # DO NOT call pc.setUp() or getFieldSplitSubKSP() here -- both were
    # confirmed to fail with PETSc error 73 ("Not for unassembled
    # matrix"/"MAT_COPY_VALUES not allowed for unassembled matrix") when
    # attempted before any real assembly has happened (additive: fails
    # silently -- creates default ILU sub-solvers instead of the
    # requested LU, confirmed via PC.view(); schur: fails outright, since
    # even the CHEAP SELFP Schur approximation needs to read real matrix
    # values to build itself). Sub-solver configuration is deferred to
    # _configure_sub_solvers(), called by the caller ONCE, immediately
    # after the FIRST real problem.solve() (which triggers genuine
    # assembly for the first time).

    return dict(V_T_list=V_T_list, T_fns=T_fns, bcs_list=bcs_list,
               Jc_fns=Jc_fns, n_fns=n_fns, problem=problem, snes=snes,
               cell_idx_list=cell_idx_list, layer_assign=layer_assign,
               pc_fieldsplit_type=pc_fieldsplit_type)


def _configure_sub_solvers(fx):
    """Call ONCE, immediately after the first real problem.solve() (which
    triggers genuine matrix assembly for the first time) -- see the long
    comment in build_fieldsplit_monolithic_problem for why this cannot be
    done any earlier."""
    pc = fx["snes"].getKSP().getPC()
    for sub_ksp in pc.getFieldSplitSubKSP():
        sub_ksp.setType("preonly")
        sub_pc = sub_ksp.getPC()
        sub_pc.setType("lu")
        sub_pc.setFactorSolverType("mumps")
        sub_ksp.setUp()


def main():
    n_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    fs_type = sys.argv[2] if len(sys.argv) > 2 else "additive"
    step_relax = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    dt = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
    I_now = float(sys.argv[5]) if len(sys.argv) > 5 else 19.6

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
    params.mesh_filename = f"{root}_monofs_{os.getpid()}{ext}"
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags, facet_tags = md.mesh, md.cell_tags, md.facet_tags
    uniform = base_solve.setup_problem(domain, cell_tags, facet_tags)
    ic = IcModel(params.csv_filename)
    nm = NValueModel(params.n_value_csv_filename)

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

    print(f"Building fieldsplit ({fs_type}) monolithic problem, "
          f"dt={dt}s I={I_now}A step_relax={step_relax} ...", flush=True)
    fx = build_fieldsplit_monolithic_problem(ta, domain, pc_fieldsplit_type=fs_type,
                                             verbose=False)

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
            fx["Jc_fns"][layer].x.array[cells_layer] = ((1 - relax) * fx["Jc_fns"][layer].x.array[cells_layer] + relax * Jc_vol)
            fx["n_fns"][layer].x.array[cells_layer] = ((1 - relax) * fx["n_fns"][layer].x.array[cells_layer] + relax * n_arr)

    update_coeffs(relax=1.0)
    problem = fx["problem"]
    snes = fx["snes"]
    prev_fnorm = None
    for k in range(n_iters):
        update_coeffs(relax=0.3)
        T_snap = [T_i.x.array.copy() for T_i in fx["T_fns"]]
        A_snap = ta["A_h"].x.array.copy()
        problem.solve()
        if k == 0:
            # first real assembly has now happened -- safe to configure
            # the sub-solvers properly (see _configure_sub_solvers' and
            # build_fieldsplit_monolithic_problem's docstrings for why
            # this could not be done any earlier).
            _configure_sub_solvers(fx)
        reason = snes.getConvergedReason()
        fnorm = snes.getFunction()[0].norm()
        ksp_its = snes.getKSP().getIterationNumber()
        ratio = fnorm / prev_fnorm if prev_fnorm else float("nan")
        print(f"  k={k+1:3d}  ||F||={fnorm:.6e}  ratio={ratio:.4f}  "
              f"reason={reason}  ksp_its={ksp_its}", flush=True)
        prev_fnorm = fnorm
        ok = (reason not in {-3, -4, -6, -7, -8, -9, -10, -11}
              and np.all(np.isfinite(ta["A_h"].x.array))
              and all(np.all(np.isfinite(T_i.x.array)) for T_i in fx["T_fns"]))
        if not ok:
            print("  STOPPED"); break
        if step_relax != 1.0:
            for T_i, snap in zip(fx["T_fns"], T_snap):
                T_i.x.array[:] = (1 - step_relax) * snap + step_relax * T_i.x.array
                T_i.x.scatter_forward()
            ta["A_h"].x.array[:] = (1 - step_relax) * A_snap + step_relax * ta["A_h"].x.array
            ta["A_h"].x.scatter_forward()

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
