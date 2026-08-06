"""
monolithic_fixed_bc_test.py -- 2026-08-05 overnight follow-up, part 3.

Proof-of-concept test of the fix for the BC-scoping bug found in
monolithic_jacobian_inspect.py / monolithic_jacobian_inspect2.py: builds
a CORRECTED monolithic block-Newton system inline, giving each layer its
own DISTINCT function space object (V_T_0 .. V_T_5, rather than reusing
ta_solve.py's single shared V_T) so each layer's Dirichlet BCs can be
correctly scoped to only that layer's own block. Does NOT modify
ta_solve.py (whose existing per-layer LinearProblem construction already
works correctly, precisely because it solves one layer at a time and
never combines BC lists across layers -- this fix is only needed for the
monolithic case, which combines all 6 layers' equations into one system).

Uses the ORIGINAL frozen-Jc/n formulation (mirroring monolithic_ta.py's
own math exactly, just rebuilt with distinct spaces) -- simpler than
carrying through today's differentiable Jc/n machinery for this first
correctness check. If this fixes the residual blowup, applying the same
distinct-space construction to monolithic_ta_diff.py's differentiable
version is a mechanical extension, not a new idea.
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


def build_fixed_monolithic_problem(ta, domain, verbose=False):
    """Rebuild the per-layer T system with DISTINCT function spaces, then
    the monolithic block Newton problem on top of it. Returns a dict with
    everything the step loop below needs -- deliberately NOT reusing
    ta["layer_T_fns"]/ta["layer_bcs"] (those still reference the shared
    V_T space and are left untouched for any other code path that might
    need them)."""
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

    # Reconstruct the layer_assign / z tolerance logic exactly as
    # ta_solve.setup_ta_problem does, so cell/dof selection matches.
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

    V_T_list, T_fns, bcs_list, gradT_exprs, cell_idx_list = [], [], [], [], []
    for i in range(params.n_layers):
        V_T_i = fem.functionspace(domain, elem_T)   # DISTINCT space -- the fix
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

        T_i = fem.Function(V_T_i, name=f"T_layer{i}_fixed")
        T_i.x.array[:] = 0.0
        # warm-start from the existing (shared-space) layer T, via spatial
        # interpolation -- robust regardless of any dof-numbering
        # difference between the old shared V_T and this new V_T_i.
        T_i.interpolate(ta["layer_T_fns"][i])

        V_T_list.append(V_T_i)
        T_fns.append(T_i)
        bcs_list.append(bcs_i)
        gradT_exprs.append(fem.Expression(ufl.grad(T_i), Vdg3_pts(Vdg0, domain)))
        cell_idx_list.append(idx_i)

    Jc_fns, n_fns, F_list = [], [], []
    for i, T_i in enumerate(T_fns):
        Jc_fn = fem.Function(Vdg0, name=f"jc_fixed_{i}")
        n_fn = fem.Function(Vdg0, name=f"n_fixed_{i}")
        Jc_fn.x.array[:] = 1.0
        n_fn.x.array[:] = 2.0
        Jc_fns.append(Jc_fn)
        n_fns.append(n_fn)

        dx = ufl.Measure("dx", domain=domain,
                         metadata={"quadrature_degree": mono.MONO_QUADRATURE_DEGREE})
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

    dx = ufl.Measure("dx", domain=domain,
                     metadata={"quadrature_degree": mono.MONO_QUADRATURE_DEGREE})
    v_A = ufl.TestFunction(ta["V_A"])
    J_dir = ufl.cross(ufl.grad(T_fns[0]), n_hat_ufl)
    for T_i in T_fns[1:]:
        J_dir = J_dir + ufl.cross(ufl.grad(T_i), n_hat_ufl)
    J_s_symbolic = J_dir * (delta_SC / Lambda)
    F_A = ((1.0 / mono.mu0) * ufl.inner(ufl.curl(ta["A_h"]), ufl.curl(v_A)) * dx
           + params.gauge_regularization * ufl.inner(ta["A_h"], v_A) * dx
           - ufl.inner(J_s_symbolic, v_A) * dx)
    F_list.append(F_A)

    u_list = T_fns + [ta["A_h"]]
    bcs = [bc for bcs_i in bcs_list for bc in bcs_i]   # NOW correctly
                                                       # scoped -- each
                                                       # bc's function_space
                                                       # is a DISTINCT
                                                       # object per layer.
    bcs.append(ta["bc_A"])

    opts = dict(mono.DEFAULT_SNES_OPTIONS)
    if verbose:
        opts["snes_monitor"] = None
    problem = NonlinearProblem(F_list, u_list, bcs=bcs, kind="mpi",
                               petsc_options_prefix="monofix_ta_", petsc_options=opts)
    snes = problem.solver
    snes.setTolerances(max_it=1)

    return dict(V_T_list=V_T_list, T_fns=T_fns, bcs_list=bcs_list,
               Jc_fns=Jc_fns, n_fns=n_fns, problem=problem, snes=snes,
               cell_idx_list=cell_idx_list, layer_assign=layer_assign)


def Vdg3_pts(Vdg0, domain):
    import basix.ufl
    from dolfinx import fem
    elem = basix.ufl.element("DG", domain.basix_cell(), 0, shape=(3,))
    Vdg3 = fem.functionspace(domain, elem)
    return Vdg3.element.interpolation_points


def main():
    n_check_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    jc_n_relax = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    step_relax = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    import numpy as np
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    import monolithic_ta
    from ic_model import IcModel, NValueModel, angle_with_normal_deg
    from newton_ta import ta_transient_seed_cold, _picard_bootstrap

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_monofix_{os.getpid()}{ext}"
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

    print("Building FIXED monolithic problem (distinct per-layer spaces)...",
          flush=True)
    fx = build_fixed_monolithic_problem(ta, domain, verbose=False)

    # sanity: confirm the fix actually changed the scoping property.
    print("Distinct space objects?",
          len(set(id(v) for v in fx["V_T_list"])) == 6)

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
            Jc_fn = fx["Jc_fns"][layer]
            n_fn = fx["n_fns"][layer]
            # NOTE: Jc_fn/n_fn are Vdg0 over the FULL mesh, same indexing
            # (cells_layer) as ta["coil_cells"]-based code elsewhere.
            Jc_fn.x.array[cells_layer] = ((1 - relax) * Jc_fn.x.array[cells_layer]
                                          + relax * Jc_vol)
            n_fn.x.array[cells_layer] = ((1 - relax) * n_fn.x.array[cells_layer]
                                         + relax * n_arr)

    update_coeffs(relax=1.0)

    print(f"jc_n_relax={jc_n_relax}  step_relax={step_relax}", flush=True)
    problem = fx["problem"]
    snes = fx["snes"]
    prev_fnorm = None
    for k in range(n_check_iters):
        update_coeffs(relax=jc_n_relax)
        T_snap = [T_i.x.array.copy() for T_i in fx["T_fns"]]
        A_snap = ta["A_h"].x.array.copy()
        problem.solve()
        reason = snes.getConvergedReason()
        fnorm = snes.getFunction()[0].norm()
        ratio = fnorm / prev_fnorm if prev_fnorm else float("nan")
        print(f"  k={k+1:3d}  ||F||={fnorm:.6e}  ratio={ratio:.4f}  reason={reason}",
              flush=True)
        prev_fnorm = fnorm
        ok = (reason not in {-3, -4, -6, -7, -8, -9, -10, -11}
              and np.all(np.isfinite(ta["A_h"].x.array))
              and all(np.all(np.isfinite(T_i.x.array)) for T_i in fx["T_fns"]))
        if ok and step_relax != 1.0:
            for T_i, snap in zip(fx["T_fns"], T_snap):
                T_i.x.array[:] = (1.0 - step_relax) * snap + step_relax * T_i.x.array
                T_i.x.scatter_forward()
            ta["A_h"].x.array[:] = (1.0 - step_relax) * A_snap + step_relax * ta["A_h"].x.array
            ta["A_h"].x.scatter_forward()
        if not ok:
            print("  STOPPED (failure/nonfinite)")
            break

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
