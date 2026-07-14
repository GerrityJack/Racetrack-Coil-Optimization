"""
solve.py
=========
Magnetostatic A-formulation solve for the racetrack coil, using Nedelec
(N1curl) elements -- see the weak-form derivation discussed alongside
this script. This is the single-current entry point (run it directly
for one solve at params.I_design); solve_sweep.py reuses the
setup_mesh/setup_problem/solve_at_current functions below to re-solve
at several currents without rebuilding the mesh or FEM problem each time.

Strong form (mu = mu0 everywhere -- no magnetic materials):

    curl( (1/mu0) curl(A) ) = J        in Omega
    n x A = 0                          on the outer (truncation) boundary

Weak form (regularized to remove the curl-curl null space {grad phi}):

    find A in V_h subset H(curl; Omega) such that, for all v in V_h
    with n x v = 0 on the outer boundary,

        (1/mu0) * (curl A, curl v)_Omega + eps * (A, v)_Omega
            = (J, v)_Omega

This script targets the dolfinx 0.11 API (confirmed against the
official 0.11.0 docs: dolfinx.io.gmshio was renamed to dolfinx.io.gmsh
and its read functions now return a MeshData dataclass instead of a
tuple; fem.petsc.LinearProblem now requires a petsc_options_prefix
keyword; FiniteElement.interpolation_points is now a property, not a
method). The companion build_mesh.py and current_source.py modules have
been independently validated in this environment (gmsh is pure-Python
installable): the CAD volumes match analytic values to ~1e-15 relative
error, and the analytic current-direction field is continuous and
unit-norm everywhere. This file itself has still not been executed (no
dolfinx available in this sandbox) -- please report back if you hit any
further API mismatches.
"""
import numpy as np
import ufl
import basix.ufl
from dolfinx import fem, io, mesh as dmesh
from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "mesh"),
           os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
import build_mesh
from current_source import build_unit_current_direction

mu0 = 4 * np.pi * 1e-7


def evaluate_at_points(domain, function, points, comm):
    """
    Evaluate `function` (a dolfinx Function) at arbitrary `points` (N,3),
    using the standard dolfinx bounding-box-tree point-location pattern.
    MPI-safe: every rank evaluates the points whose owning cell it holds,
    then results are gathered to rank 0. Points not found on any rank
    (e.g. just outside the meshed domain) come back as NaN.

    Returns (values, mask) on rank 0 -- values: (N, value_size) with NaN
    rows where nothing was found; mask: (N,) bool, True where resolved.
    On other ranks, returns (None, None).
    """
    from dolfinx import geometry

    points = np.asarray(points, dtype=np.float64)
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    local_idx, local_cells = [], []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            local_idx.append(i)
            local_cells.append(links[0])

    value_size = function.function_space.value_size
    if local_idx:
        local_vals = function.eval(points[local_idx], np.array(local_cells, dtype=np.int32))
        # dolfinx eval squeezes the single-point case to shape (value_size,);
        # normalise so row indexing below stays correct
        local_vals = np.asarray(local_vals).reshape(-1, value_size)
    else:
        local_vals = np.zeros((0, value_size))

    gathered = comm.gather((local_idx, local_vals), root=0)
    if comm.rank != 0:
        return None, None

    values = np.full((points.shape[0], value_size), np.nan)
    mask = np.full(points.shape[0], False)
    for idx_list, vals in gathered:
        for k, i in enumerate(idx_list):
            if not mask[i]:  # first rank to claim a point wins
                values[i] = vals[k]
                mask[i] = True
    return values, mask


def setup_mesh(comm):
    """Build (rank 0 only) and load the mesh. Call this ONCE -- the mesh
    doesn't depend on current, so solve_sweep.py reuses this same mesh
    across every current it solves at."""
    if comm.rank == 0:
        build_mesh.build(write_path=params.mesh_filename)
    comm.barrier()

    # dolfinx >= 0.10: dolfinx.io.gmshio was renamed to dolfinx.io.gmsh,
    # and read_from_msh now returns a MeshData dataclass rather than a
    # (mesh, cell_tags, facet_tags) tuple.
    mesh_data = gmshio.read_from_msh(params.mesh_filename, comm, rank=0, gdim=3)
    return mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags


def setup_problem(domain, cell_tags, facet_tags):
    """
    Build the function space, current source (unit direction, magnitude
    1), boundary condition, and weak form ONCE. The bilinear form a_form
    doesn't involve J at all (only L_form does), so solving at a
    different current is just: rescale J's array, call problem.solve()
    again -- see solve_at_current(). No need to rebuild any of this.

    Returns a dict with everything solve_at_current()/main() need.
    """
    elem = basix.ufl.element("N1curl", domain.basix_cell(), params.nedelec_degree)
    V = fem.functionspace(domain, elem)

    J_unit, coil_cells = build_unit_current_direction(
        domain, cell_tags, params.coil_marker, params.a, params.b)
    J = fem.Function(J_unit.function_space, name="J")
    J_unit_array = J_unit.x.array.copy()

    # ---- boundary condition: n x A = 0 on the outer box -----------
    # For H(curl) (edge) elements, this is implemented by zeroing every
    # DOF associated with an edge lying on the outer boundary facets --
    # exactly the same locate_dofs_topological call used for Lagrange
    # spaces; dolfinx resolves it correctly for Nedelec dofmaps too.
    fdim = domain.topology.dim - 1
    # outer_boundary_marker includes x=0, y=0 PEC planes in eighth-symmetry
    # mode. The z=g PMC face carries no marker — left free (natural BC).
    boundary_facets = facet_tags.find(params.outer_boundary_marker)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0
    bc = fem.dirichletbc(u_zero, boundary_dofs)

    A = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a_form = (1.0 / mu0) * ufl.inner(ufl.curl(A), ufl.curl(v)) * ufl.dx \
        + params.gauge_regularization * ufl.inner(A, v) * ufl.dx
    L_form = ufl.inner(J, v) * ufl.dx

    # SOLVER CHOICE -- this matters a lot, see below.
    #
    # Default: direct LU (MUMPS). For the "smoke"/"medium" mesh tiers in
    # params.py this is safe and reliable. It gets risky memory-wise at
    # the much larger "fine" tier (a direct 3D factorization has very
    # bad fill-in and can need tens of GB of RAM) -- if you scale the
    # mesh up, switch away from MUMPS *before* you run out of memory,
    # not after; see the GAMG/AMS notes below.
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    #
    # Why not the iterative CG+GAMG combination this project shipped
    # with earlier: it actively DIVERGES on this curl-curl system
    # (PETSc KSP converged reason -4 = KSP_DIVERGED_DTOL, the residual
    # grew past the divergence tolerance -- confirmed by an actual run,
    # not a guess). This isn't a tuning problem, it's a fundamental
    # mismatch: classical algebraic multigrid assumes something close
    # to an elliptic operator with a small near-null space, but the
    # curl-curl operator's near-null space is the *entire* space of
    # gradients -- AMG's coarsening/smoothing doesn't respect that
    # structure, and combined with how weak gauge_regularization is
    # (deliberately, to barely perturb the physics), GAMG has nothing
    # to grip onto. Silently wrong/garbage field values, not a crash,
    # are the dangerous failure mode here -- ALWAYS check
    # ksp.getConvergedReason() (done automatically below; the run that
    # surfaced this had reason -4 at every single current and 1473
    # "successfully" written coil values that were actually numerical
    # garbage).
    #
    # If/when you outgrow MUMPS (mesh too large for direct factorization),
    # the correct replacement is HYPRE's AMS (auxiliary-space Maxwell
    # solver), which is specifically designed for H(curl)/Nedelec
    # systems and respects the gradient near-null space GAMG ignores.
    # This needs the discrete gradient operator and nodal coordinates
    # threaded into the PETSc PC explicitly (dolfinx's elasticity/
    # Maxwell demos show the pattern via
    # `dolfinx.fem.petsc.discrete_gradient` + `PETSc.PC.setHYPREDiscreteGradient`
    # and `PC.setCoordinates`) -- worth setting up once you actually
    # need production-scale meshes, but NOT done here (untested in this
    # environment -- don't trust hand-written AMS setup code you
    # haven't independently verified converges; validate it against a
    # small case where you also have a trusted MUMPS answer to compare).
    problem = LinearProblem(a_form, L_form, bcs=[bc],
                             petsc_options_prefix="racetrack_coil_",
                             petsc_options=petsc_options)

    return dict(V=V, J=J, J_unit_array=J_unit_array, coil_cells=coil_cells,
                bc=bc, problem=problem, petsc_options=petsc_options)


def solve_at_current(domain, setup, I_amps, comm, verbose_label=None, log=None):
    """
    Rescale J to the current I_amps (A) and (re-)solve. Returns B_h, a
    DG0 vector Function (curl of degree-1 N1curl is exactly piecewise
    constant, so this recovery is exact -- no projection error).

    Raises RuntimeError on a non-converged solve, rather than printing
    a number that's easy to scroll past -- a solve.py run with GAMG
    that "succeeded" but actually diverged (PETSc reason -4) is exactly
    what produced numerical-garbage field values silently downstream.

    log: optional diagnostics.SolveLog -- if given, records solver
    options, PETSc converged reason, iteration count, the actual
    residual history (real for an iterative solver, a single trivial
    point for a direct solve), an independently-computed true residual,
    field magnitude stats, and wall-clock time for this solve.
    """
    import time
    from diagnostics import ConvergenceRecorder

    J_mag = I_amps / (params.t * params.w)
    setup["J"].x.array[:] = setup["J_unit_array"] * J_mag

    ksp = setup["problem"].solver
    recorder = ConvergenceRecorder().attach(ksp)

    t0 = time.time()
    A_h = setup["problem"].solve()
    wall_time = time.time() - t0

    label = verbose_label or f"I={I_amps:.1f} A"

    if comm.rank == 0:
        reason = ksp.getConvergedReason()
        print(f"  [{label}] KSP converged reason: {reason}, "
              f"iterations: {ksp.getIterationNumber()}")
        if reason <= 0:
            raise RuntimeError(
                f"KSP did NOT converge at {label} (PETSc converged reason "
                f"{reason} -- negative means diverged/failed, not just slow; "
                "see https://petsc.org/release/manualpages/KSP/KSPConvergedReason "
                "for the code meanings). The resulting field values are "
                "numerical garbage, not physics -- do not use them. If you "
                "changed the solver away from the MUMPS default in "
                "setup_problem(), that's the first thing to revert; if you "
                "kept MUMPS, look for an unusually distorted/degenerate mesh "
                "or a much larger problem than MUMPS can handle.")

    Vb = fem.functionspace(domain, ("DG", 0, (3,)))
    B_h = fem.Function(Vb, name="B")
    curl_expr = fem.Expression(ufl.curl(A_h), Vb.element.interpolation_points)
    B_h.interpolate(curl_expr)

    if log is not None and comm.rank == 0:
        try:
            x_petsc_vec = A_h.x.petsc_vec
        except AttributeError:
            x_petsc_vec = A_h.vector  # older dolfinx attribute name
        log.log_solve(label, I_amps, ksp, recorder, setup["problem"],
                      x_petsc_vec, B_h, setup["coil_cells"], wall_time,
                      setup["petsc_options"])

    return A_h, B_h


def extract_and_save(domain, cell_tags, coil_cells, B_h, comm, out_npz_path,
                      I_solved, include_grid_slices=True):
    """
    Extract coil-cell B values and volumes (and, optionally, the two
    point-evaluated grid slices used by plot_fields.py) and save to a
    portable .npz that needs no dolfinx to read. include_grid_slices=False
    (used by solve_sweep.py) skips the grid evaluation -- the quench
    analysis only needs coil-cell data, and skipping the grid saves
    time/space when solving at a dozen different currents.

    Cell volumes (via ufl.CellVolume, interpolated into DG0 the same way
    B = curl(A) is) are exported so validation/biot_savart_check.py can
    treat each coil cell as a discrete current element (moment = J_vec *
    volume) for an independent cross-check against the FEM field.
    """
    coil_centroids_local = dmesh.compute_midpoints(domain, domain.topology.dim, coil_cells)
    B_block = B_h.x.array.reshape(-1, 3)
    coil_B_local = B_block[coil_cells]

    Vvol = fem.functionspace(domain, ("DG", 0))
    vol_h = fem.Function(Vvol)
    vol_expr = fem.Expression(ufl.CellVolume(domain), Vvol.element.interpolation_points)
    vol_h.interpolate(vol_expr)
    coil_vol_local = vol_h.x.array[coil_cells]

    gathered_coil = comm.gather((coil_centroids_local, coil_B_local, coil_vol_local), root=0)

    save_kwargs = {}
    if include_grid_slices:
        margin = 1.15
        xs_top = np.linspace(-params.b * margin, params.b * margin, params.topview_nx)
        ys_top = np.linspace(-params.b * 0.55 * margin, params.b * 0.55 * margin, params.topview_ny)
        Xtop, Ytop = np.meshgrid(xs_top, ys_top)
        top_points = np.column_stack([Xtop.ravel(), Ytop.ravel(), np.zeros(Xtop.size)])
        top_B, top_mask = evaluate_at_points(domain, B_h, top_points, comm)

        xs_side = np.linspace(-params.b * margin, params.b * margin, params.sideview_nx)
        zs_side = np.linspace(-params.b * 0.3, params.b * 0.3, params.sideview_nz)
        Xside, Zside = np.meshgrid(xs_side, zs_side)
        side_points = np.column_stack([Xside.ravel(), np.zeros(Xside.size), Zside.ravel()])
        side_B, side_mask = evaluate_at_points(domain, B_h, side_points, comm)

        if comm.rank == 0:
            save_kwargs.update(
                top_X=Xtop, top_Y=Ytop, top_B=top_B.reshape(*Xtop.shape, 3),
                top_mask=top_mask.reshape(Xtop.shape),
                side_X=Xside, side_Z=Zside, side_B=side_B.reshape(*Xside.shape, 3),
                side_mask=side_mask.reshape(Xside.shape))

    if comm.rank == 0:
        coil_centroids = np.concatenate([c for c, _, _ in gathered_coil], axis=0)
        coil_B = np.concatenate([b for _, b, _ in gathered_coil], axis=0)
        coil_volume = np.concatenate([v for _, _, v in gathered_coil], axis=0)

        np.savez(out_npz_path,
                 I_solved=I_solved,
                 a=params.a, b=params.b, t=params.t, w=params.w,
                 coil_centroids=coil_centroids, coil_B=coil_B,
                 coil_volume=coil_volume,
                 **save_kwargs)
        print(f"  Wrote {out_npz_path} ({coil_centroids.shape[0]} coil cells"
              + (f", grid slices {Xtop.shape}/{Xside.shape}" if include_grid_slices else "")
              + ")")


def main():
    comm = MPI.COMM_WORLD

    domain, cell_tags, facet_tags = setup_mesh(comm)
    setup = setup_problem(domain, cell_tags, facet_tags)

    log = None
    if comm.rank == 0:
        from diagnostics import SolveLog
        log = SolveLog()
        try:
            n_nodes = domain.topology.index_map(0).size_global
            n_cells = domain.topology.index_map(domain.topology.dim).size_global
        except Exception:
            n_nodes, n_cells = None, None
        log.set_header(
            mode="single solve (solve.py)",
            num_coil_cells=len(setup["coil_cells"]),
            num_V_dofs=setup["V"].dofmap.index_map.size_global,
            mesh_nodes=n_nodes, mesh_cells=n_cells,
            petsc_options=setup["petsc_options"],
            gauge_regularization=params.gauge_regularization,
            I_design=params.I_design,
        )

    A_h, B_h = solve_at_current(domain, setup, params.I_design, comm,
                                 verbose_label="single solve", log=log)

    with io.XDMFFile(comm, params.output_filename, "w") as xf:
        xf.write_mesh(domain)
        xf.write_function(B_h)

    extract_and_save(domain, cell_tags, setup["coil_cells"], B_h, comm,
                      params.fields_npz_filename, params.I_design,
                      include_grid_slices=True)

    if comm.rank == 0:
        print(f"Total current I = {params.I_design:.4g} A")
        print(f"Wrote {params.output_filename}")
        log.write(os.path.join(params.SOLVE_DIR, "solve_log.txt"),
                  os.path.join(params.SOLVE_DIR, "solve_log.json"))
        log.plot(os.path.join(params.SOLVE_DIR, "convergence_plot.png"))

    return domain, A_h, B_h


if __name__ == "__main__":
    main()
