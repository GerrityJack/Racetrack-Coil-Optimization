"""
ta_solve.py  — homogenized T-A formulation, single BDF1 step
=============================================================
Models screening-current-induced field (SCIF) by taking one implicit
Euler step from the zero-field-cooled state (T=0, A=0) at t=0 to the
end-of-ramp state at t = params.ramp_duration.

Required additions to params.py (see params_ta_additions.py):
    delta_SC, ramp_duration, ta_n_picard, ta_picard_tol,
    ta_eps_reg, n_value_csv_filename
"""

import numpy as np
import ufl
import basix.ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc
import sys, os, time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"),
           os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
from current_source import tangent_xy, normal_xy, build_unit_current_direction
from ic_model import IcModel, NValueModel, angle_with_normal_deg, compute_rho_hts

mu0 = 4.0 * np.pi * 1e-7


# ── n̂ as UFL expression ────────────────────────────────────────────────────

def build_n_hat_ufl(domain):
    """
    Tape broad-face normal as a UFL expression.
      straight sections (|x| ≤ L): n̂ = ŷ = (0, 1, 0)
      end caps:                     n̂ = radial from cap centre
    """
    x = ufl.SpatialCoordinate(domain)
    L_val = float(params.b - params.a)
    L_c   = fem.Constant(domain, PETSc.ScalarType(L_val))

    cx      = ufl.conditional(ufl.gt(x[0], L_c), L_c, -L_c)
    dx_cap  = x[0] - cx
    rho_cap = ufl.sqrt(dx_cap**2 + x[1]**2 + ufl.as_ufl(1e-30))

    in_straight = ufl.le(abs(x[0]), L_c)   # abs() — not ufl.abs()

    return ufl.as_vector([
        ufl.conditional(in_straight, 0.0,  dx_cap / rho_cap),
        ufl.conditional(in_straight, 1.0,  x[1]   / rho_cap),
        ufl.as_ufl(0.0),
    ])


# ── numpy n̂ at coil-cell centroids ─────────────────────────────────────────

def _coil_n_hat_array(domain, coil_cells):
    """(N_coil, 3) n̂ at coil cell centroids (numpy)."""
    pts = dmesh.compute_midpoints(domain, domain.topology.dim, coil_cells)
    L   = float(params.b - params.a)
    nx, ny = normal_xy(pts[:, 0], pts[:, 1], L)
    return np.column_stack([nx, ny, np.zeros(len(coil_cells))])


def _coil_t_hat_array(domain, coil_cells):
    """(N_coil, 3) tangent t̂ at coil cell centroids (numpy)."""
    pts = dmesh.compute_midpoints(domain, domain.topology.dim, coil_cells)
    L   = float(params.b - params.a)
    tx, ty = tangent_xy(pts[:, 0], pts[:, 1], L)
    return np.column_stack([tx, ty, np.zeros(len(coil_cells))])


# ── One-time problem setup ───────────────────────────────────────────────────

def setup_ta_problem(domain, cell_tags, facet_tags, uniform_setup,
                     per_layer=None):
    """
    Build function spaces, BCs, function containers, and DG0 arrays
    needed by the Picard loop.  Call once; pass result to
    solve_ta_at_current().

    uniform_setup: dict returned by solve.setup_problem() — provides the
    V_A space and the coil_cells array so we don't duplicate mesh work.

    per_layer: if True, solve a SEPARATE T problem in every z-layer
    (each tape gets its own Dirichlet BCs at its own edges and responds
    to its own local ρ(B)) instead of solving one representative central
    tape and replicating its J to the other layers.  Defaults to
    params.ta_per_layer (False if unset).
    """
    if per_layer is None:
        per_layer = bool(getattr(params, "ta_per_layer", False))
    V_A       = uniform_setup["V"]
    coil_cells = uniform_setup["coil_cells"]

    delta_SC = params.delta_SC
    Lambda   = params.t          # tape pitch = homogenisation cell size

    # ── Scalar CG1 space for T ───────────────────────────────────────────
    elem_T = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    V_T    = fem.functionspace(domain, elem_T)

    # ── Dirichlet BCs for T on top/bottom of a SINGLE tape (z = ±w/2) ────
    # The T-A homogenization models one tape of width w (in z).  The BCs
    # encode I = (T_bot − T_top) × δ_SC for that tape, so they belong at
    # the TAPE edges (z = ±w/2), NOT the outer faces of the full n_layers
    # stack (z = ±n_layers·w/2).  Applying them at ±n_layers·w/2 dilutes
    # the effective j/jc by n_layers, causing catastrophic Picard divergence.
    #
    # Strategy: solve T in the central z-layer only (z ∈ [−w/2, +w/2]).
    # Pin T = 0 in all coil cells outside that layer and in all non-coil
    # cells.  After the T solve, replicate the central-layer J to all other
    # z-layers (they are geometrically identical tapes → same redistribution).
    w_tape = float(params.w)              # tape width = one z-layer height
    z_top  =  w_tape / 2.0               # +2 mm   (top of central layer)
    z_bot  = -w_tape / 2.0               # −2 mm   (bottom of central layer)
    # BC-plane tolerance: must stay below the smallest sub-slab thickness
    # so the tape-edge dof selection never catches interior node planes.
    _grading = getattr(params, "mesh_z_grading", None)
    if _grading:
        _min_slab = w_tape * min(float(f) for f in _grading)
    else:
        _min_slab = w_tape / max(1, int(getattr(params, "mesh_nz_per_layer", 1)))
    tol_z = min(w_tape / 8.0, 0.4 * _min_slab)

    T_top_val = fem.Constant(domain, PETSc.ScalarType(0.0))
    T_bot_val = fem.Constant(domain, PETSc.ScalarType(0.0))

    top_dofs = fem.locate_dofs_geometrical(
        V_T, lambda x: np.abs(x[2] - z_top) < tol_z)
    bot_dofs = fem.locate_dofs_geometrical(
        V_T, lambda x: np.abs(x[2] - z_bot) < tol_z)

    if top_dofs.size == 0 or bot_dofs.size == 0:
        raise RuntimeError(
            f"T-A setup: failed to find Dirichlet DOFs on single-tape faces.\n"
            f"  z_top={z_top:.4f} m,  z_bot={z_bot:.4f} m,  tol={tol_z:.4f} m\n"
            f"  top_dofs={top_dofs.size}  bot_dofs={bot_dofs.size}\n"
            "Check that params.w is correct and the mesh has nodes at z=±w/2.")

    bc_T_top = fem.dirichletbc(T_top_val, top_dofs, V_T)
    bc_T_bot = fem.dirichletbc(T_bot_val, bot_dofs, V_T)

    # ── Identify central-layer coil cells and build layer replication map ─
    tdim = domain.topology.dim
    num_cells = domain.topology.index_map(tdim).size_local
    cell_centroids = dmesh.compute_midpoints(
        domain, tdim, np.arange(num_cells, dtype=np.int32))

    # Central-layer coil cells: STRICT layer assignment (nearest layer
    # centre in z).  Do NOT select by |z| ≤ w/2 + tol: that sweeps in the
    # adjacent layers' boundary cell rows (centroids at |z| ≈ 2–2.5 mm),
    # whose T gradient is a BC artifact (T drops from the tape-edge BC
    # value to the pinned 0 across half a cell, giving ΔJ ≈ −2·J_unif of
    # fake "screening").  Those cells then become KD-tree replication
    # sources and pollute J_s and the SCIF sum — this bug inflated the
    # SCIF by orders of magnitude before 2026-07-10.
    zc_coil   = cell_centroids[coil_cells, 2]
    z_centers = np.array([(t_ + b_) / 2.0 for t_, b_ in
                          zip(params.layer_z_tops, params.layer_z_bottoms)])
    layer_assign  = np.argmin(np.abs(zc_coil[:, None] - z_centers[None, :]),
                              axis=1)
    central_layer = int(np.argmin(np.abs(z_centers)))
    coil_cells_T  = coil_cells[layer_assign == central_layer]

    # KD-tree: for every coil cell, find its twin in the central layer —
    # same (x, y) AND the same position within the tape width.  With
    # mesh_nz_per_layer > 1 there are several central cells per (x, y)
    # column (one per sub-slab), so the match must include the tape-local
    # z offset (z − layer centre) or the replication would copy from an
    # arbitrary slab.
    from scipy.spatial import cKDTree
    rel_central = cell_centroids[coil_cells_T].copy()
    rel_central[:, 2] -= z_centers[central_layer]
    rel_all = cell_centroids[coil_cells].copy()
    rel_all[:, 2] -= z_centers[layer_assign]
    _tree      = cKDTree(rel_central)
    _, layer_map = _tree.query(rel_all, k=1)  # layer_map[i] → idx in coil_cells_T

    # ── Pin T to zero everywhere outside the central layer ────────────────
    all_T_dofs         = np.arange(V_T.dofmap.index_map.size_local, dtype=np.int32)
    central_T_dofs     = fem.locate_dofs_topological(V_T, tdim, coil_cells_T)
    non_central_T_dofs = np.setdiff1d(all_T_dofs, central_T_dofs).astype(np.int32)

    T_zero_fn = fem.Function(V_T); T_zero_fn.x.array[:] = 0.0
    bc_T_noncoil = fem.dirichletbc(T_zero_fn, non_central_T_dofs)
    # Order: non-central/noncoil first (lower priority), top/bot last (win).
    all_T_bcs = [bc_T_noncoil, bc_T_top, bc_T_bot]

    # ── n̂ arrays (numpy) ─────────────────────────────────────────────────
    n_hat_coil = _coil_n_hat_array(domain, coil_cells)   # (N_coil, 3)
    t_hat_coil = _coil_t_hat_array(domain, coil_cells)   # (N_coil, 3)

    # ── DG0 coefficient fields ────────────────────────────────────────────
    Vdg0 = fem.functionspace(domain, ("DG", 0))
    Vdg3 = fem.functionspace(domain, ("DG", 0, (3,)))

    rho_fn   = fem.Function(Vdg0, name="rho_hts")
    J_s_fn   = fem.Function(Vdg3, name="J_s")
    rho_fn.x.array[:] = 0.0
    J_s_fn.x.array[:] = 0.0

    # Coil subdomain indicator (DG0, 1 in coil cells, 0 elsewhere)
    coil_ind = fem.Function(Vdg0, name="coil_indicator")
    coil_ind.x.array[:] = 0.0
    coil_ind.x.array[coil_cells] = 1.0
    coil_ind.x.scatter_forward()

    # Exact DG0 cell volumes at coil cells (for Biot-Savart SCIF sums —
    # replaces the mean-nearest-neighbour-spacing³ estimate).
    vol_fn = fem.Function(Vdg0)
    vol_fn.interpolate(fem.Expression(
        ufl.CellVolume(domain), Vdg0.element.interpolation_points))
    coil_vols = vol_fn.x.array[coil_cells].copy()

    # ── T and A Function containers ───────────────────────────────────────
    T_h = fem.Function(V_T, name="T")
    T_h.x.array[:] = 0.0

    A_h = fem.Function(V_A, name="A")
    A_h.x.array[:] = 0.0

    # ── n̂ UFL ─────────────────────────────────────────────────────────────
    n_hat_ufl = build_n_hat_ufl(domain)

    # n̂ / t̂ arrays for both the full coil and the central-layer subset
    n_hat_coil_T = _coil_n_hat_array(domain, coil_cells_T)
    t_hat_coil_T = _coil_t_hat_array(domain, coil_cells_T)

    # ── Cached interpolation objects (reused every Picard iteration) ──────
    # T_h and A_h are updated in place, so these Expressions stay valid.
    gradT_fn   = fem.Function(Vdg3)
    gradT_expr = fem.Expression(ufl.grad(T_h),
                                Vdg3.element.interpolation_points)
    B_fn      = fem.Function(Vdg3, name="B")
    curl_expr = fem.Expression(ufl.curl(A_h),
                               Vdg3.element.interpolation_points)

    # ── Per-layer T problems (per_layer=True mode) ────────────────────────
    # One T Function + BC set per z-layer.  Adjacent tapes share their
    # interface nodes in the CG1 space, and the two tapes need OPPOSITE
    # T values there (+I/2δ on the lower tape's top = −I/2δ on the upper
    # tape's bottom), so the layers cannot be solved in one system —
    # each layer gets its own solve with everything outside it pinned.
    layer_T_fns, layer_bcs, layer_gradT_exprs, layer_cell_idx = [], [], [], []
    if per_layer:
        for i in range(params.n_layers):
            idx_i   = np.nonzero(layer_assign == i)[0]    # rows in coil_cells
            cells_i = coil_cells[idx_i]
            if cells_i.size == 0:
                raise RuntimeError(f"per-layer T-A setup: layer {i} has no cells")

            dofs_layer = fem.locate_dofs_topological(V_T, tdim, cells_i)
            z_hi = float(params.layer_z_tops[i])
            z_lo = float(params.layer_z_bottoms[i])
            top_i = np.intersect1d(dofs_layer, fem.locate_dofs_geometrical(
                V_T, lambda x, z=z_hi: np.abs(x[2] - z) < tol_z)).astype(np.int32)
            bot_i = np.intersect1d(dofs_layer, fem.locate_dofs_geometrical(
                V_T, lambda x, z=z_lo: np.abs(x[2] - z) < tol_z)).astype(np.int32)
            if top_i.size == 0 or bot_i.size == 0:
                raise RuntimeError(
                    f"per-layer T-A setup: no edge DOFs for layer {i} "
                    f"(z=[{z_lo*1e3:.1f},{z_hi*1e3:.1f}] mm, "
                    f"top={top_i.size} bot={bot_i.size})")

            non_layer = np.setdiff1d(all_T_dofs, dofs_layer).astype(np.int32)
            # pin first (lower priority), tape-edge BCs last (win)
            bcs_i = [fem.dirichletbc(T_zero_fn, non_layer),
                     fem.dirichletbc(T_top_val, top_i, V_T),
                     fem.dirichletbc(T_bot_val, bot_i, V_T)]

            T_i = fem.Function(V_T, name=f"T_layer{i}")
            T_i.x.array[:] = 0.0
            layer_T_fns.append(T_i)
            layer_bcs.append(bcs_i)
            layer_gradT_exprs.append(fem.Expression(
                ufl.grad(T_i), Vdg3.element.interpolation_points))
            layer_cell_idx.append(idx_i)

    ta = dict(
        V_T=V_T, V_A=V_A,
        all_T_bcs=all_T_bcs,
        T_top_val=T_top_val, T_bot_val=T_bot_val,
        top_dofs=top_dofs, bot_dofs=bot_dofs,
        n_hat_ufl=n_hat_ufl,
        n_hat_coil=n_hat_coil, t_hat_coil=t_hat_coil,     # all coil cells
        n_hat_coil_T=n_hat_coil_T, t_hat_coil_T=t_hat_coil_T,  # central layer
        coil_cells=coil_cells,
        coil_cells_T=coil_cells_T,   # central z-layer cells (T problem domain)
        layer_map=layer_map,          # layer_map[i] → index into coil_cells_T
        rho_fn=rho_fn, J_s_fn=J_s_fn, coil_ind=coil_ind,
        T_h=T_h, A_h=A_h,
        Vdg0=Vdg0, Vdg3=Vdg3,
        delta_SC=delta_SC, Lambda=Lambda,
        z_top=z_top, z_bot=z_bot,
        cell_centroids=cell_centroids,
        coil_centroids=cell_centroids[coil_cells].copy(),
        coil_vols=coil_vols,
        gradT_fn=gradT_fn, gradT_expr=gradT_expr,
        B_fn=B_fn, curl_expr=curl_expr,
        unique_T_idx=np.unique(layer_map),
        per_layer=per_layer,
        layer_T_fns=layer_T_fns, layer_bcs=layer_bcs,
        layer_gradT_exprs=layer_gradT_exprs,
        layer_cell_idx=layer_cell_idx,   # rows in coil_cells per layer
    )

    # Build the T/A solver objects ONCE — reused for every current in a
    # sweep.  The A-matrix never changes, so its MUMPS factorisation is
    # computed on the first solve and reused for every subsequent one.
    _build_problems(domain, ta, uniform_setup)

    return ta


# ── ρ_HTS helper ─────────────────────────────────────────────────────────────

def _update_rho(ta, J_coil, B_coil, ic_model, n_model, eps_reg, relax=None):
    """
    Recompute rho_fn (DG0) from current J and B at coil cell centroids.
    J_coil: (N_coil, 3) physical SC-layer current density [A/m²]
    B_coil: (N_coil, 3) magnetic flux density [T]

    relax: optional under-relaxation factor β for ρ, applied in LOG space
    (ρ spans decades): ρ ← exp((1−β)·ln ρ_prev + β·ln ρ_new).  Damps the
    ρ-chatter of cells flickering across the eps_reg floor at the
    penetration front — an oscillation that T-relaxation alone cannot
    damp because its amplitude is independent of α (measured 2026-07-10).
    The fixed point is unchanged.  None = plain update (used for seeding).
    """
    delta_SC = ta["delta_SC"]
    Lambda   = ta["Lambda"]
    n_hat    = ta["n_hat_coil"]

    B_mag  = np.linalg.norm(B_coil, axis=-1)
    theta  = angle_with_normal_deg(B_coil, n_hat)

    Ic_arr, _ = ic_model.critical_current(B_mag, theta)           # (N,) A
    Jc_vol    = Ic_arr / (delta_SC * ic_model.tape_width)         # A/m²
    n_arr, _  = n_model.n_value(B_mag, theta)                     # (N,)

    # In-plane |J| (component perpendicular to n̂)
    J_dot_n = np.einsum("ij,ij->i", J_coil, n_hat)
    J_inplane = J_coil - J_dot_n[:, None] * n_hat
    Jmag = np.linalg.norm(J_inplane, axis=-1)

    # Floor regularisation: cells below the critical current are assigned
    # at least the critical-state resistivity (E_c/Jc at j_norm = 1).
    # eps_reg = 1.0 means the floor is E_c/Jc — the most physically
    # motivated choice.
    #
    # SMOOTH floor (ta_floor_smooth_p): a hard max(j/jc, eps) has a kink
    # at j = jc where dρ/dj jumps from 0 to (n−1)·ρ/j (huge for n ≈ 20).
    # Cells at the penetration front hover exactly there and flip states
    # every Picard iteration — a persistent |ΔB|/|B| ≈ 7e-4 oscillation
    # that neither T-relaxation nor ρ-relaxation damps (measured
    # 2026-07-10 with the High-Field dataset).  The soft-max
    # (eps^p + (j/jc)^p)^(1/p) is C^∞ and within ~2× of the hard floor at
    # the transition for p = 16 (exact away from it) — inside the floor's
    # own modelling uncertainty.  p = 0/None restores the hard max.
    jr = Jmag / Jc_vol
    p  = float(getattr(params, "ta_floor_smooth_p", 16.0) or 0.0)
    if p > 0:
        j_norm = (eps_reg**p + jr**p) ** (1.0 / p)
    else:
        j_norm = np.maximum(jr, eps_reg)

    # Power-law resistivity in the SC layer [Ω·m].
    # Clip the exponent argument to avoid overflow (j_norm >> 1 cells).
    # exp((n-1)*ln(j_norm)) is safe; direct power may overflow float64 for
    # j_norm > ~100 and n-1 > 30.
    log_j    = np.log(np.maximum(j_norm, 1e-30))
    rho_SC   = (1e-4 / Jc_vol) * np.exp((n_arr - 1.0) * log_j)

    # Scale to homogenised-volume resistivity
    rho_homog = rho_SC * (delta_SC / Lambda)

    # Log-space under-relaxation against the previous iterate (see docstring)
    if relax is not None and ta.get("_rho_prev") is not None:
        rho_homog = np.exp((1.0 - relax) * np.log(ta["_rho_prev"])
                           + relax * np.log(rho_homog))
    ta["_rho_prev"] = rho_homog

    ta["rho_fn"].x.array[:] = 0.0
    ta["rho_fn"].x.array[ta["coil_cells"]] = rho_homog
    ta["rho_fn"].x.scatter_forward()

    return Jmag, Jc_vol, n_arr


# ── J_s update ───────────────────────────────────────────────────────────────

def _update_Js(ta, J_all):
    """
    Set J_s_fn (DG0 vector) from an already-computed SC-layer current
    density J_all (N_coil, 3): J_s = (δ/Λ) × J_all.  Non-coil cells
    remain zero.
    """
    scale = ta["delta_SC"] / ta["Lambda"]
    Js_vals = ta["J_s_fn"].x.array.reshape(-1, 3)
    Js_vals[:] = 0.0
    Js_vals[ta["coil_cells"]] = J_all * scale
    ta["J_s_fn"].x.scatter_forward()


# ── Extract J at coil centroids from T ───────────────────────────────────────

def _J_from_T(ta, domain):
    """
    Evaluate ∇T × n̂ at coil cell centroids in the SC layer [A/m²].
    Returns (N_coil, 3) array for ALL coil cells.

    Replicated mode (per_layer=False): T is only solved in the central
    z-layer (coil_cells_T); J from the central layer is copied to every
    other coil cell via layer_map (nearest neighbour in the x-y plane).
    This assumes all z-layers share the central layer's geometry and
    field environment — an approximation for a non-uniform stack.

    Per-layer mode (per_layer=True): each layer has its own T solution;
    J is evaluated in each layer from its own ∇T — no replication.
    """
    gradT_fn = ta["gradT_fn"]

    if ta["per_layer"]:
        n_hat_all = ta["n_hat_coil"]
        J_coil = np.zeros((len(ta["coil_cells"]), 3))
        for i, (expr, idx_i) in enumerate(zip(ta["layer_gradT_exprs"],
                                              ta["layer_cell_idx"])):
            gradT_fn.interpolate(expr)
            g = gradT_fn.x.array.reshape(-1, 3)[ta["coil_cells"][idx_i]]
            J_coil[idx_i] = np.cross(g, n_hat_all[idx_i])
        return J_coil

    gradT_fn.interpolate(ta["gradT_expr"])
    gradT_all = gradT_fn.x.array.reshape(-1, 3)

    # J in the central-layer cells (shape: N_T × 3)
    gradT_central = gradT_all[ta["coil_cells_T"]]          # (N_T, 3)
    n_hat_T       = ta["n_hat_coil_T"]                     # (N_T, 3)
    J_central     = np.cross(gradT_central, n_hat_T)       # (N_T, 3)

    # Replicate to all coil cells using the pre-built layer_map
    # layer_map[i] is the index into coil_cells_T for the i-th coil cell
    J_coil    = J_central[ta["layer_map"]]                 # (N_all, 3)

    return J_coil


# ── Bore SCIF Biot-Savart (full two-coil system from quarter-domain cells) ──

def dB_bore_from_dJ(cents, dJ_s, dV, bore_pt=None):
    """
    Biot-Savart ΔB at the bore midplane from screening currents dJ_s
    (A/m², homogenised) defined on the QUARTER-coil cells of the
    eighth-symmetry FEM domain (x≥0, y≥0, coil 1).

    The full two-coil system is reconstructed by explicit mirror images
    (8 pieces = 4 quadrants × 2 coils).  The current density transforms
    under each mirror the way the physical loop current does:
      x-mirror  (−x, y, z):  J → (+Jx, −Jy, −Jz)
      y-mirror  (x, −y, z):  J → (−Jx, +Jy, −Jz)
      z-mirror  (x, y, 2g−z) [PMC image = coil 2]:  J → (+Jx, +Jy, −Jz)
    Omitting the quadrant images (as the pre-2026-07-09 code did)
    under-reports ΔBz by exactly 4×: every quadrant contributes the
    same Bz at an on-axis point (Bz is even under all the mirrors).
    """
    g = float(params.coil_half_gap)
    if bore_pt is None:
        bore_pt = np.array([0.0, 0.0, g])
    bore_pt = np.asarray(bore_pt, dtype=float).reshape(3)

    dB = np.zeros(3)
    for sx in (1, -1):
        for sy in (1, -1):
            c = cents.copy()
            c[:, 0] *= sx
            c[:, 1] *= sy
            J = dJ_s.copy()
            if sx < 0:
                J[:, 1] *= -1.0; J[:, 2] *= -1.0
            if sy < 0:
                J[:, 0] *= -1.0; J[:, 2] *= -1.0
            for coil in (1, 2):
                cc = c.copy()
                Jc = J
                if coil == 2:
                    cc[:, 2] = 2.0 * g - cc[:, 2]
                    Jc = J * np.array([1.0, 1.0, -1.0])
                r     = bore_pt[None, :] - cc
                r_mag = np.linalg.norm(r, axis=1)
                r_hat = r / np.maximum(r_mag, 1e-12)[:, None]
                dB   += np.sum(
                    (1e-7 * dV / np.maximum(r_mag, 1e-12)**2)[:, None]
                    * np.cross(Jc, r_hat), axis=0)
    return dB


# ── Build weak forms ─────────────────────────────────────────────────────────

def _build_problems(domain, ta, uniform_setup):
    """
    Build the T and A solvers ONCE and store them in the ta dict.

    T-problem: LinearProblem (MUMPS).  Its matrix depends on rho_fn,
    which changes every Picard iteration, so it must be reassembled and
    refactorised each solve — LinearProblem already does exactly that.

    A-problem: the bilinear form is CONSTANT (curl-curl + gauge
    regularisation, no coefficients that change), so we assemble the
    matrix and hand it to a bare PETSc KSP.  MUMPS factorises it on the
    first solve and reuses the factorisation for every later solve
    (every Picard iteration, every sweep current, and the uniform seed
    solve, which shares the same matrix) — only the RHS is reassembled.
    """
    from dolfinx.fem import petsc as fem_petsc

    dx        = ufl.Measure("dx", domain=domain)
    n_hat_ufl = ta["n_hat_ufl"]
    dt_val    = float(params.ramp_duration)
    dt_const  = fem.Constant(domain, PETSc.ScalarType(dt_val))

    # ── T-problem ────────────────────────────────────────────────────────
    T_trial = ufl.TrialFunction(ta["V_T"])
    phi_T   = ufl.TestFunction(ta["V_T"])

    J_T_trial = ufl.cross(ufl.grad(T_trial), n_hat_ufl)
    J_T_test  = ufl.cross(ufl.grad(phi_T),   n_hat_ufl)

    # Bilinear: weighted Laplacian in the n̂-perpendicular plane
    a_T = ta["rho_fn"] * ufl.inner(J_T_trial, J_T_test) * dx

    # Linear (RHS): BDF1 history term — B^k · φ n̂ / Δt
    # coil_ind restricts the integral to the coil domain.
    L_T = (-(1.0 / dt_const) * ta["coil_ind"] *
           ufl.inner(ufl.curl(ta["A_h"]), phi_T * n_hat_ufl) * dx)

    mumps_opts = {"ksp_type": "preonly",
                  "pc_type": "lu",
                  "pc_factor_mat_solver_type": "mumps"}

    if ta["per_layer"]:
        # One LinearProblem per z-layer: same bilinear/linear forms, only
        # the BC sets differ.  Each refactorises per solve (ρ changes every
        # iteration anyway — same as the single-tape mode).
        prob_T = None
        prob_T_layers = [
            LinearProblem(a_T, L_T, bcs=ta["layer_bcs"][i],
                          petsc_options_prefix=f"ta_T{i}_",
                          petsc_options=mumps_opts)
            for i in range(params.n_layers)]
    else:
        prob_T = LinearProblem(
            a_T, L_T,
            bcs=ta["all_T_bcs"],
            petsc_options_prefix="ta_T_",
            petsc_options=mumps_opts)
        prob_T_layers = None

    # ── A-problem (constant matrix → factorise once) ─────────────────────
    A_trial = ufl.TrialFunction(ta["V_A"])
    v_A     = ufl.TestFunction(ta["V_A"])

    a_A = ((1.0 / mu0) * ufl.inner(ufl.curl(A_trial), ufl.curl(v_A)) * dx
           + params.gauge_regularization * ufl.inner(A_trial, v_A) * dx)
    bc_A = uniform_setup["bc"]

    a_A_form = fem.form(a_A)
    A_mat = fem_petsc.assemble_matrix(a_A_form, bcs=[bc_A])
    A_mat.assemble()

    ksp_A = PETSc.KSP().create(domain.comm)
    ksp_A.setOperators(A_mat)
    ksp_A.setType("preonly")
    pc = ksp_A.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")

    # Two RHS forms sharing the one factorised matrix:
    #   L_A    — screening-current source (Picard iterations)
    #   L_seed — uniform-J source (seed solve; same J Function that
    #            solve.py rescales per current)
    L_A_form    = fem.form(ufl.inner(ta["J_s_fn"], v_A) * dx)
    L_seed_form = fem.form(ufl.inner(uniform_setup["J"], v_A) * dx)
    # dolfinx 0.11: create_vector takes the function space, not the form
    b_A = fem_petsc.create_vector(ta["V_A"])

    ta.update(prob_T=prob_T, prob_T_layers=prob_T_layers,
              a_A_form=a_A_form, bc_A=bc_A,
              ksp_A=ksp_A, A_mat=A_mat, b_A=b_A,
              L_A_form=L_A_form, L_seed_form=L_seed_form)


def _solve_A(ta, L_form):
    """
    Solve the A-problem with the cached factorisation: reassemble only
    the RHS from L_form, back-substitute into ta["A_h"].
    """
    from dolfinx.fem import petsc as fem_petsc

    b = ta["b_A"]
    with b.localForm() as bl:
        bl.set(0.0)
    fem_petsc.assemble_vector(b, L_form)
    fem_petsc.apply_lifting(b, [ta["a_A_form"]], bcs=[[ta["bc_A"]]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, [ta["bc_A"]])

    A_h = ta["A_h"]
    try:
        x_vec = A_h.x.petsc_vec
    except AttributeError:
        x_vec = A_h.vector   # older dolfinx attribute name
    ta["ksp_A"].solve(b, x_vec)
    A_h.x.scatter_forward()


# ── Picard solve ─────────────────────────────────────────────────────────────

def solve_ta_at_current(domain, ta, uniform_setup,
                         I_amps, ic_model, n_model, verbose=True,
                         warm_start=True):
    """
    Run the T-A Picard loop for transport current I_amps.
    Returns (A_h, B_h, T_h, info) at convergence, where info is a dict
    with n_iters (Picard iterations used), converged (bool), and
    rel_err (final raw |ΔB|/|B|) and drift (final EMA drift — the
    quantity compared against ta_picard_tol).

    warm_start: if True and a previous converged T is stored in ta
    (from an earlier call, e.g. the previous sweep current), seed T by
    scaling it with I_new/I_old instead of restarting from the ZFC
    state.  The T Dirichlet BCs are proportional to I, so the scaled T
    satisfies the new BCs exactly.  Falls back to the cold start when
    no previous state exists.
    """
    comm      = domain.comm
    delta_SC  = ta["delta_SC"]
    Lambda    = ta["Lambda"]
    eps_reg   = getattr(params, "ta_eps_reg",    1.0)
    n_picard  = getattr(params, "ta_n_picard",   40)
    tol       = getattr(params, "ta_picard_tol", 1e-3)

    per_layer = ta["per_layer"]
    prob_T    = ta["prob_T"]
    B_h       = ta["B_fn"]

    if verbose and comm.rank == 0:
        mode = "per-layer T" if per_layer else "replicated central-tape T"
        print(f"\n[T-A] I = {I_amps:.1f} A   Δt = {params.ramp_duration:.1f} s   "
              f"δ_SC = {delta_SC*1e6:.1f} µm   [{mode}]")

    # ── Set Dirichlet BC values on T ─────────────────────────────────────
    # T_bot > T_top so that J_x = -∂T/∂z > 0 in the top straight section
    T_amp = I_amps / (2.0 * delta_SC)
    ta["T_bot_val"].value = +T_amp
    ta["T_top_val"].value = -T_amp
    if verbose and comm.rank == 0:
        print(f"[T-A] T BCs: T_bot = +{T_amp:.3e},  T_top = -{T_amp:.3e}  "
              f"(single-tape z = [{ta['z_bot']*1e3:.1f}, {ta['z_top']*1e3:.1f}] mm)")

    prev_key = "_T_prev_layers" if per_layer else "_T_prev"
    warm = (warm_start and ta.get("_I_prev") is not None
            and ta.get(prev_key) is not None)
    if warm:
        # ── Warm start: T from previous converged state, scaled by I ────
        I_ratio = I_amps / ta["_I_prev"]
        if verbose and comm.rank == 0:
            print(f"[T-A] Warm start from I = {ta['_I_prev']:.1f} A "
                  f"(T scaled ×{I_ratio:.3f})")
        if per_layer:
            for T_i, prev in zip(ta["layer_T_fns"], ta[prev_key]):
                T_i.x.array[:] = prev * I_ratio
                T_i.x.scatter_forward()
        else:
            ta["T_h"].x.array[:] = ta[prev_key] * I_ratio
            ta["T_h"].x.scatter_forward()

        J_coil_prev = _J_from_T(ta, domain)      # (N_coil, 3)
        _update_Js(ta, J_coil_prev)
        _solve_A(ta, ta["L_A_form"])             # A from screening J
    else:
        # ── Cold start: seed A from the uniform-J solve, T = 0 (ZFC) ────
        if verbose and comm.rank == 0:
            print(f"[T-A] Seeding A from uniform-J solve ...")
        J_mag_homog = I_amps / (params.t * params.w)
        uniform_setup["J"].x.array[:] = (
            uniform_setup["J_unit_array"] * J_mag_homog)
        _solve_A(ta, ta["L_seed_form"])          # same matrix, uniform RHS

        ta["T_h"].x.array[:] = 0.0
        ta["T_h"].x.scatter_forward()
        for T_i in ta["layer_T_fns"]:
            T_i.x.array[:] = 0.0
            T_i.x.scatter_forward()

        # Seed J is the tangent-aligned uniform SC-layer current density
        J_mag_SC = I_amps / (delta_SC * params.w)    # |J| in the SC layer
        J_coil_prev = ta["t_hat_coil"] * J_mag_SC    # (N_coil, 3)
        _update_Js(ta, J_coil_prev)

    # ── B from seed solve (needed for Jc(B) at first ρ computation) ─────
    B_h.interpolate(ta["curl_expr"])
    B_coil_prev = B_h.x.array.reshape(-1, 3)[ta["coil_cells"]].copy()

    if verbose and comm.rank == 0:
        Bmag_mean = np.mean(np.linalg.norm(B_coil_prev, axis=-1))
        print(f"[T-A] Seed |B| mean at coil cells: {Bmag_mean:.4f} T")

    # Compute initial ρ from seed J + B  (CRITICAL — avoids ρ → 0)
    ta["_rho_prev"] = None    # fresh relaxation history per solve
    Jmag0, Jc0, n0 = _update_rho(ta, J_coil_prev, B_coil_prev,
                                  ic_model, n_model, eps_reg)
    if verbose and comm.rank == 0:
        rho_vals = ta["rho_fn"].x.array[ta["coil_cells"]]
        print(f"[T-A] Initial ρ_homog: min={rho_vals.min():.2e}  "
              f"max={rho_vals.max():.2e}  mean={rho_vals.mean():.2e} Ω·m")
        print(f"[T-A] Seed j/jc mean: {np.mean(Jmag0/Jc0):.3f}  "
              f"n mean: {np.mean(n0):.2f}")

    # ── Picard loop ───────────────────────────────────────────────────────
    B_prev = B_coil_prev.copy()
    converged = False
    t0_picard = time.time()

    # Relaxation control (settled 2026-07-11 after the High-Field dataset
    # exposed every failure mode of fancier schemes):
    #   Phase 1: α = alpha_high (0.30) for the ramp-up, until |ΔB| first
    #            stops decreasing.
    #   Phase 2: α = alpha_low FIXED.  With the smooth j/jc floor
    #            (ta_floor_smooth_p) and log-space ρ relaxation
    #            (ta_rho_relax) a fixed moderate α converges cleanly —
    #            measured: SCIF frozen to <0.01 mT within ~60 iterations.
    #            Aggressive α-throttling misreads slow physical transients
    #            (broad marginal flux fronts, j/jc ≈ 1 over ~30% of cells)
    #            as stalls and freezes them; α-raising re-excites cycles.
    # No mid-flight α throttling: every adaptive scheme tried (magnitude
    # stall detection, correlation-gated halving with dwell) misfired on
    # transiently anti-correlated updates during normal convergence and
    # froze the iteration at α ≈ 0.01.  corr is still computed and printed
    # as a diagnostic.
    alpha_high  = float(getattr(params, "ta_picard_alpha",       0.30))
    alpha_low   = float(getattr(params, "ta_picard_alpha_fine",  0.15))
    alpha_relax = alpha_high
    phase2      = False
    prev_dB_vec = None
    prev_dB_mag = np.inf

    # Convergence is judged on the OBSERVABLE, not on the raw B vector.
    # On sharp-front cases the front configuration wanders chaotically
    # among near-degenerate states: raw |ΔB|/|B| floors at ~6-10e-4 with a
    # red spectrum, so no B-space filter converges — while the integrated
    # quantities (bore SCIF, bore field, j/jc) are frozen to <0.01 mT
    # (measured 2026-07-11: SCIF = +81.98 mT identically across runs and
    # hundreds of iterations).  Criterion: the EMA-smoothed bore SCIF must
    # move less than ta_scif_stall_mT over a 10-iteration window.
    # The B-vector EMA drift is still computed and reported as diagnostic.
    B_ema     = None
    ema_hist  = []
    scif_ema  = None
    scif_hist = []
    scif_tol  = float(getattr(params, "ta_scif_stall_mT", 0.05))
    J_unif_vec = ta["t_hat_coil"] * (I_amps / (delta_SC * params.w))

    for k in range(n_picard):

        # ── (a) Solve T-equation(s) (linear with frozen ρ, A) ────────────
        if per_layer:
            for i, (T_i, prob_i) in enumerate(zip(ta["layer_T_fns"],
                                                   ta["prob_T_layers"])):
                T_old_i = T_i.x.array.copy()
                T_sol_i = prob_i.solve()
                T_i.x.array[:] = ((1.0 - alpha_relax) * T_old_i
                                  + alpha_relax * T_sol_i.x.array[:])
                T_i.x.scatter_forward()
                if np.any(np.isnan(T_i.x.array)):
                    raise RuntimeError(
                        f"[T-A k={k+1}] T solve produced NaN in layer {i}.")
        else:
            T_old = ta["T_h"].x.array.copy()
            T_sol = prob_T.solve()

            # Relaxed update: T_h ← (1-α)·T_old + α·T_new
            ta["T_h"].x.array[:] = (
                (1.0 - alpha_relax) * T_old + alpha_relax * T_sol.x.array[:]
            )
            ta["T_h"].x.scatter_forward()

            if np.any(np.isnan(ta["T_h"].x.array)):
                raise RuntimeError(
                    f"[T-A k={k+1}] T solve produced NaN. "
                    "Check Dirichlet BCs, rho_fn values, and mesh.")

        # ── (b) Compute J from T at coil centroids (all layers) ──────────
        J_coil = _J_from_T(ta, domain)   # (N_coil, 3), replicated from central layer

        # ── (c) Update J_s_fn and solve A-equation (cached factorisation) ─
        _update_Js(ta, J_coil)
        _solve_A(ta, ta["L_A_form"])

        if np.any(np.isnan(ta["A_h"].x.array)):
            raise RuntimeError(
                f"[T-A k={k+1}] A solve produced NaN.")

        # ── (d) Recompute B ───────────────────────────────────────────────
        B_h.interpolate(ta["curl_expr"])
        B_coil = B_h.x.array.reshape(-1, 3)[ta["coil_cells"]]

        # ── (e) Check convergence ─────────────────────────────────────────
        # α-normalisation: with relaxation, per-iteration |ΔB| scales ~α
        # for the same distance to the fixed point, so shrinking α below
        # the historical fine value (0.08) must NOT make convergence
        # easier.  The factor is 1 for α ≥ 0.08 (legacy behaviour) and
        # α_fine/α for smaller α.
        dB_vec   = (B_coil - B_prev).ravel()
        dB       = np.linalg.norm(dB_vec)
        B_norm   = np.linalg.norm(B_coil) + 1e-30
        rel_err  = (dB / B_norm) * max(1.0, alpha_low / alpha_relax)

        # B-vector EMA drift — diagnostic only (see header comment)
        if B_ema is None:
            B_ema = B_coil.copy()
        else:
            B_ema = 0.9 * B_ema + 0.1 * B_coil
        ema_hist.append(B_ema.copy())
        ema_hist = ema_hist[-11:]
        if len(ema_hist) == 11:
            drift = np.linalg.norm(ema_hist[-1] - ema_hist[0]) / B_norm
        else:
            drift = np.inf

        # Observable stall — the actual convergence criterion: EMA-smoothed
        # bore SCIF must move < scif_tol [mT] over the last 10 iterations.
        dBz_now = dB_bore_from_dJ(
            ta["coil_centroids"],
            (J_coil - J_unif_vec) * (delta_SC / Lambda),
            ta["coil_vols"])[2] * 1e3                       # [mT]
        scif_ema = dBz_now if scif_ema is None else \
            0.8 * scif_ema + 0.2 * dBz_now
        scif_hist.append(scif_ema)
        scif_hist = scif_hist[-11:]
        scif_stall = (abs(scif_hist[-1] - scif_hist[0])
                      if len(scif_hist) == 11 else np.inf)

        # ── Relaxation control (see header comment) ──────────────────────
        if prev_dB_vec is not None and dB > 0:
            corr = float(np.dot(dB_vec, prev_dB_vec)
                         / (dB * np.linalg.norm(prev_dB_vec) + 1e-300))
        else:
            corr = 1.0
        if not phase2 and k >= 4 and dB >= 0.95 * prev_dB_mag:
            phase2 = True
            alpha_relax = alpha_low
            if verbose and comm.rank == 0:
                print(f"  [T-A] ramp-up done — fixed fine relaxation "
                      f"α={alpha_low:.2f}")
        prev_dB_vec = dB_vec
        prev_dB_mag = dB

        if verbose and comm.rank == 0:
            # Diagnostics on the central layer only (where T is solved)
            unique_T_idx = ta["unique_T_idx"]          # indices into coil_cells
            J_diag  = J_coil[unique_T_idx]             # central layer J rows
            B_diag  = B_coil[unique_T_idx]
            n_diag  = ta["n_hat_coil"][unique_T_idx]
            Jmag_d  = np.linalg.norm(J_diag, axis=-1)
            B_mag_d = np.linalg.norm(B_diag, axis=-1)
            theta_d = angle_with_normal_deg(B_diag, n_diag)
            Jc_now, _ = ic_model.critical_current(B_mag_d, theta_d)
            Jc_vol    = Jc_now / (delta_SC * ic_model.tape_width)
            j_over_jc = np.mean(Jmag_d / (Jc_vol + 1e-30))
            if per_layer:
                T_min = min(T_i.x.array.min() for T_i in ta["layer_T_fns"])
                T_max = max(T_i.x.array.max() for T_i in ta["layer_T_fns"])
            else:
                T_arr = ta["T_h"].x.array
                T_min, T_max = T_arr.min(), T_arr.max()
            st = f"{scif_stall:.3f}" if np.isfinite(scif_stall) else " -- "
            print(f"  [k={k+1:02d}] SCIF = {scif_ema:+8.2f} mT "
                  f"(stall {st})  |ΔB|/|B| = {rel_err:.2e}  "
                  f"⟨|J|/Jc⟩_central = {j_over_jc:.3f}  corr = {corr:+.2f}")

        # ── (f) Update ρ from new J and B (log-space under-relaxed) ─────
        _update_rho(ta, J_coil, B_coil, ic_model, n_model, eps_reg,
                    relax=float(getattr(params, "ta_rho_relax", 0.5)))

        B_prev = B_coil.copy()

        if scif_stall < scif_tol and k >= 25:
            converged = True
            if verbose and comm.rank == 0:
                print(f"[T-A] Converged at k={k+1} "
                      f"({time.time()-t0_picard:.1f} s)")
            break

    if not converged and verbose and comm.rank == 0:
        print(f"[T-A] WARNING: did not converge in {n_picard} iterations "
              f"(SCIF stall = {scif_stall:.3f} mT/10it vs tol "
              f"{scif_tol:.3f}; raw |ΔB|/|B| = {rel_err:.2e})")

    # Store converged state for warm-starting the next current
    if per_layer:
        ta["_T_prev_layers"] = [T_i.x.array.copy() for T_i in ta["layer_T_fns"]]
        # Combined T for output/plots.  The layer fields overlap only at
        # the pinned interface nodes (where +I/2δ and −I/2δ sum to 0);
        # interior nodes belong to exactly one layer.
        ta["T_h"].x.array[:] = np.sum(
            [T_i.x.array for T_i in ta["layer_T_fns"]], axis=0)
        ta["T_h"].x.scatter_forward()
    else:
        ta["_T_prev"] = ta["T_h"].x.array.copy()
    ta["_I_prev"] = float(I_amps)

    info = dict(n_iters=k + 1, converged=converged, rel_err=float(rel_err),
                drift=float(drift), scif_mT=float(scif_ema),
                scif_stall_mT=float(scif_stall))
    return ta["A_h"], B_h, ta["T_h"], info


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Reuse solve.py's mesh-loading and problem-setup logic exactly —
    # avoids duplicating the gmsh import alias and MeshData handling.
    from dolfinx.io import gmsh as gmshio   # correct alias for this env
    import solve as base_solve
    import build_mesh

    comm = MPI.COMM_WORLD

    # Build mesh if needed (same call as solve.py main())
    if comm.rank == 0:
        build_mesh.build(write_path=params.mesh_filename)
    comm.barrier()

    mesh_data  = gmshio.read_from_msh(
        params.mesh_filename, comm, rank=0, gdim=3)
    domain     = mesh_data.mesh
    cell_tags  = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags

    ic_mod = IcModel(csv_path=params.shanghai_csv_filename)
    n_mod  = NValueModel(csv_path=params.n_value_csv_filename)

    # Uniform-J A-form setup (provides V_A, bc, coil_cells, J_unit_array)
    uniform_setup = base_solve.setup_problem(domain, cell_tags, facet_tags)

    # T-A problem setup
    ta = setup_ta_problem(domain, cell_tags, facet_tags, uniform_setup)

    A_h, B_h, T_h, _info = solve_ta_at_current(
        domain, ta, uniform_setup,
        I_amps   = params.I_design,
        ic_model = ic_mod,
        n_model  = n_mod,
        verbose  = True,
    )

    if comm.rank == 0:
        coil_cells = ta["coil_cells"]
        centroids  = dmesh.compute_midpoints(
            domain, domain.topology.dim, coil_cells)
        B_coil = B_h.x.array.reshape(-1, 3)[coil_cells]

        # ── Screening-current J from T ────────────────────────────────────
        # J_TA  = ∇T × n̂  (replicated to all coil cells from central layer)
        # J_unif = I / (δ_SC × n_layers × w)  in the t̂ direction (uniform J)
        delta_SC  = float(params.delta_SC)
        n_layers  = int(params.n_layers)
        w_tape    = float(params.w)
        J_TA_arr  = _J_from_T(ta, domain)              # (N, 3)  A/m² in SC layer
        # Uniform-J reference for ΔJ computation.
        # J_unif is the per-tape SC-layer current density (A/m²): I/(δ_SC × w).
        # This is what the T field gives in the ABSENCE of screening currents
        # (linear T profile from -T_amp to +T_amp across tape width w).
        # Do NOT divide by n_layers here — n_layers is already accounted for
        # by the layer_map replication that spreads J_central to all layers.
        # Using I/(δ_SC × n_layers × w) would give a 7× underestimate of the
        # uniform baseline, making ΔJ dominated by transport current, not SCIF.
        J_unif_mag = params.I_design / (delta_SC * w_tape)   # ~50 GA/m²
        J_unif_arr = ta["t_hat_coil"] * J_unif_mag           # (N, 3) per-tape uniform J
        dJ_arr     = J_TA_arr - J_unif_arr             # (N, 3)  screening ΔJ in SC layer

        # Convert to homogenised volume current density for Biot-Savart:
        #   J_s = (δ_SC / Λ) × J_SC   where Λ = t (turn pitch)
        Lambda     = float(params.t)
        scale      = delta_SC / Lambda
        dJ_s       = dJ_arr * scale                     # (N, 3)  A/m²

        # Exact per-cell volumes (DG0 CellVolume, computed once in setup)
        dV = ta["coil_vols"]                           # (N,) m³

        # ── Bore SCIF via cell Biot-Savart ────────────────────────────────
        # Full two-coil system reconstructed by mirror images (4 quadrants
        # × 2 coils) — see dB_bore_from_dJ.
        dB_bore = dB_bore_from_dJ(centroids, dJ_s, dV)
        bore_pt = np.array([[0.0, 0.0, float(params.coil_half_gap)]])

        # ── Mean |B| at bore from uniform-J (Biot-Savart reference) ──────
        from coil2_field import compute_both_coils_field
        B_bore_uniform, _, _ = compute_both_coils_field(
            bore_pt, I_per_turn=params.I_design)
        Bz_bore_uniform = float(B_bore_uniform[0, 2])
        Bz_bore_TA      = Bz_bore_uniform + float(dB_bore[2])

        # ── Save ──────────────────────────────────────────────────────────
        out = os.path.join(params.SOLVE_DIR, "racetrack_ta_fields.npz")
        np.savez(out,
                 I_solved       = params.I_design,
                 ramp_duration  = params.ramp_duration,
                 delta_SC       = params.delta_SC,
                 coil_centroids = centroids,
                 coil_B         = B_coil,
                 T_field        = T_h.x.array.copy(),
                 J_TA_coil      = J_TA_arr,
                 J_unif_coil    = J_unif_arr,
                 dB_bore        = dB_bore,
                 Bz_bore_uniform= Bz_bore_uniform,
                 Bz_bore_TA     = Bz_bore_TA)
        print(f"\nSaved T-A fields to {out}")

        # ── Console summary ───────────────────────────────────────────────
        dBz_coil = B_coil[:, 2] - np.mean(B_coil[:, 2])
        print(f"\n{'='*55}")
        print(f"  SCREENING CURRENT SUMMARY")
        print(f"{'='*55}")
        print(f"  Transport current I     = {params.I_design:.1f} A")
        print(f"  Ramp duration Δt        = {params.ramp_duration:.0f} s")
        print(f"  Uniform-J bore Bz       = {Bz_bore_uniform:+.4f} T")
        print(f"  SCIF ΔBz at bore        = {dB_bore[2]*1e3:+.2f} mT")
        print(f"  SCIF |ΔB| at bore       = {np.linalg.norm(dB_bore)*1e3:.2f} mT")
        print(f"  T-A bore Bz             = {Bz_bore_TA:+.4f} T")
        print(f"  Relative SCIF           = {abs(dB_bore[2])/abs(Bz_bore_uniform)*100:.3f}%")
        print(f"{'='*55}")

        # ── J-map in central z-layer ──────────────────────────────────────
        tol_z   = w_tape / 8.0
        mask_cl = np.abs(centroids[:, 2]) <= w_tape / 2.0 + tol_z
        if mask_cl.sum() >= 3:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xy   = centroids[mask_cl, :2] * 1e3
            Jx   = J_TA_arr[mask_cl, 0]
            Jx_u = J_unif_arr[mask_cl, 0]
            dJx  = Jx - Jx_u

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            for ax_i, (vals, title, cmap) in zip(axes, [
                (Jx   / 1e9, "J_x (T-A)  [GA/m²]",       "RdBu_r"),
                (Jx_u / 1e9, "J_x (uniform)  [GA/m²]",   "RdBu_r"),
                (dJx  / 1e6, "ΔJ_x = J_TA − J_unif  [MA/m²]", "RdBu_r"),
            ]):
                vmax = np.abs(vals).max() or 1.0
                sc = ax_i.scatter(xy[:, 0], xy[:, 1], c=vals,
                                  cmap=cmap, s=8, linewidths=0,
                                  vmin=-vmax, vmax=vmax)
                fig.colorbar(sc, ax=ax_i, label=title)
                ax_i.set_xlabel("x [mm]"); ax_i.set_ylabel("y [mm]")
                ax_i.set_title(title); ax_i.set_aspect("equal")
                ax_i.grid(True, alpha=0.2)

            fig.suptitle(
                f"Central z-layer screening currents  "
                f"(I = {params.I_design:.0f} A, Δt = {params.ramp_duration:.0f} s)",
                fontsize=11)
            plt.tight_layout()
            jmap_path = os.path.join(params.SOLVE_DIR, "ta_J_central_layer.png")
            fig.savefig(jmap_path, dpi=150)
            plt.close(fig)
            print(f"Saved J_x map → {jmap_path}")

    return A_h, B_h, T_h


if __name__ == "__main__":
    main()
