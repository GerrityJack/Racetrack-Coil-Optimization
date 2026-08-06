"""
monolithic_jacobian_inspect.py -- 2026-08-05 overnight follow-up.

Directly inspects the assembled block Jacobian of monolithic_ta.py's
system (original, frozen Jc/n -- the same pathology was already confirmed
present here, so no need to also drag in the differentiable version for
this check) at the point where the residual has already exploded (a few
outer iterations in, from the standard 30-iteration bootstrap), rather
than continuing to infer its badness indirectly from linesearch ynorm
values. Checks: overall condition number, per-block (T_0..T_5 vs A)
diagonal scale, and off-diagonal T-A coupling block scale.

Uses the CHEAP dt=60s/I=19.6A case (already confirmed to show the
identical pathology as the expensive dt=600s/I=196A production case) for
speed.
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
    n_outer_before_inspect = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    import numpy as np
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    import monolithic_ta
    from ic_model import IcModel, NValueModel
    from newton_ta import ta_transient_seed_cold, _picard_bootstrap

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_jacinspect_{os.getpid()}{ext}"
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
    B_coil = B_now

    print(f"Running {n_outer_before_inspect} outer iterations to reach the "
          f"already-exploded regime before inspecting...", flush=True)
    for k in range(n_outer_before_inspect):
        for layer in range(6):
            monolithic_ta._update_mono_coefficients(ta, ic, nm, layer, B_coil, relax=0.3)
        T_snap = [T_i.x.array.copy() for T_i in ta["layer_T_fns"]]
        A_snap = ta["A_h"].x.array.copy()
        problem.solve()
        fnorm = snes.getFunction()[0].norm()
        print(f"  outer={k+1}: ||F||={fnorm:.4e}  reason={snes.getConvergedReason()}",
              flush=True)
        for T_i, snap in zip(ta["layer_T_fns"], T_snap):
            T_i.x.array[:] = 0.7 * snap + 0.3 * T_i.x.array
            T_i.x.scatter_forward()
        ta["A_h"].x.array[:] = 0.7 * A_snap + 0.3 * ta["A_h"].x.array
        ta["A_h"].x.scatter_forward()
        J_coil = ta_solve._J_from_T(ta, domain)
        ta_solve._update_Js(ta, J_coil)
        ta["B_fn"].interpolate(ta["curl_expr"])
        B_coil = ta["B_fn"].x.array.reshape(-1, 3)[ta["coil_cells"]]

    print("\n" + "=" * 78)
    print("INSPECTING THE ASSEMBLED JACOBIAN AT THIS (already-diverged) STATE")
    print("=" * 78)

    # One more coefficient refresh (matching what the next outer iteration
    # would do) then assemble F/J at the CURRENT state without taking a step.
    for layer in range(6):
        monolithic_ta._update_mono_coefficients(ta, ic, nm, layer, B_coil, relax=0.3)

    x = snes.getSolution()
    Fvec = snes.getFunction()[0]
    Jmat, Pmat = snes.getJacobian()[0:2]
    snes.computeFunction(x, Fvec)
    snes.computeJacobian(x, Jmat, Pmat)
    print(f"||F(x)|| at inspection point = {Fvec.norm():.6e}")

    # Block sizes: V_T (per layer, but they share ONE function space -- each
    # layer's T_i lives in the SAME V_T, so each contributes V_T's full dof
    # count to the concatenated block vector) and V_A.
    n_T = ta["V_T"].dofmap.index_map.size_local
    n_A = ta["V_A"].dofmap.index_map.size_local
    print(f"\nBlock sizes: n_T (per layer) = {n_T}, n_A = {n_A}, "
          f"6 T-layers + A = {6*n_T + n_A} total dofs")

    ai, aj, av = Jmat.getValuesCSR()
    from scipy.sparse import csr_matrix
    J_sp = csr_matrix((av, aj, ai))
    print(f"Jacobian shape: {J_sp.shape}, nnz={J_sp.nnz}")

    # Per-block diagonal scale (offsets: T_0=[0,n_T), T_1=[n_T,2n_T), ...,
    # A=[6*n_T, 6*n_T+n_A))
    diag = J_sp.diagonal()
    offsets = [i * n_T for i in range(7)] + [6 * n_T + n_A]
    names = [f"T_{i}" for i in range(6)] + ["A"]
    print("\nPer-block diagonal magnitude (min/mean/max of |diag| within each block):")
    for i, name in enumerate(names):
        lo, hi = offsets[i], offsets[i + 1]
        d = np.abs(diag[lo:hi])
        d_nz = d[d > 0]
        if d_nz.size:
            print(f"  {name:4s} [{lo:6d}:{hi:6d}]  min={d_nz.min():.3e}  "
                  f"mean={d_nz.mean():.3e}  max={d_nz.max():.3e}")
        else:
            print(f"  {name:4s} [{lo:6d}:{hi:6d}]  ALL ZERO DIAGONAL")

    # Off-diagonal T-A coupling block scale: rows in the A block, columns in
    # a T block (dF_A/dT_i), and vice versa (dF_Ti/dA).
    A_lo, A_hi = offsets[6], offsets[7]
    print("\nOff-diagonal coupling block magnitudes (nonzero entries only):")
    for i in range(6):
        T_lo, T_hi = offsets[i], offsets[i + 1]
        block_AT = J_sp[A_lo:A_hi, T_lo:T_hi]
        block_TA = J_sp[T_lo:T_hi, A_lo:A_hi]
        for label, blk in [(f"dF_A/dT_{i}", block_AT), (f"dF_T{i}/dA", block_TA)]:
            data = blk.data
            if data.size:
                print(f"  {label:12s}: nnz={data.size:6d}  "
                      f"min|.|={np.abs(data).min():.3e}  "
                      f"max|.|={np.abs(data).max():.3e}")
            else:
                print(f"  {label:12s}: EMPTY (no coupling entries at all!)")

    # Condition number: this system is small enough (a few thousand to
    # ~1e4 dofs) for a dense conversion + SVD-based estimate to be
    # tractable, but do it via sparse SVD (a few extreme singular values
    # only) to avoid an expensive full dense solve if the matrix turns out
    # larger than expected.
    print(f"\nEstimating condition number (this may take a moment for "
          f"n={J_sp.shape[0]})...", flush=True)
    from scipy.sparse.linalg import svds
    try:
        # largest singular value
        s_max = svds(J_sp.asfptype(), k=1, which="LM", return_singular_vectors=False)
        # smallest singular value (shift-invert-free estimate; may be slow
        # or fail to converge for a genuinely near-singular matrix, which
        # is itself diagnostic)
        s_min = svds(J_sp.asfptype(), k=1, which="SM", return_singular_vectors=False)
        print(f"  sigma_max ~ {s_max[0]:.6e}")
        print(f"  sigma_min ~ {s_min[0]:.6e}")
        print(f"  condition number estimate ~ {s_max[0]/max(s_min[0],1e-300):.6e}")
    except Exception as e:
        print(f"  svds failed/did not converge: {e!r} -- itself consistent "
              f"with a genuinely near-singular or very poorly scaled matrix.")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
