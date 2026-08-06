"""
monolithic_fixed_bc_verify.py -- confirms the distinct-per-layer-space fix
(monolithic_fixed_bc_test.py) actually produces nonzero T-A coupling and a
non-trivial (non-identity) T-diagonal in the assembled Jacobian, the same
check monolithic_jacobian_inspect.py/2.py ran against the buggy version.
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
from monolithic_fixed_bc_test import build_fixed_monolithic_problem  # noqa: E402


def main():
    import numpy as np
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio
    from scipy.sparse import csr_matrix

    import params
    import build_mesh
    import solve as base_solve
    import ta_solve
    from ic_model import IcModel, NValueModel, angle_with_normal_deg
    from newton_ta import ta_transient_seed_cold, _picard_bootstrap

    comm = MPI.COMM_WORLD
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_monofixverify_{os.getpid()}{ext}"
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

    fx = build_fixed_monolithic_problem(ta, domain, verbose=False)

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
    snes = fx["snes"]
    for _ in range(2):
        problem.solve()

    x = snes.getSolution()
    Fvec = snes.getFunction()[0]
    Jmat, Pmat = snes.getJacobian()[0:2]
    snes.computeFunction(x, Fvec)
    snes.computeJacobian(x, Jmat, Pmat)
    print(f"||F|| = {Fvec.norm():.6e}")
    print("Jmat type:", Jmat.getType())

    n_T = fx["V_T_list"][0].dofmap.index_map.size_local
    n_A = ta["V_A"].dofmap.index_map.size_local
    ai, aj, av = Jmat.getValuesCSR()
    J_sp = csr_matrix((av, aj, ai))
    print(f"Jacobian shape: {J_sp.shape}")

    diag = J_sp.diagonal()
    offsets = [i * n_T for i in range(7)] + [6 * n_T + n_A]
    names = [f"T_{i}" for i in range(6)] + ["A"]
    print("\nPer-block diagonal magnitude:")
    for i, name in enumerate(names):
        lo, hi = offsets[i], offsets[i + 1]
        d = np.abs(diag[lo:hi])
        d_nz = d[d > 0]
        n_exactly_1 = int(np.sum(d == 1.0))
        if d_nz.size:
            print(f"  {name:4s}  min={d_nz.min():.3e}  mean={d_nz.mean():.3e}  "
                  f"max={d_nz.max():.3e}  n_exactly_1.0={n_exactly_1}/{len(d)}")

    A_lo, A_hi = offsets[6], offsets[7]
    print("\nOff-diagonal T-A coupling blocks:")
    for i in range(6):
        T_lo, T_hi = offsets[i], offsets[i + 1]
        b_AT = J_sp[A_lo:A_hi, T_lo:T_hi]
        b_TA = J_sp[T_lo:T_hi, A_lo:A_hi]
        for label, blk in [(f"dF_A/dT_{i}", b_AT), (f"dF_T{i}/dA", b_TA)]:
            if blk.nnz:
                print(f"  {label:12s}: nnz={blk.nnz:6d}  "
                      f"max|.|={np.abs(blk.data).max():.3e}  "
                      f"nonzero_count={int(np.sum(np.abs(blk.data) > 1e-300))}")
            else:
                print(f"  {label:12s}: EMPTY")

    try:
        os.remove(params.mesh_filename)
    except OSError:
        pass


if __name__ == "__main__":
    main()
