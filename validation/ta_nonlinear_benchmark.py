"""
ta_nonlinear_benchmark.py -- error hunt 2026-10-04: is the T-A solver right
in the NONLINEAR (power-law) regime that sets every load?
==============================================================================
validation/ta_rho_scaling_benchmark.py checked the solver against an
independent discrete-tape reference in the OHMIC limit (n = 1). The loads
that decide designs live in the strongly nonlinear regime (n ~ 21, cells
near Jc), which that test did not touch. Same benchmark geometry (single
layer, 40 tapes, long straight section), same reference model (each tape a
zero-thickness sheet, M elements across its width, exact log-kernel mutual
inductances, images for the other leg and coil 2), now with

    E(J) = E_c * u * g(|u|)^(n-1),   u = J / Jc,   n = 21, Jc constant,

solved for ONE backward-Euler step from the virgin state by Newton's method
with continuation in n. Two references:
  floor : g = (eps^p + |u|^p)^(1/p) with eps = RHO_FLOOR, p = 16 -- exactly
          the regularised resistivity ta_solve._update_rho uses, so FEM vs
          this tests the IMPLEMENTATION;
  pure  : g = |u| (eps = 1e-3 only to keep the Jacobian regular) -- the
          physical answer, so (floor - pure) is the error the floor causes.
FEM: production code path, stub Ic (constant) and n (= 21) models, floor
ladder 1.0 -> RHO_FLOOR on one mesh.

Compared: worst |J|/Jc (the design criterion) and the stack-mean |J|/Jc
profile across the tape width, in the straight section near x = 0.

Env: NB_DT (s, default chosen from the reference scan), NB_I (A),
     NB_FLOOR (default 0.6), NB_REF_ONLY=1 (reference scan only).
Output: validation/ta_nonlinear_benchmark.csv
"""
import os
import sys
import csv
import time

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VAL = os.path.join(_ROOT, "validation")
for _p in (_ROOT, _VAL, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
import ta_rho_scaling_benchmark as lin

E_C = 1e-4
N_EXP = 21.0
P_SOFT = 16.0
IC_A = lin.IC_A                      # 100 A per tape
I_A = float(os.environ.get("NB_I", 50.0))
FLOOR = float(os.environ.get("NB_FLOOR", 0.6))
OUT_CSV = os.path.join(_VAL, "ta_nonlinear_benchmark.csv")


def _kernel(M):
    """Element geometry and the (n_el x n_el) inductance matrix [H/m]."""
    w, lam, dsc = params.w, params.t, params.delta_SC
    N = lin.N_TAPES
    y = params.a_inner_list[0] + (np.arange(N) + 0.5) * lam
    zb = params.layer_z_bottoms[0]
    edges = zb + w * np.linspace(0.0, 1.0, M + 1)
    za, zt = edges[:-1], edges[1:]
    g = params.coil_half_gap
    Y = np.repeat(y, M)
    ZA, ZT = np.tile(za, N), np.tile(zt, N)

    def block(y_src, za_src, zt_src, sign):
        d = np.abs(Y[:, None] - y_src[None, :])
        return sign * lin._mean_log(ZA[:, None], ZT[:, None],
                                    za_src[None, :], zt_src[None, :], d)
    K = (block(Y, ZA, ZT, +1) + block(-Y, ZA, ZT, -1)
         + block(Y, 2 * g - ZT, 2 * g - ZA, +1)
         + block(-Y, 2 * g - ZT, 2 * g - ZA, -1))
    L = -(lin.MU0 / (2 * np.pi)) * K
    area = dsc * np.tile(zt - za, N)
    return 0.5 * (za + zt), L, area


def _F(u, n, eps):
    """E/E_c and its derivative for the (soft-floored) power law."""
    au = np.abs(u)
    s = (eps ** P_SOFT + au ** P_SOFT)
    gg = s ** (1.0 / P_SOFT)
    dg = np.sign(u) * au ** (P_SOFT - 1.0) * s ** (1.0 / P_SOFT - 1.0)
    F = u * gg ** (n - 1.0)
    dF = gg ** (n - 1.0) + u * (n - 1.0) * gg ** (n - 2.0) * dg
    return F, dF


def reference(dt, I, eps, M=40, verbose=False):
    """One BE step from the virgin state. Returns (z, u[N, M]) with
    u = J/Jc, by Newton with continuation n = 1 -> 21."""
    N = lin.N_TAPES
    z, L, area = _kernel(M)
    Jc = IC_A / (params.delta_SC * params.w)
    ne = N * M
    tape = np.repeat(np.arange(N), M)
    # residual scaled by E_c:  F(u) + (Jc/(E_c dt)) L (area*u) - e_k = 0
    Lsc = (Jc / (E_C * dt)) * L * area[None, :]
    u = np.full(ne, I / IC_A)
    e = np.zeros(N)
    for n in (1, 2, 3, 5, 8, 12, 16, 19, 21):
        for it in range(200):
            F, dF = _F(u, n, eps)
            R = F + Lsc @ u - e[tape]
            C = (area * Jc * u).reshape(N, M).sum(1) - I
            res = np.concatenate([R, C / (IC_A * 1e-3)])
            nr = np.linalg.norm(res)
            if nr < 1e-10 * max(1.0, np.sqrt(ne)):
                break
            Jm = np.zeros((ne + N, ne + N))
            Jm[:ne, :ne] = Lsc + np.diag(dF)
            Jm[np.arange(ne), ne + tape] = -1.0
            Jm[ne + tape, np.arange(ne)] = area * Jc / (IC_A * 1e-3)
            step = np.linalg.solve(Jm, -res)
            lam = 1.0
            for _ in range(30):                         # backtracking
                un, en = u + lam * step[:ne], e + lam * step[ne:]
                Fn, _ = _F(un, n, eps)
                Rn = Fn + Lsc @ un - en[tape]
                Cn = (area * Jc * un).reshape(N, M).sum(1) - I
                if np.linalg.norm(np.concatenate([Rn, Cn / (IC_A * 1e-3)])) \
                        < (1 - 1e-4 * lam) * nr:
                    break
                lam *= 0.5
            u, e = un, en
        if verbose:
            print(f"    n={n:4.0f}: {it+1:3d} Newton its, |res| {nr:.2e}, "
                  f"max|u| {np.abs(u).max():.4f}")
    return z, u.reshape(N, M)


def fem_run(dt, floors):
    """Production T-A solve on the benchmark mesh, stub Jc and n = 21,
    floor ladder; returns per-floor (slab profile of |u|, max |u|)."""
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio
    import dolfinx.mesh as dmesh
    import build_mesh
    import solve as base_solve
    import ta_solve

    class StubN:
        def n_value(self, B, theta):
            return np.full(np.shape(B), N_EXP), 0.0

    comm = MPI.COMM_WORLD
    path = os.path.join(_ROOT, "mesh", f"bench_nl_{os.getpid()}.msh")
    build_mesh.build(write_path=path)
    md = gmshio.read_from_msh(path, comm, rank=0, gdim=3)
    os.remove(path)
    dom = md.mesh
    uni = base_solve.setup_problem(dom, md.cell_tags, md.facet_tags)
    ta = ta_solve.setup_ta_problem(dom, md.cell_tags, md.facet_tags, uni)
    ta["dt_const"].value = dt
    cells = ta["coil_cells"]
    cents = dmesh.compute_midpoints(dom, 3, cells)
    t_hat = ta_solve._coil_t_hat_array(dom, cells)
    sel = cents[:, 0] < lin.X_CUT
    zed = params.layer_z_bottoms[0] + params.w * np.concatenate(
        [[0.0], np.cumsum(lin.Z_GRADING)])
    slab = np.clip(np.searchsorted(zed, cents[sel, 2]) - 1, 0,
                   len(lin.Z_GRADING) - 1)
    Jc = IC_A / (params.delta_SC * params.w)
    out = []
    for k, (eps, alpha, its) in enumerate(floors):
        params.ta_eps_reg = eps
        params.ta_picard_alpha = params.ta_picard_alpha_fine = alpha
        params.ta_n_picard = its
        diag = os.path.join(_VAL, f"ta_nl_bench_diag_f{eps:g}.csv")
        if os.path.exists(diag):
            os.remove(diag)
        t0 = time.time()
        ta_solve.solve_ta_at_current(
            dom, ta, uni, I_amps=I_A, ic_model=lin.StubIc(IC_A),
            n_model=StubN(), verbose=False, warm_start=(k > 0),
            min_iters=its, diag_log_path=diag)
        J = ta_solve._J_from_T(ta, dom)
        u = np.abs(np.einsum("ij,ij->i", J, t_hat)) / Jc
        us = u[sel]
        prof = np.array([us[slab == i].mean() for i in range(len(lin.Z_GRADING))])
        with open(diag) as fh:
            sc = np.array([float(r["scif_mT"]) for r in csv.DictReader(fh)])
        tail = sc[-30:]
        if os.environ.get("NB_DUMP") == "1":      # per-cell dump
            np.savez(os.path.join(_VAL, f"ta_nl_bench_cells_f{eps:g}.npz"),
                     u=u, cents=cents, vols=ta["coil_vols"],
                     verts=dom.geometry.x[dom.geometry.dofmaps[0][cells]])
        out.append(dict(eps=eps, prof=prof, umax=float(us.max()),
                        umax_all=float(u.max()),
                        scif_range=float(tail.max() - tail.min()),
                        wall=time.time() - t0))
        print(f"  FEM floor {eps}: max|u| (x<{lin.X_CUT*1e3:.0f}mm) "
              f"{us.max():.4f}, all cells {u.max():.4f}, SCIF range30 "
              f"{tail.max()-tail.min():.3f} mT  ({time.time()-t0:.0f}s)")
    return out


def _binned(z, u):
    zed = params.layer_z_bottoms[0] + params.w * np.concatenate(
        [[0.0], np.cumsum(lin.Z_GRADING)])
    p = np.abs(u).mean(axis=0)
    return np.array([p[(z >= lo) & (z < hi)].mean()
                     for lo, hi in zip(zed[:-1], zed[1:])])


def main():
    lin.apply_geometry()
    print(f"geometry: {lin.N_TAPES} tapes, Ic={IC_A} A/tape, I={I_A} A, "
          f"n={N_EXP:g}, floor={FLOOR}")
    if os.environ.get("NB_DT"):
        dts = [float(os.environ["NB_DT"])]
    else:
        dts = [1.0, 10.0, 100.0, 1000.0, 1e4]
    scan = {}
    for dt in dts:
        t0 = time.time()
        _, up = reference(dt, I_A, 1e-3)
        _, uf = reference(dt, I_A, FLOOR)
        scan[dt] = (up, uf)
        print(f"reference dt={dt:8.1f}s: max|u| pure {np.abs(up).max():.4f}, "
              f"floor {FLOOR} {np.abs(uf).max():.4f}  ({time.time()-t0:.0f}s)")
    if os.environ.get("NB_REF_ONLY") == "1":
        return
    dt = float(os.environ.get("NB_DT", 100.0))
    if dt not in scan:
        scan[dt] = (reference(dt, I_A, 1e-3)[1], reference(dt, I_A, FLOOR)[1])
    z, _ = reference(dt, I_A, FLOOR, M=40)
    up, uf = scan[dt]
    ladder = [(1.0, 0.10, 150), (0.9, 0.10, 100), (0.8, 0.05, 150),
              (0.7, 0.02, 250), (0.6, 0.005, 500)]
    ladder = [s for s in ladder if s[0] >= FLOOR - 1e-9]
    if os.environ.get("NB_LADDER"):               # "eps:alpha:iters,..."
        ladder = [tuple(float(v) if j < 2 else int(v)
                        for j, v in enumerate(item.split(":")))
                  for item in os.environ["NB_LADDER"].split(",")]
    if os.environ.get("NB_FLOOR_STOP"):           # stop the ladder early
        ladder = [s for s in ladder
                  if s[0] >= float(os.environ["NB_FLOOR_STOP"]) - 1e-9]
    fem = fem_run(dt, ladder)
    rows = []
    pp, pf = _binned(z, up), _binned(z, uf)
    for r in fem:
        rf = (reference(dt, I_A, r["eps"])[1] if abs(r["eps"] - FLOOR) > 1e-9
              else uf)
        pr = _binned(z, rf)
        rms = np.sqrt(np.mean((r["prof"] - pr) ** 2)) / np.sqrt(np.mean(pr ** 2))
        print(f"floor {r['eps']}: FEM max|u| {r['umax']:.4f} vs ref(same floor) "
              f"{np.abs(rf).max():.4f} vs pure {np.abs(up).max():.4f};  "
              f"profile RMS vs ref(same floor) {rms:.3f}")
        rows.append(dict(dt_s=dt, I_A=I_A, floor=r["eps"],
                         fem_umax=r["umax"], fem_umax_all=r["umax_all"],
                         ref_floor_umax=float(np.abs(rf).max()),
                         ref_pure_umax=float(np.abs(up).max()),
                         profile_rms_vs_ref_floor=rms,
                         scif_range30_mT=r["scif_range"],
                         fem_profile=" ".join(f"{v:.4f}" for v in r["prof"]),
                         ref_floor_profile=" ".join(f"{v:.4f}" for v in pr),
                         ref_pure_profile=" ".join(f"{v:.4f}" for v in pp)))
    with open(OUT_CSV, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
