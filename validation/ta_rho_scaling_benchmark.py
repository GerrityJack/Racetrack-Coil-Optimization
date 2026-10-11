"""
ta_rho_scaling_benchmark.py — is the resistivity in the T-equation right?
==========================================================================
2026-10-03/04.  Until 2026-10-04 solve/ta_solve.py `_update_rho()` used

    rho_fn = rho_SC * (delta_SC / Lambda)

in the T-equation  ∫ rho_fn (∇T×n̂)·(∇φ×n̂) = −(1/Δt) ∫ curl(A − A_prev)·φn̂,
while Vargas-Llanos et al. 2022 (physics/Vargas-Llanos_2022_*.pdf, eqs. 4,
5, 8, 10) put δ/Λ only on the A-equation source and use plain rho_HTS in
the T-equation.  The factor is uniform over the coil, so it is exactly a
75x change in the effective time step.

Test: OHMIC limit (n = 1, constant rho, via stub Ic/n models), where the
physics is linear and the only scale is the screening time constant, so a
75x error in rho is a 75x shift in that constant — far larger than any
geometry/mesh error.  One backward-Euler step of length dt from the virgin
state at transport current I, on a single-layer racetrack with a long
straight section.

Reference (independent, no FEM, no homogenisation): the cross-section of
the straight section at x = 0 as N DISCRETE tapes (zero-thickness sheets at
radial pitch Lambda, width w along z), each split into M elements along z.
Per-unit-length mutual inductances are exact log-kernel averages over the
element pairs.  Images: the other leg (y -> -y, current reversed; the y=0
PEC plane) and coil 2 (z -> 2g - z, same current; the z=g PMC plane).  Per
tape:  rho_SC J_m = -A_m/dt + e  (e = the tape's uniform voltage gradient),
sum_m J_m δ h_m = I.  This is unambiguous SC-layer physics: E = rho_SC J_SC.

FEM: the production code path (build_mesh -> setup_ta_problem ->
solve_ta_at_current), unmodified.  Two variants on ONE mesh (same process):
    code: stub Ic = IC_A       -> rho_fn as the solver now computes it
                                  (= rho_SC since the 2026-10-04 fix)
    old : stub Ic = IC_A·Λ/δ   -> rho_fn = rho_SC·δ/Λ, emulating the removed
                                  factor (before the fix, IC_A gave this)

RESULT (2026-10-04, run before the fix, variants then named asis/paper):
at dt = 375 s the old factor matched the reference at dt/75 = 5 s to 1.5 %
RMS (16 % off the reference at 375 s); rho_SC matched the reference at
375 s to 0.01 % and at 5 s to 0.8 %.  The factor was then removed.
Compared: J·t̂ across the tape width, averaged over the stack, in the
straight section near x = 0, plus a dt sweep of the reference.

Default dt = 375 s = 75 x 5 s.  The reference (physical rho_SC) is nearly
uniform there; since the δ/Λ factor is algebraically a dt/75 rescaling,
"old" instead reproduces the reference at 5 s (edge/centre ~1.6).  Picard stability needs alpha ~ rho_fn*dt (explicit
inductive coupling), which is why the variants get different alpha; at
dt = 375 s both are cheap.  A dt = 5 s "code" run (BENCH_DT=5
BENCH_VARIANTS=code BENCH_ALPHA=code:0.05 BENCH_NPICARD=code:250) checks
it against a strongly screened reference directly.

Env: BENCH_VARIANTS="code,old", BENCH_DT (s), BENCH_I (A),
     BENCH_ALPHA="code:0.2,old:0.05", BENCH_NPICARD="code:120,old:250",
     BENCH_REF_ONLY=1 (reference sweep only, no FEM).
Run (direct binary — see CLAUDE.md "Operational lessons"):
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \
        validation/ta_rho_scaling_benchmark.py
Output: validation/ta_rho_scaling_benchmark.csv,
        visualization/ta_rho_scaling_benchmark.png
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
for _p in (_ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params

MU0 = 4e-7 * np.pi

# ── Benchmark geometry: one layer, long straight section ───────────────────
N_TAPES = 40                 # turns in the single layer (pack = 3 mm)
A_M = 0.008                  # params.a  (end-cap mid-radius of the pack)
L_STRAIGHT = 0.030           # b - a = half-length of the straight section
GAP_M = 0.020                # coil_half_gap (coil 2 mirrored at z = 2g)
Z_GRADING = [0.1] * 10       # 10 uniform slabs across the tape width
X_CUT = 0.008                # sample cells with centroid x < X_CUT (mid-straight)

IC_A = 100.0                 # stub Ic -> rho_SC = E_c/Jc, Jc = Ic/(δ w)
I_A = float(os.environ.get("BENCH_I", 50.0))
DT_S = float(os.environ.get("BENCH_DT", 375.0))
VARIANTS = os.environ.get("BENCH_VARIANTS", "code,old").split(",")


def _kv(env, default):
    out = dict(default)
    for item in os.environ.get(env, "").split(","):
        if ":" in item:
            k, v = item.split(":")
            out[k] = v
    return out


ALPHA = {k: float(v) for k, v in
         _kv("BENCH_ALPHA", {"code": 0.2, "old": 0.05}).items()}
NPICARD = {k: int(v) for k, v in
           _kv("BENCH_NPICARD", {"code": 120, "old": 250}).items()}

OUT_CSV = os.path.join(_ROOT, "validation", "ta_rho_scaling_benchmark.csv")
OUT_PNG = os.path.join(_ROOT, "visualization", "ta_rho_scaling_benchmark.png")


def apply_geometry():
    params.a = A_M
    params.b = A_M + L_STRAIGHT
    params.coil_half_gap = GAP_M
    params.n_turns = [N_TAPES]
    params.mesh_nz_per_layer = 3
    params.mesh_z_grading = list(Z_GRADING)
    params.mesh_size_min_factor = 0.25
    params.mesh_size_max_factor = 0.60
    params.mesh_dist_min_factor = 1.50
    params.mesh_dist_max_factor = 2.00
    params.box_scale = 4.0
    params.recompute_derived()


# ── Reference: discrete-tape integral equation ─────────────────────────────

def _F2(u, d):
    """Second antiderivative in u of ln sqrt(u² + d²)."""
    u = np.asarray(u, float)
    r2 = u * u + d * d
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(r2 > 0, np.log(np.where(r2 > 0, r2, 1.0)), 0.0)
        at = np.where(d > 0, u * np.arctan(u / np.where(d > 0, d, 1.0)), 0.0)
    return 0.25 * (u * u - d * d) * lg - 0.75 * u * u + d * at


def _mean_log(z1a, z1b, z2a, z2b, d):
    """Mean of ln|r - r'| over two parallel z-segments offset by d in y."""
    I = (_F2(z1b - z2a, d) - _F2(z1b - z2b, d)
         - _F2(z1a - z2a, d) + _F2(z1a - z2b, d))
    return I / ((z1b - z1a) * (z2b - z2a))


def reference(rho_sc, dt, I, M=40):
    """Return (z_centres, J[N, M]) in A/m² (SC layer) for one BE step."""
    w, lam, dsc = params.w, params.t, params.delta_SC
    y = params.a_inner_list[0] + (np.arange(N_TAPES) + 0.5) * lam
    zb = params.layer_z_bottoms[0]
    edges = zb + w * np.linspace(0.0, 1.0, M + 1)
    za, zt = edges[:-1], edges[1:]
    h = zt - za
    g = params.coil_half_gap

    # element lists
    Y = np.repeat(y, M)
    ZA = np.tile(za, N_TAPES)
    ZT = np.tile(zt, N_TAPES)
    n = len(Y)

    def block(y_src, za_src, zt_src, sign):
        d = np.abs(Y[:, None] - y_src[None, :])
        return sign * _mean_log(ZA[:, None], ZT[:, None],
                                za_src[None, :], zt_src[None, :], d)

    K = (block(Y, ZA, ZT, +1)                          # real currents
         + block(-Y, ZA, ZT, -1)                       # other leg (y=0 PEC)
         + block(Y, 2 * g - ZT, 2 * g - ZA, +1)        # coil 2 (z=g PMC)
         + block(-Y, 2 * g - ZT, 2 * g - ZA, -1))      # coil 2, other leg
    Lmat = -(MU0 / (2 * np.pi)) * K                    # H/m per element pair
    area = dsc * np.tile(h, N_TAPES)                   # element SC area

    # unknowns: J (n) then e (N_TAPES)
    # rho J_i + (1/dt) sum_j L_ij area_j J_j - e_k(i) = 0
    # sum_{i in k} area_i J_i = I
    S = np.zeros((n + N_TAPES, n + N_TAPES))
    S[:n, :n] = rho_sc * np.eye(n) + Lmat * area[None, :] / dt
    tape = np.repeat(np.arange(N_TAPES), M)
    S[np.arange(n), n + tape] = -1.0
    S[n + tape, np.arange(n)] = area
    rhs = np.zeros(n + N_TAPES)
    rhs[n:] = I
    sol = np.linalg.solve(S, rhs)
    return 0.5 * (za + zt), sol[:n].reshape(N_TAPES, M)


def profile_metrics(J, I):
    """Edge/centre ratio of the stack-mean |J| profile, normalised."""
    J0 = I / (params.delta_SC * params.w)
    p = J.mean(axis=0) / J0 if J.ndim == 2 else J / J0
    m = len(p)
    edge = 0.5 * (p[0] + p[-1])
    centre = p[m // 2 - 1: m // 2 + 1].mean() if m % 2 == 0 else p[m // 2]
    return p, edge / centre


# ── FEM (production code path) ─────────────────────────────────────────────

class StubIc:
    def __init__(self, ic):
        self.ic = float(ic)
        self.tape_width = params.w

    def critical_current(self, B, theta):
        return np.full(np.shape(B), self.ic), 0.0


class StubN:
    def n_value(self, B, theta):
        return np.ones(np.shape(B)), 0.0


def fem_runs(variants):
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio
    import dolfinx.mesh as dmesh
    import build_mesh
    import solve as base_solve
    import ta_solve

    comm = MPI.COMM_WORLD
    mesh_path = os.path.join(_ROOT, "mesh", f"bench_rho_{os.getpid()}.msh")
    t0 = time.time()
    build_mesh.build(write_path=mesh_path)
    md = gmshio.read_from_msh(mesh_path, comm, rank=0, gdim=3)
    os.remove(mesh_path)
    domain = md.mesh
    uni = base_solve.setup_problem(domain, md.cell_tags, md.facet_tags)
    ta = ta_solve.setup_ta_problem(domain, md.cell_tags, md.facet_tags, uni)
    ta["dt_const"].value = DT_S
    params.ta_eps_reg = 1.0          # irrelevant for n = 1 (rho is constant)
    cells = ta["coil_cells"]
    cents = dmesh.compute_midpoints(domain, domain.topology.dim, cells)
    t_hat = ta_solve._coil_t_hat_array(domain, cells)
    print(f"mesh: {domain.topology.index_map(3).size_global} cells, "
          f"{len(cells)} coil cells  ({time.time()-t0:.0f}s)")

    sel = cents[:, 0] < X_CUT
    z_edges = params.layer_z_bottoms[0] + params.w * np.concatenate(
        [[0.0], np.cumsum(Z_GRADING)])
    slab = np.clip(np.searchsorted(z_edges, cents[sel, 2]) - 1,
                   0, len(Z_GRADING) - 1)
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
    lam_over_d = params.t / params.delta_SC
    out = {}
    for v in variants:
        ic = IC_A * lam_over_d if v == "old" else IC_A
        params.ta_picard_alpha = params.ta_picard_alpha_fine = ALPHA[v]
        params.ta_n_picard = NPICARD[v]
        diag = os.path.join(_ROOT, "validation", f"ta_rho_bench_diag_{v}.csv")
        if os.path.exists(diag):
            os.remove(diag)
        for k in ("_I_prev", "_T_prev", "_T_prev_layers", "_rho_prev"):
            ta.pop(k, None)
        t1 = time.time()
        ta_solve.solve_ta_at_current(domain, ta, uni, I_amps=I_A,
                                     ic_model=StubIc(ic), n_model=StubN(),
                                     verbose=False, warm_start=False,
                                     min_iters=NPICARD[v], diag_log_path=diag)
        J = ta_solve._J_from_T(ta, domain)
        Jt = np.einsum("ij,ij->i", J, t_hat)[sel]
        Jt *= np.sign(Jt.mean())
        vols = ta["coil_vols"][sel] if "coil_vols" in ta else np.ones(len(Jt))
        prof = np.array([np.average(Jt[slab == i], weights=vols[slab == i])
                         for i in range(len(Z_GRADING))])
        # convergence: raw |dB|/|B| and bore-SCIF spread over the last 10%
        with open(diag) as fh:
            rows_d = list(csv.DictReader(fh))
        sc = np.array([float(r["scif_mT"]) for r in rows_d])
        re = np.array([float(r["rel_err"]) for r in rows_d])
        tail = sc[-max(10, len(sc) // 10):]
        rho_used = float(ta["rho_fn"].x.array[cells].mean())
        out[v] = dict(z=z_mid, prof=prof, rho_used=rho_used,
                      scif_tail_range=float(tail.max() - tail.min()),
                      scif_final=float(sc[-1]), rel_err=float(re[-1]),
                      wall=time.time() - t1)
        print(f"[{v}] rho_fn={rho_used:.3e} Ohm·m  SCIF={sc[-1]:.5f} mT "
              f"(tail range {tail.max()-tail.min():.2e})  "
              f"|dB|/|B| final {re[-1]:.2e}  ({out[v]['wall']:.0f}s)")
    return out


def main():
    apply_geometry()
    lam_over_d = params.t / params.delta_SC
    Jc = IC_A / (params.delta_SC * params.w)
    rho_sc = 1e-4 / Jc
    J0 = I_A / (params.delta_SC * params.w)
    print(f"geometry: N={N_TAPES} a={params.a*1e3:.1f}mm L={L_STRAIGHT*1e3:.0f}mm "
          f"g={GAP_M*1e3:.0f}mm  rho_SC={rho_sc:.3e} Ohm·m  dt={DT_S:g}s  "
          f"I={I_A:g}A  Lambda/delta={lam_over_d:.0f}")

    # reference dt sweep (physical rho_SC)
    sweep = np.logspace(-1, 4, 26)
    ratios = []
    for dt in sweep:
        _, J = reference(rho_sc, dt, I_A)
        ratios.append(profile_metrics(J, I_A)[1])
    print("reference edge/centre ratio vs dt:")
    for dt, r in zip(sweep, ratios):
        print(f"   dt={dt:9.2f}s  edge/centre={r:7.3f}")

    zr, Jr = reference(rho_sc, DT_S, I_A)
    _, Jr75 = reference(rho_sc, DT_S / lam_over_d, I_A)
    pr, rr = profile_metrics(Jr, I_A)
    pr75, rr75 = profile_metrics(Jr75, I_A)
    print(f"reference @ dt={DT_S:g}s: edge/centre {rr:.3f};  "
          f"@ dt/{lam_over_d:.0f}={DT_S/lam_over_d:.3g}s: {rr75:.3f}")
    if os.environ.get("BENCH_REF_ONLY") == "1":
        return

    fem = fem_runs(VARIANTS)

    # compare on the FEM z-points (slab centres): bin the reference
    rows = []
    edges = params.layer_z_bottoms[0] + params.w * np.concatenate(
        [[0.0], np.cumsum(Z_GRADING)])
    def binned(p):
        return np.array([p[(zr >= lo) & (zr < hi)].mean()
                         for lo, hi in zip(edges[:-1], edges[1:])])
    ref_b = binned(pr)
    ref75_b = binned(pr75)
    for v, d in fem.items():
        pf = d["prof"] / J0
        if len(pf) != len(ref_b):
            print(f"[{v}] {len(pf)} z-points vs {len(ref_b)} slabs — "
                  "skipping binned comparison")
            continue
        e_ref = np.sqrt(np.mean((pf - ref_b) ** 2)) / np.sqrt(np.mean(ref_b ** 2))
        e_75 = np.sqrt(np.mean((pf - ref75_b) ** 2)) / np.sqrt(np.mean(ref75_b ** 2))
        er = 0.5 * (pf[0] + pf[-1]) / pf[len(pf)//2 - 1: len(pf)//2 + 1].mean()
        print(f"[{v}] edge/centre={er:.3f}   rel. RMS diff vs reference "
              f"@dt: {e_ref:.3f}   @dt/{lam_over_d:.0f}: {e_75:.3f}")
        rows.append(dict(variant=v, dt_s=DT_S, I_A=I_A, rho_SC=rho_sc,
                         rho_fn=d["rho_used"], edge_centre=er,
                         rms_vs_ref_dt=e_ref, rms_vs_ref_dt_over75=e_75,
                         scif_final_mT=d["scif_final"],
                         scif_tail_range_mT=d["scif_tail_range"],
                         rel_err_final=d["rel_err"],
                         wall_s=d["wall"],
                         profile=" ".join(f"{x:.4f}" for x in pf)))
    rows.append(dict(variant="reference", dt_s=DT_S, I_A=I_A, rho_SC=rho_sc,
                     edge_centre=rr,
                     profile=" ".join(f"{x:.4f}" for x in ref_b)))
    rows.append(dict(variant=f"reference_dt/{lam_over_d:.0f}",
                     dt_s=DT_S / lam_over_d, I_A=I_A, rho_SC=rho_sc,
                     edge_centre=rr75,
                     profile=" ".join(f"{x:.4f}" for x in ref75_b)))
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "variant", k))
    with open(OUT_CSV, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {OUT_CSV}")

    # figure (project dark theme)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="#111")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d0d1a")
        for s in ax.spines.values():
            s.set_color("#444")
        ax.tick_params(colors="white")
    zmm = (zr - params.layer_z_bottoms[0]) * 1e3
    ax1.plot(zmm, pr, color="#6ec6ff", lw=2, label=f"reference, dt={DT_S:g}s")
    ax1.plot(zmm, pr75, color="#ff9e57", lw=2, ls="--",
             label=f"reference, dt={DT_S/lam_over_d:.2g}s (= dt/{lam_over_d:.0f})")
    mk = {"code": ("o", "#6ec6ff"), "old": ("s", "#ff9e57")}
    for v, d in fem.items():
        ax1.plot((d["z"] - params.layer_z_bottoms[0]) * 1e3, d["prof"] / J0,
                 mk[v][0], color=mk[v][1], ms=8, mfc="none", mew=2,
                 label=f"FEM {v} (rho_fn={d['rho_used']:.2g})")
    ax1.set_xlabel("position across tape width z (mm)", color="white")
    ax1.set_ylabel("stack-mean J / (I/δw)", color="white")
    ax1.set_title("Current profile across the tape, mid-straight",
                  color="white")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax2.semilogx(sweep, ratios, color="#6ec6ff", lw=2, label="reference")
    for v, d in fem.items():
        pf = d["prof"] / J0
        er = 0.5 * (pf[0] + pf[-1]) / pf[len(pf)//2 - 1: len(pf)//2 + 1].mean()
        ax2.axhline(er, color=mk[v][1], ls=":", lw=1.5, label=f"FEM {v}")
    ax2.axvline(DT_S, color="#888", lw=1)
    ax2.set_xlabel("time step dt (s)", color="white")
    ax2.set_ylabel("edge / centre current ratio", color="white")
    ax2.set_title("Reference dt sweep (physical rho_SC)", color="white")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, facecolor=fig.get_facecolor())
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
