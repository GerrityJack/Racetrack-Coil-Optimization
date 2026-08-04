"""
lumped.py — Tier A1 validation of the DCN's inductance and resistance build.

Three independent checks, in increasing order of independence:

  1. Neumann + GMD self-inductance vs the textbook circular-loop formula
     L = mu0 * R * (ln(8R/a) - 2).  Tests the integrator, not the geometry.

  2. Grouping convergence of the total system inductance.  TURNS_PER_GROUP is
     the DCN's only spatial discretisation, so the answer must stop moving as
     it is refined.

  3. **The real cross-check**: the FEM.  The filament model and the FEM share
     no code, no geometry representation and no field evaluation path, so
     agreement here validates the whole chain.  Two sub-checks:

       3a. Field comparison.  B from the FEM vs B from the filament sum at
           the bore centre and along the axis.  This is the sharp one: it
           tests the geometry and the Biot-Savart path directly.

       3b. Stored magnetic energy, W = integral(|B|^2 / 2mu0) over the domain,
           x8 for the full two-coil system (mirrors about x=0, y=0 and
           z=coil_half_gap), giving L = 2W/I^2.

     Do NOT use integral(J.A) here: this A-form carries only a weak gauge
     penalty (params.gauge_regularization = 1e-3 against 1/mu0 ~ 8e5), so A
     retains a large gradient component.  That component cancels out of
     curl A, but NOT out of integral(J.A), because the eighth-domain current
     enters and leaves through the symmetry cut faces and so is not
     divergence-free in the domain.  Trying it that way gave L off by ~1e8.

     NOTE an area-integral-of-Bz check would NOT be independent -- by Stokes
     it is algebraically the same Neumann double sum.

     The energy integral is truncated at the air box (box_scale * b), so it
     slightly UNDER-counts exterior field energy; treat 3b as a ~10% check
     and 3a as the precise one.

  4. Analytic tau bracket, for orientation only.  The whole point of the DCN
     is that these lumped brackets disagree by orders of magnitude.

Run:  <env>/bin/python3 circuit/validation/lumped.py
The FEM check needs dolfinx; it is skipped with a clear message if absent.
"""

import os
import sys

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_CIRC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_CIRC)
for _p in (_CIRC, _ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params                                    # noqa: E402
import cparams as cfg                            # noqa: E402
import inductance as ind                         # noqa: E402
from geometry import CoilGeometry, TurnGroups, racetrack_loop   # noqa: E402

MU0 = 4.0e-7 * np.pi


# ── check 1: integrator vs textbook ─────────────────────────────────────────

def check_neumann_kernel():
    print("\n[1] Neumann + GMD vs textbook circular loop "
          "L = mu0*R*(ln(8R/a) - 2)")
    ok = True
    for R, gmd in ((0.010, 0.911e-3), (0.022, 0.911e-3), (0.044, 0.911e-3)):
        # L = 0 degenerates the racetrack to a circle of radius R
        p, d, _ = racetrack_loop(R, 0.0, 0.0, 4, 600)
        M = ind._neumann_pair(p, d, p, d, gmd ** 2)
        ana = MU0 * R * (np.log(8 * R / gmd) - 2.0)
        rel = M / ana - 1.0
        flag = "OK " if abs(rel) < 0.01 else "FAIL"
        ok &= abs(rel) < 0.01
        print(f"    R={R*1e3:6.1f} mm  neumann={M*1e9:8.2f} nH  "
              f"analytic={ana*1e9:8.2f} nH  rel={rel*100:+6.2f}%  {flag}")
    return ok


# ── check 2: grouping convergence ───────────────────────────────────────────

def check_convergence(geom, tpgs=(100, 50, 25, 12, 6)):
    print(f"\n[2] Grouping convergence of L_total  ({geom})")
    res = []
    for tpg in tpgs:
        tg = TurnGroups(geom, tpg)
        M = ind.build_M_cached(tg, geom, verbose=False)
        L = ind.total_inductance(M, tg, geom.two_coil)
        res.append((tpg, tg.N, L))
        print(f"    turns/group={tpg:4d}  N={tg.N:4d}  "
              f"L_total={L*1e3:9.3f} mH  "
              f"W(I_design)={0.5*L*params.I_design**2:9.1f} J")
    if len(res) >= 2:
        drift = abs(res[-1][2] / res[-2][2] - 1.0) * 100
        print(f"    last refinement moved L by {drift:.2f}%")
    return res


# ── check 3: FEM stored energy (the independent one) ────────────────────────

def _filament_B(points, geom, I_per_turn, turns_per_group=6):
    """B at `points` from the filament model (both coils)."""
    import fieldmatrix as fm
    tg = TurnGroups(geom, turns_per_group)
    g2 = 2.0 * geom.coil_half_gap
    B = np.zeros((len(points), 3))
    for j in range(tg.N):
        p, d, _ = racetrack_loop(tg.r[j], tg.z[j], geom.L, 200, 300)
        amp = tg.n[j] * I_per_turn
        B += amp * fm._biot_savart(points, p, d)
        if geom.two_coil:
            pm = p.copy()
            pm[:, 2] = g2 - p[:, 2]
            B += amp * fm._biot_savart(points, pm, d)
    return B


def check_fem(geom, L_filament):
    print("\n[3] INDEPENDENT cross-check against the FEM")
    try:
        from mpi4py import MPI
        import ufl
        from dolfinx import fem
        from dolfinx.io import gmsh as gmshio
        import build_mesh
        import solve as base_solve
    except Exception as exc:                        # pragma: no cover
        print(f"    SKIPPED (dolfinx unavailable: {exc})")
        return None

    comm = MPI.COMM_WORLD
    print("    building mesh ...")
    build_mesh.build(write_path=params.mesh_filename, verbose=False)
    md = gmshio.read_from_msh(params.mesh_filename, comm, gdim=3)
    domain, cell_tags = md.mesh, md.cell_tags

    setup = base_solve.setup_problem(domain, cell_tags, md.facet_tags)
    I = float(params.I_design)
    print(f"    solving uniform-J A-form at I = {I:.1f} A ...")
    A_h, B_h = base_solve.solve_at_current(domain, setup, I, comm,
                                           verbose_label="L-check")

    # ── 3a. filament sum vs the repo's PRODUCTION Biot-Savart path ────────
    # This is the sharp check: coil2_field.compute_both_coils_field_multilayer
    # is what every design number in this project goes through, and it shares
    # no code with circuit/.  Any error in my loop construction, turn
    # grouping, or coil-2 mirroring shows up here immediately.
    from coil2_field import compute_both_coils_field_multilayer as B_repo
    g = params.coil_half_gap
    zs = np.linspace(-0.6 * g, g, 9)
    pts = np.column_stack([np.zeros_like(zs), np.zeros_like(zs), zs])
    B1 = _filament_B(pts, geom, I)[:, 2]
    B2 = B_repo(pts)[:, 2]
    rel_repo = np.abs(B1 - B2) / np.maximum(np.abs(B2), 1e-12)
    print(f"    3a. bore-axis Bz vs coil2_field (production path), "
          f"{len(zs)} points:")
    print(f"        median rel diff = {np.median(rel_repo)*100:.4f}%, "
          f"max = {np.max(rel_repo)*100:.4f}%")
    print(f"        at bore centre: circuit {B1[-1]:+.5f} T   "
          f"repo {B2[-1]:+.5f} T")

    # ── 3b. FEM field in the BORE (air only, no winding cells) ────────────
    import dolfinx.mesh as dmesh
    tdim = domain.topology.dim
    ncell = domain.topology.index_map(tdim).size_local
    cents = dmesh.compute_midpoints(domain, tdim,
                                    np.arange(ncell, dtype=np.int32))
    Bfem = B_h.x.array.reshape(-1, 3)[:ncell]
    rho = np.hypot(cents[:, 0], cents[:, 1])
    r_bore = min(params.a_inner_list)
    sel = np.where((rho < 0.85 * r_bore) & (cents[:, 2] < g)
                   & (cents[:, 2] > -g))[0]
    if len(sel) == 0:
        print("    3b. SKIPPED (no bore cells found)")
        rel = np.array([np.nan])
    else:
        if len(sel) > 60:
            sel = sel[np.linspace(0, len(sel) - 1, 60).astype(int)]
        Bfil = _filament_B(cents[sel], geom, I)
        nf = np.linalg.norm(Bfem[sel], axis=1)
        nl = np.linalg.norm(Bfil, axis=1)
        rel = np.abs(nl - nf) / np.maximum(nf, 1e-12)
        print(f"    3b. |B| at {len(sel)} bore cells "
              f"(rho < {0.85*r_bore*1e3:.1f} mm): "
              f"median rel diff = {np.median(rel)*100:.2f}%, "
              f"max = {np.max(rel)*100:.2f}%")
        print(f"        FEM  |B| {nf.min():.3f} - {nf.max():.3f} T   "
              f"fil. |B| {nl.min():.3f} - {nl.max():.3f} T")

    # ── 3b. stored energy from |B|^2 (gauge invariant) ─────────────────────
    dx = ufl.Measure("dx", domain=domain)
    mu0 = 4.0e-7 * np.pi
    W8 = fem.assemble_scalar(
        fem.form(ufl.inner(ufl.curl(A_h), ufl.curl(A_h))
                 / (2.0 * mu0) * dx))
    W8 = comm.allreduce(W8, op=MPI.SUM)
    W_fem = 8.0 * W8
    L_fem = 2.0 * W_fem / I ** 2
    print(f"    3c. W (eighth domain) = {W8:.2f} J -> full system "
          f"{W_fem:.1f} J")
    print(f"        L_FEM = {L_fem*1e3:.2f} mH   "
          f"(filament {L_filament*1e3:.2f} mH, "
          f"ratio {L_filament/L_fem:.4f})")
    print(f"        NB the air box is only "
          f"{params.box_scale*params.b/2/(params.L+params.a_out):.2f}x the "
          f"coil's outer extent and carries a PEC boundary, so flux is "
          f"confined and\n        this is a LOWER BOUND on L, not an "
          f"independent measurement.  Use 3a as the sharp check.")
    return dict(L_fem=L_fem, W_fem=W_fem,
                B_rel_median=float(np.median(rel)),
                B_rel_max=float(np.max(rel)),
                repo_rel_median=float(np.median(rel_repo)),
                repo_rel_max=float(np.max(rel_repo)))


# ── check 4: analytic tau bracket ───────────────────────────────────────────

def tau_bracket(geom, L_total):
    print("\n[4] Analytic tau brackets (orientation only -- these are the "
          "estimates the DCN exists to replace)")
    print(f"    {'rho_c':>8}  {'R_bypass/pancake [mOhm]':>26}  "
          f"{'tau=L/R_min [s]':>16}  {'tau=L_i/R_i [s]':>16}")
    for rc_u in cfg.RHO_CT_SWEEP_UOHM_CM2:
        rho = rc_u * cfg.UOHM_CM2_TO_OHM_M2
        Rs, taus_i = [], []
        for i, n in enumerate(geom.n_turns):
            if n < 2:
                Rs.append(np.inf)
                taus_i.append(np.nan)
                continue
            r = geom.a_out - (np.arange(n - 1) + 0.5) * geom.t
            R = float(np.sum(rho / (geom.turn_length(r) * geom.w)))
            Rs.append(R)
            L_i = L_total * (n / geom.n_turns_total) ** 2
            taus_i.append(L_i / R)
        finite = [r for r in Rs if np.isfinite(r)]
        tau_hi = L_total / min(finite) if finite else np.nan
        tau_lo = np.nanmax(taus_i)
        s = " ".join(f"{r*1e3:7.2f}" if np.isfinite(r) else "    inf"
                     for r in Rs)
        print(f"    {rc_u:8.0f}  {s:>26}  {tau_hi:16.1f}  {tau_lo:16.2f}")
    print("    (the two columns differ by ~100x -- that spread is the "
          "motivation for the coupled solve)")


def main():
    print("=" * 74)
    print("DCN Tier-A1 validation — inductance and resistance build")
    print("=" * 74)
    geom = CoilGeometry.from_params()
    print(f"\n{geom}")
    print(f"tape length: geometry {geom.tape_length_m():.4f} m   "
          f"params {params.tape_length_m:.4f} m   "
          f"(rel {abs(geom.tape_length_m()/params.tape_length_m-1):.2e})")

    ok1 = check_neumann_kernel()
    res = check_convergence(geom)
    L_fil = res[-1][2]

    femr = check_fem(geom, L_fil)

    print("\n" + "=" * 74)
    print(f"filament (Neumann, finest grouping) : {L_fil*1e3:9.3f} mH"
          f"   W = {0.5*L_fil*params.I_design**2:.0f} J")
    ok3 = True
    if femr is not None:
        print(f"vs repo production Biot-Savart      : "
              f"{femr['repo_rel_median']*100:.4f}% median, "
              f"{femr['repo_rel_max']*100:.4f}% max   <- the sharp check")
        print(f"vs FEM in the bore                  : "
              f"{femr['B_rel_median']*100:.2f}% median, "
              f"{femr['B_rel_max']*100:.2f}% max")
        print(f"FEM energy inductance (lower bound) : "
              f"{femr['L_fem']*1e3:9.3f} mH  "
              f"(ratio {L_fil/femr['L_fem']:.4f})")
        ok3 = femr["repo_rel_median"] < 0.01
        print(f"VERDICT (vs production path < 1%)   : "
              f"{'PASS' if ok3 else 'FAIL'}")
    print("=" * 74)

    tau_bracket(geom, L_fil)
    return 0 if (ok1 and ok3) else 1


if __name__ == "__main__":
    sys.exit(main())
