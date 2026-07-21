"""
optimize_geometry.py — one-file configuration sweep
====================================================
Evaluates every candidate geometry (a, b, n_turns) from opt_config.py:

  1. apply geometry (params.recompute_derived) + build mesh
  2. ONE uniform-J FEM solve — the problem is linear in I (no iron), so
     B(x; I) = (I/I_ref)·B_ref(x) exactly; everything below reuses it
  3. per-cell quench current: bisection on  I = Ic(|B_ref|·I/I_ref, θ)
     → I_quench = min over cells; operating point I_op = I_quench / SF
  4. target-area field + peak-to-peak uniformity at the bore-box grid
     (multi-filament Biot-Savart: one racetrack filament per ~100 turns,
     so the layer distribution is resolved; both coils, mirror image)
  5. Bean-state screening estimate (see bean_moments): the analytic
     fully-penetrated current redistribution across each tape gives a
     per-cell magnetization → dipole-sum SCIF at the bore AND the
     SCIF-corrected uniformity used for the PASS check, plus the local
     current-amplification factor 1/i relevant to screening-current
     stress.  Ranking proxy — finalists still get the per-layer T-A.
  6. mechanical screen at I_op (cap hoop max, delamination tension max;
     uniform-J forces — local screening enhancement reported as J×)
  7. objective = |B_target(I_op)| for candidates that PASS uniformity
     (SCIF-corrected) and the Ic-clipping hygiene check

Outputs: ranked console table, opt_results.csv, opt_overview.png.

Run from Racetrack_v4 root:
    conda run -n fenicsx-env python3 optimize/optimize_geometry.py

NOT included in the screen (check finalists separately): full T-A SCIF,
thermal/quench dynamics, structural analysis.
"""
import os, sys, csv, time, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve"),
           os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params
import opt_config as cfg

from mpi4py import MPI
import dolfinx.mesh as dmesh
import solve as base_solve
import build_mesh
from ic_model import IcModel, NValueModel, angle_with_normal_deg
from current_source import normal_xy, tangent_xy
from coil2_field import compute_field_from_coil_at_z

I_REF = 200.0   # reference solve current [A] (results are exact in I)


# ── candidate list ───────────────────────────────────────────────────────────

def candidates():
    if cfg.CANDIDATES is not None:
        return list(cfg.CANDIDATES)
    return [dict(a=a, b=b, n_turns=list(nt))
            for a, b, nt in itertools.product(
                cfg.A_VALUES, cfg.B_VALUES, cfg.N_TURNS_VARIANTS)]


def apply_geometry(cand):
    params.a = float(cand["a"])
    params.b = float(cand["b"])
    params.n_turns = list(cand["n_turns"])
    if "coil_half_gap" in cand:
        params.coil_half_gap = float(cand["coil_half_gap"])
    for k, v in cfg.SCREEN_MESH_OVERRIDES.items():
        setattr(params, k, v)
    params.recompute_derived()


# ── multi-filament Biot-Savart for the target box ───────────────────────────

def filament_stack():
    """(r_center, z_center, turns) sub-filaments resolving the layer
    distribution: one racetrack filament per ~FILAMENT_TURNS_PER_GROUP
    turns, placed at the group's radial centreline and its layer's z."""
    fils = []
    for i, n_i in enumerate(params.n_turns):
        n_sub = max(1, int(round(n_i / cfg.FILAMENT_TURNS_PER_GROUP)))
        edges = np.linspace(params.a_out - n_i * params.t, params.a_out,
                            n_sub + 1)
        z_c = 0.5 * (params.layer_z_tops[i] + params.layer_z_bottoms[i])
        for j in range(n_sub):
            fils.append((0.5 * (edges[j] + edges[j + 1]), z_c, n_i / n_sub))
    return fils


def target_box_field(I_per_turn):
    """B on the target-box grid at the bore midplane, both coils
    (coil 2 = mirror image about z = coil_half_gap).
    Returns (B, grid_points)."""
    g = float(params.coil_half_gap)
    xs = np.linspace(-cfg.TARGET_X_M / 2, cfg.TARGET_X_M / 2, cfg.TARGET_NX)
    ys = np.linspace(-cfg.TARGET_Y_M / 2, cfg.TARGET_Y_M / 2, cfg.TARGET_NY)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, g)])

    L = params.b - params.a
    B = np.zeros_like(pts)
    for r_c, z_c, turns in filament_stack():
        kw = dict(I_per_turn=I_per_turn, n_turns=turns,
                  a=r_c, b=r_c + L, n_straight=150, n_cap=100)
        B += compute_field_from_coil_at_z(pts, z_center=z_c, **kw)
        B += compute_field_from_coil_at_z(pts, z_center=2 * g - z_c, **kw)
    return B, pts


# ── Bean-state screening estimate (analytic, no T-A solve) ──────────────────
# The converged per-layer T-A showed the tapes are DEEPLY field-penetrated:
# the strip penetration field B_c = μ0·(Ic/w)/π is ~50 mT while the local
# perpendicular field B_n is 1–5 T.  In that regime the current profile
# across the tape width is the fully-penetrated Bean state — J = ±Jc(B,θ)
# split so the net equals the transport current, reversal side set by
# sign(B_n) — and each tape column carries the analytic magnetization
#     |M| = Jc_hom · (w/4) · (1 − i²)     [A/m],   i = I/Ic(B,θ)
# directed along −sign(B_n)·n̂ (opposing the ramped flux).  A taper
# min(1, |B_n|/(3·B_c)) handles the few near-B_n=0 cells that are not
# fully penetrated.  The SCIF and its uniformity impact then follow from
# a dipole sum with the same 8-piece mirror expansion as dB_bore_from_dJ.
# This is a RANKING PROXY calibrated against the T-A solver — finalists
# still get the real per-layer T-A.
# CALIBRATION (2026-07-14, baseline geometry, 200 A): Bean-dipole
# estimate +92.3 mT vs converged per-layer T-A +82.0 mT — within 13%
# (T-A/Bean ratio 0.89).  Estimates are reported RAW (uncalibrated).

def bean_moments(B_op, n_hat, theta, ic_model, I_op):
    """Per-cell magnetization vector [A/m] and distribution diagnostics."""
    Bmag = np.linalg.norm(B_op, axis=1)
    Ic, _ = ic_model.critical_current(Bmag, theta)
    i_loc = np.clip(I_op / Ic, 0.0, 0.999)
    B_n   = np.einsum("ij,ij->i", B_op, n_hat)

    Bc    = 4e-7 * np.pi * (Ic / params.w) / np.pi     # strip penetration field
    taper = np.clip(np.abs(B_n) / (3.0 * Bc), 0.0, 1.0)

    M_mag = (Ic / (params.t * params.w)) * (params.w / 4.0) * (1.0 - i_loc**2)
    M_vec = (-np.sign(B_n) * taper * M_mag)[:, None] * n_hat

    diag = dict(
        J_amp_max=float(np.max(1.0 / np.maximum(i_loc, 1e-3))),
        J_amp_mean=float(np.mean(1.0 / np.maximum(i_loc, 1e-3))),
        frac_unpenetrated=float(np.mean(taper < 0.99)))
    return M_vec, diag


def dipole_field_mirrored(points, cents, M_vec, vols):
    """
    B [T] at `points` from per-cell magnetization M_vec [A/m] on the
    quarter-domain cells, expanded to the full two-coil system.
    Mirror transforms for the (in-plane) moments — axial-vector rules
    consistent with the current-pattern mirrors used in dB_bore_from_dJ:
      x-mirror: m → (−mx, +my, mz);  y-mirror: m → (+mx, −my, mz);
      PMC z-mirror (coil 2): m → (−mx, −my, mz).
    """
    g = float(params.coil_half_gap)
    pts = np.asarray(points, dtype=float)

    # Assemble ALL 8 mirror images as one source array (single pass)
    src_c, src_m = [], []
    for sx in (1, -1):
        for sy in (1, -1):
            c = cents.copy(); c[:, 0] *= sx; c[:, 1] *= sy
            m = M_vec.copy()
            if sx < 0:
                m[:, 0] *= -1.0
            if sy < 0:
                m[:, 1] *= -1.0
            for coil in (1, 2):
                cc, mm = c, m
                if coil == 2:
                    cc = c.copy(); cc[:, 2] = 2.0 * g - cc[:, 2]
                    mm = m * np.array([-1.0, -1.0, 1.0])
                src_c.append(cc)
                src_m.append(mm * vols[:, None])          # A·m² per cell
    src_c = np.vstack(src_c)
    mom   = np.vstack(src_m)

    B = np.zeros_like(pts)
    for i0 in range(0, len(pts), 512):                    # chunk points
        p = pts[i0:i0 + 512]
        r = p[:, None, :] - src_c[None, :, :]
        rm = np.linalg.norm(r, axis=2)
        rh = r / rm[:, :, None]
        mdr = np.einsum("pnk,nk->pn", rh, mom)
        B[i0:i0 + 512] += 1e-7 * np.sum(
            (3.0 * mdr[:, :, None] * rh - mom[None, :, :])
            / rm[:, :, None]**3, axis=1)
    return B


# ── quench current from the linear FEM field ─────────────────────────────────

def quench_current(B_unit_mag, theta, ic_model):
    """
    Per-cell quench current by bisection on f(I) = I − Ic(B_unit·I, θ).
    B_unit_mag: |B| per cell at 1 A/turn.  Returns (I_q_cells, clip_frac
    evaluated at the global quench point).
    """
    lo = np.full_like(B_unit_mag, 1.0)
    hi = np.full_like(B_unit_mag, cfg.I_MAX_SEARCH_A)
    ic_hi, _ = ic_model.critical_current(B_unit_mag * hi, theta)
    never = hi <= ic_hi          # not quenched even at I_MAX_SEARCH
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        ic_mid, _ = ic_model.critical_current(B_unit_mag * mid, theta)
        below = mid <= ic_mid    # still superconducting at mid
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    I_q = np.where(never, np.inf, 0.5 * (lo + hi))

    iq_min = float(np.min(I_q))
    _, extra = ic_model.critical_current(
        B_unit_mag * min(iq_min, cfg.I_MAX_SEARCH_A), theta)
    try:
        clip_frac = float(np.mean(extra))
    except Exception:
        clip_frac = float("nan")
    return I_q, clip_frac


# ── stress screen (compact form of validation/mechanical_stress_check) ──────

def stress_screen(cents, B_unit, vol, I_op, f_scale=None):
    """f_scale: optional per-cell force multiplier (screening-current
    stress: Bean saturated bands carry Jc = (1/i)·Je locally)."""
    L  = params.b - params.a
    Je = I_op / (params.t * params.w)
    tx, ty = tangent_xy(cents[:, 0], cents[:, 1], L)
    nx, ny = normal_xy(cents[:, 0], cents[:, 1], L)
    t_hat = np.column_stack([tx, ty, np.zeros(len(cents))])
    n_hat = np.column_stack([nx, ny, np.zeros(len(cents))])
    f = np.cross(Je * t_hat, B_unit * I_op)          # [N/m³] at I_op
    f_n = np.einsum("ij,ij->i", f, n_hat)
    if f_scale is not None:
        f_n = f_n * f_scale

    cap = np.abs(cents[:, 0]) > L
    cx = np.where(cents[cap, 0] > 0, L, -L)
    r_c = np.hypot(cents[cap, 0] - cx, cents[cap, 1])
    hoop_max = float(np.max(f_n[cap] * r_c)) if cap.any() else 0.0

    # transverse tension across the winding, worst layer (straight legs)
    straight = np.abs(cents[:, 0]) < 0.6 * L
    zc = np.array([(t_ + b_) / 2 for t_, b_ in
                   zip(params.layer_z_tops, params.layer_z_bottoms)])
    assign = np.argmin(np.abs(cents[:, 2][:, None] - zc[None, :]), axis=1)
    delam_max = 0.0
    for li in range(params.n_layers):
        m = straight & (assign == li)
        if m.sum() < 5:
            continue
        bins = np.linspace(params.a_inner_list[li], params.a_out, 25)
        ib = np.digitize(cents[m, 1], bins)
        fb = np.array([f_n[m][ib == k].mean() if (ib == k).any() else 0.0
                       for k in range(1, len(bins))])
        sig = -np.cumsum((fb * np.diff(bins))[::-1])[::-1]
        delam_max = max(delam_max, float(sig.max()))
    return hoop_max, delam_max


# ── evaluate one candidate ───────────────────────────────────────────────────

def evaluate(cand, ic_model, comm):
    t0 = time.time()
    label = (f"a={cand['a']*1e3:.0f} b={cand['b']*1e3:.0f} "
             f"nt={sum(cand['n_turns'])}")
    try:
        apply_geometry(cand)
    except AssertionError as e:
        return dict(label=label, feasible=False, reason=str(e))

    # mesh + one FEM solve at I_REF
    from dolfinx.io import gmsh as gmshio
    build_mesh.build(write_path=params.mesh_filename)
    md = gmshio.read_from_msh(params.mesh_filename, comm, rank=0, gdim=3)
    domain = md.mesh
    us = base_solve.setup_problem(domain, md.cell_tags, md.facet_tags)
    _, B_h = base_solve.solve_at_current(domain, us, I_REF, comm,
                                         verbose_label=label)
    cells = us["coil_cells"]
    cents = dmesh.compute_midpoints(domain, domain.topology.dim, cells)
    B_ref = B_h.x.array.reshape(-1, 3)[cells]
    B_unit = B_ref / I_REF                            # per A/turn

    import ufl
    from dolfinx import fem
    Vv = fem.functionspace(domain, ("DG", 0))
    vf = fem.Function(Vv)
    vf.interpolate(fem.Expression(ufl.CellVolume(domain),
                                  Vv.element.interpolation_points))
    vol = vf.x.array[cells]

    # quench
    L = params.b - params.a
    nx, ny = normal_xy(cents[:, 0], cents[:, 1], L)
    n_hat = np.column_stack([nx, ny, np.zeros(len(cells))])
    theta = angle_with_normal_deg(B_ref, n_hat)       # I-independent
    I_q_cells, clip_frac = quench_current(
        np.linalg.norm(B_unit, axis=1), theta, ic_model)
    I_q = float(np.min(I_q_cells))

    # Operating point = min of the quench-, hoop-, and delamination-
    # limited currents.  Stress ∝ I² exactly (linear field), so the
    # stress limits have closed forms from a single reference evaluation.
    hoop_1, delam_1 = stress_screen(cents, B_unit, vol, 1.0)   # at 1 A
    I_hoop  = np.sqrt(cfg.SIGMA_HOOP_MAX_PA  / hoop_1)  if hoop_1  > 0 else np.inf
    I_delam = np.sqrt(cfg.SIGMA_DELAM_MAX_PA / delam_1) if delam_1 > 0 else np.inf
    limits  = {"quench/SF": I_q / cfg.SAFETY_FACTOR,
               "hoop": I_hoop, "delam": I_delam}
    binding = min(limits, key=limits.get)
    I_op    = limits[binding]

    # Screening-current stress: the Bean saturated bands carry
    # Jc = (1/i)·Je locally, and those bands form contiguous slabs
    # through the stack — re-check the delamination limit with the
    # amplified force.  The amplification depends on I (through i and
    # Jc(B)), so iterate the fixed point twice.
    B_umag = np.linalg.norm(B_unit, axis=1)
    amp_cap = float(getattr(cfg, "SCREENING_AMP_CAP", 5.0))
    use_local = getattr(cfg, "SCREENING_STRESS_MODE", "local") == "local"
    delam_scr_1 = None
    for _ in range(2):
        Ic_now, _ = ic_model.critical_current(B_umag * I_op, theta)
        amp = np.clip(Ic_now / I_op, 1.0, amp_cap)   # = 1/i, ≥ 1, capped
        _, delam_scr_1 = stress_screen(cents, B_unit, vol, 1.0, f_scale=amp)
        I_delam_scr = (np.sqrt(cfg.SIGMA_DELAM_MAX_PA / delam_scr_1)
                       if delam_scr_1 > 0 else np.inf)
        if use_local and I_delam_scr < I_op:
            I_op, binding = I_delam_scr, "delam-scr"
        else:
            break
    delam_scr = delam_scr_1 * I_op**2                # reported either way

    # target-area field + Bean-state screening (radial current
    # redistribution): SCIF at the bore and its uniformity impact
    B_box, box_pts = target_box_field(I_op)
    B_op = B_unit * I_op
    M_vec, bean = bean_moments(B_op, n_hat, theta, ic_model, I_op)
    dB_box  = dipole_field_mirrored(box_pts, cents, M_vec, vol)
    g = float(params.coil_half_gap)
    scif_mT = float(dipole_field_mirrored(
        np.array([[0.0, 0.0, g]]), cents, M_vec, vol)[0, 2] * 1e3)

    Bmag0 = np.linalg.norm(B_box, axis=1)
    Bmag  = np.linalg.norm(B_box + dB_box, axis=1)
    B_tgt = float(np.mean(np.abs((B_box + dB_box)[:, 2])))
    unif0 = float((Bmag0.max() - Bmag0.min()) / Bmag0.mean() * 100.0)
    unif  = float((Bmag.max() - Bmag.min()) / Bmag.mean() * 100.0)

    hoop, delam = hoop_1 * I_op**2, delam_1 * I_op**2
    peak_B = float(np.max(B_umag) * I_op)            # data-ceiling indicator

    # hygiene: FEM vs filament bore-field cross-check (model error bar).
    # Evaluate a few mm INSIDE the domain and OFF all symmetry planes —
    # points on the domain boundary defeat the FEM point locator.
    fem_dev = float("nan")
    chk_pt = np.array([[0.004, 0.003, g - 0.004]])
    fem_val, fem_mask = base_solve.evaluate_at_points(
        domain, B_h, chk_pt, comm)
    if fem_mask is not None and fem_mask[0]:
        Bz_fem = fem_val[0, 2] / I_REF * I_op
        L_ = params.b - params.a
        Bz_fil = 0.0
        for r_c, z_c, turns in filament_stack():
            kw = dict(I_per_turn=I_op, n_turns=turns, a=r_c, b=r_c + L_,
                      n_straight=150, n_cap=100)
            Bz_fil += compute_field_from_coil_at_z(
                chk_pt, z_center=z_c, **kw)[0, 2]
            Bz_fil += compute_field_from_coil_at_z(
                chk_pt, z_center=2 * g - z_c, **kw)[0, 2]
        fem_dev = abs(Bz_fem - Bz_fil) / abs(Bz_fil) * 100.0

    # clip fraction re-evaluated at the operating field
    _, extra = ic_model.critical_current(
        np.linalg.norm(B_unit, axis=1) * I_op, theta)
    try:
        clip_op = float(np.mean(extra))
    except Exception:
        clip_op = float("nan")

    ok_unif = unif <= cfg.UNIFORMITY_MAX_PCT
    ok_clip = clip_op <= cfg.MAX_CLIP_FRACTION
    passed  = ok_unif and ok_clip

    return dict(label=label, feasible=True,
                a_mm=cand["a"]*1e3, b_mm=cand["b"]*1e3,
                n_turns=str(cand["n_turns"]), n_total=sum(cand["n_turns"]),
                tape_km=params.tape_length_m/1e3,
                I_quench_A=I_q, I_op_A=I_op, binding=binding,
                B_target_T=B_tgt,
                uniformity_pct=unif, uniformity_noscif_pct=unif0,
                scif_mT=scif_mT,
                J_amp_max=bean["J_amp_max"], J_amp_mean=bean["J_amp_mean"],
                frac_unpen=bean["frac_unpenetrated"],
                hoop_MPa=hoop/1e6, delam_MPa=delam/1e6,
                delam_scr_MPa=delam_scr/1e6,
                peak_B_T=peak_B, fem_dev_pct=fem_dev,
                clip_frac=clip_op,
                ok_unif=ok_unif, ok_clip=ok_clip, PASS=passed,
                eval_s=time.time() - t0)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    comm = MPI.COMM_WORLD
    ic_model = IcModel()
    cands = candidates()
    print(f"Evaluating {len(cands)} candidates "
          f"(SF={cfg.SAFETY_FACTOR}, box "
          f"{cfg.TARGET_X_M*1e3:.0f}×{cfg.TARGET_Y_M*1e3:.0f} mm, "
          f"uniformity ≤ {cfg.UNIFORMITY_MAX_PCT}%)\n")

    rows = []
    for i, cand in enumerate(cands):
        r = evaluate(cand, ic_model, comm)
        rows.append(r)
        if not r.get("feasible", False):
            print(f"[{i+1}/{len(cands)}] {r['label']}: INFEASIBLE "
                  f"({r.get('reason','')})")
            continue
        marks = "".join(m if ok else m.upper() for m, ok in
                        [("u", r["ok_unif"]), ("c", r["ok_clip"])])
        print(f"[{i+1}/{len(cands)}] {r['label']}: "
              f"B_tgt={r['B_target_T']:.2f} T @ I_op={r['I_op_A']:.0f} A "
              f"({r['binding']}-limited; Iq={r['I_quench_A']:.0f})  "
              f"unif={r['uniformity_pct']:.3f}% "
              f"(geom {r['uniformity_noscif_pct']:.3f}%)  "
              f"SCIF={r['scif_mT']:+.0f} mT  "
              f"J×{r['J_amp_mean']:.1f}  "
              f"hoop={r['hoop_MPa']:.0f}  delam={r['delam_MPa']:.1f}"
              f"(scr {r['delam_scr_MPa']:.1f}) MPa  "
              f"peakB={r['peak_B_T']:.1f} T  "
              f"femΔ={r['fem_dev_pct']:.1f}%  "
              f"clip={r['clip_frac']:.2f}  "
              f"[{'PASS' if r['PASS'] else 'fail:'+marks}]  "
              f"{r['eval_s']:.0f} s")

    good = [r for r in rows if r.get("feasible")]
    good.sort(key=lambda r: (not r["PASS"], -r["B_target_T"]))

    # CSV
    os.makedirs(os.path.dirname(os.path.join(_ROOT, cfg.OUT_CSV)),
                exist_ok=True)
    keys = [k for k in good[0].keys() if k not in ("feasible",)]
    with open(os.path.join(_ROOT, cfg.OUT_CSV), "w", newline="") as fcsv:
        wcsv = csv.DictWriter(fcsv, fieldnames=keys, extrasaction="ignore")
        wcsv.writeheader()
        wcsv.writerows(good)

    # ranked table
    print(f"\n{'='*100}")
    print(f"  RANKED (PASS first, then |B_target| at I_op)")
    print(f"{'='*100}")
    print(f"  {'label':32s} {'B_tgt':>6} {'I_op':>6} {'limit':>10} "
          f"{'unif%':>7} {'SCIF':>6} {'peakB':>6} {'clip':>5} "
          f"{'tape':>6} {'PASS':>5}")
    for r in good:
        print(f"  {r['label']:32s} {r['B_target_T']:>6.2f} "
              f"{r['I_op_A']:>6.0f} {r['binding']:>10} "
              f"{r['uniformity_pct']:>7.3f} {r['scif_mT']:>+6.0f} "
              f"{r['peak_B_T']:>6.1f} {r['clip_frac']:>5.2f} "
              f"{r['tape_km']:>5.2f}k "
              f"{'YES' if r['PASS'] else '-':>5}")

    _plot(good)
    print(f"\nCSV → {cfg.OUT_CSV}\nPlot → {cfg.OUT_PNG}")


def _plot(rows):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.patch.set_facecolor("#111")
    for ax in axes:
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.grid(True, alpha=0.2, color="#555")

    x = np.arange(len(rows))
    colors = ["limegreen" if r["PASS"] else "tomato" for r in rows]
    labels = [r["label"].replace(" nt=", "\n") for r in rows]

    axes[0].bar(x, [r["B_target_T"] for r in rows], color=colors)
    axes[0].axhline(10.0, color="lime", ls=":", lw=1.2, label="10 T target")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=6,
                                                   color="white")
    axes[0].set_ylabel("|B| target area @ I_op  [T]", color="white")
    axes[0].set_title("Objective (green = all constraints pass)",
                      color="white", fontsize=10)
    axes[0].legend(fontsize=8, labelcolor="white", facecolor="#222")

    axes[1].bar(x, [r["uniformity_pct"] for r in rows], color=colors)
    axes[1].axhline(cfg.UNIFORMITY_MAX_PCT, color="yellow", ls="--", lw=1.2,
                    label=f"{cfg.UNIFORMITY_MAX_PCT}% limit")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=6,
                                                   color="white")
    axes[1].set_ylabel("peak-to-peak uniformity  [%]", color="white")
    axes[1].set_title("Uniformity over target box", color="white",
                      fontsize=10)
    axes[1].legend(fontsize=8, labelcolor="white", facecolor="#222")

    for r in rows:
        axes[2].scatter(r["tape_km"], r["B_target_T"],
                        c="limegreen" if r["PASS"] else "tomato", s=45)
        axes[2].annotate(f"{r['a_mm']:.0f}/{r['n_total']}",
                         (r["tape_km"], r["B_target_T"]),
                         fontsize=6, color="white",
                         xytext=(4, 3), textcoords="offset points")
    axes[2].set_xlabel("tape length  [km]", color="white")
    axes[2].set_ylabel("|B| target @ I_op  [T]", color="white")
    axes[2].set_title("Field vs conductor cost (a[mm]/turns)",
                      color="white", fontsize=10)

    fig.suptitle("Configuration sweep — uniform-J screen "
                 f"(SF={cfg.SAFETY_FACTOR}; SCIF & T-A on finalists only)",
                 color="white", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(_ROOT, cfg.OUT_PNG), dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
