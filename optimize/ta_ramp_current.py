"""
ta_ramp_current.py — largest current with NO cell above Jc at the end of a
constant-power ramp (T-A), for one candidate. Drop-in for
ta_safe_current.evaluate() (same return keys).
=============================================================================
2026-09-27. Criterion (user decision): the magnet is ramped by a
constant-POWER supply within t_ramp, and at the moment the ramp ends
no coil cell may exceed its local Jc(B, theta) (E_c = 1 uV/cm, measured
Shanghai 20 K data + Kim extrapolation above 8 T). t_ramp was 2 h until
2026-10-05; it is now 30 min (user decision), i.e. a 1 h surrogate step.

Constant power into an inductive coil: (1/2) L I^2 = P t, so
    I(t) = I_op * sqrt(t / t_ramp),   dI/dt at the end = I_op / (2 t_ramp)
i.e. the final ramp RATE equals that of a linear ramp over 2 * t_ramp. The
end-of-ramp overload is set by the electric field at that moment
(J/Jc = (E/E_c)^(1/n)), so the SURROGATE used here is one backward-Euler
T-A step from the virgin state over dt_eff = 2 * t_ramp = 1 h (the single-
step form every production T-A solve already uses, with the step length
changed). It must be validated against a real march along the sqrt(t)
profile (ramp_hold_test.py) before its numbers are trusted, and finalists
must be re-checked with that march at a lower rho-floor and a finer mesh.

Numerics follow the 2026-09-26/27 findings (CLAUDE.md, "10 T with NO cell
above Jc: ramp + hold"), re-tuned 2026-10-04 after the T-equation
resistivity fix (ta_solve._update_rho lost a spurious delta/Lambda factor):
  * rho-floor 0.6 (ta_eps_reg), reached on the first (cold) solve by
    continuation 1.0 -> 0.9 -> 0.8 -> 0.7 -> 0.6.  With the corrected rho
    the floor matters far more.  Legacy 6-layer design, 4 h step, floors
    1.0/0.9/0.8/0.7/0.6/0.5 (optimize/runs/ramp_hold/floortest_*):
      126.6 A  worst load 0.274/0.284/0.557/0.837/0.838/-
      180 A    worst load 0.508/0.508/0.741/0.881/0.881/0.883,
               B 10.68/10.67/10.59/10.41/10.16/10.12 T,
               uniformity 1.16/1.07/0.93/0.77/0.47/0.45 %
      300 A    worst load 1.028/0.980/0.979/0.977/0.978/0.978,
               B 17.81/17.80/17.77/17.74/17.64/17.63 T
    The worst load converges once the floor is ~0.1 below the load
    itself; B and uniformity need 0.6 (0.7 overstates B by up to 3.5 %,
    0.6 -> 0.5 changes it by <= 0.4 %).
  * alpha sized to rho_floor*dt, forced full-length Picard. The ladder
    was tuned at dt = 4 h; for a shorter step the alphas of the floors
    <= 0.7 are scaled by dt/4 h and their iteration counts by 4 h/dt
    (stable alpha ~ rho_floor * dt, and rho_floor ~ eps^(n-1), so only
    the low floors are near the limit). Cost: at dt = 1 h the floor-0.7
    and floor-0.6 stages take 4x the iterations of the 4 h settings.
  * convergence (2026-10-05, error hunt): the I_op solve must have a flat
    raw SCIF trace (range over the last 30 iterations < SCIF_CONV_MT) AND
    a settled gated load (range over the last 30 iterations, and its
    trend extrapolated to the fixed point, both < LOAD_CONV_TOL). The
    error hunt's nonlinear benchmark showed relaxation hotspots that the
    global SCIF cannot see. Reported as `converged`; the runner does not
    accept unconverged designs.
  * the gate (2026-10-05, user decision) is the LOAD_PCTL = 99.99th
    percentile BY CELL COUNT (nearest rank: an actual cell's value) of
    load = |J_inplane|/Jc over the coil, sliver cells excluded. Below
    10,000 coil cells this is the worst cell; above, it skips the top
    1-3 cells. It keeps the "no cell above Jc" meaning while dropping
    single-element artefacts: eval 166 (29,367 cells) 1.103 (sliver)
    -> 0.993 (true value 0.994). Volume weighting was tried and rejected:
    the most-loaded cells are the small tape-edge cells, so it ran ~7 %
    below the worst cell (legacy 185 A: 0.854 vs 0.921), nearly as loose
    as p99.9. The plain worst cell is still reported (ta_worst_load).
    See docs/error_hunt_2026-10-04.md, "Actions taken".

Search for I_op: the worst load rises only slowly with current once the
coil is penetrated (roughly load ~ I^0.15), and steeply below that. A
secant search in log(load) vs log(I) with bracketing, starting from the
uniform-J quench current, reaches |load - LOAD_MAX| < LOAD_TOL in ~3-4
solves (see _next_current). The reported I_op is always a current that was actually solved and
satisfied the criterion (never an interpolated value).

Mesh: like ta_safe_current.py this uses whatever mesh settings params
holds when called -- the caller must force the medium tier in memory
(params.py on disk is the xdense tier; see ta_safe_margin_search.py).
"""
import os
import csv
import time
import tempfile

import numpy as np

import params
import opt_config as cfg
import optimize_geometry as og
import build_mesh
import solve as base_solve
import ta_solve
import dolfinx.mesh as dmesh
from current_source import normal_xy
from ic_model import angle_with_normal_deg, NValueModel
from ta_safe_current import _apply, _BASE_MESH_FILENAME

T_RAMP_S = float(os.environ.get("TA_RAMP_T_RAMP_S", 1800.0))   # 30 min
DT_EFF_S = 2.0 * T_RAMP_S           # constant-power end rate == linear over 2*t_ramp
LOAD_MAX = float(os.environ.get("TA_RAMP_LOAD_MAX", 1.0))       # worst J/Jc allowed
LOAD_TOL = 0.01                     # stop when a passing point is within 1%
LOAD_AIM = LOAD_MAX * (1.0 - LOAD_TOL / 2.0)   # search target: the middle
                                    # of the accepted window, not its edge
EPS_FLOOR = float(os.environ.get("TA_RAMP_EPS", 0.6))
MAX_SOLVES = int(os.environ.get("TA_RAMP_MAX_SOLVES", 5))
I_FLOOR_A = 5.0
SCIF_CONV_MT = 0.5                  # max SCIF range over last 30 iters
# Gated load statistic: percentile by cell count over the coil cells,
# slivers excluded: tet quality gamma = 3 * inradius / circumradius (gmsh's
# measure, 1 = regular) below SLIVER_Q. Not V/l_max^3, which also flags the
# thin graded tape-edge cells (28 of 4,463 on the legacy medium mesh vs 0).
LOAD_PCTL = float(os.environ.get("TA_RAMP_LOAD_PCTL", 99.99))
SLIVER_Q = 1e-3
# Local convergence of the gated load over the last CONV_WINDOW iterations
# of the I_op solve: its range, and |slope|/alpha (the remaining distance
# to the fixed point for the measured ~(1 - alpha)^k decay), both below
# LOAD_CONV_TOL.
CONV_WINDOW = 30
LOAD_CONV_TOL = float(os.environ.get("TA_RAMP_LOAD_CONV_TOL", 0.005))
# Cold ladder above the target floor: (floor, alpha, forced iterations),
# then STAGE_FLOOR = (alpha, forced iterations) for every solve at
# EPS_FLOOR. Tuned at dt = 4 h (_REF_DT_S); see the module docstring for
# the scaling to shorter steps.
_REF_DT_S = 14400.0
_DT_SCALE = min(1.0, DT_EFF_S / _REF_DT_S)


def _for_dt(eps, alpha, n_it):
    if eps > 0.7 or _DT_SCALE == 1.0:
        return alpha, n_it
    return alpha * _DT_SCALE, int(round(n_it / _DT_SCALE))


COLD_LADDER = [(e,) + _for_dt(e, a, n) for e, a, n in
               [(1.0, 0.10, 150), (0.9, 0.10, 100), (0.8, 0.10, 120),
                (0.7, 0.05, 150)]]
STAGE_FLOOR = _for_dt(EPS_FLOOR, 0.01, 300)
# TA_RAMP_ITER_SCALE < 1 shortens every solve (and MAX_SOLVES) for machinery
# tests only (run_optimization.py --quick-test) -- results are then NOT converged.
_scale = float(os.environ.get("TA_RAMP_ITER_SCALE", "1"))
if _scale != 1.0:
    COLD_LADDER = [(e, a, max(20, int(n * _scale))) for e, a, n in COLD_LADDER]
    STAGE_FLOOR = (STAGE_FLOOR[0], max(20, int(STAGE_FLOOR[1] * _scale)))
    MAX_SOLVES = min(MAX_SOLVES, 3)


def _cell_ctx(domain, ta):
    """Per-coil-cell data for the load statistic: tape normals (as
    ta_safe_current._local_margin computes them) and the non-sliver
    mask."""
    cells = ta["coil_cells"]
    cents = dmesh.compute_midpoints(domain, domain.topology.dim, cells)
    nx, ny = normal_xy(cents[:, 0], cents[:, 1], params.L)
    v = domain.geometry.x[domain.geometry.dofmaps[0][cells]]
    assert v.shape[1] == 4, "expected linear tetrahedra"
    vol_geo = np.abs(np.linalg.det(v[:, 1:] - v[:, :1])) / 6.0
    area = sum(0.5 * np.linalg.norm(np.cross(v[:, j] - v[:, i],
                                             v[:, k] - v[:, i]), axis=1)
               for i, j, k in ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)))
    a, b, c = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]
    sq = lambda u: (u ** 2).sum(1)[:, None]
    circ = np.linalg.norm(sq(a) * np.cross(b, c) + sq(b) * np.cross(c, a)
                          + sq(c) * np.cross(a, b), axis=1) / 12.0
    # gamma = 3 r_in / R with r_in = 3V/area, R = circ/V (no division by V)
    quality = 9.0 * vol_geo ** 2 / (area * circ + 1e-300)
    return dict(n_hat=np.column_stack([nx, ny, np.zeros_like(nx)]),
                keep=quality >= SLIVER_Q)


def _cell_loads(ctx, J, B, ic_model):
    """Per-cell load |J_inplane| / Jc(B, theta) (same formula as
    ta_safe_current._local_margin, inverted)."""
    n_hat = ctx["n_hat"]
    Ic_A, _ = ic_model.critical_current(np.linalg.norm(B, axis=1),
                                        angle_with_normal_deg(B, n_hat))
    Jc = Ic_A / (params.delta_SC * params.w)
    J_in = J - np.einsum("ij,ij->i", J, n_hat)[:, None] * n_hat
    return np.linalg.norm(J_in, axis=1) / Jc


def gated_load(load, keep=None, pctl=None):
    """Nearest-rank percentile of the per-cell load by cell count: the
    k-th largest load, k = ceil(N * (100 - pctl) / 100) (k = 1, the worst
    cell, for N < 100/(100 - pctl)). keep: optional mask of cells to use
    (slivers excluded)."""
    pctl = LOAD_PCTL if pctl is None else pctl
    if keep is not None:
        load = load[keep]
    k = max(1, int(np.ceil(len(load) * (100.0 - pctl) / 100.0 - 1e-9)))
    return float(np.partition(load, -k)[-k])


def _solve(domain, ta, uniform_setup, I, ic_model, n_model, eps, alpha,
           n_it, warm, diag_path, load_fn=None):
    """One forced-length solve. With load_fn(J, B) -> per-cell loads, also
    returns the gated load of each of the last CONV_WINDOW iterations and
    the last iteration's per-cell loads."""
    params.ta_eps_reg = eps
    params.ta_picard_alpha = params.ta_picard_alpha_fine = alpha
    params.ta_n_picard = n_it
    if os.path.exists(diag_path):
        os.remove(diag_path)
    trace, last = [], {}

    def cb(k, J, B):
        if load_fn is not None and k >= n_it - CONV_WINDOW:
            last["loads"] = load_fn(J, B)
            trace.append(last["gated_fn"](last["loads"]))

    if load_fn is not None:
        last["gated_fn"] = load_fn.gated
    _, B_h, _, info = ta_solve.solve_ta_at_current(
        domain, ta, uniform_setup, I_amps=I, ic_model=ic_model,
        n_model=n_model, verbose=False, warm_start=warm, min_iters=n_it,
        diag_log_path=diag_path, iter_callback=cb)
    with open(diag_path) as fh:
        scif = [float(r["scif_mT"]) for r in csv.DictReader(fh)]
    s = np.array(scif[-30:])
    return B_h, info, float(s.max() - s.min()), trace, last.get("loads")


def _load_at(domain, ta, uniform_setup, I, ic_model, n_model, first,
             diag_path, ctx):
    """One end-of-ramp evaluation at current I. Returns (gated load, worst
    cell load, B_h, scif range, conv) with conv = dict(load_range,
    load_resid) of the gated load over the last CONV_WINDOW iterations."""
    if first:
        warm = False
        for eps, alpha, n_it in COLD_LADDER:
            if eps > EPS_FLOOR:
                _solve(domain, ta, uniform_setup, I, ic_model, n_model, eps,
                       alpha, n_it, warm, diag_path)
                warm = True
    def load_fn(J, B):
        return _cell_loads(ctx, J, B, ic_model)
    load_fn.gated = lambda L: gated_load(L, ctx["keep"])

    B_h, info, rng, trace, loads = _solve(
        domain, ta, uniform_setup, I, ic_model, n_model, EPS_FLOOR,
        *STAGE_FLOOR, True, diag_path, load_fn=load_fn)
    g = np.array(trace)
    slope = np.polyfit(np.arange(len(g)), g, 1)[0] if len(g) > 1 else np.inf
    conv = dict(load_range=float(g.max() - g.min()),
                load_resid=float(abs(slope) / STAGE_FLOOR[0]))
    return float(g[-1]), float(loads.max()), B_h, rng, conv


def _next_current(trials, lo, hi):
    """Next trial current. trials = [(I, load, ...)], lo/hi = the passing /
    failing bracket ends (or None).

    With a bracket: false position in log(load) vs log(I) through the
    bracket ends, aimed at LOAD_AIM, with the Illinois rule (the end not
    refreshed by the last two trials gets half weight) and each step kept
    >= 10 % of the bracket (in log I) away from both ends. Until 2026-09-29
    this was plain false position aimed at LOAD_MAX: with the load curving
    near the crossing it approached 1.0 from above in ~1 % steps (99.6 ->
    96.3 -> 95.4 A at loads 1.008 -> 1.002 -> 1.0005, eval 104 of the
    2026-09-28 run) until the budget ran out, and the design was scored at
    its first, far too low, passing current. On synthetic load curves
    matching that run this cut the mean I_op shortfall from ~12 % to ~3 %
    and the cases > 10 % short from ~38 % to ~3 %, with fewer solves.
    Without a bracket: a log-log step with the typical penetrated-coil
    slope 0.15, at most x4 / /4."""
    if lo is None or hi is None:
        ref = trials[-1]
        I_new = ref[0] * (LOAD_AIM / ref[1]) ** (1.0 / 0.15)
        return min(max(I_new, ref[0] / 4.0), ref[0] * 4.0)
    xl, xh = np.log(lo[0]), np.log(hi[0])
    yl, yh = np.log(lo[1] / LOAD_AIM), np.log(hi[1] / LOAD_AIM)
    if len(trials) >= 3:
        if all(t[1] > LOAD_MAX for t in trials[-2:]):
            yl *= 0.5
        elif all(t[1] <= LOAD_MAX for t in trials[-2:]):
            yh *= 0.5
    x = xl - yl * (xh - xl) / (yh - yl) if yh > yl else 0.5 * (xl + xh)
    x = min(max(x, xl + 0.1 * (xh - xl)), xh - 0.1 * (xh - xl))
    return float(np.exp(x))


def evaluate(design, ic_model, comm, verbose=False, save_fields_path=None,
             n_model=None, min_iters=None):
    t0 = time.time()
    label = (f"a={design['a']*1e3:.1f} b={design['b']*1e3:.1f} "
             f"gap={design['coil_half_gap']*1e3:.1f} "
             f"nt={sum(design['n_turns'])}")
    try:
        _apply(design)
    except AssertionError as e:
        return dict(label=label, feasible=False, reason=str(e))

    from dolfinx.io import gmsh as gmshio
    root, ext = os.path.splitext(_BASE_MESH_FILENAME)
    params.mesh_filename = f"{root}_taramp{os.getpid()}_{int(t0*1000)%100000}{ext}"
    diag_path = os.path.join(tempfile.gettempdir(),
                             f"taramp_{os.getpid()}_{int(t0*1000)%100000}.csv")
    saved = {k: getattr(params, k) for k in
             ("ta_eps_reg", "ta_picard_alpha", "ta_picard_alpha_fine",
              "ta_n_picard")}
    trials = []            # (I, gated load, scif range, worst cell, conv)
    try:
        build_mesh.build(write_path=params.mesh_filename)
        md = gmshio.read_from_msh(params.mesh_filename, comm, rank=0, gdim=3)
        domain = md.mesh
        uniform_setup = base_solve.setup_problem(domain, md.cell_tags,
                                                 md.facet_tags)
        if n_model is None:
            n_model = NValueModel(csv_path=params.n_value_csv_filename)
        ta = ta_solve.setup_ta_problem(domain, md.cell_tags, md.facet_tags,
                                       uniform_setup)
        ta["dt_const"].value = DT_EFF_S
        ctx = _cell_ctx(domain, ta)

        # uniform-J reference: starting bracket + hoop stress
        _, B_h_ref = base_solve.solve_at_current(
            domain, uniform_setup, og.I_REF, comm, verbose_label=label)
        cells = uniform_setup["coil_cells"]
        cents = dmesh.compute_midpoints(domain, domain.topology.dim, cells)
        B_unit = B_h_ref.x.array.reshape(-1, 3)[cells] / og.I_REF
        import ufl
        from dolfinx import fem
        Vv = fem.functionspace(domain, ("DG", 0))
        vf = fem.Function(Vv)
        vf.interpolate(fem.Expression(ufl.CellVolume(domain),
                                      Vv.element.interpolation_points))
        vol = vf.x.array[cells]
        nx, ny = normal_xy(cents[:, 0], cents[:, 1], params.b - params.a)
        n_hat = np.column_stack([nx, ny, np.zeros(len(cells))])
        I_q_cells, clip_frac = og.quench_current(
            np.linalg.norm(B_unit, axis=1),
            angle_with_normal_deg(B_unit, n_hat), ic_model)
        I_q_uniform = float(np.min(I_q_cells))
        if not np.isfinite(I_q_uniform):
            I_q_uniform = cfg.I_MAX_SEARCH_A

        # ── secant search in log(load) vs log(I) ────────────────────────
        best = None                        # (I, load) passing, highest I
        lo = hi = None                     # bracketing (I, load)
        I_try = 0.5 * I_q_uniform
        for k in range(MAX_SOLVES):
            I_try = float(np.clip(I_try, I_FLOOR_A, cfg.I_MAX_SEARCH_A))
            load, worst, B_h, rng, conv = _load_at(
                domain, ta, uniform_setup, I_try, ic_model, n_model,
                first=(k == 0), diag_path=diag_path, ctx=ctx)
            trials.append((I_try, load, rng, worst, conv))
            if verbose:
                print(f"  [ta_ramp] I={I_try:7.2f}A  load p{LOAD_PCTL:g}="
                      f"{load:.4f} (worst cell {worst:.4f})  SCIF range="
                      f"{rng:.2f}mT  load range={conv['load_range']:.4f} "
                      f"resid={conv['load_resid']:.4f}")
            if load <= LOAD_MAX:
                if best is None or I_try > best[0]:
                    best = (I_try, load)
                lo = (I_try, load) if lo is None or I_try > lo[0] else lo
                if LOAD_MAX - load < LOAD_TOL * LOAD_MAX:
                    break
            else:
                hi = (I_try, load) if hi is None or I_try < hi[0] else hi
            if lo is not None and hi is not None and hi[0] / lo[0] < 1.03:
                break
            I_try = _next_current(trials, lo, hi)

        if best is None:
            # nothing passed within the budget: report the lowest-current
            # trial as-is, flagged (not a fabricated placeholder)
            I_op, load_op = min(trials, key=lambda t: t[0])[:2]
            search_ok = False
        else:
            I_op, load_op = best
            search_ok = True
        # final state must correspond to I_op (the loop may have ended on a
        # different, failing trial)
        if trials[-1][0] != I_op:
            load_op, worst, B_h, rng, conv = _load_at(
                domain, ta, uniform_setup, I_op, ic_model, n_model,
                first=False, diag_path=diag_path, ctx=ctx)
            trials.append((I_op, load_op, rng, worst, conv))
        _, load_op, scif_range, worst_op, conv_op = trials[-1]
        scif_ok = scif_range < SCIF_CONV_MT
        load_ok = (conv_op["load_range"] < LOAD_CONV_TOL
                   and conv_op["load_resid"] < LOAD_CONV_TOL)

        # ── field, uniformity, stress at I_op (end-of-ramp state) ──────
        delta_SC = float(ta["delta_SC"])
        J_TA = ta_solve._J_from_T(ta, domain)
        J_unif = ta["t_hat_coil"] * (I_op / (delta_SC * params.w))
        dJ_s = (J_TA - J_unif) * (delta_SC / float(params.t))
        coil_cells = ta["coil_cells"]
        centroids = dmesh.compute_midpoints(domain, domain.topology.dim,
                                            coil_cells)
        B_box_u, box_pts = og.target_box_field(I_op)
        dB_box = np.array([ta_solve.dB_bore_from_dJ(
            centroids, dJ_s, ta["coil_vols"], bore_pt=p) for p in box_pts])
        Btot = B_box_u + dB_box
        B_tgt = float(np.mean(np.abs(Btot[:, 2])))
        Bmag = np.linalg.norm(Btot, axis=1)
        unif = float((Bmag.max() - Bmag.min()) / Bmag.mean() * 100.0)
        hoop_1, delam_1 = og.stress_screen(cents, B_unit, vol, 1.0)

        if save_fields_path:
            np.savez(save_fields_path, I_solved=I_op, delta_SC=delta_SC,
                     coil_centroids=centroids,
                     coil_B=B_h.x.array.reshape(-1, 3)[coil_cells],
                     J_TA_coil=J_TA, J_unif_coil=J_unif,
                     dV=ta["coil_vols"], box_ptp_pct=unif, a=params.a,
                     b=params.b, coil_half_gap=params.coil_half_gap,
                     n_turns=np.array(params.n_turns))
    finally:
        for k, v in saved.items():
            setattr(params, k, v)
        for p in (params.mesh_filename, diag_path):
            try:
                os.remove(p)
            except OSError:
                pass

    return dict(label=label, feasible=True,
                a_mm=design["a"] * 1e3, b_mm=design["b"] * 1e3,
                n_turns=str(design["n_turns"]),
                n_total=sum(design["n_turns"]),
                tape_km=params.tape_length_m / 1e3,
                I_quench_uniform_A=I_q_uniform, I_op_A=I_op,
                binding="ta_end_of_ramp_load",
                B_target_T=B_tgt, uniformity_pct=unif,
                hoop_MPa=hoop_1 * I_op ** 2 / 1e6,
                delam_MPa=delam_1 * I_op ** 2 / 1e6,
                delam_scr_MPa=delam_1 * I_op ** 2 / 1e6,
                ta_worst_margin=1.0 / worst_op,
                ta_constraint_margin=1.0 / load_op,
                ta_worst_load=worst_op,
                ta_load_gated=load_op, load_pctl=LOAD_PCTL,
                load_range=conv_op["load_range"],
                load_resid=conv_op["load_resid"],
                n_sliver_cells=int((~ctx["keep"]).sum()),
                t_ramp_s=T_RAMP_S,
                bisection_exhausted=not search_ok,
                search_ok=search_ok,
                scif_range_mT=scif_range,
                scif_converged=scif_ok, load_converged=load_ok,
                converged=scif_ok and load_ok,
                trials=";".join(f"{t[0]:.1f}:{t[1]:.4f}" for t in trials),
                clip_frac=clip_frac,
                n_ta_solves=len(trials), eval_s=time.time() - t0)
