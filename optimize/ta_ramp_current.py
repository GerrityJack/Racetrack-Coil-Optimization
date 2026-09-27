"""
ta_ramp_current.py — largest current with NO cell above Jc at the end of a
constant-power ramp (T-A), for one candidate. Drop-in for
ta_safe_current.evaluate() (same return keys).
=============================================================================
2026-09-27. Criterion (user decision): the magnet is ramped by a
constant-POWER supply within t_ramp = 2 h, and at the moment the ramp ends
no coil cell may exceed its local Jc(B, theta) (E_c = 1 uV/cm, measured
Shanghai 20 K data + Kim extrapolation above 8 T).

Constant power into an inductive coil: (1/2) L I^2 = P t, so
    I(t) = I_op * sqrt(t / t_ramp),   dI/dt at the end = I_op / (2 t_ramp)
i.e. the final ramp RATE equals that of a linear ramp over 2 * t_ramp. The
end-of-ramp overload is set by the electric field at that moment
(J/Jc = (E/E_c)^(1/n)), so the SURROGATE used here is one backward-Euler
T-A step from the virgin state over dt_eff = 2 * t_ramp = 4 h (the single-
step form every production T-A solve already uses, with the step length
changed). It must be validated against a real march along the sqrt(t)
profile (ramp_hold_test.py) before its numbers are trusted, and finalists
must be re-checked with that march at a lower rho-floor and a finer mesh.

Numerics follow the 2026-09-26/27 findings (CLAUDE.md, "10 T with NO cell
above Jc: ramp + hold"):
  * rho-floor 0.9 (ta_eps_reg), reached by continuation from 1.0 on the
    first (cold) solve -- the end-of-ramp worst load was floor-independent
    to ~0.2% from 0.9 down; the production floor 1.0 under-states it ~5%.
  * alpha sized to rho_floor*dt (stable, converges well within the forced
    iteration count at dt = 4 h), forced full-length Picard.
  * convergence judged on the raw SCIF trace (range over the last 30
    iterations), NOT rel_err; reported as scif_range_mT / converged.
  * the gate is the WORST cell (no percentile), load = |J_inplane|/Jc.

Search for I_op: the worst load rises only slowly with current once the
coil is penetrated (roughly load ~ I^0.15), and steeply below that. A
secant search in log(load) vs log(I) with bracketing, starting from the
uniform-J quench current, reaches |load - LOAD_MAX| < LOAD_TOL in ~3-4
solves. The reported I_op is always a current that was actually solved and
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
from ta_safe_current import _apply, _local_margin, _BASE_MESH_FILENAME

T_RAMP_S = float(os.environ.get("TA_RAMP_T_RAMP_S", 7200.0))   # 2 h
DT_EFF_S = 2.0 * T_RAMP_S           # constant-power end rate == linear over 2*t_ramp
LOAD_MAX = float(os.environ.get("TA_RAMP_LOAD_MAX", 1.0))       # worst J/Jc allowed
LOAD_TOL = 0.01                     # stop when a passing point is within 1%
EPS_FLOOR = float(os.environ.get("TA_RAMP_EPS", 0.9))
MAX_SOLVES = int(os.environ.get("TA_RAMP_MAX_SOLVES", 5))
I_FLOOR_A = 5.0
SCIF_CONV_MT = 0.5                  # max SCIF range over last 30 iters
# (alpha, forced iterations) for: the cold solve at floor 1.0, then every
# solve at EPS_FLOOR. Stable alpha scales ~ rho_floor*dt; at dt = 4 h these
# are well inside the stable range measured on 2026-09-27.
STAGE_COLD = (0.10, 150)
STAGE_FLOOR = (0.10, 200)
# TA_RAMP_ITER_SCALE < 1 shortens every solve (and MAX_SOLVES) for machinery
# tests only (run_optimization.py --quick-test) -- results are then NOT converged.
_scale = float(os.environ.get("TA_RAMP_ITER_SCALE", "1"))
if _scale != 1.0:
    STAGE_COLD = (STAGE_COLD[0], max(20, int(STAGE_COLD[1] * _scale)))
    STAGE_FLOOR = (STAGE_FLOOR[0], max(20, int(STAGE_FLOOR[1] * _scale)))
    MAX_SOLVES = min(MAX_SOLVES, 3)


def _solve(domain, ta, uniform_setup, I, ic_model, n_model, eps, alpha,
           n_it, warm, diag_path):
    params.ta_eps_reg = eps
    params.ta_picard_alpha = params.ta_picard_alpha_fine = alpha
    params.ta_n_picard = n_it
    if os.path.exists(diag_path):
        os.remove(diag_path)
    _, B_h, _, info = ta_solve.solve_ta_at_current(
        domain, ta, uniform_setup, I_amps=I, ic_model=ic_model,
        n_model=n_model, verbose=False, warm_start=warm, min_iters=n_it,
        diag_log_path=diag_path)
    with open(diag_path) as fh:
        scif = [float(r["scif_mT"]) for r in csv.DictReader(fh)]
    s = np.array(scif[-30:])
    return B_h, info, float(s.max() - s.min())


def _load_at(domain, ta, uniform_setup, I, ic_model, n_model, first,
             diag_path):
    """One end-of-ramp evaluation at current I. Returns (worst load,
    margin array, B_h, scif range)."""
    if first:
        _solve(domain, ta, uniform_setup, I, ic_model, n_model, 1.0,
               *STAGE_COLD, False, diag_path)
    B_h, info, rng = _solve(domain, ta, uniform_setup, I, ic_model, n_model,
                            EPS_FLOOR, *STAGE_FLOOR, True, diag_path)
    m, clip, _, _ = _local_margin(domain, ta, B_h, ic_model)
    return float((1.0 / m).max()), m, B_h, rng, clip


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
    trials = []            # (I, worst load, scif range)
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
            load, m_arr, B_h, rng, clip_ta = _load_at(
                domain, ta, uniform_setup, I_try, ic_model, n_model,
                first=(k == 0), diag_path=diag_path)
            trials.append((I_try, load, rng))
            if verbose:
                print(f"  [ta_ramp] I={I_try:7.2f}A  worst load={load:.4f}  "
                      f"SCIF range={rng:.2f}mT")
            if load <= LOAD_MAX:
                if best is None or I_try > best[0]:
                    best = (I_try, load)
                lo = (I_try, load) if lo is None or I_try > lo[0] else lo
                if LOAD_MAX - load < LOAD_TOL * LOAD_MAX:
                    break
            else:
                hi = (I_try, load) if hi is None or I_try < hi[0] else hi
            # next trial: secant through the bracket if we have one, else
            # a log-log step with the typical penetrated-coil slope 0.15
            if lo is not None and hi is not None:
                s = np.log(hi[1] / lo[1]) / np.log(hi[0] / lo[0])
                s = max(s, 0.05)
                I_new = lo[0] * np.exp(np.log(LOAD_MAX / lo[1]) / s)
                I_new = min(max(I_new, lo[0] * 1.01), hi[0] * 0.99)
                if hi[0] / lo[0] < 1.03:
                    break
            else:
                ref = trials[-1]
                I_new = ref[0] * (LOAD_MAX / ref[1]) ** (1.0 / 0.15)
                I_new = min(max(I_new, ref[0] / 4.0), ref[0] * 4.0)
            I_try = I_new

        if best is None:
            # nothing passed within the budget: report the lowest-current
            # trial as-is, flagged (not a fabricated placeholder)
            I_op, load_op = min(trials)[0], min(trials)[1]
            search_ok = False
        else:
            I_op, load_op = best
            search_ok = True
        # final state must correspond to I_op (the loop may have ended on a
        # different, failing trial)
        if trials[-1][0] != I_op:
            load_op, m_arr, B_h, rng, clip_ta = _load_at(
                domain, ta, uniform_setup, I_op, ic_model, n_model,
                first=False, diag_path=diag_path)
            trials.append((I_op, load_op, rng))
        scif_range = trials[-1][2]

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
                ta_worst_margin=1.0 / load_op,
                ta_constraint_margin=1.0 / load_op,
                ta_worst_load=load_op,
                bisection_exhausted=not search_ok,
                search_ok=search_ok,
                scif_range_mT=scif_range,
                converged=scif_range < SCIF_CONV_MT,
                trials=";".join(f"{i:.1f}:{l:.4f}" for i, l, _ in trials),
                clip_frac=clip_frac,
                n_ta_solves=len(trials), eval_s=time.time() - t0)
