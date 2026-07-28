"""
day_search.py — 2026-07-26 day-long search / validate / relax pipeline
==========================================================================
Three chained phases, run unattended over the course of a day:

  PHASE A — widen the search.  Every prior CMA-ES run (84+ runs, 30k+
    evaluations) converged toward small coil radius `a` because tape_km is
    the only geometric term in the fitness function and no uniformity
    signal has ever been in it (see cmaes_search.py's fitness() comment).
    The 2026-07-24 a-isolation sweep then showed small `a` is a genuine,
    controlled-experiment-confirmed uniformity failure mode, not just an
    unlucky sample. Phase A re-runs the discrete n_layers outer loop
    (6,8,10,12,14,16 -- 14/16 never tried before; 4 excluded, see below)
    with: (1) cmaes_search.py's new CMAES_MIN_A_M soft floor active
    (discourages, doesn't forbid, re-exploring the known-bad region),
    (2) CMAES_TIGHT_BOUNDS off (the tight zone-out was tuned purely from
    tape-optimal, pre-uniformity-aware runs), (3) a cold start with `a`
    seeded near the one confirmed-good design's scale instead of hugging
    the bend-radius floor.
    n_layers=4 is deliberately EXCLUDED even though it's an even (buildable)
    count: it was already investigated and REJECTED for a DIFFERENT reason
    independent of uniformity value -- gmsh mesh generation was found to be
    non-reproducible across separate process launches for that specific
    design, amplifying normal mesh noise into a >20x swing in its SCIF/
    uniformity number (see CLAUDE.md's "n_layers=4 investigation" section).
    Re-running its coarse screen wouldn't change that finding.

  PHASE B — validate.  The coarse screen's uniformity_pct is known
    unreliable by up to ~10x (anti-correlated with truth on this project's
    own 5-design dataset). The only trustworthy check is a full per-layer
    T-A solve (optimize/ta_validate.py, new this session, factors out
    solve/ta_solve.py's box-uniformity extension into a standalone,
    subprocess-safe call). Every layer count's Phase-A best (plus the
    known-good 6-layer champion as a fixed reference point) is T-A
    validated TWICE each (mesh generation is not perfectly reproducible
    across separate builds -- see the n_layers=4 episode) before any
    ranking decision is made. The winner is the lowest tape_km design
    whose BOTH T-A repeats pass the uniformity target.

  PHASE C — relax assumptions on the winner, per 2026-07-26 project
    direction (three things flagged explicitly, in order):
      1. Search-space bounds -- addressed by Phase A itself.
      2. Safety/stress margins -- SAFETY_FACTOR=1.818 and
         SIGMA_HOOP_MAX_PA=400 MPa have been fixed constants through every
         run to date. A fast fixed-geometry sensitivity table (no
         re-optimization, ~seconds) shows how I_op/B_target/hoop/clip_frac
         move if the winner's OWN geometry is re-evaluated at other values;
         separately, a handful of SHORT warm-started re-optimizations at
         relaxed values quantify actual tape-length savings on the table
         (not just what happens to the same geometry).
      3. Ic dataset extrapolation -- every call site in this project uses
         IcModel.critical_current(clip_B=True) (the default), which FLAT-
         CLAMPS Ic to its B=8T measured value above 8T rather than
         extrapolating a further decrease. Since Ic vs B is decreasing in
         the measured range, a flat clamp is an OPTIMISTIC assumption for
         any cell operating above 8T (peak_B_T), not a conservative one --
         the opposite of what "extrapolated" suggests at first read.
         ConservativeIcModel (below) instead continues Ic linearly at the
         B=8T slope, and the winner is re-evaluated under it end-to-end
         (quench, I_op, B_target, hoop, clip_frac) via the SAME
         optimize_geometry.evaluate() call, just swapping the ic_model
         argument -- an apples-to-apples comparison.

None of this ever writes params.py to disk -- every phase mutates the
`params` module in-process (a subprocess's own copy) or passes overrides
via env vars, exactly like cmaes_search.py's existing worker pattern. This
was a deliberate choice after the 2026-07-26 crash left params.py in a
stale, half-mutated state from an in-place-edit-style probe script.

Run (survives terminal loss; each phase writes incrementally):
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \\
        optimize/day_search.py > optimize/day_search_stdout.log 2>&1 &
    disown

Progress: optimize/day_search_log.txt (one line per milestone);
optimize/day_search_logs/*.log (full per-job CMA-ES output);
optimize/day_search_report.md (final written at the end of Phase C, but
Phases A/B's tables are appended to it incrementally so a partial run
still leaves a readable report).
"""
import os, sys, csv, json, re, time, signal, subprocess, traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "optimize"), os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import opt_config as cfg

PYTHON_BIN = "/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3"

LOG_PATH = os.path.join(_ROOT, "optimize", "day_search_log.txt")
LOGS_DIR = os.path.join(_ROOT, "optimize", "day_search_logs")
RESULTS_CSV = os.path.join(_ROOT, "optimize", "day_search_results.csv")
HISTORY_CSV = os.path.join(_ROOT, "optimize", "day_search_history.csv")
REPORT_PATH = os.path.join(_ROOT, "optimize", "day_search_report.md")

os.makedirs(LOGS_DIR, exist_ok=True)


def _log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def _report(md):
    with open(REPORT_PATH, "a") as f:
        f.write(md + "\n")


# ══════════════════════════════════════════════════════════════════════════
# PHASE A -- widened search
# ══════════════════════════════════════════════════════════════════════════

LAYER_COUNTS = [6, 8, 10, 12, 14, 16]
PER_JOB_MINUTES_CAP = {6: 90, 8: 100, 10: 110, 12: 120, 14: 135, 16: 150}
PHASE_A_MAX_EVALS = 2500
PHASE_A_BUDGET_HOURS = 13.0
A_START_M = 0.024          # seed `a` near the one confirmed-good design's
                            # scale (6L champion a=22.2mm), not the bend-
                            # radius floor every prior search hugged
REF_TOTAL_TURNS = 1350
MIN_STRAIGHT_M = 0.005
MIN_BORE_CLEAR_M = 0.0075
MIN_FACE_GAP_M = 0.003

# The known-good 6-layer champion (2026-07-24, T-A box uniformity 0.731%
# on its original validation, 0.828% on this session's re-check with
# ta_validate.py -- see the smoke test in day_search's launch notes).
# Always included in Phase B regardless of what Phase A finds, so there is
# always a validated fallback.
CHAMPION_6L = dict(a=0.022227029065529628, b=0.02726822715975084,
                   coil_half_gap=0.013500289306395013,
                   n_turns=[285, 285, 379, 379, 2, 2],
                   I_design=224.28825989070785)


def smart_x0(n_layers, w=0.004, t=75e-6):
    n_per_layer = max(1, round(REF_TOTAL_TURNS / n_layers))
    n_turns = [n_per_layer] * n_layers
    bend_floor = n_per_layer * t / 2 + MIN_BORE_CLEAR_M + 0.0003
    a = max(bend_floor, A_START_M)
    b = a + MIN_STRAIGHT_M + 0.0022
    z_top = n_layers * w / 2.0
    gap = z_top + MIN_FACE_GAP_M / 2.0 + 0.0015
    return dict(a=round(a, 6), b=round(b, 6),
               coil_half_gap=round(gap, 6), n_turns=n_turns)


def _run_cmaes_job(job_name, x0, seed, max_evals, cap_minutes, extra_env=None):
    """Launch cmaes_search.py as a subprocess with a warm start + per-job
    output-path overrides (avoids the shared-path race documented in
    CLAUDE.md), wait up to cap_minutes, SIGTERM then SIGKILL on timeout.
    Returns (best_row_dict_or_None, timed_out_bool, elapsed_seconds)."""
    n_layers = len(x0["n_turns"])
    n_pairs = n_layers // 2
    mean_pair = sum(x0["n_turns"][0::2]) / n_pairs
    a_std0 = round(x0["a"] * 0.10, 6)
    b_std0 = round(x0["b"] * 0.10, 6)
    n_std0 = round(mean_pair * 0.20, 3)

    results_csv = os.path.join(_ROOT, "optimize", f"day_search_{job_name}_results.csv")
    history_csv = os.path.join(_ROOT, "optimize", f"day_search_{job_name}_history.csv")
    if os.path.exists(results_csv):
        os.remove(results_csv)

    env = os.environ.copy()
    env["CMAES_SWEEP_OVERRIDE_JSON"] = json.dumps(
        dict(x0=x0, seed=seed, max_evals=max_evals))
    env["CMAES_A_STD0_OVERRIDE"] = str(a_std0)
    env["CMAES_B_STD0_OVERRIDE"] = str(b_std0)
    env["CMAES_N_STD0_OVERRIDE"] = str(n_std0)
    env["CMAES_OUT_CSV_OVERRIDE"] = results_csv
    env["CMAES_OUT_LOG_OVERRIDE"] = history_csv
    if extra_env:
        env.update(extra_env)

    log_path = os.path.join(LOGS_DIR, f"{job_name}.log")
    cap_s = cap_minutes * 60
    t0 = time.monotonic()
    timed_out = False
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            [PYTHON_BIN, "-u", "optimize/cmaes_search.py"],
            cwd=_ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
        try:
            proc.wait(timeout=cap_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    dt = time.monotonic() - t0

    best = None
    if os.path.exists(results_csv):
        with open(results_csv) as f:
            rows = list(csv.DictReader(f))
        best = rows[0] if rows else None
    return best, timed_out, dt


def run_phase_a():
    _log(f"PHASE A: widened search over n_layers={LAYER_COUNTS}, "
        f"CMAES_MIN_A_M={cfg.CMAES_MIN_A_M*1e3:.0f}mm floor, tight-bounds OFF, "
        f"a seeded >= {A_START_M*1e3:.0f}mm")
    _report(f"## Phase A -- widened search ({time.strftime('%Y-%m-%d %H:%M:%S')})\n\n"
            f"n_layers tried: {LAYER_COUNTS} (a-floor={cfg.CMAES_MIN_A_M*1e3:.0f}mm, "
            f"tight-bounds off, a seeded {A_START_M*1e3:.0f}mm)\n\n"
            f"| n_layers | tape_km | B_target_T | unif%(proxy) | hoop_MPa | a_mm | "
            f"b_mm | gap_mm | n_turns | status |\n"
            f"|---|---|---|---|---|---|---|---|---|---|")

    t_start = time.monotonic()
    budget_s = PHASE_A_BUDGET_HOURS * 3600
    results = {}

    for n_layers in LAYER_COUNTS:
        elapsed = time.monotonic() - t_start
        remaining_s = budget_s - elapsed
        if remaining_s <= 300:
            _log(f"Phase A budget ({PHASE_A_BUDGET_HOURS}h) nearly exhausted -- "
                f"skipping remaining layer counts: "
                f"{LAYER_COUNTS[LAYER_COUNTS.index(n_layers):]}")
            break
        cap_minutes = min(PER_JOB_MINUTES_CAP.get(n_layers, 120),
                          max(5, int(remaining_s / 60)))
        x0 = smart_x0(n_layers)
        seed = n_layers * 1000 + 726
        job_name = f"phaseA_n{n_layers:02d}"
        _log(f"[n_layers={n_layers:2d}] starting: a={x0['a']*1e3:.2f}mm "
            f"b={x0['b']*1e3:.2f}mm gap={x0['coil_half_gap']*1e3:.2f}mm "
            f"n_turns={x0['n_turns']} seed={seed} cap={cap_minutes}min "
            f"max_evals={PHASE_A_MAX_EVALS}")
        try:
            best, timed_out, dt = _run_cmaes_job(
                job_name, x0, seed, PHASE_A_MAX_EVALS, cap_minutes,
                extra_env={"CMAES_TIGHT_BOUNDS_OVERRIDE": "false"})
        except Exception:
            _log(f"[n_layers={n_layers:2d}] FAILED:\n{traceback.format_exc()}")
            continue

        tag = "TIMED OUT" if timed_out else "finished"
        if best:
            results[n_layers] = best
            _log(f"[n_layers={n_layers:2d}] {tag} in {dt/60:.1f} min -- "
                f"best tape={float(best['tape_km']):.4f}km "
                f"B={float(best['B_target_T']):.2f}T "
                f"unif={float(best['uniformity_pct']):.3f}% "
                f"hoop={float(best['hoop_MPa']):.0f}MPa "
                f"a={float(best['a_mm']):.2f}mm b={float(best['b_mm']):.2f}mm "
                f"gap={float(best['gap_mm']):.2f}mm n_turns={best['n_turns']}")
            _report(f"| {n_layers} | {float(best['tape_km']):.4f} | "
                    f"{float(best['B_target_T']):.2f} | "
                    f"{float(best['uniformity_pct']):.3f} | "
                    f"{float(best['hoop_MPa']):.0f} | "
                    f"{float(best['a_mm']):.2f} | {float(best['b_mm']):.2f} | "
                    f"{float(best['gap_mm']):.2f} | `{best['n_turns']}` | {tag} |")
        else:
            _log(f"[n_layers={n_layers:2d}] {tag} in {dt/60:.1f} min -- "
                f"NO all-pass design found")
            _report(f"| {n_layers} | -- | -- | -- | -- | -- | -- | -- | -- | "
                    f"no all-pass design ({tag}) |")

    _log(f"PHASE A finished, {len(results)}/{len(LAYER_COUNTS)} layer counts "
        f"produced an all-pass design, {(time.monotonic()-t_start)/3600:.2f}h elapsed")
    return results


# ══════════════════════════════════════════════════════════════════════════
# PHASE B -- T-A validation
# ══════════════════════════════════════════════════════════════════════════

TA_VALIDATE_REPEATS = 2
TA_VALIDATE_TIMEOUT_S = 1500
BOX_UNIF_TARGET_PCT = 1.0

_RESULT_RE = re.compile(
    r"TA_VALIDATE_RESULT label=(?P<label>'.*?') repeat=(?P<repeat>\d+) "
    r"box_ptp_pct=(?P<box>[-\d.]+) onaxis_scif_pct=(?P<onaxis>[-\d.]+) "
    r"Bz_bore_uniform=(?P<bzu>[-\d.]+) Bz_bore_TA=(?P<bzta>[-\d.]+) "
    r"converged=(?P<conv>True|False) n_iters=(?P<niters>\d+) "
    r"solve_s=(?P<solves>[-\d.]+)")


def ta_validate(design, label, repeats=TA_VALIDATE_REPEATS):
    env = os.environ.copy()
    env["TA_VALIDATE_JSON"] = json.dumps(dict(label=label, **design))
    env["TA_VALIDATE_REPEATS"] = str(repeats)
    log_path = os.path.join(LOGS_DIR,
                            f"ta_validate_{re.sub(r'[^A-Za-z0-9]+', '_', label)}.log")
    rows = []
    try:
        out = subprocess.run(
            [PYTHON_BIN, "-u", "optimize/ta_validate.py"],
            cwd=_ROOT, env=env, capture_output=True, text=True,
            timeout=TA_VALIDATE_TIMEOUT_S)
        with open(log_path, "w") as f:
            f.write(out.stdout)
            f.write(out.stderr)
        for m in _RESULT_RE.finditer(out.stdout):
            rows.append(dict(
                box_ptp_pct=float(m.group("box")),
                onaxis_scif_pct=float(m.group("onaxis")),
                Bz_bore_uniform=float(m.group("bzu")),
                Bz_bore_TA=float(m.group("bzta")),
                converged=m.group("conv") == "True",
                n_iters=int(m.group("niters")),
                solve_s=float(m.group("solves"))))
    except subprocess.TimeoutExpired:
        _log(f"[ta_validate:{label}] TIMED OUT after {TA_VALIDATE_TIMEOUT_S}s")
    except Exception:
        _log(f"[ta_validate:{label}] FAILED:\n{traceback.format_exc()}")
    return rows


def run_phase_b(phase_a_results):
    _log("PHASE B: T-A-validating shortlist (2 repeats each) -- "
        "the coarse screen's own uniformity_pct is not trustworthy, see "
        "CLAUDE.md")
    _report(f"\n## Phase B -- T-A box-uniformity validation "
            f"({time.strftime('%Y-%m-%d %H:%M:%S')})\n\n"
            f"Target: box peak-to-peak <= {BOX_UNIF_TARGET_PCT}%, BOTH repeats.\n\n"
            f"| label | tape_km | repeat0 box% | repeat1 box% | verdict |\n"
            f"|---|---|---|---|---|")

    candidates = [("6L_champion (reference)", CHAMPION_6L, 0.22586182219270562)]
    for n_layers, best in phase_a_results.items():
        design = dict(a=float(best["a_mm"]) * 1e-3, b=float(best["b_mm"]) * 1e-3,
                      coil_half_gap=float(best["gap_mm"]) * 1e-3,
                      n_turns=json.loads(best["n_turns"].replace("'", '"')
                                        if "'" in best["n_turns"] else best["n_turns"]),
                      I_design=float(best["I_op_A"]))
        candidates.append((f"phaseA_{n_layers}L", design, float(best["tape_km"])))

    scored = []
    for label, design, tape_km in candidates:
        rows = ta_validate(design, label)
        if len(rows) < 2:
            _log(f"[{label}] validation incomplete ({len(rows)}/2 repeats) -- skipped")
            _report(f"| {label} | {tape_km:.4f} | -- | -- | INCOMPLETE |")
            continue
        box0, box1 = rows[0]["box_ptp_pct"], rows[1]["box_ptp_pct"]
        passes = box0 <= BOX_UNIF_TARGET_PCT and box1 <= BOX_UNIF_TARGET_PCT
        verdict = "PASS" if passes else "FAIL"
        _log(f"[{label}] tape={tape_km:.4f}km box_ptp%: {box0:.3f}, {box1:.3f} "
            f"-> {verdict}")
        _report(f"| {label} | {tape_km:.4f} | {box0:.3f} | {box1:.3f} | {verdict} |")
        scored.append(dict(label=label, design=design, tape_km=tape_km,
                           box0=box0, box1=box1, passes=passes))

    passing = [s for s in scored if s["passes"]]
    if not passing:
        _log("PHASE B: no candidate passed both T-A repeats -- falling back "
            "to the 6L champion")
        winner = next(s for s in scored if s["label"].startswith("6L_champion"))
    else:
        winner = min(passing, key=lambda s: s["tape_km"])
    _log(f"PHASE B WINNER: {winner['label']} tape={winner['tape_km']:.4f}km "
        f"box_ptp%=[{winner['box0']:.3f},{winner['box1']:.3f}]")
    _report(f"\n**Phase B winner: {winner['label']}, tape={winner['tape_km']:.4f}km, "
            f"box p2p=[{winner['box0']:.3f}%, {winner['box1']:.3f}%]**\n")
    return winner


# ══════════════════════════════════════════════════════════════════════════
# PHASE C -- relax assumptions
# ══════════════════════════════════════════════════════════════════════════

SF_SWEEP = [1.3, 1.5, 1.818, 2.2]        # 1.818 = current baseline
HOOP_SWEEP_MPA = [350, 400, 450, 500]     # 400 = current baseline
PHASE_C_MAX_EVALS = 1200
PHASE_C_CAP_MINUTES = 60


class ConservativeIcModel:
    """Wraps an IcModel, replacing the flat clamp above B_max with a linear
    continuation using the local slope at B_max (per angle) instead of
    assuming Ic stops changing exactly at the edge of the measured data.
    Every call site in this project uses clip_B=True (flat clamp) by
    default -- since Ic is DECREASING with B in the measured range, that
    clamp is an OPTIMISTIC assumption for cells above 8T, not conservative.
    This is the 'what if that optimism is wrong' sensitivity check."""

    def __init__(self, base_ic_model, dB=0.1):
        self.base = base_ic_model
        self.B_max = base_ic_model.B_max
        self.B_min = base_ic_model.B_min
        self.Ic_min = base_ic_model.Ic_min
        self.Ic_max = base_ic_model.Ic_max
        self.dB = dB

    def critical_current(self, B_tesla, theta_deg, clip_B=True):
        import numpy as np
        B = np.atleast_1d(np.asarray(B_tesla, dtype=np.float64))
        theta = np.atleast_1d(np.asarray(theta_deg, dtype=np.float64))
        over = B > self.B_max
        Ic_at_max, _ = self.base.critical_current(
            np.full_like(B, self.B_max), theta, clip_B=False)
        Ic_at_edge, _ = self.base.critical_current(
            np.full_like(B, self.B_max - self.dB), theta, clip_B=False)
        slope = (Ic_at_max - Ic_at_edge) / self.dB
        Ic_extrap = Ic_at_max + slope * (B - self.B_max)
        Ic_normal, _ = self.base.critical_current(
            np.clip(B, self.B_min, self.B_max), theta, clip_B=False)
        Ic = np.where(over, Ic_extrap, Ic_normal)
        Ic = np.clip(Ic, 0.02 * self.Ic_max, self.Ic_max)
        frac_clipped = float(np.mean(B < self.B_min))
        return Ic, frac_clipped


def run_ic_sensitivity(winner_design):
    _log("PHASE C.1: Ic-extrapolation-beyond-8T sensitivity check on the winner")
    _report(f"\n## Phase C -- relax assumptions "
            f"({time.strftime('%Y-%m-%d %H:%M:%S')})\n\n"
            f"### C.1 -- Ic dataset extrapolation beyond 8T\n\n"
            f"Every call site in this project uses `clip_B=True` (flat clamp "
            f"at the B=8T measured value). Since Ic decreases with B in the "
            f"measured range, that clamp is OPTIMISTIC above 8T, not "
            f"conservative. Comparing against a linear continuation at the "
            f"B=8T slope:\n\n"
            f"| model | I_quench_A | I_op_A | B_target_T | hoop_MPa | clip_frac |\n"
            f"|---|---|---|---|---|---|")
    try:
        from mpi4py import MPI
        import optimize_geometry as og
        from ic_model import IcModel
        comm = MPI.COMM_WORLD
        base_ic = IcModel()
        cons_ic = ConservativeIcModel(base_ic)

        cand = dict(a=winner_design["a"], b=winner_design["b"],
                   n_turns=winner_design["n_turns"],
                   coil_half_gap=winner_design["coil_half_gap"])
        r_base = og.evaluate(cand, base_ic, comm)
        r_cons = og.evaluate(cand, cons_ic, comm)

        _log(f"  baseline (flat clamp):  I_q={r_base['I_quench_A']:.0f}A "
            f"I_op={r_base['I_op_A']:.0f}A B={r_base['B_target_T']:.2f}T "
            f"hoop={r_base['hoop_MPa']:.0f}MPa clip={r_base['clip_frac']:.3f}")
        _log(f"  conservative (linear):  I_q={r_cons['I_quench_A']:.0f}A "
            f"I_op={r_cons['I_op_A']:.0f}A B={r_cons['B_target_T']:.2f}T "
            f"hoop={r_cons['hoop_MPa']:.0f}MPa clip={r_cons['clip_frac']:.3f}")
        _report(f"| flat clamp (current default) | {r_base['I_quench_A']:.0f} | "
                f"{r_base['I_op_A']:.0f} | {r_base['B_target_T']:.2f} | "
                f"{r_base['hoop_MPa']:.0f} | {r_base['clip_frac']:.3f} |")
        _report(f"| linear continuation (conservative) | "
                f"{r_cons['I_quench_A']:.0f} | {r_cons['I_op_A']:.0f} | "
                f"{r_cons['B_target_T']:.2f} | {r_cons['hoop_MPa']:.0f} | "
                f"{r_cons['clip_frac']:.3f} |")
        pct_drop = (100.0 * (r_base["B_target_T"] - r_cons["B_target_T"])
                   / r_base["B_target_T"])
        _report(f"\nB_target drops {pct_drop:.1f}% under the conservative "
                f"extrapolation ({'still >= 10T' if r_cons['B_target_T'] >= 10.0 else 'FALLS BELOW the 10T floor'}).\n")
        return dict(base=r_base, conservative=r_cons)
    except Exception:
        _log(f"PHASE C.1 FAILED:\n{traceback.format_exc()}")
        return None


def run_margin_sensitivity_fixed_geometry(winner_design):
    """Fast, fixed-geometry (no re-optimization) sweep: how do I_op/
    B_target/hoop/clip move if the WINNER's OWN geometry is re-evaluated
    at other SAFETY_FACTOR / SIGMA_HOOP_MAX_PA values? Seconds, not
    minutes -- complements the slower tape-savings re-optimization below."""
    _log("PHASE C.2: fixed-geometry safety-factor / hoop-cap sensitivity")
    _report(f"\n### C.2 -- fixed-geometry margin sensitivity (winner's "
            f"geometry held fixed, only I_op/hoop-cap logic varies)\n\n"
            f"| SAFETY_FACTOR | hoop_cap_MPa | I_op_A | B_target_T | "
            f"hoop_MPa | clip_frac | binding |\n|---|---|---|---|---|---|---|")
    try:
        from mpi4py import MPI
        import optimize_geometry as og
        from ic_model import IcModel
        comm = MPI.COMM_WORLD
        ic_model = IcModel()
        cand = dict(a=winner_design["a"], b=winner_design["b"],
                   n_turns=winner_design["n_turns"],
                   coil_half_gap=winner_design["coil_half_gap"])

        rows = []
        for sf in SF_SWEEP:
            cfg.SAFETY_FACTOR = sf
            r = og.evaluate(cand, ic_model, comm)
            rows.append((sf, cfg.SIGMA_HOOP_MAX_PA / 1e6, r))
            _report(f"| {sf} | {cfg.SIGMA_HOOP_MAX_PA/1e6:.0f} | "
                    f"{r['I_op_A']:.0f} | {r['B_target_T']:.2f} | "
                    f"{r['hoop_MPa']:.0f} | {r['clip_frac']:.3f} | "
                    f"{r['binding']} |")
        cfg.SAFETY_FACTOR = 1.818   # restore baseline before the hoop sweep

        for hoop_mpa in HOOP_SWEEP_MPA:
            cfg.SIGMA_HOOP_MAX_PA = hoop_mpa * 1e6
            r = og.evaluate(cand, ic_model, comm)
            rows.append((cfg.SAFETY_FACTOR, hoop_mpa, r))
            _report(f"| {cfg.SAFETY_FACTOR} | {hoop_mpa} | "
                    f"{r['I_op_A']:.0f} | {r['B_target_T']:.2f} | "
                    f"{r['hoop_MPa']:.0f} | {r['clip_frac']:.3f} | "
                    f"{r['binding']} |")
        cfg.SIGMA_HOOP_MAX_PA = 400e6   # restore baseline
        _log("PHASE C.2 done")
        return rows
    except Exception:
        _log(f"PHASE C.2 FAILED:\n{traceback.format_exc()}")
        return None


def run_margin_reopt(winner_design, winner_tape_km):
    """Short warm-started re-optimizations at relaxed SF/hoop values --
    quantifies actual TAPE-LENGTH savings (not just what happens to the
    same geometry), which is what 'relax the margin' really means for a
    design that's still being optimized, not yet built."""
    _log("PHASE C.3: warm-started re-optimization at relaxed margins "
        "(tape-savings-vs-risk)")
    _report(f"\n### C.3 -- tape-length savings from relaxed margins "
            f"(short warm-started re-optimization, {PHASE_C_MAX_EVALS} evals "
            f"each, {PHASE_C_CAP_MINUTES}min cap)\n\n"
            f"| variant | SAFETY_FACTOR | hoop_cap_MPa | tape_km | "
            f"delta_vs_baseline | B_target_T | hoop_MPa |\n|---|---|---|---|---|---|---|")

    n_layers = len(winner_design["n_turns"])
    x0 = dict(a=winner_design["a"], b=winner_design["b"],
             coil_half_gap=winner_design["coil_half_gap"],
             n_turns=winner_design["n_turns"])
    n_pairs = n_layers // 2
    mean_pair = sum(winner_design["n_turns"][0::2]) / n_pairs

    variants = [
        ("SF=1.3 (relaxed)", dict(SAFETY_FACTOR_OVERRIDE="1.3")),
        ("SF=1.5 (relaxed)", dict(SAFETY_FACTOR_OVERRIDE="1.5")),
        ("hoop=450MPa (relaxed)", dict(SIGMA_HOOP_MAX_PA_OVERRIDE=str(450e6))),
        ("hoop=500MPa (relaxed)", dict(SIGMA_HOOP_MAX_PA_OVERRIDE=str(500e6))),
    ]

    results = []
    for i, (name, extra_env) in enumerate(variants):
        job_name = f"phaseC_variant{i}"
        seed = 30000 + i
        try:
            best, timed_out, dt = _run_cmaes_job(
                job_name, x0, seed, PHASE_C_MAX_EVALS, PHASE_C_CAP_MINUTES,
                extra_env=extra_env)
        except Exception:
            _log(f"[{name}] FAILED:\n{traceback.format_exc()}")
            continue
        sf = extra_env.get("SAFETY_FACTOR_OVERRIDE", cfg.SAFETY_FACTOR)
        hoop = float(extra_env.get("SIGMA_HOOP_MAX_PA_OVERRIDE", cfg.SIGMA_HOOP_MAX_PA)) / 1e6
        if best:
            tape = float(best["tape_km"])
            delta = 100.0 * (tape - winner_tape_km) / winner_tape_km
            _log(f"[{name}] {'TIMED OUT' if timed_out else 'finished'} "
                f"in {dt/60:.1f}min -- tape={tape:.4f}km ({delta:+.1f}% vs "
                f"winner) B={float(best['B_target_T']):.2f}T "
                f"hoop={float(best['hoop_MPa']):.0f}MPa")
            _report(f"| {name} | {sf} | {hoop:.0f} | {tape:.4f} | "
                    f"{delta:+.1f}% | {float(best['B_target_T']):.2f} | "
                    f"{float(best['hoop_MPa']):.0f} |")
            results.append(dict(name=name, tape_km=tape, best=best))
        else:
            _log(f"[{name}] no all-pass design found")
            _report(f"| {name} | {sf} | {hoop:.0f} | -- | -- | -- | -- |")
    return results


# ══════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.monotonic()
    _log("=" * 70)
    _log("day_search.py starting: Phase A (widen search) -> "
        "Phase B (T-A validate) -> Phase C (relax assumptions)")
    _log("=" * 70)
    with open(REPORT_PATH, "w") as f:
        f.write(f"# Day search report -- started "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    phase_a_results = run_phase_a()
    winner = run_phase_b(phase_a_results)
    run_ic_sensitivity(winner["design"])
    run_margin_sensitivity_fixed_geometry(winner["design"])
    run_margin_reopt(winner["design"], winner["tape_km"])

    _report(f"\n## Summary\n\n"
            f"Winner: **{winner['label']}**, tape={winner['tape_km']:.4f}km, "
            f"T-A box p2p uniformity=[{winner['box0']:.3f}%, {winner['box1']:.3f}%]\n\n"
            f"Design: a={winner['design']['a']*1e3:.3f}mm "
            f"b={winner['design']['b']*1e3:.3f}mm "
            f"gap={winner['design']['coil_half_gap']*1e3:.3f}mm "
            f"n_turns={winner['design']['n_turns']} "
            f"I_design={winner['design'].get('I_design', 'n/a')}\n\n"
            f"params.py was NOT modified by this pipeline -- review this "
            f"report, then promote the winner manually if it should become "
            f"the new champion.\n")

    dt_total = (time.monotonic() - t0) / 3600
    _log(f"day_search.py FINISHED in {dt_total:.2f}h total. "
        f"Report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
