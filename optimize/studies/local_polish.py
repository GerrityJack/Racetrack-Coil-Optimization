"""
local_polish.py -- tightly-scoped local refinement of the 6-layer champion
==========================================================================
2026-07-30. The perturbation study (optimize/studies/perturbation_study.py)
proved the champion is NOT a converged local optimum: the neighbour
n_turns=[295,295,369,369,2,2] -- 10 turns per pancake moved from the
inner pair to the outer pair -- beats it on tape AND B_target AND hoop
AND T-A box uniformity simultaneously. That means a small amount of
budget spent locally should pay off, without re-opening the global search.

Design of this run, and why each choice (all lessons already paid for --
see CLAUDE.md):

- **Warm start from the DOMINATING NEIGHBOUR**, not the champion, since it
  is strictly better on every metric.
- **Proportional step sizes** via CMAES_A_STD0_OVERRIDE /
  CMAES_B_STD0_OVERRIDE / CMAES_N_STD0_OVERRIDE (STD0_FRAC of the warm
  start's OWN value), NOT the fixed-mm CMAES_A_STD0/CMAES_B_STD0 or the
  bound-range-derived turn step. A fixed absolute step caused the
  n_layers=5 round-2 regression, and an oversized turn step caused the
  focused_refinement_6_9 n_layers=9 run to wander off and never return.
  This is a POLISH: steps must be small relative to the design's scale.
- **`a` keeps its existing soft floor** (cfg.CMAES_MIN_A_M = 22.2mm).
  The perturbation study showed uniformity degrades steeply for smaller
  `a` (0.83% -> 1.14% at -0.5mm) and improves outward, so the floor
  blocks exactly the bad direction while leaving the good one free.
- **`coil_half_gap` is left free but is expected to sit at its 3mm
  face-gap floor** -- smaller gap means more field, hence fewer turns and
  less tape, so the (uniformity-blind) fitness drives it down on its own;
  and the perturbation study confirmed the floor IS the uniformity
  optimum for this axis (steepest axis, ~0.7pp/mm, monotone worsening
  upward). Nothing to force.
- **Unique CMAES_OUT_CSV_OVERRIDE / CMAES_OUT_LOG_OVERRIDE** so this run
  cannot clobber the shared optimize/runs/cmaes_{results,history}.csv
  (the race-condition bug of 2026-07-23). Only one job runs at a time
  here, but the hygiene is free.

CRITICAL -- the fitness function carries NO uniformity signal (deliberate;
two successive guessed proxies were both wrong). So Phase 1's "best" is
best-by-TAPE only. Phase 2 therefore T-A-validates a BATCH of the run's
top distinct candidates, plus the champion and the dominating neighbour as
references, and the winner is the lowest-tape design that actually PASSES
real box uniformity. Never promote Phase 1's output directly.

Run:
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \
        optimize/studies/local_polish.py
(direct binary, not `conda run` -- see CLAUDE.md on output buffering)

Outputs: optimize/runs/local_polish/{polish_log.txt, polish_history.csv,
polish_results.csv, finalists.csv} and per-candidate T-A logs.
"""
import os, sys, json, re, csv, time, subprocess, traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

PYTHON_BIN = "/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3"
RUN_DIR = os.path.join(_ROOT, "optimize", "runs", "local_polish")
LOGS_DIR = os.path.join(RUN_DIR, "ta_logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_PATH = os.path.join(RUN_DIR, "polish_log.txt")
HISTORY_CSV = os.path.join(RUN_DIR, "polish_history.csv")
RESULTS_CSV = os.path.join(RUN_DIR, "polish_results.csv")
FINALISTS_CSV = os.path.join(RUN_DIR, "finalists.csv")
SEARCH_LOG = os.path.join(RUN_DIR, "cmaes_stdout.log")

# ── Phase 1 settings ────────────────────────────────────────────────────────
# The dominating neighbour found by the perturbation study (champion
# geometry, turn pairs shifted 285/379 -> 295/369).
X0 = dict(a=0.022227029065529628, b=0.02726822715975084,
          coil_half_gap=0.013500289306395013,
          n_turns=[295, 295, 369, 369, 2, 2])
STD0_FRAC = 0.05          # a/b step = 5% of their own warm-start value
N_STD0 = 15.0             # turn-pair step [turns]; the study showed 10-turn
                          # shifts are the relevant scale
MAX_EVALS = 1000
SEED = 73024
N_WORKERS = 6
SEARCH_TIMEOUT_S = 3 * 3600

# ── Phase 2 settings ────────────────────────────────────────────────────────
N_FINALISTS = 6           # top distinct candidates by tape from Phase 1
TA_REPEATS = 2
TA_TIMEOUT_S = 1800
UNIF_LIMIT_PCT = 1.0
B_FLOOR_T = 10.0
HOOP_MAX_MPA = 400.0
# minimum separation for two candidates to count as "distinct" finalists
DEDUPE_TAPE_KM = 0.0008


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ── Phase 1 ─────────────────────────────────────────────────────────────────

def run_search():
    env = os.environ.copy()
    env["CMAES_SWEEP_OVERRIDE_JSON"] = json.dumps(
        dict(x0=X0, seed=SEED, max_evals=MAX_EVALS))
    env["CMAES_A_STD0_OVERRIDE"] = str(X0["a"] * STD0_FRAC)
    env["CMAES_B_STD0_OVERRIDE"] = str(X0["b"] * STD0_FRAC)
    env["CMAES_N_STD0_OVERRIDE"] = str(N_STD0)
    env["CMAES_N_WORKERS_OVERRIDE"] = str(N_WORKERS)
    env["CMAES_OUT_CSV_OVERRIDE"] = RESULTS_CSV
    env["CMAES_OUT_LOG_OVERRIDE"] = HISTORY_CSV

    _log(f"PHASE 1: local CMA-ES polish, {MAX_EVALS} evals, seed {SEED}")
    _log(f"  x0 = a={X0['a']*1e3:.3f}mm b={X0['b']*1e3:.3f}mm "
         f"gap={X0['coil_half_gap']*1e3:.3f}mm n_turns={X0['n_turns']}")
    _log(f"  step: a={X0['a']*STD0_FRAC*1e3:.3f}mm "
         f"b={X0['b']*STD0_FRAC*1e3:.3f}mm turns={N_STD0:.0f} "
         f"(proportional -- NOT the bound-range defaults)")

    t0 = time.time()
    with open(SEARCH_LOG, "w") as f:
        p = subprocess.Popen([PYTHON_BIN, "-u", "optimize/cmaes_search.py"],
                             cwd=_ROOT, env=env, stdout=f,
                             stderr=subprocess.STDOUT)
        try:
            p.wait(timeout=SEARCH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _log("  search exceeded its cap -- terminating "
                 "(progress is preserved by cmaes_search's incremental flush)")
            p.terminate()
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
    _log(f"PHASE 1 done in {(time.time()-t0)/60:.1f} min "
         f"(rc={p.returncode}, full log -> {SEARCH_LOG})")


def _f(row, key, default=float("nan")):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def pick_finalists():
    """Top distinct candidates by tape among this run's constraint-satisfying
    evaluations. B_target/hoop/tape are the metrics that were never broken;
    uniformity_pct is deliberately IGNORED here (see module docstring)."""
    if not os.path.exists(HISTORY_CSV):
        _log(f"  !! {HISTORY_CSV} missing -- Phase 1 produced no history")
        return []
    rows = []
    with open(HISTORY_CSV) as f:
        for r in csv.DictReader(f):
            if str(r.get("feasible", "")).strip().lower() not in ("true", "1"):
                continue
            tape, B = _f(r, "tape_km"), _f(r, "B_target_T")
            hoop = _f(r, "hoop_MPa")
            if not (tape == tape and B == B and hoop == hoop):
                continue
            if B < B_FLOOR_T or hoop > HOOP_MAX_MPA:
                continue
            rows.append(r)
    rows.sort(key=lambda r: _f(r, "tape_km"))
    _log(f"  {len(rows)} evaluations satisfy B>={B_FLOOR_T}T and "
         f"hoop<={HOOP_MAX_MPA}MPa")

    picked = []
    for r in rows:
        t = _f(r, "tape_km")
        if any(abs(t - _f(p, "tape_km")) < DEDUPE_TAPE_KM for p in picked):
            continue
        picked.append(r)
        if len(picked) >= N_FINALISTS:
            break
    return picked


# ── Phase 2 ─────────────────────────────────────────────────────────────────

_RESULT_RE = re.compile(
    r"TA_VALIDATE_RESULT label=(?P<label>'.*?') repeat=(?P<repeat>\d+) "
    r"box_ptp_pct=(?P<box>[-\d.]+) onaxis_scif_pct=(?P<onaxis>[-\d.]+) "
    r"Bz_bore_uniform=(?P<bzu>[-\d.]+) Bz_bore_TA=(?P<bzta>[-\d.]+) "
    r"converged=(?P<conv>True|False) n_iters=(?P<niters>\d+) "
    r"solve_s=(?P<solves>[-\d.]+)")


def ta_validate(design, label):
    env = os.environ.copy()
    env["TA_VALIDATE_JSON"] = json.dumps(dict(label=label, **design))
    env["TA_VALIDATE_REPEATS"] = str(TA_REPEATS)
    log_path = os.path.join(LOGS_DIR,
                            re.sub(r"[^A-Za-z0-9+.-]+", "_", label) + ".log")
    boxes = []
    try:
        out = subprocess.run([PYTHON_BIN, "-u", "optimize/ta_validate.py"],
                             cwd=_ROOT, env=env, capture_output=True,
                             text=True, timeout=TA_TIMEOUT_S)
        with open(log_path, "w") as f:
            f.write(out.stdout + "\n--- stderr ---\n" + out.stderr)
        boxes = [float(m.group("box")) for m in _RESULT_RE.finditer(out.stdout)]
    except subprocess.TimeoutExpired:
        _log(f"  [{label}] T-A TIMED OUT")
    except Exception:
        _log(f"  [{label}] T-A FAILED:\n{traceback.format_exc()}")
    return boxes


def main():
    t_start = time.time()
    open(LOG_PATH, "w").close()
    run_search()

    finalists = pick_finalists()
    if not finalists:
        _log("no constraint-satisfying candidate found -- stopping")
        return

    cands = []
    for i, r in enumerate(finalists):
        cands.append(dict(
            label=f"polish{i+1}",
            a=_f(r, "a_mm") / 1e3, b=_f(r, "b_mm") / 1e3,
            coil_half_gap=_f(r, "gap_mm") / 1e3,
            n_turns=json.loads(r["n_turns"]),
            I_design=_f(r, "I_op_A"),
            tape_km=_f(r, "tape_km"), B_target_T=_f(r, "B_target_T"),
            hoop_MPa=_f(r, "hoop_MPa")))

    # references, so the comparison is apples-to-apples on the same meshes
    cands.append(dict(label="ref_champion", a=0.022227029065529628,
                      b=0.02726822715975084, coil_half_gap=0.013500289306395013,
                      n_turns=[285, 285, 379, 379, 2, 2], I_design=224.28825989070785,
                      tape_km=0.2259, B_target_T=10.005, hoop_MPa=114.1))
    cands.append(dict(label="ref_shift_in", a=0.022227029065529628,
                      b=0.02726822715975084, coil_half_gap=0.013500289306395013,
                      n_turns=[295, 295, 369, 369, 2, 2], I_design=223.9,
                      tape_km=0.2235, B_target_T=10.215, hoop_MPa=110.9))

    _log(f"PHASE 2: T-A box uniformity on {len(cands)} candidates "
         f"({TA_REPEATS} independent-mesh repeats each)")
    for c in cands:
        _log(f"  {c['label']:<14s} tape={c['tape_km']:.4f}km "
             f"B={c['B_target_T']:.3f}T hoop={c['hoop_MPa']:.0f}MPa "
             f"a={c['a']*1e3:.3f} b={c['b']*1e3:.3f} gap={c['coil_half_gap']*1e3:.3f} "
             f"n={c['n_turns']}")

    for c in cands:
        design = {k: c[k] for k in
                  ("a", "b", "coil_half_gap", "n_turns", "I_design")}
        boxes = ta_validate(design, c["label"])
        c["box_repeats"] = boxes
        c["box_mean"] = sum(boxes) / len(boxes) if boxes else float("nan")
        c["box_worst"] = max(boxes) if boxes else float("nan")
        c["passes"] = bool(boxes) and c["box_worst"] <= UNIF_LIMIT_PCT
        _log(f"  [{c['label']}] box p2p = "
             f"{', '.join(f'{b:.3f}' for b in boxes) or 'FAILED'} "
             f"-> {'PASS' if c['passes'] else 'FAIL'}")
        with open(FINALISTS_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, extrasaction="ignore", fieldnames=[
                "label", "tape_km", "B_target_T", "hoop_MPa", "box_mean",
                "box_worst", "passes", "a", "b", "coil_half_gap", "n_turns"])
            w.writeheader()
            for x in cands:
                if "box_mean" in x:
                    w.writerow(x)

    passing = [c for c in cands if c.get("passes")]
    passing.sort(key=lambda c: c["tape_km"])
    _log("")
    _log(f"{'label':<14}{'tape_km':>9}{'B_T':>8}{'hoop':>6}"
         f"{'box%(worst)':>13}  verdict")
    for c in sorted(cands, key=lambda c: c.get("box_worst", 9e9)):
        if "box_mean" not in c:
            continue
        _log(f"{c['label']:<14}{c['tape_km']:9.4f}{c['B_target_T']:8.3f}"
             f"{c['hoop_MPa']:6.0f}{c['box_worst']:13.3f}  "
             f"{'PASS' if c['passes'] else 'FAIL'}")
    if passing:
        w = passing[0]
        _log("")
        _log(f"WINNER (lowest tape that actually passes T-A box uniformity): "
             f"{w['label']} tape={w['tape_km']:.4f}km B={w['B_target_T']:.3f}T "
             f"hoop={w['hoop_MPa']:.0f}MPa box={w['box_worst']:.3f}%")
        _log(f"  a={w['a']*1e3:.4f}mm b={w['b']*1e3:.4f}mm "
             f"gap={w['coil_half_gap']*1e3:.4f}mm n_turns={w['n_turns']}")
    else:
        _log("NO candidate passed real box uniformity -- champion stands")
    _log(f"total {(time.time()-t_start)/60:.1f} min -> {FINALISTS_CSV}")


if __name__ == "__main__":
    main()
