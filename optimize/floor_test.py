"""
floor_test.py — diagnostic: is the 50-turn search floor an artificial wall?
==============================================================================
Context (2026-07-23): CMAES_TIGHT_N_BOUNDS = (50, 500) is an empirically
"zoned-out" search range (opt_config.py), NOT a physical constraint --
params.recompute_derived() only asserts n_turns[i] >= 1. But across 60,930
cumulative evaluations, roughly HALF the layers in every well-converged
design at n_layers >= 6 sit exactly at that 50 floor (e.g. n=12's champion:
4 of 12 layers in [50,61]; n=10's: 4 of 10 layers in [50,55]). That's the
same signature that previously flagged the old a/b box bounds as wrong
(pinned-at-boundary => probably not the true optimum) -- see
CLAUDE.md's "run 1 vs run 3" history for the precedent.

This script warm-starts each of the already-converged champions (6, 8, 9,
10, 12 layers) and reruns with the turn floor dropped from 50 to 1 (true
physical minimum), everything else held at its current tight/pinned
settings (gap still pinned near floor, a/b still unbounded with a small
proportional step). If many turn counts drift toward the new floor (1)
rather than settling back near 50, that confirms 50 was an artificial wall
and the true optimum wants fewer effective layers than the design currently
uses -- direct evidence for the "maybe higher layer counts are actually
approximating a smaller effective layer count" hypothesis. If they instead
settle back up near 50-100 despite having room to go lower, that's evidence
50 was already close to the true optimum for those layers.

Runs IN PARALLEL with optimize/focused_refinement_6_9.py's 9/6 run (started
2026-07-23 09:39, ~4.5h expected) via CMAES_N_WORKERS_OVERRIDE=2 (primary
job uses 6, this uses 2, exactly saturating this 8-core machine without
oversubscribing). Time caps are deliberately short -- this is a diagnostic
looking for a DIRECTION (do turns drift toward 1?), not a full polish.

Run:
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \\
        optimize/floor_test.py \\
        > optimize/floor_test_stdout.log 2>&1 &

Progress: optimize/floor_test_log.txt (summary); optimize/floor_test_logs/
n_layers_NN.log (full per-job output). Feeds the same cumulative
optimize/cmaes_all_evaluations.csv as any other cmaes_search.py run --
pull results by run_tag, not cmaes_results.csv (last job only).

How to read the result: for each job, look at the final n_turns in its
summary line -- specifically whether the layers that started near 50 (see
the JOBS list below) drifted DOWN toward 1 or stayed put/drifted up. A
drift down across multiple layer counts is the signal the hypothesis
predicts.
"""
import os, sys, csv, json, time, signal, subprocess, traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import opt_config as cfg

PYTHON_BIN = "/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3"

TOTAL_BUDGET_HOURS = 4.0
# Lower layer counts cost less per FEM eval (fewer mesh cells) -- shorter
# caps for them; this is a direction-finding diagnostic, not a full search.
PER_JOB_MINUTES_CAP = {6: 30, 8: 40, 9: 45, 10: 50, 12: 60}
MAX_EVALS = 4000
STD0_FRAC = 0.05          # same proportional a/b step as the primary run
GAP_MARGIN_M = 0.001      # keep gap pinned near floor, as established
N_WORKERS = 2             # leaves the primary run's 6 workers undisturbed
TURN_FLOOR_OVERRIDE = "1,500"   # was (50, 500); floor dropped to physical min

# Warm starts: each layer count's current best design (2026-07-23, from
# optimize/cmaes_all_evaluations.csv). Kept at full precision.
JOBS = [
    dict(n_layers=12, seed=12112,
         x0=dict(a=0.020798306934719015, b=0.028365753705744407,
                 coil_half_gap=0.025611979074903857,
                 n_turns=[58, 251, 226, 250, 241, 50, 129, 57, 115, 50, 72, 50])),
    dict(n_layers=10, seed=10110,
         x0=dict(a=0.020623039279382215, b=0.026934913612960074,
                 coil_half_gap=0.021504820390494547,
                 n_turns=[272, 190, 242, 272, 82, 58, 51, 147, 50, 50])),
    dict(n_layers=9, seed=9109,
         x0=dict(a=0.01243270988543402, b=0.018798156912111256,
                 coil_half_gap=0.019500166088372918,
                 n_turns=[155, 194, 242, 241, 84, 130, 59, 111, 58])),
    dict(n_layers=8, seed=8108,
         x0=dict(a=0.014820826388410298, b=0.022118537106350228,
                 coil_half_gap=0.01752223820011793,
                 n_turns=[166, 222, 115, 240, 241, 246, 51, 238])),
    dict(n_layers=6, seed=6106,
         x0=dict(a=0.012908431166261252, b=0.020950054349354264,
                 coil_half_gap=0.013915040783036767,
                 n_turns=[187, 223, 256, 258, 245, 50])),
]

LOG_DIR = os.path.join(_ROOT, "optimize", "floor_test_logs")
SUMMARY_LOG = os.path.join(_ROOT, "optimize", "floor_test_log.txt")
RESULTS_CSV = os.path.join(_ROOT, cfg.CMAES_OUT_CSV)


def _log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(SUMMARY_LOG, "a") as f:
        f.write(line + "\n")


def _run_job(job, cap_minutes):
    n_layers = job["n_layers"]
    x0 = job["x0"]
    a_std0 = round(x0["a"] * STD0_FRAC, 6)
    b_std0 = round(x0["b"] * STD0_FRAC, 6)

    env = os.environ.copy()
    env["CMAES_SWEEP_OVERRIDE_JSON"] = json.dumps(
        dict(x0=x0, seed=job["seed"], max_evals=MAX_EVALS))
    env["CMAES_A_STD0_OVERRIDE"] = str(a_std0)
    env["CMAES_B_STD0_OVERRIDE"] = str(b_std0)
    env["CMAES_TIGHT_GAP_MARGIN_M_OVERRIDE"] = str(GAP_MARGIN_M)
    env["CMAES_TIGHT_N_BOUNDS_OVERRIDE"] = TURN_FLOOR_OVERRIDE
    env["CMAES_N_WORKERS_OVERRIDE"] = str(N_WORKERS)

    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"n_layers_{n_layers:02d}.log")
    cap_s = cap_minutes * 60

    _log(f"[n_layers={n_layers:2d}] starting: turn bounds={TURN_FLOOR_OVERRIDE} "
        f"(was 50,500) a={x0['a']*1e3:.2f}mm b={x0['b']*1e3:.2f}mm "
        f"gap={x0['coil_half_gap']*1e3:.2f}mm n_turns={x0['n_turns']} "
        f"seed={job['seed']} workers={N_WORKERS} cap={cap_minutes}min")

    t0 = time.time()
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
    dt = time.time() - t0

    best = None
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV) as f:
            rows = list(csv.DictReader(f))
        best = rows[0] if rows else None

    tag = ("TIMED OUT (progress preserved via incremental flush)"
           if timed_out else "finished")
    if best:
        n_before = job["x0"]["n_turns"]
        import ast
        n_after = ast.literal_eval(best["n_turns"])
        drift = [f"{b}->{a}" for b, a in zip(n_before, n_after)]
        _log(f"[n_layers={n_layers:2d}] {tag} in {dt/60:.1f} min -- "
            f"best tape={float(best['tape_km']):.4f}km "
            f"(was {job.get('start_tape', '?')}) "
            f"unif={float(best['uniformity_pct']):.3f}% "
            f"hoop={float(best['hoop_MPa']):.0f}MPa "
            f"turns before->after: [{', '.join(drift)}]")
    else:
        _log(f"[n_layers={n_layers:2d}] {tag} in {dt/60:.1f} min -- "
            f"NO all-pass design found (check cmaes_all_evaluations.csv "
            f"for this run's run_tag)")
    return best


def main():
    t_start = time.time()
    budget_s = TOTAL_BUDGET_HOURS * 3600
    _log(f"Starting floor-test diagnostic: {len(JOBS)} jobs "
        f"({[j['n_layers'] for j in JOBS]}), {TOTAL_BUDGET_HOURS}h total "
        f"budget, turn floor {TURN_FLOOR_OVERRIDE} (was 50,500), "
        f"{N_WORKERS} workers/job (parallel to the primary 9/6 run)")

    for i, job in enumerate(JOBS):
        elapsed = time.time() - t_start
        remaining_s = budget_s - elapsed
        if remaining_s <= 180:
            skipped = [j["n_layers"] for j in JOBS[i:]]
            _log(f"Total time budget ({TOTAL_BUDGET_HOURS}h) nearly "
                f"exhausted -- skipping remaining jobs: {skipped}")
            break
        cap_minutes = min(PER_JOB_MINUTES_CAP.get(job["n_layers"], 40),
                          max(5, int(remaining_s / 60)))
        try:
            _run_job(job, cap_minutes)
        except Exception:
            _log(f"[n_layers={job['n_layers']:2d}] FAILED:\n"
                f"{traceback.format_exc()}")
            continue

    _log(f"Floor-test diagnostic finished, total "
        f"{(time.time()-t_start)/3600:.2f}h")


if __name__ == "__main__":
    main()
