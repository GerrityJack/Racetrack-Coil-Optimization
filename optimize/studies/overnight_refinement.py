"""
overnight_refinement.py — time-boxed extended refinement across layer counts
==============================================================================
As of 2026-07-22, layer counts 6, 7, and 8 have each been through a proper
"extended refinement" round (2000-2500 evals, tight bounds, warm-started
from the n_layers-sweep's cold-start result) and show a clean, converged
trend: 6 (0.1464km) < 7 (0.1884km) < 8 (0.1971km). Layer counts 3, 4, 9, 10,
12 only have the CHEAP cold-start sweep result (250-500 evals,
sweep_n_layers.py) -- an apples-to-oranges comparison against 6/7/8's much
more thorough search. n_layers=5 got a round 2 refinement attempt that
REGRESSED (0.373km -> 0.417km) because its warm-start step size
(CMAES_A_STD0/CMAES_B_STD0, fixed absolute metres) was oversized relative to
its own small a/b scale -- see opt_config.py's CMAES_X0 comment block for
the full ledger. This script:

  1. Gives 3, 4, 9, 10, 12 the same extended-refinement treatment 6/7/8 got,
     warm-started from their actual cold-start-sweep best (not a blind
     smart_x0 restart).
  2. Redoes n_layers=5 from ROUND 1's endpoint (not round 2's) with a
     PROPORTIONAL step size (CMAES_A_STD0_OVERRIDE/CMAES_B_STD0_OVERRIDE,
     added to opt_config.py specifically for this) to avoid repeating the
     round-2 regression.

Design principles carried over from the session's hard-won fixes (see
CLAUDE.md for the full incident writeups -- do not re-break these):
  - Launches cmaes_search.py via the fenicsx-env python BINARY DIRECTLY,
    not `conda run -n fenicsx-env python3 ...` -- conda run buffers ALL
    subprocess stdout until the wrapped process exits, confirmed
    independent of any in-script buffering fix, so `conda run` makes a
    multi-hour job's progress completely unobservable until it's done.
  - Each job is TIME-BOXED (PER_JOB_MINUTES_CAP), not just eval-boxed.
    cmaes_search.py already flushes cmaes_history.csv/
    cmaes_all_evaluations.csv every 20 evals (FLUSH_EVERY), so a timeout
    kill (SIGTERM, then SIGKILL if it doesn't exit) loses at most ~20
    evals of progress, never the whole run.
  - Each job's own CMAES_A_STD0/CMAES_B_STD0 are set PROPORTIONAL to that
    job's own warm-start a/b (STD0_FRAC), not a fixed mm number -- the
    exact bug that caused the n_layers=5 round-2 regression.

Run overnight (this is the ~10 hour job -- start it in the evening, check
optimize/overnight_refinement_log.txt / optimize/overnight_logs/*.log in the
morning):

    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \\
        optimize/overnight_refinement.py \\
        > optimize/overnight_stdout.log 2>&1 &

(background it with `&`, or launch via `run_in_background` if driven by
Claude Code -- either way, checking on it mid-run works because of the
direct-binary launch above; `tail -f optimize/overnight_refinement_log.txt`
or any of the per-job logs in optimize/overnight_logs/ shows live progress).

Progress: optimize/overnight_refinement_log.txt (one line per job
start/finish, same format as sweep_n_layers_log.txt). Per-job full logs:
optimize/overnight_logs/n_layers_NN.log. Every job still feeds the same
cumulative optimize/runs/cmaes_all_evaluations.csv / visualization/
cmaes_param_map.png as any other cmaes_search.py run.

After it finishes (or if you need to check mid-run / after an interrupted
run), pull each job's best result out of the cumulative master log by its
run_tag (printed in the per-job log's final lines) rather than
cmaes_results.csv, which only ever holds the LAST job's result.
"""
import os, sys, csv, json, time, signal, subprocess, traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import opt_config as cfg

PYTHON_BIN = "/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3"

TOTAL_BUDGET_HOURS = 10.0

# Hard wall-clock ceiling per job [minutes]. Higher layer counts cost more
# per FEM eval (mesh has more cells) -- see sweep_n_layers.py's
# per_layer_evals() for the same reasoning at the cold-start budget scale.
PER_JOB_MINUTES_CAP = {3: 60, 4: 60, 5: 75, 9: 90, 10: 100, 12: 120}

MAX_EVALS = 4000   # generous cap; PER_JOB_MINUTES_CAP is what actually binds
STD0_FRAC = 0.05   # proportional step size: 5% of this job's own a/b value

# Warm starts: each job's best known design as of 2026-07-22.
# 3/4/9/10/12: cold-start sweep result (optimize/sweep_n_layers_log.txt).
# 5: ROUND 1's endpoint (0.373km) -- NOT round 2 (0.417km, a regression;
#    see opt_config.py's CMAES_X0 comment for why).
#
# NOTE: CMAES_TIGHT_BOUNDS=True (opt_config.py default) caps every
# per-layer turn count to CMAES_TIGHT_N_BOUNDS=(50,500) -- pycma's
# `geno()` raises ValueError if x0 falls outside the configured bounds, so
# any cold-start value above 500 is clipped here to exactly 500 (verified
# by smoke test: n_layers=3's 635 and n_layers=10's 541 both tripped this
# before being clipped -- this bit the n_layers=6 extended-refinement
# launch too, see CLAUDE.md). n_layers=12's gap (0.0255m) sat AT its tight
# floor to 4-decimal precision -- nudged +0.2mm so the starting point
# itself isn't immediately borderline-infeasible.
JOBS = [
    dict(n_layers=3, seed=7003,
         x0=dict(a=0.0287, b=0.0374, coil_half_gap=0.0080,
                 n_turns=[420, 491, 500])),          # 635 -> 500 (tight cap)
    dict(n_layers=4, seed=7004,
         x0=dict(a=0.0256, b=0.0360, coil_half_gap=0.0096,
                 n_turns=[352, 312, 458, 492])),
    dict(n_layers=5, seed=7005,
         x0=dict(a=0.016225348571759877, b=0.05499433483759851,
                 coil_half_gap=0.01154555376755883,
                 n_turns=[294, 349, 351, 73, 351])),
    dict(n_layers=9, seed=7009,
         x0=dict(a=0.0306, b=0.0391, coil_half_gap=0.0197,
                 n_turns=[245, 240, 244, 258, 223, 412, 65, 172, 494])),
    dict(n_layers=10, seed=7010,
         x0=dict(a=0.0307, b=0.0420, coil_half_gap=0.0237,
                 n_turns=[376, 406, 500, 108, 286, 69, 384, 320, 233, 98])),  # 541 -> 500
    dict(n_layers=12, seed=7012,
         x0=dict(a=0.0250, b=0.0381, coil_half_gap=0.0257,      # 0.0255 -> 0.0257 (was at floor)
                 n_turns=[287, 112, 178, 229, 360, 99, 157, 210, 318,
                          111, 86, 198])),
]

LOG_DIR = os.path.join(_ROOT, "optimize", "runs", "overnight", "overnight_logs")
SUMMARY_LOG = os.path.join(_ROOT, "optimize", "runs", "overnight", "overnight_refinement_log.txt")
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

    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"n_layers_{n_layers:02d}.log")
    cap_s = cap_minutes * 60

    _log(f"[n_layers={n_layers:2d}] starting: a={x0['a']*1e3:.2f}mm "
        f"b={x0['b']*1e3:.2f}mm gap={x0['coil_half_gap']*1e3:.2f}mm "
        f"a_std0={a_std0*1e3:.3f}mm b_std0={b_std0*1e3:.3f}mm "
        f"n_turns={x0['n_turns']} seed={job['seed']} cap={cap_minutes}min")

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

    tag = ("TIMED OUT (killed cleanly -- progress preserved via the "
           "incremental flush)" if timed_out else "finished")
    if best:
        _log(f"[n_layers={n_layers:2d}] {tag} in {dt/60:.1f} min -- "
            f"best tape={float(best['tape_km']):.4f}km "
            f"B={float(best['B_target_T']):.2f}T "
            f"unif={float(best['uniformity_pct']):.3f}% "
            f"hoop={float(best['hoop_MPa']):.0f}MPa "
            f"a={float(best['a_mm']):.2f}mm b={float(best['b_mm']):.2f}mm "
            f"gap={float(best['gap_mm']):.2f}mm n_turns={best['n_turns']}")
    else:
        _log(f"[n_layers={n_layers:2d}] {tag} in {dt/60:.1f} min -- "
            f"NO all-pass design in cmaes_results.csv (check "
            f"cmaes_all_evaluations.csv for this run's run_tag -- an "
            f"early kill may still have found feasible-but-violating "
            f"points worth resuming from)")
    return best


def main():
    t_start = time.time()
    budget_s = TOTAL_BUDGET_HOURS * 3600
    _log(f"Starting overnight extended refinement: {len(JOBS)} jobs "
        f"({[j['n_layers'] for j in JOBS]}), {TOTAL_BUDGET_HOURS}h total "
        f"budget, {STD0_FRAC*100:.0f}% proportional step size")

    for i, job in enumerate(JOBS):
        elapsed = time.time() - t_start
        remaining_s = budget_s - elapsed
        if remaining_s <= 300:
            skipped = [j["n_layers"] for j in JOBS[i:]]
            _log(f"Total time budget ({TOTAL_BUDGET_HOURS}h) nearly "
                f"exhausted -- skipping remaining jobs: {skipped}")
            break
        cap_minutes = min(PER_JOB_MINUTES_CAP.get(job["n_layers"], 90),
                          max(5, int(remaining_s / 60)))
        try:
            _run_job(job, cap_minutes)
        except Exception:
            _log(f"[n_layers={job['n_layers']:2d}] FAILED:\n"
                f"{traceback.format_exc()}")
            continue

    _log(f"Overnight refinement finished, total "
        f"{(time.time()-t_start)/3600:.2f}h")


if __name__ == "__main__":
    main()
