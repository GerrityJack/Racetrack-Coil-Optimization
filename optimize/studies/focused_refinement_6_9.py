"""
focused_refinement_6_9.py — targeted follow-up on n_layers=9 and 6
====================================================================
Context (2026-07-23): the overnight n_layers sweep (overnight_refinement.py)
found 9 layers reaching tape=0.1524km in just 91 minutes (timed out, not
converged) -- within 4% of the 6-layer champion (0.1464km, from a full
2000+-eval extended-refinement round) despite 9 getting far less budget and
sitting between two much worse, FULLY CONVERGED neighbors (8: 0.1971km,
10: 0.2431km, both plateaued in their own extended rounds). That non-
monotonic jump is the signature this project has repeatedly seen before
(CLAUDE.md's run-1-vs-run-3 history, n_layers=5's round-2 regression) when
a design is sitting in a local basin rather than the true optimum for its
layer count -- so 9 is treated here as the most promising open lead, not
settled, and gets the lion's share of the budget. 6 gets a shorter polish
pass since it's already well converged (top-5% of 935 all-pass evals cluster
tightly around its current champion).

Two changes from the overnight run's methodology:
  1. coil_half_gap is pinned near its physical floor via
     CMAES_TIGHT_GAP_MARGIN_M_OVERRIDE (see opt_config.py) instead of
     searching a 10mm window -- every layer count's best designs sit within
     ~0.4mm of the floor across 60,930 cumulative evaluations, so this
     dimension is no longer worth CMA-ES's budget. Cuts the effective search
     from 10 to ~9 live dimensions.
  2. n_layers=9 gets a much longer cap (180 min) than the overnight run's
     90 min, since it was still actively improving (not plateaued) when
     killed -- see optimize/overnight_logs/n_layers_09.log's tail.

Same launch/logging conventions as overnight_refinement.py (direct python
binary for live output, time-boxed with clean SIGTERM/SIGKILL, proportional
a/b step sizes via CMAES_A_STD0_OVERRIDE/CMAES_B_STD0_OVERRIDE, incremental
CSV flushing means a killed job loses at most ~20 evals) -- see that file's
docstring for the full rationale on each of these.

Run:
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \\
        optimize/focused_refinement_6_9.py \\
        > optimize/focused_refinement_stdout.log 2>&1 &

Progress: optimize/focused_refinement_log.txt (summary, one line per
job start/finish); optimize/focused_refinement_logs/n_layers_NN.log
(full per-job CMA-ES output, live). Both jobs still feed the same
cumulative optimize/runs/cmaes_all_evaluations.csv / visualization/
cmaes_param_map.png as any other cmaes_search.py run -- pull results by
run_tag from there, not cmaes_results.csv (which only holds the LAST job's
best design).
"""
import os, sys, csv, json, time, signal, subprocess, traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import opt_config as cfg

PYTHON_BIN = "/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3"

TOTAL_BUDGET_HOURS = 5.0
PER_JOB_MINUTES_CAP = {9: 180, 6: 90}
MAX_EVALS = 6000   # generous cap; PER_JOB_MINUTES_CAP is what actually binds
STD0_FRAC = 0.05   # proportional step size: 5% of this job's own a/b value
GAP_MARGIN_M = 0.001   # pin gap within 1mm of its physical floor

# Warm starts: each job's current best known design (2026-07-23).
JOBS = [
    dict(n_layers=9, seed=9009,
         x0=dict(a=0.01243270988543402, b=0.018798156912111256,
                 coil_half_gap=0.019500166088372918,
                 n_turns=[155, 194, 242, 241, 84, 130, 59, 111, 58])),
    dict(n_layers=6, seed=6006,
         x0=dict(a=0.012908431166261252, b=0.020950054349354264,
                 coil_half_gap=0.013915040783036767,
                 n_turns=[187, 223, 256, 258, 245, 50])),
]

LOG_DIR = os.path.join(_ROOT, "optimize", "runs", "focused_refinement", "focused_refinement_logs")
SUMMARY_LOG = os.path.join(_ROOT, "optimize", "runs", "focused_refinement", "focused_refinement_log.txt")
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

    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"n_layers_{n_layers:02d}.log")
    cap_s = cap_minutes * 60

    _log(f"[n_layers={n_layers:2d}] starting: a={x0['a']*1e3:.2f}mm "
        f"b={x0['b']*1e3:.2f}mm gap={x0['coil_half_gap']*1e3:.2f}mm "
        f"a_std0={a_std0*1e3:.3f}mm b_std0={b_std0*1e3:.3f}mm "
        f"gap_margin={GAP_MARGIN_M*1e3:.1f}mm "
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
            f"cmaes_all_evaluations.csv for this run's run_tag)")
    return best


def main():
    t_start = time.time()
    budget_s = TOTAL_BUDGET_HOURS * 3600
    _log(f"Starting focused refinement: {len(JOBS)} jobs "
        f"({[j['n_layers'] for j in JOBS]}), {TOTAL_BUDGET_HOURS}h total "
        f"budget, {STD0_FRAC*100:.0f}% proportional step size, "
        f"gap pinned within {GAP_MARGIN_M*1e3:.1f}mm of floor")

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

    _log(f"Focused refinement finished, total "
        f"{(time.time()-t_start)/3600:.2f}h")


if __name__ == "__main__":
    main()
