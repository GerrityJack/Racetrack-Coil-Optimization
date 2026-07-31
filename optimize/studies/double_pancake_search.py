"""
double_pancake_search.py — re-optimize under the three new physical
constraints (2026-07-23)
==============================================================================
Practical manufacturing constraints added to the model this session:

  1. Minimum bend radius 7.5mm (REBCO tape cracks below this) --
     cmaes_search.py's geometry_violation() min_clear raised 3mm -> 7.5mm.
  2. Double-pancake construction: every pancake is one of a PAIR of
     adjacent layers (2i, 2i+1) wound as a single continuous piece with the
     SAME turn count (outer edges already match; the inner ends are
     joined) -- cmaes_search.py now optimizes one turn variable per pair,
     not per layer, and asserts N_LAYERS is even. This eliminates every
     odd layer count (3,5,7,9) outright -- including n_layers=9, which was
     this project's most promising open lead before this change.
  3. Turn-count floor removed 50 -> 1 (no material basis for 50; see
     CLAUDE.md's floor-test diagnostic, which -- despite being corrupted
     by a since-fixed race condition on the two DIAGNOSTIC jobs that
     overlapped with a concurrently-running job -- still showed the
     pattern motivating this: designs sitting pinned on an arbitrary
     wall). Sensor-array clearance (7x14x1mm) needs NO separate
     constraint -- verified against the racetrack geometry directly:
     each straight section is 2L long (4L total tape run per turn, see
     params.py's tape_length_m formula), so even the tightest prior
     L~6.3mm design has 2L~12.6mm of straight bore, well over the 7mm
     sensor dimension; the 14mm dimension needs bore diameter
     2*a_inner_min >= 14mm, satisfied once (1) holds (2*7.5=15mm); the
     1mm dimension fits in the existing 3mm coil-to-coil face gap.

None of the four already-explored even layer counts (6, 8, 10, 12) satisfy
pairing as previously recorded, and all sit deeply infeasible on the new
7.5mm bend radius (the old searches converged right at the old 3mm floor).
This script rebuilds a sensible starting point for each: turns pair-
averaged from the prior champion (see module docstring math in the
commit), and `a` recomputed directly from the new floor
(pack_thickness/2 + 7.5mm + a small margin) rather than left deeply
infeasible -- `b` shifted by the same amount to preserve the prior
straight length L. `coil_half_gap` is untouched (that floor is unaffected
by bend radius or pairing) and pinned near it as before.

Runs strictly SEQUENTIALLY (unlike the earlier parallel floor_test.py /
focused_refinement_6_9.py pair, which corrupted each other's results via
a shared-output-path race condition -- see CLAUDE.md/opt_config.py for the
postmortem) so there is no possibility of that bug recurring here. Each
job still sets CMAES_OUT_CSV_OVERRIDE/CMAES_OUT_LOG_OVERRIDE (the actual
fix for that bug) out of caution/hygiene even though sequential execution
alone already prevents the collision.

Turn-count step size (CMAES_N_STD0_OVERRIDE) is set proportionally (20% of
each job's mean pair value) -- the SECOND bug found this session was an
unscaled turn step size (~135-150, the full bound-range) that turned an
intended "polish" run into an unconstrained fresh search. A slightly
looser fraction than a/b's (10% here vs the usual 5%) is deliberate: the
turn distribution needs real room to move under the new constraints, this
isn't a small local polish.

Run:
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \\
        optimize/double_pancake_search.py \\
        > optimize/double_pancake_stdout.log 2>&1 &

Progress: optimize/double_pancake_log.txt (summary); optimize/
double_pancake_logs/n_layers_NN.log (full per-job output). Every job feeds
the same cumulative optimize/runs/cmaes_all_evaluations.csv / visualization/
cmaes_param_map.png -- pull results by run_tag, not cmaes_results.csv
(overwritten by each job in turn, even with the per-job override, since
the override paths below are also reused across jobs for simplicity in
this strictly-sequential script).
"""
import os, sys, csv, json, time, signal, subprocess, traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import opt_config as cfg

PYTHON_BIN = "/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3"

TOTAL_BUDGET_HOURS = 8.0
PER_JOB_MINUTES_CAP = {6: 90, 8: 100, 10: 110, 12: 120}
MAX_EVALS = 6000
A_STD0_FRAC = 0.10   # proportional step size for a/b (10%, looser than the
B_STD0_FRAC = 0.10   # usual 5% -- this is a real re-search, not a polish)
N_STD0_FRAC = 0.20   # proportional step size for turns (20% of mean pair)
GAP_MARGIN_M = 0.001   # keep gap pinned near its (unchanged) floor

# Warm starts: `a` recomputed from the new 7.5mm bend-radius floor using
# each job's pair-averaged turn distribution; `b` shifted to preserve the
# prior straight length L; `coil_half_gap` unchanged (still at its own,
# separate floor). See module docstring for the arithmetic.
JOBS = [
    dict(n_layers=6, seed=60623,
         x0=dict(a=0.0174375, b=0.0254791, coil_half_gap=0.013915040783036767,
                 n_turns=[205, 205, 257, 257, 148, 148])),
    dict(n_layers=8, seed=80823,
         x0=dict(a=0.016950, b=0.0242477, coil_half_gap=0.01752223820011793,
                 n_turns=[194, 194, 178, 178, 244, 244, 144, 144])),
    dict(n_layers=10, seed=101023,
         x0=dict(a=0.0174375, b=0.0237494, coil_half_gap=0.021504820390494547,
                 n_turns=[231, 231, 257, 257, 70, 70, 99, 99, 50, 50])),
    dict(n_layers=12, seed=121223,
         x0=dict(a=0.016725, b=0.0242924, coil_half_gap=0.025611979074903857,
                 n_turns=[154, 154, 238, 238, 146, 146, 93, 93, 82, 82, 61, 61])),
]

LOG_DIR = os.path.join(_ROOT, "optimize", "runs", "double_pancake", "double_pancake_logs")
SUMMARY_LOG = os.path.join(_ROOT, "optimize", "runs", "double_pancake", "double_pancake_log.txt")
RESULTS_CSV = os.path.join(_ROOT, "optimize", "runs", "double_pancake", "double_pancake_results.csv")
HISTORY_CSV = os.path.join(_ROOT, "optimize", "runs", "double_pancake", "double_pancake_history.csv")


def _log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(SUMMARY_LOG, "a") as f:
        f.write(line + "\n")


def _run_job(job, cap_minutes):
    n_layers = job["n_layers"]
    x0 = job["x0"]
    n_pairs = n_layers // 2
    mean_pair = sum(x0["n_turns"][0::2]) / n_pairs
    a_std0 = round(x0["a"] * A_STD0_FRAC, 6)
    b_std0 = round(x0["b"] * B_STD0_FRAC, 6)
    n_std0 = round(mean_pair * N_STD0_FRAC, 3)

    env = os.environ.copy()
    env["CMAES_SWEEP_OVERRIDE_JSON"] = json.dumps(
        dict(x0=x0, seed=job["seed"], max_evals=MAX_EVALS))
    env["CMAES_A_STD0_OVERRIDE"] = str(a_std0)
    env["CMAES_B_STD0_OVERRIDE"] = str(b_std0)
    env["CMAES_N_STD0_OVERRIDE"] = str(n_std0)
    env["CMAES_TIGHT_GAP_MARGIN_M_OVERRIDE"] = str(GAP_MARGIN_M)
    env["CMAES_OUT_CSV_OVERRIDE"] = RESULTS_CSV
    env["CMAES_OUT_LOG_OVERRIDE"] = HISTORY_CSV

    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"n_layers_{n_layers:02d}.log")
    cap_s = cap_minutes * 60

    _log(f"[n_layers={n_layers:2d}] starting: a={x0['a']*1e3:.3f}mm "
        f"b={x0['b']*1e3:.3f}mm gap={x0['coil_half_gap']*1e3:.3f}mm "
        f"a_std0={a_std0*1e3:.3f}mm b_std0={b_std0*1e3:.3f}mm "
        f"n_std0={n_std0:.2f} n_turns={x0['n_turns']} "
        f"seed={job['seed']} cap={cap_minutes}min")

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
        _log(f"[n_layers={n_layers:2d}] {tag} in {dt/60:.1f} min -- "
            f"best tape={float(best['tape_km']):.4f}km "
            f"B={float(best['B_target_T']):.2f}T "
            f"unif={float(best['uniformity_pct']):.3f}% "
            f"hoop={float(best['hoop_MPa']):.0f}MPa "
            f"a={float(best['a_mm']):.2f}mm b={float(best['b_mm']):.2f}mm "
            f"gap={float(best['gap_mm']):.2f}mm n_turns={best['n_turns']}")
    else:
        _log(f"[n_layers={n_layers:2d}] {tag} in {dt/60:.1f} min -- "
            f"NO all-pass design found (check cmaes_all_evaluations.csv "
            f"for this run's run_tag)")
    return best


def main():
    t_start = time.time()
    budget_s = TOTAL_BUDGET_HOURS * 3600
    _log(f"Starting double-pancake re-search: {len(JOBS)} jobs "
        f"({[j['n_layers'] for j in JOBS]}), {TOTAL_BUDGET_HOURS}h total "
        f"budget -- 7.5mm bend radius, double-pancake pairing, "
        f"turn floor removed (1,500)")

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

    _log(f"Double-pancake re-search finished, total "
        f"{(time.time()-t_start)/3600:.2f}h")


if __name__ == "__main__":
    main()
