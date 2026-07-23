"""
sweep_restarts.py — many diverse CMA-ES cold-start restarts, time-boxed
==========================================================================
Launches successive optimize/cmaes_search.py subprocess runs, each from an
independently, randomly sampled starting point spanning the practical
parameter space, for up to TOTAL_BUDGET_HOURS wall-clock time. Built for
the 2026-07-21 diverse-restart study: 4 hand-picked cold starts (run via
manual opt_config.py edits) all landed worse than run 3's 0.373km optimum
and none reproduced its gap-at-floor behavior -- this automates many more
restarts unattended to sweep much more of the space and build up
optimize/cmaes_all_evaluations.csv (the cumulative "parameter-space map"
dataset) far beyond what a handful of manual restarts could cover.

Each restart is a COMPLETELY SEPARATE OS process (conda run ... cmaes_
search.py), never a repeated in-process call to cmaes_search.main() --
that function's module-level globals (_eval_count, _history, _best) are
only ever reset by a fresh process/import, not by calling main() again,
so looping in-process would silently corrupt bookkeeping across restarts.

Starting points are handed to each subprocess via the CMAES_SWEEP_OVERRIDE_JSON
environment variable (see opt_config.py's "sweep override" section) --
opt_config.py's own on-disk CMAES_X0/SEED/MAX_EVALS are NEVER rewritten by
this script, so a partial or interrupted sweep leaves no residue to clean
up, and the file's defaults stay exactly the run-3 warm start throughout.

Sampling: a ~ U(0.015, 0.090) m, b ~ a + U(0.010, 0.150) m,
coil_half_gap ~ U(*cfg.CMAES_HALF_GAP_BOUNDS), each n_turns[i] ~
DiscreteU(*cfg.CMAES_N_BOUNDS) independently -- a plain uniform random
design over the same box CMA-ES itself is allowed to explore (not a
Latin Hypercube -- simpler, and with dozens of restarts planned, coverage
does not hinge on LHS's variance-reduction benefit over plain uniform
sampling). Reproducible via MASTER_SEED.

Run from Racetrack_v4 root, in the background (this is meant to run
unattended for hours):
    conda run -n fenicsx-env python3 optimize/sweep_restarts.py

Progress: optimize/sweep_restarts_log.txt (one line per completed restart:
index, elapsed, x0 sampled, best tape_km/B_target/uniformity/hoop found).
Per-restart full console output: optimize/sweep_logs/restart_<i>.log.
All evaluations still accumulate into optimize/cmaes_all_evaluations.csv
and visualization/cmaes_param_map.png exactly as any other cmaes_search.py
run -- nothing extra needed there.
"""
import os, sys, csv, json, time, traceback, subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import opt_config as cfg

TOTAL_BUDGET_HOURS = 11.5     # leaves ~30 min margin under a 12h absence
PER_RESTART_MAX_EVALS = 300   # breadth over depth -- run 3 itself needed
                              # ~700 evals to fully settle, but each restart
                              # here just needs to show which basin it heads
                              # toward, not fully polish it
MASTER_SEED = 2026072107      # reproducible sequence of starting points
N_LAYERS = 7

SWEEP_LOG = os.path.join(_ROOT, "optimize", "sweep_restarts_log.txt")
SWEEP_LOGS_DIR = os.path.join(_ROOT, "optimize", "sweep_logs")
RESULTS_CSV = os.path.join(_ROOT, cfg.CMAES_OUT_CSV)


def _sample_x0(rng):
    a = float(rng.uniform(0.015, 0.090))
    b = a + float(rng.uniform(0.010, 0.150))
    gap = float(rng.uniform(*cfg.CMAES_HALF_GAP_BOUNDS))
    n_turns = [int(rng.integers(cfg.CMAES_N_BOUNDS[0],
                                cfg.CMAES_N_BOUNDS[1] + 1))
              for _ in range(N_LAYERS)]
    return dict(a=round(a, 6), b=round(b, 6),
               coil_half_gap=round(gap, 6), n_turns=n_turns)


def _log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(SWEEP_LOG, "a") as f:
        f.write(line + "\n")


def _run_one_restart(i, x0, seed):
    env = os.environ.copy()
    env["CMAES_SWEEP_OVERRIDE_JSON"] = json.dumps(dict(
        x0=x0, seed=seed, max_evals=PER_RESTART_MAX_EVALS))

    # cmaes_results.csv is only (re)written if this run finds >=1 all-pass
    # design (_write_csv() skips it otherwise) -- remove any pre-existing
    # copy first so a restart that finds nothing can't be misreported as
    # having reproduced a PREVIOUS restart's leftover result.
    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)

    os.makedirs(SWEEP_LOGS_DIR, exist_ok=True)
    log_path = os.path.join(SWEEP_LOGS_DIR, f"restart_{i:03d}.log")
    with open(log_path, "w") as logf:
        subprocess.run(
            ["conda", "run", "-n", "fenicsx-env", "python3",
             "optimize/cmaes_search.py"],
            cwd=_ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT,
            check=True)

    if not os.path.exists(RESULTS_CSV):
        return None
    with open(RESULTS_CSV) as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def main():
    rng = np.random.default_rng(MASTER_SEED)
    t_start = time.time()
    budget_s = TOTAL_BUDGET_HOURS * 3600.0
    durations = []

    _log(f"Starting sweep: budget {TOTAL_BUDGET_HOURS} h, "
        f"{PER_RESTART_MAX_EVALS} evals/restart, master seed {MASTER_SEED}")

    i = 0
    while True:
        elapsed = time.time() - t_start
        remaining = budget_s - elapsed
        avg_dur = sum(durations) / len(durations) if durations else None
        if avg_dur is not None and remaining < 1.3 * avg_dur:
            _log(f"Stopping: {remaining/60:.1f} min left, avg restart "
                f"takes {avg_dur/60:.1f} min -- not enough margin left.")
            break
        if remaining <= 0:
            _log("Stopping: time budget exhausted.")
            break

        i += 1
        x0 = _sample_x0(rng)
        seed = 5000 + i
        _log(f"[restart {i:03d}] starting: a={x0['a']*1e3:.1f}mm "
            f"b={x0['b']*1e3:.1f}mm gap={x0['coil_half_gap']*1e3:.1f}mm "
            f"n_turns={x0['n_turns']} seed={seed} "
            f"(elapsed {elapsed/3600:.2f}h / {TOTAL_BUDGET_HOURS}h)")

        t0 = time.time()
        try:
            best = _run_one_restart(i, x0, seed)
        except Exception:
            _log(f"[restart {i:03d}] FAILED:\n{traceback.format_exc()}")
            continue
        dt = time.time() - t0
        durations.append(dt)

        if best:
            _log(f"[restart {i:03d}] done in {dt/60:.1f} min -- "
                f"best tape={float(best['tape_km']):.3f}km "
                f"B={float(best['B_target_T']):.2f}T "
                f"unif={float(best['uniformity_pct']):.3f}% "
                f"hoop={float(best['hoop_MPa']):.0f}MPa "
                f"a={float(best['a_mm']):.1f}mm b={float(best['b_mm']):.1f}mm "
                f"gap={float(best['gap_mm']):.1f}mm")
        else:
            _log(f"[restart {i:03d}] done in {dt/60:.1f} min -- "
                f"no feasible design found")

    total_h = (time.time() - t_start) / 3600.0
    _log(f"Sweep finished: {i} restarts attempted in {total_h:.2f} h.")


if __name__ == "__main__":
    main()
