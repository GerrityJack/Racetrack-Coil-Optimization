"""
overnight_newton_validation.py — unattended overnight validation of the
2026-08-05 t_relax fix (see CLAUDE.md's "FOUND IT" entry) across a spread of
dt values, a reproducibility repeat, a full ramp+hold schedule, and a
t_relax sensitivity check.

Launches transient/validation/run_one_schedule.py as a SEPARATE OS process
per job (never in-process, matching optimize/studies/double_pancake_search.py's
established pattern) so one job's failure/hang can't corrupt the next job's
`ta`/`params` state. Each job is individually time-boxed
(subprocess.Popen + .wait(timeout=...), SIGTERM then SIGKILL on overrun,
matching optimize/studies/overnight_refinement.py) and the whole run has a
global wall-clock budget using time.monotonic() (NOT time.time() -- CLAUDE.md
records a real bug from using time.time() for a budget check that a paused/
resumed run then blew through).

Launch via the direct python binary, NOT `conda run` (CLAUDE.md: `conda run`
buffers ALL subprocess stdout until exit, making a live log tail useless):

    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \\
        transient/studies/overnight_newton_validation.py \\
        > transient/runs/newton_overnight/orchestrator_stdout.log 2>&1 &
    disown

Progress: transient/runs/newton_overnight/overnight_log.txt (one line per
job start/finish, like optimize's sweep_n_layers_log.txt). Full per-job
output: transient/overnight_logs/<tag>.log. Machine-readable per-job result:
transient/runs/newton_overnight/<tag>.json (written by run_one_schedule.py
itself, so a job's result survives even if this orchestrator is killed).
Aggregate summary (rewritten after EVERY job, not just at the end, so a kill
loses at most one job's worth of aggregation, mirroring cmaes_search.py's
FLUSH_EVERY incremental-write lesson): transient/runs/newton_overnight/summary.csv
"""
import csv
import json
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRANS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_TRANS)

PY = "/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3"
RUNNER = os.path.join(_TRANS, "validation", "run_one_schedule.py")
RESULTS_DIR = os.path.join(_TRANS, "runs", "newton_overnight")
LOG_DIR = os.path.join(_TRANS, "overnight_logs")
PROGRESS_LOG = os.path.join(RESULTS_DIR, "overnight_log.txt")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TOTAL_BUDGET_HOURS = 11.0

# (tag, n_ramp, n_hold, t_hold, max_outer, t_relax, timeout_minutes)
# I_op=196.0 (params.I_design), t_ramp=600.0 (params.ramp_duration) fixed
# for every job -- only the schedule granularity, hold phase, and t_relax
# vary, isolating exactly what each job is meant to test.
JOBS = [
    # dt=150s (n_ramp=4), the case both the isolated-step and full-ramp
    # tests validated -- run TWICE (independent mesh each time) as the
    # reproducibility check this project's history says is never optional.
    ("dt150_tr015_run1", 4, 0, 0.0, 100, 0.15, 60),
    ("dt150_tr015_run2", 4, 0, 0.0, 100, 0.15, 60),
    # Finer dt -- generalization test across the range a real multi-step
    # schedule would actually use (25-150s per CLAUDE.md's own framing).
    ("dt100_tr015", 6, 0, 0.0, 130, 0.15, 80),
    ("dt075_tr015", 8, 0, 0.0, 130, 0.15, 95),
    ("dt050_tr015", 12, 0, 0.0, 150, 0.15, 130),
    ("dt025_tr015", 24, 0, 0.0, 150, 0.15, 180),
    # Full ramp + hold -- the actual production shape (charge to I_op, then
    # sit there), not just a bare ramp.
    ("full_ramp_hold_dt150", 4, 4, 200.0, 100, 0.15, 90),
    # t_relax sensitivity around the value that converged the isolated step
    # (0.15) -- does a full ramp behave better/worse at neighbouring values?
    ("dt150_tr010", 4, 0, 0.0, 100, 0.10, 60),
    ("dt150_tr020", 4, 0, 0.0, 100, 0.20, 60),
]


def log_progress(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")


def write_summary(results):
    fields = ["tag", "ok", "error", "n_steps", "n_converged", "wall_s",
             "n_ramp", "n_hold", "t_relax", "max_outer",
             "worst_step_scif_mT", "final_step_scif_mT",
             "final_step_stop_reason"]
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = dict(tag=r.get("tag"), ok=r.get("ok"), error=r.get("error"),
                      n_steps=r.get("n_steps"), n_converged=r.get("n_converged"),
                      wall_s=round(r.get("wall_s", 0.0), 1) if r.get("wall_s") else None,
                      n_ramp=r.get("args", {}).get("n_ramp"),
                      n_hold=r.get("args", {}).get("n_hold"),
                      t_relax=r.get("args", {}).get("t_relax"),
                      max_outer=r.get("args", {}).get("max_outer"))
            steps = r.get("steps") or []
            if steps:
                row["final_step_scif_mT"] = round(steps[-1]["scif_mT"], 2)
                row["final_step_stop_reason"] = steps[-1]["stop_reason"]
                row["worst_step_scif_mT"] = round(
                    max(abs(s["scif_mT"]) for s in steps), 2)
            w.writerow(row)


def run_job(tag, n_ramp, n_hold, t_hold, max_outer, t_relax, timeout_min):
    out_json = os.path.join(RESULTS_DIR, f"{tag}.json")
    log_path = os.path.join(LOG_DIR, f"{tag}.log")
    # Remove any pre-existing json from a prior interrupted run, same
    # discipline sweep_restarts.py's postmortem established: never let a
    # stale result file be silently mistaken for this run's output if the
    # job fails before writing (run_one_schedule.py writes it in a finally
    # block, so this should be rare, but cheap to guarantee).
    if os.path.exists(out_json):
        os.remove(out_json)

    cmd = [PY, RUNNER, "--tag", tag, "--n-ramp", str(n_ramp),
          "--n-hold", str(n_hold), "--t-hold", str(t_hold),
          "--max-outer", str(max_outer), "--t-relax", str(t_relax),
          "--out-json", out_json]

    log_progress(f"START {tag}  n_ramp={n_ramp} n_hold={n_hold} t_hold={t_hold} "
                f"max_outer={max_outer} t_relax={t_relax} "
                f"timeout={timeout_min}min -> {log_path}")

    t0 = time.monotonic()
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                cwd=_ROOT)
        try:
            proc.wait(timeout=timeout_min * 60)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    elapsed = time.monotonic() - t0

    if os.path.exists(out_json):
        with open(out_json) as f:
            result = json.load(f)
    else:
        result = dict(tag=tag, ok=False,
                     error="timed_out, no result written" if timed_out
                     else "process exited without writing a result",
                     args=dict(n_ramp=n_ramp, n_hold=n_hold, t_hold=t_hold,
                              max_outer=max_outer, t_relax=t_relax),
                     steps=[], wall_s=elapsed)

    status = "TIMEOUT" if timed_out else ("OK" if result.get("ok") else "FAILED")
    n_conv = result.get("n_converged", 0)
    n_steps = result.get("n_steps", len(result.get("steps", [])))
    log_progress(f"FINISH {tag}  status={status}  "
                f"{n_conv}/{n_steps} steps converged  "
                f"wall={elapsed:.0f}s  error={result.get('error')}")
    return result


def main():
    t_start = time.monotonic()
    log_progress(f"=== overnight Newton-hybrid validation starting, "
                f"{len(JOBS)} jobs, budget {TOTAL_BUDGET_HOURS}h ===")

    results = []
    for tag, n_ramp, n_hold, t_hold, max_outer, t_relax, timeout_min in JOBS:
        elapsed_h = (time.monotonic() - t_start) / 3600.0
        if elapsed_h >= TOTAL_BUDGET_HOURS:
            log_progress(f"SKIP {tag} and all remaining jobs -- total budget "
                        f"({TOTAL_BUDGET_HOURS}h) exhausted after {elapsed_h:.2f}h")
            break
        result = run_job(tag, n_ramp, n_hold, t_hold, max_outer, t_relax,
                        timeout_min)
        results.append(result)
        write_summary(results)   # incremental -- never lose completed jobs

    log_progress(f"=== overnight run finished: {len(results)}/{len(JOBS)} "
                f"jobs completed, {(time.monotonic()-t_start)/3600.0:.2f}h "
                f"elapsed. Summary: {SUMMARY_CSV} ===")


if __name__ == "__main__":
    main()
