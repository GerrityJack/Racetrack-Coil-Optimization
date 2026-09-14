"""
run_4p2K_overnight.py — staged overnight orchestrator for the 4.2K-model
T-A-safe-current search
=============================================================================
2026-09-13. Per explicit user direction: (1) try the 4.2K-improved model
at the champion's CURRENT 6-layer topology first; (2) if that search
stalls, try new starting points / wider bounds (still 6 layers); (3) only
if THAT doesn't work either, escalate to more layers, continuing until
B_target_T >= 10T (with hoop/uniformity also satisfied) or the night runs
out. This script drives `ta_safe_margin_search_4p2K.py` (itself the
4.2K-model sibling of `ta_safe_margin_search.py`) through that staged
sequence, entirely unattended -- each stage is time-boxed via a hard
subprocess timeout (the search script checkpoints and exits cleanly on
SIGTERM, so a timeout never loses progress within a stage), and the
decision to stop or escalate is made automatically from each stage's own
results.csv.

Prior-evidence context for the escalation targets (all UNDER THE OLD 20K
MODEL, from this project's own past runs, both killed mid-flight by a
harness SIGTERM, not by design -- see `optimize/runs/ta_safe_margin_8layer/`
`_10layer/`): 8 layers reached B_target_T=5.03 (909 evals), 10 layers
reached 4.43 (only 60 evals, far less mature). Both better than 6 layers'
own historical ceiling (4.10T, ~1247 evals) but nowhere near 10T -- this
is why the user's own reference point (a real 20-pancake/40-layer design
reaching 25T, cited in the same Tsuchiya et al. paper the 4.2K ratio was
built from) suggests a genuinely large layer count may be needed, not
just "a few more". Stage 4/5 below warm-start from those exact prior best
designs (translated into this session's own 4.2K-model search).

Run (background, direct binary):
    PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
    nohup $PY optimize/studies/run_4p2K_overnight.py \
        > optimize/runs/ta_safe_margin_4p2K/orchestrator_stdout.log 2>&1 &
"""
import os
import sys
import csv
import json
import time
import subprocess
import signal

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_OPT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_OPT)

PY = "/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3"
SEARCH_SCRIPT = os.path.join(_HERE, "ta_safe_margin_search_4p2K.py")
BASE_OUT = os.path.join(_OPT, "runs", "ta_safe_margin_4p2K")
os.makedirs(BASE_OUT, exist_ok=True)
MASTER_LOG = os.path.join(BASE_OUT, "orchestrator.log")

B_TARGET_GOAL = 10.0

_stop_requested = False


def _request_stop(signum, frame):
    global _stop_requested
    _stop_requested = True
    log("[stop requested] will not launch a new stage; current stage's own "
        "subprocess will be sent SIGTERM (checkpoints cleanly).")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(MASTER_LOG, "a") as fh:
        fh.write(line + "\n")


def read_best(out_dir):
    """Returns (B_target_T, all_constraints_ok, row_dict) or (None, None, None)."""
    csv_path = os.path.join(out_dir, "results.csv")
    if not os.path.exists(csv_path):
        return None, None, None
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None, None, None
    row = rows[0]
    try:
        b = float(row.get("B_target_T", "") or "nan")
    except ValueError:
        b = float("nan")
    ok = str(row.get("all_constraints_ok", "")).strip().lower() == "true"
    return b, ok, row


def run_stage(name, out_dir, duration_s, extra_env=None, max_evals=None,
             n_workers=4):
    """Launches ta_safe_margin_search_4p2K.py as a subprocess with the
    given env overrides, waits up to duration_s (sending SIGTERM at the
    deadline -- the search script's own signal handler checkpoints and
    exits within seconds, not instantly killed), and returns
    (B_target_T, all_constraints_ok, row_dict)."""
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env["TA_SAFE_OUT_DIR"] = out_dir
    if max_evals is not None:
        env["TA_SAFE_MAX_EVALS"] = str(max_evals)
    env["TA_SAFE_N_WORKERS"] = str(n_workers)
    if extra_env:
        env.update(extra_env)

    log(f"=== STAGE {name}: launching (budget {duration_s/60:.0f} min) "
        f"out_dir={out_dir} ===")
    log(f"    env overrides: "
        f"{ {k: v for k, v in (extra_env or {}).items()} }")

    log_path = os.path.join(out_dir, "log.txt")
    t0 = time.time()
    with open(log_path, "ab") as logfh:
        # 2026-09-13 bugfix (found live during this project's own use of
        # this script): a plain proc.kill() only kills the immediate
        # child. If a ProcessPoolExecutor worker pool is still busy when
        # the 180s post-SIGTERM grace period expires, those workers are
        # never signaled and are orphaned -- 20 such leaked workers from
        # this script's own stages 1-4 were found still running 6+ hours
        # later, consuming real CPU/RAM. start_new_session=True + killing
        # the whole process GROUP on every termination path fixes this.
        proc = subprocess.Popen([PY, SEARCH_SCRIPT], env=env,
                                stdout=logfh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            proc.wait(timeout=duration_s)
        except subprocess.TimeoutExpired:
            log(f"    stage {name}: budget exhausted, sending SIGTERM to "
                f"process group (script checkpoints and exits cleanly) ...")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                log(f"    stage {name}: did not exit within 180s of "
                    f"SIGTERM, SIGKILL-ing the whole process group.")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=30)
    dt = time.time() - t0
    b, ok, row = read_best(out_dir)
    log(f"=== STAGE {name} finished after {dt/60:.1f} min: "
        f"best B_target_T={b}  all_constraints_ok={ok} ===")
    if row:
        log(f"    best design: a={row.get('a_mm')}mm b={row.get('b_mm')}mm "
            f"gap={row.get('gap_mm')}mm n_turns={row.get('n_turns')} "
            f"I_op={row.get('I_op_A')}A tape={row.get('tape_km')}km "
            f"uniformity={row.get('uniformity_pct')}% hoop={row.get('hoop_MPa')}MPa")
    return b, ok, row


def goal_met(b, ok):
    return (b is not None and ok and b >= B_TARGET_GOAL)


def main():
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    log("#" * 90)
    log("4.2K overnight search orchestrator starting")
    log(f"Goal: B_target_T >= {B_TARGET_GOAL} with all_constraints_ok (hoop, "
        f"uniformity) also satisfied")
    log("#" * 90)

    overall_best = (None, None, None, None)  # (b, ok, row, stage_name)

    def consider(b, ok, row, name):
        nonlocal overall_best
        if b is not None and (overall_best[0] is None or
                              (row and row.get("fitness") and
                               overall_best[2] and
                               float(row.get("fitness", "inf")) <
                               float(overall_best[2].get("fitness", "inf")))):
            overall_best = (b, ok, row, name)

    stages = [
        # (name, out_subdir, duration_s, env_overrides, max_evals)
        ("1_6layer_champion_start",
         "stage1_6layer", 5400, {}, 300),

        ("2_6layer_alt_start_wider_bounds",
         "stage2_6layer_altstart", 3600,
         dict(TA_SAFE_X0_OVERRIDE_JSON=json.dumps(dict(
             a=0.045, b=0.070, gap=0.0155,
             n_turns=[600, 600, 600, 600, 300, 300])),
              TA_SAFE_A_STD0_M="0.010", TA_SAFE_B_STD0_M="0.015",
              TA_SAFE_GAP_STD0_M="0.004", TA_SAFE_N_STD0="100",
              TA_SAFE_GAP_RANGE_WIDE_M="0.100"),
         300),

        ("3_8layer_warmstart_from_20K_best",
         "stage3_8layer", 7200,
         dict(CMAES_X0_JSON_OVERRIDE=json.dumps(dict(
             a=0.03560, b=0.04109, coil_half_gap=0.01775,
             n_turns=[719, 719, 728, 728, 72, 72, 657, 657])),
              TA_SAFE_X0_OVERRIDE_JSON=json.dumps(dict(
                  a=0.03560, b=0.04109, gap=0.01775,
                  n_turns=[719, 719, 728, 728, 72, 72, 657, 657])),
              TA_SAFE_A_STD0_M="0.008", TA_SAFE_B_STD0_M="0.012",
              TA_SAFE_GAP_STD0_M="0.003", TA_SAFE_N_STD0="80"),
         400),

        ("4_10layer_warmstart_from_20K_best",
         "stage4_10layer", 7200,
         dict(CMAES_X0_JSON_OVERRIDE=json.dumps(dict(
             a=0.03830, b=0.04528, coil_half_gap=0.02197,
             n_turns=[609, 609, 639, 639, 76, 76, 550, 550, 571, 571])),
              TA_SAFE_X0_OVERRIDE_JSON=json.dumps(dict(
                  a=0.03830, b=0.04528, gap=0.02197,
                  n_turns=[609, 609, 639, 639, 76, 76, 550, 550, 571, 571])),
              TA_SAFE_A_STD0_M="0.008", TA_SAFE_B_STD0_M="0.012",
              TA_SAFE_GAP_STD0_M="0.003", TA_SAFE_N_STD0="80"),
         400),

        ("5_12layer_fresh_wide_search",
         "stage5_12layer", 7200,
         dict(CMAES_X0_JSON_OVERRIDE=json.dumps(dict(
             a=0.041, b=0.049, coil_half_gap=0.026,
             n_turns=[550, 550, 550, 550, 80, 80, 550, 550, 550, 550, 550, 550])),
              TA_SAFE_X0_OVERRIDE_JSON=json.dumps(dict(
                  a=0.041, b=0.049, gap=0.026,
                  n_turns=[550, 550, 550, 550, 80, 80, 550, 550, 550, 550, 550, 550])),
              TA_SAFE_A_STD0_M="0.008", TA_SAFE_B_STD0_M="0.012",
              TA_SAFE_GAP_STD0_M="0.003", TA_SAFE_N_STD0="80"),
         500),
    ]

    for name, subdir, duration_s, env_overrides, max_evals in stages:
        if _stop_requested:
            log("[stop requested] ending orchestration before next stage.")
            break
        out_dir = os.path.join(BASE_OUT, subdir)
        b, ok, row = run_stage(name, out_dir, duration_s, env_overrides,
                               max_evals=max_evals)
        consider(b, ok, row, name)
        if goal_met(b, ok):
            log(f"*** GOAL MET at stage {name}: B_target_T={b} with all "
                f"constraints satisfied. Stopping orchestration. ***")
            break
    else:
        log("All stages exhausted without reaching the 10T goal with all "
            "constraints satisfied.")

    log("#" * 90)
    log("FINAL SUMMARY")
    b, ok, row, name = overall_best
    if row is not None:
        log(f"Best design found across ALL stages (by fitness, "
            f"all_constraints_ok rows only), from stage '{name}':")
        for k in ("a_mm", "b_mm", "gap_mm", "n_turns", "n_total", "tape_km",
                  "B_target_T", "uniformity_pct", "hoop_MPa", "I_op_A",
                  "ta_worst_margin", "n_ta_solves"):
            log(f"  {k}: {row.get(k)}")
        log(f"Goal (B_target_T >= {B_TARGET_GOAL}, all constraints ok): "
            f"{'MET' if goal_met(b, ok) else 'NOT MET'}")
    else:
        log("No stage produced any all_constraints_ok=True candidate at all.")
    log("#" * 90)


if __name__ == "__main__":
    main()
