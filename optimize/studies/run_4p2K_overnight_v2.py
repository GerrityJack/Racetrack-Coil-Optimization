"""
run_4p2K_overnight_v2.py -- continuing layer-count escalation for the 4.2K
T-A-safe-current search, per explicit user direction (2026-09-13):
"continue on that path until we start getting some mean fields above 10T"

Picks up exactly where run_4p2K_overnight.py's 5-stage run left off
(6/8/10/12 layers -> best B_target_T 3.67/3.83/5.56/5.54/6.72). That run's
own per-stage budgets (60-126 min) were visibly evaluation-starved at
higher layer counts (10L: 30 evals, 12L: 30 evals, vs 8L's 63) because
mesh size and per-solve cost both grow with layer count -- this run uses
LONGER per-stage budgets so each layer count gets a fairer shake, and
keeps escalating N_LAYERS by 4 (14->wait, even-only: uses +4 so 12->16->
20->24->...) automatically, warm-starting each new stage from the
PREVIOUS stage's own best design (extended with 2 more pairs at that
design's own "big pair" turn count), until either B_target_T >= 10.0 is
found anywhere (the user's literal goal -- reported immediately,
regardless of whether hoop/uniformity also pass) or MAX_LAYERS is hit.

Rough extrapolation from the 4 completed points (log-log power-law fit,
B ~ N^0.82): 6->3.8T, 12->6.72T suggests ~20 layers (10 pancakes) for
10T -- notably close to the Tsuchiya et al. paper's own cited 20-pancake/
25T reference design that motivated this whole line of investigation.
Treat this as a rough guide for how many stages to expect, not a
guarantee -- CMA-ES progress at each layer count is itself budget-
limited, and the true relationship could easily be shallower.

Appends to the SAME master log as the first overnight run
(optimize/runs/ta_safe_margin_4p2K/orchestrator.log) for one continuous
record; new stage output dirs (stage6_16layer, stage7_20layer, ...) so
nothing from the first run is overwritten.

Run (background, direct binary):
    PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
    nohup $PY optimize/studies/run_4p2K_overnight_v2.py \
        > optimize/runs/ta_safe_margin_4p2K/orchestrator_v2_stdout.log 2>&1 &
"""
import os
import sys
import ast
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
MAX_LAYERS = 40          # matches the cited real 40-layer/25T reference design
LAYER_STEP = 4
# 2026-09-13, revised after measuring real per-solve cost at 16 layers:
# solve time scales SUPER-linearly with turn count (power-law fit on the
# only two real data points available, 6L=42.7s/solve @ 1726 turns and
# 12L=201s/solve @ 5798 turns, gives time ~ turns^1.28) -- extrapolated to
# 16L's actual candidates (~8700 turns) that's ~340s/solve, and a full
# candidate (up to ~9-14 T-A solves for the bisection) costs ~50-80
# minutes. With 4 workers and CMA-ES's own default popsize (~11 at this
# dimensionality), the busiest worker needs ~3 candidates/generation --
# ~150-240 min for ONE generation. The original STAGE_BUDGET_S=9000
# (150min) would fit at most one generation; GEN_TIMEOUT_S=5400 (90min,
# set in run_stage() below) would likely discard it as a false "hang"
# before it even finished. Both raised accordingly -- this makes each
# stage a genuinely multi-hour commitment at these layer counts, which is
# simply the real cost of forcing every solve to full length (see
# ta_safe_margin_search_4p2K.py's own docstring for why that's required
# for trustworthiness) at a mesh size this large; see also POPSIZE_OVERRIDE
# below, which trades some CMA-ES statistical robustness for more
# generations per stage in this expensive regime.
# 2026-09-13, revised AGAIN under time pressure (user wants a real shot
# at >10T by morning): 16 layers plateaued at 6.99T across several
# generations under the 6h budget -- with ~8-9 hours left tonight, 6h/
# stage would only reach ~20-24 layers, and the 6L->16L trend (3.8, 5.56,
# 5.54, 6.72, 6.99T) is not obviously going to cross 10T that soon.
# Shortened to 3h/stage to fit ~3 more stages (16->20->24->28) instead of
# ~1-2, trading generation depth per layer count for BREADTH across layer
# counts -- higher chance of finding the real inflection point before
# running out of night. Honest tradeoff, not a free lunch: fewer
# generations per stage means each layer count's true potential is less
# thoroughly explored.
STAGE_BUDGET_S = 10800     # 3 hours
MAX_EVALS_PER_STAGE = 600  # not the real limiter, the timeout is
N_WORKERS = 4
POPSIZE_OVERRIDE = 8      # vs pycma's own default (~11 at N_PAIRS=8) --
                          # fewer candidates/generation means more
                          # generations fit in the same stage budget
BISECT_TOL_A_OVERRIDE = 6.0  # vs ta_safe_current.py's own default 3.0 --
                          # cuts ~1 bisection solve/candidate (~10-15% of
                          # per-candidate cost), no accuracy cost on the
                          # eventual winner (re-solved precisely anyway)
W_TAPE = 0.004
MIN_COIL_GAP_M = 0.003

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


def gap_floor(n_layers):
    return n_layers * W_TAPE / 2.0 + MIN_COIL_GAP_M / 2.0


def read_best(out_dir):
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


def read_best_feasible_by_field(out_dir):
    """Best (highest B_target_T) among FEASIBLE rows, regardless of
    all_constraints_ok -- used to pick the next stage's warm start even
    when nothing this stage fully passed (which is expected, since 10T
    is a hard floor most stages won't clear)."""
    csv_path = os.path.join(out_dir, "history.csv")
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    feas = [r for r in rows if str(r.get("feasible", "")).strip().lower() == "true"]
    if not feas:
        return None

    def f(r):
        try:
            return float(r.get("B_target_T", "") or -1)
        except ValueError:
            return -1
    feas.sort(key=f, reverse=True)
    return feas[0]


def next_design(prev_design, prev_layers):
    """prev_design: dict(a, b, coil_half_gap, n_turns). Returns
    (new_design, new_layers) extending by LAYER_STEP layers (LAYER_STEP//2
    new pairs), each new pair's turn count set to the mean of the
    previous design's own "big" pairs (excluding any outlier thin pair --
    every winning design so far has had exactly one much-thinner pair),
    gap set to the new layer count's own floor plus the SAME margin above
    that floor the previous design used (preserves whatever face-gap
    choice CMA-ES converged on), a/b carried over unchanged (CMA-ES's own
    step sizes will re-explore from there)."""
    old_n_turns = list(prev_design["n_turns"])
    new_layers = prev_layers + LAYER_STEP
    old_pairs = [old_n_turns[2 * i] for i in range(prev_layers // 2)]
    max_pair = max(old_pairs)
    big_pairs = [p for p in old_pairs if p > 0.3 * max_pair]
    rep = int(round(sum(big_pairs) / len(big_pairs))) if big_pairs else \
        int(round(sum(old_pairs) / len(old_pairs)))
    n_new_pairs = LAYER_STEP // 2
    new_pairs = old_pairs + [rep] * n_new_pairs
    new_n_turns = []
    for p in new_pairs:
        new_n_turns += [p, p]

    old_gap = prev_design["coil_half_gap"]
    old_floor = gap_floor(prev_layers)
    new_floor = gap_floor(new_layers)
    new_gap = new_floor + max(0.0, old_gap - old_floor)

    new_design = dict(a=prev_design["a"], b=prev_design["b"],
                      coil_half_gap=new_gap, n_turns=new_n_turns)
    return new_design, new_layers


def run_stage(name, out_dir, duration_s, design, max_evals=None,
             n_workers=N_WORKERS, alt_stds=None):
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env["TA_SAFE_OUT_DIR"] = out_dir
    if max_evals is not None:
        env["TA_SAFE_MAX_EVALS"] = str(max_evals)
    env["TA_SAFE_N_WORKERS"] = str(n_workers)
    # 2026-09-13: the 16-layer stage's very first generation already hit
    # the sibling script's default GEN_TIMEOUT_S=2700s (45 min) ceiling on
    # at least one worker (mesh size / per-solve cost both grow with
    # layer count, well past what 2700s was set against). Give later,
    # bigger-mesh stages more room before a slow-but-real solve gets
    # mistaken for a hang and its whole generation discarded.
    # 2026-09-13: must stay meaningfully BELOW STAGE_BUDGET_S (now 3h) --
    # equal-or-larger would mean a slow generation gets discarded by
    # GEN_TIMEOUT_S at almost exactly the moment the stage's own SIGTERM
    # fires anyway, wasting the near-complete work for nothing.
    env["TA_SAFE_GEN_TIMEOUT_S"] = str(int(os.environ.get(
        "TA_SAFE_GEN_TIMEOUT_S_OVERRIDE", 7200)))
    if POPSIZE_OVERRIDE is not None and "TA_SAFE_POPSIZE" not in env:
        env["TA_SAFE_POPSIZE"] = str(POPSIZE_OVERRIDE)
    env["TA_SAFE_BISECT_TOL_A"] = str(os.environ.get(
        "TA_SAFE_BISECT_TOL_A_OVERRIDE", BISECT_TOL_A_OVERRIDE))

    cmaes_x0 = dict(a=design["a"], b=design["b"],
                    coil_half_gap=design["coil_half_gap"],
                    n_turns=design["n_turns"])
    ta_safe_x0 = dict(a=design["a"], b=design["b"], gap=design["coil_half_gap"],
                      n_turns=design["n_turns"])
    env["CMAES_X0_JSON_OVERRIDE"] = json.dumps(cmaes_x0)
    env["TA_SAFE_X0_OVERRIDE_JSON"] = json.dumps(ta_safe_x0)
    stds = alt_stds or dict(TA_SAFE_A_STD0_M="0.008", TA_SAFE_B_STD0_M="0.012",
                            TA_SAFE_GAP_STD0_M="0.003", TA_SAFE_N_STD0="80")
    env.update(stds)

    log(f"=== STAGE {name}: launching (budget {duration_s/60:.0f} min) "
        f"out_dir={out_dir} ===")
    log(f"    x0: a={design['a']*1e3:.2f}mm b={design['b']*1e3:.2f}mm "
        f"gap={design['coil_half_gap']*1e3:.2f}mm n_turns={design['n_turns']}")

    log_path = os.path.join(out_dir, "log.txt")
    t0 = time.time()
    with open(log_path, "ab") as logfh:
        # 2026-09-13 bugfix: start_new_session=True puts this subprocess (and
        # every ProcessPoolExecutor worker IT spawns) in its own process
        # group. Discovered live tonight: a plain proc.kill() only kills
        # the immediate child -- if that child is mid-generation (workers
        # still busy) when the 180s grace period after SIGTERM expires,
        # proc.kill() sends SIGKILL to the main process only, and its
        # worker subprocesses (which never got a chance to run their own
        # cleanup) are orphaned and keep running indefinitely. Found 20
        # such leaked workers from stages 1-4 of the FIRST overnight run,
        # still consuming ~4.4GB RAM and real CPU 6+ hours later, actively
        # slowing down this (stage 6) run. Fix: kill the whole process
        # GROUP, not just the one PID, at every termination point below.
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
        f"best (all_constraints_ok) B_target_T={b}  all_constraints_ok={ok} ===")
    best_feas = read_best_feasible_by_field(out_dir)
    if best_feas:
        log(f"    best FEASIBLE row (any constraint status): "
            f"B_target_T={best_feas.get('B_target_T')} "
            f"a={best_feas.get('a_mm')}mm b={best_feas.get('b_mm')}mm "
            f"gap={best_feas.get('gap_mm')}mm n_turns={best_feas.get('n_turns')} "
            f"I_op={best_feas.get('I_op_A')}A hoop={best_feas.get('hoop_MPa')}MPa "
            f"unif={best_feas.get('uniformity_pct')}%")
    return b, ok, row, best_feas


def main():
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    log("#" * 90)
    log("4.2K overnight search orchestrator v2 (layer-count continuation) starting")
    log(f"Goal: B_target_T >= {B_TARGET_GOAL} ANYWHERE (reported immediately, "
        f"regardless of hoop/uniformity) -- per explicit user direction")
    log(f"Escalating by {LAYER_STEP} layers/stage, {STAGE_BUDGET_S/60:.0f} min/stage, "
        f"up to {MAX_LAYERS} layers")
    log("#" * 90)

    # 2026-09-13: skipping stage 6 (16 layers)'s remaining ~3h budget --
    # it plateaued at 6.99T across several generations under the old
    # (longer) budget, and time is short (user wants a real shot at >10T
    # by morning). Starting straight from its own best design so stage 7
    # (20 layers) launches immediately instead of waiting out more
    # diminishing-returns time at 16.
    current_design = dict(
        a=31.130821333791943e-3, b=40.01565009808927e-3,
        coil_half_gap=33.97411042487954e-3,
        n_turns=[550, 550, 584, 584, 17, 17, 507, 507, 507, 507, 613, 613,
                 629, 629, 605, 605])
    current_layers = 16
    stage_idx = 7  # stage 6 (16L) already explored, see comment above

    while current_layers < MAX_LAYERS:
        if _stop_requested:
            log("[stop requested] ending orchestration before next stage.")
            break

        design, new_layers = next_design(current_design, current_layers)
        name = f"{stage_idx}_{new_layers}layer_escalation"
        out_dir = os.path.join(BASE_OUT, f"stage{stage_idx}_{new_layers}layer")

        b, ok, row, best_feas = run_stage(name, out_dir, STAGE_BUDGET_S, design,
                                          max_evals=MAX_EVALS_PER_STAGE)

        candidate_b = None
        if best_feas is not None:
            try:
                candidate_b = float(best_feas.get("B_target_T", "") or "nan")
            except ValueError:
                candidate_b = None

        if candidate_b is not None and candidate_b >= B_TARGET_GOAL:
            log(f"*** GOAL MET (raw field) at stage {name}: "
                f"B_target_T={candidate_b} -- see whether "
                f"all_constraints_ok in the row above for full-pass status. "
                f"Stopping orchestration. ***")
            break

        if b is not None and ok and b >= B_TARGET_GOAL:
            log(f"*** GOAL MET (all constraints) at stage {name}: "
                f"B_target_T={b}. Stopping orchestration. ***")
            break

        # Warm-start the NEXT stage from whatever this stage's best
        # feasible design was (even if not all_constraints_ok -- 10T is a
        # hard floor most stages below the true crossing won't clear).
        # If NOTHING feasible was found at all this stage (shouldn't
        # normally happen), keep the same base design so escalation still
        # proceeds rather than getting stuck.
        if best_feas is not None:
            current_design = dict(
                a=float(best_feas["a_mm"]) / 1e3,
                b=float(best_feas["b_mm"]) / 1e3,
                coil_half_gap=float(best_feas["gap_mm"]) / 1e3,
                n_turns=ast.literal_eval(best_feas["n_turns"]))
        else:
            current_design = design
        current_layers = new_layers
        stage_idx += 1

    else:
        log(f"Reached MAX_LAYERS={MAX_LAYERS} without hitting the 10T goal.")

    log("#" * 90)
    log("v2 ORCHESTRATOR DONE")
    log("#" * 90)


if __name__ == "__main__":
    main()
