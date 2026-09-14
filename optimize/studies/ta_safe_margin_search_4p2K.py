"""
ta_safe_margin_search_4p2K.py — CMA-ES search using the T-A-resolved-safe
operating current, under the 4.2K temperature-scaled Ic/n models
=============================================================================
2026-09-13. Sibling of `ta_safe_margin_search.py` (kept separate, not
modified in place, so that script's own 20K-based run history/checkpoints
for 6/8/10-layer stages stay intact and resumable). Same architecture,
same `ta_safe_current.evaluate()` fitness function, same wide-bounds
philosophy, same worker-pool crash/hang resilience and checkpointing --
the differences are exactly the two things needed to make a 4.2K-based
search trustworthy, both established this session:

1. **Fixed relaxation pair.** The 4.2K-scaled n(B,theta) model pushes n to
   ~50-60, well outside the n=13-34 range params.py's committed default
   `ta_picard_alpha=(0.30,0.15)` was tuned/validated for -- confirmed
   (`optimize/studies/ta_thermal_validation.py`, same day) to give a raw
   |ΔB|/|B| residual around 1.7-2.0 (never converging) at that default,
   vs. 0.01-0.05 at `alpha=(0.03,0.01)` -- the SAME pair that fixed the
   `transient/` short-dt problem. Forced in-memory below, exactly like
   the mesh-tier override already established in the sibling script.
2. **Matched Ic+n models, forced full-length solves.** Ic-only scaling
   (leaving n at its 20K measured value) was found to give a physically
   inconsistent, WORSE-than-20K margin (mixing a boosted Jc with an
   unmatched n is not a preview of real 4.2K behaviour -- see CLAUDE.md's
   "Cryogenic (4.2K)..." section). `ta_safe_current.evaluate()` gained
   `n_model=`/`min_iters=` kwargs this session specifically so a caller
   can pass the MATCHED n-model and force every solve to the full
   `params.ta_n_picard` iterations (the EMA-smoothed-SCIF stall flag was
   found to never fire True in this regime within 150 iterations even
   when the raw residual is already good -- forcing full length costs
   nothing extra here and removes a systematic false-stall risk for
   candidate geometries never directly tested).

CONTEXT for why this search exists at all: a same-day user challenge
("real magnets reach 10T+, what are THEY doing differently if not
temperature?") led to checking the champion's own reference [30] in the
Tsuchiya paper -- "a large 20-pancake prototype reaching up to 25T" --
vs. this project's 6 layers (3 pancake pairs). Prior 20K-model evidence
(this project's own `ta_safe_margin_8layer`/`_10layer` runs, both killed
mid-flight by a harness SIGTERM, not by design) found 8 layers reaching
5.03T and 10 layers reaching 4.43T (fewer evals, less mature) -- better
than 6 layers' 4.10T ceiling, but still far below 10T. Tonight's task
(per explicit user direction): try the 4.2K-improved model at the
CURRENT 6-layer topology first, with new starting points / wider bounds
if the champion-neighborhood search stalls; only escalate to more layers
if that doesn't work. See `run_4p2K_overnight.sh` for the actual staged
orchestration built on top of this script.

Env vars (all identical in meaning to the sibling script):
    CMAES_X0_JSON_OVERRIDE   (opt_config.py) -- sets N_LAYERS/N_PAIRS via
                             a full a/b/coil_half_gap/n_turns dict; MUST
                             be set before this script is even launched
                             (read at opt_config.py import time)
    TA_SAFE_X0_OVERRIDE_JSON -- this script's own CMA-ES starting point
                             (a/b/gap/n_turns, same n_turns length as
                             above)
    TA_SAFE_A_STD0_M / TA_SAFE_B_STD0_M / TA_SAFE_GAP_STD0_M / TA_SAFE_N_STD0
    TA_SAFE_N_MIN / TA_SAFE_PAIR2_MAX
    TA_SAFE_MAX_EVALS / TA_SAFE_N_WORKERS / TA_SAFE_SEED / TA_SAFE_POPSIZE
    TA_SAFE_OUT_DIR          -- defaults to runs/ta_safe_margin_4p2K here
                             (NOT the sibling script's runs/ta_safe_margin)
    TA_SAFE_IC_EXTRAP        -- base 20K Ic extrapolation before the 4.2K
                             ratio is applied (default "kim")

Run (background, direct binary -- see CLAUDE.md's "Operational lessons"):
    PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
    nohup $PY optimize/studies/ta_safe_margin_search_4p2K.py \
        > $TA_SAFE_OUT_DIR/log.txt 2>&1 &
"""
import os
import sys
import csv
import time

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_OPT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_OPT)
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "mesh"),
           os.path.join(_ROOT, "solve"), os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import cma
from mpi4py import MPI

import params

# ── force medium mesh tier in-memory (params.py on disk is still the
# 2026-09-03 xdense one-off tier -- see CLAUDE.md's "Mesh resolution") ──
params.mesh_size_min_factor = 0.25
params.mesh_size_max_factor = 0.60
params.mesh_dist_min_factor = 1.50
params.mesh_dist_max_factor = 2.00
params.box_scale = 4.0
params.mesh_z_grading = [0.075, 0.15, 0.55, 0.15, 0.075]

# ── force the relaxation pair validated for the 4.2K model's n~50-60
# regime (see module docstring point 1) -- in-memory, applies in the main
# process AND every spawned worker (this module is re-imported fresh in
# each child) ─────────────────────────────────────────────────────────
params.ta_picard_alpha = 0.03
params.ta_picard_alpha_fine = 0.01

import opt_config as cfg
import ta_safe_current
from cmaes_search import (decode, encode_x0, geometry_violation, N_PAIRS,
                          N_LAYERS)
from ic_extrapolation import make_ic_model
from ic_model import NValueModel
from ic_temperature_scaling import (Fujikura4p2KScaledIcModel,
                                     Fujikura4p2KScaledNValueModel)

OUT_DIR = os.environ.get("TA_SAFE_OUT_DIR",
                         os.path.join(_OPT, "runs", "ta_safe_margin_4p2K"))
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "results.csv")
OUT_LOG = os.path.join(OUT_DIR, "history.csv")
CHECKPOINT_PATH = os.path.join(OUT_DIR, "cma_checkpoint.pkl")

A_STD0 = float(os.environ.get("TA_SAFE_A_STD0_M", 0.005))
B_STD0 = float(os.environ.get("TA_SAFE_B_STD0_M", 0.008))
GAP_STD0 = float(os.environ.get("TA_SAFE_GAP_STD0_M", 0.0025))
N_STD0 = float(os.environ.get("TA_SAFE_N_STD0", 50.0))

N_BOUNDS_WIDE = (int(os.environ.get("TA_SAFE_N_MIN", 1)), 3000)
GAP_RANGE_WIDE_M = float(os.environ.get("TA_SAFE_GAP_RANGE_WIDE_M", 0.060))

MAX_EVALS = int(os.environ.get("TA_SAFE_MAX_EVALS", 700))
N_WORKERS = int(os.environ.get("TA_SAFE_N_WORKERS", 4))
SEED = int(os.environ.get("TA_SAFE_SEED", 90902))
FLUSH_EVERY = 1

_ic_model = None
_n_model = None
_comm = None
_eval_count = 0
_history = []
_best = dict(fitness=np.inf)
_run_tag = "ta_safe_4p2K"
_last_flushed_idx = 0

HISTORY_FIELDNAMES = [
    "run_tag", "eval", "fitness", "feasible", "all_constraints_ok",
    "a_mm", "b_mm", "gap_mm", "face_gap_mm", "n_turns", "n_total",
    "tape_km", "B_target_T", "uniformity_pct", "hoop_MPa",
    "I_quench_uniform_A", "I_op_A", "ta_worst_margin",
    "ta_constraint_margin", "bisection_exhausted", "n_ta_solves", "eval_s",
]


def bounds_and_stds_wide():
    n_bounds = N_BOUNDS_WIDE
    gap_lo = N_LAYERS * params.w / 2.0 + cfg.MIN_COIL_GAP_M / 2.0
    gap_hi = gap_lo + GAP_RANGE_WIDE_M
    lo = [cfg.CMAES_A_BOUNDS[0], cfg.CMAES_B_BOUNDS[0],
          gap_lo] + [n_bounds[0]] * N_PAIRS
    n_hi = [n_bounds[1]] * N_PAIRS
    pair2_max = os.environ.get("TA_SAFE_PAIR2_MAX")
    if pair2_max is not None and N_PAIRS >= 1:
        n_hi[-1] = float(pair2_max)
    hi = [cfg.CMAES_A_BOUNDS[1], cfg.CMAES_B_BOUNDS[1], gap_hi] + n_hi
    stds = [A_STD0, B_STD0, GAP_STD0] + [N_STD0] * N_PAIRS
    return lo, hi, stds


def _evaluate_candidate(x):
    a, b, gap, n_turns = decode(x)
    viol, face_gap = geometry_violation(a, b, gap, n_turns)
    if viol > 0.0:
        f = cfg.CMAES_INFEASIBLE_PENALTY_KM * (1.0 + viol)
        return dict(a=a, b=b, gap=gap, n_turns=n_turns, face_gap=face_gap,
                    f=f, feasible=False, r=None, g=None)

    cand = dict(a=a, b=b, coil_half_gap=gap, n_turns=n_turns)
    try:
        r = ta_safe_current.evaluate(cand, _ic_model, _comm,
                                     n_model=_n_model,
                                     min_iters=params.ta_n_picard)
    except Exception as e:
        print(f"  [eval error] {type(e).__name__}: {e}", flush=True)
        f = cfg.CMAES_INFEASIBLE_PENALTY_KM
        return dict(a=a, b=b, gap=gap, n_turns=n_turns, face_gap=face_gap,
                    f=f, feasible=False, r=None, g=None)

    if not r.get("feasible", False):
        f = cfg.CMAES_INFEASIBLE_PENALTY_KM
        return dict(a=a, b=b, gap=gap, n_turns=n_turns, face_gap=face_gap,
                    f=f, feasible=False, r=None, g=None)

    hoop_max_mpa = cfg.SIGMA_HOOP_MAX_PA / 1e6
    g_field = max(0.0, (cfg.B_TARGET_MIN_T - r["B_target_T"]) / cfg.B_TARGET_MIN_T)
    g_hoop = max(0.0, (r["hoop_MPa"] - hoop_max_mpa) / hoop_max_mpa)
    g_unif = max(0.0, (r["uniformity_pct"] - cfg.UNIFORMITY_MAX_PCT) / cfg.UNIFORMITY_MAX_PCT)
    g = [g_field, g_hoop, g_unif]
    penalty = (cfg.CMAES_PENALTY_KM_FIELD * g_field ** 2
              + cfg.CMAES_PENALTY_KM_HOOP * g_hoop ** 2
              + cfg.CMAES_PENALTY_KM_UNIFORMITY * g_unif ** 2)
    f = r["tape_km"] + penalty
    return dict(a=a, b=b, gap=gap, n_turns=n_turns, face_gap=face_gap,
                f=f, feasible=True, r=r, g=g)


def _append_master_log():
    global _last_flushed_idx
    new_rows = _history[_last_flushed_idx:]
    if not new_rows:
        return
    write_header = not os.path.exists(OUT_LOG)
    with open(OUT_LOG, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in new_rows:
            writer.writerow(row)
    _last_flushed_idx = len(_history)


def _write_best_csv():
    if _best.get("fitness", np.inf) >= np.inf:
        return
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_FIELDNAMES)
        writer.writeheader()
        writer.writerow(_best)


def _record(x, a, b, gap, n_turns, face_gap, f, feasible, r, g=None):
    global _eval_count
    _eval_count += 1
    all_pass = feasible and all(gi <= 1e-9 for gi in (g or [1, 1, 1]))
    row = dict.fromkeys(HISTORY_FIELDNAMES, "")
    row.update(run_tag=_run_tag, eval=_eval_count, fitness=f,
              feasible=feasible, all_constraints_ok=all_pass,
              a_mm=a * 1e3, b_mm=b * 1e3, gap_mm=gap * 1e3,
              face_gap_mm=face_gap * 1e3,
              n_turns=str(n_turns), n_total=sum(n_turns))
    if r is not None:
        row.update(tape_km=r["tape_km"], B_target_T=r["B_target_T"],
                  uniformity_pct=r["uniformity_pct"],
                  hoop_MPa=r["hoop_MPa"],
                  I_quench_uniform_A=r.get("I_quench_uniform_A"),
                  I_op_A=r["I_op_A"], ta_worst_margin=r["ta_worst_margin"],
                  ta_constraint_margin=r.get("ta_constraint_margin"),
                  bisection_exhausted=r.get("bisection_exhausted"),
                  n_ta_solves=r["n_ta_solves"], eval_s=r["eval_s"])
    _history.append(row)

    if all_pass and f < _best["fitness"]:
        _best.clear()
        _best.update(row)
        _best["fitness"] = f

    tag = "OK " if all_pass else ("INFEAS" if not feasible else "viol")
    extra = (f"tape={r['tape_km']:.3f}km B={r['B_target_T']:.2f}T "
             f"unif={r['uniformity_pct']:.2f}% hoop={r['hoop_MPa']:.0f}MPa "
             f"I_op={r['I_op_A']:.1f}A "
             f"margin(p{int(ta_safe_current.MARGIN_PERCENTILE)})="
             f"{r.get('ta_constraint_margin', float('nan')):.3f} "
             f"worst={r['ta_worst_margin']:.3f} "
             f"{'[BISECT_EXHAUSTED]' if r.get('bisection_exhausted') else ''} "
             f"({r['n_ta_solves']} TA solves, {r['eval_s']:.0f}s)"
             if r is not None else "no evaluation (geometry pre-check)")
    print(f"  [{_eval_count:4d}] {tag:6s} f={f:8.3f}  "
         f"a={a*1e3:.2f} b={b*1e3:.2f} gap={gap*1e3:.2f} "
         f"(face={face_gap*1e3:.2f}mm) n={n_turns}  {extra}", flush=True)

    if _eval_count % FLUSH_EVERY == 0:
        _append_master_log()
        _write_best_csv()


def _worker_init():
    global _ic_model, _n_model, _comm
    _comm = MPI.COMM_WORLD
    base_ic = make_ic_model(os.environ.get("TA_SAFE_IC_EXTRAP", "kim"))
    base_n = NValueModel(csv_path=params.n_value_csv_filename)
    _ic_model = Fujikura4p2KScaledIcModel(base_ic)
    _n_model = Fujikura4p2KScaledNValueModel(base_n)


_stop_requested = False


def _request_stop(signum, frame):
    global _stop_requested
    _stop_requested = True
    print("\n[stop requested] finishing current step, will checkpoint and "
         "exit before starting a new generation ...", flush=True)


def _save_checkpoint(es):
    import pickle
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump({"es": es, "run_tag": _run_tag}, fh)
    os.replace(tmp, CHECKPOINT_PATH)


def _load_checkpoint():
    import pickle
    if not os.path.exists(CHECKPOINT_PATH):
        return None, None
    try:
        with open(CHECKPOINT_PATH, "rb") as fh:
            data = pickle.load(fh)
        return data["es"], data["run_tag"]
    except Exception as e:
        print(f"[checkpoint] found {CHECKPOINT_PATH} but failed to load "
             f"({type(e).__name__}: {e}) -- starting fresh instead.",
             flush=True)
        return None, None


def _restore_eval_count():
    if not os.path.exists(OUT_LOG):
        return 0
    with open(OUT_LOG, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return int(rows[-1]["eval"]) if rows else 0


def _restore_best():
    if not os.path.exists(OUT_CSV):
        return dict(fitness=np.inf)
    with open(OUT_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return dict(fitness=np.inf)
    row = rows[0]
    row["fitness"] = float(row["fitness"])
    return row


def _run_parallel(es, n_workers):
    import multiprocessing as mp
    import concurrent.futures as cf

    # 2026-09-13: forced full-length (min_iters=params.ta_n_picard=150)
    # solves at the retuned alpha, plus up to MAX_BISECT=12 bisection
    # steps (raised from the old MAX_BISECT=5 this same session -- see
    # ta_safe_current.py), mean a single candidate can legitimately need
    # up to ~14 T-A solves x 150 forced iterations each. Measured
    # 2026-09-12/13 at medium tier: ~66-90s/solve for a modest-size
    # candidate -- worst case ~14*90s=~21min, more for a larger mesh.
    # GEN_TIMEOUT_S must cover that legitimate worst case, not just the
    # sibling (20K, old MAX_BISECT=5, early-exit-eligible) script's 900s.
    GEN_TIMEOUT_S = int(os.environ.get("TA_SAFE_GEN_TIMEOUT_S", 2700))

    def _kill_and_replace_pool(pool, reason):
        print(f"  [GENERATION FAILED] {reason} Force-killing worker "
             f"processes and recreating the pool; this generation's "
             f"candidates are not recorded.", flush=True)
        try:
            for p in list(getattr(pool, "_processes", {}).values()):
                try:
                    p.kill()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        return cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                      initializer=_worker_init)

    ctx = mp.get_context("spawn")
    pool = cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                  initializer=_worker_init)
    try:
        while not es.stop():
            if _stop_requested:
                print("[stop requested] exiting before next generation "
                     "(checkpoint already reflects the last completed one).",
                     flush=True)
                break
            solutions = es.ask()
            futures = [pool.submit(_evaluate_candidate, x) for x in solutions]
            t0 = time.time()
            results = []
            failure = None
            for fut in futures:
                remaining = GEN_TIMEOUT_S - (time.time() - t0)
                try:
                    results.append(fut.result(timeout=max(1.0, remaining)))
                except cf.TimeoutError:
                    failure = (f"A worker did not return within "
                              f"{GEN_TIMEOUT_S}s (likely a non-converging "
                              f"solve, not a crash).")
                    break
                except Exception as e:
                    failure = (f"{type(e).__name__}: {e} -- a worker "
                              f"process almost certainly crashed natively "
                              f"(SEGV/MPI_Abort in PETSc/MUMPS, not a "
                              f"Python exception).")
                    break
            if failure is not None:
                pool = _kill_and_replace_pool(pool, failure)
                continue
            fitnesses = []
            for x, res in zip(solutions, results):
                _record(x, res["a"], res["b"], res["gap"], res["n_turns"],
                        res["face_gap"], res["f"], res["feasible"],
                        res["r"], res.get("g"))
                fitnesses.append(res["f"])
            es.tell(solutions, fitnesses)
            _save_checkpoint(es)
    finally:
        try:
            for p in list(getattr(pool, "_processes", {}).values()):
                try:
                    p.kill()
                except Exception:
                    pass
        except Exception:
            pass
        pool.shutdown(wait=False, cancel_futures=True)


def _encode_x0_override():
    raw = os.environ.get("TA_SAFE_X0_OVERRIDE_JSON")
    if not raw:
        return None
    import json
    d = json.loads(raw)
    full = d["n_turns"]
    assert len(full) == 2 * N_PAIRS, (
        f"TA_SAFE_X0_OVERRIDE_JSON's n_turns has {len(full)} entries but "
        f"N_PAIRS={N_PAIRS} (N_LAYERS={N_LAYERS}) was computed from "
        f"opt_config.CMAES_X0 -- set CMAES_X0_JSON_OVERRIDE too (same "
        f"n_turns, note its gap key is 'coil_half_gap' not 'gap') so "
        f"N_LAYERS is established BEFORE cmaes_search.py's import-time "
        f"computation, or this override silently uses the wrong layer count.")
    pairs = [(full[2 * i] + full[2 * i + 1]) / 2.0 for i in range(N_PAIRS)]
    return np.array([d["a"], d["b"], d["gap"], *pairs], dtype=float)


def main():
    global _run_tag, _eval_count, _best

    import signal
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    print("=" * 90)
    print("T-A-safe-current CMA-ES search UNDER THE 4.2K MODEL -- "
         "optimize/studies/ta_safe_margin_search_4p2K.py")
    print(f"alpha=(ta_picard_alpha={params.ta_picard_alpha}, "
         f"ta_picard_alpha_fine={params.ta_picard_alpha_fine})  "
         f"min_iters=ta_n_picard={params.ta_n_picard} (forced full-length)")
    print("=" * 90)

    es, loaded_run_tag = _load_checkpoint()
    if es is not None:
        expected_dim = 3 + N_PAIRS
        assert es.N == expected_dim, (
            f"Checkpoint at {CHECKPOINT_PATH} is {es.N}-dimensional but "
            f"this process's N_PAIRS={N_PAIRS} (N_LAYERS={N_LAYERS}) implies "
            f"{expected_dim} -- CMAES_X0_JSON_OVERRIDE must be set to the "
            f"SAME n_turns length used when this checkpoint was created, "
            f"on every resume, not just the first launch.")
        _run_tag = loaded_run_tag
        _eval_count = _restore_eval_count()
        _best = _restore_best()
        print(f"[checkpoint] RESUMED from {CHECKPOINT_PATH}")
        if _best.get("fitness", np.inf) < np.inf:
            print(f"  best-so-far (from results.csv): fitness="
                 f"{_best['fitness']:.3f} B_target_T={_best.get('B_target_T')} "
                 f"n_turns={_best.get('n_turns')}")
        print(f"  run_tag={_run_tag}  eval_count={_eval_count}  "
             f"generations completed={es.countiter}  "
             f"evals CMA-ES has counted={es.countevals}")
        print(f"  To force a fresh start instead, remove this file first.\n",
             flush=True)
    else:
        _run_tag = os.environ.get(
            "TA_SAFE_RUN_TAG", f"ta_safe_4p2K_{time.strftime('%Y%m%d_%H%M%S')}")
        x0 = _encode_x0_override()
        if x0 is None:
            x0 = encode_x0()
        lo, hi, stds = bounds_and_stds_wide()

        opts = {
            "bounds": [lo, hi],
            "CMA_stds": stds,
            "seed": SEED,
            "maxfevals": MAX_EVALS,
            "verbose": -3,
        }
        _popsize = os.environ.get("TA_SAFE_POPSIZE")
        if _popsize:
            opts["popsize"] = int(_popsize)

        print(f"x0: a={x0[0]*1e3:.2f}mm b={x0[1]*1e3:.2f}mm "
             f"gap={x0[2]*1e3:.2f}mm pairs={list(x0[3:])}")
        print(f"step sizes: a={A_STD0*1e3:.2f}mm b={B_STD0*1e3:.2f}mm "
             f"gap={GAP_STD0*1e3:.2f}mm n={N_STD0:.0f} turns")
        print(f"popsize={opts.get('popsize', 'default')}")
        es = cma.CMAEvolutionStrategy(x0, 1.0, opts)

    print(f"N_LAYERS={N_LAYERS}  N_PAIRS={N_PAIRS}")
    print(f"budget: {MAX_EVALS} evaluations, {N_WORKERS} parallel workers")
    _pctile = ta_safe_current.MARGIN_PERCENTILE
    _cell_desc = ("EVERY cell" if _pctile <= 0
                 else f"{100 - _pctile:.0f}% of cells (p{_pctile:.0f})")
    print(f"target: B >= {cfg.B_TARGET_MIN_T} T AND T-A local margin "
         f">= {ta_safe_current.MARGIN_REQUIRED:.4f} (J/Jc <= 0.65) for "
         f"{_cell_desc}, hoop <= {cfg.SIGMA_HOOP_MAX_PA/1e6:.0f} MPa, "
         f"uniformity <= {cfg.UNIFORMITY_MAX_PCT}%")
    print(f"outputs: {OUT_CSV}, {OUT_LOG}")
    print(f"checkpoint: {CHECKPOINT_PATH}\n", flush=True)
    t0 = time.time()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    _run_parallel(es, N_WORKERS)

    dt = time.time() - t0
    _append_master_log()
    _write_best_csv()

    print(f"\n{'='*90}\nFinished: {_eval_count} evaluations in {dt/60:.1f} min\n{'='*90}")
    try:
        print(f"CMA-ES stop reason(s): {es.stop()}")
    except Exception as _e:
        print(f"(could not read es.stop(): {_e})")
    if _best.get("fitness", np.inf) < np.inf:
        print("\nBest feasible (field+hoop-satisfying) T-A-safe design found:")
        for k in ("a_mm", "b_mm", "gap_mm", "face_gap_mm", "n_turns",
                  "n_total", "tape_km", "B_target_T", "uniformity_pct",
                  "hoop_MPa", "I_op_A", "ta_worst_margin", "n_ta_solves"):
            if k in _best:
                print(f"  {k:18s} {_best[k]}")
    else:
        print("\nNo evaluation satisfied both B_target and hoop within budget "
             "-- see history.csv for the closest candidates (smallest "
             "fitness among feasible=True rows).")


if __name__ == "__main__":
    main()
