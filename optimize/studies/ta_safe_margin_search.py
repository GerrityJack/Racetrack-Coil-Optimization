"""
ta_safe_margin_search.py — CMA-ES search using the T-A-resolved-safe
operating current (optimize/ta_safe_current.py), not the uniform-J one
========================================================================
2026-09-02: this is the search this project's own history flagged as the
correct fix for known-open-issue #6 (CLAUDE.md, "Ramp-up power analysis")
-- the champion's 65%-of-Ic margin was only ever checked under a
uniform-J approximation. `ta_safe_current.evaluate()` replaces
`optimize_geometry.evaluate()`'s I_op with the largest current at which a
genuine T-A solve's worst-cell local margin stays >= 1/0.65 = 1.5385
EVERYWHERE, then reports B_target_T at THAT current -- so a candidate
that needs a big current derating to be locally safe shows up directly
as a field shortfall in the fitness function, exactly like any other
infeasible candidate.

WHY THIS IS A SEPARATE SCRIPT, NOT A FLAG ON cmaes_search.py: each
evaluation here is a full mesh build + several T-A Picard solves (a cold
solve plus up to ~6 bisection/confirmation solves, warm-started against
each other) -- measured at ~161s for one candidate in a direct smoke test
(2026-09-02), not the ~5-10s of the uniform-J screen. That is ~20x more
expensive per evaluation.

2026-09-02, SAME-DAY REVISION: this was originally scoped as a small
LOCAL POLISH around the champion (matching optimize/runs/local_polish/'s
pattern), on the assumption the champion was already close to a fix. A
direct smoke test of ta_safe_current.evaluate() on the champion's own
EXACT geometry found otherwise: the highest current at which every cell's
T-A-resolved local margin holds >= 1.5385 (J/Jc <= 0.65) is ~17.3A --
giving B_target_T ~= 0.92T, not 10T. That is an ~11x field gap, not the
~2x-ish gap the champion's own uniform-J-vs-T-A margin numbers (0.709 vs
1.5385, i.e. ~2.2x tighter) might suggest -- a small percent-level
geometric perturbation cannot plausibly close an 11x gap. Per explicit
user direction, this is now instead a WIDE search: much larger step
sizes (A_STD0/B_STD0/GAP_STD0/N_STD0 below) and the FULL bound range
(cfg.CMAES_N_BOUNDS / cfg.CMAES_HALF_GAP_BOUNDS, NOT the tight
champion-neighborhood bounds cfg.CMAES_TIGHT_BOUNDS would otherwise
apply), still warm-started at the champion (a real, working, manufacturable
baseline) but free to move far from it if the fitness landscape wants to.

HONEST EXPECTATION, stated up front: whether ANY reachable geometry in
this search's bounds gets to 10T while keeping every cell's local margin
safe is genuinely unknown -- this project's own "Proxy graveyard" section
found repeatedly that nothing about T-A's real behavior can be assumed
from a cheap proxy or from a nearby design's behavior. Given the
per-evaluation cost above, this search's evaluation budget (MAX_EVALS
below) is still thin for a genuinely wide 6-dimensional CMA-ES run (a
proper from-scratch search elsewhere in this project's history used
tens of thousands of evaluations). This run may find a genuinely better
design, or it may run out of budget without closing the gap -- BOTH are
useful, honest outcomes; do not treat "the search did not find a fix" as
this script failing to run correctly.

Uses the SAME separated field/hoop penalty factors as cmaes_search.py
(CMAES_PENALTY_KM_FIELD, CMAES_PENALTY_KM_HOOP, opt_config.py) plus its
own geometry_violation() pre-filter, decode()/encode_x0() vector coding,
and N_PAIRS/N_LAYERS -- all imported directly from cmaes_search.py (pure
functions, no shared mutable state) rather than re-implemented.

Outputs (own paths -- never overwrites the main cmaes_search.py outputs):
    optimize/runs/ta_safe_margin/results.csv   (best design, overwritten each flush)
    optimize/runs/ta_safe_margin/history.csv   (every evaluation, this run)
    optimize/runs/ta_safe_margin/log.txt       (stdout, tee'd)

Run (background, direct binary -- NOT `conda run`, see CLAUDE.md's
"Operational lessons" on stdout buffering):
    PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
    nohup $PY optimize/studies/ta_safe_margin_search.py \
        > optimize/runs/ta_safe_margin/log.txt 2>&1 &
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
import opt_config as cfg
import ta_safe_current
from cmaes_search import (decode, encode_x0, geometry_violation, N_PAIRS,
                          N_LAYERS)
from ic_extrapolation import make_ic_model

# 2026-09-03: overridable so a second, concurrent instance (e.g. an
# 8-layer run alongside a 6-layer one) doesn't clobber the same
# log.txt/history.csv/results.csv -- see CLAUDE.md's "Operational
# lessons" on exactly this class of concurrent-output-path bug.
OUT_DIR = os.environ.get("TA_SAFE_OUT_DIR",
                         os.path.join(_OPT, "runs", "ta_safe_margin"))
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "results.csv")
OUT_LOG = os.path.join(OUT_DIR, "history.csv")

# ── search scope: WIDE, warm-started at the champion but free to roam ───────
# 2026-09-02 revision (see docstring): the champion's own T-A-safe field is
# ~0.92T, an ~11x gap to the 10T target -- closing that needs step sizes
# and bounds comparable to cmaes_search.py's own COLD-START defaults, not
# a tight local-polish neighborhood. x0 is still the champion (a real,
# working, manufacturable starting point), but sigma is now large enough
# for CMA-ES to actually leave that neighborhood.
# 2026-09-03: after 4 generations (0.92 -> 2.32 -> 3.25 -> 3.51T, each
# restart warm-started from the previous best), the design's SHAPE has
# converged consistently: a in the low-40s mm, b in the 50s mm, gap
# pinned near its floor, and a two-heavy/one-thin turn-pair pattern
# scaled up from the champion's own shape. Gen4 plateaued at 2.82T for 83
# straight evaluations without re-surpassing gen3's 3.51T peak -- a real
# stall, not just restart catch-up. Halving these step sizes trades some
# exploration range for faster re-convergence/refinement within the
# now-well-established basin, overridable per-restart via env vars
# without another code edit.
A_STD0 = float(os.environ.get("TA_SAFE_A_STD0_M", 0.005))     # 5 mm (was 10mm)
B_STD0 = float(os.environ.get("TA_SAFE_B_STD0_M", 0.008))     # 8 mm (was 15mm)
GAP_STD0 = float(os.environ.get("TA_SAFE_GAP_STD0_M", 0.0025))  # 2.5mm (was 5mm)
N_STD0 = float(os.environ.get("TA_SAFE_N_STD0", 50.0))          # turns (was 100)

# 2026-09-02: opt_config.py's CMAES_N_BOUNDS=(1,900) upper and
# CMAES_HALF_GAP_BOUNDS's range width (45-15.5=29.5mm) have NO physical
# derivation in this codebase -- unlike the bend-radius/face-gap/hoop
# floors (real material/manufacturing limits, left untouched below),
# these ceilings just happened to be generous enough for every
# champion-neighborhood search this project has run so far. This search
# is explicitly hunting for "however big it needs to be" to close an ~11x
# field gap, so raise them well past any value seen in gen-1 testing
# (turns up to 494/pair, a/b up to 36/51mm) rather than risk silently
# capping the one direction that might actually close the gap.
N_BOUNDS_WIDE = (int(os.environ.get("TA_SAFE_N_MIN", 1)), 3000)  # vs. opt_config's (1, 900)
GAP_RANGE_WIDE_M = 0.060           # vs. opt_config's implied 29.5mm range
# 2026-09-03: TA_SAFE_N_MIN lets a restart forbid the "collapse one pair
# to near-nothing" pattern CMA-ES keeps converging on (49, 60, 39, 68
# turns seen across generations tonight) -- per-cell worst-margin location
# analysis on the all-time-best design found that thin pair running hot
# across its ENTIRE radial width (not just an edge -- it's too thin to
# have any self-shielding bulk), a real contributor to I_op's ceiling
# that nothing in the fitness function currently discourages (I_op only
# enters the objective indirectly, through the bisected B_target).

MAX_EVALS = int(os.environ.get("TA_SAFE_MAX_EVALS", 700))
N_WORKERS = int(os.environ.get("TA_SAFE_N_WORKERS", 4))
SEED = int(os.environ.get("TA_SAFE_SEED", 90902))
FLUSH_EVERY = 1   # every evaluation -- each one is minutes of compute,
                  # losing even one to an unattended-run crash is wasteful

_ic_model = None
_comm = None
_eval_count = 0
_history = []
_best = dict(fitness=np.inf)
_run_tag = "ta_safe"
_last_flushed_idx = 0

# Fixed column set for history.csv / results.csv -- see _record()'s
# 2026-09-02 bugfix comment for why this must be fixed, not inferred.
HISTORY_FIELDNAMES = [
    "run_tag", "eval", "fitness", "feasible", "all_constraints_ok",
    "a_mm", "b_mm", "gap_mm", "face_gap_mm", "n_turns", "n_total",
    "tape_km", "B_target_T", "uniformity_pct", "hoop_MPa",
    "I_quench_uniform_A", "I_op_A", "ta_worst_margin",
    "ta_constraint_margin", "n_ta_solves", "eval_s",
]


def bounds_and_stds_wide():
    """Uses this script's own N_BOUNDS_WIDE/GAP_RANGE_WIDE_M (wider than
    EITHER of opt_config.py's tight-or-not CMAES_N_BOUNDS/
    CMAES_HALF_GAP_BOUNDS -- see the comment above those constants) so a
    candidate genuinely needing more turns or more axial room than any
    prior champion-neighborhood search ever tried isn't silently
    excluded. Physical floors (bend radius, face gap, hoop stress) are
    untouched -- those come from geometry_violation()/stress_screen(),
    not from these search-range bounds, and stay enforced exactly as
    before regardless of how wide this range is."""
    n_bounds = N_BOUNDS_WIDE
    gap_lo = N_LAYERS * params.w / 2.0 + cfg.MIN_COIL_GAP_M / 2.0
    gap_hi = gap_lo + GAP_RANGE_WIDE_M
    lo = [cfg.CMAES_A_BOUNDS[0], cfg.CMAES_B_BOUNDS[0],
          gap_lo] + [n_bounds[0]] * N_PAIRS
    n_hi = [n_bounds[1]] * N_PAIRS
    # 2026-09-03: quench-location analysis on gen12's best (turn-floor
    # fixed) found the worst-cell problem didn't go away, it MOVED --
    # 23/25 worst cells are now in layers 4/5 (pair index N_PAIRS-1 by
    # decode()'s fixed layer-pair mapping: pair0->layers0,1, pair1->
    # layers2,3, pair2->layers4,5), the single most axially-exposed
    # position (farthest from the coil's own mirror plane), clustered at
    # the OUTER radial edge at surprisingly LOW field (0.6-1.3T) -- same
    # angle-driven mechanism as the champion's outer-edge hot spot, just
    # now dominant. Unlike the thin-layer fix, this is about WHICH pair
    # carries the load, not how thin it's allowed to be -- cap that
    # specific pair's turn ceiling so CMA-ES is forced to put more of the
    # budget in the safer, mirror-plane-adjacent pairs instead.
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
        r = ta_safe_current.evaluate(cand, _ic_model, _comm)
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
    g_hoop  = max(0.0, (r["hoop_MPa"] - hoop_max_mpa) / hoop_max_mpa)
    # 2026-09-03: uniformity_pct here is ta_safe_current.py's REAL T-A box
    # calculation (uses the actual solved screening-current state at
    # I_op) -- not one of the cheap proxies CLAUDE.md's "Proxy graveyard"
    # falsified -- so folding it into the fitness isn't repeating that
    # mistake. Started at a low coefficient (CMAES_PENALTY_KM_UNIFORMITY)
    # per user direction: field is the urgent constraint right now.
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
    # Fixed, complete field set on EVERY row (blank for whatever an
    # infeasible/pre-check-failed row doesn't have) -- 2026-09-02 bugfix:
    # previously this dict only grew when r was not None, so feasible
    # rows had ~19 keys and infeasible rows had ~11, and
    # _append_master_log()'s DictWriter derived its fieldnames from
    # whichever row happened to flush first, producing a CSV with
    # inconsistent row widths (unparseable by pandas: "Expected 11
    # fields... saw 19"). Every row now has the same HISTORY_FIELDNAMES.
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
             f"({r['n_ta_solves']} TA solves, {r['eval_s']:.0f}s)"
             if r is not None else "no evaluation (geometry pre-check)")
    print(f"  [{_eval_count:4d}] {tag:6s} f={f:8.3f}  "
         f"a={a*1e3:.2f} b={b*1e3:.2f} gap={gap*1e3:.2f} "
         f"(face={face_gap*1e3:.2f}mm) n={n_turns}  {extra}", flush=True)

    if _eval_count % FLUSH_EVERY == 0:
        _append_master_log()
        _write_best_csv()


def _worker_init():
    """No cfg.SCREEN_MESH_OVERRIDES here -- unlike cmaes_search.py's
    worker init, this evaluator needs the FULL-fidelity production mesh
    (see ta_safe_current.py's docstring).

    2026-09-03: this used to also rename params.mesh_filename with a
    per-worker PID suffix here -- removed. ta_safe_current.py's own
    evaluate() already derives a fresh, unique (pid + timestamp) filename
    from a FIXED base on every call; this rename only fed that logic a
    moving target (see ta_safe_current.py's _BASE_MESH_FILENAME bugfix
    comment for what that combination actually did over a long-lived
    worker's lifetime)."""
    global _ic_model, _comm
    _comm = MPI.COMM_WORLD
    _ic_model = make_ic_model(os.environ.get("TA_SAFE_IC_EXTRAP", "kim"))


def _run_parallel(es, n_workers):
    """2026-09-03 resilience fix: as the wide search samples ever-larger
    candidates (more turns, bigger mesh), a worker can hit a genuine
    native-code crash in PETSc/MUMPS (confirmed: `Caught signal number 11
    SEGV` -- a real segfault, not a Python exception _evaluate_candidate's
    own try/except can catch, since a SEGV kills the OS process outright).
    That poisons the whole ProcessPoolExecutor (`BrokenProcessPool`), and
    an unguarded pool.map() call propagates that as an unhandled
    exception all the way out of main(), which is what actually ended a
    live overnight run several hours in. Now: a broken pool is caught,
    logged, and replaced with a fresh one: es.ask()'s solutions for that
    one generation are lost (not tell()-'d), but the run itself survives
    and continues to the next generation instead of dying outright."""
    import multiprocessing as mp
    import concurrent.futures as cf

    # 2026-09-03 hang-protection fix: found a live worker (a=52.6, b=72.7,
    # gap=13.7, nt=3562) stuck for 40+ minutes still burning CPU -- past
    # the KSP solve, presumably inside a non-converging/oscillating T-A
    # Picard loop that never hit either the stall-convergence criterion or
    # a hard iteration cap in reasonable time. This is DIFFERENT from the
    # SEGV case above: the worker isn't dead, it's just never coming back,
    # and pool.map()'s blocking iteration has no timeout -- the whole
    # overnight run silently stalled with no crash to catch. Submitting
    # per-task and bounding each generation's total wall time lets a
    # generation that blows the budget be detected and its (genuinely
    # hung, not just slow) workers force-killed and replaced, instead of
    # blocking forever.
    GEN_TIMEOUT_S = 900   # 15 min -- generous vs. the ~60-260s/candidate
                          # normally seen, but catches a true hang instead
                          # of waiting on it indefinitely

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
    """TA_SAFE_X0_OVERRIDE_JSON, if set, restarts from an explicit
    (a, b, coil_half_gap, n_turns) design instead of opt_config.py's
    champion -- e.g. the best design a prior run of this same script
    found, so a restart (to pick up a code fix or widen bounds further)
    doesn't throw away that progress. JSON: {"a":.., "b":.., "gap":..,
    "n_turns":[...]} (a/b/gap in metres, n_turns full per-layer list --
    same convention as CMAES_X0 in opt_config.py)."""
    raw = os.environ.get("TA_SAFE_X0_OVERRIDE_JSON")
    if not raw:
        return None
    import json
    d = json.loads(raw)
    full = d["n_turns"]
    pairs = [(full[2 * i] + full[2 * i + 1]) / 2.0 for i in range(N_PAIRS)]
    return np.array([d["a"], d["b"], d["gap"], *pairs], dtype=float)


def main():
    global _run_tag
    _run_tag = f"ta_safe_{time.strftime('%Y%m%d_%H%M%S')}"

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

    print("=" * 90)
    print("T-A-safe-current CMA-ES wide search -- optimize/studies/ta_safe_margin_search.py")
    print("=" * 90)
    print(f"x0: a={x0[0]*1e3:.2f}mm b={x0[1]*1e3:.2f}mm "
         f"gap={x0[2]*1e3:.2f}mm pairs={list(x0[3:])}")
    print(f"step sizes: a={A_STD0*1e3:.2f}mm b={B_STD0*1e3:.2f}mm "
         f"gap={GAP_STD0*1e3:.2f}mm n={N_STD0:.0f} turns")
    print(f"budget: {MAX_EVALS} evaluations, {N_WORKERS} parallel workers, "
         f"popsize={opts.get('popsize', 'default')}")
    _pctile = ta_safe_current.MARGIN_PERCENTILE
    _cell_desc = ("EVERY cell" if _pctile <= 0
                 else f"{100 - _pctile:.0f}% of cells (p{_pctile:.0f})")
    print(f"target: B >= {cfg.B_TARGET_MIN_T} T AND T-A local margin "
         f">= {ta_safe_current.MARGIN_REQUIRED:.4f} (J/Jc <= 0.65) for "
         f"{_cell_desc}, hoop <= {cfg.SIGMA_HOOP_MAX_PA/1e6:.0f} MPa, "
         f"uniformity <= {cfg.UNIFORMITY_MAX_PCT}%")
    print(f"penalty weights: field={cfg.CMAES_PENALTY_KM_FIELD:.0f} "
         f"hoop={cfg.CMAES_PENALTY_KM_HOOP:.0f} "
         f"uniformity={cfg.CMAES_PENALTY_KM_UNIFORMITY:.0f}")
    print(f"outputs: {OUT_CSV}, {OUT_LOG}\n", flush=True)

    es = cma.CMAEvolutionStrategy(x0, 1.0, opts)
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

        # ── save a reproducible field snapshot of the winner, so the
        # poster J/Jc cross-section figures (visualization/for poster/
        # make_jjc_cross_sections.py) can be regenerated for THIS design
        # without re-running the whole search -- see ta_safe_current.py's
        # save_fields_path docstring for exactly what it writes.
        import ast
        global _ic_model, _comm
        _comm = MPI.COMM_WORLD
        _ic_model = make_ic_model(os.environ.get("TA_SAFE_IC_EXTRAP", "kim"))
        best_design = dict(a=_best["a_mm"] / 1e3, b=_best["b_mm"] / 1e3,
                          coil_half_gap=_best["gap_mm"] / 1e3,
                          n_turns=ast.literal_eval(_best["n_turns"]))
        snap_path = os.path.join(OUT_DIR, "best_design_fields.npz")
        print(f"\nRe-evaluating winner once more to save {snap_path} ...")
        r = ta_safe_current.evaluate(best_design, _ic_model, _comm,
                                     verbose=True, save_fields_path=snap_path)
        print(f"  confirmed: I_op={r['I_op_A']:.2f}A B_target={r['B_target_T']:.3f}T "
             f"ta_worst_margin={r['ta_worst_margin']:.4f}")
    else:
        print("\nNo evaluation satisfied both B_target and hoop within budget "
             "-- see history.csv for the closest candidates (smallest "
             "fitness among feasible=True rows).")


if __name__ == "__main__":
    main()
