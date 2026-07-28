"""
cmaes_search.py — CMA-ES search over (a, b, coil_half_gap, n_turns)
====================================================================
Minimizes tape length subject to:

    B_target_T     >= cfg.B_TARGET_MIN_T       (mean |Bz| over the target
                                                box at I_op, 10 T)
    uniformity_pct <= cfg.UNIFORMITY_MAX_PCT    (peak-to-peak over the box,
                                                SCIF-corrected, 1%)
    hoop_MPa       <= cfg.SIGMA_HOOP_MAX_PA/1e6 (end-cap curved sections
                                                only, B x J x bending
                                                radius, 400 MPa)
    face_gap_mm    >= cfg.MIN_COIL_GAP_M*1e3    (coil-to-coil FACE-TO-FACE
                                                clearance, 3 mm -- distinct
                                                from coil_half_gap, which is
                                                a half CENTRE-to-centre
                                                distance)

Operating point I_op = I_quench / SAFETY_FACTOR (cfg.SAFETY_FACTOR = 1.818,
i.e. the worst-margin cell runs at 55% of local Ic — the midpoint of the
requested 50-60% band; every other cell has more margin).  Delamination
stress is computed and reported but not enforced (see opt_config.py).

All physics reuses optimize_geometry.evaluate() unchanged — this script
only supplies a continuous parameterization and a penalty-method fitness
function on top of it.  Evaluations are independent black-box calls
(one FEM solve + quench bisection + Biot-Savart, no gradients), which is
exactly the setting CMA-ES targets, and matches the earlier recommendation
over gradient-based methods (BFGS etc.) given there is no adjoint through
the Picard/quench-bisection chain.

Variables are NOT individually normalized; instead CMA_stds sets each
coordinate's initial spread (pycma's built-in mechanism for handling
variables on different scales without hand-rolled rescaling), and n_turns
are rounded to the nearest int only at evaluation time (CMA-ES sees a
continuous — if slightly noisy — landscape internally).

Reference: Hansen, N. (2016), "The CMA Evolution Strategy: A Tutorial",
arXiv:1604.00772.

Runtime: each fitness evaluation is one coarse-mesh FEM solve, ~5-10 s.
Population size defaults to pycma's 4 + 3*ln(10) ~= 11 per generation, so
cfg.CMAES_MAX_EVALS = 400 is ~36 generations, ~35-65 min serial. This
script is currently single-process; evaluations across a generation are
independent and can be parallelized later if the budget needs to grow
(see optimize/evaluate.py's docstring).

Run from Racetrack_v4 root:
    conda run -n fenicsx-env python3 optimize/cmaes_search.py

Outputs: optimize/cmaes_results.csv (best design), optimize/cmaes_history.csv
(every evaluation), and four figures in visualization/ (cmaes_convergence,
cmaes_constraints, cmaes_variables, cmaes_overview).
"""
import os, sys, csv, time

# 2026-07-22: force line-buffered stdout. Without this, print() output sits
# fully buffered whenever stdout is redirected to a file (the normal case
# for a backgrounded run) and only appears when the process exits --
# confirmed repeatedly across multi-hour runs where `tail`/`cat` on the log
# showed nothing until completion. Enables live progress monitoring.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "mesh"), os.path.join(_ROOT, "solve"),
           os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import cma

import params
import opt_config as cfg
import optimize_geometry as og
from mpi4py import MPI
from ic_model import IcModel

N_LAYERS = len(cfg.CMAES_X0["n_turns"])
assert N_LAYERS % 2 == 0, (
    f"N_LAYERS={N_LAYERS} is odd -- double-pancake construction (2026-07-23) "
    "requires every pancake to be one of a PAIR of adjacent layers wound as "
    "a single continuous piece (outer edges already match; the inner ends "
    "are joined), which is only physically buildable for an even layer "
    "count. Pick an even N_LAYERS.")
N_PAIRS = N_LAYERS // 2
# vector layout: [a, b, coil_half_gap, p_1, ..., p_N_PAIRS] -- ONE turn-count
# variable per double-pancake PAIR (layers 2i, 2i+1 share it), not one per
# physical layer. decode() expands each pair value to both of its layers.
IDX_A, IDX_B, IDX_GAP, IDX_N0 = 0, 1, 2, 3


# ── vector <-> (a, b, coil_half_gap, n_turns) coding ────────────────────────

def encode_x0():
    """cfg.CMAES_X0['n_turns'] is still given as the FULL per-layer list
    (length N_LAYERS) for readability -- averages each adjacent pair down
    to the single shared value the search actually optimizes over. If the
    warm start already satisfies pairing exactly (both halves of every
    pair equal), this is a no-op; otherwise it's the natural starting
    point for a design that's about to be forced to pair up anyway."""
    x0 = cfg.CMAES_X0
    full = x0["n_turns"]
    pairs = [(full[2 * i] + full[2 * i + 1]) / 2.0 for i in range(N_PAIRS)]
    return np.array([x0["a"], x0["b"], x0["coil_half_gap"], *pairs],
                    dtype=float)


def decode(x):
    a = float(x[IDX_A])
    b = float(x[IDX_B])
    gap = float(x[IDX_GAP])
    pair_vals = [max(1, int(round(xi)))
                 for xi in x[IDX_N0:IDX_N0 + N_PAIRS]]
    n_turns = []
    for p in pair_vals:
        n_turns.append(p)
        n_turns.append(p)
    return a, b, gap, n_turns


def gap_bounds_for_n_layers():
    """The true minimum coil_half_gap (>=3mm face-to-face clearance) depends
    on N_LAYERS (z_top = N_LAYERS*w/2), so a static config range tuned for
    one layer count is WRONG for another -- e.g. cfg.CMAES_HALF_GAP_BOUNDS
    was tuned for N_LAYERS=7 (floor 15.5mm); at N_LAYERS=12 the true floor
    is ~27.5mm, well inside that "box", which would let pycma sample gaps
    that are always penalized as infeasible. Compute the floor from
    N_LAYERS directly and preserve the ORIGINAL range width above it, so
    behavior at N_LAYERS=7 is unchanged (7*0.004/2 + 0.003/2 = 0.0155,
    exactly cfg.CMAES_HALF_GAP_BOUNDS[0]) when CMAES_TIGHT_BOUNDS is off.
    With CMAES_TIGHT_BOUNDS on, uses CMAES_TIGHT_GAP_MARGIN_M instead of
    the full original width -- see opt_config.py's "zone-out" section."""
    floor = N_LAYERS * params.w / 2.0 + cfg.MIN_COIL_GAP_M / 2.0
    if getattr(cfg, "CMAES_TIGHT_BOUNDS", False):
        width = cfg.CMAES_TIGHT_GAP_MARGIN_M
    else:
        width = cfg.CMAES_HALF_GAP_BOUNDS[1] - cfg.CMAES_HALF_GAP_BOUNDS[0]
    return floor, floor + width


def bounds_and_stds():
    # a, b: no box bound (None = unbounded in pycma) per project direction;
    # step size is set directly rather than derived from a bound range.
    tight = getattr(cfg, "CMAES_TIGHT_BOUNDS", False)
    n_bounds = cfg.CMAES_TIGHT_N_BOUNDS if tight else cfg.CMAES_N_BOUNDS

    gap_lo, gap_hi = gap_bounds_for_n_layers()
    lo = [cfg.CMAES_A_BOUNDS[0], cfg.CMAES_B_BOUNDS[0],
          gap_lo] + [n_bounds[0]] * N_PAIRS
    hi = [cfg.CMAES_A_BOUNDS[1], cfg.CMAES_B_BOUNDS[1],
          gap_hi] + [n_bounds[1]] * N_PAIRS
    gap_std = cfg.CMAES_SIGMA0_FRAC * (gap_hi - gap_lo)
    # 2026-07-23: n_std defaults to the bound-range-derived value (a wide,
    # cold-start-appropriate step), same as before -- but this was found to
    # be badly oversized for WARM-STARTED "polish" runs (proportional a/b
    # step via CMAES_A_STD0_OVERRIDE/CMAES_B_STD0_OVERRIDE, but turns still
    # got the full-range step): a run intended as a local refinement near a
    # known-good design instead sampled turn counts essentially uniformly
    # across the whole bound range from eval 1, and after 3000+ evals had
    # drifted to a WORSE optimum than its own starting point without
    # finding its way back. CMAES_N_STD0_OVERRIDE lets an orchestrator set
    # this proportionally (like the a/b overrides) for that case.
    n_std_override = getattr(cfg, "CMAES_N_STD0_OVERRIDE", None)
    n_std = (n_std_override if n_std_override is not None
             else cfg.CMAES_SIGMA0_FRAC * (n_bounds[1] - n_bounds[0]))
    stds = [cfg.CMAES_A_STD0, cfg.CMAES_B_STD0, gap_std] + [n_std] * N_PAIRS
    return lo, hi, stds


# ── cheap pre-check (mirrors params.recompute_derived()'s assertions plus
# the new coil-to-coil face-gap requirement) ────────────────────────────────
# Lets CMA-ES see a smooth penalty gradient for infeasible geometry without
# paying for a full mesh + FEM solve on every out-of-bounds sample.

def geometry_violation(a, b, gap, n_turns):
    L = b - a
    pack_thickness = max(n * params.t for n in n_turns)
    a_out = a + pack_thickness / 2
    a_inner_min = a_out - max(n * params.t for n in n_turns)

    min_L = 0.005            # 5 mm minimum straight section
    # 2026-07-23: REBCO tape minimum bend radius -- the innermost turn of
    # any layer (always the layer with the most turns, since every layer
    # shares the same outer edge a_out and eats inward by its own
    # n_i*t) cannot bend tighter than 7.5mm without risking cracking the
    # tape. Was 3mm (an arbitrary bore-clearance number with no material
    # basis) -- this is now a real mechanical limit, not a convenience
    # value, and is expected to bind hard (every design found so far sits
    # right at the old 3mm floor).
    min_clear = 0.0075        # 7.5 mm minimum bend radius (innermost turn)
    min_a = 0.003             # a is a radius -- must stay physically positive
    z_top = N_LAYERS * params.w / 2.0
    face_gap = 2.0 * (gap - z_top)   # coil-to-coil FACE-TO-FACE clearance

    viol = 0.0
    if a < min_a:
        viol += ((min_a - a) / min_a) ** 2
    if L < min_L:
        viol += ((min_L - L) / min_L) ** 2
    if a_inner_min < min_clear:
        viol += ((min_clear - a_inner_min) / min_clear) ** 2
    if face_gap < cfg.MIN_COIL_GAP_M:
        viol += ((cfg.MIN_COIL_GAP_M - face_gap) / cfg.MIN_COIL_GAP_M) ** 2
    # 2026-07-26: soft floor on `a` -- see opt_config.py's CMAES_MIN_A_M
    # comment. Smooth quadratic, same style as the other terms; 0 disables.
    min_a_soft = getattr(cfg, "CMAES_MIN_A_M", 0.0)
    if min_a_soft > 0.0 and a < min_a_soft:
        viol += ((min_a_soft - a) / min_a_soft) ** 2
    return viol, face_gap


# ── fitness ──────────────────────────────────────────────────────────────────

_ic_model = None
_comm = None
_eval_count = 0
_history = []
_best = dict(fitness=np.inf)
_run_tag = "run"
_last_flushed_idx = 0    # _history[:_last_flushed_idx] already on disk
FLUSH_EVERY = 20          # ~1.5-2 generations; see _record()


def _evaluate_candidate(x):
    """Pure, worker-safe evaluation: decode x, run geometry_violation() and
    (if feasible) og.evaluate(), compute the constraint penalty, and return
    a plain-data dict. Reads only this process's own _ic_model/_comm/params
    globals (set once by main() in the parent serial path, or _worker_init()
    in a parallel-pool worker) and never touches _history/_best/_eval_count
    -- safe to call from a ProcessPoolExecutor worker. Identical evaluation
    logic to the old inline body of fitness(), just without the bookkeeping
    side effects."""
    a, b, gap, n_turns = decode(x)

    viol, face_gap = geometry_violation(a, b, gap, n_turns)
    if viol > 0.0:
        f = cfg.CMAES_INFEASIBLE_PENALTY_KM * (1.0 + viol)
        return dict(a=a, b=b, gap=gap, n_turns=n_turns, face_gap=face_gap,
                    f=f, feasible=False, r=None, g=None)

    cand = dict(a=a, b=b, n_turns=n_turns, coil_half_gap=gap)
    try:
        r = og.evaluate(cand, _ic_model, _comm)
    except AssertionError:
        f = cfg.CMAES_INFEASIBLE_PENALTY_KM
        return dict(a=a, b=b, gap=gap, n_turns=n_turns, face_gap=face_gap,
                    f=f, feasible=False, r=None, g=None)

    if not r.get("feasible", False):
        f = cfg.CMAES_INFEASIBLE_PENALTY_KM
        return dict(a=a, b=b, gap=gap, n_turns=n_turns, face_gap=face_gap,
                    f=f, feasible=False, r=None, g=None)

    hoop_max_mpa = cfg.SIGMA_HOOP_MAX_PA / 1e6
    g = [
        max(0.0, (cfg.B_TARGET_MIN_T - r["B_target_T"]) / cfg.B_TARGET_MIN_T),
        max(0.0, (r["hoop_MPa"] - hoop_max_mpa) / hoop_max_mpa),
    ]
    # 2026-07-24: NO uniformity signal is currently in the fitness
    # function. History, in order:
    #   1. r["uniformity_pct"] (the coarse Bean-state on-axis proxy) used
    #      to be here. Full T-A box-uniformity validation found it not
    #      just noisy but ANTI-correlated with the real target: the
    #      design with the best on-axis proxy score (n_layers=10, on-axis
    #      1.37%) has the WORST true box uniformity of every design tried
    #      (9.18% peak-to-peak, vs. 0.44-1.06% for 4/6/8 layers). Removed.
    #   2. A penalty on peak per-pair turn concentration was added in its
    #      place, reasoning from the same (flawed) on-axis data. Once box
    #      uniformity was actually measured, peak turns turned out NOT to
    #      track it either (n_layers=12 has fewer peak turns than 4/6/8
    #      but worse box uniformity). Removed too.
    #   3. What DOES track true box uniformity, cleanly, across all five
    #      layer counts tested: coil radius `a` (bigger a -> better box
    #      uniformity -- the target box is a FIXED 30x6mm, so a smaller
    #      coil means the box eats into proportionally more of the
    #      near-field region). No penalty on `a` has been added here yet
    #      -- `a` is already a free, unbounded search variable, and biasing
    #      it upward through the objective would fight directly against
    #      minimizing tape_km (smaller coils generally need less tape),
    #      which is a real physical tradeoff this project hasn't resolved
    #      into a single proxy term yet, not just an implementation gap.
    # r["uniformity_pct"] is still computed and recorded in the CSV logs
    # for reference. Until a fast, TRUSTWORTHY uniformity signal exists,
    # T-A-validate box uniformity (see ta_solve.py's box-uniformity
    # extension) on a shortlist of low-tape candidates AFTER the coarse
    # search, rather than trusting anything in the coarse screen's own
    # ranking for uniformity. See CLAUDE.md's "Coarse-screen SCIF proxy
    # found unreliable" section for the complete investigation.
    penalty = cfg.CMAES_PENALTY_KM * sum(gi ** 2 for gi in g)
    f = r["tape_km"] + penalty
    return dict(a=a, b=b, gap=gap, n_turns=n_turns, face_gap=face_gap,
                f=f, feasible=True, r=r, g=g)


def fitness(x):
    """Serial-path objective handed to cma.optimize(). Thin wrapper around
    _evaluate_candidate(): same eval-count increment + _record() call, same
    order, as before the extraction -- this keeps CMAES_N_WORKERS=1 output
    byte-for-byte identical to the pre-parallelization code."""
    global _eval_count
    _eval_count += 1
    res = _evaluate_candidate(x)
    _record(x, res["a"], res["b"], res["gap"], res["n_turns"],
            res["face_gap"], res["f"], res["feasible"], res["r"],
            res.get("g"))
    return res["f"]


def _record(x, a, b, gap, n_turns, face_gap, f, feasible, r, g=None):
    all_pass = feasible and all(gi <= 1e-9 for gi in (g or [1, 1, 1]))
    row = dict(run_tag=_run_tag, eval=_eval_count, fitness=f,
               feasible=feasible, all_constraints_ok=all_pass,
               a_mm=a * 1e3, b_mm=b * 1e3, gap_mm=gap * 1e3,
               face_gap_mm=face_gap * 1e3,
               n_turns=str(n_turns), n_total=sum(n_turns))
    if r is not None:
        row.update(tape_km=r["tape_km"], B_target_T=r["B_target_T"],
                   uniformity_pct=r["uniformity_pct"],
                   hoop_MPa=r["hoop_MPa"], delam_MPa=r["delam_MPa"],
                   delam_scr_MPa=r["delam_scr_MPa"],
                   I_quench_A=r["I_quench_A"], I_op_A=r["I_op_A"],
                   binding=r["binding"], clip_frac=r["clip_frac"])
    _history.append(row)

    if all_pass and f < _best["fitness"]:
        _best.clear()
        _best.update(row)
        _best["fitness"] = f

    tag = "OK " if all_pass else ("INFEAS" if not feasible else "viol")
    extra = (f"tape={r['tape_km']:.3f}km B={r['B_target_T']:.2f}T "
             f"unif={r['uniformity_pct']:.3f}% hoop={r['hoop_MPa']:.0f}MPa"
             if r is not None else "no evaluation (geometry pre-check)")
    print(f"  [{_eval_count:4d}] {tag:6s} f={f:8.3f}  "
          f"a={a*1e3:.1f} b={b*1e3:.1f} gap={gap*1e3:.1f} "
          f"(face={face_gap*1e3:.1f}mm) n={n_turns}  {extra}")

    # 2026-07-22: periodic incremental flush to disk -- previously
    # cmaes_history.csv/cmaes_all_evaluations.csv/cmaes_results.csv were
    # only ever written once, at the very end of main(), meaning a crash
    # or kill mid-run (e.g. a multi-hour extended-refinement run) lost
    # EVERY evaluation done so far with no way to recover it. Every
    # FLUSH_EVERY evaluations (~1.5-2 generations), append any not-yet-
    # written rows to the master log and rewrite this run's own CSVs, so
    # at most FLUSH_EVERY evaluations' worth of work is ever at risk.
    if _eval_count % FLUSH_EVERY == 0:
        _append_master_log()
        _write_csv(quiet=True)


# ── opt-in generation-parallel evaluation (cfg.CMAES_N_WORKERS > 1) ─────────
# Default (CMAES_N_WORKERS = 1) never touches any of this -- main() takes
# the plain es.optimize(fitness) branch, byte-for-byte the original path.

def _worker_init():
    """ProcessPoolExecutor initializer, runs once per worker process before
    any task is sent to it. Mirrors what main() does once in the serial
    path (_comm, _ic_model, SCREEN_MESH_OVERRIDES), plus the one extra step
    required only for parallel workers: giving each worker a distinct
    mesh_filename so concurrent build_mesh.build(write_path=...) calls
    (inside og.evaluate()) can never race on the shared default path
    mesh/racetrack_mesh.msh -- params is a plain module, so each OS process
    has its own independent copy of params.mesh_filename to repoint."""
    global _ic_model, _comm
    _comm = MPI.COMM_WORLD
    _ic_model = IcModel()
    for k, v in cfg.SCREEN_MESH_OVERRIDES.items():
        setattr(params, k, v)
    root, ext = os.path.splitext(params.mesh_filename)
    params.mesh_filename = f"{root}_w{os.getpid()}{ext}"


def _run_parallel(es, n_workers):
    """One generation (population) per round-trip to a process pool.
    pool.map() preserves input order, so fitnesses[i] always corresponds to
    solutions[i] regardless of completion order -- required for
    es.tell(solutions, fitnesses). es.stop()/opts['maxfevals'] are driven
    by es.countevals, which tell() increments by len(fitnesses) each call
    -- the same mechanism .optimize() itself relies on -- so budget
    semantics, including the same "last generation may overshoot
    maxfevals" behavior that already exists in the serial path, are
    unchanged by using this manual loop instead.

    2026-07-22: added es.logger.add(es) after tell(), matching what pycma's
    own optimize() does internally via _prepare_callback_list() (which
    appends self.logger.add as a callback, then invokes it as f(self) --
    i.e. the `es` argument IS required; `es.logger.add()` with no argument
    silently returns without writing anything -- verified directly:
    logger.add() bare produced zero files in outcmaes/ over 6 generations,
    while logger.add(es) with the same opts wrote all 9 files as expected).
    Without this, this manual loop never wrote to outcmaes/ at all --
    confirmed by checking outcmaes/fit.dat's mtime during a live
    2500-eval parallel run and finding it ~19h stale. Every invocation so
    far that used CMAES_N_WORKERS>1 (both extended-refinement rounds) was
    affected; this only fixes it going forward, it doesn't recover missed
    data from already-completed runs."""
    global _eval_count
    import multiprocessing as mp
    import concurrent.futures as cf

    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                initializer=_worker_init) as pool:
        while not es.stop():
            solutions = es.ask()
            results = list(pool.map(_evaluate_candidate, solutions))
            fitnesses = []
            for x, res in zip(solutions, results):
                _eval_count += 1
                _record(x, res["a"], res["b"], res["gap"], res["n_turns"],
                        res["face_gap"], res["f"], res["feasible"],
                        res["r"], res.get("g"))
                fitnesses.append(res["f"])
            es.tell(solutions, fitnesses)
            es.logger.add(es)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    global _ic_model, _comm, _run_tag
    _comm = MPI.COMM_WORLD
    _ic_model = IcModel()
    _run_tag = f"run_{time.strftime('%Y%m%d_%H%M%S')}"

    for k, v in cfg.SCREEN_MESH_OVERRIDES.items():
        setattr(params, k, v)

    x0 = encode_x0()
    lo, hi, stds = bounds_and_stds()

    opts = {
        "bounds": [lo, hi],
        "CMA_stds": stds,
        "seed": cfg.CMAES_SEED,
        "maxfevals": cfg.CMAES_MAX_EVALS,
        "verbose": -3,   # we print our own per-eval log
    }
    if cfg.CMAES_POPSIZE is not None:
        opts["popsize"] = cfg.CMAES_POPSIZE

    print(f"Starting CMA-ES: {len(x0)} variables, budget "
          f"{cfg.CMAES_MAX_EVALS} evaluations, target B >= "
          f"{cfg.B_TARGET_MIN_T} T, uniformity <= "
          f"{cfg.UNIFORMITY_MAX_PCT}%, hoop <= "
          f"{cfg.SIGMA_HOOP_MAX_PA/1e6:.0f} MPa, coil-coil face gap >= "
          f"{cfg.MIN_COIL_GAP_M*1e3:.0f} mm\n")

    es = cma.CMAEvolutionStrategy(x0, 1.0, opts)
    t0 = time.time()

    n_workers = getattr(cfg, "CMAES_N_WORKERS", 1)
    if n_workers > 1:
        # Set BEFORE the pool is created so spawned children inherit these
        # in os.environ at interpreter boot -- earlier than their own
        # numpy/dolfinx/PETSc/MKL imports, which matters because a spawned
        # child must re-import this module to locate _worker_init/
        # _evaluate_candidate by name, meaning module-level imports happen
        # before _worker_init()'s body ever runs. Scoped to this branch
        # only -- the serial path's environment is never touched.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        print(f"Parallel mode: {n_workers} worker processes per generation.")
        _run_parallel(es, n_workers)
    else:
        es.optimize(fitness)   # UNCHANGED -- exact original serial path

    dt = time.time() - t0

    print(f"\n{'='*90}\nCMA-ES finished: {_eval_count} evaluations in "
          f"{dt/60:.1f} min\n{'='*90}")

    if _best["fitness"] < np.inf:
        print("\nBest feasible design found:")
        for k in ("a_mm", "b_mm", "gap_mm", "face_gap_mm", "n_turns",
                  "n_total", "tape_km", "B_target_T", "uniformity_pct",
                  "hoop_MPa", "delam_MPa", "I_quench_A", "I_op_A",
                  "binding", "clip_frac"):
            if k in _best:
                print(f"  {k:16s} {_best[k]}")
    else:
        print("\nNo feasible design satisfied all constraints within the "
              "evaluation budget — widen bounds or raise CMAES_MAX_EVALS.")

    _write_csv()
    _append_master_log()
    print(f"This run's {_last_flushed_idx} rows are in the cumulative "
          f"master log -> {cfg.CMAES_MASTER_LOG} "
          f"(flushed incrementally every {FLUSH_EVERY} evals during the "
          f"run, not just now at the end)")
    _make_plots()
    _make_param_map()


def _append_master_log():
    """Append any NOT-YET-WRITTEN rows (tracked via _last_flushed_idx) to
    the cumulative, never-overwritten master log (cfg.CMAES_MASTER_LOG) --
    the parameter-space "map" dataset spanning every cmaes_search.py run to
    date, each tagged with run_tag. Called both periodically during a run
    (see _record()) and once more at the end -- the index tracking means
    calling it twice never double-writes a row, so both call sites can use
    the exact same function."""
    global _last_flushed_idx
    new_rows = _history[_last_flushed_idx:]
    if not new_rows:
        return
    fields = ["run_tag", "eval", "fitness", "feasible", "all_constraints_ok",
              "a_mm", "b_mm", "gap_mm", "face_gap_mm", "n_turns", "n_total",
              "tape_km", "B_target_T", "uniformity_pct", "hoop_MPa",
              "delam_MPa", "delam_scr_MPa", "I_quench_A", "I_op_A",
              "binding", "clip_frac"]
    path = os.path.join(_ROOT, cfg.CMAES_MASTER_LOG)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(new_rows)
    _last_flushed_idx = len(_history)


def _write_csv(quiet=False):
    """Rewrites (not appends -- this file always reflects THIS run only,
    unlike the cumulative master log) cfg.CMAES_OUT_LOG/CMAES_OUT_CSV.
    quiet=True for the periodic mid-run calls from _record() (avoids a
    print every FLUSH_EVERY evals); the final call from main() still
    prints normally."""
    if not _history:
        return
    keys = list(_history[0].keys())
    for row in _history[1:]:
        for k in row:
            if k not in keys:
                keys.append(k)
    path = os.path.join(_ROOT, cfg.CMAES_OUT_LOG)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(_history)
    if not quiet:
        print(f"\nFull evaluation history -> {cfg.CMAES_OUT_LOG}")

    if _best["fitness"] < np.inf:
        best_path = os.path.join(_ROOT, cfg.CMAES_OUT_CSV)
        with open(best_path, "w", newline="") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=list(_best.keys()))
            w.writeheader()
            w.writerow(_best)
        if not quiet:
            print(f"Best design            -> {cfg.CMAES_OUT_CSV}")


# ── visualization ────────────────────────────────────────────────────────────
# Dark theme matching the rest of the project (see CLAUDE.md figure style):
# #111 figure bg, #0d0d1a axes bg, white labels, #444 spines, magma/plasma
# colormaps, monospace stats panels in #1a1a2e boxes.

def _style_axes(ax):
    ax.set_facecolor("#0d0d1a")
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.grid(True, alpha=0.2, color="#555")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")


def _make_plots():
    if not _history:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = getattr(params, "VIZ_DIR", os.path.join(_ROOT, "visualization"))
    os.makedirs(viz_dir, exist_ok=True)

    ev = np.array([r["eval"] for r in _history])
    fit = np.array([r["fitness"] for r in _history])
    ok = np.array([r["all_constraints_ok"] for r in _history])
    feas = np.array([r["feasible"] for r in _history])
    colors = np.where(ok, "limegreen", np.where(feas, "orange", "tomato"))

    def _best_so_far():
        best = np.full(len(fit), np.nan)
        cur = np.inf
        for i in range(len(fit)):
            if ok[i] and fit[i] < cur:
                cur = fit[i]
            best[i] = cur if np.isfinite(cur) else np.nan
        return best

    # ── 1. convergence: fitness history + best-feasible-so-far ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#111")
    for ax in axes:
        _style_axes(ax)

    axes[0].scatter(ev, fit, c=colors, s=18, alpha=0.85)
    axes[0].plot(ev, _best_so_far(), color="cyan", lw=1.8,
                 label="best feasible so far")
    axes[0].set_xlabel("evaluation #")
    axes[0].set_ylabel("fitness  (tape_km + penalty)")
    axes[0].set_title("Convergence", color="white", fontsize=10)
    axes[0].legend(fontsize=8, labelcolor="white", facecolor="#222")
    axes[0].set_yscale("log")

    feasible_rows = [r for r in _history if r.get("feasible")]
    if feasible_rows:
        fev = np.array([r["eval"] for r in feasible_rows])
        b_t = np.array([r["B_target_T"] for r in feasible_rows])
        unif = np.array([r["uniformity_pct"] for r in feasible_rows])
        hoop = np.array([r["hoop_MPa"] for r in feasible_rows])
        fcolors = np.array([("limegreen" if r["all_constraints_ok"]
                            else "orange") for r in feasible_rows])

        axes[1].plot(fev, (cfg.B_TARGET_MIN_T - b_t) / cfg.B_TARGET_MIN_T
                    * 100, color="dodgerblue", lw=1, alpha=0.6,
                    label="B_target deficit %")
        axes[1].plot(fev, unif - cfg.UNIFORMITY_MAX_PCT, color="gold",
                    lw=1, alpha=0.6, label="uniformity excess (pp)")
        axes[1].plot(fev, hoop - cfg.SIGMA_HOOP_MAX_PA / 1e6,
                    color="orchid", lw=1, alpha=0.6,
                    label="hoop excess (MPa)")
        axes[1].axhline(0.0, color="white", ls="--", lw=1)
        axes[1].set_xlabel("evaluation #")
        axes[1].set_ylabel("constraint margin (<=0 means satisfied)")
        axes[1].set_title("Constraint margins vs. evaluation",
                          color="white", fontsize=10)
        axes[1].legend(fontsize=7, labelcolor="white", facecolor="#222")

    fig.suptitle("CMA-ES convergence", color="white", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(viz_dir, "cmaes_convergence.png"), dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)

    # ── 2. per-constraint detail (feasible evals only) ──────────────────────
    if feasible_rows:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.patch.set_facecolor("#111")
        for ax in axes.flat:
            _style_axes(ax)

        axes[0, 0].scatter(fev, b_t, c=fcolors, s=16)
        axes[0, 0].axhline(cfg.B_TARGET_MIN_T, color="cyan", ls="--", lw=1.2,
                           label=f"{cfg.B_TARGET_MIN_T:.0f} T target")
        axes[0, 0].set_xlabel("evaluation #")
        axes[0, 0].set_ylabel("B_target  [T]")
        axes[0, 0].set_title("Mean |Bz| over target box", color="white",
                             fontsize=10)
        axes[0, 0].legend(fontsize=8, labelcolor="white", facecolor="#222")

        axes[0, 1].scatter(fev, unif, c=fcolors, s=16)
        axes[0, 1].axhline(cfg.UNIFORMITY_MAX_PCT, color="gold", ls="--",
                           lw=1.2, label=f"{cfg.UNIFORMITY_MAX_PCT:.1f}% limit")
        axes[0, 1].set_xlabel("evaluation #")
        axes[0, 1].set_ylabel("uniformity  [%]")
        axes[0, 1].set_title("Peak-to-peak uniformity (SCIF-corrected)",
                             color="white", fontsize=10)
        axes[0, 1].legend(fontsize=8, labelcolor="white", facecolor="#222")

        axes[1, 0].scatter(fev, hoop, c=fcolors, s=16)
        axes[1, 0].axhline(cfg.SIGMA_HOOP_MAX_PA / 1e6, color="orchid",
                           ls="--", lw=1.2,
                           label=f"{cfg.SIGMA_HOOP_MAX_PA/1e6:.0f} MPa limit")
        axes[1, 0].set_xlabel("evaluation #")
        axes[1, 0].set_ylabel("hoop stress  [MPa]  (end-cap curves only)")
        axes[1, 0].set_title("Hoop stress (B x J x bending radius)",
                             color="white", fontsize=10)
        axes[1, 0].legend(fontsize=8, labelcolor="white", facecolor="#222")

        fgap = np.array([r["face_gap_mm"] for r in feasible_rows])
        axes[1, 1].scatter(fev, fgap, c=fcolors, s=16)
        axes[1, 1].axhline(cfg.MIN_COIL_GAP_M * 1e3, color="tomato",
                           ls="--", lw=1.2,
                           label=f"{cfg.MIN_COIL_GAP_M*1e3:.0f} mm minimum")
        axes[1, 1].set_xlabel("evaluation #")
        axes[1, 1].set_ylabel("coil-coil face gap  [mm]")
        axes[1, 1].set_title("Coil-to-coil clearance", color="white",
                             fontsize=10)
        axes[1, 1].legend(fontsize=8, labelcolor="white", facecolor="#222")

        fig.suptitle("Constraint detail (green = all pass, orange = "
                     "feasible but violating)", color="white", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(viz_dir, "cmaes_constraints.png"), dpi=150,
                    facecolor=fig.get_facecolor())
        plt.close(fig)

    # ── 3. design-variable trajectories ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor("#111")
    for ax in axes.flat:
        _style_axes(ax)

    a_mm = np.array([r["a_mm"] for r in _history])
    b_mm = np.array([r["b_mm"] for r in _history])
    gap_mm = np.array([r["gap_mm"] for r in _history])
    n_tot = np.array([r["n_total"] for r in _history])

    axes[0, 0].scatter(ev, a_mm, c=colors, s=14)
    axes[0, 0].set_xlabel("evaluation #"); axes[0, 0].set_ylabel("a  [mm]")
    axes[0, 0].set_title("Cap radius a", color="white", fontsize=10)

    axes[0, 1].scatter(ev, b_mm, c=colors, s=14)
    axes[0, 1].set_xlabel("evaluation #"); axes[0, 1].set_ylabel("b  [mm]")
    axes[0, 1].set_title("Centre->cap-centre b", color="white", fontsize=10)

    axes[1, 0].scatter(ev, gap_mm, c=colors, s=14)
    axes[1, 0].set_xlabel("evaluation #")
    axes[1, 0].set_ylabel("coil_half_gap  [mm]")
    axes[1, 0].set_title("Coil-to-coil half gap", color="white", fontsize=10)

    axes[1, 1].scatter(ev, n_tot, c=colors, s=14)
    axes[1, 1].set_xlabel("evaluation #")
    axes[1, 1].set_ylabel("total turns")
    axes[1, 1].set_title("Total turn count", color="white", fontsize=10)

    fig.suptitle("Design-variable trajectories  (green=pass, orange=viol, "
                "red=infeasible)", color="white", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(viz_dir, "cmaes_variables.png"), dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)

    # ── 4. overview: objective vs. tape cost, with best marked ──────────────
    if feasible_rows:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.patch.set_facecolor("#111")
        for ax in axes:
            _style_axes(ax)

        tape = np.array([r["tape_km"] for r in feasible_rows])
        axes[0].scatter(tape, b_t, c=fcolors, s=22)
        axes[0].axhline(cfg.B_TARGET_MIN_T, color="cyan", ls=":", lw=1)
        if _best["fitness"] < np.inf:
            axes[0].scatter([_best["tape_km"]], [_best["B_target_T"]],
                           marker="*", s=400, color="white",
                           edgecolor="black", linewidth=1, zorder=5,
                           label="best")
            axes[0].legend(fontsize=8, labelcolor="white", facecolor="#222")
        axes[0].set_xlabel("tape length  [km]")
        axes[0].set_ylabel("B_target  [T]")
        axes[0].set_title("Objective vs. cost (all feasible evaluations)",
                          color="white", fontsize=10)

        axes[1].scatter(unif, hoop, c=fcolors, s=22)
        axes[1].axvline(cfg.UNIFORMITY_MAX_PCT, color="gold", ls=":", lw=1)
        axes[1].axhline(cfg.SIGMA_HOOP_MAX_PA / 1e6, color="orchid", ls=":",
                        lw=1)
        if _best["fitness"] < np.inf:
            axes[1].scatter([_best["uniformity_pct"]], [_best["hoop_MPa"]],
                           marker="*", s=400, color="white",
                           edgecolor="black", linewidth=1, zorder=5)
        axes[1].set_xlabel("uniformity  [%]")
        axes[1].set_ylabel("hoop stress  [MPa]")
        axes[1].set_title("Constraint space (dotted lines = limits)",
                          color="white", fontsize=10)

        n_ok = int(ok.sum())
        stats = (f"evaluations: {len(_history)}\n"
                f"feasible:    {int(feas.sum())}\n"
                f"all-pass:    {n_ok}\n"
                f"best tape:   {_best.get('tape_km', float('nan')):.3f} km\n"
                f"best B:      {_best.get('B_target_T', float('nan')):.2f} T\n"
                f"best unif:   {_best.get('uniformity_pct', float('nan')):.3f} %\n"
                f"best hoop:   {_best.get('hoop_MPa', float('nan')):.0f} MPa\n"
                f"best gap:    {_best.get('face_gap_mm', float('nan')):.1f} mm")
        fig.text(0.5, -0.02, stats, ha="center", va="top", color="white",
                 fontsize=8, family="monospace",
                 bbox=dict(boxstyle="round", facecolor="#1a1a2e",
                          edgecolor="#444"))

        fig.suptitle("CMA-ES search overview (green=all pass, orange=viol)",
                     color="white", fontsize=12)
        fig.tight_layout(rect=(0, 0.05, 1, 0.93))
        fig.savefig(os.path.join(viz_dir, "cmaes_overview.png"), dpi=150,
                    facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    print(f"\nFigures -> {viz_dir}/cmaes_convergence.png, "
          f"cmaes_constraints.png, cmaes_variables.png, cmaes_overview.png")


def _make_param_map():
    """Cumulative parameter-space map across EVERY cmaes_search.py run to
    date (reads cfg.CMAES_MASTER_LOG, not just this run's _history) -- shows
    which regions of (a, b, coil_half_gap, n_total) have been explored and
    whether they were good, constraint-violating, or infeasible, colored by
    run so it's clear which run found what."""
    master_path = os.path.join(_ROOT, cfg.CMAES_MASTER_LOG)
    if not os.path.exists(master_path):
        return
    with open(master_path, newline="") as fcsv:
        all_rows = list(csv.DictReader(fcsv))
    if not all_rows:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _f(row, key, default=np.nan):
        v = row.get(key, "")
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _b(row, key):
        return str(row.get(key, "")).strip().lower() == "true"

    a_mm = np.array([_f(r, "a_mm") for r in all_rows])
    b_mm = np.array([_f(r, "b_mm") for r in all_rows])
    gap_mm = np.array([_f(r, "gap_mm") for r in all_rows])
    n_tot = np.array([_f(r, "n_total") for r in all_rows])
    tape = np.array([_f(r, "tape_km") for r in all_rows])
    feas = np.array([_b(r, "feasible") for r in all_rows])
    ok = np.array([_b(r, "all_constraints_ok") for r in all_rows])
    run_tags = [r.get("run_tag", "?") for r in all_rows]
    unique_runs = sorted(set(run_tags))
    run_idx = np.array([unique_runs.index(t) for t in run_tags])

    colors = np.where(ok, "limegreen", np.where(feas, "orange", "tomato"))
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    # Per-run marker shapes only make sense for a handful of runs -- with
    # dozens (e.g. an overnight sweep_restarts.py session), a legend with
    # one entry per run is unreadable and swamps the figure. Above this
    # threshold, fall back to a single uniform marker colored by outcome
    # only; the point is the aggregate map, not which run found what.
    MAX_RUNS_FOR_MARKERS = 10
    by_run = len(unique_runs) <= MAX_RUNS_FOR_MARKERS
    pt_size = 20 if len(all_rows) < 5000 else 6
    pt_alpha = 0.75 if len(all_rows) < 5000 else 0.35

    viz_dir = getattr(params, "VIZ_DIR", os.path.join(_ROOT, "visualization"))
    os.makedirs(viz_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.patch.set_facecolor("#111")
    for ax in axes.flat:
        _style_axes(ax)

    def _scatter_by_run(ax, x, y, mask=None):
        idx = np.arange(len(x)) if mask is None else np.where(mask)[0]
        if by_run:
            for i, tag in enumerate(unique_runs):
                m = idx[run_idx[idx] == i]
                ax.scatter(x[m], y[m], c=colors[m],
                          marker=markers[i % len(markers)], s=pt_size,
                          alpha=pt_alpha, edgecolor="none", label=tag)
        else:
            ax.scatter(x[idx], y[idx], c=colors[idx], marker="o",
                      s=pt_size, alpha=pt_alpha, edgecolor="none")

    _scatter_by_run(axes[0, 0], a_mm, b_mm)
    axes[0, 0].set_xlabel("a  [mm]"); axes[0, 0].set_ylabel("b  [mm]")
    axes[0, 0].set_title("Cap radius vs. centre-cap-centre length",
                         color="white", fontsize=10)

    _scatter_by_run(axes[0, 1], a_mm, gap_mm)
    axes[0, 1].set_xlabel("a  [mm]")
    axes[0, 1].set_ylabel("coil_half_gap  [mm]")
    axes[0, 1].set_title("Cap radius vs. coil-to-coil half gap",
                         color="white", fontsize=10)

    _scatter_by_run(axes[1, 0], n_tot, gap_mm)
    axes[1, 0].set_xlabel("total turns")
    axes[1, 0].set_ylabel("coil_half_gap  [mm]")
    axes[1, 0].set_title("Turn count vs. coil-to-coil half gap",
                         color="white", fontsize=10)

    fm = feas & np.isfinite(tape)
    _scatter_by_run(axes[1, 1], tape, n_tot, mask=fm)
    axes[1, 1].set_xlabel("tape length  [km]")
    axes[1, 1].set_ylabel("total turns")
    axes[1, 1].set_title("Tape cost vs. turn count (feasible only)",
                         color="white", fontsize=10)

    from matplotlib.lines import Line2D
    color_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="limegreen",
              markersize=7, label="all constraints pass", linestyle="none"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="orange",
              markersize=7, label="feasible, violating", linestyle="none"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",
              markersize=7, label="geometrically infeasible",
              linestyle="none"),
    ]
    if by_run:
        run_handles = [Line2D([0], [0], marker=markers[i % len(markers)],
                             color="w", markerfacecolor="gray",
                             markersize=7, label=tag, linestyle="none")
                      for i, tag in enumerate(unique_runs)]
    else:
        run_handles = []
    fig.legend(handles=color_handles + run_handles, loc="lower center",
              ncol=min(4, len(color_handles) + len(run_handles)),
              fontsize=8, labelcolor="white", facecolor="#1a1a2e",
              edgecolor="#444", bbox_to_anchor=(0.5, -0.02))

    n_ok = int(ok.sum())
    fig.suptitle(f"Cumulative parameter-space map — {len(all_rows)} "
                f"evaluations across {len(unique_runs)} run(s), "
                f"{n_ok} all-pass", color="white", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    out_path = os.path.join(_ROOT, cfg.CMAES_PARAM_MAP_PNG)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(),
               bbox_inches="tight")
    plt.close(fig)
    print(f"Cumulative parameter-space map -> {cfg.CMAES_PARAM_MAP_PNG} "
          f"({len(all_rows)} evaluations, {len(unique_runs)} runs)")


if __name__ == "__main__":
    main()
