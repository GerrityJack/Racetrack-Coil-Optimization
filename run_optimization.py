#!/usr/bin/env python3
"""
run_optimization.py — ONE-FILE entry point for the racetrack coil design search
================================================================================

    python run_optimization.py            # check setup, then search (resumes automatically)
    python run_optimization.py --check    # only check the setup, then exit
    python run_optimization.py --smoke    # check setup + evaluate the start design once
    python run_optimization.py --fresh    # ignore any saved checkpoint and start over
    python run_optimization.py --workers 4 --max-evals 300

Run it with the Python of the `fenicsx-env` conda environment (see README):

    conda env create -f environment.yml          # once
    conda activate fenicsx-env
    python run_optimization.py

What it optimizes
-----------------
A two-coil REBCO racetrack magnet (4 mm tape, 20 K). Variables: end-cap
radius a, straight-section parameter b, coil half-gap, and one turn count
per double-pancake pair (N_LAYERS is fixed by START_DESIGN below).

For every candidate the T-A screening-current solver finds the LARGEST
transport current at which NO coil cell is above its local critical current
density at the moment a 2-hour constant-power ramp ends
(optimize/ta_ramp_current.py). The mean field in the 30x6 mm target box,
uniformity and hoop stress are evaluated at that current. The fitness is

    tape length [km]
    + W_FIELD      * (shortfall below B_MIN_T      / B_MIN_T)^2
    + W_UNIFORMITY * (excess above UNIFORMITY_MAX / UNIFORMITY_MAX)^2
    + W_HOOP       * (excess above HOOP_MAX       / HOOP_MAX)^2

i.e. reach 10 T first, then use as little tape as possible, with a
uniformity penalty. Geometrically impossible designs (bend radius < 7.5 mm,
coil face gap < 3 mm, ...) are rejected in milliseconds without a solve.

Everything is saved as it goes (optimize/runs/full_config_search/):
    history.csv     every evaluated design, one row each
    best.csv        best design so far (all constraints satisfied)
    checkpoint.pkl  CMA-ES state after every generation -> re-running this
                    file resumes where it stopped (Ctrl+C, crash, power cut)
    run.log         a copy of everything printed to the terminal

Cost: each candidate takes several T-A solves (~20-60 min on a modern
8-core machine for 16 layers). Finalists must then be validated with a
real sqrt(t) ramp march and a finer mesh -- see README, "Validating a
design".
"""
# ─────────────────────────────────────────────────────────────────────────────
#  USER SETTINGS — the only part of this file you normally edit
# ─────────────────────────────────────────────────────────────────────────────
# Starting design: the 16-layer design (8 double pancakes), SI units.
START_DESIGN = dict(
    a=0.03113, b=0.04002, coil_half_gap=0.03397,
    n_turns=[550, 550, 584, 584, 17, 17, 507, 507,
             507, 507, 613, 613, 629, 629, 605, 605])

B_MIN_T = 10.0            # required mean |Bz| over the target box. The
                          # project's history says a search converges ONTO
                          # this floor; use ~10.3 for build-tolerance margin.
UNIFORMITY_MAX_PCT = 0.8  # peak-to-peak |B| over the box [%] (spec is < 1 %)
HOOP_MAX_MPA = 400.0      # end-cap hoop stress limit

W_FIELD = 3000.0          # penalty weights [km of tape per (relative
W_UNIFORMITY = 50.0       #   violation)^2]. Field dominates until 10 T is
W_HOOP = 20.0             #   reached; uniformity was 20 in earlier searches.
PENALTY_FAILED_KM = 1000.0  # fitness for a candidate whose evaluation failed

RAMP_TIME_S = 7200.0      # constant-power ramp duration (2 h)

MAX_EVALS = 300           # total candidate evaluations (T-A-solved ones)
N_WORKERS = None          # parallel candidates; None = choose from CPUs/RAM
POPSIZE = None            # CMA-ES population; None = pycma default
SEED = 20260927

# Initial CMA-ES step sizes (roughly how far it explores at first)
STEP_A_M, STEP_B_M, STEP_GAP_M, STEP_TURNS = 0.003, 0.004, 0.0025, 60.0
# Search box (physical limits are enforced separately, see geometry_violation)
A_RANGE_M = (0.005, 0.100)
B_RANGE_M = (0.006, 0.150)
GAP_EXTRA_RANGE_M = 0.060  # half-gap may go up to its floor + this
TURNS_RANGE = (1, 1500)

CANDIDATE_TIMEOUT_H = 6.0    # one candidate taking longer = a hung solve
HEARTBEAT_MIN = 10.0         # print a "still running" line this often
MIN_RAM_PER_WORKER_GB = 2.5  # used to pick N_WORKERS automatically
OUT_SUBDIR = "full_config_search"
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import csv
import time
import json
import signal
import argparse
import traceback
import datetime as _dt

ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (ROOT, os.path.join(ROOT, "physics"), os.path.join(ROOT, "mesh"),
           os.path.join(ROOT, "solve"), os.path.join(ROOT, "optimize"),
           os.path.join(ROOT, "sweep")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
OUT_DIR = os.environ.get("RUN_OPT_OUT_DIR",
                         os.path.join(ROOT, "optimize", "runs", OUT_SUBDIR))
HISTORY_CSV = os.path.join(OUT_DIR, "history.csv")
BEST_CSV = os.path.join(OUT_DIR, "best.csv")
CHECKPOINT = os.path.join(OUT_DIR, "checkpoint.pkl")
RUN_LOG = os.path.join(OUT_DIR, "run.log")

HISTORY_FIELDS = [
    "eval", "generation", "timestamp", "status", "fitness",
    "all_constraints_ok", "a_mm", "b_mm", "gap_mm", "face_gap_mm",
    "n_layers", "n_turns", "n_total", "tape_km", "I_op_A", "B_target_T",
    "uniformity_pct", "hoop_MPa", "worst_load", "ta_converged",
    "n_ta_solves", "trials", "eval_min", "error"]

# medium mesh tier (params.py on disk may hold a much slower one-off tier)
MESH_SETTINGS = dict(mesh_size_min_factor=0.25, mesh_size_max_factor=0.60,
                     mesh_dist_min_factor=1.50, mesh_dist_max_factor=2.00,
                     box_scale=4.0, mesh_nz_per_layer=3,
                     mesh_z_grading=[0.075, 0.15, 0.55, 0.15, 0.075])
TAPE_W = 0.004        # m (params.w)
TAPE_T = 75e-6        # m (params.t)
MIN_FACE_GAP_M = 0.003
MIN_BEND_RADIUS_M = 0.0075


# ── terminal output (also copied to run.log) ─────────────────────────────────

class _Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, "a", buffering=1)

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()


def now():
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def say(msg=""):
    print(f"[{now()}] {msg}" if msg else "", flush=True)


def banner(title):
    print("\n" + "=" * 78 + f"\n  {title}\n" + "=" * 78, flush=True)


def fmt_dur(s):
    s = int(max(s, 0))
    h, m = divmod(s // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s % 60:02d}s"


def die(msg, hint=None):
    print("\n" + "!" * 78, file=sys.stderr)
    print(f"  ERROR: {msg}", file=sys.stderr)
    if hint:
        print(f"  HOW TO FIX: {hint}", file=sys.stderr)
    print("!" * 78 + "\n", file=sys.stderr, flush=True)
    sys.exit(1)


# ── setup checks ────────────────────────────────────────────────────────────

def preflight():
    banner("1/3  Checking your setup")
    env_hint = ("create and activate the conda environment:  "
                "conda env create -f environment.yml && conda activate "
                "fenicsx-env   (then run this file again with that python)")
    if sys.version_info < (3, 10):
        die(f"Python {sys.version.split()[0]} is too old (need >= 3.10).",
            env_hint)
    say(f"python {sys.version.split()[0]}  ({sys.executable})")

    missing = []
    versions = {}
    for mod in ("numpy", "scipy", "mpi4py", "petsc4py", "dolfinx", "ufl",
                "basix", "gmsh", "cma"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "ok")
        except Exception as e:           # noqa: BLE001 - report any import failure
            missing.append(f"{mod} ({type(e).__name__}: {e})")
    if missing:
        die("these required packages could not be imported:\n    "
            + "\n    ".join(missing), env_hint)
    say("packages: " + ", ".join(f"{k} {v}" for k, v in versions.items()))
    if not str(versions["dolfinx"]).startswith("0.11"):
        say(f"WARNING: dolfinx {versions['dolfinx']} -- this code is written "
            f"for dolfinx 0.11 (environment.yml). It may fail.")

    import params
    for label, path in (("Ic(B,theta) data", params.shanghai_csv_filename),
                        ("n-value data", params.n_value_csv_filename)):
        if not os.path.exists(path):
            die(f"{label} file not found: {path}",
                "make sure the repository was copied completely (physics/*.csv).")
    say("tape data files found (physics/*.csv)")

    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        test = os.path.join(OUT_DIR, ".write_test")
        open(test, "w").close()
        os.remove(test)
    except OSError as e:
        die(f"cannot write to the output folder {OUT_DIR}: {e}",
            "run from a folder you have write permission to.")
    say(f"output folder: {OUT_DIR}")

    n_cpu = os.cpu_count() or 1
    ram_gb = _ram_gb()
    say(f"machine: {n_cpu} CPU threads, "
        f"{ram_gb:.1f} GB RAM" if ram_gb else f"machine: {n_cpu} CPU threads")
    _check_start_design()
    return n_cpu, ram_gb


def _ram_gb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (ValueError, OSError, AttributeError):
        return None


def _check_start_design():
    n = START_DESIGN["n_turns"]
    if len(n) % 2:
        die(f"START_DESIGN has {len(n)} layers; the number of layers must be "
            f"even (double pancakes).", "edit START_DESIGN at the top of this file.")
    if any(n[2 * i] != n[2 * i + 1] for i in range(len(n) // 2)):
        die("START_DESIGN: both layers of each double pancake (layers 1&2, "
            "3&4, ...) must have the same turn count.",
            "edit START_DESIGN at the top of this file.")
    viol, face_gap, why = geometry_violation(
        START_DESIGN["a"], START_DESIGN["b"], START_DESIGN["coil_half_gap"], n)
    if viol > 0:
        die(f"START_DESIGN is not buildable: {why}",
            "edit START_DESIGN at the top of this file.")
    say(f"start design OK: {len(n)} layers, {sum(n)} turns, "
        f"face gap {face_gap*1e3:.2f} mm")


# ── design encoding and geometry rules ──────────────────────────────────────

N_LAYERS = len(START_DESIGN["n_turns"])
N_PAIRS = N_LAYERS // 2


def gap_floor():
    return N_LAYERS * TAPE_W / 2.0 + MIN_FACE_GAP_M / 2.0


def encode(d):
    pairs = [d["n_turns"][2 * i] for i in range(N_PAIRS)]
    return [d["a"], d["b"], d["coil_half_gap"], *pairs]


def decode(x):
    a, b, gap = float(x[0]), float(x[1]), float(x[2])
    n = []
    for v in x[3:3 + N_PAIRS]:
        k = max(1, int(round(v)))
        n += [k, k]
    return a, b, gap, n


def geometry_violation(a, b, gap, n_turns):
    """Same physical rules as optimize/cmaes_search.py: returns (violation
    size, face gap, reason). 0 violation = buildable."""
    t = TAPE_T
    a_inner_min = a - max(n_turns) * t / 2.0      # innermost turn radius
    face_gap = 2.0 * (gap - N_LAYERS * TAPE_W / 2.0)
    viol, why = 0.0, []
    if a < 0.003:
        viol += ((0.003 - a) / 0.003) ** 2
        why.append(f"a={a*1e3:.2f} mm too small")
    if b - a < 0.005:
        viol += ((0.005 - (b - a)) / 0.005) ** 2
        why.append(f"straight section b-a={1e3*(b-a):.2f} mm < 5 mm")
    if a_inner_min < MIN_BEND_RADIUS_M:
        viol += ((MIN_BEND_RADIUS_M - a_inner_min) / MIN_BEND_RADIUS_M) ** 2
        why.append(f"bend radius {a_inner_min*1e3:.2f} mm < 7.5 mm")
    if face_gap < MIN_FACE_GAP_M:
        viol += ((MIN_FACE_GAP_M - face_gap) / MIN_FACE_GAP_M) ** 2
        why.append(f"face gap {face_gap*1e3:.2f} mm < 3 mm")
    return viol, face_gap, "; ".join(why)


# ── worker side ─────────────────────────────────────────────────────────────

_W = {}


def _worker_init(ramp_time_s, in_worker=True):
    """Runs once in every worker process."""
    import warnings
    warnings.filterwarnings("ignore")
    if in_worker:
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # main process handles Ctrl+C
        sys.stdout = _Tee(sys.stdout, RUN_LOG)          # worker lines -> run.log too
    os.environ["TA_RAMP_T_RAMP_S"] = str(ramp_time_s)
    import params
    for k, v in MESH_SETTINGS.items():
        setattr(params, k, v)
    params.recompute_derived()
    from mpi4py import MPI
    from ic_extrapolation import make_ic_model
    import ta_ramp_current
    _W.update(comm=MPI.COMM_WORLD, ic=make_ic_model("kim"),
              ev=ta_ramp_current)


def evaluate_candidate(x, eval_id):
    """Evaluate one candidate. Never raises: any failure is returned as a
    status so one bad design cannot stop the search."""
    t0 = time.time()
    a, b, gap, n = decode(x)
    viol, face_gap, why = geometry_violation(a, b, gap, n)
    base = dict(a=a, b=b, gap=gap, n_turns=n, face_gap=face_gap)
    if viol > 0:
        return dict(base, status="rejected_geometry", error=why,
                    fitness=PENALTY_FAILED_KM * (1.0 + viol), r=None,
                    eval_min=0.0)
    if os.environ.get("RUN_OPT_DRY_RUN") == "1":
        time.sleep(float(os.environ.get("RUN_OPT_DRY_SLEEP", "0.2")))
        fault = os.environ.get("RUN_OPT_DRY_FAULT")    # test error handling
        if fault and eval_id % 5 == 0:
            if fault == "crash":
                os._exit(11)                            # like a native segfault
            try:
                raise RuntimeError("injected test failure")
            except RuntimeError as e:
                return dict(base, status="solver_error", error=str(e),
                            fitness=PENALTY_FAILED_KM, r=None, eval_min=0.0)
        B = 4.0 + 2.0e-4 * sum(n) / (1.0 + 30.0 * abs(a - 0.031))
        r = dict(tape_km=sum(n) * 2 * 3.1416 * a / 1e3, I_op_A=80.0,
                 B_target_T=B, uniformity_pct=0.5, hoop_MPa=50.0,
                 ta_worst_load=1.0, converged=True, n_ta_solves=0,
                 trials="dry-run")
        g_f = max(0.0, (B_MIN_T - B) / B_MIN_T)
        return dict(base, status="ok", error="", r=r, ok=g_f == 0,
                    fitness=r["tape_km"] + W_FIELD * g_f ** 2, eval_min=0.0)
    print(f"[{now()}]   -> eval {eval_id} started   a={a*1e3:.2f} "
          f"b={b*1e3:.2f} gap={gap*1e3:.2f} mm, {sum(n)} turns", flush=True)
    try:
        r = _W["ev"].evaluate(dict(a=a, b=b, coil_half_gap=gap, n_turns=n),
                              _W["ic"], _W["comm"])
    except Exception as e:                # noqa: BLE001 - keep the search alive
        tb = traceback.format_exc(limit=3).strip().splitlines()[-1]
        print(f"[{now()}]   !! eval {eval_id} FAILED: {type(e).__name__}: "
              f"{e}", flush=True)
        return dict(base, status="solver_error", error=f"{type(e).__name__}: "
                    f"{e} | {tb}", fitness=PENALTY_FAILED_KM, r=None,
                    eval_min=(time.time() - t0) / 60)
    if not r.get("feasible", False):
        return dict(base, status="rejected_model", error=r.get("reason", ""),
                    fitness=PENALTY_FAILED_KM, r=None,
                    eval_min=(time.time() - t0) / 60)
    g_f = max(0.0, (B_MIN_T - r["B_target_T"]) / B_MIN_T)
    g_u = max(0.0, (r["uniformity_pct"] - UNIFORMITY_MAX_PCT) / UNIFORMITY_MAX_PCT)
    g_h = max(0.0, (r["hoop_MPa"] - HOOP_MAX_MPA) / HOOP_MAX_MPA)
    fitness = (r["tape_km"] + W_FIELD * g_f ** 2 + W_UNIFORMITY * g_u ** 2
               + W_HOOP * g_h ** 2)
    ok = g_f == 0 and g_u == 0 and g_h == 0
    print(f"[{now()}]   <- eval {eval_id} done in {fmt_dur(time.time()-t0)}: "
          f"B={r['B_target_T']:.2f} T at {r['I_op_A']:.1f} A, "
          f"unif={r['uniformity_pct']:.2f}%, tape={r['tape_km']:.3f} km  "
          f"{'PASSES ALL' if ok else 'violates constraints'}", flush=True)
    return dict(base, status="ok", error="", fitness=fitness, ok=ok, r=r,
                eval_min=(time.time() - t0) / 60)


# ── main process: bookkeeping ───────────────────────────────────────────────

class Search:
    def __init__(self):
        self.n_eval = 0          # every candidate, including geometry rejects
        self.n_solved = 0        # candidates that went through the T-A evaluator
        self.best = None
        self.t_start = time.time()
        self.eval_minutes = []
        self.stop_requested = False
        self.n_workers = 1

    def load_history(self):
        if not os.path.exists(HISTORY_CSV):
            return
        with open(HISTORY_CSV, newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.n_eval = int(rows[-1]["eval"]) if rows else 0
        self.n_solved = sum(r["status"] != "rejected_geometry" for r in rows)
        for row in rows:
            if row["eval_min"] and row["status"] == "ok":
                self.eval_minutes.append(float(row["eval_min"]))
            if row["all_constraints_ok"] == "True" and (
                    self.best is None
                    or float(row["fitness"]) < float(self.best["fitness"])):
                self.best = row

    def record(self, res, generation):
        self.n_eval += 1
        if res["status"] != "rejected_geometry":
            self.n_solved += 1
        r = res.get("r") or {}
        row = dict.fromkeys(HISTORY_FIELDS, "")
        row.update(eval=self.n_eval, generation=generation, timestamp=now(),
                   status=res["status"], fitness=res["fitness"],
                   all_constraints_ok=bool(res.get("ok", False)),
                   a_mm=res["a"] * 1e3, b_mm=res["b"] * 1e3,
                   gap_mm=res["gap"] * 1e3, face_gap_mm=res["face_gap"] * 1e3,
                   n_layers=len(res["n_turns"]), n_turns=str(res["n_turns"]),
                   n_total=sum(res["n_turns"]),
                   eval_min=round(res.get("eval_min", 0.0), 2),
                   error=res.get("error", ""))
        if r:
            row.update(tape_km=r["tape_km"], I_op_A=r["I_op_A"],
                       B_target_T=r["B_target_T"],
                       uniformity_pct=r["uniformity_pct"],
                       hoop_MPa=r["hoop_MPa"], worst_load=r["ta_worst_load"],
                       ta_converged=r["converged"],
                       n_ta_solves=r["n_ta_solves"], trials=r["trials"])
            self.eval_minutes.append(res.get("eval_min", 0.0))
        new = not os.path.exists(HISTORY_CSV)
        with open(HISTORY_CSV, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        if row["all_constraints_ok"] and (
                self.best is None
                or float(row["fitness"]) < float(self.best["fitness"])):
            self.best = row
            with open(BEST_CSV, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
                w.writeheader()
                w.writerow(row)
            say(f"*** NEW BEST design (eval {row['eval']}): tape "
                f"{float(row['tape_km']):.3f} km, B={float(row['B_target_T']):.2f} T, "
                f"unif={float(row['uniformity_pct']):.2f}%, n={row['n_turns']}")
        return row


def save_checkpoint(es):
    import pickle
    tmp = CHECKPOINT + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(dict(es=es, n_layers=N_LAYERS,
                         start_design=START_DESIGN), fh)
    os.replace(tmp, CHECKPOINT)


def load_checkpoint():
    import pickle
    if not os.path.exists(CHECKPOINT):
        return None
    try:
        with open(CHECKPOINT, "rb") as fh:
            d = pickle.load(fh)
    except Exception as e:                # noqa: BLE001
        say(f"WARNING: could not read the checkpoint ({type(e).__name__}: {e}); "
            f"starting a new search. (Old file kept as checkpoint.pkl.bad)")
        os.replace(CHECKPOINT, CHECKPOINT + ".bad")
        return None
    if d.get("n_layers") != N_LAYERS:
        die(f"the saved checkpoint is for a {d.get('n_layers')}-layer search "
            f"but START_DESIGN now has {N_LAYERS} layers.",
            "run with --fresh to start a new search (the old history.csv "
            "is kept), or restore the old START_DESIGN.")
    return d["es"]


def pick_workers(n_cpu, ram_gb):
    if N_WORKERS:
        return int(N_WORKERS)
    by_cpu = max(1, n_cpu // 2)
    by_ram = max(1, int((ram_gb or 8) // MIN_RAM_PER_WORKER_GB))
    return max(1, min(by_cpu, by_ram, 8))


def _child(x, eval_id, ramp_time_s, conn):
    """Entry point of the separate process that evaluates ONE candidate."""
    try:
        _worker_init(ramp_time_s)
        conn.send(evaluate_candidate(x, eval_id))
    except BaseException as e:            # noqa: BLE001 - report, never hang
        try:
            conn.send(dict(status="solver_error",
                           error=f"{type(e).__name__}: {e}"))
        except Exception:                 # noqa: BLE001
            pass
    finally:
        conn.close()


def _cleanup_child(pid):
    """Delete the temporary mesh / trace files of a child that was killed or
    crashed (its own cleanup code never ran)."""
    import glob
    import tempfile
    for pat in (os.path.join(ROOT, "mesh", f"*_taramp{pid}_*"),
                os.path.join(tempfile.gettempdir(), f"taramp_{pid}_*")):
        for f in glob.glob(pat):
            try:
                os.remove(f)
            except OSError:
                pass


def _failed(x, status, error):
    a, b, gap, n = decode(x)
    return dict(a=a, b=b, gap=gap, n_turns=n,
                face_gap=geometry_violation(a, b, gap, n)[1], status=status,
                error=error, fitness=PENALTY_FAILED_KM, r=None, eval_min=0.0)


def run_generation(es, search, n_workers, ctx):
    """Evaluate one CMA-ES generation. Every T-A candidate runs in its OWN
    process, so a native crash or a hung solve costs only that candidate."""
    xs = es.ask()
    gen = es.countiter + 1
    first_id = search.n_eval + 1
    results = [None] * len(xs)
    queue = []
    for i, x in enumerate(xs):                  # geometry rejects: instant
        if geometry_violation(*decode(x))[0] > 0:
            results[i] = evaluate_candidate(x, first_id + i)
        else:
            queue.append(i)
    say(f"generation {gen}: {len(xs)} candidates (evals {first_id}-"
        f"{first_id + len(xs) - 1}); {len(xs) - len(queue)} rejected for "
        f"geometry, {len(queue)} to solve on up to {n_workers} workers")
    running = {}                                # i -> (process, conn, t_start)
    t0 = last_beat = time.time()
    while queue or running:
        while queue and len(running) < n_workers:
            i = queue.pop(0)
            recv, send = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_child, args=(xs[i], first_id + i,
                                                 RAMP_TIME_S, send), daemon=True)
            p.start()
            send.close()
            running[i] = (p, recv, time.time())
        time.sleep(2.0)
        for i, (p, recv, ts) in list(running.items()):
            if recv.poll():
                try:
                    res = recv.recv()
                except (EOFError, OSError):
                    res = None
                p.join(timeout=30)
                if res is None:
                    say(f"   !! eval {first_id + i}: its process died (exit code "
                        f"{p.exitcode}) -- usually a native crash in PETSc/MUMPS "
                        f"or running out of memory. Marked failed; the search "
                        f"continues.")
                    res = _failed(xs[i], "worker_crash",
                                  f"process exit code {p.exitcode}")
                    _cleanup_child(p.pid)
                elif "a" not in res:
                    say(f"   !! eval {first_id + i} failed: {res.get('error')}")
                    res = _failed(xs[i], "solver_error", res.get("error", ""))
                results[i] = res
                del running[i]
            elif not p.is_alive():
                say(f"   !! eval {first_id + i}: its process died (exit code "
                    f"{p.exitcode}) -- usually a native crash in PETSc/MUMPS or "
                    f"running out of memory. Marked failed; the search continues.")
                results[i] = _failed(xs[i], "worker_crash",
                                     f"process exit code {p.exitcode}")
                _cleanup_child(p.pid)
                del running[i]
            elif time.time() - ts > CANDIDATE_TIMEOUT_H * 3600:
                say(f"   !! eval {first_id + i} exceeded {CANDIDATE_TIMEOUT_H} h "
                    f"(probably a non-converging solve) -- killed, marked failed.")
                p.kill()
                p.join(timeout=30)
                results[i] = _failed(xs[i], "timeout",
                                     f"exceeded {CANDIDATE_TIMEOUT_H} h")
                _cleanup_child(p.pid)
                del running[i]
        if search.stop_requested:
            for p, _, _ in running.values():
                p.kill()
                p.join(timeout=30)
                _cleanup_child(p.pid)
            return None, xs
        if time.time() - last_beat > HEARTBEAT_MIN * 60:
            n_done = sum(r is not None for r in results)
            say(f"   ... still running: {n_done}/{len(xs)} candidates of "
                f"generation {gen} finished, {len(running)} in progress, "
                f"{fmt_dur(time.time() - t0)} elapsed")
            last_beat = time.time()
    fits = []
    for x, res in zip(xs, results):
        search.record(res, gen)
        fits.append(res["fitness"])
    return fits, xs


def summary(search, es, budget):
    done = search.n_solved
    mins = [m for m in search.eval_minutes if m > 0]
    avg = f"{sum(mins) / len(mins):.1f} min" if mins else "n/a"
    left = max(0, budget - done)
    eta = ""
    if mins and left:
        eta = (f", roughly {fmt_dur(left * sum(mins) / len(mins) * 60 / max(1, search.n_workers))}"
               f" to go")
    say(f"progress: {done} of {budget} solved candidates (budget checked after "
        f"each whole generation; {search.n_eval - done} more rejected instantly "
        f"for geometry), generation {es.countiter}, "
        f"{fmt_dur(time.time() - search.t_start)} this session, avg {avg} per "
        f"solved candidate{eta}")
    if search.best:
        b = search.best
        say(f"best so far (eval {b['eval']}): tape {float(b['tape_km']):.3f} km, "
            f"B={float(b['B_target_T']):.2f} T at {float(b['I_op_A']):.1f} A, "
            f"unif={float(b['uniformity_pct']):.2f}%, hoop={float(b['hoop_MPa']):.0f} MPa")
    else:
        say("no design satisfies all constraints yet (field >= "
            f"{B_MIN_T} T is usually the first to be reached)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="only check the setup")
    ap.add_argument("--smoke", action="store_true",
                    help="check the setup and evaluate the start design once")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore the saved checkpoint and start a new search")
    ap.add_argument("--workers", type=int, help="parallel candidates")
    ap.add_argument("--max-evals", type=int, help="evaluation budget")
    ap.add_argument("--dry-run", action="store_true",
                    help="test the search machinery with a fake, instant "
                         "evaluator (no physics; writes to runs/"
                         f"{OUT_SUBDIR}_dryrun)")
    args = ap.parse_args()
    if args.dry_run:
        global OUT_DIR, HISTORY_CSV, BEST_CSV, CHECKPOINT, RUN_LOG
        OUT_DIR = OUT_DIR + "_dryrun"
        HISTORY_CSV, BEST_CSV, CHECKPOINT, RUN_LOG = (
            os.path.join(OUT_DIR, f) for f in
            ("history.csv", "best.csv", "checkpoint.pkl", "run.log"))
        os.environ["RUN_OPT_DRY_RUN"] = "1"
        os.environ["RUN_OPT_OUT_DIR"] = OUT_DIR

    os.makedirs(OUT_DIR, exist_ok=True)
    sys.stdout = _Tee(sys.stdout, RUN_LOG)
    banner(f"Racetrack coil optimization  --  {now()}")
    n_cpu, ram_gb = preflight()
    if args.check:
        say("setup check passed. Run without --check to start the search.")
        return

    global N_WORKERS
    if args.workers:
        N_WORKERS = args.workers
    n_workers = pick_workers(n_cpu, ram_gb)
    budget = args.max_evals or MAX_EVALS
    threads = max(1, n_cpu // n_workers)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)       # inherited by worker processes

    if args.smoke:
        banner("Smoke test: evaluating the start design once")
        say(f"this runs the full T-A evaluation on one design ({threads} "
            f"threads); expect 20-120 min depending on the machine")
        _worker_init(RAMP_TIME_S, in_worker=False)
        res = evaluate_candidate(encode(START_DESIGN), 0)
        if res["status"] != "ok":
            die(f"the start design could not be evaluated: {res['error']}",
                "see the messages above; check the conda environment.")
        r = res["r"]
        say(f"smoke test OK: I_op={r['I_op_A']:.1f} A, B={r['B_target_T']:.2f} T, "
            f"uniformity={r['uniformity_pct']:.2f}%, worst load "
            f"{r['ta_worst_load']:.3f}, trials {r['trials']}")
        return

    import cma
    import multiprocessing as mp
    banner("2/3  Setting up the search")
    search = Search()
    es = None if args.fresh else load_checkpoint()
    if es is not None:
        search.load_history()
        say(f"RESUMING a saved search: {search.n_solved} solved candidates and "
            f"{es.countiter} generations already done (use --fresh to restart)")
    else:
        old = [p for p in (CHECKPOINT, HISTORY_CSV, BEST_CSV) if os.path.exists(p)]
        if old:
            arch = os.path.join(OUT_DIR, "archive_" + time.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(arch)
            for p in old:
                os.replace(p, os.path.join(arch, os.path.basename(p)))
            say(f"previous search results moved to {arch}")
        lo = [A_RANGE_M[0], B_RANGE_M[0], gap_floor()] + [TURNS_RANGE[0]] * N_PAIRS
        hi = [A_RANGE_M[1], B_RANGE_M[1], gap_floor() + GAP_EXTRA_RANGE_M] + \
            [TURNS_RANGE[1]] * N_PAIRS
        opts = dict(bounds=[lo, hi], seed=SEED, verbose=-9,
                    CMA_stds=[STEP_A_M, STEP_B_M, STEP_GAP_M] + [STEP_TURNS] * N_PAIRS)
        if POPSIZE:
            opts["popsize"] = POPSIZE
        es = cma.CMAEvolutionStrategy(encode(START_DESIGN), 1.0, opts)
        say(f"NEW search from the {N_LAYERS}-layer start design")
    say(f"objective: minimize tape; require B >= {B_MIN_T} T, uniformity <= "
        f"{UNIFORMITY_MAX_PCT}%, hoop <= {HOOP_MAX_MPA:.0f} MPa, and no cell above "
        f"Jc at the end of a {RAMP_TIME_S/3600:.1f} h constant-power ramp")
    say(f"weights: field {W_FIELD:g}, uniformity {W_UNIFORMITY:g}, hoop {W_HOOP:g}")
    say(f"budget {budget} evaluations, population {es.popsize}, {n_workers} "
        f"workers x {threads} threads")
    say(f"outputs in {OUT_DIR}  (history.csv, best.csv, checkpoint.pkl, run.log)")

    search.t_start = time.time()
    search.n_workers = n_workers

    def _on_signal(signum, frame):
        if search.stop_requested:
            say("second interrupt -- exiting now.")
            os._exit(130)
        search.stop_requested = True
        say("STOP requested: the current generation is abandoned; the search "
            "will resume from the last completed generation next time you run "
            "this file. (Press Ctrl+C again to exit immediately.)")
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    banner("3/3  Searching  (Ctrl+C stops safely; re-run to resume)")
    ctx = mp.get_context("spawn")
    while search.n_solved < budget and not es.stop():
        fits, xs = run_generation(es, search, n_workers, ctx)
        if fits is None:
            break
        es.tell(xs, fits)
        save_checkpoint(es)
        summary(search, es, budget)
        if search.stop_requested:
            break

    banner("Search finished" if not search.stop_requested else "Search stopped")
    summary(search, es, budget)
    if not search.stop_requested and es.stop():
        say(f"CMA-ES stop reason: {dict(es.stop())}")
    say(f"all results: {HISTORY_CSV}")
    if search.best:
        say(f"best design: {BEST_CSV} -- validate it before trusting it "
            f"(README, 'Validating a design').")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:                # noqa: BLE001 - last-resort message
        traceback.print_exc()
        die(f"unexpected {type(e).__name__}: {e}",
            "the search state up to the last completed generation is saved; "
            "re-running resumes it. If this repeats, send run.log to the "
            "maintainer.")
