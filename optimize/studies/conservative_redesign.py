"""
conservative_redesign.py -- find a design that meets 10 T under a REAL Ic model
==============================================================================
2026-07-31.

**Why.** Every design in this project's history was optimized with
`clip_B=True`, which flat-clamps Ic at the measured 8 T ceiling. Hold-out
validation against the measured data itself
(`optimize/studies/ic_extrapolation_validation.py`) showed that clamp
over-predicts Ic by **+26.7 % at a 1.6x extrapolation and +54 % at 2.7x** --
it is badly optimistic, not conservative. Under a validated model the
current champion reaches only 9.40 T (kim) / 8.94 T (scaling, 45 T) at 65 %
of Ic, versus the 10.21 T the flat clamp claimed at 55 %. The 10 T
constraint is genuinely unmet.

**Ic model used here: `kim`** -- Jc0/(1+B/B0), the best performer in
hold-out testing (MAPE 4.1 %, bias -3.3 %, i.e. very slightly conservative).
`scaling:45` is used afterwards as a conservative cross-check on the winner.

**Operating point: 62.5 % of local Ic** (`SAFETY_FACTOR = 1.6`), the
midpoint of the 60-65 % band, chosen the same way the original 55 % was
picked as the midpoint of 50-60 %.

**Why sweep the layer count.** A quick scan showed that simply adding turns
to the 6-layer design is an inefficient way to buy field: pack thickness is
`max(n_i)*t`, so more turns per layer thickens the pack, and the 7.5 mm
bend-radius floor then forces `a` outward, which costs field efficiency AND
raises the peak field (lowering Ic and hence I_op). Doubling the tape that
way bought only ~+1 T. Adding LAYERS instead adds turns at CONSTANT pack
thickness, so it does not fight the bend-radius constraint. Hence a
discrete outer loop over even layer counts (odd counts are unbuildable
under double-pancake pairing).

**Two-phase, because the fitness has no uniformity signal.** Phase A
minimizes tape subject to B_target/hoop only (`cmaes_search.py`'s objective
is deliberately uniformity-free -- two guessed proxies were both wrong).
Phase B then T-A-validates the per-layer-count winners for true box
uniformity, and the reported design is the lowest-tape one that actually
passes everything.

Run:
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \\
        optimize/studies/conservative_redesign.py

Progress: optimize/runs/conservative_redesign/log.txt
Per-job CMA-ES logs: .../jobs/n_layers_NN.log
"""
import os, sys, csv, json, re, time, subprocess, traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "optimize"), os.path.join(_ROOT, "physics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import params

PYTHON_BIN = "/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3"
RUN_DIR = os.path.join(_ROOT, "optimize", "runs", "conservative_redesign")
JOBS_DIR = os.path.join(RUN_DIR, "jobs")
TA_DIR = os.path.join(RUN_DIR, "ta_logs")
for d in (RUN_DIR, JOBS_DIR, TA_DIR):
    os.makedirs(d, exist_ok=True)
LOG_PATH = os.path.join(RUN_DIR, "log.txt")
SUMMARY_CSV = os.path.join(RUN_DIR, "summary.csv")

IC_EXTRAP = "kim"
IC_EXTRAP_CHECK = "scaling:45"
SAFETY_FACTOR = 1.6              # 62.5% of local Ic
LAYER_COUNTS = [6, 8, 10, 12]
MAX_EVALS = 1200
N_WORKERS = 6
SEED = 731
PER_JOB_MIN = {6: 35, 8: 40, 10: 45, 12: 50}
TOTAL_BUDGET_H = 4.5
N_TURN_BOUNDS = "1,600"

# ── staying ANCHORED on the champion (2026-07-31, after a false start) ──────
# The fitness function has NO uniformity signal (deliberately -- two guessed
# proxies were both wrong). local_polish.py proved what that means: a
# tape-only search reliably FLATTENS the turn taper toward equal pairs,
# which is 17% cheaper in tape and catastrophic for box uniformity
# (3.6-8.6% vs a 1% target). The champion's steep taper is doing essential
# uniformity work.
#
# Therefore this search must stay LOCAL around the champion's validated
# SHAPE, not explore freely. A first attempt did neither and was scrapped:
# it seeded a FLAT profile ([340,340,2] vs the champion's [295,369,2]) and,
# because CMAES_TIGHT_N_BOUNDS_OVERRIDE was set but CMAES_N_STD0_OVERRIDE
# was NOT, the turn step fell back to the bound-range default
# (0.3 * 699 ~= 210 turns). Within 103 evaluations it had wandered to
# profiles like [544,148,167] at ~2x the champion's tape -- the same
# oversized-step failure that wrecked the n_layers=9 polish.
#
# Fixes: seed from the champion's own pair values, extend to more layers by
# REPEATING the large pair (adding turns via LAYERS, at constant pack
# thickness, so the bend-radius floor never binds), pin the gap exactly at
# its face-gap floor (the perturbation study showed that IS the uniformity
# optimum for that axis), and use small PROPORTIONAL step sizes.
CH_PAIR_INNER = 295              # champion pair nearest the midplane
CH_PAIR_MAIN = 369               # champion's large pair
CH_PAIR_TAIL = 2                 # champion's vestigial outermost pair
CH_A = 0.022227029065529628
CH_B = 0.02726822715975084
N_STD0 = 25.0                    # turn step [turns] -- LOCAL, not 210
STD0_FRAC = 0.05                 # a/b step as a fraction of their own value

# constraint floors (mirror cmaes_search.geometry_violation)
MIN_BEND_M = 0.0075
MIN_STRAIGHT_M = 0.005041        # the champion's own straight length
MIN_FACE_GAP_M = 0.003
MIN_A_M = 0.0222                 # cfg.CMAES_MIN_A_M soft floor


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def smart_x0(n_layers):
    """Warm start ANCHORED on the champion's validated taper (see the
    CH_PAIR_* comment block). At 6 layers this reproduces the champion
    exactly; at higher layer counts the LARGE pair is repeated, so extra
    turns arrive via extra layers at constant pack thickness rather than by
    thickening any layer (which is what makes the bend-radius floor push
    `a` outward and kill the returns). Gap sits exactly on its face-gap
    floor -- the perturbation study showed that IS its uniformity optimum."""
    n_pairs = n_layers // 2
    pairs = [CH_PAIR_INNER] + [CH_PAIR_MAIN] * (n_pairs - 2) + [CH_PAIR_TAIL]
    n_turns = [v for p in pairs for v in (p, p)]
    pack = max(n_turns) * params.t
    a = max(MIN_A_M, CH_A, MIN_BEND_M + pack / 2.0 + 0.0003)
    b = a + (CH_B - CH_A)
    gap = n_layers * params.w / 2.0 + MIN_FACE_GAP_M / 2.0
    return dict(a=round(a, 8), b=round(b, 8), coil_half_gap=round(gap, 8),
                n_turns=n_turns)


def check_x0(n_layers, x0):
    """Verify the warm start is buildable AND inside the bounds pycma will
    enforce -- a warm start outside bounds raises ValueError in geno(), a
    mistake this project has hit twice."""
    n_turns = x0["n_turns"]
    a, b, gap = x0["a"], x0["b"], x0["coil_half_gap"]
    pack = max(n_turns) * params.t
    bend = a - pack / 2.0
    straight = b - a
    face = 2.0 * (gap - n_layers * params.w / 2.0)
    lo, hi = (int(v) for v in N_TURN_BOUNDS.split(","))
    ok = (bend >= MIN_BEND_M - 1e-9 and straight >= 0.005 - 1e-9
          and face >= MIN_FACE_GAP_M - 1e-9 and a >= MIN_A_M - 1e-9
          and all(lo <= n <= hi for n in n_turns))
    return ok, dict(bend_mm=bend * 1e3, straight_mm=straight * 1e3,
                    face_mm=face * 1e3, max_turns=max(n_turns))


def run_job(n_layers, deadline):
    x0 = smart_x0(n_layers)
    ok, m = check_x0(n_layers, x0)
    _log(f"n_layers={n_layers}: x0 a={x0['a']*1e3:.2f}mm b={x0['b']*1e3:.2f}mm "
         f"gap={x0['coil_half_gap']*1e3:.2f}mm turns={x0['n_turns']}")
    _log(f"   bend={m['bend_mm']:.2f}mm straight={m['straight_mm']:.2f}mm "
         f"face={m['face_mm']:.2f}mm  buildable={ok}")
    if not ok:
        _log(f"   SKIPPED -- warm start not buildable/in-bounds")
        return None

    res_csv = os.path.join(JOBS_DIR, f"n{n_layers:02d}_results.csv")
    hist_csv = os.path.join(JOBS_DIR, f"n{n_layers:02d}_history.csv")
    for p in (res_csv, hist_csv):
        if os.path.exists(p):
            os.remove(p)     # never read a stale previous run's file

    env = os.environ.copy()
    env["CMAES_IC_EXTRAP"] = IC_EXTRAP
    env["SAFETY_FACTOR_OVERRIDE"] = str(SAFETY_FACTOR)
    env["CMAES_SWEEP_OVERRIDE_JSON"] = json.dumps(
        dict(x0=x0, seed=SEED + n_layers, max_evals=MAX_EVALS))
    env["CMAES_N_WORKERS_OVERRIDE"] = str(N_WORKERS)
    env["CMAES_TIGHT_N_BOUNDS_OVERRIDE"] = N_TURN_BOUNDS
    # keep the search LOCAL -- without these the step sizes fall back to
    # bound-range defaults (~210 turns) and the search flattens the taper
    env["CMAES_N_STD0_OVERRIDE"] = str(N_STD0)
    env["CMAES_A_STD0_OVERRIDE"] = str(x0["a"] * STD0_FRAC)
    env["CMAES_B_STD0_OVERRIDE"] = str(x0["b"] * STD0_FRAC)
    env["CMAES_OUT_CSV_OVERRIDE"] = res_csv
    env["CMAES_OUT_LOG_OVERRIDE"] = hist_csv

    cap_s = min(PER_JOB_MIN[n_layers] * 60, max(60, deadline - time.monotonic()))
    log_path = os.path.join(JOBS_DIR, f"n_layers_{n_layers:02d}.log")
    t0 = time.monotonic()
    with open(log_path, "w") as f:
        p = subprocess.Popen([PYTHON_BIN, "-u", "optimize/cmaes_search.py"],
                             cwd=_ROOT, env=env, stdout=f,
                             stderr=subprocess.STDOUT)
        try:
            p.wait(timeout=cap_s)
        except subprocess.TimeoutExpired:
            _log(f"   time cap ({cap_s/60:.0f} min) hit -- terminating "
                 f"(progress preserved by incremental flush)")
            p.terminate()
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
    dt = (time.monotonic() - t0) / 60
    best = read_best(hist_csv)
    if best is None:
        _log(f"   n_layers={n_layers}: no constraint-satisfying design "
             f"({dt:.1f} min)")
        return None
    _log(f"   n_layers={n_layers}: tape={best['tape_km']:.4f}km "
         f"B={best['B_target_T']:.2f}T hoop={best['hoop_MPa']:.0f}MPa "
         f"a={best['a_mm']:.2f}mm turns={best['n_turns']} ({dt:.1f} min)")
    return best


def read_best(hist_csv):
    """Lowest-tape evaluation that satisfies B_target and hoop. Read from the
    HISTORY (not results.csv) so the choice is explicit here rather than
    depending on the search's own internal notion of 'best'."""
    if not os.path.exists(hist_csv):
        return None
    best = None
    with open(hist_csv) as f:
        for r in csv.DictReader(f):
            if str(r.get("feasible", "")).strip().lower() not in ("true", "1"):
                continue
            try:
                tape = float(r["tape_km"]); B = float(r["B_target_T"])
                hoop = float(r["hoop_MPa"])
            except (KeyError, ValueError, TypeError):
                continue
            if B < 10.0 or hoop > 400.0:
                continue
            if best is None or tape < best["tape_km"]:
                best = dict(tape_km=tape, B_target_T=B, hoop_MPa=hoop,
                            a_mm=float(r["a_mm"]), b_mm=float(r["b_mm"]),
                            gap_mm=float(r["gap_mm"]),
                            n_turns=json.loads(r["n_turns"]),
                            I_op_A=float(r["I_op_A"]))
    return best


_TA_RE = re.compile(r"box_ptp_pct=(?P<box>[-\d.]+)")


def ta_validate(design, label, repeats=2):
    env = os.environ.copy()
    env["TA_VALIDATE_JSON"] = json.dumps(dict(label=label, **design))
    env["TA_VALIDATE_REPEATS"] = str(repeats)
    lp = os.path.join(TA_DIR, f"{label}.log")
    try:
        out = subprocess.run([PYTHON_BIN, "-u", "optimize/ta_validate.py"],
                             cwd=_ROOT, env=env, capture_output=True,
                             text=True, timeout=2700)
        with open(lp, "w") as f:
            f.write(out.stdout + "\n--- stderr ---\n" + out.stderr)
        return [float(m.group("box")) for m in _TA_RE.finditer(out.stdout)]
    except Exception:
        _log(f"   [{label}] T-A failed:\n{traceback.format_exc()}")
        return []


def main():
    t_start = time.monotonic()
    deadline = t_start + TOTAL_BUDGET_H * 3600
    open(LOG_PATH, "w").close()
    _log("Conservative redesign: 10 T under a VALIDATED Ic extrapolation")
    _log(f"  Ic model = {IC_EXTRAP} (hold-out MAPE 4.1%, bias -3.3%)")
    _log(f"  operating point = {1/SAFETY_FACTOR:.1%} of local Ic "
         f"(SAFETY_FACTOR={SAFETY_FACTOR})")
    _log(f"  layer counts = {LAYER_COUNTS}, {MAX_EVALS} evals each")

    results = {}
    for n in LAYER_COUNTS:
        if time.monotonic() >= deadline:
            _log(f"total budget exhausted -- skipping n_layers={n}")
            continue
        _log("")
        try:
            r = run_job(n, deadline)
        except Exception:
            _log(f"n_layers={n} FAILED:\n{traceback.format_exc()}")
            r = None
        if r:
            results[n] = r
            with open(SUMMARY_CSV, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["n_layers", "tape_km",
                    "B_target_T", "hoop_MPa", "a_mm", "b_mm", "gap_mm",
                    "I_op_A", "n_turns", "box_ptp"])
                w.writeheader()
                for k, v in sorted(results.items()):
                    w.writerow(dict(n_layers=k, box_ptp="", **v))

    if not results:
        _log("no layer count produced a design meeting 10 T -- stopping")
        return

    _log("")
    _log("PHASE B: T-A box-uniformity validation of each layer count's best")
    for n, r in sorted(results.items()):
        design = dict(a=r["a_mm"] / 1e3, b=r["b_mm"] / 1e3,
                      coil_half_gap=r["gap_mm"] / 1e3,
                      n_turns=r["n_turns"], I_design=r["I_op_A"])
        boxes = ta_validate(design, f"n{n:02d}")
        r["box_ptp"] = max(boxes) if boxes else float("nan")
        _log(f"  n_layers={n}: box p2p = "
             f"{', '.join(f'{b:.3f}' for b in boxes) or 'FAILED'}"
             f"  -> {'PASS' if boxes and max(boxes) <= 1.0 else 'FAIL'}")

    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_layers", "tape_km", "B_target_T",
            "hoop_MPa", "a_mm", "b_mm", "gap_mm", "I_op_A", "n_turns",
            "box_ptp"])
        w.writeheader()
        for k, v in sorted(results.items()):
            w.writerow(dict(n_layers=k, **v))

    _log("")
    _log(f"{'n_layers':>9}{'tape_km':>9}{'B_T':>7}{'hoop':>6}{'a_mm':>7}"
         f"{'box%':>8}  verdict")
    passing = []
    for n, r in sorted(results.items()):
        ok = r.get("box_ptp", 9e9) <= 1.0
        if ok:
            passing.append((r["tape_km"], n, r))
        _log(f"{n:>9}{r['tape_km']:>9.4f}{r['B_target_T']:>7.2f}"
             f"{r['hoop_MPa']:>6.0f}{r['a_mm']:>7.2f}"
             f"{r.get('box_ptp', float('nan')):>8.3f}  "
             f"{'PASS' if ok else 'FAIL (uniformity)'}")
    if passing:
        passing.sort()
        tape, n, r = passing[0]
        _log("")
        _log(f"WINNER: n_layers={n}, tape={tape:.4f}km, "
             f"B={r['B_target_T']:.2f}T, hoop={r['hoop_MPa']:.0f}MPa, "
             f"box={r['box_ptp']:.3f}%")
        _log(f"  a={r['a_mm']:.4f}mm b={r['b_mm']:.4f}mm "
             f"gap={r['gap_mm']:.4f}mm I_op={r['I_op_A']:.2f}A")
        _log(f"  n_turns={r['n_turns']}")
        _log("")
        _log(f"NEXT: cross-check this design under the conservative "
             f"'{IC_EXTRAP_CHECK}' Ic model before adopting it.")
    else:
        _log("")
        _log("No design passed BOTH 10 T and box uniformity.")
    _log(f"total {(time.monotonic()-t_start)/60:.1f} min -> {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
