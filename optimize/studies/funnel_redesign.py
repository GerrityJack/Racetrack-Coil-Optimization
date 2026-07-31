"""
funnel_redesign.py -- 3-stage funnel: cheap geometry filter -> FEM -> T-A
=========================================================================
2026-07-31, fourth and best-founded attempt at the conservative redesign.

**What the three failed attempts taught us.**
  1. Free search, flat seed, huge turn step -> wandered to nonsense.
  2. Champion seed + local steps -> geometry stayed put but the taper still
     FLATTENED (1.25 -> 1.00 over 280 evals). A tape-only objective has a
     permanent gradient toward flat; small steps only slow it.
  3. Taper held FIXED, only its scale searched -> reached 10 T at 0.289 km
     (8 layers) but box uniformity was 1.71 %, a clear FAIL. **The
     champion's taper is NOT a transferable shape** -- proportionally
     identical profiles score 1.66-1.71 % at 8 and 10 layers versus 0.69 %
     at 6. It is tuned to its own stack, not a general recipe.

So uniformity must be IN the loop. It could not be before, because the
only fast metric available was the Bean-state proxy, which this project
found unreliable by ~10x and even anti-correlated with truth.

**The opening.** Decomposing the champion's T-A result shows where its
advantage actually lives:

    design        uniform-J box p2p    T-A total   screening
    champion 6L        1.420 %          0.688 %     -0.732 pp
    8L  s=0.85         2.256 %          1.713 %     -0.543 pp
    10L s=0.675        3.072 %          1.660 %     -1.412 pp

The champion wins on the UNIFORM-CURRENT field (1.42 % vs 2.26/3.07 %),
not on some subtle screening effect -- screening merely improves every
design by 0.5-1.4 pp. And the uniform-J box field is computed by
multi-filament Biot-Savart -- no FEM, no mesh, and cross-validated to
0.01 % between independent meshes. It is *exact for what it computes*,
unlike the Bean proxy. (It is NOT free, though: ~2.4 ms per evaluation
point, so ~140 ms on the coarse screening grid. See the S1_EVALS comment.)

Better still, **uniform-J box uniformity is CURRENT-INDEPENDENT**: for a
uniform current density B is exactly linear in I, so the peak-to-peak /
mean ratio is a pure function of geometry. No I_op, no quench solve, no
FEM is needed to evaluate it.

**Honest limitation, stated up front.** Uniform-J does NOT preserve
ranking against T-A: 10L scores worse on uniform-J (3.07) yet better on
T-A (1.66) than 8L (2.26 -> 1.71), because the screening contribution
varies between designs. So it is used ONLY as a generous FILTER (keep
designs whose uniform-J is at least as good as the champion's 1.42 %),
never as the final word. Every survivor is still T-A validated.

**The funnel.**
  Stage 1 (~140 ms/design) CMA-ES over (a, b-a, per-pair turns) for every
                        even layer count, gap pinned at its face-gap floor.
                        Minimizes tape subject to uniform-J box p2p <=
                        UNIF_J_MAX and enough field at a nominal current.
                        ~2500 evaluations per layer count, ~6 min.
  Stage 2 (~5 s/design) full optimize_geometry.evaluate() on the best
                        survivors under the validated `kim` Ic model:
                        real quench-limited I_op, B_target, hoop.
  Stage 3 (~3-5 min)    T-A box uniformity on the finalists, 2 independent
                        meshes each. This is the only number that decides.

Run:
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \\
        optimize/studies/funnel_redesign.py
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

PYTHON_BIN = "/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3"
RUN_DIR = os.path.join(_ROOT, "optimize", "runs", "funnel_redesign")
TA_DIR = os.path.join(RUN_DIR, "ta_logs")
os.makedirs(TA_DIR, exist_ok=True)
LOG_PATH = os.path.join(RUN_DIR, "log.txt")
S1_CSV = os.path.join(RUN_DIR, "stage1_survivors.csv")
S2_CSV = os.path.join(RUN_DIR, "stage2_fem.csv")
S3_CSV = os.path.join(RUN_DIR, "stage3_final.csv")

IC_EXTRAP = "kim"
IC_CHECK = "scaling:45"
FRAC_IC = 0.625
B_FLOOR = 10.0
HOOP_MAX = 400.0
UNIF_LIMIT = 1.0                 # the real (T-A) target
UNIF_J_MAX = 1.45                # stage-1 filter: at least as good as the
                                 # champion's own uniform-J (1.420%)
LAYER_COUNTS = [6, 8, 10, 12]
# 2026-07-31: stage 1 was FIRST written as brute-force random sampling on the
# claim that each evaluation cost "milliseconds". MEASURED, it is ~500 ms --
# the multi-filament Biot-Savart costs ~2.4 ms per evaluation POINT, and
# 80,000 samples projected to 11.5 HOURS. (FILAMENT_TURNS_PER_GROUP barely
# matters: 450 -> 415 ms from 100 -> 500, so the cost is per-point, not
# per-filament.) Two fixes: a coarser 11x5 box grid (137 ms/eval, only
# -0.055pp error on the champion -- fine for a FILTER whose survivors are
# all T-A validated anyway), and CMA-ES instead of random sampling, which
# needs ~2500 evaluations rather than 20,000 to optimize 5-8 variables.
# Net: ~6 min per layer count instead of ~3 hours.
S1_EVALS = 2500                  # stage-1 CMA-ES evaluations per layer count
S1_NX, S1_NY = 11, 5             # stage-1 box grid (coarse; filter only)
I_NOM_A = 195.0                  # nominal current for the stage-1 field screen.
                                 # Real quench-limited I_op came out 195-216 A
                                 # across every design measured under `kim` at
                                 # 62.5% Ic, so screening at the LOW end keeps
                                 # designs that stage 2 will likely confirm.
N_STAGE2 = 8                     # FEM evaluations per layer count
N_STAGE3 = 5                     # T-A validations overall

# geometry bounds for stage 1
A_RANGE = (0.020, 0.032)
STRAIGHT_RANGE = (0.005, 0.014)  # b - a
PAIR_RANGE = (1, 460)
MIN_BEND_M = 0.0075
MIN_FACE_GAP_M = 0.003
SEED = 20260731


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ── stage 1: cheap, current-independent geometry filter ─────────────────────

def make_stage1(params, cfg):
    from coil2_field import compute_both_coils_field_multilayer
    xs = np.linspace(-cfg.TARGET_X_M / 2, cfg.TARGET_X_M / 2, S1_NX)
    ys = np.linspace(-cfg.TARGET_Y_M / 2, cfg.TARGET_Y_M / 2, S1_NY)
    X, Y = np.meshgrid(xs, ys)

    def evaluate(a, b, gap, n_turns):
        """(uniform-J box p2p %, |Bz| mean per amp-turn, tape_km) or None."""
        pack = max(n_turns) * params.t
        if a - pack / 2.0 < MIN_BEND_M or (b - a) < 0.005:
            return None
        params.a = a; params.b = b; params.coil_half_gap = gap
        params.n_turns = list(n_turns); params.I_design = 1.0
        try:
            params.recompute_derived()
        except AssertionError:
            return None
        pts = np.column_stack([X.ravel(), Y.ravel(),
                               np.full(X.size, gap)])
        B = compute_both_coils_field_multilayer(pts, I_per_turn=1.0)
        mag = np.linalg.norm(B, axis=1)
        mean = mag.mean()
        if mean <= 0:
            return None
        return ((mag.max() - mag.min()) / mean * 100.0,
                abs(B[:, 2].mean()), params.tape_length_m / 1e3)

    return evaluate


def stage1(params, cfg, rng):
    """CMA-ES on the cheap, current-independent metric. Variables:
    [a, straight (=b-a), pair_1..pair_Npairs]. gap is pinned at the exact
    face-gap floor for that layer count (the perturbation study showed that
    IS its uniformity optimum, and it keeps coil 2 as close as allowed)."""
    import cma
    ev = make_stage1(params, cfg)
    survivors = []
    for n_layers in LAYER_COUNTS:
        n_pairs = n_layers // 2
        gap = n_layers * params.w / 2.0 + MIN_FACE_GAP_M / 2.0
        found = []

        def fitness(x):
            a = float(x[0])
            straight = float(x[1])
            b = a + straight
            pairs = [max(1, int(round(v))) for v in x[2:]]
            n_turns = [v for p in pairs for v in (p, p)]
            if not (A_RANGE[0] <= a <= A_RANGE[1]) or \
                    not (STRAIGHT_RANGE[0] <= straight <= STRAIGHT_RANGE[1]):
                return 1e3
            r = ev(a, b, gap, n_turns)
            if r is None:
                return 1e3
            unif, bz_per_at, tape = r
            Bz = bz_per_at * I_NOM_A
            pen = 0.0
            if unif > UNIF_J_MAX:
                pen += 50.0 * (unif - UNIF_J_MAX) ** 2
            if Bz < B_FLOOR:
                pen += 50.0 * (B_FLOOR - Bz) ** 2
            if pen == 0.0:
                found.append((tape, unif, bz_per_at, a, b, gap, n_turns))
            return tape + pen

        x0 = [0.0230, 0.0060] + [260.0] * n_pairs
        std0 = [0.0020, 0.0015] + [70.0] * n_pairs
        es = cma.CMAEvolutionStrategy(
            x0, 1.0, dict(CMA_stds=std0, popsize=14, seed=SEED + n_layers,
                          maxfevals=S1_EVALS, verbose=-9,
                          bounds=[[A_RANGE[0], STRAIGHT_RANGE[0]]
                                  + [PAIR_RANGE[0]] * n_pairs,
                                  [A_RANGE[1], STRAIGHT_RANGE[1]]
                                  + [PAIR_RANGE[1]] * n_pairs]))
        t0 = time.monotonic()
        while not es.stop():
            xs = es.ask()
            es.tell(xs, [fitness(x) for x in xs])
        found.sort(key=lambda t: t[0])
        keep, seen = [], set()
        for t in found:
            key = round(t[0], 4)
            if key in seen:
                continue
            seen.add(key)
            keep.append(t)
            if len(keep) >= 40:
                break
        _log(f"  n_layers={n_layers}: {len(found)} feasible on the cheap "
             f"metric, keeping {len(keep)} "
             f"({(time.monotonic()-t0)/60:.1f} min)")
        if keep:
            _log(f"    best cheap design: tape={keep[0][0]:.4f}km "
                 f"unifJ={keep[0][1]:.3f}% "
                 f"pairs={[keep[0][6][2*i] for i in range(n_pairs)]}")
        for t in keep:
            survivors.append(dict(n_layers=n_layers, tape_km=t[0],
                                  unifJ_pct=t[1], bz_per_At=t[2],
                                  a=t[3], b=t[4], gap=t[5], n_turns=t[6]))
    return survivors


def main():
    t0 = time.monotonic()
    open(LOG_PATH, "w").close()
    from mpi4py import MPI
    import params, opt_config as cfg
    import optimize_geometry as og
    from ic_extrapolation import make_ic_model
    comm = MPI.COMM_WORLD
    rng = np.random.default_rng(SEED)

    _log("Funnel redesign: cheap geometry filter -> FEM -> T-A")
    _log(f"  Ic model={IC_EXTRAP}, operating point={FRAC_IC:.1%} of local Ic")
    _log(f"  stage-1 filter: uniform-J box p2p <= {UNIF_J_MAX}% "
         f"(champion's own = 1.420%)")

    _log("")
    _log("STAGE 1 (cheap, current-independent geometry filter)")
    survivors = stage1(params, cfg, rng)
    if not survivors:
        _log("no geometry passed the uniform-J filter -- widen it and retry")
        return
    with open(S1_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_layers", "tape_km", "unifJ_pct",
                                          "bz_per_At", "a", "b", "gap",
                                          "n_turns"])
        w.writeheader()
        for s in survivors:
            w.writerow({**s, "n_turns": json.dumps(s["n_turns"])})
    _log(f"  {len(survivors)} survivors -> {S1_CSV}")

    _log("")
    _log(f"STAGE 2 (full FEM evaluate, {IC_EXTRAP} Ic model)")
    ic = make_ic_model(IC_EXTRAP)
    cfg.SAFETY_FACTOR = 1.0 / FRAC_IC
    rows = []
    for n_layers in LAYER_COUNTS:
        subset = [s for s in survivors if s["n_layers"] == n_layers]
        # order by tape but skip the ones with too little field to ever reach
        # 10 T -- bz_per_At is the cheap indicator of that
        subset.sort(key=lambda s: s["tape_km"])
        tried = 0
        for s in subset:
            if tried >= N_STAGE2:
                break
            r = og.evaluate(dict(a=s["a"], b=s["b"], n_turns=s["n_turns"],
                                 coil_half_gap=s["gap"]), ic, comm)
            tried += 1
            if not r.get("feasible"):
                continue
            row = dict(n_layers=n_layers, tape_km=r["tape_km"],
                       B_target_T=r["B_target_T"], hoop_MPa=r["hoop_MPa"],
                       I_op_A=r["I_op_A"], unifJ_pct=s["unifJ_pct"],
                       a_mm=s["a"] * 1e3, b_mm=s["b"] * 1e3,
                       gap_mm=s["gap"] * 1e3,
                       n_turns=json.dumps(s["n_turns"]), box_ptp="")
            rows.append(row)
            flag = "OK " if (r["B_target_T"] >= B_FLOOR
                             and r["hoop_MPa"] <= HOOP_MAX) else "-- "
            _log(f"  {flag}n={n_layers} tape={r['tape_km']:.4f} "
                 f"B={r['B_target_T']:5.2f}T hoop={r['hoop_MPa']:4.0f} "
                 f"unifJ={s['unifJ_pct']:.2f}% "
                 f"pairs={[s['n_turns'][2*i] for i in range(n_layers//2)]}")
    with open(S2_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    ok = [r for r in rows if r["B_target_T"] >= B_FLOOR
          and r["hoop_MPa"] <= HOOP_MAX]
    if not ok:
        _log("")
        _log("No stage-1 survivor reached 10 T under the conservative Ic "
             "model. The uniform-J filter and the field requirement are in "
             "tension -- relax UNIF_J_MAX or widen the search.")
        return
    ok.sort(key=lambda r: r["tape_km"])
    _log(f"  {len(ok)} designs meet 10 T and hoop")

    _log("")
    _log(f"STAGE 3 (T-A box uniformity, the deciding number)")
    finalists = ok[:N_STAGE3]
    for i, r in enumerate(finalists):
        d = dict(a=r["a_mm"] / 1e3, b=r["b_mm"] / 1e3,
                 coil_half_gap=r["gap_mm"] / 1e3,
                 n_turns=json.loads(r["n_turns"]), I_design=r["I_op_A"])
        boxes = ta_validate(d, f"f{i+1}_n{r['n_layers']:02d}")
        r["box_ptp"] = max(boxes) if boxes else float("nan")
        _log(f"  n={r['n_layers']} tape={r['tape_km']:.4f} "
             f"unifJ={r['unifJ_pct']:.2f}% -> T-A box = "
             f"{', '.join(f'{b:.3f}' for b in boxes) or 'FAILED'}  "
             f"{'PASS' if boxes and max(boxes) <= UNIF_LIMIT else 'FAIL'}")
        with open(S3_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(finalists[0].keys()))
            w.writeheader(); w.writerows(finalists)

    passing = [r for r in finalists
               if r["box_ptp"] == r["box_ptp"] and r["box_ptp"] <= UNIF_LIMIT]
    _log("")
    if not passing:
        _log("No finalist passed T-A box uniformity.")
        _log("Best T-A seen: " + ", ".join(
            f"n={r['n_layers']} {r['box_ptp']:.3f}%" for r in finalists))
        return
    passing.sort(key=lambda r: r["tape_km"])
    w = passing[0]
    _log(f"WINNER: n_layers={w['n_layers']} tape={w['tape_km']:.4f}km "
         f"B={w['B_target_T']:.2f}T hoop={w['hoop_MPa']:.0f}MPa "
         f"box={w['box_ptp']:.3f}%")
    _log(f"  a={w['a_mm']:.4f}mm b={w['b_mm']:.4f}mm gap={w['gap_mm']:.4f}mm "
         f"I_op={w['I_op_A']:.2f}A")
    _log(f"  n_turns={w['n_turns']}")

    _log("")
    _log(f"conservative cross-check ('{IC_CHECK}'):")
    ic2 = make_ic_model(IC_CHECK)
    r2 = og.evaluate(dict(a=w["a_mm"] / 1e3, b=w["b_mm"] / 1e3,
                          n_turns=json.loads(w["n_turns"]),
                          coil_half_gap=w["gap_mm"] / 1e3), ic2, comm)
    _log(f"  B_target={r2['B_target_T']:.2f}T I_op={r2['I_op_A']:.1f}A "
         f"({'still >=10T' if r2['B_target_T'] >= 10 else 'BELOW 10T'})")
    _log(f"total {(time.monotonic()-t0)/60:.1f} min")


_TA_RE = re.compile(r"box_ptp_pct=(?P<box>[-\d.]+)")


def ta_validate(design, label, repeats=2):
    env = os.environ.copy()
    env["TA_VALIDATE_JSON"] = json.dumps(dict(label=label, **design))
    env["TA_VALIDATE_REPEATS"] = str(repeats)
    try:
        out = subprocess.run([PYTHON_BIN, "-u", "optimize/ta_validate.py"],
                             cwd=_ROOT, env=env, capture_output=True,
                             text=True, timeout=2700)
        with open(os.path.join(TA_DIR, f"{label}.log"), "w") as f:
            f.write(out.stdout + "\n--- stderr ---\n" + out.stderr)
        return [float(m.group("box")) for m in _TA_RE.finditer(out.stdout)]
    except Exception:
        _log(f"   [{label}] T-A failed:\n{traceback.format_exc()}")
        return []


if __name__ == "__main__":
    main()
