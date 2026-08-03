"""
jitter_margin_design.py -- build-tolerance study of the MARGIN-AWARE design
======================================================================
Does the margin-aware design (a=26.0mm, b=31.4mm, gap=13.7mm,
n_turns=[382,382,478,478,3,3], I_op=196.0A, tape=0.3372km) survive
realistic manufacturing error?

Its predecessor did NOT: 0 of 14 perturbed builds reached 10 T, because
the search had converged exactly onto the B >= 10 T constraint. This
design was built against MARGIN-AWARE constraints derived from that
jitter response (B >= 10.3 T nominal; bend radius >= 7.5mm at a-0.2mm AND
t+2% simultaneously; face gap 3.4mm nominal; straight 5.4mm nominal).
**This run tests whether that margin arithmetic actually holds** -- the
previous design passed every static check and still failed here, so the
claim needs the empirical test, not the derivation.

**Why re-do this rather than reuse the old champion's jitter set.** Two
problems with that earlier test, both fixed here:

  1. It jittered TURN COUNTS by +-2%. That is not a real build error --
     turns are wound and counted, so the number is exact. Including it
     manufactured a failure mode that does not exist.
  2. It omitted TAPE THICKNESS. Real REBCO tape has lot-to-lot thickness
     variation (~+-2% on 75 um), and because pack thickness is
     max(n_i)*t, that error scales the ENTIRE winding pack -- it moves
     every layer's radius and the bend radius at once. It is plausibly
     the dominant geometric error and was simply missing.

Also fixed: each jittered design's quench-limited I_op is recomputed under
the validated `kim` Ic model, because **box uniformity depends on the
OPERATING CURRENT** (the old champion reads 0.688% at 223.9A but 0.268% at
208.7A). Evaluating a perturbed geometry at the nominal current would
measure the wrong thing.

**Error model** (1-sigma-ish, meant as plausible build tolerances):
    a     +-0.20 mm   mandrel / winding-form machining
    b     +-0.20 mm   straight-section length
    gap   +-0.20 mm   coil-to-coil assembly shim
    t     +-2 %       tape thickness, lot variation
    turns  exact      (wound and counted)

**Spec-floor caveat, which this design makes acute.** The nominal design
sits ON three floors: face gap 3.001mm (margin 0.001), straight length
5.041mm (margin 0.041), bend radius 7.815mm (margin 0.315). So roughly
half of any symmetric tolerance distribution lands OUT OF SPEC on gap and
straight length. Those samples are still evaluated (a coil built 0.1mm
tight is physically fine, just outside the stated clearance), but they are
FLAGGED -- the manufacturability conclusion is separate from the
uniformity one, and matters regardless of how the uniformity comes out.
A bend-radius violation is different: that risks cracking the tape, a real
physical failure, and is flagged more strongly.

Two groups:
  `axis`   one variable at a time, +- its tolerance -> identifies which
           error actually matters
  `random` all variables at once -> the realistic combined case

Run:
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \\
        optimize/studies/jitter_new_design.py
"""
import os, sys, csv, json, re, time, subprocess, traceback
from concurrent.futures import ThreadPoolExecutor

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
RUN_DIR = os.path.join(_ROOT, "optimize", "runs", "jitter_margin_design")
TA_DIR = os.path.join(RUN_DIR, "ta_logs")
os.makedirs(TA_DIR, exist_ok=True)
LOG_PATH = os.path.join(RUN_DIR, "log.txt")
CSV_PATH = os.path.join(RUN_DIR, "results.csv")

IC_EXTRAP = "kim"
FRAC_IC = 0.65
B_FLOOR = 10.0
UNIF_LIMIT = 1.0
TA_REPEATS = 2
N_WORKERS = 2
SEED = 731202

# nominal design
NOM = dict(a=0.0260, b=0.0314, gap=0.0137, t=75e-6,
           n_turns=[382, 382, 478, 478, 3, 3])

TOL_A = 0.0002      # m
TOL_B = 0.0002      # m
TOL_GAP = 0.0002    # m
TOL_T_FRAC = 0.02   # fractional

MIN_BEND_M = 0.0075
MIN_STRAIGHT_M = 0.005
MIN_FACE_GAP_M = 0.003
N_LAYERS = 6
W_TAPE = 0.004


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def margins(d):
    pack = max(d["n_turns"]) * d["t"]
    bend = d["a"] - pack / 2.0
    straight = d["b"] - d["a"]
    face = 2.0 * (d["gap"] - N_LAYERS * W_TAPE / 2.0)
    flags = []
    if bend < MIN_BEND_M:
        flags.append(f"BEND {bend*1e3:.3f}<7.5")
    if straight < MIN_STRAIGHT_M:
        flags.append(f"straight {straight*1e3:.3f}<5.0")
    if face < MIN_FACE_GAP_M:
        flags.append(f"gap {face*1e3:.3f}<3.0")
    return bend, straight, face, flags


def build_candidates():
    rng = np.random.default_rng(SEED)
    cands = [dict(label="nominal", group="ref", **{k: NOM[k] for k in NOM})]
    # one-at-a-time
    for name, key, delta in (("a-0.2mm", "a", -TOL_A), ("a+0.2mm", "a", +TOL_A),
                             ("b-0.2mm", "b", -TOL_B), ("b+0.2mm", "b", +TOL_B),
                             ("gap-0.2mm", "gap", -TOL_GAP),
                             ("gap+0.2mm", "gap", +TOL_GAP)):
        d = dict(NOM); d[key] = NOM[key] + delta
        cands.append(dict(label=name, group="axis", **d))
    for name, f in (("t-2pct", 1 - TOL_T_FRAC), ("t+2pct", 1 + TOL_T_FRAC)):
        d = dict(NOM); d["t"] = NOM["t"] * f
        cands.append(dict(label=name, group="axis", **d))
    # all-axes random, normal with sigma = the stated tolerance
    for i in range(6):
        d = dict(NOM)
        d["a"] = NOM["a"] + rng.normal(0, TOL_A)
        d["b"] = NOM["b"] + rng.normal(0, TOL_B)
        d["gap"] = NOM["gap"] + rng.normal(0, TOL_GAP)
        d["t"] = NOM["t"] * (1 + rng.normal(0, TOL_T_FRAC))
        cands.append(dict(label=f"jit{i+1}", group="random", **d))
    return cands


_TA_RE = re.compile(r"box_ptp_pct=(?P<box>[-\d.]+)")


def ta_validate(c):
    design = dict(label=c["label"], a=c["a"], b=c["b"],
                  coil_half_gap=c["gap"], n_turns=c["n_turns"],
                  I_design=c["I_op_A"],
                  mesh={"t": c["t"]})     # tape pitch goes through the same
                                          # params-override hook
    env = os.environ.copy()
    env["TA_VALIDATE_JSON"] = json.dumps(design)
    env["TA_VALIDATE_REPEATS"] = str(TA_REPEATS)
    try:
        out = subprocess.run([PYTHON_BIN, "-u", "optimize/ta_validate.py"],
                             cwd=_ROOT, env=env, capture_output=True,
                             text=True, timeout=2700)
        with open(os.path.join(TA_DIR, f"{c['label']}.log"), "w") as f:
            f.write(out.stdout + "\n--- stderr ---\n" + out.stderr)
        return [float(m.group("box")) for m in _TA_RE.finditer(out.stdout)]
    except Exception:
        _log(f"   [{c['label']}] T-A failed:\n{traceback.format_exc()}")
        return []


def main():
    t0 = time.monotonic()
    open(LOG_PATH, "w").close()
    from mpi4py import MPI
    import params, opt_config as cfg
    import optimize_geometry as og
    from ic_extrapolation import make_ic_model
    comm = MPI.COMM_WORLD

    _log("Build-tolerance (jitter) study of the 2026-07-31 design")
    _log(f"  Ic model={IC_EXTRAP}, operating point={FRAC_IC:.0%} of local Ic")
    _log(f"  tolerances: a,b,gap +-0.2mm; tape thickness +-2%; turns EXACT")
    ic = make_ic_model(IC_EXTRAP)
    cfg.SAFETY_FACTOR = 1.0 / FRAC_IC
    t_nominal = params.t

    cands = build_candidates()
    _log("")
    _log("STEP 1: geometry margins + FEM (each design gets its OWN I_op)")
    for c in cands:
        bend, straight, face, flags = margins(c)
        c.update(bend_mm=bend * 1e3, straight_mm=straight * 1e3,
                 face_mm=face * 1e3, flags=";".join(flags))
        params.t = c["t"]                      # tape pitch is a params global
        try:
            r = og.evaluate(dict(a=c["a"], b=c["b"], n_turns=c["n_turns"],
                                 coil_half_gap=c["gap"]), ic, comm)
        finally:
            params.t = t_nominal
        if not r.get("feasible"):
            _log(f"  {c['label']:<11s} INFEASIBLE ({r.get('reason')})")
            c["feasible"] = False
            continue
        c.update(feasible=True, tape_km=r["tape_km"],
                 B_target_T=r["B_target_T"], hoop_MPa=r["hoop_MPa"],
                 I_op_A=r["I_op_A"])
        _log(f"  {c['label']:<11s} B={r['B_target_T']:6.2f}T "
             f"I_op={r['I_op_A']:6.1f}A hoop={r['hoop_MPa']:4.0f} "
             f"bend={bend*1e3:5.3f} face={face*1e3:5.3f}"
             + (f"  [{c['flags']}]" if flags else ""))

    runnable = [c for c in cands if c.get("feasible")]
    _log("")
    _log(f"STEP 2: T-A box uniformity on {len(runnable)} designs "
         f"({N_WORKERS} parallel, {TA_REPEATS} meshes each)")

    def job(c):
        boxes = ta_validate(c)
        c["box_ptp"] = max(boxes) if boxes else float("nan")
        c["unif_ok"] = bool(boxes) and c["box_ptp"] <= UNIF_LIMIT
        c["field_ok"] = c["B_target_T"] >= B_FLOOR
        _log(f"  {c['label']:<11s} box = "
             f"{', '.join(f'{b:.3f}' for b in boxes) or 'FAILED':<16s} "
             f"B={c['B_target_T']:6.2f}T  "
             f"unif={'OK' if c['unif_ok'] else 'FAIL'} "
             f"field={'OK' if c['field_ok'] else 'FAIL'}")
        _write(cands)
        return c

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        list(ex.map(job, runnable))
    _write(cands)

    # ── summary ─────────────────────────────────────────────────────────────
    nom = next(c for c in cands if c["label"] == "nominal")
    _log("")
    _log(f"{'label':<11}{'group':<8}{'B_T':>7}{'box%':>8}{'dbox':>8}"
         f"{'spec':>6}  notes")
    for c in sorted(runnable, key=lambda c: (c["group"], c["label"])):
        if "box_ptp" not in c:
            continue
        spec = "ok" if not c["flags"] else "OUT"
        _log(f"{c['label']:<11}{c['group']:<8}{c['B_target_T']:>7.2f}"
             f"{c['box_ptp']:>8.3f}{c['box_ptp']-nom['box_ptp']:>+8.3f}"
             f"{spec:>6}  {c['flags']}")

    rnd = [c for c in runnable if c["group"] == "random" and "box_ptp" in c]
    ax = [c for c in runnable if c["group"] == "axis" and "box_ptp" in c]
    _log("")
    if ax:
        worst = max(ax, key=lambda c: abs(c["box_ptp"] - nom["box_ptp"]))
        _log(f"dominant single-axis error: {worst['label']} "
             f"({worst['box_ptp']-nom['box_ptp']:+.3f}pp)")
    if rnd:
        vals = [c["box_ptp"] for c in rnd]
        npass = sum(1 for c in rnd if c["unif_ok"] and c["field_ok"])
        _log(f"all-axes samples: box {min(vals):.3f}-{max(vals):.3f}% "
             f"(nominal {nom['box_ptp']:.3f}%), "
             f"{npass}/{len(rnd)} meet BOTH uniformity and 10 T")
        n_out = sum(1 for c in rnd if c["flags"])
        _log(f"  {n_out}/{len(rnd)} land OUT OF SPEC on a clearance floor "
             f"(the nominal design sits on three of them)")
    _log(f"total {(time.monotonic()-t0)/60:.1f} min -> {CSV_PATH}")


def _write(rows):
    fields = ["label", "group", "a", "b", "gap", "t", "n_turns", "feasible",
              "bend_mm", "straight_mm", "face_mm", "flags", "tape_km",
              "B_target_T", "hoop_MPa", "I_op_A", "box_ptp", "unif_ok",
              "field_ok"]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "n_turns": json.dumps(r["n_turns"])})


if __name__ == "__main__":
    main()
