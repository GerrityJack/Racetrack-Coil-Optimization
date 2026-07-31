"""
ta_in_loop_search.py -- T-A in the loop, anchored on the only passing design
=============================================================================
2026-07-31, fifth attempt. The previous four all tried to avoid paying for
T-A during the search, and all four failed for the same underlying reason:
**no cheap surrogate for box uniformity exists.** The tally:

    proxy                     verdict
    on-axis SCIF              anti-correlated with truth (best on-axis had
                              the WORST box uniformity)
    peak turns per pair       built from the on-axis data; also wrong
    Bean-state correction     unreliable by up to ~10x for compact coils
    uniform-J box field       NEW this session, and also insufficient:
                              screening ranges from -1.50 to +0.57 pp, so a
                              design at uniform-J 1.43% can land anywhere
                              from 0.69% to 2.00% T-A. Three funnel
                              finalists at uniform-J 1.41-1.45% scored
                              1.89-2.12% T-A. All FAIL.

So this study puts the real T-A solve in the loop and simply accepts the
cost (~3-5 min per candidate). That limits us to tens of evaluations, so
the candidates are chosen by physical reasoning rather than blind search.

**Anchor: the current champion**, the ONLY design in this project's history
with a validated passing box uniformity (0.688%). Under the validated `kim`
Ic model at 65% of local Ic it reaches 9.40 T -- about 6% short of the
10 T floor. The question is therefore narrow and well-posed: *which
direction out of the champion buys ~6% more field without spending its
uniformity margin?*

**Directions tested, and why each is plausible** (all from measured data,
not speculation):
  - `turns_scale`: the perturbation study measured [299,299,392,392,2,2]
    (+4% turns) at 0.774% uniformity -- still passing -- so there is turn
    headroom before the bend radius binds at max_pair = 392 for a = 22.227mm.
  - `a_grow`: the perturbation study also showed uniformity IMPROVES as `a`
    grows (0.828% -> 0.675% at +0.5mm -> 0.487% at +1.0mm). Growing `a`
    simultaneously raises the bend-radius ceiling on turns
    (max_pair = 2*(a - 7.5mm)/t), so it buys uniformity margin AND permits
    the extra turns that buy field. This is the most promising direction
    and is sampled most densely.
  - Each `a_grow` variant scales the turn profile up to just inside its own
    bend-radius ceiling, so the field gain is maximal for that radius.

Everything keeps the champion's turn RATIO (295:369:2) since that shape is
validated; `gap` stays pinned at the 3mm face-gap floor (its measured
uniformity optimum); `b - a` stays at the champion's straight length.

Operating point is 65% of local Ic -- the TOP of the requested 60-65%
band, since the gap to close is a field shortfall.

Run:
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \\
        optimize/studies/ta_in_loop_search.py
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
RUN_DIR = os.path.join(_ROOT, "optimize", "runs", "ta_in_loop")
TA_DIR = os.path.join(RUN_DIR, "ta_logs")
os.makedirs(TA_DIR, exist_ok=True)
LOG_PATH = os.path.join(RUN_DIR, "log.txt")
CSV_PATH = os.path.join(RUN_DIR, "results.csv")

IC_EXTRAP = "kim"
IC_CHECK = "scaling:45"
FRAC_IC = 0.65                   # top of the requested 60-65% band
B_FLOOR = 10.0
HOOP_MAX = 400.0
UNIF_LIMIT = 1.0
TA_REPEATS = 2
N_TA_WORKERS = 2                 # T-A solves in parallel

CH_A = 0.022227029065529628
CH_B = 0.02726822715975084
CH_STRAIGHT = CH_B - CH_A
CH_RATIO = (295.0, 369.0, 2.0)   # champion pair ratio, preserved
MIN_BEND_M = 0.0075
MIN_FACE_GAP_M = 0.003
N_LAYERS = 6
TAPE_MARGIN = 0.0003             # keep max_pair this far inside the ceiling


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def max_pair_for_a(a, t):
    """Largest allowed max-pair turn count at radius `a` (bend-radius floor)."""
    return int((a - MIN_BEND_M - TAPE_MARGIN) * 2.0 / t)


def build_candidates(params):
    """Champion-ratio profiles at a range of radii, each scaled to just
    inside that radius's own bend-radius ceiling."""
    gap = N_LAYERS * params.w / 2.0 + MIN_FACE_GAP_M / 2.0
    cands = [dict(label="champion_ref", a=CH_A, b=CH_B,
                  n_turns=[295, 295, 369, 369, 2, 2], gap=gap)]
    for da_mm, fill in ((0.0, 1.00), (0.0, 0.995),
                        (0.5, 1.00), (1.0, 1.00), (1.5, 1.00),
                        (2.0, 1.00), (3.0, 1.00), (4.0, 1.00)):
        a = CH_A + da_mm * 1e-3
        cap = max_pair_for_a(a, params.t)
        scale = fill * cap / CH_RATIO[1]
        pairs = [max(1, int(round(r * scale))) for r in CH_RATIO]
        if pairs[1] > cap:
            pairs[1] = cap
        n_turns = [v for p in pairs for v in (p, p)]
        lbl = f"a+{da_mm:.1f}mm" + ("" if fill == 1.0 else f"_f{fill}")
        if da_mm == 0.0 and fill == 1.0:
            lbl = "a+0.0mm_maxturns"
        cands.append(dict(label=lbl, a=a, b=a + CH_STRAIGHT,
                          n_turns=n_turns, gap=gap))
    return cands


_TA_RE = re.compile(r"box_ptp_pct=(?P<box>[-\d.]+)")


def ta_validate(design, label):
    env = os.environ.copy()
    env["TA_VALIDATE_JSON"] = json.dumps(dict(label=label, **design))
    env["TA_VALIDATE_REPEATS"] = str(TA_REPEATS)
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


def main():
    t0 = time.monotonic()
    open(LOG_PATH, "w").close()
    from concurrent.futures import ThreadPoolExecutor
    from mpi4py import MPI
    import params, opt_config as cfg
    import optimize_geometry as og
    from ic_extrapolation import make_ic_model
    comm = MPI.COMM_WORLD

    _log("T-A-in-the-loop search anchored on the champion")
    _log(f"  Ic model={IC_EXTRAP}, operating point={FRAC_IC:.0%} of local Ic")
    _log(f"  NOTE: no cheap uniformity proxy survives -- uniform-J was the "
         f"4th to fail (screening spans -1.50 to +0.57 pp)")

    ic = make_ic_model(IC_EXTRAP)
    cfg.SAFETY_FACTOR = 1.0 / FRAC_IC
    cands = build_candidates(params)

    _log("")
    _log("STEP 1: FEM screen (field / stress) -- cheap, weeds out anything "
         "that cannot reach 10 T before paying for T-A")
    keep = []
    for c in cands:
        r = og.evaluate(dict(a=c["a"], b=c["b"], n_turns=c["n_turns"],
                             coil_half_gap=c["gap"]), ic, comm)
        if not r.get("feasible"):
            _log(f"  {c['label']:<20s} INFEASIBLE ({r.get('reason')})")
            continue
        c.update(tape_km=r["tape_km"], B_target_T=r["B_target_T"],
                 hoop_MPa=r["hoop_MPa"], I_op_A=r["I_op_A"])
        pairs = [c["n_turns"][2 * i] for i in range(N_LAYERS // 2)]
        ok = r["B_target_T"] >= B_FLOOR and r["hoop_MPa"] <= HOOP_MAX
        _log(f"  {c['label']:<20s} tape={r['tape_km']:.4f} "
             f"B={r['B_target_T']:6.2f}T hoop={r['hoop_MPa']:4.0f} "
             f"I_op={r['I_op_A']:6.1f} pairs={pairs} "
             f"{'-> T-A' if ok else ''}")
        if ok or c["label"] == "champion_ref":
            keep.append(c)

    if not keep:
        _log("")
        _log("Nothing reaches 10 T along these directions. The field target "
             "and this coil family are in tension -- see the summary.")
        return

    _log("")
    _log(f"STEP 2: T-A box uniformity on {len(keep)} candidates "
         f"({N_TA_WORKERS} in parallel, {TA_REPEATS} meshes each)")

    def job(c):
        d = dict(a=c["a"], b=c["b"], coil_half_gap=c["gap"],
                 n_turns=c["n_turns"], I_design=c["I_op_A"])
        boxes = ta_validate(d, c["label"])
        c["box_ptp"] = max(boxes) if boxes else float("nan")
        c["passes"] = bool(boxes) and c["box_ptp"] <= UNIF_LIMIT \
            and c["B_target_T"] >= B_FLOOR
        _log(f"  {c['label']:<20s} B={c['B_target_T']:6.2f}T "
             f"tape={c['tape_km']:.4f} -> box = "
             f"{', '.join(f'{b:.3f}' for b in boxes) or 'FAILED'}  "
             f"{'PASS' if c['passes'] else 'fail'}")
        _write(keep)
        return c

    with ThreadPoolExecutor(max_workers=N_TA_WORKERS) as ex:
        list(ex.map(job, keep))
    _write(keep)

    _log("")
    _log(f"{'label':<20}{'tape_km':>9}{'B_T':>8}{'hoop':>6}{'box%':>8}  verdict")
    for c in sorted(keep, key=lambda c: c.get("box_ptp", 9e9)):
        if "box_ptp" not in c:
            continue
        _log(f"{c['label']:<20}{c['tape_km']:>9.4f}{c['B_target_T']:>8.2f}"
             f"{c['hoop_MPa']:>6.0f}{c['box_ptp']:>8.3f}  "
             f"{'PASS' if c.get('passes') else 'fail'}")

    winners = [c for c in keep if c.get("passes")]
    _log("")
    if not winners:
        _log("No candidate meets BOTH 10 T and 1% box uniformity.")
        near = [c for c in keep if c.get("box_ptp", 9e9) <= UNIF_LIMIT]
        if near:
            best = max(near, key=lambda c: c["B_target_T"])
            _log(f"  Best uniformity-passing design reaches only "
                 f"{best['B_target_T']:.2f} T ({best['label']}).")
        return
    winners.sort(key=lambda c: c["tape_km"])
    w = winners[0]
    _log(f"WINNER: {w['label']} tape={w['tape_km']:.4f}km "
         f"B={w['B_target_T']:.2f}T hoop={w['hoop_MPa']:.0f}MPa "
         f"box={w['box_ptp']:.3f}%")
    _log(f"  a={w['a']*1e3:.4f}mm b={w['b']*1e3:.4f}mm "
         f"gap={w['gap']*1e3:.4f}mm I_op={w['I_op_A']:.2f}A")
    _log(f"  n_turns={w['n_turns']}")
    _log("")
    _log(f"conservative cross-check ('{IC_CHECK}'):")
    ic2 = make_ic_model(IC_CHECK)
    r2 = og.evaluate(dict(a=w["a"], b=w["b"], n_turns=w["n_turns"],
                          coil_half_gap=w["gap"]), ic2, comm)
    _log(f"  B_target={r2['B_target_T']:.2f}T I_op={r2['I_op_A']:.1f}A "
         f"({'still >=10T' if r2['B_target_T'] >= 10 else 'BELOW 10T'})")
    _log(f"total {(time.monotonic()-t0)/60:.1f} min -> {CSV_PATH}")


def _write(rows):
    fields = ["label", "tape_km", "B_target_T", "hoop_MPa", "I_op_A",
              "a", "b", "gap", "n_turns", "box_ptp", "passes"]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "n_turns": json.dumps(r["n_turns"])})


if __name__ == "__main__":
    main()
