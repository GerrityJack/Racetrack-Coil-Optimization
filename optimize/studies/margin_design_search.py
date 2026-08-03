"""
margin_design_search.py -- a design with BUILD MARGIN, not one on the floors
============================================================================
2026-07-31. The jitter study killed the previous design: **0 of 14
perturbed builds reached 10 T** (9.82-10.02 T), because the search
minimized tape subject to B >= 10 T and therefore converged EXACTLY ONTO
that constraint (nominal 10.03 T, 0.3 % margin). Uniformity was never the
problem -- 13 of 14 passed easily at 0.33-0.53 %.

So this search optimizes against MARGIN-AWARE constraints derived directly
from the measured jitter response, rather than against the nominal
constraints:

  requirement                          derivation
  ---------------------------------    ------------------------------------
  B_target >= 10.3 T nominal           worst measured jitter cost was
                                       -0.21 T, so 10.3 keeps the build
                                       above 10.0
  bend radius >= 7.5 mm AT WORST       worst case is a-0.2mm AND t+2%
    => a >= 7.7mm + N*3.825e-5         simultaneously (tape thickness is
                                       the dominant error and thickens the
                                       pack by 2%)
  face gap >= 3.0 mm at gap-0.2mm      => nominal face gap >= 3.4 mm
    => coil_half_gap >= 13.70 mm
  straight >= 5.0 mm at (a+.2, b-.2)   => nominal b-a >= 5.4 mm
  box uniformity (T-A) <= 1%           validated on finalists, as always

The binding tension: more turns buy field but need a LARGER `a` for
bend margin (a >= 7.7mm + N*3.825e-5), and larger `a` costs tape. So the
search is a 2-D scan over `a` and how far the turn count is backed off its
own bend-margin ceiling.

Turn RATIO is held at the champion's 295:369:2 -- the only profile family
with T-A-validated passing uniformity. (The funnel's flat-ish
[243,243,219] looked good on the cheap metric and scored 2.0 % on T-A.)

Ic model `kim` at 65 % of local Ic, as for the previous design.

Run:
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \\
        optimize/studies/margin_design_search.py
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
RUN_DIR = os.path.join(_ROOT, "optimize", "runs", "margin_design")
TA_DIR = os.path.join(RUN_DIR, "ta_logs")
os.makedirs(TA_DIR, exist_ok=True)
LOG_PATH = os.path.join(RUN_DIR, "log.txt")
CSV_PATH = os.path.join(RUN_DIR, "results.csv")

IC_EXTRAP = "kim"
IC_CHECK = "scaling:45"
FRAC_IC = 0.65

# margin-aware targets (see docstring)
B_TARGET_MIN = 10.30
HOOP_MAX = 400.0
UNIF_LIMIT = 1.0
GAP_M = 0.0137                 # face gap 3.4mm nominal
STRAIGHT_M = 0.0054            # 5.4mm nominal
TOL_A = 0.0002
TOL_T_FRAC = 0.02
BEND_FLOOR = 0.0075

CH_RATIO = (295.0, 369.0, 2.0)
N_LAYERS = 6
TA_REPEATS = 2
N_TA_WORKERS = 2
N_FINALISTS = 4


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def max_pair_with_margin(a, t):
    """Largest max-pair turn count whose bend radius still clears the floor
    in the WORST case: a at -0.2mm and tape 2% thick simultaneously."""
    return int(((a - TOL_A) - BEND_FLOOR) * 2.0 / (t * (1 + TOL_T_FRAC)))


def worst_bend(a, n_max, t):
    return (a - TOL_A) - n_max * t * (1 + TOL_T_FRAC) / 2.0


def design_for(a, fill, t):
    cap = max_pair_with_margin(a, t)
    n_main = max(1, int(round(cap * fill)))
    scale = n_main / CH_RATIO[1]
    pairs = [max(1, int(round(r * scale))) for r in CH_RATIO]
    pairs[1] = n_main
    n_turns = [v for p in pairs for v in (p, p)]
    return dict(a=a, b=a + STRAIGHT_M, coil_half_gap=GAP_M, n_turns=n_turns)


_TA_RE = re.compile(r"box_ptp_pct=(?P<box>[-\d.]+)")


def ta_validate(c):
    d = dict(label=c["label"], a=c["a"], b=c["b"],
             coil_half_gap=c["coil_half_gap"], n_turns=c["n_turns"],
             I_design=c["I_op_A"])
    env = os.environ.copy()
    env["TA_VALIDATE_JSON"] = json.dumps(d)
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

    _log("Margin-aware design search (the previous design had 0.3% field "
         "margin and 0/14 builds reached 10 T)")
    _log(f"  targets: B >= {B_TARGET_MIN} T, worst-case bend >= 7.5mm, "
         f"face gap 3.4mm nominal, straight 5.4mm nominal")
    _log(f"  Ic={IC_EXTRAP} at {FRAC_IC:.0%} of local Ic")

    ic = make_ic_model(IC_EXTRAP)
    cfg.SAFETY_FACTOR = 1.0 / FRAC_IC
    t = params.t

    _log("")
    _log("STAGE 1: FEM scan over (a, turn fill fraction)")
    rows = []
    for a_mm in (23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 27.0, 28.0):
        a = a_mm * 1e-3
        for fill in (1.00, 0.92, 0.85):
            d = design_for(a, fill, t)
            n_max = max(d["n_turns"])
            wb = worst_bend(a, n_max, t)
            if wb < BEND_FLOOR - 1e-9:
                continue
            r = og.evaluate(dict(a=d["a"], b=d["b"], n_turns=d["n_turns"],
                                 coil_half_gap=d["coil_half_gap"]), ic, comm)
            if not r.get("feasible"):
                continue
            pairs = [d["n_turns"][2 * i] for i in range(N_LAYERS // 2)]
            ok = (r["B_target_T"] >= B_TARGET_MIN
                  and r["hoop_MPa"] <= HOOP_MAX)
            rows.append(dict(label=f"a{a_mm:.1f}_f{fill:.2f}",
                             a=d["a"], b=d["b"],
                             coil_half_gap=d["coil_half_gap"],
                             n_turns=d["n_turns"], tape_km=r["tape_km"],
                             B_target_T=r["B_target_T"],
                             hoop_MPa=r["hoop_MPa"], I_op_A=r["I_op_A"],
                             worst_bend_mm=wb * 1e3, ok_stage1=ok))
            _log(f"  {'OK ' if ok else '-- '}a={a_mm:4.1f} fill={fill:.2f} "
                 f"tape={r['tape_km']:.4f} B={r['B_target_T']:6.2f}T "
                 f"hoop={r['hoop_MPa']:4.0f} worstbend={wb*1e3:5.2f} "
                 f"pairs={pairs}")

    good = [r for r in rows if r["ok_stage1"]]
    if not good:
        _log("")
        _log(f"Nothing reaches {B_TARGET_MIN} T with the margin constraints. "
             f"Widen the `a` range or relax the margin target.")
        _write(rows)
        return
    good.sort(key=lambda r: r["tape_km"])
    _log("")
    _log(f"  {len(good)} designs meet B >= {B_TARGET_MIN} T with full "
         f"build margin; cheapest {min(r['tape_km'] for r in good):.4f} km")

    finalists = good[:N_FINALISTS]
    _log("")
    _log(f"STAGE 2: T-A box uniformity on {len(finalists)} finalists")

    def job(c):
        boxes = ta_validate(c)
        c["box_ptp"] = max(boxes) if boxes else float("nan")
        c["passes"] = bool(boxes) and c["box_ptp"] <= UNIF_LIMIT
        _log(f"  {c['label']:<14s} tape={c['tape_km']:.4f} "
             f"B={c['B_target_T']:6.2f}T -> box = "
             f"{', '.join(f'{b:.3f}' for b in boxes) or 'FAILED'}  "
             f"{'PASS' if c['passes'] else 'fail'}")
        _write(rows)
        return c

    with ThreadPoolExecutor(max_workers=N_TA_WORKERS) as ex:
        list(ex.map(job, finalists))
    _write(rows)

    winners = [c for c in finalists if c.get("passes")]
    _log("")
    if not winners:
        _log("No margin-aware design passed T-A box uniformity.")
        return
    winners.sort(key=lambda c: c["tape_km"])
    w = winners[0]
    _log(f"WINNER: {w['label']} tape={w['tape_km']:.4f}km "
         f"B={w['B_target_T']:.2f}T hoop={w['hoop_MPa']:.0f}MPa "
         f"box={w['box_ptp']:.3f}% worst-case bend={w['worst_bend_mm']:.2f}mm")
    _log(f"  a={w['a']*1e3:.4f}mm b={w['b']*1e3:.4f}mm "
         f"gap={w['coil_half_gap']*1e3:.4f}mm I_op={w['I_op_A']:.2f}A")
    _log(f"  n_turns={w['n_turns']}")
    _log("")
    _log(f"conservative Ic cross-check ('{IC_CHECK}'):")
    ic2 = make_ic_model(IC_CHECK)
    r2 = og.evaluate(dict(a=w["a"], b=w["b"], n_turns=w["n_turns"],
                          coil_half_gap=w["coil_half_gap"]), ic2, comm)
    _log(f"  B_target={r2['B_target_T']:.2f}T "
         f"({'>=10T' if r2['B_target_T'] >= 10 else 'below 10T'})")
    _log(f"total {(time.monotonic()-t0)/60:.1f} min -> {CSV_PATH}")


def _write(rows):
    fields = ["label", "tape_km", "B_target_T", "hoop_MPa", "I_op_A",
              "worst_bend_mm", "a", "b", "coil_half_gap", "n_turns",
              "ok_stage1", "box_ptp", "passes"]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "n_turns": json.dumps(r["n_turns"])})


if __name__ == "__main__":
    main()
