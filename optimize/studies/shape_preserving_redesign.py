"""
shape_preserving_redesign.py -- 10 T under a real Ic model, taper PRESERVED
===========================================================================
2026-07-31, third attempt at the conservative redesign. The first two both
failed the same way, and the reason is structural, not a tuning mistake:

  attempt 1  seeded a FLAT profile and left the turn step at its
             bound-range default (~210 turns). Wandered to [544,148,167]
             at 2x the champion's tape within 103 evaluations.
  attempt 2  seeded the champion EXACTLY and used a local 25-turn step.
             Geometry stayed local (a 22.5-22.9mm vs 22.227), but over
             ~280 evaluations the turn profile still flattened from
             [295,369,2] (ratio 1.25) to [344,344,6] (ratio 1.00). Of 167
             designs meeting 10 T, NONE retained a champion-like taper.

**Why that keeps happening.** `cmaes_search.py`'s fitness minimizes tape
subject to B_target and hoop, with NO uniformity term (deliberate -- two
guessed uniformity proxies were both wrong, see CLAUDE.md). Flattening the
taper is genuinely cheaper in tape at equal field, so the objective has a
persistent gradient toward it. A small step size slows that drift; it
cannot stop it. And `local_polish.py` already measured where it leads:
flat profiles score 3.6-8.6% box uniformity against a 1% target, while the
champion's taper scores 0.69%.

**So the shape is constrained here rather than searched.** The turn
profile is FIXED to the champion's validated shape and only a single scale
factor is optimized alongside the geometry:

    n_turns(pairs) = round( s * [295, 369, ..., 369, 2] )

At 6 layers this is exactly the champion's profile. At higher layer counts
the LARGE pair is repeated, so extra turns arrive via extra LAYERS at
constant pack thickness -- which is what keeps the 7.5 mm bend-radius floor
from pushing `a` outward and killing the returns (adding turns per layer
needed 2x the tape for +1 T; adding layers reached 10.96 T at 0.34 km).

Because the shape is fixed and tape rises monotonically with `s` while
B_target also rises with `s`, the minimum-tape design is simply the
SMALLEST `s` that still reaches 10 T -- a 1-D bisection, ~10 evaluations
per layer count instead of 1200. Cheap enough to then T-A-validate every
winner properly.

`a`, `b` are held at the champion's values (both sit on validated floors:
`a` at the soft a-floor / bend-radius limit, `b` at the 5 mm straight-length
floor) and `gap` at each layer count's exact 3 mm face-gap floor, which the
perturbation study showed is that axis's uniformity optimum.

Ic model: `kim` (hold-out MAPE 4.1%, bias -3.3%); operating point 62.5% of
local Ic. The winner is cross-checked under the conservative `scaling:45`.

Run:
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \\
        optimize/studies/shape_preserving_redesign.py
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
RUN_DIR = os.path.join(_ROOT, "optimize", "runs", "shape_preserving")
TA_DIR = os.path.join(RUN_DIR, "ta_logs")
os.makedirs(TA_DIR, exist_ok=True)
LOG_PATH = os.path.join(RUN_DIR, "log.txt")
CSV_PATH = os.path.join(RUN_DIR, "results.csv")

IC_EXTRAP = "kim"
IC_CHECK = "scaling:45"
FRAC_IC = 0.625                  # 62.5% of local Ic -> SAFETY_FACTOR 1.6
LAYER_COUNTS = [6, 8, 10, 12]
B_FLOOR = 10.0
HOOP_MAX = 400.0
UNIF_LIMIT = 1.0

CH_A = 0.022227029065529628
CH_B = 0.02726822715975084
CH_PROFILE = (295, 369, 2)       # (inner pair, large pair, vestigial pair)
MIN_FACE_GAP_M = 0.003


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def design_for(n_layers, s):
    """Champion-shaped turn profile scaled by s, at that layer count."""
    import params
    n_pairs = n_layers // 2
    inner, main, tail = CH_PROFILE
    pairs = [inner] + [main] * (n_pairs - 2) + [tail]
    pairs = [max(1, int(round(p * s))) for p in pairs]
    n_turns = [v for p in pairs for v in (p, p)]
    gap = n_layers * params.w / 2.0 + MIN_FACE_GAP_M / 2.0
    return dict(a=CH_A, b=CH_B, coil_half_gap=gap, n_turns=n_turns)


def buildable(d):
    import params
    pack = max(d["n_turns"]) * params.t
    return (d["a"] - pack / 2.0) >= 0.0075 - 1e-9 and (d["b"] - d["a"]) >= 0.005 - 1e-9


def main():
    t0 = time.monotonic()
    open(LOG_PATH, "w").close()
    from mpi4py import MPI
    import optimize_geometry as og, opt_config as cfg
    from ic_extrapolation import make_ic_model
    comm = MPI.COMM_WORLD

    _log("Shape-preserving redesign: minimum tape reaching 10 T with the "
         "champion's taper held FIXED")
    _log(f"  Ic model={IC_EXTRAP}, operating point={FRAC_IC:.1%} of local Ic")
    _log(f"  profile per pair = s x [{CH_PROFILE[0]}, "
         f"{CH_PROFILE[1]}...(repeated), {CH_PROFILE[2]}]")
    ic = make_ic_model(IC_EXTRAP)
    cfg.SAFETY_FACTOR = 1.0 / FRAC_IC

    def ev(d, model=None):
        r = og.evaluate(dict(a=d["a"], b=d["b"], n_turns=d["n_turns"],
                             coil_half_gap=d["coil_half_gap"]),
                        model or ic, comm)
        return r

    rows = []
    for n in LAYER_COUNTS:
        _log("")
        _log(f"--- n_layers={n} ---")
        # bracket: find an s that reaches B_FLOOR
        s_hi = None
        for s in (0.7, 0.85, 1.0, 1.2, 1.5, 1.9):
            d = design_for(n, s)
            if not buildable(d):
                _log(f"  s={s:.2f}: not buildable (bend radius) -- stop raising")
                break
            r = ev(d)
            if not r.get("feasible"):
                _log(f"  s={s:.2f}: infeasible ({r.get('reason')})")
                continue
            _log(f"  s={s:.2f}  tape={r['tape_km']:.4f}km  "
                 f"B={r['B_target_T']:.2f}T  hoop={r['hoop_MPa']:.0f}MPa")
            if r["B_target_T"] >= B_FLOOR and r["hoop_MPa"] <= HOOP_MAX:
                s_hi = s
                break
        if s_hi is None:
            _log(f"  n_layers={n}: cannot reach {B_FLOOR} T with this shape")
            continue

        # bisect down to the smallest s meeting the floor
        s_lo = max(0.3, s_hi - 0.35)
        best = None
        for _ in range(7):
            s_mid = 0.5 * (s_lo + s_hi)
            d = design_for(n, s_mid)
            r = ev(d) if buildable(d) else None
            if r and r.get("feasible") and r["B_target_T"] >= B_FLOOR \
                    and r["hoop_MPa"] <= HOOP_MAX:
                s_hi = s_mid
                best = (s_mid, d, r)
            else:
                s_lo = s_mid
        if best is None:
            d = design_for(n, s_hi)
            best = (s_hi, d, ev(d))
        s, d, r = best
        pairs = [d["n_turns"][2 * i] for i in range(n // 2)]
        _log(f"  -> min scale s={s:.4f}: tape={r['tape_km']:.4f}km "
             f"B={r['B_target_T']:.2f}T hoop={r['hoop_MPa']:.0f}MPa "
             f"I_op={r['I_op_A']:.1f}A pairs={pairs}")
        rows.append(dict(n_layers=n, scale=s, tape_km=r["tape_km"],
                         B_target_T=r["B_target_T"], hoop_MPa=r["hoop_MPa"],
                         I_op_A=r["I_op_A"], a_mm=d["a"] * 1e3,
                         b_mm=d["b"] * 1e3,
                         gap_mm=d["coil_half_gap"] * 1e3,
                         n_turns=json.dumps(d["n_turns"]), box_ptp=""))
        _write(rows)

    if not rows:
        _log("no layer count reached 10 T -- stopping")
        return

    _log("")
    _log("PHASE B: T-A box-uniformity validation (2 independent meshes each)")
    for row in rows:
        d = dict(a=row["a_mm"] / 1e3, b=row["b_mm"] / 1e3,
                 coil_half_gap=row["gap_mm"] / 1e3,
                 n_turns=json.loads(row["n_turns"]),
                 I_design=row["I_op_A"])
        boxes = ta_validate(d, f"n{row['n_layers']:02d}")
        row["box_ptp"] = max(boxes) if boxes else float("nan")
        _log(f"  n_layers={row['n_layers']}: box p2p = "
             f"{', '.join(f'{b:.3f}' for b in boxes) or 'FAILED'} -> "
             f"{'PASS' if boxes and max(boxes) <= UNIF_LIMIT else 'FAIL'}")
        _write(rows)

    _log("")
    _log(f"{'n_lay':>6}{'tape_km':>9}{'B_T':>7}{'hoop':>6}{'box%':>8}  verdict")
    ok = []
    for row in rows:
        p = row["box_ptp"] == row["box_ptp"] and row["box_ptp"] <= UNIF_LIMIT
        if p:
            ok.append((row["tape_km"], row))
        _log(f"{row['n_layers']:>6}{row['tape_km']:>9.4f}"
             f"{row['B_target_T']:>7.2f}{row['hoop_MPa']:>6.0f}"
             f"{row['box_ptp']:>8.3f}  {'PASS' if p else 'FAIL'}")
    if not ok:
        _log("")
        _log("No shape-preserving design passed both 10 T and box uniformity.")
        return
    ok.sort(key=lambda t: t[0])
    tape, w = ok[0]
    _log("")
    _log(f"WINNER: n_layers={w['n_layers']} tape={tape:.4f}km "
         f"B={w['B_target_T']:.2f}T hoop={w['hoop_MPa']:.0f}MPa "
         f"box={w['box_ptp']:.3f}%")
    _log(f"  a={w['a_mm']:.4f}mm b={w['b_mm']:.4f}mm gap={w['gap_mm']:.4f}mm "
         f"I_op={w['I_op_A']:.2f}A")
    _log(f"  n_turns={w['n_turns']}")

    # conservative cross-check
    _log("")
    _log(f"cross-check under the conservative '{IC_CHECK}' Ic model:")
    ic2 = make_ic_model(IC_CHECK)
    r2 = ev(dict(a=w["a_mm"] / 1e3, b=w["b_mm"] / 1e3,
                 coil_half_gap=w["gap_mm"] / 1e3,
                 n_turns=json.loads(w["n_turns"])), model=ic2)
    _log(f"  B_target={r2['B_target_T']:.2f}T  I_op={r2['I_op_A']:.1f}A  "
         f"hoop={r2['hoop_MPa']:.0f}MPa  "
         f"({'still >=10T' if r2['B_target_T'] >= 10 else 'BELOW 10T'})")
    _log(f"total {(time.monotonic()-t0)/60:.1f} min -> {CSV_PATH}")


def _write(rows):
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_layers", "scale", "tape_km",
            "B_target_T", "hoop_MPa", "I_op_A", "a_mm", "b_mm", "gap_mm",
            "n_turns", "box_ptp"])
        w.writeheader(); w.writerows(rows)


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
