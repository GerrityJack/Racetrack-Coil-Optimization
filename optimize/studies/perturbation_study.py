"""
perturbation_study.py -- "was the champion just lucky?" (2026-07-30)
=====================================================================
The 6-layer champion (a=22.227mm, b=27.268mm, gap=13.500mm,
n_turns=[285,285,379,379,2,2]) was found by a CMA-ES search whose fitness
function carries NO uniformity signal at all (see CLAUDE.md's "Box
uniformity is the real target" section) -- yet it is the only design in
this project's history whose T-A box peak-to-peak uniformity actually
passes (0.73-0.83%), with every alternative tried landing at 3-9%.
Landing that close to a uniformity sweet spot by chance is suspicious.

The `a`-isolation sweep already showed a smooth V-shaped bowl in `a` with
its minimum essentially ON the champion, but at COARSE spacing (+-3mm)
and along ONE axis only, with the manufacturing constraints ignored (its
-3/-6mm points violate the 7.5mm bend radius). This study asks the
sharper question: perturb the champion by SMALL, MANUFACTURABLE amounts
along every axis independently AND all axes at once, and see whether the
box uniformity degrades smoothly (=> genuine local optimum, trustworthy)
or erratically (=> the 0.73% was a knife-edge numerical fluke).

Note the champion sits ON three constraint boundaries simultaneously:
  straight length  L = b - a  = 5.041mm  (floor 5.0mm)
  face gap  2*(gap - z_top)   = 3.001mm  (floor 3.0mm)
  bend radius  a - max(n)*t/2 = 8.015mm  (floor 7.5mm, 0.5mm margin)
so several axes can only be perturbed in ONE direction without leaving
the buildable set. Perturbations are chosen accordingly; every candidate
is checked against all three floors before it is run, and any that
violates is reported, not silently dropped.

Method per candidate (identical to the project's validated pipeline):
  1. optimize_geometry.evaluate()  -- coarse screen, in-process, gives
     the quench-limited I_op plus tape_km / B_target_T / hoop_MPa (the
     three metrics that were NEVER broken; its uniformity_pct is ignored)
  2. optimize/ta_validate.py in a FRESH subprocess at that I_op -- the
     full per-layer T-A solve and true 30x6mm box peak-to-peak, 2 repeats
     (independent mesh builds) so mesh-noise fragility is measured, not
     assumed (the n_layers=4 episode: 0.79% in-process vs 2.19% on an
     independent mesh)

Run:
    /home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \
        optimize/perturbation_study.py
(direct binary, NOT `conda run` -- see CLAUDE.md on output buffering)

Outputs: optimize/perturbation_results.csv, optimize/perturbation_log.txt,
per-candidate T-A logs in optimize/perturbation_logs/.
"""
import os, sys, json, re, csv, time, subprocess, traceback
from concurrent.futures import ThreadPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "mesh"),
           os.path.join(_ROOT, "solve"), os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np

PYTHON_BIN = "/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3"
LOGS_DIR = os.path.join(_ROOT, "optimize", "runs", "perturbation", "perturbation_logs")
LOG_PATH = os.path.join(_ROOT, "optimize", "runs", "perturbation", "perturbation_log.txt")
CSV_PATH = os.path.join(_ROOT, "optimize", "runs", "perturbation", "perturbation_results.csv")
os.makedirs(LOGS_DIR, exist_ok=True)

TA_REPEATS = 2
TA_TIMEOUT_S = 1800
N_TA_WORKERS = 3          # 8-core machine; each T-A solve is ~65 s/repeat

# ── the design under test ───────────────────────────────────────────────────
CHAMPION = dict(a=0.022227029065529628, b=0.02726822715975084,
                coil_half_gap=0.013500289306395013,
                n_turns=[285, 285, 379, 379, 2, 2])

# constraint floors (mirrors cmaes_search.geometry_violation)
MIN_L_M = 0.005            # straight section
MIN_BEND_M = 0.0075        # innermost-turn bend radius
MIN_FACE_GAP_M = 0.003     # coil-to-coil face-to-face
T_PITCH = 75e-6
W_TAPE = 0.004


def margins(a, b, gap, n_turns):
    """(straight_mm, bend_mm, face_gap_mm) and whether all floors hold."""
    L = b - a
    bend = a - max(n_turns) * T_PITCH / 2.0
    z_top = len(n_turns) * W_TAPE / 2.0
    face = 2.0 * (gap - z_top)
    ok = (L >= MIN_L_M - 1e-9 and bend >= MIN_BEND_M - 1e-9
          and face >= MIN_FACE_GAP_M - 1e-9)
    return L * 1e3, bend * 1e3, face * 1e3, ok


def cand(label, group, a=None, b=None, gap=None, n_turns=None):
    return dict(label=label, group=group,
                a=CHAMPION["a"] if a is None else a,
                b=CHAMPION["b"] if b is None else b,
                coil_half_gap=CHAMPION["coil_half_gap"] if gap is None else gap,
                n_turns=list(CHAMPION["n_turns"] if n_turns is None else n_turns))


def build_candidates():
    A, B, G = CHAMPION["a"], CHAMPION["b"], CHAMPION["coil_half_gap"]
    mm = 1e-3
    C = [cand("champion", "baseline")]

    # 1. rigid radial translation: a and b shifted together, so b-a (straight
    #    length), n_turns and gap are all exactly preserved -- the same
    #    single-variable isolation the earlier coarse sweep used, but at fine
    #    spacing and staying inside the 7.5mm bend-radius floor (a may only
    #    drop 0.51mm before that floor bites).
    for d in (-0.5, -0.25, +0.25, +0.5, +1.0, +2.0):
        C.append(cand(f"a{d:+.2f}mm", "a_translate", a=A + d * mm, b=B + d * mm))

    # 2. straight length alone: b moved, a fixed. Only upward -- the champion
    #    sits 0.04mm above the 5mm straight-length floor.
    for d in (+0.5, +1.5, +3.0):
        C.append(cand(f"b{d:+.2f}mm", "b_only", b=B + d * mm))

    # 3. coil-to-coil gap alone. Only upward -- face gap is AT its 3mm floor.
    for d in (+0.5, +1.0, +2.0, +4.0):
        C.append(cand(f"gap{d:+.2f}mm", "gap_only", gap=G + d * mm))

    # 4. turn distribution, double-pancake pairing preserved (both layers of
    #    each pair must share a turn count). Pair-1 capped at 392 by the bend
    #    radius (a - n*t/2 >= 7.5mm).
    for nt, lbl in (([275, 275, 389, 389, 2, 2], "turns_shift_out"),
                    ([295, 295, 369, 369, 2, 2], "turns_shift_in"),
                    ([285, 285, 379, 379, 10, 10], "turns_grow_pair3"),
                    ([271, 271, 360, 360, 2, 2], "turns_scale-5pct"),
                    ([299, 299, 392, 392, 2, 2], "turns_scale+4pct")):
        C.append(cand(lbl, "turns", n_turns=nt))

    # 5. simultaneous jitter on every axis at once -- the real robustness
    #    test. Biased to the feasible side where a floor blocks one direction;
    #    resampled until all three floors hold.
    rng = np.random.default_rng(20260730)
    n_jit, tries = 0, 0
    while n_jit < 4 and tries < 200:
        tries += 1
        a_j = A * (1.0 + rng.uniform(-0.005, 0.010))
        b_j = a_j + (B - A) * (1.0 + rng.uniform(0.0, 0.05))
        g_j = G * (1.0 + rng.uniform(0.0, 0.02))
        nt_j = []
        for pair in (285, 379, 2):
            v = max(1, int(round(pair * (1.0 + rng.uniform(-0.02, 0.02)))))
            nt_j += [v, v]
        if margins(a_j, b_j, g_j, nt_j)[3]:
            n_jit += 1
            C.append(cand(f"jitter{n_jit}", "jitter", a=a_j, b=b_j, gap=g_j,
                          n_turns=nt_j))
    return C


# ── logging ─────────────────────────────────────────────────────────────────

def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ── phase 2: T-A validation subprocess ──────────────────────────────────────

_RESULT_RE = re.compile(
    r"TA_VALIDATE_RESULT label=(?P<label>'.*?') repeat=(?P<repeat>\d+) "
    r"box_ptp_pct=(?P<box>[-\d.]+) onaxis_scif_pct=(?P<onaxis>[-\d.]+) "
    r"Bz_bore_uniform=(?P<bzu>[-\d.]+) Bz_bore_TA=(?P<bzta>[-\d.]+) "
    r"converged=(?P<conv>True|False) n_iters=(?P<niters>\d+) "
    r"solve_s=(?P<solves>[-\d.]+)")


def ta_validate(c):
    design = dict(label=c["label"], a=c["a"], b=c["b"],
                  coil_half_gap=c["coil_half_gap"], n_turns=c["n_turns"],
                  I_design=c["I_op_A"])
    env = os.environ.copy()
    env["TA_VALIDATE_JSON"] = json.dumps(design)
    env["TA_VALIDATE_REPEATS"] = str(TA_REPEATS)
    log_path = os.path.join(
        LOGS_DIR, f"{re.sub(r'[^A-Za-z0-9+-]+', '_', c['label'])}.log")
    rows = []
    try:
        out = subprocess.run([PYTHON_BIN, "-u", "optimize/ta_validate.py"],
                             cwd=_ROOT, env=env, capture_output=True,
                             text=True, timeout=TA_TIMEOUT_S)
        with open(log_path, "w") as f:
            f.write(out.stdout)
            f.write("\n--- stderr ---\n")
            f.write(out.stderr)
        for m in _RESULT_RE.finditer(out.stdout):
            rows.append(dict(box_ptp_pct=float(m.group("box")),
                             onaxis_scif_pct=float(m.group("onaxis")),
                             Bz_bore_uniform=float(m.group("bzu")),
                             Bz_bore_TA=float(m.group("bzta")),
                             converged=m.group("conv") == "True",
                             n_iters=int(m.group("niters")),
                             solve_s=float(m.group("solves"))))
    except subprocess.TimeoutExpired:
        _log(f"  [{c['label']}] T-A TIMED OUT after {TA_TIMEOUT_S}s")
    except Exception:
        _log(f"  [{c['label']}] T-A FAILED:\n{traceback.format_exc()}")
    return rows


# ── output ──────────────────────────────────────────────────────────────────

FIELDS = ["label", "group", "a_mm", "b_mm", "gap_mm", "n_turns",
          "straight_mm", "bend_mm", "face_gap_mm", "buildable",
          "tape_km", "I_op_A", "B_target_T", "hoop_MPa", "clip_frac",
          "box_ptp_0", "box_ptp_1", "box_ptp_mean", "box_ptp_spread",
          "onaxis_scif_pct", "ta_converged"]


def write_csv(rows):
    with open(CSV_PATH, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


def main():
    t_start = time.time()
    open(LOG_PATH, "w").close()
    cands = build_candidates()
    _log(f"perturbation study around the 6-layer champion: "
         f"{len(cands)} candidates, {TA_REPEATS} T-A repeats each")

    for c in cands:
        L, bend, face, ok = margins(c["a"], c["b"], c["coil_half_gap"],
                                    c["n_turns"])
        c.update(a_mm=c["a"] * 1e3, b_mm=c["b"] * 1e3,
                 gap_mm=c["coil_half_gap"] * 1e3, n_turns_s=str(c["n_turns"]),
                 straight_mm=L, bend_mm=bend, face_gap_mm=face, buildable=ok)
        if not ok:
            _log(f"  !! {c['label']} violates a floor "
                 f"(L={L:.3f} bend={bend:.3f} face={face:.3f} mm) -- "
                 f"kept and flagged, not dropped")

    # ── phase 1: coarse screen (sequential, mutates the shared params module
    #    and the shared default mesh path, so it must NOT run concurrently)
    _log("PHASE 1: coarse screen (tape / I_op / B_target / hoop) ...")
    from mpi4py import MPI
    import optimize_geometry as og
    from ic_model import IcModel
    comm = MPI.COMM_WORLD
    ic = IcModel()

    for c in cands:
        try:
            r = og.evaluate(dict(a=c["a"], b=c["b"],
                                 coil_half_gap=c["coil_half_gap"],
                                 n_turns=c["n_turns"]), ic, comm)
        except Exception:
            _log(f"  [{c['label']}] screen FAILED:\n{traceback.format_exc()}")
            c["screen_ok"] = False
            continue
        if not r.get("feasible", False):
            _log(f"  [{c['label']}] INFEASIBLE: {r.get('reason')}")
            c["screen_ok"] = False
            continue
        c["screen_ok"] = True
        c.update(tape_km=r["tape_km"], I_op_A=r["I_op_A"],
                 B_target_T=r["B_target_T"], hoop_MPa=r["hoop_MPa"],
                 clip_frac=r["clip_frac"])
        _log(f"  {c['label']:<20s} tape={r['tape_km']:.4f}km "
             f"I_op={r['I_op_A']:.1f}A B={r['B_target_T']:.3f}T "
             f"hoop={r['hoop_MPa']:.0f}MPa")

    runnable = [c for c in cands if c.get("screen_ok")]
    _log(f"PHASE 2: T-A box uniformity on {len(runnable)} candidates, "
         f"{N_TA_WORKERS} concurrent ...")

    done = [0]

    def _job(c):
        rows = ta_validate(c)
        boxes = [r["box_ptp_pct"] for r in rows]
        c["box_ptp_0"] = boxes[0] if len(boxes) > 0 else float("nan")
        c["box_ptp_1"] = boxes[1] if len(boxes) > 1 else float("nan")
        c["box_ptp_mean"] = float(np.mean(boxes)) if boxes else float("nan")
        c["box_ptp_spread"] = (float(max(boxes) - min(boxes))
                               if len(boxes) > 1 else float("nan"))
        c["onaxis_scif_pct"] = rows[0]["onaxis_scif_pct"] if rows else float("nan")
        c["ta_converged"] = all(r["converged"] for r in rows) if rows else False
        done[0] += 1
        _log(f"  [{done[0]}/{len(runnable)}] {c['label']:<20s} "
             f"box_ptp={c['box_ptp_mean']:.3f}% "
             f"(spread {c['box_ptp_spread']:.3f}) "
             f"converged={c['ta_converged']}")
        write_csv(cands)          # incremental: never lose finished work
        return c

    with ThreadPoolExecutor(max_workers=N_TA_WORKERS) as ex:
        list(ex.map(_job, runnable))

    write_csv(cands)

    # ── summary ─────────────────────────────────────────────────────────────
    base = next((c for c in cands if c["label"] == "champion"), None)
    _log("")
    _log(f"{'label':<20s} {'grp':<14s} {'a_mm':>7s} {'tape_km':>8s} "
         f"{'B_T':>6s} {'box%':>7s} {'d_box':>7s}")
    b0 = base.get("box_ptp_mean", float("nan")) if base else float("nan")
    for c in sorted(cands, key=lambda c: c.get("box_ptp_mean", 9e9)):
        if "box_ptp_mean" not in c:
            continue
        _log(f"{c['label']:<20s} {c['group']:<14s} {c['a_mm']:7.3f} "
             f"{c.get('tape_km', float('nan')):8.4f} "
             f"{c.get('B_target_T', float('nan')):6.3f} "
             f"{c['box_ptp_mean']:7.3f} "
             f"{c['box_ptp_mean'] - b0:+7.3f}")
    _log(f"total {(time.time() - t_start)/60:.1f} min -> {CSV_PATH}")


if __name__ == "__main__":
    main()
