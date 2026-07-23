"""
sweep_n_layers.py — search over the NUMBER of pancake layers
==============================================================
Everything up to 2026-07-22 treated n_layers as a fixed constant (7,
inherited from params.py's original baseline) -- cmaes_search.py's
N_LAYERS is derived once from len(opt_config.CMAES_X0["n_turns"]), and
every run (initial search, diverse restarts, the 76-restart overnight
sweep, both extended-refinement rounds) always supplied exactly 7
turn-count values. Verified 2026-07-22 that the underlying pipeline
(build_mesh.py, optimize_geometry.evaluate()) already handles an
arbitrary layer count correctly with no code changes needed there --
params.n_layers = len(params.n_turns) is computed fresh every call. The
one real bug this exposed: cfg.CMAES_HALF_GAP_BOUNDS was a static range
tuned for N_LAYERS=7 (floor 15.5mm); at a different layer count the true
minimum coil_half_gap (for the 3mm face-gap constraint) is different --
fixed in cmaes_search.py's gap_bounds_for_n_layers(), which now computes
the floor from N_LAYERS directly (reproduces the old static bounds
exactly at N_LAYERS=7, so nothing about the existing 7-layer behavior
changed).

CMA-ES itself can't treat "number of layers" as a continuous/optimized
dimension -- its covariance adaptation needs a FIXED dimensionality for
the whole run. So this sweeps n_layers as a DISCRETE OUTER loop: one full,
independent CMA-ES search per candidate layer count (each still using
generation-parallel evaluation across CMAES_N_WORKERS processes), then
compares the best tape_km found at each layer count.

Starting points are NOT random cold starts -- by now we know the
characteristic shape of a good design (a pinned near the inner-radius
floor, straight length just above its 5mm floor, gap pinned near its
own face-gap floor -- see the round-1/round-2 extended-refinement
results). smart_x0_for_n_layers() reconstructs that same shape for each
candidate layer count directly from the physical floors, redistributing
the current best known design's total turn count evenly across the new
layer count as a starting turns-per-layer guess. This should converge
much faster per layer count than a blind random restart would.

Run from Racetrack_v4 root, in the background (each layer count is a
full CMA-ES search, so this takes a while -- see the printed per-count
budget/estimate):
    conda run -n fenicsx-env python3 optimize/sweep_n_layers.py

Progress: optimize/sweep_n_layers_log.txt (one line per completed layer
count). Every run still feeds the same cumulative
cmaes_all_evaluations.csv/cmaes_param_map.png as any other cmaes_search.py
run (each tagged with its own run_tag, as always).
"""
import os, sys, csv, json, time, traceback, subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import opt_config as cfg

# Candidate layer counts to test -- spans below/above the historical
# default of 7. Higher counts cost much more per evaluation (12 layers
# measured ~43s/eval vs ~5-10s at 7 layers on this coarse screen mesh, so
# this sweep intentionally gives higher layer counts a SMALLER eval budget
# (see PER_LAYER_EVALS) rather than a uniform one -- otherwise a handful
# of expensive layer counts would dominate total wall-clock time.
# 2026-07-22: 3 already completed (best 0.370km, see optimize/
# sweep_n_layers_log.txt) before this restart -- not re-run. 7 skipped
# entirely per project direction: we already have a far more thoroughly
# optimized 7-layer design (0.188km, from the extended-refinement rounds,
# not this sweep's from-scratch smart start) -- redoing it here at only
# 400 evals from a fresh uniform-turns start would just be worse and
# redundant. That 0.188km design remains the reference every OTHER layer
# count is compared against.
LAYER_COUNTS = [4, 6, 8, 10, 12]     # 2026-07-23: odd counts (5, 9) removed
                                      # -- double-pancake construction makes
                                      # them physically unbuildable
                                      # (cmaes_search.py now asserts on
                                      # odd N_LAYERS)

# The best known design as of 2026-07-22 (round-1 extended refinement from
# run 3): a=15.5mm, b=22.8mm, gap=17.0mm, n_turns totaling 1310 across 7
# layers. Used only as the TOTAL TURN COUNT reference for redistributing
# across a different layer count -- not literally reused as-is.
REF_TOTAL_TURNS = 1310

MIN_STRAIGHT_M = 0.005       # geometry_violation()'s floor
MIN_BORE_CLEAR_M = 0.0075    # geometry_violation()'s min bend radius floor
                              # (2026-07-23: REBCO min bend radius, was 3mm)
MIN_FACE_GAP_M = 0.003       # cfg.MIN_COIL_GAP_M

SWEEP_LOG = os.path.join(_ROOT, "optimize", "sweep_n_layers_log.txt")
SWEEP_LOGS_DIR = os.path.join(_ROOT, "optimize", "sweep_n_layers_logs")
RESULTS_CSV = os.path.join(_ROOT, cfg.CMAES_OUT_CSV)


def per_layer_evals(n_layers):
    """Smaller budget for pricier (more-layer) geometries so the sweep's
    wall-clock time isn't dominated by a couple of expensive counts."""
    if n_layers <= 5:
        return 500
    if n_layers <= 8:
        return 400
    return 250


def smart_x0_for_n_layers(n_layers, w=0.004, t=75e-6):
    """Reconstruct the characteristic shape of a good design (a and gap
    pinned near their physical floors, straight length just above its
    floor) for a NEW layer count, redistributing REF_TOTAL_TURNS evenly
    across n_layers as the starting per-layer guess. CMA-ES refines the
    exact distribution/values from here -- this just avoids a blind
    random cold start for each of the 9 layer counts.

    Margins above each floor calibrated 2026-07-22 against the actual best
    known 7-layer design (a=15.5mm/b=22.8mm/gap=17.0mm, tape=0.188km):
    that design sits at inner-radius margin=0.094mm (essentially AT its
    floor), straight-length margin=2.24mm, gap margin=1.50mm above their
    respective floors -- MARGIN_A tightened from a previous 2mm guess to
    0.3mm to match (kept slightly off zero, not exactly 0, so the starting
    point itself isn't immediately borderline-infeasible)."""
    n_per_layer = max(50, round(REF_TOTAL_TURNS / n_layers))
    n_turns = [n_per_layer] * n_layers

    MARGIN_A = 0.0003        # ~0.09mm discovered; small non-zero buffer
    MARGIN_B_EXTRA = 0.0022  # matches the discovered 2.24mm straight-length margin
    MARGIN_GAP = 0.0015      # matches the discovered 1.50mm gap margin

    a = n_per_layer * t / 2 + MIN_BORE_CLEAR_M + MARGIN_A
    b = a + MIN_STRAIGHT_M + MARGIN_B_EXTRA
    z_top = n_layers * w / 2.0
    gap = z_top + MIN_FACE_GAP_M / 2.0 + MARGIN_GAP
    return dict(a=round(a, 6), b=round(b, 6),
               coil_half_gap=round(gap, 6), n_turns=n_turns)


def _log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(SWEEP_LOG, "a") as f:
        f.write(line + "\n")


def _run_one_layer_count(n_layers, seed):
    x0 = smart_x0_for_n_layers(n_layers)
    max_evals = per_layer_evals(n_layers)

    env = os.environ.copy()
    env["CMAES_SWEEP_OVERRIDE_JSON"] = json.dumps(dict(
        x0=x0, seed=seed, max_evals=max_evals))

    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)

    os.makedirs(SWEEP_LOGS_DIR, exist_ok=True)
    log_path = os.path.join(SWEEP_LOGS_DIR, f"n_layers_{n_layers:02d}.log")
    with open(log_path, "w") as logf:
        subprocess.run(
            ["conda", "run", "-n", "fenicsx-env", "python3",
             "optimize/cmaes_search.py"],
            cwd=_ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT,
            check=True)

    if not os.path.exists(RESULTS_CSV):
        return None, x0, max_evals
    with open(RESULTS_CSV) as f:
        rows = list(csv.DictReader(f))
    return (rows[0] if rows else None), x0, max_evals


def main():
    _log(f"Starting n_layers sweep: {LAYER_COUNTS}, "
        f"ref_total_turns={REF_TOTAL_TURNS}")

    for n_layers in LAYER_COUNTS:
        seed = 6000 + n_layers
        x0 = smart_x0_for_n_layers(n_layers)
        max_evals = per_layer_evals(n_layers)
        _log(f"[n_layers={n_layers:2d}] starting: a={x0['a']*1e3:.1f}mm "
            f"b={x0['b']*1e3:.1f}mm gap={x0['coil_half_gap']*1e3:.1f}mm "
            f"n_turns={x0['n_turns']} max_evals={max_evals} seed={seed}")

        t0 = time.time()
        try:
            best, _, _ = _run_one_layer_count(n_layers, seed)
        except Exception:
            _log(f"[n_layers={n_layers:2d}] FAILED:\n{traceback.format_exc()}")
            continue
        dt = time.time() - t0

        if best:
            _log(f"[n_layers={n_layers:2d}] done in {dt/60:.1f} min -- "
                f"best tape={float(best['tape_km']):.3f}km "
                f"B={float(best['B_target_T']):.2f}T "
                f"unif={float(best['uniformity_pct']):.3f}% "
                f"hoop={float(best['hoop_MPa']):.0f}MPa "
                f"a={float(best['a_mm']):.1f}mm b={float(best['b_mm']):.1f}mm "
                f"gap={float(best['gap_mm']):.1f}mm "
                f"n_turns={best['n_turns']}")
        else:
            _log(f"[n_layers={n_layers:2d}] done in {dt/60:.1f} min -- "
                f"no feasible design found")

    _log("n_layers sweep finished.")


if __name__ == "__main__":
    main()
