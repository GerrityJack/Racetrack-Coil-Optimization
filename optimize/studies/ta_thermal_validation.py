"""
ta_thermal_validation.py — forced-full-length T-A validation matrix for the
4.2K cryogenic-operating-temperature investigation
=============================================================================
2026-09-12 (started by Claude Code, continuing the same-day 4.2K
investigation documented in CLAUDE.md's "Cryogenic (4.2K)
operating-temperature investigation" / known-open-issue #8).

That investigation found every 4.2K variant tried (Ic-only, Ic+n matched;
medium/fine mesh; bisection/fixed-current) hit a distinct solver-reliability
problem before producing a trustworthy number: a physically-inconsistent
Jc/n mismatch, a false-stall convergence signature (the EMA-smoothed-SCIF
`converged` flag firing while the raw |ΔB|/|B| residual was still
oscillating 140-280%), a bisection-budget (MAX_BISECT=5) exhaustion that
silently returned a placeholder value, and non-monotonic/non-converged
fixed-current results. It also found the 20K BASELINE itself is not
mesh-converged (I_op swung 70% between medium and fine mesh in
ta_safe_current.py) -- so no 4.2K-vs-20K comparison means anything until
that is fixed first.

This script is the "same class of work already done once for the
transient/ short-dt problem" CLAUDE.md says is needed: forced full-length
Picard runs that bypass the EMA stall flag (ta_solve.solve_ta_at_current's
new `min_iters`/`diag_log_path` kwargs, added this session specifically
for this) and check the raw residual directly, run as a systematic matrix
rather than one-off scripts.

Matrix, in order (each phase's results feed the next):
  A. z-resolution convergence check on the 20K baseline, in-plane mesh
     HELD FIXED at "medium" -- CLAUDE.md's own text says z-resolution
     (mesh_nz_per_layer/mesh_z_grading), not in-plane refinement, is the
     axis that actually matters for this solve. Three z-configs at the
     champion's own I=25A reference point, cold start, forced to
     ta_n_picard iterations.
  B. Same z-configs, same I=25A, cold start, forced full length, for the
     4.2K Ic-only and Ic+n-matched models -- does the false-stall /
     non-convergence signature persist once z-resolution is genuinely
     converged, or was it partly a mesh artifact?
  C. Relaxation-pair sweep on the 4.2K Ic+n-matched model (n~50-60, well
     outside the n=13-34 range ta_picard_alpha=(0.30,0.15) was tuned and
     validated for) at whichever z-config phase A/B found best-converged
     -- same idea as the transient/ short-dt fix (a ~10x smaller pair
     fixed THAT regime; this is a different stiff-n regime, not
     short-dt, so the right pair is not assumed, it's swept and checked
     against the raw residual).
  D. Direct fixed-current sweep (NOT ta_safe_current.py's 5-iteration
     bisection, which this same investigation showed can silently return
     a placeholder) at the best z-config/alpha found, for both 20K and
     the best 4.2K model -- maps margin(I) cleanly enough to read off a
     genuine T-A-safe I_op without the bisection-budget failure mode.

Every solve's raw per-iteration trace is logged to
optimize/runs/ta_thermal_validation/diag/<tag>.csv; every solve's summary
row is appended to optimize/runs/ta_thermal_validation/summary.csv
immediately (flushed every row, not just at the end -- a multi-hour run
must survive a restart without losing prior results, per CLAUDE.md's
"Operational lessons").

Deliberately does NOT touch params.py on disk -- current committed state
is CLAUDE.md's documented 2026-09-03 "xdense" one-off tier (confirmed live
via mesh_size_min_factor=0.05 on disk, 2026-09-12), ~20-30x medium's solve
time. Mesh tier is forced in-memory here, exactly the same pattern
optimize/studies/ta_safe_margin_search.py already uses for the same
reason.

Run (long -- direct python binary, see CLAUDE.md "Operational lessons" on
`conda run` stdout buffering):
    PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
    nohup $PY optimize/studies/ta_thermal_validation.py \
        > optimize/runs/ta_thermal_validation/log.txt 2>&1 &
"""
import os
import sys
import csv
import time
import traceback

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_OPT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_OPT)
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "mesh"),
           os.path.join(_ROOT, "solve"), os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from mpi4py import MPI

import params

# ── force a KNOWN in-plane mesh tier in-memory (see docstring) ─────────────
_TIER_FACTORS = dict(
    medium=dict(mesh_size_min_factor=0.25, mesh_size_max_factor=0.60,
                mesh_dist_min_factor=1.50, mesh_dist_max_factor=2.00,
                box_scale=4.0),
    fine=dict(mesh_size_min_factor=0.10, mesh_size_max_factor=1.0 / 3.0,
              mesh_dist_min_factor=1.00, mesh_dist_max_factor=3.00,
              box_scale=6.0),
)
_ZGRADE_PRESETS = dict(
    # CLAUDE.md's "current production default" 5-slab grading
    medium5=[0.075, 0.15, 0.55, 0.15, 0.075],
    # the 2026-09-03 xdense investigation's denser 7-slab edge grading,
    # same in-plane tier -- tests whether MORE z-resolution changes the
    # 20K/4.2K answer
    xdense7=[0.05, 0.075, 0.10, 0.55, 0.10, 0.075, 0.05],
    # params.py's own "brute-force uniform nz=5" comparison basis
    uniform5=None,
)


def apply_mesh_config(tier, zgrade_key):
    for k, v in _TIER_FACTORS[tier].items():
        setattr(params, k, v)
    if zgrade_key == "uniform5":
        params.mesh_nz_per_layer = 5
        params.mesh_z_grading = None
    else:
        params.mesh_nz_per_layer = 3
        params.mesh_z_grading = _ZGRADE_PRESETS[zgrade_key]
    params.recompute_derived()


import build_mesh
import solve as base_solve
import ta_solve
from dolfinx.io import gmsh as gmshio
from ic_model import IcModel, NValueModel
from ic_extrapolation import make_ic_model
from ic_temperature_scaling import (Fujikura4p2KScaledIcModel,
                                     Fujikura4p2KScaledNValueModel)
from ta_safe_current import _local_margin

OUT_DIR = os.path.join(_ROOT, "optimize", "runs", "ta_thermal_validation")
DIAG_DIR = os.path.join(OUT_DIR, "diag")
os.makedirs(DIAG_DIR, exist_ok=True)
SUMMARY_CSV = os.path.join(OUT_DIR, "summary.csv")

MARGIN_REQUIRED = 1.0 / 0.65
DEFAULT_ALPHA = (0.30, 0.15)   # params.py's own committed default


def record(row):
    new = not os.path.exists(SUMMARY_CSV)
    with open(SUMMARY_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def build_setup(mesh_tag):
    comm = MPI.COMM_WORLD
    mesh_path = os.path.join(
        OUT_DIR, f"mesh_{mesh_tag}_{os.getpid()}.msh")
    build_mesh.build(write_path=mesh_path)
    md = gmshio.read_from_msh(mesh_path, comm, rank=0, gdim=3)
    domain = md.mesh
    uniform_setup = base_solve.setup_problem(domain, md.cell_tags,
                                             md.facet_tags)
    ta = ta_solve.setup_ta_problem(domain, md.cell_tags, md.facet_tags,
                                   uniform_setup)
    try:
        os.remove(mesh_path)
    except OSError:
        pass
    return domain, uniform_setup, ta


def make_models(model_key):
    base_ic = make_ic_model("kim")
    base_n = NValueModel(csv_path=params.n_value_csv_filename)
    if model_key == "20K":
        return base_ic, base_n
    if model_key == "4p2K_iconly":
        return Fujikura4p2KScaledIcModel(base_ic), base_n
    if model_key == "4p2K_icn":
        return (Fujikura4p2KScaledIcModel(base_ic),
                Fujikura4p2KScaledNValueModel(base_n))
    raise ValueError(model_key)


def run_one(domain, uniform_setup, ta, ic_model, n_model, I_amps,
            alpha_high, alpha_low, min_iters, warm_start, tag, phase):
    params.ta_picard_alpha = alpha_high
    params.ta_picard_alpha_fine = alpha_low
    diag_path = os.path.join(DIAG_DIR, f"{tag}.csv")
    if os.path.exists(diag_path):
        os.remove(diag_path)
    print(f"[{phase}] >>> {tag}  I={I_amps:g}A  alpha=({alpha_high},"
          f"{alpha_low})  warm={warm_start}  min_iters={min_iters}")
    t0 = time.time()
    try:
        A_h, B_h, T_h, info = ta_solve.solve_ta_at_current(
            domain, ta, uniform_setup, I_amps=I_amps, ic_model=ic_model,
            n_model=n_model, verbose=False, warm_start=warm_start,
            min_iters=min_iters, diag_log_path=diag_path)
    except Exception as e:
        wall = time.time() - t0
        print(f"[{phase}] !!! {tag} FAILED after {wall:.1f}s: {e}")
        traceback.print_exc()
        record(dict(phase=phase, tag=tag, I_A=I_amps, alpha_high=alpha_high,
                     alpha_low=alpha_low, min_iters=min_iters,
                     warm_start=warm_start, error=str(e), wall_s=wall))
        return None
    wall = time.time() - t0
    margin_arr, clip_ta, cents, J_arr = _local_margin(domain, ta, B_h,
                                                       ic_model)
    p5 = float(np.percentile(margin_arr, 5))
    row = dict(phase=phase, tag=tag, I_A=I_amps, alpha_high=alpha_high,
               alpha_low=alpha_low, min_iters=min_iters,
               warm_start=warm_start, n_iters=info["n_iters"],
               converged_flag=info["converged"],
               rel_err_final=info["rel_err"], scif_mT=info["scif_mT"],
               scif_stall_mT=info["scif_stall_mT"],
               worst_margin=float(margin_arr.min()), p5_margin=p5,
               median_margin=float(np.median(margin_arr)),
               clip_frac=clip_ta, wall_s=wall, error="")
    record(row)
    print(f"[{phase}] <<< {tag}  done in {wall:.1f}s  n_iters={info['n_iters']} "
          f"converged_flag={info['converged']}  rel_err={info['rel_err']:.3e}  "
          f"worst_margin={row['worst_margin']:.3f}  p5_margin={p5:.3f}")
    return row


I_REF = 25.0   # CLAUDE.md's own fixed-current reference point for this investigation

PHASE_A_CONFIGS = [
    ("medium", "medium5"),
    ("medium", "uniform5"),
    ("medium", "xdense7"),
    ("fine", "medium5"),
]

PHASE_B_MODELS = ["4p2K_iconly", "4p2K_icn"]

ALPHA_SWEEP = [(0.30, 0.15), (0.15, 0.08), (0.10, 0.05), (0.05, 0.02),
               (0.03, 0.01)]

I_SWEEP = [5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]


def phase_ab():
    """z-resolution convergence on 20K, then the same grid for both 4.2K
    variants, all at I_REF, cold start, forced full length."""
    results = {}
    for tier, zgrade in PHASE_A_CONFIGS:
        apply_mesh_config(tier, zgrade)
        mesh_tag = f"{tier}_{zgrade}"
        print(f"\n=== building mesh: {mesh_tag} "
              f"(mesh_size_min_factor={params.mesh_size_min_factor}, "
              f"nz_per_layer={params.mesh_nz_per_layer}, "
              f"z_grading={params.mesh_z_grading}) ===")
        t0 = time.time()
        domain, uniform_setup, ta = build_setup(mesh_tag)
        print(f"=== mesh {mesh_tag} built+setup in {time.time()-t0:.1f}s ===")
        min_iters = int(params.ta_n_picard)

        for model_key in ["20K"] + PHASE_B_MODELS:
            ic_model, n_model = make_models(model_key)
            tag = f"phaseAB_{mesh_tag}_{model_key}_I{I_REF:g}_default_alpha"
            phase = "A" if model_key == "20K" else "B"
            row = run_one(domain, uniform_setup, ta, ic_model, n_model,
                          I_REF, DEFAULT_ALPHA[0], DEFAULT_ALPHA[1],
                          min_iters, warm_start=False, tag=tag, phase=phase)
            results[(mesh_tag, model_key)] = row
    return results


def phase_c(mesh_tag, tier, zgrade):
    """Relaxation-pair sweep on the 4.2K Ic+n-matched model."""
    apply_mesh_config(tier, zgrade)
    print(f"\n=== phase C: rebuilding mesh {mesh_tag} for alpha sweep ===")
    domain, uniform_setup, ta = build_setup(f"C_{mesh_tag}")
    ic_model, n_model = make_models("4p2K_icn")
    min_iters = int(params.ta_n_picard)
    rows = []
    for ah, al in ALPHA_SWEEP:
        tag = f"phaseC_{mesh_tag}_4p2K_icn_I{I_REF:g}_a{ah}_{al}"
        row = run_one(domain, uniform_setup, ta, ic_model, n_model, I_REF,
                      ah, al, min_iters, warm_start=False, tag=tag,
                      phase="C")
        rows.append(row)
    return rows, (domain, uniform_setup, ta)


def phase_d(domain, uniform_setup, ta, mesh_tag, alpha_pair, model_key):
    """Direct fixed-current sweep (bypasses ta_safe_current.py's fragile
    5-iteration bisection) to map margin(I) cleanly for one model."""
    ic_model, n_model = make_models(model_key)
    min_iters = int(params.ta_n_picard)
    prev_I = None
    for I_amps in I_SWEEP:
        tag = f"phaseD_{mesh_tag}_{model_key}_I{I_amps:g}_a{alpha_pair[0]}_{alpha_pair[1]}"
        warm = prev_I is not None
        run_one(domain, uniform_setup, ta, ic_model, n_model, I_amps,
                alpha_pair[0], alpha_pair[1], min_iters, warm_start=warm,
                tag=tag, phase="D")
        prev_I = I_amps


def main():
    t_start = time.time()
    print(f"=== ta_thermal_validation starting, PID={os.getpid()} ===")
    print(f"summary CSV: {SUMMARY_CSV}")

    results = phase_ab()

    # Pick the z-config with the smallest 20K rel_err_final (best-converged
    # baseline) as phase C/D's mesh -- if ties/errors, fall back to medium5.
    best_mesh_tag, best_tier, best_zgrade = None, "medium", "medium5"
    best_rel_err = np.inf
    for (mesh_tag, model_key), row in results.items():
        if model_key == "20K" and row is not None and row.get("error", "") == "":
            if row["rel_err_final"] < best_rel_err:
                best_rel_err = row["rel_err_final"]
                best_mesh_tag = mesh_tag
    if best_mesh_tag is not None:
        best_tier, best_zgrade = best_mesh_tag.split("_", 1)
    print(f"\n=== phases A/B done ({time.time()-t_start:.1f}s elapsed). "
          f"Best-converged 20K mesh config: {best_tier}/{best_zgrade} "
          f"(rel_err={best_rel_err:.3e}) ===\n")

    c_rows, (domain, uniform_setup, ta) = phase_c(
        f"{best_tier}_{best_zgrade}", best_tier, best_zgrade)

    # Pick the smallest-rel_err alpha pair from phase C for phase D.
    best_alpha = DEFAULT_ALPHA
    best_c_rel_err = np.inf
    for row in c_rows:
        if row is not None and row.get("error", "") == "":
            if row["rel_err_final"] < best_c_rel_err:
                best_c_rel_err = row["rel_err_final"]
                best_alpha = (row["alpha_high"], row["alpha_low"])
    print(f"\n=== phase C done. Best-converged alpha for 4.2K Ic+n: "
          f"{best_alpha} (rel_err={best_c_rel_err:.3e}) ===\n")

    print("=== phase D: 20K fixed-current sweep (bypass bisection) ===")
    phase_d(domain, uniform_setup, ta, f"{best_tier}_{best_zgrade}",
            best_alpha, "20K")
    print("=== phase D: 4.2K Ic+n fixed-current sweep (bypass bisection) ===")
    phase_d(domain, uniform_setup, ta, f"{best_tier}_{best_zgrade}",
            best_alpha, "4p2K_icn")

    print(f"\n=== ta_thermal_validation ALL PHASES DONE in "
          f"{time.time()-t_start:.1f}s ===")


if __name__ == "__main__":
    main()
