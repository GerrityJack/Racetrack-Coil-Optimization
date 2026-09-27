"""
relaxed_margin_test.py — how far must the 65%-of-local-Ic T-A condition be
relaxed for the champion (4mm tape, 20K, 6 layers) to reach 10T?
=============================================================================
2026-09-24. Project direction: tape stays 4mm, cooling stays 20K, but the
65%-of-local-Ic requirement (ta_safe_current.MARGIN_REQUIRED = 1/0.65) may
be relaxed. Quick test, NOT a search: fixed champion geometry (params.py
defaults), direct fixed-current sweep (no bisection), same validated
harness as ta_thermal_validation.py's Phase D (medium/medium5 mesh forced
in-memory, alpha=(0.03,0.01), forced full-length Picard).

At each current, records the full per-cell local-margin distribution
(Jc/|J_inplane|) and B_target (uniform-J Biot-Savart + T-A SCIF
correction, same formula as ta_safe_current.evaluate()), so any relaxed
threshold / percentile combination can be read off afterwards.

Run:
    PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
    $PY optimize/studies/relaxed_margin_test.py
"""
import os
import sys
import csv
import json
import time

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_OPT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_OPT)
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "mesh"),
           os.path.join(_ROOT, "solve"), os.path.join(_ROOT, "optimize"),
           _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import dolfinx.mesh as dmesh

import params
from ta_thermal_validation import (apply_mesh_config, build_setup,
                                   make_models)
import ta_solve
import optimize_geometry as og
from ta_safe_current import _local_margin

OUT_DIR = os.path.join(_ROOT, "optimize", "runs", "relaxed_margin")
os.makedirs(OUT_DIR, exist_ok=True)
# Optional design override: RELAX_DESIGN_JSON='{"a":..,"b":..,
# "coil_half_gap":..,"n_turns":[..]}' (SI units); RELAX_TAG names the CSV.
DESIGN = json.loads(os.environ.get("RELAX_DESIGN_JSON", "null"))
OUT_CSV = os.path.join(OUT_DIR, os.environ.get("RELAX_TAG", "champion")
                       + "_20K_sweep.csv")

I_SWEEP = [float(x) for x in os.environ.get(
    "RELAX_I_SWEEP", "25,50,75,100,125,150,175,196,220").split(",")]
ALPHA = (0.03, 0.01)
# RELAX_ALPHA="0.30,0.15" overrides the Picard relaxation pair. NOTE
# (2026-09-26): at alpha_fine=0.01 the SCIF error decays only as
# (1-0.01)^k, so 150 forced iterations leave ~25% of the initial error --
# check the RELAX_DIAG trace, not rel_err, before trusting a result.
if os.environ.get("RELAX_ALPHA"):
    ALPHA = tuple(float(x) for x in os.environ["RELAX_ALPHA"].split(","))
# Optional ramp-time sweep (2026-09-26): RELAX_DT_LIST="600,6000,..." solves
# EVERY current at EVERY ramp time (dt of the single implicit step, i.e.
# E = -dA/dt), each one COLD-started so no solve inherits another's
# screening state. Unset = the historical single dt=params.ramp_duration.
# Optional rho-floor sweep: RELAX_EPS_LIST="1.0,0.7,0.5" (params.ta_eps_reg,
# read fresh by every solve). The floor gives sub-critical cells at least
# rho = (E_c/Jc)*eps^(n-1); eps=1 (production) makes sub-Jc tape ohmic at the
# critical-state resistivity, which lets screening currents drain away on
# long ramps -- a result is only physical if it is insensitive to eps.
EPS_LIST = [float(x) for x in os.environ.get("RELAX_EPS_LIST", "").split(",")
            if x.strip()] or [None]
DT_LIST = [float(x) for x in os.environ.get("RELAX_DT_LIST", "").split(",")
           if x.strip()]
# Optional: save a field snapshot at the LAST sweep current, in the schema
# transient/validation/plot_quench_location.py reads.
SAVE_NPZ = os.environ.get("RELAX_SAVE_NPZ")


def b_target(domain, ta, I):
    """Box-mean |Bz| and peak-to-peak uniformity, uniform-J + T-A SCIF
    correction -- identical to ta_safe_current.evaluate()."""
    delta_SC = float(ta["delta_SC"])
    J_TA = ta_solve._J_from_T(ta, domain)
    J_unif = ta["t_hat_coil"] * (I / (delta_SC * params.w))
    dJ_s = (J_TA - J_unif) * (delta_SC / float(params.t))
    cents = dmesh.compute_midpoints(domain, domain.topology.dim,
                                    ta["coil_cells"])
    B_unif, pts = og.target_box_field(I)
    dB = np.array([ta_solve.dB_bore_from_dJ(cents, dJ_s, ta["coil_vols"],
                                            bore_pt=p) for p in pts])
    Btot = B_unif + dB
    Bmag = np.linalg.norm(Btot, axis=1)
    return (float(np.mean(np.abs(Btot[:, 2]))),
            float(np.mean(np.abs(B_unif[:, 2]))),
            float((Bmag.max() - Bmag.min()) / Bmag.mean() * 100.0))


def main():
    if DESIGN is not None:
        params.a = float(DESIGN["a"])
        params.b = float(DESIGN["b"])
        params.coil_half_gap = float(DESIGN["coil_half_gap"])
        params.n_turns = list(DESIGN["n_turns"])
    apply_mesh_config("medium", "medium5")   # calls recompute_derived()
    print(f"champion: a={params.a*1e3:.1f}mm b={params.b*1e3:.1f}mm "
          f"gap={params.coil_half_gap*1e3:.1f}mm n_turns={params.n_turns} "
          f"w={params.w*1e3:.0f}mm")
    domain, uniform_setup, ta = build_setup("relaxed")
    ic_model, n_model = make_models("20K")
    params.ta_picard_alpha, params.ta_picard_alpha_fine = ALPHA
    # RELAX_N_PICARD: override the forced Picard length (default
    # params.ta_n_picard=150). RELAX_DIAG=1 writes a per-iteration raw
    # residual trace per solve to OUT_DIR/diag/<tag>_I<I>_dt<dt>_eps<eps>.csv.
    if os.environ.get("RELAX_N_PICARD"):
        params.ta_n_picard = int(os.environ["RELAX_N_PICARD"])
    min_iters = int(params.ta_n_picard)
    diag_dir = None
    if os.environ.get("RELAX_DIAG") == "1":
        diag_dir = os.path.join(OUT_DIR, "diag")
        os.makedirs(diag_dir, exist_ok=True)

    new = not os.path.exists(OUT_CSV)
    prev = None
    runs = [(I, dt, eps) for I in I_SWEEP for dt in (DT_LIST or [None])
            for eps in EPS_LIST]
    # RELAX_SEQUENCE="I:dt:eps,I:dt:eps,..." overrides the grid with an
    # explicit ordered list (e.g. an eps_reg continuation at fixed I/dt).
    if os.environ.get("RELAX_SEQUENCE"):
        runs = [tuple(float(v) for v in step.split(":"))
                for step in os.environ["RELAX_SEQUENCE"].split(",")]
    cold = bool(DT_LIST) or EPS_LIST != [None]
    # RELAX_CONTINUE=1: continuation -- each solve after the first warm-starts
    # from the previous converged state (e.g. stepping eps_reg down at fixed
    # I/dt), instead of cold-starting. Needed because a cold solve at a
    # weaker floor does not converge.
    if os.environ.get("RELAX_CONTINUE") == "1":
        cold = False
    for I, dt, eps in runs:
        if dt is not None:
            ta["dt_const"].value = dt
        if eps is not None:
            params.ta_eps_reg = eps
        t0 = time.time()
        diag_path = None
        if diag_dir:
            diag_path = os.path.join(diag_dir, (
                f"{os.environ.get('RELAX_TAG', 'champion')}_I{I:g}"
                f"_dt{float(ta['dt_const'].value):g}"
                f"_eps{float(params.ta_eps_reg):g}.csv"))
            if os.path.exists(diag_path):
                os.remove(diag_path)
        _, B_h, _, info = ta_solve.solve_ta_at_current(
            domain, ta, uniform_setup, I_amps=I, ic_model=ic_model,
            n_model=n_model, verbose=False,
            warm_start=(prev is not None and not cold),
            min_iters=min_iters, diag_log_path=diag_path)
        prev = I
        m, clip, _, _ = _local_margin(domain, ta, B_h, ic_model)
        Bt, Bu, unif = b_target(domain, ta, I)
        # fraction of local Ic in use = 1/margin
        load = 1.0 / m
        row = dict(
            I_A=I, ramp_s=float(ta["dt_const"].value),
            eps_reg=float(params.ta_eps_reg), B_target_T=Bt, B_uniformJ_T=Bu, unif_pct=unif,
            worst_margin=m.min(), p1_margin=np.percentile(m, 1),
            p5_margin=np.percentile(m, 5), median_margin=np.median(m),
            max_load=load.max(), p99_load=np.percentile(load, 99),
            p95_load=np.percentile(load, 95),
            frac_over_Ic=float(np.mean(load > 1.0)),
            frac_over_90=float(np.mean(load > 0.9)),
            frac_over_80=float(np.mean(load > 0.8)),
            frac_over_65=float(np.mean(load > 0.65)),
            n_cells=len(m), rel_err=info["rel_err"],
            scif_mT=info["scif_mT"], clip_frac=clip,
            wall_s=time.time() - t0)
        with open(OUT_CSV, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
                new = False
            w.writerow(row)
        print(f"I={I:6.1f}A  ramp={row['ramp_s']:.0f}s  eps={row['eps_reg']:g}  B={Bt:6.3f}T  unif={unif:.3f}%  "
              f"worst={row['worst_margin']:.3f} p5={row['p5_margin']:.3f} "
              f"med={row['median_margin']:.2f}  maxload={row['max_load']:.2f} "
              f"p95load={row['p95_load']:.2f}  >Ic={row['frac_over_Ic']:.1%} "
              f">80%={row['frac_over_80']:.1%}  rel_err={info['rel_err']:.2e}  "
              f"({row['wall_s']:.0f}s)")

    if SAVE_NPZ:
        cells = ta["coil_cells"]
        np.savez(SAVE_NPZ, I_solved=I, delta_SC=float(ta["delta_SC"]),
                 coil_centroids=dmesh.compute_midpoints(
                     domain, domain.topology.dim, cells),
                 coil_B=B_h.x.array.reshape(-1, 3)[cells],
                 J_TA_coil=ta_solve._J_from_T(ta, domain),
                 J_unif_coil=ta["t_hat_coil"] * (
                     I / (float(ta["delta_SC"]) * params.w)),
                 dV=ta["coil_vols"], box_ptp_pct=unif, a=params.a,
                 b=params.b, coil_half_gap=params.coil_half_gap,
                 n_turns=np.array(params.n_turns))
        print(f"saved {SAVE_NPZ}")


if __name__ == "__main__":
    main()
