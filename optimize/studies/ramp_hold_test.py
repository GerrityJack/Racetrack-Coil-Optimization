"""
ramp_hold_test.py — time-marched ramp + hold at fixed geometry (20 K).
======================================================================
2026-09-26. Question: does a design reach 10 T with NO coil cell above its
local Jc (E_c = 1 uV/cm criterion) in a realistic operating scenario?

With the power-law E-J model, J/Jc = (E/E_c)^(1/n), so "J > Jc" means the
local electric field exceeds E_c. During a ramp E is set by the ramp rate;
after the ramp the magnet is HELD at constant current and E relaxes (flux
creep). A single implicit step over the whole ramp (every production T-A
number in this project) only sees the ramp. This script marches in time:
each step is one backward-Euler step of length dt at transport current I,
with the previous step's converged A as ta["A_prev"] (the hook
setup_ta_problem already provides; E_induced = -(A - A_prev)/dt).

Steps: RH_SEQUENCE="I:dt:eps:adv,..."
    adv=1  new time step (A_prev <- current converged A; for the very
           first step A_prev stays 0 = virgin/ZFC state)
    adv=0  re-solve the SAME time step with a different eps (rho-floor
           continuation), A_prev unchanged
    optional 5th/6th fields: alpha (used for BOTH Picard phases) and
    forced Picard length for that step, e.g. "185:3600:0.8:0:0.02:400".
    Stability needs alpha roughly proportional to rho_floor*dt: the
    T-equation sees the self-field from the previous iterate (explicit
    inductive coupling, loop gain ~ mu0*w*D/(rho*dt)), so a lower floor or
    a shorter step needs a smaller alpha and proportionally more iterations.
Every solve warm-starts from the previous one (the first is cold).

Other env vars: RH_DESIGN_JSON (geometry, as relaxed_margin_test.py),
RH_ALPHA="0.10,0.05", RH_N_PICARD (forced Picard length, default 150),
RH_TAG (output names), RH_SAVE_STEPS="3,5" (npz snapshot after those step
indices), RH_MESH="medium" / RH_ZGRID="medium5".

Convergence: rel_err is NOT a usable check (see CLAUDE.md, 2026-09-26);
each row reports scif_range_mT = max-min of the raw SCIF over the last 30
Picard iterations and scif_slope_mT = mean SCIF change per iteration over
them. Outputs: optimize/runs/ramp_hold/<tag>.csv, diag/<tag>_s<k>.csv.
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
from ta_thermal_validation import apply_mesh_config, build_setup, make_models
import ta_solve
import ta_implicit
from ta_safe_current import _local_margin
from relaxed_margin_test import b_target

OUT_DIR = os.path.join(_ROOT, "optimize", "runs", "ramp_hold")
DIAG_DIR = os.path.join(OUT_DIR, "diag")
os.makedirs(DIAG_DIR, exist_ok=True)

TAG = os.environ.get("RH_TAG", "champion_ramphold")
DESIGN = json.loads(os.environ.get("RH_DESIGN_JSON", "null"))
ALPHA = tuple(float(x) for x in
              os.environ.get("RH_ALPHA", "0.10,0.05").split(","))
N_PICARD = int(os.environ.get("RH_N_PICARD", "150"))
SAVE_STEPS = {int(x) for x in os.environ.get("RH_SAVE_STEPS", "").split(",")
              if x.strip()}
# RH_SOLVER=implicit uses solve/ta_implicit.py (implicit T-A coupling,
# per-cell Newton rho update; the step's alpha field is ignored and its
# iteration field is the number of OUTER iterations).
SOLVER = os.environ.get("RH_SOLVER", "picard")
RHO_UPDATE = os.environ.get("RH_RHO_UPDATE", "newton")
STEPS = []
for s in os.environ["RH_SEQUENCE"].split(","):
    f = s.split(":")
    STEPS.append((float(f[0]), float(f[1]), float(f[2]), int(f[3]),
                  (float(f[4]), float(f[4])) if len(f) > 4 else ALPHA,
                  int(f[5]) if len(f) > 5 else N_PICARD,
                  f[6] if len(f) > 6 else SOLVER))


def trace_stats(path, n=30):
    k, scif = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            k.append(int(r["k"]))
            scif.append(float(r["scif_mT"]))
    s = np.array(scif[-n:])
    return float(s.max() - s.min()), float((s[-1] - s[0]) / max(len(s) - 1, 1))


def efield_ratio(ta, J_arr):
    """E/E_c per coil cell from the rho actually used last (rho_fn is the
    homogenised resistivity; the SC-layer value is rho*Lambda/delta)."""
    rho_h = ta["rho_fn"].x.array[ta["coil_cells"]]
    rho_sc = rho_h * float(ta["Lambda"]) / float(ta["delta_SC"])
    n_hat = ta["n_hat_coil"]
    Jd = np.einsum("ij,ij->i", J_arr, n_hat)
    Jm = np.linalg.norm(J_arr - Jd[:, None] * n_hat, axis=-1)
    return rho_sc * Jm / 1e-4


def main():
    if DESIGN is not None:
        params.a = float(DESIGN["a"])
        params.b = float(DESIGN["b"])
        params.coil_half_gap = float(DESIGN["coil_half_gap"])
        params.n_turns = list(DESIGN["n_turns"])
    apply_mesh_config(os.environ.get("RH_MESH", "medium"),
                      os.environ.get("RH_ZGRID", "medium5"))
    print(f"design: a={params.a*1e3:.2f}mm b={params.b*1e3:.2f}mm "
          f"gap={params.coil_half_gap*1e3:.2f}mm n_turns={params.n_turns} "
          f"alpha={ALPHA} n_picard={N_PICARD}")
    domain, uniform_setup, ta = build_setup("rh")
    ic_model, n_model = make_models("20K")
    # RH_IC: Ic(B) extrapolation above the 8 T data ceiling (default kim,
    # as make_models); e.g. RH_IC=scaling:45 for the conservative model
    if os.environ.get("RH_IC"):
        from ic_extrapolation import make_ic_model
        ic_model = make_ic_model(os.environ["RH_IC"])
        print(f"Ic model: {os.environ['RH_IC']}")

    out_csv = os.path.join(OUT_DIR, f"{TAG}.csv")
    new = not os.path.exists(out_csv)
    t_phys = 0.0
    first = True
    for idx, (I, dt, eps, adv, alpha, n_it, solver) in enumerate(STEPS):
        params.ta_picard_alpha, params.ta_picard_alpha_fine = alpha
        params.ta_n_picard = n_it
        if adv and not first:
            ta["A_prev"].x.array[:] = ta["A_h"].x.array
            ta["A_prev"].x.scatter_forward()
        if adv:
            t_phys += dt
        ta["dt_const"].value = dt
        params.ta_eps_reg = eps
        diag = os.path.join(DIAG_DIR, f"{TAG}_s{idx}.csv")
        if os.path.exists(diag):
            os.remove(diag)
        t0 = time.time()
        try:
            if solver == "implicit":
                _, B_h, _, info = ta_implicit.solve_ta_implicit(
                    domain, ta, uniform_setup, I_amps=I, ic_model=ic_model,
                    n_model=n_model, n_outer=n_it, warm_start=not first,
                    rho_update=RHO_UPDATE, diag_log_path=diag,
                    verbose=os.environ.get("RH_VERBOSE") == "1")
            else:
                _, B_h, _, info = ta_solve.solve_ta_at_current(
                    domain, ta, uniform_setup, I_amps=I, ic_model=ic_model,
                    n_model=n_model, verbose=False, warm_start=not first,
                    min_iters=n_it, diag_log_path=diag)
        except (RuntimeError, FloatingPointError) as e:   # NaN etc. -- state is unusable after this
            print(f"[{idx}] FAILED (eps={eps:g}, alpha={alpha}): {e}")
            break
        first = False
        m, clip, cents, J_arr = _local_margin(domain, ta, B_h, ic_model)
        load = 1.0 / m
        e_rat = efield_ratio(ta, J_arr)
        Bt, Bu, unif = b_target(domain, ta, I)
        rng, slope = trace_stats(diag)
        row = dict(step=idx, solver=solver, I_A=I, dt_s=dt, t_phys_s=t_phys, eps_reg=eps,
                   adv=adv, alpha=alpha[1], n_iter=n_it, B_target_T=Bt, B_uniformJ_T=Bu, unif_pct=unif,
                   max_load=load.max(), p999_load=np.percentile(load, 99.9),
                   p99_load=np.percentile(load, 99),
                   p95_load=np.percentile(load, 95),
                   median_load=np.median(load),
                   n_over_Ic=int(np.sum(load > 1.0)),
                   frac_over_Ic=float(np.mean(load > 1.0)),
                   frac_over_95=float(np.mean(load > 0.95)),
                   frac_over_90=float(np.mean(load > 0.90)),
                   max_E_over_Ec=float(e_rat.max()),
                   n_cells=len(m), scif_mT=info["scif_mT"],
                   scif_range_mT=rng, scif_slope_mT=slope,
                   rel_err=info["rel_err"], clip_frac=clip,
                   wall_s=time.time() - t0)
        with open(out_csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
                new = False
            w.writerow(row)
        print(f"[{idx}] I={I:6.1f}A dt={dt:8.0f}s t={t_phys:8.0f}s "
              f"eps={eps:g} a={alpha[1]:g}x{n_it}  B={Bt:6.3f}T unif={unif:.3f}%  "
              f"maxload={row['max_load']:.3f} p99={row['p99_load']:.3f} "
              f">Ic={row['n_over_Ic']}/{len(m)}  maxE/Ec={row['max_E_over_Ec']:.2g}  "
              f"SCIF={info['scif_mT']:.1f}mT range30={rng:.2f} "
              f"slope={slope:+.3f}  ({row['wall_s']:.0f}s)")
        if idx in SAVE_STEPS:
            path = os.path.join(OUT_DIR, f"{TAG}_s{idx}_fields.npz")
            cells = ta["coil_cells"]
            np.savez(path, I_solved=I, delta_SC=float(ta["delta_SC"]),
                     coil_centroids=cents,
                     coil_B=B_h.x.array.reshape(-1, 3)[cells],
                     J_TA_coil=J_arr,
                     J_unif_coil=ta["t_hat_coil"] * (
                         I / (float(ta["delta_SC"]) * params.w)),
                     dV=ta["coil_vols"], box_ptp_pct=unif, load=load,
                     E_over_Ec=e_rat, a=params.a, b=params.b,
                     coil_half_gap=params.coil_half_gap,
                     n_turns=np.array(params.n_turns))
            print(f"    saved {path}")


if __name__ == "__main__":
    main()
