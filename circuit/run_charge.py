"""
run_charge.py — startup ramp of the NI coil (DCN model).

Answers the two questions this phase exists for:

  1. CHARGING DELAY / FIELD LAG.  The supply reaches I_op at t_ramp, but the
     bore field does not: some of the transport current is still leaking
     radially through the turn-to-turn contacts instead of going around the
     turns.  The field deficit decays as exp(-t/tau) once the ramp stops, and
     tau is fitted from that decay.

  2. AC LOSS / CRYOGENIC HEAT LOAD.  Contact loss (I_r^2 R_ct summed over
     every turn) and superconducting power-law loss, integrated over the
     schedule.

Both are swept over the contact resistivity, which is the NI free parameter
and is NOT derivable from anything in params.py (params.t is documented as a
bare 75 um pitch with no modelled insulation).

Ramp and hold are integrated as SEPARATE solve_ivp calls so the stiff solver
never steps across the dI/dt discontinuity at the end of the ramp.

Run:  <env>/bin/python3 circuit/run_charge.py
"""

import csv
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

sys.stdout.reconfigure(line_buffering=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "physics"),
           os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params                     # noqa: E402
import cparams as cfg             # noqa: E402
import dcn as dcn_mod             # noqa: E402
from geometry import CoilGeometry  # noqa: E402


def ramp_schedule(I_op, t_ramp):
    """Linear 0 -> I_op over t_ramp, flat afterwards."""
    def I_of_t(t):
        return I_op * min(1.0, max(0.0, t / t_ramp))
    return I_of_t


def integrate(dcn, I_of_t, t_span, i0, t_eval, max_step=None):
    sol = solve_ivp(lambda t, y: dcn.rhs(t, y, I_of_t),
                    t_span, i0, t_eval=t_eval,
                    method=cfg.ODE_METHOD, rtol=cfg.ODE_RTOL,
                    atol=cfg.ODE_ATOL,
                    max_step=max_step if max_step else np.inf)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol


def run_one(dcn, I_op, t_ramp, t_hold, n_out=400, verbose=True):
    """Charge then hold.  Returns a dict of time series."""
    I_of_t = ramp_schedule(I_op, t_ramp)
    N = dcn.N

    t_ramp_eval = np.linspace(0.0, t_ramp, n_out // 2)
    # LOG-spaced in the hold phase: the field deficit decays exponentially, so
    # a linear grid resolves it only when tau is comparable to t_hold.  With a
    # linear grid the tau fit returned nan for every fast case (small rho_c),
    # purely because too few samples landed inside the decay.
    t_hold_eval = t_ramp + np.geomspace(1e-3 * t_hold, t_hold, n_out // 2)

    t0 = time.time()
    s1 = integrate(dcn, I_of_t, (0.0, t_ramp), np.zeros(N), t_ramp_eval,
                   max_step=cfg.ODE_MAX_STEP_FRAC * t_ramp)
    s2 = integrate(dcn, I_of_t, (t_ramp, t_ramp + t_hold), s1.y[:, -1],
                   t_hold_eval,
                   max_step=cfg.ODE_MAX_STEP_FRAC * (t_ramp + t_hold))
    wall = time.time() - t0

    t = np.concatenate([s1.t, s2.t])
    Y = np.concatenate([s1.y, s2.y], axis=1)          # (N, T)

    I = np.array([I_of_t(tt) for tt in t])
    Bz = np.array([dcn.bore_Bz(Y[:, k]) for k in range(len(t))])
    i_sp = np.array([dcn.spiral_current(Y[:, k]) for k in range(len(t))])
    P = np.array([dcn.power(Y[:, k], I[k]) for k in range(len(t))])
    P_c, P_s = P[:, 0], P[:, 1]
    E_c = np.concatenate([[0.0], np.cumsum(0.5 * (P_c[1:] + P_c[:-1])
                                           * np.diff(t))])
    E_s = np.concatenate([[0.0], np.cumsum(0.5 * (P_s[1:] + P_s[:-1])
                                           * np.diff(t))])

    Bz_dc = dcn.bore_Bz(np.full(N, I_op))
    out = dict(t=t, I=I, Bz=Bz, Bz_dc=Bz_dc, i_spiral=i_sp,
               i_radial=I - i_sp, P_contact=P_c, P_sc=P_s,
               E_contact=E_c, E_sc=E_s, Y=Y, wall_s=wall,
               I_op=I_op, t_ramp=t_ramp, t_hold=t_hold)
    out.update(analyse(out))
    if verbose:
        print(f"    solved in {wall:.1f} s   tau={out['tau_s']:.2f} s   "
              f"lag(99%)={out['lag99_s']:.1f} s   "
              f"E_contact={E_c[-1]:.1f} J   E_sc={E_s[-1]:.2f} J",
              flush=True)
    return out


def analyse(r):
    """Fit tau from the post-ramp field deficit and measure the 99% lag."""
    t, Bz, Bz_dc = r["t"], r["Bz"], r["Bz_dc"]
    hold = t >= r["t_ramp"]
    deficit = np.abs(Bz_dc - Bz)
    scale = max(abs(Bz_dc), 1e-30)

    # tau: least-squares slope of log(deficit) over the part of the hold where
    # the deficit is resolvable (between 50% and 0.5% of its post-ramp value)
    d_hold = deficit[hold]
    t_hold = t[hold]
    tau = np.nan
    if len(d_hold) > 5 and d_hold[0] > 0:
        m = (d_hold < 0.5 * d_hold[0]) & (d_hold > 5e-3 * d_hold[0]) & \
            (d_hold > 1e-9 * scale)
        if m.sum() >= 3:
            p = np.polyfit(t_hold[m], np.log(d_hold[m]), 1)
            tau = -1.0 / p[0] if p[0] < 0 else np.nan
    if not np.isfinite(tau):
        print("      WARNING: tau fit failed (too few resolvable points in "
              "the decay window) — reported as nan, not as zero")

    # field lag: time after the ramp ends until |Bz| is within 1% of DC
    lag99 = np.nan
    within = hold & (deficit <= 0.01 * abs(Bz_dc))
    if within.any():
        lag99 = float(t[within][0] - r["t_ramp"])

    return dict(tau_s=float(tau), lag99_s=float(lag99),
                Bz_at_ramp_end=float(Bz[hold][0]),
                field_deficit_at_ramp_end_pct=float(
                    100.0 * deficit[hold][0] / abs(Bz_dc)),
                E_contact_total_J=float(r["E_contact"][-1]),
                E_sc_total_J=float(r["E_sc"][-1]))


def main():
    print("=" * 78)
    print("NI charging transient — DCN model")
    print("=" * 78)
    geom = CoilGeometry.from_params()
    I_op = float(params.I_design)
    print(f"{geom}")
    print(f"I_op = {I_op:.2f} A   (params.I_design, the champion's "
          f"quench-limited operating point)\n")

    rows = []
    series = {}
    for rc in cfg.RHO_CT_SWEEP_UOHM_CM2:
        print(f"--- rho_c = {rc:.0f} uOhm.cm^2 " + "-" * 40)
        d = dcn_mod.build(geom, rho_ct_uohm_cm2=rc, verbose=True)
        for t_ramp in cfg.RAMP_SWEEP_S:
            t_hold = max(cfg.CHARGE_HOLD_S, 6.0 * t_ramp)
            print(f"  ramp {t_ramp:7.0f} s (hold {t_hold:.0f} s):", flush=True)
            r = run_one(d, I_op, t_ramp, t_hold)
            rows.append(dict(
                rho_ct_uohm_cm2=rc, t_ramp_s=t_ramp, t_hold_s=t_hold,
                tau_s=r["tau_s"], lag99_s=r["lag99_s"],
                field_deficit_at_ramp_end_pct=
                    r["field_deficit_at_ramp_end_pct"],
                Bz_dc_T=r["Bz_dc"],
                E_contact_J=r["E_contact_total_J"],
                E_sc_J=r["E_sc_total_J"],
                mean_P_contact_W=r["E_contact_total_J"] / (t_ramp + t_hold),
                wall_s=r["wall_s"]))
            if t_ramp == cfg.CHARGE_RAMP_S:
                series[rc] = r

    out_csv = os.path.join(cfg.RUNS_DIR, "charge_sweep.csv")
    with open(out_csv, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nwrote {out_csv}")

    np.savez_compressed(
        os.path.join(cfg.RUNS_DIR, "charge_series.npz"),
        **{f"rc{int(k)}_{f}": v for k, r in series.items()
           for f, v in r.items()
           if isinstance(v, np.ndarray) or np.isscalar(v)})
    print(f"wrote {os.path.join(cfg.RUNS_DIR, 'charge_series.npz')}")

    print("\n" + "=" * 78)
    print(f"{'rho_c':>8} {'t_ramp':>8} {'tau':>9} {'lag99':>9} "
          f"{'deficit@end':>12} {'E_contact':>11} {'E_sc':>9}")
    print(f"{'uO.cm2':>8} {'s':>8} {'s':>9} {'s':>9} "
          f"{'%':>12} {'J':>11} {'J':>9}")
    print("-" * 78)
    for r in rows:
        print(f"{r['rho_ct_uohm_cm2']:8.0f} {r['t_ramp_s']:8.0f} "
              f"{r['tau_s']:9.2f} {r['lag99_s']:9.1f} "
              f"{r['field_deficit_at_ramp_end_pct']:12.3f} "
              f"{r['E_contact_J']:11.2f} {r['E_sc_J']:9.3f}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
