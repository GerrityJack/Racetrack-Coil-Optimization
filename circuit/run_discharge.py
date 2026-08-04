"""
run_discharge.py — sudden discharge / shutdown of the NI coil (DCN model).

Scenario: the coil is at DC steady state (every turn carrying I_op, zero
radial current) and the supply is OPENED at t = 0.  The terminal current goes
to zero, but the coil current cannot: the stored energy circulates through
the turn-to-turn contacts and decays on the same L/R time constant that
governs charging.

This is the case that actually sizes the thermal problem.  ALL of the stored
energy ends up in the winding -- there is no external dump resistor in an NI
coil, that is the entire point of the technique -- so the energy balance

    integral P_contact dt  ==  W_stored = 1/2 L I_op^2

is both a physical result and a hard self-check on the whole assembly (M, R
and the integrator together).  It is asserted below.

NOTE this is an ISOTHERMAL electromagnetic model.  It tells you how many
joules land in the winding and how fast, but not the temperature rise -- the
thermal problem is deliberately out of scope for this phase.

Run:  <env>/bin/python3 circuit/run_discharge.py
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

import params                      # noqa: E402
import cparams as cfg              # noqa: E402
import dcn as dcn_mod              # noqa: E402
import inductance as ind           # noqa: E402
from geometry import CoilGeometry  # noqa: E402


def run_one(dcn, I_op, t_end, n_out=600, verbose=True):
    """Open the supply at t=0 from the DC steady state."""
    N = dcn.N
    i0 = np.full(N, I_op)             # DC state: no radial current
    # LOG-spaced: the dissipated power peaks at t=0+ and decays as exp(-2t/tau),
    # so a linear grid over-estimates the energy integral by trapezoid error on
    # a strongly convex curve (it broke the energy balance by 9% at the
    # shortest tau).  Log spacing resolves the initial spike.
    t_eval = np.concatenate([[0.0],
                             np.geomspace(1e-4 * t_end, t_end, n_out - 1)])

    t0 = time.time()
    sol = solve_ivp(lambda t, y: dcn.rhs(t, y, lambda _t: 0.0),
                    (0.0, t_end), i0, t_eval=t_eval,
                    method=cfg.ODE_METHOD, rtol=cfg.ODE_RTOL,
                    atol=cfg.ODE_ATOL, max_step=cfg.ODE_MAX_STEP_FRAC * t_end)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    wall = time.time() - t0

    t, Y = sol.t, sol.y
    Bz = np.array([dcn.bore_Bz(Y[:, k]) for k in range(len(t))])
    i_sp = np.array([dcn.spiral_current(Y[:, k]) for k in range(len(t))])
    P = np.array([dcn.power(Y[:, k], 0.0) for k in range(len(t))])
    P_c, P_s = P[:, 0], P[:, 1]
    E_c = np.concatenate([[0.0], np.cumsum(0.5 * (P_c[1:] + P_c[:-1])
                                           * np.diff(t))])
    E_s = np.concatenate([[0.0], np.cumsum(0.5 * (P_s[1:] + P_s[:-1])
                                           * np.diff(t))])

    Bz0 = Bz[0]
    tau = np.nan
    m = (np.abs(Bz) < 0.6 * abs(Bz0)) & (np.abs(Bz) > 0.02 * abs(Bz0))
    if m.sum() >= 3:
        p = np.polyfit(t[m], np.log(np.abs(Bz[m])), 1)
        tau = -1.0 / p[0] if p[0] < 0 else np.nan

    out = dict(t=t, Bz=Bz, i_spiral=i_sp, P_contact=P_c, P_sc=P_s,
               E_contact=E_c, E_sc=E_s, Y=Y, wall_s=wall, tau_s=float(tau),
               Bz0=float(Bz0), I_op=I_op,
               E_contact_total_J=float(E_c[-1]),
               E_sc_total_J=float(E_s[-1]),
               P_contact_peak_W=float(P_c.max()))
    if verbose:
        print(f"    solved in {wall:.1f} s   tau={tau:.2f} s   "
              f"E_dissipated={E_c[-1]:.1f} J   peak P={P_c.max():.1f} W",
              flush=True)
    return out


def main():
    print("=" * 78)
    print("NI sudden discharge (supply opened) — DCN model")
    print("=" * 78)
    geom = CoilGeometry.from_params()
    I_op = float(params.I_design)
    print(f"{geom}")
    print(f"I_op = {I_op:.2f} A\n")

    rows = []
    series = {}
    for rc in cfg.RHO_CT_SWEEP_UOHM_CM2:
        print(f"--- rho_c = {rc:.0f} uOhm.cm^2 " + "-" * 40)
        d = dcn_mod.build(geom, rho_ct_uohm_cm2=rc, verbose=True)
        L_tot = ind.total_inductance(d.M, d.groups, geom.two_coil)
        W_stored = 0.5 * L_tot * I_op ** 2

        # integrate long enough to dissipate essentially everything
        r = run_one(d, I_op, t_end=max(cfg.DISCHARGE_HOLD_S, 400.0))
        # extend if the energy balance says we stopped early
        while r["E_contact_total_J"] < 0.97 * W_stored and \
                r["t"][-1] < 2.0e4:
            r = run_one(d, I_op, t_end=r["t"][-1] * 3.0, verbose=False)

        bal = r["E_contact_total_J"] / W_stored
        print(f"    W_stored = {W_stored:.1f} J   "
              f"dissipated = {r['E_contact_total_J']:.1f} J   "
              f"balance = {bal*100:.2f}%   "
              f"{'OK' if 0.95 < bal < 1.05 else 'CHECK'}")
        rows.append(dict(rho_ct_uohm_cm2=rc, tau_s=r["tau_s"],
                         W_stored_J=W_stored,
                         E_dissipated_J=r["E_contact_total_J"],
                         energy_balance_pct=bal * 100.0,
                         P_peak_W=r["P_contact_peak_W"],
                         t_end_s=float(r["t"][-1]),
                         wall_s=r["wall_s"]))
        series[rc] = r

    out_csv = os.path.join(cfg.RUNS_DIR, "discharge_sweep.csv")
    with open(out_csv, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nwrote {out_csv}")

    np.savez_compressed(
        os.path.join(cfg.RUNS_DIR, "discharge_series.npz"),
        **{f"rc{int(k)}_{f}": v for k, r in series.items()
           for f, v in r.items()
           if isinstance(v, np.ndarray) or np.isscalar(v)})

    print("\n" + "=" * 78)
    print(f"{'rho_c':>8} {'tau':>9} {'W_stored':>11} {'E_diss':>11} "
          f"{'balance':>9} {'P_peak':>10}")
    print(f"{'uO.cm2':>8} {'s':>9} {'J':>11} {'J':>11} {'%':>9} {'W':>10}")
    print("-" * 78)
    for r in rows:
        print(f"{r['rho_ct_uohm_cm2']:8.0f} {r['tau_s']:9.2f} "
              f"{r['W_stored_J']:11.1f} {r['E_dissipated_J']:11.1f} "
              f"{r['energy_balance_pct']:9.2f} {r['P_peak_W']:10.1f}")
    print("=" * 78)
    print("\nAll of this energy lands in the winding -- an NI coil has no "
          "external dump path.\nThe temperature rise it causes is NOT "
          "modelled here (isothermal, EM only).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
