"""
power_ramp.py -- 2026-08-08, constant-supply-power ramp-up analysis for the
NI coil (DCN model, Phase A).

QUESTION THIS ANSWERS
----------------------
A real supply delivers P(t) = I(t) * V_terminal(t), not a prescribed
current schedule. Given a POWER limit (not a current or voltage limit),
what is the highest constant power P that ramps the coil from 0 to
I_design as fast as possible without the local transport current, at any
turn group and any instant, coming close to its local Ic(B, theta)?

WHY NOT THE NAIVE P = L*I*dI/dt
---------------------------------
That single-lumped-inductor formula is singular at I=0 (dI/dt -> infinity
for any finite P as I -> 0) and ignores the NI winding's parallel
turn-to-turn contact path entirely. The DCN model (circuit/dcn.py,
VALIDATED -- 0.00-0.08% energy-balance closure, 0.18%/0.44% median/max
Biot-Savart agreement, He et al. 2025 benchmark to 4.2%) already gives
the EXACT self-consistent supply voltage as `terminal_voltage()`, and
that expression is LINEAR in I(t) at fixed state i_turn (turn-to-turn
contact resistance is a pure geometry/material property, not a function
of I or field):

    V(I, i) = c1*I - c2(i)
    c1 = k * sum(n * R_ct)          (constant)
    c2(i) = k * sum(n * R_ct * i)   (state-dependent)
    k = 2 for the two-coil series device, matching DCN.terminal_voltage.

Constant power P0 = I * V(I, i) is then a plain quadratic in I at every
instant, given the current turn-group spiral currents i_turn(t):

    c1*I^2 - c2(i)*I - P0 = 0
    I = [c2(i) + sqrt(c2(i)^2 + 4*c1*P0)] / (2*c1)      (positive root)

At i=0 (cold start) this gives I(0) = sqrt(P0/c1), FINITE -- no
singularity, because the real parallel contact path draws a finite
current from cold even though the spiral/inductive branch cannot. This
control law is solved algebraically at every ODE RHS evaluation; no
change to the branch ODE's physics, and no need to touch the
exploratory, short-dt-fragile T-A transient solver (transient/) for the
ramp trajectory itself.

QUENCH MARGIN -- WHAT THIS SCRIPT DOES AND DOES NOT CHECK
------------------------------------------------------------
The DCN model resolves inter-turn (radial-leakage) current sharing only
-- it has no notion of intra-tape screening-current concentration (that
is the T-A formulation's job, see solve/ta_solve.py and CLAUDE.md's "The
T-A formulation"). The margin reported here (min over turn-group sample
points of Ic(B,theta)/|i_turn|, using local_Ic_n -- the SAME per-turn,
per-sample-point Ic(B,theta) evaluation the DCN already uses for its own
V_sc superconducting-branch voltage) is therefore a NECESSARY, not
SUFFICIENT, safety check. Per this project's own established practice
(docs/HISTORY.md, "Proxy graveyard": every cheap uniformity/quench proxy
tried in this project has needed independent T-A confirmation before
being trusted), the P recommended here MUST be spot-checked against a
T-A run at the resulting critical current/ramp-rate before being
trusted -- see transient/validation/ for that follow-up.

SAFETY THRESHOLD
------------------
Per user direction, "close to quenching" is defined identically to the
champion design's own steady-state operating philosophy: the coil
operates at I_design = 65% of local Ic (CLAUDE.md, "Current design"), so
the required margin Ic/|i_turn| >= 1/0.65 ~= 1.538 must hold EVERYWHERE,
at every instant of the ramp AND the post-ramp hold, not just at the
final steady state.

CONTACT RESISTIVITY rho_c
----------------------------
The NI free parameter, not derivable from anything in params.py (same
caveat as circuit/run_charge.py). Swept across the project's existing
validated range [30, 100, 400] uOhm.cm^2 rather than assumed at the
nominal 100.

Run:  <env>/bin/python3 circuit/power_ramp.py
"""

import csv
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import lu_solve

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

MARGIN_REQUIRED = 1.0 / 0.65      # matches the champion's 65%-of-Ic operating point
T_HOLD_S = 1800.0                 # long enough to resolve tau at every rho_c tried
N_OUT_RAMP = 150
N_OUT_HOLD = 150


# ── the control law ─────────────────────────────────────────────────────────

def _coeffs(dcn):
    """c1 (const), and a function c2(i_turn) -- both from DCN.terminal_voltage's
    own linear-in-I structure. k=2 for the two-coil series device."""
    k = 2.0 if dcn.geom.two_coil else 1.0
    c1 = k * float(np.sum(dcn.n * dcn.R_ct))
    return k, c1


def _make_control(dcn, P0):
    """Returns I_from_state(i_turn) closing over this dcn/P0."""
    k, c1 = _coeffs(dcn)
    n_Rct = dcn.n * dcn.R_ct

    def I_from_state(i_turn):
        c2 = k * float(np.sum(n_Rct * i_turn))
        disc = c2 * c2 + 4.0 * c1 * P0
        return (c2 + np.sqrt(disc)) / (2.0 * c1)

    return I_from_state


def _rhs_power(t, i_turn, dcn, I_from_state):
    I = I_from_state(i_turn)
    b = dcn.R_ct * (I - i_turn) - dcn.V_sc(i_turn)
    return lu_solve(dcn._lu, b)


def _rhs_fixed(t, i_turn, dcn, I_fixed):
    b = dcn.R_ct * (I_fixed - i_turn) - dcn.V_sc(i_turn)
    return lu_solve(dcn._lu, b)


def _margin(dcn, i_turn):
    """min over (turn group, sample point) of Ic(B,theta)/|i_turn|."""
    Ic, _, _, _ = dcn.local_Ic_n(i_turn)
    m = Ic / np.maximum(np.abs(i_turn)[:, None], 1e-12)
    idx = np.unravel_index(np.argmin(m), m.shape)
    return float(m[idx]), idx


# ── one full ramp+hold run at a given constant power ────────────────────────

def run_power_ramp(dcn, P0, I_design, t_max_ramp_guess, verbose=False):
    """Integrate the ramp phase (constant power, event-terminated at
    I=I_design) then a fixed-I_design hold phase. Returns a dict of time
    series plus the minimum quench margin observed anywhere in either
    phase."""
    N = dcn.N
    I_from_state = _make_control(dcn, P0)

    def event_reached(t, i_turn, dcn_, I_from_state_):
        return I_from_state_(i_turn) - I_design
    event_reached.terminal = True
    event_reached.direction = 1

    t_span = (0.0, t_max_ramp_guess)
    i0 = np.zeros(N)

    if I_from_state(i0) >= I_design:
        # P is so large that even the cold-start (i=0) control current
        # already reaches I_design -- solve_ivp's event detector only
        # catches a SIGN CHANGE, so it would miss a crossing that already
        # happened at t=0. Treat the ramp as instantaneous instead of
        # integrating a phase that never really exists.
        t_ramp_end = 0.0
        i_ramp_end = i0
    else:
        sol = solve_ivp(_rhs_power, t_span, i0, args=(dcn, I_from_state),
                        method=cfg.ODE_METHOD, rtol=cfg.ODE_RTOL, atol=cfg.ODE_ATOL,
                        events=event_reached, dense_output=True,
                        max_step=cfg.ODE_MAX_STEP_FRAC * t_max_ramp_guess)
        if not sol.success:
            raise RuntimeError(f"ramp phase failed: {sol.message}")
        if len(sol.t_events[0]) == 0:
            # never reached I_design within the guessed span -- signal caller
            # to retry with a longer span rather than silently truncating.
            return dict(reached=False, t_ramp_end=np.nan)

        t_ramp_end = float(sol.t_events[0][0])
        i_ramp_end = sol.y_events[0][0]

    if t_ramp_end == 0.0:
        t_ramp_eval = np.zeros(1)
        i_ramp = i0.reshape(-1, 1)
    else:
        t_ramp_eval = np.linspace(0.0, t_ramp_end, N_OUT_RAMP)
        i_ramp = sol.sol(t_ramp_eval)
    I_ramp = np.array([I_from_state(i_ramp[:, j]) for j in range(len(t_ramp_eval))])

    sol2 = solve_ivp(_rhs_fixed, (t_ramp_end, t_ramp_end + T_HOLD_S), i_ramp_end,
                     args=(dcn, I_design), method=cfg.ODE_METHOD,
                     rtol=cfg.ODE_RTOL, atol=cfg.ODE_ATOL, dense_output=True,
                     max_step=cfg.ODE_MAX_STEP_FRAC * T_HOLD_S)
    if not sol2.success:
        raise RuntimeError(f"hold phase failed: {sol2.message}")
    t_hold_eval = t_ramp_end + np.geomspace(1e-3 * T_HOLD_S, T_HOLD_S, N_OUT_HOLD)
    i_hold = sol2.sol(t_hold_eval)
    I_hold = np.full(len(t_hold_eval), I_design)

    t = np.concatenate([t_ramp_eval, t_hold_eval])
    I = np.concatenate([I_ramp, I_hold])
    Y = np.concatenate([i_ramp, i_hold], axis=1)

    margins = np.empty(len(t))
    worst = (np.inf, None, None)
    for j in range(len(t)):
        m, idx = _margin(dcn, Y[:, j])
        margins[j] = m
        if m < worst[0]:
            worst = (m, idx, t[j])

    min_margin, worst_idx, t_worst = worst
    i_spiral_mean = np.array([dcn.spiral_current(Y[:, j]) for j in range(len(t))])
    Bz = np.array([dcn.bore_Bz(Y[:, j]) for j in range(len(t))])

    if verbose:
        print(f"    P={P0:9.1f} W  t_ramp={t_ramp_end:8.2f} s  "
              f"min_margin={min_margin:.4f} (need >= {MARGIN_REQUIRED:.4f})  "
              f"at t={t_worst:.2f}s group={worst_idx[0]}", flush=True)

    return dict(reached=True, t=t, I=I, Y=Y, margins=margins,
               t_ramp_end=t_ramp_end, min_margin=min_margin,
               worst_group=int(worst_idx[0]) if worst_idx else None,
               worst_sample=int(worst_idx[1]) if worst_idx else None,
               t_worst=float(t_worst), i_spiral_mean=i_spiral_mean, Bz=Bz,
               P0=P0)


def run_power_ramp_auto_span(dcn, P0, I_design, t_guess0, verbose=False):
    """Retries run_power_ramp with a growing t_span until the I=I_design
    event actually fires (guards against an under-guessed span silently
    truncating the ramp)."""
    t_guess = t_guess0
    for _ in range(6):
        r = run_power_ramp(dcn, P0, I_design, t_guess, verbose=verbose)
        if r["reached"]:
            return r
        t_guess *= 4.0
    raise RuntimeError(f"P={P0}: ramp never reached I_design even at "
                       f"t_span={t_guess:.1f}s")


# ── search for the fastest safe P ────────────────────────────────────────────

def find_Pmax(dcn, I_design, P_lo, P_hi, t_guess0, n_scan=14, verbose=True):
    """Coarse log-spaced scan first (to check the margin-vs-P relationship
    is monotonic, not assumed), then bisection within the bracket found."""
    Ps = np.geomspace(P_lo, P_hi, n_scan)
    scan = []
    if verbose:
        print(f"  coarse scan, P in [{P_lo:.1f}, {P_hi:.1f}] W:")
    for P in Ps:
        r = run_power_ramp_auto_span(dcn, P, I_design, t_guess0, verbose=verbose)
        scan.append((P, r["min_margin"], r["t_ramp_end"]))

    feasible = [s for s in scan if s[1] >= MARGIN_REQUIRED]
    infeasible = [s for s in scan if s[1] < MARGIN_REQUIRED]
    if not feasible:
        raise RuntimeError(f"No P in the scanned range satisfies the margin "
                           f"requirement -- lower P_lo. Scan: {scan}")
    if not infeasible:
        print(f"  WARNING: every scanned P up to {P_hi:.1f} W stayed safe -- "
              f"P_hi is not large enough to bracket the constraint.")
        return feasible[-1][0], scan

    P_safe = max(s[0] for s in feasible)
    P_unsafe = min(s[0] for s in infeasible if s[0] > P_safe) if any(
        s[0] > P_safe for s in infeasible) else min(s[0] for s in infeasible)
    if P_unsafe <= P_safe:
        # non-monotonic in this bracket -- fall back to the largest safe
        # scan point rather than bisecting blindly.
        print(f"  WARNING: margin-vs-P not monotonic in the scanned range "
              f"(safe point {P_safe:.1f}W sits above an unsafe point at "
              f"{P_unsafe:.1f}W) -- returning the largest confirmed-safe "
              f"scanned P instead of bisecting.")
        return P_safe, scan

    if verbose:
        print(f"  bisecting between safe P={P_safe:.1f} W and "
              f"unsafe P={P_unsafe:.1f} W")
    lo, hi = P_safe, P_unsafe
    for _ in range(12):
        mid = np.sqrt(lo * hi)   # log-midpoint
        r = run_power_ramp_auto_span(dcn, mid, I_design, t_guess0, verbose=verbose)
        scan.append((mid, r["min_margin"], r["t_ramp_end"]))
        if r["min_margin"] >= MARGIN_REQUIRED:
            lo = mid
        else:
            hi = mid
    return lo, scan


# ── driver ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("Constant-power ramp-up analysis -- DCN model (Phase A)")
    print("=" * 78)
    geom = CoilGeometry.from_params()
    I_design = float(params.I_design)
    print(f"{geom}")
    print(f"I_design = {I_design:.2f} A   MARGIN_REQUIRED = {MARGIN_REQUIRED:.4f} "
          f"(= 1/0.65, the champion's steady-state design margin)\n")

    # crude first guess for the ramp time span, from the naive L*I*dI/dt
    # energy estimate (ignoring dissipation, which only makes the true
    # answer longer -- fine as a starting guess since the search retries
    # with a larger span if the event never fires).
    import inductance as ind

    rows = []
    all_scans = {}
    for rc in cfg.RHO_CT_SWEEP_UOHM_CM2:
        print(f"--- rho_c = {rc:.0f} uOhm.cm^2 " + "-" * 40)
        d = dcn_mod.build(geom, rho_ct_uohm_cm2=rc, verbose=True)
        L = ind.total_inductance(d.M, d.groups, geom.two_coil)
        E_design = 0.5 * L * I_design ** 2
        P_lo = E_design / 3600.0    # ~1 hour ramp, generous lower bound
        P_hi = E_design / 2.0       # ~2 second ramp, generous upper bound
        t_guess0 = max(30.0, E_design / P_lo)

        t0 = time.time()
        P_max, scan = find_Pmax(d, I_design, P_lo, P_hi, t_guess0)
        r_best = run_power_ramp_auto_span(d, P_max, I_design, t_guess0, verbose=False)
        wall = time.time() - t0

        rows.append(dict(rho_ct_uohm_cm2=rc, P_max_W=P_max,
                         t_ramp_s=r_best["t_ramp_end"],
                         min_margin=r_best["min_margin"],
                         worst_group=r_best["worst_group"],
                         t_worst_s=r_best["t_worst"], wall_s=wall))
        all_scans[rc] = scan
        print(f"  => P_max = {P_max:.1f} W   t_ramp = {r_best['t_ramp_end']:.1f} s   "
              f"min_margin = {r_best['min_margin']:.4f}   "
              f"(wall {wall:.1f}s)\n", flush=True)

    out_csv = os.path.join(cfg.RUNS_DIR, "power_ramp_results.csv")
    with open(out_csv, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"wrote {out_csv}")

    save_dict = {}
    for rc, scan in all_scans.items():
        save_dict[f"rc{int(rc)}_P"] = np.array([s[0] for s in scan])
        save_dict[f"rc{int(rc)}_margin"] = np.array([s[1] for s in scan])
        save_dict[f"rc{int(rc)}_tramp"] = np.array([s[2] for s in scan])
    np.savez_compressed(os.path.join(cfg.RUNS_DIR, "power_ramp_scans.npz"), **save_dict)
    print(f"wrote {os.path.join(cfg.RUNS_DIR, 'power_ramp_scans.npz')}")

    print("\n" + "=" * 78)
    print(f"{'rho_c':>8} {'P_max':>10} {'t_ramp':>9} {'min_margin':>11} "
          f"{'worst_grp':>10} {'t_worst':>9}")
    print(f"{'uO.cm2':>8} {'W':>10} {'s':>9} {'':>11} {'':>10} {'s':>9}")
    print("-" * 78)
    for r in rows:
        print(f"{r['rho_ct_uohm_cm2']:8.0f} {r['P_max_W']:10.1f} "
              f"{r['t_ramp_s']:9.2f} {r['min_margin']:11.4f} "
              f"{r['worst_group']:10d} {r['t_worst_s']:9.2f}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
