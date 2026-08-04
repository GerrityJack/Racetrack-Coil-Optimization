"""
he2025_racetrack.py — Tier A2: reproduce the published NI racetrack benchmark.

Reference
---------
Z. He, Y. Liu, C. Yang, J. Yang, J. Ou, C. Zhang, M. Yan, L. Li,
"A Simulation Model for the Transient Characteristics of No-Insulation
Superconducting Coils Based on T-A Formulation", Energies 2025, 18, 3669.
doi:10.3390/en18143669.  Table 2 (racetrack NI coil, tested in LN2 at 77 K):

    tape thickness            0.2  mm
    tape width               12    mm
    number of turns         115
    Ic of coil (77 K, s.f.) 190    A
    inner / outer radius     49 / 73 mm
    field constant            0.81 mT/A
    coil self-inductance      3.05 mH
    coil time constant       13.53 s      <- MEASURED, from sudden discharge
    contact resistivity     399    uOhm.cm^2

WHY THIS TEST MATTERS
---------------------
Everything else in circuit/ is self-consistent but self-referential.  This is
the only check against a physical coil that somebody actually built and
measured, and its measured tau is exactly the quantity the whole Phase A
exists to predict.  This project's history is a chain of proxies that looked
fine until they were checked against ground truth; this is the ground truth.

WHAT IS FITTED AND WHAT IS PREDICTED
------------------------------------
The paper gives inner/outer radius but NOT the straight-leg length, so one
geometric parameter is genuinely unknown.  With one free parameter and three
published numbers (field constant, self-inductance, tau) the table is
over-determined, so BOTH natural fits are run and reported:

    A. fit the straight length to the field constant -> predict L and tau
    B. fit the straight length to the self-inductance -> predict the field
       constant and tau

Reporting only whichever fit looks better would be cherry-picking, so both
are always printed.

RESULT (see the verdict block): the two fits disagree, because the paper's
own L, rho_c and tau are not mutually consistent under the standard NI
relation tau = L / sum(R_ct) -- by roughly a factor of 3.  That is a property
of the published table, not of this model, and it bounds how tightly this
benchmark can calibrate rho_c -> tau.  It is reported, not hidden.

The 20 K Shanghai Ic data in physics/ does not apply to a 77 K coil, so the
superconducting branch is suppressed here (perfect conductor).  That is not a
dodge: at the sudden-discharge operating point the contact branch dominates
the time constant by orders of magnitude, and suppressing V_sc isolates
exactly the inductance + contact-resistance chain this test is meant to
validate.

Run:  <env>/bin/python3 circuit/validation/he2025_racetrack.py
"""

import os
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

sys.stdout.reconfigure(line_buffering=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_CIRC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_CIRC)
for _p in (_CIRC, _ROOT, os.path.join(_ROOT, "physics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cparams as cfg                                  # noqa: E402
import inductance as ind                               # noqa: E402
import fieldmatrix as fm                               # noqa: E402
from geometry import CoilGeometry, TurnGroups, racetrack_loop   # noqa: E402

# ── published values ────────────────────────────────────────────────────────
PAPER = dict(t=0.2e-3, w=12e-3, n_turns=115,
             Ic_A=190.0, a_inner=49e-3, a_outer=73e-3,
             field_const_mT_per_A=0.81, L_self_mH=3.05,
             tau_s=13.53, rho_ct_uohm_cm2=399.0,
             I_charge=150.0, ramp_rate_A_per_s=10.0)

TURNS_PER_GROUP = 5


def make_geom(L_straight):
    """The paper's coil as a CoilGeometry, for a given straight-leg length."""
    return CoilGeometry(a_out=PAPER["a_outer"], n_turns=[PAPER["n_turns"]],
                        t=PAPER["t"], w=PAPER["w"], L=L_straight,
                        coil_half_gap=0.0, two_coil=False, layer_z=[0.0],
                        label="He2025 racetrack")


def field_constant(L_straight):
    """Centre field per amp of transport current [mT/A]."""
    geom = make_geom(L_straight)
    tg = TurnGroups(geom, TURNS_PER_GROUP)
    pt = np.array([[0.0, 0.0, 0.0]])
    Bz = 0.0
    for j in range(tg.N):
        p, d, _ = racetrack_loop(tg.r[j], tg.z[j], geom.L, 200, 300)
        Bz += tg.n[j] * fm._biot_savart(pt, p, d)[0, 2]
    return abs(Bz) * 1e3          # T/A -> mT/A


def self_inductance(L_straight, turns_per_group=TURNS_PER_GROUP):
    geom = make_geom(L_straight)
    tg = TurnGroups(geom, turns_per_group)
    M = ind.build_M(tg, geom, verbose=False)
    return ind.total_inductance(M, tg, two_coil=False), geom, tg, M


def discharge_tau(geom, tg, M, rho_ct_uohm_cm2, I0, t_end):
    """Sudden-discharge time constant with the SC branch suppressed."""
    rho = rho_ct_uohm_cm2 * cfg.UOHM_CM2_TO_OHM_M2
    R_ct = tg.contact_resistance(rho)
    A = M * tg.n[None, :]
    from scipy.linalg import lu_factor, lu_solve
    lu = lu_factor(A)

    def rhs(_t, i):
        return lu_solve(lu, R_ct * (0.0 - i))

    t_eval = np.concatenate([[0.0], np.geomspace(1e-4 * t_end, t_end, 500)])
    sol = solve_ivp(rhs, (0.0, t_end), np.full(tg.N, I0), t_eval=t_eval,
                    method="BDF", rtol=1e-8, atol=1e-8)
    if not sol.success:
        raise RuntimeError(sol.message)
    cur = tg.n @ sol.y / tg.n.sum()
    m = (cur < 0.6 * I0) & (cur > 0.02 * I0)
    p = np.polyfit(sol.t[m], np.log(cur[m]), 1)
    return -1.0 / p[0], sol.t, cur


def _series_R(geom):
    rho = PAPER["rho_ct_uohm_cm2"] * cfg.UOHM_CM2_TO_OHM_M2
    r_k = geom.a_out - (np.arange(PAPER["n_turns"] - 1) + 0.5) * geom.t
    return float(np.sum(rho / (geom.turn_length(r_k) * geom.w)))


def evaluate(L_straight, label):
    L_ind, geom, tg, M = self_inductance(L_straight)
    fc = field_constant(L_straight)
    tau, _, _ = discharge_tau(geom, tg, M, PAPER["rho_ct_uohm_cm2"],
                              PAPER["I_charge"], t_end=400.0)
    R = _series_R(geom)
    print(f"\n--- {label} ---")
    print(f"  straight half-length L   = {L_straight*1e3:7.2f} mm   "
          f"(turn length {geom.turn_length(0.061):.4f} m)")
    print(f"  field constant           = {fc:7.3f} mT/A   "
          f"published {PAPER['field_const_mT_per_A']:.2f}   "
          f"error {fc/PAPER['field_const_mT_per_A']-1:+7.1%}")
    print(f"  self-inductance          = {L_ind*1e3:7.3f} mH     "
          f"published {PAPER['L_self_mH']:.2f}   "
          f"error {L_ind*1e3/PAPER['L_self_mH']-1:+7.1%}")
    print(f"  discharge tau            = {tau:7.3f} s      "
          f"published {PAPER['tau_s']:.2f}  (MEASURED)  "
          f"error {tau/PAPER['tau_s']-1:+7.1%}")
    print(f"  series contact R         = {R*1e3:7.3f} mOhm  "
          f"-> naive L/R = {L_ind/R:.2f} s")
    return dict(L_straight=L_straight, fc=fc, L_ind=L_ind, tau=tau, R=R)


def main():
    print("=" * 76)
    print("Tier A2 — He et al., Energies 2025, 18, 3669, Table 2 "
          "(racetrack NI coil)")
    print("=" * 76)
    print("Published: L = 3.05 mH, field constant = 0.81 mT/A, "
          "tau = 13.53 s (measured)")
    print("           115 turns, 0.2 mm x 12 mm tape, 49/73 mm radii, "
          "rho_c = 399 uOhm.cm^2")
    print("The straight-leg length is NOT published; it is the single free "
          "parameter.\n")

    target_fc = PAPER["field_const_mT_per_A"]
    lo, hi = 1e-4, 0.30
    print(f"bracket: field constant spans {field_constant(hi):.3f} - "
          f"{field_constant(lo):.3f} mT/A over L = 300 - 0 mm")
    print(f"         self-inductance spans {self_inductance(lo)[0]*1e3:.3f} - "
          f"{self_inductance(hi)[0]*1e3:.3f} mH over the same range")

    A = evaluate(brentq(lambda L: field_constant(L) - target_fc, lo, hi,
                        xtol=1e-6),
                 "Fit A: straight length chosen to match the FIELD CONSTANT")
    B = evaluate(brentq(lambda L: self_inductance(L)[0] * 1e3
                        - PAPER["L_self_mH"], lo, hi, xtol=1e-6),
                 "Fit B: straight length chosen to match the SELF-INDUCTANCE")

    # grouping convergence (fit A geometry)
    print("\ngrouping convergence of L_self (fit A geometry):")
    for tpg in (10, 5, 2, 1):
        Lx, _, _, _ = self_inductance(A["L_straight"], tpg)
        print(f"    turns/group={tpg:3d}: {Lx*1e3:.3f} mH")

    print("\n" + "=" * 76)
    print("VERDICT")
    print("=" * 76)
    print(f"Fit A reproduces the MEASURED tau to "
          f"{abs(A['tau']/PAPER['tau_s']-1)*100:.1f}% but overshoots the "
          f"published L by {abs(A['L_ind']*1e3/PAPER['L_self_mH']-1)*100:.0f}%.")
    print(f"Fit B reproduces L exactly (by construction) but misses the "
          f"measured tau by "
          f"{abs(B['tau']/PAPER['tau_s']-1)*100:.0f}% and the field constant "
          f"by {abs(B['fc']/target_fc-1)*100:.0f}%.")
    print()
    print("Root cause — the published table is INTERNALLY INCONSISTENT under")
    print("the standard NI relation tau = L / sum(R_ct):")
    print(f"    L(published) / R(fit B) = "
          f"{PAPER['L_self_mH']*1e-3/B['R']:.2f} s, "
          f"against a measured tau of {PAPER['tau_s']:.2f} s "
          f"-- a factor of {PAPER['tau_s']/(PAPER['L_self_mH']*1e-3/B['R']):.1f}.")
    print("No choice of the one free geometric parameter can satisfy all")
    print("three published numbers at once.  Either their rho_c was")
    print("back-fitted through a different contact-area definition, or L and")
    print("tau were measured under conditions this lumped relation does not")
    print("describe.")
    print()
    ok = abs(A["tau"] / PAPER["tau_s"] - 1) < 0.30
    print(f"WHAT THIS DOES VALIDATE: the model structure and the tau "
          f"prediction\n  ({'PASS' if ok else 'FAIL'}: fit A predicts the "
          f"measured tau to "
          f"{abs(A['tau']/PAPER['tau_s']-1)*100:.1f}%).")
    print("WHAT IT DOES NOT: an absolute rho_c -> tau calibration.  Carry a")
    print("  factor ~2-3 uncertainty on any tau quoted from an ASSUMED rho_c.")
    print("=" * 76)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
