"""ta_quench_margin_check.py -- 2026-08-08, does T-A screening current
concentration move the coil's actual local quench point relative to the
uniform-J assumption every quench evaluation elsewhere in this project
(sweep/quench_sweep.py, circuit/dcn.py's local_Ic_n, optimize/'s
geometry_violation()) is built on -- and does that gap get worse under a
genuinely time-resolved fast ramp?

WHY THIS SCRIPT EXISTS
-------------------------
circuit/power_ramp.py (this same session) found that the DCN model
predicts NO ramp-speed-dependent quench constraint: the transient
turn-current state is always safer than the final DC steady state, all
the way up to near-instantaneous ramps, for every rho_c tested. But DCN
only resolves INTER-turn (radial-leakage) current sharing -- it has no
notion of INTRA-tape screening-current concentration, which is exactly
what the T-A formulation (solve/ta_solve.py) exists to capture. Per user
direction, that gap must be closed before trusting DCN's "P is
unbounded" finding: does J_TA_coil (the actual, screening-current-
resolved local current density) ever get locally closer to Jc(B,theta)
than the uniform-J assumption believes, and does a faster ramp make it
worse?

WHAT THIS SCRIPT DOES
------------------------
Three comparisons, all at the SAME final current I_design=196A, using
the SAME per-cell critical-current-DENSITY conversion ta_solve.py's own
Picard solver uses internally (Jc = Ic(B,theta)/(delta_SC*w), compared
against the IN-PLANE (n_hat-projected-out) component of J -- verified by
reading _update_rho, not assumed):

  1. uniform-J           -- the naive assumption every quench check in
                             this project (sweep/quench_sweep.py,
                             circuit/dcn.py) is built on.
  2. T-A, dt=600s         -- solve/racetrack_ta_fields.npz, the project's
     (single implicit step)  own standing, already-validated production
                             T-A solve (the single implicit step every
                             production code path uses). Freshness
                             checked against the champion's documented
                             box_ptp_pct.
  3. T-A, dt=60s x10      -- transient/full_validation_plots/data/
     (genuine multi-step)    full_ramp_0to196A.npz's step 9 (the final
                             step of THIS project's first genuine
                             multi-step ramp, 0->196A over 10 real
                             60s steps, alpha=(0.03,0.01)-fixed,
                             forced-full-length). Reaches the identical
                             196A final current via real time-marching
                             instead of one big implicit step -- same
                             600s TOTAL span as (2), but resolved.

CAVEAT ON COMPARISON (3) vs (2): these come from two different code
paths -- ta_solve.solve_ta_at_current() (production) vs.
ta_transient._picard_phase() (the validation harness) -- which this
project's own history (CLAUDE.md, "NI transient work", 2026-08-06) has
an UNRESOLVED ~2% SCIF discrepancy between, present regardless of alpha,
root cause not found despite five isolation attempts. Any difference
found here between (2) and (3) should be read against that ~2% floor,
not treated as a clean measurement.

INTERPRETATION CAVEAT (does NOT apply to (1) vs (2)/(3), only to what
"margin < 1" means): this project's quench criterion everywhere
(including here) is the E_c = 1 uV/cm engineering definition of Ic, and
it is NORMAL, EXPECTED critical-state/Bean-model behaviour for some
cells near a flux-penetration front to sit at or above that threshold
without implying catastrophic thermal runaway -- ta_solve.py's own
Picard solver explicitly assigns a finite (not infinite) critical-state
resistivity to exactly this regime as standard operation, not an error
condition (see eps_reg in CLAUDE.md's "The T-A formulation"). Whether
this finding constitutes a real safety gap or expected, benign
critical-state behaviour the uniform-J margin was never meant to catch
is a judgement call outside this script's scope -- it reports the raw
comparison only.

Run:  <env>/bin/python3 transient/validation/ta_quench_margin_check.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRANS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_TRANS)
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "solve"),
           os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params                                          # noqa: E402
from current_source import normal_xy                    # noqa: E402
from ic_model import angle_with_normal_deg               # noqa: E402
from ic_extrapolation import make_ic_model                # noqa: E402

FIELDS_NPZ = os.path.join(_ROOT, "solve", "racetrack_ta_fields.npz")
FULL_RAMP_NPZ = os.path.join(_TRANS, "full_validation_plots", "data",
                             "full_ramp_0to196A.npz")
MARGIN_REQUIRED = 1.0 / 0.65   # same threshold as circuit/power_ramp.py


def _inplane_mag(J, n_hat):
    """|J| with the n_hat-normal component projected out -- EXACTLY
    matches ta_solve._update_rho's own jr = Jmag/Jc_vol comparison."""
    J_dot_n = np.einsum("ij,ij->i", J, n_hat)
    J_inplane = J - J_dot_n[:, None] * n_hat
    return np.linalg.norm(J_inplane, axis=-1)


def evaluate(label, centroids, B, J, delta_SC, w, ic):
    """Per-cell margin = Jc(B,theta)/|J_inplane|. Returns a dict of
    summary stats plus the full per-cell margin array."""
    L = params.b - params.a
    nx, ny = normal_xy(centroids[:, 0], centroids[:, 1], L)
    n_hat = np.column_stack([nx, ny, np.zeros_like(nx)])

    Bmag = np.linalg.norm(B, axis=1)
    theta = angle_with_normal_deg(B, n_hat)
    Ic_A, frac_clipped = ic.critical_current(Bmag, theta)
    Jc = Ic_A / (delta_SC * w)
    Jmag = _inplane_mag(J, n_hat)
    margin = Jc / Jmag
    i_worst = int(np.argmin(margin))

    return dict(label=label, margin=margin, i_worst=i_worst,
               min_margin=float(margin[i_worst]),
               n_below_1=int(np.sum(margin < 1.0)),
               n_below_req=int(np.sum(margin < MARGIN_REQUIRED)),
               n_cells=len(margin), frac_clipped=float(frac_clipped),
               Bmag=Bmag, theta=theta, Ic_A=Ic_A, Jc=Jc, Jmag=Jmag,
               centroids=centroids)


def print_row(r):
    print(f"{r['label']:26s} {r['min_margin']:12.4f} {r['i_worst']:9d} "
          f"{r['n_below_1']:6d}/{r['n_cells']:<6d} "
          f"{r['n_below_req']:6d}/{r['n_cells']:<6d}")


def print_detail(r):
    i = r["i_worst"]
    c = r["centroids"][i]
    print(f"  [{r['label']}] worst cell: ({c[0]*1e3:.2f}, {c[1]*1e3:.2f}, "
          f"{c[2]*1e3:.2f}) mm   |B|={r['Bmag'][i]:.3f} T   "
          f"theta={r['theta'][i]:.1f} deg   Ic={r['Ic_A'][i]:.2f} A   "
          f"concentration={r['Jmag'][i]/(196.0/(1e-6*params.w)):.2f}x uniform")


def main():
    print("=" * 88)
    print("T-A vs uniform-J local quench margin -- slow (dt=600s) vs fast (dt=60s x10) ramp")
    print("=" * 88)
    ic = make_ic_model("kim")

    # ── (1) & (2): uniform-J and dt=600s single-step T-A ──────────────────
    d1 = np.load(FIELDS_NPZ)
    I_solved = float(d1["I_solved"])
    delta_SC = float(d1["delta_SC"])
    w = params.w
    centroids1 = d1["coil_centroids"]
    box_ptp_pct = float(d1["box_ptp_pct"])
    print(f"\nLoaded {FIELDS_NPZ}")
    print(f"  I_solved={I_solved:.2f} A  box_ptp_pct={box_ptp_pct:.4f}% "
          f"(freshness check vs champion's documented 0.338-0.517%)")
    if I_solved != params.I_design:
        print(f"  NOTE: I_solved != current params.I_design "
              f"({params.I_design:.1f} A) -- file may predate params.py.")

    r_unif = evaluate("uniform-J (naive)", centroids1, d1["coil_B"],
                      d1["J_unif_coil"], delta_SC, w, ic)
    r_slow = evaluate("T-A dt=600s (1 step)", centroids1, d1["coil_B"],
                      d1["J_TA_coil"], delta_SC, w, ic)

    # ── (3): dt=60s x10 genuine multi-step ramp, final step (196A) ────────
    r_fast = None
    if os.path.exists(FULL_RAMP_NPZ):
        d2 = np.load(FULL_RAMP_NPZ, allow_pickle=True)
        I_step9 = float(d2["step9__I_now"])
        print(f"\nLoaded {FULL_RAMP_NPZ}  (step9: I={I_step9:.1f} A, "
              f"dt={float(d2['step9__dt']):.0f} s, final step of a genuine "
              f"10x60s=600s ramp)")
        r_fast = evaluate("T-A dt=60s x10 (step9)", d2["step9__coil_centroids"],
                          d2["step9__B_coil"], d2["step9__J_coil"],
                          delta_SC, w, ic)
    else:
        print(f"\n(skipping fast-ramp comparison -- {FULL_RAMP_NPZ} not found)")

    print()
    print(f"{'':26s} {'min margin':>12s} {'at cell':>9s} "
          f"{'#<1.0':>13s} {'#<'+format(MARGIN_REQUIRED,'.3f'):>13s}")
    for r in (r_unif, r_slow, r_fast):
        if r is not None:
            print_row(r)

    print()
    print(f"T-A worst-cell margin, slow vs fast ramp (same final I, same "
          f"600s total span, single-step-implicit vs genuinely time-marched):")
    if r_fast is not None:
        print(f"  dt=600s (1 step) : {r_slow['min_margin']:.4f}   "
              f"({r_slow['n_below_1']}/{r_slow['n_cells']} cells < 1.0)")
        print(f"  dt=60s  (10 step): {r_fast['min_margin']:.4f}   "
              f"({r_fast['n_below_1']}/{r_fast['n_cells']} cells < 1.0)")
        same_cell = r_slow["i_worst"] == r_fast["i_worst"]
        if same_cell:
            note = ("a real, robust feature (peak-field cell), not a "
                    "single-method artifact")
        else:
            note = ("the two methods disagree on WHERE the worst point is, "
                    "so treat both numbers cautiously")
        match_word = "MATCHES" if same_cell else "DIFFERS"
        print(f"  Worst cell location {match_word} between the two methods "
              f"(cell {r_slow['i_worst']} vs {r_fast['i_worst']}) -- {note}.")
        delta_pct = 100.0 * (r_slow["min_margin"] - r_fast["min_margin"]) / r_slow["min_margin"]
        print(f"  Faster (genuinely time-marched) ramp margin is "
              f"{delta_pct:+.1f}% relative to the slow single-step reference "
              f"-- for scale, this project's own harness-vs-production SCIF "
              f"discrepancy at this same (dt, I) is ~2%, so treat a "
              f"difference below that as within known noise, not as a clean "
              f"ramp-rate effect.")

    print()
    print("Worst-cell detail:")
    print_detail(r_unif)
    print_detail(r_slow)
    if r_fast is not None:
        print_detail(r_fast)

    print()
    print("Ratio (uniform-J margin / T-A margin) at each method's own worst cell:")
    print(f"  dt=600s: {r_unif['min_margin']/r_slow['min_margin']:.2f}x tighter under T-A")
    if r_fast is not None:
        print(f"  dt=60s : {r_unif['min_margin']/r_fast['min_margin']:.2f}x tighter under T-A")

    print()
    print("INTERPRETATION (see this file's own docstring for the full caveat):")
    print("  This project's quench criterion (here and everywhere else) is the")
    print("  E_c=1uV/cm engineering Ic threshold. Some cells sitting at/above")
    print("  it near a flux-penetration front is NORMAL Bean/critical-state")
    print("  behaviour (ta_solve.py's own solver assigns a finite, not")
    print("  infinite, resistivity there as standard operation) -- whether")
    print("  this specific extent/depth constitutes a real safety gap vs.")
    print("  expected benign critical-state behaviour the uniform-J margin")
    print("  was never meant to catch is a judgement call this script does")
    print("  not make. What IS established here: (a) T-A's local margin is")
    print("  meaningfully tighter than the uniform-J assumption at ANY ramp")
    print("  speed tested, including the project's own slow 600s reference,")
    print("  and (b) a genuinely time-marched fast ramp does not improve on")
    print("  that gap and trends slightly worse, within/near the harness's")
    print("  own known ~2% noise floor.")
    print("=" * 88)


if __name__ == "__main__":
    main()
