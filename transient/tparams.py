"""
tparams.py — configuration for the T-A no-insulation TRANSIENT model.

Imports params.py for the coil geometry but NEVER writes to it on disk (the
optimize/ta_validate.py convention, adopted after the 2026-07-27 params.py
corruption incident).  In-process mutation only, and only via
params.recompute_derived().
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# NOTE: circuit/ is deliberately NOT added here.  This repo has no packages
# and every module is imported by bare name off a flat sys.path, so
# circuit/run_charge.py and transient/run_charge.py collide -- whichever
# directory sits earlier on the path wins, silently.  (It did: a smoke test
# of the transient driver ran circuit's charge driver instead.)  Anything
# that genuinely needs both sides, i.e. validation/dcn_crosscheck.py, loads
# the circuit modules by explicit file path under distinct names.
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "mesh"),
           os.path.join(_ROOT, "solve"), os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RUNS_DIR = os.path.join(_HERE, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)


# ── radial binning of the winding ───────────────────────────────────────────
# The T-A mesh cannot resolve individual turns: mesh_size_min is ~0.25 x the
# pack thickness and there are only ~4400 coil cells over the whole eighth
# domain, so one cell already spans many turns.  The He et al. paper
# homogenises for exactly the same reason.  Bins are per LAYER.
N_RADIAL_BINS = 8


# ── contact resistivity ─────────────────────────────────────────────────────
# Same units and meaning as circuit/cparams.py; kept separate so the two
# models can be pointed at different values deliberately.
RHO_CT_UOHM_CM2 = 100.0
UOHM_CM2_TO_OHM_M2 = 1e-10
RHO_CT_INSULATED_UOHM_CM2 = 1e12       # the insulated limit (I_r -> 0)


# ── the E_p ambiguity ───────────────────────────────────────────────────────
# He et al. Eq. (12) writes E_p = rho_sc * J * d / L without saying whether J
# is the SC-layer current density or the homogenised engineering one -- a
# factor of Lambda/delta_SC = 75 straight onto the radial current.
#
# ta_solve's rho_fn is ALREADY rho_sc * (delta_SC/Lambda), and _J_from_T
# returns the SC-layer J, so the physically-derived electric field along the
# tape is
#       E = rho_sc * J_sc = rho_fn * (Lambda/delta_SC) * J_sc
# i.e. EP_FACTOR = Lambda/delta_SC.  That is the default because it is
# derived, not guessed.  EP_FACTOR = 1.0 reproduces the paper's literal
# reading.  validation/dcn_crosscheck.py is what decides between them: the
# DCN's ladder is the same relation with E_p from a lumped power law instead
# of the field solve, and it was validated independently in Phase A.
# RESOLVED EMPIRICALLY 2026-08-03 -- "paper" (F = 1) is the default.
# Measured on the champion at rho_c = 100 uOhm.cm^2, 600 s ramp, against the
# Phase A circuit model's validated I_r of ~3.4 A during the ramp:
#     F = Lambda/delta = 75 ("derived") -> I_r = 18-35 A (up to 37% of I),
#         29-38 of 48 bins clipped, closure unstable
#     F = 1             ("paper")       -> I_r = 3.9 / 2.1 / 2.1 A over the
#         ramp and 0.40 A on the flat top, zero clipping, stable
# CAVEAT on how much this proves: at F = 1, E_p drops ~75x and becomes small
# next to E_i, so the closure largely reduces to the inductive behaviour the
# DCN already captures.  The agreement therefore shows F = 1 is the
# self-consistent choice, NOT that the factor is independently confirmed.
EP_FACTOR_MODE = "paper"       # "paper" | "derived"


# ── coupling / relaxation ───────────────────────────────────────────────────
# The linear (inductive) part of the closure is now solved IMPLICITLY every
# Picard iteration (ni_circuit.py's _solve_I_z) -- only the nonlinear part
# (E_p, from the T-solve) is still Picard-lagged, so this relaxation now only
# damps E_p-driven oscillation, not the inductive coupling.
NI_RELAX = 0.10

# Picard iterations during which the circuit is FROZEN at I_z = I (the
# insulated state), letting T/A/rho settle before the closure is switched on.
#
# 2026-08-04: these are now CEILINGS, not the actual determinant of when the
# closure switches on -- ta_transient._picard_phase() runs the warmup phase
# to genuine convergence using ta_solve's own two-phase relaxation and
# observable-stall criterion, exactly as a normal (non-NI) solve would, and
# only switches to the closure once THAT reports converged (or hits the
# ceiling below). The original fixed counts (30 / 8) were a guess make
# BEFORE that convergence check existed, using only the FINE relaxation
# (alpha=0.15) throughout with no fast ramp-up phase -- nowhere near enough
# for a cold start to settle (the base solver typically needs k~25-80), so
# the closure was switching on into a state still in its own early, wildly
# swinging transient. That was the actual root cause of the persistent
# oscillation found in validation/ni_closure_stability_check.py, not a defect
# in the closure's own linear algebra. Set generously; the observable-stall
# criterion decides the real cutoff.
NI_WARMUP_ITERS = 80
NI_WARMUP_ITERS_FIRST = 150     # the first step starts from ZFC

# Physical band on the imposed per-bin current, as a fraction of I beyond
# [0, I].  The ladder forbids |I_r| > |I|; excursions past this are clipped
# and COUNTED (see diagnostics n_clipped).
I_Z_BAND = 0.5

# Picard iterations per time step.  Warm continuation (T, rho and A carried
# over between steps) should make these far cheaper than the ~80 a cold solve
# needs.
N_PICARD_STEP = 60
N_PICARD_FIRST = 150           # the first step starts cold


# ── time schedule ───────────────────────────────────────────────────────────
RAMP_S = 600.0
HOLD_S = 200.0
N_STEPS_RAMP = 24
N_STEPS_HOLD = 12


# ── convergence ─────────────────────────────────────────────────────────────
# Reuses ta_solve's observable-stall idea (params.ta_scif_stall_mT) but on a
# per-step basis with a much lower iteration floor, since a warm-started step
# begins close to its answer.
STEP_STALL_MT = 0.5
STEP_MIN_ITERS = 6
