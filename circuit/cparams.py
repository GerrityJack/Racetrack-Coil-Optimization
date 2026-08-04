"""
cparams.py — configuration for the DCN (distributed circuit network) model.

This is the ONLY file to edit for a new circuit run.  It imports params.py
for the coil geometry but NEVER writes to it (the ta_validate.py convention,
adopted after the 2026-07-27 params.py corruption incident).

Everything here is pure-numpy configuration: no dolfinx, no FEM.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "physics"), os.path.join(_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RUNS_DIR  = os.path.join(_HERE, "runs")
CACHE_DIR = os.path.join(_HERE, "runs", "cache")
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ── Turn grouping ───────────────────────────────────────────────────────────
# Turns are lumped into radial groups; every turn in a group is assumed to
# carry the same current.  This is the DCN's only spatial discretisation, so
# it is the convergence knob: halve it and the answer must not move.
#
# NOTE this is deliberately much finer than opt_config.FILAMENT_TURNS_PER_GROUP
# (=100), which exists for a completely different purpose (target-box field
# accuracy).  Cost is O(N_group^2) in the inductance build.
TURNS_PER_GROUP = 20


# ── Turn-to-turn contact resistivity ────────────────────────────────────────
# The NI free parameter.  params.t (75 um) is documented as a bare tape pitch
# with NO modelled insulation, so there is nothing in params.py from which
# this could be derived -- it must be assumed and swept.
#
# Literature range for bare REBCO co-wound contacts, and the two values the
# He et al. (2025) paper measured on its own coils:
#     pancake  coil:  90 uOhm.cm^2
#     racetrack coil: 399 uOhm.cm^2
RHO_CT_UOHM_CM2       = 100.0            # nominal
RHO_CT_SWEEP_UOHM_CM2 = [30., 100., 400.]

# rho_c -> Ohm.m^2.  1 uOhm.cm^2 = 1e-6 Ohm * 1e-4 m^2 = 1e-10 Ohm.m^2
UOHM_CM2_TO_OHM_M2 = 1e-10

# The insulated limit: used by the regression test that must reproduce the
# existing (insulated) model.  Not infinity -- a large finite value keeps the
# ODE well conditioned.
RHO_CT_INSULATED_UOHM_CM2 = 1e12


# ── Ic / n-value model ──────────────────────────────────────────────────────
# 'flat' is the historical default and is measurably +26.7% OPTIMISTIC above
# 8 T (2026-08-03 hold-out validation).  'kim' scored best (4.1% MAPE) and is
# itself mildly conservative.  See optimize/ic_extrapolation.py.
IC_EXTRAPOLATION = "kim"

# Samples around one turn at which Ic(B, theta) is evaluated.  The turn
# resistance is the sum over segments, so this captures the peak-field point
# that actually drives the resistive transition.
N_TURN_SAMPLES = 16


# ── Biot-Savart discretisation ──────────────────────────────────────────────
# Segments per racetrack loop for the Neumann inductance integral and the
# field matrix.  n_cap dominates accuracy (the caps are where the geometry
# actually curves).
N_STRAIGHT = 40
N_CAP      = 60

# Self / near-neighbour regularisation.  A filament pair at zero separation
# gives a divergent Neumann integral; the physical cure is the geometric mean
# distance (GMD) of the conductor cross-section.  For a rectangle of sides
# (t, w), GMD ~ 0.2235 * (t + w).  Applied to EVERY pair as
# |r_a - r_b| -> sqrt(d^2 + gmd^2), which also correctly makes adjacent turns
# (75 um apart, far closer than the ~0.9 mm GMD) nearly as strongly coupled
# as a turn is to itself -- which is the physical truth for stacked tape.
GMD_FACTOR = 0.2235


# ── Current schedules ───────────────────────────────────────────────────────
# (t_ramp is the linear ramp time to I_op; t_hold is the flat top after it.)
CHARGE_RAMP_S = 600.0
CHARGE_HOLD_S = 600.0

# Sudden discharge: the supply opens at t=0 from a fully charged steady state.
# In an NI coil the stored energy then circulates through the contacts.
DISCHARGE_HOLD_S = 300.0

# Ramp durations to sweep for the field-lag study.
RAMP_SWEEP_S = [60., 200., 600., 1800.]


# ── ODE solver ──────────────────────────────────────────────────────────────
# The n-value power law (n ~ 20-30 near I = Ic) is genuinely stiff.  This is
# the repo's first use of scipy.integrate -- there is no precedent to copy.
ODE_METHOD = "BDF"
ODE_RTOL   = 1e-6
ODE_ATOL   = 1e-6          # amps
ODE_MAX_STEP_FRAC = 0.02   # max step as a fraction of the total span


# ── Figure style (matches visualization/: dark theme, dpi 150) ──────────────
FIG_BG   = "#111"
AXES_BG  = "#0d0d1a"
FIG_DPI  = 150
SERIES_COLORS = ["#00acc1", "#fdd835", "#8e24aa", "#43a047",
                 "#e53935", "#fb8c00", "#3949ab", "#00897b"]
