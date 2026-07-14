"""
opt_config.py — THE file to edit for configuration optimization.
================================================================
Defines the candidate geometries (a, b, n_turns) and the constraints /
targets used by optimize_geometry.py.  Nothing else needs touching.

A candidate is a dict: {"a": [m], "b": [m], "n_turns": [list, top→bottom]}.
Either list them explicitly in CANDIDATES, or leave CANDIDATES = None and
the grid product of A_VALUES × B_VALUES × N_TURNS_VARIANTS is used.
"""

# ── Candidate geometries ─────────────────────────────────────────────────────
# Explicit list takes precedence when not None:
CANDIDATES = None

# Grid mode (used when CANDIDATES is None):
A_VALUES = [0.045, 0.050, 0.055]          # cap radius [m]
B_VALUES = [0.080]                         # centre → cap centre [m]
N_TURNS_VARIANTS = [
    [500, 500, 500, 400, 400, 250, 100],   # current baseline (2650)
    [500, 500, 500, 500, 400, 250, 100],   # +100 mid          (2750)
    [550, 550, 500, 400, 400, 250, 100],   # +100 top          (2850)
    [500, 500, 500, 400, 400, 400, 250],   # fill bottom       (2950)
]

# ── Operating-point rule ─────────────────────────────────────────────────────
SAFETY_FACTOR   = 1.15     # I_op = I_quench_min / SAFETY_FACTOR
I_MAX_SEARCH_A  = 1500.0   # upper bound for the quench-current root find

# ── Target area & uniformity (box centred on the bore midplane) ─────────────
TARGET_X_M      = 0.015    # box full width in x [m]
TARGET_Y_M      = 0.006    # box full width in y [m]
TARGET_NX       = 61
TARGET_NY       = 25
UNIFORMITY_MAX_PCT = 1.0   # peak-to-peak |B| limit over the box [%]

# ── Mechanical allowables (screen; see validation/mechanical_stress_check) ──
SIGMA_HOOP_MAX_PA  = 500e6   # tape lengthwise tension
SIGMA_DELAM_MAX_PA =  30e6   # transverse tension (delamination — weak axis)

# Screening-current stress policy — THE biggest design lever in the screen.
# The Bean bands locally carry Jc = (1/i)·Je; whether the delamination
# interface sees the full local amplification or the width-averaged load
# depends on load transfer across the stiff 4 mm tape — a mechanical
# question this screen cannot settle.  Options:
#   "local"    — full 1/i amplification (conservative bound).  With the
#                current tape this limits I_op to ~130 A → ~5.4 T target,
#                but entirely within the validated 0–8 T Ic data.
#   "averaged" — width-averaged load (uniform-J delamination, optimistic
#                bound).  I_op ~290 A → ~12.5 T target, but peak conductor
#                fields far beyond the 8 T data ceiling.
# Both delamination numbers are always reported.
SCREENING_STRESS_MODE = "local"
SCREENING_AMP_CAP     = 5.0   # cap on the 1/i amplification factor

# ── Ic data hygiene ──────────────────────────────────────────────────────────
MAX_CLIP_FRACTION  = 0.25  # flag candidates where more Ic evaluations than
                           # this were clipped to the CSV's 8 T ceiling at
                           # the quench point (quench value is a floor
                           # estimate there, not a prediction)

# ── Screening fidelity ───────────────────────────────────────────────────────
# The optimizer uses the fast uniform-J FEM screen: coarse z-mesh (the T-A
# tape-width refinement is irrelevant for uniform J), one solve per
# geometry (field is exactly linear in I).  SCIF (~1% with the current
# tape) and its uniformity impact must be checked on FINALISTS with the
# per-layer T-A solver.
SCREEN_MESH_OVERRIDES = dict(mesh_nz_per_layer=1, mesh_z_grading=None)
FILAMENT_TURNS_PER_GROUP = 100   # radial sub-filaments per layer for the
                                 # target-box Biot-Savart (uniformity)

# ── Output ───────────────────────────────────────────────────────────────────
OUT_CSV = "optimize/opt_results.csv"
OUT_PNG = "visualization/opt_overview.png"
