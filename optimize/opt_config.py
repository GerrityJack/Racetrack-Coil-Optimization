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
# 2026-07-21 design direction: operate at 50-60% of local Ic anywhere in the
# winding.  I_quench is (by construction of quench_current()'s bisection) the
# current at which the worst-margin cell reaches i = I/Ic = 1, so I_op =
# I_quench / SAFETY_FACTOR puts that worst cell at i = 1/SAFETY_FACTOR
# exactly; every other cell has more margin (smaller i).  SAFETY_FACTOR =
# 1.818 -> worst-cell i = 0.55, the midpoint of the 50-60% band.
SAFETY_FACTOR   = 1.818    # I_op = I_quench_min / SAFETY_FACTOR  (=> i=0.55)
I_MAX_SEARCH_A  = 1500.0   # upper bound for the quench-current root find

# ── Target area & uniformity (box centred on the bore midplane) ─────────────
# 2026-07-21: box updated to 30 x 6 mm.  ASSUMPTION (matches the prior box's
# convention of Y being the shorter side): TARGET_X_M carries the 30 mm side,
# TARGET_Y_M keeps the 6 mm side.  Flag if the axes should be swapped.
TARGET_X_M      = 0.030    # box full width in x [m]
TARGET_Y_M      = 0.006    # box full width in y [m]
TARGET_NX       = 61
TARGET_NY       = 25
UNIFORMITY_MAX_PCT = 1.0   # peak-to-peak |B| limit over the box [%]
B_TARGET_MIN_T  = 10.0     # required mean |Bz| over the target box at I_op

# ── Mechanical allowables (screen; see validation/mechanical_stress_check) ──
# 2026-07-21: only hoop stress (B x J x bending radius, curved end-cap
# sections only -- see the `cap` mask in stress_screen()) is an active
# constraint right now, tightened 500 -> 400 MPa.  Delamination is a
# material property (~30 MPa) rather than a design lever and is NOT being
# enforced for this optimization round -- set effectively unbounded so it
# never binds I_op.  Revisit before finalizing the design.
SIGMA_HOOP_MAX_PA  = 400e6    # tape lengthwise tension, end-cap curves only
SIGMA_DELAM_MAX_PA = 1e12     # disabled per 2026-07-21 project direction

# ── evaluate.py's FROZEN contract (external optimization team) ──────────────
# evaluate.py (the 2026-07-14 handoff entry point) originally read
# SAFETY_FACTOR/TARGET_X_M/TARGET_Y_M directly.  Those three were repurposed
# above for the 2026-07-21 internal CMA-ES direction (different safety-factor
# philosophy, different box), which would otherwise silently change
# evaluate.py's behaviour for the external team with no code change on their
# end.  evaluate.py now reads ONLY these EVALUATE_* constants instead, kept
# equal to its original documented defaults -- do not repurpose these.
EVALUATE_SAFETY_FACTOR = 1.15
EVALUATE_TARGET_X_M    = 0.015
EVALUATE_TARGET_Y_M    = 0.006

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

# ── CMA-ES search (cmaes_search.py) ──────────────────────────────────────────
# Continuous search over (a, b, n_turns) minimizing tape length subject to
# B_TARGET_MIN_T, UNIFORMITY_MAX_PCT, and SIGMA_HOOP_MAX_PA above.  The
# 7-pancake stack structure (n_layers) is kept fixed; only the per-layer
# turn counts vary.  n_turns are treated as continuous by CMA-ES and rounded
# to the nearest int only when evaluated.
# 2026-07-21 warm start: best design from the first (bounded) run rather
# than the original baseline -- 0.773 km @ 10.08 T, was still the best of
# the two runs so far (the unbounded rerun didn't converge in 410 evals;
# a/b never settled -- see cmaes_variables.png). Restarting from here with
# a bigger budget lets CMA-ES refine around a known-good point instead of
# re-discovering it.
CMAES_X0 = dict(a=0.030776, b=0.060448, coil_half_gap=0.018491,
               n_turns=[207, 511, 190, 76, 467, 462, 374])

# 2026-07-21: the first run pinned a and b at their box bounds (30/60 mm)
# for most of the search -- per project direction, a and b now have NO
# artificial search bound; (None, None) means unbounded in pycma.  The only
# limits left on them are physical: b > a + 5mm (straight section must
# exist) and inner radius > 3mm clearance (geometry_violation() in
# cmaes_search.py), both enforced as penalties, not box bounds.  Since more
# tape directly costs the objective, there's also a natural restoring force
# against a/b growing without bound.  CMAES_A_STD0/CMAES_B_STD0 set the
# initial CMA-ES step size directly (no bound range left to derive it from).
CMAES_A_BOUNDS = (None, None)       # cap radius -- unconstrained
CMAES_B_BOUNDS = (None, None)       # centre -> cap-centre -- unconstrained
CMAES_A_STD0 = 0.020                 # initial step size for a [m]
CMAES_B_STD0 = 0.020                 # initial step size for b [m]
CMAES_N_BOUNDS = (50, 900)          # per-layer turn count search range

# 2026-07-21: coil-to-coil axial gap made a free variable (was fixed at
# 0.030 m).  Constraint is a >= 3 mm FACE-TO-FACE clearance between the two
# coil packs, i.e. 2*(coil_half_gap - n_layers*w/2) >= MIN_COIL_GAP_M -- not
# the same as coil_half_gap itself.  With n_layers=7, w=4mm fixed
# (n_layers*w/2 = 14 mm), the exact minimum coil_half_gap is 15.5 mm; the
# lower bound below matches that with no slack (cmaes_search.py's
# geometry_violation() penalizes any encroachment).
MIN_COIL_GAP_M = 0.003               # required face-to-face clearance
CMAES_HALF_GAP_BOUNDS = (0.0155, 0.045)   # coil_half_gap search range [m]
CMAES_SIGMA0_FRAC = 0.3             # initial step size as a fraction of each
                                    # variable's (upper - lower) bound range
CMAES_POPSIZE   = None              # None => pycma default (4 + 3*ln(N))
CMAES_MAX_EVALS = 800                # doubled 2026-07-21: unbounded a/b
                                     # needs more budget to converge than a
                                     # boxed search (~5-10 s/eval)
CMAES_SEED      = 43                 # bumped from 42: new run, new warm
                                     # start -- avoids replaying run 2's
                                     # exact (unproductive) early samples
CMAES_PENALTY_KM = 200.0            # penalty scale [km-equivalent] per unit
                                    # (normalized) constraint violation^2 --
                                    # large relative to typical tape lengths
                                    # (O(1-3 km)) so any violation dominates
CMAES_INFEASIBLE_PENALTY_KM = 1000.0  # flat penalty for geometrically
                                      # infeasible / failed evaluations
CMAES_OUT_CSV = "optimize/cmaes_results.csv"     # this run's best design
CMAES_OUT_LOG = "optimize/cmaes_history.csv"     # this run's full history

# Cumulative, NEVER-overwritten log of every evaluation across every
# cmaes_search.py run to date (each row tagged with a run_tag) -- the
# "parameter-space map" dataset for identifying good/bad regions across
# runs.  See cmaes_param_map.png (built from this file, not just the
# current run) in visualization/.
CMAES_MASTER_LOG = "optimize/cmaes_all_evaluations.csv"
CMAES_PARAM_MAP_PNG = "visualization/cmaes_param_map.png"
