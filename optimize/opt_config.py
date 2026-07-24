"""
opt_config.py — THE file to edit for configuration optimization.
================================================================
Defines the candidate geometries (a, b, n_turns) and the constraints /
targets used by optimize_geometry.py.  Nothing else needs touching.

A candidate is a dict: {"a": [m], "b": [m], "n_turns": [list, top→bottom]}.
Either list them explicitly in CANDIDATES, or leave CANDIDATES = None and
the grid product of A_VALUES × B_VALUES × N_TURNS_VARIANTS is used.
"""
import os as _os

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
# 2026-07-23: *** THE ENTIRE CHAMPION LEDGER BELOW PREDATES THREE NEW
# PHYSICAL CONSTRAINTS AND IS NO LONGER VALID/BUILDABLE AS-IS: ***
#   1. Minimum bend radius raised 3mm -> 7.5mm (REBCO tape cracks below
#      this; see geometry_violation()'s min_clear in cmaes_search.py) --
#      every design below sat right at the old 3mm floor, so all are now
#      significantly infeasible on this constraint alone.
#   2. Double-pancake construction: cmaes_search.py now optimizes ONE turn
#      count per PAIR of adjacent layers (2i, 2i+1 forced equal), not one
#      per layer -- required so each double pancake can be wound as a
#      single continuous piece (outer edges already match; inner ends are
#      joined). This makes odd N_LAYERS physically unbuildable (3,5,7,9
#      are eliminated outright) and none of the even-layer designs below
#      (6,8,10,12) satisfy pairing as recorded.
#   3. Turn-count floor removed (CMAES_N_BOUNDS/CMAES_TIGHT_N_BOUNDS lower
#      bound 50 -> 1, the true physical minimum) -- the old 50 floor had
#      no material basis and roughly half the layers in every converged
#      design sat pinned exactly on it (see CLAUDE.md's floor-test
#      diagnostic writeup).
# None of the numbers below should be used as a direct warm start anymore
# without re-deriving `a` for the new bend-radius floor and averaging each
# pair's turns first (see cmaes_search.py's encode_x0(), which does the
# pair-averaging automatically if you pass one of these lists directly --
# but the resulting `a` will start deeply infeasible on constraint 1).
# Kept for historical reference only.
#
# 2026-07-22: default CMAES_X0 restored to the CURRENT OVERALL BEST design
# (6-layer champion, 0.1464km @ 10.10T, unif=0.94%, hoop=59MPa) so a plain
# `python3 cmaes_search.py` with no env-var override does something useful
# by default -- polishing the current champion -- rather than reproducing
# the n_layers=5 round-2 config (which was a dead-end regression, see
# CLAUDE.md's "n_layers sweep + extended refinement" section).
#
# Full champion ledger (all from extended-refinement rounds, tight bounds,
# 2000-2500 evals each -- see CLAUDE.md for full history/why):
#   6 layers (BEST OVERALL, 0.1464km @ 10.10T, unif=0.94%, hoop=59MPa):
#     a=0.012908431166261252, b=0.020950054349354264,
#     coil_half_gap=0.013915040783036767, n_turns=[187,223,256,258,245,50]
#   7 layers (0.1884km @ 10.16T, unif=0.68%, hoop=78MPa):
#     a=0.015543508841320996, b=0.02278499488641108,
#     coil_half_gap=0.017004169493203968,
#     n_turns=[241,258,332,307,52,69,51]
#   8 layers (0.1971km @ 10.24T, unif=0.68%, hoop=74MPa):
#     a=0.014820826388410298, b=0.022118537106350228,
#     coil_half_gap=0.01752223820011793,
#     n_turns=[166,222,115,240,241,246,51,238]
#   5 layers, round 1 (0.3734km @ 10.18T, unif=0.90%, hoop=79MPa) -- USE
#     THIS ONE, not round 2:
#     a=0.016225348571759877, b=0.05499433483759851,
#     coil_half_gap=0.01154555376755883, n_turns=[294,349,351,73,351]
#   5 layers, round 2 (REGRESSION, do not use as a warm start -- 0.4173km,
#     worse than round 1): the warm-start step size (CMAES_A_STD0=6mm,
#     CMAES_B_STD0=12mm, both FIXED absolute values) was ~37%/22% of this
#     design's small a/b scale -- big enough to jump CMA-ES into a worse
#     basin (a=19.6mm/b=59.4mm) instead of refining round 1's point. This
#     is exactly the bug CMAES_A_STD0_OVERRIDE/CMAES_B_STD0_OVERRIDE below
#     were added to prevent (proportional step size instead of fixed mm).
#   3/4/9/10/12 layers: only cold-start sweep results exist (250-500
#     evals each, optimize/sweep_n_layers_log.txt) -- NOT yet given the
#     extended-refinement treatment. See CLAUDE.md's overnight-run plan.
#
# ── 2026-07-24: CURRENT CHAMPION, under the three 2026-07-23 constraints ──
# (bend radius 7.5mm, double-pancake pairing, no turn floor). Full ledger,
# one CMA-ES job per buildable (even) layer count, ~90-135 min each,
# n_turns given as the full per-layer list (pairs already equal):
#   10 layers (BEST OVERALL, 0.1938km @ 10.05T, unif=0.88%, hoop=71MPa):
#     a=0.015638780170509824, b=0.02064141470385894,
#     coil_half_gap=0.021924337907097105,
#     n_turns=[152,152,217,217,212,212,211,211,2,2]
#   8 layers (0.2049km @ 10.22T, unif=0.98%, hoop=85MPa):
#     a=0.01691878415679104, b=0.023162907260678892,
#     coil_half_gap=0.017526252266569582,
#     n_turns=[171,171,250,250,243,243,67,67]
#   12 layers (0.2380km @ 10.01T, unif=0.94%, hoop=111MPa):
#     a=0.020143788523549656, b=0.025226049597542666,
#     coil_half_gap=0.025505016270093442,
#     n_turns=[231,231,326,326,45,45,106,106,5,5,1,1]
#   6 layers (0.2258km @ 10.00T, unif=0.99%, hoop=114MPa):
#     a=0.02219635247570603, b=0.02726759479379641,
#     coil_half_gap=0.0135001375408541580,
#     n_turns=[285,285,379,379,2,2]
# All four show the same pattern: the last pair collapses toward the new
# turn floor (2, 67->via n=8's own tightest pair, 5&1, 2) -- a recurring
# signal that even n_layers=10's own true optimum may want fewer than 5
# effective pairs; see CLAUDE.md for the discussion. n_layers=12 was
# skipped by the orchestrator's own stale wall-clock total-budget check
# (a bug found this run, see CLAUDE.md) and had to be launched manually.
CMAES_X0 = dict(a=0.015638780170509824, b=0.02064141470385894,
               coil_half_gap=0.021924337907097105,
               n_turns=[152, 152, 217, 217, 212, 212, 211, 211, 2, 2])

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
CMAES_A_STD0 = 0.006                 # 2026-07-22: n_layers=5 ROUND 2 --
                                     # a/gap already flattened in round 1,
                                     # smaller step for local refinement
CMAES_B_STD0 = 0.012                 # b was STILL DECLINING in round 1
                                     # (55mm, trending down) -- kept larger
                                     # than a's step so it can keep moving
CMAES_N_BOUNDS = (1, 900)           # per-PAIR turn count search range.
                                    # 2026-07-23: floor dropped 50 -> 1 (the
                                    # true physical minimum, params.py only
                                    # asserts n>=1) -- the old 50 had no
                                    # material basis, and roughly half the
                                    # layers in every converged design sat
                                    # pinned exactly on it (see CLAUDE.md's
                                    # floor-test diagnostic).

# 2026-07-21: coil-to-coil axial gap made a free variable (was fixed at
# 0.030 m).  Constraint is a >= 3 mm FACE-TO-FACE clearance between the two
# coil packs, i.e. 2*(coil_half_gap - n_layers*w/2) >= MIN_COIL_GAP_M -- not
# the same as coil_half_gap itself.  With n_layers=7, w=4mm fixed
# (n_layers*w/2 = 14 mm), the exact minimum coil_half_gap is 15.5 mm; the
# lower bound below matches that with no slack (cmaes_search.py's
# geometry_violation() penalizes any encroachment).
MIN_COIL_GAP_M = 0.003               # required face-to-face clearance
CMAES_HALF_GAP_BOUNDS = (0.0155, 0.045)   # coil_half_gap search range [m]

# ── "zone-out" tight bounds (opt-in) ─────────────────────────────────────────
# 2026-07-22: derived empirically from the top 5% of all-pass designs across
# 30k+ evaluations spanning many independent runs and layer counts (see
# CLAUDE.md's margin-above-floor analysis): those top designs sit within a
# few mm of their PHYSICAL floors (inner-radius, straight-length, face-gap)
# and never use more than ~420 turns/layer, out of the (50,900) range
# CMA-ES was actually allowed to search. Rationale for zoning out the rest:
# every independent search so far (84+ runs, 4 hand-picked diverse restarts,
# a 76-restart random overnight sweep, and the n_layers sweep) has
# converged toward this same near-floor region regardless of starting
# point -- concentrating budget there via tighter bounds should let a fixed
# evaluation budget converge much faster instead of re-discovering the
# region from scratch every time. Opt-in (CMAES_TIGHT_BOUNDS=False by
# default) -- does not affect any existing default behavior.
CMAES_TIGHT_BOUNDS = True   # 2026-07-22: enabled for the n_layers=6/8
                            # extended-refinement follow-up (restore to
                            # False afterward unless kept on deliberately)
CMAES_TIGHT_N_BOUNDS = (1, 500)              # 2026-07-23: floor 50 -> 1 (see
                                             # CMAES_N_BOUNDS above); upper
                                             # still 500 (top 5% maxed at 423)
CMAES_TIGHT_GAP_MARGIN_M = 0.010             # gap in [floor, floor+10mm]; top
                                             # 20%'s max observed margin was 11.3mm
CMAES_SIGMA0_FRAC = 0.3             # initial step size as a fraction of each
                                    # variable's (upper - lower) bound range
CMAES_POPSIZE   = None              # None => pycma default (4 + 3*ln(N))
CMAES_MAX_EVALS = 2500                # generous default budget for polishing
                                      # the current champion (see CMAES_X0)
CMAES_SEED      = 606                 # fresh seed for the 6-layer polish
CMAES_PENALTY_KM = 200.0            # penalty scale [km-equivalent] per unit
                                    # (normalized) constraint violation^2 --
                                    # large relative to typical tape lengths
                                    # (O(1-3 km)) so any violation dominates
CMAES_INFEASIBLE_PENALTY_KM = 1000.0  # flat penalty for geometrically
                                      # infeasible / failed evaluations
CMAES_OUT_CSV = "optimize/cmaes_results.csv"     # this run's best design
CMAES_OUT_LOG = "optimize/cmaes_history.csv"     # this run's full history

# 2026-07-23: path overrides, REQUIRED whenever more than one
# cmaes_search.py process may be alive at once (e.g. two orchestrators
# launched in parallel). Found the hard way: two orchestrators both left
# CMAES_OUT_CSV/CMAES_OUT_LOG at these shared defaults while running jobs
# concurrently, and each job's periodic write silently clobbered the
# other's file -- one orchestrator's "best design for job X" summary line
# ended up reading back a DIFFERENT job's in-progress result (confirmed:
# a reported 12-layer turns list only had 9 entries after zip() truncated
# against a stray 9-layer read). CMAES_MASTER_LOG above was NOT affected
# (append-only, every row tagged by run_tag) -- only these two per-run
# snapshot files. Any orchestrator launching concurrent jobs MUST set
# unique values for both env vars per job (e.g. suffix by n_layers).
_out_csv_override = _os.environ.get("CMAES_OUT_CSV_OVERRIDE")
if _out_csv_override:
    CMAES_OUT_CSV = _out_csv_override
_out_log_override = _os.environ.get("CMAES_OUT_LOG_OVERRIDE")
if _out_log_override:
    CMAES_OUT_LOG = _out_log_override

# Cumulative, NEVER-overwritten log of every evaluation across every
# cmaes_search.py run to date (each row tagged with a run_tag) -- the
# "parameter-space map" dataset for identifying good/bad regions across
# runs.  See cmaes_param_map.png (built from this file, not just the
# current run) in visualization/.
CMAES_MASTER_LOG = "optimize/cmaes_all_evaluations.csv"
CMAES_PARAM_MAP_PNG = "visualization/cmaes_param_map.png"

# 2026-07-21: opt-in generation-parallel evaluation. Default 1 = the exact
# original serial path (es.optimize(fitness), no process pool, no env var
# changes) -- fully backward compatible. Set >1 to evaluate each CMA-ES
# generation's population concurrently across OS processes (see
# _run_parallel() in cmaes_search.py). Try 6 on this 8-core machine
# (leaves headroom for the main process + OS).
CMAES_N_WORKERS = 6

# ── sweep override (optional; used by sweep_restarts.py) ────────────────────
# If CMAES_SWEEP_OVERRIDE_JSON is set, override CMAES_X0/SEED/MAX_EVALS from
# it. This lets sweep_restarts.py hand each subprocess a different cold-start
# point via an env var instead of rewriting this file per-restart (fragile
# given how much hand-editing/commentary this block already has). Absent
# (the normal case), this does nothing -- default behavior for every other
# caller of this module is completely unaffected.
import json as _json
_override = _os.environ.get("CMAES_SWEEP_OVERRIDE_JSON")
if _override:
    _o = _json.loads(_override)
    if "x0" in _o:
        CMAES_X0 = _o["x0"]
    if "seed" in _o:
        CMAES_SEED = _o["seed"]
    if "max_evals" in _o:
        CMAES_MAX_EVALS = _o["max_evals"]

# 2026-07-22: standalone std0 overrides for orchestrators (e.g.
# overnight_refinement.py) that need a PROPORTIONAL step size (some
# fraction of the warm-start's own a/b value) rather than a fixed absolute
# CMAES_A_STD0/CMAES_B_STD0 in metres. This was the direct cause of the
# n_layers=5 round-2 regression (0.373km -> 0.417km, see CLAUDE.md): a
# fixed 6mm/12mm step size was ~37%/22% of that design's small ~16mm/55mm
# a/b scale -- big enough to jump CMA-ES out of the round-1 basin into a
# worse one, despite being intended as a "finer" refinement. Set as plain
# floats in metres (already computed as warm_start_value * fraction by the
# caller) via env vars so per-job values don't require editing this file.
# Absent (the normal case): CMAES_A_STD0/CMAES_B_STD0 above are used
# unchanged -- fully backward compatible.
_a_std0_override = _os.environ.get("CMAES_A_STD0_OVERRIDE")
if _a_std0_override:
    CMAES_A_STD0 = float(_a_std0_override)
_b_std0_override = _os.environ.get("CMAES_B_STD0_OVERRIDE")
if _b_std0_override:
    CMAES_B_STD0 = float(_b_std0_override)

# 2026-07-23: turn-count (per-pair) step size override, same rationale as
# the a/b overrides above -- added after discovering the un-overridden
# formula (CMAES_SIGMA0_FRAC * bound_range) gives n_std ~= 135-150, which
# is enormous next to typical per-pair turn counts (50-250). A warm-started
# "polish" run that only proportionally scaled a/b's step size still
# sampled turns almost uniformly across the WHOLE bound range from eval 1
# -- not a local refinement at all -- and after 3000+ evals had drifted to
# a worse optimum than its own starting point. CMAES_N_STD0 is None by
# default (falls back to the old bound-range-derived formula, unchanged
# behavior for cold starts, which legitimately want a wide initial step).
# Set via this override for any warm-started polish run.
CMAES_N_STD0 = None
_n_std0_override = _os.environ.get("CMAES_N_STD0_OVERRIDE")
if _n_std0_override:
    CMAES_N_STD0 = float(_n_std0_override)

# 2026-07-23: gap-margin override, for orchestrators that want to pin
# coil_half_gap near its physical floor instead of letting CMA-ES search it.
# Motivation: across 60,930 cumulative evaluations spanning every layer
# count tried (3,4,5,6,7,8,9,10,12), the best designs land within ~0.4mm of
# gap_bounds_for_n_layers()'s floor (2*n_layers*w/2 + MIN_COIL_GAP_M/2) in
# EVERY case -- it has stopped being a variable worth spending search budget
# on. This does not remove gap from the CMA-ES vector (would require
# touching the fixed [a,b,gap,n...] encoding and every downstream CSV/plot
# column that assumes it) -- it just shrinks CMAES_TIGHT_GAP_MARGIN_M (and
# with it, gap's initial step size, computed proportionally in
# bounds_and_stds()) so CMA-ES effectively treats it as fixed while staying
# fully backward compatible for any caller that doesn't set this env var.
_gap_margin_override = _os.environ.get("CMAES_TIGHT_GAP_MARGIN_M_OVERRIDE")
if _gap_margin_override:
    CMAES_TIGHT_GAP_MARGIN_M = float(_gap_margin_override)

# 2026-07-23: turn-floor override, for the floor-test diagnostic
# (optimize/floor_test.py). The CMAES_TIGHT_N_BOUNDS lower bound of 50 is
# NOT physical -- params.recompute_derived() only asserts n_turns[i] >= 1 --
# it was an empirical "zone-out" choice. But across 60k+ evaluations, many
# layers in near-optimal designs (roughly half, in every layer count >= 6)
# sit exactly AT that 50 floor, the same warning sign that flagged the old
# a/b box bounds as wrong. This override lets a diagnostic run drop the
# floor toward the true physical minimum (1) to see whether tape length
# keeps improving -- if so, 50 was an artificial wall, not a converged
# optimum. Format: "lo,hi". Absent: CMAES_TIGHT_N_BOUNDS above is unchanged.
_tight_n_override = _os.environ.get("CMAES_TIGHT_N_BOUNDS_OVERRIDE")
if _tight_n_override:
    _lo, _hi = _tight_n_override.split(",")
    CMAES_TIGHT_N_BOUNDS = (int(_lo), int(_hi))

# 2026-07-23: worker-count override, so a diagnostic run launched IN
# PARALLEL with a primary cmaes_search.py job (e.g. floor_test.py alongside
# focused_refinement_6_9.py) can use fewer workers and avoid oversubscribing
# this 8-core machine. Absent: CMAES_N_WORKERS above (6) is unchanged.
_n_workers_override = _os.environ.get("CMAES_N_WORKERS_OVERRIDE")
if _n_workers_override:
    CMAES_N_WORKERS = int(_n_workers_override)
