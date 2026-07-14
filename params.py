"""
params.py — single source of truth for all simulation parameters.
All other scripts import this module; nothing is hardcoded elsewhere.
"""
import os, math

# ── Tape / winding geometry ──────────────────────────────────────────────────
a  = 0.050     # m — short radius (end-cap centreline)
b  = 0.080     # m — long radius  (centre → cap centre)
t  = 75e-6     # m — single tape thickness (radial pitch, no gap)
w  = 0.004     # m — tape width (out-of-plane depth per layer)

delta_SC = 1.0e-6   # m — REBCO superconducting layer thickness (~1 µm).
                     # Sets the T Dirichlet BC amplitude: T = ±I/(2·delta_SC).

# ── Layer definition  [top → bottom] ────────────────────────────────────────
# Each entry is the number of turns in that z-layer.
# All layers share the SAME outer radial edge; inner edge = a_out − n_i·t.
n_turns = [500, 500, 500, 400, 400, 250, 100]

# ── Derived winding quantities ────────────────────────────────────────────────
# Computed by recompute_derived() at the BOTTOM of this file.  To change
# geometry programmatically (the optimizer does this), mutate a / b /
# n_turns / t / w and call params.recompute_derived() — every derived
# quantity below (and the mesh sizing) is refreshed.

# ── Current ───────────────────────────────────────────────────────────────────
I_design = 200.0    # A/turn — default operating / visualisation current
# J_mag (homogenised current density) is set by recompute_derived()

# ── Two-coil configuration ────────────────────────────────────────────────────
two_coil_mode  = True
coil_half_gap  = 0.03    # m — half centre-to-centre separation
                           # (coil 1 at z=0, coil 2 at z=2g, midplane at z=g)

# ── Eighth-symmetry mode ──────────────────────────────────────────────────────
# FEM restricted to (x≥0, y≥0, z≤coil_half_gap); 8× finer mesh same cost.
# PMC at z=coil_half_gap handles coil-2 automatically (image principle).
use_eighth_symmetry = True

# ── Ramp parameters ──────────────────────────────────────────────────────────
ramp_duration = 600.0   # s — ramp time from 0 → I_design (default: 10 min).
                          # Controls screening current magnitude: slower ramp
                          # → deeper penetration → less SCIF at end of ramp.
                          # Scan this to bracket the worst-case SCIF.

ta_picard_tol = 1e-4  # legacy raw-|ΔB| tolerance — DIAGNOSTIC ONLY since
                         # 2026-07-11.  On sharp-front cases the front wanders
                         # chaotically among near-degenerate states (raw
                         # |ΔB|/|B| floors at ~6-10e-4, red spectrum) while
                         # every observable is frozen, so convergence is now
                         # judged on the observable: see ta_scif_stall_mT.
ta_scif_stall_mT = 0.05  # CONVERGENCE CRITERION: the EMA-smoothed bore SCIF
                         # [mT] must move less than this over 10 consecutive
                         # Picard iterations (earliest at k=25).  Measured:
                         # converged runs are stable to <0.01 mT.
ta_eps_reg  = 1.0   # j/jc floor: sub-critical cells get ρ = E_c/Jc (critical-state ρ)
                         # uses max(|J|/Jc, eps_reg) — NOT additive — so the power law
                         # is exact for j ≥ eps_reg×Jc and floored at eps_reg = 1.0 means
                         # the floor equals the critical-state resistivity E_c/Jc.
ta_n_picard          = 150   # maximum Picard iterations (phase 1 + phase 2);
                             # tol 1e-4 needs ~50-80 cold, fewer warm-started
ta_per_layer         = True  # one T problem per z-layer (no replication).
                             # At mesh_nz_per_layer ≥ 3 and tol 1e-4 the two
                             # modes agree within ~10% on the bore SCIF, but
                             # the replicated mode plateaus at |ΔB|/|B| ≈ 2-3e-4
                             # (never reaches 1e-4) while per-layer converges
                             # cleanly in ~70 iterations — so per-layer is the
                             # production default.  False = legacy replicated
                             # central-tape mode (~2× faster per iteration).
ta_picard_alpha      = 0.30  # phase-1 relaxation (fast ramp-up)
ta_picard_alpha_fine = 0.15  # phase-2 relaxation, held FIXED (with the smooth
                             # floor + rho relaxation this converges cleanly;
                             # aggressive throttling freezes slow flux-front
                             # transients — see ta_solve.py control comment)
ta_floor_smooth_p    = 16.0  # soft-max exponent for the j/jc floor:
                             # (eps^p + (j/jc)^p)^(1/p).  The hard max() has
                             # a kink at j = jc that makes front cells flip
                             # states every iteration (persistent 7e-4
                             # oscillation); the smooth floor restores Picard
                             # contraction.  0/None = legacy hard max.
ta_rho_relax         = 0.5   # log-space under-relaxation of ρ(J,B) updates —
                             # damps front cells flickering across the
                             # eps_reg floor (chatter amplitude is independent
                             # of the T-relaxation α, so α alone cannot fix it)
                       # to avoid ρ_HTS → 0 when J → 0. Typical range 1e-5–1e-3.

# ── Ic(B,θ) & n(B,θ) datasets ─────────────────────────────────────────────────────────
# ONE dataset for the whole pipeline (quench analysis AND T-A solver) —
# the Shanghai Superconductor High Field / Low Temperature 20 K tape
# (2026-07-10).  Ic file is Format A (A/cm width, scaled by tape width w);
# the "(1)" file carries the n-value column.  Data covers 0–8 T: fields
# above 8 T are still extrapolated.
PHYSICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "physics")
csv_filename = os.path.join(
    PHYSICS_DIR,
    "Shanghai Superconductor High Field Low Temperature 2G HTS 20 K Angle Dependence.csv")
shanghai_csv_filename = csv_filename   # T-A solver uses the same tape as quench

n_value_csv_filename = os.path.join(
    PHYSICS_DIR,
    "Shanghai Superconductor High Field Low Temperature 2G HTS 20 K Angle Dependence (1).csv")

# ── Current sweep ─────────────────────────────────────────────────────────────
sweep_I_min  = 200     # A/turn — lower bound of quench search
sweep_I_max  = 600     # A/turn — upper bound (raise if too many never-quenched)
sweep_I_step =  25     # A/turn — step size in the quench region
sweep_currents = (
    [50, 100]          # low-current anchors for accurate linear interpolation
    + list(range(sweep_I_min, sweep_I_max + sweep_I_step, sweep_I_step))
)
# Fine interpolation steps between each pair of solved currents when
# searching for the exact quench crossing current.
quench_search_steps_per_interval = 20

# ── Solver ────────────────────────────────────────────────────────────────────
use_iterative_solver = False   # True → HYPRE AMS (O(N) memory, slower per solve)
gauge_regularization = 1e-3    # penalises the gauge null-space; raise if outlier_ratio > 20
nedelec_degree       = 1

# ── Coil z-resolution ─────────────────────────────────────────────────────────
# Number of mesh sub-slabs across each tape width w (per z-layer).  The
# screening-current penetration profile lives across the tape width, so this
# is the resolution that matters for the T-A solve.  Each layer is built as
# mesh_nz_per_layer stacked extrusions of w/nz, forcing conforming node
# planes every w/nz.  1 = legacy behaviour (~2 cell rows per layer).
mesh_nz_per_layer = 3

# Graded slab thicknesses (fractions of w, must sum to 1, bottom → top of
# each tape).  Overrides mesh_nz_per_layer when set; None = uniform slabs.
# Screening penetrates from the tape edges (flux-free core half-width =
# a·sqrt(1−i²), i = I/Ic — Brandt/Norris strip theory): for i ~ 0.2-0.75
# the reversal/penetration zone is ~0.1-1 mm from each edge, so put small
# cells there and a coarse bulk.  The default (0.3 mm edge cells) matches
# brute-force uniform nz=5 within 4% on the bore SCIF at half the DOFs and
# 1/3 the runtime, and resolves the edge current-reversal zone that
# uniform nz=3 truncates (validation/ta_z_grading_study.py).  Note: on
# this and finer meshes the Picard plateaus just above 1e-4, so solves may
# report non-convergence at k=150 while the SCIF is already settled.
mesh_z_grading = [0.075, 0.15, 0.55, 0.15, 0.075]

# ── Mesh tier  (uncomment ONE block) ─────────────────────────────────────────
#   Tier     DOFs (1/8 domain)   RAM    Time/solve
#   smoke    ~9k                 <1GB   ~2s
#   medium   ~48k                ~3GB   ~5s
#   fine     ~70k                ~5GB   ~10s
#
# current: medium — factors used by recompute_derived():
mesh_size_min_factor = 0.25   # × pack_thickness
mesh_size_max_factor = 0.60   # × b
mesh_dist_min_factor = 1.50   # × pack_thickness
mesh_dist_max_factor = 2.00   # × b
box_scale     = 4.0

# smoke:
# mesh_size_min = 0.5*pack_thickness; mesh_size_max = 0.8*b
# mesh_dist_min = 2.0*pack_thickness; mesh_dist_max = 1.5*b; box_scale = 3.0

# fine:
# mesh_size_min = 0.10*pack_thickness; mesh_size_max = b/3
# mesh_dist_min = 1.00*pack_thickness; mesh_dist_max = 3.0*b; box_scale = 6.0

# ── Physical group markers (gmsh tags) ───────────────────────────────────────
coil_marker           = 1
air_marker            = 2
outer_boundary_marker = 3   # PEC n×A=0: outer box + x=0 + y=0 symmetry planes
pmc_boundary_marker   = 4   # PMC n×H=0 (natural BC): z=coil_half_gap midplane

# ── File paths ────────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.abspath(__file__))
MESH_DIR    = os.path.join(_HERE, "mesh")
PHYSICS_DIR = os.path.join(_HERE, "physics")
SOLVE_DIR   = os.path.join(_HERE, "solve")
SWEEP_DIR   = os.path.join(_HERE, "sweep")
FIELD_SWEEP_DIR = os.path.join(SWEEP_DIR, "field_sweep")
VIZ_DIR     = os.path.join(_HERE, "visualization")
VALIDATION_DIR = os.path.join(_HERE, "validation")

mesh_filename   = os.path.join(MESH_DIR,  "racetrack_mesh.msh")
output_filename = os.path.join(SOLVE_DIR, "racetrack_fields.xdmf")
fields_npz_filename = os.path.join(SOLVE_DIR, "racetrack_fields.npz")

for _d in (MESH_DIR, SOLVE_DIR, SWEEP_DIR, FIELD_SWEEP_DIR, VIZ_DIR, VALIDATION_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Visualisation grid parameters ────────────────────────────────────────────
topview_nx  = 200;  topview_ny  = 100
sideview_nx = 220;  sideview_nz = 90
quiver_stride = 8


# ── Derived-quantity computation ─────────────────────────────────────────────
def recompute_derived():
    """
    Recompute every quantity derived from the base geometry
    (a, b, t, w, n_turns, I_design) and refresh the mesh sizing.
    Called once at import; call again after mutating geometry
    programmatically (e.g. the configuration optimizer).
    Raises AssertionError for infeasible geometry.
    """
    global n_layers, n_turns_total, pack_thickness_list, pack_thickness
    global a_out, a_center_list, a_inner_list
    global _z_top, layer_z_bottoms, layer_z_tops
    global L, pack_x_mm, pack_y_mm, pack_z_mm, bore_inner_r_mm, tape_length_m
    global J_mag, mesh_size_min, mesh_size_max, mesh_dist_min, mesh_dist_max

    n_layers            = len(n_turns)
    n_turns_total       = sum(n_turns)
    pack_thickness_list = [n * t for n in n_turns]   # radial depth per layer
    pack_thickness      = max(pack_thickness_list)
    a_out               = a + pack_thickness / 2     # outer radial edge [m]
    a_center_list       = [a_out - n*t/2 for n in n_turns]
    a_inner_list        = [a_out - n*t   for n in n_turns]

    # Z positions: layer 0 = top (highest z); stack centred at z = 0
    _z_top          = n_layers * w / 2
    layer_z_bottoms = [_z_top - (i+1)*w for i in range(n_layers)]
    layer_z_tops    = [_z_top - i*w     for i in range(n_layers)]

    # Geometry summary quantities
    L               = b - a
    pack_x_mm       = 2 * (L + a_out) * 1e3
    pack_y_mm       = 2 * a_out * 1e3
    pack_z_mm       = n_layers * w * 1e3
    bore_inner_r_mm = min(a_inner_list) * 1e3
    tape_length_m   = sum(n*(4*L + 2*math.pi*ac)
                          for n, ac in zip(n_turns, a_center_list))

    J_mag = I_design / (t * w)     # homogenised current density [A/m²]

    mesh_size_min = mesh_size_min_factor * pack_thickness
    mesh_size_max = mesh_size_max_factor * b
    mesh_dist_min = mesh_dist_min_factor * pack_thickness
    mesh_dist_max = mesh_dist_max_factor * b

    # Sanity checks
    assert all(n >= 1 for n in n_turns), "all n_turns values must be >= 1"
    assert L > 0, "b must exceed a (straight length L = b - a > 0)"
    assert all(ai > 0 for ai in a_inner_list), \
        "inner radius ≤ 0 for some layer — reduce n_turns or increase a"


recompute_derived()
