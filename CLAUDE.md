# Racetrack_v4 — Project Briefing for Claude Code

## What this project is

FEniCSx (dolfinx) FEM simulation pipeline for a **REBCO high-temperature
superconducting racetrack coil magnet**. Primary goal: characterise the
screening-current-induced field (SCIF) of the coil using the homogenised
**T-A formulation** (Vargas-Llanos et al. 2022, *Supercond. Sci. Technol.*
35, 124001).

**Design targets:** 10 T bore field, <1% field uniformity, maximum quench
safety factor.

**Runtime environment:** `fenicsx-env` conda on WSL (Ubuntu).
Always activate with `conda run -n fenicsx-env python3 ...` or work inside
the activated environment — **except for any long-running script**, where
you need the direct binary path instead; see "Operational lessons" below.

**Project history:** this file only covers current state and standing
reference material. The full chronological narrative — every search run,
bug hunt, rejected design, and retraction that produced the current design
and the current understanding of the solver's limits — is archived in
`docs/HISTORY.md`. When a section below states a conclusion without much
justification, that's deliberate: the reasoning is one `docs/HISTORY.md`
lookup away.

---

## Repository layout

```
Racetrack_v4/
├── params.py                    # All geometry, material, solver parameters
├── docs/
│   └── HISTORY.md                # Full chronological project narrative (archive)
├── mesh/
│   └── build_mesh.py            # Gmsh mesh builder (eighth-symmetry domain)
├── physics/
│   ├── current_source.py        # Arc-length, tangent/normal, turn/layer index helpers
│   ├── ic_model.py              # IcModel (Ic(B,θ) interpolation) + NValueModel
│   │                            #   (n(B,θ) interpolation) + compute_rho_hts()
│   └── digitized_IC_data/       # 2026-09-12 — hand-digitized Ic(B,θ) and
│                                #   n(B) points from OTHER tapes' published
│                                #   data, for the 4.2K temperature-scaling
│                                #   investigation (see "Cryogenic (4.2K)
│                                #   operating-temperature investigation"
│                                #   below) — {5,10,15,20,24}T_{4p2K,20K}.csv
│                                #   (Fujikura FESC-SCH04(40) Ic ratio, from
│                                #   the Tsuchiya et al. 2026 IEEE paper) and
│                                #   nvalue_Bc_{4p2K,...,77p3K}.csv (from the
│                                #   Fujikura manufacturer datasheet, both
│                                #   PDFs kept here too for provenance)
├── solve/
│   ├── solve.py                 # Uniform-J A-form FEM solve (the baseline)
│   ├── ta_solve.py              # T-A Picard solve  ← primary new file
│   ├── ta_sweep.py              # Current sweep using ta_solve
│   └── ta_postprocess.py        # Post-processing from saved .npz
├── sweep/
│   ├── solve_sweep.py           # Uniform-J field sweep
│   └── quench_sweep.py          # Quench current analysis
├── optimize/                    # design search — reorganized 2026-07-30
│   ├── opt_config.py            # the ONLY file to edit for a new search
│   ├── optimize_geometry.py     # coarse uniform-J screen (~5 s/design)
│   ├── cmaes_search.py          # the CMA-ES search itself
│   ├── evaluate.py              # frozen external-team entry point
│   ├── ta_validate.py           # full T-A box-uniformity ground truth
│   ├── ic_extrapolation.py      # KimIcModel / ScalingLawIcModel / BetaIcModel
│   ├── ic_temperature_scaling.py # Fujikura4p2KScaledIcModel/NValueModel
│   │                            #   (2026-09-12) — see "Cryogenic (4.2K)
│   │                            #   operating-temperature investigation" below
│   ├── ta_safe_current.py       # T-A-resolved safe I_op (2026-09-02) —
│   │                            #   see "Ramp-up power analysis" below
│   ├── studies/                 # one-off orchestrators (day_search.py,
│   │                            #   double_pancake_search.py,
│   │                            #   perturbation_study.py, margin_design_search.py,
│   │                            #   ta_safe_margin_search.py, …)
│   └── runs/                    # every log + CSV, grouped by study
│       ├── cmaes_all_evaluations.csv   # CUMULATIVE, append-only, ~100k+ rows
│       ├── ta_safe_margin/, ta_safe_margin_8layer/  # see "Ramp-up power
│       │                            #   analysis" below
│       └── perturbation/, day_search/, double_pancake/, …
├── circuit/                     # NI transient Phase A — lumped DCN circuit model
│   ├── geometry.py, inductance.py, fieldmatrix.py, dcn.py
│   ├── power_ramp.py            # constant-power ramp-up analysis (2026-08-08)
│   └── validation/              # A1 (self-check), A2 (He et al. 2025 benchmark)
├── transient/                   # NI transient Phase B — T-A + circuit closure (exploratory)
│   ├── ta_transient.py, ni_circuit.py, newton_ta.py, induction.py
│   └── validation/, studies/
└── visualization/
    ├── plot_fields.py            # field_top.png, field_side.png (dark theme)
    ├── field_uniformity.py       # uniformity.png (dark theme)
    ├── plot_3d.py                # geometry.png, field_3d.png, quench plots
    └── for poster/                # 2026-09-03, white-background poster figures
        ├── make_jjc_cross_sections.py    # jjc_{major,minor}_axis_cross_section.png
        ├── make_pca_3d_trajectory.py     # pca_3d_trajectory.png
        ├── make_penalty_evolution.py     # penalty_evolution.png
        ├── make_external_field_3d.py     # external_field_3d.png (2026-09-04)
        ├── make_field_comparison.py, make_geometry_definition.py,
        │   make_technical_diagram.py, make_design_progression.py,
        │   make_pca_convergence.py, make_optimal_configuration.py,
        │   make_results_table.py
        └── *_medium_backup.png/.npz      # pre-xdense-mesh reference copies,
                                           # see "Mesh resolution" below
```

**Key output files:**
- `solve/racetrack_ta_fields.npz` — T-A solve results (coil_B, J_TA_coil,
  J_unif_coil, dB_bore, Bz_bore_uniform, Bz_bore_TA, T_field, ...)
- `sweep/quench_results.csv` — per-cell quench currents
- `visualization/ta_*.png` — T-A figures

---

## Current design — the champion (as of 2026-08-03)

**n_layers=6 (double-pancake: 3 pairs), a=26.0mm, b=31.4mm,
coil_half_gap=13.7mm, n_turns=[382,382,478,478,3,3],
I_design=196.0 A (65% of local Ic under the Kim Ic(B) model), tape=0.3372 km.**

| metric | nominal | limit | across 15 jitter samples |
|---|---|---|---|
| B_target (Kim Ic model) | 10.49 T | ≥ 10 T | 10.10–10.49 T — 15/15 PASS |
| box peak-to-peak uniformity (T-A) | 0.495% | ≤ 1% | 0.338–0.517% — 15/15 PASS |
| hoop stress | 113 MPa | ≤ 400 MPa | 102–113 MPa |
| bend radius | 8.075 mm | ≥ 7.5 mm | 7.545–8.434 mm |
| face gap | 3.40 mm | ≥ 3.0 mm | 3.00–3.84 mm |

This is the first design in the project's history validated against BOTH
a realistic critical-current model **and** build tolerance (±0.2mm on
a/b/gap, ±2% on tape thickness, exact turn counts) — see
`optimize/studies/margin_design_search.py` and
`optimize/studies/jitter_margin_design.py`. Its predecessor
(`[329,329,411,411,2,2]`, 0.2596 km) hit 10.03 T with only 0.3% margin and
then failed catastrophically under the same jitter test (0/14 builds
reached 10 T) — the search that produced it minimized tape subject to
`B ≥ 10 T` and converged exactly onto that constraint with nothing asking
for margin. This design was built from margin-aware constraints derived
directly from that failure, not from tighter constants picked by eye.

**Caveats, in priority order:**
1. **Ic model uncertainty is real, ±0.5 T.** Under the more conservative
   `scaling:45` Ic extrapolation this design gives 9.44 T instead of
   10.49 T. Kim is the measurably better model (hold-out MAPE 4.1% vs.
   6.1%) and itself mildly conservative, so state the design as
   **~10.5 T ± 0.5 T of model uncertainty**, closable only by measured Ic
   data above 8 T. See "Ic(B) extrapolation above 8 T" below.
2. **No fast uniformity proxy has ever survived validation for this
   project** (four have been falsified — see "Proxy graveyard" below).
   `optimize/cmaes_search.py`'s fitness function carries **no uniformity
   signal at all**, deliberately. Any future search's finalists MUST be
   validated with `optimize/ta_validate.py` before being trusted.
3. **The NI (no-insulation) transient work in `circuit/`/`transient/` does
   NOT change this design.** At DC steady state the radial current
   vanishes, so every number in the table above is unaffected — it only
   adds a transient (ramp/discharge) constraint on top. See "NI transient
   work" below for what is and isn't validated there.
4. Assembly tolerance on tape thickness is the single dominant build
   error (asymmetric — `t+2%` is far worse than `t-2%`); gap/face-gap/
   bend-radius margins in the table above are the buffer against it.
   Tighter machining than the assumed ±0.2mm/±2% would recover tape.
5. **2026-08-08: the "quench safety factor" row above is uniform-J
   based, and a same-day T-A cross-check found that assumption to be
   materially optimistic locally.** `transient/validation/
   ta_quench_margin_check.py` compared T-A's actual (screening-current-
   resolved) local current density against Ic(B,θ) at the same I_design
   operating point and found the worst-cell margin is ~2.26-2.33x
   TIGHTER than the uniform-J number (0.71-0.69 vs. 1.60), with 27-34%
   of coil cells locally at or above the E_c-defined Ic — present even
   at the design's own slow (600s) reference ramp, not a fast-ramp
   artifact. Whether this is a real safety gap or expected, benign
   Bean/critical-state behaviour the uniform-J margin was never built to
   catch is an OPEN judgement call, deliberately not resolved this
   session — see "Ramp-up power analysis" below and `docs/HISTORY.md`'s
   2026-08-08 entry for the full finding and its caveats. **2026-09-02/03
   update: an overnight search for a geometry that closes this gap found
   a strong negative signal instead (~1247 T-A-evaluated candidates,
   T-A-safe operating current confined to 5-54A everywhere tried, best
   B_target 4.10T) — see "Ramp-up power analysis" below's 2026-09-02/03
   entry and "Known open issues" #7. Still unresolved, now more
   consequential.**

`params.py`, `optimize/opt_config.py`'s `CMAES_X0`, and the `cmaes_*` /
geometry / field figures in `visualization/` are all set to/regenerated
from this design.

---

## Key geometry parameters (from params.py, current champion)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `a` | 0.0260 m | Coil inner radius |
| `b` | 0.0314 m | Coil outer radius |
| `t` | 75 µm | Tape pitch (radial, = Λ in T-A formulation) |
| `w` | 4 mm | Tape width (axial z-extent of one layer) |
| `n_turns` | [382,382,478,478,3,3] | Turns per z-layer (3 double-pancake pairs) |
| `n_layers` | 6 | Number of z-stacked layers (must be even — double-pancake) |
| `n_turns_total` | 1726 | Sum of n_turns |
| `I_design` | 196.0 A | Operating transport current per turn (65% of local Ic) |
| `delta_SC` | 1 µm | REBCO SC layer thickness |
| `coil_half_gap` | 13.7 mm | Half the coil-to-coil centre separation |
| `B_target` | 10 T | Design bore field |

**Two-coil mode:** Coil 1 at z=0, Coil 2 at z=2×coil_half_gap. Fields are
superposed using Biot-Savart — **use
`coil2_field.compute_both_coils_field_multilayer()`**, not the older
single-filament `compute_both_coils_field()`, for any near-coil field
evaluation (see "Multi-filament Biot-Savart" below; every production
code path already does this).

These are `params.py` module-level values, not hardcoded constants —
`optimize/` mutates `a`/`b`/`t`/`w`/`n_turns` programmatically and calls
`params.recompute_derived()` to refresh every derived quantity and the
mesh sizing.

---

## Eighth-symmetry FEM domain

The mesh covers x≥0, y≥0, z≤coil_half_gap (one eighth of the full geometry).
The coil winding is in the domain. Symmetry BCs (PEC/PMC) are applied on
the cut faces. Cell markers: `coil_marker=1`, `outer_boundary_marker=3`.

---

## dolfinx API — confirmed quirks in this environment

These are NOT standard dolfinx — specific to the build in `fenicsx-env`:

1. **`interpolation_points` is a property**, not a method.
   ```python
   Vdg3.element.interpolation_points   # correct
   Vdg3.element.interpolation_points() # WRONG — TypeError
   ```

2. **`fem.dirichletbc` signature depends on value type:**
   ```python
   fem.dirichletbc(fn_Function, dofs)           # no V when value is Function
   fem.dirichletbc(constant_Constant, dofs, V)  # V required for Constant
   ```

3. **`LinearProblem` requires `petsc_options_prefix` as mandatory kwarg:**
   ```python
   LinearProblem(a, L, bcs=bcs,
                 petsc_options_prefix="my_prefix_",
                 petsc_options={...})
   ```
   Without it: `TypeError: missing 1 required keyword-only argument`.

4. **`domain.geometry.dofmap` is deprecated** — use `domain.geometry.dofmaps[0]`.

5. **`from dolfinx.io import gmsh as gmshio`** — this is the correct alias
   in this env (not `import gmshio` directly).

6. **`fem.functionspace(domain, ("DG", 0, (3,)))`** works for vector DG0.

7. **DG0 vector array layout:** block format — reshape(-1, 3) gives
   (cell_0_x, cell_0_y, cell_0_z, cell_1_x, ...).

---

## The T-A formulation (ta_solve.py)

### Physics

The T-A homogenisation models screening currents in the REBCO tape as a
bulk anisotropic conductor. The scalar potential T (current vector potential,
A/m) satisfies Faraday's law; A (magnetic vector potential) satisfies
Ampere's law with source J_s = (δ_SC/Λ) × ∇T×n̂.

**Key relationship:** `J_SC = ∇T × n̂`  (SC-layer current density, A/m²)
**T BCs:** `T_bot = +I/(2δ_SC)`, `T_top = −I/(2δ_SC)` at z = ±w/2
**A-form source:** `J_s = (δ_SC/Λ) × J_SC`

### Single-tape architecture (critical design decision)

The T-A formulation models **one tape of width w**. The z-stacked layers
are SEPARATE tapes — the T BCs must be at ±w/2 (single tape edges),
NOT ±n_layers×w/2 (full stack edges). Using the full stack gives the wrong,
below-critical j/jc.

**Implementation:** `ta_per_layer = True` (production default) solves T
independently in **every** z-layer — no cross-layer replication, and it
converges better than the older KD-tree-replicated single-tape mode. All
non-coil DOFs are pinned to T=0.

### Picard iteration details

- **Seed:** Uniform A-form solve → B field → ρ(B, J_uniform)
- **Per iteration:**
  1. Solve T-equation (linear, frozen ρ) with MUMPS
  2. Relaxed update: `T_h = (1-α)×T_old + α×T_new`
  3. Compute J from T
  4. Solve A-equation → new B
  5. Update ρ(J, B) using power-law with j_norm floor
  6. Check convergence (see below)

- **Two-phase relaxation:** Phase 1 α=0.30 (`ta_picard_alpha`, fast
  ramp-up) → Phase 2 α=0.15 (`ta_picard_alpha_fine`, fixed — NOT
  adaptive), triggered once |ΔB| stops decreasing monotonically.
  **Every adaptive α-throttle tried on the sharp-flux-front dataset
  misfired** (throttling freezes slow physical transients; raising
  re-excites limit cycles) — do not reintroduce one without testing
  against this dataset.

- **ρ regularisation:** `j_norm = max(|J|/Jc, eps_reg)` with
  `eps_reg=1.0` (a MAX floor at critical-state ρ = E_c/Jc, not additive),
  a smooth soft-max floor (`ta_floor_smooth_p=16`) instead of a hard
  kink, and log-space ρ under-relaxation (`ta_rho_relax=0.5`).
  `rho_SC = (E_c/Jc) × exp((n-1)×log(j_norm))` (not `j_norm**(n-1)`) to
  avoid float64 overflow at large n×j.

- **Convergence criterion (`ta_scif_stall_mT=0.05`):** the flux front
  wanders chaotically among near-degenerate states — raw `|ΔB|/|B|`
  floors at ~6-10e-4 with no clean decay, so no B-vector residual
  criterion converges. Instead: converged = EMA-smoothed bore SCIF moving
  < 0.05 mT over a 10-iteration window (earliest k=25). `ta_picard_tol`
  is diagnostic-only. **This scheme is validated ONLY at
  `dt = params.ramp_duration` (600 s), the single implicit step every
  production code path uses** — see "NI transient work" below for why
  shorter `dt` is a separate, unresolved problem.

### Convergence and performance

At `I_design`, `dt=600s`: cold start ~60-80 Picard iterations, warm start
(from a previous converged state, T scaled by `I_new/I_old`) far fewer.
The A-problem's bilinear form is constant, so its matrix is factorised
once (MUMPS) and every later solve — every Picard iteration, every sweep
current — is a cheap back-substitution (`_solve_A()`); this was the
dominant historical cost and is now solved. The T-problem (ρ-dependent)
stays a small, cheap `LinearProblem` refactorised every iteration.
`solve_ta_at_current(..., warm_start=True)` (default) verified to agree
with a cold start to 0.01% in SCIF.

---

## SCIF computation

```
ΔJ = J_TA − J_uniform_per_tape
J_uniform_per_tape = I / (δ_SC × w)   # ~50 GA/m²  (NOT divided by n_layers!)
ΔJ_s = (δ_SC/Λ) × ΔJ
ΔB_bore = μ₀/4π × Σ_cells (ΔJ_s × r̂/r²) × ΔV   (both coils)
```

**CRITICAL:** `J_uniform` must be `I/(δ_SC × w)`, NOT
`I/(δ_SC × n_layers × w)` — the n_layers factor was a historical bug that
made ΔJ dominated by transport current rather than the screening
component, giving 1000× wrong SCIF%. Every SCIF number reported before
2026-07-10 in this project's history was artifact-dominated by this and
two other now-fixed bugs (missing mirror-quadrant images in the Biot-Savart
sum; approximate instead of exact `ufl.CellVolume` cell volumes) — see
`docs/HISTORY.md` if you need the full bug chronology, and item 8-11 in
"Bugs fixed" below for the one-line versions.

**On-axis SCIF is NOT a usable stand-in for real (box peak-to-peak)
uniformity — it can be actively anti-correlated with it.** The design with
the best on-axis SCIF in this project's history had the *worst* box
uniformity of every design tried. `ta_solve.py`'s `dB_bore_from_dJ()`
supports evaluating over an arbitrary grid; always use the box grid (30×6mm,
matching `opt_config.py`'s `TARGET_X_M`/`TARGET_Y_M`), never the single
on-axis point, when uniformity is the question.

**Mesh resolution:** the screening profile lives across the tape width,
so `params.mesh_nz_per_layer` / `params.mesh_z_grading` (graded sub-slabs
across each tape, current production default
`[0.075, 0.15, 0.55, 0.15, 0.075]`) is the resolution axis that matters,
not in-plane refinement (which converges quickly and is not the
bottleneck). Box uniformity converges to roughly a **0.39-0.44% band with
~±0.2pp of irreducible alignment-driven scatter that does not shrink with
further refinement** on the current champion — it's a near-cancelling
dipole sum, sensitive to exactly where cell boundaries fall relative to
the ~1mm penetration front. **gmsh mesh generation is not perfectly
reproducible across separate OS processes** (only within one process) —
this caused two false alarms in this project's history (see "Operational
lessons" below); never trust a single cross-process uniformity number
near a constraint boundary without an independent-mesh repeat.

**2026-09-03/04 — diagonal "islands" in J/Jc cross-section plots
confirmed to be a mesh-resolution artifact, not real physics.** The
poster J/Jc cross-section figures
(`visualization/for poster/make_jjc_cross_sections.py`) showed diagonal
streaks of intermediate J/Jc crossing what should be smooth Bean
critical-state bands. A one-off finer mesh tier ("xdense":
`mesh_size_min_factor=0.05`, `mesh_dist_max_factor=4.00`, `box_scale=6.0`,
7-slab `mesh_z_grading=[0.05,0.075,0.10,0.55,0.10,0.075,0.05]`, roughly
denser than the existing `fine` tier) was solved at the champion's exact
`I_design=196A` and compared directly against the standing `medium`-tier
result (preserved as `racetrack_ta_fields_medium_backup.npz` and the
`*_medium_backup.png` figures). The xdense solve converged cleanly
(k=69, 1502s, SCIF +643.6mT vs. the established 641.3mT medium reference
— ~0.4% apart, within this section's own documented ~0.2pp irreducible
scatter band; box uniformity 0.425%) and the diagonal streaks are
**absent** at xdense resolution — both cross-sections resolve into
smooth, physically coherent bands with no cross-hatching. **Practical
consequence: `params.py` currently has these xdense settings live
(uncommitted, staged as a TEMPORARY block with the `medium` production
defaults commented out immediately below it) — revert to the `medium`
block before any further production solve/search that isn't specifically
re-checking this**, since xdense is ~20-30x the medium tier's solve time
for a difference that only matters for this one visual-artifact
question, not for any reported number in this file.

**Ic dataset:** the whole pipeline uses ONE tape's measured data — the
"Shanghai Superconductor High Field Low Temperature 2G HTS 20 K" CSVs in
`physics/`. Data covers 0–8 T; see "Ic(B) extrapolation above 8 T" below
for how the pipeline handles the champion's actual ~10-11 T peak field.

**Mechanical stress:** `validation/mechanical_stress_check.py` screens
hoop, delamination-direction tension/compression, and coil-coil axial
attraction; stress ∝ I², so it's an active, not incidental, constraint
near the design's operating current. `optimize/`'s search enforces hoop
stress only (400 MPa cap, end-cap curved sections) — delamination is
treated as a material property, not a design lever, per project
direction (still computed/reported, not enforced).

---

## n-value / Ic model

**Data file:** `physics/Shanghai_Superconductor_Low_Field_High_Temperature_2G_HTS_20_K_Angle_Dependence__1_.csv`
1100 points, 20 B-values × 55 angles, n=13-34, at ~20 K.

**`NValueModel`** (in `physics/ic_model.py`, after the `IcModel` class):
Uses `RectBivariateSpline` with σ=0.8 Gaussian smoothing. Import:
```python
from physics.ic_model import IcModel, NValueModel
```

**`n_value_csv_filename`** must be set in `params.py` (points to the same
CSV as `shanghai_csv_filename` but with a different column parsed).

---

## Ic(B) extrapolation above 8 T

The measured dataset only covers 0–8 T, but the champion's peak field is
~10-11 T. Every Ic lookup in this project used to default to
`clip_B=True` — flat-clamping Ic to its measured 8 T value above that
field. **This is optimistic, not conservative** (Ic decreases with B in
the measured range), and matters: 11.8% of the champion's own
quench-relevant Ic evaluations clip to the 8 T boundary.

**Hold-out validation** (fit a candidate Jc(B) form on low-field data,
score it on measured high-field points it never saw — the same
extrapolation ratio as 8T→~10.7T peak field) settled which model to trust:

| form | hold-out MAPE @1.6x extrapolation | reading |
|---|---|---|
| flat clamp (old default) | 26.7% | badly OPTIMISTIC |
| power law | 6.9% | slightly optimistic |
| **Kim, `Jc0/(1+B/B0)`** | **4.1%** | **BEST, mildly conservative** |
| pinning-force scaling law (fixed Bc2) | 6.1% | good, more conservative |
| Beta / max-entropy (Long 2013) | 5.5-14.3% | does not beat Kim — the data doesn't reach the field range that constrains its irreversibility-field parameter |

**Use `KimIcModel` (`optimize/ic_extrapolation.py`, `make_ic_model`)** as
the default for any B_target/quench evaluation above 8 T. It's selectable
in `cmaes_search.py` via `CMAES_IC_EXTRAP` (default `flat` = old,
optimistic, historical behaviour — deliberately NOT changed as the
default there, since every existing search result assumes it; switch it
explicitly when re-optimizing under the realistic model).

**Any `(1 − B/B_scale)`-type model (scaling law, Beta) MUST have its
cutoff field (Bc2 / B_irr) bounded well above the data ceiling** — this
project hit the identical bug twice, at the identical angle (88° in this
dataset): an unconstrained fit picks a spuriously low cutoff there,
collapsing Ic just above the data and silently wrecking the whole quench
bisection. Keep the floor at ≥20 T (REBCO at 20 K has B_irr ≈ 30-45 T).

---

## Manufacturing constraints (hard, current)

1. **Minimum bend radius 7.5 mm** (REBCO tape cracks tighter than this).
   Enforced in `cmaes_search.py`'s `geometry_violation()` against
   `a_inner_min` (the innermost turn's radius, from the layer with the
   most turns).
2. **Double-pancake construction.** Every pancake must be one of a PAIR
   of adjacent layers wound as one continuous piece of tape, which
   requires both layers in a pair to have the SAME turn count. Implemented
   as `N_PAIRS = N_LAYERS // 2` turn variables in the optimizer;
   `N_LAYERS` must be even (`cmaes_search.py` asserts this at import
   time). This eliminates all odd layer counts.
3. **7×14×1mm sensor array clearance** — checked directly against the
   geometry and found to need no dedicated constraint; automatically
   satisfied once (1) and the 3mm face-gap floor hold.
4. **Turn-count floor is 1** (not a fixed 50 — the old 50 had no material
   basis; `params.py` only ever asserted `n≥1`).

---

## Configuration optimizer (`optimize/`)

**Purpose:** minimize tape length subject to hard constraints:
`B_target_T >= 10.0 T`, operate at 50-60% of local Ic anywhere in the
winding (`SAFETY_FACTOR` in `opt_config.py` sets `I_op = I_quench/SF`),
hoop stress ≤ 400 MPa (end-cap curved sections only), coil-to-coil face
gap ≥ 3mm, plus the manufacturing constraints above.
**No uniformity signal is in the fitness function** — see "Proxy
graveyard" below for why, and always validate finalists with
`optimize/ta_validate.py`.

- `optimize/opt_config.py` — the ONLY file to edit for a new search:
  candidates/bounds, safety factor, target box (30×6mm, `TARGET_X_M`/
  `TARGET_Y_M`), stress allowables, Ic-clip hygiene limit.
- `optimize/optimize_geometry.py` — coarse uniform-J screen (~5s/design):
  one uniform-J FEM solve (field is exactly linear in I), per-cell quench
  bisection, multi-filament Biot-Savart target-box field.
- `optimize/cmaes_search.py` — the CMA-ES search (`pycma`, conda-forge).
  Variables: `a`, `b`, `coil_half_gap`, and `N_PAIRS` turn-pair values,
  all continuous (turns rounded to int only at evaluation). `a`/`b` are
  intentionally UNBOUNDED — physical limits (`b > a`, bend radius, etc.)
  are a smooth penalty in `geometry_violation()`, not a box bound.
  Run: `conda run -n fenicsx-env python3 optimize/cmaes_search.py`
  (~5-10s/eval; use `CMAES_N_WORKERS>1` for generation-parallel
  evaluation, 6 is a good default on an 8-core machine). Outputs
  `optimize/runs/cmaes_results.csv` (best design, overwritten each run —
  do NOT rely on it alone for a multi-job orchestration; pull by
  `run_tag` from the cumulative master log instead),
  `optimize/runs/cmaes_history.csv` (this run only), and figures in
  `visualization/`: `cmaes_convergence.png`, `cmaes_constraints.png`,
  `cmaes_variables.png`, `cmaes_overview.png` (this run),
  `cmaes_param_map.png` (cumulative across every run ever, from
  `optimize/runs/cmaes_all_evaluations.csv` — the append-only master log,
  ~100k+ rows, tagged by `run_tag`; never overwritten).
- `optimize/evaluate.py` — frozen entry point for the external
  optimization team, deliberately decoupled from the internal search's
  constants (its own `EVALUATE_*` constants in `opt_config.py`) so
  repurposing `opt_config.py` for internal searches can't silently change
  its behaviour.
- `optimize/ta_validate.py` — standalone extraction of `ta_solve.py`'s
  box-uniformity machinery; the ONLY trustworthy uniformity check, never
  touches `params.py` on disk. ~30-80s per solve. Always run 2
  independent-mesh repeats on anything near a constraint boundary.
- `optimize/ic_extrapolation.py` — `KimIcModel`/`ScalingLawIcModel`/
  `BetaIcModel`/`make_ic_model`; see "Ic(B) extrapolation" above.

**Multi-filament Biot-Savart:** `coil2_field.compute_both_coils_field()`
(single filament at nominal radius `a`) is only valid when the winding
pack is small relative to `a`/`coil_half_gap` — true at the ~50-80mm
scale this project started at, false at the ~15-30mm scale the optimizer
converges to. **Use `compute_both_coils_field_multilayer()`** (resolves
each layer's own radial/z position, grouped into sub-filaments) for any
near-coil field evaluation; every production code path already does.

Full run-by-run search history (every CMA-ES run, every rejected
champion, every proxy investigation) is in `docs/HISTORY.md`.

---

## Proxy graveyard — no fast uniformity proxy exists

Four cheap stand-ins for the true (T-A) box peak-to-peak uniformity have
each been tried and falsified against `ta_validate.py`, in order:

1. **On-axis SCIF** — anti-correlated with box uniformity in places (the
   best on-axis design had the worst box uniformity found).
2. **Peak-turns-per-pancake-pair penalty** — built from the same bad
   on-axis data, inherited its error.
3. **Bean-state dipole correction** (`bean_moments()`/
   `dipole_field_mirrored()` in `optimize_geometry.py`) — off by up to
   ~10x at the compact coil scale (a≈15-25mm) the search converges to;
   only ever validated at the original ~50-80mm scale.
4. **Uniform-J box field as a coarse pre-filter** — designs that scored
   well pre-filter (1.41-1.45%) scored 1.65-2.12% under real T-A (all
   FAIL).

**T-A (`ta_validate.py`) is the only arbiter.** A solve is only ~30-80s
at this coil scale, so the practical search shape is "tens of
physically-chosen or CMA-ES-found candidates, each T-A-validated
directly" — not thousands of proxy-filtered ones, and not a fifth guess
at a cheap formula.

**Also retracted:** "box uniformity tracks coil radius `a` monotonically
(bigger = better)" — isolating `a` with everything else held fixed gives
a **V-shaped bowl with an interior minimum**, not a monotone curve. Do
not reintroduce an `a`-maximizing bias or an `a`-floor on this basis.

---

## NI (no-insulation) transient work — `circuit/` and `transient/`

**Project direction:** the coil is committed to no-insulation (NI)
winding. At DC steady state the radial current vanishes, so every design
number in "Current design" above is unaffected — this adds a transient
(ramp/discharge) constraint, it does not change the design.

**TL;DR on Phase B (T-A transient), as of 2026-08-05:** Phase A
(`circuit/`) is solid and unaffected by any of this. Phase B is not.
Five different fixes were tried for the short-dt/multi-step convergence
problem — Picard relaxation tuning, a Gauss-Seidel Newton hybrid, a fully
monolithic block Newton, adaptive step-size marching, and (as a
diagnostic, not a fix) forcing deterministic threading/hashing — and
**none of them produced a trustworthy multi-step transient result.** The
deepest finding isn't any one of those failures — it's that this solver's
cold-start convergence is **genuinely non-deterministic across separate
process launches** (the identical configuration succeeds ~2/5 and fails
~3/5 of the time), for a reason that survived direct testing of every
obvious explanation (mesh non-reproducibility, threading, hash-seeding).
Treat every `transient/` multi-step number as provisional until that's
understood; a single successful run proves nothing. Full blow-by-blow,
including two retracted conclusions, is in `docs/HISTORY.md`'s 2026-08-05
entries — read those before trusting or extending anything here.

### `circuit/` — Phase A, lumped DCN circuit model (VALIDATED)

Reduced-order model: per-turn mutual inductance via a Neumann double
integral with geometric-mean-distance regularisation (GMD =
0.2235(t+w) ≈ 0.91mm — necessary since GMD ≫ the 75µm turn pitch, so
adjacent turns really are nearly as coupled as a turn to itself), a
per-unit-current field matrix for Ic(B,θ) lookups, and a DCN ladder
(`i_k + j_k = I(t)` exactly at every rung for an NI winding) solved with
`scipy.integrate.solve_ivp(method="BDF")` (the power-law nonlinearity,
n~13-22, is stiff — RK45 will not converge).

**Validated:** filament sum matches the production Biot-Savart path to
0.18% median / 0.44% max; energy balance on discharge closes to
0.00-0.08%; the model structure and tau prediction reproduce a published
benchmark (He et al. 2025) to 4.2% (though the paper's own table is
internally inconsistent, so an absolute rho_c→tau calibration is not
independently confirmed — carry a factor ~2-3 uncertainty on any tau from
an assumed contact resistivity).

**Champion results:** self-inductance **419.7 mH**, stored energy
**8.07 kJ** at 196A. `tau = 1330 / rho_c[µΩ·cm²]` seconds (confirmed
1/rho_c scaling). Ramp-end field deficit and contact heat both scale off
one number, tau: `E_contact = 2·W_stored·tau/t_ramp` (verified against
the DCN to 7%). **Sudden discharge puts all 8.07 kJ into the winding**
(NI has no external dump path by design) — this is the thermal design
case and is NOT yet modeled (isothermal, EM-only).

Run:
```bash
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
$PY circuit/validation/lumped.py; $PY circuit/validation/he2025_racetrack.py
$PY circuit/run_charge.py; $PY circuit/run_discharge.py; $PY circuit/postprocess.py
```

### `transient/` — Phase B, T-A with the circuit closure (EXPLORATORY, NOT VALIDATED for general use)

This adds the NI circuit closure directly to the T-A Picard/Newton solve,
to get hysteretic loss and current redistribution across the tape width
during a ramp — information the lumped model cannot see.

**The blocking finding:** the base T-A Picard solver (`ta_solve.py`) has
only ever been validated at `dt = params.ramp_duration` (600s, a single
implicit step over the whole ramp) — every prior use of the solver in
this project used exactly that. Genuine multi-step time-marching needs
shorter `dt`, and the Picard scheme **does not reliably converge there**:
extensive testing (uniform relaxation, per-layer relaxation, Anderson
acceleration, different current-jump sizes, different schedules) all
failed to fix it. A quasi-Newton reformulation
(`transient/newton_ta.py`, `hybrid_step()`) **does** fix accuracy at the
validated dt=600s case (reproduces Picard's own 641.26 mT ground truth to
1.16%, confirmed by directly transplanting a wrong Newton-hybrid state
into the validated Picard solver and watching it self-correct in 44
iterations) — but **does not fix the original short-dt / multi-step
motivating problem**, tested directly and confirmed still failing.

**RETRACTED, 2026-08-05 (overnight, same day): the paragraph below (kept
struck-through for the record, see replacement after it) was wrong on two
counts — a genuine implementation bug AND a false-positive convergence
read, not a real finding about the monolithic approach's viability.**
~~A fully monolithic reformulation (`transient/monolithic_ta.py`,
2026-08-05) was tried and does NOT fix it either. All six layer-T
unknowns and the shared A unknown were solved as ONE PETSc SNES block
system (true simultaneous coupling, not Gauss-Seidel), with Jc(B)/n(B)
refreshed every Newton step instead of once per outer sweep. Undamped, it
diverges in 2 steps; damped (blending the joint step with its pre-step
state), it converges *cleanly* — but to three different wrong answers at
three different damping strengths (+6800/+3669/+2012 mT against a
+641 mT truth), each a monotonic function of the damping constant, which
proves the plateau is a scheme artifact, not a slow-but-correct answer.~~

**What actually happened, found during the differentiable-Jacobian work
below**: `monolithic_ta.py` has a genuine Dirichlet-BC block-scoping bug,
present since the file was first written. All 6 layers' T-functions share
ONE `fem.functionspace` object (`ta_solve.setup_ta_problem`'s single
`V_T`), and `dolfinx.fem.bcs.DirichletBC` carries no reference to which
block it targets — so the flat-concatenated BC list applies EVERY layer's
"pin outside my own cells" condition to EVERY layer's block. Verified by
direct Jacobian inspection: every T-layer's diagonal was uniformly 1.0
(including dofs that layer's own construction meant to leave free — not
just the genuinely-pinned majority), and all 12 T-A off-diagonal coupling
blocks were EXACTLY zero — i.e. this was never actually a coupled
monolithic system at all, contrary to the claim above; it was an almost
fully decoupled, near-trivial-identity system. Separately, and
independently of the bug: the "converges cleanly to three different
answers" claim was never checked against the raw PETSc residual, only
against the same SCIF-EMA-stall diagnostic used everywhere else in this
file — and the raw residual, when checked (re-running the exact
dt=600s/I=196A configuration behind the +6800/+3669/+2012mT numbers with
`snes_monitor` enabled), explodes then plateaus at 3.3-4.0 BILLION at
every damping level tested, never decreasing. None of the three damping
levels ever converged in any residual sense — they are three different
points along three different diverging trajectories whose SCIF-drift-rate
happened to slow down enough to trip the stall criterion, not three
distinct fixed points. **The "proves it's a scheme artifact" conclusion's
own premise (genuine convergence at each level) does not hold.**

A structural fix (giving each layer a distinct function-space object) was
implemented as a proof-of-concept and verified directly (exact match
between "dofs at diagonal=1.0" and the true pinned-dof count; all 12
coupling blocks now genuinely nonzero) — but is NOT yet wired into
`monolithic_ta.py`'s actual `build_monolithic_problem`, and fixing the bug
alone does not produce convergence: the correctly-coupled system's
residual is far larger from the first step (previously-hidden real
coupling now contributes) and, under the old (buggy-system-tuned) damping
parameters, is chaotic rather than convergent. A leading untested
hypothesis is a severe (6-8 order of magnitude) natural-scale mismatch
between the T-blocks and the A-block — MUMPS auto-scaling gave a modest
~2x improvement (real but not dominant).
**Whether a correctly-assembled monolithic T-A Newton system can converge
at all is genuinely OPEN — not the closed "no" the retracted paragraph
above claimed, and not a "yes" either.** See
`transient/validation/monolithic_diff_investigation_2026-08-05.md` for
the complete, dated arc (this is a `transient/`-local file, not yet
folded into `docs/HISTORY.md`).

**Update, 2026-08-06 (this paragraph was stale — PCFIELDSPLIT was tried,
not "not yet tried"):** real `PCFIELDSPLIT` (block-Jacobi/additive, then
Schur) was implemented and tested on the bug-fixed system (Part 6 of the
investigation file). Block-Jacobi produced the first genuine, substantial
residual decrease seen anywhere in this whole investigation — but not
reliably: identical launches of the identical configuration sometimes
show real convergence progress and sometimes fail immediately, the same
cross-process floating-point-sensitivity chaos documented elsewhere in
this project. Schur hit a real PETSc bootstrapping obstacle (its
`SELFP` approximation needs an already-assembled matrix, but the very
first solve fails outright under Schur's own defaults) and was not
resolved. A follow-up (Part 7, blocked; Part 8, resolved the block —
see the investigation file for the dated detail) got a working bit-level
diff of the pre-solve iteration-1 system by bypassing `SNES`'s own
introspection (`computeFunction`/`computeJacobian`, which crash/error
before any real solve) and instead calling dolfinx's module-level
`assemble_residual`/`assemble_jacobian` directly — the same functions
SNES calls internally, without going through its lifecycle state. See
the investigation file's Part 8 for what that comparison found.

**"This changes the standing recommendation on H-formulation too" (below)
should be read with the above in mind — it is NOT retracted, since Picard
and the Gauss-Seidel Newton hybrid's own pathologies stand independently
of the monolithic bug, but it can no longer cite the monolithic result as
a third independent confirmation**: since that result rested on a system
that was never actually coupled the way it was believed to be, and never
actually converged, it isn't evidence about whether tight T-A coupling
sidesteps the underlying problem — that question is open, not answered.
The original paragraph is kept below for the historical record of the
reasoning, with this caveat attached: since Picard (sequential) and the
Gauss-Seidel Newton hybrid have hit the identical pathology, the problem
isn't which fields are unknowns or how tightly they're coupled — it's a
Newton-type large step interacting badly with an observable (SCIF) that's
a near-cancellation invisible to any residual norm. An H-formulation would
face the identical stiff power law and the identical near-cancelling
observable, so it would not be expected to sidestep this either. See
`docs/HISTORY.md`'s 2026-08-05 entry for the full investigation.

**Practical consequence: do not trust any `transient/` SCIF, radial
current, or ramp-trajectory number without first checking
`docs/HISTORY.md`'s dated entries for exactly what was validated (or
retracted) at that specific `dt`/schedule** — several results in that
history were later found to be false stalls (a scheme that "converged"
cleanly to a value later proven wrong) or genuine instabilities that only
appeared after hundreds of iterations. The standing lesson from this
whole investigation: **a solver reporting "converged" is not the same
claim as "the answer is correct"** — always check a new formulation
against independent ground truth, not just its own status flag.

**Adaptive step-size marching (`transient/adaptive_march.py`, 2026-08-05)
was tried and INITIALLY LOOKED like it broke this impasse — that
conclusion was RETRACTED the same day after a re-check.** A
Newton-iteration-count step controller wrapped around the unmodified base
Picard machinery (`ta_transient._picard_phase`) first appeared to march
the INSULATED case through the entire 600s ramp cleanly from a large
first step (dt≈60s → +828.50 mT). But that result came from a run that
also tested two other `dt_init` values in the SAME process, and when the
dt≈60s reference case was independently re-run in true isolation, **it
did not reproduce** — instead failing repeatedly at the ramp start with
negative SCIF values, a signature seen nowhere else except marginal/
questionable states. The likely cause is NOT the same-process PETSc-reuse
issue that was correctly caught and controlled for on the other two
values (dt≈60s was the *first* configuration tried in both the original
sweep and the failed re-check, so process-reuse doesn't distinguish them)
— three candidate explanations were tested directly and ALL REJECTED:
cross-process gmsh mesh non-reproducibility (two independent builds of
the identical geometry gave byte-identical mesh files), multi-threaded
BLAS/MUMPS floating-point non-associativity, and Python per-process
hash-seed randomisation (forcing `OMP_NUM_THREADS=1` +
`PYTHONHASHSEED=0` did not stabilise the outcome — if anything it did
worse, 0/5 successes vs. baseline's 2/5, on ten repeats of the identical
single first-step configuration). **The system's cold-start convergence
is genuinely non-deterministic across separate process launches, for a
reason that remains unidentified after direct testing of the obvious
candidates.** No adaptive-marching SCIF number from this whole
investigation (+190, +70, +828.50 mT, or any of the isolated re-checks)
should be treated as validated. A genuinely isolated decoupling
diagnostic (varying `dt` and the target current independently) did
produce one clean result — only a large `dt` AND a large `I` together
converged, in that one run — but per the finding above, a single run of
this configuration is not evidence of anything; the same test could as
easily have failed. See `docs/HISTORY.md`'s 2026-08-05 entries for the
complete, honest arc: the `dt_const` bug fix (real, keep it), the
retraction, and the two rounds of hypothesis-testing that narrowed but
did not close the question.

**Practical consequence, the most important takeaway:** any future claim
that a transient schedule or solver change "works" must be based on
multiple repeated isolated runs of that specific configuration, not one
— a positive result here is exactly as likely to be a fluke as a
negative one is to be a contaminated run. **Not yet done:** bit-level
comparison of the assembled matrix/RHS between a successful and a failed
run at the very first Picard iteration (before any nonlinear amplification
has a chance to act) — the only diagnostic left that could actually
localise the source, and a genuine scope decision given the cost of
today's investigation relative to what it settled. The short-dt/
multi-step convergence problem remains OPEN, now with two ruled-out
explanations and one confirmed real bug fix (`dt_const`) as the net
progress.

Run:
```bash
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
$PY transient/validation/insulated_limit.py   # B0 gate — must stay PASS
```

### 2026-08-05 (continued): half-domain hypothesis rejected; nondeterminism root-caused, not fixed

Two more directed investigations, run as parallel background jobs with
their own isolated git worktrees, each with multiple internal rounds of
self-correction. Full raw evidence in
`transient/validation/half_domain_investigation_2026-08-05.md` and
`transient/validation/nondeterminism_investigation_2026-08-05.md` — read
those before extending either conclusion; both contain honest retractions
of their own early results, not just final numbers.

**Half-domain (drop the x=0/y=0 symmetry cuts) does NOT fix short-dt
non-convergence — REJECTED on two independent lines of evidence.** The
hypothesis: the eighth-symmetry domain's x=0/y=0 mirrors cut directly
through each turn's own current loop, forcing the discretized source to
have nonzero divergence, which pollutes the raw vector potential A by
~1e10 (`transient/induction.py`'s finding). A half-domain (only the
z=coil_half_gap mirror, ~3.5x the cells — `mesh/build_mesh_half.py`,
production path untouched) measurably cut that gauge pollution by ~114x
(8.76e7 vs ~1e10) — the mechanism is real — but the short-dt Picard
failure signature (smooth initial decay into unbounded chaotic wandering)
was qualitatively IDENTICAL on the half-domain, and a follow-up test that
additionally froze Jc(B)/n(B) for the entire step (removing the other
candidate lag mechanism entirely) *still* showed 0/5 converged with the
same signature. Both confirm the a priori argument already in this file:
the T-equation only ever consumes `curl(A_h - A_prev)`, which is
gauge-invariant, so A's gauge freedom was never the mechanism. The
half-domain infrastructure is left in the repo (validated boundary
tagging, current-direction generalization, working per-layer T-A setup)
for any FUTURE hypothesis that specifically needs a domain without the
x/y cuts — this is not that case.

**Cross-process nondeterminism: root cause substantially identified —
default multi-threading's floating-point non-associativity — but this is
NOT a bug to fix.** Genuinely single-threaded execution (verified by
observing zero CPU ticks on every non-main thread over sustained windows
via `/proc/<pid>/task/*/stat`, not by trusting env vars — a prior same-day
attempt at this exact test was silently a no-op due to a bash
array-export-across-process-boundary bug, caught and corrected) makes the
canonical repro case (dt=60s, I=19.6A, cold start) **perfectly
deterministic — and deterministically non-convergent**: 0/18 converged
across 3 independent batches, 2 env-var variants, 2 load regimes
(p≈6e-7 against the ~40% baseline rate), with every failure converging to
the IDENTICAL final SCIF to printed precision. Mechanistic grounding: an
iteration-1 bit-level dump/diff (the diagnostic this file previously
flagged as "not yet done") showed a near/below-machine-epsilon difference
in the assembled T-equation matrix/RHS becomes a ~1e-3-to-1e-4 RELATIVE
difference in the solved T after just ONE linear solve — a ~1e17-1e19-fold
amplification, confirming extreme ill-conditioning in the linear
sub-problem (plausibly the smoothed critical-state floor's huge dynamic
range in `rho_fn` across cells near j/jc≈1). Honest caveat: this
amplification is generic to ANY two independent runs, including
fail-vs-fail controls — it does not by itself discriminate which runs go
on to succeed. **Conclusion: this is a chaotic map whose one
deterministic trajectory from this exact cold start fails, and whose only
source of run-to-run variation — uncontrolled thread-scheduling noise —
is what currently gives it a ~40% chance of escaping onto a converging
trajectory instead. Eliminating the nondeterminism does not fix this
solver; it locks it onto its own worst-case trajectory.** Two new inert,
env-var-gated diagnostic hooks were added for any future work on this:
`TA_MUMPS_EXTRA_OPTS` (`solve/ta_solve.py`, JSON dict of MUMPS ICNTL/CNTL
overrides) and `DUMP_ITER1_MATRIX_PATH` (`transient/ta_transient.py`,
dumps the first Picard iteration's assembled matrix/RHS/solution per
layer). Separately, confirmed as a real, independent structural hazard
regardless of the above: `~/.cache/fenics` (the FFCx JIT cache) is a
single global, unversioned directory shared by every process on this
machine, with no locking — the same class of concurrent-output-clobbering
bug already flagged for `cmaes_search.py` in "Operational lessons" below,
here affecting compiled-form caching instead of CSV/log output.

Neither investigation found something actionable as a quick fix. What
they add: the short-dt problem is no longer "an unexplained reliability
bug" — it is a characterised chaotic Picard map, currently kept alive at
a ~40% success rate only by an accident of its own floating-point
environment, with the ill-conditioning of its linear sub-problem now
quantified. The two real remaining levers — a genuine differentiable
Jacobian for Ic(B)/n(B) through the measured-CSV spline models (removing
outer-loop coefficient freezing entirely, a substantial rewrite), or
deliberately engineering a small controlled perturbation as an escape
mechanism in place of relying on uncontrolled thread noise — are each
correctly scoped as separate, user-reviewed decisions, not next steps
taken here.

### 2026-08-05 (same day, evening/overnight): a differentiable Jacobian
was built — the attempt to use it surfaced a real bug in
`monolithic_ta.py`, retroactively correcting the monolithic-Newton
conclusion above

Full arc in `transient/validation/monolithic_diff_investigation_2026-08-05.md`
— read it before extending or citing any of this; it contains a retracted
false-positive result and exactly how it was caught, not just the final
state.

**Built**: `physics/entropy_ic_model.py` — `Jc(B,theta)` fit to the Long
(2013) maximum-entropy Beta equation (a small field-offset regularization
added, disclosed, so a best-fit `alpha<1` doesn't diverge at the
zero-field-cooled start — MAPE 2.29% over the full measured 0-8T grid,
`Jc(0)` accurate to 1-3%); `n(B,theta)` fit to a smooth empirical
Hill-type decay (explicitly NOT from the entropy paper — MAPE 0.47%).
Both pure algebra in B, wired as genuine UFL expressions of
`curl(A_h)` into a new `transient/monolithic_ta_diff.py` (purely
additive, does not modify `monolithic_ta.py`/`newton_ta.py`/`ta_solve.py`)
— the one place in this project a differentiable Jc(B)/n(B) can matter at
all, since the production Gauss-Seidel path never has B as a live unknown
during a T-solve. A unit bug (Ic in amps vs. the required volumetric
A/m²) was found and fixed.

**A result that looked like a genuine breakthrough was retracted after
direct verification.** Testing initially showed 8/8 independent process
launches converging to SCIF values agreeing to 0.003% (vs. this project's
~20-40% historical success rate with 400+mT scatter even among
successes) — but this was a false positive: the raw PETSc residual was
exploding exponentially (to ~1e95) the entire time, while the SCIF-EMA-
stall diagnostic (borrowed from the Picard scheme, where it's meaningful)
coincidentally plateaued and reported `converged=True`. Caught by
enabling `snes_monitor` and looking at the actual residual, which nobody
had done for ANY monolithic-Newton attempt in this project before,
including the original one.

**Doing that retroactively on the ORIGINAL `monolithic_ta.py`'s own
historical result found the same false-positive signature underneath
it.** The "converges cleanly to three different wrong answers at three
damping strengths" claim in this file (just above, now marked RETRACTED)
never checked the raw residual either — it explodes then plateaus at
3.3-4.0 billion at every damping level, never decreasing, when actually
checked. Digging into why surfaced the real cause: a genuine Dirichlet-BC
block-scoping bug (all 6 T-layers sharing one function space, so BCs
meant for one layer silently apply to all of them), present since
`monolithic_ta.py` was first written — verified by direct Jacobian
inspection (every T-A coupling block was EXACTLY zero; the system was
never actually coupled at all, contrary to its own docstring's claim). A
structural fix is implemented and verified correct
(`transient/validation/monolithic_fixed_bc_test.py`/`_verify.py`,
proof-of-concept only, not yet wired into `monolithic_ta.py` itself) —
but fixing the bug does not by itself produce convergence; the
correctly-coupled system's residual is far larger and, under old
(buggy-system-tuned) damping, chaotic rather than convergent. A severe
(6-8 order of magnitude) T-block/A-block natural-scale mismatch is the
leading untested hypothesis (MUMPS auto-scaling gave a modest ~2x
improvement; `PCFIELDSPLIT`/proper block preconditioning is the next
lever).

**Net effect on this file's standing conclusions**: whether a correctly-
assembled monolithic T-A Newton system can converge is now genuinely
OPEN — neither the closed "no" the original paragraph claimed, nor a
"yes". The broader "problem isn't which fields are unknowns, it's an
SCIF/near-cancellation issue no residual norm sees" conclusion is NOT
retracted (Picard and the Gauss-Seidel Newton hybrid's pathologies are
unaffected by this bug), but it can no longer cite the monolithic result
as independent third confirmation, since that result was never what it
appeared to be. `physics/entropy_ic_model.py`'s fitted models remain
solid, reusable infrastructure regardless of how the monolithic question
resolves.

### 2026-08-06: PCFIELDSPLIT tried (real but unreliable), a direct-assembly
bypass unblocked the stuck bit-level diff, two "smarter warm-start"
remedies both gave mixed/null results, and — the actual breakthrough —
the deterministic failure was traced and root-caused to an ordinary
relaxation-parameter instability, not irreducible chaos

Full arc across two files, in order: `transient/validation/
monolithic_diff_investigation_2026-08-05.md` (Parts 6-8) and
`transient/validation/nondeterminism_investigation_2026-08-05.md` (the
whole "2026-08-06" tail, several dated continuations) — read both before
citing or extending anything here; each contains its own retractions and
methodological corrections, not just final numbers.

**PCFIELDSPLIT, n-continuation, the Bean seed, and jitter-retry were each
tried and each fell short, in informative ways:**
- **Real PCFIELDSPLIT** (block-Jacobi) on the bug-fixed monolithic system
  gave the first genuine residual decrease in this whole investigation —
  but unreliably (same cross-process chaos as everywhere else). Schur hit
  an unresolved PETSc bootstrapping issue. A follow-up bypassed the SNES
  introspection crash that had blocked a bit-level pre-solve diff (by
  calling dolfinx's `assemble_residual`/`assemble_jacobian` directly
  instead of `snes.computeFunction`/`computeJacobian`) and found the
  RHS/residual is already O(1) relatively different between independent
  launches BEFORE the monolithic system is even assembled — the
  divergence originates upstream, in the Picard bootstrap seed phase, not
  in the monolithic solver itself. A follow-up trace showed this
  saturates by bootstrap iteration ~20-25, in two regimes: an ~11-order-
  of-magnitude jump in the first 1-2 iterations (matching the established
  per-solve amplification), then slower geometric growth to full
  decorrelation.
- **n-value continuation** (ramp the power-law exponent up from a mild
  n=3 instead of starting at the full physical n=13-34) gave a real,
  consistent 5-10x reduction in divergence mid-bootstrap (iterations
  10-20) — but the advantage was gone by the iteration-30 handoff point
  in every tested ramp length (5 configs, dwell time 2-30 iterations,
  non-monotonic result). **An analytic Bean-like critical-state seed**
  (replacing cold T=0 with an edge-weighted initial profile, deliberately
  NOT the exact Norris closed form — reconstructing that from memory
  risked a real transcription error, caught mid-derivation) did even
  worse: inconsistent even mid-run, and the A-field was over 2x worse
  than doing nothing at handoff. Both structurally different levers
  landing on the same negative result argues the ~20-25-iteration
  saturation window is a property of the map itself, not the cold-start
  recipe.
- **Jitter-retry** (cold-reset and retry with a small explicit
  perturbation if a step fails to converge) DID roughly double the raw
  success rate for the canonical `dt=60s, I=19.6A` repro case (37.5% ->
  75% over up to 5 attempts) — but could not be shown to add anything
  beyond what plain retrying (benefiting from ambient thread noise alone)
  would already achieve, and — flagged directly by the user, who had
  seen this project already get burned once by trusting an EMA-smoothed
  convergence flag without checking the raw signal — a follow-up found
  that DIFFERENT "converged" retries were landing on WILDLY different
  final SCIF values (e.g. -21.5, -1.3, +74.2 mT — different signs, not
  just different magnitudes) even in a ZERO-jitter control arm. Forcing
  convergence via retry, jittered or not, was NOT trustworthy as tested.

**The fix, and its full validation — consolidated final account
(superseding several rounds of same-day correction-on-correction; see
`transient/validation/nondeterminism_investigation_2026-08-05.md` for
the complete, honest, self-correcting arc if the reasoning history
itself is ever needed):**

**Root cause.** Traced the canonical `dt=60s, I=19.6A` repro case's full
trajectory under the project's verified-single-threaded recipe
(bit-identical across independent launches — the strongest form of
determinism checked, the entire per-iteration sequence, not just
pass/fail). It is not a slow drift and not the period-2 limit cycle this
project fixed at `dt=600s` (ruled out via autocorrelation — this is
higher-dimensional chaotic wandering). `T` overshoots to roughly
**+150x/-100x its own boundary-condition scale within the first 5
iterations** and stays parked in a persistent, bounded, large-amplitude
attractor thereafter — `|dB|/|B|` never drops below ~50-90% per
iteration at EITHER validated relaxation setting (`alpha=0.30` fast
phase, `0.15` careful phase). The two-phase scheme provides **zero
effective damping** at short `dt`, full stop, independent of noise.

**Fix.** A ~10x smaller, still-fixed, still-two-phase relaxation pair —
**`alpha=(0.03, 0.01)`** vs. the `dt=600s`-tuned `(0.30, 0.15)` — set via
`params.ta_picard_alpha`/`ta_picard_alpha_fine` before calling the
UNMODIFIED `_picard_phase`. Deliberately NOT an adaptive throttle (this
project's history explicitly warns against reintroducing one). At
`alpha=(0.10, 0.05)`, only ~3x smaller, the map is STILL not
contractive — this is a genuine threshold effect, not a smooth function
of alpha.

**Critical methodological lesson, learned the hard way and now
load-bearing for every number below:** `_picard_phase`'s own EMA-based
`converged` flag is NOT trustworthy in this regime at ANY alpha,
including the fix — it fires 300-1000+ iterations before genuine
settling. Every validated number below comes from FORCED full-length
runs (`min_iters=max_iters`, bypassing the stall check entirely) and raw
diagnostics (`T_max`/`T_min` per layer — not just pooled — `|dB|/|B|`,
raw non-EMA SCIF), never the flag alone.

**Full validation, accuracy-prioritised (four stages, all forced-full-
length):**
- **Generalisation across `(dt, I)`**: genuinely converges (verified via
  raw diagnostics) across `dt` in {600, 300, 150, 100, 60}s at
  `I=19.6A`, and across `I` in {49, 98, 196}A at `dt=60s` — a real,
  meaningfully wide operating window, with smooth physical trends in
  both SCIF and the (understood, non-alarming — see below) minority-
  layer overshoot depth. **Fails again at `dt=30s`** — confirmed
  genuine, not premature-stopping (`T` still 6-11x boundary scale even
  forced to 1200 iterations) — a real, now well-characterised boundary,
  consistent with the `1/dt`-forcing-coefficient mechanism (that term is
  2x larger again at `dt=30s` than at the just-barely-sufficient
  `dt=60s`).
- **Multi-threaded reliability, at the TRUE convergence horizon** (not
  the premature one): 5 independent noisy launches at `dt=60s, I=19.6A`
  agree to **0.004%** (122.424-122.429mT) — tighter than the original
  (premature-stop) 0.15% finding, not weaker.
- **Genuine multi-step ramp — the first ever run in this project's
  history** (every prior test, this session and before, was a single
  first step only): 5 steps at `dt=60s`, current stepping 19.6→98A,
  `A_prev` genuinely carried forward between steps. **All 5 converge
  cleanly** — `T_max/amp` settles to ~1.000 from step 2 onward,
  `|dB|/|B|` shrinks monotonically step to step (0.069→0.011), SCIF
  trajectory is smooth and physically sensible (124.7→539.7mT). This
  closes the single largest gap in the whole investigation.
- **Real cost**: ~700-1500 iterations for genuine convergence, not the
  ~460 first assumed — a materially larger, but bounded and now
  well-characterised, price.

**A resolved false alarm, worth recording precisely because it looked
serious before it was understood:** comparing this harness's converged
values against the project's established `641.26mT` reference
(`dt=600s, I=196A`) at first showed a ~2% gap, blamed on `alpha`. It was
NOT an alpha problem — running DEFAULT alpha through the same harness
ALSO gives ~653.9mT, matching the fix's own result to <0.02%. The 2% gap
is between this test harness (`_picard_phase`, `ta_transient.py`) and
the SEPARATE, independent production implementation
(`ta_solve.solve_ta_at_current()`, still reproducibly giving `641.27mT`
today) — present regardless of alpha choice. Root cause not found
despite five targeted isolation tests, though a small (~0.002%),
deterministic (confirmed via bit-identical repeats, not noise)
seed-level difference was traced amplifying to the visible gap over
iterations — consistent with, not contradicting, this project's
established sensitivity findings. **Practical upshot: comparisons
BETWEEN `_picard_phase`-based runs (different alpha, different `(dt,
I)`, different repeats) are valid and are what every claim above rests
on; comparisons AGAINST `solve_ta_at_current()`-sourced reference values
are not, until this specific harness discrepancy is understood.** A
secondary, now-understood curiosity surfaced by the same investigation:
the two 3-turn "closure" layers (of six, turn counts
382/382/478/478/3/3) show large but STABLE `T` excursions (up to -86x
boundary scale) at genuine convergence under the fix — confirmed absent
under default alpha at `dt=600s`, so specific to the short-dt/small-alpha
regime, shrinking smoothly as current increases, not a divergence.

**Status: this is now a genuinely, substantively validated fix** — not
a single-point curiosity — across `dt` in [60s, 600s], `I` in
[19.6, 196]A, single- and multi-threaded execution, and genuine
multi-step ramps. **Not yet tested (as of 2026-08-06)**: multi-threaded
execution of a full ramp (only single-threaded); a ramp crossing the
`dt=30s` boundary mid-sequence; the NI circuit closure (out of scope
here, as for everything else in this section — insulated limit only); a
real production ramp schedule (0 → full design current) rather than the
specific 5-step test schedule used; whether an even smaller alpha
rescues `dt=30s`; and the root cause of the harness-vs-`solve_ta_at_current`
discrepancy. See the 2026-08-07 entry immediately below for the first
two of these.

### 2026-08-07: full production-scale ramp under genuine multi-threaded execution — reliable run-to-run, but a new ~2.2% single-vs-multi-threaded gap; dt=30s-crossing test in progress

Closes two of the "not yet tested" items above, one fully and one
partially. Full evidence in
`transient/validation/nondeterminism_investigation_2026-08-05.md`'s
2026-08-07 entry.

**Multi-threaded full ramp (10 steps, 0→196A, dt=60s): reliable, but not
yet accurate to the single-threaded reference.** Ran
`transient/validation/full_ramp_run.py` TWICE under ordinary unforced
execution (no single-thread env-var pins), same schedule as the
2026-08-06/07 single-threaded Stage 1 reference. Both runs converged
cleanly (`finite=True` all 10 steps, `T_max/amp`→1.000 by step 4, smooth
saturating SCIF) and agreed with EACH OTHER to 0.001-0.03% at every step
— as tight as the earlier single-step multi-threaded result. **But both
sit ~2.2% below the single-threaded reference at every comparable step**
(730.1mT vs. 746.2mT final-step SCIF) — a real, systematic gap, not
noise, structurally the same shape of problem as the already-open
harness-vs-`solve_ta_at_current()` discrepancy above (unresolved there
too). **Practical upshot: trust multi-threaded execution for reliability
(run-to-run agreement), not yet for exact quantitative agreement with a
single-threaded number to better than ~2%.**

**dt=30s-crossing mid-ramp test: a warm-started crossing does NOT
reproduce the cold-start chaos.** `transient/validation/dt_crossing_ramp.py`
— steps 0-2 clean dt=60s baseline (reproduces the Stage-1 reference to
3+ sig figs), steps 3-4 deliberately dropped to dt=30s (the same dt
confirmed "fully chaotic" — dB_rel=1.02, T_max/amp=10.8 — in the
cold-start/I=19.6A boundary sweep), steps 5-9 back to dt=60s.
**Result, checked on raw `dB_rel` not just the printed diagnostic**:
step 3 settles to `dB_rel=0.162` (elevated, transitional-like, closer to
the `dt=35s` reference point than to chaos), step 4 (second consecutive
dt=30s step) settles fully clean (`dB_rel=0.012`), and steps 5-9 recover
completely (final SCIF 748.6mT, within 0.3% of the 746.2mT single-threaded
reference). **The earlier dt=30s "fully chaotic" characterization is
cold-start/low-current(I=19.6A)-specific, not a universal dt=30s
failure** — a warm-started, higher-current mid-ramp crossing produces a
milder, self-correcting response instead. This does NOT mean dt=30s is
safe in general (a poor/cold entry into dt=30s, or many consecutive
dt=30s steps, is untested) — it means the boundary depends on the state a
step starts from, not on dt alone. See
`nondeterminism_investigation_2026-08-05.md`'s second 2026-08-07 entry
for the full per-step table and reasoning.

**First validation of the NI radial-current closure itself (`ni_circuit.py`)
under the alpha fix: clean, at modest scope.** Every result above used the
INSULATED limit only — `circuit.update()` was never called and
`per_turn_bc=True` (required for the closure) was never combined with the
alpha fix until now. `transient/validation/ni_closure_smoke_check.py`
(new) ran 3 forced-full-length steps at the known-clean `dt=60s`,
I=19.6/39.2/58.8A (deliberately NOT `tparams.py`'s own default ramp
schedule, which uses `dt=25s`/`16.7s` — below the validated floor even
before adding the closure). **Result, checked on raw `dB_rel`**: warmup
phase (`circuit.freeze()`, insulated-equivalent) reproduces the
insulated-limit reference almost exactly (0.069/0.033/0.024 vs.
0.069/0.038/0.023); closure phase (`circuit.update()`, the actual radial
coupling) is slightly higher but still clean (0.080/0.047/0.033, well
inside the established 0.02-0.08 clean band). SCIF is genuinely ~2x
higher than the insulated case at the same steps — expected, physically
real (radial redistribution changing the azimuthal current), not a bug
signature. Zero clipping; `I_r_mean` (2.8→3.4A) lands close to Phase A's
independently-validated ~3.4A reference. **Narrow scope**: one run,
single-threaded, current only to 58.8A (30% of design), 3 steps, `dt=60s`
only. **Not yet tested**: full design current (196A), a full production
ramp, multi-threaded reliability, and the closure's behavior at short
`dt` (the already-known-hard problem for the insulated case) — none of
these should be assumed clean on this one result.

---

## Ramp-up power analysis (2026-08-08, `circuit/power_ramp.py` + `transient/validation/ta_quench_margin_check.py`)

**Question:** for a supply that delivers constant POWER (not a
prescribed current schedule, `P = I·V`), what's the fastest safe ramp to
`I_design` without approaching quench?

**Phase A (DCN, `circuit/power_ramp.py`): finds no ramp-speed-dependent
quench constraint at all.** `DCN.terminal_voltage()` is exact and linear
in `I` at fixed turn-current state, so constant power is a plain
quadratic in `I` solved algebraically at every ODE step — no need for
the naive, `I=0`-singular `P=L·I·dI/dt` formula, and no need to touch
`transient/`'s short-dt-fragile solver for the control law itself. Swept
across rho_c ∈ {30,100,400} µΩ·cm² and P from ~2W to >4000W (ramp times
~1hr down to sub-second): the per-turn-group margin is IDENTICALLY the
DC steady-state floor (2.225) at every tested P — the transient state is
always safer than the final DC state (no overshoot). **This means P is
effectively unbounded by DCN's own physics** — expected, since DCN only
resolves inter-turn (radial-leakage) sharing, not intra-tape screening
current concentration, so it cannot see the mechanism that could
actually bound P.

**T-A cross-check (required before trusting the above): finds a real
gap between T-A and the uniform-J assumption, present at ANY ramp
speed — not a fast-ramp-specific effect.** Comparing per-cell
`Jc(B,θ)=Ic(B,θ)/(δ_SC·w)` (the same normalisation `ta_solve.py`'s own
Picard solver uses) against the in-plane local current density at three
points — uniform-J, the production T-A solve at I_design/dt=600s, and
the final step of the project's first genuine multi-step ramp
(dt=60s×10, same 600s total span):

| | worst-cell margin | cells < 1.0 | cells < 1.538 (design threshold) |
|---|---|---|---|
| uniform-J | 1.600 | 0/4463 | 0/4463 |
| T-A, dt=600s (1 step) | 0.709 | 1200/4463 (27%) | 2054/4463 (46%) |
| T-A, dt=60s×10 (step 9) | 0.687 | 1500/4463 (34%) | 2153/4463 (48%) |

Both T-A methods agree on WHERE the worst cell is (the coil's own
peak-field cell, ~26.5mm radius, |B|~11.3T) despite coming from two
different code paths — a real, robust feature. The genuinely
time-marched fast ramp trends slightly worse (+3.1%) than the slow
single-step reference, not better, though this sits near this project's
own already-documented ~2% unresolved harness-vs-production SCIF
discrepancy (see "NI transient work" above), so read it as "does not
improve on the gap" rather than a cleanly-confirmed ramp-rate effect.

**Interpretation caveat, deliberately left open (per user direction —
recorded, not escalated, this session):** this project's quench
criterion everywhere is the E_c=1µV/cm engineering Ic definition. Some
cells at/above that threshold near a flux-penetration front is NORMAL
Bean/critical-state behaviour — `ta_solve.py`'s own solver assigns a
finite, not infinite, critical-state resistivity there as standard
operation (`eps_reg`), not an error condition. Whether "27-34% of cells
locally over Ic, even at the slow reference ramp" is a real,
previously-undetected safety gap in the champion design, or expected
critical-state physics the uniform-J-based 65%-of-Ic margin was never
built to catch, is an OPEN judgement call outside this session's scope.

**Practical consequence:** DCN alone cannot answer the ramp-power
question — its physics has nothing to say about the actual candidate
binding constraint. No "fastest safe P" number is recommended from this
session; that requires resolving the interpretation caveat above first.
Full evidence and reasoning: `docs/HISTORY.md`'s 2026-08-08 entry.

Run:
```bash
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
$PY circuit/power_ramp.py
$PY transient/validation/ta_quench_margin_check.py
```

### 2026-09-02/03: search for a geometry that closes the T-A-vs-uniform-J
margin gap directly — overnight effort, strong negative signal, not yet
conclusive

Rather than leave the interpretation caveat above open indefinitely, a
CMA-ES search (`optimize/studies/ta_safe_margin_search.py`,
`optimize/ta_safe_current.py`) was built to attack the gap at its
SOURCE: instead of the uniform-J `I_op = I_quench/SAFETY_FACTOR`
approximation every other search in this project uses,
`ta_safe_current.evaluate()` finds the largest current at which a
genuine T-A solve's worst-cell local margin, `Jc(B,θ)/|J_inplane|`,
stays ≥ 1.5385 (65% of local Ic) **everywhere in the coil**, then reports
`B_target_T` at THAT current — so a candidate needing a big derating to
be locally safe shows up directly as a field shortfall in the fitness
function. Expensive by construction (~70-160s/eval: a full mesh build
plus several warm-started T-A Picard solves), so run as a wide,
warm-started-at-the-champion CMA-ES search, not a from-scratch one.

**A smoke test on the champion's own exact geometry found the gap much
larger than the uniform-J-vs-T-A margin ratio (0.709 vs 1.5385, ~2.2x)
suggested:** the champion's T-A-safe current is only ~17.3A → 0.92T, an
~11x field shortfall, not a ~2x one. Scoped as a wide search accordingly
(full `a`/`b`/`coil_half_gap`/turn-pair bounds, large step sizes, still
warm-started at the champion).

**Ran overnight across 21 separate process launches** (`run_tag`s
`ta_safe_20260902_230407` through `ta_safe_20260903_174242`, archived per
generation in `optimize/runs/ta_safe_margin/gen*_archive/`) —
**restarting this many times is itself informative**: each restart
followed a generation that hit the 900s per-worker timeout (a
non-converging T-A solve at some candidate geometry, not a crash), and
the final launch ended when the harness session hosting it was
terminated (see "Operational lessons" below on why a harness-managed
background run cannot be paused, only restarted from its last flush).
**Across ~1247 T-A-evaluated candidates with a valid `I_op` reading,
spanning `a` from ~42mm to ~55mm, `b` from ~49mm to ~73mm, and turn
splits from near-uniform to sharply tapered — a far wider region than
the champion's own 26.0/31.4mm — the T-A-safe operating current stayed
confined to 5.0-54.2A (median 35.0A), and zero of them satisfied
`all_constraints_ok`.** The best `B_target_T` found anywhere in the
entire run was **4.10 T** (`a`=49.4mm, `b`=54.6mm,
`n_turns=[1032,1032,153,153,442,442]`, `I_op`=52.5A), less than half the
10T floor.

**Reading this, deliberately not overclaimed:** a search finding nothing
after ~1200+ real (not proxy) T-A evaluations spanning a genuinely wide
region is a MUCH stronger negative signal than "the search ran out of
budget" — the near-flat 5-54A ceiling on `I_op` across wildly different
`a`/`b`/turn distributions is the standout feature, and it did not
respond to any geometry lever this search varied. That specifically
raises (but does not confirm — no run has yet tested it) the possibility
that the ceiling is closer to a structural property of the current
6-layer double-pancake topology (e.g. a self-field effect from having
only 6 z-layers concentrate all the transport current) than a
geometry-tuning problem this search's variables can reach. **Per this
reasoning, `optimize/opt_config.py` gained a `CMAES_X0_JSON_OVERRIDE`
env-var hook (2026-09-03) to let a future run vary `N_LAYERS` itself**
(spreading the same ampere-turns across more, thinner layers) — an
`optimize/runs/ta_safe_margin_8layer/` variant was started on this basis
but only reached 2 generations before the effort moved on to other
work, so this hypothesis is queued, not tested.

**Practical consequence: do not treat "closing the T-A-vs-uniform-J
margin gap" as a matter of re-running this search longer or with a
bigger budget** — the evidence so far points at a structural limit, not
an unexplored corner of `a`/`b`/`gap`/turn-split space. The interpretation
caveat two sections up remains open, and this result makes it MORE
consequential, not less: if the 65%-of-local-Ic target genuinely cannot
be met anywhere near 10T within this topology, the resolution is either
(a) the critical-state-physics reading of the caveat is correct and the
uniform-J-based margin was simply never the right check, or (b) the
champion's real safety margin is as tight as `ta_quench_margin_check.py`
found and a design change (more layers, a different topology, or a
relaxed/re-justified target) is needed — this session did not determine
which.

Run (long — launch via the direct python binary, see "Operational
lessons" below):
```bash
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
$PY optimize/studies/ta_safe_margin_search.py
```

---

## Cryogenic (4.2K) operating-temperature investigation (2026-09-12, continued same day by Claude Code) — RESOLVED: 4.2K gives only ~9% more T-A-safe current, not enough to close issues #6/#7's gap. Mesh convergence on the 20K baseline itself is still an open loose end.

**Motivation:** issues #6/#7 above (the T-A-vs-uniform-J margin gap) leave
the champion's real safety margin either genuinely tight or a benign
critical-state artifact, unresolved. Since this project's own Ic/n-value
data only goes down to 15K, and REBCO Jc/n both rise substantially at
liquid-helium temperature, operating at 4.2K was proposed as a way to
close the gap directly rather than resolve the interpretation question.
**This session could not produce a trustworthy answer, in either
direction** — not "4.2K doesn't help," but "every attempt to check hit a
solver-reliability wall before producing a number worth trusting." Full
arc below, kept in the same warts-and-all style as the `transient/`
sections above because the failure mode rhymes with that investigation.

**Data sourcing, in order of what was tried:**
1. A candidate 4.2K Ic(B,θ) image (sample SP06-M3-609-1-MS) was
   **REJECTED**: at the same B/θ/T its absolute Ic (~800 A/4mm at 4.2K,
   1T, θ=0°) was LOWER than this project's own Shanghai tape at the
   *warmer* 20K (1185 A/4mm) — a physical impossibility for the same
   tape, proving only that it's a different, weaker conductor. Absolute
   values from an unrelated tape cannot be transplanted onto this
   project's model.
2. **Tsuchiya, Nojima, Zampa, Mizuno, Kohama, Awaji, "High-Field Angular
   Ic Metrology for REBCO Tapes: Combined Pulsed-Current and Rotation
   Techniques," IEEE Trans. Appl. Supercond. 36(5), Art. 8001107, Aug
   2026, DOI 10.1109/TASC.2026.3667136** — a genuine multi-temperature
   (4.2-77K) Ic(B,θ) characterization of Fujikura FESC-SCH04(40) tape up
   to 24T (`physics/digitized_IC_data/High-Field_Angular_Ic_Metrology_*.pdf`).
   Cross-check against this project's own measured data: Fujikura
   20K/5T/θ=90° digitizes to ~1827 A/4mm vs. Shanghai's own MEASURED 1728
   A/4mm — 6% apart, i.e. comparable-class modern 4mm 2G HTS tape (unlike
   SP06 above). The paper's own Fig. 4 (8-panel Ic(θ) grid, 4.2-77.3K,
   fields to 24T) was hand-digitized by the user into
   `physics/digitized_IC_data/{5,10,15,20,24}T_{4p2K,20K}.csv` (plain
   `theta_deg,Ic_A_per_4mm` point lists).
3. **`Fujikura4p2KScaledIcModel`** (`optimize/ic_temperature_scaling.py`)
   borrows the Ic(4.2K)/Ic(20K) RATIO from those digitized points — never
   Fujikura's absolute Ic — and applies it multiplicatively to this
   project's own measured 20K Shanghai grid (itself already
   Kim-extrapolated above 8T by `ic_extrapolation.py`). Ratio found:
   **1.8-3.0x across the (B,θ) grid sampled, median ~2.2x** — notably
   smaller than an initial eyeball estimate straight off the source
   image (5-8x), which turned out to be a misread of the 20K panel's
   axis scale; the digitized-CSV number is the one to trust. Bounded
   flat extrapolation below 5T / above 24T and outside the digitized
   theta range (never extrapolates the ratio itself, only clamps the
   query point) — same discipline as `ic_extrapolation.py`.

**First finding: naive Ic-only scaling made local margins WORSE, not
better — a real modeling gap, not proof against 4.2K.** Fixed-current
check on the champion (I=25A, medium mesh, cold start): scaling only
Ic to 4.2K while leaving n(B,θ) at its 20K measured value dropped the
worst-cell margin from 20K's 1.16 to **0.78** — worse. Root cause,
verified by inspecting `ta_solve.py`'s `_update_rho()`: the T-A local
resistivity is ρ=(E_c/Jc)(J/Jc)^(n-1), and the self-consistent screening-
current distribution depends on Jc and n *together* — mixing 4.2K's
higher Jc with 20K's lower (softer) n is a physically inconsistent
hybrid the coupled nonlinear solve is sensitive to, not evidence about
real 4.2K physics.

**Fix attempt: digitize a properly tape-matched n(B,T), from the actual
manufacturer datasheet.** The user located
**Fujikura Ltd., "Introduction of Fujikura RE-based High Temperature
Superconductor" (Superconductor Business Division, Rev. AUG2025)**
(`physics/digitized_IC_data/Fujikura_superconductor_EN.pdf`) — page 14
has an n-value-B-T chart for the **AP (FESC) series** (artificial
pinning, 2.4µm SC layer, Ic≈200A@77K self-field) — the tape family that
actually matches the IEEE paper's FESC-SCH04(40) sample, **not** the
Non-AP (FYSC, 1.9µm layer, no artificial pinning) chart on pages 12-13
the user first screenshotted from the same PDF. AP vs Non-AP have
measurably different flux-pinning landscapes (that's the whole point of
adding pinning centers), so this distinction mattered.
Digitized (pymupdf render + per-curve color-classification pixel
extraction) into `physics/digitized_IC_data/nvalue_Bc_{4p2K,10K,20K,30K,
40K,50K,65K,77p3K}.csv`. **Hit and fixed one real extraction artifact
worth remembering for any future re-digitization from this kind of
chart:** each curve's own text label (e.g. "4.2K") is printed in the
same color as the curve, positioned right at the plot's edge — naive
whole-image color-matching scoops up the label text as fake data,
corrupting the high-field tail. Fixed by restricting extraction to an
interior window (B=0.75-17.5T) plus a median-based despike pass; the
cleaned data is smooth and monotonic in the expected direction (higher T
→ lower n, 77.3K flattening at n≈1 near B≈14.5T matching the chart's own
floor). Resulting ratio n(4.2K)/n(20K) at B//c: **~2.25-3.0x** across
0.75-17.5T. Wired into **`Fujikura4p2KScaledNValueModel`** (same module,
same bounded-flat-extrapolation policy) — applied **isotropically** (no
AP-type B//ab n-value chart exists in this datasheet), a disclosed
simplification, not a validated angular n(T) model.

**Second finding: matching n did NOT fix it — margins got worse again,
and a new red flag appeared.** Same fixed-current check (I=25A, medium
mesh), now with both Ic and n properly scaled together:

| config | Picard iters | worst margin | p5 margin | median margin |
|---|---|---|---|---|
| 20K (baseline) | 73 | 1.16 | 1.69 | 28.5 |
| 4.2K, Ic only (mismatched) | 102 | 0.81 | 1.63 | 2.9 |
| 4.2K, Ic + n (matched) | 147 | **0.50** | 1.62 | 3.1 |

Iteration count climbed monotonically with each added realism. Root
cause: **this project's own measured n-value data spans only n=13-34**
(the range the T-A Picard scheme's relaxation parameters,
`ta_picard_alpha`/`ta_picard_alpha_fine`, and its EMA stall criterion
were tuned and validated against). The Fujikura ratio pushes n to
**~50-60** at 4.2K — a much stiffer E-J power law than the solver has
ever been checked at, in this project or the one the ratio was borrowed
from.

**Fine-mesh re-check (params.py "fine" tier, ~70k DOF vs medium's ~48k)
using the real bisection-based `ta_safe_current.evaluate()`** surfaced
two further, independent reliability problems rather than resolving
anything:
- **The 20K baseline itself is not mesh-converged.** I_op/B_target went
  from 25.4A/1.36T (medium) to **43.3A/2.31T** (fine) — a 70% swing from
  mesh refinement alone, on the well-validated n=13-34 regime. Neither
  number should be trusted as final; this is the same "never trust a
  single-mesh number near a constraint boundary" lesson this file
  already states elsewhere, now confirmed for `ta_safe_current.py`
  specifically (it had not been mesh-convergence-checked before).
- **4.2K matched (Ic+n), fine mesh: I_op=18.5A, B_target=0.98T — but the
  raw Picard residual never actually settled.** At k=63, when the
  EMA-based stall criterion declared "converged," `|ΔB|/|B|` was still
  oscillating between 1.4 and 2.8 (140-280%), not decaying — the
  identical false-stall signature the `transient/` short-dt investigation
  already caught and documented (see above): quiet enough envelope to
  fool the stall detector, not actually converged in any residual sense.
- **4.2K, Ic-only (n at 20K), fine mesh, via bisection: a degenerate,
  non-physical result caused by the bisection algorithm's fixed
  5-iteration budget, not the physics.** The bisection tested
  137.2→71.1→38.0→21.5→13.3→9.1A (halving from an I_hi bracket inflated
  by the boosted Jc), found none satisfied the margin target, exhausted
  its iteration budget before reaching anywhere near the 5A floor, and
  fell back to evaluate()'s literal placeholder
  (`I_op_A=5.0`, `margin=1.53846...`=1/0.65 exactly) — **not a
  measurement.** `ta_safe_current.py`'s `MAX_BISECT=5` was tuned around
  20K's own scale, where the true crossing sits close enough to I_hi for
  5 halvings to resolve it; a substantially boosted Jc inflates I_hi far
  past where 5 halvings can reach the answer.
- **Direct fixed-current follow-up (no bisection) at 4.2K Ic-only, fine
  mesh, I∈{4,6,8,10}A: still no value satisfies the target, and the
  results themselves are internally inconsistent.** p5 margin stayed
  below the 1.5385 target at every current tested (best: 1.41 at 4A).
  Worst-cell margin was **non-monotonic** with current — 0.38 → 0.11 →
  0.23 → 0.58 as I dropped 10→8→6→4A — and I=10A failed to converge
  within the 150-iteration cap. This matches this project's own
  documented "the flux front wanders chaotically among near-degenerate
  states" finding (see "Convergence criterion" under "The T-A
  formulation" above), not a clean physical trend a smooth current sweep
  should show.

**Conclusion, deliberately left open:** every 4.2K variant tried
(Ic-only mismatched, Ic+n matched; medium and fine mesh; bisection and
direct fixed-current) has shown a solver-reliability red flag
(inconsistent-physics mismatch, false stall, bisection-budget exhaustion,
non-monotonic/non-converged results), a failure to beat the 20K
baseline, or both. **No test in this session produced a number worth
trusting, in either direction.** This is not evidence that cryogenic
operation doesn't help the real coil — it is evidence that
`ta_safe_current.py`'s Picard relaxation parameters, stall criterion, and
fixed bisection budget were tuned for this project's own historical
operating regime (n=13-34, currents where the uniform-J bracket lands
close to the true answer) and have not been validated outside it.
Getting a trustworthy answer needs the same class of work already done
once for the `transient/` short-dt problem — forced full-length runs
that bypass the EMA stall flag and check the raw residual directly, and
likely a retuned (smaller) relaxation pair for the much stiffer n~50-60
regime — not another one-off script. **Not yet done**, and the natural
next step if this path continues.

New files from this session: `optimize/ic_temperature_scaling.py`
(`Fujikura4p2KScaledIcModel`, `Fujikura4p2KScaledNValueModel`);
`physics/digitized_IC_data/` (digitized CSVs plus both source PDFs, kept
for provenance and any future re-digitization).

### 2026-09-12 (continued, same day, by Claude Code): the solver-reliability
wall above was root-caused and fixed the same way the `transient/`
short-dt problem was — a forced-full-length validation matrix, not
another one-off script. Result: 4.2K genuinely does not fix issues #6/#7.

Built `optimize/studies/ta_thermal_validation.py` (a systematic matrix,
not a one-off) plus two small, additive, backward-compatible instrumentation
hooks needed to run it: `ta_solve.solve_ta_at_current()` gained
`min_iters` (forces a full-length run past the point the EMA-smoothed-SCIF
stall flag would normally declare "converged" — default value 25
reproduces every existing caller's behaviour exactly) and `diag_log_path`
(per-iteration raw-residual CSV, opt-in, no effect when omitted). All
results: `optimize/runs/ta_thermal_validation/summary.csv` (35 rows, zero
errors) + per-solve raw traces in `.../diag/*.csv`.

**Phase A/B — z-resolution AND in-plane resolution convergence check on
the 20K baseline (fixed I=25A, cold start, forced to the full 150
iterations), isolating each axis separately:**

| mesh config | 20K worst-margin | 20K raw \|ΔB\|/\|B\| | wall/solve |
|---|---|---|---|
| medium in-plane, medium5 z-grading (production default) | 1.162 | 0.0103 | 66s |
| medium in-plane, uniform5 z (ungraded, nz=5) | 1.295 | 1.056 | 66s |
| medium in-plane, xdense7 z-grading | 1.025 | 0.036 | 91s |
| **fine** in-plane, medium5 z-grading | **0.925** | 0.127 | 435s |

Two findings, both new:
1. z-grading choice materially changes both the raw residual and the
   worst-margin at FIXED in-plane resolution — the production `medium5`
   grading is in fact the best-conditioned of the three z-configs tried
   (lowest residual by ~3-100x), not an arbitrary historical choice.
2. **This file's own claim that "in-plane refinement ... converges
   quickly and is not the bottleneck" (see "Mesh resolution" under "SCIF
   computation" above) does NOT hold for `ta_safe_current.py`'s local
   margin check.** Fine in-plane (holding z-grading fixed) swings the
   20K worst-margin by ~20% (1.162→0.925) and, notably, makes the raw
   residual WORSE (0.0103→0.127), not better — i.e. none of the four
   configs tried reaches genuine raw-residual convergence at 150 forced
   iterations, `medium/medium5` is merely the least-bad of the four. This
   independently confirms (via a DIFFERENT method — direct fixed-current,
   not bisection) the 2026-09-12 fine-mesh finding above that
   `ta_safe_current.py`'s 20K baseline is not mesh-converged, and extends
   it: the non-convergence is present even isolating in-plane resolution
   alone, and even the historically-reported "converged" 20K numbers in
   this investigation were most likely EMA-stall false positives, the
   same failure mode already caught for the 4.2K variants. **This remains
   an open loose end** — no Richardson-style extrapolation across 3+
   successively finer meshes was done (fine in-plane took 435-495s/solve
   vs medium's 66-90s; the committed-on-disk `xdense` in-plane tier,
   confirmed still live in `params.py` this session — see below — would
   be the natural next data point but is ~20-30x medium's cost and was
   not run here).

**Phase B (same configs, 4.2K variants, DEFAULT alpha=(0.30,0.15)):**
confirms the physically-inconsistent-mismatch finding above independent
of mesh choice — 4.2K Ic-only worst-margin 0.55-0.83 and 4.2K Ic+n-matched
worst-margin 0.68-0.83 across all four mesh configs, raw residual
1.7-1.98 (never converging) in every single one. Mesh choice does not
rescue the DEFAULT-alpha failure; the alpha itself was the problem (next).

**Phase C — relaxation-pair sweep on the 4.2K Ic+n-matched model, at the
best-converged mesh (`medium`/`medium5`), fixed I=25A, forced full length:**

| alpha (high, low) | raw \|ΔB\|/\|B\| | worst-margin |
|---|---|---|
| (0.30, 0.15) — production default | 1.828 | 0.767 |
| (0.15, 0.08) | 0.0236 | 1.243 |
| (0.10, 0.05) | 0.0241 | 1.245 |
| (0.05, 0.02) | 0.0173 | 1.310 |
| **(0.03, 0.01)** — the SAME pair the `transient/` short-dt investigation found | **0.0138** | **1.981** |

A ~2x-to-~130x drop in raw residual and a real, physical worst-margin
recovery — this is the same class of fix as the `transient/` short-dt
relaxation-parameter root cause (default alpha provides ~zero effective
damping once n leaves the historically-validated 13-34 range; a ~10x
smaller pair fixes it here too, independently confirming that finding's
generality). **Caveat, honestly flagged rather than smoothed over:**
worst-margin is still trending upward as alpha shrinks (0.77→1.24→1.25→
1.31→1.98), not clearly plateaued at 150 forced iterations — smaller
alpha needs proportionally more iterations to reach the same depth of
convergence, so `(0.03, 0.01)`'s lower residual may mean "still slowly
converging" rather than "has reached the fixed point already." Treat the
Phase D numbers below as good-enough-to-be decisive on the HEADLINE
question (is 4.2K a big win — no), not as a final, arbitrarily-precise
answer.

**Phase D — direct fixed-current sweep (bypasses `ta_safe_current.py`'s
own `MAX_BISECT` entirely, sidestepping the exhaustion bug below) at
`alpha=(0.03,0.01)`, `medium`/`medium5` mesh, I ∈ {5,8,10,15,20,25,30,35,40}A,
each forced to 150 iterations:**

Both curves are clean and monotonic (no repeat of the fine-mesh
non-monotonic 4.2K-Ic-only finding above — that pathology does not
reproduce once alpha is fixed), raw residual 0.010-0.048 throughout —
genuinely, substantially better converged than anything in this
investigation before today.

- **20K**: crosses the MARGIN_REQUIRED=1.5385 threshold at **I_op ≈
  17.7A** (linearly interpolated between the 15A/1.785 and 20A/1.331
  points) — matches the champion's own previously-reported ~17.3A T-A-safe
  current (from the 2026-09-02 smoke test, different alpha/mesh) to
  within 3%, a strong independent cross-check that this harness is
  measuring the same real quantity.
- **4.2K, Ic+n-matched**: crosses the same threshold at **I_op ≈ 19.3A**
  (between 15A/1.990 and 20A/1.466).
- **Ratio: 19.3/17.7 ≈ 1.09 — only a ~9% improvement in T-A-safe
  operating current from 4.2K operation**, despite the underlying
  Ic/n ratios being ~2.2-3x. B_target scales roughly linearly with I_op
  in this low-current regime (screening current is a small correction
  relative to transport current well below I_design), so this implies
  roughly 0.92T → ~1.0T, nowhere close to closing the ~11x gap to 10T
  that issues #6/#7 identified.

**Answer to known-open-issue #8, replacing "genuinely unanswered":**
**4.2K operation does not meaningfully help.** Once the solver-reliability
wall is actually fixed (not worked around), the honest comparison shows
cryogenic operation buys single-digit-percent headroom on the T-A-safe
current, not the order-of-magnitude the champion's field-margin gap
(issues #6/#7) would need. The interpretation question those issues
raise — whether the uniform-J-based 65%-of-Ic target was ever the right
check, versus the champion's real quench margin being genuinely tight —
is **still open**, but temperature is no longer a candidate lever to
close it with.

**Separately, found and fixed while investigating this: `optimize/
ta_safe_current.py`'s `MAX_BISECT=5` could silently return a fabricated
placeholder instead of a real measurement.** With `I_lo=I_FLOOR_A=5A` and
`I_hi` inflated by a boosted Jc (up to `cfg.I_MAX_SEARCH_A`=1500A), 5
halvings cannot resolve a 3A tolerance across that range (needs up to 9).
The old fallback then reported `I_op_A=I_FLOOR_A, margin=1/0.65` exactly
— indistinguishable from a genuine measurement — without ever actually
solving at that current. Fixed: `MAX_BISECT` raised to 12 (covers the
full possible bracket with margin), and the fallback now does one real
confirming solve at `I_FLOOR_A` and reports its ACTUAL margin, tagged via
a new `bisection_exhausted` field. **Not empirically re-triggered by this
session's own work** (Phase D bypassed the bisection by design) — the fix
is code-review/reasoning-validated, not exercised against a live
budget-exhaustion case.

**Also surfaced, unrelated to the fix above but relevant to trusting ANY
current search result: `params.py` on disk is still the 2026-09-03/04
one-off "xdense" mesh-artifact-investigation tier (confirmed live via
`mesh_size_min_factor=0.05` on 2026-09-12), not the "medium" production
default this file's own "Mesh resolution" section says to revert to.**
`git status` on 2026-09-12 shows `params.py` as UNMODIFIED relative to
HEAD — i.e. xdense is the currently-COMMITTED state, not merely an
uncommitted local edit left over from the investigation. Every script in
this investigation (and `optimize/studies/ta_safe_margin_search.py`,
independently, per its own 2026-09-08 comment) works around this
in-memory rather than relying on the file; any NEW script that reads
`params.py`'s mesh settings directly (without such an override) will
silently inherit the ~20-30x-slower xdense tier. Whether to actually
commit a revert to medium is a decision left to the user — not done by
this session, since another investigation's state may still depend on
the file as committed.

New files from this session: `optimize/studies/ta_thermal_validation.py`,
`optimize/studies/summarize_ta_thermal_validation.py`,
`optimize/runs/ta_thermal_validation/` (summary.csv + per-solve diag
CSVs). Modified: `solve/ta_solve.py` (additive `min_iters`/
`diag_log_path` kwargs on `solve_ta_at_current()`, default behaviour
unchanged), `optimize/ta_safe_current.py` (bisection budget/fallback fix
above).

### 2026-09-13: sanity-checked the ~9% number against the source paper's
own low-field data — confirms it, doesn't undermine it

User pushback (reasonable): real REBCO magnets DO reach 10T+ (some well
beyond, e.g. 25-45T), so if 4.2K genuinely only buys ~9% here, something
else must differ — worth checking the ~9% wasn't an artifact of
`Fujikura4p2KScaledIcModel`'s conservative flat-hold below its digitized
floor (B_ratio_lo=5T), since the champion's own T-A-safe operating point
(I≈17.7-19.3A) sits at peak coil field ≈1.1-1.2T — entirely inside that
flat-held, never-actually-measured region (confirmed: `clip_frac=1.0` for
EVERY 4.2K point in the Phase D sweep, including up to I=40A/peak
B≈2.4T). The worry: if the true low-field ratio is LARGER than the
flat-held 5T value, the 9% number understates the real benefit.

**Checked directly against the source paper's own physics, not another
guess.** Tsuchiya et al. Section III-B / Table II gives a generalized
p-q scaling-law fit, `Ic(B) = Ic0(1+B/B0)^-p (1-B/Birr)^q`, fit to the
FULL measured field-dependence curve (Fig. 3), from near self-field
through 24T+ — a fundamentally better-grounded source for the low-field
region than the 5-slice angular dataset (`{5,10,15,20,24}T_*.csv`) this
project digitized for the Ic MODEL itself. At θ=0° (B⊥tape):

| B (T) | Ic₄.₂K (A) | Ic₂₀K (A) | ratio |
|---|---|---|---|
| 0 (self-field) | 3050 | 1800 | 1.69 |
| 1.0-1.2 (champion's own T-A-safe operating field) | 2140-2240 | 1244-1307 | 1.71-1.72 |
| 5 (this project's flat-hold floor) | 1194 | 680 | 1.76 |
| 24 | 365 | 178 | 2.06 |

**The ratio INCREASES with B, not decreases — the opposite of the
worried-about direction.** The true ratio at the champion's actual
operating field (~1.1-1.2T) is ~1.71-1.72, slightly BELOW the flat-held
5T value (~1.76) this project's model used for that entire region — i.e.
the flat-hold was mildly GENEROUS to 4.2K, not conservative against it.
The paper's own text independently corroborates this: "Ic at self-fields
at 77K and 20K differs by roughly one order of magnitude, and that at
4.2K is about 1.5 times larger [than 20K]" — a self-field ratio in the
same ~1.5-1.7x ballpark, well below the ~2.2x median this project's
digitized 5-24T angular ratio grid gives. **Conclusion stands, now on
firmer ground: the ~9% T-A-safe-current improvement from 4.2K is not an
extrapolation artifact.** Physically, this matches ordinary REBCO Jc(T)
behaviour: Jc(T) is comparatively flat well below Tc/2 (~46K for
Tc≈92K), so most of the temperature-scaling "gain" happens between
~40-77K, not between 4.2K and 20K — both already deep in the flat part
of the curve.

**So what ARE real 10T+ (and 25-45T) REBCO magnets doing differently, if
not mainly temperature?** Not exhaustively checked this session, but the
same paper's own reference list points at one directly: citation [30],
"First validation of robust REBCO insert concept on a large **20-pancake**
prototype reaching up to 25 T" — 20 pancakes (40 layers) for a 25T
insert, vs. this project's **6 layers** (3 double-pancake pairs).
Spreading the same ampere-turns across many more, thinner pancakes
reduces the self-field/screening-current concentration PER PANCAKE — this
is exactly the mechanism the 2026-09-02/03 `ta_safe_margin_search.py`
overnight search flagged (without confirming) as a possible structural
property of the 6-layer topology, and exactly why an `N_LAYERS`-varying
follow-up was queued (`CMAES_X0_JSON_OVERRIDE`, the `8layer` variant,
only 2 generations run) but not pursued at scale. Other real, plausible
differences not investigated this session: striated/filamented tape
(reduces screening-current magnitude directly — this project assumes
unstriated 4mm tape), hybrid designs with a background field from a
separate (LTS or resistive) magnet rather than one winding generating the
entire target field, and a quench/margin QUALIFICATION criterion based
on actual thermal-runaway analysis rather than this project's static
"no T-A-resolved cell exceeds 65% of local Ic" test (the still-open
critical-state-physics interpretation question in issues #6/#7). The
`N_LAYERS` lever is the best-evidenced next step of these — it's already
partially built and directly supported by a real published high-field
design, not a new hypothesis.

---

## Operational lessons (env quirks and process gotchas)

- **`conda run` buffers ALL subprocess stdout until the process exits**,
  regardless of `sys.stdout.reconfigure(line_buffering=True)` inside the
  script or any of `conda run`'s own `--no-capture-output`/`-s`/
  `--live-stream` flags. Launch any long-running script via the
  environment's **direct python binary**
  (`/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3`) instead
  of `conda run -n fenicsx-env python3 ...` if you need to check on it
  mid-run — essentially always true for multi-hour searches/solves.
- **Flush results incrementally.** Any long-running search/solve script
  should write its CSV/results periodically (see `cmaes_search.py`'s
  `_record()`, `FLUSH_EVERY=20`), not only once at the end — a crash or
  kill partway through a multi-hour run must not lose everything.
- **gmsh mesh generation is not reproducible across separate OS
  processes** (only within one process). Never trust a single
  cross-process uniformity/field number near a constraint boundary —
  always do an independent-mesh repeat before promoting a design.
- **`SIGSTOP`/`SIGCONT` is safe** to pause a shell-launched (`nohup`/
  `disown`) background run — confirmed zero progress lost across a 1.5h
  pause. **It is NOT safe on a Claude Code harness-managed background
  task** — `SIGSTOP` reports as a failed exit (147) and the harness reaps
  the entire process tree; there is no in-place pause for a
  harness-launched run, only finish-or-kill-and-restart-from-flushed-CSV.
- **Concurrent `cmaes_search.py` processes must not share output
  paths** — `CMAES_OUT_CSV`/`CMAES_OUT_LOG` are hardcoded defaults that
  get fully overwritten on every flush; two processes writing the same
  path silently clobber each other. Use the `CMAES_OUT_CSV_OVERRIDE`/
  `CMAES_OUT_LOG_OVERRIDE` env vars for any orchestrator launching
  concurrent jobs. The cumulative master log
  (`cmaes_all_evaluations.csv`) is append-only and tagged by `run_tag`,
  so it's unaffected either way and is the source of truth to recover
  from if this happens.
- **`circuit/run_charge.py` and `transient/run_charge.py` share a module
  name** in this flat, package-less repo (bare `sys.path` imports
  everywhere) — loading one can silently import the other. Anything
  needing both loads by explicit file path under distinct names.
- **Any `(1 − B/B_scale)`-type Ic extrapolation model needs its cutoff
  field bounded well above the measured data ceiling** — see "Ic(B)
  extrapolation" above; this project's dataset reliably breaks an
  unconstrained fit at angle 88°.
- **A step size that's "small" in absolute terms can still be huge
  relative to a design's own scale.** A fixed-mm CMA-ES step size used
  for a warm-started "polish" run at a compact (a≈15mm) design let the
  search escape its own basin into a worse one. Always set warm-start
  step sizes as a fraction of the warm-start value itself
  (`CMAES_A_STD0_OVERRIDE`/`CMAES_B_STD0_OVERRIDE`/`CMAES_N_STD0_OVERRIDE`
  in `opt_config.py`), and assert on the value the solver will actually
  use rather than trusting an env var was read correctly — this project
  shipped a step-size override that silently did nothing for a week
  because of an attribute-name typo, caught only by that assertion habit.

---

## Running the pipeline

```bash
# Single T-A solve at I_design
conda run -n fenicsx-env python3 solve/ta_solve.py

# Sweep 150-400 A (11 points, ~25 s solver time + plot generation)
conda run -n fenicsx-env python3 solve/ta_sweep.py

# Post-process saved .npz
conda run -n fenicsx-env python3 solve/ta_postprocess.py

# Baseline uniform-J solve (for comparison)
conda run -n fenicsx-env python3 solve/solve.py

# Quench analysis (uses v3 or v4 sweep results)
conda run -n fenicsx-env python3 sweep/quench_sweep.py
```

---

## Bugs fixed (history, for context)

1. **`petsc_options_prefix` missing** → added `"ta_T_"` and `"ta_A_"` prefixes
   to both `LinearProblem` constructors.
2. **T BCs at wrong z-boundaries** (PRIMARY bug) → BCs were at
   ±n_layers×w/2 instead of ±w/2. Fixed by restricting T to per-layer
   solves with the correct single-tape BC.
3. **eps_reg additive instead of floor, value=1e-4 instead of 1.0** →
   changed to `j_norm = max(|J|/Jc, eps_reg)` with eps_reg=1.0. Also
   switched to `exp((n-1)*log(j_norm))` to prevent float64 overflow.
4. **Picard stagnated in a period-2 limit cycle** → two-phase adaptive
   relaxation (see "The T-A formulation" above).
5. **J_uniform wrong by a factor of n_layers in SCIF computation** →
   fixed (see "SCIF computation" above).
6. **`domain.geometry.dofmap` deprecation** → changed to `dofmaps[0]`.
7. **`quench_results.csv` empty strings** in `quench_current_A` for
   never-quenched cells → guard with `.strip()` before `float()`.
8. **SCIF Biot-Savart missed 3 mirrored quadrants** (×4 under-report of
   ΔBz) → `dB_bore_from_dJ()` expands to all 8 mirror pieces.
9. **Cell volumes estimated as (mean NN spacing)³** (~×2.1 net SCIF
   error) → exact DG0 `ufl.CellVolume` (`ta["coil_vols"]`).
10. **`coil_cells_T` swept in adjacent-layer boundary cells** (the
    dominant SCIF bug, 2026-07-10) → strict nearest-layer-centre cell
    assignment.
11. **Coil 2 translated instead of mirrored** in several visualization
    scripts (`plot_3d.py`, `plot_fields.py`, `plot_field_poster.py`) —
    picture-only bug, no physics number was ever affected. Fixed with a
    shared `_mirror_z(z, g) = 2*g - z` helper.
12. **`CMAES_N_STD0_OVERRIDE` was inert for ~a week** — `opt_config.py`
    parsed it into `cfg.CMAES_N_STD0`, but `cmaes_search.py` read the
    nonexistent `CMAES_N_STD0_OVERRIDE` attribute via `getattr(..., None)`,
    silently falling back to the oversized cold-start default. Fixed;
    see "Operational lessons" above for the general habit this taught.
13. **`ta_safe_current.py`'s `MAX_BISECT=5` bisection budget could
    silently return a fabricated placeholder instead of a real
    measurement** when a boosted Jc (e.g. the 4.2K temperature-scaled
    model) inflated the search bracket past what 5 halvings can resolve
    — `I_op_A=I_FLOOR_A, margin=1/0.65` exactly, reported as if measured.
    Fixed 2026-09-12: `MAX_BISECT` raised to 12 (covers the full possible
    bracket up to `cfg.I_MAX_SEARCH_A`), and the exhausted-budget fallback
    now does one real confirming solve at `I_FLOOR_A` instead of
    fabricating a value, flagged via a new `bisection_exhausted` field.
    See "Cryogenic (4.2K) operating-temperature investigation" above.

---

## Confirmed working imports

```python
from dolfinx.io import gmsh as gmshio            # mesh loading
from physics.ic_model import IcModel, NValueModel # Ic and n models
from physics.current_source import (
    arc_length_xy, normal_xy, turn_index_xy,
    turn_index_xy_multilayer, layer_index_from_z,
    expand_to_full_domain)
from ta_solve import (setup_ta_problem, solve_ta_at_current, _J_from_T,
                      dB_bore_from_dJ)
# solve_ta_at_current returns (A_h, B_h, T_h, info);
# info = dict(n_iters, converged, rel_err)
from coil2_field import compute_both_coils_field_multilayer   # Biot-Savart
```

`angle_with_normal_deg` lives in `physics/ic_model.py` with signature
`(B_array, n_hat_array)`.

---

## Figure style

All visualization figures use a **dark theme**: `#111` figure background,
`#0d0d1a` axes background, `magma` colormap for field magnitudes,
`plasma` for screening current magnitudes, white labels, `#444` spine color.
Stats panels use monospace font in a `#1a1a2e` rounded box.

Output directory: `params.VIZ_DIR` (= `visualization/`).
T-A figures are prefixed `ta_`: `ta_field_top.png`, `ta_field_side.png`,
`ta_uniformity.png`, `ta_field_3d.png`, `ta_scif_3d.png`,
`ta_bore_vs_current.png`, `ta_scif_vs_current.png`.
CMA-ES search figures are prefixed `cmaes_`: `cmaes_convergence.png`,
`cmaes_constraints.png`, `cmaes_variables.png`, `cmaes_overview.png` (this
run only), and `cmaes_param_map.png` (cumulative across every run — see
"Configuration optimizer" above).

---

## Current status and next steps

**Done / current state:**
- T-A Picard solver converges cleanly and fast at the project's standard
  `dt=600s` operating point (see "The T-A formulation").
- The CMA-ES design search (`optimize/`), the manufacturing constraints,
  and the proxy-graveyard lesson are all settled, current practice — see
  the dedicated sections above.
- **Current champion (see "Current design" above) is the first design
  validated against a realistic Ic(B) extrapolation AND build tolerance**
  — 15/15 jitter samples pass both B_target and uniformity.
- NI transient Phase A (`circuit/`) is validated and gives the champion's
  tau/energy/loss numbers. Phase B (`transient/`) has a validated
  point-fix at dt=600s only; an apparent adaptive-marching fix was tried,
  initially looked promising, and was RETRACTED the same day after its
  own reference case failed to reproduce in isolation — see "NI transient
  work" above. A same-day follow-up REJECTED the half-domain
  (drop-x/y-symmetry) hypothesis for the short-dt problem on two
  independent lines of evidence, and substantially root-caused the
  cross-process nondeterminism to default multi-threading's
  floating-point non-associativity — but reframed it as load-bearing
  noise keeping a deterministically-failing chaotic map alive at ~40%
  success, not a bug to fix. See the "2026-08-05 (continued)" entry at
  the end of "NI transient work" above.
- **2026-08-06: the short-dt failure was root-caused (an ordinary
  relaxation-parameter instability, not irreducible chaos) and a fix —
  `alpha=(0.03, 0.01)`, a ~10x smaller two-phase relaxation pair — was
  found and, after a same-day correction (an initial "fixed" claim
  turned out to rest on premature-stopped runs) and a full accuracy-
  first re-validation, genuinely validated.** Works across `dt` in
  [60s, 600s] and `I` in [19.6, 196]A (tested), including a genuine
  multi-step ramp (the project's first) and multi-threaded reliability
  at the true convergence horizon (0.004% spread across 5 independent
  runs). Fails at `dt=30s` — a real, characterised boundary, not another
  premature-stop artifact. Real cost: ~700-1500 iterations, not the
  ~460 first assumed. See "NI transient work" above's final consolidated
  account and `transient/validation/nondeterminism_investigation_2026-08-05.md`
  for the complete evidence and reasoning history.
- **2026-08-07: multi-threaded execution of the full production ramp
  confirmed reliable (0.001-0.03% run-to-run agreement across two
  independent launches) but NOT yet quantitatively matched to the
  single-threaded reference (~2.2% systematic gap, unresolved, same
  shape as the existing harness-vs-production discrepancy). Also:
  a ramp deliberately crossing the dt=30s "fully chaotic" boundary
  mid-sequence, warm-started at higher current, did NOT reproduce that
  chaos** — the earlier characterization was specific to a cold start at
  I=19.6A; a warm-started crossing produces a milder, self-correcting
  transitional response and the ramp recovers fully. See the two
  2026-08-07 entries at the end of "NI transient work" above.
- **2026-09-12: a 4.2K operating-temperature model was built
  (`optimize/ic_temperature_scaling.py`) and tested; the initial pass
  produced no trustworthy result in either direction, but a same-day
  follow-up (forced-full-length validation matrix,
  `optimize/studies/ta_thermal_validation.py`) root-caused the
  solver-reliability wall and got a genuine answer: 4.2K gives only
  ~9% more T-A-safe operating current, not enough to close known-open-
  issues #6/#7's field-margin gap.** See "Cryogenic (4.2K)
  operating-temperature investigation" above and known-open-issue #8
  (now RESOLVED). The investigation also surfaced, as a byproduct, that
  `ta_safe_current.py` itself has never been mesh-convergence-checked
  (its 20K baseline swung 70% between medium and fine mesh via
  bisection, and independently confirmed non-converged even isolating
  in-plane resolution alone via direct fixed-current solves — still
  open) and that its fixed 5-iteration bisection budget could silently
  return a placeholder value instead of a real one when a candidate's Jc
  is far outside the scale it was tuned around (fixed, `MAX_BISECT=12`
  plus a real confirming solve on exhaustion — see "Bugs fixed" #13).

**Known open issues, in priority order:**
1. **Ic model uncertainty (~±0.5T on B_target)** — closable only by
   extending the measured Ic/n-value dataset above 8T. Until then, quote
   the champion as ~10.5T ± 0.5T, not a bare 10.49T.
2. **`transient/`'s multi-step convergence problem: RESOLVED at the
   relaxation-parameter level, and substantively validated — not fully
   general (`dt=30s` remains a real, characterised failure boundary).**
   Consolidated final account (this item accumulated many rounds of
   same-day correction on 2026-08-06 as the investigation deepened; the
   full honest arc — including two retracted "breakthroughs" — is
   preserved in `transient/validation/nondeterminism_investigation_2026-08-05.md`
   and `docs/HISTORY.md`, not repeated here):
   - **Earlier remedies that did NOT work**, tried across several
     sessions: Picard relaxation tuning (the wrong VALUES, not the
     wrong idea — see the actual fix below), a Gauss-Seidel Newton
     hybrid, a fully monolithic block Newton, adaptive step-size
     marching, and forced-deterministic threading as a diagnostic.
     Also tried and falling short in informative ways: real
     PCFIELDSPLIT (real but unreliable residual decrease), n-value
     continuation and an analytic Bean-like seed (both real mid-run
     effects that didn't survive to the point that mattered), and
     jitter-retry (doubled raw success rate but could not be shown
     trustworthy — different "converged" retries landed on wildly
     different SCIF values).
   - **Root cause**: NOT irreducible chaos. The two-phase Picard
     relaxation scheme, tuned for `dt=600s` (`alpha=0.30`/`0.15`),
     provides zero effective damping at `dt=60s` and below — `T`
     overshoots ~100-150x its own boundary-condition scale within 5
     iterations and stays trapped in a bounded but never-settling
     attractor. Confirmed by tracing the fully deterministic (forced
     single-threaded, bit-identical across repeats) failure trajectory
     directly.
   - **Fix**: a ~10x smaller, still-fixed, non-adaptive relaxation pair,
     `alpha=(0.03, 0.01)`.
   - **Validation, accuracy-first, forced-full-length (bypassing the
     internal EMA `converged` flag entirely, since it was shown to fire
     300-1000+ iterations before genuine settling at every tested
     point)**: genuinely converges across `dt` in {600,300,150,100,60}s
     at `I=19.6A` and `I` in {49,98,196}A at `dt=60s`; FAILS again,
     genuinely (not a premature-stop artifact), at `dt=30s`; 5
     independent multi-threaded launches at the true convergence horizon
     agree to 0.004%; a genuine 5-step multi-step ramp (this project's
     first) converges cleanly step to step with state properly carried
     forward. Real cost: ~700-1500 iterations, not the ~460 first
     assumed.
   - **A resolved false alarm**: comparing against the established
     `641.26mT` reference at `dt=600s, I=196A` showed an apparent ~2%
     gap, at first blamed on `alpha`. It is NOT an alpha problem —
     default alpha through the same test harness (`_picard_phase`) also
     gives ~653.9mT, agreeing with the fix to <0.02%. The gap is between
     this harness and the SEPARATE, independent production
     implementation (`ta_solve.solve_ta_at_current()`, still reliably
     giving `641.27mT` today) — present regardless of alpha, root cause
     not found despite five isolation attempts. Comparisons WITHIN the
     harness remain valid; comparisons against `solve_ta_at_current()`-
     sourced numbers are not, until this is understood.
   - **Separately, still open and unrelated to any of the above**:
     building a differentiable Jc(B)/n(B) model
     (`physics/entropy_ic_model.py`, solid, reusable) surfaced a real
     Dirichlet-BC bug in `monolithic_ta.py` (all 6 T-layers sharing one
     function space, so it was never actually coupled) and showed its
     historical "converges to 3 wrong answers" result never held up
     against the raw residual (explodes then plateaus at billions).
     Whether a correctly-assembled monolithic T-A Newton system can
     converge at all is genuinely OPEN — not resolved by, and not
     needed for, the relaxation-parameter fix above.
   - **2026-08-07 update**: multi-threaded execution of a full ramp is
     now tested — reliable run-to-run (0.001-0.03%), but with a new,
     unresolved ~2.2% gap against the single-threaded reference. A ramp
     crossing the `dt=30s` boundary mid-sequence
     (`transient/validation/dt_crossing_ramp.py`) is also tested —
     warm-started at higher current, it does NOT reproduce the cold-start
     chaos (mild transitional response, self-corrects within one more
     dt=30s step, full recovery once back at dt=60s). **The dt=30s
     "fully chaotic" finding is now known to be cold-start/low-current-
     specific, not a universal dt=30s failure** — the boundary depends on
     the state a step starts from, not on dt in isolation. Not the same
     as "dt=30s is safe": a poor/cold entry or many consecutive dt=30s
     steps remains untested.
   - **Not yet tested**: the NI circuit closure (out of scope, insulated
     limit only); a real production ramp schedule starting from a true
     cold zero-current state (every test so far starts from `I=19.6A`,
     not `I=0`); a smaller alpha rescuing `dt=30s` from a cold start; the
     harness-discrepancy root cause; the new single-vs-multi-threaded
     accuracy gap; a cold or poor-state entry into `dt=30s` mid-ramp; many
     consecutive `dt=30s` steps.
3. **No fast uniformity proxy exists** (see "Proxy graveyard"). If a
   genuinely fast search is wanted again, this needs new derivation work,
   not another guess — three previous guesses were each wrong in ways
   that weren't obvious until checked against T-A.
4. NI thermal design case (sudden discharge, all 8.07kJ into the
   winding) is not yet modeled — isothermal/EM-only so far, no
   temperature-rise estimate.
5. `visualization/plot_convergence_poster.py` is unreliable — it naively
   replays the cumulative master log's `all_constraints_ok` flag, which
   mixes many now-obsolete constraint eras. Needs rebuilding from only
   genuinely T-A-validated points.
6. **2026-08-08: T-A shows a materially tighter local quench margin than
   the uniform-J assumption every OTHER quench check in this project
   relies on (worst-cell margin 0.69-0.71 vs. 1.60, ~2.3x tighter),
   present at ANY ramp speed including the design's own slow reference —
   not a fast-ramp-specific finding.** Whether this is a real,
   previously-undetected gap in the champion's stated quench safety
   margin, or expected/benign Bean critical-state behaviour the
   uniform-J margin was never built to catch, is an OPEN judgement call,
   deliberately not resolved this session. No ramp-power recommendation
   should be made until it is. See "Ramp-up power analysis" above and
   `docs/HISTORY.md`'s 2026-08-08 entry.
7. **2026-09-02/03: an overnight search for a geometry that closes issue
   #6's margin gap found a strong negative signal, not a fix — and
   raised, without confirming, a structural (not geometry-tuning)
   explanation.** ~1247 real T-A-evaluated candidates
   (`optimize/studies/ta_safe_margin_search.py`, `optimize/runs/
   ta_safe_margin/`) across a wide `a`/`b`/`coil_half_gap`/turn-split
   region found the T-A-safe operating current confined to 5.0-54.2A
   (median 35A) everywhere tried — best `B_target_T` found was 4.10T, and
   zero candidates passed all constraints. This search's variables
   (`a`/`b`/`gap`/turn split) could not move the ceiling; a follow-up
   varying `N_LAYERS` itself is queued (`CMAES_X0_JSON_OVERRIDE` in
   `opt_config.py`, an `8layer` variant started but only reached 2
   generations) but not yet run at scale. See "Ramp-up power analysis"
   above for the full account. **This makes issue #6 more consequential,
   not less** — treat both as one open question, not two.
8. **RESOLVED 2026-09-12 (same day, continued): 4.2K operation does NOT
   meaningfully raise the achievable field.** The solver-reliability wall
   (physically-inconsistent Jc/n mismatch, false-stall convergence,
   bisection-budget exhaustion, non-monotonic results) was root-caused
   the same way the `transient/` short-dt problem was: a forced-full-length
   validation matrix (`optimize/studies/ta_thermal_validation.py`) found
   the SAME relaxation-parameter root cause (default `alpha=(0.30,0.15)`
   provides ~zero damping once n leaves the historically-validated 13-34
   range; `alpha=(0.03,0.01)` fixes it here too). With that fix, a clean,
   monotonic, low-residual direct current sweep gives 20K T-A-safe I_op ≈
   17.7A (matches the champion's own previously-reported ~17.3A to
   within 3%, cross-validating the harness) vs. 4.2K Ic+n-matched I_op ≈
   19.3A — **only a ~9% improvement**, despite Ic/n themselves being
   ~2.2-3x higher at 4.2K. Translates to roughly 0.92T → ~1.0T, nowhere
   near closing issues #6/#7's ~11x gap to 10T. **Temperature alone is
   not a sufficient lever for closing that gap** (see issue #9 below for
   what tonight's follow-up search found when combining it with more
   layers); the interpretation question issues #6/#7 raise (uniform-J
   margin never the right check, vs. a genuinely tight real quench
   margin) is unchanged and still open. Also fixed in the same session:
   `ta_safe_current.py`'s `MAX_BISECT` budget bug (see "Bugs fixed"
   #13). **Still open, found as a byproduct**: the 20K baseline itself
   is not mesh-converged even isolating in-plane resolution alone (0.925
   fine vs 1.162 medium worst-margin at otherwise-identical settings) —
   no Richardson-style multi-mesh extrapolation has been done; see
   "Cryogenic (4.2K) operating-temperature investigation" above for the
   full account.
9. **2026-09-12/13 overnight: staged CMA-ES search combining the 4.2K
   model with issue #7's queued `N_LAYERS` lever — real, substantial
   progress (3.8x the 6-layer-at-4.2K ceiling), still well short of
   10T.** Prompted directly by a user challenge: real REBCO magnets DO
   reach 10T+ (some far beyond), so if 4.2K alone barely helps, something
   else must differ. Checking the source paper's (Tsuchiya et al.) own
   reference list surfaced one candidate answer directly: citation [30]
   is "a large 20-pancake prototype reaching up to 25T" — 20 pancakes
   (40 layers) vs. this project's 6. `optimize/studies/
   ta_safe_margin_search_4p2K.py` (sibling of the 20K-era search,
   `ta_safe_margin_search.py`, kept separate so that script's own
   checkpoints/history stay intact) reruns the SAME wide-bounds T-A-safe-
   current CMA-ES search but with the matched 4.2K Ic+n models, the
   validated `alpha=(0.03,0.01)`, and every solve forced to the full
   `ta_n_picard` iterations (the EMA stall flag was found to never fire
   True in this regime regardless of `min_iters`, so this costs nothing
   extra and removes a systematic false-stall risk). `ta_safe_current.
   evaluate()` gained `n_model=`/`min_iters=` kwargs the same session
   specifically to make this possible (additive, default behaviour
   unchanged). `optimize/studies/run_4p2K_overnight.py` orchestrated 5
   time-boxed stages overnight, per explicit user direction (try 4.2K at
   the current 6-layer topology first; try new starting points/wider
   bounds if that stalls; only then escalate layers), each stage's
   subprocess checkpointing cleanly on a timeout-triggered SIGTERM:

   | stage | layers | evals | best B_target_T | binding constraint |
   |---|---|---|---|---|
   | 1: 6L, champion warm start | 6 | 99 | 3.67 | uniformity (2.73% vs 0.8% cap) + field |
   | 2: 6L, alt start + wider bounds | 6 | 18 | 3.83 | field only (hoop/unif both clear) |
   | 3: 8L, warm start from 20K-era best | 8 | 63 | 5.56 | field + uniformity |
   | 4: 10L, warm start from 20K-era best | 10 | 30 | 5.54 | field + uniformity |
   | 5: 12L, fresh wide search | 12 | 30 | **6.72** | field only (hoop 25MPa, unif 1.18% both clear-ish) |

   No stage produced an `all_constraints_ok=True` candidate (10T is the
   hard field floor); every stage's own budget (90-126 min) bought only
   18-99 evaluations — a handful of CMA-ES generations at most, not the
   500-1200-eval depth the 20K-era searches reached, because every solve
   here is deliberately forced to full length. **Reading the trend
   honestly**: 6→8 layers was a big jump (3.8T→5.56T), 8→10 was flat
   (5.56T→5.54T, likely just under-evaluated — only 30 evals vs 63), and
   12 broke through again to 6.72T — consistent with real progress that
   is simply starved of evaluation budget at each step, not with a hard
   ceiling near 5-6T. Stage 5's winner (a=32.19mm, b=45.25mm,
   gap=26.56mm, n_turns=[600,600,559,559,37,37,519,519,583,583,601,601],
   I_op=67.04A) has a genuinely comfortable margin at its own T-A-safe
   operating point — **0/10927 coil cells locally quenched** (worst
   margin 1.006), a categorically different, safer situation than the
   champion's 27-34%-over-Ic picture at its own (much more aggressive)
   I_design — see `visualization/quench_location_4p2K_winner_*.png`,
   reproducible via `optimize/studies/regenerate_4p2K_winner_fields.py`
   + `plot_quench_location_4p2K_winner.py`. Field snapshot:
   `optimize/runs/ta_safe_margin_4p2K/winner_fields.npz`.

   **Honest interpretation**: this is real, substantial, quantified
   progress (3.8T → 6.72T, ~77% higher than the 4.2K-at-6-layers ceiling
   from issue #8), directly connecting two previously-separate threads
   (temperature and layer count) — but 6.72T is still 33% short of 10T,
   and the citation that motivated this ([30]'s 40-layer design) is
   more than 3x this search's best (12 layers) tried so far. **Not yet
   done**: any stage beyond 12 layers; a longer per-stage budget to let
   CMA-ES actually converge within a layer count rather than being
   time-boxed mid-search; and reconciling the "which pairs should be
   thick vs thin" pattern CMA-ES keeps finding (one persistently thin
   pair, e.g. 37/37 turns here) against the per-cell worst-margin
   location analysis `ta_safe_margin_search.py`'s own comments already
   flagged for the 20K case. Whether continuing this same escalation
   (16, 20, ... layers) reaches 10T, plateaus short of it, or needs the
   striation/hybrid-field/margin-criterion levers instead (see issue #8's
   entry above) is still open.

   **2026-09-13 continuation**: the escalation was continued past 12
   layers, per explicit user direction ("continue on that path until we
   start getting some mean fields above 10T"). Progress: 6L→3.8T,
   8L→5.56T, 10L→5.54T (likely just under-evaluated, 30 evals vs 8L's
   63), 12L→6.72T, **16L→6.99T** (best so far, `a`=31.13mm, `b`=40.02mm,
   gap=33.97mm, `n_turns=[550,550,584,584,17,17,507,507,507,507,613,613,
   629,629,605,605]`, I_op=63.06A, hoop=23.7MPa, uniformity=0.13% — both
   comfortably clear, only field is short; worst-cell margin 0.962,
   slightly below 1.0, unlike 12L's cleanly-safe 1.006). **This
   "still climbing" read did NOT hold up — see the 2026-09-14 morning
   update below, which found 16L was the PEAK, not a waypoint.** Two
   operational bugs were found and fixed
   live during this continuation, both now fixed in
   `optimize/studies/run_4p2K_overnight_v2.py` and (for consistency)
   `run_4p2K_overnight.py`: (1) a force-killed stage leaked its entire
   worker pool (`proc.kill()` only kills the immediate child, not
   ProcessPoolExecutor descendants — fixed via `start_new_session=True` +
   `os.killpg()`; 20 such orphaned workers, leaked across stages 1-5 of
   the first overnight run, were found still consuming ~4.4GB RAM and
   real CPU 6+ hours later), and (2) per-solve cost scales
   SUPER-linearly with turn count (time ~ turns^1.28 fit from the only
   two measured points, 6L@1726 turns=42.7s/solve, 12L@5798
   turns=201s/solve) — the original per-generation timeout (2700s) and
   per-stage budget (9000s) were both sized for 6-12 layers and started
   silently discarding whole generations at 16L; raised to 10800s and
   21600s respectively, plus `TA_SAFE_POPSIZE` dropped 11→8 to fit more
   generations per stage in this now-expensive-per-eval regime.

   **2026-09-14 morning update — RESOLVED (for this topology/search):
   the layer-count lever has a real ceiling around 16 layers / ~7T, not
   a continued path to 10T.** Per user direction overnight ("get a run
   above 10T by morning if possible"), budgets were cut 6h→3h/stage and
   the bisection tolerance loosened (3A→6A, `TA_SAFE_BISECT_TOL_A` env
   var added to `ta_safe_current.py`, default unchanged) to cover more
   layer counts before morning, at the cost of shallower CMA-ES
   convergence per stage. Full overnight table:

   | layers | best B_target_T |
   |---|---|
   | 6 | 3.83 |
   | 8 | 5.56 |
   | 10 | 5.54 |
   | 12 | 6.72 |
   | **16** | **6.99 (peak)** |
   | 20 | 5.66 |
   | 24 | 5.95 |
   | 28 | 5.58 |
   | 32 | inconclusive — paused before any generation completed |

   **10T was not reached.** More importantly, the search **peaked at 16
   layers and every stage since (20L, 24L, 28L) landed BELOW it**,
   clustered in a 5.6-6.0T band with no recovery across three
   consecutive stages — a real signal, not single-stage noise from an
   under-evaluated budget. This does not prove 20-28L are worse than 16L
   AT FULL CONVERGENCE (each stage here only got a shortened 3h budget,
   vs. 16L's own 6h), but three independent stages all landing in the
   same band well below the 16L peak is meaningfully different from "one
   unlucky stage." **Practical reading: within this search's reachable
   region (given the time actually spent), layer count alone tops out
   around 12-16 layers at ~7T, roughly 30% short of 10T** — continuing
   to add layers is not a reliable path to closing that gap by itself.
   Combined with issue #10 below (a real 10T HTS racetrack magnet reaches
   its target mainly via ~18x higher current-per-turn through a THICKER
   conductor, not more layers or lower temperature), the evidence now
   points at conductor cross-section — never a search variable in this
   project (`params.w`=4mm single tape has always been fixed) — as the
   more promising untried lever, not further layer escalation. **Per
   explicit user direction, the run was paused (not stopped) at this
   point** (`kill -STOP` on the whole process tree — safe, zero progress
   lost, resumable) rather than continued or abandoned; all data above
   is preserved in `optimize/runs/ta_safe_margin_4p2K/stage{6..10}_*/`.

   **2026-09-14/15 — dedicated overnight optimization AT 16 layers
   (resumed from the same checkpoint, not restarted) did not beat the
   original find, but did surface a cleaner alternative.** Per user
   direction ("do more optimization on the winning layers at 4mm 4.2K"),
   the layer-escalation orchestrator was retired and
   `ta_safe_margin_search_4p2K.py` was run directly against
   `stage6_16layer`'s own checkpoint (preserving its 32 prior
   evaluations/4 generations), given a full night with no per-generation
   or per-stage wall-clock ceiling (2h generation-hang-protection only),
   `TA_SAFE_BISECT_TOL_A=6.0` kept from the speed-up work. Result after
   64 more evaluations (96 total): **best B_target_T unchanged at
   6.987T (eval 13)** — genuinely re-confirmed as at least a strong local
   optimum for this layer count, not just an early lucky find abandoned
   too soon. One close, meaningfully DIFFERENT alternative did emerge:
   eval 83, B_target_T=6.713T (a=36.71mm, b=50.10mm, gap=33.64mm,
   n_turns=[673,673,679,679,21,21,509,509,629,629,582,582,625,625,672,672],
   I_op=54.58A, uniformity=0.607%, hoop=22.7MPa) — 4% less field but
   worst-cell margin **1.034** (genuinely 0 locally-quenched cells) vs.
   eval 13's 0.962 (a handful of cells just over their local Ic) — a
   small-field-for-cleaner-margin tradeoff worth knowing about if margin
   headroom ever matters more than the last 0.3T. No PETSc/solver crashes
   across the whole run; occasional
   benign per-candidate failures (DOF-alignment misses, degenerate-mesh
   KSP divergence) were caught and penalized by the search's own existing
   error handling as designed, not new problems. **Paused (not stopped)
   again** at this point, same safe `kill -STOP` discipline, all 96 rows
   preserved in `optimize/runs/ta_safe_margin_4p2K/stage6_16layer/`.

10. **2026-09-13: a real, currently-being-built 10T HTS racetrack dipole
    (found via user-provided PDF, not web access — IEEE Xplore blocks
    this session's fetch tooling) gives a concrete external benchmark,
    and it points at a lever this whole search has never varied.** Sorti
    et al., "Final Design and Production of a 10 T HTS Energy-Saving
    Dipole Magnet for the Italian Facility IRIS," IEEE Trans. Appl.
    Supercond. 36(3), Art. 4601205, May 2026, DOI 10.1109/TASC.2025.3628303
    (`physics/Final_Design_and_Production_of_a_10_T_HTS_Energy-Saving_
    Dipole_Magnet_for_the_Italian_Facility_IRIS.pdf`) — the IRIS/ESMA
    project (INFN Milan + ASG Superconductors), currently in fabrication,
    not yet tested. Same basic architecture as this project (flat
    REBCO racetrack pancakes, non-insulated/metal-as-insulation winding,
    conduction-cooled) — a genuinely comparable real design, not just
    another simulation.

    | | ESMA (real, in production) | This project's 2026-09-13 best (16L) |
    |---|---|---|
    | Field | 10 T (target) | 6.99 T (search-found) |
    | Layers/racetracks | 12 | 16 |
    | Turns | 4200 (350/racetrack × 12) | 8024 |
    | **Current/turn** | **1150 A** | **63 A** |
    | Conductor | 2× face-to-face 12mm YBCO tapes + steel co-wind (162µm stack) | 1× 4mm tape, single layer |
    | Coil scale | 130mm mandrel radius, 240mm straight | ~31-45mm radius |
    | Operating temp | 20 K | 4.2 K (this project's model) |
    | Tape length | ~6.24km cable (12.48km raw REBCO, 2 tapes/cable) | 1.96 km |
    | Critical current margin | >20% (≤80% Ic) | 35% (65%-of-Ic target) |

    **The ~18x gap in current-per-turn, not layer count or temperature,
    is the single biggest difference.** ESMA reaches this by using a
    THICKER conductor (two 12mm-wide tapes stacked face-to-face) rather
    than more layers of thin tape — raising the local Ic enough to carry
    far more current per turn at a comparable safety margin. This
    project's entire search space (this session and all prior ones) has
    only ever varied `a`/`b`/`coil_half_gap`/turn-count/`N_LAYERS` — the
    conductor cross-section itself (`params.w`=4mm single tape) has never
    been a search variable. **Per explicit user direction (2026-09-13),
    this is NOT being added to the running search** — documented here as
    a candidate lever for a FUTURE decision, not applied.

    Also worth carrying forward: ESMA's own paper independently
    corroborates that this project's margin/quench concerns (issues
    #6/#7 above) are a real class of problem, not an artifact of this
    project's own modeling choices — they found NO quench-protection
    scenario avoided a "disruptive mechanical shockwave" even with
    non-insulated windings, had to switch from active to preventive
    protection as a result, and their thermal budget is razor-thin (15W
    max during ramp, mechanical safety factors of only 1.5-2× in places).
    A real, funded, currently-being-fabricated 10T HTS racetrack magnet
    is wrestling with the same class of margin/protection difficulty
    this project's own T-A-vs-uniform-J investigation surfaced.

    **2026-09-14 — the ESMA-motivated hypothesis (conductor width, not
    layer count or temperature) was directly tested, and it dwarfs every
    other lever tried this session.** Quick, one-off test (NOT part of
    the search — `optimize/studies/test_12mm_tape_width.py` /
    `eval_12mm_tape_validated.py`, geometry unchanged, no re-optimization,
    per explicit user direction): took the 16-layer winner's EXACT
    radial geometry (`a`/`b`/`n_turns` from issue #9's eval 13) and
    widened `params.w` from 4mm to 12mm (3x), with two physically-required
    adjustments, both disclosed: (1) Ic scaled 3x — the Shanghai CSV is
    Format B (absolute amps for the real 4mm tape; `tape_width` is
    IGNORED for Format B, confirmed in `ic_model.py`'s own docstring), so
    widening `params.w` alone does nothing to modeled Ic; a linear
    Ic-scales-with-width assumption was applied instead (disclosed, NOT
    measured data — same epistemic status as the 4.2K ratio-transplant
    model), `n(B,theta)` left unscaled (reflects pinning uniformity, not
    cross-sectional area); (2) `coil_half_gap` raised from 34.0mm to
    98.0mm — 16 layers × 12mm = 192mm of axial stack vs. the original
    64mm, physically cannot fit the old gap, so it was raised to the new
    manufacturing floor plus the same margin-above-floor the original
    design had.

    First attempt (20K, DEFAULT alpha=(0.30,0.15)) did NOT converge (raw
    |dB|/|B|=0.277, SCIF oscillating chaotically 857-1458mT even at
    k=150) — same class of problem as the whole 4.2K investigation:
    pushing to this much higher current (I_hi=476.6A, vs. the original
    design's 44-63A) pushes a large fraction of cells to/above their
    local Ic (naive worst_margin=0.42-1.9, wildly inconsistent across
    reruns), a stiffer regime default alpha was never validated for.
    An alpha sweep (`test_12mm_tape_alpha_sweep.py`, same methodology as
    the temperature investigation) found the SAME `alpha=(0.03,0.01)`
    pair that already fixed BOTH the `transient/` short-dt problem and
    the 4.2K regime also converges here — 4 independent alpha values
    ≤0.05 all converged to a consistent worst_margin~1.70-1.89 cluster
    (raw residual ~1e-3, vs. ~0.19 for every alpha≥0.10 tested), smooth
    monotonic SCIF decay, not chaotic oscillation.

    **Validated result, `alpha=(0.03,0.01)`, forced full-length, raw
    residual 2.67e-3:**

    | | 4mm @ 20K | 4mm @ 4.2K | **12mm @ 20K** |
    |---|---|---|---|
    | I_op | 44.2 A | 63.1 A | **476.6 A** |
    | **B_target** | 4.93 T | 6.99 T | **17.38 T** |
    | Worst-cell margin | 0.92 | 0.96 | **1.72** |
    | Uniformity | 0.075% | 0.130% | 0.604% (still passes) |
    | Hoop stress | 12.6 MPa | 23.7 MPa | 181 MPa (still passes, 400 cap) |

    **Tripling tape width took the exact same design from 4.93T to
    17.38T (3.5x) at 20K alone — no re-optimization, not even the 4.2K
    boost — comfortably clearing 10T with genuine safety margin to
    spare (worst-cell 1.72, well above the 1.5385 requirement).** This
    dwarfs every other lever tried this session: the entire overnight
    layer-escalation effort topped out at 1.42x (6.99T @ 16L); 4.2K alone
    gave 1.29-1.43x depending on the design. Field snapshots:
    `optimize/runs/ta_safe_margin_4p2K/winner_12mm_tape_VALIDATED_fields.npz`
    (this result) and `winner_at_20K_fields.npz` (the 4mm/20K reference).

    **Caveats, not yet resolved:** (a) the linear Ic-vs-width scaling is
    an assumption, not measured data — this project has no real 12mm-tape
    Ic(B,theta) measurements; (b) the 98mm gap this needs was not checked
    against face-gap/manufacturing constraints (`geometry_violation()`
    was bypassed for this one-off test, per explicit "don't touch the
    optimization" direction); (c) `a`/`b`/`n_turns` were never
    re-optimized FOR a 12mm tape — CMA-ES has only ever searched with
    `params.w` fixed at 4mm, so this number is a lower bound on what a
    genuine 12mm-tape search could find, not an upper one. **Per explicit
    user direction, conductor width has NOT been added as a search
    variable** — this remains a documented, validated, but unexplored
    lever for a future decision, same status as before, just now with a
    concrete number behind it instead of just the ESMA analogy.

**Full history, every rejected design, every retracted claim, and the
reasoning behind every conclusion above:** `docs/HISTORY.md`.
