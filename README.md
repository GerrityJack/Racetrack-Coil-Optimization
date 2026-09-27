# Racetrack HTS Coil Optimization

Finite-element design and optimization of a two-coil **REBCO racetrack
magnet** (4 mm tape, 20 K, no-insulation double pancakes). Built on
FEniCSx (dolfinx 0.11). The core physics model is the homogenised
**T-A formulation** (Vargas-Llanos et al., *Supercond. Sci. Technol.* 35,
124001, 2022), which resolves the screening currents inside each tape
that decide whether any part of the winding exceeds its critical current.

**Design goal:**

| requirement | value |
|---|---|
| mean \|Bz\| over the 30 × 6 mm target box between the coils | ≥ 10 T |
| peak-to-peak field uniformity over that box | < 1 % (search limit 0.8 %) |
| local current density | **no cell above its local Jc(B, θ)** at the moment a **2 h constant-power ramp** ends |
| end-cap hoop stress | ≤ 400 MPa |
| manufacturing | bend radius ≥ 7.5 mm, coil face gap ≥ 3 mm, double pancakes (even layer count, paired turn counts) |
| objective | use as little tape as possible |

Tape width (4 mm) and operating temperature (20 K) are fixed project
decisions.

---

## Quick start: running the optimization

Everything is driven by **one file, `run_optimization.py`**.

**1. Install the environment (once).** Linux or WSL with conda/mamba:

```bash
conda env create -f environment.yml     # creates "fenicsx-env" (dolfinx 0.11, PETSc, gmsh, pycma, ...)
conda activate fenicsx-env
```

**2. Check the setup** (a few seconds). This checks the Python version,
the required packages and their versions, the tape data files, write
access to the output folder, CPU/RAM, and that the start design is
buildable. Every problem is reported with a "HOW TO FIX" line.

```bash
python run_optimization.py --check
```

**3. Optional tests:**

```bash
python run_optimization.py --dry-run   # seconds: exercises the whole search machinery with a fake evaluator
python run_optimization.py --smoke     # one REAL evaluation of the start design (roughly 20-120 min)
```

**4. Run the search:**

```bash
python run_optimization.py                          # uses the settings at the top of the file
python run_optimization.py --workers 4 --max-evals 300
```

- **Resuming is automatic.** The CMA-ES state is saved after every
  generation. If the run is stopped (Ctrl+C, crash, reboot, power cut),
  running the same command again continues from the last completed
  generation.
- **Ctrl+C stops safely.** The generation in progress is abandoned, and
  its candidates are re-proposed on the next run. Press Ctrl+C twice to
  exit immediately.
- `--fresh` starts a new search. The previous results are moved into an
  `archive_<timestamp>/` folder, not deleted.
- **Failures don't stop the search.** Each candidate runs in its own
  process. A solver error, a native PETSc/MUMPS crash, an out-of-memory
  kill, or a solve hung past `CANDIDATE_TIMEOUT_H` only marks that one
  candidate as failed.
- **What the terminal shows:** a timestamped line when each candidate
  starts and finishes (field, current, uniformity, tape, pass/fail); a
  heartbeat every 10 min; and after each generation a progress line
  (evaluations done, average time per candidate, estimated time left)
  plus the best design so far. Everything printed also goes to `run.log`.

**Outputs** (`optimize/runs/full_config_search/`):

| file | content |
|---|---|
| `history.csv` | every candidate: geometry, status, fitness, I_op, B, uniformity, hoop stress, worst-cell load, T-A convergence flag, run time, error text |
| `best.csv` | best design that satisfies all constraints |
| `checkpoint.pkl` | CMA-ES state (used for resuming) |
| `run.log` | copy of the terminal output |

**Settings.** Edit the block at the top of `run_optimization.py`:

| setting | default | meaning |
|---|---|---|
| `START_DESIGN` | 16-layer design (below) | starting geometry; its number of layers fixes the layer count of the search |
| `B_MIN_T` | 10.0 | field floor. Searches converge *onto* this floor, so use ~10.3 T for build-tolerance margin (see Limitations) |
| `UNIFORMITY_MAX_PCT` | 0.8 | uniformity limit |
| `W_FIELD`, `W_UNIFORMITY`, `W_HOOP` | 3000, 50, 20 | penalty weights (see below) |
| `RAMP_TIME_S` | 7200 | constant-power ramp duration |
| `MAX_EVALS` | 300 | budget, counted in T-A-solved candidates (geometry rejects are free) |
| `N_WORKERS` | auto | parallel candidates; auto = min(CPUs/2, RAM/2.5 GB, 8) |
| step sizes, ranges, timeouts | | see the comments in the file |

**Cost.** Every candidate takes 3–5 T-A solves (a secant search for its
current). For the 16-layer design that's roughly 20–60 min per candidate
on a modern 8-core machine, and much longer on a small laptop. A
generation of ~11 candidates on 4 workers therefore takes a few hours; a
few hundred evaluations is a multi-day run. Each worker needs about
1–2.5 GB of RAM.

---

## What the optimizer does

**Variables:** end-cap radius `a`, straight-section parameter `b`, coil
half-gap, and one turn count per double pancake (layers 2i and 2i+1 share
a count). The layer count is fixed by `START_DESIGN`; to try a different
layer count, give a start design with that many layers.

**Per candidate** (`optimize/ta_ramp_current.py`):
1. **Geometry check** (instant): bend radius, face gap, and straight
   section. Impossible designs get a penalty without a solve.
2. **Operating current.** A secant search finds the largest current I_op
   at which **no coil cell exceeds its local Jc** at the end of a 2 h
   constant-power ramp. Each trial current is one T-A solve (see next
   section for how the ramp is modelled). The reported I_op is always a
   current that was actually solved and passed.
3. **At I_op:** mean box field (Biot-Savart plus the T-A screening-current
   correction), box uniformity from the same T-A state, and hoop stress.

**Fitness** (minimized, CMA-ES via pycma):

```
tape_km + W_FIELD·(field shortfall/10 T)² + W_UNIFORMITY·(uniformity excess/0.8 %)² + W_HOOP·(hoop excess/400 MPa)²
```

Reaching 10 T dominates everything until it's met; after that the search
minimizes tape. The uniformity weight is 50 (earlier searches used 20),
which puts somewhat more value on uniformity. A 0.1 % excess costs about
as much as ~0.8 km of tape.

---

## The starting point: the 16-layer design

`START_DESIGN` in `run_optimization.py` (8 double pancakes):

| parameter | value |
|---|---|
| `a` / `b` / coil half-gap | 31.13 mm / 40.02 mm / 33.97 mm (face gap 3.94 mm) |
| `n_turns` | [550, 550, 584, 584, 17, 17, 507, 507, 507, 507, 613, 613, 629, 629, 605, 605] |
| total turns / tape | 8024 / 1.96 km |
| innermost bend radius | 7.54 mm (on the 7.5 mm floor) |

It came from the 2026-09-13/14 layer-count search, which was run under a
4.2 K model. Re-solved at 20 K it gave ~10 T at ~88–92 A, with excellent
uniformity (~0.1 %). Two caveats:
- Those numbers used a single 600 s ramp step and solver settings that
  were later found to stop short of convergence (see Limitations).
- It has **not yet been evaluated under the current criterion**.

`python run_optimization.py --smoke` does exactly that. Expect a lower
field at the end of a 2 h ramp than the old numbers suggest; see the next
section. Its per-turn (uniform-current) load at 10 T is only ~30 % of Ic,
about half the 6-layer design's, so it has much more headroom than the
legacy design.

---

## How the ramp-up time affects the design

This is the most important physics result of the project, and the reason
for the current criterion.

**Why cells go above Jc during a ramp.** A REBCO tape's electric field
rises steeply but smoothly with current, following the measured power law
E = E_c·(J/Jc)ⁿ, with n ≈ 13–34 from the tape data. Jc is *defined* as
the current density at which E = E_c = 1 µV/cm; that's the standard
threshold used when Ic is measured. Ramping the magnet induces an
electric field in the tape. Where screening currents have penetrated, the
local current settles at

  J/Jc = (E/E_c)^(1/n)

so a cell exceeds Jc whenever the local electric field exceeds 1 µV/cm.
**How far over Jc a cell goes is set by how fast the current is changing
at that moment**, not by the design's current rating alone.

**Measured on the 6-layer design** (converged T-A, ρ-floor-converged,
2026-09-26/27):

| scenario | worst cell (J/Jc) | cells above Jc |
|---|---|---|
| 600 s linear ramp to 185 A (end of ramp)¹ | 1.27 | 23 % |
| 1 h linear ramp to 185 A (end of ramp) | 1.165 | 7.5 % |
| … then held at 185 A for 1 h | 1.032 | 2 cells |
| … held 2 h | 0.990 | 0 |
| … held 5 h | 0.946 | 0 |
| … held 11 h | 0.902 | 0 |
| 2 h constant-power ramp, largest current with 0 cells above Jc at ramp end | 0.998 at 126.6 A (7.1 T) | 0 |

¹ At floor 1.0. Its worst load was unchanged at floor 0.9 (measured at
196 A), but the over-Jc fraction rose (25 → 30 %). Every other row was
checked at floor ≤ 0.8.

**Rules of thumb:**
- The worst-cell load drops about **10 % per 10× slower ramp**, which is
  (10)^(1/n) with n ≈ 21, exactly the power-law scaling.
- **After the ramp stops, loads relax below Jc** by flux creep:
  logarithmically, fast at first, then slowly. Real HTS magnets reach
  their DC operating state this way.
- **Near the crossing, the worst load rises only slowly with current**
  (~I^0.3). So asking for "no cell above Jc at the instant the ramp ends"
  costs a lot of field. The 6-layer design gives 10.4 T with a 1 h ramp
  plus a ≥ 2 h hold, but only 7.1 T under the strict end-of-ramp rule.
- **Fully relaxed limit:** with unlimited ramp and hold time, screening
  currents vanish, and the condition becomes simply "turn current < turn
  Ic". That's the fast uniform-current screen in
  `optimize/optimize_geometry.py`. It's a necessary condition only,
  because creep is logarithmic and the fully relaxed state is never
  actually reached.

**The ramp modelled in the search.** A constant-power supply feeding an
inductive coil gives ½LI² = P·t, so

  I(t) = I_op·√(t / t_ramp), with P = L·I_op²/(2·t_ramp)

The ramp is fast at first and gentle near the top, which is the natural
taper. Its final ramp rate, I_op/(2·t_ramp), is that of a linear ramp
over **2·t_ramp**. The search therefore models a 2 h constant-power ramp
as **one implicit T-A step of 4 h** from the virgin state. This is a
surrogate that still has to be validated: see "Validating a design".

---

## Validating a design

The search uses a fast surrogate and a moderate mesh, so a finalist must
be re-checked before it's trusted:

1. **Real ramp march** (`optimize/studies/ramp_hold_test.py`). Step along
   the √t constant-power profile with the previous state carried forward
   (`RH_SEQUENCE="I:dt:eps:adv[:alpha:iters]"`; the file header documents
   the format), and compare the end-of-ramp worst load with the 4 h
   surrogate. *This comparison has not yet been done even for the 6-layer
   design.*
2. **ρ-floor** (`ta_eps_reg`): repeat at 0.8 and 0.7. The search uses 0.9.
   Worst loads were floor-independent to < 1 % from 0.9 down; field and
   screening current are still weakly floor-dependent.
3. **Mesh:** repeat with the denser across-width grading (`RH_ZGRID=xdense7`)
   and an independently generated mesh (a separate process — gmsh is not
   bit-reproducible across processes).
4. **Uniformity** with `optimize/ta_validate.py`, the full T-A box check.
   It is the only uniformity number to trust (see "Proxy graveyard" in
   CLAUDE.md).
5. **Load map:** `optimize/studies/plot_quench_location_ramphold.py <npz>`
   shows where the most-loaded cells are.

**Solver settings that matter** (learned the hard way, 2026-09-26/27):
- **Judge convergence on the SCIF trace**, the per-iteration
  screening-field value (flat over the last ~30 iterations), not on
  `rel_err`. `rel_err` can sit flat while the answer is still drifting.
- **Picard relaxation must scale with ρ_floor·dt.** The T-equation sees
  the self-field from the previous iteration, so its loop gain is
  ~μ₀·w·D/(ρ·dt). Lower floors and shorter steps need a smaller α and
  proportionally more iterations. Examples that work: α = 0.10 at dt = 4 h
  and floor 0.9; 0.02 at 1 h and floor 0.8; 0.005 at 1 h and floor 0.7.
- **The production floor 1.0 is wrong for holds and long ramps.** It
  makes sub-critical tape ohmic and drains the screening currents.

---

## Legacy: the 6-layer design

**Superseded as the starting point.** It is still the geometry in
`params.py`, so solver/visualization scripts that read `params.py` still
use it.

| parameter | value |
|---|---|
| `a` / `b` / coil half-gap | 26.0 mm / 31.4 mm / 13.7 mm |
| `n_turns` | [382, 382, 478, 478, 3, 3] (3 double pancakes) |
| total turns / tape | 1726 / 0.337 km |
| old rating | 196 A, 10.49 T, uniformity 0.495 %, validated against build tolerance (15/15 jitter samples) — under the **old** criterion: 65 % of local Ic with *uniform* current |

Under the T-A current distribution that old rating is not what it
seemed: 23–34 % of cells are above Jc at the end of a ramp. Under the
current criteria:
- **Strict** (no cell above Jc at the end of a 2 h constant-power ramp):
  **7.1 T** at 126.6 A.
- **With a hold** (1 h ramp to 185 A, then ≥ 2 h at constant current):
  **10.38 T**, with 0 cells above Jc from 2 h of hold onward. This result
  was checked for floor, mesh, Ic model, and time-step sensitivity; see
  CLAUDE.md, "10 T with NO cell above Jc: ramp + hold".

---

## Repository layout

```
Racetrack-Coil-Optimization/
├── run_optimization.py        ← START HERE: the one-file optimization runner
├── params.py                  ← geometry/material/solver parameters (holds the legacy 6-layer design;
│                                 its mesh block is a slow one-off "xdense" tier — the runner and the
│                                 study scripts force the "medium" tier in memory)
├── environment.yml            ← conda environment (dolfinx 0.11 pinned)
├── CLAUDE.md                  ← detailed technical notes and current findings
├── docs/HISTORY.md            ← full chronological project history
├── mesh/build_mesh.py         ← gmsh mesh builder (eighth-symmetry domain)
├── physics/
│   ├── ic_model.py            ← Ic(B,θ) and n(B,θ) interpolation of the measured tape data
│   ├── current_source.py      ← racetrack geometry helpers (tangent/normal, turn/layer index)
│   ├── coil2_field.py         ← multi-filament Biot-Savart (both coils)
│   ├── entropy_ic_model.py    ← smooth analytic Jc(B,θ)/n(B,θ) fits (experimental use)
│   ├── Shanghai*.csv          ← measured 20 K tape data, 0-8 T (the only Ic data used)
│   └── digitized_IC_data/     ← other tapes' published data (4.2 K investigation, closed)
├── solve/
│   ├── solve.py               ← uniform-current magnetostatic solve
│   ├── ta_solve.py            ← T-A Picard solver (the production screening-current solver)
│   ├── ta_implicit.py         ← experimental implicit T-A coupling (GMRES) — not used for results
│   └── ta_sweep.py, ta_postprocess.py
├── optimize/
│   ├── ta_ramp_current.py     ← per-design evaluator used by run_optimization.py
│   ├── ta_safe_current.py     ← older evaluator (65 %-of-Ic, single 600 s step)
│   ├── optimize_geometry.py   ← fast uniform-current screen (field, stress, per-turn Ic)
│   ├── ta_validate.py         ← full T-A box-uniformity check
│   ├── ic_extrapolation.py    ← Ic(B) above 8 T (Kim model = default)
│   ├── opt_config.py, cmaes_search.py  ← older uniform-current CMA-ES search and shared constants
│   ├── studies/               ← one-off studies; the ones used now:
│   │   ├── ramp_hold_test.py        ← time-marched ramp + hold (validation tool)
│   │   ├── test_ta_ramp_current.py  ← run the evaluator on one design
│   │   ├── relaxed_margin_test.py   ← fixed-geometry current / ramp / floor sweeps
│   │   ├── plot_quench_location_ramphold.py
│   │   └── ta_safe_margin_search.py ← older multi-worker search (TA_SAFE_EVALUATOR=ramp)
│   └── runs/                  ← all logs and CSVs, one folder per study
├── circuit/                   ← lumped no-insulation circuit model (inductance, τ, constant-power ramp)
├── transient/                 ← T-A + no-insulation transient solver (exploratory)
├── sweep/, validation/        ← older sweeps and cross-checks
└── visualization/             ← figures (quench_location_* = per-cell load maps)
```

---

## The model in brief

**Geometry and symmetry.** Two identical racetrack coils face each other
across the target box. The FEM domain is one eighth of the magnet:
- x = 0 and y = 0 are mirror planes (n×A = 0);
- the midplane is a symmetry plane (natural boundary condition), which
  includes the second coil by the image principle.

Quantities summed over the winding are expanded to all 8 mirror images.

**Magnetostatics.** ∇×(1/μ₀ ∇×A) = J on a tetrahedral mesh with edge
elements. With no iron, the field is exactly linear in current. The fast
screen exploits this: one solve per geometry gives field, stress (∝ I²)
and per-turn Ic.

**T-A screening-current model** (`solve/ta_solve.py`):
- Each tape's superconducting-layer current is J = ∇T × n̂. The transport
  current is imposed through T = ±I/(2δ_SC) on each tape's two edges, with
  one T problem per layer.
- The vector potential A is driven by the homogenised current
  (δ_SC/Λ)·J.
- The resistivity is ρ = (E_c/Jc)·max(J/Jc, ε)^(n−1), using measured
  Jc(B,θ) and n(B,θ).
- The floor ε (`ta_eps_reg`) keeps sub-critical cells from having zero
  resistivity. See Limitations.
- Time is one backward-Euler step per time step,
  E = −(A − A_prev)/dt. A single step from the virgin state models a
  ramp; further steps with `A_prev` carried forward model ramps and holds.
- T and A are iterated to a fixed point (Picard). The A-matrix is
  factorised once and reused.

**Local load.** For each coil cell, load = |J_in-plane| / Jc(B, θ), with
Jc = Ic/(δ_SC·w) from the measured 20 K data (Kim extrapolation above
8 T). "No cell above Jc" means max load ≤ 1 over every mesh cell.

**Target field and uniformity.** The mean |Bz| and the peak-to-peak |B|
over a 30 × 6 mm grid of points between the coils: multi-filament
Biot-Savart for the transport current plus the screening-current
correction from the T-A state.

**Mechanics.** End-cap hoop stress (B×J×r, self-supporting turns) is
enforced. Delamination stress is computed but not enforced.

---

## Limitations

**Model scope**
- **Ic data ends at 8 T.** Above that, Ic is extrapolated with the Kim
  model, the best of five models in hold-out tests. At a fixed current
  the end-of-ramp loads barely depend on the Ic model, since they are set
  by the electric field (tested: Kim vs. `scaling:45` gave identical
  loads and field). Anything that picks the current *from* Ic still
  carries ~±0.5 T of model uncertainty.
- **Isothermal, electromagnetic only.** There's no heating, no quench
  propagation, and no protection analysis. Briefly exceeding Jc during a
  ramp means some local dissipation, and whether that's thermally
  acceptable is not modelled.
- **No-insulation radial currents are not in the T-A model.** Those runs
  are in the insulated limit. The radial currents vanish at DC, but
  during a ramp they delay the field. The lumped circuit model in
  `circuit/` gives τ ≈ 1330/ρ_c[µΩ·cm²] s for the 6-layer design.
- **Homogenised winding.** No individual-tape resolution; the across-width
  resolution is a few graded sub-slabs per tape.
- **Symmetry.** The model assumes equal-sense coils and screening currents
  that share the transport current's mirror symmetry.
- **Mechanics.** Only a hoop-stress screen; no structural analysis,
  delamination limit, or cooldown/prestress.

**Numerics**
- **Ramp surrogate not yet validated.** The 2 h constant-power ramp is
  modelled as one 4 h implicit step. It needs a √t-profile march on at
  least one design before its numbers are trusted.
- **ρ-floor.** The search uses 0.9. Worst loads were floor-independent
  below that, but the screening field and B still move by ~1–2 % between
  0.9 and 0.6. Floors below ~0.6 cannot currently be solved with the
  staggered Picard scheme; a fully implicit T-A solve would be the fix
  (`solve/ta_implicit.py` is an unfinished start).
- **Mesh.** Medium tier. The denser across-width grading changed the
  worst-cell load by ~4 % in one test. In-plane refinement hasn't been
  rechecked with properly converged solves. A per-cell maximum is
  inherently sensitive to individual mesh cells.
- **gmsh meshes are not bit-reproducible across processes.** Repeat
  near-boundary results with an independent mesh.
- **Unreliable older results.** Earlier results produced with
  α = (0.03, 0.01) and 150 fixed iterations, or with the floor at 1.0 on
  long ramps and holds, stopped short of convergence and are unreliable.
  This includes parts of `optimize/runs/relaxed_margin/`, the 4.2 K
  validation study, and the older T-A searches. See CLAUDE.md for the
  list.

**Search**
- **Cost.** Each candidate is several T-A solves (tens of minutes), so
  budgets are hundreds of evaluations, not thousands.
- **No fast uniformity proxy exists.** Four were tried and each failed
  against the full T-A check, so uniformity is evaluated with T-A for
  every candidate.
- **Build tolerance is not in the fitness.** The previous champion found
  that way converged exactly onto B = 10 T and failed a ±0.2 mm / ±2 %
  jitter test (0/14 builds reached 10 T). Use `B_MIN_T` ≈ 10.3 T or
  re-check finalists with a jitter study.
- **Fixed layer count per run.** The layer count is set by
  `START_DESIGN`, not optimized.

---

## Further reading

- **CLAUDE.md:** detailed technical notes. It covers the solver's known
  quirks, every current finding with its evidence, and the open issues.
- **docs/HISTORY.md:** the full chronological narrative (every search,
  bug hunt, rejected design and retracted conclusion).
- Vargas-Llanos et al., *Supercond. Sci. Technol.* 35, 124001 (2022):
  the T-A homogenisation.
