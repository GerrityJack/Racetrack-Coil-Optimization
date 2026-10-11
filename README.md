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
| local current density | **no cell above its local Jc(B, θ)** at the moment a **30 min constant-power ramp** ends (2 h until 2026-10-05) |
| end-cap hoop stress | ≤ 400 MPa |
| manufacturing | bend radius ≥ 7.5 mm, coil face gap ≥ 3 mm, double pancakes (even layer count, paired turn counts) |
| objective | use as little tape as possible (both coils; see note below) |

Tape width (4 mm) and operating temperature (20 K) are fixed project
decisions.

> **Tape length is for both coils since 2026-10-05.** `tape_km` and
> `params.tape_length_m` now cover the whole magnet (two identical
> coils). Every tape length reported before that date, in this README,
> CLAUDE.md, docs/HISTORY.md and all older run CSVs, is **per coil**
> (half the magnet). The 16-layer design's 1.96 km is 3.92 km for the
> magnet. `params.tape_length_per_coil_m` keeps the old quantity.

> **2026-10-04: T-A results before this date are invalid.** The
> T-equation's resistivity was multiplied by δ_SC/Λ (1/75). The
> formulation paper applies that factor only to the A-equation source.
> Because the factor is uniform, every T-A step behaved like a time step
> 75× shorter than the one set (the 4 h surrogate modelled ~190 s), which
> overstated the per-cell loads. An independent discrete-tape benchmark
> confirmed this (`validation/ta_rho_scaling_benchmark.py`), and the factor
> is now removed. Every load, operating current and field number below that
> was measured before the fix, including the optimization results in
> `optimize/runs/full_config_search/`, must be re-run before use. Do not
> resume that search's checkpoint; start a new one.

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
python run_optimization.py --quick-test   # tiny REAL search, ~1-3 h: coarse mesh, 6-layer design,
                                          # 6/8 layers, short solves -- tests every code path end to
                                          # end (numbers are NOT meaningful; output in *_quicktest/)
```

Recommended order on a new machine: `--check`, `--dry-run`,
`--quick-test`, then the real search.

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

**Continuing a search on another machine.** The results in
`optimize/runs/search_ramp30min/` (the output folder since 2026-10-05;
`full_config_search/` holds the old, invalid 2 h-ramp search) should be
committed to git, so the search moves between machines with the
repository:

```bash
git pull                                 # gets history.csv, best.csv, checkpoint.pkl, run.log
conda env update -f environment.yml      # needs pycma >= 4.5 (installed from pip)
python run_optimization.py --check       # must print "saved search found ... RESUMES it"
python run_optimization.py               # continues where it stopped -- NO --fresh
```

When you stop, commit the folder and push it back (`git add
optimize/runs/search_ramp30min && git commit && git push`). Don't edit
the settings block for a continued run unless asked: any change to a
limit or weight re-scores the saved results. If `--check` says the
checkpoint can't be loaded, upgrade pycma (`pip install 'cma>=4.5'`).
Don't use `--fresh`, which starts over.

**Outputs** (`optimize/runs/search_ramp30min/`):

| file | content |
|---|---|
| `history.csv` | every candidate: round, layer count, geometry, status, fitness, I_op, B, uniformity, hoop stress, gated load (`load_gated`) and worst-cell load, T-A convergence flag and its parts (`scif_range_mT`, `load_range`, `load_resid`), sliver count, run time, error text. Status `unconverged` = solved but not converged; its numbers are kept but not used |
| `best.csv` | best design that satisfies all constraints |
| `checkpoint.pkl` | CMA-ES state (used for resuming) |
| `run.log` | copy of the terminal output |

**Settings.** Edit the block at the top of `run_optimization.py`:

| setting | default | meaning |
|---|---|---|
| `START_DESIGN` | 16-layer design (below) | starting geometry; it seeds every layer count |
| `INITIAL_LAYERS` | [12, 16, 20] | layer counts searched in the first round |
| `LAYERS_RANGE` | (4, 40) | layer counts the search may move to |
| `LAYER_STEP_COARSE` / `_FINE` | 4 / 2 | neighbour spacing when moving toward better layer counts |
| `GENS_PER_ROUND`, `MIN_ACTIVE` | 2, 2 | generations per layer count per round; layer counts kept after each round |
| `B_MIN_T` | 10.0 | field floor (project decision, 2026-09-29: keep 10.0; build-tolerance margin is checked when validating finalists, see Limitations). Changing any limit or weight re-scores the saved results on the next resume |
| `UNIFORMITY_MAX_PCT` | 0.8 | uniformity limit |
| `W_FIELD`, `W_UNIFORMITY`, `W_HOOP` | 6000, 100, 40 | penalty weights (see below; doubled with the switch to two-coil tape, same trade-offs as the old 3000/50/20) |
| `RAMP_TIME_S` | 1800 | constant-power ramp duration (30 min). Changing it changes the physics: start a new output folder |
| `MAX_EVALS` | 300 | budget, counted in T-A-solved candidates (geometry rejects are free) |
| `N_WORKERS` | auto | parallel candidates; auto = min(CPUs/2, RAM/2.5 GB, 8) |
| step sizes, ranges, timeouts | | see the comments in the file |

**Cost.** Every candidate takes 3–5 T-A solves (a secant search for its
current). With the 2 h ramp this was roughly 20–60 min per 16-layer
candidate on a modern 8-core machine (median 136 min per candidate in
the 2026-10-04 run). **The 30 min ramp roughly triples that:** its 1 h
surrogate step needs a 4× smaller relaxation α, so 4× the iterations, at
the floor-0.7 and floor-0.6 stages, which dominate the cost. A
generation of ~11 candidates on 4 workers therefore takes a few hours; a
few hundred evaluations is a multi-day run. Each worker needs about
1–2.5 GB of RAM.

---

## What the optimizer does

**Variables:** end-cap radius `a`, straight-section parameter `b`, coil
half-gap, one turn count per double pancake (layers 2i and 2i+1 share a
count), **and the number of layers**.

**How the layer count is searched.** Each layer count has a different
number of variables, so each gets its own CMA-ES search (an "island"),
and the islands compete:
1. **Round 1** searches 12, 16 and 20 layers. Each starts from the
   16-layer design, re-profiled to that layer count: the turn profile is
   stretched over the stack with the same total turns, and the coil gap
   and `a`/`b` are adjusted to stay buildable.
2. **Every round**, each active layer count gets 2 CMA-ES generations.
   The layer counts are then ranked by their best fitness and the worse
   half is retired (at least 2 are kept).
3. **The untried neighbours** of the best layer count are added: ±4
   layers, then ±2 once both ±4 neighbours are known to be worse. New
   layer counts start from the current best design, re-profiled.

So the effort moves toward the best layer count, coarse to fine. The
terminal prints a table of every layer count tried (status, generations,
best fitness, best design) after each round. A dry run with fake physics
that peaks at 24 layers moved 16 → 20 → 24, bracketed with 22/26, and
retired the others.

**Per candidate** (`optimize/ta_ramp_current.py`):
1. **Geometry check** (instant): bend radius, face gap, and straight
   section. Impossible designs get a penalty without a solve.
2. **Operating current.** A secant search finds the largest current I_op
   at which **no coil cell exceeds its local Jc** at the end of a 30 min
   constant-power ramp. Each trial current is one T-A solve (see next
   section for how the ramp is modelled). The reported I_op is always a
   current that was actually solved and passed.
   - "No cell" is measured as the **99.99th percentile of J/Jc by cell
     count**, with sliver elements excluded: the worst cell for meshes
     under 10,000 coil cells, the 2nd–3rd worst on the largest ones. A
     single bad element can't decide the result (error hunt 2026-10-05:
     a sliver gave 1.103 where the true value was 0.994; this gate gives
     0.993). The plain worst cell is still recorded.
   - **Convergence gate:** at I_op the screening field must be flat
     (range < 0.5 mT over the last 30 iterations) and the gated load must
     have settled (range, and slope/α extrapolated to the fixed point,
     both < 0.005). Otherwise the design is recorded as `unconverged` and
     treated as a failed evaluation.
3. **At I_op:** mean box field (Biot-Savart plus the T-A screening-current
   correction), box uniformity from the same T-A state, and hoop stress.

**Fitness** (minimized, CMA-ES via pycma):

```
tape_km + W_FIELD·(field shortfall/10 T)² + W_UNIFORMITY·(uniformity excess/0.8 %)² + W_HOOP·(hoop excess/400 MPa)²
```

Reaching 10 T dominates everything until it's met; after that the search
minimizes tape (`tape_km` = both coils). The uniformity weight is 100
(50 on per-coil tape before 2026-10-05; earlier searches used 20). A
0.1 % excess costs about as much as ~1.6 km of magnet tape (~0.8 km per
coil).

---

## The starting point: the 16-layer design

`START_DESIGN` in `run_optimization.py` (8 double pancakes):

| parameter | value |
|---|---|
| `a` / `b` / coil half-gap | 31.13 mm / 40.02 mm / 33.97 mm (face gap 3.94 mm) |
| `n_turns` | [550, 550, 584, 584, 17, 17, 507, 507, 507, 507, 613, 613, 629, 629, 605, 605] |
| total turns / tape | 8024 / 1.96 km per coil (3.92 km magnet) |
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
over **2·t_ramp**. The search therefore models a 30 min constant-power
ramp as **one implicit T-A step of 1 h** from the virgin state. This
surrogate is **non-conservative**: the 2026-10-05 error hunt compared it
with a real √t march (6-layer design, 2 h ramp, 180 A) and the march's
end-of-ramp load was ~4 % higher on the worst cell (~2–3 % on the 99.9th
percentile), because screening currents induced early in the ramp are
still there at the end. Finalists must be confirmed with a march (see
"Validating a design").

---

## Validating a design

The search uses a fast surrogate and a moderate mesh, and leaves out
several expensive checks on purpose (error hunt 2026-10-05,
`docs/error_hunt_2026-10-04.md`). **Every finalist must go through all of
these before it's trusted.** `optimize/studies/ramp_hold_test.py` reports
the search's own gate (`gated_load`) next to the worst cell for each step.

1. **Real ramp march — required; the search's surrogate is
   non-conservative.** Step along the √t constant-power profile with the
   previous state carried forward
   (`RH_SEQUENCE="I:dt:eps:adv[:alpha:iters]"`; the file header documents
   the format). For a 30 min ramp to I_op: steps at t = 450/900/1350/1800 s,
   I = I_op·√(t/1800), dt = 450 s, then refine to 8 steps once to check the
   march itself is step-converged. Reduce α with dt (stable α ∝ ρ_floor·dt)
   and check the load trace, not only SCIF. Pass = gated load ≤ 1 at the
   end of the march. Also note the mid-ramp peak (on the 6-layer design
   the load peaked before the end). Plan: calibrate march/surrogate on 3–4
   designs of different layer counts; if the ratio is stable, tighten the
   search's limit (`TA_RAMP_LOAD_MAX`) by it.
2. **ρ-floor** (`ta_eps_reg`): repeat at 0.5. The search uses 0.6 (since
   2026-10-04). On the 6-layer design, B and uniformity changed by
   ≤ 0.4 % and ≤ 0.03 points between 0.6 and 0.5.
3. **Uniformity on the denser across-width mesh — required.** Repeat with
   `RH_ZGRID=xdense7`. Uniformity is NOT mesh-converged on the production
   grading: 0.475 % (medium5) vs 0.955 % (xdense7) on the 6-layer design
   at 180 A, while loads and B agreed to 0.5 %. Treat the search's
   uniformity as ±0.5 points. Then `optimize/ta_validate.py`, the full T-A
   box check (see "Proxy graveyard" in CLAUDE.md).
4. **Fine in-plane mesh** (tier `fine`) and an independently generated
   mesh (a separate process — gmsh is not bit-reproducible across
   processes). Robust loads (p99.9) and B were mesh-converged in the
   error hunt; the single worst cell was not. On a finer mesh α may need
   to shrink further (the fine-mesh solve at α = 0.01 had not settled).
5. **Air box** — repeat with `RH_BOX_SCALE=8` (the search uses 4). The
   small box puts the FEM field 5–8 % low in the winding. Loads moved only
   ~0.2 % on the 6-layer design, but the hoop stress (computed from the
   FEM field) is ~8 % low; check it with box 8.
6. **α check for the regime used.** A "flat SCIF" does not prove local
   convergence: too large an α makes spurious hotspots the SCIF can't
   see. For each (dt, floor, mesh) used in validation, repeat the last
   stage with α halved and confirm the gated load doesn't move.
7. **Load map:** `optimize/studies/plot_quench_location_ramphold.py <npz>`
   shows where the most-loaded cells are.

**Solver settings that matter** (learned the hard way, 2026-09-26/27):
- **Judge convergence on the SCIF trace**, the per-iteration
  screening-field value (flat over the last ~30 iterations), not on
  `rel_err`. `rel_err` can sit flat while the answer is still drifting.
- **Picard relaxation must scale with ρ_floor·dt.** The T-equation sees
  the self-field from the previous iteration, so its loop gain is
  ~μ₀·w·D/(ρ·dt). Lower floors and shorter steps need a smaller α and
  proportionally more iterations. Examples that worked before the
  2026-10-04 ρ fix: α = 0.10 at dt = 4 h and floor 0.9; 0.02 at 1 h and
  floor 0.8; 0.005 at 1 h and floor 0.7. The fix made ρ 75× larger, so the
  same α now has more stability margin. Measured after the fix (6-layer
  design, dt = 4 h): α = 0.10 converges at floors 1.0–0.8, 0.05 at 0.7 and
  0.01 at 0.6. For the 1 h step the evaluator divides the floor-0.7 and
  floor-0.6 α by 4 (0.0125, 0.0025) and multiplies their iterations by 4
  (600, 1200); the floors ≥ 0.8 are far from the limit and unchanged.
  These 1 h values follow from the scaling rule and have not yet been
  checked with a forced long run.
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
| total turns / tape | 1726 / 0.337 km per coil (0.674 km magnet) |
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
- **Ramp surrogate is non-conservative.** The 30 min constant-power ramp
  is modelled as one 1 h implicit step. Against a √t march (6-layer
  design, 2 h ramp) the surrogate's end-of-ramp load was ~4 % low on the
  worst cell; since load rises only as ~I^0.15, that is ~20–25 % in
  allowed current. Finalists must be confirmed with a march.
- **ρ-floor.** The search uses 0.6, reached by continuation 1.0 → 0.9 →
  0.8 → 0.7 → 0.6 on each design's first solve. After the 2026-10-04 ρ fix
  the floor matters much more (6-layer design, floors 0.9 → 0.5):
  - The worst load converges once the floor is ~0.1 below the load
    itself. At 300 A (load 0.98) floor 0.9 is enough; at 180 A (load
    0.88) it needs 0.7.
  - B and uniformity need 0.6. At 180 A: B 10.67 / 10.59 / 10.41 / 10.16 /
    10.12 T and uniformity 1.07 / 0.93 / 0.77 / 0.47 / 0.45 % at floors
    0.9 / 0.8 / 0.7 / 0.6 / 0.5.
  - Floor 0.5 is solvable with α = 0.002 but costs ~1500 iterations.
- **Mesh.** Medium tier, air box 4·b. Error hunt 2026-10-05 (6-layer
  design, 180 A): B and the 99.9th-percentile load agree to ≤ 1.5 %
  across medium / fine in-plane / xdense7 across-width / box 8; the single
  worst cell does not (0.881 / 0.970 / 0.877 / 0.879). The search's gate
  (99.99th percentile by cell count) only removes the top 1–3 cells on
  large meshes, so it is still exposed to most of that scatter; finalists
  need the mesh checks in "Validating a design". **Uniformity is not converged
  across the tape width** (0.475 % vs 0.955 % with xdense7).
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
  jitter test (0/14 builds reached 10 T). The floor stays at 10.0 T
  (project decision), so finalists near 10 T MUST get a jitter study.
- **Layer-count search is greedy.** Retired layer counts are never
  revisited. A layer count that looked poor after only 2 generations
  could be dropped too early. Raise `GENS_PER_ROUND` or `MIN_ACTIVE` for a
  more thorough (slower) search. With ~3 layer counts active, one round
  is roughly 40–50 solved candidates, so `MAX_EVALS=300` is ~6 rounds.

---

## Further reading

- **CLAUDE.md:** detailed technical notes. It covers the solver's known
  quirks, every current finding with its evidence, and the open issues.
- **docs/HISTORY.md:** the full chronological narrative (every search,
  bug hunt, rejected design and retracted conclusion).
- Vargas-Llanos et al., *Supercond. Sci. Technol.* 35, 124001 (2022):
  the T-A homogenisation.
