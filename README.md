# Racetrack HTS Coil — Magnetostatics, Screening Currents & Quench Analysis

FEniCSx (dolfinx) FEM pipeline for a two-coil REBCO racetrack magnet:

- **A-formulation magnetostatics** (Nedelec elements, uniform current) — the baseline
- **Homogenised T-A formulation** (Vargas-Llanos et al. 2022) — non-uniform
  screening currents and the screening-current-induced field (SCIF)
- **Ic(B,θ) quench analysis** from manufacturer tape data

**Design targets:** 10 T bore field at the midplane between the coils,
<1 % field uniformity, maximum quench safety margin.

**Environment setup:** create the `fenicsx-env` conda environment from
the pinned spec, then run every script through it from the repo root:

```bash
conda env create -f environment.yml
conda run -n fenicsx-env python3 <script>
```

The pins matter — the code depends on dolfinx 0.11.0 API behaviour
(see the version notes in `environment.yml`).

**→ For a complete explanation of the physics, the model stack, and
every major assumption, read [The physics, explained](#the-physics-explained).**

---

## The design problem — start here

**Minimize REBCO tape length**, subject to:

| constraint | limit |
|---|---|
| mean \|Bz\| over the target box (30×6 mm) at I_op | ≥ 10 T — **evaluated with a validated Ic extrapolation, not the flat clamp** |
| peak-to-peak uniformity over that box | ≤ 1 % — **must be checked with the T-A solver, not the coarse screen** (see below) |
| hoop stress, end-cap curved sections only | ≤ 400 MPa |
| operating point | worst-margin cell at 60–65 % of local Ic (2026-07-31; was 50–60 %) |
| coil-to-coil **face-to-face** clearance | ≥ 3 mm |
| innermost turn bend radius (REBCO cracks below this) | ≥ 7.5 mm |
| double-pancake construction | layers paired (2i, 2i+1) share a turn count → **`n_layers` must be even** |

**Variables:** `a` (end-cap radius), `b` (centre → cap-centre length),
`coil_half_gap`, and one turn count per double-pancake pair.

**Current best design — 6 layers (3 double pancakes), 2026-07-31:**
tape = 0.3372 km, B_target = 10.49 T, hoop = 113 MPa, T-A box uniformity
= 0.495%. `a` = 26.0 mm, `b` = 31.4 mm, `coil_half_gap` = 13.7 mm,
`n_turns = [382,382,478,478,3,3]`, I_op = 196.0 A at 65% of local Ic.

**This is the first design validated against both a realistic
critical-current model and build tolerance.**

| metric | nominal | limit | across 15 jitter samples |
|---|---|---|---|
| B_target (Kim Ic) | 10.49 T | ≥ 10 T | 10.10–10.49 — **15/15 pass** |
| box uniformity (T-A) | 0.495% | ≤ 1% | 0.338–0.517 — **15/15 pass** |
| hoop stress | 113 MPa | ≤ 400 | 102–113 |
| bend radius | 8.075 mm | ≥ 7.5 | 7.545–8.434 |
| face gap | 3.40 mm | ≥ 3.0 | 3.00–3.84 |

> **Why it is deliberately not minimal.** Its predecessor
> (`a`=23.227 mm, `[329,329,411,411,2,2]`, 0.2596 km) reached 10.03 T —
> a 0.3% margin — and then failed build tolerance outright: **0 of 14**
> perturbed builds reached 10 T, 5 of 6 were out of spec on a clearance
> floor, and 2 violated the bend radius. The cause was structural: the
> search minimized tape subject to B ≥ 10 T and converged *exactly onto*
> the constraint. This design targets margin explicitly (B ≥ 10.3 T
> nominal; bend radius ≥ 7.5 mm under simultaneous −0.2 mm radius and
> +2% tape thickness). Tape thickness is the dominant build error and is
> asymmetric — on the predecessor `t+2%` alone failed uniformity (1.33%);
> here it is benign (0.356%).
>
> Roughly half the increase over the original 0.2235 km design is the
> realistic Ic model, half is build tolerance never previously budgeted.
> The tolerance half scales with the assumed ±0.2 mm / ±2% figures —
> tighter machining recovers much of it.

> **Ic extrapolation — five models, bias-corrected, agree (2026-08-03).**
> Long's maximum-entropy Beta model (*Entropy* 2013, 15(7), 2585,
> doi:10.3390/e15072585, Eq. 2: `Jc ~ b^(α−1)(1−b)^(β−1)`, `b = B/B_irr`)
> was implemented and hold-out tested. It does **not** beat the Kim model
> here (MAPE 5.46% per-angle / 14.26% with α,β shared, vs Kim's 4.14%) —
> the pinning force is still rising at 8 T, so B_irr is never pinned by a
> visible peak. The paper's method is sound; our data does not reach the
> field range that constrains it.
>
> Its value is as an independent check. Because the hold-out framework
> *measures* each model's bias, each B_target can be corrected by its own:
>
> | Ic model | bias | B_raw | B_corrected |
> |---|---|---|---|
> | flat clamp | +26.7% | 14.69 T | 10.77 T |
> | kim | −3.3% | 10.48 T | 10.82 T |
> | scaling:45 | −5.8% | 9.43 T | 9.97 T |
> | beta (Long) | −5.1% | 9.75 T | 10.25 T |
> | beta_shared (Long) | −13.0% | 8.76 T | 9.89 T |
>
> Five models spanning a 6 T raw range collapse to **9.89–10.82 T** once
> bias-corrected — including the flat clamp, from the opposite side. Best
> estimate: **B_target ≈ 10.3 ± 0.5 T**, clearing the 10 T floor. The
> biases were measured at a 1.6× extrapolation while the real use is
> 1.34×, so the correction is slightly over-generous — a consistency
> check, not a substitute for measured Ic data above 8 T.

**How to evaluate a design.** Two tools, and the difference matters:

| tool | speed | trust |
|---|---|---|
| `optimize/optimize_geometry.py` `evaluate()` — coarse uniform-J screen | ~5 s | `tape_km`, `B_target_T`, `hoop_MPa` are reliable. Its **`uniformity_pct` is NOT** — found wrong by up to ~10× and even anti-correlated with truth for compact coils. |
| `optimize/ta_validate.py` — full per-layer T-A solve, true box peak-to-peak | ~60–150 s | The ground truth. Run on any finalist. |

Never promote a design on the screen's uniformity number alone, and
cross-check near-boundary designs against an **independently generated
mesh** (a separate process) — gmsh is not bit-reproducible across
processes, and one design once swung 0.79 % → 2.19 % between meshes.

---

## Legacy external-team entry point (`optimize/evaluate.py`)

**Frozen 2026-07-14 contract, kept working and deliberately unchanged.**
It implements a *different* problem from the one above — maximize field
at a 1.15 safety factor over a 15×6 mm box — and reads its own dedicated
`EVALUATE_SAFETY_FACTOR`/`EVALUATE_TARGET_X_M`/`EVALUATE_TARGET_Y_M`
constants in `optimize/opt_config.py` so it does not drift when the
shared constants are repurposed by the internal search. Its uniformity
number comes from the same coarse Bean-state proxy flagged as unreliable
above.

**The problem it solves.**  Maximize the magnetic field in the target area, subject
to (1) never exceeding the tape's quench limit anywhere in the winding
(with a 1.15 safety factor) and (2) keeping the field uniform to < 1 %
peak-to-peak over the target box.  Mechanical stress limits exist in the
codebase but are EXCLUDED from this optimization by current project
direction.

**Design variables** (all other geometry is derived automatically):

| variable | meaning | type / bounds |
|---|---|---|
| `a` | end-cap radius [m] | continuous; > 0 (baseline 0.050) |
| `b` | centre → cap-centre length [m] | continuous; must exceed `a` (baseline 0.080) |
| `n_turns` | turns per pancake layer, top → bottom | list of integers ≥ 1; the list LENGTH (number of pancakes) may also vary (baseline `[500,500,500,400,400,250,100]`) |

Geometric feasibility (each layer's inner radius must stay positive:
`a + max(nᵢ)·t/2 − nᵢ·t > 0`) is checked inside the evaluator —
infeasible designs return `feasible: false` with objective `None`;
treat as a constraint violation, not an error.

**One evaluation = one file:** `optimize/evaluate.py`

```bash
conda run -n fenicsx-env python3 optimize/evaluate.py \
    --a 0.050 --b 0.080 --n-turns 500,500,500,400,400,250,100 --json
```

prints a brief human-readable summary and (with `--json`) the full
machine-readable result dict.  Or from Python:

```python
from evaluate import evaluate_configuration   # optimize/ on sys.path
r = evaluate_configuration(0.050, 0.080, [500,500,500,400,400,250,100])
r["objective_B_target_T"]   # maximize this
r["pass_constraints"]       # uniformity ≤ 1 % (screening-corrected)
r["feasible"]               # geometry valid
```

**Cost and parallelism.**  ~8–12 s per evaluation on one core-group
(a single small FEM solve — the problem is exactly linear in current, so
the quench current is a root-find on one solve and everything else is a
scaling).  Evaluations are fully independent: parallelise across
processes freely.  Baseline result for reference: a = 50 mm, b = 80 mm,
baseline turns → **B_target = 13.4 T at I_op = 339 A, uniformity
0.21 %, PASS**.

**One data caveat**  The tape's measured
critical-current data ends at 8 T; above that the model extrapolates.
The result dict reports `clip_fraction` (how much of the quench
evaluation relied on extrapolation) and `peak_conductor_B_T` — designs
with large values are *floor estimates*, not predictions, and the
optimum will likely sit in that regime until extended tape data arrives.
Consider reporting Pareto results (objective vs `clip_fraction`) rather
than a single winner.

Constants (target-box size, uniformity limit, safety factor) live in
`optimize/opt_config.py` as `EVALUATE_SAFETY_FACTOR`/`EVALUATE_TARGET_X_M`/
`EVALUATE_TARGET_Y_M` — dedicated constants kept equal to the values above,
frozen so this entry point's contract doesn't drift when the shared
`SAFETY_FACTOR`/`TARGET_X_M`/`TARGET_Y_M` values are repurposed elsewhere
(see the CMA-ES search below, which does exactly that internally).  The
full physics and assumption ledger is in [The physics,
explained](#the-physics-explained) at the bottom of this README.  A
batch/grid screening driver with plots also exists
(`optimize/optimize_geometry.py`) if you want a coarse map before running
your algorithms.

---

## Internal CMA-ES search (`optimize/cmaes_search.py`, 2026-07-21, updated 2026-07-30)

A separate, actively-evolving internal design study under a different
objective — **do not confuse this with the frozen `evaluate.py` handoff
above**, which is unaffected by it. Uses [pycma](https://github.com/CMA-ES/pycma)
(installed into `fenicsx-env` from conda-forge) to search continuously over
geometry variables plus the coil-to-coil gap and the per-layer turn counts,
instead of the grid in `optimize_geometry.py`.

**2026-07-23 — three practical manufacturing constraints added; the
previous champion ledger is obsolete.** Every design found before this
date predates all three of the following and is no longer valid/buildable
— see CLAUDE.md's "Practical manufacturing constraints" section for the
full reasoning.

**First genuinely validated design (6 pancake layers, i.e. 3 double
pancakes): tape = 0.2258 km, B_target = 10.00 T, hoop = 114 MPa, box
peak-to-peak uniformity = 0.83 % — a real, T-A-validated PASS** —
a=22.20mm, b=27.27mm, coil_half_gap=13.50mm,
n_turns=[285,285,379,379,2,2] (found 2026-07-24). **Superseded
2026-07-30** by the same geometry with the turn split shifted to
[295,295,369,369,2,2] — better on all four metrics; see the perturbation
study below. **This was the first
genuinely validated design of the entire optimization effort** — every
earlier "champion" (10, 8, and a rejected 4-layer design) turned out to
be an artifact of a broken proxy once actually checked against the real
30×6mm target box.

**2026-07-27 — widened search re-confirms this design as the best found
anywhere.** A follow-up search (`optimize/studies/day_search.py`) re-ran the
discrete n_layers outer loop (6, 8, 10, 12, and two new counts, 14 and
16) with a coil-radius floor tuned against real T-A checks, then
T-A-validated every winner. Every alternative came in at 3.1–4.5% true
box uniformity — 4 to 6× worse than the champion's 0.83%. See CLAUDE.md's
"2026-07-27" section for the full table. **The same run also surfaced an
important open risk on the champion itself: its B_target relies on an
optimistic Ic extrapolation above the measured 8 T tape data. Under a
conservative extrapolation instead, B_target drops from 10.00 T to
6.51 T.** This is now the top-priority open item — see [Known
limitations](#known-limitations) below.

**2026-07-30 — perturbation study: the champion is a real point in a
well-behaved landscape, but it is NOT a converged local optimum.**
`optimize/studies/perturbation_study.py` perturbed the champion by small,
*buildable* amounts along each axis independently and along all axes at
once — 23 candidates, each given a full T-A box-uniformity solve with 2
independent-mesh repeats (`visualization/perturbation_study.png`).

*What held up:* every solve converged, and repeat-to-repeat spread was
≤0.003 pp for 22 of 23 candidates (the champion itself reproduced at
0.828 %, matching the earlier re-check exactly). Every axis varies
smoothly and monotonically — none of the knife-edge mesh fragility that
invalidated an earlier 4-layer design. The design region is genuine.

*What did not:* the champion is **dominated** by a neighbour just 10
turns per pancake away —

| design | tape | B_target | hoop | box uniformity |
|---|---|---|---|---|
| champion `[285,285,379,379,2,2]` | 0.2259 km | 10.005 T | 114 MPa | 0.828 % |
| **`[295,295,369,369,2,2]`** | **0.2235 km** | **10.215 T** | **111 MPa** | **0.687 %** |

better on *all four* metrics simultaneously. 8 of 22 perturbations had
better uniformity than the champion. This is the expected consequence of
searching with no uniformity signal: CMA-ES stopped where the tape and
field constraints cornered it, and 0.83 % was luck in the narrow sense
that it happened to pass.

*Where the champion sits in the uniformity bowl:* on the **inner wall**
in `a` (minimum ≈ +1 mm outward, 0.487 %) and below the optimum in `b`
(minimum ≈ +1.5 mm, 0.591 %). Only `coil_half_gap` is genuinely optimal
— it is the steepest axis (≈ 0.7 pp/mm, monotone worsening upward) and
the 3 mm face-gap manufacturing floor pins it exactly at its best value.

*Tolerance sensitivity (practical concern):* all four all-axes jitter
samples — perturbations of ≲0.3 mm plus a few turns — **failed** the 1 %
target (1.09, 1.23, 1.25, 1.93 %). Stated plainly, that group is
**biased pessimistic**: the face-gap floor means jitter could only
*increase* gap, the single steepest axis. It is not a symmetric
tolerance estimate, but it does say assembly gap tolerance is tight in
the direction one can actually err.

**2026-07-30 — local polish (`optimize/studies/local_polish.py`): the
dominating neighbour is promoted, and the tape/uniformity tradeoff is
now quantified.** A tightly-scoped 1000-evaluation CMA-ES refinement was
warm-started from `[295,295,369,369,2,2]` with *proportional* step sizes
(5 % of `a`/`b`, 15 turns), then its six best distinct candidates were
T-A-validated alongside both references on the same pipeline:

| design | tape | B_target | hoop | T-A box p2p | verdict |
|---|---|---|---|---|---|
| **`[295,295,369,369,2,2]`** | **0.2235 km** | **10.215 T** | **111 MPa** | **0.688 %** | **PASS — new champion** |
| `[285,285,379,379,2,2]` (old) | 0.2259 km | 10.005 T | 114 MPa | 0.828 % | PASS |
| polish1 `[291,291,291,291,1,1]` | 0.1863 km | 10.001 T | 97 MPa | 3.660 % | FAIL |
| polish2 … polish6 (flat profiles) | 0.187–0.191 km | ~10.0 T | 97–98 MPa | 3.59–8.59 % | FAIL |

The search — which optimizes tape/field/hoop with **no uniformity term**
— reliably drove the turn profile toward *equal* pairs, buying a genuine
**17 % tape reduction** (0.1863 vs 0.2235 km) at 10 T and lower hoop
stress. Every one of those designs fails real box uniformity by 3.6–8.6×
the target. The champion's steep taper (295 → 369 → 2) is not incidental;
it is doing essential uniformity work, and **tape length and uniformity
are in direct conflict along this axis.** Both references reproduced
their known values exactly (0.828 %, 0.688 %), confirming the failures
are physics, not a pipeline artifact.

Net result: the perturbation study's dominating neighbour is confirmed
and promoted (a −1.1 % tape, +2.1 % field, −3 MPa, −0.14 pp uniformity
improvement over the old champion, all at once), and no flatter profile
is admissible without a uniformity-aware objective.

**Important — the coarse optimizer screen's `uniformity_pct` metric was
found unreliable by up to ~10x**, and a same-day replacement heuristic
(penalizing peak per-layer turn concentration) was *also* found wrong
once real box uniformity was measured: the design with the best on-axis
SCIF (10 layers, 1.37%) has the *worst* true box uniformity of every
design tried (9.18%, vs. 0.44-1.06% for 4/6/8 layers) — on-axis SCIF is
a near-cancelling sum that does not represent the real target.

An initial follow-up concluded that coil radius `a` cleanly tracked true
box uniformity ("bigger = better", since the target box is a fixed size
regardless of coil scale). **That conclusion was wrong and has been
retracted** — it came from comparing 5 designs that differed in `a`
*and* layer count *and* gap *and* turn distribution at once. Isolating
`a` properly (translating the whole coil radially, everything else held
fixed) gives a **V-shaped bowl with an interior minimum**, not a
monotone trend: moving `a` either smaller *or* larger degrades
uniformity. There is still no known fast proxy for box uniformity.
`cmaes_search.py`'s fitness function currently
carries **no uniformity signal at all** — deliberately, rather than a
third guessed proxy. **See CLAUDE.md's "Box uniformity is the real
target" section for the full investigation, 5-design data table, and
current methodology** before trusting any `uniformity_pct` number from
this optimizer, past or future.

1. **Minimum bend radius 7.5 mm** — REBCO tape cracks below this. Raised
   from an arbitrary 3 mm bore-clearance value that had no material basis.
   Every design found before today sat right at that old 3 mm floor, so
   all are now significantly infeasible on this alone.
2. **Double-pancake construction** — the search now optimizes **one turn
   count per PAIR of adjacent layers** (layers 2i and 2i+1 forced equal),
   not one per layer, since each double pancake must be wound as a single
   continuous piece (outer edges already match; the inner ends are
   joined). This makes **odd `n_layers` physically unbuildable**
   (`cmaes_search.py` asserts on it) — 3, 5, 7, and 9 layer designs are
   eliminated outright, including the 9-layer design that had been the
   most promising open lead just before this change.
3. **Turn-count floor removed** (50 → 1, the true physical minimum —
   `params.py` only asserts `n ≥ 1`). The old 50 had no material basis,
   and roughly half the layers in every previously-converged design sat
   pinned exactly on it.

A 7×14×1 mm sensor array also needs clearance, but **needs no separate
constraint** — verified directly against the racetrack geometry: each
straight section is `2L` long (not `L`), so even the tightest prior
design has ~12–16 mm of straight bore, well over the 7 mm sensor
dimension; the 14 mm dimension needs bore diameter `2×a_inner_min ≥ 14mm`,
automatically satisfied once constraint 1 holds (2×7.5=15mm); the 1 mm
dimension fits in the existing 3 mm coil-to-coil face gap.

**Objective:** minimize tape length, subject to:

| constraint | limit |
|---|---|
| mean \|Bz\| over the target box (30×6 mm) at I_op | ≥ 10 T — **evaluated with a validated Ic extrapolation, not the flat clamp** |
| peak-to-peak uniformity over that box (SCIF-corrected) | ≤ 1% |
| hoop stress, end-cap curved sections only (B×J×bending radius) | ≤ 400 MPa |
| coil-to-coil **face-to-face** clearance (not `coil_half_gap` itself) | ≥ 3 mm |
| innermost turn of any layer (bend radius) | ≥ 7.5 mm |
| operating point | worst-margin cell at 50–60% of local Ic (`SAFETY_FACTOR = 1.818`) |

Delamination stress is still computed/reported but not enforced (current
project direction — see `optimize/opt_config.py` for the reasoning).

**Variables:** `a`, `b`, `coil_half_gap` (previously fixed at 30 mm — now
free, so the optimizer can bring the two coils closer as long as the 3 mm
face-gap constraint holds), and **one `n_turns` value per double-pancake
pair** of layers (so an `n_layers`-layer design has `n_layers/2` free turn
variables, each applied to both layers in its pair) — `n_layers` itself
must be even. `a` and `b` are **intentionally unbounded** — a first
bounded run pinned both at their box edges, so per current project
direction only the real physical limits apply (non-degenerate straight
section, positive inner-radius clearance ≥ 7.5mm bend radius), enforced
as a smooth penalty rather than a search box.

```bash
conda run -n fenicsx-env python3 optimize/cmaes_search.py
```

~5–10 s per evaluation (one coarse-mesh FEM solve). Outputs
`optimize/runs/cmaes_results.csv` (this run's best design), `optimize/
runs/cmaes_history.csv` (this run's full history — overwritten each run), and
five figures in `visualization/`: `cmaes_convergence.png`,
`cmaes_constraints.png`, `cmaes_variables.png`, `cmaes_overview.png` (all
this-run-only), plus `cmaes_param_map.png`.

**Cumulative log across runs:** every run also appends to
`optimize/runs/cmaes_all_evaluations.csv` (never overwritten, each row tagged
with the run's start timestamp) — this is the persistent record of every
design point ever evaluated, good or bad, across every run of this search.
`cmaes_param_map.png` is built from that cumulative file, not just the
latest run, so it shows the full explored region to date (colored by
outcome, one marker shape per run) — check it before starting a new run to
avoid re-exploring territory already shown to be infeasible or dominated.

As with the grid screen, this uses the fast coarse uniform-J mesh — verify
the eventual finalist with the full per-layer T-A solver before treating it
as final.

---

## Pipeline

```bash
# Baseline uniform-J solve (single current)
conda run -n fenicsx-env python3 solve/solve.py

# Uniform-J current sweep + quench analysis
conda run -n fenicsx-env python3 sweep/solve_sweep.py
conda run -n fenicsx-env python3 sweep/quench_sweep.py

# T-A screening-current solve at I_design
conda run -n fenicsx-env python3 solve/ta_solve.py

# T-A sweep 150–400 A → all ta_* figures + CSV  (~25 s solver time)
conda run -n fenicsx-env python3 solve/ta_sweep.py

# Post-process saved T-A results (no re-solve)
conda run -n fenicsx-env python3 solve/ta_postprocess.py

# Uniform-J visualisation
conda run -n fenicsx-env python3 visualization/plot_fields.py
conda run -n fenicsx-env python3 visualization/plot_3d.py
conda run -n fenicsx-env python3 visualization/field_uniformity.py
```

**Validate one candidate design's true box uniformity** (the check that
matters — see [the design problem](#the-design-problem--start-here)):

```bash
TA_VALIDATE_JSON='{"label":"champion","a":0.022227,"b":0.027268,
  "coil_half_gap":0.013500,"n_turns":[295,295,369,369,2,2],
  "I_design":223.88}' \
/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 optimize/ta_validate.py
```

> **Launch long runs via the environment's python binary directly, not
> `conda run`.** `conda run` buffers *all* subprocess output until the
> process exits — confirmed independent of anything in the script, and
> not fixable with `--no-capture-output`/`--live-stream` or
> `sys.stdout.reconfigure()`. Use
> `/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 <script>`
> for anything you need to monitor mid-run (every CMA-ES search, sweep
> or study). The short commands above are fine either way.
>
> **Do not pause a run with `SIGSTOP` if it was launched as a Claude
> Code background task** — the resulting exit code is read as a failure
> and the whole process tree is reaped, so the run ends rather than
> suspending. `SIGSTOP`/`SIGCONT` remains safe for a `disown`ed shell
> job (verified on a 1.5 h pause with zero progress lost).

## Repository layout

```
Racetrack_v4/
├── params.py                  ← SINGLE source of truth; edit this first
├── mesh/build_mesh.py         ← gmsh mesh (eighth-symmetry domain)
├── physics/
│   ├── current_source.py      ← tangent/normal/arc-length helpers, symmetry expansion
│   ├── ic_model.py            ← IcModel + NValueModel (Ic(B,θ), n(B,θ) from CSV)
│   ├── coil2_field.py         ← Biot-Savart reference field (both coils)
│   └── *.csv                  ← Shanghai Superconductor 20 K tape data (0–8 T)
├── solve/
│   ├── solve.py               ← uniform-J A-form FEM solve
│   ├── ta_solve.py            ← T-A Picard solver (screening currents)
│   ├── ta_sweep.py            ← T-A current sweep + all ta_* figures
│   ├── ta_postprocess.py      ← plots from saved racetrack_ta_fields.npz
│   └── diagnostics.py         ← solver logging / residual checks
├── sweep/
│   ├── solve_sweep.py         ← uniform-J field sweep
│   └── quench_sweep.py        ← per-cell quench currents + performance summary
├── optimize/                  ← design search (see below)
├── validation/                ← Biot-Savart cross-check, mesh convergence, …
└── visualization/             ← output figures (ta_* = T-A results)
```

`optimize/` is split three ways (reorganized 2026-07-30) — the top level
holds only the reusable tools, so one-off study scripts and run
artifacts don't bury them:

```
optimize/
├── opt_config.py              ← the ONLY file to edit for a new search
├── optimize_geometry.py       ← coarse uniform-J screen (~5 s/design)
├── cmaes_search.py            ← the CMA-ES search itself
├── evaluate.py                ← frozen external-team entry point
├── ta_validate.py             ← full T-A box-uniformity ground truth
├── studies/                   ← one-off orchestrators, each a historical run
│   ├── day_search.py                 ← 2026-07-27 widened 6–16 layer search
│   ├── double_pancake_search.py      ← 2026-07-23 re-search under new constraints
│   ├── perturbation_study.py         ← 2026-07-30 champion robustness study
│   ├── sweep_n_layers.py             ← discrete layer-count outer loop
│   ├── sweep_restarts.py             ← random cold-start sweep
│   ├── overnight_refinement.py, focused_refinement_6_9.py, floor_test.py
│   └── regenerate_champion_plots.py  ← rebuild cmaes_*.png from the master log
└── runs/                      ← every log + CSV, grouped by study
    ├── cmaes_all_evaluations.csv     ← CUMULATIVE master log, append-only,
    │                                   ~101k evaluations, never overwritten
    ├── cmaes_results.csv / cmaes_history.csv   ← latest run only (overwritten)
    ├── perturbation/, day_search/, double_pancake/, sweep_n_layers/,
    └── sweep_restarts/, overnight/, floor_test/, n4_trial/, n8_focused/, …
```

Every artifact path flows through constants in `opt_config.py`
(`CMAES_MASTER_LOG`, `CMAES_OUT_CSV`, `CMAES_OUT_LOG`, `OUT_CSV`), so
the visualization scripts needed no changes and future moves only touch
that one file.

---

## Current configuration (from params.py)

**`params.py` currently holds the margin/jitter-aware champion design
(as of 2026-08-03), not the original hand-picked baseline.** This
supersedes the `[295,295,369,369,2,2]` champion narrated in "Internal
CMA-ES search" above (that section is a historical record of how the
search evolved through 2026-07-30 and is kept as-is; the design it ends
on was itself later superseded — see [The design problem](#the-design-problem--start-here)
at the top of this README and `CLAUDE.md`'s "Current design" section for
the reasoning behind the final margin-aware version). The baseline
values (`a=50mm`, `b=80mm`, 7-layer `[500,500,500,400,400,250,100]`
stack) are kept below for reference — they're what `evaluate.py`'s
worked example and the physics-explanation section further down use —
but are no longer what a fresh solve/visualization run in this repo
actually produces.

| Parameter | Current champion (params.py, as of 2026-08-03) | Original baseline (evaluate.py example) |
|---|---|---|
| `n_turns` | `[382, 382, 478, 478, 3, 3]`  (6 layers = 3 double pancakes, top→bottom) | `[500, 500, 500, 400, 400, 250, 100]`  (7 layers) |
| `n_turns_total` | 1726 | 2650 |
| `a` / `b` | 26.0 mm / 31.4 mm | 50 mm / 80 mm |
| `t` / `w` | 75 µm / 4 mm  (tape pitch Λ / tape width) | same |
| `delta_SC` | 1 µm  (REBCO superconducting layer thickness) | same |
| `I_design` | 196.0 A/turn (65% of local Ic under the Kim Ic(B) model) | 200 A/turn |
| `coil_half_gap` | 13.7 mm  (face-to-face gap 3.40 mm nominal, 3.00–3.84 mm across 15 jitter samples) | 30 mm |
| Tape length | 0.3372 km | ~1194 m |
| B_target @ I_op | 10.49 T nominal, quoted as **~10.5 ± 0.5 T** of Ic-model uncertainty (Kim model, hold-out MAPE 4.1% — the best-validated extrapolation above the measured 8 T ceiling; see [Ic(B,θ) data — limitations](#icbθ-data--limitations)) | 13.4 T @ I_op=339A |
| Box peak-to-peak uniformity | 0.495% (T-A validated PASS; 0.338–0.517% across 15 jitter samples, 15/15 pass) | 0.21% |
| Hoop stress | 113 MPa (102–113 MPa across jitter samples) | — |
| `ramp_duration` | 600 s  (ramp 0 → I; sets screening-current depth) | same |
| `mesh_z_grading` | `[0.075, 0.15, 0.55, 0.15, 0.075]`  (graded sub-slabs per tape width: 0.3 mm edge cells, coarse bulk) | same |
| T-A sweep range | 150–400 A in 25 A steps (`SWEEP_CURRENTS` in ta_sweep.py) | same |

This is the first design in the project's history validated against
**both** a realistic critical-current model **and** build tolerance
(±0.2mm on a/b/gap, ±2% on tape thickness, exact turn counts) — its
predecessor reached 10.03 T with only 0.3% margin and then failed
catastrophically under the same jitter test (0/14 builds reached 10 T).
See `CLAUDE.md`'s "Current design" section for the full margin-search
reasoning.

All layers share the same outer radial edge `a_out`; the inner edge of
layer i is `a_out − n_i·t`. The stack is centred at z = 0, one tape-width
`w` per layer. Change only `n_turns` in params.py to try a new stack —
everything else is derived (call `params.recompute_derived()` after
mutating any of `a`/`b`/`t`/`w`/`n_turns` programmatically).

---

## Eighth-symmetry FEM domain

The mesh covers the octant (x ≥ 0, y ≥ 0, z ≤ coil_half_gap) — an 8×
DOF reduction:

| Plane | BC | Meaning |
|---|---|---|
| x = 0, y = 0 | PEC  n×A = 0 | quadrant mirrors of the racetrack |
| z = coil_half_gap | PMC  n×H = 0 (natural) | same-polarity image = coil 2 (Helmholtz pair) |

Coil 2's field is therefore included automatically in the FEM. The coil
cells in the mesh are **one quadrant of coil 1** — any quantity summed over
them (Biot-Savart integrals, quench statistics over the full magnet, …)
must expand to the full system: 4 quadrants × 2 coils. For on-axis Bz every
mirror piece contributes equally (Bz is even under all three mirrors).
`dB_bore_from_dJ()` in ta_solve.py and `expand_to_full_domain()` in
current_source.py implement this.

---

## T-A screening-current model (ta_solve.py)

The T-A formulation homogenises the REBCO winding and resolves the
**non-uniform current distribution** inside the tape that a uniform-J model
ignores. Two coupled fields:

- **T** — current vector potential (scalar here, CG1). The SC-layer current
  density is `J_SC = ∇T × n̂`, with n̂ the tape broad-face normal.
  Transport current is imposed through Dirichlet BCs on the tape edges:
  `T = ±I/(2δ_SC)` at each tape's top/bottom faces. Two modes:
  **per-layer** (default, `params.ta_per_layer = True`): one T problem per
  z-layer, each with its own edge BCs and local ρ(B); converges cleanly.
  Adjacent tapes need opposite T values on their shared interface nodes,
  so the layers are solved as separate systems.  **replicated** (legacy,
  `ta_per_layer = False`): T solved only in the central z-layer; its J
  copied to the other 6 layers via a KD-tree (x, y, z-within-tape) lookup
  — ~2× faster per iteration but an approximation for the asymmetric
  stack, and its Picard plateaus around |ΔB|/|B| ≈ 2–3e-4.
- **A** — magnetic vector potential (N1curl), driven by the homogenised
  source `J_s = (δ_SC/Λ)·J_SC` with Λ = t (tape pitch).

The tape's E(J) power law enters as a field-dependent resistivity
`ρ = (E_c/Jc)·(|J|/Jc)^(n−1)`, with Jc(B,θ) and n(B,θ) interpolated from
the manufacturer CSV, floored at the critical-state value (`ta_eps_reg`).
One implicit-Euler (BDF1) step of Faraday's law from the zero-field-cooled
state to t = `ramp_duration` gives the end-of-ramp screening state; the
T-equation RHS is `−B_n/Δt`.

**Picard loop** (fixed point in ρ): solve T (linear, frozen ρ) → relax
`T ← (1−α)T_old + αT_new` → J from T → solve A → B → update ρ(J,B) → repeat
until `|ΔB|/|B| < ta_picard_tol`. Two-phase relaxation (α = 0.30 → 0.08)
suppresses a period-2 limit cycle. ~26–31 iterations cold, ~9–11 warm.

**SCIF**: `ΔJ = J_TA − J_unif` with `J_unif = I/(δ_SC·w)·t̂` (per-tape,
NOT divided by n_layers), then a cell-wise Biot-Savart sum of
`(δ_SC/Λ)·ΔJ` over the full mirrored system with exact DG0 cell volumes
gives ΔB at the bore midplane.

### Solver efficiency (reworked 2026-07-09)

- The A-problem matrix is constant → assembled once, **MUMPS factorised
  once**, reused for every Picard iteration, every sweep current, and the
  uniform seed solve (only the RHS is reassembled — `_solve_A()`).
- The T-problem matrix changes with ρ each iteration → stays a
  LinearProblem (small scalar CG1 system).
- **Warm start** across sweep currents: T seeded from the previous
  converged state scaled by I_new/I_old (the T BCs are ∝ I). Verified
  bias-free: warm and cold agree to 0.01 % in SCIF at tol = 1e-4.
- Full 11-point sweep: **~25 s solver time** (was minutes to ~28 min).

---

## Results at 200 A, Δt = 600 s (unified dataset, 2026-07-11)

- Tape: Shanghai Superconductor **High Field Low Temperature 20 K**
  (one CSV pair for the whole pipeline — quench AND T-A).
  Ic(0 T) = 1976 A, Ic(5 T, 0°) = 546 A per 4 mm tape → i = I/Ic ≈ 0.37
  at 200 A (sub-critical).  Data covers 0–8 T only.
- Uniform-J bore Bz = −7.62 T (76 % of the 10 T target)
- **Bore SCIF ΔBz = +82.0 mT = 1.08 %** (per-layer T-A, graded mesh,
  converged k ≈ 80; reproducible to <0.01 mT).  At the 1 % uniformity
  target this is design-relevant — the spatially-resolved ΔB map over the
  bore box is the next required analysis.
- Mechanical stress screen (`validation/mechanical_stress_check.py`):
  cap hoop max 233 MPa (allow ~500), transverse delamination tension
  12 MPa (allow ~30), leg line load 1.3 MN/m, coil-coil attraction
  273 kN.  OK at 200 A, but stress ∝ I² → active constraint near 10 T.
- Quench limit 219.8 A/turn (predates the dataset switch — rerun).

**History warning:** every SCIF number before 2026-07-10 was
artifact-dominated ("+1.95 %", an interim "~16 %", the "≤0.3 %" bound
from the superseded datasets, any old `ta_sweep_results.csv`) — see
CLAUDE.md for the bug history (boundary-cell replication artifact,
missing quadrant mirrors, wrong cell volumes) and for the 2026-07-11
Picard robustness rework (fixed α, smooth j/jc floor, ρ relaxation,
observable-stall convergence criterion).

---

## NI (no-insulation) transient work — `circuit/` and `transient/`

The coil is committed to no-insulation (NI) winding. At DC steady state
the radial current vanishes, so the design above is unaffected — this
adds a transient (ramp/discharge) constraint on top, not a change to it.

- **`circuit/` (Phase A, lumped DCN circuit model) — VALIDATED.** Reduced-
  order per-turn mutual-inductance model with a DCN ladder solved via
  `scipy.integrate.solve_ivp(method="BDF")`. Matches the production
  Biot-Savart field path to 0.18% median, reproduces a published benchmark
  (He et al. 2025) to 4.2%, and gives the champion's self-inductance
  (419.7 mH), stored energy (8.07 kJ @ 196A), and `tau = 1330/rho_c` scaling.
- **`transient/` (Phase B, T-A + circuit closure) — EXPLORATORY, use with
  caution.** Adds the NI circuit closure to the T-A Picard/Newton solve
  for hysteretic loss and current redistribution during a ramp. The base
  T-A solver was only ever validated at a single implicit step
  (`dt=600s`); genuine multi-step time-marching at shorter `dt` needed a
  from-scratch relaxation-parameter fix (`alpha=(0.03,0.01)`, found and
  validated 2026-08-06) after five other approaches failed. As of
  2026-08-07: validated across `dt` in [60s,600s] and `I` in [19.6,196]A,
  including a genuine multi-step ramp and reliable multi-threaded
  execution (0.001-0.03% run-to-run) — but a new, unresolved ~2.2% gap
  exists between single- and multi-threaded runs, `dt=30s` is a
  confirmed hard failure boundary, and the NI circuit closure itself is
  still out of scope (insulated limit only so far).

**Do not trust any `transient/` SCIF or ramp-trajectory number without
first reading `CLAUDE.md`'s "NI transient work" section** for exactly
what's validated (or retracted) at that specific `dt`/schedule — several
early results in this effort were later found to be false stalls or
process-launch nondeterminism, not real physics. Full dated arc in
`docs/HISTORY.md` and `transient/validation/nondeterminism_investigation_2026-08-05.md`.

### How NI leakage current is modeled (`transient/ni_circuit.py`)

With no turn-to-turn insulation, the tape-to-tape contact forms a
resistive path in parallel with the intended azimuthal (spiral) path —
at every turn boundary, transport current can either continue along the
spiral or "leak" radially through the contact resistance. KCL at every
turn: `i_turn + I_r = I(t)`. `circuit/dcn.py` (Phase A) models this as a
lumped ladder network; `ni_circuit.py` (Phase B) couples the same
relation directly into the T-A field solve, so the induced field
(`E_i = -(A-A_prev)·t̂/dt`) and resistive field (`E_p = ρ·(J_sc·t̂)`) that
drive the leakage split come from the actual per-bin screening-current
solution rather than a lumped inductance matrix:

```
I_r,k = (E_p,k - E_i,k) * l_turn,k / R_ct,k        # radial leakage
I_z,k = I(t) - I_r,k                                # remaining azimuthal
                                                     # current → T Dirichlet BC
```

Faster ramps raise `E_i`, diverting more current radially — this is why
the coil's actual (spiral) field lags the supply current during a fast
ramp (`visualization/plot_ramp_field_animation.py`) and is the physical
basis of NI coils' self-protecting behaviour. Implementation note: `I_z`
must be solved as an exact 48×48 linear system per Picard iteration
(only `E_p` Picard-lagged) — an earlier explicit/elementwise update
diverged, because the per-turn mutual-inductance matrix is dense and far
from diagonally dominant (GMD ≈ 0.91 mm ≫ the 75 µm turn pitch), which
is exactly the case classical Jacobi-iteration theory says won't
converge. The closure itself is validated only at low scope so far
(`ni_closure_smoke_check.py`: 3 steps, up to 58.8 A / 30% of I_design,
`dt=60s` only) — every ramp/hysteresis result below still runs in the
**insulated limit** (closure off), not with leakage active.

### Ramp-up power analysis (2026-08-08) — `circuit/power_ramp.py`, `transient/validation/ta_quench_margin_check.py`

For a supply delivering constant power (not a prescribed current
schedule), Phase A (DCN) finds **no ramp-speed-dependent quench
constraint at all** — `DCN.terminal_voltage()` is exact and linear in
`I`, so constant power reduces to an algebraic quadratic at every ODE
step, and the per-turn-group margin sits at the DC steady-state floor
(2.225) at every tested power (~2 W to >4000 W, rho_c ∈ {30,100,400}
µΩ·cm²) — the transient state is never less safe than the final DC
state. Expected: DCN only resolves inter-turn (radial-leakage) current
sharing, not intra-tape screening-current concentration, so it can't see
the mechanism that could actually bound P.

A T-A cross-check tells a different story, though: comparing per-cell
`Jc(B,θ)` against the actual local in-plane current density finds the
**local** quench margin is materially tighter than the uniform-J
assumption every other check in this project relies on — present at
**any** ramp speed, not just fast ones:

| | worst-cell margin | cells < 1.0 | cells < 1.538 (design threshold) |
|---|---|---|---|
| uniform-J | 1.600 | 0/4463 | 0/4463 |
| T-A, `dt=600s` (1 step) | 0.709 | 1200/4463 (27%) | 2054/4463 (46%) |
| T-A, `dt=60s`×10 (step 9) | 0.687 | 1500/4463 (34%) | 2153/4463 (48%) |

Both T-A methods agree on *where* the worst cell is (the coil's own
peak-field cell, ~26.5 mm radius, |B|~11.3 T) via two independent code
paths. **Open judgement call, deliberately not resolved:** this
project's quench criterion is the E_c=1µV/cm engineering Ic definition,
and some cells sitting at/above it near a flux-penetration front is
normal Bean/critical-state behaviour (`ta_solve.py`'s own solver assigns
a finite, not infinite, critical-state resistivity there as standard
operation). Whether this is a real, previously-undetected gap in the
champion's stated quench margin, or expected physics the uniform-J
65%-of-Ic margin was never built to catch, is unresolved — no "fastest
safe ramp power" number should be treated as settled until it is. Report
figures: `visualization/ramp_power_report/` (`00_summary_dashboard.png`
ties every number above together).

### Ramp-down / hysteresis-loop cross-check (2026-08-08 to 08-10) — EXPLORATORY, latest work, not yet folded into `docs/HISTORY.md`

`transient/validation/full_ramp_up_down_run.py` runs the T-A transient
solver (insulated limit, `alpha=(0.03,0.01)`, `dt=60s`,
forced-full-length per step) through repeated 39.2→196→~2 A up/down
cycles, tracking the bore SCIF — the closest thing this project's
transient solver has to a magnetization — as a physical sanity check
against the analytically-derived Bean critical-state hysteresis loop
(`visualization/plot_hysteresis_loop.py`, an independently re-derived
1D Bean slab model, deliberately not the exact Norris self-field
solution — see that script's docstring for why).

Run out to **three full cycles** (`full_ramp_3cycle.npz`, all 30 steps
numerically clean, `dB_rel` 0.003–0.024): the loop does not close after
one pass, but the gap between successive cycles **shrinks** by a
consistent ~0.65× ratio at both the peak (757.0 → 647.6 → 576.1 mT) and
the remanent point (−367.1 → −464.8 → −529.8 mT) — a decaying-ratio
signature consistent with converging toward a stable minor loop, not
constant drift or a runaway instability (though 2 successive deltas is
not a tight extrapolation, and the loop still hasn't closed within 3
cycles). See `visualization/plot_hysteresis_comparison.py`'s docstring
for the full reasoning and `visualization/ramp_power_report/06_hysteresis_loop_vs_simulation.png`
for the figure. As with everything else in this section: insulated
limit only (no NI leakage closure active), and this specific multi-cycle
result predates any write-up in `docs/HISTORY.md` — treat it as
provisional pending that.

---

## Ic(B,θ) data — limitations

The manufacturer CSV covers **0–8 T** at ~20 K. Peak winding fields exceed
this, so Ic above 8 T is extrapolated/clamped (~48 % of evaluations at
higher currents) — quench predictions there are a floor estimate. Extended
measurements (15–20 T) are planned; until then apply an extra safety factor.

## Known limitations

- **CONFIRMED 2026-07-31 — the champion does NOT reach 10 T under a
  physical Ic extrapolation.** `optimize/studies/ic_scaling_law_test.py`
  fits the pinning-force scaling law `Jc = C·B^(p-1)(1−B/B_c2)^q` per
  angle to the measured 0–8 T data (p ≈ 0.61–0.66, sub-1 % RMS) and uses
  it above 8 T. B_target at the target box:

  | model | 55 % Ic | 60 % Ic | 65 % Ic |
  |---|---|---|---|
  | flat clamp (current default, optimistic) | 10.21 T | 11.32 T | 12.42 T |
  | scaling law, B_c2 = 25 T | 6.95 | 7.81 | 8.67 |
  | scaling law, B_c2 = 45 T | 7.18 | 8.06 | 8.94 |
  | scaling law, B_c2 = 100 T | 7.28 | 8.17 | 9.06 |

  Even relaxing the operating point to 65 % of Ic, the design reaches only
  **~8.7–9.1 T**. The B_c2 band is tight (~0.4 T), so this is not an
  artifact of the unconstrained-parameter problem — B_c2 is *fixed* at
  three values precisely because q and B_c2 are degenerate over a 1–8 T
  fit window (a free fit picks physically meaningless parameters; see the
  script's docstring). Naive turn-scaling is an inefficient fix: doubling
  the tape buys only ~+1 T, because more turns push `a` outward via the
  bend-radius floor and raise the peak field, lowering I_op. The design
  needs re-optimization under this Ic model, not just more tape.
- **Superseded framing, kept for context — found 2026-07-27.** Every Ic lookup in this project defaults to
  `clip_B=True`, flat-clamping Ic to its measured 8 T value for any cell
  above that field — this is *optimistic*, not conservative, since Ic
  decreases with B in the measured range. Re-evaluating the champion's
  fixed geometry (`optimize/studies/day_search.py` Phase C, `ConservativeIcModel`)
  under a conservative linear continuation of the measured 8 T slope
  instead drops B_target from 10.00 T to **6.51 T (-34.9 %)**, below the
  design floor. 11.8 % of the champion's own quench-point Ic evaluations
  already clip to the 8 T boundary — not a remote edge case. Resolve by
  extending the Ic/n-value measurement dataset above 8 T, or by
  re-optimizing under the conservative extrapolation (see CLAUDE.md's
  2026-07-27 section for the one extra data point gathered so far: at
  safety factor 1.3 the same geometry reaches 14.71 T, clip_frac 0.231).
- **HIGH, found 2026-07-30 — the champion is not a converged local
  optimum, and its uniformity margin is thin against build tolerance.**
  The perturbation study (`optimize/studies/perturbation_study.py`, 23
  designs × full T-A) found a neighbour, `n_turns=[295,295,369,369,2,2]`,
  that beats it on tape *and* field *and* hoop stress *and* uniformity
  simultaneously; 8 of 22 perturbations had better uniformity. Separately,
  all four all-axes jitter samples (≲0.3 mm plus a few turns) exceeded the
  1 % uniformity target — biased pessimistic, since the face-gap floor
  lets jitter only *increase* gap, the steepest axis (≈0.7 pp/mm), but it
  means **assembly gap tolerance is tight**. Re-optimize locally before
  treating this geometry as final; the obvious untested candidate is
  `turns_shift_in` scaled slightly down to land exactly on 10 T.
- **Ic dataset inconsistency** (see above) — resolve before trusting
  operating-point conclusions
- **CSV ceiling at 8 T** — quench limit unreliable above that field
- **Homogenised winding** — no individual-tape quench propagation
- **Single BDF1 step** — end-of-ramp snapshot, no full ramp history
- **On-axis bore SCIF is a near-cancelling sum** — the robust statement
  is the ≤0.3 % bound.  With the graded default mesh the sub-critical
  value is converged to ≈ +17 ± 1 mT at 200 A
  (`validation/ta_z_grading_study.py`: graded matches brute-force
  uniform nz=5 within 4 % at half the DOFs); screening forms a ~1 mm
  current-reversal zone at one tape edge that coarser meshes truncate
- **Layer replication (legacy mode, `ta_per_layer = False`)** — copied
  pattern ignores layer-to-layer field variation and its Picard plateaus
  at |ΔB|/|B| ≈ 2–3e-4; at nz ≥ 3 it agrees with the per-layer default
  within ~10 % (validation/ta_per_layer_comparison.py)
- **1/8 symmetry assumes equal-sense coils** (Helmholtz pair) and that the
  screening pattern shares the transport current's mirror symmetry
- **RESOLVED 2026-07-24 — bore-box homogeneity is now computed properly**
  (it was previously only the on-axis SCIF point). `ta_validate.py`
  evaluates `dB_bore_from_dJ()` over a grid spanning the real 30×6 mm
  target box. This mattered enormously: on-axis SCIF and true box
  peak-to-peak are *anti*-correlated across the designs tested — the
  design with the best on-axis number (10 layers, 1.37 %) had the worst
  box uniformity (9.18 %). **Always use the box metric.**
- **RESOLVED 2026-07-22 — single-filament Biot-Savart broke down at small
  coil scale:** `physics/coil2_field.py`'s original
  `compute_both_coils_field()` treats the whole winding as one filament at
  radius `a` — valid only when the winding-pack cross-section is small
  compared to `a`/`coil_half_gap` (true at the original ~50-80 mm scale,
  false for the much smaller CMA-ES-optimized coils above, a≈13-25 mm).
  Fixed via `compute_both_coils_field_multilayer()` (resolves each layer's
  own z-center and radial sub-filament group, matching
  `optimize_geometry.py`'s own approach) — every near-coil field
  evaluation in the repo (`visualization/plot_fields.py`,
  `visualization/field_uniformity.py`, `sweep/quench_sweep.py`,
  `solve/ta_postprocess.py`, `solve/ta_sweep.py`, `solve/ta_solve.py`) now
  uses it. `field_uniformity.py` also now matches the optimizer's exact
  30×6 mm target box and applies the same Bean-state SCIF correction —
  verified result went from a spurious FAIL (6.74 %) to a PASS (0.56 %)
  closely agreeing with the optimizer's own reported 0.68-0.94 % range.
  `optimize_geometry.py` itself was already correct (built its own
  multi-filament sum from scratch) and needed no change.
- **RESOLVED 2026-07-27 — coil 2 was mirrored incorrectly in
  visualization code (picture-only bug, no physics numbers affected):**
  `visualization/plot_3d.py`'s `_expand_to_full_system()` and several
  copies of the same pattern (`plot_fields.py`'s layer-shading loop,
  `plot_field_poster.py`) placed coil 2 by *translating* coil 1's
  geometry by `+2·coil_half_gap` instead of *mirroring* it about the
  midplane — correct only for a palindromic layer stack, which the
  champion's `[285,285,379,379,2,2]` is not, so figures silently drew
  coil 2's layers in the wrong relative order (thin layer appearing at
  the wrong face). `physics/coil2_field.py`'s
  `compute_both_coils_field_multilayer()` — the function every real
  design number (B_target, uniformity, tape optimization, T-A SCIF)
  actually goes through — was always correct. Fixed with a shared
  `_mirror_z(z, g) = 2·g − z` helper; every affected figure regenerated.

---

# The physics, explained

This section is the complete story of what this repository computes, how
each piece works, why we trust it, and where the biggest assumptions
live.  It is written to be read start-to-finish.

## 1. The design problem

Two identical racetrack-shaped coils face each other across a 60 mm gap
(a Helmholtz-like pair).  Each coil is wound from REBCO
high-temperature-superconductor tape — 4 mm wide, 75 µm thick per turn —
stacked into 7 pancake layers with different turn counts.  Between the
coils, at the midplane, sits the *target area*: a small box where the
experiment happens.  The design must deliver **10 T** there, uniform to
**< 1 %** across the box, without ever exceeding the tape's critical
current anywhere in the winding (quench), and without breaking the tape
mechanically.  The free design variables are the cap radius `a`, the
length parameter `b`, and the per-layer turn counts `n_turns`.

Four distinct pieces of physics decide whether a design works, and the
code implements them as four stacked models plus an optimizer that ties
them together.

## 2. Layer 1 — Magnetostatics (where the field comes from)

**Model.** Maxwell's magnetostatics with no magnetic materials:
∇×(1/μ₀ ∇×A) = J, solved by finite elements (Nédélec/edge elements for
the vector potential A, with a small gauge-regularisation term) on a
tetrahedral mesh of one-eighth of the geometry.  The winding is
*homogenised*: instead of 2650 individual tapes we prescribe a smeared
current density J = I/(t·w) following the racetrack direction.

**Symmetry.**  The magnet has three mirror symmetries, so the FEM domain
is the octant (x ≥ 0, y ≥ 0, z ≤ gap-midplane).  The x = 0 and y = 0
cuts are "perfect electric conductor" boundaries (n×A = 0) and the
midplane is a "perfect magnetic conductor" (natural boundary), which by
the image principle automatically includes the *entire second coil* —
we never mesh it.  Anything summed over the winding (forces, Biot–Savart
integrals) must be expanded back to all 8 mirror images; forgetting the
quadrant images was historically a factor-of-4 bug in the SCIF.

**The most important property: linearity.**  With no iron, B is exactly
proportional to the current.  One FEM solve per geometry gives B per
ampere everywhere, and then: the field at any current is a scaling; the
quench current is a one-dimensional root-find; mechanical stress scales
exactly as I².  This is what makes the optimizer cheap (~8 s per
candidate geometry instead of minutes).

**Validation.**  The FEM field agrees with an independent Biot–Savart
integration at the few-% level (historically ~4 % median); the optimizer
re-checks this per candidate (`fem_dev_pct` column).

## 3. Layer 2 — Superconductor electrodynamics (how the current actually distributes)

A superconducting tape does not carry current uniformly.  Ramping the
magnet changes the flux through each tape, which by Faraday's law drives
*screening currents*: the current bunches toward the tape edges and can
locally reverse, up to the critical density ±Jc.  Two consequences
matter to us: the screening currents create their own field error at the
target (the **screening-current-induced field, SCIF**), and they
locally amplify |J| — and hence the Lorentz force — inside the tape
(**screening-current stress**).

**Model.**  The homogenised T–A formulation (Vargas-Llanos et al. 2022).
Each tape's sheet current is written as J = ∇T × n̂, where T is a scalar
"current potential" living on the tape plane and n̂ is the tape's face
normal; the transport current enters as boundary values T = ±I/(2δ_SC)
on the two tape edges.  The superconductor's electrical behaviour is the
measured power law E = E_c·(J/Jc)ⁿ, with Jc(B, θ) and n(B, θ)
interpolated from the manufacturer's 20 K dataset (θ = field angle to
the tape normal).  One implicit-Euler step of Faraday's law takes the
system from the zero-field-cooled state to the end of the 600 s ramp.
T (one problem per pancake — adjacent tapes need opposite edge values,
so layers are solved separately) and the vector potential A are iterated
to a joint fixed point (Picard), with the A-matrix factorised once and
reused.

**Numerics that mattered** (each of these was forced by a measured
failure, not taste): a *fixed* relaxation factor after ramp-up — every
adaptive scheme misread slow physical flux-front transients as stalls
and froze them; a smooth soft-max floor on j/jc — the hard kink at
j = jc made front cells flip states forever; log-space under-relaxation
of the resistivity; and a convergence criterion on the *observable*
(the bore SCIF must stall to < 0.05 mT per 10 iterations) because the
raw field residual never converges — the flux front wanders chaotically
among near-degenerate states at the 10⁻⁴ level while every integrated
quantity is frozen.

**Mesh.**  The screening profile lives across the 4 mm tape width, so
each layer is meshed as graded sub-slabs — 0.3 mm cells at the tape
edges (sized by Brandt/Norris strip theory: the penetration zone is
(1−√(1−i²))·w/2 ≈ 0.1–1 mm here), coarse in the bulk.  This matches
brute-force uniform refinement within 4 % on the SCIF at a third of the
cost.

**What the solution looks like — and why it is right.**  The textbook
"current peaks at both tape edges" profile is the *zero-perpendicular-
field* case.  Inside this winding every tape sees 1–5 T of perpendicular
field, vastly more than the ~50 mT strip penetration field, so the tapes
sit in the *fully-penetrated Bean state*: J = +Jc over part of the
width and −Jc over the rest, split so the net equals the transport
current, with the reversal side set by the sign of B_n.  The computed
profiles show exactly this, flipping sides between the top and bottom
pancakes as B_n flips — a signature the solver was never told about.

**Result with the current tape:** SCIF ≈ +82 mT ≈ 1.1 % of the bore
field at 200 A — at the scale of the uniformity budget, which is why the
optimizer carries a screening estimate for every candidate.

## 4. Layer 3 — Quench (how hard we can push)

A REBCO tape quenches (goes resistive) when its current exceeds the
critical current Ic, which drops steeply with local field magnitude and
angle.  The static criterion used here: the coil quenches at the current
where, for the *worst cell in the winding*, I = Ic(B(I), θ).  Because
B ∝ I, this is a per-cell 1-D root-find on the single reference solve.
The operating point takes the minimum over all cells divided by a
safety factor (1.15 by default).

**The biggest data assumption in the whole project lives here**: the
manufacturer dataset covers 0–8 T, while the conductor peak field at
the operating point is well above that.  Ic beyond 8 T is extrapolated
(clamped), so quench currents are *floor estimates*; the optimizer
reports the fraction of clipped Ic evaluations (`clip` column) and
flags candidates that rely too heavily on extrapolation.

## 5. Layer 4 — Mechanics (does the tape survive)

The Lorentz force density f = J×B is enormous at these fields
(~GN/m³).  Four screens, all from the same field solve:

1. **Cap hoop stress** σ = f_n·r — the curved end-caps carry outward
   load as conductor tension, evaluated per cell with the conservative
   assumption that each turn supports itself.  Limit ~500 MPa
   (lengthwise tape strength).
2. **Transverse (delamination) stress** — in the straight legs the
   outward force passes turn-to-turn through the stack: 1-D equilibrium
   σ_n(y) = −∫f_n dy from the free outer surface.  This acts along the
   tape's weakest axis (limit ~30 MPa).  The optimizer additionally
   re-evaluates this with the Bean amplification (saturated bands carry
   Jc = (1/i)·J_e locally — "screening-current stress", a documented
   killer of real REBCO magnets).
3. **Straight-leg line load** (~MN/m) — straight conductors cannot react
   transverse load without curvature; this number is the requirement
   handed to the support structure.
4. **Coil–coil axial attraction** (~hundreds of kN) and its bearing
   pressure.

Because stress ∝ I² exactly, the stress-limited currents have closed
forms, and in the current design space they — not quench — set the
operating point.

## 6. Layer 5 — The optimizer screen (putting it together)

For each candidate (a, b, n_turns), `optimize/optimize_geometry.py`
runs: one coarse-mesh FEM solve → per-cell quench root-find → operating
current I_op = min(quench/SF, hoop-limited, delamination-limited
including screening amplification) → target-box field and uniformity
from a multi-filament Biot–Savart (one racetrack filament per ~100
turns, so the layer distribution is resolved) → Bean-state screening
magnetization per cell → dipole-sum SCIF and SCIF-corrected uniformity
→ pass/fail and ranking.  ~8 s per candidate.

The Bean screening proxy was calibrated head-to-head against the full
T-A solver at the baseline geometry: +92 mT vs +82 mT (13 %).  Shortlist
finalists still get the full T-A, graded-mesh stress maps, and a real
structural review.

## 7. The assumption ledger (ranked by how much they could move answers)

1. **Ic data ends at 8 T** while operating points need 10–14 T at the
   conductor — quench margins beyond 8 T are extrapolations.  *Mitigation:
   clip-fraction flag; extended measurements planned.*
2. **Static quench criterion** — no thermal runaway dynamics, no
   normal-zone propagation, no hot-spot analysis.  Fine for ranking;
   a real protection study is separate work.
3. **Homogenised winding** — no individual tape resolution; the T-A
   tape-width resolution is finite (graded sub-slabs); insulation,
   joints, and terminal effects are absent.
4. **Single ramp step, no history** — SCIF is the end-of-ramp snapshot;
   relaxation/drift after the ramp and ramp-rate dependence are not
   modelled (matters for NMR-class stability specs, not for reaching
   10 T).
5. **Mechanical screen, not structural analysis** — self-supporting-turn
   hoop (conservative), no load sharing, no cooldown prestress, no
   bending solution for the legs, no stress concentrations.
   **The screening-current-stress policy is the single biggest design
   lever in the optimizer** (`SCREENING_STRESS_MODE` in opt_config.py):
   whether the delamination interface sees the full local Bean
   amplification ("local", conservative) or the width-averaged load
   ("averaged", optimistic) swings the achievable target field between
   ~5.4 T (entirely within validated Ic data) and ~12.5 T (heavily
   extrapolated Ic).  Settling it needs either a mechanical model of
   load transfer across the tape width or vendor delamination data
   under non-uniform loading.
6. **Bean fully-penetrated proxy in the optimizer** — ±13 % on SCIF vs
   the T-A reference; assumes |B_n| ≫ 50 mT (true everywhere here;
   a taper handles the exceptions and a diagnostic reports the
   unpenetrated fraction).
7. **Mirror symmetry** — equal-sense coils, and screening patterns
   assumed to share the transport current's symmetry.
8. **Field-model accuracy** — FEM validated to ~4 % vs Biot–Savart;
   uniformity numbers are relative and far more accurate than absolute
   field values; the per-candidate `fem_dev_pct` column tracks this.

## 8. Validation summary

| Claim | How it was checked |
|---|---|
| FEM field correct | Independent Biot–Savart: ~4 % (historic) + per-candidate cross-check |
| SCIF machinery correct | Quarter-cell uniform-current sum ×4 reproduces full Biot–Savart; exact cell volumes match analytic coil volume to 0.02 % |
| T-A profiles physical | Fully-penetrated Bean shape, ±Jc plateaus at local Jc, reversal side flips with sign(B_n); Norris transport limit overlaid |
| Mesh sufficiency | z-resolution study nz = 1…5 + graded; SCIF converged (strong tape ±5 %); in-plane converged at 0.5 % |
| Per-layer vs replicated T-A | Agree within ~10 % once the tape width is resolved |
| Solver convergence real | Observable frozen to < 0.01 mT across runs and 500-iteration continuation |
| Bean optimizer proxy | +92.3 vs +82.0 mT against converged T-A (13 %) at the original ~50–80 mm coil scale — **breaks down badly (up to ~10×, sign-inverted ranking) for the compact a ≈ 15–25 mm designs the search converges to; do not trust its `uniformity_pct`** |
| Warm-start unbiased | Warm vs cold starts agree to 0.01 % at tight tolerance |
| Champion is a real optimum, not numerical luck | 23-design perturbation study, full T-A each with 2 independent-mesh repeats: all converged, spread ≤ 0.003 pp on 22 of 23, every axis smooth and monotone. **But it is not converged** — a neighbour dominates it on all four metrics (2026-07-30) |
