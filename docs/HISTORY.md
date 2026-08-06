# Racetrack_v4 — Detailed Project History

Archived from CLAUDE.md on 2026-08-04 during a cleanup pass, to keep
CLAUDE.md itself as a lean current-state reference. This file is the
full, chronological narrative of every investigation, search run,
bug hunt, and retraction that produced the current design and the
current understanding of the solver's limitations — kept for anyone
who needs the detailed reasoning behind a conclusion CLAUDE.md only
states in summary. Nothing below has been re-edited; it is preserved
as originally written, including sections that were later retracted
or superseded (each says so inline).

For the current design, current constraints, and current open
issues, see CLAUDE.md — this file is reference/archive only.

---

### CMA-ES continuous search (`optimize/cmaes_search.py`, 2026-07-21)

**New project direction, distinct from evaluate.py's 2026-07-14 handoff
spec above** — the objective flipped from "maximize B_target" to
**minimize tape length** subject to hard constraints:

  - `B_target_T >= 10.0 T` (mean |Bz| over the target box at I_op)
  - `uniformity_pct <= 1.0%` (peak-to-peak, SCIF-corrected) over a
    **30 x 6 mm** box (was 15x6mm — `TARGET_X_M`/`TARGET_Y_M` in
    opt_config.py; ASSUMPTION: 30mm is the X/long side, 6mm stays Y,
    matching the prior box's convention — flag if this should be swapped)
  - Operate at 50–60% of local Ic anywhere in the winding: implemented as
    `SAFETY_FACTOR = 1.818` (`I_op = I_quench/SAFETY_FACTOR`), which by
    construction of `quench_current()`'s bisection puts the single
    worst-margin cell at exactly i = 0.55 (every other cell has more
    margin) — the midpoint of the requested band
  - **Hoop stress only**, end-cap curved sections only (`B x J x bending
    radius`, the existing `cap` mask in `stress_screen()`), tightened
    500→400 MPa (`SIGMA_HOOP_MAX_PA`). Delamination is a material property
    (~30 MPa), not a design lever — disabled (`SIGMA_DELAM_MAX_PA = 1e12`)
    per explicit project direction; still computed/reported, not enforced
  - **Coil-to-coil axial gap is now a free variable** (was fixed at
    `coil_half_gap = 0.030`): >= 3 mm **face-to-face** clearance required
    (distinct from `coil_half_gap`, which is a half CENTRE-to-centre
    distance — face_gap = `2*(coil_half_gap − n_layers*w/2)`). Minimum
    feasible `coil_half_gap` is 15.5 mm given n_layers=7, w=4mm fixed.

**evaluate.py's contract is UNCHANGED** — it now reads dedicated
`EVALUATE_SAFETY_FACTOR` (1.15) / `EVALUATE_TARGET_X_M` / `EVALUATE_TARGET_Y_M`
(15x6mm) constants instead of the shared ones above, specifically so the
external team's tool doesn't silently change behavior when opt_config.py is
repurposed for this internal search (verified: still reproduces the
documented baseline, B_target=13.4T @ I_op=339A, unif=0.212%).

**Variables** (9 base + coil_half_gap = 10 continuous, CMA-ES via `pycma`,
installed into `fenicsx-env` from conda-forge — not on PyPI-via-pip in this
env, no system pip): `a`, `b`, `coil_half_gap`, `n_turns[7]`. `n_turns` are
rounded to nearest int only at evaluation; CMA-ES sees them as continuous.
**`a` and `b` are intentionally UNBOUNDED** (`CMAES_A_BOUNDS = (None, None)`
in opt_config.py) per project direction — the first run pinned both at
their box edges (30/60mm) for most of the search, so real limits are now
physical only (`b > a + 5mm`; inner-radius clearance > 3mm), enforced as a
smooth penalty in `geometry_violation()`, not a box bound. Initial CMA-ES
step size for a/b set directly (`CMAES_A_STD0`/`CMAES_B_STD0` = 0.02m)
since there's no bound range to derive it from.

Constraint handling is a straightforward quadratic penalty added to
tape_km (not augmented Lagrangian) — see `fitness()` in cmaes_search.py.
A cheap pure-numpy pre-check (`geometry_violation()`, mirrors
`params.recompute_derived()`'s assertions plus the new face-gap rule)
skips the expensive FEM solve for geometrically invalid samples while
still handing CMA-ES a smooth gradient-like signal.

Run: `conda run -n fenicsx-env python3 optimize/cmaes_search.py`
(~5-10s/eval, ~400 evals/~40-50 min serial, single-process — evaluations
are independent and parallelizable later if the budget needs to grow).
Outputs `optimize/runs/cmaes_results.csv` (best design),
`optimize/runs/cmaes_history.csv` (every evaluation), and four dark-theme
figures in `visualization/`: `cmaes_convergence.png` (fitness history +
constraint margins), `cmaes_constraints.png` (per-constraint detail),
`cmaes_variables.png` (design-variable trajectories), `cmaes_overview.png`
(objective-vs-cost and constraint-space scatter with the best design
starred, plus a stats panel).

**Run history so far** (each run's full evaluation set is preserved, never
overwritten — see "Cumulative history" below):
- Run 1 (bounded a/b 30-70/60-140mm, 410 evals): best a=30.8mm, b=60.4mm,
  coil_half_gap=18.5mm (face gap 9.0mm), n_turns=[207,511,190,76,467,462,374]
  (2287 total) → **tape=0.773km, B_target=10.08T, uniformity=0.867%,
  hoop=162MPa** — ~35% less tape than the 1.15-safety-factor/13.4T baseline
  while meeting the new 10T floor, but a and b sat at their box bounds for
  most of the run (see `cmaes_variables.png` from this run).
- Run 2 (a/b unbounded per project direction, 410 evals, same budget,
  cold-started from the original baseline): WORSE result (tape=0.871km @
  10.65T) — a/b never converged within budget once the box was removed
  (still oscillating over a wide range at eval 410); removing the bound
  enlarged the effective search space and 410 evals wasn't enough. Best
  design from run 1 remained the overall best.
- Run 3 (a/b unbounded, warm-started from run 1's best instead of the
  baseline, budget doubled to 800, `CMAES_SEED` bumped 42→43): **best yet,
  and this time fully converged** — a=21.0mm, b=32.8mm, coil_half_gap=
  15.5mm (face gap 3.0mm — right at the 3mm floor, now the genuinely
  binding physical constraint on that variable, not an artifact),
  n_turns=[329,390,107,208,126,392,349] (1901 total) → **tape=0.373km,
  B_target=11.14T, uniformity=0.373%, hoop=109MPa** — roughly half the
  tape of run 1, comfortably clearing every constraint. Unlike run 2,
  `cmaes_variables.png` shows a, b, and coil_half_gap all genuinely
  settle (not still wandering) by ~eval 700 of 800 — this is a real
  interior optimum for a/b, not a budget-starved search. 810 evals took
  108.6 min (some overshoot past the 800 budget from the last generation).
  `cmaes_param_map.png` (now 1630 evaluations across all 3 runs) shows
  run 3 found a genuinely new region (small a/b/gap) neither run 1 nor
  run 2 explored.

**Cumulative history / parameter-space map:** every run appends its full
evaluation set (tagged with a `run_tag` = start timestamp) to
`optimize/runs/cmaes_all_evaluations.csv` (`cfg.CMAES_MASTER_LOG`) — this file
is NEVER overwritten, unlike the per-run `cmaes_history.csv`. Run 1's data
would otherwise have been lost (its `cmaes_history.csv` was overwritten by
run 2 before this existed) — it was recovered by parsing run 1's saved
stdout log; both runs are now backfilled into the master log under
`run_tag`s `run1_bounded_2026-07-21` / `run2_unbounded_2026-07-21`.
`cmaes_search.py`'s `_make_param_map()` reads this cumulative file (not
just the current run) and produces `visualization/cmaes_param_map.png`:
4 panels (a-vs-b, a-vs-gap, turns-vs-gap, tape-vs-turns) colored by
outcome (green=all-pass / orange=feasible-but-violating / red=infeasible)
with a distinct marker shape per run, so it's visible at a glance which
run explored which region and what was learned. Use this file as the
running "map" of good/bad regions for future searches — don't re-explore
territory it already shows is infeasible or dominated.

**Generation-parallel evaluation (`cmaes_search.py`, 2026-07-21):** opt-in
via `cfg.CMAES_N_WORKERS` (default 1 = original serial path, unchanged).
`>1` evaluates each generation's population across a
`ProcessPoolExecutor` (manual `ask()`/`tell()` loop instead of
`es.optimize()`). Each worker gets a PID-suffixed `mesh_filename`
(`_worker_init()`) so concurrent `build_mesh.build()` calls can't race on
the shared default mesh path — verified in an isolated copy: same
seed/config at `N_WORKERS=1` vs `=4` produced byte-identical candidate
sequences and the same best design, with real speedup (96s→36s for the
same 30 evals). 6 workers is a good default on this 8-core machine.

**Live progress logging fix (2026-07-22):** `_run_parallel()`'s manual
ask/tell loop never wrote to `outcmaes/` at all — pycma's own
`.optimize()` auto-registers `self.logger.add` as a post-`tell()`
callback (via `_prepare_callback_list()`), invoked as `f(self)`, i.e. the
`es` argument is required. `es.logger.add()` with no argument silently
no-ops (confirmed directly: 6 generations of bare `.add()` produced zero
files in `outcmaes/`, while `.add(es)` with identical opts wrote all 9
expected files). Every `CMAES_N_WORKERS>1` run before this fix (both
extended-refinement rounds) was affected — `outcmaes/*.dat` sat stale for
the entire run and only ever reflected whatever the last *serial* run had
written, which made mid-run progress unreadable. Fixed by adding
`es.logger.add(es)` after `es.tell(...)` in `_run_parallel()`; verified
with a real (non-bare-pycma) end-to-end smoke test that `outcmaes/fit.dat`
now updates live during a parallel run. Doesn't recover missed data from
already-completed runs, only fixes future ones.

**Two more logging/data-loss fixes (2026-07-22), found after repeatedly
being unable to check on multi-hour runs mid-flight:**

1. **`conda run` buffers ALL subprocess output until the process exits —
   confirmed independent of anything in the script.** `sys.stdout.
   reconfigure(line_buffering=True)` (added to `cmaes_search.py`) is
   necessary but NOT sufficient: verified directly that a trivial
   `print()`-in-a-loop script run via `conda run -n fenicsx-env python3
   -c "..."` shows ZERO output until the process finishes, REGARDLESS of
   `sys.stdout.reconfigure()` inside the script AND regardless of `conda
   run`'s own `--no-capture-output`/`-s`/`--live-stream` flag (tested,
   didn't help either). The same script run with the environment's
   **python binary directly** (`/home/gerrityjack/miniconda3/envs/
   fenicsx-env/bin/python3`, bypassing `conda run` entirely) shows live
   output immediately, and combined with the `reconfigure()` call, is
   fully line-buffered. **Every long-running script in this project
   should be launched via the direct binary path, not `conda run -n
   fenicsx-env python3 ...`, if you need to check on it mid-run** (which
   is essentially always true for multi-hour CMA-ES/sweep runs) — `conda
   run`'s wrapping process is the actual cause, not anything fixable from
   inside the Python process. Every run before this finding (both
   extended-refinement rounds, the n_layers=6/8 refinements, the overnight
   sweep) was launched via `conda run` and had this problem; it wasn't a
   fluke or a fixable-in-script issue.

2. **Incremental CSV/master-log flushing** (`cmaes_search.py`'s
   `_record()`, `FLUSH_EVERY = 20`): previously `cmaes_history.csv`/
   `cmaes_all_evaluations.csv`/`cmaes_results.csv` were only ever written
   ONCE, at the very end of `main()` after `es.optimize()`/
   `_run_parallel()` fully returns — meaning a crash or kill partway
   through a multi-hour run lost EVERY evaluation done so far, with no
   way to recover any of it. Fixed by appending any not-yet-written rows
   (tracked via a `_last_flushed_idx` pointer, so incremental + the final
   call never double-writes) to the master log, and rewriting this run's
   own CSVs, every `FLUSH_EVERY` evaluations (~1.5-2 generations).
   Verified end-to-end: exactly the expected row count in both the
   per-run CSV and the master log for that run's tag, no duplicates.
   Figures are NOT regenerated incrementally (too expensive per
   checkpoint) — only the CSVs, which is what actually matters for not
   losing data; figures can always be rebuilt from the CSVs after the
   fact.

**Diverse-restart study (2026-07-21):** run 3's optimum was reached by
always warm-starting from the previous best, which never tests a
different basin. 4 manual cold starts from deliberately different
regions all landed WORSE than run 3 and none reproduced its gap-at-floor
behavior: "large_sparse" (a=65/b=120mm start) → 1.166km @ 12.34T;
"small_wide_gap" (a=25/b=45mm start) → 0.743km @ 10.38T (gap backed off
the floor to 9.1mm); "high_turn_dense" (a=45/b=70mm, 600 turns/layer
start) → 0.755km @ 12.25T (unif right at the 1% limit, gap off the floor
again); a 4th ("asymmetric_reverse") was stopped early once confirmed
working, superseded by the automated sweep below. Moderate confidence
run 3's basin (small a/b, gap pinned at its physical floor) is a strong,
possibly global, optimum — nothing found so far competes, and the
gap-at-floor signature hasn't reappeared anywhere else.

**Automated overnight sweep (`optimize/studies/sweep_restarts.py`, 2026-07-21):**
built to scale the diverse-restart idea up unattended — time-boxed
(`TOTAL_BUDGET_HOURS`, default 11.5) loop of independent CMA-ES cold
starts, each a **separate OS process** (never a repeated in-process call
to `cmaes_search.main()` — its globals `_eval_count`/`_history`/`_best`
only reset via a fresh process/import, not by calling `main()` again).
Starting points are sampled uniformly (`a~U(15,90)mm`,
`b~a+U(10,150)mm`, `gap~U(*CMAES_HALF_GAP_BOUNDS)`, each `n_turns[i]~
DiscreteU(*CMAES_N_BOUNDS)`) and handed to each subprocess via the
`CMAES_SWEEP_OVERRIDE_JSON` env var (see `opt_config.py`'s "sweep
override" section, right after `CMAES_N_WORKERS`) — `opt_config.py`'s
on-disk `CMAES_X0`/`SEED`/`MAX_EVALS` are NEVER rewritten by the sweep,
so it can be interrupted anytime with zero cleanup needed. 300 evals per
restart (breadth over depth — enough to reveal which basin a cold start
heads toward without fully polishing it). Progress:
`optimize/sweep_restarts_log.txt` (one line per completed restart);
per-restart full logs in `optimize/sweep_logs/`. Every restart still
feeds the same cumulative `cmaes_all_evaluations.csv`/`cmaes_param_map.png`
as any other run. **Caught one real bug during smoke-testing**: reading
`cmaes_results.csv` right after a subprocess exits is unsafe without
first deleting any pre-existing copy — that file is only rewritten if the
run finds ≥1 all-pass design, so a restart that finds nothing would
otherwise silently be reported as having reproduced the PREVIOUS
restart's leftover result. Fixed by removing the file before each
subprocess launch and treating "file doesn't exist after" as "no
feasible design found," not an error.

**Overnight sweep RESULTS (completed 2026-07-22, 05:58, 11.37h, 76
restarts attempted — 75 succeeded, 1 hit a rare gmsh meshing error
("Unknown node 0 in element...") on a degenerate candidate geometry,
caught by the sweep's error handling and skipped, sweep continued
normally):**
- **Run 3's design (0.373km @ 11.14T) remains the best found anywhere in
  26,180 cumulative evaluations across 84 runs (9,026 all-pass).** The
  closest any of the 76 random cold starts got was 0.430km @ 10.53T
  (a=25.4mm, b=36.4mm, gap=17.2mm) — about 15% worse in tape length.
- 5 of 76 restarts (~7%) independently converged into run 3's general
  basin (a<30mm, b<45mm) from random, uncorrelated starting points
  spanning a∈[15,90]mm, b−a∈[10,150]mm, gap∈[15.5,45]mm, n_turns∈[50,900]
  per layer — that basin is a recurring attractor, not a one-off.
- **This substantially raises confidence that run 3's optimum is at or
  very near the global optimum within the practical design space**: 76
  independent searches covering a wide, randomly-sampled swath of the
  space never beat it, and its neighborhood keeps getting rediscovered.
  Not proof (CMA-ES restarts are still a heuristic search, not exhaustive
  enumeration), but strong empirical evidence.
- `cmaes_param_map.png` now aggregates 26,180 points across 84 runs — a
  clean "efficient frontier" is visible in the tape-vs-turns panel, and
  the a/b panel shows a sharp infeasibility boundary (the
  inner-radius/straight-length constraints) with the all-pass region
  concentrated at small-to-moderate a/b, exactly where run 3 sits.
  **Fixed a legend-scalability bug** in `_make_param_map()` while
  processing this: per-run marker shapes are unreadable past ~10 runs
  (84 runs produced an unusable legend covering two subplots) — now
  falls back to a single uniform marker colored by outcome only above
  `MAX_RUNS_FOR_MARKERS = 10`, since the point of this figure at scale is
  the aggregate coverage map, not which specific run found what.

## Multi-filament Biot-Savart fix (2026-07-22)

`physics/coil2_field.py`'s original `compute_both_coils_field()` treats the
**entire winding as one filament at a single nominal radius `a`** — its own
docstring states this is valid only when the winding-pack cross-section is
small compared to `a`/`coil_half_gap`. That was true at the original
~50-80mm coil scale but breaks down hard for the CMA-ES-optimized designs
below, where `a` is only ~13-25mm and the pack thickness is a comparable
~20-30mm. Symptom that surfaced it: `visualization/field_uniformity.py`
reported a spurious FAIL (6.74%) directly contradicting the optimizer's own
PASS (0.68-0.94%) for the same design.

**Fix:** `compute_both_coils_field_multilayer()` (new function in
`coil2_field.py`) resolves each layer's own z-center and radial center
(mirroring `optimize_geometry.py`'s internal `filament_stack()`), grouping
each layer's `n_turns[i]` into sub-filaments of `cfg.FILAMENT_TURNS_PER_GROUP`
(~100) turns each, placed at that sub-group's own radial centerline, summed
over both coils via z-mirroring. The old `compute_both_coils_field()` is
kept (still valid at large coil scale) but every near-coil field evaluation
in the repo now calls the multilayer version instead:
`visualization/plot_fields.py`, `visualization/field_uniformity.py`,
`sweep/quench_sweep.py`, `solve/ta_postprocess.py`, `solve/ta_sweep.py`,
`solve/ta_solve.py`. (`visualization/plot_field_poster.py` and
`visualization/plot_3d.py` use actual per-cell FEM data directly, not this
filament approximation, so they were unaffected.) `optimize_geometry.py`
itself already built its own correct multi-filament sum and needed no
change — it was the reference this fix was calibrated against.

`field_uniformity.py` was also updated to match the optimizer's exact
constraint check: box size `TARGET_X_M`/`TARGET_Y_M` (30×6mm, was a
hardcoded 15×6mm) and a new `_scif_correction()` applying the same
Bean-state dipole correction the optimizer uses (imported from
`optimize_geometry.py`: `bean_moments()`, `dipole_field_mirrored()`).
Verified progression on the same design: spurious FAIL 6.74% → PASS 1.51%
(multi-filament fix alone) → PASS 0.56% (+ box size + SCIF correction),
in close agreement with the optimizer's own 0.68-0.94% depending on
layer count. The SCIF correction is applied to the box-interior field only,
not the surrounding/background visualization grid (cosmetic, not worth the
extra compute there).

---

## n_layers sweep — searching over the number of pancake layers (2026-07-22)

CMA-ES cannot treat "number of layers" as a continuous/optimized search
dimension — its covariance adaptation needs fixed dimensionality for an
entire run. `optimize/studies/sweep_n_layers.py` handles this as a **discrete outer
loop**: one full, independent CMA-ES search per candidate layer count.
`params.n_layers = len(params.n_turns)` is already computed fresh on every
call, and `build_mesh.py`/`optimize_geometry.evaluate()` handle an
arbitrary layer count with no code changes — the one real bug this exposed
was `cfg.CMAES_HALF_GAP_BOUNDS` being a static range tuned for
`N_LAYERS=7`; fixed via `gap_bounds_for_n_layers()` in `cmaes_search.py`,
which computes the coil_half_gap floor directly from `N_LAYERS` (reproduces
the old static bounds exactly at 7 layers, so existing behavior is
unchanged there).

`sweep_n_layers.py`'s `smart_x0_for_n_layers()` builds a physically-informed
starting point per layer count instead of a blind cold start: `a` and
`coil_half_gap` pinned near their physical floors (inner-radius clearance,
face-gap), straight length just above its own floor, total turn count
(`REF_TOTAL_TURNS=1310`, from the then-best 7-layer design) redistributed
evenly across the candidate layer count. Margins calibrated directly
against a converged design's actual measured margins-above-floor
(`MARGIN_A=0.3mm`, `MARGIN_B_EXTRA=2.24mm`, `MARGIN_GAP=1.50mm`).

**Cold-start sweep results** (250-500 evals each, smart-start, NOT the
extended-refinement treatment — see below):

| n_layers | tape_km | B_target_T | unif_pct | hoop_MPa | a_mm | b_mm | gap_mm |
|---|---|---|---|---|---|---|---|
| 3 | 0.370 | 10.06 | 0.913 | 149 | 28.7 | 37.4 | 8.0 |
| 4 | 0.356 | 10.58 | 0.529 | 143 | 25.6 | 36.0 | 9.6 |
| 5 | 1.011 | 10.83 | 0.963 | 209 | 37.4 | 60.1 | 20.4 |
| 6 | 0.792 | 10.18 | 0.493 | 207 | 41.8 | 48.7 | 22.6 |
| 8 | 0.442 | 11.46 | 0.375 | 161 | 27.4 | 33.9 | 17.6 |
| 9 | 0.632 | 10.09 | 0.487 | 233 | 30.6 | 39.1 | 19.7 |
| 10 | 0.792 | 12.91 | 0.664 | 210 | 30.7 | 42.0 | 23.7 |
| 12 | 0.561 | 10.13 | 0.527 | 146 | 25.0 | 38.1 | 25.5 |

(7 was skipped in this sweep — a far more thoroughly optimized 7-layer
design already existed from the extended-refinement rounds below; redoing
it at only 400 cold-start evals would be worse and redundant.)

**Extended-refinement rounds** (2000-2500 evals, `CMAES_TIGHT_BOUNDS=True`
— see the "zone-out" section below — warm-started from the cold-start
result, launched via the direct python binary for live logging):

| n_layers | tape_km | B_target_T | unif_pct | hoop_MPa | a_mm | b_mm | gap_mm | n_turns |
|---|---|---|---|---|---|---|---|---|
| **6 (BEST OVERALL)** | **0.1464** | 10.10 | 0.939 | 59 | 12.9 | 21.0 | 13.9 | [187,223,256,258,245,50] |
| 7 | 0.1884 | 10.16 | 0.678 | 78 | 15.5 | 22.8 | 17.0 | [241,258,332,307,52,69,51] |
| 8 | 0.1971 | 10.24 | 0.677 | 74 | 14.8 | 22.1 | 17.5 | [166,222,115,240,241,246,51,238] |

This confirms a clean, converged trend: **6 < 7 < 8** in tape cost, all
three thoroughly refined and showing plateau/near-plateau behavior in
`cmaes_variables.png` (not still declining at the budget cap, unlike round
1 of several of these — see individual run notes in
`optimize/opt_config.py`'s `CMAES_X0` comment for the full history).

**n_layers=5 — a cautionary tale about step size, not a real result yet:**
- Round 1 (from the 1.011km cold start, 2010 evals): converged to
  **0.3734km** @ 10.18T, unif=0.90%, hoop=79MPa — notably worse than 6/7/8,
  breaking the smooth trend. But `cmaes_variables.png` showed `b` and total
  turns still visibly declining at the eval-2000 cap, no plateau — the same
  under-convergence signature the 7-layer search showed before its own
  round 2 found a dramatic further improvement (0.373km → 0.188km at the
  time). This suggested 5 layers might not really be worse, just less
  converged.
- Round 2 (warm-started from round 1's endpoint, 2510 evals, smaller but
  still FIXED absolute step size `CMAES_A_STD0=6mm`/`CMAES_B_STD0=12mm`):
  **regressed to 0.4173km** @ 10.29T, unif=0.80%, hoop=98MPa — WORSE than
  round 1, converging to a and b that were 20-25% away from round 1's point
  (a=19.6mm/b=59.4mm vs round1's a=16.2mm/b=55.0mm).
- **Root cause:** those "smaller" step sizes were still 37%/22% of this
  design's own small a/b scale (16mm/55mm) — large enough, in absolute
  terms, to let CMA-ES escape round 1's basin into a worse one, despite
  being intended as a fine local refinement. The step size needs to be
  proportional to the design's own scale, not a fixed mm number picked by
  eye. Fixed for future rounds via `CMAES_A_STD0_OVERRIDE`/
  `CMAES_B_STD0_OVERRIDE` env vars in `opt_config.py` (see below) — set
  directly in metres by the caller as `warm_start_value * fraction`.
- **n_layers=5's true status is still open**: round 1's 0.373km is the
  best DATA POINT so far, but neither round properly tested "5 layers, well
  converged, correctly-scaled refinement." Queued in
  `optimize/studies/overnight_refinement.py` below, redone from round 1's endpoint
  (not round 2's) with a proportional step size.

**"Zone-out" / tight-bounds mechanism (`CMAES_TIGHT_BOUNDS` in
opt_config.py):** derived empirically from the top 5% of all-pass designs
across 30k+ evaluations spanning many independent runs and layer counts —
those designs sit within a few mm of their physical floors (inner-radius,
straight-length, face-gap) and never use more than ~420 turns/layer, out of
the (50,900) range CMA-ES was actually allowed to search. Every independent
search to date (84+ runs, 4 hand-picked diverse restarts, a 76-restart
random overnight sweep, and this n_layers sweep) converged toward this same
near-floor region regardless of starting point — concentrating budget there
via `CMAES_TIGHT_N_BOUNDS=(50,500)` and `CMAES_TIGHT_GAP_MARGIN_M=0.010`
lets a fixed evaluation budget converge much faster instead of
re-discovering the region from scratch every run. Opt-in
(`CMAES_TIGHT_BOUNDS=False` by default in the underlying mechanism,
currently left `True` on disk since every extended-refinement round has
used it) — does not affect other callers' default behavior.
**Gotcha, bit twice:** pycma's `geno()` raises `ValueError: argument of
inverse must be within the given bounds` if a warm-start `x0` has any
per-layer turn count above the tight cap (500) — hit when launching
n_layers=6's extended refinement (cold-start had turns up to 667) and again
in the `overnight_refinement.py` smoke test (n_layers=3's cold start had a
635) — always clip warm-start turn counts to the tight bound before use if
`CMAES_TIGHT_BOUNDS` is on.

---

## Live-progress-logging and data-loss fixes for long-running searches (2026-07-22)

Two fixes, both critical for any unattended multi-hour `cmaes_search.py`
run, found after repeatedly being unable to check on such runs mid-flight:

1. **`conda run` buffers ALL subprocess stdout until the process exits —
   confirmed independent of anything in the script.**
   `sys.stdout.reconfigure(line_buffering=True)` (added to
   `cmaes_search.py`) is necessary but NOT sufficient: a trivial
   `print()`-in-a-loop script run via `conda run -n fenicsx-env python3 -c
   "..."` shows ZERO output until the process finishes, regardless of the
   reconfigure call AND regardless of `conda run`'s own
   `--no-capture-output`/`-s`/`--live-stream` flags (tested, none helped).
   The same script run with the environment's **python binary directly**
   (`/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3`, bypassing
   `conda run` entirely) shows live output immediately, and combined with
   `reconfigure()`, is fully line-buffered.
   **Every long-running script in this project should be launched via the
   direct binary path, not `conda run -n fenicsx-env python3 ...`, if you
   need to check on it mid-run** — which is essentially always true for
   multi-hour CMA-ES/sweep runs. `conda run`'s wrapping process is the
   actual cause, not anything fixable from inside the Python process.
   `optimize/studies/overnight_refinement.py` (below) launches this way by design.

2. **Incremental CSV/master-log flushing** (`cmaes_search.py`'s
   `_record()`, `FLUSH_EVERY = 20`): previously `cmaes_history.csv`/
   `cmaes_all_evaluations.csv`/`cmaes_results.csv` were written ONCE, at the
   very end of `main()` — a crash or kill partway through a multi-hour run
   lost EVERY evaluation done so far, with no way to recover any of it.
   Fixed by appending any not-yet-written rows (tracked via a
   `_last_flushed_idx` pointer, so incremental + the final call never
   double-writes) to the master log, and rewriting this run's own CSVs,
   every `FLUSH_EVERY` evaluations (~1.5-2 generations). Figures are NOT
   regenerated incrementally (too expensive per checkpoint) — only the
   CSVs, which is what actually matters for not losing data; figures can
   always be rebuilt from the CSVs after the fact. This is what makes
   `overnight_refinement.py`'s per-job timeout-and-kill safe: a killed job
   loses at most ~20 evals of progress, never the whole run.

---

## Overnight extended-refinement run (`optimize/studies/overnight_refinement.py`, 2026-07-22)

**Purpose:** finish the n_layers sweep's job properly. Layer counts 6, 7,
8 have each had a full extended-refinement round; 3, 4, 9, 10, 12 only have
the cheap 250-500-eval cold-start result (apples-to-oranges vs 6/7/8); and
5's only refinement attempt regressed (see above) because of a step-size
bug that's now fixed. This script gives 3, 4, 9, 10, 12 the same
extended-refinement treatment 6/7/8 got, and redoes 5 from round 1's
endpoint with a **proportional** step size instead of round 2's oversized
fixed one.

**What it does, concretely:**
- Runs 6 jobs sequentially (never in parallel with each other — each job
  itself already uses `CMAES_N_WORKERS=6` internal parallelism on this
  8-core machine): n_layers = 3, 4, 5, 9, 10, 12, each warm-started from its
  actual best known design (hardcoded in the script's `JOBS` list, sourced
  from `sweep_n_layers_log.txt` for 3/4/9/10/12 and round 1's endpoint for
  5 — NOT round 2's).
- Each job's `CMAES_A_STD0`/`CMAES_B_STD0` are set to `STD0_FRAC=5%` of
  that job's OWN warm-start `a`/`b` value (via the new
  `CMAES_A_STD0_OVERRIDE`/`CMAES_B_STD0_OVERRIDE` env vars added to
  `opt_config.py` for exactly this) — proportional, not the fixed-mm
  approach that caused n_layers=5's round-2 regression.
- Each job is **time-boxed** (`PER_JOB_MINUTES_CAP`: 60/60/75/90/100/120 min
  for n=3/4/5/9/10/12 respectively, cheaper layer counts get less time,
  matching the per-eval cost scaling `sweep_n_layers.py` already uses) —
  launched via `subprocess.Popen` with a hard `.wait(timeout=...)`, SIGTERM
  then SIGKILL if it doesn't exit. Because of fix #2 above, a timed-out job
  still has all its progress (data-wise) preserved up to the last flush.
- The whole run is also **globally time-boxed**
  (`TOTAL_BUDGET_HOURS=10.0`) — if a job overruns badly, remaining jobs are
  skipped rather than pushing the total run past its overnight window.
- Launches every job via the direct python binary (fix #1 above), so
  progress is genuinely observable mid-run, not just at the end.

**Verified before being handed off** (2026-07-22): syntax-checked; all 6
jobs' warm-start `x0` confirmed within `bounds_and_stds()`'s actual bounds
(two needed a fix — n=3's cold-start turn count of 635 and n=10's 541 both
exceed the tight 500 cap, clipped to exactly 500 in the script; n=12's gap
warm-start sat AT its tight-bounds floor to 4-decimal precision, nudged
+0.2mm); a live 1-minute-capped smoke test of the n=3 job confirmed the
full pipeline works end-to-end — parallel workers started, real
evaluations ran and were logged live, the timeout killed the subprocess
cleanly, and 40 evaluations were preserved via the incremental flush.

**To launch** (from the repo root, in a fresh conversation/session, in the
background so it survives the session ending):

```bash
/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3 \
    optimize/overnight_refinement.py \
    > optimize/overnight_stdout.log 2>&1 &
disown
```

(`disown` so it isn't killed if the launching shell exits — or launch it
via Claude Code's own background-task mechanism, which survives
independently of the shell.)

**To check progress:** `tail -f optimize/overnight_refinement_log.txt` (one
line per job start/finish, like `sweep_n_layers_log.txt`) or any
`optimize/overnight_logs/n_layers_NN.log` (full per-job CMA-ES output,
live). Expect roughly one job finishing every 1-2 hours.

**To read results after it finishes (or check mid-run):** do NOT rely on
`optimize/runs/cmaes_results.csv` alone — it only ever holds the LAST job's
best design (each job overwrites it, same convention as
`sweep_n_layers.py`). Instead, pull each job's result from
`optimize/runs/cmaes_all_evaluations.csv` by its `run_tag` (each job's log ends
with a line reporting how many rows were appended to that file), or just
read the summary line in `optimize/overnight_refinement_log.txt` for each
completed job — same format as the extended-refinement round summaries
tabulated above.

**After it completes:** update the champion ledger in this section and in
`optimize/opt_config.py`'s `CMAES_X0` comment if any layer count beats
6 layers' current 0.1464km, or if 5/3/4/9/10/12 reach a clear, converged
number worth recording (check `cmaes_variables.png` from that job's run for
plateau vs still-declining, same diagnostic used throughout this section).
Regenerate `visualization/*` figures from whichever design ends up the
overall best (see "Running the pipeline" below) — the currently-plotted
figures are still from the 6-layer champion.

**IMPORTANT — everything in this section (and the n_layers-sweep table
above it) predates 2026-07-23's practical manufacturing constraints and is
now OBSOLETE.** See the next section for what superseded it. In
particular: the "current best design overall" (6 layers, 0.1464km) no
longer satisfies the bend-radius or double-pancake constraints below, and
`optimize/studies/overnight_refinement.py` was killed mid-run rather than left to
finish, because the constraint change made its target (an odd-layer-count
design, n_layers=9) unbuildable regardless of how well-optimized it got.

---

## Practical manufacturing constraints (2026-07-23)

Three real hardware requirements, added mid-session after direct user
input (not derived from any FEM/optimization result):

1. **Minimum bend radius 7.5mm.** REBCO tape cracks if bent tighter than
   this. `cmaes_search.py`'s `geometry_violation()` already tracked the
   relevant quantity (`a_inner_min`, the innermost turn's radius — always
   the layer with the most turns, since every layer shares the same outer
   edge `a_out` and eats inward by its own `n_i·t`) as a bore-clearance
   check against an arbitrary 3mm; raised straight to 7.5mm (`min_clear`
   in `geometry_violation()`). Every design found before today sat right
   at the old 3mm floor (the "zone-out" empirical bounds work explicitly
   documented this), so this is a hard, binding change — expect
   noticeably longer tape lengths than the pre-2026-07-23 champion ledger,
   not a bug if that happens.
2. **7×14×1mm sensor array clearance — verified to need NO new
   constraint**, not just assumed. Checked directly against the racetrack
   geometry (`params.py`'s `tape_length_m = sum(n*(4L + 2π·a_c))` — each
   straight section is `2L` long, not `L`): even the tightest pre-2026-07-23
   design (L≈6.3mm) gives `2L≈12.6mm` of straight bore, well over the 7mm
   sensor dimension. The 14mm dimension needs bore diameter
   `2×a_inner_min ≥ 14mm`, automatically satisfied once (1) holds
   (2×7.5=15mm). The 1mm dimension fits inside the existing 3mm
   `MIN_COIL_GAP_M` coil-to-coil face gap.
3. **Double-pancake construction.** Every pancake must be one of a PAIR
   of adjacent layers (2i, 2i+1) wound as a single continuous piece of
   tape — outer edges already match, but the inner ends need to be
   joined, which only works if both layers in a pair have the SAME turn
   count (so their `a_inner` also match, turn-for-turn). Implemented in
   `cmaes_search.py`: the optimization vector now carries
   `N_PAIRS = N_LAYERS // 2` turn variables instead of `N_LAYERS`;
   `decode()` expands each pair value to both of its layers before
   evaluation; `encode_x0()` averages each pair when reading a (possibly
   unpaired) warm-start `n_turns` list. **`N_LAYERS` must be even** —
   `cmaes_search.py` raises `AssertionError` at import time otherwise.
   This eliminates every odd layer count (3, 5, 7, 9) outright, including
   **n_layers=9**, which had been the most promising open lead (0.152km,
   within 4% of the 6-layer champion) right up until this constraint was
   added — that whole line of investigation is now moot, not just
   deprioritized.

**Turn-count floor also removed** (50 → 1) at the same time, independent
of the three constraints above but decided together: `CMAES_N_BOUNDS`/
`CMAES_TIGHT_N_BOUNDS` lower bound in `opt_config.py`, `params.py` only
ever asserted `n≥1` — the old 50 had no material basis, and a same-day
diagnostic (`floor_test.py`, see the postmortem below — its actual
NUMBERS were corrupted by a race condition, but the motivating pattern
was independently confirmed by inspecting the un-corrupted per-layer-index
data: roughly half the layers in every converged design, across every
layer count ≥6, sat pinned exactly on the old 50 floor) supported dropping
it.

### Same-session postmortem: two bugs found running the first attempt at this

Before landing on the clean re-implementation above, an initial attempt
ran two orchestrators in parallel (`focused_refinement_6_9.py` targeting
n_layers=9/6, `floor_test.py` diagnosing the turn floor on 12/10/9/8/6)
and produced results that looked exciting (n_layers=12 "improving" from
0.274km to 0.199km) before turning out to be wrong. Both orchestrators
and their full process trees (including `ProcessPoolExecutor` worker
pools) were killed once this was caught. Root causes, both now fixed and
worth not re-breaking:

- **Race condition on shared output paths.** `cfg.CMAES_OUT_CSV`
  (`optimize/runs/cmaes_results.csv`) and `cfg.CMAES_OUT_LOG`
  (`optimize/runs/cmaes_history.csv`) are shared, hardcoded defaults —
  `_write_csv()` in `cmaes_search.py` fully overwrites (not appends) both
  on every periodic flush. Two `cmaes_search.py` subprocesses alive at
  once (one from each orchestrator, both happening to be on n_layers=9 at
  the same time) silently clobbered each other's files. Caught by a
  telltale symptom: a reported "12-layer" turns list had only 9 entries
  after `zip()` silently truncated against a stray 9-layer read.
  `optimize/runs/cmaes_all_evaluations.csv` (the cumulative master log) was
  NOT affected — it's append-only and every row is tagged with its own
  `run_tag`, so it remained the source of truth for recovering the real
  numbers. **Fix:** `CMAES_OUT_CSV_OVERRIDE`/`CMAES_OUT_LOG_OVERRIDE` env
  vars added to `opt_config.py` — any orchestrator launching concurrent
  `cmaes_search.py` jobs MUST set unique values per job. (This session's
  actual double-pancake re-search, `double_pancake_search.py`, sidesteps
  the whole class of bug by running strictly sequentially instead — the
  override exists for future parallel use.)
- **Oversized turn-count step size for warm-started "polish" runs.** The
  `focused_refinement_6_9.py` n_layers=9 job proportionally scaled its
  `a`/`b` step size (`CMAES_A_STD0_OVERRIDE`/`CMAES_B_STD0_OVERRIDE`, the
  fix from the earlier n_layers=5 round-2 regression) but left the
  turn-count step size at its old formula,
  `CMAES_SIGMA0_FRAC × (bound_range)` ≈ 135–150 — enormous next to a
  typical per-layer turn count of 50–250. Eval #1 of an intended "local
  refinement" run sampled turns of `[316,114,221,350,...]`, nothing like
  the `[155,194,242,241,...]` warm start; after 3000+ evaluations and 2+
  hours it was stuck at 0.171km, WORSE than the 0.152km it started from,
  having never found its way back. Same failure mode as the n_layers=5
  regression, just in a dimension that hadn't been fixed yet. **Fix:**
  `CMAES_N_STD0_OVERRIDE` env var added to `opt_config.py` (mirrors the
  a/b overrides) — any future warm-started polish run must set this
  proportionally, not rely on the bound-range-derived default (which
  remains correct/intended for genuine cold starts).
  **⚠ That "fix" was INERT until 2026-07-30** — `cmaes_search.py` read it
  under the wrong attribute name, so it never had any effect and every
  polish run between those dates got the oversized default anyway. See
  the 2026-07-30 bug section below.

### Current re-search (`optimize/studies/double_pancake_search.py`, 2026-07-23)

Rebuilds a starting point for each of the four already-well-explored even
layer counts (6, 8, 10, 12 — all from clean, non-corrupted earlier runs)
under all three new constraints: turns pair-averaged from the prior
(pre-constraint) champion, `a` recomputed directly from the new 7.5mm
floor (`pack_thickness/2 + 7.5mm + 0.3mm margin`) rather than left deeply
infeasible, `b` shifted by the same delta to preserve the prior straight
length. All four warm starts verified feasible (`geometry_violation()` =
0) and within bounds before launch. Odd counts are not included (not
buildable). Runs strictly sequentially — full worker parallelism
(`CMAES_N_WORKERS=6`) per job, no risk of the race condition above. 8h
total budget, 90-120 min per job depending on layer count.

Expect tape lengths noticeably above the old (now-invalid) champions —
the 7.5mm bend radius is a real, binding cost, not a search artifact.

Progress: `optimize/runs/double_pancake/double_pancake_log.txt` (summary),
`optimize/double_pancake_logs/n_layers_NN.log` (full per-job output). Feeds
the same cumulative `optimize/runs/cmaes_all_evaluations.csv` — pull results by
`run_tag`, not `optimize/double_pancake_results.csv` alone (only reflects
the most recently finished job).

### Results (completed 2026-07-24) — new champion: 10 layers, 0.1938km

All four jobs finished. **n_layers=10 is the new overall best**, beating
6/8/12 clearly:

| n_layers | tape_km | B_target_T | unif% | hoop_MPa | a_mm | b_mm | gap_mm | n_turns (per-layer) | run_tag |
|---|---|---|---|---|---|---|---|---|---|
| **10 (BEST)** | **0.1938** | 10.05 | 0.88 | 71 | 15.64 | 20.64 | 21.92 | [152,152,217,217,212,212,211,211,2,2] | run_20260723_162315 |
| 8 | 0.2049 | 10.22 | 0.98 | 85 | 16.92 | 23.16 | 17.53 | [171,171,250,250,243,243,67,67] | run_20260723_140808 |
| 12 | 0.2380 | 10.01 | 0.94 | 111 | 20.14 | 25.23 | 25.51 | [231,231,326,326,45,45,106,106,5,5,1,1] | run_20260724_000938 |
| 6 | 0.2258 | 10.00 | 0.99 | 114 | 22.20 | 27.27 | 13.50 | [285,285,379,379,2,2] | run_20260723_124414 |

Turn PAIRS (one value per double-pancake, i.e. what CMA-ES actually
optimizes): 10→[152,217,212,211,2]; 8→[171,250,243,67];
12→[231,326,45,106,5,1]; 6→[285,379,2].

**Every one of the four designs sheds its last pair down to the new turn
floor** (2, 67, 1, 2 respectively — note n=8's *last* pair is 67, not
near-floor, but its *third* pair distribution still shows the same
tapering shape). This is a real, recurring signature, not noise — it
showed up independently in all four separately-seeded, separately-run
searches. It suggests even the n_layers=10 winner may not be using its
full pair count efficiently, i.e. the true unconstrained optimum might
sit at effectively "9.x" pairs' worth of winding. Worth testing directly
(e.g. rerun n_layers=10 with `CMAES_TIGHT_N_BOUNDS` upper bound tightened
around the other 4 pairs to force more budget there) as a follow-up, not
done yet.

**Third bug found this run, for the record:** the orchestrator's
*total* budget check in `main()` uses wall-clock `time.time()`, unlike
each job's own per-job cap (`subprocess.wait(timeout=...)`, which uses
Python's monotonic clock and correctly ignores time spent suspended). A
long gap where the user paused the run via `SIGSTOP` (intentional, see
below) meant wall-clock time between launch and the third job finishing
exceeded the nominal 8h total budget purely from elapsed calendar time,
even though actual working time was much less — so the orchestrator
correctly finished n_layers=10 but then skipped n_layers=12 and exited.
Not a data-integrity bug (nothing was lost or corrupted, unlike the
earlier two), just a scheduling one. Recovered by manually launching
n_layers=12 as a standalone `cmaes_search.py` process (same env-var
override mechanism, wrapped in `timeout 120m` for a clean self-imposed
cap) — see `optimize/runs/double_pancake/double_pancake_log.txt` for the exact commands used.
**Fix for next time:** if `main()`'s total-budget check is ever revisited,
use `time.monotonic()` instead of `time.time()` for `t_start`/`elapsed`,
consistent with the per-job timeout.

**Pause/resume tested and confirmed safe mid-run:** the run was paused
for ~1.5h (14:14→15:49) via `SIGSTOP` on the full process tree (orchestrator
+ `cmaes_search.py` subprocess + all `ProcessPoolExecutor` workers) and
resumed with `SIGCONT` — zero progress lost, evaluations resumed exactly
where they left off. This is a confirmed-safe pattern for any future
long-running search if compute needs to be freed up temporarily —
**but ONLY for a job launched from a shell (`nohup`/`disown`), NOT for
one launched as a Claude Code background task.** Corrected 2026-07-30:
`SIGSTOP` on a harness-managed background task makes it report exit code
147, which the background-task manager treats as a failure and responds
to by reaping the entire process tree — the run *ends* instead of
suspending, and `SIGCONT` cannot bring it back. (No data was lost when
this happened, thanks to `_record()`'s incremental CSV flushing, but the
last candidate had to be re-run.) To pause a harness-launched run,
there is no in-place option: let it finish, or kill it and restart from
its flushed CSV state.

`optimize/opt_config.py`'s `CMAES_X0` and this doc's status section
have been updated to the n_layers=10 champion.
`visualization/cmaes_{convergence,constraints,variables,overview}.png`
and `cmaes_param_map.png` have been regenerated from this run.

**SUPERSEDED same day, see below** — the 10-layer champion's own two
small pairs turned out to be a deadweight artifact once compared
apples-to-apples against an 8-layer truncation; the 0.8%-uniformity
retarget and the 8-layer champion below replace this as current.

---

## Tightened uniformity target, the 8-layer champion, and an n_layers=4 investigation that got rejected (2026-07-24)

### 1. Is the 10-layer champion's last pair deadweight?

Direct test: took the 10-layer champion's design, dropped its collapsed
5th pair (2,2 turns), re-evaluated as 8 layers with `coil_half_gap`
**compensated** to preserve the same physical clearance between the
near-midplane layer and the midplane (`recompute_derived()` always
re-centers the winding stack around coil 1's fixed z=0, so just changing
`n_layers` without compensating gap silently moves every layer — an
earlier *uncompensated* version of this test gave a badly misleading
uniformity blowup, a pure geometry-bookkeeping artifact). Properly
compensated (`gap = z_top(8L) + [gap(10L) − z_top(10L)]`), the truncated
8-layer design scored statistically indistinguishable from the 10-layer
original. **Conclusion: the dropped pair was genuine deadweight.**

Layer-position convention used throughout this section (confirmed
directly from `params.recompute_derived()`): layer index 0 (first entry
in `n_turns`) sits at the **top of the local stack, closest to the
midplane**; the last entry sits farthest from it. Coil 1's own center is
always pinned at global z=0, independent of `n_layers`.

### 2. Uniformity target tightened to 0.8%

Repeated evaluation of the identical 10-layer champion design (three
separate process launches, same exact geometry) gave uniformity readings
of 0.88%, 1.12%, and 1.27% — a ~0.4 percentage-point spread on a design
sitting right at the 1.0% limit. `UNIFORMITY_MAX_PCT` in `opt_config.py`
was tightened **1.0% → 0.8%** so the optimizer builds in real margin
against this (see §4 below for an important correction to what this
noise actually was).

### 3. Focused n_layers=8 search under the new target — current champion

Warm-started from the gap-compensated 8-layer truncation
(§1), `CMAES_SEED=80824`, run_tag `run_20260724_103646`. Finished
naturally at `CMAES_MAX_EVALS=2500` (~102 min, inside its 150 min cap).
Converged smoothly: 0.3645km (eval 136) → 0.2690km → 0.2334km → **0.2327km
final**, no regression.

**CURRENT CHAMPION: 8 layers, tape=0.2327km @ 10.00T, unif=0.588%,
hoop=114MPa** — a=21.97mm, b=27.57mm, gap=17.50mm,
n_turns=[284,284,385,385,3,3,6,6]. Two pairs again collapsed near the
turn floor (3, 6) — the same recurring "sheds a pair" signature. Under
the *looser* 1.0% target this same search space produces designs around
0.19-0.20km — tightening the uniformity margin costs roughly **+20-25%
tape** for a comparable design. That is the real, measured price of
building in margin, not a modeling artifact.

`params.py`, `opt_config.py`'s `CMAES_X0`, and the `cmaes_*` /
geometry / field figures all reflect this design as of this writing.

### 4. n_layers=4 investigation — promoted, then REJECTED after further testing

The 8-layer champion's own two small pairs (3,3 and 6,6 — the two pairs
**farthest** from the midplane, per §1's convention) show the same
"collapsing toward deadweight" look. Direct truncation test (gap
re-compensated the same way, floor for n_layers=4 = 9.5mm): tape=0.2282km
(only ~2% less — those pairs were already small), but B_target=9.93T
(just under the 10T floor) and unif=0.88% (passes the *old* 1.0% target,
fails the *current* 0.8% one) — NOT a clean pass this time, unlike §1's
10L→8L case, but close enough on both fronts to warrant a real search
rather than dismissing it.

**The trial** (`CMAES_SEED=40824`, run_tag `run_20260724_122521`) was
launched and monitored live. Partway through, the population visibly
converged to a single point (a≈22.8mm, b≈28.0mm, n_turns=[289,289,387,387])
yet kept reporting **wildly different uniformity readings for that
IDENTICAL candidate — 0.49% to 1.29% across repeats within the live
search**. Per project direction, the run was stopped early (further
CMA-ES search can't distinguish real improvement from noise once the
population has collapsed like this) in favor of directly characterizing
the noise on the best point found (eval 1631, tape=0.2352km, unif=0.79%).

**First characterization — misleadingly reassuring.** A 20-repeat
in-process test (`optimize/n4_robustness_test.py`: loop calling
`og.evaluate()` on the byte-identical candidate 20 times within one
Python process) showed **essentially ZERO variance**: uniformity
std=0.0002 percentage points, 20/20 pass, mean 0.790%. Taken at face
value, this looked like a clean, reproducible pass with the same tape
length as the 8-layer champion at ~1% less tape. On this basis, the
design was **promoted**: `params.py`, `opt_config.py`'s `CMAES_X0`, and
all figures were updated to this n_layers=4 design as the new champion.

**Second characterization — the promotion was wrong.** Cross-checking the
*same exact* design against an **independently generated mesh** (running
`solve.py` as a separate process — the same check `field_uniformity.py`
performs, which the project already treats as an authoritative
cross-check against the optimizer's own internal number) gave
**uniformity = 2.19%, a clear FAIL** — a >20x swing from the 0.79% both
the optimizer's own evaluation *and* the 20x in-process repeat had agreed
on. Isolated the discrepancy to the SCIF (Bean-state dipole) correction
specifically: the raw (pre-SCIF) target-box field agreed to 0.01% between
the two mesh realizations (confirming the plain Biot-Savart part is
solid), but the SCIF correction itself swung from +0.04pp (evaluate()'s
own mesh) to +1.44pp (the independent mesh) — the signature of a
near-cancelling dipole sum (the same class of sensitivity documented
elsewhere in this file for the full T-A SCIF calculation) being amplified
by small, normal cross-process mesh differences.

**Root cause, confirmed:** gmsh mesh generation is **not perfectly
reproducible across separate process launches**, even for byte-identical
geometry and identical `build_mesh.build()` calls — only *within* a
single process (which is exactly why the 20x in-process repeat test saw
none of this: every repeat reused the same process's mesh generation
state). During the live CMA-ES search, each of `CMAES_N_WORKERS=6`
parallel pool workers is its own OS process — so the 0.49–1.29% swings
seen live were real, not a fluke, and came from exactly this mechanism.

**Is this a general problem with this session's other results?** Checked
directly: the SAME independent-mesh cross-check applied to the 8-layer
champion gives uniformity=0.72% (vs. the optimizer's own 0.59%) — both
still comfortably under 0.8%, nothing like the 4-layer design's >20x
swing. **This is a real, n_layers=4-specific fragility** (plausibly because
its much higher per-layer current density, concentrated into only 2
distinct z-positions instead of 4, pushes the Bean-state dipole sum
further into the near-cancelling regime), **not a general flaw in this
session's work.**

**VERDICT: REJECTED.** `params.py`, `opt_config.py`'s `CMAES_X0`, and all
figures were reverted to the 8-layer champion (§3). The n_layers=4
trial's data is preserved (`optimize/n4_trial_*`,
`optimize/n4_robustness_test.py`) for the record but is NOT the current
design.

**General lesson for future work — do not skip this again:** a design's
own single-mesh uniformity number, and even a same-process repeated-
evaluation test, are NOT sufficient validation for a design sitting near
a constraint boundary. Always cross-check against an *independently
generated mesh* (e.g. `solve.py` + `field_uniformity.py`, run as a
separate process from whatever produced the candidate) before promoting
any design, especially one with high per-layer current density or few
distinct z-positions. This is cheap (~1-2 min) relative to the cost of
being wrong.

**Immediately superseded by a much bigger finding — see the next
section.** The mesh-fragility investigation above turned out to be a
comparatively minor issue.

---

## Coarse-screen SCIF proxy found unreliable by up to ~10x; methodology changed (2026-07-24)

### How this was found

Prompted by a practical engineering objection — a double-pancake with
only 3 or 6 turns is not something you'd actually build — the 8-layer
champion's two small pairs were tested for whether they were worth
keeping at all, using the project's authoritative validation tool (the
full per-layer T-A Picard solver, `solve/ta_solve.py`) instead of the
coarse optimizer screen. It converges cleanly in well under two minutes
for these compact designs, so it was run on **every layer count tried
this session**, using each one's actual champion geometry and I_op:

| Layers | Peak turns/layer | Coarse-screen `uniformity_pct` | **TRUE on-axis SCIF (T-A)** | Tape |
|---|---|---|---|---|
| 6  | 379 | ~0.99% (reported PASS) | **5.68%** (fails badly) | 0.226km |
| 8  | 385 | 0.588% (reported PASS, best of the five) | **5.54%** (fails badly) | 0.233km |
| 10 | 217 | 0.88% (reported PASS, only 3rd best) | **1.37%** (best by far) | 0.194km |
| 12 | 326 | 0.94% (reported PASS) | **5.78%** (fails badly) | 0.238km |
| 4  | 387 | 0.79% (reported PASS) | **5.61%** (fails badly) | 0.235km |

The coarse screen said all five designs comfortably pass. The T-A ground
truth says only n_layers=10 comes anywhere close. Worse than being
merely noisy, **the proxy is actively anti-correlated with truth on this
data**: it ranked the design with the *worst* true SCIF (8 layers) as
its *best* score, and ranked the design with by far the *best* true
SCIF (10 layers) only third out of five.

### Root cause: peak per-layer turn concentration, not layer count

The pattern is not about how many layers a design has — it's about how
evenly the total current is spread across them. Every design tried with
peak turns/layer in the ~320-390 range shows ~5.5-5.8% true SCIF,
*regardless of whether that's spread across 4, 6, 8, or 12 layers*. The
one outlier, n_layers=10, has a much more evenly-distributed turn
profile (max 217, vs. 320-390 for the others) — and its true SCIF is
~4x lower. This is physically sensible: screening-current effects scale
with local current density, and spreading the same total transport
current more evenly lowers the peak concentration that drives it. The
coarse Bean-state proxy (`bean_moments()`/`dipole_field_mirrored()` in
`optimize_geometry.py`) is derived from exactly this same physical
picture, but its *absolute calibration* breaks down badly at the compact
scale (a≈15-25mm) the CMA-ES search converges to — it was only ever
validated against the original ~50-80mm baseline design (13% agreement
there). A focused investigation (checking Ic-extrapolation clipping
fraction, the `i_loc`/taper distribution, and per-cell magnetization
stats) did not turn up a single obvious formula bug explaining the ~10x
gap — it did not look like a quick, confidently-correctable fix, so none
was attempted blind. (**This is exactly why we validated with T-A rather
than guessing at the proxy** — see the mesh-fragility section above for
what happened last time an unvalidated number was trusted.)

### What was NOT changed (still trustworthy)

`tape_km` (pure geometry), `B_target_T` (the multi-filament Biot-Savart
field, cross-validated to 0.01% between independent mesh realizations
earlier this session), and `hoop_MPa` (linear Lorentz-force calc off the
FEM reference field) are **not** affected by this bug — none of them
depend on the Bean-state SCIF correction. Only `uniformity_pct` is
unreliable.

### Methodology change made to `optimize/cmaes_search.py`

1. **`uniformity_pct` removed from the CMA-ES fitness/constraint list.**
   It's still computed and logged (for reference/future recalibration)
   but no longer penalizes candidates — keeping a misleading signal in
   the objective actively steered the search in the wrong direction, as
   the table above shows.
2. **New constraint: `cfg.MAX_TURNS_PER_PAIR_TARGET = 250`** (in
   `opt_config.py`) penalizes peak per-pair turn concentration directly
   in the fitness function — the validated true driver. **This is
   PROVISIONAL**, interpolated between one good T-A data point (217) and
   several bad ones (320-390) — not a precisely calibrated threshold.
   Refine with more T-A checks (e.g. designs with max~250, ~280) before
   trusting it too literally. Set to `None`/`0` to disable.
3. `all_constraints_ok` (the "all-pass"/green-vs-orange classification
   throughout the CSV logs and figures) now means "B_target + hoop +
   peak-turns all satisfied" — no longer requires the broken uniformity
   check. Old runs' *recorded* `all_constraints_ok` values predate this
   change and still reflect the old (misleading) definition; the
   `cmaes_*` figures for the current champion (see below) were
   regenerated with `all_constraints_ok`/`fitness` **recomputed** from
   each row's raw metrics under the new definition, not read from the
   stale stored values.

### Current champion — promoted, T-A-validated

**n_layers=10, tape=0.1938km, a=15.64mm, b=20.64mm, gap=21.92mm,
n_turns=[152,152,217,217,212,212,211,211,2,2], TRUE on-axis SCIF=1.37%.**
This is simultaneously the best tape length of everything tried this
session (on the metrics that were never broken) *and* by far the best
validated screening behavior. `params.py` and `opt_config.py`'s
`CMAES_X0` are set to it; `cmaes_*`, `geometry.png`, `field_top.png`,
`field_side.png`, `field_3d.png` are regenerated from it.
`uniformity.png` was NOT regenerated for it — it's built on the same
broken Bean-state proxy `field_uniformity.py` shares with the optimizer
(see that file's docstring: it deliberately mirrors
`optimize_geometry.py`'s SCIF correction to agree with it) and should
not be trusted until that proxy is fixed. **This design's true 1.37% is
an ON-AXIS point value, not yet the true peak-to-peak BOX uniformity —
no tool exists yet to compute the latter from a T-A solve** (see next
steps). It's a strong leading indicator (4x better than every
alternative), not a final confirmed PASS against the 1.0%/0.8% target.

### Recommended next steps

1. **Build a T-A-based box-uniformity tool.** `ta_solve.py`'s
   `dB_bore_from_dJ(cents, dJ_s, dV, bore_pt=None)` already accepts an
   arbitrary evaluation point — extend it to a grid over the 30×6mm
   target box (the way `field_uniformity.py` does for the broken proxy)
   to get a true peak-to-peak number. `dV` (`ta["coil_vols"]`) isn't
   currently saved in `racetrack_ta_fields.npz` and would need adding.
   **DONE, same day — see the next section.** This turned out to be the
   single most important step: the on-axis number was actively
   misleading, not just imprecise.
2. ~~Refine `MAX_TURNS_PER_PAIR_TARGET`~~ **MOOT — that constraint has
   been REMOVED.** It was built entirely from on-axis SCIF data; once
   real box uniformity was measured, peak-turns turned out not to track
   it at all. See the next section.
3. ~~Re-run searches under the new methodology~~ **Still valid, but the
   "methodology" it refers to has changed again — see next section for
   what's actually in the fitness function now (just B_target + hoop;
   no uniformity signal at all until a trustworthy one exists).**
4. Consider whether the Bean-state proxy formula itself is worth properly
   re-deriving/recalibrating for this compact-coil regime (a≈15-25mm) —
   still open, now higher priority since no working proxy replaced it.

---

## Box uniformity is the real target — on-axis SCIF was actively misleading (2026-07-24, later the same day)

### The correction

The section above treated on-axis SCIF as if it were a usable stand-in
for real uniformity, with an open caveat that it hadn't been checked
against the actual peak-to-peak box target. It was checked, the same
day, and **the caveat mattered enormously**: `ta_solve.py` was extended
to evaluate `dB_bore_from_dJ()` over a grid spanning the real 30×6mm
target box (matching `opt_config.py`'s `TARGET_X_M`/`TARGET_Y_M`) instead
of just the single on-axis point, combined with the uniform-J box field
the same way the coarse proxy does. Applied to every layer count tried
this session:

| Layers | a (mm) | Peak turns/layer | On-axis SCIF | **Box peak-to-peak (TRUE)** | Tape |
|---|---|---|---|---|---|
| 4  | 22.80 | 387 | 5.61% | **0.436% — PASS (best)** | 0.235km |
| 6  | 22.20 | 379 | 5.68% | **0.731% — PASS** | 0.226km |
| 8  | 21.97 | 385 | 5.54% | 1.059% (nearly) | 0.233km |
| 12 | 20.14 | 326 | 5.78% | 2.404% (fail) | 0.238km |
| 10 | 15.64 | 217 | 1.37% | **9.18% (fail, badly)** | 0.194km |

The on-axis metric and the box metric are not just uncorrelated here —
**they're inverted**: n_layers=10 has the *best* on-axis number (a
near-cancelling sum that happened to land favorably) and the *worst* box
number by a wide margin. The peak-per-pair-turns penalty added to
`cmaes_search.py` earlier the same day, reasoning from the on-axis data,
is consequently **also wrong** — box uniformity doesn't track peak turns
(12 has fewer peak turns than 4/6/8 but worse box uniformity; 10 has by
far the fewest peak turns and by far the worst box uniformity). **That
penalty has been removed from the fitness function.**

### What actually tracks box uniformity: coil radius `a`

Sorted by `a`, box uniformity is close to perfectly monotonic — bigger
coil radius, better uniformity — independent of layer count or peak
turns. This is physically sensible: the target box is a **fixed** 30×6mm
regardless of coil size, so a smaller coil means that fixed-size box eats
into proportionally more of the near-field region where gradients are
steep. No penalty on `a` has been added to the optimizer yet: `a` is
already a free, unbounded search variable, and biasing it upward directly
fights against minimizing `tape_km` (smaller coils generally need less
tape for the same field) — a genuine physical tradeoff this project
hasn't resolved into a single fast proxy term, not just an
implementation gap. Until it does, `cmaes_search.py`'s fitness function
carries **no uniformity signal at all** (only B_target and hoop) —
deliberately, rather than another guessed-at proxy that might again be
wrong in a way that isn't obvious until checked.

### Current champion: n_layers=6

**tape=0.2258km, a=22.20mm, b=27.27mm, gap=13.50mm,
n_turns=[285,285,379,379,2,2], B=10.00T, hoop=114MPa, box peak-to-peak
uniformity=0.731% (T-A-validated PASS).** This is the ONLY tape-optimal
design among those with a validated passing uniformity (4 layers is
slightly better on uniformity alone, 0.436%, but costs more tape,
0.235km vs 0.226km). `params.py`, `opt_config.py`'s `CMAES_X0`, and all
figures (`cmaes_*`, `geometry.png`, `field_top.png`, `field_side.png`,
`field_3d.png`) are set to/regenerated from it. This is the **first, and
so far only, genuinely validated design of the entire 2026-07-23/24
optimization effort** — every other champion recorded in this document's
history turned out to be an artifact of one broken proxy or another once
actually checked.

### Recommended next steps

1. **Every champion in this session's history was found under an
   objective with no working uniformity signal** (first the broken
   on-axis proxy, then the equally-broken peak-turns proxy, now nothing
   at all). A fresh search — even a purely tape/B_target/hoop-driven one
   with no uniformity term — followed by T-A box-uniformity validation of
   its top few candidates, is likely to find something at least as good
   as 0.226km, possibly better, since the search space hasn't actually
   been explored with real information about what drives uniformity.
2. **Investigate whether `a` alone is sufficient**, or whether it's a
   proxy for something else (e.g. the ratio of box size to some coil
   dimension, or gap, which correlates with `a` in this family of designs
   since larger coils also need larger `n_layers*w/2`-driven gaps). Only
   5 data points exist; a deliberate sweep at fixed layer count varying
   only `a` would isolate this cleanly.
3. **Build a fast, trustworthy uniformity proxy** if a genuinely fast
   (coarse-screen-speed) search is still wanted — informed by the `a`
   correlation above, not another guess. Until then, treat the coarse
   screen as tape/B_target/hoop-only and always T-A-validate finalists.
4. Consider whether the Bean-state proxy formula itself is worth properly
   re-deriving/recalibrating — now higher priority, since two successive
   guessed replacements (on-axis SCIF, peak-turns) both turned out wrong.

---

## `a`-isolation sweep: the "bigger `a` is monotonically better" claim above is WRONG (2026-07-24, later still)

Item 2 above ("investigate whether `a` alone is sufficient") was tested directly:
took the 6-layer champion and radially translated the WHOLE coil (`a` and `b`
shifted together by the same delta, so `b−a`, `n_turns`, `gap`, and
`I_design` are all held exactly fixed — a clean single-variable isolation,
unlike the cross-layer-count comparison the monotonic claim was based on,
where `a`, gap, and turn distribution all varied together). 5 T-A box-
uniformity solves, ~25-80s each (`solve/ta_solve.py` is fast enough now to
use directly instead of any proxy):

| delta | a (mm) | box peak-to-peak |
|---|---|---|
| −6mm | 16.20 | 8.478% |
| −3mm | 19.20 | 1.900% |
| **0 (champion)** | **22.20** | **0.731% (best)** |
| +3mm | 25.20 | 1.015% |
| +6mm | 28.20 | 2.443% |
| +9mm | 31.20 | 3.496% |

**This is a smooth, symmetric-ish V-shape (bowl) with its minimum sitting
almost exactly ON the champion's own `a` — not a monotonic curve.** Moving
`a` either smaller OR larger, with everything else held fixed, makes box
uniformity worse. The earlier "bigger coil radius, better uniformity"
conclusion was an artifact of comparing 5 designs that differed in `a`
*and* layer count *and* gap *and* turn distribution simultaneously — `a`
happened to correlate with the ranking in that particular set of 5, but
isolating it properly shows it is not a simply-increasing lever at all.

**Implication:** the 6-layer champion's specific (a, b, gap, n_turns)
combination is a genuine, fairly narrow local optimum for box uniformity
(±3mm of `a` alone costs 0.3-1.2 percentage points), not an accident of
"large enough `a`." It was found by CMA-ES optimizing tape/B_target/hoop
only, with no uniformity signal — landing this close to a real uniformity
sweet spot by chance is notable, and means a fresh blind search is NOT
guaranteed to reproduce anything this good on uniformity; it needs to be
checked directly, not assumed.

**Methodology consequence:** since a single T-A box-uniformity solve only
costs ~30-80s (confirmed by this sweep, not the ~280s worst-case quoted
elsewhere in this doc for a harder-converging tape/mesh combination), the
practical path is not "build a fast coarse proxy, trust its ranking" —
it's "run the blind tape/B/hoop-only CMA-ES search as already configured,
then T-A-validate a batch of its best candidates directly" (minutes of
compute, not a new proxy-derivation effort). This is what task tracking
for this session now does: launch the search, then screen its top-N
all-pass-by-tape candidates through `ta_solve.py`'s box-uniformity check
and keep whichever one is both low-tape and genuinely passing.

---

## 2026-07-27: widened search confirms the champion, plus a real risk found on it

Same day as the last section but a new session: `params.py` was found in
a corrupted, half-mutated state at the start (a WSL crash killed a script
mid-way through in-place-editing it to spot-check a CMA-ES candidate,
leaving a comment block describing the 6-layer champion sitting above
completely different live values tagged `[eval404 candidate check]`).
Restored to the exact champion values by cross-referencing
`optimize/runs/cmaes_all_evaluations.csv` (`run_20260723_124414`, eval 1759) —
this is the same incident that motivated `optimize/studies/day_search.py` below
never writing to `params.py` on disk at all, only mutating in-process
module state or passing overrides via env vars, so it can't happen again.

### `optimize/studies/day_search.py` — widened search (6/8/10/12/14/16 layers), T-A validated

Built to directly test the "re-run searches now that box uniformity is
understood" recommendation from the section above. Three phases, run
unattended: **A** — re-run the discrete n_layers outer loop (6, 8, 10, 12
now with 14 and 16 added for the first time) with a new soft floor on
coil radius `a` (`CMAES_MIN_A_M` in `opt_config.py`) informed directly by
the a-isolation sweep above, since every blind search to date wastes
budget converging toward small `a` with no uniformity signal to stop it.
**B** — T-A-validate every Phase-A winner (2 repeats each, mesh
generation isn't perfectly reproducible run-to-run) via the new
`optimize/ta_validate.py`, a standalone extraction of `ta_solve.py`'s
box-uniformity machinery that never touches `params.py` on disk. **C** —
stress-test the winner against relaxed safety-factor/hoop assumptions and
Ic-extrapolation risk.

**The `a`-floor mechanism needed two live corrections, both confirmed by
direct T-A checks, not assumed:** CMA-ES pins any free variable with a
soft floor exactly at that floor once nothing else opposes it (zero
fitness gradient past the constraint boundary) — confirmed 3 times,
including one case where a promising mid-run snapshot (`a`=22.22mm)
regressed back to the floor by the run's own final convergence. First
floor (18mm) produced a design real-checked at 12.14% box uniformity
(vs. the champion's 0.73-0.83%) — raised to 21.5mm, still real-checked at
3.24% — raised again in-place (editing `opt_config.py` while the running
orchestrator's later, not-yet-started jobs picked up the new value
automatically, since each job is a fresh subprocess reading the file
fresh) to 22.2mm, matching the champion's own radius almost exactly.
**A separate, real bug was also caught and killed mid-run**: one job's
population fully collapsed to an identical candidate yet kept reporting
different B_target/hoop/uniformity values for it — the same cross-process
gmsh mesh-noise signature documented in the n_layers=4 episode above,
confirming CMA-ES had nothing left to learn from continuing that job.

**Phase B result — champion re-confirmed, nothing beat it:**

| candidate | a (mm) | tape_km | T-A box p2p uniformity | verdict |
|---|---|---|---|---|
| **6L champion (reference)** | 22.227 | 0.226 | **0.83%** | **PASS** |
| phaseA 6L | 21.50 (pinned at floor) | 0.178 | 4.46-4.49% | FAIL |
| phaseA 8L | 21.50 (pinned at floor) | 0.188 | 3.41% | FAIL |
| phaseA 10L | 23.47 (settled naturally) | 0.212 | 3.47-3.48% | FAIL |
| phaseA 12L | 24.81 (settled naturally) | 0.232 | 4.10% | FAIL |
| phaseA 14L | ~22.2 (at floor) | 0.225 | 3.09-3.10% | FAIL |
| phaseA 16L | ~22.2 (at floor) | 0.221 | 3.85% | FAIL |

Confirms (again) that `a` alone doesn't determine uniformity — phaseA_8L
sits at essentially the champion's own radius yet scores 4-5x worse, so
the champion's specific (a, b, gap, n_turns) combination remains a narrow,
not-yet-reproduced-elsewhere optimum. Full writeup, including the
now-superseded first attempt (18mm floor) and every intermediate check:
`optimize/runs/day_search/day_search_report.md`.

**Phase C — the most important finding of this session, an unresolved
risk on the CHAMPION ITSELF, independent of the search above:** every Ic
lookup in this entire project defaults to `clip_B=True`
(`physics/ic_model.py`), which flat-clamps Ic to its measured B=8T value
for any cell above that field — since Ic decreases with B in the measured
range, this is an OPTIMISTIC assumption for cells above 8T, not a
conservative one (the opposite of what "clipping/extrapolating" usually
implies). Re-evaluating the champion's fixed geometry under a linear
continuation of the measured B=8T slope instead (a new `ConservativeIcModel`
wrapper in `optimize/studies/day_search.py`, swapped in for the normal `IcModel`
with no other code changes needed since `optimize_geometry.evaluate()`
takes the Ic model as a parameter): **B_target drops from 10.00T to
6.51T (-34.9%), falling below the 10T design floor.** 11.8% of the
champion's own quench-point Ic evaluations already clip to the 8T
boundary (`clip_frac=0.118`) — not a remote edge case. Session was
stopped (by explicit user request, to go work on poster figures) before
finishing the rest of Phase C (a fixed-geometry safety-factor sensitivity
sweep got one data point — SF=1.3 reaches B_target=14.71T at clip_frac
0.231 on the same geometry — before stopping; the warm-started
tape-savings-vs-relaxed-margin re-optimizations never ran).
**This is the top-priority open item for this design** — see "Known
design issues" below.

### Visualization mirror-symmetry bug — found, fixed (real bug, predates this session)

While building poster figures, noticed the "thin" (2-turn) layer sitting
at the *bottom* of both coils in `field_3d_poster.png`, when the two
coils should be mirror-symmetric about the midplane (thin layer at the
*outer* face of each coil, symmetric). Root cause: `visualization/
plot_3d.py`'s `_expand_to_full_system()` (and, copying its pattern, the
guide-line loop in `plot_field_3d()`, the layer-shading loop in
`plot_fields.py`'s `plot_side()`, and everything in the new
`plot_field_poster.py`) placed coil 2 by **translating** coil 1's
geometry by `+2*coil_half_gap`, not by mirroring it about the midplane.
Translation and mirror give the same picture only for a palindromic
layer stack; the champion's `[285,285,379,379,2,2]` is not one, so this
silently drew coil 2's layers in the wrong relative order. Confirmed via
`physics/coil2_field.py`'s `compute_both_coils_field_multilayer()` — the
function actually used for every real design number (B_target,
uniformity, tape optimization, T-A SCIF) — which correctly places each
layer at `2*g - z_c` (a true mirror, matching the PMC boundary condition
the FEM solve itself uses at the midplane specifically because the real
two-coil system is mirror-symmetric). **This was a picture-only bug** —
every actual physics number in this project was unaffected, since none
of them went through the buggy code path. Fixed by adding a shared
`_mirror_z(z, g) = 2*g - z` helper to `plot_3d.py` and using it
everywhere the old `+2g` pattern appeared; regenerated `field_3d.png`,
`geometry.png`, `field_top.png`, `field_side.png`, `quench_3d.png`, and
`field_3d_poster.png`. `visualization/plot_3d.py`'s
`plot_quench_2d()` has an unrelated, pre-existing bug (indexes
`params.a_center_list` with a stale layer index from an old
`sweep/quench_results.csv` that predates the current 6-layer geometry) —
noticed but not fixed, since `sweep/quench_results.csv` needs regenerating
from the current champion first (`sweep/quench_sweep.py`).

### New poster figures (`visualization/`)

Built for an external poster, iterated live with the user across several
rounds each:
- **`field_3d_poster.png`** — white background, solid coils rendered as a
  scattered point cloud (tried filled/translucent surfaces first; kept
  hitting mplot3d depth-sorting artifacts — a cutaway face blending into
  the outer wall with no shading cue, then a "shell of the outer layer
  cutting into the center" artifact — abandoned in favor of points, which
  have no opaque shell to misrender), a wedge cut out of one end-cap
  exposing the interior (points within a WEDGE_HALF_ANGLE_DEG angular
  sector simply dropped), a wireframe outline per layer (both the outer
  curved surface and the exposed cross-section at the wedge, so
  layer-to-layer separation and differing bore sizes are visible even
  where two adjacent layers share the same turn count), and finally a
  translucent (`FILL_ALPHA=0.25`) solid fill added back underneath the
  points/wireframe for a "filled" look without reintroducing the earlier
  opacity artifacts. Coil-2 gap is exaggerated for visibility
  (`GAP_VISUAL_STRETCH_MM`, display-only, applied on top of the correct
  mirrored base position). Color scale uses `PowerNorm(gamma=0.5)` pinned
  to a true 0T floor (not the data's sampled minimum) so low-field
  structure doesn't collapse into uniform-looking purple.
- **`optimization_summary_poster.png`** (`visualization/
  plot_optimization_poster.py`) — tape length vs. uniformity for the 7
  T-A-validated candidates above, PLUS a faint backdrop of all ~86,000
  feasible coarse-screen evaluations from the cumulative master log
  (clearly labeled as NOT independently verified) — added after initial
  feedback that showing only 7 points made the search look far smaller
  than it was.
- **`constraint_failures_poster.png`** (`visualization/
  plot_constraint_failures.py`) — every evaluation in the master log
  (~101,657, tape length in meters, linear axis capped at 1000m),
  colored by which constraint it failed first (B_target, hoop, or the
  coarse uniformity proxy, priority-ordered). Notable implementation
  point: infeasible-geometry evaluations (which never got a FEM solve)
  are INCLUDED, not excluded — `tape_length_m` (`params.py`) turns out to
  be a pure closed-form function of `(a, b, n_turns, t)` with no FEM
  dependency for ANY row, feasible or not, so the exact same formula
  already used for feasible rows' `tape_km` was applied directly to
  infeasible rows' stored `a_mm`/`b_mm`/`n_turns` fields — not an
  estimate, identical precision to every other point in the figure.

**Not resolved / left as a known issue:** `visualization/
plot_convergence_poster.py` (the "best tape found so far" staircase)
was found to have a much bigger problem than initially scoped when asked
to refresh it with today's data — the cumulative master log mixes many
now-obsolete constraint eras (pre-double-pancake, pre-7.5mm-bend-radius,
multiple now-superseded uniformity-proxy definitions), so naively
replaying the whole log's `all_constraints_ok` flag surfaces old,
invalid "improvements" as if they were real (a 0.146km pre-double-pancake
design, since fully invalidated, showed up as the "current best"). Only
today's 7 known-bad `day_search.py` run_tags were ever excluded before
the session moved on to other work — the deeper problem (every other
"champion" in this project's history except the current one turned out
to be a proxy artifact once checked) is unaddressed. Do not trust this
figure's output until it's rebuilt to only count genuinely T-A-validated
points, or is clearly re-scoped to "coarse-screen progress only."

---

## 2026-07-30: champion perturbation study + `optimize/` reorganization

### The question

Every champion in this project's history turned out to be a proxy
artifact once checked (see the four-investigation chain of 2026-07-24).
The current 6-layer champion survived that, and the `a`-isolation sweep
suggested it sat near a uniformity sweet spot — but it was found by a
CMA-ES search whose fitness function carries **no uniformity signal at
all**, so "did it just get lucky?" was still open.

### `optimize/studies/perturbation_study.py`

23 candidates: the champion, plus small **buildable** perturbations along
each axis independently and along all axes at once. Each got a full
per-layer T-A box-uniformity solve via `ta_validate.py` in a fresh
subprocess, 2 independent-mesh repeats (~65 s each, 3 concurrent).
Per candidate the coarse screen runs first (sequentially — it mutates
the shared `params` module and the default mesh path) to get that
geometry's own quench-limited `I_op`, `tape_km`, `B_target_T`, `hoop_MPa`;
only its `uniformity_pct` is ignored.

**The champion sits ON three constraint floors simultaneously** — straight
length `b−a` = 5.041 mm (floor 5.0), face gap = 3.001 mm (floor 3.0),
bend radius `a − max(n)·t/2` = 8.015 mm (floor 7.5, only 0.5 mm margin).
So `b` and `gap` can only be perturbed upward and `a` only 0.5 mm down.
Perturbations were chosen accordingly; all 23 verified buildable first.

### Results (`optimize/runs/perturbation/perturbation_results.csv`,
`visualization/perturbation_study.png`)

**Not a fluke — the measurement and the landscape are both solid.** All
23 converged; independent-mesh repeat spread ≤ 0.003 pp on 22 of 23. The
champion reproduced at 0.828 %, matching the 2026-07-27 re-check exactly
(0.8281/0.8283). Every axis varies smoothly and monotonically — none of
the knife-edge mesh fragility that killed the n_layers=4 design.

**But it is NOT a converged local optimum.** `[295,295,369,369,2,2]`
(10 turns per pancake moved from the inner pair to the outer pair)
beats it on **all four** metrics at once:

| design | tape | B_target | hoop | box p2p |
|---|---|---|---|---|
| champion `[285,285,379,379,2,2]` | 0.2259 km | 10.005 T | 114 MPa | 0.828 % |
| **`[295,295,369,369,2,2]`** | **0.2235 km** | **10.215 T** | **111 MPa** | **0.687 %** |

8 of 22 perturbations had better uniformity than the champion. Expected:
with no uniformity signal, CMA-ES stopped where tape/field cornered it.

**Bowl position per axis** (champion = 0.828 %):
- `a` (rigid radial translation, `b−a`/gap/turns fixed): champion is on
  the **inner wall**. −0.50/−0.25/0/+0.25/+0.50/+1.00/+2.00 mm →
  1.140 / 0.981 / 0.828 / 0.832 / 0.675 / **0.487** / 0.705 %. Minimum
  ≈ +1 mm outward. The earlier coarse ±3 mm sweep concluded the minimum
  sat *on* the champion — at fine spacing that is wrong.
- `b` alone: 0/+0.5/+1.5/+3.0 mm → 0.828 / 0.742 / **0.591** / 0.946 %.
  Also an interior minimum, ≈ +1.5 mm.
- `coil_half_gap` alone: **the one axis the champion gets right.**
  0/+0.5/+1.0/+2.0/+4.0 mm → 0.828 / 1.191 / 1.624 / 2.612 / 5.129 %.
  Steepest axis (≈ 0.7 pp/mm), monotone worsening upward, and the 3 mm
  face-gap floor pins it exactly at its optimum.
- turns (pairing preserved): shift_out `[275,275,389,389,2,2]` 1.982 %
  (much worse); shift_in 0.687 %; grow_pair3 `[…,10,10]` 0.897 % (so the
  2-turn vestigial pair is genuinely near-optimal, not an artifact);
  scale−5 % 0.563 % but B = 9.52 T (fails); scale+4 % 0.774 %.

**Tolerance sensitivity — the practical concern.** All four all-axes
jitter samples (≲0.3 mm plus a few turns) **failed** the 1 % target:
1.085 / 1.230 / 1.247 / 1.930 %. jitter4 was also the only candidate in
the study with meaningful mesh spread (0.086 pp vs ≤0.003 elsewhere).
**State the bias honestly whenever quoting this:** the face-gap floor
means jitter could only *increase* gap, the single steepest axis, so the
group is pessimistic by construction and is not a symmetric tolerance
estimate. It does show assembly gap tolerance is tight in the direction
one can actually err.

**Untested obvious follow-up:** combine `turns_shift_in`'s field
headroom (10.215 T) with a slight turn scale-down to land exactly on
10 T — likely ~0.219 km at better uniformity than the champion.

`params.py` and `opt_config.py`'s `CMAES_X0` were **left on the
champion** — the dominating neighbour has not yet been re-validated as a
finalist in its own right (it was one perturbation, not a search).

### Local polish (`optimize/studies/local_polish.py`) — new champion, and
### the tape/uniformity tradeoff quantified

Two phases, one script. **Phase 1:** 1000-eval CMA-ES warm-started from
the dominating neighbour `[295,295,369,369,2,2]` with PROPORTIONAL step
sizes (5% of `a`/`b` = 1.11/1.36 mm; 15 turns — set via the overrides,
and *verified in `bounds_and_stds()`'s output*, which is how the bug
below was caught). **Phase 2:** the six best distinct candidates by tape,
plus the old champion and the neighbour as references, each given a full
T-A box-uniformity solve with 2 independent-mesh repeats. The winner is
the lowest-tape design that actually PASSES — never Phase 1's output,
since the fitness has no uniformity signal.

| design | tape | B | hoop | T-A box p2p | verdict |
|---|---|---|---|---|---|
| **`[295,295,369,369,2,2]`** | **0.2235 km** | **10.215 T** | **111 MPa** | **0.688 %** | **PASS — NEW CHAMPION** |
| `[285,285,379,379,2,2]` (old) | 0.2259 km | 10.005 T | 114 MPa | 0.828 % | PASS |
| polish1 `[291,291,291,291,1,1]` | 0.1863 km | 10.001 T | 97 MPa | 3.660 % | FAIL |
| polish6 | 0.1905 km | 10.026 T | 97 MPa | 3.594 % | FAIL |
| polish4 | 0.1888 km | 10.042 T | 97 MPa | 3.902 % | FAIL |
| polish2 | 0.1871 km | 10.019 T | 98 MPa | 4.487 % | FAIL |
| polish5 | 0.1897 km | 10.048 T | 97 MPa | 4.772 % | FAIL |
| polish3 | 0.1880 km | 10.014 T | 97 MPa | 8.587 % | FAIL |

**The key physical finding: tape length and box uniformity are in direct
conflict along the turn-taper axis.** The tape-only search reliably
flattens the profile toward equal pairs — a genuine **17 % tape saving**
at 10 T with LOWER hoop stress — and every such design fails uniformity
by 3.6–8.6×. The champion's steep taper (295 → 369 → 2) is doing
essential uniformity work, not wasting tape. **Do not "optimize away"
the taper.** Both references reproduced their known values exactly
(0.828 %, 0.688 %), so the failures are physics, not pipeline error.

Promoted: `params.py` (`n_turns`, `I_design` = 223.88086308072167) and
`opt_config.py`'s `CMAES_X0`. `a`/`b`/`coil_half_gap` are UNCHANGED —
only the turn split moved. All figures regenerated; `cmaes_*.png` were
rebuilt from the champion's original search
(`regenerate_champion_plots.py`), not the polish run, since the polish
run's own best is a rejected design.

**Also fixed a long-standing figure failure:** `visualization/plot_3d.py`
had been dying in `plot_quench_2d()` with `IndexError: index 6 is out of
bounds` — it reads `sweep/quench_results.csv`, which was stale from a
≥7-layer geometry, and indexed the now-length-6 `params.a_center_list`.
Regenerating the CSV (`sweep/quench_sweep.py`) fixes it; `plot_3d.py`
now completes all four figures. Note `quench_sweep.py` re-solves
`racetrack_fields.npz` at near-quench current as its last act, so re-run
`solve/solve.py` afterward before regenerating any field figure.

## 2026-07-31: mesh stabilization + physical Jc(B) extrapolation

### Mesh convergence of the champion's box uniformity — CONSTRAINT MET

`optimize/studies/mesh_convergence_champion.py` (+ `_followup.py`) varied
mesh resolution on the FIXED champion geometry, 15 configs. `ta_validate.py`
gained an optional `"mesh"` key that sets any `params` attribute before
`recompute_derived()` (so `ta_n_picard` goes through it too), and now also
reports `n_coil_cells`/`n_dofs`.

**All 15 configs are below the 1% target, spanning 0.547-0.996%.**

- **In-plane: converged.** 0.686 -> 0.661 -> 0.642% over a 12x refinement
  (56k cells / 268k dofs), steps -0.025 then -0.019pp.
- **z (tape width): NON-MONOTONIC.** nz2 0.549, nz3 0.547, graded5 0.686,
  graded7 **0.986** (3 reps, +-0.008pp), graded9 **0.613** (2 reps,
  +-0.001pp). **Do not read a 3-point trend here as convergence** — an
  earlier reading of this data as "monotonically increasing and
  unconverged" was WRONG and was refuted by the 4th point. graded7 is a
  reproducible mesh-ALIGNMENT outlier, not a trend: the box metric is a
  near-cancelling dipole sum, sensitive to where cell boundaries land
  relative to the ~1mm penetration front. That also explains why
  refining in-plane "fixed" graded7 (0.996 -> 0.736) — it just perturbed
  the alignment.
- **Solver is not a factor.** Raising `ta_n_picard` 150->400 changed
  nothing (0.686->0.687, 0.986->0.961); solves converge at 73-98 iters.
  The concern that the `ta_scif_stall_mT` criterion (which watches the
  ON-AXIS SCIF) might stop before the BOX metric settles is cleared.
- **Best estimate ~0.62-0.69%**, with ~±0.2pp of irreducible
  alignment-driven scatter that does NOT shrink with refinement. Quote a
  band, and remember the tolerance jitter result: ≲0.3mm of build error
  still pushes this over 1%.
- `optimize/studies/mesh_convergence_balanced.py` (fine-in-plane z-ladder)
  is written but was NOT run — it would confirm, not change, the verdict.

### Physical Jc(B) extrapolation above 8 T — the champion FALLS SHORT of 10 T

`optimize/studies/ic_scaling_law_test.py` fits the pinning-force scaling
law `Jc = C·B^(p-1)·(1−B/Bc2)^q` per angle over B∈[1,8]T,
continuity-matched at 8T, and uses it ONLY above 8T (below, the measured
interpolation is unchanged — the law diverges as B->0 for p<1). Fits are
excellent, p ≈ 0.61-0.66.

**Bc2 MUST be fixed, never free-fit.** Over a 1-8T window with Bc2 >> 8T
the `(1−B/Bc2)^q` factor only varies 0.93-1.0, so q and Bc2 are strongly
degenerate. A free fit at angle 88° chose Bc2 = 10.23T, q = 0.18 — 0.51%
RMS *inside* the window, physical nonsense outside it (above 10.23T the
factor clips and Ic collapses to the floor). Because 88° is near the
ab-plane Ic peak, that one bad angle dominated the quench bisection and
produced the tell-tale absurdity of the OPTIMISTIC variant giving a LOWER
operating current than the pessimistic one. Now Bc2 is fixed at
25/45/100T and only C,p,q are fitted; the spread between them IS the
uncertainty band, and it is tight (~0.4T in B_target).

**Champion, B_target at the target box:**

| model | 55% Ic | 60% Ic | 65% Ic |
|---|---|---|---|
| flat clamp (current default, OPTIMISTIC) | 10.21 T | 11.32 T | 12.42 T |
| scaling law, Bc2=25T | 6.95 | 7.81 | 8.67 |
| scaling law, Bc2=45T | 7.18 | 8.06 | 8.94 |
| scaling law, Bc2=100T | 7.28 | 8.17 | 9.06 |

**Under a physical extrapolation the champion reaches only ~8.7-9.1 T
even at 65% of Ic — it does NOT meet the 10 T constraint.** The flat
clamp was carrying ~1.2-3.3 T of unearned field. (The older
`ConservativeIcModel` linear continuation gave 6.51T at 55%; the scaling
law's 6.95-7.28T is slightly better and better-motivated.)

**Naive turn-scaling is NOT an efficient fix** (quick scan, Bc2=45T,
65% Ic, taper shape / `b−a` / gap held fixed, `a` raised only as needed
to hold the 7.5mm bend radius):

| turns x | tape_km | B_target | I_op |
|---|---|---|---|
| 1.00 | 0.2235 | 8.94 T | 200 A |
| 1.15 | 0.2746 | 9.39 | 191 |
| 1.30 | 0.3363 | 9.45 | 179 |
| 1.45 | 0.4019 | 9.64 | 171 |
| 1.60 | 0.4732 | 9.91 | 166 |

Doubling the tape buys only +1 T and still misses 10 T: more turns force
`a` outward (bend radius), which costs field efficiency, AND raise the
peak field, which lowers Ic and hence I_op (200 -> 166 A). Strong
negative feedback. **This is a crude one-parameter family** (fixed
taper/`b−a`/gap/layer count) so it does NOT prove 10T is unreachable —
it proves the design needs genuine RE-OPTIMIZATION under the scaling-law
Ic model, not just more tape. Layer count, taper shape, and gap are all
still free.

## 2026-07-31 (part 2): redesign under a validated Ic model — NEW DESIGN

### Is the extrapolation realistic? Tested, not argued

`optimize/studies/ic_extrapolation_validation.py` does hold-out validation
on the measured data: fit each candidate Jc(B) form on a LOW-field subset,
score it on measured high-field points it never saw. At the split matching
real use (fit <=5T, predict 7-8T -- a 1.6x extrapolation, same factor as
8T -> 13T), over all 43 angles:

| form | MAPE | bias | reading |
|---|---|---|---|
| **flat clamp** (project default) | 26.7% | **+26.7%** | badly OPTIMISTIC |
| power law | 6.9% | +6.8% | slightly optimistic |
| **kim** `Jc0/(1+B/B0)` | **4.1%** | −3.3% | BEST, mildly conservative |
| scaling law (Bc2=45T) | 6.1% | −5.8% | good, more conservative |

At a 2.7x split the flat clamp over-predicts by **+54%** (up to +88% on
individual angles). **So: the extrapolation is realistic, the flat clamp is
decisively unsafe, and the 10T shortfall is real — the scaling law errs
CONSERVATIVE, so its numbers are under-estimates, not over-estimates.**

Both wrappers live in `optimize/ic_extrapolation.py`
(`KimIcModel`, `ScalingLawIcModel`, `make_ic_model`), selectable in
`cmaes_search.py` via `CMAES_IC_EXTRAP` (default `flat` = historical
behavior, unchanged). **Bc2 must be FIXED, never free-fit** — over a 1-8T
window q and Bc2 are degenerate and an unconstrained fit picked
Bc2=10.23T at angle 88°, collapsing Ic just above the data and corrupting
the quench solve (it made the OPTIMISTIC variant give a LOWER operating
current than the pessimistic one).

### THE NEW DESIGN (`optimize/studies/ta_in_loop_search.py`)

**n_layers=6, a=23.227mm, b=28.268mm, gap=13.500mm,
n_turns=[329,329,411,411,2,2], I_op=204.57A (65% of local Ic),
tape=0.2596km.**

| metric | value | limit | |
|---|---|---|---|
| B_target (kim Ic) | 10.03 T | >= 10 T | PASS |
| box p2p uniformity (T-A) | **0.442%** | <= 1% | PASS (2 meshes: .442/.442) |
| hoop | 102 MPa | <= 400 | PASS |
| bend radius | 7.82 mm | >= 7.5 | PASS |

**CAVEAT:** under the more conservative `scaling:45` model it gives 9.34T.
Kim is the measurably better model AND itself slightly conservative, so
10.03T is the better estimate; the honest statement is **~10T with ±0.5T
of MODEL uncertainty**, closable only by measured Ic data above 8T. NO
candidate reached 10T under `scaling:45` — even +55% tape only got to
9.57T.

**Why growing `a` is the winning direction:** it improves box uniformity
AND raises the bend-radius ceiling (`max_pair = 2(a−7.5mm)/t`), so more
turns fit — buying field and uniformity together. Adding turns at fixed
`a` is far weaker (0.2330km reached only 9.77T).

Also found: **uniformity depends on the OPERATING CURRENT.** The old
champion measures 0.688% at I=223.9A but **0.268%** at I=208.7A (the
conservative model's lower I_op) — less drive, weaker screening. Never
compare uniformity numbers taken at different currents.

### Build-tolerance (jitter) study — the design FAILS on FIELD, not uniformity

`optimize/studies/jitter_new_design.py`. Two fixes over the old champion's
jitter test: (a) it jittered TURN COUNTS ±2%, which is not a real build
error (turns are wound and counted, so the number is exact) — dropped;
(b) it omitted TAPE THICKNESS, a real ±2% lot variation that scales the
whole pack (`pack = max(n_i)·t`) and moves every layer at once — added.
Each jittered design also gets its OWN quench-limited I_op recomputed,
since box uniformity depends on operating current.

Tolerances: a, b, gap ±0.2mm; tape thickness ±2%; turns exact.

**Result: 0 of 14 perturbed designs reach 10 T** (range 9.82-10.02 T),
while **13 of 14 pass uniformity comfortably** (0.33-0.53%).

**Root cause: the search minimized tape subject to B >= 10 T, so it
converged EXACTLY ONTO that constraint** (nominal 10.03 T, 0.3% margin).
Nothing in the objective asked for margin, so none was bought, and every
build error costs more than 0.3%. **Any future search must target a
nominal ABOVE the floor** (~10.3 T) rather than on it.

**Dominant single-axis error: TAPE THICKNESS, asymmetrically.** `t−2%` is
benign (0.489%); `t+2%` is the only one-at-a-time perturbation that fails
uniformity (1.33%) AND consumes the entire bend-radius margin (7.506mm vs
the 7.5 floor) AND costs 0.15 T. It is also the error least under a
builder's control — set by the tape lot, not by machining.

**Manufacturability is a separate failure:** the nominal sits ON three
floors (face gap margin 0.001mm, straight length 0.041mm, bend radius
0.315mm), so ~half of any symmetric tolerance distribution is out of spec
before physics enters. 5 of 6 random samples landed out of spec; 2
violated the bend radius (a tape-cracking risk, not just paperwork).

### Mesh convergence of the new design — 0.39-0.44%, and a false alarm

`optimize/studies/mesh_convergence_new_design.py`. The jitter study's
nominal read 0.938/0.942% where `ta_in_loop_search.py` had measured
0.442% for the same geometry, and two further processes agreed at ~0.94%.
**I wrongly concluded the 0.442% was irreproducible and that a 0.29 µm
gap difference caused it.** Both claims were wrong:

- The controlled test (`gapA_prod` vs `gapB_prod`, same run, only the
  0.29 µm gap differing) gives **0.416% for BOTH**. The gap was a red
  herring; it merely correlated with which mesh those runs happened to
  draw.
- The real correlation is with mesh CELL COUNT: 4375 -> 0.416%,
  4390 -> 0.442%, **4400 -> 0.94%**. gmsh is not deterministic across
  processes, and this design sits near a meshing tipping point where one
  of the two topologies is anomalous.
- The refinement ladder settles it: **7 converged configs span
  0.360-0.438%** (2.7k to 21k cells, 8x range), with the FINEST
  (`both_fine`, 21003 cells) at **0.386%**. So ~0.39-0.44% is the
  converged answer and 0.94% was a bad mesh draw.

**Caveat on the summary line:** the 8th config (`inplane_finer`) reported
3.460% with `conv=False` — it hit the 150-iteration Picard cap because
that config did not set `ta_n_picard=400`. It is a FAILED SOLVE, not a
data point, and it makes the script's "all configs below 1.0%: False"
line misleading. Re-run it with a raised cap before quoting any range
from that CSV.

**Lesson: a non-converged solve must be excluded, not averaged in** — and
when several processes agree on an anomalous value, that can still be a
shared bad mesh draw rather than the truth. Only refinement settles it.

## 2026-08-03: Long (2013) max-entropy Beta model tested — Kim still wins

Paper: **N. J. Long, "Maximum Entropy Distributions Describing Critical
Currents in Superconductors", Entropy 2013, 15(7), 2585-2605,
doi:10.3390/e15072585.** (MDPI blocks scraping; the PDF is fetchable from
`https://mdpi-res.com/entropy/entropy-15-02585/article_deploy/entropy-15-02585.pdf`
and this repo has no pdftotext/pypdf — a small zlib+regex extractor in
scratch worked.)

**Its Eq. 2:** maximum-entropy inference with logarithmic constraints
⟨ln b⟩ and ⟨ln(1−b)⟩ gives the field dependence as a BETA distribution

    Jc(b) ~ b^(α−1) (1−b)^(β−1),     b = B / B_irr

with ⟨ln b⟩ = ψ(α) − ψ(α+β), ⟨ln(1−b)⟩ = ψ(β) − ψ(α+β) (ψ = digamma).
Multiplying by B recovers the 40-year-old pinning-force form
F_p ~ b^m (1−b)^n. Two things distinguish it from our `scaling:45`:
the normalizing field is the IRREVERSIBILITY field B_irr (where Jc → 0),
not Bc2; and the paper shows angular data collapses to a COMMON curve —
one (α,β) with angle-dependent B_irr (their Bi-2223: α=1.8, β=9.6,
B_irr 1.73 T at 0° to 11.1 T at 90°).

Implemented as `BetaIcModel` (per-angle) and `BetaSharedIcModel` (shared
α,β — the paper's angular-scaling claim) in `optimize/ic_extrapolation.py`.

**Hold-out result: it does NOT beat Kim for this tape.**

| form | MAPE @1.6x | MAPE @2.7x |
|---|---|---|
| **kim** | **4.14 %** | **6.98 %** |
| beta (per-angle, free B_irr) | 5.46 % | 22.72 % |
| beta_shared (α,β shared) | 14.26 % | 28.66 % |

**Why.** The pinning force F_p = Ic·B is still RISING at 8 T at every
angle, so its peak (at b = α/(α+β−1)) lies above the measured data and
B_irr is never pinned by a visible maximum — the free per-angle fit rails
B_irr at its upper bound for most angles and collapses onto a power law
(its 5.46 % is indistinguishable from scaling100's 5.47 %). Sharing (α,β)
is better conditioned (B_irr 36-38 T, physically sensible for REBCO at
20 K) but forces one curve shape onto a genuinely anisotropic tape, which
costs more than it gains. **The paper's method is sound; our data simply
does not reach the field range that constrains it.**

**BUG RE-INTRODUCED AND CAUGHT — the same one, at the same angle.** With
`BIRR_BOUNDS` lower bound at 9 T, the free per-angle fit chose
B_irr = 10.23 T at angle 88°, collapsing Ic from 894 A at 10 T to 39.6 A
at 12 T — at the angle where Ic is HIGHEST — which dominated the quench
bisection and gave the champion a spurious B_target of 5.59 T. **This is
the identical angle and the identical 10.23 T value the free-Bc2 scaling
fit picked earlier** (see ic_scaling_law_test.py's docstring). Angle 88°
in this dataset reliably induces it. Fixed by raising the floor to 20 T
(REBCO at 20 K has B_irr ≈ 30-45 T). Lesson: **any (1−B/B_scale) model
must have its cutoff field bounded well above the data ceiling**, or one
bad angle silently wrecks the whole quench calculation.

### Bias-corrected synthesis — the most defensible B_target estimate yet

Because the hold-out framework MEASURES each model's bias, each model's
B_target can be corrected by its own bias. On the current champion:

| Ic model | MAPE | bias | B_raw | B_bias-corrected |
|---|---|---|---|---|
| flat clamp | 26.7 % | +26.7 % | 14.69 T | 10.77 T |
| kim | 4.1 % | −3.3 % | 10.48 T | 10.82 T |
| scaling:45 | 6.1 % | −5.8 % | 9.43 T | 9.97 T |
| beta (Long) | 5.5 % | −5.1 % | 9.75 T | 10.25 T |
| beta_shared (Long) | 14.3 % | −13.0 % | 8.76 T | 9.89 T |

**Five models spanning a 6 T raw range collapse to 9.89-10.82 T once
bias-corrected**, including the flat clamp approaching from the opposite
side. That is strong evidence the champion genuinely sits at
**B_target ≈ 10.3 ± 0.5 T** and clears the 10 T floor.

Caveat on the correction: the biases were measured at a 1.6x
extrapolation (≤5 T → 7-8 T), while the real use is 8 T → ~10.7 T peak
field (1.34x, gentler), so the true biases are somewhat smaller and the
correction is slightly over-generous. Treat it as a consistency check,
not a replacement for measured Ic data above 8 T.

Note the conservative models give a LOWER I_op (170-185 A vs kim's 196 A),
and box uniformity IMPROVES at lower current (measured earlier: 0.688 % at
224 A vs 0.268 % at 209 A on the old champion), so the uniformity
validation is unaffected — no re-run needed.

### FINAL DESIGN: margin-aware, and jitter-VALIDATED

`optimize/studies/margin_design_search.py` +
`optimize/studies/jitter_margin_design.py`.

**a=26.0mm, b=31.4mm, gap=13.7mm, n_turns=[382,382,478,478,3,3],
I_op=196.0A (65% Ic), tape=0.3372km.**

| metric | nominal | limit | across 15 jitter samples |
|---|---|---|---|
| B_target (kim) | 10.49 T | >= 10 T | 10.10-10.49 — **15/15 PASS** |
| box p2p (T-A) | 0.495% | <= 1% | 0.338-0.517 — **15/15 PASS** |
| hoop | 113 MPa | <= 400 | 102-113 |
| bend radius | 8.075 mm | >= 7.5 | 7.545-8.434 |
| face gap | 3.40 mm | >= 3.0 | 3.00-3.84 |

Contrast with the predecessor: **0/14** builds reached 10 T, 5/6 out of
spec, 2 bend-radius violations. Here: **15/15 on both constraints, 0/6
out of spec.** The margin arithmetic was tested, not assumed.

**How it was built:** margin-aware constraints derived from the measured
jitter response, not the nominal limits —
  - B >= 10.3 T nominal (worst measured jitter cost was -0.21 T)
  - bend >= 7.5mm at a-0.2mm AND t+2% SIMULTANEOUSLY, i.e.
    `a >= 7.7mm + N*3.825e-5` — this is what forces the larger radius
  - face gap 3.4mm nominal, straight 5.4mm nominal
Turn ratio held at the champion's 295:369:2 (the only T-A-validated
profile family). Stage 1 scanned (a, turn-fill); only 3 of 24 cleared
10.3 T, all needing turns at the margin-aware ceiling — **field requires
turns at the ceiling, so margin must come from RADIUS, not from backing
turns off.**

`t+2%` was the predecessor's worst failure (1.33% uniformity, ate the
whole bend margin, -0.15 T). Here it is benign: 0.356%, 10.15 T.

**Cost: 0.3372km vs 0.2596km (+30%).** Roughly half the total increase
over the original 0.2235km design is the realistic Ic model, half is
build tolerance never previously budgeted. The tolerance half scales
directly with the assumed ±0.2mm / ±2% numbers — **those are MY
assumptions, not measured shop capability**; tighter machining recovers
much of it. Worth revisiting with real tolerances before treating 0.3372
as the true cost.

`scaling:45` cross-check still gives 9.44 T — unchanged in character;
~10.5 T ± 0.5 T of MODEL uncertainty, closable only by Ic data above 8 T.

### FOUR failed approaches, and the lesson: NO fast uniformity proxy exists

| attempt | approach | outcome |
|---|---|---|
| 1 | free CMA-ES search | wandered off; turn step fell back to the ~210 bound-range default |
| 2 | champion seed + local 25-turn step | geometry stayed local but the taper FLATTENED anyway (1.25→1.00 over 280 evals) |
| 3 | taper shape FIXED, only scale searched | reached 10T at 0.289km but **1.66-2.12% uniformity, all FAIL** |
| 4 | uniform-J box field as a cheap filter | 5 finalists at uniform-J 1.41-1.45% scored **1.65-2.12% T-A, all FAIL** |

Attempt 3 disproved that the champion's taper is a transferable shape:
proportionally identical profiles give 1.66-1.71% at 8/10 layers vs 0.69%
at 6. Attempt 4 killed the last plausible cheap proxy.

**Proxy graveyard (four now):** on-axis SCIF (anti-correlated — the best
on-axis design had the WORST box uniformity), peak-turns-per-pair (built
from that same bad data), the Bean-state correction (~10x error for
compact coils), and the uniform-J box field (screening spans −1.50 to
+0.57 pp). **T-A is the only arbiter. Put it in the loop; do not filter
ahead of it.** A T-A solve is only ~3-5 min, so tens of physically-chosen
candidates is the right shape for a search — not thousands of blind ones.

Also RETRACTED here: "box uniformity tracks coil radius `a` cleanly
(bigger = better)". Isolating `a` gives a V-shaped bowl with an interior
minimum. That claim has been removed from `params.py` and
`opt_config.py`.

### A cost mis-estimate worth not repeating

Attempt 4's stage-1 filter was written on my claim that it cost
"milliseconds". Measured: **519 ms** — the multi-filament Biot-Savart
costs ~2.4 ms per evaluation POINT, and 80,000 samples projected to
**11.5 hours**. (FILAMENT_TURNS_PER_GROUP barely matters: 450→415 ms from
100→500, so the cost is per-point, not per-filament.) Fixed by using
CMA-ES (~2500 evals) instead of brute force plus a coarser 11x5 screening
grid. **Measure the per-evaluation cost before sizing any sweep.**

### BUG: `CMAES_N_STD0_OVERRIDE` was inert since the day it was added

Found while setting up the local polish run below, by asserting on
`bounds_and_stds()`'s returned step sizes instead of trusting the env var
to have worked. `opt_config.py` parses `CMAES_N_STD0_OVERRIDE` into
`cfg.CMAES_N_STD0` (no `_OVERRIDE` suffix), but
`cmaes_search.bounds_and_stds()` read `getattr(cfg,
"CMAES_N_STD0_OVERRIDE", None)` — an attribute that never exists — so
`getattr` silently returned `None` and the oversized bound-range default
(`CMAES_SIGMA0_FRAC × (n_hi − n_lo)` ≈ 150 turns) was used **regardless
of what any caller set**.

The `a`/`b` overrides were never affected (they overwrite
`CMAES_A_STD0`/`CMAES_B_STD0` directly, which is what
`bounds_and_stds()` actually reads) — only the turn-count one.

Consequence: the 2026-07-23 fix intended to prevent exactly the
`focused_refinement_6_9.py` n_layers=9 regression (an intended local
polish that sampled turns nearly uniformly across the whole bound range
from eval 1, drifted to 0.171 km — worse than the 0.152 km it started
from — and never recovered) **was itself inert**. Every warm-started
"polish" run since then silently got the cold-start turn step. Fixed by
reading `cfg.CMAES_N_STD0`; verified both directions (override → 15.0,
absent → 149.7 unchanged, so cold starts behave exactly as before).

**Lesson worth keeping: assert on the value the solver will actually
use, not on the env var being set.** Two of this project's worst
regressions trace to step sizes that were believed to be small and
weren't.

### `optimize/` reorganized

Top level now holds only the 5 reusable tools (`opt_config.py`,
`optimize_geometry.py`, `cmaes_search.py`, `evaluate.py`,
`ta_validate.py`). One-off orchestrators → `optimize/studies/`; every
log/CSV → `optimize/runs/<study>/`. Notes for future moves:
- Study scripts' `_ROOT` is now `dirname` **×3** (studies/ → optimize/ →
  repo root); their `sys.path` entries and their
  `subprocess` launches of `optimize/cmaes_search.py` (relative, with
  `cwd=_ROOT`) were already correct and unchanged.
- Every artifact path flows through `opt_config.py` constants
  (`CMAES_MASTER_LOG`, `CMAES_OUT_CSV`, `CMAES_OUT_LOG`, `OUT_CSV`), so
  **no visualization script needed editing** — keep it that way.
- Verified: all scripts compile, all four config paths resolve to
  existing files, `_ROOT` correct in every moved script, and a bounded
  end-to-end `cmaes_search.py` run wrote to the new paths correctly.

**Two gotchas hit while verifying, worth not repeating:**
1. **There is no `CMAES_MAX_EVALS_OVERRIDE` env var.** Guessing one
   silently launches a full 2500-eval search. The supported way to bound
   a smoke test is `CMAES_SWEEP_OVERRIDE_JSON='{"max_evals": 6}'`.
2. **Any `cmaes_search.py` run overwrites `visualization/cmaes_*.png`
   with its own history and appends to the cumulative master log.** After
   a smoke test, restore the master log from a backup and re-run
   `optimize/studies/regenerate_champion_plots.py` to rebuild the
   champion's figures from `run_20260723_124414`.

---

## 2026-08-03 (part 2): NO-INSULATION transient — Phase A, the DCN circuit model

New top-level package `circuit/`. **Nothing outside it was modified** — the
existing model is untouched and runs exactly as before (verified: `git status`
shows only `circuit/` added). Pure numpy/scipy, no dolfinx: `physics/
coil2_field.py`, `physics/ic_model.py` and `optimize/ic_extrapolation.py` are
all import-clean, so the circuit model runs without the FEM stack.
`optimize/optimize_geometry.py` is NOT importable this way (it pulls in
`mpi4py`/`dolfinx.mesh` at module level), so `filament_stack()` was not
reused — `coil2_field.compute_both_coils_field_multilayer()` re-inlines the
identical grouping logic and IS clean.

**Project direction (2026-08-03): the coil is committed to no-insulation
(NI).** At DC steady state the radial current vanishes, so **every existing
design number survives unchanged** (10.49 T, 0.495% box uniformity, 113 MPa,
0.3372 km). This adds a transient constraint; it invalidates nothing.

### THE INDUCTANCE WAS WRONG BY 8x IN A FIRST ESTIMATE — the correct value

A quick flux-linkage estimate gave 54 mH / 1.04 kJ. **That was wrong twice
over**: a 26x26 integration grid over a region containing the winding's own
1/d filament singularities, and a missing factor 2 (coil 2's turns link flux
as well). The converged answer:

| quantity | value |
|---|---|
| self-inductance (both coils, series) | **419.7 mH** |
| stored energy at 196 A | **8.07 kJ** |

Converges cleanly in the turn grouping: 432.5 → 422.4 → 420.2 → 419.8 →
419.7 mH at 100/50/25/12/6 turns per group (last step 0.01%).

**Do not check an inductance with a coarse area integral of Bz.** It is not
even an independent check — by Stokes it is algebraically the same Neumann
double sum — and it is wildly sensitive to grid resolution near conductors.

### `circuit/` — what is in it

- `geometry.py` — `CoilGeometry` (decoupled from `params.py` global state, so
  the benchmark coil can be built without touching it), `TurnGroups`,
  `racetrack_loop()`. Reproduces `params.tape_length_m` to 4e-16.
- `inductance.py` — per-turn mutual inductance by the Neumann double integral
  with a geometric-mean-distance (GMD = 0.2235(t+w) = 0.911 mm) regularisation
  applied to EVERY pair. Not a fudge: at a 75 µm pitch with 4 mm tape the GMD
  is >10x the turn spacing, so adjacent turns correctly come out nearly as
  coupled as a turn is to itself. Cached to `runs/cache/`.
- `fieldmatrix.py` — per-unit-current field matrix, vectorised over field
  points (`coil2_field`'s helper loops one point at a time, which makes an
  N^2 build infeasible). **Same GMD regularisation is mandatory here**: sample
  points lie on their own group's centreline, and without it the local |B|
  came out at 24.9 T against a true peak near 10.7 T, poisoning every
  Ic(B,theta) lookup. With it: 12.3 T.
- `dcn.py` — the ladder. In an NI winding the contact resistance bridges the
  same two nodes as the turn it parallels, so KCL gives `i_k + j_k = I(t)`
  exactly at every rung — the He et al. paper's `I_z = I - I_r` is a derived
  consequence here, not an imposed closure. Eliminating `j` leaves
  `A di/dt = R_ct*(I - i) - V_sc(i)` with `A[k,j] = n_j M[k,j]`, solved with
  `solve_ivp(method="BDF")` (the repo's first use of `scipy.integrate`; the
  power law with n ~ 13-22 is stiff, RK45 will not do). ~0.3 ms per RHS
  evaluation, ~0.5 s per scenario.
  The power law is written as a VOLTAGE, not a resistance — algebraically
  identical for i != 0 but finite at i = 0, which every charge run starts at.

### Validation

- **A1 (`validation/lumped.py`)** — Neumann+GMD reproduces the textbook
  circular-loop self-inductance to 0.01-0.5%; the filament sum matches the
  repo's PRODUCTION Biot-Savart path (`compute_both_coils_field_multilayer`,
  which every design number goes through) to **0.18% median / 0.44% max** on
  the bore axis. That is the sharp check and it PASSES.
  Two checks that are NOT usable here, both worth not repeating:
  `integral(J.A)` from the FEM is **not gauge-invariant** for this A-form
  (`gauge_regularization = 1e-3` against `1/mu0 ~ 8e5`, and the eighth-domain
  current is not divergence-free because it crosses the symmetry cut faces) —
  it gave L off by 1e8. And the FEM energy `integral(|B|^2/2mu0)` is only a
  LOWER BOUND: the air box is just 1.27x the coil's outer extent with a PEC
  boundary, so flux is confined (it gives 343 mH vs 420, and the FEM bore
  field runs ~10% below the filament model for the same reason).
- **Energy balance on discharge** — the strongest self-check, since it
  exercises M, R and the integrator together: dissipated energy vs
  `1/2 L I^2` closes to **0.00-0.08%**. It initially read 9% high at the
  shortest tau purely from trapezoid error on a linearly-sampled exponential;
  both drivers now use LOG-spaced output grids (this also fixed `tau` fits
  returning `nan` for every fast case).

### A2 — the published benchmark, and an inconsistency IN THE PAPER

`validation/he2025_racetrack.py` against He et al. Table 2 (115 turns,
0.2x12 mm tape, 49/73 mm, 77 K, rho_c = 399 µΩ·cm², L = 3.05 mH, field
constant 0.81 mT/A, **tau = 13.53 s measured**). The straight-leg length is
not published, so it is the one free parameter — and the table is
over-determined, so BOTH fits are run and reported rather than the flattering
one:

| fit | field const | L_self | tau vs MEASURED |
|---|---|---|---|
| A: match field constant | 0.810 (exact) | 5.855 mH (+92%) | **12.96 s (−4.2%)** |
| B: match self-inductance | 0.996 (+23%) | 3.050 (exact) | 3.90 s (−71%) |

**No choice of the free parameter satisfies all three published numbers.**
Root cause: the table is internally inconsistent under the standard NI
relation `tau = L/sum(R_ct)` — their own L and rho_c give 3.81 s against a
measured 13.53 s, a factor of 3.5. Either their rho_c was back-fitted through
a different contact-area definition, or L and tau were measured under
conditions this lumped relation does not describe.

**What A2 validates:** the model structure and the tau prediction (4.2% on
the measured value). **What it does not:** an absolute rho_c → tau
calibration. **Carry a factor ~2-3 uncertainty on any tau quoted from an
ASSUMED rho_c.**

### Results for the champion

**tau = 1330 / rho_c[µΩ·cm²] seconds** — exact 1/rho_c scaling confirmed
(44.33x30 = 13.30x100 = 3.33x400 = 1330), the L/R signature.

| rho_c | tau | 600 s ramp: field deficit at ramp end | contact loss | mean power |
|---|---|---|---|---|
| 30 | 44.3 s | **5.41%** | 1045 J | 1.7 W |
| 100 | 13.3 s | 1.62% | 331 J | 0.55 W |
| 400 | 3.3 s | 0.41% | 84 J | 0.14 W |

Closed form, verified against the DCN to 7%: **`E_contact = 2 W_stored *
tau/t_ramp`**. One number, tau, sets both the field lag and the ramp heat
load.

**Sudden discharge: all 8.07 kJ lands in the winding** (an NI coil has no
external dump path — that is the entire point of the technique). Peak power
574 W (rho_c=30) to 7.65 kW (rho_c=400). This is the thermal design case;
the temperature rise is NOT modelled (isothermal, EM only).

### The 3-turn-pancake worry was WRONG — retracted

I flagged the champion's two 3-turn pancakes as an NI risk on the grounds
that 2 turn-to-turn interfaces means almost no bypass resistance, hence a
wildly different per-pancake time constant. **The model says no**, and the
reasoning was based on the wrong topology: a low-resistance bypass around a
*small* number of turns also bypasses a proportionally *small* EMF, so the
two scale together and there is no per-pancake time constant at all — the
pancakes are series-connected and strongly mutually coupled, so there is one
system tau. `visualization/circuit_turn_currents.png` shows all six layers
charging on top of each other. The radial profile does have real structure
(inner turns carry ~3% more than outer, because outer turns are longer and so
have LOWER contact resistance per turn), just not a pathology.

### AC loss — only HALF the answer, and it is the easy half

`E_sc` from the DCN is 0.003-0.06 J and is **not** the hysteretic loss. A DCN
carries one lumped current per turn, so it cannot represent current
distribution across the tape width — the magnetization loss that dominates
REBCO AC loss is invisible to it by construction. What is reported is the
contact loss (exact) plus the power-law flux-flow loss along the tape
(negligible at i/Ic ~ 0.5). **The hysteretic half needs Phase B (T-A).**
Stabilizer eddy/coupling loss is outside both models, so every loss number
here is a lower bound.

### Assumption ledger (NI model)

Uniform rho_c (the paper attributes its own pancake discrepancy to
non-uniformity); no pancake-to-pancake contact (double-pancake construction
insulates between pancakes — an assumption, not a derivation); no normal-metal
stabilizer conductivity; isothermal at 20 K; filament approximation for the
winding (inherited from the repo's own production field path, accuracy
0.18% against it).

### Open / next

Phase B (`transient/`, T-A with the paper's local circuit closure) is NOT
started. It is what supplies hysteretic loss, current distribution across the
tape width, and uniformity during the ramp, and it cross-checks tau
independently. The `solve/ta_solve.py` hooks it needs (a stored mutable
`dt_const`, a zeroed `A_prev`, and the history term `curl(A_h - A_prev)`)
have NOT been added yet.

Also worth noting: NI is self-protecting, so the current `SAFETY_FACTOR =
1.818` (55% of local Ic), chosen for an insulated coil with no protection
margin, is plausibly over-conservative. Relaxing it is the cheapest tape
saving available anywhere in this project. Not investigated.

Run:
```bash
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
$PY circuit/validation/lumped.py            # A1
$PY circuit/validation/he2025_racetrack.py  # A2
$PY circuit/run_charge.py ; $PY circuit/run_discharge.py ; $PY circuit/postprocess.py
```
Outputs: `circuit/runs/*.csv|log`, `visualization/circuit_{charge,discharge,turn_currents}.png`.

---

## 2026-08-03 (part 3): NI transient — Phase B, T-A with the circuit closure

New package `transient/`. `solve/ta_solve.py` gained THREE hooks, all inert
by default and **proved inert, not assumed**:
1. `ta["A_prev"]` — a zeroed N1curl Function, the previous time step's A.
2. The T right-hand side history term changed from `curl(A_h)` to
   `curl(A_h - A_prev)`. Identical while `A_prev` is zero, and exact (no
   interpolation error, unlike storing a DG0 `B_prev`).
3. `setup_ta_problem(..., per_turn_bc=True)` makes the tape-edge Dirichlet
   values Functions instead of Constants, so the imposed current can vary
   with radial position. `solve_ta_at_current()` now RAISES on such a setup
   rather than silently solving with stale Constants.

**Tier B0 (`transient/validation/insulated_limit.py`) PASSES**, and the way
it passes matters: comparing `box_ptp_pct` before/after would be a weak test,
because gmsh is not reproducible across processes and this design's own
scatter (0.34-0.52%) swamps any subtle bug. Instead it assembles the T RHS
from the new form and from a locally reconstructed old form on the same mesh
and state, and they are **BIT-IDENTICAL** (max difference exactly 0.0).
`box_ptp_pct` came out 0.517% vs the recorded 0.495% — reported, not
asserted.

### THE BLOCKING FINDING: `E_i = -dA_z/dt` CANNOT BE EVALUATED IN THIS REPO

He et al. Eq. (11) takes the induced field pointwise from A. **That is
invalid in this A-formulation.** `solve.py` solves

    (1/mu0)(curl A, curl v) + gauge_regularization*(A, v) = (J, v)

with `gauge_regularization = 1e-3` against `1/mu0 ~ 8e5`. The curl-curl
null space (gradients) is penalised only by that 1e-3 term, and the
eighth-symmetry domain's current is NOT divergence-free (it crosses the
symmetry cut faces), so a large gradient component is amplified into A.
Measured directly on the champion at 196 A:

| quantity | value |
|---|---|
| `A.t_hat` over the coil cells, mean | **-1.2e8** |
| physical `Phi_turn / l_turn` | **0.115 Wb/m** |
| ratio | **~1e10** |

**A is meaningful in this repo ONLY through `curl A`.** Every pre-existing
user takes its curl, so this never surfaced before. It is the same root cause
that made an `integral(J.A)` inductance check come out 1e8 too large
(see `circuit/validation/lumped.py`).

Using raw `A.t_hat` anyway gave radial currents of 39 A against a 65 A
transport current (the validated circuit model says ~2-3 A) with 38 of 48
bins clipped.

**Fix (`transient/induction.py`):** E_i is a purely inductive, geometric
quantity, so it is taken from a per-bin mutual-inductance matrix built with
the SAME Neumann+GMD machinery as Phase A. Independent confirmation: that
matrix gives **L_total = 423.27 mH against circuit/'s 419.7 mH (0.85%)** on a
completely different grouping. The T-A still does the induction correctly
INTERNALLY — its T RHS takes a curl and so is gauge-invariant; only the
extraction of a per-bin scalar E_i is replaced.
**Stated plainly: this makes the induced term common to both models, so
`dcn_crosscheck.py` no longer independently validates the induction. What it
isolates is the effect of E_p.**

### The E_p ambiguity — INITIAL "resolution" RETRACTED (see next section)

Eq. (12)'s `E_p = rho_sc*J*d/L` never says whether J is SC-layer or
engineering current density — a factor `Lambda/delta_SC = 75`. Both were run
on the champion (rho_c = 100, 600 s ramp) against the DCN's validated
I_r ~ 3.4 A, using the smoke test's TIGHT iteration caps (45/20, chosen to
test mechanics fast):

| E_p factor | I_r over the ramp | clipped bins | behaviour (at the smoke test's cap) |
|---|---|---|---|
| 75 ("derived") | 18-35 A (up to 37% of I) | 29-38 of 48 | looked unstable |
| 1 ("paper") | 3.9 / 2.1 / 2.1 A, 0.40 A on the flat top | 0 | **looked stable — WRONG, see below** |

**This conclusion was WRONG.** `tparams.EP_FACTOR_MODE` was set to `"paper"`
on the strength of it. See the next section — the "stable" result was an
artifact of stopping the loop before the instability had time to appear, not
evidence of a converged state. `EP_FACTOR_MODE` is left at `"paper"` in the
file (still the more literal reading, and still cheaper per-iteration since
it makes `E_p` smaller), but **it is NOT validated and must not be quoted as
resolved.**

**Why F=75 blows up, which is itself the interesting physics:** the screening
currents drive local |J| to Jc at the flux front, and with n ~ 13-22 the
power-law rho makes a bin average of rho*J dominated by those front cells
rather than by the transport current. The volume-weighted average (already a
fix over an arithmetic one, which diverged to NaN in 3 Picard iterations) does
not remove this. **Whether that average is the circuit-relevant turn voltage
at all is exactly the ambiguity the paper leaves open, and it was not settled
from the paper's text.**

### Stability machinery the closure needed (all in tparams.py)

The closure is EXPLICIT — it changes the T Dirichlet data between Picard
iterations — and this Picard loop is documented as delicate.
- `NI_WARMUP_ITERS` (8; 30 on the first step): the circuit is FROZEN at the
  insulated state for the first iterations of every step so T/A/rho settle.
  Enabling it against an unsettled flux front diverges.
- `NI_RELAX = 0.10` fixed under-relaxation on I_z. Do not make it adaptive
  without testing — CLAUDE.md records that every adaptive scheme tried on
  this loop misfired.
- `I_Z_BAND = 0.5`: clip |I_r| > |I| (which the ladder forbids) and COUNT the
  clips as `n_clipped`. A silently clipped solution looks converged.

### 2026-08-04: the NI closure Picard loop does NOT converge — RETRACTS the "stable" conclusion above

Follow-up work checked two things the smoke test left open: whether
`hysteretic_power()` is correct, and whether the "stable" `EP_FACTOR_MODE`
result held up under the project's real iteration budget rather than the
smoke test's tight cap.

**The loss formula itself is fine.** `validation/loss_sanity_check.py`
applies it to `ta_solve.solve_ta_at_current()`'s own, independently
well-converged steady solve (k=80, the project's standard reference path,
no NI closure involved) and gets `P_sc = 34.6 W` (0.10 W/m) — sane, and
correctly small relative to the critical-state floor power `E_c*Jc` since
this design runs sub-critical (j/jc mean 0.59, 26% of cells over-critical).
So the formula is not the problem.

**The closure is the problem.** Rerunning step 1 of the ramp
(`transient/validation/ni_closure_stability_check.py`) with the project's
real budget (`N_PICARD_FIRST=150`, not the smoke test's 45) instead of
converging shows a **persistent, non-decaying oscillation**: I_z per bin
swings by tens of amps, 20-30 of 48 bins sit in the physical clip band every
iteration, with no trend toward convergence over 250 iterations. Four
things were tested, independently, and NONE stabilised it:

1. Under-relaxation `NI_RELAX` 0.10 -> 0.02 (5x stronger) — same character.
2. Moving the closure update to AFTER the physics pass each iteration
   (removing the one-iteration lag between the J used to set the BC and the
   J the BC produces) — same character.
3. Contact resistivity `rho_c` 100 -> 2000 uOhm.cm^2 (20x WEAKER NI
   coupling) — same persistent oscillation and clip band. **This is the
   most important negative result**: if the oscillation were ordinary
   physical stiffness from strong radial current, weakening the coupling
   should visibly help. It did not — this points at an implementation-level
   instability in the closure, not stiffness in the physics being modelled.

**Consequence: every transient/ number produced before this check is
unvalidated**, read off a state that never converged — the smoke-test bore
field, the radial current profile, and especially `P_sc = 1182 W` (which is
now understood as noise from an unconverged, oscillating state, not a real
measurement). The apparent stability of `EP_FACTOR_MODE="paper"` documented
above was the same artifact: the smoke test's 45-iteration cap simply didn't
run long enough for the oscillation to appear.

**What is NOT affected:** `ta_solve.py`'s own Picard loop (no NI closure)
converges cleanly, as it always has — confirmed directly in this same check.
Phase A (`circuit/`, the DCN) is a completely separate, independently
validated model; none of its tau/energy numbers depend on this closure.

**This needs a structural fix, not a parameter sweep** — the four negative
results above rule out the cheap fixes. Plausible directions, not yet tried:
treating the circuit update implicitly together with T (closer to a Newton
step than an explicit outer-loop update), per-bin adaptive relaxation
informed by local sensitivity (mirroring the phase-1/phase-2 mechanism that
stabilised the base Picard loop), or coarser radial binning to reduce the
number of coupled degrees of freedom the closure has to stabilise
simultaneously. `ni_closure_stability_check.py` is a regression probe for
any future fix attempt — it should show clean convergence once one works,
and its env vars (`NI_RELAX_OVERRIDE`, `RHO_CT_OVERRIDE`, `REORDER_CLOSURE`)
let a fix attempt vary the three ruled-out suspects without editing the file.

**Open, in priority order:** (1) find a structural fix for the closure
instability — this blocks everything else in Phase B; (2) once fixed, rerun
and actually trust a hysteretic loss number; (3) `dcn_crosscheck.py` should
NOT be run for a real comparison until the closure converges — its result
would just be comparing the DCN against another unconverged snapshot; (4)
the full `run_charge.py` schedule has not been run to completion and should
not be, for the same reason.

### 2026-08-04 (session 2): TWO real bugs found and fixed in the closure, then a THIRD, BIGGER, pre-existing issue found underneath both

Went looking for a structural fix per the open item above. Found and fixed
two real bugs in the closure itself, then discovered the closure was never
the deepest problem — the same instability exists in the completely
UNMODIFIED, non-NI base solver at short time steps. Read this in order; the
first two fixes are real and worth keeping, but the third finding is what
actually matters for whether Phase B can proceed as designed.

**Bug 1 — the closure's E_i term was elementwise-Jacobi on a
non-diagonally-dominant matrix.** The original `ni_circuit.py` computed
`E_i` each Picard iteration from a finite difference using the PREVIOUS
iteration's not-yet-converged `I_z`, combined with `E_p` via elementwise
per-bin arithmetic. The per-turn mutual inductance matrix
(`induction.BinInductance.A`) is dense and far from diagonally dominant (GMD
regularisation ~0.91mm >> the 75um turn pitch, so most turns are nearly as
coupled to each other as to themselves) — an elementwise/Jacobi update on
such a matrix is exactly the textbook non-convergent case; no relaxation
strength fixes a matrix-conditioning problem. **Fixed:** since `E_i` is
LINEAR in `I_z`, solve the inductive part of the closure EXACTLY each
iteration (a 48x48 dense linear solve, `ni_circuit.NICircuit._solve_I_z`),
leaving only the nonlinear `E_p` term Picard-lagged. Verified the resulting
matrix is well-conditioned in isolation (condition number 1.18, max
eigenvalue 1.13) — the linear algebra itself was never the problem, only how
it was being iterated.

**Bug 2 — the transient driver's warmup phase used only the FINE
relaxation, never the base solver's fast ramp-up.** Fixing bug 1 alone made
things WORSE (48/48 bins clipping, `E_p` spiking to 1e4 V/m — physically
~1e-4 V/m is expected). Instrumenting `E_p` during the "frozen" warmup phase
showed the real cause: `ta_transient.py`'s `step()` used only
`alpha_fine=0.15` throughout, never implementing `ta_solve.py`'s own
two-phase scheme (fast `alpha=0.30` ramp-up until `|dB|` stops decreasing,
THEN switch to `0.15`). Without the fast phase, the fixed 30-iteration
warmup was nowhere near enough for the base (insulated-equivalent) state to
settle — SCIF was still swinging by thousands of mT iteration-to-iteration,
the same pattern the base solver's own k=1-10 shows before ITS phase-2
switch. The closure was switching on into a state still in its own early
transient. **Fixed:** `ta_transient._picard_phase()` now reuses the base
solver's exact two-phase relaxation and observable-stall convergence
criterion for BOTH the frozen warmup and the closure-coupled phase, instead
of a fixed iteration count. `NI_WARMUP_ITERS`/`NI_WARMUP_ITERS_FIRST` in
`tparams.py` are now generous CEILINGS, not the actual determinant of when
the closure switches on.

**Finding 3 — the REAL root cause, underneath both fixes: the base T-A
solver's Picard convergence machinery has never been validated at any `dt`
other than `params.ramp_duration` (600s), and does not reliably converge at
shorter `dt`.** After fixing bugs 1 and 2, the warmup phase (now correctly
running the base solver's own two-phase relaxation) STILL failed to
converge in 150 iterations at `dt=100s`. Direct comparison against the
completely UNMODIFIED `ta_solve.solve_ta_at_current()` proved this is not
about the NI code at all:

- At `dt=600s` (the only value ever used anywhere in this project before
  now — `ta_solve.py`'s whole design is "a single implicit BDF1 step over
  the full ramp"), the unmodified base solver converges cleanly, k=64-65,
  matching this project's entire prior history.
- At `dt=100s`, using the IDENTICAL unmodified solver machinery (no NI, no
  per-turn BCs, just a shorter `dt`), the SCIF does NOT converge even after
  **400 iterations** — it wanders persistently between ~10 and ~400 mT with
  no decaying trend.
- A `dt` scan (600/400/300/200/150s, 150-iteration budget each) shows this
  is NOT a simple "shorter dt is harder" relationship: 600, 300, and 150s
  converged (k=65, 53, 89 respectively — note 300s converged FASTER than
  600s); 400s and 200s did NOT converge within the same 150-iteration
  budget. This is the same "near-degenerate flux-front configurations, red
  spectrum wandering" phenomenon this project's own history documents
  extensively for hard-converging cases at the standard dt=600 — except at
  other `dt` values that wandering phase can apparently extend well past
  budgets that have always been sufficient at dt=600, and which `dt` values
  are affected does not follow an obvious pattern.

**What this means:** the NI closure was never the deepest issue. ANY genuine
multi-step time-marching scheme built on `ta_solve.py`'s Picard loop needs
it to converge at a RANGE of `dt` values smaller than the full ramp duration
— that is the whole point of taking multiple steps through a ramp instead of
one. That has never been tested or tuned for, because every existing caller
of this solver (the entire SCIF/optimization history, `ta_sweep.py`
included) uses `dt=ramp_duration` unconditionally. This is a genuinely
bigger task than closing out the NI circuit's own convergence — it likely
needs either (a) a properly characterised, possibly much larger and
dt-dependent iteration budget, (b) a different or adaptive relaxation
scheme specifically for the short-dt regime (untested territory — recall
CLAUDE.md's own warning that every adaptive scheme tried on the dt=600 case
misfired, so a naive adaptive fix here should not be assumed to work
either), or (c) reconsidering the time-stepping granularity (fewer, larger
sub-steps closer to the validated dt=600 regime, at the cost of temporal
resolution).

**Status:** bugs 1 and 2 are real, fixed, and worth keeping regardless of
what happens next — they were both genuine defects independent of finding
3. But `ni_closure_stability_check.py` cannot report a clean PASS until
finding 3 is resolved, since its warmup phase sits on exactly the dt-scan
problem above. Scratch diagnostic scripts used for the dt scan were
temporary (in `/tmp`, not committed); the permanent regression probe
(`transient/validation/ni_closure_stability_check.py`) now calls the real
`ta_transient.step()` directly rather than a hand-rolled copy, so it will
correctly reflect any future fix to either the closure or the base solver's
short-dt convergence.

### 2026-08-04 (session 2, continued): six further convergence strategies tried, ALL FAILED — this is a genuinely hard problem, not a quick fix

Per explicit direction to invest in fixing the short-dt convergence, tested
six candidate remedies against the established hard case (I=32.667 A,
dt=100 s, cold start — the actual first-step conditions of a 6-step ramp
schedule), all using the SAME unmodified base solver machinery (no NI code)
so the comparison isolates the relaxation/acceleration scheme itself. Every
one was judged against the SAME criterion this project already uses:
EMA-smoothed bore SCIF stalling below its tolerance over a sliding window.
**None converged.** Summary (std = spread of the EMA-SCIF over the last
100-150 iterations once any short-lived transient has passed; smaller is
better, and the target is stall < 0.05 mT, i.e. essentially zero):

| scheme | iters tested | outcome |
|---|---|---|
| base solver's own 2-phase alpha (0.30 to /rho_relax=0.5) | 400 | wanders, std~100, no decay trend even by k=400 |
| base solver's own 2-phase alpha, MUCH bigger budget | 1000 | still wandering, std=100.3 over last 100 (range -22 to +520 mT) |
| fixed alpha=0.05 (skip phase-1) | 500 | clean **period-2 limit cycle** (154<->245 mT) — the exact failure mode the original 2-phase scheme was built to escape, reintroduced by going too conservative |
| fixed alpha=0.08 (skip phase-1) | 400 | **best result found**: std=38.98, settling into a ~150-345 mT band — real improvement over the 2-phase default, but never crosses the stall threshold |
| fixed alpha=0.10 | 800 | std still large (range 20-570 mT), no clear convergence even at k=700+ |
| gentle-ramp schedules (small alpha first ~15 iters, then 0.10-0.12 steady) | 800 | no better than plain fixed alpha |
| Anderson acceleration, undamped (Walker & Ni 2011, history depth m=3/5/8) | 300 each | std 68-98 mT, **no better than plain fixed alpha=0.08** |
| Anderson acceleration, damped (beta=0.3/0.5/0.7, m=5) | 300 each | std 47-97 mT, beta=0.3 comparable to (not better than) alpha=0.08 |

Anderson acceleration was the natural next step per this project's own
"Potential next steps" list ("Newton linearisation or Anderson acceleration
... is the remaining convergence bottleneck") and is a well-established,
principled technique (a multi-secant quasi-Newton extrapolation using a
least-squares combination of past iterates) — not another ad hoc guess. It
was implemented correctly (verified against the textbook formula: the
undamped case is exactly `x_bar + f_bar` where `(x_bar, f_bar)` are the
history combination that minimises `||f_bar||`; damping multiplies only the
`f_bar` term by `beta`) and safeguarded (falls back to plain relaxation if
the residual explodes or the least-squares solve is ill-conditioned). **It
did not help.** This is a genuine, informative negative result, not a
failed implementation — see below for why.

**Important clarification found along the way: dt is not "just a numerical
parameter" here — it is a real physical input (the ramp rate), so different
dt values are EXPECTED to converge to genuinely DIFFERENT SCIF answers, not
the same one via different numerical paths.** A short dt represents a
faster ramp (same current change, less time), which drives a larger
induced E-field and a physically different screening response. This
reframes the non-monotonic dt-scan result from the previous entry (600s
converges, 400s doesn't, 300s does, 200s doesn't, 150s does): it is not
that some dt values are "broken" while others are "correct" — every dt is
attempting to converge to ITS OWN physically valid answer, and the
recurring difficulty is that the fixed-point map's own conditioning (via
the power-law rho(J) with n~13-22) apparently gets harder as dt shrinks
below 600s, consistent with a genuinely larger effective forcing term in
the T-equation's RHS (which scales as 1/dt) driving bigger nonlinear
excursions per Picard iteration.

**Why the raw residual not decreasing is not, by itself, damning:** this
project's own history documents that even the VALIDATED dt=600 case has a
raw B/T residual that never cleanly goes to zero — "the front configuration
wanders chaotically among near-degenerate states — raw |ΔB|/|B| floors at
~6-10e-4 with a red spectrum... while the bore SCIF is frozen." The
Anderson tests' `|f|` (T-space residual norm) stayed around 1e10 throughout
with no visible decay in EVERY variant tried, matching that same
documented pattern. So a non-decaying raw residual is consistent with
this solver's known behaviour even in the GOOD case — what actually
distinguishes success from failure is whether the OBSERVABLE (bore SCIF)
settles, and at dt=100 it does not, under any scheme tried.

**Conclusion: this is not fixable by tuning the existing Picard/relaxation
framework, damped or accelerated.** Six genuinely different remedies,
spanning the full space of "reasonable next things to try" for a
relaxed-fixed-point scheme, converged on the same negative result. The
practical paths remaining are qualitatively different in scope from
everything tried so far:

1. **A true Newton-Krylov reformulation** (`dolfinx.fem.petsc.NonlinearProblem`
   + PETSc SNES, with the T-equation's actual Jacobian derived via
   `ufl.derivative` instead of the current "linear-with-frozen-rho, iterate
   to a fixed point" Picard scheme). This is the textbook answer for a
   stiff power-law nonlinearity, has fundamentally better convergence
   theory than Picard+acceleration, but is a substantial rewrite of the
   solver core — plausibly multiple days of work, and carries real risk of
   its own convergence difficulties in a different form (Newton needs a
   reasonable initial guess and a correct Jacobian; an incorrect Jacobian
   fails silently as slow/no convergence, not a crash).
2. **Report the SCIF as an inherently banded/uncertain quantity during
   short-dt sub-steps**, rather than insisting on a single converged
   number — in the same spirit as this project's existing practice of
   quoting bounds instead of point values for other near-degenerate/
   resolution-limited quantities (e.g. the on-axis SCIF history). This is
   far cheaper (no solver rewrite) but changes what a transient run can
   promise: an uncertainty band on the ramp trajectory, not a precise curve.
3. **Reconsider the time-stepping schedule itself** — e.g. deliberately
   choosing sub-step sizes that land near dt values empirically observed to
   converge well (300s, 150s) rather than a uniform partition, or taking
   one large step close to the validated 600s regime and only sub-stepping
   near the target current where warm-starting from an already-converged
   neighbour is expected to help. Untested; the one non-uniform-schedule
   idea tried so far (many small EQUAL steps, tested at n_steps=6/12/24/48)
   did NOT help — the wandering did not visibly shrink with smaller
   uniform steps, and n=48's trace showed a violent sign-flipping swing to
   -1122 mT, suggesting this is not simply "smaller steps are gentler."
   A schedule specifically chosen to hit the empirically-good dt values is
   a different, untested idea from uniform fine-stepping.

**This is now a scope decision, not a technical one** — the six approaches
tried represent the reasonable space of "improve the existing iteration"
fixes, and none worked. Further progress needs either a substantially
larger engineering investment (option 1) or an honest change to what
Phase B's transient output claims to deliver (option 2), or a targeted,
different experiment on the schedule itself (option 3, cheap to try next
if there's appetite for one more test before committing to 1 or 2).

### 2026-08-04 (session 2, part 3): Newton-Krylov — CORE HYPOTHESIS VALIDATED, outer-loop robustness still open

Direction chosen: option 1 (Newton-Krylov). New, additive module
`transient/newton_ta.py` — does NOT modify `solve/ta_solve.py`. Reuses
`ta_solve.setup_ta_problem()`'s output unchanged; the existing
`prob_T_layers` (Picard, `LinearProblem`) and the new `newton_problems`
(`NonlinearProblem`/SNES) coexist on the same `ta` dict. Every existing
production path (`ta_sweep.py`, `optimize/`, `solve_ta_at_current()`) is
untouched and still gives byte-identical results.

**Approach: quasi-Newton, not full Newton.** A full Jacobian would need
`d(Ic)/dB`/`d(n)/dB` from the measured-CSV spline models — not something
UFL's `ufl.derivative` can differentiate symbolically. Instead `Jc`/`n` are
FROZEN as DG0 coefficients (updated between outer iterations from the
measured Ic(B,theta)/n(B,theta) models, exactly as `_update_rho` already
does), and only the power-law-in-J term is written as genuine UFL algebra
in `J = curl(T)`:
```
rho(J) = (E_c/Jc) * exp((n-1)*log(smooth_floor(|J|/Jc))) * (delta_SC/Lambda)
```
`ufl.derivative` handles this automatically — no hand-derived spline
Jacobian needed. This linearises exactly the dominant stiffness driver (the
`n~13-22` exponent) while Picard-lagging the milder `Ic(B)`/`n(B)`
dependence: a nested loop, OUTER Picard-on-Jc/n, INNER exact Newton-per-
layer given frozen Jc/n.

**Result: CORE VALIDATION PASSED.** A single-layer smoke test at the exact
hard case (dt=100s, I=32.667A, cold start — the case Picard never converged
in 1000+ iterations) converged in 11 SNES iterations, ~10 orders of
magnitude residual drop. The full 6-layer hybrid, run at dt=600s (the
established regression point, Picard's own answer: SCIF=172.77 mT at
k=71), reached **SCIF=171.49 mT after just 2 OUTER iterations** (each
needing only 1-16 cheap inner Newton iterations) -- under 1% off the
Picard-validated ground truth. This is a genuine, decisive confirmation the
reformulation is physically correct, not just "SNES reports converged."

**Three real bugs found and fixed building this (all worth knowing for any
future Newton/UFL work in this project):**
1. FFCx's automatic quadrature-degree estimation cannot handle the
   transcendental `exp(log(...))` power-law form -- left to guess, it tried
   to allocate a 125000x125000 quadrature table (116 GiB). Fixed with an
   explicit `quadrature_degree` on the measure
   (`newton_ta.NEWTON_QUADRATURE_DEGREE`).
2. Writing `rho = (E_c/Jc)*...` symbolically means the division is
   evaluated EVERYWHERE the form is integrated, including cells where `Jc`
   has no physical meaning for that layer's own NonlinearProblem (other
   layers' cells, where `T` is Dirichlet-pinned to zero). `Jc=0` there gives
   `1/0=inf`, and `0 (from curl(T)=0) * inf = NaN` in IEEE arithmetic, not
   zero. Fixed by giving `Jc_fn` a SAFE NONZERO default (1.0) everywhere --
   the zero-`T` pinning makes the true contribution zero regardless of the
   placeholder value.
3. **The one that actually mattered for correctness**: a first version of
   `rho_expr` omitted the `(delta_SC/Lambda)` homogenisation factor that
   `ta_solve._update_rho` applies (`rho_homog = rho_SC * (delta_SC/Lambda)`)
   -- rho came out ~75x too large, and the outer loop converged CLEANLY
   (SNES genuinely happy) to a self-consistent but PHYSICALLY WRONG answer
   (SCIF=13.6 mT instead of ~172 mT). This is the important lesson: **"SNES
   reports converged" and "the formulation is correct" are different
   claims** -- always check a new nonlinear formulation against an
   independent ground-truth number, not just its own solver status. This
   session's whole history is proxies that looked fine until checked;
   this is the same lesson applying to a Newton reformulation, not just to
   the old Picard/proxy work.

**Two robustness patches applied, both real and worth keeping:**
- A cold `T=0` start is too far from the solution for line-search Newton
  (first attempt: `SNES_DIVERGED_LINE_SEARCH` on the very first solve). Fixed
  by bootstrapping with a SHORT run of the EXISTING, validated two-phase
  Picard scheme (`ta_transient._picard_phase`, reused unmodified, NOT a
  second hand-rolled version -- a first attempt at a hand-rolled fixed-alpha
  bootstrap reintroduced the period-2 divergence the two-phase scheme exists
  to prevent, diverging to NaN within 15 iterations at dt=600, the
  supposedly "easy" case). The bootstrap only needs to get "close enough"
  for Newton, not converge on its own.
- Once an individual layer's Newton solve reports non-convergence
  (`SNES_DIVERGED_LINE_SEARCH` or `_MAX_IT`), retry once with the line
  search switched to `"basic"` (plain, unglobalized step) and a higher
  iteration cap -- standard PETSc troubleshooting for exactly this failure
  signature (a robust globalized method failing its own merit-function test
  right at the edge of convergence, not the Newton direction itself being
  bad).

**OPEN: the outer Jc(B)/n(B) Picard loop's own robustness right at its
fixed point.** Past the iteration that lands on the correct answer (k=2,
SCIF=171.49 mT), continuing to iterate hits a genuinely hard regime:
`SNES_DIVERGED_DTOL` (residual increase past the divergence tolerance),
observed specifically on the layer whose frozen `n` reaches its highest
range (~27-28) -- plausible, since the `(n-1)` exponent in `d(rho)/dJ`
makes the Jacobian most sensitive exactly there, and this may be landing
near the same "near-degenerate marginal flux front" regime documented
throughout this project's history for the Picard scheme, just manifesting
as a Newton-conditioning issue instead of an oscillation. Tried and
REJECTED: keeping a failed layer's previous `T` and retrying next outer
iteration -- this does not recover, it lets 4 of 6 layers get permanently
stuck while Jc/n keeps moving around them, and the whole state drifts away
from the correct answer (observed: SCIF drifting to -571 mT over 30
iterations once stuck). The right fix is most likely a smarter STOPPING
criterion (recognise that k=2's answer is already correct and exit there,
rather than trying to iterate to a formal multi-iteration stall check that
requires pushing past the point of numerical practicality) rather than a
robustness patch on continuing past it -- this echoes the project's own
established lesson that "solves may report non-convergence with the
observable already settled" (documented for the Picard scheme's SCIF-stall
criterion; the same principle likely applies here, just needs a properly
designed stopping rule for THIS nested scheme, not another ad hoc retry).

**Status:** the reformulation itself is validated and should not need
revisiting. What remains is a bounded, well-characterised stopping/
robustness question for the outer loop, not an open-ended one -- next
step is a smarter convergence check (e.g. accept the result once even 1-2
outer iterations show sub-mT SCIF change, rather than requiring a 6-
iteration window that forces the loop past its own practical fixed point),
not further blind SNES-option tuning.

### 2026-08-04 (session 2, part 4): outer-loop stopping mechanism fixed for dt=600; dt=100 improved but not yet validated

Continued directly from the open item above. Diagnosed the actual mechanism
of the outer-loop failure (not another blind parameter search) and fixed
the stopping logic properly.

**Root cause of the outer-loop instability:** the ORIGINAL "keep this one
failed layer's previous T, retry next iteration" safeguard was not just
unhelpful, it was the actual cause of a cascading collapse. Once one layer's
T goes stale while every OTHER layer's T (and the shared A-equation, which
couples all layers together) keeps advancing, that layer's J no longer
matches the field it's embedded in -- corrupting the NEXT iteration's Jc/n
update for every layer, causing more failures, compounding. Verified
directly: tried at THREE different Jc/n relaxation strengths (0.5, 0.3,
0.2) and every one showed the identical pattern -- SCIF lands within 1% of
truth at outer iteration k=2, then whichever layer fails first triggers a
cascade that drifts the SCIF to -400 to -620 mT over the following 10-30
iterations. Relaxation strength was never the actual lever.

**Fix, in two parts:**
1. **Revert the WHOLE outer iteration, not just the failed layer.** Snapshot
   every layer's T, Jc, n, AND the shared A field before each outer
   iteration; on any failure, restore ALL of them together and stop. This
   keeps every layer mutually consistent with the field it was solved
   against, which is what the per-layer revert broke.
2. **Detect trouble one step earlier than outright SNES failure.** The
   iteration that actually corrupts the state (k=3 in the dt=600 case) often
   has SNES report SUCCESS (reason=3) for every layer -- just after an
   iteration-count SPIKE (28 iterations for a layer that needed 1-2 the
   previous iteration). By the time the FOLLOWING iteration's SNES call
   fails outright, the damage is already done. Added a spike check
   (`its > max(10, 3*previous_max)`) that treats a large iteration-count
   jump as a stop signal with the SAME revert-and-stop handling, even when
   SNES's own status claims success -- this is the same lesson the missing
   delta_SC/Lambda bug taught in a different form: **a solver's own
   "converged" status is not the same claim as "this is correct."**

**Verified: `converged` must mean "trustworthy," not "did not crash."** A
first attempt tried to auto-classify a reverted stop as trustworthy via a
crude check (are the last two successful iterations' SCIF within 5 mT of
each other?). Tested directly against the one case with known ground truth:
it FAILED, flagging the dt=600 result (150.65 -> 171.53 mT, a 21 mT gap
between the last two points, yet independently confirmed correct to <1%) as
"not trustworthy." No 2-point heuristic reliably distinguishes "still
settling" from "converging non-monotonically." **Fixed by removing the
auto-classifier entirely**: `converged` is now True ONLY when the real,
formal multi-iteration SCIF-stall criterion fires on an uninterrupted run.
Any revert-and-stop ALWAYS returns `converged=False`, with `stop_reason`
(`"newton_failure_reverted"` / `"iteration_spike_reverted"` / `"stall"` /
`"max_outer"`) and `scif_hist_tail` (the last up to 6 EMA-smoothed SCIF
values) exposed so a caller can judge the trajectory -- or check against
independent ground truth, which is what actually validated the dt=600
result, not the solver's own status flag.

**Result, dt=600 regression: SOLID, repeated 3x.** `171.46 / 171.48 / 171.53
mT` across three independent runs (fresh mesh each time), all within 1% of
the Picard-validated `172.77 mT`. `stop_reason=iteration_spike_reverted`
every time (the formal stall window never got the chance to fire before the
spike check caught it first) -- expected and fine, per the `converged`
semantics above; this is judged on the SCIF value against ground truth, not
the flag. This is now a genuinely validated, reproducible result at the
project's own long-standing dt=600 operating point, reached via 2-3 OUTER
iterations (vs Picard's 71).

**Result, dt=100 (the actual target): IMPROVED, NOT YET VALIDATED.** No
longer wanders indefinitely with zero decay trend (Picard's failure mode,
confirmed to persist past 1000 iterations) -- it now reaches a well-defined
stop with clear diagnostics every time. But the STOPPED VALUE is not
reproducible run to run: three independent runs gave `194.13 / -187.35 /
533.29 mT` -- wildly different, including a SIGN FLIP. There is no ground
truth to check against here (Picard never converged at dt=100 at all), but
this run-to-run spread is itself informative: it means dt=100 is genuinely
still-unsettled physics being cut off by the stop mechanism at different
points each time (consistent with mesh-regeneration randomness perturbing
which iteration first hits the spike/failure threshold), not numerical
noise around a stable answer the way dt=600's <1%-spread result is.

**Honest interpretation:** the Newton reformulation has CONCLUSIVELY solved
the problem this work originally set out to fix at the project's
established operating point (dt=600) -- and done so with a principled,
understood, reproducible mechanism, not a lucky parameter choice. Whether
dt=100 specifically needs further engineering (more bootstrap iterations,
a schedule that never takes a bare cold jump that large/fast, or something
else) or reflects a genuinely harder physical regime (a very fast ramp
rate driving sharper flux-front gradients) is still open. Given dt=100 was
never the goal in itself -- it was chosen as "the hardest case Picard could
not solve" to stress-test the fix -- the practical next step for actually
using this in a real ramp schedule is to test it at the dt values a real
multi-step schedule would use (e.g. 25-150s for a 4-24 step, 600s ramp) and
see where the boundary between "reproduces reliably" and "still spread
across runs" actually falls, rather than assuming dt=100 specifically must
be solved before this is useful.

**Status of `transient/newton_ta.py`:** the module is now internally
consistent and its `converged`/`stop_reason`/`scif_hist_tail` diagnostics
are trustworthy (verified against ground truth, not just asserted). Not yet
wired into the NI circuit closure (`ni_circuit.py`) or a full multi-step
`march()` -- both remain future work, now on solid footing rather than on
top of an unvalidated solver.

### Module-name collision (bit once, fixed)

`circuit/run_charge.py` and `transient/run_charge.py` share a module name and
this repo has **no packages anywhere** — everything is a bare import off a
flat `sys.path`. A transient smoke test silently ran circuit's driver
instead. `transient/tparams.py` therefore does NOT put `circuit/` on the
path; anything needing both sides loads circuit modules by explicit file path
under distinct names (`_load_circuit_module`). Watch for this with any future
same-named module.

Run:
```bash
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
$PY transient/validation/insulated_limit.py    # B0 gate — must stay PASS
$PY transient/run_charge.py
$PY transient/validation/dcn_crosscheck.py     # B2
```

---

## 2026-08-05: first multi-step `newton_ta.march()` run — mechanics work, but the outer loop does not converge in this regime

Built and ran `transient/newton_ta.march()` (added the previous session,
never executed) on a 4-step, dt=150s/step, insulated ramp
(I: 49→98→147→196A, `transient/validation/newton_march_check.py`). Two real
bugs found and fixed BEFORE trusting any output, per explicit instruction to
verify along the way rather than just report a final number:

1. **`ta_transient.ramp_schedule()`'s `n_hold = n_hold or tp.N_STEPS_HOLD`
   silently discarded an explicit `n_hold=0`** (0 is falsy) — requesting a
   ramp-only schedule instead appended `tp.N_STEPS_HOLD` extra steps, each
   with `dt = t_hold/n_hold = 0.0/N = 0`, which fed a `1/dt` term straight
   into a NaN (`SNES reason=-4`) and crashed the run. Fixed to
   `n_hold if n_hold is not None else tp.N_STEPS_HOLD` (same pattern already
   used correctly for `t_ramp`/`t_hold`); `n_ramp` had the identical latent
   bug, fixed too.
2. **`newton_ta.step()`'s final summary `print` crashed with
   `TypeError: unsupported format string passed to NoneType.__format__`**
   when the very first outer iteration reverts before ever completing an
   A-solve (`scif_ema` stays `None`) — exactly what bug 1's `dt=0` step
   triggered. The returned dict already handled this case; only the print
   didn't. Fixed with a guarded format string.

**After fixing both, the march runs end-to-end with no crash, and the
mechanics are verified correct**, not just assumed: the applied current
genuinely advances step to step (49/98/147/196A, confirmed distinct — this
is the regression test for bug-fix #1 from the PRIOR session, "BC set every
step, not just first"), and T/Jc/n/A are genuinely warm-started (later steps
need far fewer SNES iterations per layer than a cold start, e.g. 1 vs the
first step's 8-14).

**But none of the 4 steps reached the formal SCIF-stall convergence
criterion.** Step 1 (cold bootstrap) stopped via `newton_failure_reverted`
after 3 outer iterations; steps 2-4 (warm-started) each stopped via
`iteration_spike_reverted` after just 1. **Checked whether the spike guard
is just over-tuned for this new regime** (it was calibrated on the single
validated dt=600s case) via `transient/validation/spike_check_diag.py`:
re-ran step 2 with the guard disabled and a bigger outer-iteration budget.
**Result: not over-tuned.** Without the guard, the same step still fails —
one iteration later, via an outright Newton divergence
(`newton_failure_reverted`, SNES reason ≤ 0) instead of the guard's
early-warning trip. So this is a genuine convergence difficulty specific to
the warm-started, sequential-current-ramp regime at dt=150s, not a false
alarm from an overly-sensitive heuristic — consistent with (and an extension
of) the earlier finding that this solver's convergence behavior is
`dt`-dependent and has never been characterized outside dt=600s and the
single dt=100s stress test.

**Also found: SCIF values from this unconverged regime are not even
run-to-run reproducible.** Step 1 (I=49A, dt=150s, cold start) gave three
different values across three independent process launches: +125.45 mT
(the march run), -130.99 mT (an earlier march run before the schedule fix),
-0.70 mT (`spike_check_diag.py`). This is consistent with — not a
contradiction of — the state never reaching a genuine fixed point (gmsh's
documented cross-process mesh non-reproducibility is only visible at this
scale when the underlying solve itself hasn't settled; it stays invisible
at true convergence, as the dt=600 regression's <1%-spread repeats show).
**No SCIF number from a multi-step march should be treated as physically
meaningful yet.**

**Status:** the `march()`/`step()` machinery itself (schedule handling, BC
tracking, warm-starting, revert/spike safety, diagnostics) is now verified
correct end-to-end. What remains open is the SAME class of problem the
single-step dt-scan already flagged as open (see the 2026-08-04 entries) —
Newton fixed the inner power-law nonlinearity decisively at dt=600, but the
outer Jc(B)/n(B) Picard-lag loop's own robustness across a REALISTIC
multi-step schedule (dt=150s here) has not been achieved. Candidate next
steps, not yet tried: a larger `max_outer` budget combined with weaker
Jc/n relaxation (trading more, gentler outer iterations for stability, the
same lever that fixed the single-step dt=600 case's own early instability);
per-step bootstrap iterations tuned per step rather than a fixed 30; or
accepting dt=150 is still too aggressive for the warm-started multi-step
path specifically and profiling a schedule with more, smaller steps nearer
the step-to-step current increments already shown to behave.

Run: `<env>/bin/python3 transient/validation/newton_march_check.py` (smoke
test) and `transient/validation/spike_check_diag.py` (spike-guard
diagnostic).

### 2026-08-05 (continued): outer-loop fix attempts — root cause isolated, not yet fixed

Three further diagnostics, each testing one specific lever rather than
guessing, per explicit direction to invest in the fix:

1. **`relax_sweep_diag.py`: Jc/n relaxation strength (0.5/0.3/0.2/0.1) makes
   NO difference.** All four converge to nearly the SAME failure (SCIF
   -141 to -144 mT, `iteration_spike_reverted` at n_outer=1) on the
   identical warm-started step. Rules out relaxation magnitude as the
   lever — a clean re-test under the CURRENT whole-iteration revert logic,
   not a repeat of the pre-fix finding CLAUDE.md already recorded that
   relaxation "didn't matter" (that older finding was contaminated by the
   since-fixed per-layer-revert bug and doesn't actually apply here).
2. **`di_sweep_diag.py`: current-jump size (+6/+11/+21/+31/+49A) also makes
   NO difference.** Every jump, including the gentlest (+6A, from
   49→55A — trivial by any reasonable standard), fails identically. Rules
   out "the jump was too big" as the explanation.
3. **`spike_floor_diag.py`, the decisive test: the spike guard IS
   miscalibrated for warm starts, AND there is a separate, real underlying
   failure — both true at once.** A warm-started step's first outer
   iteration is anomalously easy (its=1, since T is already close to the
   new BC), so `prev_max_iters` resets to 1 and the guard's relative
   threshold (`its > max(10, 3*prev_max_iters)`) trips on completely
   ordinary iteration-2 behaviour. Disabling the guard for the gentlest
   possible jump (+6A) with a much bigger budget did NOT let it converge:
   SCIF still swung -89 → -37 mT between outer iterations 1 and 2 (not
   settling), and outer iteration 3 then hit a genuine
   `SNES_DIVERGED_DTOL` (reason=-9) on the layer with the highest frozen
   `n` (~27, vs 20-26 on the other five layers).

**This is the SAME failure signature CLAUDE.md already documented as an
OPEN item for the dt=600 single-step case** ("the layer whose frozen `n`
reaches its highest range (~27-28)... the `(n-1)` exponent... makes the
Jacobian most sensitive exactly there"). It is not a dt=150-specific or
multi-step-specific bug — it is the SAME outer-loop Jacobian-sensitivity
problem, present at dt=600 too. What differs is that at dt=600 (cold
start, single step) the SCIF happens to already be within 1% of the true
answer by the time this divergence hits (k=2), so the existing "accept the
best reverted state" design recovers a trustworthy number anyway. Here
(dt=150, warm-started), the SCIF has NOT settled when the same divergence
hits (still moving 50+ mT per outer iteration) — so there is no good
fallback state to accept. **The dt=600 workaround (stop early, trust the
last good state) does not transfer to this regime because "early" here
isn't yet close to converged.**

**Root cause, best current understanding:** the single shared
`jc_n_relax` factor damps every layer's Jc/n update equally, but the
instability is concentrated in whichever layer's frozen `n` is highest —
uniformly weaker relaxation (tested, no effect) doesn't target that. A
PER-LAYER relaxation (weaker specifically for high-`n` layers, where the
power-law Jacobian is most sensitive) is the next untried, targeted lever
— cheap to test, unlike a further architecture change. The bigger
alternative (fold Jc(B)/n(B) into the Newton residual too, via a
finite-difference or automatic-differentiation Jacobian through the spline
models, removing the outer Picard-lag loop entirely) would eliminate the
class of problem outright but is a substantially larger rewrite, flagged
here as a scope decision, not attempted.

**Status: root cause isolated and characterized, not yet fixed.** The
`march()` mechanics (schedule, BC tracking, warm-start plumbing, revert
safety) remain verified correct; what's blocked is the outer loop's
robustness on the highest-`n` layer, in BOTH single-step and multi-step
regimes, not just the multi-step case as first suspected.

Added `jc_n_relax` kwarg to `newton_ta.step()` (default `None` = unchanged
behaviour, falls through to `params.ta_rho_relax`) to make this and future
relaxation experiments possible without editing the function. `spike_check`
kwarg (added same day) similarly defaults to `True` = unchanged behaviour.

Run: `<env>/bin/python3 transient/validation/relax_sweep_diag.py`,
`di_sweep_diag.py`, `spike_floor_diag.py`.

### 2026-08-05 (continued further): per-layer relaxation ALSO fails — every cheap lever is now exhausted

Added `high_n_relax_factor`/`high_n_threshold` kwargs to `newton_ta.step()`:
extra-damp (multiply relax by `high_n_relax_factor`) specifically whichever
layer's CURRENT frozen `n` exceeds `high_n_threshold` (24.0), leaving the
other layers at the normal relax — a targeted version of the uniform
relaxation sweep that already failed, since the instability is concentrated
in one layer, not spread evenly.

**First attempt (spike_check=True, the default) was confounded**: every
factor tripped the (already-known-miscalibrated) spike guard at outer
iteration 1→2 regardless of damping strength, so it tested the guard, not
the actual divergence. Re-ran with `spike_check=False` to test the real
question. **Result: factor ∈ {0.3, 0.1, 0.03, 0.01} all fail IDENTICALLY**
— same layer (the highest-`n` one), same `SNES_DIVERGED_DTOL` (reason=-9),
same outer iteration (n_outer=2), SCIF converging to the same ~169-170 mT
regardless of how hard that layer's coefficients were damped (factor=0.01
is a 99% reduction in its update speed — effectively frozen — and made
NO measurable difference).

**This is decisive, not just another negative data point: damping speed
has ZERO effect on the outcome.** That rules out "the coefficients are
moving too fast for Newton to track" as the mechanism entirely — if it
were, more damping would have to help at least a little, and it did not,
down to a 100x range in damping strength. The divergence is not about HOW
FAST Jc/n reach their target; it happens at essentially the SAME resulting
state regardless of the path taken there. This points toward the target
state itself being genuinely ill-conditioned for the line-search Newton
solve on that layer (consistent with — and now sharpening — the project's
existing "near-degenerate marginal flux front" language used elsewhere for
the base Picard scheme's own hard cases), not a step-size/damping problem
at all.

**Every cheap, parameter-level lever tried across this whole investigation
has now failed, cleanly and for a specific, checked reason, not just "we
tried some numbers and gave up":**

| lever tested | result |
|---|---|
| uniform Jc/n relaxation (0.5→0.1) | no effect — same failure point every time |
| current-jump size (+6A→+49A) | no effect — even the gentlest jump fails |
| spike-guard threshold | miscalibrated for warm starts, but disabling it does NOT rescue convergence — a real failure sits underneath |
| per-layer relaxation on the worst layer (1.0→0.01) | no effect — same layer, same failure, same SCIF, regardless of damping |

**This is now a scope decision, in the same sense CLAUDE.md's earlier
Picard-acceleration investigation reached one** (six remedies tried,
none worked, conclusion: needs Newton-Krylov or a fundamentally different
approach — see the 2026-08-04 "six further convergence strategies" entry).
The two live options, neither attempted yet:

1. **Fold Jc(B)/n(B) into the Newton residual itself**, removing the outer
   Picard-lag loop entirely (a genuinely monolithic Newton solve). Would
   need a Jacobian contribution through the measured-CSV spline models —
   not UFL-differentiable, so this means either a finite-difference
   Jacobian block or precomputing dJc/dB, dn/dB from the spline objects
   directly and assembling a custom Jacobian. Substantially larger than
   anything tried in this investigation; would eliminate the class of
   problem outright if it works, but carries its own convergence risk
   (an approximate/FD Jacobian can itself misbehave).
2. **Investigate WHY line-search Newton specifically fails
   (`SNES_DIVERGED_DTOL`) on the highest-`n` layer at that state** — e.g.
   inspect the linear system's conditioning at the failing iterate, try a
   different SNES type (trust-region `newtontr` instead of `newtonls`,
   which handles exactly this "line search's merit function rejects every
   step" failure mode differently), or a continuation/homotopy approach
   that ramps the EXPONENT `n` itself rather than jumping straight to its
   measured value. Cheaper to try than option 1, not yet attempted.

Neither was pursued without a direction call, since option 1 is a large
investment and option 2, while cheap, is still unproven speculation rather
than something already isolated by a diagnostic the way every other lever
above was.

Run: `<env>/bin/python3 transient/validation/high_n_relax_diag.py`.

### 2026-08-05 (continued further still): FOUND IT — T itself was never damped, only Jc/n

Prompted by a direct challenge to the "it's just the nonlinearity" framing:
if the power-law nonlinearity's stiffness were the mechanism, SOME amount
of damping should have helped SOME amount. A 100x range in Jc/n damping
(1.0 -> 0.01) producing an IDENTICAL failure -- same layer, same
`SNES_DIVERGED_DTOL`, same SCIF to <1 mT -- is not what stiffness looks
like. It's the signature of a lever that was never connected to the actual
unstable variable.

**Re-examined what the ORIGINAL (validated) Picard scheme actually damps:
TWO things, every iteration** -- `rho`/Jc/n AND `T` itself
(`T_i = (1-alpha)*T_old + alpha*T_new`, unconditionally, every Picard
iteration). The Newton-hybrid only ever damped Jc/n; each layer's Newton
solve is EXACT and its result was accepted outright, with no damping
before it drives the shared A-equation. **That is the missing half of the
original scheme's stability mechanism.** Six layers' exact,
individually-consistent-only-with-their-own-frozen-Jc/n T solutions,
fed straight into the shared A-equation every iteration with no smoothing
at all, is a fundamentally different (and less stable) coupling than the
original scheme's damped field evolution.

**Added `t_relax` kwarg to `newton_ta.step()`/`march()`**: blends each
layer's freshly-solved T with its start-of-iteration value
(`T = (1-t_relax)*T_old + t_relax*T_new`) before computing J/A/B, applied
AFTER Newton's exact per-layer solve -- Newton still handles the dominant
power-law nonlinearity exactly; only the outer field-coupling update gets
damped, restoring the missing half of the original two-part damping.

**Tested on the identical case that failed under every previous lever**
(I=49->55A, dt=150s, warm-started, spike_check off so the real behaviour
is visible):

| t_relax | outcome |
|---|---|
| 1.0 (no T-damping, prior behaviour) | fails at n_outer=2 |
| 0.5 | fails at n_outer=31 (15x improvement) |
| 0.3 | fails at n_outer=59 (nearly the full 60-iter budget) |
| **0.15** | **`converged=True`, `stop=stall`, n_outer=41, SCIF=129.41 mT** |
| 0.08 | ran out of budget at n_outer=60, SCIF=129.27 mT -- clearly converging to the SAME ~129 mT answer as 0.15, just needed a few more iterations |

**This is the first genuine, formal-stall-criterion convergence anywhere
in the multi-step warm-started investigation.** Two independent t_relax
values landing on the same ~129 mT answer is corroborating evidence it is
a real fixed point, not an artifact of one specific relaxation choice.

**Full 4-step ramp re-run with `t_relax=0.15`** (49->98->147->196A,
dt=150s/step, `spike_check=False`, `max_outer=60`,
`transient/validation/newton_march_check.py`): **zero crashes across all
four steps** -- a complete reversal from every previous multi-step attempt,
which died after 1-2 outer iterations on EVERY step. None of the four
steps reached the strict 0.05 mT stall criterion within the 60-iteration
budget (each hit `max_outer` instead), but every step showed smooth,
monotonic (or band-bounded) settling -- SCIF trajectories like
2586 mT -> 365 mT over step 2's 60 iterations, never a crash, never
unbounded growth. **This is now purely a budget question** (more outer
iterations, or a tighter/adaptive stall check that recognises a settling
trend), not a stability question -- the instability itself is fixed.

**Status: the outer-loop fix is validated and works.** Recommended
follow-ups, not yet done: (1) raise `max_outer` for a production run (the
isolated single-step test needed 41 iterations from an already-decent
warm start; a full ramp's early cold-start-adjacent steps may need
noticeably more since step 1 itself didn't fully settle in 60 either);
(2) re-tune `min_outer`/the stall window now that trajectories are smooth
rather than chaotic -- a smooth monotonic decay may allow a looser/faster
stall detection than the noisy case the original 6-iteration-window
criterion was designed for; (3) sweep `t_relax` further (0.1-0.2) to see
if a value exists that converges the FULL ramp within a reasonable budget,
not just the isolated step; (4) the spike guard remains miscalibrated for
warm starts specifically (still true, unrelated to this fix) and was
disabled (`spike_check=False`) for all of this validation -- either fix
its threshold or leave it off by default for warm-started steps.

Run: `<env>/bin/python3 transient/validation/t_relax_diag.py` (isolated
step sweep) and `newton_march_check.py` (full 4-step ramp, now using
`t_relax=0.15, spike_check=False`).

## 2026-08-04: overnight validation of the t_relax fix — zero crashes across 9 independent runs, 74 step-solves

Built `transient/validation/run_one_schedule.py` (CLI-driven, single-job
runner, writes a JSON result + full log) and
`transient/studies/overnight_newton_validation.py` (orchestrator: launches
each job as a SEPARATE subprocess, time-boxed, incremental CSV aggregation,
global budget on `time.monotonic()` -- same pattern as
`optimize/studies/double_pancake_search.py`/`overnight_refinement.py`, and
the same reasons: process isolation, crash-safety, and the project's own
recorded `time.time()`-vs-`time.monotonic()` budget bug). Smoke-tested the
CLI runner and the orchestrator's subprocess/timeout/aggregation machinery
on trivial configs BEFORE launching the real run, per this project's
established "verify before handing off" discipline.

Launched via the direct python binary + `nohup ... & disown` (never
`conda run` -- buffers all output). 9 jobs, 11h budget, actually finished
in **3.87h** with **zero crashes, zero NaN, zero unbounded blowups across
74 individual step-solves**:

| job | steps | converged (stall) | final-step SCIF | final stop_reason |
|---|---|---|---|---|
| dt150 t_relax=0.15 run1 | 4 | 0/4 | 548.10 mT | max_outer |
| dt150 t_relax=0.15 run2 | 4 | 2/4 | 556.05 mT | max_outer |
| dt100 t_relax=0.15 | 6 | 3/6 | 486.63 mT | **stall** |
| dt075 t_relax=0.15 | 8 | 1/8 | 569.27 mT | max_outer |
| dt050 t_relax=0.15 | 12 | 3/12 | 468.58 mT | newton_failure_reverted |
| dt025 t_relax=0.15 | 24 | 7/24 | 345.56 mT | **stall** |
| full_ramp_hold dt150+200s hold | 8 | 1/8 | 274.97 mT (after hold decay) | newton_failure_reverted |
| dt150 t_relax=0.10 | 4 | 0/4 | 574.66 mT | max_outer |
| dt150 t_relax=0.20 | 4 | 0/4 | 579.51 mT | max_outer |

**17 of 74 step-solves (23%) reached the formal 0.05 mT stall criterion.**
The other 77% did NOT diverge -- they either plateaued in a stable band at
`max_outer` (the dominant outcome) or, at dt=50 and in the hold phase,
eventually reverted via `newton_failure_reverted` but only after
**100-138 outer iterations** (vs 1-2 before the t_relax fix), with the
reverted SCIF still tracing a smooth, physically sensible curve right up
to the point of failure. **The fix's core claim -- eliminating the
1-2-iteration collapse -- holds up over a full night of independent runs
across the entire realistic dt range (25-150s), both t_relax values near
0.15, and a ramp+hold schedule**, not just the single isolated case it was
first validated on.

**t_relax is not sensitive within 0.10-0.20**: the four dt=150 final-step
values across two t_relax values and one repeat (548/556/575/580 mT) sit
in a tight ~5.5% band -- the fix does not need precise tuning.

**Physical sanity check, not just numerical stability**: the hold-phase
job (constant I=196A for 200s after the ramp) shows SCIF **decaying
monotonically** (507->447->395->330->275 mT over the 4 hold sub-steps) --
exactly the expected physical relaxation of ramp-induced screening current
toward the (much smaller) DC steady-state value once the current stops
changing. This is independent evidence the model is doing something
physically sensible, not merely avoiding a crash.

**Open item, not resolved tonight: dt=25's final value (345.56 mT) is
noticeably lower than every other dt's final value (468-580 mT)**, a
bigger gap than the dt=50-vs-75-vs-100-vs-150 spread among themselves.
CLAUDE.md's own established framing is that different dt SHOULD give
genuinely different physical answers (a faster ramp drives a larger
induced E-field) -- but this specific gap is large enough, and dt=25 is
distinctive enough (the only job with a majority of genuinely
stall-converged steps, 7/24, vs single digits elsewhere), that it deserves
a dedicated check before being accepted at face value: e.g. does it
persist on a repeat run (mesh non-reproducibility), or is finer dt itself
systematically pulling the answer down for a real physical reason. Not
tested yet.

**What tonight did NOT test**: whether the `newton_failure_reverted` steps'
returned SCIF is actually TRUSTWORTHY, only that it is smooth/bounded --
`converged=False` on those steps means exactly what it always has (see the
step()-level `converged` semantics comment): not auto-validated. The dt=600
regression case showed a reverted stop CAN be accurate to <1%, but that was
checked against independent ground truth, not assumed from smoothness
alone. None of tonight's `newton_failure_reverted`/`max_outer` values
should be treated as final numbers without that same check.

**Also not tested tonight (deliberately out of scope, per the earlier
recommended-next-steps list)**: raising `max_outer` further to see if more
budget alone closes the gap to full stall convergence on the
still-plateaued steps; re-tuning `min_outer`/the stall window now that
trajectories are smooth; fixing the still-miscalibrated spike guard for
warm starts (left `spike_check=False` for all of tonight's runs);
wiring `t_relax` into the NI circuit closure (`ni_circuit.py`) -- Phase B's
actual end goal, still not started.

All artifacts preserved: `transient/runs/newton_overnight/summary.csv`
(aggregate), `transient/runs/newton_overnight/<tag>.json` (per-job full
step data), `transient/overnight_logs/<tag>.log` (full per-job stdout),
`transient/runs/newton_overnight/overnight_log.txt` (chronological
start/finish log). No stray mesh files, no changes outside `transient/`
(verified via `git status` before and after).

## 2026-08-04 (later the same day): the overnight run's "converged" values are WRONG — the outer loop drifts to a different fixed point, unrelated to inner-solve precision

Directly prompted by the obvious next question after the overnight summary:
**"the model is converging, but is it accurate?"** It is not, and this
invalidates every "converged"/"stall" result from the overnight run above.

**Ground truth, established first:** ran the UNMODIFIED, validated Picard
path (`ta_solve.solve_ta_at_current()`) at the actual operating point used
all night, I=196A (`params.I_design`), single implicit dt=600s step from
ZFC. Converged cleanly (k=80, the project's standard cost), on-axis
SCIF=+641.26 mT. Field diagnostics at this state (mean over coil cells):
`|B|`=4.13T (11.8% of cells >8T, matching the champion's long-documented
`clip_frac`≈0.118), `J/Jc` mean=0.586, 26.0% over-critical -- this
reproduces `validation/loss_sanity_check.py`'s independently-recorded
numbers (`j/jc mean 0.59, 26% over-critical`) essentially exactly, so the
ground truth itself is solid, not a fluke of this run.

**Newton-hybrid at the SAME discretization (single dt=600s step,
t_relax=0.15): converged=True, stop_reason=stall, SCIF=+540.89 mT -- a
clean formal convergence this time, no revert needed at all.** Looked like
a genuine win. **15.65% off from Picard. That is the first sign something
is wrong, not a rounding difference.**

**Field-level comparison (`accuracy_diagnose.py`) localizes it precisely.**
Per-layer J/Jc mean, Picard vs Newton-hybrid-at-its-"stall":

| layer | turns | Picard | Newton | 
|---|---|---|---|
| 0 | 382 | 0.204 | 0.196 |
| 1 | 382 | 0.349 | 0.339 |
| 2 | 478 | 0.653 | 0.592 |
| 3 | 478 | 0.919 | 0.787 |
| **4** | **3** | **0.949** | **0.293** |
| **5** | **3** | **0.919** | **0.280** |

Layers 4/5 -- the 3-turn vestigial pancakes this project's history already
flags repeatedly as numerically fragile -- are off by ~3x. Layers 0-3 are
off by a smaller but still real 4-14%.

**Root-cause investigation, in order, each a real test not a guess:**

1. **Hypothesis: "false stall," t_relax's damping shrinks the observable
   change enough to fool the 6-iteration stall window while the state is
   still far from its fixed point.** Tested directly
   (`thin_layer_trend.py`): called `step()` in repeated 30-iteration
   chunks (never letting the internal stall check fire early:
   `stall_tol=1e-9`), tracking per-layer J/Jc and SCIF every chunk out to
   ~390 total outer iterations. **Confirmed, dramatically.** SCIF did NOT
   plateau near 541 mT -- it kept decaying smoothly (544 -> 394 -> 290 ->
   213 -> 155 -> 111 -> 78 -> 53 -> 34 -> 20 -> 10 -> 2 -> **-3.9 mT by
   iteration 390**), a strikingly clean geometric decay (ratio ~0.75 per
   30-iteration chunk, holding for 8+ consecutive chunks). Three
   independent Aitken Delta-squared extrapolations from non-overlapping
   windows of this sequence all agreed: true asymptote approx **-24 mT**
   (-24.75, -24.99, -23.29 mT). **The "converged, stall" result at 540.89
   mT was a false stall** -- t_relax's per-iteration damping shrinks the
   OBSERVED step size for reasons unrelated to proximity to the true fixed
   point, so the stall criterion is not relaxation-invariant and triggers
   far too early once t_relax is small. Layers 4/5's own J/Jc had ALREADY
   visibly plateaued by iteration ~90 (essentially flat at 0.27/0.26 the
   entire remaining 300 iterations) while the aggregate SCIF kept moving
   for another 300 iterations -- meaning the per-layer mean is a poor
   proxy for whether the SCIF-relevant (near-cancelling, spatially
   resolved) state has actually settled, the same "summary observable
   doesn't reflect true convergence status" lesson this project's history
   already teaches for the base Picard scheme, just inverted (there the
   observable freezes while the state wanders; here the simple per-layer
   statistic froze while the spatially-resolved SCIF kept moving).

2. **Hypothesis: the true fixed point (~-24 mT) is itself WRONG, caused by
   accumulated inner-Newton-solve imprecision** -- `DEFAULT_SNES_OPTIONS`
   deliberately loosened `snes_rtol` 1e-8->1e-6 (see the module docstring)
   to avoid spurious line-search failures near convergence; a small
   systematic (not randomly-cancelling) per-iteration under-convergence
   COULD compound over hundreds of outer iterations into exactly this kind
   of slow directional drift. **Tested directly and REFUTED**
   (`tight_tol_trend.py`): reran the identical chunked trend with
   `snes_rtol=snes_atol=1e-12`, `snes_stol=1e-13` (tighter than the
   ORIGINAL pre-loosening values). Chunks 0, 1, and 2 matched the loose-
   tolerance run to 3-4 decimal places on BOTH the SCIF (543.96 vs 543.94,
   393.51 vs 393.53, 290.04 vs 290.01 mT) and every per-layer J/Jc value.
   **Tightening the inner solve by six orders of magnitude changed
   nothing.** The drift is not an inner-solve precision artifact.

**Conclusion: the drift toward a wrong fixed point is a genuine,
deterministic property of the OUTER loop's iteration scheme itself** (the
Jc(B)/n(B) Picard-lag + `t_relax` T-damping combination), not a tolerance
or under-iteration issue. Plausible mechanism, not yet confirmed: Picard's
own scheme damps the FULL rho(J,B) (a function of both J and B, evaluated
with the CURRENT J) directly every iteration; the Newton-hybrid instead
damps only Jc(B)/n(B) (functions of B alone) while letting Newton solve
the J-dependence of rho EXACTLY and SELF-CONSISTENTLY given that frozen
snapshot -- these are structurally different fixed-point iterations, and
given the extreme nonlinearity (n~13-27), the self-consistent "solve T
exactly given frozen Jc" subproblem may not have a unique solution, so
repeated exact re-solves along a slowly-drifting Jc/n path can walk onto a
different branch than Picard's own combined-damping path would reach.
**Not yet verified against a hand-derived mechanism -- this is the
leading hypothesis, not a confirmed root cause.**

**Practical consequence: EVERY "converged"/"stall" result in the overnight
run above (2026-08-04, "zero crashes across 74 step-solves") is
UNVALIDATED, not just under-tested.** The overnight run's real, still-true
finding is narrower than first reported: `t_relax` reliably prevents the
short-term (1-2 iteration) crash/revert this whole investigation set out
to fix. It does NOT reliably produce an accurate converged answer, and a
"stall" stop_reason no longer means "trustworthy" the way it did for the
single dt=600 regression case found earlier the same day (which WAS
independently checked against Picard and agreed to <1% -- that check
remains valid; what changed is that "stall" fired correctly there only
because that specific run happened to still be close to Picard's basin
when it triggered, not because the stall criterion is reliable in
general).

**Open, in priority order:**
1. Fix the stall/convergence check so it is not fooled by `t_relax` --
   candidates: track per-layer state (not just the aggregate SCIF) for
   stall; require the window to shrink relative to `t_relax` itself (a
   smaller `t_relax` needs a proportionally larger min_outer/window); or
   switch to a relaxation-invariant criterion (e.g. the residual of the
   FULL nonlinear system, not the observable).
2. Confirm or refute the "structurally different fixed-point iteration"
   hypothesis above -- e.g. check whether Picard's OWN scheme, run from
   the SAME state Newton drifted to, moves back toward 641 mT (would
   confirm Newton found a spurious branch) or also drifts toward -24 mT
   (would mean -24 mT is a second genuine physical fixed point, and the
   real question becomes which one the physical ramp process actually
   reaches).
3. Do NOT trust any multi-step march result, γ-relax value, or dt
   generalization claim from the overnight run until (1) is fixed and the
   whole sweep is re-validated against Picard ground truth at a spot-check
   of currents, not just I=196A.

Run: `<env>/bin/python3 transient/validation/accuracy_check_I196.py`
(the regression check that first caught the 15.65% gap),
`accuracy_diagnose.py` (per-layer field comparison),
`thin_layer_trend.py` (the chunked long-run trend + Aitken extrapolation),
`tight_tol_trend.py` (the tolerance-tightening refutation).

## 2026-08-04 (still later): DECISIVE — 641.26 mT is the ONE true answer; the Newton-hybrid's outer loop is genuinely unstable, not just slow or under-iterated

Direct test of open item 2 above. `transient/validation/picard_from_newton_state.py`:
drifted the Newton-hybrid (t_relax=0.15) to a clearly-wrong state (~240
outer iters, SCIF=+52.92 mT, far from Picard's 641.26 mT), then
TRANSPLANTED that exact T field into a fresh Picard `ta` (same mesh),
re-seeded A/B/rho consistently from it (verified: seeded Picard SCIF =
+52.89 mT, matching the source state to 0.03 mT -- the transplant is
correct, not an artifact of re-seeding from scratch), and ran
`ta_transient._picard_phase` -- the SAME unmodified, validated Picard
iteration -- from there.

**Picard converged cleanly in 44 iterations to SCIF = +641.25 mT** --
matching the from-ZFC ground truth (641.26 mT) to 0.01 mT, from a starting
point over 10x further from that answer than where it started.

**This is decisive on two points at once:**
1. **641.26 mT is the ONE true physical fixed point, not one of several.**
   The -24 mT the Newton-hybrid was drifting toward is not a legitimate
   alternate solution -- there is no genuine non-uniqueness here.
2. **The Newton-hybrid's outer loop is not merely slow or under-iterated
   -- it is locally UNSTABLE near the true solution.** It started at
   SCIF=+543.92 mT after only 30 iterations (already reasonably close to
   641.26 mT) and moved MONOTONICALLY FURTHER away for the next 200+
   iterations, on a clean geometric trend, while Picard -- given the
   EXACT SAME physical state 240 iterations further from the truth --
   immediately and robustly corrected course. The two schemes are solving
   the identical PDE; the difference is entirely in which quantity gets
   damped between outer iterations: Picard damps the FULL rho(J,B)
   (a function of both current AND field, evaluated at the current
   iterate) every step; the Newton-hybrid damps only Jc(B)/n(B)
   (field-only) while letting Newton resolve the J-dependence of rho
   EXACTLY and self-consistently each iteration. That structural
   difference is the leading suspect for the instability -- an "exact
   inner solve, lagged coefficients" scheme can have a locally repelling
   fixed point even when the underlying PDE's solution is unique and the
   naive Picard equivalent is stable, especially with this problem's
   extreme nonlinearity (n up to 27).

**This reframes the open items from the previous entries.** A better
stall/convergence DETECTOR (item 1) cannot fix a trajectory that is
genuinely diverging from the correct answer -- extrapolating a divergent
sequence just gets you a precise wrong number faster. The real fix target
is the outer loop's STABILITY, not its convergence *detection*.

**Follow-up, concluded: smaller `t_relax` does NOT stabilize it.**
`small_trelax_trend.py` reran the identical chunked march at t_relax=0.05
(3x stronger damping than 0.15). Result: still drifts away from 641.26 mT,
at a COMPARABLE rate to t_relax=0.15 -- chunk-over-chunk drops of
~166/~110/~80 mT at t_relax=0.05 vs ~150/~104/~77 mT at t_relax=0.15 over
the first three chunks (0.05's trajectory starts closer, 573.79 mT vs
543.94 mT at chunk 0, simply because a smaller relax factor makes LESS
progress in the first 30 iterations overall -- but the DIRECTION and
per-chunk MAGNITUDE of drift once underway are not meaningfully smaller).
**A 3x reduction in damping strength barely dented the drift rate** --
inconsistent with a simple "not enough damping of the right kind"
explanation, and pointing instead at a genuinely strong instability that
naive t_relax tuning cannot practically reach into stable territory
(reaching a dramatically smaller t_relax would also make forward progress
on genuinely well-behaved iterations impractically slow, defeating the
original point of the reformulation).

**CONCLUSION FOR THIS INVESTIGATION: t_relax-based T-damping is not a
viable fix for the outer loop's instability.** The instability is
structural: damping only Jc(B)/n(B) while resolving rho(J)'s
J-dependence exactly via Newton, each outer iteration, is not equivalent
to (and is evidently far less stable than) Picard's own scheme of damping
the FULL rho(J,B) together. The fact that Newton's PER-LAYER inner solve
is individually exact (and insensitive to tolerance, per
`tight_tol_trend.py`) does not help -- the instability lives entirely in
how the outer loop composes those exact solves over many iterations.

**Recommended next direction, not yet attempted:** damp an effective
rho-like quantity BETWEEN outer iterations the way Picard does, instead of
damping Jc/n and trusting Newton's exact J-resolution. Concretely: after
each layer's Newton solve, compute the resulting rho(J) it implies, blend
THAT with the previous iteration's rho (Picard's own relaxation target),
and feed the BLENDED rho back into the next Newton solve as an additional
frozen coefficient multiplier -- closer in spirit to Picard's actual
damping mechanism while still using Newton for the per-iteration linear
algebra. This has NOT been implemented or tested. A cheaper alternative
worth trying first: skip the Newton/quasi-Newton hybrid's outer loop
entirely for now and use Picard for the outer Jc/n and rho evolution,
reserving Newton only as a candidate for accelerating the (currently
linear, damped) inner T-solve within Picard's own iteration structure --
a smaller, more surgical change than the current from-scratch outer loop.

**Status of the whole Newton-Krylov investigation:** the original
motivating problem (Picard's own convergence failure at short dt, e.g.
dt=100s single-step) is STILL only solved by the now-invalidated Newton-
hybrid + t_relax combination -- i.e., effectively still UNSOLVED. The
core Newton reformulation's PER-LAYER exactness remains a validated,
useful building block (confirmed via `tight_tol_trend.py` and the
original dt=600 regression), but the OUTER LOOP built around it needs a
different damping structure before any of its numbers, single-step or
multi-step, can be trusted. Every result in the 2026-08-05 entries above
this one (the overnight validation, the t_relax fix, all dt/schedule
generalization claims) should be treated as SUPERSEDED, not as a
still-valid partial result -- they were built on the same flawed outer
loop this entry's investigation traces to its root cause.

Run: `<env>/bin/python3 transient/validation/picard_from_newton_state.py`,
`small_trelax_trend.py`.

## 2026-08-04 (still later): a genuine, working fix at dt=600 -- newton_ta.hybrid_step(). Does NOT fix the original dt=100 problem.

Direct response to "keep working on it until the problems are solved."
Replaced `newton_ta.step()`/`march()` (the disproven `t_relax` scheme --
kept in the module, unused by anything new, as a documented dead end) with
`hybrid_step()`/`hybrid_march()`: a Newton-INFORMED Picard hybrid, not a
Newton-dominant one. Built through three iterations, each tested against
the I=196A/dt=600s Picard ground truth (641.26-641.31 mT across repeats)
before moving on -- do not skip this discipline for any future change here.

**Design, final form:** per outer iteration, (1) Picard's own LINEAR
per-layer solve (`ta["prob_T_layers"]`, unmodified, using `rho_fn` exactly
as the previous iteration left it) plus the two-phase alpha T-relaxation
-- this is the ONLY thing that advances the persisted state; (2) AFTER
that, run Newton's exact per-layer solve informationally (never written to
the persisted T -- snapshotted and restored), to get a more accurate J
estimate than Picard's own linear solve would give; (3) blend
`J_for_rho = (1-newton_blend)*J_picard + newton_blend*J_informed` and feed
THAT into `ta_solve._update_rho()` (unmodified), which is what the NEXT
iteration's step 1 will use.

**Three bugs found and fixed en route, each isolated by a targeted test
before moving to the next, not guessed at:**
1. **Lag bug (first version).** Computing rho from Newton's J and using it
   in the SAME iteration's T-solve (rather than the next one) overshot to
   +910 mT, crossed back down through the true 641 mT, and kept drifting
   through 240+ mT before being killed. Root cause: Picard's own iteration
   has a natural ONE-ITERATION LAG (rho used to solve iteration k's T was
   computed from iteration k-1's state) that turns out to be load-bearing
   for stability, not incidental. Fixed by moving the Newton-informing pass
   to run AFTER the Picard update, so its rho feeds only the FOLLOWING
   iteration.
2. **Jc/n cross-lag (second version).** With the lag fixed, the trajectory
   stayed close to 641 mT far longer but still drifted, stabilizing at a
   WRONG value (~545 mT, ~15% low). Hypothesis: Newton's own `Jc_fn`/`n_fn`
   carried their OWN separate linear relaxation memory (`relax=None` ->
   `ta_rho_relax`=0.5), running in parallel with `rho_fn`'s log-space
   relaxation -- two independently-lagged coefficient tracks able to drift
   apart. Fixed by making Newton's Jc/n memoryless (`relax=1.0`, full
   overwrite from the current B every iteration -- they are ONLY an input
   to the advisory solve, so no reason to lag them separately). Tested:
   nearly IDENTICAL trajectory to before -- this hypothesis was WRONG, or
   at least not the dominant effect; kept the fix (still principled, still
   correct) but it did not by itself solve the bias.
3. **Undamped-J bug (root cause, third version).** Picard's OWN
   rho-informing J is computed from its DAMPED T (inherits the alpha
   relaxation) -- it is itself an already-damped quantity. Newton's
   J_informed is a fully undamped, exact resolve -- a stronger
   perturbation per iteration than Picard's careful T/rho co-damping
   balance assumes, even with the lag and Jc/n-memory issues fixed. Fixed
   by blending J_informed with Picard's own (already-damped) J_coil via
   `newton_blend` rather than using J_informed raw.

**Result, swept 0.5/0.3/0.15, single dt=600s step, I=196A, cold start, ALL
THREE reached a genuine formal stall (not a false one, not max_outer):**

| newton_blend | SCIF | diff vs 641.29 mT ground truth | n_outer to stall |
|---|---|---|---|
| 0.50 | 628.58 mT | 1.98% | 94 |
| 0.30 | 659.02 mT | 2.77% | 13 |
| **0.15 (now the default)** | **648.69 mT** | **1.16%** | **36** |

This is the first scheme in this whole investigation that BOTH converges
genuinely AND lands close to ground truth (previous best, `t_relax`, was
stable-looking but 15.65% wrong on the SAME test). `newton_blend=0.15` is
now the default in both `hybrid_step()` and `hybrid_march()`.

**THE ORIGINAL PROBLEM IS STILL NOT SOLVED.** Tested `hybrid_step()` at
the actual motivating hard case -- dt=100s, I=32.667 A, cold start (the
case Picard itself never converged even at 1000+ iterations, "SCIF
wandered chaotically, std~100 mT, no decaying trend"). Ran 100+ outer
iterations: **SCIF ranged from -146 to +557 mT with no decaying trend at
all**, `|dB|` staying large (50-180) throughout, never settling --
qualitatively IDENTICAL to Picard's own documented failure mode at this
dt. This is not surprising in hindsight: `newton_blend=0.15` means the
scheme is still ~85% Picard by construction (Picard's own linear solve is
the ONLY thing that advances the state; Newton only nudges what rho_fn
targets). Whatever makes Picard's own iteration chaotic at short dt is
apparently NOT primarily about the coefficient-freezing approximation
this hybrid improves -- it survives essentially unchanged.

**Honest status of the whole investigation:** the hybrid is a genuine,
validated improvement over the `t_relax` scheme -- it actually works, at
the one operating point (dt=600) this project has ever validated anything
against. But the motivating question that started this entire session
("why can't Picard converge at the short dt a real multi-step ramp
needs") remains open. `hybrid_step()`/`hybrid_march()` should NOT be
assumed to fix multi-step/short-dt schedules just because they fix the
single dt=600 accuracy problem -- that has been tested and disproven for
dt=100 specifically, and by extension should not be assumed for any
dt significantly shorter than 600s without its own direct test.

**What remains untried, in rough order of promise:**
1. Sweep `newton_blend` at the dt=100 case itself (higher blend = more
   Newton influence = further from "still 85% Picard" -- untested whether
   this helps, hurts, or is irrelevant at short dt specifically).
2. A genuinely different mechanism for the short-dt regime specifically,
   e.g. the previously-flagged full Newton-Krylov reformulation with a
   real (possibly finite-difference) Jacobian for the Ic(B)/n(B) spline
   dependence, removing coefficient-freezing entirely rather than just
   damping it better.
3. Revisit the time-stepping schedule itself (per the much earlier
   "reconsider the time-stepping granularity" idea) -- a schedule that
   only ever takes sub-steps at dt values closer to 600s, rather than
   assuming any dt down to 25-100s must work directly.

Run: `<env>/bin/python3 transient/validation/hybrid_accuracy_check.py`
(dt=600 regression, the test that caught bugs 1 and 3),
`hybrid_blend_sweep.py` (the blend sweep above),
`hybrid_dt100_check.py` (the dt=100 test that shows this is NOT solved).

---


## 2026-08-05: fully monolithic block-Newton T-A -- tried, and it does NOT
## fix the transient convergence problem either (new architecture, same
## failure signature)

### The question that prompted this

Direct user question: given the transient (Phase B) work's convergence
history -- Picard fails at short dt, `newton_ta.py`'s per-layer Newton
fixes the dominant power-law nonlinearity but its outer Gauss-Seidel loop
(solve each layer's T to convergence with A frozen, then solve A with
every T frozen, repeat) still fails or drifts to wrong answers away from
the one validated dt=600s point -- would switching to an H-formulation
help convergence, or is it better to keep iterating on T-A?

Answer given first, then tested rather than left as opinion: H-formulation
was judged UNLIKELY to help, because the documented instability traces to
the OUTER loop that keeps the measured, non-UFL-differentiable
Ic(B,theta)/n(B,theta) splines consistent with the field state -- an
H-formulation at this turn density still needs the same homogenised bulk
description reading from the same splines, so it inherits the identical
non-differentiable-coefficient problem. The more promising, cheaper,
UNTRIED lever flagged in CLAUDE.md's Newton-Krylov section was to fold
Jc(B)/n(B) into a single monolithic residual, removing the outer
Picard-lag entirely. User asked to actually try it rather than leave it as
a hypothesis.

### `transient/monolithic_ta.py` -- what was built

A genuinely different architecture from `newton_ta.py`, not a variant of
it: ALL SIX layer-T unknowns and the shared A unknown are solved as ONE
PETSc SNES BLOCK system (`dolfinx.fem.petsc.NonlinearProblem([F_T0, ...,
F_T5, F_A], [T_0, ..., T_5, A_h], kind="mpi")`), so the Jacobian includes
the TRUE cross-coupling terms dF_A/dT_i and dF_Ti/dA every single Newton
step -- something neither Picard (T then A, heavily damped) nor
`newton_ta.py` (T_i to convergence with A frozen, then A, Gauss-Seidel)
ever captures directly. Feasibility of dolfinx 0.11's block NonlinearProblem
API (`kind="mpi"`, list-of-forms/list-of-Functions) was confirmed on a toy
2-unknown coupled nonlinear system before touching the real physics --
converged to the analytically known fixed point exactly.

Jc(B)/n(B) still cannot be symbolically differentiated (same reason as
`newton_ta.py`: measured-CSV scipy splines, not UFL-expressible) -- frozen
per-layer DG0 coefficients, same as before. What's new is WHEN they refresh:
every Newton step, not once per outer Gauss-Seidel sweep, via
`snes.setTolerances(max_it=1)` + a Python loop that reads the just-updated
B field and rewrites the coefficients between calls (confirmed on the same
toy system: repeated max_it=1 solves warm-start correctly and converge to
the exact fixed point in ~8 calls; `getConvergedReason()==-5`,
SNES_DIVERGED_MAX_IT, on every call before the true fixed point is EXPECTED
under this pattern, not a failure -- verified directly, not assumed).

### Result: diverges undamped, and drifts to a DIFFERENT wrong answer at every damping level tried

Regression case: single implicit dt=600s step, I=params.I_design (196A),
cold start -- the one operating point this project has ever validated a
T-A number against (`transient/validation/monolithic_accuracy_check.py`,
mirroring `accuracy_check_I196.py`'s pattern exactly). Ground truth on the
CURRENT champion geometry: Picard, k=90, **on-axis SCIF = +641.17 mT**
(matches the historical ~641.26 mT reference to <0.02%, confirming the
setup is consistent with prior sessions' work).

The raw (undamped) block-Newton step overshot to **+11589 mT** after ONE
step (18x the truth) and failed outright on the next. Line search
formally ACCEPTED that first step (reduced the coupled system's raw
residual norm) -- but the residual norm has no visibility into SCIF, a
near-cancelling small difference of much larger current densities, so "the
line search accepted it" and "this step is physically reasonable" are
different claims here, the same lesson this project's history keeps
re-learning in new forms.

Added `step_relax` (blends the accepted joint (T, A) Newton step with the
pre-step state, since T and A are now solved simultaneously rather than
sequentially -- a different hypothesis from `newton_ta.py`'s disproven
`t_relax`, which only had T to damp because Gauss-Seidel let it drift out
of sync with A in the first place). Swept on the identical case
(`transient/validation/monolithic_step_relax_sweep.py`, one shared mesh,
`debug=True` printing per-layer j/jc every step):

| step_relax | outcome | SCIF vs +641.17 mT truth |
|---|---|---|
| 1.0 (undamped) | diverged within 2 outer iterations | +11589 -> hard failure |
| 0.3 | smooth asymptotic climb, still rising at failure (k=17) | ~+6800 mT and climbing |
| 0.1 | clean asymptotic plateau (k=60) | **+3669 mT** |
| 0.03 | clean asymptotic plateau (k=45) | **+2012 mT** |

Every one of the damped runs shows the SAME shape: a long, smooth,
geometrically-decaying climb to a plateau, then eventually a hard
`SNES_DIVERGED_LINE_SEARCH` failure right at that plateau (the identical
"line search gets numerically finicky once the residual is already small"
signature `newton_ta.py`'s `newton_solve_layer` docstring already
documents for per-layer solves, now showing up at the OUTER, joint-step
level instead).

**The decisive point is not that these runs failed to reach 641 mT -- it's
that the three damped runs converged CLEANLY to three DIFFERENT wrong
values, and that value is a monotonic function of `step_relax`.** A
genuine fixed point of the underlying PDE cannot depend on the pseudo-time
damping used to reach it. This is therefore proof the plateau is a
numerical artifact of the coefficient-refresh-every-step scheme, not a
slow-but-correct answer waiting for more iterations or a smaller step
size -- turning `step_relax` down further would not be expected to find
641 mT, only a smaller spurious plateau (consistent with the trend
6800 -> 3669 -> 2012 as damping strengthens). This is the SAME failure
signature already on record for `newton_ta.py`'s disproven `t_relax`
scheme (which converged cleanly to -24 mT against the identical 641 mT
truth) -- now confirmed in a SECOND, structurally different "more
monolithic" architecture, which rules out "the Gauss-Seidel T/A split was
the problem" as an explanation, since this scheme has no such split at
all.

### What this changes about the standing recommendation

**Revised understanding, weaker than "H-formulation won't help" and more
specific:** the instability is not really about which fields are treated
as unknowns, or how tightly/loosely/simultaneously they are coupled --
Picard (sequential, heavily damped), `newton_ta.py` (Gauss-Seidel,
per-layer-exact), and now a fully monolithic block Newton (simultaneous,
every-step coefficient refresh) have ALL been tried, spanning the entire
reasonable space of "how coupled should the solve be," and every one that
refreshes Jc(B)/n(B) faster or with less of Picard's OWN specific damping
structure than Picard itself uses lands on the same pathology: a scheme
that looks stable (or clearly converges) but is quietly wrong, by an
amount that depends on the scheme's own tuning parameters rather than on
the physics. The common thread across every failure in this whole Phase B
investigation (Picard's own short-dt wandering, `t_relax`'s clean-but-wrong
-24 mT, and now this) is a Newton-type method's large corrective step
interacting badly with an observable (SCIF) that is a near-cancellation
invisible to any residual norm the solver actually globalizes on. An
H-formulation reformulation would face the IDENTICAL stiff power-law
constitutive law and the IDENTICAL near-cancelling screening-current
signature (the physics, not the formulation, produces the cancellation),
so there is no longer a principled reason to expect it to sidestep this --
the recommendation to continue with T-A + the circuit closure rather than
reformulate stands, now on stronger evidence than the original reasoning
alone.

**What has NOT been tried and would be the next genuinely different
lever, if this is revisited:** a true pseudo-transient continuation /
homotopy on the coefficient freezing itself (gradually un-freezing Jc/n's
field-dependence via a continuation parameter from 0 to 1, rather than a
fixed-strength lag or a fixed-strength step-damping), or accepting that
SCIF from any short-dt/multi-step regime must be reported as a band
(informed by exactly this damping-dependent spread) rather than a point
value -- in the same spirit as this project's existing practice of
quoting bounds instead of single numbers for other near-degenerate/
resolution-limited quantities.

Run: `<env>/bin/python3 transient/validation/monolithic_accuracy_check.py`
(dt=600 regression, the test that caught the undamped divergence),
`monolithic_step_relax_sweep.py` (the damping sweep above).

---

## 2026-08-05 (continued): adaptive step-size marching -- works, but not the
## way "smaller steps are safer" predicts; the cold start is the hard part

### Why this was tried

Direct follow-up to the user's "what's different between our method and
standard practice" question. Two real gaps from standard T-A/H-formulation
transient practice were identified: (1) this project takes ONE giant
implicit step over the whole 600s ramp instead of many adaptively-sized
small ones, and (2) the material data (Ic(B,theta)/n(B,theta)) is a
measured, non-differentiable spline, forcing every solver into some form
of coefficient freezing instead of genuine exact Newton. User chose to
test (1) first, since it's cheaper and doesn't touch the material-model
question -- if small-enough steps alone stabilise the EXISTING, unmodified
Picard machinery, the bigger differentiable-surrogate rewrite may not be
needed.

### `transient/adaptive_march.py` -- what was built

A classic Newton-iteration-count step-size controller wrapped around
`ta_transient._picard_phase` (unmodified, the same validated per-step
Picard machinery every other production path uses), marching the
INSULATED base solver (no NI closure, `per_turn_bc=False`) from
zero-field-cooled through a linear ramp: propose a step, run
`_picard_phase` with a bounded iteration budget, accept-and-grow-dt if it
converged easily, accept-and-hold-dt if it converged but needed more
iterations, or revert-everything-and-shrink-dt if it didn't converge at
all. Reverting restores T (every layer), A, A_prev, AND the Picard
relaxation history array `ta["_rho_prev"]` -- omitting that last one would
leave a retry's log-space rho relaxation lagging against a state that no
longer exists.

### BUG (caught before trusting any number): `dt_const` was never actually set

First two runs (a 20-step march at dt_init=30s, then a refinement at
dt_init=15s) both "succeeded" cleanly -- zero rejects, smooth SCIF
trajectories -- but landed on wildly different, monotonically shrinking
answers as steps got finer (641 mT single-step reference -> 190 mT @
dt=30 -> 70 mT @ dt=15), which was almost read as "the single-step ground
truth was never a resolved simulation of the ramp and may have been wrong
all along." That conclusion would have been WRONG, and was caught before
being stated as a finding: `_picard_phase(ta, ..., dt, ...)` takes `dt`
only to pass through to its NI-closure `closure` callback -- it does NOT
set the FEM's `ta["dt_const"].value` itself (`ta_transient.step()` does
that, one explicit line, immediately before calling `_picard_phase`).
`adaptive_march.py`'s first version never did this, so every single step
in both runs actually solved with `dt_const` stuck at whatever
`_build_problems()` initialised it to (`params.ramp_duration` = 600s),
regardless of the intended `dt_try` -- the ADAPTIVE BOOKKEEPING (advancing
`t`, computing `I_next`) was correct, but the PHYSICS SOLVED each step was
not. The tell that caught it: iteration count per step was suspiciously
constant (~46) regardless of the current level, which isn't what varying
`dt` should produce. Fixed with one line
(`ta["dt_const"].value = float(dt_try)` immediately before the
`_picard_phase` call). **Lesson: a smooth-looking, zero-reject trajectory
is not evidence of correctness by itself -- always check WHY a number
moved the way it did, especially when the direction of a trend
contradicts prior physical reasoning (finer discretisation is not
supposed to change an answer by 3-9x).**

### After the fix: a real, reproducible, and non-obvious pattern

Single-step reference (unchanged, from earlier this session):
**+641.17 mT** (Picard, k=90, dt=600s, cold start).

| dt_init | own process? | outcome |
|---|---|---|
| 60s | first sweep (3 values, 1 process) | 11 steps, 2 mid-ramp rejects (both self-corrected: shrink to 30s once, regrow to 60s), completed the WHOLE ramp, **final SCIF = +828.50 mT** |
| 30s | first sweep (1 process) | **FAILED outright** at t=0 (ZFC) -- every shrink down to the dt=1.0s floor still didn't converge |
| 15s | first sweep (1 process) | did NOT fail -- found dt=3.75s workable at t=0, then struggled again on the next step |

**A confound was caught before trusting the dt=30/dt=15 comparison**: all
three values were run back-to-back in ONE process, each calling
`ta_solve.setup_ta_problem()` (and its identically-prefixed PETSc
solver objects) fresh but in the SAME process. PETSc's options database is
process-global, and this project's own established convention -- for
exactly this class of risk -- is to run independent configurations as
separate OS processes (see the earlier "concurrent cmaes_search.py
processes must not share output paths" lesson, a different but related
same-process-contamination issue). The tell: at the LITERAL SAME nominal
configuration (dt=3.75s, t=0, same target current, same seed), the
dt_init=30 chain (which passed through dt=3.75 as its 4th shrink attempt)
FAILED there, while the dt_init=15 chain (which reached the SAME dt=3.75
as its 3rd shrink attempt) SUCCEEDED. Re-derivation of the state-reset
logic did not find an obvious bug (every retry of a `first=True` step
fully re-seeds from scratch via `_seed_cold`+`_update_rho`, so prior
failed attempts should not leak into the next one) -- rather than assume
either explanation, this was re-tested with each dt_init in its OWN
process (`transient/validation/adaptive_march_single.py`).

**Isolated-process result: the dt_init=15 failure REPRODUCED cleanly (own
process, own mesh, `FAILED: step at t=0.0s did not converge even at the
dt floor (1.0s) after 5 rejects`) -- confirming the earlier "success at
3.75s" was the same-process artifact, not real.** dt_init=30 in its own
process eventually found ONE workable dt (1.875s) for the first step
after 4 rejects, but the resulting SCIF was **-246.58 mT** -- a sign flip
from every other early-ramp reading, itself a red flag of a marginal,
possibly non-physical state rather than a clean answer -- and then hit
the SAME severe difficulty on the very next step (95+ CPU-minutes,
essentially no further ramp progress). That run was killed rather than
let it grind further once the qualitative picture was clear.

### The actual finding, stated carefully

**The base Picard solver gets HARDER to converge as the step size shrinks
specifically in the earliest, lowest-current part of the ramp (near zero
field, near zero current) -- the opposite of the general "smaller steps
are safer" intuition that motivated trying adaptive stepping in the first
place, and the opposite of what happened at dt_init=60, which sailed
through the ENTIRE ramp including the mid-ramp region this project has
separately documented as dt-sensitive (400s/200s failing while 600s/300s/
150s worked).** This is now well-supported (reproduced in isolated
processes, at two different dt_init values, both failing or barely
succeeding at exactly the same point) but the MECHANISM is not confirmed,
only hypothesised:
- The T-equation's forcing term scales as `1/dt * curl(A_h - A_prev)`. In
  the earliest ramp, the true physical EMF signal is small (I is barely
  moving), so a small `dt` inflates `1/dt` while the actual signal being
  amplified is itself small and plausibly noisy/poorly-resolved relative
  to whatever numerical error is present -- amplifying relative error
  rather than physical signal.
- The uniform-J `_seed_cold` bootstrap (used to seed A before the very
  first Picard iteration) may be a much worse RELATIVE approximation of
  the true state for a tiny target current than for a moderate one, even
  though the absolute seed magnitude scales with the target current in
  both cases.

Neither hypothesis has been tested directly (e.g. by instrumenting the
per-cell j/jc distribution or the seed-vs-converged-state error at
several current levels) -- flagged as the natural next diagnostic, not
done here.

### What this means practically

Adaptive stepping is validated as a REAL, useful mechanism for the
MID-ramp dt-sensitivity this project has documented before (dt_init=60's
2 self-correcting rejects around t=120s/t=270s are exactly that pattern,
handled automatically). It is NOT validated as a blanket "use small steps
everywhere" fix, and naively starting a march with a small first step
actively makes things WORSE, not better, at the one place a real ramp
schedule cannot avoid: the very start. **The practical recipe supported
by today's evidence: take one deliberately large first step (dt on the
order of 60s, matched to how far this specific solver needs to get from
I=0 before its usual dt-sensitivity dynamics apply) to clear the
zero-field-cooled regime, then hand off to normal grow/shrink adaptive
control for the rest of the ramp** -- not a uniform small-step schedule,
and not the single dt=600s whole-ramp step either.

**Not yet done**: confirming dt_init=60 is itself the right size (was
picked as "clearly bigger than the known-bad 200-400s zone's neighbours"
by inspection, not derived); testing whether an even larger first step
(say 90-120s) does better or starts running into the SAME mid-ramp
dt-sensitivity earlier; running the NI circuit closure on top of a
validated adaptive schedule now that the base insulated case has one;
diagnosing the actual mechanism behind the cold-start pathology rather
than the two hypotheses above.

Run: `<env>/bin/python3 transient/validation/adaptive_march_check.py`
(the dt_init=30 run that first surfaced the dt_const bug),
`transient/validation/adaptive_march_single.py <dt_init_seconds>` (run
as SEPARATE processes -- this is now the correct way to compare different
dt_init values, not a same-process sweep).

---

## 2026-08-05 (continued further): the "large first step" recipe RETRACTED --
## it did not reproduce, and the real cause is likely mesh sensitivity, not
## a fixable bug

### What was attempted

Direct follow-up on the "adaptive marching" entry above's two flagged
open items: (1) stress-test whether dt_init=60s generalises (try 90s,
120s, and re-verify 60s itself in true isolation -- it had only ever been
tested inside the 3-value same-process sweep), and (2) decouple `dt` and
the target current `I` to isolate the cold-start mechanism (adaptive
marching's linear ramp always changes both together at a fixed rate).

### The decoupling diagnostic (`transient/validation/first_step_diagnostic.py`)
### -- a real result, genuinely isolated (unlike the sweep confound)

Four single first-steps from ZFC, each its OWN process (so the earlier
same-process PETSc-reuse concern does not apply -- each of these is a
single, first-ever `setup_ta_problem()` call in a fresh process, exactly
like the isolated single-value adaptive-march runs):

| dt | I | implied rate (I/dt) | result |
|---|---|---|---|
| 60s | 19.6A | 0.327 A/s (= the ramp's natural rate) | **converged, 56 iters** |
| 15s | 19.6A | 1.307 A/s (4x faster) | FAILED, capped at 150 |
| 60s | 4.9A | 0.082 A/s (4x slower) | FAILED, capped at 150 |
| 15s | 4.9A | 0.327 A/s (same rate as the success!) | FAILED, capped at 150 |

The last row rules out the cleanest candidate explanations: neither "small
dt is the problem" nor "small I is the problem" nor "matching the natural
ramp rate is what matters" survives -- (15s, 4.9A) has the IDENTICAL
implied rate as the only success, yet fails. Only the specific combination
of both a large absolute `dt` AND a large absolute `I` succeeded in this
one test. This result stands as a genuine, uncontaminated data point, but
see below for why it should not be over-trusted either.

### The stress test that RETRACTS the "large first step" recipe

`transient/validation/adaptive_march_single.py 60` -- a true single-value
isolated re-run of the exact configuration that gave the clean
"+828.50 mT, 11 steps, 2 self-correcting rejects" result documented in the
adaptive-marching entry above. **It did not reproduce.** Instead: REJECT
at t=0 (dt=60, CAP at 150 iters), REJECT again after shrinking to dt=30,
only accepted at dt=15 with SCIF **-550.29 mT** (a sign flip seen nowhere
else in this project except marginal/questionable states), then repeated
rejects and negative-SCIF accepts all the way through t=195s of the 600s
target before the run was killed (170+ CPU-minutes, clearly not heading
toward a clean completion). Every accepted SCIF value in this run was
negative; the original successful run's were all smoothly positive and
monotonically evolving. These are not the same trajectory in any
meaningful sense, despite being nominally the identical configuration.

**Why the earlier "same-process PETSc reuse" explanation does NOT cover
this:** that hypothesis was specifically about a SECOND or THIRD
`setup_ta_problem()` call in one process picking up state left behind by
an earlier one. `dt_init=60` was the FIRST value tried in the original
3-value sweep -- at the moment it ran, no prior `setup_ta_problem()` call
existed in that process, exactly like this isolated single-value rerun.
Both are "fresh first call" scenarios. So this specific discrepancy needs
a different explanation.

**Leading candidate, not confirmed:** each process builds its own mesh via
`build_mesh.build()`, and this project has repeatedly documented (the
n_layers=4 mesh-fragility episode, the day_search Phase-A mesh-noise
episode, the new-design mesh-convergence false alarm) that **gmsh mesh
generation is not perfectly reproducible across separate process
launches, even for byte-identical geometry.** Combined with this system's
own extensively-documented "near-degenerate marginal flux front,
chaotically wandering, red-spectrum" sensitivity, it is plausible that two
separate mesh draws of the IDENTICAL nominal geometry landed on opposite
sides of a convergence/non-convergence boundary for this specific
cold-start configuration. If true, this is a more fundamental and more
concerning finding than a fixable implementation bug: it would mean the
cold-start convergence behaviour of this solver is sensitive to WHICH
mesh realisation gets drawn, not just to solver tuning parameters -- not
tested directly here (would need e.g. re-running `dt_init=60` several more
times in isolation and checking whether outcomes cluster into "clean
success" and "chaotic failure" bins, or whether there is some other,
undiscovered cause).

### What this means for everything claimed in the adaptive-marching entry above

**The "practical recipe: one deliberately large first step, then normal
adaptive grow/shrink" conclusion is RETRACTED as stated.** It was drawn
from a single confounded run whose OTHER two values (dt_init=15/30) were
already known to be unreliable, and whose reference value (dt_init=60)
has now failed to reproduce in a genuinely clean re-test. **No adaptive
marching number produced today (+190 mT, +70 mT, +828.50 mT, or the
failed/negative-SCIF isolated dt_init=60 attempt) should be treated as a
validated transient result.** The ONLY thing from this whole line of
investigation that remains solidly validated, reproduced repeatedly
across this project's history without this kind of cross-run instability,
is the original single-implicit-step-over-the-whole-ramp Picard result
(+641.17 mT this session, matching the historical ~641.26 mT reference).

**What IS still probably true, with lower confidence than stated
earlier:** the decoupling diagnostic's clean-process 2x2 result is
internally consistent (one success, three failures, no confound in how
it was run) and is suggestive that both dt and I need to be large
together for this specific first-step scenario -- but given dt_init=60's
own failure to reproduce moments later, that 2x2 result itself should be
treated as ONE observation on ONE mesh draw, not a settled characterised
of the system, until repeated.

### Honest summary of the whole 2026-08-05 adaptive-marching investigation

1. A real implementation bug was found and fixed (`dt_const` never being
   set) -- this fix is correct and should be kept.
2. A real same-process PETSc-reuse confound was found and correctly
   identified as invalidating the dt_init=15 "success at 3.75s" result.
3. What was NOT caught in time: the REFERENCE case (dt_init=60) was never
   independently re-verified before being written up as a validated
   recipe in CLAUDE.md and this file. It has now been re-verified, and it
   failed to reproduce -- for a reason that is NOT the same confound
   already identified, and is most likely cross-process mesh sensitivity
   interacting with this system's known chaotic cold-start dynamics.
4. **Lesson, worth stating plainly for future work on this solver:**
   before writing up ANY multi-step transient result as a working
   "recipe," re-run the SPECIFIC reference case in complete isolation at
   least once, not just the disputed comparison cases -- a positive
   result is just as capable of being a lucky mesh draw as a negative one
   is of being a contaminated one, and this session checked the latter
   diligently while assuming the former was safe.

**Status: adaptive step-size marching, as investigated today, is
NEITHER a confirmed fix NOR a confirmed failure for the transient
convergence problem.** It remains a plausible direction (the mid-ramp
self-correction behaviour seen in the original dt_init=60 run, even if
that specific run's overall number is not trustworthy, is still a
believable local mechanism), but claiming it "works" requires a
repeatable demonstration this investigation did not achieve. The
short-dt/multi-step convergence problem should be treated as OPEN, at the
same status as before this whole 2026-08-05 exploration began, with the
addition of the `dt_const` bug fix (genuinely useful, keep it) and the
mesh-sensitivity concern (a new, real risk to design around in any future
attempt, e.g. by always running multiple mesh realisations before trusting
a convergence outcome).

Run: `<env>/bin/python3 transient/validation/first_step_diagnostic.py <dt> <I>`
(the decoupling diagnostic, genuinely isolated),
`transient/validation/adaptive_march_single.py 60` (repeat this several
times to test the mesh-sensitivity hypothesis directly -- not done here).

---

## 2026-08-05 (continued further still): mesh-sensitivity hypothesis
## TESTED DIRECTLY AND NOT SUPPORTED -- the dt_init=60 discrepancy remains
## unexplained

### The test

The entry above proposed cross-process gmsh mesh non-reproducibility as
the "leading candidate, not confirmed" explanation for why an isolated
re-run of `dt_init=60` failed to reproduce the original clean result.
That was an assumption carried over from this project's documented history
for OTHER designs (n_layers=4, day_search Phase-A), never directly tested
for THIS geometry. Tested directly now, two ways:

1. `transient/validation/mesh_reproducibility_check.py` -- two
   `build_mesh.build()` calls in ONE process. Byte-identical files, 1718
   nodes / 8136 cells both times, zero coordinate difference. This only
   tests within-process reproducibility, which was never actually in
   question (this project's own history already establishes that gmsh is
   reproducible within a single process -- see the n_layers=4 episode's
   "every repeat reused the same process's mesh generation state").
2. **The real test**: two independent `python3` process launches (not
   dolfinx/gmsh calls within one script), each building the identical
   geometry to its own file, then a plain `diff`. **Byte-identical.**

**Two separate mesh draws of this design's geometry, across two genuinely
separate OS processes, produced the exact same mesh.** The mesh-sensitivity
hypothesis is NOT supported by direct evidence for this case. This does
not prove gmsh is ALWAYS reproducible for this geometry (one repeat is
one data point, and this project's history shows OTHER geometries hitting
non-reproducibility only intermittently, near specific "meshing tipping
points" -- this design may simply not be near one), but it removes mesh
non-reproducibility as the leading explanation. **The `dt_init=60`
discrepancy documented in the entry above remains genuinely unexplained.**

### Other candidate explanations, and how to test each (not yet done)

1. **Multi-threaded BLAS/MUMPS floating-point non-associativity, amplified
   by this system's known chaotic/near-degenerate sensitivity.** Every
   solve in this investigation showed ~700-800% CPU from a single Python
   process (internal OpenMP/BLAS threading inside MUMPS's factorisation).
   Floating-point addition is not associative, so the exact summation
   order in a multi-threaded reduction can vary run to run depending on
   OS thread scheduling, even for bit-identical inputs -- normally an
   utterly negligible (~1e-15 relative) effect, but potentially enough to
   flip a marginal case in a system already documented to have cells
   sitting exactly at the j/jc=1 smoothed-floor transition, where small
   perturbations are known to matter (the same class of sensitivity
   documented for the base Picard scheme's own red-spectrum wandering).
   **Test:** force fully serial execution (`OMP_NUM_THREADS=1`,
   `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and MUMPS's own
   thread-count option if exposed via PETSc options) and repeat the
   identical `dt_init=60` isolated run 2-3 times. If forcing serial
   execution makes repeats agree (same outcome, ideally same trajectory),
   this is confirmed as at least a contributing mechanism.
2. **Python hash-seed randomisation affecting internal set/dict iteration
   order.** CPython randomises `hash()` per process by default (since
   3.3, unless `PYTHONHASHSEED` is fixed), which affects `set` iteration
   order (dict iteration order is insertion-order-stable since 3.7 and NOT
   affected, but any `set()` used anywhere in the dolfinx/gmsh/PETSc
   Python-level pipeline for e.g. building local-to-global DOF maps or
   partitioning could, in principle, produce a different DOF/cell
   ordering across two otherwise-identical processes -- not a different
   MESH, but a different ASSEMBLY ORDER, which again bottoms out in
   floating-point non-associativity). **Test:** set `PYTHONHASHSEED=0`
   (or any fixed value) for both process launches and repeat. Cheap,
   two-line change (an env var), no code modification needed. Should be
   tested alongside or independently of (1) to see which (if either)
   matters.
3. **A genuine, not-yet-found deterministic bug specific to warm/cold
   continuation logic inside `adaptive_march.py`** -- e.g. some piece of
   `ta` state not being reset identically between the two runs' `first`
   step handling. Lower prior than (1)/(2) given the isolated single-run
   script does nothing unusual on its first call (same code path both
   times, no branching on anything process-specific), but not ruled out.
   **Test:** would need step-by-step diffing of the two runs' intermediate
   states (T, A, rho arrays after the seed and after the first Picard
   iteration) rather than just comparing final outcomes -- more
   expensive, only worth doing if (1) and (2) both come back negative.

**Recommended order, if this is pursued further:** (1) and (2) are both
cheap (env vars only, no new code) and can be combined into one test
(fix both simultaneously, see if repeats now agree) before trying them
separately to isolate which one matters -- only fall back to (3) if a
fully-deterministic environment still fails to reproduce.

---

## 2026-08-05 (continued yet further): threading/hash-seed hypotheses
## TESTED AND REJECTED -- the non-determinism is real, its source is not

### The test

Ten repeats of `transient/validation/first_step_diagnostic.py 60 19.6`
(the cheap, single-step version of the exact configuration that produced
inconsistent outcomes throughout this investigation), each its own
process, its own mesh:

| batch | env | n_iters per rep | converged |
|---|---|---|---|
| baseline | default (threaded BLAS/MUMPS, random `PYTHONHASHSEED`) | 92, 64, 150, 150, 150 | **2/5** |
| forced-deterministic | `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `PYTHONHASHSEED=0` | 150, 150, 150, 150, 150 | **0/5** |

### Verdict

**Both hypotheses (multi-threaded floating-point non-associativity,
per-process hash-seed randomisation) are REJECTED as the explanation.**
If either were the driver, forcing determinism should have made the
outcome consistent (all-succeed or all-fail, matching each other) --
instead the "fixed" batch was WORSE than baseline (0/5 vs 2/5) and still
internally non-deterministic in the sense that different reps under the
IDENTICAL forced-deterministic settings could in principle have differed
in trajectory even though all five happened to fail outright here. The
baseline batch alone is already the decisive data point: two runs of the
literal same script, same command, same input, same machine, back to
back, gave 92 and 64 Picard iterations to converge, and the next three
gave outright non-convergence. That spread cannot be explained by
anything this test controlled for.

**The actual source of the run-to-run variation in this system's
cold-start convergence remains genuinely unidentified.** Ruled out so
far: same-process PETSc options-database reuse (confirmed real, but
doesn't explain single-process-first-call cases); cross-process gmsh mesh
non-reproducibility (tested directly, mesh files were byte-identical);
BLAS/MUMPS threading non-determinism; Python hash-seed randomisation.
What has NOT been tested: whether `_seed_cold`'s uniform-J seed or the
subsequent `_update_rho` call reads any wall-clock-dependent or
otherwise-uncontrolled state; whether this is genuine sensitive
dependence on floating-point-level initial conditions inherent to the
nonlinear Picard map itself (i.e. real mathematical chaos needing no
external source of variation at all -- two "identical" floating-point
computations on the same hardware are not guaranteed bit-identical
outputs even from the same input if ANY part of the call stack, including
libraries several layers below application code, makes a
non-deterministic choice not caught by the tests here); or something
entirely unexamined.

### Practical implication -- the most important takeaway of the whole day

**No single run of this solver's cold-start behaviour can be trusted as
representative, full stop.** The identical nominal configuration succeeds
roughly 20-40% of the time and fails the rest, for reasons not explained
by anything tested. Any future claim that a transient schedule or
solver modification "works" MUST be based on multiple repeated isolated
runs of the specific configuration, not one -- this was already the
lesson from the retracted "large first step" recipe, and today's testing
shows the underlying instability is deeper and more persistent than that
single incident suggested; it is a property of the system at this
operating point, not a one-off fluke that better bookkeeping fixes.

### Recommendation

Given the cost of today's investigation (multiple multi-hour and
multi-CPU-minute runs, several retracted conclusions) relative to what it
established, further root-cause chasing on the cold-start non-determinism
is a genuine scope decision, not a quick next step. The two remaining
untested candidates (uncontrolled state in the seed/BC-setting path;
genuine floating-point-chaos with no external cause) would need
substantially more careful instrumentation than an env-var A/B test to
distinguish -- e.g. bit-for-bit comparison of the assembled matrices
and RHS vectors between a successful and failed run at iteration 1,
before any Picard nonlinearity has had a chance to amplify anything.
Whether that investment is worth making now, versus treating the
multi-step transient problem as genuinely open and moving to other work,
is a call for project direction, not something to keep pursuing by default
momentum.

Run: `<env>/bin/python3 transient/validation/first_step_diagnostic.py 60 19.6`
(repeat several times to see the non-determinism directly -- do not expect
a consistent answer).

---

## 2026-08-05/06 (continued across two more sessions, full detail NOT
## folded into this file -- see the pointers below): the bit-for-bit
## diagnostic recommended just above WAS eventually done, and it led to a
## real fix

This file's own last recommendation (immediately above) -- bit-for-bit
comparison of assembled matrices/RHS between a successful and failed
run -- was picked up in later sessions, but the full narrative from here
onward lives in `transient/validation/monolithic_diff_investigation_
2026-08-05.md` (Parts 6-9: real PCFIELDSPLIT, a direct-assembly bypass
for the SNES introspection crash, and localising the divergence's origin
to the Picard bootstrap seed phase) and `transient/validation/
nondeterminism_investigation_2026-08-05.md`'s later dated continuations
(through 2026-08-06), rather than being transcribed here -- read those
files directly for the complete, dated, self-correcting arc (each
contains its own retractions, same as everything else in this project's
history).

**The short version, since it changes this file's own standing
conclusion:** the "genuine floating-point-chaos with no external cause"
candidate this file left as untested-but-suspected was investigated
further and found to be the WRONG framing. Tracing the fully
deterministic (forced single-threaded, bit-identical across repeats)
failing trajectory directly showed it is not chaos in any deep,
irreducible sense -- `T` overshoots to ~100-150x its own
boundary-condition scale within the first 5 Picard iterations and then
stays trapped in a persistent, bounded attractor, because the project's
`dt=600s`-tuned relaxation factors (`alpha=0.30`/`0.15`) provide
literally zero damping at `dt=60s`. A ~10x smaller, still-fixed
(non-adaptive) relaxation pair, `alpha=(0.03, 0.01)`, restores genuine
contraction: verified bit-identical convergence single-threaded AND 5/5
genuine convergence under normal multi-threaded execution, landing on
SCIF values within 0.15% of each other -- the first remedy in the
project's whole multi-session non-determinism investigation to fix both
the determinism problem and the accuracy problem, with no jitter, retry,
or noise-reliance at all. Several noise-control remedies (n-value
continuation, an analytic Bean-like seed, jitter-retry) were tried first
and each gave real-but-inconclusive mid-run effects that did not survive
to the point that mattered -- the eventual fix came from abandoning
noise-control entirely and root-causing the deterministic failure
instead, which is why it is recorded here rather than as one more entry
in this file's long list of noise-control attempts.

**Still open, so this is a fix at one point, not a closed problem**: only
validated at the single `(dt=60s, I=19.6A)` canonical repro case: whether
it generalises across the `(dt, I)` range a real ramp needs, and whether
it holds up within an actual multi-step time-march (everything tested so
far, in every session, has only ever been a single first step from cold
start), are both untested. See CLAUDE.md's "NI (no-insulation) transient
work" section's final dated entry for the current standing summary.

**CORRECTION, same day, later: even "validated at one point" above was
premature.** User-directed thorough testing found every SCIF/convergence
number here came from a run that stopped well before genuine settling --
a `dt` generalisation sweep found the fix fails again (same signature) at
`dt=30s`; per-layer inspection (never done before) found two of the six
tape layers still overshooting -82x to -86x their own boundary-condition
scale at the point every check above called "converged"; forcing runs far
past `_picard_phase`'s own stall check shows they DO settle, but only
after ~750 iterations, not ~460, and the truly-converged SCIF differs
materially from every number above (+124.6mT, not +131-134mT, at the
canonical point). At the actual validated `dt=600s, I=196A` production
point the same treatment gives a genuinely flat +653.9mT -- closer to
this project's `641.26mT` ground truth than the premature estimate but
still 1.97% off, unexplained. The relaxation-parameter root-cause
diagnosis stands; "a validated fix" does not, yet. See
`transient/validation/nondeterminism_investigation_2026-08-05.md`'s final
entries for the ongoing, corrected arc.

---
