# Error hunt, night of 2026-10-04/05

Goal: find errors that, like the δ/Λ resistivity factor (fixed 2026-10-04),
silently invalidate the optimization. Every check compares against
something independent of the code path being tested: a measured dataset,
an independent solver, an analytic result, or a convergence study.

Compute runs: `optimize/runs/error_hunt/queue{1,2,3}.sh` and their `.log`
files; T-A rows in `optimize/runs/ramp_hold/eh_*.csv`.

## Headline (for the morning)

Nothing found tonight invalidates results the way the δ/Λ factor did, but
two findings move the design verdicts by amounts comparable to the
margins, both in the NON-conservative direction:

1. **The single-step ramp surrogate under-predicts the end-of-ramp worst
   load by ~4 %** (0.881 vs 0.917, legacy, 180 A, vs a √t constant-power
   march) and over-predicts B by ~1.7 %. Early-ramp screening currents
   persist; the surrogate starts from a virgin state. Given load ∝
   ~I^0.15–0.18, this is ~20–25 % in allowed current.
2. **The pass/fail verdict rests on a single cell, which is fragile.** On
   the fine in-plane mesh the worst cell jumped 0.881 → 0.970 while the p99
   stayed exactly 0.786. The
   nonlinear benchmark shows (a) a too-large relaxation α produces spurious
   hotspots (up to 2× the true load) that the SCIF-based convergence test
   cannot see, and (b) even converged, the single worst cell carries ~6 %
   mesh noise (p99.9 is accurate). Slivers add occasional outliers (seen:
   1.103 vs 0.994).

3. **Uniformity is not mesh-converged across the tape width:** the denser
   xdense7 grading doubles it at floor 0.6 (0.475 → 0.955 %, legacy 180 A)
   while loads and B agree to 0.5 %. Uniformity verdicts against the 0.8 %
   limit are uncertain by ~±0.5 pp on the production mesh.

Also found: the FEM air box is too small (field 9 % low at the bore, 5–8 %
in the winding) — real, but loads move only ~0.2 % because the load depends
on E, not on Jc's absolute value; hoop stress is ~8 % low.

Verified correct (independent checks): the T-A implementation in the
nonlinear regime, Ic angle convention, tape normal, current and Jc
normalisation, screening sum and mirror images, B_target and uniformity
(Biot-Savart), time-step handling, optimizer scoring, cross-process
reproducibility.

### Recommendations, in priority order
1. Evaluate end-of-ramp loads with a short march (e.g. 3–4 steps along the
   √t profile) instead of the single 4 h step, or calibrate a correction
   on a few designs. Check the march's own step convergence once (8 steps).
2. Replace the single-worst-cell criterion with a robust statistic (p99.9,
   or the mean of the top ~0.1 % of cells by volume) and exclude
   degenerate elements; add a local convergence test (that statistic stable
   over the last N iterations), and gate scoring on convergence.
3. Validate α for each (dt, floor) regime the evaluator uses with a
   forced long run (the benchmark's α was too large at dt = 100 s).
4. box_scale ≥ 8 in all tiers (cheap: +40 % cells).
5. Still pending tonight: across-width mesh (xdense7) and fine in-plane
   mesh checks, box_scale 16 (queue 4, `optimize/runs/error_hunt/queue4.log`).

## Summary

| # | check | verdict | impact |
|---|---|---|---|
| 1 | FEM air box (box_scale 4) | **ERROR, low impact** | FEM field 9 % low at the bore, 5–8 % low in the winding; loads change only ~0.2 % (load depends on E, not Jc); hoop stress ~8 % low |
| 2 | Ic(B,θ) angle convention | correct | — |
| 3 | tape-normal orientation (straight + caps, numpy + UFL) | correct | — |
| 4 | current / Jc normalisation, δ cancellation | correct | — |
| 5 | screening (SCIF) Biot-Savart sum, mirror images | correct | — |
| 6 | B_target / uniformity (Biot-Savart) vs FEM | correct to 0.3 % / 0.01 pp once the box is large | — |
| 7 | Kim extrapolation above 8 T vs Fujikura 20 K data shape | mildly optimistic (~3–6 % at 10–12 T) | known ±0.5 T caveat |
| 8 | n-value above 8 T | held flat (~20) | slightly conservative for E < E_c |
| 9 | optimizer scoring / pass-fail consistency | correct | — |
| 10 | sliver elements | ~0–1 per mesh, mesh varies across processes | occasional spurious worst cell (seen: 1.103 vs true 0.994) |
| 11 | reproducibility across processes (legacy, 180 A, floor ladder to 0.6) | good: B 10.161/10.161 T, unif 0.475/0.475 %, load 0.882/0.881 | — |
| 17 | time step actually honoured (dt_const set once, never reset; optimizer passes 7200 s) | correct | — |
| 18 | unconverged T-A solves in scoring | flag recorded but ignored by the scorer; 6/200 unconverged in updated_10_4, none passing | recommend gating on it |
| 12 | 2 h √t ramp march vs the 4 h single-step surrogate | **ERROR (non-conservative)** | surrogate under-predicts the end-of-ramp worst load by ~4 % (0.881 vs 0.917) and over-predicts B by ~1.7 %; ≈20–25 % in allowed current |
| 13 | across-width mesh (xdense7 vs medium5) | loads/B converged (worst 0.877 vs 0.881, B 10.155 vs 10.161 T); **uniformity NOT: 0.955 vs 0.475 %** at floor 0.6 | uniformity verdicts (0.8 % limit) uncertain by ~±0.5 pp on the production mesh |
| 14 | fine in-plane mesh (tier `fine`, box 6) | p99 identical (0.786), B +0.3 %, uniformity 0.499 vs 0.475 %; **single worst cell 0.970 vs 0.881** (E/E_c 0.53 vs 0.13: a local outlier) | robust load statistics are mesh-converged; the single worst cell is not |
| 15 | air-box impact on loads (box 8) | small | 180 A, floor 0.6: worst 0.879 vs 0.881, B 10.181 vs 10.161 T, unif 0.472 vs 0.475 % |
| 16 | nonlinear (n = 21) benchmark vs discrete-tape reference | **implementation correct** (0.1 % at floors 1.0/0.9/0.7); **too-large α gives spurious hotspots** (up to 1.65 vs true 0.80) invisible to the SCIF convergence check; even converged, the single worst cell carries ~6 % mesh noise (p99.9 matches) | α must be validated per regime; use a robust statistic (p99.9 / top-N mean) and a local convergence check |

## Details

### 1. FEM air box too small (box_scale 4) — ERROR

`validation/eh_field_crosscheck.py` compares the FEM field at the 30×6 mm
box points with the independent Biot-Savart value the optimizer reports.
At the production box_scale 4 (outer PEC box edge = 4·b) they disagreed by
almost 1 T. `validation/eh_box_size_check.py` (uniform-J, legacy, 180 A):

| box_scale | box edge | box mean Bz vs Biot-Savart | coil peak \|B\| | coil mean \|B\| |
|---|---|---|---|---|
| 4 (all production runs) | 126 mm | −9.2 % | 10.97 T | 4.09 T |
| 8 | 251 mm | −0.8 % | 11.42 T | 4.42 T |
| 16 | 502 mm | −0.3 % | 11.50 T | 4.42 T |

The PEC box (n×A = 0, B·n = 0) forces the return flux into a box barely
larger than the coil. Reported B_target and uniformity are Biot-Savart and
therefore fine (box 16: FEM vs BS 0.3 % in B, 1.178 vs 1.171 % in
uniformity). But the FEM field is what sets Jc(B) and the screening drive
in every cell, so loads have been computed in a field 5–8 % too low, which
I first expected to make them optimistic. **Measured impact (box 8 vs 4, legacy, 180 A,
floor 0.6): worst load 0.879 vs 0.881, p99 0.796 vs 0.786, B_target 10.181
vs 10.161 T, uniformity 0.472 vs 0.475 %. Negligible.** Reason: with the
power law, load = (E/E_c)^(1/n); Jc's absolute value cancels, and a 5–8 %
change in the driving field changes the load by only ~1.08^(1/21) ≈ 0.4 %.
What the box error does affect: the hoop stress (computed from the FEM
field, ~8 % low; irrelevant at 70–110 MPa vs 400 MPa) and any quantity that
uses |B| in the coil directly. Recommend box_scale ≥ 8 anyway (+40 % cells).

### 5/6. Screening sum and B_target

Mirror transforms (x: J→(+,−,−), y: (−,+,−), coil 2: (+,+,−)), prefactor
and cell volumes are correct. With a large box, FEM and Biot-Savart agree on
the screening correction to ~5 mT (0.05 % of B) and on uniformity to
0.01 pp.

### 7. Ic above 8 T

Field dependence at θ = 0, ratio to 5 T: Shanghai + Kim 2.10 (1.1 T) /
0.29 (24 T); Fujikura 20 K (Tsuchiya et al. p-q fit) 1.88 / 0.26. Kim
falls slightly more slowly: mildly optimistic at 10–24 T (~3–12 %).

### 10. Slivers

gmsh γ < 1e-3 elements: 0–1 per mesh (legacy 0, 16L 1, 28L 0 in one
process); meshes are reproducible within a process but differ across
processes, so a given design sometimes gets one. One sliver gave a
spurious worst load of 1.103 (true 0.994) in the eval-166 reproduction.
Recommend excluding near-degenerate cells (or using a high percentile)
in the worst-cell criterion.

### 11. Reproducibility

Fresh process / fresh mesh, legacy 180 A floor ladder: identical to
< 0.1 % in B, uniformity and worst load. Meshing noise is not a concern
for this design.

### 12. Single-step surrogate vs a √t constant-power march — ERROR

Legacy design, 2 h constant-power ramp to 180 A, floor 0.7, medium mesh.
March: four backward-Euler steps of 1800 s along I = 180·√(t/7200)
(90 → 127.3 → 155.9 → 180 A), A_prev carried forward
(`optimize/runs/ramp_hold/eh_march_legacy180.csv`). Surrogate: one 4 h step
from the virgin state (`floortest_legacy6L_180A.csv`).

| | worst load | p99 | B_target | uniformity | SCIF |
|---|---|---|---|---|---|
| march, t = 1800 s (90 A) | 0.895 | 0.823 | 4.84 T | 0.60 % | 453 mT |
| march, t = 3600 s (127.3 A) | 0.904 | 0.813 | 6.99 T | 0.34 % | 502 mT |
| march, t = 5400 s (155.9 A) | 0.927 | 0.817 | 8.74 T | 0.45 % | 452 mT |
| **march, t = 7200 s (180 A)** | **0.917** | 0.808 | **10.24 T** | 0.59 % | 383 mT |
| surrogate, floor 0.7 | 0.881 | 0.779 | 10.41 T | 0.77 % | 217 mT |
| surrogate, floor 0.6 | 0.881 | 0.786 | 10.16 T | 0.47 % | 451 mT |

Mechanism: a constant-power ramp is fastest at the start (dI/dt ∝ 1/√t);
the screening currents induced early decay only by flux creep and are
still present at the end. The surrogate (virgin state, final ramp rate)
misses them. The load also peaks mid-ramp (0.927 at 155.9 A). With loads
rising only as ~I^0.15–0.18, a 4 % load error is ~20–25 % in allowed
current. Caveats: four steps only (finer steps probably raise the early
loads further), floor 0.7 (B not floor-converged; the surrogate's B drops
2.4 % from 0.7 to 0.6, so the march's floor-0.6 B is likely ~10.0 T).
Recommendation: replace the single-step surrogate with a short march in
the evaluator, or calibrate a correction factor on several designs.

### 16. Nonlinear benchmark (n = 21, power law) — implementation OK, local-convergence hazard

`validation/ta_nonlinear_benchmark.py`: same 40-tape geometry as the ohmic
benchmark, constant Jc (Ic = 100 A/tape), n = 21, I = 50 A, one 100 s
backward-Euler step from the virgin state. The independent reference
(discrete tapes, Newton with continuation in n) is converged to 0.3 % in
its z-resolution (M = 40 vs 80). Reference worst load: 0.7985 (pure power
law); with the solver's floor: 0.512 / 0.589 / 0.769 / 0.795 / 0.798 at
floors 1.0 / 0.9 / 0.8 / 0.7 / 0.6 — an independent confirmation that
the worst load converges once the floor is ~0.1–0.2 below it.

FEM (production code path, stub Jc and n), first run, α = 0.10/0.10/0.05/
0.02/0.005 at floors 1.0/0.9/0.8/0.7/0.6:

| floor | FEM worst (x < 8 mm) | FEM worst (all cells) | reference | profile RMS |
|---|---|---|---|---|
| 1.0 | 0.512 | 0.523 | 0.512 | 0.1 % |
| 0.9 | 0.587 | 0.603 | 0.589 | 0.7 % |
| 0.8 | 0.770 | **0.898** | 0.769 | 1.5 % |
| 0.7 | **1.293** | **1.293** | 0.795 | 5.7 % |
| 0.6 | **1.653** | **1.653** | 0.798 | 12.7 % |

Second run (fresh process/mesh, NB_DUMP=1): floor 0.7 worst in x < 8 mm
0.797 (correct), but 15 cells at x ≈ 11–12 mm on the bottom tape edge at
0.918; per-cell values move ±20 % between floors 0.8 and 0.7 while the
reference moves 3 %. So at low floors individual cell values are noisy
and mesh-dependent, while the global SCIF (the convergence test) is flat.
|ΔB|/|B| is ~1 throughout in this small-field geometry and not
diagnostic. Most likely cause: α too large for this short dt (stable α
scales with ρ_floor·dt; dt = 100 s here vs 14400 s in production), i.e. a
local limit cycle that the global SCIF criterion cannot see. Re-test with
α = 0.01 (floor 0.8) / 0.002 (floor 0.7) running
(`optimize/runs/error_hunt/nl_alpha_small.log`; large-α dumps saved in
`validation/nl_alpha_large/`).

**Small-α re-test** (α = 0.01 × 400 at floor 0.8, 0.002 × 2000 at 0.7;
same mesh — gmsh was reproducible here): floor 0.7 worst in x < 8 mm 0.796
(reference 0.795), all cells 0.843 (was 0.918 with α = 0.02), p99.9 0.803.
So the large hotspots were a relaxation (stability) artifact, but even a
converged solve shows ~6 % scatter in the single worst cell along a
straight section where the true answer is x-invariant; the p99.9 matches
the reference. Two consequences: (1) α must be validated for every
(dt, floor, geometry) regime — a "converged" SCIF does not prove it;
(2) the single-worst-cell criterion inherits several % of mesh noise,
which the steep load-vs-current curve turns into tens of % in allowed
current. A high percentile or the mean over the top few cells would be
far more robust.

Production relevance: the legacy floor ladders (0.881 / 0.881 / 0.883 at
floors 0.7 / 0.6 / 0.5, and reproduced in a fresh process) show no sign of
this. But the evaluator's convergence test is the same global SCIF range,
and the design verdict is a single worst cell, so a local limit cycle would
pass undetected. Recommend a local convergence check: the worst load (or
the p99.9) must be stable over the last N iterations, not just the SCIF.

### 13. Across-width mesh (xdense7 vs medium5)

Legacy, 180 A, 4 h step, floor ladder (`eh_zgrid_xdense7_legacy180.csv` vs
`floortest_legacy6L_180A.csv`):

| floor | worst (med5 / xd7) | p99 | B_target (T) | uniformity % |
|---|---|---|---|---|
| 0.7 | 0.881 / 0.863 | 0.779 / 0.783 | 10.413 / 10.413 | 0.767 / 0.822 |
| 0.6 | 0.881 / 0.877 | 0.786 / 0.792 | 10.161 / 10.155 | **0.475 / 0.955** |

Loads and B are converged to < 0.5 %; box uniformity is not (it is the
near-cancelling quantity CLAUDE.md already warns about, and its sensitivity
grows at low floor where the screening currents are larger).

### 14. Fine in-plane mesh (tier `fine`: size_min 0.10·pack, box_scale 6)

Legacy, 180 A, 4 h step (`eh_mesh_fine_legacy180.csv`, 21,899 coil cells
vs 4,463):

| floor | worst (med / fine) | p99 | B_target (T) | uniformity % |
|---|---|---|---|---|
| 0.7 | 0.881 / 0.874 | 0.779 / 0.778 | 10.413 / 10.457 | 0.767 / 0.841 |
| 0.6 | 0.881 / **0.970** | 0.786 / 0.786 | 10.161 / 10.193 | 0.475 / 0.499 |

The p99 load, B and (in-plane) uniformity are converged; the single worst
cell is not (its E/E_c = 0.53 vs 0.13, i.e. an isolated outlier — likely the
same local-instability mechanism as the benchmark: the stable α shrinks as
elements get smaller, and α = 0.01 was tuned on the medium mesh). The old
"in-plane refinement moves margins by 20 %" does not reproduce for robust
statistics; it was most likely the unconverged-solve artefact of that time.

## Actions taken (2026-10-05, evening)

User decisions: implement the low-cost recommendations now, keep the
expensive ones for finalist validation, and shorten the ramp.

**Implemented in the search** (`optimize/ta_ramp_current.py`,
`run_optimization.py`; `solve/ta_solve.py` gained an opt-in
`iter_callback`, default behaviour unchanged):

1. **Ramp 2 h -> 30 min** (user decision): surrogate step 4 h -> 1 h.
   Floors <= 0.7 get α ÷ 4 and iterations × 4 (stable α ∝ ρ_floor·dt):
   floor 0.7 α 0.0125 × 600, floor 0.6 α 0.0025 × 1200; floors ≥ 0.8
   unchanged. **Roughly 3× the cost per design**, from the time-step
   change, not from the error-hunt fixes. The 1 h α values come from the
   scaling rule and have NOT been checked with a forced long run.
2. **Gate = 99.99th percentile of J/Jc by cell count (nearest rank),
   slivers excluded** (user choice). That is the worst cell below 10,000
   coil cells and the 2nd-3rd worst on the largest meshes. Checked on
   saved fields: equals the worst cell on every legacy medium-mesh
   snapshot; eval 166 (29,367 cells) 1.103 -> 0.993 (true 0.994).
   *Volume-weighted* percentiles were tried and rejected: the most-loaded
   cells are the small tape-edge cells, so volume weighting ran ~7 % below
   the worst cell (legacy 185 A: 0.854 vs 0.921), nearly as loose as
   p99.9. The plain worst cell is still recorded.
3. **Sliver test** = gmsh-style γ = 3·r_in/R < 1e-3 (legacy medium mesh:
   0 cells). V/l_max³ was rejected: it flags 28 thin tape-edge cells.
4. **Local convergence check**: over the last 30 iterations of the I_op
   solve, the gated load's range and |slope|/α (remaining distance to
   the fixed point for the measured ~(1-α)^k decay) must both be < 0.005,
   in addition to the SCIF range < 0.5 mT. The tolerance is a first
   guess; check how often it trips in the first real evaluations.
5. **Convergence gating**: unconverged designs get status `unconverged`,
   the failed-evaluation fitness, and can never pass; their numbers stay
   in history.csv. New columns: load_gated, scif_range_mT, load_range,
   load_resid, n_sliver_cells.
6. New output folder `optimize/runs/search_ramp30min/`; the runner
   refuses to resume a history written by the old evaluator.
7. `optimize/studies/ramp_hold_test.py` also reports `gated_load` and the
   sliver count, so finalist checks use the same statistic.

Tested: dry run (fake physics) and a coarse-mesh, cut-iteration real
evaluation of the legacy design (all code paths; numbers meaningless).
NOT yet run: a real-length evaluation at the 1 h step.

**Deferred to finalist validation** (README, "Validating a design"):
√t ramp march (required, surrogate non-conservative, plus a step-
convergence check and a march/surrogate calibration on 3-4 designs);
xdense7 uniformity (required, ±0.5 pp); fine in-plane and independent
mesh; air box 8 (incl. hoop stress); α halving check per regime; floor
0.5. box16 (queue 4) never finished: the process died at 14:02 during
its floor-0.9 step; low priority.
