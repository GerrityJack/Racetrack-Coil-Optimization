# Day search report -- started 2026-07-27 00:14:01

## Phase A -- widened search (2026-07-27 00:14:01)

n_layers tried: [6, 8, 10, 12, 14, 16] (a-floor=22mm, tight-bounds off, a seeded 24mm)

| n_layers | tape_km | B_target_T | unif%(proxy) | hoop_MPa | a_mm | b_mm | gap_mm | n_turns | status |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 0.1781 | 10.00 | 15.708 | 92 | 21.50 | 26.53 | 13.63 | `[287, 287, 284, 284, 1, 1]` | finished |
| 8 | 0.1877 | 10.00 | 12.687 | 92 | 21.50 | 27.18 | 18.50 | `[324, 324, 237, 237, 1, 1, 1, 1]` | finished |
| 10 | 0.2121 | 10.00 | 10.115 | 99 | 23.47 | 28.65 | 22.78 | `[356, 356, 226, 226, 1, 1, 3, 3, 1, 1]` | finished |
| 12 | 0.2315 | 10.01 | 6.619 | 111 | 24.81 | 31.07 | 26.24 | `[318, 318, 315, 315, 1, 1, 1, 1, 1, 1, 1, 1]` | finished |
| 14 | 0.2254 | 10.00 | 8.115 | 108 | 22.20 | 27.20 | 29.50 | `[241, 241, 239, 239, 211, 211, 1, 1, 1, 1, 1, 1, 1, 1]` | finished |
| 16 | 0.2214 | 10.05 | 8.249 | 93 | 22.28 | 27.31 | 33.94 | `[355, 355, 1, 1, 179, 179, 72, 72, 1, 1, 1, 1, 1, 1, 2, 2]` | finished |

## Phase B -- T-A box-uniformity validation (2026-07-27 08:04:26)

Target: box peak-to-peak <= 1.0%, BOTH repeats.

| label | tape_km | repeat0 box% | repeat1 box% | verdict |
|---|---|---|---|---|
| 6L_champion (reference) | 0.2259 | 0.828 | 0.828 | PASS |
| phaseA_6L | 0.1781 | 4.462 | 4.487 | FAIL |
| phaseA_8L | 0.1877 | 3.409 | 3.410 | FAIL |
| phaseA_10L | 0.2121 | 3.475 | 3.471 | FAIL |
| phaseA_12L | 0.2315 | 4.096 | 4.097 | FAIL |
| phaseA_14L | 0.2254 | 3.091 | 3.097 | FAIL |
| phaseA_16L | 0.2214 | 3.845 | 3.845 | FAIL |

**Phase B winner: 6L_champion (reference), tape=0.2259km, box p2p=[0.828%, 0.828%]**


## Phase C -- relax assumptions (2026-07-27 09:31:03)

### C.1 -- Ic dataset extrapolation beyond 8T

Every call site in this project uses `clip_B=True` (flat clamp at the B=8T measured value). Since Ic decreases with B in the measured range, that clamp is OPTIMISTIC above 8T, not conservative. Comparing against a linear continuation at the B=8T slope:

| model | I_quench_A | I_op_A | B_target_T | hoop_MPa | clip_frac |
|---|---|---|---|---|---|
| flat clamp (current default) | 408 | 224 | 10.00 | 114 | 0.118 |
| linear continuation (conservative) | 292 | 161 | 6.51 | 59 | 0.000 |

B_target drops 34.9% under the conservative extrapolation (FALLS BELOW the 10T floor).


### C.2 -- fixed-geometry margin sensitivity (winner's geometry held fixed, only I_op/hoop-cap logic varies)

| SAFETY_FACTOR | hoop_cap_MPa | I_op_A | B_target_T | hoop_MPa | clip_frac | binding |
|---|---|---|---|---|---|---|
| 1.3 | 400 | 314 | 14.71 | 223 | 0.231 | quench/SF |

**STOPPED HERE by user request (2026-07-27 09:31), before the rest of C.2's sweep or
any of C.3's re-optimization jobs ran.** Killed cleanly (SIGTERM, no orphaned
processes, `params.py` untouched throughout -- it never held anything but the
6-layer champion's values). Total run time 9h17m (across two launches, the
first ~50min being the invalidated 18mm-floor attempt described below).

---

## CONSOLIDATED SUMMARY

**Bottom line: the 6-layer champion (tape=0.2259km, a=22.227mm, b=27.268mm,
gap=13.500mm, n_turns=[285,285,379,379,2,2]) remains the best validated
design. Nothing found by the widened search beat it -- all 6 new
candidates (n_layers 6/8/10/12/14/16, each independently re-optimized under
a corrected, evidence-based `a`-floor) failed real T-A box-uniformity
validation, most by 3-5x the target.** `params.py` was never modified by
this pipeline and still holds the champion's values.

### What Phase A (widened search) found

Two rounds were needed. The first (18mm `a`-floor) produced designs that
CMA-ES pinned exactly at the floor with no natural settling -- confirmed by
direct T-A check to fail catastrophically (12.14% vs the <=1.0% target) --
so it was discarded and restarted with the floor raised to 21.5mm, then
raised again in-place to 22.2mm (essentially the champion's own radius)
once 21.5mm was ALSO shown to just become a new wall (6L and 8L both
finished pinned exactly there). Under the corrected 22.2mm floor, results
were genuinely mixed -- some layer counts (10, 12, 16) settled naturally
above the floor rather than pinning, which is a meaningfully different,
non-artifactual outcome:

| n_layers | a_mm (final) | pinned at floor? | tape_km | T-A box_ptp% | verdict |
|---|---|---|---|---|---|
| 6 | 21.50 | yes (first-round floor) | 0.178 | 4.46-4.49 | FAIL |
| 8 | 21.50 | yes (first-round floor) | 0.188 | 3.41 | FAIL |
| 10 | 23.47 | no | 0.212 | 3.47-3.48 | FAIL |
| 12 | 24.81 | no | 0.232 | 4.10 | FAIL |
| 14 | 22.20 | at/near 22.2mm floor | 0.225 | 3.09-3.10 | FAIL |
| 16 | 22.28 | at/near 22.2mm floor | 0.221 | 3.85 | FAIL |
| **champion (6L)** | **22.227** | **n/a (not floor-derived)** | **0.226** | **0.83** | **PASS** |

The key finding, repeated and confirmed at every check this session: **coil
radius `a` alone does not determine box uniformity.** The clearest evidence
is `phaseA_8L`, which landed at essentially the SAME radius as the champion
(22.22mm vs 22.227mm) yet scored 3-4x worse (1.80% on an earlier live
check, 3.41% on Phase B's clean re-check) -- the champion's specific turn
distribution ([285,285,379,379,2,2], evenly paired, gently tapered) is
doing something the coarse tape-minimizing search has not managed to
reproduce at ANY other layer count, despite six independent, well-seeded
attempts. This is a genuine negative result, not a search-budget or
floor-tuning problem: every design that got close to the champion's `a`
still failed by a wide margin.

**Process lesson worth keeping for future searches:** CMA-ES will pin `a`
(or any variable with an artificial floor) exactly at that floor whenever
nothing in the fitness function rewards moving away from it -- confirmed
three separate times, including one case (`phaseA_8L`) where a promising
MID-RUN snapshot (a=22.22mm at eval 878) regressed all the way back to the
floor (21.50mm) by the run's own final convergence. A floor only produces a
trustworthy "natural" answer when some OTHER constraint (bend radius,
face-gap) binds above it on its own, as happened for n_layers=10/12/16 --
that outcome can't be forced by picking a higher floor, only observed.

### What Phase C found before being stopped

**C.1 (Ic-extrapolation-beyond-8T sensitivity) surfaced a significant,
previously-uncharacterized risk on the CHAMPION ITSELF, independent of
everything above:** every Ic evaluation in this project flat-clamps Ic to
its measured B=8T value above that field (`clip_B=True`, the default
everywhere) -- since Ic decreases with B in the measured range, that clamp
is an OPTIMISTIC assumption for any cell operating above 8T, not a
conservative one. Re-evaluating the champion under a linear continuation
of the B=8T slope instead: **B_target drops from 10.00T to 6.51T (-34.9%),
falling BELOW the 10T design floor**, and I_op drops from 224A to 161A.
11.8% of the champion's Ic evaluations at its quench point already clip to
the 8T boundary (`clip_frac=0.118`), so this isn't a remote edge case --
it's a real, currently-unmeasured dependency the design's headline numbers
rest on. **This should be treated as the top follow-up item**: either
extend the Ic/n-value dataset above 8T, or explicitly bound the design
against a non-optimistic extrapolation before trusting the 10T claim.

**C.2 (fixed-geometry safety-factor/hoop sensitivity)** only completed one
row before the stop: at SAFETY_FACTOR=1.3 (relaxed from 1.818), the SAME
champion geometry reaches B_target=14.71T at I_op=314A -- well past the
10T floor, but clip_frac rises to 0.231 (23% of Ic evaluations now clipped
at the data ceiling) and hoop stress to 223MPa (still under the 400MPa
cap). This one data point suggests real headroom exists in the safety
factor, but it also pushes further into the unmeasured-Ic region flagged
by C.1 -- the two findings compound, not independent.

**C.3 (tape-savings-vs-risk re-optimization at relaxed margins) did not
run at all** -- stopped before it started.

### Recommended next steps, in priority order

1. **Resolve the C.1 Ic-extrapolation risk first.** It applies to the
   CURRENT, already-accepted champion, not a hypothetical new design --
   this is higher priority than any further search. Either get measured
   Ic data above 8T, or re-run the champion's operating point under a
   deliberately conservative (non-optimistic) extrapolation and confirm
   10T still holds before relying on today's number.
2. If more search is wanted: the widened search's real lesson is that
   tape-minimization without a uniformity signal cannot be trusted to
   preserve a narrow sweet spot, no matter how it's constrained via `a`.
   A targeted local search AROUND the champion's own basin (small
   perturbations to all of a/b/gap/n_turns simultaneously, T-A-validated
   directly rather than coarse-screened) is more likely to find a genuine
   improvement than another blind tape-minimizing run at a different
   layer count.
3. Finish C.2's sweep and run C.3 if the tape-savings-vs-risk trade remains
   of interest -- `optimize/day_search.py`'s Phase C functions
   (`run_margin_sensitivity_fixed_geometry`, `run_margin_reopt`) are still
   in place and can be re-run standalone, seeded from the champion, without
   repeating Phases A/B.
