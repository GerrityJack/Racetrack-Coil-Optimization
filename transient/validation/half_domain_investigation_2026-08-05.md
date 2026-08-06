# 2026-08-05: half-domain (drop x/y symmetry) hypothesis for short-dt non-convergence -- TESTED

## The hypothesis

The FEM domain uses three mirror-symmetry cuts (x=0, y=0, z=coil_half_gap).
The x=0/y=0 planes cut directly through each turn's own racetrack current
loop (straight legs, end-cap tips), forcing the discretized current source
to have current crossing those internal faces. Per `transient/induction.py`'s
docstring, this pollutes the raw vector potential A with a huge (~1e10)
gauge/null-space component, since the curl-curl A-equation's weak form can
only represent divergence-free sources through its main term -- anything
with nonzero divergence is picked up entirely by the tiny
`gauge_regularization` term (1e-3 vs ~8e5). The z=coil_half_gap plane sits
in the physical air gap between the two coils and does not cut through any
turn's own loop.

Hypothesis: dropping the x=0/y=0 mirrors (keeping z=coil_half_gap) -- i.e.
modeling one full closed racetrack turn instead of a quarter of one, ~3.5x
the cell count of the eighth-domain -- might fix or improve the T-A
transient solver's short-dt Picard non-convergence, IF that non-convergence
is driven by the same gauge/divergence artifact.

## Reasoning caveat, checked before building anything

The base T-equation's Picard right-hand side only ever consumes
`curl(A_h - A_prev)` (see `solve/ta_solve.py`'s `L_T` form), which is
gauge-invariant in the continuum -- naively, short-dt non-convergence
should be COMPLETELY ORTHOGONAL to the gauge pollution, since the solver
never touches raw A. The one place raw A's gauge pollution was known to
matter (extracting E_i for the NI circuit closure) was already worked
around cheaply in `transient/induction.py` via a separate mutual-inductance
matrix, without touching the domain.

Checked whether this a priori argument could be defeated at the discrete
level via ill-conditioning degrading curl(A) itself (not just the
null-space component), amplified by the T-equation's 1/dt forcing term at
short dt -- see "Gauge-ratio measurement" below.

## Step 1: does the base insulated solver already fail at short dt with NO NI closure?

Answer, from `docs/HISTORY.md`'s existing 2026-08-05 record (verified by
reading it in full before starting, not re-derived from scratch):
**YES.** `transient/validation/first_step_diagnostic.py` runs exactly the
insulated-limit case (`per_layer=True, per_turn_bc=False` -- no NI closure
machinery at all) and this project's own prior investigation already found:

- `first_step_diagnostic.py 60 19.6` (dt=60s, I=19.6A), 10 independent
  process launches: baseline (default threading/hash-seed) converged 2/5,
  forced-deterministic-threading converged 0/5 -- non-deterministic either
  way, for a reason neither threading nor hash-seeding explains.
- `hybrid_dt100_check.py`-class case (dt=100s, I=32.667A cold start): Picard
  never converges even at 1000+ iterations, SCIF wanders chaotically with
  no decaying trend.
- A **fully monolithic block-Newton** scheme (every layer's T AND the
  shared A solved as ONE simultaneous PETSc SNES system, no Gauss-Seidel
  split at all) hits the SAME qualitative failure signature (clean
  convergence to a value that is a monotonic function of a damping
  parameter, not the physics) -- this rules out "sequential T-then-A
  coupling" as the mechanism, since the monolithic scheme has no such
  split.

This is strong prior evidence the short-dt problem is intrinsic to the
Picard/Newton nonlinear iteration itself and orthogonal to A's gauge
pollution -- but per this project's own standard, an a priori argument is
not a substitute for the actual test, so the half-domain was built and
tested directly (below).

## Step 2: half-domain mesh built

New files (production eighth-domain path untouched):
- `mesh/build_mesh_half.py` -- full x/y racetrack footprint per layer (no
  quadrant clip), air box spanning full x/y, z in [-box/2, g] (z-mirror
  kept). Boundary classification reuses the SAME z~=g-vs-else logic as
  `build_mesh.py`'s eighth-symmetry branch, and it is correct here
  unmodified: with no more x=0/y=0 internal cut faces, every non-PMC
  boundary face genuinely IS the far-field outer box boundary.
  `solve/solve.py`'s `setup_problem()` needed ZERO changes -- it reads
  `facet_tags.find(params.outer_boundary_marker)` generically.
- Verified `physics/current_source.py`'s `tangent_xy`/`normal_xy` and
  `solve/ta_solve.py`'s `build_n_hat_ufl` are already quadrant-agnostic
  (built from `abs(x)`/`arctan2`, no x>=0/y>=0 assumption) -- confirmed by
  reading, not assumed.
- Mesh size at production resolution: 30,000 total cells / 16,446 coil
  cells / 5,552 T-dofs / 36,602 A-dofs, vs. the eighth-domain's 8,136 /
  4,463 / 1,718 / 10,551 -- a 3.2-3.7x increase (not exactly 4x, expected
  given discretization/boundary-layer differences), builds in ~7.6s.

## Step 3: steady-state validation gate (dt=600s, I=196A, cold start)

Ran `transient/validation/half_domain_steady_check.py` (production mesh
resolution, 30,000 cells / 16,446 coil cells) -- the SAME unmodified
`ta_solve.solve_ta_at_current()` entry point used for every production
steady-state result, pointed at the half-domain mesh. IMPORTANT CAVEAT
found while running this: `solve_ta_at_current()`'s own INTERNAL per-
iteration SCIF/stall diagnostic calls `ta_solve.dB_bore_from_dJ()`, which
hardcodes an 8-piece (4 quadrants x 2 coils) mirror expansion assuming its
input cell centroids cover only ONE quadrant -- true for the eighth-domain,
but WRONG for the half-domain (whose centroids already cover the full
racetrack loop), so the internal trajectory the solver printed and used
for its OWN stall criterion was inflated by roughly the quadrant-mirror
factor. This does not corrupt the underlying T/A/rho state (the criterion
only gates WHEN to stop, not the physics), but it does mean the solver's
`ta_scif_stall_mT=0.05` threshold was effectively ~4x tighter than intended
in physical units, and the run hit the `n_picard=150` cap without formally
declaring `converged=True` (`wall=674s`) even though the LAST ~60
iterations were plainly oscillating in a tight band rather than trending.
A CORRECTED, half-domain-aware mirror function (z-mirror only --
`dB_bore_from_dJ_half()` in the check script) was used for the final
reported numbers below, evaluated on the (near-fixed-point, tightly
bounded) end-of-run state:

| quantity | half-domain (this test) | eighth-domain reference |
|---|---|---|
| converged (formal) | False (150-cap, oscillating band) | True (k=80-90, various sessions) |
| \|B\| mean, coil cells | 7.67 T | 4.13 T |
| frac(\|B\|>8T) | 21.5% | 11.8% |
| J/Jc mean | 0.623 | 0.586 |
| frac(J/Jc>1) | 22.8% | 26.0% |
| on-axis SCIF (correct mirror count) | +514.77 mT | ~641 mT |

**Read as a qualified pass, not a clean one.** SCIF (the quantity this
whole investigation is actually about) and the J/Jc distribution land in
the same ballpark (20% low, 14% low respectively) as the eighth-domain
reference -- consistent with a genuinely comparable physical state on a
differently-shaped, differently-meshed domain, not a broken setup. The
\|B\| mean and frac(\|B\|>8T) are further off (~1.85x, ~1.8x) than
expected from meshing differences alone, and this gap is NOT fully
explained (candidate: an unweighted arithmetic mean over coil cells is
more sensitive than a volume-weighted one to how the two independently-
generated meshes happen to distribute cell density between the
higher-field end-cap region and the lower-field straight legs -- plausible
but NOT verified here). **Flagged honestly as an open discrepancy** rather
than smoothed over; it does not change any conclusion below, since the
short-dt test (Step 5) does not depend on this gate passing cleanly, but
it means "the half-domain reproduces the eighth-domain's physics" should
be read as "close enough to trust the domain/BC construction is not
grossly wrong," not as "independently re-validated to the champion's own
precision standard."

## Step 4: gauge-ratio measurement

Direct measurement of `mean(|A|) / mean(|curl(A)|)` at the same converged
(near-fixed-point) state above:

| domain | \|A\|:\|curl(A)\| ratio |
|---|---|
| eighth-domain (documented, `transient/induction.py`) | ~1e10 |
| half-domain (measured here) | **8.76e7** |

**A genuine, large (~114x) reduction in gauge pollution**, in the expected
direction and roughly the expected order of magnitude for removing 2 of
the 3 current-crossing internal cut faces. This CONFIRMS the mechanism the
hypothesis is built on: dropping the x=0/y=0 mirrors measurably reduces
the divergence-driven gauge blow-up. It does NOT eliminate it (8.76e7 is
still enormous, not O(1)) -- residual pollution likely comes from
discretization error in the T-A current representation and/or the z=g PMC
face's own treatment, not investigated further here since it wasn't the
question.

## Step 5: short-dt convergence comparison

Used the canonical repro case from `docs/HISTORY.md`
(`first_step_diagnostic.py 60 19.6` -- dt=60s, I=19.6A, insulated limit,
cold start from ZFC), which the project's own prior investigation
documents as the config with a historical ~20-40% success rate. Built
`transient/validation/half_domain_first_step_diagnostic.py`, the identical
test on the half-domain mesh (coarsened 2x in-plane mesh size/dist factors
only -- `mesh_z_grading`, the resolution axis CLAUDE.md identifies as the
one that actually matters for the screening profile, was left at
production settings; 10,026 total / 5,735 coil cells at this coarsening,
vs. the eighth-domain's 8,136 / 4,463). 6 independent process launches
each (own process, own mesh, per this project's established
cross-process-independence requirement):

| domain | config | converged | n_iters (in order) |
|---|---|---|---|
| eighth (this session, today) | dt=60 I=19.6 | **1/6** | 150(F), 28(T), 150(F), 150(F), 150(F), 150(F) |
| half (this session, today) | dt=60 I=19.6, coarsen=2x | **0/6** | 150(F), 150(F), 150(F), 150(F), 150(F), 150(F) |
| eighth (`docs/HISTORY.md`, prior session) | dt=60 I=19.6 | 2/5 | 92, 64, 150(F), 150(F), 150(F) |

Combining this session's eighth-domain result with history's gives 3/11
(27%) for the eighth domain under default (non-forced) threading/hashing,
consistent with the project's documented "20-40%" range. **0/6 on the
half-domain is not, by itself, statistically distinguishable from the
eighth-domain's own ~20-27% baseline rate** (P(0 successes in 6 trials |
p=0.2) is ~26% -- unsurprising even with no real difference) -- so the
raw pass/fail count alone is suggestive but not decisive.

**The decisive evidence is the failure SIGNATURE, not just the count.**
Re-ran one half-domain failing case with `NI_TRACE=1` to see the full
per-iteration trajectory (`half_trace_run.log`): SCIF decays smoothly and
plausibly from the seed (+12675 mT at k=1 down to +141-190 mT by k~18-19,
looking exactly like it is about to converge) and then, for the remaining
~130 iterations, **wanders chaotically with no decaying trend** -- bouncing
between roughly -70 and +550 mT, `|dB|` staying large (70-150) throughout,
never settling, ending at -70.37 mT purely because that's where the
150-iteration cap happened to land. This is QUALITATIVELY IDENTICAL, not
just similarly bad, to `docs/HISTORY.md`'s own description of the
eighth-domain's dt=100s/I=32.667A failure ("SCIF wandered chaotically,
std~100 mT, no decaying trend") and its fully-monolithic-Newton failure
mode. **Reducing A's gauge pollution by ~114x did not change the
qualitative character of the instability at all** -- same smooth-then-
chaotic shape, same order-of-magnitude wander amplitude, same failure to
ever decay.

## Conclusion

**The half-domain hypothesis is REJECTED, on two independent lines of
evidence, not just the a priori orthogonality argument:**

1. Pass/fail rate shows no improvement (0/6 vs. the eighth-domain's own
   ~20-27% baseline measured the same day, same script family) -- weak
   evidence alone, but directionally against the hypothesis, not for it.
2. **The failure signature is unchanged** -- smooth initial decay followed
   by unbounded chaotic wandering, identical in character to the
   eighth-domain's own documented failure modes across THREE structurally
   different solvers now (Picard, a Gauss-Seidel Newton hybrid, and a
   fully monolithic block-Newton with no T/A split at all) -- despite a
   measured, genuine, ~114x reduction in exactly the gauge-pollution
   quantity the hypothesis targeted. If the gauge artifact were a material
   contributor to the instability, a 114x reduction in it should have
   produced SOME visible change in the trajectory's character (a longer
   smooth-decay phase, a smaller wander amplitude, more iterations before
   the onset of chaos, etc.). None of that is observed.

This confirms, empirically, what the a priori reasoning in this write-up's
second section already argued: the short-dt Picard/Newton non-convergence
is intrinsic to how the T-equation's coefficients (rho via Jc(B)/n(B),
frozen/lagged every outer iteration because the measured-CSV spline models
are not UFL-differentiable) get updated between outer iterations, not to
anything about the vector potential A's gauge freedom. The half-domain
mesh, BC-construction, and diagnostic code built here (`build_mesh_half.py`,
`half_domain_steady_check.py`, `half_domain_first_step_diagnostic.py`) are
left in the repo as validated, reusable infrastructure (correct boundary
tagging, correct current-direction generalization, working per-layer T-A
setup) in case a FUTURE hypothesis specifically needs a domain without the
x/y symmetry cuts -- but the current short-dt non-convergence problem is
not that case.

**Not pursued further at the time this section was first written:**
CLAUDE.md's own flagged next step -- a bit-level diff of the assembled
matrix/RHS between a successful and failed run at the very first Picard
iteration -- was judged out of scope here: it is a nondeterminism-focused
diagnostic (explicitly assigned to a parallel investigation in this
project's current work), not a half-domain-focused one. Per this project's
own established "cheap lever exhaustion" pattern (see `docs/HISTORY.md`'s
Newton-Krylov entries), the two genuinely untried, more expensive levers
were flagged as remaining: (1) fold Jc(B)/n(B) into the Newton residual
itself via a finite-difference or precomputed-derivative Jacobian through
the measured-CSV spline models, removing outer-loop coefficient-freezing
entirely; or (2) accept that short-dt/multi-step schedules need dt values
kept close to the validated 600s regime. A cheap, BOUNDED follow-up
testing the DIAGNOSIS behind lever (1) -- without committing to building
it -- was requested and is reported below.

## Follow-up: does freezing Jc(B)/n(B) for the whole step change the failure signature?

**Question.** This write-up's conclusion diagnoses the instability as
living in how Jc(B)/n(B) get re-evaluated from the CURRENT B and lagged
into the next outer iteration (`ta_solve._update_rho`'s normal behaviour),
rather than in anything about A's gauge freedom. This follow-up tests
that diagnosis directly, cheaply, without building the large differentiable-
Jacobian rewrite: if the per-iteration Jc/n re-evaluation is a material
driver of the chaotic wandering, FREEZING Jc(B) and n(B) at their seed-B
values for the entire step (removing ALL B-dependence from the coefficient
lookup for that step, leaving only rho(J)'s power-law dependence on the
CURRENT current density, which Newton already linearizes exactly elsewhere
in this project without fixing short-dt convergence) should measurably
soften or change the failure -- e.g. a longer smooth-decay phase, a smaller
wander amplitude, or outright convergence.

**Method.** `transient/validation/frozen_jcn_diagnostic.py` -- a
throwaway, standalone script (NOT a change to `ta_solve.py` or
`ta_transient.py`): identical setup to `first_step_diagnostic.py` (eighth-
domain, own mesh per process, `per_layer=True, per_turn_bc=False`, cold
start from ZFC, dt=60s/I=19.6A, `max_iters=150, min_iters=6, scif_tol=0.5`
-- the exact same stopping criteria for a fair comparison), except
`Jc_vol`/`n_arr` are computed ONCE from the seed (uniform-J) B field
(`ic.critical_current`/`nm.n_value` at `B_mag_seed`, `theta_seed`) and
held fixed; every Picard iteration's rho update (`_update_rho_frozen_jcn`,
copied line-for-line from `ta_solve._update_rho`'s math with Jc_vol/n_arr
as fixed inputs) only recomputes `Jmag` from the CURRENT `J` against those
frozen arrays -- the power-law nonlinearity in `J` and the log-space rho
relaxation are otherwise untouched. 5 independent process launches.

**Result: 0/5 converged, all capped at the 150-iteration budget --
identical failure rate to the un-frozen case, and the SAME qualitative
signature.** All 5 traces show large, non-decaying `|dB|` (50-220)
straight through to k=150, with SCIF swinging over hundred-to-thousand-mT
ranges within a single run (rep 1: seed decay to ~-3550 mT by k=6, then
wanders back up through -1500, -750, ..., to +175 mT with no settling;
rep 2: ends climbing through -250 -> +1117 mT in the last 5 iterations
alone; rep 3: still climbing +72 -> +450 mT at the cap). **No run showed a
longer smooth-decay phase, a smaller wander amplitude, or anything else
suggesting the instability softened.** This is a small sample (5 runs,
one config) and should be read as directional, not a final statistical
claim -- but the direction is unambiguous: completely removing Jc/n's
per-iteration B-dependence did not change the failure's character at all.

**Conclusion of the follow-up: the per-iteration Jc(B)/n(B) re-evaluation
is NOT, by itself, a material driver of the chaotic wandering.** The
instability survives even when that specific mechanism is switched off
entirely, which sharpens (and partially revises) this write-up's own
diagnosis -- it points more squarely at the T/rho/A field-coupling loop's
own fixed-point structure (the rho(J) power-law feedback and its
interaction with the relaxed T/A updates) as the locus of the instability,
consistent with the project's broader finding that a fully monolithic
Newton scheme with NO outer Picard-lag on ANYTHING still hits the
identical failure signature. This does not by itself distinguish between
the two remaining levers CLAUDE.md already flags (a true differentiable
Jacobian vs. dt-near-600s scheduling) -- if anything it makes lever (1)
LESS obviously promising than the original diagnosis suggested, since
removing the exact coefficient it would target did not help here. Per the
explicit scope for this follow-up, this result is reported as-is and NOT
used to launch either of those larger efforts -- both remain separate,
user-reviewed decisions.

**This closes the half-domain investigation and this bounded follow-up.**
No further threads opened from here.
