# 2026-08-05: differentiable Jc(B)/n(B) Jacobian for the monolithic T-A
# solve -- built, then a false "breakthrough" caught and retracted, then a
# bigger, pre-existing problem found underneath it

## The goal

`newton_ta.py` and `monolithic_ta.py` are both explicitly "quasi-Newton":
Jc(B,theta) and n(B,theta) come from measured-CSV scipy splines
(`physics/ic_model.py`), which `ufl.derivative` cannot differentiate
symbolically, so both files freeze them as plain DG0 coefficients within
any single Newton solve, refreshed only between outer iterations. The
task: replace the spline lookups with smooth closed-form equations that
ARE UFL-differentiable, so a monolithic T-A solve (the only place this can
matter -- see below) gets a genuinely complete Jacobian for the first time
in this project.

## Why this only matters for a MONOLITHIC solve, not the production path

Worked out before writing any code, and confirmed by the frozen-Jc/n test
in `half_domain_investigation_2026-08-05.md`: in the project's sequential
Gauss-Seidel scheme (solve T with A/B frozen, then solve A with T frozen,
repeat -- what `newton_ta.py` and the entire production path use), B is
not a function of the unknown T being solved for AT ALL during any single
T-solve. `d(Jc)/dB` contributes exactly zero to that Jacobian no matter
how Jc(B) is represented. Only `monolithic_ta.py`'s block system (T and A
solved simultaneously, so B=curl(A) is a genuine live unknown) can be
affected by this. That module already exists and already only used frozen
Jc/n -- this work targets exactly that one gap.

## Part 1: the physics -- built and validated, this part is solid

`physics/entropy_ic_model.py`, two fitted models, both pure algebra in B:

**`EntropyBetaIcModel`** -- Long (2013) maximum-entropy Beta model,
Eq. 2: `Jc(b) ~ b^(alpha-1)*(1-b)^(beta-1)`, `b = B/B_irr`. Fit PER ANGLE
to the FULL measured 0-8T grid (not the 7-point >8T-only tail
`optimize/ic_extrapolation.py`'s `BetaIcModel` uses for extrapolation).

- **First fit attempt, unconstrained**: MAPE 2.29% at low/mid field, but
  alpha in [0.607, 1.051] with 41/43 angles having alpha<1 --
  **Jc(B->0) diverges** for nearly every angle. Fatal for a zero-field-
  cooled solver (the CSV's own field grid starts at B=0 exactly).
- **Second attempt, alpha constrained >=1**: fixes the divergence but
  MAPE degrades to ~8% overall (45% at 8T) -- every fit lands alpha
  pinned at its 1.0 floor, meaning the true optimum wants alpha<1 and the
  hard constraint fights the data.
- **Final form, DELIBERATE DEVIATION FROM THE PAPER'S LITERAL EQUATION**:
  added a small field-offset regularization, `b = (B+B0)/B_irr` instead
  of `b = B/B_irr` (same philosophy as `ta_solve.py`'s `eps_reg` smooth
  floor on rho). This lets alpha take its unconstrained best-fit value
  (0.3-1.02 across angles) while keeping `Jc(0)` finite. Result: **MAPE
  2.29% over the FULL 0-8T grid** (median 1.92%, worst single point
  10.22%), `Jc(0)` accurate to 1-3% at every angle (vs. literally
  divergent before). Validated ONLY against the measured 0-8T range --
  NOT validated for extrapolation above 8T; `KimIcModel` remains the
  project's validated choice there.

**`HillNModel`** -- empirical saturating (Hill-type) decay,
`n(B) = n_inf + (n0-n_inf)/(1+(B/Bn)^p)`. Explicitly **NOT** derived from
the entropy-maximization paper (that paper is Jc-specific) -- chosen
purely for smoothness/fit quality. MAPE 0.47% over the full grid (worst
single point 2.89%), smooth and bounded everywhere, no regularization
needed.

**Analytic derivative cross-check**: `EntropyBetaIcModel.dIc_dB` matches
central finite-difference exactly (0.000% relative difference) at every
tested field except B=0.01T, where the apparent mismatch was traced to
the numpy convenience wrapper's OUTPUT clip (`Ic_max` ceiling) creating a
flat plateau -- the raw, unclipped formula's derivative matches finite
difference exactly there too. `jc_ufl_expr()` (the function the solver
actually uses) never applies that output clip, only a domain clamp on `b`
into (0,1) that never engages in the physically relevant range here --
confirmed by direct inspection, not assumed.

## Part 2: wiring into a monolithic solve -- one real bug found and fixed

`transient/monolithic_ta_diff.py` (new, purely additive, does not modify
`monolithic_ta.py`/`newton_ta.py`/`ta_solve.py`): copies
`monolithic_ta.py`'s block-Newton structure, replacing the frozen
`Jc_fn`/`n_fn` DG0 constants with `EntropyBetaIcModel.jc_ufl_expr()` /
`HillNModel.n_ufl_expr()` evaluated on `Bmag_ufl = sqrt(inner(curl(A_h),
curl(A_h)) + eps)` -- genuinely live in the unknown A. Only the 9
per-layer ANGLE-dependent parameters (5 for Jc, 4 for n) stay Picard-
lagged DG0 coefficients, refreshed between outer iterations exactly as
`monolithic_ta.py` refreshes its frozen `Jc_fn`/`n_fn` today.

**Bug found and fixed**: `EntropyBetaIcModel` is correctly fit against
`IcModel.critical_current()`, i.e. Ic in AMPS (matching the CSV's native
units) -- but `rho_expr` needs the VOLUMETRIC critical current density
`Jc_vol = Ic/(delta_SC*tape_width)` (~1e11 A/m2 at this project's scale),
exactly as `ta_solve._update_rho` computes it for the production path.
Using raw amps where volumetric A/m2 was needed is an 11-order-of-
magnitude scale error. Fixed with a single scalar rescale of the `A`
(amplitude) parameter at the point of injection into the DG0 coefficient
(`_update_monodiff_coefficients`), leaving `EntropyBetaIcModel`'s own fit
(correctly validated against amps, matching `IcModel`'s convention)
untouched. Confirmed by direct instrumentation: post-fix, `j_norm =
Jmag/Jc` at sample cells ranges 0.015-6.7 (mean 1.2) -- a physically
sensible near-critical regime -- vs. an astronomically overcritical
~1e8 pre-fix.

**Surprising finding, not fully explained**: fixing this 11-order-of-
magnitude bug did NOT change the converged SCIF outcome (both landed at
~1158.6-1158.7 mT across repeated tests, before and after the fix).
Working hypothesis, not proven: the fix is a UNIFORM rescale of Jc across
every cell (a single scalar factor, not spatially varying), and since
current-redistribution physics is governed by RELATIVE overcritical-ness
between cells (driven by the ANGLE/B-direction pattern, which the fix
does not touch) rather than the absolute resistivity scale, the
redistribution pattern -- and hence SCIF -- may be largely invariant to
a uniform Jc rescale. Not chased further; flagged for anyone who needs to
understand it precisely.

## Part 3: the false "breakthrough", and how it was caught

Initial testing (`snes_linesearch_type="bt"`, the default) hit
`SNES_DIVERGED_LINE_SEARCH` (reason=-6) after a handful of outer
iterations, with the linesearch monitor showing `ynorm ~ 1e16` on the
failing step (an enormous proposed Newton correction -- a near-singular-
Jacobian signature) despite a small function norm. Pattern-matched
(WRONGLY, see below) to `newton_ta.py`'s documented "spurious reason=-6
failures once already close to the fixed point" -- tried loosening
`snes_rtol`/`snes_atol` (did not help, failed even earlier) and then
tried `snes_linesearch_type="basic"` (no globalization, always accept
the full raw Newton step, relying on the existing outer `step_relax`
Python-level damping instead).

**`basic` appeared to work spectacularly**: smooth, monotonic SCIF
convergence (e.g. 686->714->743->...->854mT with visibly decaying
increments) reaching a defined stall criterion. Damping-strength
insensitivity check (`step_relax`/`jc_n_relax` swept 0.2/0.3/0.4/0.5) gave
essentially IDENTICAL converged values (1158.62/1158.66/1158.70/1158.68
mT, spread 0.08mT) -- the textbook signature of a genuine fixed point,
NOT the damping-dependent-artifact pattern that proved
`monolithic_ta.py`'s own earlier undifferentiated attempt was a scheme
artifact (it converged to +6800/+3669/+2012mT at three different damping
strengths). An 8-repeat independent-process batch at fixed settings
(dt=60s, I=19.6A, `linesearch_type=basic`, `step_relax=jc_n_relax=0.3`)
gave **8/8 converged**, SCIF spread **+1158.667 to +1158.697 mT (0.03mT,
0.003% relative)** -- versus this project's own historical ~20-40% Picard
success rate with a 400+mT spread even among "successes" (-61.74,
+374.66, +258.37, +63.63 mT from four Picard baseline convergences run
this same session), and versus 0/18 for genuinely single-threaded
execution (this session's separate nondeterminism finding).

**This was wrong, and here is exactly how it was caught.** Testing the
historically-hardest documented case (dt=100s, I=32.667A -- the one
Picard could not converge even at 1000+ iterations) with
`build_monolithic_problem_diff(..., verbose=True)` (which turns on
`snes_monitor`, printing the RAW SNES function norm every step -- NOT
enabled during the "successful" 8/8 batch, which only ever watched SCIF)
showed the true residual EXPLODING EXPONENTIALLY every single outer
iteration: `1.25e62 -> 5.98e64 -> ... -> 7.0e96` by iteration 44, while
SCIF calmly plateaued at +1931.31mT and the code's own SCIF-stall
criterion declared `converged=True`. Re-checking the ORIGINAL "successful"
dt=60s/I=19.6A case with the monitor enabled showed the IDENTICAL
pathology: function norm `1.25e33 -> 5.8e35 -> ... -> 4.7e95` over the
same 43 iterations that "converged" to +1158.67mT.

**The `-6` failures were correct.** `bt`'s backtracking line search was
legitimately detecting that the raw Newton step does not decrease the
residual and refusing to take it. Switching to `basic` did not fix
anything -- it disabled the one mechanism that was correctly catching a
genuine divergence, and let the iteration run to a point where a DERIVED
diagnostic (SCIF, borrowed from the Picard scheme, where it is meaningful
because Picard genuinely damps toward a fixed point) coincidentally
plateaued while the actual PDE residual diverged to physically
meaningless values. The 8/8 reproducibility and the clean damping-
insensitivity sweep were both real, measured facts -- and both
completely consistent with "SCIF's drift rate under this particular
divergence is insensitive to damping strength," not with "a real fixed
point was found." **The entire finding is retracted.**

## Part 4: chasing the real cause -- two remedies tried, both ruled out

**Remedy 1: Levenberg-Marquardt / Tikhonov regularization.** Added an
optional proximal term `+ lm_lambda*inner(T_i - T_anchor_i, phi)*dx` to
each layer's residual (`T_anchor_i` a snapshot refreshed immediately
before each `problem.solve()` call -- vanishes exactly at any point where
`T_i == T_anchor_i`, so it cannot bias a genuine fixed point, only lifts
the Jacobian's diagonal). Swept `lm_lambda` in {0, 1e3, 1e6, 1e12, 1e18}
-- an 18-order-of-magnitude range -- using the LEGITIMATE `bt` linesearch
and watching the raw residual directly this time. **Result: statistically
indistinguishable blowup trajectories at every lambda value tested**
(~100-170x growth in iteration 2, ~80-165x in iteration 3, plateau
~2.2-3.2e9, failure by iteration 5-7, regardless of lambda). A
regularization term with zero measurable effect across 18 orders of
magnitude means the diagnosis (a fixable near-singular direction curable
by isotropic Tikhonov damping) was likely wrong, not that the magnitude
was mistuned.

**Remedy 2: a much better bootstrap.** Tested whether the divergence was
an artifact of starting Newton from a rough, 30-iteration-CAPPED Picard
seed (far from any true root) rather than a property of the system near
the true solution. Ran a 150-iteration bootstrap instead (5x longer;
note this ALSO did not fully converge on its own -- "CAP in 150 iters" --
consistent with this exact case's documented unreliability). **Result:
statistically identical blowup** (176x, 117x, 1.33x jump, plateau
~2.4e9, failure at iteration 6) to the 30-iteration bootstrap. Bootstrap
quality is not the cause either.

## Part 5: the real finding -- this predates today's work entirely

Directly instrumented the ORIGINAL, unmodified `monolithic_ta.py`
(frozen Jc/n, nothing touched by this investigation) with the same
residual-monitoring diagnostic, from the identical 30-iteration bootstrap
seed. **Identical pathology**: 98x, 153x, 1.4x jump, plateau at
**2.365699e9 -> 2.397960e9** (barely moving, NOT decreasing toward zero)
across 15 iterations, `reason=-5` (DIVERGED_MAX_IT, expected under the
max_it=1 scheme) throughout -- never hitting -6 in this particular run,
but also never actually converging in any residual sense. A residual
plateaued at 2.4e9 is not a solved nonlinear system by any standard
definition.

**This is not a defect introduced by making Jc(B)/n(B) differentiable.**
It is a pre-existing property of the monolithic block-Newton scheme
itself, present in the version of `monolithic_ta.py` that has been in
this repository since before today. CLAUDE.md documents that
`monolithic_ta.py` "converges cleanly... to three different wrong
answers at three different damping strengths (+6800/+3669/+2012mT
against a +641mT truth)" and concludes this proves the plateau is "a
scheme artifact, not a slow-but-correct answer" -- but that conclusion,
like today's retracted one, was reached by watching SCIF/stall behavior,
not the raw PETSc residual. **It is now a real, open question whether
`monolithic_ta.py`'s own historical "convergence" claims were ALSO a
false-positive SCIF-stall signal riding on top of an equally-diverging
residual**, never checked because nobody had instrumented `snes_monitor`
against it before this session. This has NOT yet been verified either
way for the historical dt=600s/full-current case that produced those
numbers -- only for the dt=60s/I=19.6A short-dt case checked today.

## Status and what's next

**Not fixed.** Two natural remedies (Tikhonov regularization, better
bootstrap) are ruled out by direct evidence, not abandoned by assumption.
The real locus is now understood to be a structural property of the
monolithic block-Newton residual/Jacobian itself -- independent of
whether Jc/n are frozen or differentiable -- most likely involving how
PETSc's line search/convergence machinery handles a block system whose
sub-equations (T-layers vs. A) may have very different natural scales, or
a genuine bug in the T-A coupling term assembly that has never been
caught because no one checked the raw residual before. Candidate next
steps, in rough priority order: (1) retroactively check whether
`monolithic_ta.py`'s own historically-reported dt=600s "converged"
results show the same residual-plateau pathology -- if so, this
invalidates a piece of this project's standing history, not just today's
new attempt; (2) inspect the assembled block Jacobian directly (condition
number, per-block diagonal scale spread) rather than inferring its
badness indirectly from linesearch behavior; (3) consider whether the
combined multi-field residual norm PETSc's line search judges against is
itself the problem (dominated by whichever block has the largest natural
scale) and whether a block-wise or scaled convergence criterion behaves
differently; (4) as a more radical alternative, abandon the monolithic
architecture for this purpose and investigate whether a Gauss-Seidel
scheme with a SEPARATE, small monolithic-like T-A sub-solve restricted to
just the coupling terms could isolate the live-B-dependence benefit
without inheriting the full block system's apparent fragility.

**What is solid and reusable regardless of how the above resolves**:
`physics/entropy_ic_model.py`'s two fitted models (validated fit quality,
verified analytic derivatives, correct UFL wiring including the fixed
unit bug) are correct, general-purpose infrastructure, independent of
which solver architecture eventually consumes them.

---

## 2026-08-05 (overnight continuation): the historical +6800/+3669/+2012mT result is CONFIRMED to be the same false-positive signature -- a retraction, not just a closed-out investigation

Picking up this file's own final "what's next" item #1. Reproduced the
EXACT historical regression case from `docs/HISTORY.md`'s "fully
monolithic block-Newton T-A" entry (dt=`params.ramp_duration`=600s,
I=`params.I_design`=196A, cold start, `monolithic_ta.py` -- the ORIGINAL,
frozen-Jc/n file, nothing from today's differentiable work involved) using
the project's own existing `transient/validation/monolithic_step_relax_
sweep.py` configuration, but with `snes_monitor` enabled this time
(new script: `transient/validation/monolithic_historical_residual_check.py`).

**Both tested damping levels reproduce their historical SCIF values
closely (confirming a faithful re-run, not a different setup by
accident) while showing the SAME residual-explosion-then-plateau
signature already found and retracted for today's short-dt case:**

| step_relax | SCIF here | SCIF in docs/HISTORY.md | \|\|F\|\| trajectory |
|---|---|---|---|
| 0.1 | +3664.15 mT (k=34, then reason=-6) | +3669 mT (k=60 "clean plateau") | 1.4e5 -> 6.96e6 (49.5x) -> 3.15e8 (45.3x) -> 3.22e9 (10.2x) -> plateau 3.27-3.29e9 for 30 more iterations, ratio 0.9995-1.0007 throughout, NEVER decreasing |
| 0.03 | +1990.52 mT (k=23, then reason=-6) | +2012 mT (k=45 "clean plateau") | 1.45e5 -> 2.11e6 (14.5x) -> 3.20e7 (15.2x) -> 4.53e8 (14.1x) -> 3.51e9 (7.7x) -> plateau 3.93-3.96e9 for 18 more iterations, ratio 0.9996-1.0017 throughout, NEVER decreasing |

**Verdict: CONFIRMED, not just plausible.** `docs/HISTORY.md`'s own
"clean asymptotic plateau" language describes a residual sitting flat at
3.3-4.0 BILLION -- not decreasing toward zero by any amount, let alone to
a converged value -- while SCIF alone climbed smoothly and looked
converged. This is the identical false-positive mechanism documented
earlier in this file for today's differentiable-Jc/n attempt, now shown
to also underlie the ORIGINAL project conclusion that "the plateau is a
numerical artifact of the coefficient-refresh-every-step scheme, not a
slow-but-correct answer" -- that conclusion's PREMISE (three genuinely
different converged states) does not hold: none of the three damping
levels ever converged in any residual sense. The three different SCIF
plateau VALUES are real and reproducible, but they are three different
points along three DIFFERENT DIVERGING trajectories that happen to have
their SCIF-drift-rate slow down enough to trip an EMA-stall criterion,
not three different fixed points of the same equation.

**This is a correction to this project's standing documented history,
not merely a closed-out side investigation.** `CLAUDE.md`'s "NI transient
work" section and `docs/HISTORY.md`'s 2026-08-05 "fully monolithic
block-Newton T-A" entry should be read with this in mind: the monolithic
architecture was never shown to produce even a self-consistent wrong
answer, let alone the right one -- it has never been observed to
genuinely converge (raw residual -> small) at ANY damping level, at
EITHER the dt=600s/I=196A production regime or the dt=60s/I=19.6A short-
dt regime, with EITHER frozen or differentiable Jc/n. Recommend the
parent session update CLAUDE.md accordingly when reviewing this file.

**What this changes about where the real problem lives**: since this
reproduces at the FULL production scale (196A, the actual validated
operating current, dt=600s, the one single-step regime this project has
ever trusted) and not just at today's low-current short-dt test case,
the failure is not specific to short-dt marching, low current, or
differentiable coefficients. It is a property of the monolithic block-
Newton formulation ITSELF, present across the whole range of operating
conditions tried. This raises the prior for a genuine implementation-
level or structural-scaling bug (checked next, Part 2 below) over a
purely physics-driven explanation (e.g. "the power-law is just too stiff
for any Newton scheme") -- a physics-driven cause would be expected to at
least look different in severity between a gentle, well-inside-the-
validated-regime case and an aggressive short-dt one; instead both look
identical.

---

## 2026-08-05 (overnight continuation, part 2): ROOT CAUSE FOUND -- a Dirichlet-BC block-scoping bug present since `monolithic_ta.py` was first written

Directly inspected the assembled block Jacobian (new scripts:
`transient/validation/monolithic_jacobian_inspect.py`,
`monolithic_jacobian_inspect2.py`) at a point a few outer iterations into
the standard dt=60s/I=19.6A repro case, on the ORIGINAL `monolithic_ta.py`
(frozen Jc/n -- confirmed present there first, so this is not an artifact
of today's differentiable work). Confirmed `Jmat.getType() == "seqaij"`
(a genuine monolithic AIJ matrix, not a MATNEST whose naive CSR extraction
could have been misleading) before trusting anything below.

**Finding 1: every T-A off-diagonal coupling block (dF_A/dT_i and
dF_Ti/dA, all 6 layers) is EXACTLY zero** -- not small, exactly 0.000000,
despite the sparsity pattern allocating 72012 nonzero slots per block.
`monolithic_ta.py`'s own docstring claims this Jacobian "includes the TRUE
cross-coupling terms dF_A/dT_i and dF_Ti/dA every single Newton step" --
the assembled matrix shows none of that coupling actually exists.

**Finding 2: every T-layer's diagonal block is uniformly, exactly 1.0** --
not just on the Dirichlet-pinned majority of dofs (expected, correct
behaviour for a BC row), but on EVERY SINGLE ONE of the 180 dofs layer 0's
own BC construction intended to leave FREE (a real unknown, where the
physical `rho*inner(J,J_test)` term should dominate). Sample free-dof rows
contain only entries in {0.0, 1.0} -- not the rich, widely-varying values
(rho spans ~1e-4 to astronomically large across this project's own
documented power-law regime) a real physics row would show.

**Root cause, confirmed directly, not inferred:** `ta_solve.setup_ta_problem`
builds ONE shared CG1 function space `V_T` and every layer's `T_i =
fem.Function(V_T, ...)` reuses it (`transient/validation/bcshare_check`-
style probe, run inline, not saved as this exact name -- reproduce via
the snippet embedded in this section's git history if needed: confirmed
`all(T_i.function_space is ta["V_T"] for T_i in layer_T_fns)` is True).
Each layer's own BC list (`ta["layer_bcs"][i]`) correctly pins "every dof
NOT in layer i's own coil cells" to zero, referencing dof indices in that
SHARED V_T numbering. `dolfinx.fem.bcs.DirichletBC` objects carry only
`(function_space, dof_indices, value)` -- NO reference to which specific
`Function`/unknown-block they were constructed for. `monolithic_ta.py`
builds its combined BC list for the whole 7-block system as a flat
concatenation:
```python
bcs = [bc for layer_bcs_i in ta["layer_bcs"] for bc in layer_bcs_i]
```
handed once to `NonlinearProblem(F_list, u_list, bcs=bcs, kind="mpi")`
with NO per-block scoping. Because all six layers share one function
space, dolfinx cannot restrict a given BC to "only T_0's block" -- it
applies EVERY layer's "pin everything outside my own cells" BC to EVERY
T-block. Directly verified: **all 180 of layer 0's own intended-free dofs
are ALSO listed in the union of layers 1-5's pinned sets** (each real dof
genuinely belongs to only one layer's own coil cells, so it is "outside"
the other five layers by construction) -- so in the assembled monolithic
system, each T-block ends up almost entirely Dirichlet-pinned by dofs
that were never meant to constrain it, wiping out the real per-layer
physics (and, since a Dirichlet row is a pure identity row by
construction, also wiping out that row's T-A coupling entries -- this is
the SAME bug producing BOTH findings above, not two separate bugs).

**This fully and mechanistically explains the residual-blowup pathology**
confirmed today across every configuration tried (frozen and
differentiable Jc/n, dt=60s/I=19.6A and the historical dt=600s/I=196A
production case, multiple damping strengths): a Newton step computed
against a Jacobian that is ~90%+ trivial identity (with the actual
nonlinear physics almost entirely absent) bears little relationship to
the TRUE physical residual, so accepting it (at any damping strength,
since damping only rescales a systematically-wrong direction) drives the
state away from any real solution. It also explains why three different
damping strengths produced three different plateaus: each is a different
point along a trajectory dominated by this structural error, not three
different attempts to approach the same true fixed point.

**This is a bug in `monolithic_ta.py`, present since that file was first
written (2026-08-05, earlier in this project's own day) -- not something
introduced by today's differentiable-Jc/n work, and not evidence against
the monolithic ARCHITECTURE as an idea.** `transient/monolithic_ta_diff.py`
inherited it unchanged (it builds `bcs` the identical way). Neither file's
"converges cleanly to different wrong answers" or "residual explodes"
behaviour is evidence about whether a genuinely-coupled monolithic T-A
Newton system could work -- one has never actually been assembled and
tested; what's been tested and failed is this specific, fixable BC bug's
consequences.

**Path to an actual fix, not yet implemented/tested as of this
checkpoint:** each layer needs a function space that is DISTINCT as a
Python object (even though mathematically/dofmap-identical to every other
layer's), so a `DirichletBC`'s `function_space` check can correctly scope
it to one block only. This means either (a) building 6 separate
`fem.functionspace(domain, elem_T)` calls (one per layer) instead of the
one shared `V_T` `ta_solve.setup_ta_problem` currently constructs, or (b)
some dolfinx-supported mechanism for scoping a BC to a specific
sub-block of a block `NonlinearProblem` that does not require distinct
function space objects, if one exists (not yet checked). Option (a) is a
change to shared infrastructure (`ta_solve.py`) that EVERY solver in this
project depends on (Picard, `newton_ta.py`, both monolithic files) -- it
must not break the Gauss-Seidel/production path, which currently works
correctly specifically BECAUSE it solves each layer's `LinearProblem`
SEPARATELY, one at a time, each seeing only its own BC list on its own
call (no flat-concatenation, no cross-contamination possible there,
confirmed by inspection of `ta_solve.py`'s `prob_T_layers` construction:
each layer's `LinearProblem` is built with `bcs=ta["layer_bcs"][i]` for
that `i` only). Attempting the fix next, in a way that does not touch the
existing, working per-layer LinearProblem construction.

---

## 2026-08-05 (overnight continuation, part 3): the fix is implemented as a proof-of-concept and VERIFIED STRUCTURALLY CORRECT

New files: `transient/validation/monolithic_fixed_bc_test.py` (builds a
corrected monolithic block system inline, giving each layer a DISTINCT
`fem.functionspace(domain, elem_T)` object instead of reusing
`ta_solve.py`'s single shared `V_T`, so each layer's Dirichlet BC list is
correctly scoped to only its own block -- mirrors `monolithic_ta.py`'s
frozen-Jc/n math exactly, does not touch `ta_solve.py` or any existing
file) and `monolithic_fixed_bc_verify.py` (re-runs the same Jacobian
inspection from Part 2 against the corrected system).

**Structural fix CONFIRMED working, directly, not inferred:**

| quantity | buggy (Part 2) | fixed (this section) |
|---|---|---|
| T_0 diagonal, count exactly 1.0 | 1718/1718 (ALL dofs, including the 180 meant to be free) | **1538/1718 -- matches the TRUE pinned-dof count exactly** |
| T_0 diagonal on the 180 genuinely free dofs | 1.0, zero variance | **2.05e-19 to 3.24e+05, mean 2518 -- rich, physically-varying, matching this project's known rho power-law dynamic range** |
| dF_A/dT_i (all 6 layers) | exactly 0.0, every entry | **nonzero: max magnitude 1.7e-5 to 2.3e-5, 2300-7100 genuinely nonzero entries per block** |
| dF_Ti/dA (all 6 layers) | exactly 0.0, every entry | **nonzero: max magnitude 7.8e-7 to 2.1e-5, similarly populated** |

This is a real, structural fix, not a numerical coincidence -- the exact
match between "number of dofs showing diagonal=1.0" and "number of dofs
this layer's OWN BC construction intends to pin" is the clean confirming
signature.

**However: plugging the SAME warm-start and the SAME damping parameters
(`relax=0.3`, `eps_reg`, etc. -- all tuned against the OLD, effectively-
decoupled buggy system) into the now-CORRECTLY-coupled system does NOT
produce clean convergence.** `||F||` starts far larger (~1.9e14, vs. the
buggy system's ~1e5) -- expected, since the true coupling terms (real
physics, previously zeroed) now contribute to the residual for the first
time -- and evolves erratically over 25 iterations: not the clean,
monotonic explosion the buggy version showed, but a chaotic, non-
monotonic trajectory including large drops (e.g. iteration 20: ratio
0.0006, a near-3-order-of-magnitude single-step DECREASE) as well as
large jumps (iteration 18: ratio 47x). This qualitative CHANGE in
character (trivial monotonic runaway -> genuinely chaotic, physics-
reflecting dynamics) is itself further evidence the fix engaged real
coupling that was previously entirely absent -- but it does not, on its
own and with unmodified tuning, solve the convergence problem.

**Honest interpretation:** the BC bug and the convergence problem are
now understood to be TWO SEPARATE THINGS that happened to look identical
from the outside (both produced "the residual explodes / doesn't
converge"). Fixing the BC bug does not automatically fix convergence --
it replaces "a trivially-wrong system that (unsurprisingly) can't be
trusted" with "a genuinely, correctly assembled but still very stiff and
chaotic coupled nonlinear system," which may or may not be tractable with
different tuning. This is real, separate progress (a previously-hidden
implementation bug, now understood and fixable), not a solved convergence
problem.

**Not yet done, in priority order for continuing tonight or tomorrow:**
1. A damping/relaxation sweep on the CORRECTED system specifically (the
   existing `step_relax`/`jc_n_relax` values were never tuned for genuine
   T-A coupling and there is no reason to expect they transfer).
2. If (1) does not find clean convergence quickly, check whether the
   corrected system converges from a DIFFERENT, gentler starting point
   (e.g. a much smaller current/dt than this project's hardest cases, to
   establish whether the corrected monolithic system converges AT ALL
   anywhere before concluding it's still fundamentally too stiff).
3. Wire the fix into `monolithic_ta.py`'s and `monolithic_ta_diff.py`'s
   actual `build_monolithic_problem`/`build_monolithic_problem_diff`
   functions (currently only a standalone proof-of-concept in
   `transient/validation/`, not integrated into the production files) --
   worth doing only once (1)/(2) show the corrected system is actually
   worth using.
4. Update `CLAUDE.md`'s "NI transient work" section and flag
   `docs/HISTORY.md`'s 2026-08-05 "fully monolithic block-Newton T-A"
   entry as based on an unverified premise (see Part 2's retraction) --
   left for the parent/user to review and apply, not done here since it
   touches project-wide standing documentation outside this file's scope.

---

## 2026-08-05 (overnight continuation, part 4): damping sweep on the corrected system, and where this stands for the morning

Tried `step_relax=0.05` (much gentler than the historical 0.1/0.03, given
the corrected system's initial residual is orders of magnitude larger
than the buggy system's) on top of the structural fix. **Result: the
residual STOPS exploding and stabilizes -- but at a plateau of ~3.7e14,
not a decrease toward zero** (25 iterations, ratio hovering 0.987-1.011,
essentially flat/noisy, no trend). Gentler damping tames the WILD swings
seen at `step_relax=1.0` (which included a 47x jump and a 0.0006x crash
within the same 25-iteration run) but does not, by itself, produce
convergence.

**Checked whether this enormous residual scale is present even before
any Newton step is taken** (i.e. is it a property of the warm-started
state itself, not an artifact of a bad first step) -- the direct
`snes.computeFunction()` call on the freshly-built problem's initial
state hit the same PETSc segfault pattern noted earlier in this file
(SNES internal vectors need a prior `solve()`/`setUp()` call before
`getFunction()`/`getSolution()` are safe to use) and was not chased
further given the time already spent tonight. Indirect evidence still
points the same direction: `||F||` after just ONE Newton step is already
1.9e14-3.8e14 across every run tried, regardless of damping -- consistent
with (not proof of) the true residual at the warm-started state already
being enormous once the previously-missing coupling terms are correctly
included, rather than something a bad first Newton direction manufactures
from a small starting point.

**A live, unresolved hypothesis for tomorrow, flagged but not tested
tonight:** the T-blocks' diagonal (mean ~2500-5700, verified in Part 3)
and the A-block's diagonal (mean ~1.19e10, max 4.32e11, also from Part 3)
differ by roughly 6-8 orders of magnitude. A single combined residual
norm and a single direct LU factorization treating all 7 blocks as one
undifferentiated system (which is what `kind="mpi"` + a plain `pc_type
lu` currently does) has no way to "know" these blocks live at wildly
different natural scales -- this is exactly directive item 3 from
tonight's brief (PETSc `PCFIELDSPLIT` / block-scaled preconditioning),
not yet tried. This is a genuinely different, well-motivated next lever
from anything tried tonight, and a natural place to pick back up.

## SUMMARY FOR THE MORNING

**What changed tonight, in order of confidence:**

1. **CONFIRMED (multiple independent checks, mechanistically understood,
   not inferred): `monolithic_ta.py` has had a real Dirichlet-BC block-
   scoping bug since it was first written.** All 6 layers' T functions
   share one function space object; `dolfinx.fem.bcs.DirichletBC` cannot
   disambiguate which block a BC targets when function spaces are shared;
   the combined BC list therefore applies every layer's "pin outside my
   own cells" condition to every layer's block, collapsing each T-block's
   Jacobian to near-pure identity (verified: exactly the pinned-dof count
   shows diagonal=1.0, INCLUDING dofs that specific layer's own
   construction intended to leave free) and zeroing every T-A coupling
   block entirely (verified: exactly 0.0, not just small, in every one of
   12 off-diagonal blocks). `transient/monolithic_ta_diff.py` (today's
   differentiable-Jc/n work) inherited this unchanged.

2. **CONFIRMED (two independent historical configurations re-run with
   `snes_monitor`, both reproducing their documented SCIF values closely):
   this bug's consequence -- a residual that explodes then plateaus at an
   enormous, non-decreasing value while a downstream SCIF diagnostic looks
   converged -- is present in `docs/HISTORY.md`'s own documented
   dt=600s/I=196A "monolithic block-Newton" result
   (+6800/+3669/+2012mT at three damping strengths). That result's own
   stated conclusion ("the plateau is a scheme artifact, not a slow-but-
   correct answer, proven by three genuinely different converged states")
   rests on a premise -- genuine convergence at each damping level -- that
   the raw residual shows never held. `CLAUDE.md` and `docs/HISTORY.md`
   should be corrected to reflect this; not done in this file, since it
   touches project-wide standing documentation outside a `transient/`
   working file's scope -- flagged clearly for the user/parent session to
   action.

3. **IMPLEMENTED AND STRUCTURALLY VERIFIED (not yet integrated into
   production code): a fix.** Giving each layer a distinct function
   space object (rather than sharing `ta_solve.py`'s single `V_T`)
   restores genuine per-layer physics to the T-diagonal (verified: exact
   match between "dofs at diagonal=1.0" and the TRUE pinned-dof count)
   and genuine nonzero T-A coupling (verified: every one of 12 blocks now
   has thousands of real nonzero entries). This is a real, mechanistic
   fix for the bug in (1), proof-of-concept only in
   `transient/validation/monolithic_fixed_bc_test.py` /
   `_verify.py` -- NOT wired into `monolithic_ta.py`'s or
   `monolithic_ta_diff.py`'s actual `build_monolithic_problem*` functions.

4. **STILL OPEN, genuinely unresolved: whether the CORRECTLY-coupled
   monolithic system can be made to converge at all.** With the fix
   applied, the residual is far larger from the first step (~1e14-3.8e14,
   vs. the buggy system's ~1e5) and, under the old damping parameters,
   behaves chaotically (wild swings at `step_relax=1.0`; a stable-but-
   non-decreasing plateau at `step_relax=0.05`). This is NOT evidence the
   fix is wrong (the fix is independently verified structurally correct
   in item 3) -- it means the OLD tuning, calibrated against a trivially-
   broken decoupled system, does not transfer, and the genuinely-coupled
   system's own convergence behavior has not yet been properly
   characterized. The leading untested hypothesis is a severe (6-8 order
   of magnitude) natural-scale mismatch between the T-blocks and the
   A-block, for which PETSc `PCFIELDSPLIT` / block-scaled preconditioning
   is the natural next tool, not yet tried.

**Bottom line: a real, previously-undiscovered, historically-significant
bug is found and fixed at the structural level -- this is genuine
forward progress and should be treated as such -- but "does a correctly-
built monolithic T-A Newton solve actually converge" is now an OPEN
question again, not a closed "no" (Part 2's finding) nor a closed "yes"
(nothing has converged yet either). Recommend continuing with the
field-split/preconditioning direction before concluding either way, and
separately, updating the project's standing documentation
(`CLAUDE.md`/`docs/HISTORY.md`) to reflect item 2's retraction regardless
of how the open question above eventually resolves.**

---

## 2026-08-05 (overnight continuation, part 5): quick scale-mismatch test, then stopping for the night

Cheap test of the scale-mismatch hypothesis from Part 4, before committing
to a full `PCFIELDSPLIT` implementation: MUMPS's built-in automatic matrix
scaling (`-mat_mumps_icntl_8 77`), on the structurally-fixed system, with
gentle `step_relax=0.05` damping. **Result: modest improvement, not a
fix.** Plateau drops from ~3.7e14 (Part 4, no scaling) to ~1.6-1.8e14 (this
test, with scaling) -- roughly 2x -- but still shows no decreasing trend
over 20 iterations (ratios scattered 0.91-1.10, noisy, flat). This
suggests the T-block/A-block scale disparity is a REAL contributing factor
(scaling helped, in the right direction) but likely not the SOLE or even
dominant cause -- a problem that was purely about matrix scaling would be
expected to respond much more dramatically to a well-chosen scaling
strategy than a 2x reduction that still leaves the residual 14 orders of
magnitude away from anything resembling convergence.

**Stopping here for the night.** This is a reasonable point to hand off:
further progress would need either (a) a properly wired `PCFIELDSPLIT`
with per-field Schur-complement or block-Jacobi treatment (a real
implementation effort, not a quick options-database test like the MUMPS
scaling check above), or (b) accepting that the corrected monolithic
system may simply be too stiff for Newton-type methods regardless of
correct assembly, and that this project's existing standing
recommendation (T-A + Gauss-Seidel iteration, not a monolithic
reformulation) may turn out to still be the right call -- but for a
DIFFERENT reason than previously documented (not "monolithic converges
to scheme-dependent wrong answers," which Part 2 retracted, but possibly
"a correctly-assembled monolithic system is well-posed but too stiff for
practical Newton globalization," which is a materially different and
more defensible claim, still not yet fully established either).

**Final status for whoever reads this in the morning:** the historically
significant finding (Parts 1-3: a real BC bug, present since
`monolithic_ta.py`'s creation, now understood and fixable, invalidating
this project's prior "monolithic converges to 3 different wrong answers"
conclusion) is solid and should be acted on regardless of how the
remaining convergence question resolves. The remaining question (can a
correctly-built monolithic T-A Newton system actually be made to
converge) is genuinely open, with one avenue (field-split
preconditioning) flagged as the most promising untested lever and one
cheap proxy for it (matrix scaling) already tested with modest,
inconclusive results. No file in this project should currently cite a
"monolithic block Newton converges" OR "monolithic block Newton
definitively cannot converge" claim -- both would currently be overclaims
in opposite directions.

---

## 2026-08-05 (overnight continuation, part 6, coordinator-directed): real PCFIELDSPLIT implemented and tested -- inconclusive/negative, with real PETSc engineering findings along the way

Coordinator (parent session) independently verified Part 1-3's BC-bug
finding directly (re-read `ta_solve.py` lines 117/330 confirming the
shared-`V_T` root cause), merged all files, and corrected `CLAUDE.md`
(a dated retraction of the "converges cleanly to three wrong answers"
claim) -- that documentation work is DONE, not duplicated here. Directed
continuation: implement REAL `PCFIELDSPLIT` (not the MUMPS-auto-scaling
proxy from Part 5), block-Jacobi (`additive`) first, Schur if that
doesn't move the needle, same residual-based rigor throughout.

New file: `transient/validation/monolithic_fieldsplit_test.py`.

### Key API findings (real engineering, worth keeping regardless of the final verdict)

1. **dolfinx's automatic field-split IS transfer requires BOTH
   `kind="nest"` AND a separate preconditioner form `P`** (read directly
   from `dolfinx/fem/petsc.py`: the `if kind == "nest" and self.P_mat is
   not None:` guard). `monolithic_ta.py`/`monolithic_ta_diff.py` pass
   neither `P` nor `kind="nest"` -- automatic wiring never applies to
   them; field-split ISs must be built and attached manually
   (`pc.setFieldSplitIS(...)`).

2. **Options passed through `NonlinearProblem`'s own `petsc_options` dict
   cannot configure per-split sub-solvers.** dolfinx pushes them onto the
   global options database, calls `setFromOptions()` ONCE during
   `__init__`, then immediately deletes every key and pops the prefix --
   but per-split sub-KSP/sub-PC objects do not exist yet at that point
   (created lazily inside `PCSetUp`, itself deferred to the first real
   solve). Confirmed directly: passing `fieldsplit_ksp_type`/
   `fieldsplit_pc_type`/`fieldsplit_pc_factor_mat_solver_type` this way
   produced PETSc's own "options set but not used" warning naming exactly
   those keys, and `PC.view()` showed every split silently using PETSc's
   own built-in default (`ksp_type=preonly`, `pc_type=ilu`) instead.
   **Fix: configure each sub-KSP directly via the Python API**
   (`pc.getFieldSplitSubKSP()`, then `.setType()`/`.getPC().setType()`/
   `.getPC().setFactorSolverType()` on each) -- but only AFTER the first
   real `problem.solve()` call, since the underlying matrix blocks are not
   assembled before that (`sub_ksp.setUp()` or even `pc.setUp()` called
   earlier hits PETSc error 73, "Not for unassembled matrix" /
   "MAT_COPY_VALUES not allowed for unassembled matrix").

3. **`kind="nest"` is incompatible with `PC.SchurPreType.SELFP`'s
   Schur-complement approximation for a multi-block combined field.**
   Grouping all 6 T-layer blocks into one combined "T" field (needed for
   a 2-field Schur split against "A") keeps that field internally
   `MATNEST`-structured when the parent matrix is nest; SELFP's
   approximation needs a `MatMatMult` between two such blocks, which
   PETSc does not support ("Unspecified symbolic phase for product AB
   with A nest, B nest. The product is not supported"). **Fix: use
   `kind="mpi"` (a genuine monolithic AIJ matrix) instead, with
   MANUALLY-constructed `PETSc.IS` objects** (via `IS().createStride()`
   from known dof offsets -- no `getNestISs()` available or needed for a
   plain AIJ matrix). This is, in fact, the more standard way
   `PCFIELDSPLIT` is used in practice.

### Results, block-Jacobi (`additive`) field-split, properly configured (verified via `PC.view()`: every split genuinely uses `preonly`+`lu`+MUMPS, not the silent ILU default)

**Real signal, but NOT reliable.** Multiple independent runs of the
IDENTICAL configuration (dt=60s, I=19.6A, `jc_n_relax=0.3`,
`step_relax=0.3`):
- Run 1 (3-iteration check): `||F||` ratios 0.9853, 0.9350 -- a genuine,
  consistent DECREASING trend, the first of the entire investigation
  (everything else either exploded, plateaued flat, or failed outright).
  Linear (KSP) residual within one Newton step dropped ~8 orders of
  magnitude in a single `fgmres` iteration (2.6e14 -> 2.16e6) --
  confirms the preconditioner itself is doing something substantial.
- Run 2 (reproducibility check, same config): ratio 0.9702 at k=1, but
  FAILED outright (`reason=-3`, KSP hit its 200-iteration cap) at k=3.
- Run 3 (another reproducibility check, same config): failed outright
  (`reason=-3`, `ksp_its=0`) on the very FIRST iteration -- no decrease
  observed at all.
- A 40-iteration attempt (same config) also failed immediately on
  iteration 1.

**Verdict: this is the same class of cross-process chaos documented
throughout this entire investigation (this file's Parts 1-5, and the
separate `nondeterminism_investigation_2026-08-05.md`), not a reliable
fix.** The IDENTICAL nominal configuration produces genuinely different
outcomes -- occasionally several iterations of real, substantial
residual decrease; more often an outright failure within 1-3 iterations.
A preconditioner that sometimes works and sometimes doesn't, on the
identical input, for reasons not yet isolated, is not a working fix by
this project's own standard (multiple independent repeats required
before trusting any convergence claim) -- it is, however, the closest
anything has come to genuine decrease in this entire multi-session
investigation, and worth flagging as a real, if fragile, positive
signal for whoever continues this.

### Results, Schur complement

Hit the `kind="nest"` + SELFP MatMatMult limitation (finding 3 above),
fixed by switching to `kind="mpi"` + manual ISs -- but even after that
fix, the FIRST solve (using PETSc's own defaults, before this file's
deferred sub-solver configuration can run) fails outright
(`reason=-3`, `ksp_its=0`) before any useful signal is obtained. Given
the "configure after first solve" pattern this file uses depends on that
first solve succeeding well enough to trigger real assembly, and it does
not here, resolving this would need a different bootstrapping strategy
(e.g. configuring the Schur sub-solve's approximation BEFORE the very
first solve via a route that doesn't require already-assembled data, or
pre-assembling the matrix once via a dummy/throwaway solve under a
DIFFERENT, working PC before switching to the real Schur configuration).
**Not pursued further tonight** -- per the coordinator's own explicit
"legitimate, honest stopping point if PCFIELDSPLIT doesn't resolve it
within reasonable effort" allowance, and given the substantial time
already spent on real, documented PETSc-API engineering (findings 1-3
above are genuine, reusable knowledge regardless of the inconclusive
final outcome).

## REVISED FINAL SUMMARY FOR THE MORNING (supersedes Part 4's summary; Parts 1-3's BC-bug finding is unchanged and already actioned by the coordinator)

**Unchanged and solid:** the Dirichlet-BC block-scoping bug (Parts 1-3)
is real, confirmed, and already corrected in the project's standing
documentation by the coordinator.

**The open convergence question is STILL open, now with more precise
information about why it's hard:**
- A correctly block-Jacobi-preconditioned version of the bug-fixed
  monolithic system CAN, sometimes, show several iterations of genuine,
  substantial residual decrease -- the first time this entire multi-
  session investigation has observed that at all, in any configuration.
- But this is NOT reproducible on demand: identical configurations,
  launched as separate processes, sometimes show this decrease and
  sometimes fail immediately with zero linear-solver progress. This
  matches the SAME cross-process floating-point-sensitivity chaos this
  project has documented extensively elsewhere (see
  `nondeterminism_investigation_2026-08-05.md`), now apparently present
  even in the CORRECTED, properly-preconditioned system, not just the
  buggy one.
- Schur-complement fieldsplit (the more principled treatment for a
  system with real T-A coupling, as this one now genuinely has) was not
  successfully gotten working tonight -- real, fixable PETSc API/
  bootstrapping obstacles were found and partially resolved, but time ran
  out before a working configuration was reached.

**Recommended next steps, in priority order, for whoever continues
this:**
1. Investigate WHY the additive field-split's success is itself
   nondeterministic -- this is now a NARROWER, more tractable version of
   the project's standing cross-process-chaos question (apply the SAME
   bit-level first-iteration diff technique from
   `nondeterminism_investigation_2026-08-05.md` to a success vs. a
   failure run of THIS SPECIFIC configuration).
2. Finish debugging the Schur-complement bootstrapping issue (the
   deferred-configuration pattern needs a working FIRST solve to hang
   its later reconfiguration off of; that first solve currently fails
   outright under Schur's own PETSc defaults -- needs either a
   differently-ordered setup sequence or a throwaway warm-up solve under
   a simpler, known-working PC before switching to Schur).
3. If both remain intractable, this project should treat "does a
   correctly-assembled monolithic T-A Newton system converge" as
   genuinely unresolved (not proven either way) and make forward
   progress decisions accordingly, rather than waiting for a definitive
   answer that may not be cheaply obtainable.

No file in this project should currently claim the monolithic
architecture "converges" (Parts 1-5 retracted that) OR that it
"cannot work" (tonight's block-Jacobi result, while unreliable, is real
signal against that stronger claim too). Both remain open.

---

## 2026-08-05 (overnight continuation, part 7, coordinator-directed final round): bit-level diff attempt on the field-split configuration -- blocked by a genuine PETSc introspection obstacle, not completed

### A methodological correction found first, before any new testing

While building this round's test, found that Part 6's own additive
results are not ALL mutually comparable. `smoke4`/`smoke5`/`long1`/
`repro1` used `kind="nest"` with sub-solvers configured to `preonly`+
LU+MUMPS BEFORE any solve (confirmed safe for additive specifically).
`repro2` (Part 6's "another reproducibility check") ran AFTER
`monolithic_fieldsplit_test.py`'s SHARED `build_fieldsplit_monolithic_
problem` helper was changed to `kind="mpi"` with DEFERRED sub-solver
configuration -- a change made solely to fix the unrelated Schur/
MatMatMult limitation, but which (since additive and schur share that
one helper) silently changed additive's own iteration-1 behaviour too
(iteration 1 now runs under PETSc's default ILU, not the intended LU,
until reconfigured after that first solve). **`repro2`'s "immediate
failure" is therefore not clean evidence of the same phenomenon as
`long1`'s clean, same-code immediate failure -- it is confounded by an
actual code difference and should not have been cited as supporting
Part 6's "same chaos" conclusion.** Part 6's core conclusion still holds
on the three genuinely comparable runs alone (`smoke4`/`smoke5` success,
`repro1` partial, `long1` clean immediate failure -- all identical
`kind="nest"`, early-LU-configured code) -- this correction narrows the
evidence, it does not overturn the conclusion.

### The bit-level diff attempt itself

Built `transient/validation/monolithic_fieldsplit_dump.py`, deliberately
SELF-CONTAINED (reimplements the confirmed-clean `kind="nest"` +
early-configuration construction directly, not importing the
now-`kind="mpi"` shared helper) so every run is a genuine, uncontaminated
repeat -- addressing the confound just found.

**Blocked by a real PETSc/SNES initialization constraint, not a
methodology question.** Getting the ASSEMBLED matrix/RHS that will be
fed into the FIRST Newton step requires calling `snes.computeFunction()`/
`snes.computeJacobian()` on the current (warm-started) state BEFORE
`problem.solve()` is ever called. Every attempt at this crashed:

1. First attempt (`kind="nest"`, calling `pc.setUp()` then
   `computeFunction`/`computeJacobian` directly): **segfault**, no
   Python traceback at all -- crashed inside PETSc's C layer.
2. (From earlier in the night, same underlying constraint, different
   code path): calling `pc.setUp()`/`sub_ksp.setUp()` before any
   assembly on a `kind="mpi"` matrix raised PETSc error 73 ("Not for
   unassembled matrix") -- a Python-catchable error, at least, but the
   same root problem.

Every SUCCESSFUL use of `snes.computeFunction()`/`computeJacobian()`
elsewhere in this entire investigation (Part 2's `monolithic_jacobian_
inspect.py`, Part 3's `monolithic_fixed_bc_verify.py`) was ALWAYS preceded
by at least one REAL `problem.solve()` call first -- there is no
confirmed-safe way, found tonight, to introspect this SNES block system's
assembled state before its very first solve. This is a genuine
difference from the base T-solve (a plain `LinearProblem`), where
`prob.A.getValuesCSR()` after each `.solve()` call is straightforward and
was used successfully throughout `nondeterminism_investigation_2026-08-05
.md` -- the SNES/block-system's more complex internal lifecycle makes the
identical technique substantially harder to apply at exactly the "before
iteration 1" point that matters most for this specific question.

**A concrete path that would fix this, not attempted tonight given the
time-box:** bypass SNES's own introspection entirely and assemble the UFL
residual/Jacobian forms directly via `dolfinx.fem.petsc.assemble_vector`/
`assemble_matrix` on the compiled forms `build_nest_additive_problem`
already builds (would need a small refactor to return `F_list`/the
Jacobian forms, not just the finished `NonlinearProblem`) -- this does
not depend on SNES's internal solve-lifecycle state at all, and should be
safe to call at any point, including before the first `problem.solve()`.

**Not completed tonight.** Per the explicit time-box ("report that
honestly rather than grinding indefinitely" if a clean pair isn't
obtained in reasonable time), stopping here rather than continuing to
iterate on PETSc initialization order. This round produced a real,
useful correction (the `repro2` confound) but did NOT produce the
requested success-vs-failure bit-level comparison -- that remains open,
with a specific, concrete next step identified above.

## FINAL STATUS FOR THE MORNING (this supersedes nothing from Parts 1-6's own conclusions -- it only narrows Part 6's evidence and reports tonight's last attempt as incomplete)

- Parts 1-3 (BC-scoping bug, root cause, fix, structural verification):
  solid, unchanged, already corrected in the project's standing
  documentation by the coordinator.
- Part 6 (real PCFIELDSPLIT, block-Jacobi shows real-but-unreliable
  improvement): conclusion stands, now on a narrower but still valid set
  of clean same-code comparisons (3 runs, not 4).
- Part 7 (this round): the requested bit-level first-iteration diff for
  the field-split configuration was NOT obtained -- blocked by a genuine
  PETSc/SNES introspection constraint (safely reading the assembled
  system before any solve has ever happened), not by the diff failing to
  discriminate or by running out of repeat attempts. A concrete,
  actionable next step (bypass SNES introspection, assemble the UFL
  forms directly) is identified for whoever continues this.
- The central open question from Part 6 remains exactly as open as it
  was: does a correctly-assembled, correctly-preconditioned monolithic
  T-A system converge reliably? Real signal exists that it CAN converge
  (Part 6's clean successes), but not reliably (Part 6's clean failure),
  and tonight did not determine WHY the difference occurs.

Worktree confirmed clean (checked below). No commits made. This is the
end of tonight's session -- the user will review in the morning.

---

## 2026-08-06 (coordinator-directed continuation, Part 8): the bit-level pre-solve diff Part 7 was blocked on, obtained -- RHS diverges O(1) across independent launches BEFORE the monolithic Newton solve even runs; the Jacobian data does not

Directed continuation of Part 7's own identified next step ("bypass SNES
introspection entirely and assemble the UFL residual/Jacobian forms
directly via `dolfinx.fem.petsc.assemble_vector`/`assemble_matrix` ...
should be safe to call at any point, including before the first
`problem.solve()`"). New file:
`transient/validation/monolithic_direct_assemble_dump.py`, reusing Part
6/7's `build_nest_additive_problem` construction unchanged.

### The bypass works

Rather than `assemble_vector`/`assemble_matrix` directly (which need
hand-rolled BC lifting), used dolfinx's own module-level
`assemble_residual`/`assemble_jacobian` functions (`dolfinx/fem/petsc.py`)
-- these are the EXACT functions SNES calls internally via
`solver.setFunction`/`setJacobian`; calling them ourselves, with the
`NonlinearProblem`'s own compiled `problem.F`/`problem.J`/`problem.b`/
`problem.A`/`problem.x`/`problem.u`, reproduces precisely "the system fed
to the first Newton step" without going through any SNES-internal
lifecycle state. No crash, on any of 17 runs (1 smoke test + 16-run
batch) -- this is a clean, general fix for Part 7's blocker, reusable for
any future pre-solve introspection need on this class of problem.

### A second real bug found and fixed along the way

`Jmat.convert("aij")` (used by both this file's dump and Part 7's
original, never-reached `monolithic_fieldsplit_dump.py` dump line) --
**`PETSc.Mat.convert(mat_type, out=None)` converts IN PLACE when `out`
is not supplied** (confirmed by reading the petsc4py docstring directly
after this silently broke the subsequent real `problem.solve()` calls
with PETSc error 73, "local to global mapping" missing). The dump was
mutating `problem.A` itself from `nest` to `aij`, destroying the
field-split structure the real solve still needed. Part 7's script never
reached this line (it crashed earlier), so this bug was latent, not yet
triggered, in the existing investigation file. Fixed here by converting
a `Jmat.copy()`, not `problem.A` itself.

### The result: 16/16 runs, tallied honestly

The `ksp_its > 0` "outcome" label inherited from Part 6/7's scripts
turned out not to discriminate anything here -- every run has
`ksp_its >= 1` at k=1 (all print `OUTCOME: success` under that
definition), because `snes.setTolerances(max_it=1)` means SNES's own
`reason=-5` (`DIVERGED_MAX_ITS`) fires as soon as the ONE permitted
Newton step finishes, converged or not, and that's the common case.
**The real discriminator, found while reading the tallies, is whether
the inner KSP actually converged**: `reason=-3`
(`DIVERGED_LINEAR_SOLVE`) with `ksp_its=200` means the field-split-
preconditioned `fgmres` hit its own 200-iteration cap without
satisfying `ksp_rtol=1e-8` -- a genuine linear-solve failure, not
SNES's tolerance bookkeeping. By that criterion: **3/16 runs (19%)
genuinely failed the first linear solve (runs 3, 5, 14, all
`ksp_its=200`); 13/16 (81%) genuinely converged, in as few as 1
Krylov iteration.** This is a higher clean-success rate than Part 6's
own small sample (1 clean success, 1 partial, 2 failures out of 4
comparable runs) suggested, plausibly just `n=4` vs `n=16` sampling
noise -- not asserted as a contradiction, just a note that Part 6's
rate estimate was imprecise.

### The bit-level diff itself

Compared the pre-solve (direct-assembly, before any Newton step) CSR
dumps pairwise across multiple runs, both success-vs-failure and
failure-vs-failure and success-vs-success:

| pair | class | rhs rel diff | Jacobian data rel diff |
|---|---|---|---|
| run2 vs run3 | success vs failure | 1.58 | 1.06e-6 |
| run3 vs run5 | failure vs failure | 1.01 | 9.4e-7 |
| run3 vs run14 | failure vs failure | 1.39 | 1.8e-6 |
| run2 vs run6 | success vs success | 1.43 | 9.3e-7 |
| run2 vs run9 | success vs success | 1.44 | 6.8e-7 |
| run6 vs run9 | success vs success | 1.97 | 7.3e-7 |
| run1 vs run4 | success vs success | 1.66 | 1.1e-6 |

Sparsity pattern (`indptr`/`indices`) is bit-identical across all 16 runs
(expected -- same mesh, confirmed byte-reproducible within one process
per this project's standing finding). The largest Jacobian entry
(`max|data|`) is bit-identical to the last digit across all 16 runs too
(4.3199e11 exactly) -- consistent with that entry belonging to a
mesh/geometry-only block that has no dependence on the chaotic T/A state
at all.

**The headline finding: this is qualitatively DIFFERENT from every
other bit-level diff in this project's history.** Every prior instance
(the base T-solve in `nondeterminism_investigation_2026-08-05.md`) found
a near-machine-epsilon INPUT difference that only became large (~1e-3 to
1e-4 relative) AFTER a single linear solve amplified it --
ill-conditioning turning a tiny perturbation into a moderate one. Here,
**the residual vector itself is already O(1) relatively different
(1.0-2.0x) between independent process launches of the IDENTICAL nominal
configuration, before the monolithic Newton solve has run at all** --
while the assembled Jacobian's matrix entries are still only
~1e-6-relatively different (consistent with ordinary floating-point
reduction-order noise, not yet amplified by anything). No ill-conditioned
solve is needed to explain this: by the time the monolithic system is
even built, the warm-started T/A state feeding it has ALREADY diverged
across runs.

### Why this is consistent with, and sharpens, the standing root-cause finding

This does not contradict `nondeterminism_investigation_2026-08-05.md`'s
conclusion (extreme ill-conditioning in the underlying T-equation linear
sub-problem, ~1e17-1e19-fold amplification per solve) -- it extends it.
Every run here goes through `_picard_bootstrap(..., n_iters=30, ...)`
before the monolithic system is ever assembled. Thirty compounding
applications of a solve that turns machine-epsilon differences into
~1e-3-1e-4 relative ones is easily enough to fully decorrelate two
trajectories by iteration 30 -- consistent with the O(1) relative RHS
difference measured here. **This localizes WHERE the divergence
saturates: not inside the monolithic Newton solve itself, but upstream,
during the Picard bootstrap/seed phase, well before the monolithic
system is handed anything.** The monolithic solver (Part 6's
block-Jacobi field-split) is not the source of the chaos and does not
need to be -- it inherits an already-fully-diverged input by the time it
runs, from a mechanism (the bootstrap Picard iteration) already
root-caused elsewhere in this project.

**Not done here, a concrete cheap follow-up for whoever continues
this:** dump the T/A state immediately after `_picard_bootstrap`
(iteration 30) but BEFORE any monolithic assembly, across several runs,
and check whether the divergence is already O(1) at, say, iteration 10
or 15 -- this would pin down how many bootstrap iterations it actually
takes to saturate, rather than just confirming it has saturated by 30.

### Net effect on the standing open question

Part 6's central question ("does a correctly-assembled monolithic T-A
Newton system converge reliably?") is NOT answered by this round --
that remains open. What Part 8 adds: the unreliability is not evidence
against the monolithic architecture specifically. A system fed a
genuinely different (O(1)-diverged) input on each launch cannot be
expected to behave identically launch-to-launch regardless of which
solver architecture receives it -- Picard, Gauss-Seidel Newton hybrid, or
monolithic block Newton all inherit the same already-chaotic seed. Any
future attempt to get a reproducible read on the monolithic solver's OWN
convergence properties (isolated from bootstrap-phase chaos) would need
to either fix the bootstrap phase's sensitivity first, or seed every
compared run from one single, explicitly-saved, byte-identical T/A
state (not a freshly-run bootstrap) so the monolithic solve is the only
thing allowed to vary between runs.

Scripts: `transient/validation/monolithic_direct_assemble_dump.py`
(the fixed dump), 16 raw run dumps under a session scratch directory
(not checked into the repo -- regenerate via
`<env>/bin/python3 transient/validation/monolithic_direct_assemble_dump.py <path>`,
repeated, if needed again).

---

## 2026-08-06 (continuation): the divergence-growth curve during the Picard bootstrap -- traced, and it is NOT a single jump

Part 8's own flagged follow-up, done: instead of only comparing the
bootstrap's final (iteration-30) state across runs, checkpointed T/A
state at iterations 0, 2, 5, 10, 15, 20, 25, 30 of
`_picard_bootstrap`'s 30-iteration Picard run, across 8 independent
process launches of the identical nominal configuration (dt=60s,
I=19.6A) -- 28 pairwise comparisons per checkpoint. New file:
`transient/validation/bootstrap_saturation_check.py`, using
`_picard_phase`'s own existing `closure` extension point (called once
per iteration, before that iteration's T-solve) to dump state --
`_picard_phase`/`_picard_bootstrap` themselves were NOT modified.

### The curve (max relative difference in T / A across all 28 pairs at each checkpoint)

| iteration | T rel diff (median) | T rel diff (max) | A rel diff (median) | A rel diff (max) |
|---|---|---|---|---|
| 0  | 0.0 (bit-identical) | 0.0 | 1.2e-14 | 1.9e-14 |
| 2  | 2.3e-3 | 3.8e-3 | 1.7e-3 | 4.0e-3 |
| 5  | 6.7e-3 | 1.2e-2 | 2.3e-3 | 3.9e-3 |
| 10 | 1.2e-1 | 2.8e-1 | 1.8e-2 | 4.9e-2 |
| 15 | 4.7e-1 | 1.4e0 | 1.1e-1 | 2.5e-1 |
| 20 | 9.7e-1 | 1.3e0 | 5.4e-1 | 1.2e0 |
| 25 | 1.3e0 | 1.8e0 | 5.5e-1 | 1.4e0 |
| 30 | 1.4e0 | 1.9e0 | 2.7e-1 | 6.3e-1 |

(T=0 is bit-identical at iteration 0 because every run starts from the
same all-zero cold T; the A-field at iteration 0 already carries the
~1e-14 machine-epsilon-level noise this project's nondeterminism
investigation attributes to floating-point reduction-order effects in
multi-threaded assembly.)

### Two distinct regimes, not one smooth blowup

1. **Iterations 0->2: the ~1e-14 seed noise jumps to ~1e-3 in just TWO
   Picard iterations** -- roughly an 11-order-of-magnitude amplification
   in 2 steps. This is the SAME phenomenon `nondeterminism_investigation_
   2026-08-05.md` already quantified for the base (non-bootstrap) T-solve
   (a near-machine-epsilon input becoming a ~1e-3-to-1e-4 relative output
   difference after ONE linear solve, from the smoothed critical-state
   floor's extreme dynamic range) -- confirming that same mechanism is
   exactly where this trajectory's divergence ORIGINATES, at the very
   first iterations of the bootstrap, not gradually throughout it.
2. **Iterations 2->20: roughly geometric growth from ~1e-3 to O(1)**
   (each 5-10 iteration window compounds the existing difference by
   roughly 3-20x), reaching full decorrelation (median rel diff ~1,
   individual pairs up to ~1.3-1.9, the natural ceiling for a relative
   difference between two uncorrelated-scale vectors) by iteration
   ~20-25. Iterations 25->30 show NO further growth in T (median 1.3->1.4,
   already saturated) and actually a DROP in A's median rel diff
   (0.55->0.27) -- consistent with full decorrelation already reached by
   iteration 20-25, after which the comparison is just noise around the
   saturated ceiling rather than continued growth.

### This is chaos, not numerical blow-up -- confirmed, not just asserted

The absolute state norms (`||T||_inf`, `||A||_inf`) stay bounded and of
consistent order across every run at every checkpoint (`||T||_inf` in a
tight ~[7.7e8, 1.5e9] band from iteration 10 onward, `||A||_inf` in
~[1.1e8, 2.3e8]) -- no NaN, no run trending toward infinity, no
outlier run diverging in absolute magnitude while the others stay put.
Two runs can each individually look like a perfectly reasonable,
well-behaved Picard trajectory and still have fully decorrelated from
each other by iteration 20. This is the textbook signature of sensitive
dependence on initial conditions (deterministic chaos), not a numerical
instability -- directly confirming, with a number attached to WHEN it
happens, the "chaotic map" characterization
`nondeterminism_investigation_2026-08-05.md` already gave this system.

### Net effect

This sharpens, rather than changes, Part 8's conclusion. The O(1)
divergence Part 8 found at iteration 30 does not accumulate gradually
across all 30 iterations -- it is already essentially complete by
iteration ~20, and its ORIGIN is the same ~1e17-1e19-fold-per-solve
amplification mechanism already root-caused for the base T-solve,
triggered within the bootstrap's first 1-2 iterations. Any future
attempt to get a byte-identical warm-start state for isolating the
monolithic solver's own convergence properties (Part 8's suggested next
step) would need to intervene before iteration ~2 of the bootstrap, not
just before the monolithic system is assembled at iteration 30 -- by
iteration 2 the die is already substantially cast.

Script: `transient/validation/bootstrap_saturation_check.py`. Raw
checkpoint dumps (8 runs x 8 checkpoints) under a session scratch
directory, not checked into the repo -- regenerate via
`<env>/bin/python3 transient/validation/bootstrap_saturation_check.py <prefix>`,
repeated, if needed again.

---

## 2026-08-06 (continuation): n-value continuation prototyped -- delays the divergence substantially, does NOT prevent it reaching the same handoff ceiling within a fixed 30-iteration bootstrap

Coordinator-directed prototype of the top remedy suggested for the
divergence characterised above: n-value continuation (homotopy),
standard practice in the H-formulation/T-A superconductor modelling
literature for exactly this class of stiff power-law solver failure.
Idea: start the Picard bootstrap at a mild, well-conditioned exponent
(n_start=3.0, vs. the physical n(B,theta)~13-34) and linearly ramp to
the true n over the first `ramp_iters` iterations, so the solver never
has to face the sharp near-singular j/jc=1 transition from a cold,
far-away guess.

New file: `transient/validation/bootstrap_ncontinuation_check.py`.
Implementation does NOT modify `_picard_phase`/`_picard_bootstrap`
(validated, do-not-touch code) -- it wraps the real `NValueModel` in a
`ContinuationNModel` exposing the identical `.n_value(B_mag, theta)`
interface `_update_rho` already calls, blended toward `n_start` by a
`frac` attribute the SAME per-iteration `closure` hook (used for
checkpointing in the prior round) advances each iteration. `_picard_phase`
itself runs completely unmodified.

**Scope note:** the "smarter analytic seed" idea (a Bean/Kim critical-
state initial profile instead of cold T=0) discussed alongside
continuation was NOT built as a separate analytic-profile implementation
-- deliberately, to avoid introducing a new, unverified physics formula
into a project whose whole standing lesson is "don't trust a result
until it's independently checked." It was instead treated as
approximately subsumed by testing two ramp lengths (10 and 20
iterations): a longer ramp keeps the system in a well-conditioned,
near-Bean-like state for longer before committing to the true stiff
power law, which is the same qualitative effect a smarter seed would be
reaching for by a different route.

### Result: two ramp lengths (10, 20 iterations, n_start=3.0), 6 repeats each, compared against the existing 8-run no-continuation baseline

Median relative difference in T across all pairs, per checkpoint:

| iteration | baseline (no continuation) | ramp_iters=10 | ramp_iters=20 |
|---|---|---|---|
| 0  | 0 | 0 | 0 |
| 2  | 2.3e-3 | 4.6e-3 | 5.7e-3 |
| 5  | 6.7e-3 | 5.6e-3 | 7.5e-3 |
| 10 | 1.25e-1 | 2.5e-2 (5x better) | 1.3e-2 (10x better) |
| 15 | 4.7e-1 | 6.5e-2 (7x better) | 9.6e-2 (5x better) |
| 20 | 9.7e-1 | 7.3e-1 (25% better) | 5.4e-1 (45% better) |
| 25 | 1.25e0 | 1.13e0 (10% better) | 1.24e0 (~same) |
| 30 | 1.35e0 | 1.30e0 (~same) | 1.43e0 (slightly WORSE) |

A's own curve tells the same story with a sharper reversal: at iteration
20 both continuation configs are 4-5x better than baseline (0.12-0.14 vs
0.54), but by iteration 30 ramp10 is worse than baseline (0.45 vs 0.27)
and ramp20 is substantially worse (0.64 vs 0.27, with its single worst
pair reaching 1.59 vs baseline's worst of 0.63).

### Honest read: this delays the divergence, it does not prevent it reaching the same handoff ceiling

n-continuation does exactly what the mechanism predicts during the ramp
itself: keeping the system at a soft, well-conditioned exponent measurably
slows the SAME machine-epsilon noise from being amplified, buying a real
5-10x reduction in divergence through iterations 10-20. But neither
tested schedule prevents full O(1) decorrelation by iteration 30, the
actual point a downstream solver (monolithic or otherwise) would receive
this state -- the elbow of the growth curve visibly shifts later
(compare iteration 15 in the baseline, already at 0.47, to iteration 20
in ramp20, at 0.54 -- roughly a 5-iteration delay), but once `frac`
reaches 1.0 and the system is back at the true, stiff physical n, the
SAME ill-conditioning re-asserts itself and the (smaller, but still
present) accumulated difference gets the same violent amplification.
Neither schedule tested here holds the soft regime long enough, relative
to the fixed 30-iteration total budget, to still be ahead by the time the
bootstrap ends.

This is a genuinely useful, if partial, result -- not a fix on its own,
but strong evidence the mechanism understanding is correct (the
divergence rate really is governed by how close the effective exponent
is to the stiff physical value, exactly as the amplification-source
argument predicted), and a concrete, unexplored next lever: **a longer
total iteration budget** (so the post-ramp, full-n phase has enough
additional iterations to re-settle before handoff, rather than being cut
off at 30 immediately after the ramp completes) or a ramp that reaches
n=1.0 continuation status only asymptotically near the end of a longer
run, rather than exactly at 30. Not tested here, given the scope of this
round.

Scripts: `transient/validation/bootstrap_ncontinuation_check.py`. Raw
checkpoint dumps (12 runs x 8 checkpoints) under a session scratch
directory, not checked into the repo.

---

## 2026-08-06 (continuation): the dwell-time hypothesis tested further -- does NOT hold up cleanly; n-continuation is a real middle-of-run effect but not a handoff-time fix

Follow-up to the n-continuation round above. That round's own iteration-30
numbers (baseline dwell=30: 1.35; ramp10 dwell=20: 1.30; ramp20 dwell=10:
1.43) suggested a hypothesis: divergence at handoff might be governed by
how many iterations are spent at the full physical (stiff) exponent
before handoff ("dwell time"), predicting that pushing the ramp even
later within the SAME 30-iteration total budget (shrinking dwell time
further, at NO extra iteration cost) should keep helping. Tested
ramp_iters=25 (dwell=5 at handoff) and ramp_iters=28 (dwell=2), 6
repeats each, same methodology.

### Combined final table, all five configs, T rel diff median/max at iteration 30 (handoff)

| config | dwell at iter 30 | T rel diff (median/max) | A rel diff (median/max) |
|---|---|---|---|
| baseline (no continuation) | 30 | 1.35 / 1.90 | 0.27 / 0.63 |
| ramp10 | 20 | 1.30 / 1.86 | 0.45 / 1.09 |
| ramp20 | 10 | 1.43 / 2.91 | 0.64 / 1.09 |
| ramp25 | 5  | 1.16 / 1.48 | 0.60 / 1.09 |
| ramp28 | 2  | 1.18 / 2.12 | 0.48 / 1.18 |

### The dwell-time hypothesis does NOT hold up cleanly

T's numbers are not monotonic in dwell time: ramp20 (dwell=10) is the
WORST of all five configs (1.43, even above baseline's 1.35), while
ramp25 (dwell=5) is the best (1.16) -- but ramp28 (dwell=2, even less
dwell than ramp25) is essentially tied with ramp25, not better, breaking
what a clean dwell-time relationship would predict. **A's numbers are
worse for this hypothesis: every single continuation config (0.45-0.64)
is HIGHER than the no-continuation baseline (0.27) at handoff,
regardless of ramp length** -- the exact opposite of what "less dwell
time helps" predicts, and the opposite sign from T's own (mild, noisy)
trend.

The most defensible reading, given n=6 repeats per config (15 pairs,
itself a small sample of an already-noisy saturated-ceiling
distribution -- individual pairs range 1.2-2.9 even within one config):
**this is sampling noise dominating a real signal that is, at best, weak
and inconsistent between T and A.** The middle-of-run protective effect
(5-10x reduction in divergence at iterations 10-20, reproduced cleanly
across all four continuation configs and both fields) is real and not in
doubt. Whether ANY of the tested ramp schedules meaningfully reduces
divergence specifically AT THE ITERATION-30 HANDOFF POINT is not
established by this data -- the differences between configs there are
comparable in size to the run-to-run noise within a single config.

### Recommendation: this specific lever has hit diminishing returns for now

n-value continuation, as tested here (a single linear ramp within a
fixed 30-iteration Picard bootstrap), should be considered validated as
a real mid-run stabiliser but NOT demonstrated as a fix for the
handoff-time divergence that actually matters for downstream solver
reliability. Further tuning of ramp length/shape within this same
30-iteration structure is not a promising next move -- the five points
tested already span the practical range (dwell 2 to 30) without a clean
trend emerging. Two directions that would be a genuinely different test,
not more of the same, if this is revisited:
1. Larger sample sizes (20-30 repeats per config, not 6-8) to determine
   whether the weak T-field trend and the adverse A-field trend are real
   effects or pure noise -- expensive (each repeat costs ~20-26s, so this
   is a real but bounded cost, not prohibitive).
2. A structurally different lever, not a variant of continuation: reduce
   the TOTAL number of Picard iterations needed by starting from a
   genuinely better physical guess (the analytic Bean/Kim seed idea
   deferred at the start of this line of work) rather than trying to
   outrun the stiff regime within a fixed budget.

For now, this line of investigation (n-continuation and its variants) is
a reasonable stopping point. The short-dt/multi-step transient problem
remains OPEN, exactly as CLAUDE.md's standing status already states.

Scripts (unchanged from the prior round):
`transient/validation/bootstrap_ncontinuation_check.py`.

---

## Part 9 (2026-08-06): analytic Bean-like seed prototyped -- a structurally different lever than n-continuation, with the same overall verdict: real but inconsistent mid-run effect, no reliable win at handoff

Prototype of the second remedy discussed alongside n-continuation:
replace the cold T=0 bootstrap start with an initial T(z) profile per
layer that already looks roughly critical-state-like (current
concentrated toward the tape edges, more so as local I/Ic rises), rather
than letting 30 Picard iterations discover that shape from a flat start.
New file: `transient/validation/bootstrap_beanseed_check.py`.

### Deliberately NOT the published Norris (1970) closed form

Reconstructing the exact Norris self-field strip solution from memory
was attempted while reasoning through this and abandoned: a first
attempt produced an ODD (antisymmetric) profile, which on reflection is
the signature of the DIFFERENT external-field screening problem, not the
transport-current problem this project actually has (which requires an
EVEN, same-sign profile by z<->-z symmetry -- verified by checking that
the T_bot=+T_amp/T_top=-T_amp boundary conditions imply an ODD T(z),
hence an EVEN dT/dz = J(z), the physically correct symmetry). Given a
real risk of a sign/exponent transcription error surfaced during that
reasoning, the profile actually used is a self-derived, BC-anchored
piecewise-linear approximation instead: a core region with gradient
G_core and two edge regions with gradient G_edge = ratio*G_core
(ratio = 1 + 9*I_frac, core half-width fraction f = 1 - I_frac,
I_frac = J_unif/Jc_layer clipped to 0.98), with G_core solved so the
piecewise-integrated profile reproduces T_bot/T_top EXACTLY by
construction. This is checked with a runtime assertion (not just claimed
in a docstring) comparing the profile's own endpoint values against
T_amp to 1e-6 relative tolerance -- passed on all 9 runs (1 smoke test +
8-run batch), no assertion failures.

Jc_layer reuses the SAME Jc_vol array the seed-time `_update_rho` call
already computes (median over each layer's own cells) -- no new Ic-model
call, no new physics assumption beyond the profile shape itself.

### Result: 8 repeats, compared directly against the 8-run cold-start baseline

| iteration | baseline T (med/max) | Bean-seed T (med/max) | baseline A (med/max) | Bean-seed A (med/max) |
|---|---|---|---|---|
| 0  | 0 / 0 | 3.0e-7 / 4.5e-7 | 1.2e-14 / 1.9e-14 | 1.4e-14 / 3.3e-14 |
| 2  | 2.3e-3 / 3.8e-3 | 4.6e-3 / 1.4e-2 | 1.7e-3 / 4.0e-3 | 1.2e-3 / 2.6e-3 |
| 5  | 6.7e-3 / 1.2e-2 | 1.3e-2 / 2.9e-2 | 2.3e-3 / 3.9e-3 | 2.1e-3 / 6.3e-3 |
| 10 | 1.25e-1 / 2.8e-1 | 5.4e-2 / 1.4e-1 | 1.9e-2 / 4.9e-2 | 2.0e-2 / 5.6e-2 |
| 15 | 4.7e-1 / 1.4e0 | 8.0e-1 / 1.4e0 | 1.1e-1 / 2.5e-1 | 1.5e-1 / 4.6e-1 |
| 20 | 9.7e-1 / 1.3e0 | 9.8e-1 / 2.1e0 | 5.4e-1 / 1.2e0 | 4.1e-1 / 7.7e-1 |
| 25 | 1.25e0 / 1.8e0 | 1.29e0 / 1.8e0 | 5.5e-1 / 1.4e0 | 4.1e-1 / 8.0e-1 |
| 30 | 1.35e0 / 1.9e0 | 1.26e0 / 1.7e0 | 2.7e-1 / 0.63 | 6.2e-1 / 0.76 |

**One incidental finding at iteration 0:** unlike the baseline's
literal-zero (bit-identical, since cold T=0 involves no computation at
all), the Bean seed's OWN starting point already differs by ~3e-7
relative between independent launches -- not the ~1e-14 pure
floating-point noise floor seen in the A-field seed, four orders of
magnitude below the ~1e-3 threshold that later triggers runaway growth,
but confirming the seed formula itself is not perfectly reproducible
either (it depends on Jc_layer, itself derived from a B-field that
involved a multi-threaded assembly).

### Honest read: not a clean win, and not consistent even mid-run

Unlike n-continuation (which was cleanly, consistently better than
baseline at EVERY checkpoint from iteration 10 through 20, in both
fields), the Bean seed's effect is genuinely mixed even in the middle of
the run: markedly BETTER at iteration 10 (2.3x less T divergence) but
markedly WORSE at iteration 15 (1.7x MORE T divergence) and roughly tied
elsewhere. At the handoff point (iteration 30) it's a wash: T is ~7%
better (1.26 vs 1.35, well within the noise band the n-continuation
sweep already established -- recall that swept from 1.16 to 1.43 across
five configs with no clean trend), while A is clearly WORSE (0.62 vs
0.27, more than 2x) -- the same adverse-A-field pattern every
intervention tried this session has shown at handoff.

This is a weaker, LESS consistent signal than n-continuation's own
(real, if ultimately non-persisting) mid-run effect, not a stronger one.
Both structurally different levers tried this session -- n-continuation
(changes the coefficient physics during the run) and the Bean seed
(changes only the starting point) -- land on the same overall verdict:
real perturbation of the chaotic trajectory, no reliable improvement at
the point that actually matters for a downstream solver.

### What this adds to the standing picture

Two independent, structurally different interventions both failing to
produce a reliable handoff-time improvement is itself informative: it
argues AGAINST "this is just an easy-to-fix startup transient" and FOR
treating the ~20-25-iteration saturation window characterised earlier in
this file as a fairly robust property of the underlying chaotic map at
the physical operating point, not an artifact of the specific cold-start
recipe. A genuinely different angle, not tried here and worth flagging
rather than pursuing without direction: since the FIXED 30-iteration
checkpoint itself may not be a meaningful synchronisation point across
genuinely different chaotic trajectories, comparing divergence at a
convergence-triggered (not fixed-iteration) stopping point might be a
fairer test than anything tried in this file so far -- but that is a
bigger methodological change than either of the two levers tried this
session, and is a scope decision, not a natural next increment.

Scripts: `transient/validation/bootstrap_beanseed_check.py`. Raw
checkpoint dumps (9 runs x 8 checkpoints) under a session scratch
directory, not checked into the repo.
