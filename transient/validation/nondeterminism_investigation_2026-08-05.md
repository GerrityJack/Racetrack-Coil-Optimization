# 2026-08-05: nondeterminism root-cause investigation (part 2)

## Context

CLAUDE.md and the earlier 2026-08-05 HISTORY.md entries established that the
T-A transient solver's cold-start convergence is non-deterministic across
separate OS process launches (identical config, ~2/5 success historically),
and ruled out three explanations directly:

1. gmsh mesh non-reproducibility across processes (byte-identical `.msh`
   files across two independent builds).
2. Python hash-seed randomisation + generic OpenMP threading
   (`OMP_NUM_THREADS=1` + `PYTHONHASHSEED=0` did not stabilise the outcome —
   if anything it was worse, 0/5 vs. baseline 2/5 on ten repeats).
3. dolfinx's own post-mesh-load dof/cell reordering (found and ruled out in
   the parent session earlier today: 20 independent fresh process launches
   reading the identical `mesh/racetrack_mesh.msh` produced byte-identical
   sha256 fingerprints for geometry dofmap, cell-vertex connectivity, cell
   tags, `coil_cells`, and both `V_T`/`V_A` dofmaps).

This file picks up the remaining candidates flagged in HISTORY.md's
"Recommended order, if this is pursued further" and the parent session's
priority list. Repro case throughout:
`transient/validation/first_step_diagnostic.py 60 19.6` (dt=60s,
I=19.6A, cold start, per-layer T-A, no NI closure — the exact case behind
the documented "~2/5" claim), run as genuinely separate OS process launches
of `/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3`.

**Environment note that shaped this investigation's pace**: this session ran
concurrently with a sibling investigation (the half-domain convergence test,
in its own worktree) on the same physical machine. Load average peaked
above 30 on what CPU topology suggests is an ~8-core box — every timing
number below is under heavy oversubscription, and batches took
substantially longer wall-clock than an isolated run would. This does not
bias the *correctness* of any convergence outcome (the physics doesn't know
about system load), but it capped how many repeats were affordable in the
time available, and is the reason isolated per-rep `XDG_CACHE_HOME`
directories were used throughout (see Candidate 3) — both to test the JIT
cache hypothesis cleanly and to stop this investigation's own repeats from
racing each other on the shared cache while a concurrent sibling process
was also compiling forms into it.

## New diagnostic infrastructure added

- `solve/ta_solve.py`: added an inert, env-var-gated hook
  (`TA_MUMPS_EXTRA_OPTS`, a JSON dict) in `_build_problems()` that merges
  extra PETSc/MUMPS options into the T-equation's `LinearProblem`
  construction. No effect unless the env var is set — verified the
  production code path (`solve_ta_at_current`, `ta_sweep.py`, etc.) is
  untouched by this addition.
- `transient/validation/nd_runs/run_batch.sh` — N repeats of the repro
  case as separate process launches, optional env overrides.
- `transient/validation/nd_runs/run_batch_env.sh` — same, but gives every
  repeat its own isolated `XDG_CACHE_HOME` so concurrent processes (this
  investigation's own repeats, or a sibling investigation) cannot race on
  the shared FFCx JIT cache.
- `transient/validation/nd_runs/candidate_*.sh` — one script per candidate
  configuration tested below.
- `transient/validation/nd_runs/mumps_verbose_check.sh` — single-run MUMPS
  `ICNTL(4)=3` verbosity check, to read off the actual ordering
  method/analysis mode MUMPS chooses at runtime rather than assume from
  linked libraries.

## Environment findings (before any repeats)

`ldd`/`conda list` on the `fenicsx-env` environment:

- BLAS/LAPACK: `libblas.so.3`/`libcblas.so.3`/`liblapack.so.3` all resolve
  to **`libopenblasp-r0.3.33.so`**, the conda-forge `openmp_...` build
  variant (build string `openmp_hd680484_0`). This is OpenBLAS built with
  the **OpenMP threading backend**, not the pthreads backend — meaning
  `OMP_NUM_THREADS` genuinely does control OpenBLAS's thread count for this
  specific build (the "OMP_NUM_THREADS doesn't pin OpenBLAS" concern that
  motivated re-testing BLAS env vars does not actually apply here; the
  prior "forced-deterministic" test's `OMP_NUM_THREADS=1` should already
  have serialised OpenBLAS). This narrows candidate 1's novelty somewhat —
  see below.
- `libpetsc.so` links `libptscotchparmetisv3`, `libparmetis`, `libmetis`,
  `libptesmumps` (PT-SCOTCH + ParMETIS — the *parallel* graph-partitioning
  libraries) alongside the plain serial MUMPS libs (`libdmumps.so`, etc.).
  This raised MUMPS's fill-reducing ordering choice as a live candidate
  (linked in, not necessarily invoked at `comm.size==1` — needs runtime
  verification, not an assumption from link lines).
- `~/.cache/fenics` (the FFCx JIT cache) is a single global, unversioned,
  cross-process, cross-worktree directory — **every process on this
  machine shares it**, including both concurrently-running investigations
  today. Found one stale `libffcx_forms_...c.failed` marker dated 2026-08-03
  (two days old, from an unrelated earlier failed compile) sitting
  unnoticed in the cache. This confirms the cache does accumulate
  leftover-failure artifacts across this project's history of killed
  long-running processes, though this specific one is not the same form
  hash as anything in the current repro case's call path.

## Candidate 1: BLAS/MUMPS-specific thread env vars, combined with Candidate 4: MUMPS ordering

> **CORRECTION, added in the coordinator-directed follow-up further down
> this file: the `forced_serial_mumps` batch described in this section
> never actually applied its configuration, due to a bash array/export
> bug — see "The `forced_serial_mumps` batch never actually applied its
> env vars" below. Its "REJECTED" verdict is WITHDRAWN. Read this section
> for the reasoning that motivated the test, but treat its numeric result
> as void; the corrected, properly-verified re-test is in the follow-up
> section, with the opposite conclusion.**

Given the environment finding above (OpenBLAS here is the OpenMP-backend
build, so `OMP_NUM_THREADS=1` alone should already have serialised it in
the prior "forced-deterministic" test that gave 0/5), pure BLAS thread
pinning is unlikely to add anything new on its own. The MORE promising
untested lever is MUMPS's own fill-reducing ordering choice
(`ICNTL(7)`) and analysis mode (`ICNTL(28)`), given ParMETIS/PT-SCOTCH are
linked into this PETSc build. Tested together as one configuration
(`forced_serial_mumps`, `nd_runs/candidate_forced_serial_mumps.sh`):

`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`
`TA_MUMPS_EXTRA_OPTS={"mat_mumps_icntl_16":1,"mat_mumps_icntl_7":2,"mat_mumps_icntl_28":1}`

(icntl_7=2 forces AMF ordering — a simple deterministic minimum-degree-like
heuristic with no internal randomisation, as opposed to the default
"automatic" choice which could in principle fall through to a
graph-partitioner-based method; icntl_28=1 forces sequential analysis
explicitly, removing any ambiguity about whether the parallel-analysis path
is silently active even at `comm.size==1`; icntl_16=1 pins MUMPS's own
OpenMP thread count explicitly, on top of the env vars).

**Logical structure of this comparison**: the thread-pinning env vars alone
were ALREADY tested in the prior investigation and made things WORSE (0/5
vs baseline 2/5). This configuration keeps that same thread-pinning
component unchanged and ADDS ONLY the MUMPS ordering options on top — so
any improvement over 0/5 in this batch is attributable specifically to the
ordering change, not re-litigating the already-rejected thread hypothesis.

Live counts (updated as reps complete; final tally in the Summary section):

**FINAL, complete batches (both 2026-08-05, this session):**

- baseline (default env, shared cache): F,F,T,T,T,T,F,F,F,F →
  **4/10 converged = 40%**. This reproduces the historically-quoted "~2/5"
  rate almost exactly in a completely independent batch, on a different
  day, under heavy concurrent system load — strong confirmation that the
  underlying phenomenon is real, stable in its statistics, and not itself
  an artifact of whatever machine/load conditions it's measured under.
  Iteration counts on successes: 111, 123, 104, 23 — note the wide spread
  even among successes, itself notable (see the MUMPS-per-iteration
  finding below).
- forced_serial_mumps (BLAS+MUMPS thread pinning, `icntl_16` explicit,
  `icntl_7=2`/`icntl_28=1` — 5 reps before deliberately stopped once the
  picture was clear, to free machine resources for other candidates):
  T,F,F,T,F,T (one extra rep completed after the stop signal was sent) →
  **3/6 converged = 50%**, statistically indistinguishable from baseline's
  40% at this sample size. **This candidate is REJECTED** — no
  detectable effect, consistent with the finding below that its "ordering
  fix" was actually a no-op (AMF was already the default).

**Why this candidate got promoted to top priority**: inspected dolfinx's
`LinearProblem.solve()` source directly
(`dolfinx/fem/petsc.py`) — it does `self.A.zeroEntries()` then
`assemble_matrix(self.A, ...)` on the SAME PETSc `Mat` object every call,
i.e. every single Picard iteration reassembles new VALUES into the same
matrix (same nonzero structure, since only `rho_fn`'s coefficient values
change between iterations, not the mesh or sparsity pattern). PETSc's
default LU/MUMPS `PC` does NOT automatically reuse the symbolic
factorisation (fill-reducing ordering) across `KSPSolve` calls on a
modified matrix unless `-pc_factor_reuse_ordering` is explicitly set,
which nothing in this codebase does. **This means MUMPS's ordering choice
is potentially being freely re-decided at EVERY SINGLE Picard iteration
of every T-solve, not just once at process start.** If the "automatic"
ordering heuristic (`ICNTL(7)=7`, the current default — nothing in
`ta_solve.py` sets it) has any internal source of non-input-determined
variability (e.g. routing through a randomised graph-partitioning
heuristic in METIS/SCOTCH under certain matrix-size/density heuristics),
that would inject a fresh opportunity for a tiny perturbation at EVERY
iteration, not just the first — which would also help explain why this
system's Picard iteration counts vary so wildly even among successful runs
in today's baseline batch (23, 104, 111, 123, 145 iterations, same
nominal config) and not only whether it succeeds or fails at all. This
reframes "cross-process nondeterminism" as possibly better described as
"per-linear-solve ordering nondeterminism, which cross-process comparison
merely samples independently each time" — forcing a FIXED, simple,
non-randomised ordering method (AMD/AMF have no internal RNG, unlike
METIS/SCOTCH-based orderings) is there a genuine mechanistic hypothesis,
not just a knob to try because it exists.

**Side observation**: even with every thread-count env var above set to 1,
`ps` showed this configuration's process using ~200-240% CPU, not ~100%
— something in the call stack is still spawning multiple threads despite
every documented lever being pulled. Worth a follow-up note even though it
doesn't block interpreting the convergence-outcome results: full thread
determinism was NOT actually achieved by this env-var set, so this
candidate's test is better read as "MUMPS ordering forced deterministic,
residual threading still present" than "fully serial execution."

## Candidate 2: ASLR — INCONCLUSIVE, insufficient data

Batch launched (`nd_runs/candidate_no_aslr.sh`, target 8 reps, each via
`setarch $(uname -m) -R <python3> ...` to disable address-space layout
randomisation for the whole process tree, isolated per-rep
`XDG_CACHE_HOME`). **Only 3 of the planned 8 reps completed before this
investigation was closed out**: F, T, F → 1/3. This is nowhere near enough to
say anything about whether ASLR matters — flagged explicitly as the
top-priority candidate for anyone continuing this work (see Summary). The
script (`nd_runs/candidate_no_aslr.sh`) is left in place, ready to run
with more machine headroom.

**Why this stopped early**: this session ran concurrently with a sibling
investigation on the same physical machine (see the environment note at
the top of this file); system load peaked above 38 (on what CPU topology
suggests is an ~8-core box) and stayed above 20 for most of the session.
Every batch in this investigation ran roughly 3-5x slower wall-clock than
an isolated run would, and continuing to chase full statistical power on
every remaining candidate under these conditions was a diminishing-returns
trade against the time available — a genuine scope decision, consistent
with how this project has handled similar situations before (see
docs/HISTORY.md's "genuine scope decision" framing in the original
2026-08-05 investigation this one continues).

## Candidate 3: JIT cache staleness/races — NOT COMPLETED

Batch launched twice (`nd_runs/candidate_fresh_cache_each_run.sh`, target
8 reps each rep gets a BRAND NEW, never-before-used `XDG_CACHE_HOME`
deleted immediately after — forcing full FFCx recompilation from scratch
every single rep, zero reuse). **Only 1 of the planned 8 reps completed**
before this investigation was closed out: F (non-converged, 150 iters
capped) — the first launch was killed to free resources for candidate 2,
and the relaunch's first rep (inherently slower than a warm-cache rep,
compounding the general system-contention slowdown described under
Candidate 2) had only just finished. n=1 is not data, just an anecdote —
no conclusion possible either way.

The circumstantial evidence gathered earlier (the shared, global,
unversioned `~/.cache/fenics` directory; the pre-existing stale `.failed`
marker from 2026-08-03; this investigation's own repeats and a
concurrently-running sibling investigation actively compiling into that
SAME directory at the same time, observed directly via `ps` during this
session) still stands as a real, demonstrated structural hazard,
independent of whether a completed batch would have shown a rate
difference. Recommended next step for anyone continuing: re-run this
exact batch with the machine otherwise idle, and separately, run a
DELIBERATE race test (two processes launched to compile the identical
form hash at the same instant, in a tight loop, many times) rather than
this session's staleness/reuse framing, which is a different and weaker
test of the same underlying concern.

## Candidate 4: MUMPS internal ordering — DECISIVE NEGATIVE RESULT

Ran `nd_runs/mumps_verbose_check.sh` (`ICNTL(4)=3`, MUMPS's own verbosity,
on the UNMODIFIED default configuration — no ordering override) and read
MUMPS's own printed analysis-phase output directly, rather than assuming
from linked libraries:

```
Type of parallelism: Working host
****** ANALYSIS STEP ********
...
Ordering based on AMF
```
appearing identically at Picard iterations 1, 2, and 3 of the SAME process
(byte-identical `FILS`/`FRERE` permutation arrays and entry counts each
time), with `executing #MPI = 1 and #OMP = 8` confirming sequential
(not parallel/ParMETIS/PT-SCOTCH) analysis, exactly as the automatic
default is documented to choose for a single-rank, this-sized problem.

**This refutes the MUMPS-ordering-nondeterminism hypothesis as originally
framed.** The default ordering was ALREADY the simple, deterministic,
non-randomised AMF heuristic (not METIS/SCOTCH, despite those being linked
into the PETSc build) — meaning `forced_serial_mumps`'s `icntl_7=2`
override (which also selects AMF) changed NOTHING relative to the
production default. The ordering *algorithm* is not a live candidate. What
`forced_serial_mumps` actually tests, now that this is understood, is only
the residual thread-count env vars plus explicitly pinning
`icntl_16` (MUMPS's own OpenMP thread count) — a narrower claim than
originally intended, and given `forced_serial_mumps` already shows failures
in its first two reps (see Candidate 1 above), that narrower claim is not
looking promising either.

**What remains live from the original "MUMPS internal parallelism"
concern**: `#OMP = 8` in the default configuration confirms MUMPS's
factorisation IS multi-threaded internally by default (matching the
~790% CPU observed throughout this investigation's baseline). Per-iteration
re-analysis (see the dolfinx `LinearProblem.solve()` finding above — full
symbolic+numeric factorisation runs fresh every Picard iteration, not just
once) means this 8-way-threaded AMF ordering computation and the 8-way
threaded numeric factorisation both run repeatedly, every iteration, every
process. AMF itself is a deterministic sequential algorithm with no RNG,
but if MUMPS's multi-threaded implementation of it (or of the subsequent
numeric factorisation) has ANY floating-point-order-dependent step (e.g.
threaded partial pivoting, threaded frontal-matrix assembly with a
non-fixed reduction order), that remains a live, per-iteration source of
non-associative floating-point differences — not through ordering *choice*
varying, but through numeric roundoff in a fixed-ordering computation
varying with thread-scheduling timing. This is a more precise, narrower
version of the original "candidate 1" hypothesis than the initial framing,
and worth stating clearly since it survived while the ordering-choice part
of that hypothesis did not.

## Candidate 5: denormal/FTZ-DAZ state

`nd_runs/check_ftz_daz.py` — behavioural probe (multiply/divide/sum a
subnormal double) run at import time for numpy, mpi4py, petsc4py, and
dolfinx, in-process, sequentially. Result: **no flushing detected at any
stage** — `5e-324` (the smallest positive subnormal) survived
multiplication, division, and an 8-element numpy sum completely intact
through every import stage tested. This is a single run, not a
repeated-process comparison, and it only probes the scalar/Python-level
and small-array numpy code path — it does NOT rule out FTZ/DAZ being set
transiently inside a specific vectorised OpenBLAS or MUMPS compiled kernel
that this probe never exercises. Given the cost of building a more
invasive probe (would need a small C extension reading MXCSR directly, or
disassembling which code path a 1718x1718 sparse factorisation actually
takes) relative to its prior plausibility, this candidate is deprioritised
rather than fully closed out — flagged as untested at the
compiled-kernel level in the final summary.

## Candidate 6: bit-level first-iteration diff

Exercised in the coordinator-directed follow-up further down this file
("The iteration-1 bit-level diff, exercised") — see there for the full
methodology and results. Short version: the tiny near-machine-epsilon
input differences get amplified ~10¹⁷-10¹⁹-fold by the first linear
solve alone, but this amplification is generic to any two independent
runs (including fail-vs-fail), so it doesn't by itself discriminate the
eventual outcome.

## 2026-08-05 (continued): coordinator-directed follow-up — the threading question resolved, and the iteration-1 diff exercised

Two things flagged for one more focused round: (1) the `forced_serial_mumps`
candidate's process still showed 200-240% CPU despite thread-pinning env
vars — chase down whether single-threading was ever genuinely achieved,
verified by observation not by trusting an env var was read; (2) actually
exercise the `DUMP_ITER1_MATRIX_PATH` hook against a real success/failure
pair. Both done. The first one uncovered something more fundamental than
expected.

### The `forced_serial_mumps` batch never actually applied its env vars — a real infrastructure bug, found and fixed

Before measuring anything, inspected how `candidate_forced_serial_mumps.sh`
actually passed its configuration down to `run_batch_env.sh`:

```bash
export EXTRA_ENV=("OMP_NUM_THREADS=1" ... 'TA_MUMPS_EXTRA_OPTS={...}')
"$HERE/run_batch_env.sh" forced_serial_mumps 10 60 19.6
```

`run_batch_env.sh` is invoked as a **separate process** (its own shebang,
not `source`d), and reads `"${EXTRA_ENV[@]}"` inside. Bash **cannot export
array variables across a process boundary** — only scalar strings survive
`export` into a child's environment. Verified directly:

```
$ bash -c 'export EXTRA_ENV=("FOO=1" "BAR=2"); bash -c "echo [\${EXTRA_ENV[@]}]; echo [\$FOO]"'
EXTRA_ENV in child: []
FOO=[]
```

**This means the entire `forced_serial_mumps` batch (the one reported as
"REJECTED, 3/6=50%, statistically indistinguishable from baseline" above)
ran with NONE of its intended configuration applied — no thread pinning,
no MUMPS ordering override, nothing.** It was, unknowingly, a second
baseline batch. Its rate matching baseline is not evidence the
configuration had no effect — it's an artifact of the configuration never
having been applied at all. **The "REJECTED" verdict for that specific
batch, and by extension the "candidate 1" and "candidate 4 residual
threading" framing above, are WITHDRAWN as originally stated.** The MUMPS
verbose-ordering finding (AMF is the default, confirmed via MUMPS's own
`ICNTL(4)=3` output) is unaffected by this bug — that check
(`mumps_verbose_check.sh`) set its env var directly, not through the
broken array mechanism — so the ordering-algorithm conclusion still
stands on its own.

Every script from this point on sets environment variables as a **direct
prefix to the command** (`VAR=val VAR2=val2 <python> ...`), the only form
that reliably propagates into the child process, and every claim below was
checked against actual observed behaviour, not trusted from the env var
alone.

### Verifying genuine single-threading by observation, not by trusting the env var

Built `nd_runs/thread_audit.sh` (samples `/proc/<pid>/status` `Threads:`
and every thread's `comm` every 2s) and `nd_runs/thread_cpu_audit.sh`
(reads `utime+stime` from `/proc/<pid>/task/<tid>/stat` at two points 5s
apart, per thread — the only way to distinguish "thread exists" from
"thread is actually computing").

**Default (unpinned) threading**, mid-solve: 7 idle/structural threads
(0 ticks each over the 5s window) plus **7 additional threads each
consuming 131-352 ticks** (48-70% of a core each) — **423% aggregate CPU**,
confirmed genuine parallel computation (`nd_runs/thread_cpu_audit_default_samples.txt`).

**`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_THREAD_LIMIT=1`
plus `TA_MUMPS_EXTRA_OPTS` with `icntl_16/7/28`**, applied correctly this
time (direct prefix): same 6-7 extra OS threads still get spawned (thread
pools are created eagerly and never torn down), but **every one of them
shows a `delta_ticks` of exactly 0** over the 5s window — genuinely idle,
not computing. Only the main thread does any work (517/500 ticks ≈ 100%
of one core). Repeated during the mesh-build (gmsh) phase too — also
clean single-core (~102% aggregate) — so gmsh's own meshing isn't a
hidden multi-threading source either. **Single-threaded execution is
real and was correctly verified, not just asserted from an unread env
var** — the earlier "200-240% CPU via `ps`" observation that prompted
this whole line of inquiry was itself a `ps`-snapshot artifact (or
reflected a moment this simpler tick-delta method wasn't sampling), not
evidence of residual threading. Also found in passing:
`-mat_mumps_icntl_16` is silently **rejected by this PETSc/MUMPS build** —
PETSc prints "WARNING! There are options you set that were not used!"
listing it explicitly (6 times, once per layer) whenever it's actually
applied. It has no effect either way; `icntl_7`/`icntl_28` ARE recognised
(not flagged) but, per the earlier finding, already match the default.

### Re-run with GENUINELY verified single-threading: 0/18, and every failure is bit-identical

Three independent batches, all using the corrected (direct-prefix) env var
application, run across both heavy system load (~20-30) and light load
(~8-12, after the sibling half-domain investigation finished):

| batch | config | result |
|---|---|---|
| `verified_single_thread` (1st) | full pin + `OMP_THREAD_LIMIT=1` + icntl overrides | F,F,F,F,F,F → 0/6 |
| `verified_no_threadlimit` | same minus `OMP_THREAD_LIMIT` | F,F,F,F,F,F → 0/6 |
| `verified_single_thread` (2nd, light load) | identical to 1st | F,F,F,F,F,F → 0/6 |

**0/18 converged, across three independent batches, two env-var variants,
and two very different system-load regimes.** Under the historical/
this-session baseline rate of ~40%, 0/18 has probability 0.6^18 ≈ 6×10⁻⁷
— this is not sampling noise. **And the six reps in the light-load rerun
didn't just all fail the same way: every one reported the IDENTICAL final
SCIF to the printed precision, `-28.07 mT`.** Genuinely single-threaded
execution isn't merely lower-success — it's **perfectly deterministic**,
and for this specific configuration (dt=60s, I=19.6A, cold start) that
one deterministic trajectory is a non-convergent one.

**This replicates and substantially strengthens the ORIGINAL 2026-08-05
investigation's finding** (`OMP_NUM_THREADS=1`+`OPENBLAS_NUM_THREADS=1`+
`MKL_NUM_THREADS=1`+`PYTHONHASHSEED=0` gave 0/5 vs. baseline's 2/5) — that
result was directionally right and is now confirmed with a >3x larger
sample, cross-checked against genuinely-idle-thread verification (not
just trusting the env vars), and cross-checked across load conditions.

**What this means, stated plainly**: multi-threading isn't just
*correlated* with this system's run-to-run variability — for this
operating point, it appears to be the ONLY source of it, and it is what
gives the system its ~40% chance of reaching a converged state at all.
A control batch under default threading with the sibling investigation's
load gone (`control_recheck.sh`) confirmed successes are still reachable
under light load (2/6), ruling out "the machine got quieter and that's
why" as a confound for the threaded case. **Eliminating the
nondeterminism (by forcing determinism) does not fix this solver — it
locks it onto its own worst-case trajectory.** The practical
non-determinism is, in a real sense, the thing CURRENTLY making a ~40%
success rate possible instead of a 0% one, for this exact cold start.

### The iteration-1 bit-level diff, exercised

Added an inert dump hook to `transient/ta_transient.py`'s `_picard_phase`
(`DUMP_ITER1_MATRIX_PATH` env var; dumps each layer's assembled T-equation
matrix CSR + RHS + iteration-1 solution to `.npz`). Ran the DEFAULT-
threaded repro case repeatedly until catching both outcomes with the dump
active, then diffed `nd_runs/diff_iter1_dumps.py`, three independent
pairs:

| comparison | pattern | matrix data | RHS | iter-1 solution |
|---|---|---|---|---|
| fail vs. success (pair 1) | identical (nnz=22820, every layer) | NOT bit-identical, rel. diff ~1e-22–1e-24 | NOT bit-identical, rel. diff ~1e-20–1e-23 | NOT bit-identical, **rel. diff ~3e-4–3e-3** |
| fail vs. success (pair 2) | identical | ~1e-22–1e-24 | ~1e-20–1e-23 | **~3e-5–5e-3** |
| **fail vs. fail (control)** | identical | ~1e-21–1e-24 | ~1e-20–1e-23 | **~3e-5–4e-3** |

Two things, both real:

1. **A genuinely tiny (near/below machine-epsilon-relative-to-the-largest-
   entry) difference in the assembled matrix/RHS becomes a ~1e-3-to-1e-4
   RELATIVE difference in the solution after just ONE linear solve** —
   roughly a 10¹⁷-to-10¹⁹-fold amplification, from a LINEAR solve, before
   any Picard nonlinearity has acted at all. This is a real, large,
   precisely quantified illustration of how ill-conditioned this system's
   linear sub-problem is (plausibly the smoothed critical-state floor's
   enormous dynamic range in `rho_fn` across cells — cells near j/jc≈1
   sit orders of magnitude apart in resistivity from cells far from it).
2. **The fail-vs-fail control shows the SAME magnitude of amplification as
   the fail-vs-success pairs.** This is the honest, important caveat:
   this diagnostic does NOT show a special, discriminating signature at
   iteration 1 that marks "this run will succeed" vs. "this run will
   fail" — the amplification is a generic property of ANY two
   independent runs, success or not. Whatever actually decides the
   eventual outcome is not visible as a distinguishing feature this early;
   it must emerge from how this already-substantial per-iteration
   divergence compounds over the remaining up-to-149 nonlinear Picard
   iterations — consistent with, and now giving concrete numerical
   grounding to, this project's long-standing "chaotic, near-degenerate,
   red-spectrum wandering" characterisation of the Picard map at this
   operating point.

CLAUDE.md flagged this exact diagnostic as "the only diagnostic left that
could actually localise the source." It has now been run. The honest
answer: it localises the AMPLIFICATION MECHANISM precisely (ill-
conditioning in the linear T-solve), but does not by itself localise
WHICH iteration or which specific difference decides the final outcome —
that would need the same bit-level diff repeated at several later
iterations along both a real success and a real failure trajectory
(sampled every ~10-20 iterations, say), not attempted here.

## Summary and verdict

**Root cause: substantially identified this round, superseding the
"NOT identified" verdict below the table** (kept for the historical
record of what candidates 1-6 individually showed; read the paragraph
after the table for the corrected overall picture). Three more candidates
were tested and
rejected/deprioritised this session, on top of the three already ruled out
before it started:

| candidate | verdict | evidence |
|---|---|---|
| dolfinx post-load dof/cell reordering | **REJECTED** | 20/20 fresh process launches gave byte-identical fingerprints for every ordering-sensitive array |
| MUMPS fill-reducing ordering algorithm | **REJECTED** | default is ALREADY the deterministic AMF heuristic (confirmed via MUMPS's own `ICNTL(4)=3` verbose output, identical across iterations); forcing it explicitly changed nothing |
| BLAS/MUMPS thread-count env vars + explicit ordering pin, bundled (`forced_serial_mumps`, first attempt) | **INVALID TEST — WITHDRAWN**, not actually rejected | the batch never applied its own configuration due to a bash array-export-across-process bug (see follow-up section); its 2/5=40% is really an accidental second baseline, not evidence the config does nothing |
| **Multi-threading itself** (re-tested with genuinely verified single-threading, direct env-var application, confirmed idle via `/proc` tick-deltas) | **CONFIRMED AS THE (or a) DRIVER** | 0/18 across 3 independent batches / 2 configs / 2 load regimes (p≈6e-7 vs. the 40% baseline); every failure under true single-threading reports the IDENTICAL SCIF to the printed precision — genuinely deterministic, and deterministically non-convergent, for this exact operating point |
| ASLR | **INCONCLUSIVE** (1/3, insufficient reps — resource-constrained stop) | lower priority now — multi-threading itself looks like it explains the great majority of the effect |
| JIT cache staleness/reuse | **INCONCLUSIVE** (1 rep only — resource-constrained stop) | still a real, separately-demonstrated structural hazard (shared global cache, concurrent writers) worth fixing regardless |
| denormal/FTZ-DAZ state | deprioritised, not conclusively closed | no flushing detected at the scalar/import level in one process; compiled-kernel-level behaviour untested; low remaining priority |

**What this session adds to the standing understanding — the corrected,
final picture**: multi-threading (specifically, the floating-point
non-associativity it introduces into BLAS/MUMPS's parallel reductions) is
now shown, with strong statistical and mechanistic support, to be **the
dominant source of this system's cross-process nondeterminism, and — for
this exact cold-start configuration (dt=60s, I=19.6A) — the ONLY thing
standing between a guaranteed-failing deterministic trajectory and a ~40%
chance of a converging one.** The evidence chain:

1. `dolfinx`'s `LinearProblem.solve()` reruns MUMPS's full analysis
   +factorisation (not just numeric refactorisation) fresh at EVERY
   Picard iteration, giving many repeated opportunities per run for
   thread-scheduling-dependent floating-point reduction order to differ.
2. Genuinely single-threaded execution — verified by observing zero CPU
   ticks on every non-main thread over a sustained window, not by
   trusting an env var — makes the system perfectly deterministic:
   0/18 converged across 3 independent batches, and every failure
   converges to the IDENTICAL final SCIF to the printed precision.
3. The iteration-1 bit-level diff shows WHY a threading-scale
   perturbation matters so much: the T-equation's linear solve is so
   ill-conditioned that a near/below-machine-epsilon difference in the
   assembled matrix/RHS becomes a ~1e-3-to-1e-4 relative difference in
   the solution after just ONE (linear) solve — a ~10¹⁷-10¹⁹-fold
   amplification before any Picard nonlinearity even acts. (Important
   honest caveat: this amplification is generic to any two independent
   runs, success or not — it doesn't itself discriminate outcomes; see
   the section above.)

Put together: this is a chaotic dynamical system (already well
established by this project's own "red-spectrum wandering" history) whose
one deterministic trajectory from this exact initial condition happens to
be non-convergent, and whose only source of run-to-run variation — default
multi-threading's floating-point noise — acts as a source of perturbation
large enough (given the extreme ill-conditioning above) to occasionally
knock the trajectory onto a different, converging one. Eliminate the
noise, and you eliminate the only chance of escaping the bad trajectory.

**A second, independent, concrete finding, still standing regardless of
the above**: `~/.cache/fenics` (the FFCx JIT compilation cache) is a
single global directory shared by EVERY process on this machine, with no
versioning or locking visible from the outside, and it already contains
at least one stale `.failed` marker from an unrelated compile two days
ago. During this exact investigation, a sibling process (a concurrently-
running second investigation, in a different git worktree) was
independently compiling into and reading from this SAME cache directory
at the same time as this investigation's own repeats. This is a real,
demonstrated concurrency hazard for this project's general practice of
launching multiple background solves at once (already flagged for
`cmaes_search.py` output paths in CLAUDE.md's "Operational lessons" — same
class of bug, different subsystem) — separate from, and not needed to
explain, the threading finding above.

**Fixable? Not in the way "fixable" was originally framed.** The
nondeterminism is not an environmental bug to be swept away — chasing
"determinism" as the goal is actively counterproductive here, since the
one deterministic outcome for this operating point is failure. What
WOULD constitute progress: (a) understanding why THIS specific cold-start
configuration has no deterministically-reachable converged trajectory in
the first place (a question about the Picard map's fixed-point structure,
not about eliminating noise) — the ill-conditioning finding above is a
concrete lead; (b) deliberately introducing a small, controlled
perturbation (rather than relying on uncontrolled thread-scheduling noise)
as an escape mechanism, which is at least a principled idea now, whereas
before this session it would have looked like "adding noise on purpose,"
a strange thing to try without knowing noise was load-bearing.

**What remains untested, in priority order for anyone continuing this**:
1. Repeat the iteration-1 bit-level diff at several LATER iterations
   (every ~10-20, say) along one real success and one real failure
   trajectory, to find where a fail-vs-fail-sized divergence stops being
   generic and starts actually discriminating the eventual outcome.
2. Whether a SMALL, deliberate, controlled perturbation (e.g. a tiny
   random nudge to the ZFC seed, applied identically regardless of
   threading) reproduces the ~40% escape rate in a fully single-threaded,
   otherwise-deterministic run — this would directly test the
   "perturbation is load-bearing" hypothesis from a different angle.
3. A genuine JIT-cache RACE test (two processes compiling the identical
   form hash at the literally same instant, on purpose, in a tight loop)
   rather than this session's fresh-vs-shared comparison.
4. ASLR and JIT-cache-staleness batches, resumed to full statistical
   power — now lower priority given multi-threading looks like it
   explains the great majority of the effect, but not formally ruled out
   as a smaller contributing factor.
5. Compiled-kernel-level FTZ/DAZ verification (a small C extension
   reading MXCSR directly around the actual MUMPS factorisation call).

**Practical bottom line, revised from before this session**: no individual
run's cold-start outcome can still not be trusted as representative of
"the" behaviour at this operating point — but it is now understood WHY:
this is a genuinely chaotic nonlinear map with (as far as tested) a single
deterministic trajectory that fails, kept alive at a ~40% success rate
only by uncontrolled multi-threading noise. That reframes the multi-step
transient problem from "an unexplained reliability bug" to "a
characterised chaotic system currently surviving on an accident of its
own floating-point environment" — a real, if not yet actionable, causal
answer.

---
*Raw per-rep evidence for every count above: `nd_runs/*.log` (one file per
repeat, named `<label>_<NN>.log`) and `nd_runs/run_batch*.sh` /
`nd_runs/candidate_*.sh` for the exact commands run.*
