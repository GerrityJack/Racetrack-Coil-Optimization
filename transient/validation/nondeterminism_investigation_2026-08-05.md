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

---

## 2026-08-06 (continuation, coordinator-directed): controlled-jitter retry prototyped -- raises the success rate, but the specific contribution of the jitter itself (vs. just retrying) is not established by this round

Follow-up to this file's own flagged-but-unstarted lever: "deliberately
engineering a small controlled perturbation as an escape mechanism in
place of relying on uncontrolled thread noise." Prototype: attempt the
canonical repro case (dt=60s, I=19.6A, cold start, identical to
`first_step_diagnostic.py`) once unmodified; if it fails to converge
(`_picard_phase`, `max_iters=150, min_iters=6, scif_tol=0.5`, same as the
canonical case), cold-reset and retry with a small explicit random
perturbation (1e-3 relative to T_amp, seeded and logged) added to the T
seed, up to 4 retries (5 attempts total) within the SAME process launch.
New file: `transient/validation/jitter_retry_check.py`.

### Result: 8 independent process launches

| launch | attempt-0 alone | overall (up to 5 attempts) | attempts used |
|---|---|---|---|
| 1 | converged | converged | 1 |
| 2 | converged | converged | 1 |
| 3 | FAILED | converged | 4 |
| 4 | FAILED | converged | 3 |
| 5 | FAILED | **still failed** | 5 (exhausted) |
| 6 | FAILED | converged | 2 |
| 7 | converged | converged | 1 |
| 8 | FAILED | **still failed** | 5 (exhausted) |

**Attempt-0-only success: 3/8 = 37.5%** -- a good calibration check,
consistent with this file's own historical ~40% single-attempt baseline
for this exact repro case (confirms the harness faithfully reproduces
the known benchmark before drawing any conclusion from the retry
mechanism).

**Overall success within 5 attempts: 6/8 = 75%.** Retrying clearly helps
in absolute terms -- doubling the odds of eventually getting a converged
first step for this case, a genuinely useful practical result on its
own. Two of eight launches (25%) never converged even after exhausting
the full retry budget, so this is an improvement in odds, not a
reliable fix.

### The honest caveat: this does NOT cleanly demonstrate the JITTER is doing the work, as opposed to just retrying

If each attempt (whether jittered or not) had an independent ~37.5%
chance of success purely from whatever ambient thread-scheduling
variability exists between separate `_picard_phase` calls -- which this
design does NOT rule out, since attempts 1-4 still involve fresh
mesh-independent recomputation with its own genuine thread noise on top
of the added jitter -- the naive expectation for 5 independent attempts
at p=0.375 would be 1-(0.625)^5 ≈ 90%. The observed 75% is BELOW that
naive-independence prediction, not above it (though with only 8 samples
the confidence interval on a 75% observed rate is wide enough -- roughly
±30 points -- that this is not a strong claim of underperformance
either, just clearly not evidence of jitter adding value beyond plain
retrying).

**This experiment therefore established the RETRY strategy works (real,
useful, ~2x odds improvement) but did NOT isolate whether the explicit
controlled jitter contributes anything beyond what repeated attempts
would already get from ambient noise alone.** The clean follow-up,
not done here: a head-to-head control arm with `jitter_scale=0` (retries
that only cold-reset, with no added perturbation) run through the exact
same harness, to see whether the jittered arm's success curve actually
exceeds the zero-jitter retry arm's, rather than comparing against a
single-attempt baseline that both arms would beat.

### Practical recommendation

**Adopt the retry wrapper regardless of the jitter question** -- cold-
resetting and re-attempting a non-converged short-dt first step up to
~5 times is cheap relative to the cost of a single attempt (~50-150s
here) and roughly doubles the odds of eventually getting a converged
result for this specific repro case, with no evidence of it doing harm.
Whether to bother adding the explicit jitter on top of plain retries is
NOT resolved by this round -- worth the zero-jitter control run before
concluding jitter itself is worth the added complexity, but the retry
loop itself is a low-risk, positive-expected-value practical wrapper
independent of that question.

Scripts: `transient/validation/jitter_retry_check.py`. Raw per-launch
JSON results under a session scratch directory, not checked into the
repo.

---

## 2026-08-06 (continuation, coordinator-directed pivot): the deterministic single-threaded failure traced -- it is not a slow drift or a period-2 limit cycle, it is a persistent large-amplitude chaotic attractor with T overshooting ~100x its own boundary-condition scale, from iteration 5 onward, never decaying

Directed pivot away from the jitter-retry line (which was raising the
success rate but not resolving whether that was a real fix or just
selection among noise, and could not establish that a jitter-forced
"converged" run was landing on a trustworthy answer -- see the
just-preceding section). Redirected to the more fundamental question this
file's own "Not yet done" item pointed at: what does the deterministic
failing trajectory actually DO, mechanistically, when every source of
run-to-run noise is removed?

### Method

Reused this file's own verified single-threaded recipe exactly
(`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1 OMP_THREAD_LIMIT=1`, `TA_MUMPS_EXTRA_OPTS` forcing
sequential MUMPS analysis/ordering, isolated `XDG_CACHE_HOME`), on the
SAME canonical repro case (`dt=60s, I=19.6A`, cold start) this file's
0/18 result was measured on. New file:
`transient/validation/jitter_retry_trace_check.py` (built the previous
round for a different purpose, reused here with `jitter_scale=0` -- no
injected perturbation, no retry, ONE deterministic run), instrumented via
`_picard_phase`'s own `closure` hook to independently record, every
iteration: relative field change `|dB|/|B|` (recomputed independently,
not read from `_picard_phase`'s internal state), the RAW (non-EMA)
instantaneous SCIF, and `T_max`/`T_min` across all layer T-functions.

**Reproducibility re-confirmed, at the full trajectory level this time
(not just the pass/fail outcome):** two independent process launches
under this config produced BIT-IDENTICAL 150-iteration traces -- every
recorded value, at every iteration, matched exactly. This extends the
prior 0/18 pass/fail confirmation to the strongest possible form: not
just "both runs fail," but "both runs fail via the literal same sequence
of states."

### What the trajectory actually looks like

**It never approaches anything resembling the physical solution.**
`T_max`/`T_min` (normalized by `T_amp = I/(2*delta_SC) = 9.8e6`, the
actual boundary-condition scale) overshoot to roughly **+150x/-98x
T_amp by iteration 5** -- extremely early, not a late-stage blowup --
and then stay parked in a persistent, bounded, large-amplitude band
(roughly +55x to +150x on the high side, -80x to -125x on the low side)
for the ENTIRE REMAINING 145 iterations, with no growth trend and no
decay trend. The raw (non-EMA) SCIF swings correspondingly: means per
30-iteration chunk of +308, +240, +78, +99, +161 mT (no monotonic trend)
with per-chunk standard deviations of 200-790 mT -- the same order as
the means themselves, i.e. genuinely noisy/wandering, not settling.

**`|dB|/|B|` (relative field change per iteration) never decays below
~0.5-0.9 (50-90% relative change EVERY iteration) for the entire run.**
Chunked means: 0.734, 0.779, 0.827, 0.816, 0.829 across the five
30-iteration windows -- if anything creeping UP slightly over the run,
not down. This is not "a small residual oscillation superimposed on an
otherwise-converged state" (which is what the EMA-based stall criterion
is designed to smooth past) -- the field is changing by most of its own
magnitude every single iteration, for 150 iterations straight.

**Not a simple period-2 limit cycle.** The autocorrelation of the raw
SCIF sequence is +0.545 at lag 1, decaying through +0.470, +0.323, +0.077
at lags 2-4, crossing to slightly negative (-0.01 to -0.22) at lags 5-10.
A clean period-2 cycle (the SPECIFIC failure mode this project's own
two-phase relaxation scheme was built to fix at dt=600s, per CLAUDE.md's
bug history) would show strong alternating correlation (~-1 at odd lags,
~+1 at even lags) -- this autocorrelation shape rules that out. The
dynamics here are higher-dimensional/genuinely chaotic wandering within
a bounded attractor, not a simple few-state cycle.

### Why this reframes the standing picture, not contradicts it

Prior characterisation (this file, above): a near-machine-epsilon input
difference amplifies ~1e17-1e19-fold through one ill-conditioned linear
solve, and independent noise realisations decorrelate to full O(1)
difference by iteration ~20-25 of a Picard run. That remains true and is
not retracted. What this round adds: the SPECIFIC deterministic
trajectory that noise was occasionally escaping from is not "a nearly-
good state perturbed off course" -- it is a trajectory that overshoots
by ~100x its own physical scale within the first 5 iterations and then
never leaves a persistent chaotic attractor centred on that
unphysically-large overshoot, for the entire 150-iteration budget. The
two-phase relaxation scheme (validated, reliable at dt=600s) provides
**no effective damping whatsoever** in this regime -- `|dB|/|B|` sitting
at 70-90% throughout means the Picard map is simply not contractive
here, with or without noise, at either its fast (alpha=0.30) or careful
(alpha=0.15) relaxation setting.

**Leading mechanistic hypothesis, not yet directly tested this round:**
`first_step_diagnostic.py`'s own module docstring already flagged the
candidate explanation, written before this trace existed: a small `dt`
inflates the T-equation's `1/dt` forcing coefficient
(`curl(A_h - A_prev)/dt`) relative to the resistive (damping) term. At
`dt=60s` that coefficient is 10x larger than at the validated `dt=600s`
operating point -- directly changing the effective contraction ratio of
the Picard fixed-point map, not merely making it "more noise-sensitive."
This would predict a reasonably sharp TRANSITION in behaviour somewhere
between `dt=60s` (chaotic attractor, as traced here) and `dt=600s`
(reliable convergence, this project's whole validated production path)
as `dt` is swept between them -- not yet tested.

### What noise is actually doing, under this revised picture

The ~40% success rate under normal (multi-threaded, noisy) execution is
not "noise nudging a nearly-converged trajectory the rest of the way."
It is noise occasionally sending the VERY EARLY trajectory (by iteration
2-5, per the amplification timeline this file already established) down
a qualitatively different path that avoids this chaotic attractor
entirely, into the genuine convergence basin, rather than into it. Which
basin a given launch lands in is decided essentially immediately, within
the first handful of iterations -- consistent with, and now mechanistically
sharpened by, everything already established about where the
amplification concentrates.

### Recommended next step (not yet done, flagged for direction)

A `dt` sweep (e.g. 600, 300, 150, 100, 60 -- reusing the same traced
single-threaded harness) would show whether this is a genuinely sharp
bifurcation at some critical `dt`, or a gradual degradation -- and would
directly test the `1/dt` forcing-coefficient hypothesis above. That is a
natural, well-motivated continuation of this specific finding, not
attempted in this round given the pivot's own scope.

Scripts: `transient/validation/jitter_retry_trace_check.py`. Raw traces
(`det_run1.json`, `det_run2.json`, bit-identical) under a session
scratch directory, not checked into the repo.

---

## 2026-08-06 (continuation, coordinator-directed): eliminate the noise, THEN fix the actual instability -- a genuinely smaller (fixed, non-adaptive) relaxation factor converges the canonical case cleanly and deterministically

Directed follow-up to the trace above: rather than continuing to chase
noise-driven escape mechanisms (jitter/retry), get rid of the noise
first (same verified single-threaded recipe as every result in this
file) and then try to fix the actual instability the trace just
characterised. New file: `transient/validation/alpha_sweep_trace_check.py`
-- same trace instrumentation as the previous round, but overrides
`params.ta_picard_alpha`/`params.ta_picard_alpha_fine` (the two
relaxation factors `_picard_phase` already reads via `getattr`, confirmed
from its own source -- no code change to `_picard_phase` itself) before
calling it. This is deliberately NOT an adaptive alpha-throttle -- this
project's own history warns explicitly against reintroducing one without
testing against the sharp-flux-front dataset it broke on before. This
keeps the exact same validated two-phase STRUCTURE (fast ramp-up until
`|dB|` stops decreasing, then a fixed slow phase) and only changes the
two alpha VALUES.

### First test: alpha=(0.10, 0.05) -- ~3x smaller than the dt=600s-validated (0.30, 0.15) defaults

`converged=True` at n_iters=56 -- but the SAME false-positive pattern
flagged by the user two rounds ago: the raw (non-EMA) SCIF was still
swinging by 210.5 mT over the last 10 iterations, the last iteration's
relative field change was still 67%, and `T_max` was still **31x** its
own physical boundary-condition scale (`T_amp`). Real improvement over
the default alpha (which peaked at ~150x `T_amp` and never came down),
but not real convergence -- the EMA-based stall check is not reliable in
this regime at ANY alpha tested so far, and every conclusion below relies
on the raw trace, not the `converged` flag.

### Second test: alpha=(0.03, 0.01) -- ~10x smaller than defaults, max_iters raised to 1000 -- GENUINE, CLEAN, REPRODUCIBLE CONVERGENCE

**This one is real.** Full trajectory (`T_max` normalized by `T_amp`,
`dB_rel` = relative field change, raw instantaneous SCIF):

| iter | T_max/T_amp | dB_rel | scif_raw (mT) |
|---|---|---|---|
| 5 | 31.2 | 1.84 | +1801.3 |
| 20 | 8.2 | 0.094 | +891.8 |
| 50 | 2.33 | 0.087 | +689.8 |
| 90 | 1.13 | 0.080 | +503.0 |
| 170 | 0.89 | 0.069 | +294.0 |
| 290 | 0.96 | 0.063 | +175.3 |
| 410 | 1.02 | 0.057 | +139.8 |
| 458 | 1.03 | 0.071 | +134.0 |

`T_max/T_amp` decays MONOTONICALLY (with only minor noise) from a 31x
initial overshoot down through 1.13, dips slightly below 1 (0.89, a mild
undershoot -- physically unremarkable, not a sign of instability) then
settles smoothly back up to ~1.03 and stays there. `dB_rel` drops
sharply after the first few iterations and then sits in a stable,
NON-GROWING 0.06-0.09 band for the remaining ~400 iterations. Raw SCIF
decreases smoothly and asymptotically from +1801 mT to +134 mT, tracking
`1/n`-ish decay all the way -- a textbook convergence curve, nothing
resembling the chaotic wandering seen at default alpha.

**Reproducibility: bit-identical across two independent process
launches** (both single-threaded, same verified-deterministic recipe) --
every one of the 459 recorded iterations matched exactly between the two
runs, not just the final summary numbers. Same standard this file has
applied to every other claim: not accepted on a single run.

**Final converged state: `scif=+133.9mT`, `T_max/T_amp=1.03`, at
`n_iters=459`.** A physically sensible SCIF magnitude (for context, the
validated dt=600s/I=196A production point reaches ~641mT at 10x the
current and 10x the dt -- a few-hundred-mT SCIF at this much smaller
current/dt is not an unreasonable order of magnitude) and a T value
within 3% of its own boundary-condition scale, not the 30-150x
overshoot seen at every larger-alpha configuration tried.

### What this establishes

**The underlying physics/PDE problem at dt=60s/I=19.6A is NOT inherently
chaotic or unstable.** What looked like deterministic chaos in the
previous round's trace was the two-phase relaxation scheme's alpha
values -- tuned and validated for `dt=600s` -- being genuinely too
aggressive for `dt=60s`, pushing the Picard fixed-point map's effective
contraction ratio above 1 in this regime. A ~10x smaller (still fixed,
still two-phase, still non-adaptive) relaxation factor restores
contraction and produces clean, smooth, reproducible convergence to a
sensible answer, with NO jitter, NO retries, NO reliance on
noise-driven escape -- a completely deterministic single run gets there
on its own.

**This reframes the entire session's line of investigation.** Every
prior remedy this session tried (n-continuation, the Bean seed,
jitter-retry) was implicitly treating the target `n(B,theta)` exponent,
the relaxation SCHEDULE (alpha_high -> alpha_low switch timing), or the
starting point as the lever -- never the raw MAGNITUDE of alpha itself
at this specific dt. None of those approaches touched the actual
instability; this one goes directly at it.

### Important open caveats, not yet resolved

1. **Cost.** ~460 iterations vs. this project's usual 30-80 at the
   validated dt=600s operating point -- a real, non-trivial cost
   increase, though not prohibitive (the run itself took under 3 minutes
   single-threaded).
2. **Not yet checked under normal (multi-threaded) execution.** Everything
   above is single-threaded. Whether `alpha=(0.03, 0.01)` also converges
   reliably (not just once) under ordinary multi-threaded execution --
   where the SAME floating-point noise this file has characterised all
   along is still present -- has not been tested. A smaller alpha
   generically INCREASES a fixed-point iteration's stability margin
   against small perturbations too, so there is real reason to expect
   this also fixes the noise-driven unreliability directly, without
   needing jitter or retries at all -- but that is a prediction, not yet
   an observation.
3. **Boundary not mapped.** Alpha=(0.10, 0.05) still failed (31x
   overshoot); alpha=(0.03, 0.01) converged cleanly. Where between those
   the transition actually sits, and whether it is sharp or gradual, is
   unknown -- not essential to the headline finding, but useful for
   picking an efficient (not overly conservative) production value.
4. **Only tested at ONE (dt, I) point.** Whether `alpha=(0.03, 0.01)` (or
   whatever the eventual chosen value is) generalises across the range of
   `dt`/`I` a real multi-step ramp would need, or needs to itself be
   `dt`-dependent, is untested.

### Recommended next steps, in priority order

1. Re-run `alpha=(0.03, 0.01)` (or nearby) WITHOUT forcing single-threaded
   execution, repeated several times, to see whether it also resolves the
   multi-threaded reliability problem directly -- this is the test that
   would turn this from "a clean single-threaded curiosity" into "an
   actual production fix."
2. If (1) confirms it, map the alpha boundary more precisely and check
   generalisation across a few more `(dt, I)` points before considering
   this validated for the full multi-step ramp use case.

Scripts: `transient/validation/alpha_sweep_trace_check.py`. Raw traces
(`a010_005.json`, `a003_001.json`, `a003_001_rep2.json` -- the latter two
bit-identical) under a session scratch directory, not checked into the
repo.

---

## 2026-08-06 (continuation): alpha=(0.03, 0.01) confirmed under NORMAL (multi-threaded, noisy) execution -- 5/5 genuine convergence, tightly clustered SCIF, direct resolution of the accuracy concern raised earlier in this file

Follow-up to the single-threaded confirmation above: does the same
relaxation fix also resolve the multi-threaded reliability problem this
entire file is about, not just the isolated deterministic case? Ran
`alpha_sweep_trace_check.py 60 19.6 0.03 0.01 1000` FIVE times under
ordinary, unforced execution (no `OMP_NUM_THREADS`/single-thread env vars
-- genuinely multi-threaded, confirmed by `user` time far exceeding
`real` time on every run, e.g. one run's 52.5 CPU-minutes over 7.1
wall-clock minutes, ~7.4x parallelism).

### Result: 5/5 genuine convergence, all judged by the raw diagnostics (not the EMA flag alone)

| run | n_iters | scif_final (mT) | dB_rel_last | T_max/T_amp | last10 scif spread (mT) |
|---|---|---|---|---|---|
| 1 | 458 | +131.510 | 0.044 | 1.141 | 0.885 |
| 2 | 460 | +131.319 | 0.039 | 1.144 | 0.886 |
| 3 | 458 | +131.522 | 0.030 | 1.141 | 0.892 |
| 4 | 458 | +131.517 | 0.042 | 1.146 | 0.890 |
| 5 | 459 | +131.411 | 0.027 | 1.144 | 0.890 |

Every run: `dB_rel_last` < 0.05 (vs. 0.5-0.9 for the old default-alpha
chaotic case), `T_max/T_amp` clustered at 1.14-1.15 (vs. 30-150x
overshoot before), `last10_scif_spread` a tight ~0.89 mT band across ALL
FIVE runs independently. None of these show the false-positive signature
(EMA says converged, raw signal still swinging) found earlier in this
same file at `alpha=(0.10, 0.05)` -- every one of these is a genuine,
verified convergence.

### This directly answers the accuracy question raised earlier in this file

**SCIF values across all 5 multi-threaded runs: 131.32 to 131.52 mT -- a
spread of 0.2 mT out of ~131 mT, about 0.15% relative.** Compare this to
the OLD default-alpha jitter/retry runs' converged-SCIF scatter:
-21.5, -1.3, +74.2 mT -- wildly different magnitudes AND signs, from
runs that were ALL reported "converged" by the same EMA flag. The fix
does not just raise the raw convergence rate (~40% -> 100% here, on a
small but clean sample); it resolves the accuracy/consistency problem
too -- 5 independent noisy launches now land on effectively the SAME
physical answer instead of 5 different, uncorrelated ones.

The single-threaded run from the previous round (`scif=+133.9mT`,
`n_iters=459`) is close to but not bit-identical with this multi-threaded
cluster (~131.3-131.5mT) -- expected and unremarkable: different
threading modes take different exact floating-point paths to the SAME
true fixed point, converging to values that agree to ~2%, not to machine
precision. This is categorically different from, and far tighter than,
the wild scatter this file characterised throughout its 2026-08-05
entries for the OLD default-alpha configuration.

### Net verdict

**`alpha=(0.03, 0.01)` is now a validated fix for the canonical
`dt=60s, I=19.6A` repro case specifically** -- both the single-threaded
deterministic chaos (previous round) and the multi-threaded
reliability/accuracy problem (this round) are resolved by the same
change, with no jitter, no retry, no reliance on noise at all. 5/5
genuine convergence (verified by raw diagnostics, not the EMA flag
alone) to a tightly-clustered answer is strong evidence for THIS
specific point, on the sample size tested.

**What is still open before this generalises to "the multi-step
transient problem is solved":**
1. Only tested at ONE `(dt, I)` combination. A real ramp needs many
   different `dt`/`I` values; whether `alpha=(0.03, 0.01)` is
   universally sufficient, needs to be `dt`- or `I`-dependent, or needs
   occasional retuning, is untested.
2. Cost: ~458-460 iterations here vs. ~30-80 at the validated `dt=600s`
   production point -- real, though not prohibitive (each multi-threaded
   run took ~7 minutes wall-clock, itself slower than the single-threaded
   ~3 minutes due to thread overhead on a problem this size -- worth
   knowing when choosing execution mode for any future run of this kind).
3. The alpha boundary between "still chaotic" (0.10, 0.05) and "cleanly
   convergent" (0.03, 0.01) has not been narrowed -- unknown whether a
   less conservative (faster) value in between would also work.
4. Not yet tested within an actual multi-step RAMP (this and everything
   in this file has only ever tested a single first step from cold
   start) -- the original motivating problem was genuine multi-step
   time-marching, which this result does not directly demonstrate yet.

Scripts: `transient/validation/alpha_sweep_trace_check.py`. Raw traces
(`mt_timing1.json`, `mt_batch/run{2..5}.json`) under a session scratch
directory, not checked into the repo.

---

## 2026-08-06 (continuation, coordinator-directed thorough test, Phase 1): dt/I generalization sweep -- the fix holds across current and down to dt=60s, but fails again at dt=30s; the production dt=600 point costs ~9x more iterations under the smaller alpha

User-directed thorough validation of `alpha=(0.03, 0.01)` before trusting
it beyond the single `dt=60s, I=19.6A` point. 9-point single-threaded
sweep (same verified recipe, `alpha_sweep_trace_check.py`, max_iters=1500
per point): `dt` in {600, 300, 150, 100, 60*, 30} at `I=19.6A` (*60s
already established in the prior round, not rerun), `I` in {49, 98, 196}
at `dt=60s`, plus `dt=600s, I=196A` -- the ACTUAL validated production
operating point -- as a regression sanity check.

### Results (all judged by raw diagnostics, not the `converged` flag alone)

| point | n_iters | scif (mT) | dB_rel_last | T_max/T_amp | last10 spread (mT) | verdict |
|---|---|---|---|---|---|---|
| dt=600, I=19.6 | 453 | +116.4 | 0.038 | 0.991 | 0.884 | GENUINE |
| dt=300, I=19.6 | 455 | +124.3 | 0.057 | 0.991 | 0.885 | GENUINE |
| dt=150, I=19.6 | 457 | +129.5 | 0.059 | 0.991 | 0.887 | GENUINE |
| dt=100, I=19.6 | 457 | +131.8 | 0.074 | 0.991 | 0.885 | GENUINE |
| dt=60, I=19.6 (prior round) | 458-460 | +131.3 to +133.9 | 0.03-0.07 | 1.03-1.15 | ~0.89 | GENUINE |
| **dt=30, I=19.6** | **189** | **+292.0** | **1.110** | **10.87** | **74.9** | **FALSE POSITIVE -- NOT converged** |
| dt=60, I=49 | 553 | +310.3 | 0.031 | 1.018 | 0.873 | GENUINE |
| dt=60, I=98 | 621 | +560.4 | 0.012 | 0.998 | 0.904 | GENUINE |
| dt=60, I=196 | 696 | +799.4 | 0.007 | 0.999 | 0.878 | GENUINE |
| dt=600, I=196 (production point) | 694 | +663.2 | 0.007 | 0.999 | 0.886 | GENUINE |

### Three findings

**1. Smooth, physically sensible SCIF trend across the dt sweep at fixed
I=19.6A**: 116.4 -> 124.3 -> 129.5 -> 131.8 -> ~133 mT as `dt` decreases
from 600 to 60s -- monotonic, no discontinuity, no sign of a
scheme artifact. This is what a real, physically-converged trend should
look like.

**2. The fix is NOT universal -- it fails again at dt=30s.** Every
raw diagnostic flags this point as a false positive exactly like the
old default-alpha failures: `T_max/T_amp=10.87` (an 11x overshoot,
not the ~1.0 every genuine point shows), `dB_rel_last=1.11` (110%
relative change on the last iteration), `last10_scif_spread=74.9mT`
(vs. ~0.88mT at every genuine point) -- yet the EMA flag still said
`converged=True`. **This directly confirms the `1/dt`-forcing-coefficient
mechanism proposed earlier**: `alpha=(0.03,0.01)` was enough damping for
the forcing term's size at `dt=60s`, but not at `dt=30s`, where that term
is 2x larger again. The fix needs to be `dt`-dependent (or found via a
proper boundary search), not a single universal constant.

**3. The fix generalises cleanly across CURRENT at the hard `dt=60s`
point** -- `I`=49, 98, and 196A (the full champion design current) all
converge genuinely, with `T_max/T_amp` settling closer to 1.0 (0.998-1.02)
and `dB_rel_last` getting SMALLER (0.031 -> 0.012 -> 0.007) as current
increases -- if anything easier at higher current, not harder. Current
level is not the sensitive axis; `dt` is.

### Production-point regression check: works, but costs ~9x more iterations

`dt=600s, I=196A` (the actual validated champion operating point) DOES
converge genuinely under `alpha=(0.03,0.01)`: `scif=+663.2mT`,
`T_max/T_amp=0.999`, `dB_rel_last=0.0065` -- all clean. But it needed
**694 iterations**, versus this project's documented 60-80 iterations at
this exact point under the validated DEFAULT `alpha=(0.30,0.15)` -- **an
approximately 9x increase**. There is a real, non-trivial cost to
adopting the smaller alpha universally; this points toward a `dt`-aware
(or continuation-based) alpha selection for production use, not a
blanket replacement of the validated defaults.

One nuance flagged, not resolved: `663.2mT` is close to but not
identical with this project's own established `641.26mT` ground truth at
this exact point (from the default-alpha, already-validated scheme) --
about 3.4% different. Plausibly within scheme-dependent variation
(different relaxation paths reaching a not-perfectly-unique fixed
point), or possibly `694` iterations has not yet FULLY settled given how
slowly `alpha_low=0.01` converges asymptotically -- not distinguished
here, flagged as a loose end rather than a red flag.

### Net read

The core mechanism understanding holds up under real testing: this is a
genuine relaxation-parameter/damping problem tied to the size of the
`1/dt` forcing term, not a mysterious universal chaos. The fix
generalises across current cleanly, degrades gracefully across `dt` down
to the tested boundary, and reproduces the right order of magnitude (if
not to the last mT) at the fully validated production point. It is NOT a
universal constant -- `dt=30s` needs something smaller still (or a
different, `dt`-scaled approach), and adopting it everywhere would cost
production runs at `dt=600s` a real ~9x iteration-count penalty.

Scripts: `transient/validation/alpha_sweep_trace_check.py` (unchanged).
Raw traces for all 9 points under a session scratch directory, not
checked into the repo.

---

## 2026-08-06 (continuation, user-directed): forced-full-length runs (min_iters=max_iters, bypassing `_picard_phase`'s own early-exit) reveal the "converged" flag was firing well before genuine settling -- the minority-layer anomaly DOES resolve, but the true converged SCIF differs meaningfully from every number reported so far in this file

Direct follow-up to the user's question: do these alpha=(0.03,0.01) runs actually
approach the validated one-step (`dt=600s`) ground truth? First attempt at a
3000-iteration long run (previous entry) silently stopped at iteration 459 --
`_picard_phase`'s own EMA stall check exited early despite `max_iters=3000`,
because `min_iters=6` let it. Fixed by setting `min_iters=max_iters` in
`per_layer_diag_check.py` (now also tracking SCIF per checkpoint), forcing the
solver to run the full requested length regardless of what its own stall
check thinks, and reran both `dt=60s, I=19.6A` and `dt=600s, I=196A`
(the actual validated production point) for a genuine 3000 iterations each,
single-threaded.

### dt=60s, I=19.6A: the minority layers DO settle -- by ~750 iterations, not ~460

| iter | SCIF (mT) | layer4 min/amp | layer5 min/amp |
|---|---|---|---|
| 300 | +170.5 | -81.3 | -82.1 |
| 750 | +125.1 | -82.4 | -86.4 |
| 1500 | +124.6 | -82.4 | -86.4 |
| 2250 | +124.6 | -82.4 | -86.4 |
| 2999 | +124.6 | -82.4 | -86.4 |

**Genuinely flat from iteration 750 onward** -- not decelerating-but-still-moving
as it appeared at iteration 300-458 in every earlier run, actually stable to
3-4 significant figures across 2250 more iterations. The minority-layer
concern raised earlier in this file is RESOLVED as a "needs more iterations"
issue, not an unbounded pathology -- good news, as far as it goes.

**But this reveals every earlier claim about this exact configuration was
based on a premature stopping point.** The EMA flag fired at iteration
~459-460 in every prior run at this `(dt, I)` (the single-threaded confirmation,
all 5 multi-threaded confirmation runs, the entire generalisation sweep's
`dt=60` baseline) reporting `scif≈+131-134mT`. **The TRUE, fully-settled value
is +124.6mT** -- about 6-7% lower. The earlier "5/5 genuine convergence,
SCIF values within 0.15% of each other" finding is NOT retracted as a
run-to-run CONSISTENCY result (5 independent noisy launches agreeing with
each other to 0.15% is still a real, meaningful improvement over the old
default-alpha scatter) -- but it was consistency at a shared premature
stopping point, not evidence of having reached the actual fixed point. Both
things were true at once: better raw-diagnostic checks (T_max/T_amp, dB_rel,
short-window SCIF spread) than the bare EMA flag, but still not sufficient to
catch that ~460 iterations undershoots true convergence by a wide margin at
this `(dt, I)`.

### dt=600s, I=196A (the validated production point): forced settling brings the answer MUCH closer to the known ground truth, but not exactly onto it

| iter | SCIF (mT) | layer4 min/amp | layer5 min/amp |
|---|---|---|---|
| 300 | +1145.8 | -3.43 | -3.40 |
| 750 | +659.2 | -3.32 | -3.26 |
| 1500 | +653.9 | -3.32 | -3.26 |
| 2250 | +653.9 | -3.32 | -3.26 |
| 2999 | +653.9 | -3.32 | -3.26 |

**No minority-layer pathology here at all** -- layer4/5 settle to the same
modest range as every other layer, matching the default-alpha result almost
exactly (confirming again this is a short-`dt`-specific issue, not general to
small alpha). SCIF settles cleanly and flatly from iteration 750 onward at
**+653.9mT**.

**Compared to this project's established ground truth at this exact point,
`641.26mT`** (from the validated default-alpha scheme): the forced-full-length
result (`653.9mT`) is **1.97% off** -- a real improvement over the
premature-stop estimate from the earlier round (`663.2mT` at 694 iterations,
`3.4%` off), but not an exact match. The trajectory is flat to 3-4
significant figures across the last three checkpoints (1500-2999), so this
does not look like it is still slowly drifting toward `641.26mT` given more
time -- it looks like a genuinely different, stable fixed point, about 2%
away from the reference value.

**This 2% residual gap is not explained.** Since the Picard relaxation factor
`alpha` is a pure numerical-scheme parameter that should not appear in the
underlying fixed-point equations themselves, two different alpha schedules
converging to two DIFFERENT stable fixed points (not just different
transient paths to the same one) is a genuine open question, not a rounding
detail -- plausible candidates, none confirmed: the `641.26mT` reference
value's own convergence may itself not have been checked this rigorously
(it predates this file's raw-diagnostic standard); or the smoothed
critical-state floor's log-space rho-relaxation may have some genuine
path-dependence this project has not previously had reason to probe.

### What this means, plainly

**Revise the cost estimate upward, substantially.** True convergence at
`alpha=(0.03,0.01)` needs on the order of 750-1500+ iterations depending on
`dt`, not the 460-700 assumed through every earlier round today -- a bigger
production-cost tradeoff than previously stated.

**The fix is not fully validated against the known ground truth.** It gets
much closer once genuinely converged (1.97% vs 3.4% gap), which IS a
meaningful positive result and rules out "wildly wrong" -- but a 2%,
currently-unexplained discrepancy against this project's own established
reference value is a real open item, not a footnote. Every number reported
earlier today about "genuine convergence" at this configuration should be
read as "internally consistent at iteration ~460, not fully converged" --
this file's own standing lesson (never trust a status flag over an
independent check) applied one layer too shallow, and has now been applied
one layer deeper.

**Recommended next steps, not yet done:**
1. Understand the 2% gap at the validated `dt=600s` point specifically --
   run the DEFAULT alpha to a similarly rigorous forced-full-length,
   raw-diagnostic-verified standard (has `641.26mT` itself ever been checked
   this carefully, or does it also predate this level of scrutiny?) before
   concluding which of the two numbers, if either, is "more correct."
2. Re-run the multi-threaded reliability confirmation (5 runs, previously
   showing 0.15% SCIF spread) at the TRUE convergence horizon (750+
   iterations, not ~460) to see whether the tight clustering finding still
   holds at the actual fixed point, not just at a shared premature stopping
   point.
3. Reconsider the iteration-count/cost tradeoff for any future use of this
   fix given the revised, larger true iteration requirement.

Scripts: `transient/validation/per_layer_diag_check.py` (now with SCIF
tracking and `min_iters=max_iters` forced full-length runs). Raw logs under
a session scratch directory, not checked into the repo.

---

## 2026-08-06 (continuation): the "2% gap vs 641.26mT ground truth" was a test-harness artifact, NOT an alpha-fix problem -- root cause not found, but decisively shown to be independent of alpha

Follow-up to the previous entry's forced-full-length dt=600s/I=196A
result: default alpha (0.30, 0.15), run through this file's own test
harness (`_picard_phase` from `ta_transient.py`, via
`per_layer_diag_check.py`), ALSO settles at +653.9mT, not the
project's established `641.26mT` ground truth -- meaning the ~2% gap
exists independent of which alpha is used, and was never actually
about the `alpha=(0.03,0.01)` fix at all.

### Confirmed: the official production path still gives 641.26mT today

Ran `transient/validation/accuracy_check_I196.py` (unmodified, the
original script that established the ground truth) directly. It calls
`ta_solve.solve_ta_at_current()` -- a SEPARATE, independent
implementation of the T-A Picard loop, embedded directly in `ta_solve.py`,
not the same code as `ta_transient.py`'s `_picard_phase` -- and still
reproduces **+641.27mT**, converging cleanly and monotonically over 85
iterations with well-behaved `|ΔB|/|B|` (~3-5e-3 throughout, never
spiking). `params.py` has not drifted; this is a live, current-codebase
result, not a stale historical one.

### Five isolation attempts, all correctly reproducing 641mT, failed to identify why `_picard_phase`-based scripts give 653.9mT instead

Built `transient/validation/loop_isolation_check.py` (mesh/seed built
byte-for-byte matching `solve_ta_at_current`'s own inline cold-start
block, then `_picard_phase` called on it) -- gives **+641.175mT**,
matching official almost exactly. Confirmed the loop function itself is
NOT the problem. Systematically tested every remaining hypothesis, each
its own script, each STILL giving the correct ~641mT:
- `_check2.py`: seed order swapped (A-seed before vs after setting
  `T_bot_val`/`T_top_val`) -- both orders give ~641mT.
- `_check3.py`: calling the actual `_seed_cold()` function (imported from
  `ta_transient.py`, as every earlier script today did) instead of a
  manually-inlined seed -- still ~641mT.
- `_check4.py`: explicitly setting `params.ta_picard_alpha`/
  `ta_picard_alpha_fine` before mesh-building (confirmed via direct grep
  that `ta_solve.py` never reads these outside the loop functions, so
  timing cannot matter) -- still ~641mT. Adding a diagnostic closure that
  calls `B_fn.interpolate()` and `_J_from_T()` every iteration (matching
  `per_layer_diag_check.py`'s closure) -- still ~641mT.

Yet `per_layer_diag_check.py` itself -- which uses the SAME `_seed_cold`,
the SAME seed order, the SAME explicit alpha-setting, and a closure that
was shown not to matter -- reliably gives **+653.9mT**, reproduced
**bit-identically across 3 independent single-threaded launches**
(iter-by-iter SCIF matching to the printed decimal at every one of ~20
checkpoints, e.g. all three giving exactly `+2674.838mT` at iteration 5).
This rules out run-to-run noise/multi-basin sensitivity as the
explanation -- whatever the difference is, it is fully deterministic, not
random.

### A real, small, deterministic seed-level difference was found -- amplified, not created, by the iteration

Dumping checksums (`.sum()`) of `J_coil`, `rho_fn` immediately after
seeding: `J_coil.sum()` matched EXACTLY between `per_layer_diag_check.py`
and the isolation scripts; `rho_fn.sum()` differed at the 5th significant
figure (`2.990060e-14` vs `2.990128e-14`, ~0.002% relative) -- small but
real, and NOT attributable to mesh non-reproducibility (this project's own
earlier investigation confirmed byte-identical mesh files across
processes; both these runs were single-threaded, ruling out thread-order
noise too). After just ONE Picard iteration (`max_iters=1`, forced),
SCIF is already `+9262.586mT` vs. the official log's own iteration-1 value
of `+9251.68mT` -- a 0.118% difference, roughly 50x amplified from the
seed-level ~0.002% discrepancy. This is consistent in KIND with this
whole file's established finding that the T-equation's linear sub-problem
strongly amplifies small differences -- just far more muted here (dt=600s
stays stable and cleanly convergent, unlike dt=60s/30s) -- compounding
over ~100+ iterations into the visible ~2% final gap.

**Root cause of the seed-level `rho_fn` discrepancy itself was NOT found**
despite five targeted isolation attempts. Given the amplification
mechanism is already well-understood and this is clearly consistent with
(not contradicting) everything else established today, further
bisection was stopped here as a scope decision, not because the
question is resolved.

### The critical clarification this provides

**The `alpha=(0.03,0.01)` fix is NOT the source of the ~2% discrepancy
from `641.26mT`.** Both default alpha and small alpha, run through the
SAME test harness (`_picard_phase`), converge to the SAME fixed point
(+653.9mT, agreeing with each other to <0.02%) -- the alpha-vs-alpha
comparison done throughout today's testing remains entirely valid. What
was NOT valid was comparing either of those numbers against `641.26mT`,
since that number comes from a DIFFERENT code path
(`solve_ta_at_current`) with its own small, unexplained numerical
difference from `_picard_phase` -- present regardless of which alpha is
used, including the already-validated default one.

### Practical recommendation

**For any future validation that needs to match this project's
established reference values, use `ta_solve.solve_ta_at_current()`
directly**, not a custom `_picard_phase`-based harness -- until the root
cause of this harness discrepancy is found, comparing `_picard_phase`
output against `solve_ta_at_current` reference numbers is not a like-for-
like comparison. Comparisons BETWEEN `_picard_phase`-based runs (e.g.
different alpha values, different dt, different repeats) remain valid,
since the same harness-specific offset should apply consistently, based
on the evidence gathered here (default and small alpha landing on the
same fixed point via this harness).

Scripts: `transient/validation/loop_isolation_check.py` through `_check4.py`,
`per_layer_noclosure_test.py` (all in this session, kept for reference).

---

## 2026-08-06 (continuation, Stage B): the 9-point (dt, I) generalization sweep redone with forced-full-length convergence -- alpha=(0.03,0.01) genuinely converges everywhere tested except dt=30s, which remains a real, not premature-stopping, failure

Redo of the earlier (premature-stop-affected) Phase 1 sweep, this time
using `per_layer_diag_check.py` with `min_iters=max_iters` forced
throughout (single-threaded, same verified recipe). 7 new points run;
`dt=600s` at `I=19.6A` and `I=196A` reuse the already forced-full-length-
verified results from earlier entries in this file.

### Complete, corrected 9-point table

| point | max_iters used | SCIF (mT) | worst T_max/amp | worst T_min/amp (always a 3-turn layer) | verdict |
|---|---|---|---|---|---|
| dt=600, I=19.6 | 3000+ | +124.6 | 1.03 | -86.4 (stable) | GENUINE |
| dt=300, I=19.6 | 1000 | +115.0 | 1.00 | -46.1 | GENUINE |
| dt=150, I=19.6 | 1000 | +120.3 | 1.00 | -62.9 | GENUINE |
| dt=100, I=19.6 | 1000 | +122.5 | 1.00 | -74.2 | GENUINE |
| dt=60, I=19.6 | 750-3000 | +124.6 | 1.03 | -86.4 (stable) | GENUINE |
| **dt=30, I=19.6** | 1200 | +152.9* | **6.10 / 10.83** | -93.2 | **STILL CHAOTIC -- NOT converged** |
| dt=60, I=49 | 1500 | +301.2 | 1.02 | -31.4 | GENUINE |
| dt=60, I=98 | 1500 | +551.0 | 1.00 | -11.8 | GENUINE |
| dt=60, I=196 | 1500 | +790.0 | 1.00 | -3.9 | GENUINE |
| dt=600, I=196 | 2000-8000 | +653.9 | 1.00 | -3.3 (stable) | GENUINE |

*dt=30's SCIF is not a meaningful number -- the state is not converged
(two layers show `T_max/amp` of 6.10 and 10.83, an order of magnitude
above the boundary-condition scale, matching the exact chaotic signature
characterised throughout this investigation).

### dt=30s confirmed as a genuine failure, not premature stopping

Forcing the full 1200 iterations (vs. the 189-iteration premature stop
the original unforced sweep hit) does NOT resolve it -- `T_max/amp`
reaches 6.1-10.8x at the FINAL iteration, essentially unchanged in
character from the earlier truncated run. This settles the question
raised when this point first failed: `alpha=(0.03,0.01)` is genuinely,
not just apparently, insufficient at `dt=30s` -- consistent with the
`1/dt`-forcing-coefficient mechanism (at `dt=30s` that term is 2x larger
again than at the just-barely-sufficient `dt=60s`).

### Three clean findings

1. **Current generalises cleanly.** All three tested currents at the hard
   `dt=60s` point (49, 98, 196A) converge genuinely, and interestingly
   the minority-layer overshoot SHRINKS as current increases (-31.4 ->
   -11.8 -> -3.9x boundary scale) -- higher current is, if anything,
   easier for this alpha value, not harder.
2. **`dt` generalises down to (but not including) 30s.** Every point at
   `dt` in {600, 300, 150, 100, 60} converges genuinely with this exact
   alpha pair, with a smooth trend in both SCIF (124.6 -> 115.0 -> 120.3
   -> 122.5 -> 124.6 mT -- not monotonic in one direction, actually U-
   shaped/flat across this window, unlike the earlier dt=600->60 trend
   at premature-stop iteration counts, which was monotonically
   increasing) and in the minority-layer overshoot depth (-46 -> -63 ->
   -74 -> -86x, growing smoothly as `dt` shrinks toward the eventual
   `dt=30s` failure).
3. **The minority-layer overshoot magnitude scales inversely with both
   current and `dt`** -- worse (more negative `T_min/amp`) at lower
   current and shorter `dt`, i.e., worse in the SAME regime that was
   already known to be hardest. This is a consistent, physically
   sensible trend, not noise.

### Practical read

`alpha=(0.03,0.01)` is a genuine, working fix across a meaningfully wide
`(dt, I)` window -- all of `dt` in [60s, 600s] at the tested currents --
with a real, now well-characterised boundary at `dt=30s`. Required
iteration count for genuine settling is roughly 700-1500 across this
whole window (not wildly unpredictable), a real but bounded cost. This
is the strongest, most complete evidence yet that this fix's diagnosis
(a relaxation-parameter/forcing-coefficient problem, not irreducible
chaos) is correct and the fix itself is practically useful across a real
operating range -- just not universal down to arbitrarily short `dt`.

Scripts: `transient/validation/per_layer_diag_check.py` (unchanged).

---

## 2026-08-06 (continuation, Stage C): multi-threaded reliability redone at the true convergence horizon -- even tighter clustering than the premature-stop result

5 repeats of `per_layer_diag_check.py 60 19.6 0.03 0.01 800` (forced
full-length, `min_iters=max_iters=800` -- well past the true ~750-
iteration settling point established earlier in this file), under
NORMAL (unforced, genuinely multi-threaded) execution.

### Result: 5/5 genuine convergence, 0.004% SCIF spread

| run | SCIF (mT) | layer4(3t) min/amp | layer5(3t) min/amp |
|---|---|---|---|
| 1 | 122.424 | -77.436 | -88.249 |
| 2 | 122.429 | -77.530 | -88.247 |
| 3 | 122.425 | -77.554 | -88.246 |
| 4 | 122.428 | -77.517 | -88.247 |
| 5 | 122.428 | -77.498 | -88.248 |

SCIF spread across all 5: **0.005mT out of 122.4mT, 0.004% relative** --
tighter than the earlier premature-stop result's already-good 0.15%.
Per-layer `T_max`/`T_min` also agree closely across all 5 runs (e.g.
`layer5` min: -88.246 to -88.249, agreeing to 4 significant figures).
All 5 runs show the clean, genuine-convergence signature throughout
(`T_max/amp` near 1 for every layer, no chaotic overshoot).

### Verdict

The multi-threaded reliability finding is CONFIRMED, and strengthened,
at the correct convergence horizon: independent noisy launches of
`alpha=(0.03,0.01)` converge not just consistently with each other, but
MORE tightly than at the premature stopping point tested earlier today.
This is the strongest evidence yet that the fix produces a genuine,
reproducible, trustworthy answer under realistic (multi-threaded,
unforced) execution -- not merely an artifact of a shared premature EMA
stall.

Scripts: `transient/validation/per_layer_diag_check.py` (unchanged).

---

## 2026-08-06 (continuation, Stage D): the first genuine, fully-converged multi-step ramp test in this project's history -- all 5 steps converge cleanly

Every prior test in this project -- this session, and every session
before it referenced in `docs/HISTORY.md` -- has only ever tested a
SINGLE first step from cold start. The original motivating problem for
the entire multi-session non-determinism/convergence investigation was
genuine multi-step time-marching. This is the first test of that,
properly: `transient/validation/multistep_ramp_check.py`, fixed to force
`min_iters=max_iters_per_step` (bypassing the EMA stall check entirely,
per every lesson learned today), 5 steps, all at `dt=60s` (the
established hard regime), current stepping `I=19.6, 39.2, 58.8, 78.4,
98.0A` (5 equal +19.6A increments), `A_prev` genuinely carried forward
between steps (not reset), `alpha=(0.03, 0.01)` throughout, single-
threaded, `max_iters_per_step=1000`.

### Result: 5/5 steps genuinely converged

| step | I (A) | SCIF (mT) | T_max/amp | T_min/amp (3-turn layers) | dB_rel_last_iter |
|---|---|---|---|---|---|
| 0 | 19.6 | +124.68 | 1.040 | -86.4 | 0.069 |
| 1 | 39.2 | +240.48 | 1.000 | -41.0 | 0.038 |
| 2 | 58.8 | +349.24 | 1.000 | -23.3 | 0.023 |
| 3 | 78.4 | +448.70 | 1.000 | -15.1 | 0.017 |
| 4 | 98.0 | +539.75 | 1.000 | -10.7 | 0.011 |

**All 5 steps: `converged=True`, `finite=True`, genuine convergence
confirmed by raw diagnostics** (`T_max/amp` essentially exactly 1.000
from step 1 onward; `dB_rel_last_iter` small and monotonically
DECREASING step to step, 0.069 -> 0.011). The minority-turn-layer
`T_min/amp` overshoot shrinks progressively as current increases within
the ramp (-86.4 -> -41.0 -> -23.3 -> -15.1 -> -10.7), reproducing exactly
the same current-dependence pattern Stage B found for independent
single-step tests at increasing current -- a strong internal consistency
check, not a coincidence.

**SCIF trajectory is smooth and monotonic** (124.7 -> 240.5 -> 349.2 ->
448.7 -> 539.7 mT), no jumps, no sign changes, no instability
propagating from one step into the next -- exactly what a physically
sensible current ramp should look like.

### This is the answer to the project's original motivating question

`alpha=(0.03, 0.01)` successfully handles genuine multi-step time-
marching at `dt=60s` across this current range, with `A_prev` properly
carried forward between steps -- not just isolated first steps. This
closes the single largest gap flagged throughout this entire session
(every prior "validated" claim about this fix was for one step only).
Combined with Stage B's `(dt, I)` generalisation (works down to `dt=60s`,
fails at `dt=30s`) and Stage C's multi-threaded reliability confirmation
(0.004% spread across 5 independent noisy launches at the true
convergence horizon), this is now a substantively validated result, not
a single-point curiosity -- within the tested window.

**Still not tested**: multi-threaded execution of the full ramp (only
single-threaded here); a ramp that crosses the `dt=30s` failure boundary
mid-sequence; a ramp with the NI circuit closure active (this and
everything in this file remains insulated-limit only, per this file's
long-standing scope).

Scripts: `transient/validation/multistep_ramp_check.py`.

---

## 2026-08-06/07 (final): full production-scale ramp (0->196A) and precise dt-boundary mapping, with plots

User-directed longer run, generating a permanent visual record. Two
stages, sequential, both single-threaded/forced-full-length:
`transient/validation/full_ramp_run.py` and
`transient/validation/dt_boundary_sweep.py`, plotted by
`plot_ramp_diagnostics.py` / `plot_dt_sweep_summary.py`. All output
(both raw `.npz` data and every `.png`) in
`transient/full_validation_plots/` -- 12 plots total.

### Stage 1: full ramp, 0->196A, 10 steps, dt=60s throughout

All 10 steps converged genuinely (`T_max/amp` = 1.000 from step 1
onward, `finite=True` throughout). SCIF trajectory, smooth and
saturating as expected: 124.7, 240.5, 349.2, 448.7, 539.8, 619.4, 685.5,
725.3, 739.6, 746.2 mT -- the per-step increment shrinks monotonically
(+115.8, +108.8, +99.5, +91.1, +79.6, +66.1, +39.7, +14.3, +6.6 mT), a
physically sensible saturation curve, not noise.

Steps 0-2 reproduce Stage D's earlier 5-step test almost exactly
(124.68/240.48/349.24 vs. today's 124.68/240.48/349.24) -- an
independent process, independent mesh, same result to 3+ significant
figures.

**One genuinely new, informative finding**: the full ramp's converged
SCIF at `I=196A` (**746.2mT**) differs meaningfully from Stage B's
single cold-start jump directly to `I=196A` at the same `dt=60s`
(**790.0mT**) -- about 5.9% different. This is NOT a contradiction or a
bug: it reflects genuine path-dependence -- a current that arrives
gradually (10 warm-started steps) vs. one that arrives from a single
cold jump are, physically, different histories for a critical-state
material, and NI/hysteretic systems are expected to be history-
dependent. Worth flagging as a reminder that "the SCIF at 196A" is not
a single number independent of how the ramp got there.

Plots: `ramp_scif_trend.png`, `ramp_T_extrema_trend.png` (per-layer,
all 6), `ramp_dB_rel_trend.png`, `ramp_step_summary.png`, plus field
snapshots (`ramp_snapshot_step{0,4,9}_BJ.png` for |B|/|J| at the coil,
`ramp_snapshot_step{0,4,9}_T.png` for all 6 layers' T fields) at the
first, middle, and last steps.

### Stage 2: dt-boundary sweep, 8 points from 60s down to 30s, I=19.6A fixed

| dt (s) | n_iters | T_max/amp | T_min/amp | dB_rel | verdict |
|---|---|---|---|---|---|
| 60 | 1200 | 1.041 | -86.4 | 0.063 | clean |
| 55 | 1200 | 1.049 | -87.6 | 0.075 | clean |
| 50 | 1200 | 1.059 | -88.7 | 0.059 | clean |
| 45 | 1200 | 1.072 | -89.8 | 0.077 | clean |
| 40 | 1200 | 1.092 | -90.9 | 0.069 | clean |
| **35** | 1200 | **1.896** | -92.0 | **0.272** | **transitional** |
| 32 | 1200 | 4.255 | -92.7 | 0.765 | degrading toward chaos |
| 30 | 1200 | 10.829 | -93.2 | 1.020 | fully chaotic |

**The boundary is precisely located, for the first time** (prior testing
only confirmed the two endpoints, `dt=60s` works and `dt=30s` fails,
without intermediate resolution). `dt=40s` and above are all cleanly
converged, closely spaced (`T_max/amp` 1.04-1.09, `dB_rel` 0.06-0.08,
smoothly trending). `dt=35s` is a genuine transitional point -- not
clean (`T_max/amp` jumps to 1.9) but not fully chaotic either
(`dB_rel`=0.27, well short of the ~0.5-1+ seen at `dt`<=32s). The
degradation from `dt=35s` through `dt=30s` is smooth and monotonic in
both diagnostics, consistent with a genuine bifurcation somewhere in
the `dt`=35-40s window, not a sharp cliff at one specific value.

Plots: `dt_boundary_summary.png` (3-panel: `T_max/amp`, `dB_rel`, SCIF
vs `dt`, colour-coded green/red by genuine-vs-chaotic), and
`dt_boundary_T_trend_overlay.png` (all 8 dt values' `T_max/amp` vs
iteration on one plot, showing the transition into chaos directly as a
qualitative shape change, not just an endpoint value).

### Net effect

This closes out today's full validation effort with a permanent,
plotted record. Combined with everything else established today: the
fix works cleanly and reproducibly across the full champion design
current range (0-196A) at `dt=60s`, and the `dt` boundary is now known
precisely (clean through 40s, transitional at 35s, chaotic by 30-32s)
rather than only bracketed between two far-apart tested points.

Scripts: `transient/validation/full_ramp_run.py`,
`transient/validation/dt_boundary_sweep.py`,
`transient/validation/plot_ramp_diagnostics.py`,
`transient/validation/plot_dt_sweep_summary.py`. Output:
`transient/full_validation_plots/` (data/ subfolder for raw `.npz`,
12 `.png` plots at the top level).

---

## 2026-08-07: full 10-step ramp under genuine multi-threaded execution -- 2/2 independent runs agree to 0.001%, but both disagree with the single-threaded reference by ~2%

Closes the "multi-threaded execution of a full ramp" gap flagged above --
until now, multi-threading had only been checked at a single Picard step
(the `alpha_sweep_trace_check.py` 5-run result), never across a genuine
10-step ramp. Ran `full_ramp_run.py 0.03 0.01 1000` TWICE, back to back,
under ordinary unforced execution (no `OMP_NUM_THREADS`/single-thread env
vars -- the same script and schedule as last night's single-threaded
Stage 1 reference, 0->196A, 10 steps of +19.6A, dt=60s throughout).

### Both runs converge cleanly, and agree with EACH OTHER to ~0.001-0.03%

All 10 steps `finite=True` in both runs; `T_max/amp` settles to 1.000 by
step 4 and holds through step 9 in both (vs. step 1 in the single-threaded
reference -- slower to settle, but still clean by the point that matters).
`T_min/amp` excursion at the two 3-turn closure layers shrinks smoothly
step to step in both runs (-88.3 -> -3.95), matching the shrinking-with-
current pattern already established for the single-threaded case.

| step | I (A) | rep1 SCIF (mT) | rep2 SCIF (mT) | \|diff\| (mT) |
|---|---|---|---|---|
| 0 | 19.6 | 122.169 | 122.167 | 0.002 |
| 4 | 98.0 | 531.267 | 531.181 | 0.086 |
| 8 | 176.4 | 725.915 | 726.140 | 0.225 |
| 9 | 196.0 | 730.086 | 730.078 | 0.008 |

Max per-step spread across all 10 steps: 0.225mT (step 8, ~0.03%
relative); final-step spread 0.008mT (~0.001%). This is as tight as the
single-step multi-threaded reliability result found earlier in this file
(0.15% / 0.004%) -- the fix's reliability holds up across a genuine full
ramp, not just an isolated step.

### But both multi-threaded runs sit ~2.2% below the single-threaded reference at every comparable step

Single-threaded Stage 1 (from the entry above): 124.7, 240.5, 349.2,
448.7, 539.8, 619.4, 685.5, 725.3, 739.6, **746.2** mT.
Multi-threaded (both reps, averaged): 122.2, 236.4, 343.6, 441.5, 531.2,
609.4, 673.9, 712.2, 726.0, **730.1** mT.

Final-step gap: 16.1mT / 2.16% -- small but far larger than the 0.001-
0.03% run-to-run spread within either threading mode, so this is a real,
systematic single-vs-multi-threaded difference in this harness, not
noise. Structurally the same shape of problem already on record just
above (the unresolved ~2% gap between this `_picard_phase` harness and
the separate `solve_ta_at_current()` production path) -- root cause not
investigated here either. **Practical upshot: multi-threaded execution is
now confirmed RELIABLE (tight run-to-run agreement) for the full ramp,
but not yet confirmed ACCURATE to the same value as single-threaded
execution.** Comparisons between multi-threaded runs are valid; a
multi-threaded number should not yet be treated as interchangeable with
the single-threaded reference to better than ~2%.

Scripts: `transient/validation/full_ramp_run.py` (unmodified). Output:
`transient/full_validation_plots/data/full_ramp_0to196A_multithreaded.npz`,
`..._multithreaded_rep2.npz`.

---

## 2026-08-07 (continued): a ramp deliberately crossing the dt=30s "fully chaotic" boundary mid-sequence does NOT reproduce that chaos -- the earlier boundary characterization was cold-start/low-current specific, not a universal dt=30s failure

Closes the "ramp crossing the dt=30s boundary mid-sequence" item. Script:
`transient/validation/dt_crossing_ramp.py` (new, adapted from
`full_ramp_run.py`'s state-carrying loop). Schedule: steps 0-2 at clean
dt=60s (I=19.6, 39.2, 58.8A, baseline), steps 3-4 deliberately dropped to
dt=30s (I=78.4, 98.0A -- the same dt confirmed "fully chaotic",
dB_rel=1.020, T_max/amp=10.8, in the 2026-08-06/07 dt-boundary sweep),
steps 5-9 back to dt=60s (I=117.6 -> 196.0A). Single-threaded,
forced-full-length (1000 iters/step), same rigor as every other script
in this file.

### Important methodological note before the result: this comparison is apples-to-oranges on purpose, and that is exactly the point

The dt-boundary sweep (`dt_boundary_sweep.py`) tests each dt point from a
**cold start** (`_seed_cold`) at a **fixed low current, I=19.6A**, every
time -- it never warm-starts. This new test crosses dt=30s **warm-started**
from a converged dt=60s state, **at higher current** (I=78.4A, 4x the
sweep's fixed point). These are genuinely different physical setups, and
the question this test asks is precisely whether the boundary
characterization generalizes across that difference -- it does not.

### Result: steps 3-4 show a mild, self-correcting transitional response, not chaos -- verified on raw dB_rel, not just the printed T_max/amp

| step | dt (s) | I (A) | T_max/amp | last dB_rel | max dB_rel during step | SCIF (mT) |
|---|---|---|---|---|---|---|
| 0 | 60 | 19.6 | 1.039 | 0.069 | 1.95 | 124.680 |
| 1 | 60 | 39.2 | 1.000 | 0.038 | 0.26 | 240.478 |
| 2 | 60 | 58.8 | 1.000 | 0.023 | 0.15 | 349.248 |
| **3** | **30** | **78.4** | 1.000 | **0.162** | 1.06 | 453.203 |
| **4** | **30** | **98.0** | 1.103 | **0.012** | 1.11 | 550.885 |
| 5 | 60 | 117.6 | 1.007 | 0.010 | 0.05 | 625.673 |
| 6 | 60 | 137.2 | 1.000 | 0.007 | 0.04 | 689.731 |
| 7 | 60 | 156.8 | 1.000 | 0.008 | 0.03 | 728.516 |
| 8 | 60 | 176.4 | 1.000 | 0.006 | 0.02 | 742.269 |
| 9 | 60 | 196.0 | 1.000 | 0.006 | 0.02 | 748.560 |

(`dB_rel` pulled directly from the saved per-iteration trace, not the
printed summary -- checked deliberately, per this file's own standing
lesson that a diagnostic-looking-clean printout is not sufficient on its
own.)

Step 3 (the actual dt=30s crossing) settles to `dB_rel=0.162` -- elevated
above the clean band (0.02-0.08 at dt>=40s) and closer to the
`dt=35s` "transitional" reference point (0.272) than to clean, but
nowhere near the cold-start dt=30s chaos (1.020, never settling even at
1200 forced iterations). Step 4 (second consecutive dt=30s step) settles
fully clean (`dB_rel=0.012`). Steps 5-9, back at dt=60s, are
indistinguishable from an uncontaminated clean ramp. Final-step SCIF
(748.6mT) is within 0.3% of the original single-threaded 0-196A reference
(746.2mT) -- i.e. whatever happened at steps 3-4 left no lasting trace in
the final converged state.

### Reading

**The dt=30s "fully chaotic" finding from the boundary sweep does not
generalize to a warm-started, higher-current encounter with the same
dt.** It was characterized under one specific condition (cold start,
I=19.6A) and that characterization holds for that condition (steps 0-2's
clean dt=60s baseline reproduces the earlier Stage-1 reference to 3+ sig
figs, confirming this run's setup is faithful). But crossing dt=30s
mid-ramp -- with a good state to warm-start from and 4x the forcing
current -- produces a much milder, transitional-not-chaotic response that
self-corrects within one more step at the same dt and leaves no
measurable residue once dt returns to 60s.

**This does not mean dt=30s is "safe"** -- it means the boundary is not a
single dt value in isolation; it depends on the state the step starts
from. A production ramp that dwells at dt=30s for many consecutive steps,
or that hits dt=30s from a poor/cold state, has not been tested and
should not be assumed safe on the strength of this one crossing. What
this DOES establish: a **brief, warm-started** excursion to dt=30s inside
an otherwise-clean ramp is not automatically catastrophic, which matters
practically for any future adaptive-stepping scheme that might dip below
the nominal dt=40s floor for one or two steps.

Scripts: `transient/validation/dt_crossing_ramp.py` (new). Output:
`transient/full_validation_plots/data/dt_crossing_ramp.npz`.

---

## 2026-08-07 (continued): first validation of the NI radial-current closure (`transient/ni_circuit.py`) under the alpha=(0.03,0.01) fix -- clean at dt=60s, modest current, 3 steps

Every multi-step ramp validated in this file (Stage 1, the multi-threaded
reruns, the dt-crossing test) used the INSULATED limit only -- plain
T-A, `per_turn_bc=False`, `circuit.update()` never called. The NI radial-
current closure (`transient/ni_circuit.py`, `ta_transient.step()`) has
its own separate implementation and its own separate instability history
(a Jacobi-divergence bug fixed 2026-08-04) that entirely predates the
alpha fix -- and requires `per_turn_bc=True`, a code path the alpha fix
had never touched. This closes that gap, as a first, deliberately modest
step.

Script: `transient/validation/ni_closure_smoke_check.py` (new). Same
forced-full-length methodology as every other script here (bypasses
`ta_transient`'s own EMA-based per-step `converged` flag --
`tparams.py`'s defaults, `STEP_MIN_ITERS=6`/`STEP_STALL_MT=0.5`, are far
too permissive under this project's own established lesson). Schedule
deliberately kept at the known-clean `dt=60s` and modest current
(19.6/39.2/58.8A, matching `dt_crossing_ramp.py`'s baseline steps for
direct comparability) rather than `tparams.py`'s own default ramp/hold
schedule (`dt=600/24=25s`, `200/12=16.7s` -- both BELOW the validated dt
floor even before adding the closure). The question asked here is
narrowly "does the closure converge under the fix", not "does it also
survive the separate short-dt problem" -- that combination is untested
and should not be assumed to work.

A 10-forced-iteration mechanical smoke test caught no wiring bugs (finite
throughout, `I_z` tracking close to `I_now` as physically expected,
`I_r_mean` a few amps, no clipping) before committing to the real run.

### Result: genuinely clean convergence, checked on raw `dB_rel` not just the printed diagnostic

| step | I (A) | phase | last `dB_rel` | last10 SCIF spread (mT) | I_z range (A) | I_r_mean (A) | n_clipped |
|---|---|---|---|---|---|---|---|
| 0 | 19.6 | warmup | 0.069 | 0.007 | -- | -- | -- |
| 0 | 19.6 | closure | 0.080 | 0.014 | [14.33, 19.30] | 2.845 | 0 |
| 1 | 39.2 | warmup | 0.033 | 0.018 | -- | -- | -- |
| 1 | 39.2 | closure | 0.047 | 0.015 | [32.88, 38.86] | 3.348 | 0 |
| 2 | 58.8 | warmup | 0.024 | 0.029 | -- | -- | -- |
| 2 | 58.8 | closure | 0.033 | 0.022 | [52.28, 58.45] | 3.439 | 0 |

**Warmup-phase `dB_rel` (0.069/0.033/0.024) matches the insulated-limit
`dt_crossing_ramp.py` reference almost exactly** (0.069/0.038/0.023 at
the same currents) -- expected, since warmup runs `circuit.freeze()`
(insulated-equivalent BCs), and a strong sanity check that the new
plumbing (`per_turn_bc=True`, `ta_transient.build()`) reproduces the
already-validated physics when the closure itself is inactive.
**Closure-phase `dB_rel` (0.080/0.047/0.033) is slightly higher but
comfortably inside the established "clean" band** (0.02-0.08 at
`dt>=40s` in the dt-boundary sweep), nowhere near the `dt=35s`
transitional signature (0.27) let alone chaos. SCIF spread over the last
10 iterations is sub-0.03mT at every step/phase -- genuine settling, not
an EMA-flag false positive.

`I_r_mean` (2.8 -> 3.3 -> 3.4A as current rises 19.6->58.8A) sits close in
order of magnitude to Phase A's independently-validated circuit-model
reference (~3.4A during the 196A/600s ramp) -- not a strict apples-to-
apples check (different current and dt), but a reassuring consistency
signal, not a red flag. Zero clipping at every step -- the physical
`|I_r|<=I` band was never binding.

**One expected, physically real difference from the insulated case**:
SCIF is substantially higher with the closure active (271/414/529mT vs.
125/240/349mT insulated at the same steps) -- roughly 2x at step 0. This
is NOT a bug signature; it reflects genuine new physics the insulated
model cannot see (radial current redistribution changing the azimuthal
current distribution), which is precisely what this whole effort exists
to capture. `T_min/amp` (the closure-layer excursion depth) is nearly
unchanged from the insulated case (-76/-42/-24 vs. -86/-41/-23) --
sensible, since that excursion is dominated by the 3-turn layer geometry,
not by the radial-current path.

### Reading

**First genuine validation point for the NI circuit closure under the
alpha fix: clean.** Narrow scope, deliberately: single-threaded only, one
run (not yet repeated for reliability), `dt=60s` only, current only up to
58.8A (30% of the 196A design current), 3 steps only. **Not yet tested**:
higher current (up to the full 196A design point), more steps / a full
production ramp, multi-threaded reliability, and -- the big untested
combination -- whether the closure remains stable at short `dt` (the
separate, already-characterized problem for the insulated case). None of
these should be assumed to also be clean on the strength of this one
result.

Scripts: `transient/validation/ni_closure_smoke_check.py` (new). Output:
`transient/full_validation_plots/data/ni_closure_smoke.npz`.
