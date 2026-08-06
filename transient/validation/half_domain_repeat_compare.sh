#!/bin/bash
# half_domain_repeat_compare.sh -- run N independent repeats of the
# eighth-domain and half-domain first-step diagnostics at the SAME
# (dt, I) config, each a genuinely separate process launch, and summarize
# converged/n_iters. Usage: ./half_domain_repeat_compare.sh <dt> <I> <N> <coarsen> <outdir>
set -e
DT="${1:-60}"
I="${2:-19.6}"
N="${3:-5}"
COARSEN="${4:-1.0}"
OUTDIR="${5:-/tmp/half_domain_test}"
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
ROOT=/home/gerrityjack/fenicsx/Racetrack-Coil-Optimization/.claude/worktrees/agent-ad40f7c9f7131ebd1
mkdir -p "$OUTDIR"

echo "=== EIGHTH-DOMAIN (existing script), dt=$DT I=$I, $N repeats ==="
for i in $(seq 1 "$N"); do
  echo "-- eighth rep $i --"
  $PY "$ROOT/transient/validation/first_step_diagnostic.py" "$DT" "$I" \
    > "$OUTDIR/eighth_dt${DT}_I${I}_rep${i}.log" 2>&1
  grep "^RESULT" "$OUTDIR/eighth_dt${DT}_I${I}_rep${i}.log" || echo "RESULT: CRASHED/NO OUTPUT"
done

echo "=== HALF-DOMAIN, dt=$DT I=$I coarsen=$COARSEN, $N repeats ==="
for i in $(seq 1 "$N"); do
  echo "-- half rep $i --"
  $PY "$ROOT/transient/validation/half_domain_first_step_diagnostic.py" "$DT" "$I" "$COARSEN" \
    > "$OUTDIR/half_dt${DT}_I${I}_rep${i}.log" 2>&1
  grep "^RESULT" "$OUTDIR/half_dt${DT}_I${I}_rep${i}.log" || echo "RESULT: CRASHED/NO OUTPUT"
done

echo "=== SUMMARY ==="
echo "-- eighth --"
grep -h "^RESULT" "$OUTDIR"/eighth_dt${DT}_I${I}_rep*.log 2>/dev/null
echo "-- half --"
grep -h "^RESULT" "$OUTDIR"/half_dt${DT}_I${I}_rep*.log 2>/dev/null
