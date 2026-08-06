#!/bin/bash
# run_batch_env.sh <label> <n_reps> <dt> <I> -- like run_batch.sh but reads
# extra env assignments from the EXTRA_ENV array set by the caller (avoids
# quoting headaches with `env` + multiple KEY=VAL pairs containing braces/
# quotes, e.g. TA_MUMPS_EXTRA_OPTS='{"mat_mumps_icntl_7": 2}').
# Each rep gets its OWN isolated XDG_CACHE_HOME (unless the caller already
# set one in EXTRA_ENV) so concurrent sibling processes (e.g. another
# investigation's worktree) cannot race on the shared ~/.cache/fenics JIT
# cache and contaminate the result.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
LABEL="$1"; N="$2"; DT="$3"; I="$4"

cd "$ROOT"
for i in $(seq -w 1 "$N"); do
  LOG="$HERE/${LABEL}_${i}.log"
  CACHE="$HERE/${LABEL}_${i}_cache"
  mkdir -p "$CACHE"
  ( export XDG_CACHE_HOME="$CACHE"
    for kv in "${EXTRA_ENV[@]}"; do export "$kv"; done
    "$PY" transient/validation/first_step_diagnostic.py "$DT" "$I" > "$LOG" 2>&1 )
  RES=$(grep "^RESULT" "$LOG")
  echo "[$LABEL] rep $i: $RES"
done
