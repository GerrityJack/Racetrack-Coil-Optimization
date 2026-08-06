#!/bin/bash
# Isolating test: identical to candidate_verified_single_thread.sh (env
# vars applied correctly via direct prefix, NOT the broken bash-array
# mechanism) but WITHOUT OMP_THREAD_LIMIT=1, to check whether that specific
# variable (vs. genuine single-threading in general) is responsible for
# the 0/6 result.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
LABEL="verified_no_threadlimit"; N=6; DT=60; I=19.6

cd "$ROOT"
for i in $(seq -w 1 "$N"); do
  LOG="$HERE/${LABEL}_${i}.log"
  CACHE="$HERE/${LABEL}_${i}_cache"
  mkdir -p "$CACHE"
  XDG_CACHE_HOME="$CACHE" \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    TA_MUMPS_EXTRA_OPTS='{"mat_mumps_icntl_7": 2, "mat_mumps_icntl_28": 1}' \
    "$PY" transient/validation/first_step_diagnostic.py "$DT" "$I" > "$LOG" 2>&1
  RES=$(grep "^RESULT" "$LOG")
  echo "[$LABEL] rep $i: $RES"
done
