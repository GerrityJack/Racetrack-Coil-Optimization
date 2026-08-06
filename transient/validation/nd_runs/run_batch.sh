#!/bin/bash
# run_batch.sh <label> <n_reps> <dt> <I> [extra env assignments as KEY=VAL ...]
# Runs first_step_diagnostic.py n_reps times as separate process launches,
# each with any extra env vars applied, logging to nd_runs/<label>_NN.log
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
LABEL="$1"; N="$2"; DT="$3"; I="$4"
shift 4
ENVARGS=("$@")

cd "$ROOT"
for i in $(seq -w 1 "$N"); do
  LOG="$HERE/${LABEL}_${i}.log"
  env "${ENVARGS[@]}" "$PY" transient/validation/first_step_diagnostic.py "$DT" "$I" > "$LOG" 2>&1
  RES=$(grep "^RESULT" "$LOG")
  echo "[$LABEL] rep $i: $RES"
done
