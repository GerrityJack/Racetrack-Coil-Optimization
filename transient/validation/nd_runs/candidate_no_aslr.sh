#!/bin/bash
# ASLR-disabled repro batch. setarch -R disables address-space layout
# randomization for the child process (and its children).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
LABEL="no_aslr"; N=10; DT=60; I=19.6

cd "$ROOT"
for i in $(seq -w 1 "$N"); do
  LOG="$HERE/${LABEL}_${i}.log"
  CACHE="$HERE/${LABEL}_${i}_cache"
  mkdir -p "$CACHE"
  XDG_CACHE_HOME="$CACHE" setarch "$(uname -m)" -R "$PY" \
    transient/validation/first_step_diagnostic.py "$DT" "$I" > "$LOG" 2>&1
  RES=$(grep "^RESULT" "$LOG")
  echo "[$LABEL] rep $i: $RES"
done
