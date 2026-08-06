#!/bin/bash
# Candidate 3 (JIT cache): does giving each rep a COMPLETELY FRESH,
# never-before-used cache directory (forcing full recompilation every
# single time, no reuse at all) change the success rate vs. baseline
# (which uses the single shared, long-lived ~/.cache/fenics)?
# If success rate here matches baseline: JIT cache staleness/reuse is not
# the driver. If it changes markedly: worth digging further into cache
# corruption/partial-write races specifically.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
LABEL="fresh_cache"; N=8; DT=60; I=19.6

cd "$ROOT"
for i in $(seq -w 1 "$N"); do
  LOG="$HERE/${LABEL}_${i}.log"
  CACHE="$HERE/${LABEL}_${i}_cache_$$_${i}"
  rm -rf "$CACHE"
  mkdir -p "$CACHE"
  XDG_CACHE_HOME="$CACHE" "$PY" transient/validation/first_step_diagnostic.py "$DT" "$I" > "$LOG" 2>&1
  RES=$(grep "^RESULT" "$LOG")
  echo "[$LABEL] rep $i: $RES"
  rm -rf "$CACHE"
done
