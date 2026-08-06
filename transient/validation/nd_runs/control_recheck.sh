#!/bin/bash
# Control re-check: DEFAULT threading, NO dump hook, run NOW (system load
# cleared, sibling investigation finished) to see whether the ~0% rate
# seen in the just-completed dump batch was session drift / a dump-hook
# side effect, or genuinely representative of current conditions.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
N=6

cd "$ROOT"
for i in $(seq -w 1 "$N"); do
  LOG="$HERE/control_recheck_${i}.log"
  CACHE="$HERE/control_recheck_${i}_cache"
  mkdir -p "$CACHE"
  XDG_CACHE_HOME="$CACHE" \
    "$PY" transient/validation/first_step_diagnostic.py 60 19.6 > "$LOG" 2>&1
  RES=$(grep "^RESULT" "$LOG")
  echo "[control_recheck] rep $i: $RES"
done
