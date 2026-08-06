#!/bin/bash
# Run the DEFAULT-threaded repro case repeatedly with the iteration-1
# matrix/RHS dump enabled, until we have at least one success and one
# failure dumped, for a bit-level diff.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
N="${1:-10}"

cd "$ROOT"
for i in $(seq -w 1 "$N"); do
  LOG="$HERE/dump_run_${i}.log"
  CACHE="$HERE/dump_run_${i}_cache"
  mkdir -p "$CACHE"
  DUMPDIR="$HERE/iter1_dump_${i}"
  mkdir -p "$DUMPDIR"
  XDG_CACHE_HOME="$CACHE" \
    DUMP_ITER1_MATRIX_PATH="$DUMPDIR/dump" \
    "$PY" transient/validation/first_step_diagnostic.py 60 19.6 > "$LOG" 2>&1
  RES=$(grep "^RESULT" "$LOG")
  echo "[dump] rep $i: $RES"
done
