#!/bin/bash
# thread_audit.sh -- launch the repro case under the "forced serial" env,
# then repeatedly sample /proc/<pid>/status (Threads: line) and every
# task's comm (thread name) while it runs, to see the REAL OS thread count
# and identify which library/call is spawning threads despite
# OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 and
# mat_mumps_icntl_16=1.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
CACHE="$HERE/thread_audit_cache"
mkdir -p "$CACHE"
LOG="$HERE/thread_audit_run.log"
SAMPLES="$HERE/thread_audit_samples.txt"
: > "$SAMPLES"

cd "$ROOT"
XDG_CACHE_HOME="$CACHE" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  OMP_THREAD_LIMIT=1 \
  TA_MUMPS_EXTRA_OPTS='{"mat_mumps_icntl_16": 1, "mat_mumps_icntl_7": 2, "mat_mumps_icntl_28": 1}' \
  "$PY" transient/validation/first_step_diagnostic.py 60 19.6 > "$LOG" 2>&1 &
PID=$!
echo "launched PID=$PID" | tee -a "$SAMPLES"

# Sample thread count + thread names every 2s for up to 60s (should cover
# at least one full T-solve + A-solve inside the Picard loop).
for i in $(seq 1 30); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "process exited before sampling window ended (i=$i)" >> "$SAMPLES"
    break
  fi
  NTHREADS=$(grep ^Threads: "/proc/$PID/status" 2>/dev/null || echo "Threads: ?")
  echo "--- sample $i ($(date +%H:%M:%S)) PID=$PID $NTHREADS ---" >> "$SAMPLES"
  if [ -d "/proc/$PID/task" ]; then
    for t in /proc/$PID/task/*/; do
      tid=$(basename "$t")
      comm=$(cat "$t/comm" 2>/dev/null || echo "?")
      echo "  tid=$tid comm=$comm" >> "$SAMPLES"
    done
  fi
  sleep 2
done

wait "$PID"
echo "exit code: $?" >> "$SAMPLES"
grep "^RESULT" "$LOG" >> "$SAMPLES"
echo "=== done ==="
cat "$SAMPLES"
