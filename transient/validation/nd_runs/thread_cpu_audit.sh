#!/bin/bash
# thread_cpu_audit.sh -- like thread_audit.sh, but measures per-THREAD CPU
# consumption (utime+stime deltas from /proc/<pid>/task/<tid>/stat) over a
# fixed interval, to distinguish "spawned but idle" worker threads from
# "genuinely computing in parallel despite OMP_NUM_THREADS=1" ones.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
CACHE="$HERE/thread_cpu_audit_cache"
mkdir -p "$CACHE"
LOG="$HERE/thread_cpu_audit_run.log"
OUT="$HERE/thread_cpu_audit_samples.txt"
: > "$OUT"

cd "$ROOT"
XDG_CACHE_HOME="$CACHE" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  OMP_THREAD_LIMIT=1 \
  TA_MUMPS_EXTRA_OPTS='{"mat_mumps_icntl_16": 1, "mat_mumps_icntl_7": 2, "mat_mumps_icntl_28": 1}' \
  "$PY" transient/validation/first_step_diagnostic.py 60 19.6 > "$LOG" 2>&1 &
PID=$!
echo "launched PID=$PID" | tee -a "$OUT"

# wait for the worker threads to appear (mesh build + first factorisation)
sleep 8

read_ticks() {
  # sum utime+stime (fields 14,15 of /proc/PID/task/TID/stat) per thread
  for t in /proc/$PID/task/*/; do
    tid=$(basename "$t")
    if [ -r "$t/stat" ]; then
      awk -v tid="$tid" '{print tid, $14+$15}' "$t/stat" 2>/dev/null
    fi
  done
}

echo "--- snapshot A ($(date +%H:%M:%S.%N)) ---" >> "$OUT"
A=$(read_ticks)
echo "$A" >> "$OUT"

sleep 5

echo "--- snapshot B ($(date +%H:%M:%S.%N)) ---" >> "$OUT"
B=$(read_ticks)
echo "$B" >> "$OUT"

echo "--- per-thread ticks consumed in the 5s window (CLK_TCK=$(getconf CLK_TCK)) ---" >> "$OUT"
python3 - "$OUT" <<'PYEOF'
import sys
path = sys.argv[1]
text = open(path).read()
blocks = text.split("--- snapshot")
a = {}
b = {}
for line in blocks[1].splitlines()[1:]:
    parts = line.split()
    if len(parts) == 2:
        a[parts[0]] = int(parts[1])
for line in blocks[2].splitlines()[1:]:
    parts = line.split()
    if len(parts) == 2:
        b[parts[0]] = int(parts[1])
out = []
for tid in sorted(set(a) | set(b), key=int):
    da = a.get(tid, 0)
    db = b.get(tid, 0)
    out.append(f"tid={tid} delta_ticks={db-da}")
with open(path, "a") as f:
    f.write("\n".join(out) + "\n")
PYEOF
cat "$OUT"

kill "$PID" 2>/dev/null
wait "$PID" 2>/dev/null
echo "(process killed after measurement window)" >> "$OUT"
