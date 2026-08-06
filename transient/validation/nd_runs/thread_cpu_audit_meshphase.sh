#!/bin/bash
# Same tick-delta methodology, but sampled during the MESH-BUILD phase
# (first ~4s of the process, before the Picard loop starts) under the
# forced-serial env, to check whether gmsh itself ignores
# OMP_NUM_THREADS/OPENBLAS_NUM_THREADS and threads independently.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
CACHE="$HERE/thread_cpu_audit_mesh_cache"
mkdir -p "$CACHE"
LOG="$HERE/thread_cpu_audit_mesh_run.log"
OUT="$HERE/thread_cpu_audit_mesh_samples.txt"
: > "$OUT"

cd "$ROOT"
XDG_CACHE_HOME="$CACHE" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  OMP_THREAD_LIMIT=1 \
  "$PY" transient/validation/first_step_diagnostic.py 60 19.6 > "$LOG" 2>&1 &
PID=$!
echo "launched PID=$PID" | tee -a "$OUT"

read_ticks() {
  for t in /proc/$PID/task/*/; do
    tid=$(basename "$t")
    if [ -r "$t/stat" ]; then
      awk -v tid="$tid" '{print tid, $14+$15}' "$t/stat" 2>/dev/null
    fi
  done
}

sleep 0.5
echo "--- snapshot A (t=0.5s, during mesh build) ---" >> "$OUT"
A=$(read_ticks)
echo "$A" >> "$OUT"

sleep 2.5
echo "--- snapshot B (t=3.0s, likely still mesh build) ---" >> "$OUT"
B=$(read_ticks)
echo "$B" >> "$OUT"

python3 - "$OUT" <<'PYEOF'
import sys
path = sys.argv[1]
text = open(path).read()
blocks = text.split("--- snapshot")
a, b = {}, {}
for line in blocks[1].splitlines()[1:]:
    p = line.split()
    if len(p) == 2: a[p[0]] = int(p[1])
for line in blocks[2].splitlines()[1:]:
    p = line.split()
    if len(p) == 2: b[p[0]] = int(p[1])
total = 0
out = []
for tid in sorted(set(a) | set(b), key=int):
    d = b.get(tid, 0) - a.get(tid, 0)
    total += d
    out.append(f"tid={tid} delta_ticks={d}")
out.append(f"TOTAL delta_ticks={total} over 2.5s window ({total/2.5:.0f}% avg aggregate CPU)")
with open(path, "a") as f:
    f.write("\n".join(out) + "\n")
PYEOF
cat "$OUT"
echo "--- log so far (to confirm what phase we caught) ---"
cat "$LOG"

kill "$PID" 2>/dev/null
wait "$PID" 2>/dev/null
