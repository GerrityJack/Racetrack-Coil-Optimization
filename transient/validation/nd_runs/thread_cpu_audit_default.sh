#!/bin/bash
# Same as thread_cpu_audit.sh but with DEFAULT (unpinned) threading, for
# direct contrast -- confirms the measurement methodology actually detects
# genuine multi-threaded computation when it's present.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
CACHE="$HERE/thread_cpu_audit_default_cache"
mkdir -p "$CACHE"
LOG="$HERE/thread_cpu_audit_default_run.log"
OUT="$HERE/thread_cpu_audit_default_samples.txt"
: > "$OUT"

cd "$ROOT"
XDG_CACHE_HOME="$CACHE" \
  "$PY" transient/validation/first_step_diagnostic.py 60 19.6 > "$LOG" 2>&1 &
PID=$!
echo "launched PID=$PID (DEFAULT threading env)" | tee -a "$OUT"

sleep 8

read_ticks() {
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
total = 0
for tid in sorted(set(a) | set(b), key=int):
    da = a.get(tid, 0)
    db = b.get(tid, 0)
    d = db - da
    total += d
    out.append(f"tid={tid} delta_ticks={d}")
out.append(f"TOTAL delta_ticks={total} over 5s window ({total/5.0/100.0*100:.0f}% avg aggregate CPU)")
with open(path, "a") as f:
    f.write("\n".join(out) + "\n")
PYEOF
cat "$OUT"

kill "$PID" 2>/dev/null
wait "$PID" 2>/dev/null
echo "(process killed after measurement window)" >> "$OUT"
