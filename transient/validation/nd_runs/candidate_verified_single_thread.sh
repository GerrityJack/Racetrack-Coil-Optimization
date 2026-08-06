#!/bin/bash
# Re-run of the "forced serial" config, now that thread_cpu_audit.sh /
# thread_cpu_audit_meshphase.sh have DIRECTLY VERIFIED (via /proc tick
# deltas, not env-var trust) that this exact env produces genuinely
# single-threaded execution throughout both the mesh-build and Picard-solve
# phases. Adds more reps to the existing forced_serial_mumps batch (3/6)
# under the same configuration, explicitly labelled to distinguish
# "verified single-threaded" data from the earlier, less rigorously
# confirmed batch.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
LABEL="verified_single_thread"; N=6; DT=60; I=19.6

cd "$ROOT"
for i in $(seq -w 1 "$N"); do
  LOG="$HERE/${LABEL}_${i}.log"
  CACHE="$HERE/${LABEL}_${i}_cache"
  mkdir -p "$CACHE"
  XDG_CACHE_HOME="$CACHE" \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    OMP_THREAD_LIMIT=1 \
    TA_MUMPS_EXTRA_OPTS='{"mat_mumps_icntl_16": 1, "mat_mumps_icntl_7": 2, "mat_mumps_icntl_28": 1}' \
    "$PY" transient/validation/first_step_diagnostic.py "$DT" "$I" > "$LOG" 2>&1
  RES=$(grep "^RESULT" "$LOG")
  echo "[$LABEL] rep $i: $RES"
done
