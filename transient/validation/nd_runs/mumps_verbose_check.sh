#!/bin/bash
# One-off: run the repro case with MUMPS ICNTL(4) verbosity cranked up so
# MUMPS itself prints which analysis mode (sequential vs parallel) and
# which ordering method it actually chose at runtime, rather than assuming
# from linked libraries. Single run, not a statistics batch.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
cd "$ROOT"
XDG_CACHE_HOME="$HERE/mumps_verbose_cache" \
  TA_MUMPS_EXTRA_OPTS='{"mat_mumps_icntl_4": 3}' \
  "$PY" transient/validation/first_step_diagnostic.py 60 19.6 > "$HERE/mumps_verbose.log" 2>&1
grep -iE "analysis|ordering|ICNTL\(7\)|ICNTL\(28\)|sequential|parallel|AMD|AMF|PORD|METIS|SCOTCH|QAMD" "$HERE/mumps_verbose.log" | head -40
