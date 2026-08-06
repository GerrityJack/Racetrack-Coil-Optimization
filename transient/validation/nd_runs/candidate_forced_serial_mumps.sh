#!/bin/bash
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXTRA_ENV=("OMP_NUM_THREADS=1" "OPENBLAS_NUM_THREADS=1" "MKL_NUM_THREADS=1" "NUMEXPR_NUM_THREADS=1" 'TA_MUMPS_EXTRA_OPTS={"mat_mumps_icntl_16": 1, "mat_mumps_icntl_7": 2, "mat_mumps_icntl_28": 1}')
"$HERE/run_batch_env.sh" forced_serial_mumps 10 60 19.6
