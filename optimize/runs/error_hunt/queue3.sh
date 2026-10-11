#!/bin/bash
# Error-hunt queue 3: waits for queues 1-2, then the nonlinear benchmark FEM.
cd /home/gerrityjack/PPPL/Racetrack-Coil-Optimization
while pgrep -f "error_hunt/queue[12].sh" >/dev/null; do sleep 60; done
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
LOG=optimize/runs/error_hunt/queue3.log
echo "=== $(date +%F_%T) START nonlinear benchmark dt=100" >> $LOG
NB_DT=100 $PY validation/ta_nonlinear_benchmark.py >> $LOG 2>&1
echo "=== $(date +%F_%T) END rc=$?" >> $LOG
echo "=== $(date +%F_%T) QUEUE DONE" >> $LOG
