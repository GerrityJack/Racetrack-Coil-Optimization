#!/bin/bash
# Error-hunt queue 2: waits for queue1, then air-box-size impact on loads.
cd /home/gerrityjack/PPPL/Racetrack-Coil-Optimization
while pgrep -f "error_hunt/queue1.sh" >/dev/null; do sleep 60; done
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
LOG=optimize/runs/error_hunt/queue2.log
export RH_DESIGN_JSON='{"a":0.026,"b":0.0314,"coil_half_gap":0.0137,"n_turns":[382,382,478,478,3,3]}'
LADDER="180:14400:1.0:1:0.10:150,180:14400:0.9:0:0.10:100,180:14400:0.8:0:0.10:120,180:14400:0.7:0:0.05:150,180:14400:0.6:0:0.01:300"
run() { echo "=== $(date +%F_%T) START $1" >> $LOG; shift; env "$@" $PY optimize/studies/ramp_hold_test.py >> $LOG 2>&1; echo "=== $(date +%F_%T) END rc=$?" >> $LOG; }
run box8 RH_TAG=eh_box8_legacy180 RH_BOX_SCALE=8 RH_SEQUENCE="$LADDER"
run box16 RH_TAG=eh_box16_legacy180 RH_BOX_SCALE=16 RH_SEQUENCE="$LADDER"
echo "=== $(date +%F_%T) QUEUE DONE" >> $LOG
