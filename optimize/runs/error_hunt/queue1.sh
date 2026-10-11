#!/bin/bash
# Overnight error-hunt compute queue (2026-10-04/05), runs sequentially.
cd /home/gerrityjack/PPPL/Racetrack-Coil-Optimization
PY=/home/gerrityjack/miniconda3/envs/fenicsx-env/bin/python3
LOG=optimize/runs/error_hunt/queue1.log
export RH_DESIGN_JSON='{"a":0.026,"b":0.0314,"coil_half_gap":0.0137,"n_turns":[382,382,478,478,3,3]}'
LADDER="180:14400:1.0:1:0.10:150,180:14400:0.9:0:0.10:100,180:14400:0.8:0:0.10:120,180:14400:0.7:0:0.05:150,180:14400:0.6:0:0.01:300"
run() { echo "=== $(date +%F_%T) START $1" >> $LOG; shift; env "$@" $PY optimize/studies/ramp_hold_test.py >> $LOG 2>&1; echo "=== $(date +%F_%T) END rc=$?" >> $LOG; }
# 1. reproducibility: identical config to floortest_legacy6L_180A, new process (new mesh)
run repro RH_TAG=eh_repro_legacy180 RH_SEQUENCE="$LADDER"
# 2. sqrt(t) constant-power march to 180 A over 2 h (4 x 1800 s), floor 0.7,
#    vs the 4 h single-step surrogate (floortest 180 A floor 0.7: worst 0.881)
run march RH_TAG=eh_march_legacy180 RH_SEQUENCE="90:1800:1.0:1:0.05:200,90:1800:0.9:0:0.05:150,90:1800:0.8:0:0.02:200,90:1800:0.7:0:0.005:600,127.28:1800:0.7:1:0.005:600,155.88:1800:0.7:1:0.005:600,180:1800:0.7:1:0.005:600"
# 3. denser across-width z-grading (xdense7), same ladder
run xdense7 RH_TAG=eh_zgrid_xdense7_legacy180 RH_ZGRID=xdense7 RH_SEQUENCE="$LADDER"
# 4. fine in-plane mesh tier, same ladder
run fine RH_TAG=eh_mesh_fine_legacy180 RH_MESH=fine RH_SEQUENCE="$LADDER"
echo "=== $(date +%F_%T) QUEUE DONE" >> $LOG
