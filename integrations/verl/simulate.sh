#!/bin/bash
# Run the three node entry scripts end to end on this machine, without a GPU, a model or a bucket, and print what
# SIMULATION.md shows. A stand-in server replays control policies of the environments (mrgym_verl.mixstub), a
# stand-in trainer does the policy update (mrgym_verl.sim), and a directory stands in for the bucket.
#
#   bash integrations/verl/simulate.sh <empty work folder> <python with numpy and scipy> [part ...]
#
# Parts (default: all): calibrate | smoke | train | noreward. At most two worker processes at a time, about 15 minutes.
# Nothing printed here is a result about any model.
set -uo pipefail
WORK=${1:?work folder}; PY=${2:?python}; shift 2
PARTS=${*:-calibrate smoke train noreward}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
NODE=$REPO/integrations/verl/node
mkdir -p "$WORK/out" "$WORK/node"
COMMON=(MRGYM_STUB=1 STUB_SKIP_VENV=1 "STUB_PYTHON=$PY" "OUT_URI=$WORK/out" "REPO_SRC=$REPO" "WORK_ROOT=$WORK/node" HOLD_S=0 HEARTBEAT_S=0
        SYNC_EVERY=2 SUMMARY_EVERY=0 WATCH_EVERY=2 OMP_NUM_THREADS=1 MODEL_HF=stand-in/none)
CELLS="Scanner/NoWrap-v1:2,3"
run() {  # run <log> <script> VAR=value ...: the command is printed, the script's output goes to the log
  local logf=$1 script=$2; shift 2
  echo "\$ env ${*} bash integrations/verl/node/$script"
  env -i PATH="$PATH" HOME="$HOME" "${COMMON[@]}" "$@" bash "$NODE/$script" > "$logf" 2>&1
  echo "exit code $?"
}
show() { grep -E "$2" "$1" | cut -c1-230; }
fresh_node() { echo "\$ rm -rf <work disk of the node>     # a fresh node"; rm -rf "$WORK/node"; mkdir -p "$WORK/node"; }
count() { find "$1" -name record.json 2>/dev/null | wc -l; }

for part in $PARTS; do case "$part" in
calibrate)
  echo "## calibrate.sh: killed in the middle of the plan, started again on a fresh node"
  CAL=(RUN_ID=c01 MODEL_NAME=mix:0.45:0.10 "CAL_PLAN=$CELLS Scanner/SNRSpec-v0:1" CAL_N=8 CAL_K=4 CAL_WORKERS=1 CAL_PROCS=2 STUB_SERVERS=1)
  run "$WORK/cal_1.log" calibrate.sh "${CAL[@]}" SIM_DIE_AT=calibrate SIM_DIE_AFTER_S=25
  show "$WORK/cal_1.log" "self-test ok:|records already present|servers ready|SIM_DIE_AT"
  echo "episode records in the output folder after the kill: $(count "$WORK/out/runs/c01/records")  (DONE.txt: $([ -f "$WORK/out/runs/c01/DONE.txt" ] && echo yes || echo no))"
  fresh_node
  run "$WORK/cal_2.log" calibrate.sh "${CAL[@]}"
  show "$WORK/cal_2.log" "records already present|pass [0-9] of the plan|done:|FAILED"
  grep -h "episode(s) to run" "$WORK"/node/job_c01/out/cal_p*.log | sort | uniq -c | cut -c1-200
  echo "--- $WORK/out/runs/c01/calibration_summary.md"; cat "$WORK/out/runs/c01/calibration_summary.md"
  echo "--- train_arms.json"; "$PY" -c "import json; d = json.load(open('$WORK/out/runs/c01/train_arms.json')); print({k: d[k] for k in ('train_arms', 'base_rate', 'complete')})"
  run "$WORK/cal_3.log" calibrate.sh "${CAL[@]}"
  show "$WORK/cal_3.log" "Nothing to do"
  ;;
smoke)
  echo "## train_smoke.sh"
  run "$WORK/smoke.log" train_smoke.sh RUN_ID=t01 "TRAIN_ARMS=$CELLS" SMOKE_STEPS=2 PROMPTS_PER_STEP=2 GROUP_SIZE=4
  show "$WORK/smoke.log" "=== SMOKE|^OK |^FAIL |restor|SMOKE PASS|SMOKE FAIL|timing_s"
  show "$WORK/node/job_t01/out/run_b.log" "restoring|Resuming|^step:"
  echo "--- smoke_report.json (head)"; "$PY" -c "import json; d = json.load(open('$WORK/out/runs/t01/smoke_report.json')); print({k: d[k] for k in ('episodes', 'records', 'attempts', 'evaluation_after_restore', 'outcomes')})"
  echo "--- a failing smoke names its cause: the same script when every first reply is cut by the token limit (SIM_FORCE_CUT=1)"
  run "$WORK/smoke_fail.log" train_smoke.sh RUN_ID=t02 "TRAIN_ARMS=$CELLS" SMOKE_STEPS=1 PROMPTS_PER_STEP=1 GROUP_SIZE=4 SIM_FORCE_CUT=1
  show "$WORK/smoke_fail.log" "SMOKE FAIL|FAILED" | head -2
  ;;
train)
  echo "## train.sh: killed after the rollouts of step 3, started again on a fresh node, read out on the secret split"
  [ -f "$WORK/secret.key" ] || PYTHONPATH="$REPO/src" "$PY" -m mrgym.cli epoch keygen --out "$WORK/secret.key" > /dev/null
  TR=(RUN_ID=u01 "TRAIN_ARMS=$CELLS" BASE_RATE=0.45 "TEST_ARMS=$CELLS Scanner/SNRSpec-v0:1" UNTRAINED=Scanner/SNRSpec-v0 TEST_SEEDS=0-5 "SECRET_URI=$WORK/secret.key"
      TOTAL_STEPS=6 SAVE_FREQ=2 TEST_FREQ=2 PROMPTS_PER_STEP=4 GROUP_SIZE=4 DEV_SEEDS=0-7 VAL_N=2 SIM_LR=3.0 SIM_IDLE_SHARE=0.05
      NO_REWARD_STEPS=2 HARNESS_STEPS=3 MIXED_MIN=0 WINDOW=2 COLLAPSE_STEPS=3 MIN_ENVS=1)
  run "$WORK/train_1.log" train.sh "${TR[@]}" SIM_DIE_AT_STEP=3
  show "$WORK/train_1.log" "held-out split opens|mirror:|=== TRAIN|SIM_DIE_AT"
  echo "in the output folder after the kill: LATEST = $(cat "$WORK/out/runs/u01/remote/ckpt/LATEST"); $(grep -c . "$WORK/out/runs/u01/remote/records/episodes.jsonl") episode line(s)"
  fresh_node
  run "$WORK/train_2.log" train.sh "${TR[@]}"
  show "$WORK/train_2.log" "mirror:|=== TRAIN|training reached|already evaluated|done:|FAILED"
  show "$WORK/node/job_u01/out/train.log" "restoring|Resuming|^step:"
  echo "--- $WORK/out/runs/u01/remote/metrics.txt"; cat "$WORK/out/runs/u01/remote/metrics.txt"
  echo "--- $WORK/out/runs/u01/readout.txt"; cat "$WORK/out/runs/u01/readout.txt"
  echo "--- the secret is in no uploaded file: $(grep -rl "$(cat "$WORK/secret.key")" "$WORK/out" | wc -l) file(s) contain it"
  ;;
noreward)
  echo "## train.sh on a policy that never passes: the no-reward rule stops it"
  run "$WORK/train_nr.log" train.sh RUN_ID=u02 "TRAIN_ARMS=$CELLS" TOTAL_STEPS=8 SAVE_FREQ=2 TEST_FREQ=4 PROMPTS_PER_STEP=2 GROUP_SIZE=4 DEV_SEEDS=0-1 VAL_N=1 \
      SIM_THETA0=-20 NO_REWARD_STEPS=3
  show "$WORK/train_nr.log" "stopped by a rule|done:|FAILED"
  echo "--- STOP.json"; cat "$WORK/out/runs/u02/STOP.json"; echo
  ;;
esac; echo; done
echo "processes left behind: $(ps -eo cmd | grep -cE "[s]tore\.py loop|mrgym_verl\.(mixstub|sim|stoprules)|mrgym\.cli calibrate")"
