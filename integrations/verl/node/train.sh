#!/bin/bash
# Batch evaluation worker for one 8-GPU node, preemptible: the training run. LoRA + GRPO in verl (pinned commit) on
# in-process episodes, a curriculum over the given (environment, level) cells, periodic evaluation on public dev
# instances, stop rules, and at the end one readout on secret held-out instances: trained minus base, paired.
#
# Resumable. Checkpoints, episode records and the stop-rule state are mirrored to OUT_URI/runs/RUN_ID/remote every
# SYNC_EVERY seconds. A preempted job that starts again repeats the setup, pulls the mirror, restores the newest
# checkpoint and continues; a finished readout part is not repeated. A run id that finished, failed or was stopped
# by a rule is never started again.
#
# Stop rules (integrations/verl/mrgym_verl/stoprules.py; the values of a run are fixed in its pre-registration). A
# stop is a result: the job writes STOP.json with the rule and its numbers, keeps the last checkpoint, ends with exit 0.
#   no reward in the first NO_REWARD_STEPS steps; training harness unlike the calibration (first HARNESS_STEPS steps);
#   episodes not delivering; replies cut; collapse; futility by a sequential test on the dev instances; mastered.
#
# ============================== parameters: environment variables (prepend `export NAME=value` lines, or edit here) ==
#   RUN_ID, OUT_URI, REPO_SRC      as in calibrate.sh                                       (required)
#   MODEL_HF        Hugging Face id of the model to train                                   (required)
#   TRAIN_ARMS      training cells "ID:levels ID:levels" ...                                (one of the two is required)
#   ARMS_URI        ... or the train_arms.json a calibration run wrote (sets TRAIN_ARMS and BASE_RATE)
#   DEV_ARMS / DEV_SEEDS / VAL_N   in-training evaluation: cells (default TRAIN_ARMS), public dev seeds, episodes each   - / 0-31 / 2
#   TEST_ARMS / TEST_SEEDS         final readout on the secret split test-iid; may name cells never trained on            - / 0-149
#   UNTRAINED       environment ids of TEST_ARMS that are never trained on (the transfer condition of the verdict)
#   SECRET_URI      where the secret key file is (s3://... or a path); fetched to a file of mode 0600 on the node and
#                   handed to the package as MRGYM_SECRET_FILE. Required with TEST_ARMS. Never put the key in a variable.
#   TOTAL_STEPS / SAVE_FREQ / TEST_FREQ / PROMPTS_PER_STEP / GROUP_SIZE                     100 / 5 / 10 / 16 / 8
#   SEED            seed of the curriculum sampler                                          0
#   CURRICULUM_MODE bandit | ladder | uniform                                               bandit
#   SIGNAL          pass (the environment's paid reward) | progress (not the paid reward)   pass
#   BASE_RATE       calibrated pass rate of the training cells (rule F1); from ARMS_URI when given
#   stop rules:     NO_REWARD_STEPS 4, NO_REWARD_MIN 0.02, HARNESS_STEPS 10, MIXED_MIN 0.40, SUBMIT_MIN 0.50, CUT_MAX 0.20,
#                   WINDOW 5, COLLAPSE_STEPS 10, ALPHA 0.05, LOOKS (TOTAL_STEPS / TEST_FREQ), MDE 0.05,
#                   FUTILITY_FROM_STEP (TOTAL_STEPS / 2), MASTERED 0.90, WATCH_EVERY 120 (seconds between checks)
#   readout rule:   GAIN 0.15, LOWER 0.05, HARM 0.05, MIN_ENVS 2
#   every variable of integrations/verl/overrides.sh (ROLLOUT_TP, GPU_MEM_UTIL, MAX_PROMPT_LEN, MAX_RESPONSE_LEN,
#   MAX_TOKENS_PER_TURN, TOKENS_PER_GPU, LORA_RANK, LR, CHAT_KWARGS, TOOL_FORMAT, ...) is passed through when set
#   NGPUS 8, SYNC_EVERY 45, HOLD_S 600, WORK_ROOT, VERL_REF (pinned commit)
#   MRGYM_STUB=1    local simulation (see train_smoke.sh). SIM_DIE_AT_STEP=N: the stand-in trainer is killed after the
#                   rollouts of step N and the job dies with it, without a final upload (a preemption).
# =====================================================================================================================
set -uo pipefail
# ---------------------------------------------------------------- bootstrap (the same lines in the three entry scripts)
STUB=${MRGYM_STUB:-0}; HOLD_S=${HOLD_S:-600}; BOOT_PY=${BOOT_PY:-python3}
[ -n "${RUN_ID:-}" ] && [ -n "${OUT_URI:-}" ] && [ -n "${REPO_SRC:-}" ] || { echo "[boot] FATAL: RUN_ID, OUT_URI and REPO_SRC are required"; exit 2; }
echo "[boot] $(date -u +%FT%TZ) host=$(hostname) user=$(id -un 2>/dev/null) run=$RUN_ID stub=$STUB"
pick_root() {   # the writable local disk with the most free space (no network mount, no memory-backed file system)
  df -P -l -x tmpfs -x devtmpfs -x overlay 2>/dev/null | awk 'NR>1 && $4 > 300000000 {print $4, $6}' | sort -rn | while read -r _ m; do [ -w "$m" ] && { echo "$m"; break; }; done
}
ROOT=${WORK_ROOT:-$(pick_root)}; [ -n "$ROOT" ] || ROOT=${HOME:-/root}
W=$ROOT/job_$RUN_ID
mkdir -p "$W/out" "$W/src" "$W/models" || { echo "[boot] FATAL: cannot create $W"; sleep "$HOLD_S"; exit 1; }
cd "$W" || exit 1
if [ "$STUB" != 1 ]; then    # boto3 for the uploads and uv for the environments, into the image's interpreter
  "$BOOT_PY" -c "import boto3" 2>/dev/null && "$BOOT_PY" -m uv --version >/dev/null 2>&1 || "$BOOT_PY" -m pip install -q boto3 uv 2>&1 | tail -1 \
    || "$BOOT_PY" -m pip install -q --break-system-packages boto3 uv 2>&1 | tail -1
  "$BOOT_PY" -c "import boto3" || { echo "[boot] FATAL: boto3 unavailable"; sleep "$HOLD_S"; exit 1; }
fi
s3get() { "$BOOT_PY" -c 'import boto3, sys; b, _, k = sys.argv[1][5:].partition("/"); boto3.client("s3").download_file(b, k, sys.argv[2])' "$1" "$2"; }
if [ -d "$REPO_SRC" ]; then REPO=$(cd "$REPO_SRC" && pwd)          # a checkout on this machine: used in place
else
  REPO=$W/src/repo
  if [ ! -f "$REPO/integrations/verl/node/lib.sh" ]; then
    rm -rf "$REPO"; mkdir -p "$REPO"
    case "$REPO_SRC" in
      s3://*)  s3get "$REPO_SRC" "$W/src/repo.tar.gz" && tar --no-same-owner -xzf "$W/src/repo.tar.gz" -C "$REPO" --strip-components="${REPO_STRIP:-0}" ;;
      git+*)   u=${REPO_SRC#git+}; git clone -q "${u%@*}" "$REPO" && git -C "$REPO" checkout -q "${u##*@}" ;;
      *)       tar --no-same-owner -xzf "$REPO_SRC" -C "$REPO" --strip-components="${REPO_STRIP:-0}" ;;
    esac || { echo "[boot] FATAL: cannot get the repository from $REPO_SRC"; sleep "$HOLD_S"; exit 1; }
  fi
fi
[ -f "$REPO/integrations/verl/node/lib.sh" ] || { echo "[boot] FATAL: no integrations/verl/node/lib.sh under $REPO"; sleep "$HOLD_S"; exit 1; }
# shellcheck source=/dev/null
source "$REPO/integrations/verl/node/lib.sh"
# ---------------------------------------------------------------- end of bootstrap
: "${MODEL_HF:?}"
[ -n "${TRAIN_ARMS:-}${ARMS_URI:-}" ] || { echo "[boot] FATAL: TRAIN_ARMS or ARMS_URI is required"; exit 2; }
export TOTAL_STEPS=${TOTAL_STEPS:-100} SAVE_FREQ=${SAVE_FREQ:-5} TEST_FREQ=${TEST_FREQ:-10} PROMPTS_PER_STEP=${PROMPTS_PER_STEP:-16} GROUP_SIZE=${GROUP_SIZE:-8}
export VAL_N=${VAL_N:-2} DEV_SEEDS=${DEV_SEEDS:-0-31} TEST_SEEDS=${TEST_SEEDS:-0-149} SEED=${SEED:-0}
stage() { echo; log "=== TRAIN $1 ==="; mark "stage $1"; }

start_run
stage "0 node"; need_gpus
stage "1 install"; env_venv; verl_env; get_model "$MODEL_HF"
if [ -z "${TRAIN_ARMS:-}" ]; then
  st get "$ARMS_URI" "$W/out/train_arms.json" || fail "cannot read $ARMS_URI"
  TRAIN_ARMS=$("$BOOT_PY" -c "import json; print(json.load(open('$W/out/train_arms.json'))['train_arms'])")
  BASE_RATE=${BASE_RATE:-$("$BOOT_PY" -c "import json; print(json.load(open('$W/out/train_arms.json'))['base_rate'] or '')")}
  [ -n "$TRAIN_ARMS" ] || fail "the calibration named no cell in the rule: nothing to train on"
fi
export TRAIN_ARMS DEV_ARMS=${DEV_ARMS:-$TRAIN_ARMS} TEST_ARMS=${TEST_ARMS:-}
log "training cells: $TRAIN_ARMS | dev: $DEV_ARMS | test: ${TEST_ARMS:-none} | base rate: ${BASE_RATE:-not given}"

if [ -n "$TEST_ARMS" ]; then   # the secret of the held-out split: a file on this node, never a command line, never a log
  : "${SECRET_URI:?SECRET_URI is required with TEST_ARMS}"
  mkdir -p "$W/secret" && chmod 700 "$W/secret"
  if [ -f "$SECRET_URI" ]; then cp "$SECRET_URI" "$W/secret/key"; else st get "$SECRET_URI" "$W/secret/key" >/dev/null 2>&1 || fail "cannot read the secret (SECRET_URI)"; fi
  chmod 600 "$W/secret/key"; export MRGYM_SECRET_FILE=$W/secret/key
  "$PYBIN" - "$TEST_ARMS" <<'PY' || fail "the secret does not open the held-out split"
import sys, mrgym
from mrgym_verl import instances
env_id, level = instances.parse_arms(sys.argv[1])[0]
env = mrgym.make(env_id, level=level, seed=0, split="test-iid"); env.reset(); env.close()
print("held-out split opens:", env_id, "level", level)
PY
fi

export RUN_NAME=${RUN_NAME:-$RUN_ID} MODEL_PATH=$MODEL_DIR MRGYM_MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL_HF")}
export DATA_DIR=$R/run/data CKPT_DIR=$R/run/ckpt RECORD_DIR=$R/remote/records MRGYM_CKPT_DEST=$R/remote/ckpt
mkdir -p "$CKPT_DIR"
stage "2 the adapter on CPU with the model's tokenizer"
cpu_gate "$(echo "$TRAIN_ARMS" | awk '{print $1}')"

mirror_start "$R/remote" remote
LOG=$RECORD_DIR/episodes.jsonl
log "mirror: $(grep -c . "$LOG" 2>/dev/null || echo 0) episode(s) logged so far; newest checkpoint: $(cat "$MRGYM_CKPT_DEST/LATEST" 2>/dev/null || echo none)"
if [ "$STUB" = 1 ]; then
  seeds="train:10000-$((10000 + PROMPTS_PER_STEP * (TOTAL_STEPS + 4))),dev:$DEV_SEEDS"; [ -n "$TEST_ARMS" ] && seeds="$seeds,test-iid:$TEST_SEEDS"
  serve_start stand-in "$TRAIN_ARMS $DEV_ARMS $TEST_ARMS" "$seeds" || fail "the stand-in server did not start"
  export SIM_BASE_URLS; SIM_BASE_URLS=$(printf 'http://127.0.0.1:%s/v1,' "${PORTS[@]}"); SIM_BASE_URLS=${SIM_BASE_URLS%,}
fi
"$BOOT_PY" - > "$R/remote/stop_config.json" <<PY
import json, os
e = os.environ.get
num = lambda name, default, cast=float: cast(e(name) or default)
total, freq = int("$TOTAL_STEPS"), int("$TEST_FREQ")
print(json.dumps(dict(
    episodes_per_step=int("$PROMPTS_PER_STEP") * int("$GROUP_SIZE"), group_size=int("$GROUP_SIZE"),
    no_reward_steps=num("NO_REWARD_STEPS", 4, int), no_reward_min=num("NO_REWARD_MIN", 0.02), harness_steps=num("HARNESS_STEPS", 10, int),
    base_rate=float("${BASE_RATE:-0}") or None, mixed_min=num("MIXED_MIN", 0.40), submit_min=num("SUBMIT_MIN", 0.50), cut_max=num("CUT_MAX", 0.20),
    window=num("WINDOW", 5, int), collapse_steps=num("COLLAPSE_STEPS", 10, int), alpha=num("ALPHA", 0.05), looks=num("LOOKS", max(1, total // freq), int),
    mde=num("MDE", 0.05), futility_from_step=num("FUTILITY_FROM_STEP", total // 2, int), mastered=num("MASTERED", 0.90)), indent=1))
PY
cat "$R/remote/stop_config.json"
gpu_mem_watch

STOP=$R/remote/STOP.json
if [ ! -f "$R/remote/TRAIN_DONE" ] && [ ! -f "$STOP" ]; then
  stage "3 training: $PROMPTS_PER_STEP groups x $GROUP_SIZE rollouts per step, $TOTAL_STEPS steps"
  "$PYBIN" -m mrgym_verl.stoprules --episodes "$LOG" --config "$R/remote/stop_config.json" --out "$R/remote/metrics.json" --watch \
      --pid-file "$W/out/trainer.pgid" --stop-file "$STOP" --interval "${WATCH_EVERY:-120}" > "$W/out/watch.log" 2>&1 & WATCH=$!; bg $WATCH
  launch_trainer "$W/out/train.log"
  kill "$WATCH" 2>/dev/null
  tail -n 5 "$W/out/train.log"
  "$PYBIN" -m mrgym_verl.smokecheck --log "$W/out/train.log" --episodes "$LOG" > "$W/out/verdict.txt"; VRC=$?; cat "$W/out/verdict.txt"
  st put "$W/out/train.log" "$SR/logs/train.log" >/dev/null 2>&1
  if [ -f "$STOP" ]; then log "stopped by a rule: $(cat "$STOP")"
  elif [ "$RC" = 0 ] && grep -q "step:$TOTAL_STEPS - " "$W/out/train.log"; then date -u +%FT%TZ > "$R/remote/TRAIN_DONE"; mark "training reached step $TOTAL_STEPS"
  elif [ "$RC" = 143 ] || [ "$RC" = 130 ]; then interrupted
  elif [ "$VRC" != 0 ]; then fail "training: $(cut -d'|' -f1 "$W/out/verdict.txt")"
  else fail "training ended with exit code $RC before step $TOTAL_STEPS (see train.log)"
  fi
fi
"$PYBIN" -m mrgym_verl.stoprules --episodes "$LOG" --config "$R/remote/stop_config.json" --out "$R/remote/metrics.json" > "$W/out/metrics.txt" 2>&1; cat "$W/out/metrics.txt"
if [ -f "$STOP" ]; then
  RULE=$("$BOOT_PY" -c "import json; print(json.load(open('$STOP'))['rule'])")
  if [ "$RULE" != mastered ]; then
    final_sync; st put "$STOP" "$SR/STOP.json"
    finish "STOPPED by rule $RULE: $("$BOOT_PY" -c "import json; print(json.load(open('$STOP'))['reason'])") - the held-out split was not opened"
  fi
fi

if [ -n "$TEST_ARMS" ]; then
  stage "4 readout on the held-out split: the base model and the final checkpoint, same instances, same harness"
  evaluate() {  # evaluate <name> <resume mode> <checkpoint folder> <checkpoint source>
    [ -f "$R/remote/$1/EVAL_DONE" ] && { log "$1: already evaluated"; return 0; }
    launch_trainer "$W/out/$1.log" VAL_ONLY=1 VAL_FILE=test RESUME_MODE="$2" CKPT_DIR="$3" RECORD_DIR="$R/remote/$1" MRGYM_CKPT_DEST="$4"
    st put "$W/out/$1.log" "$SR/logs/$1.log" >/dev/null 2>&1
    [ "$RC" = 143 ] || [ "$RC" = 130 ] && interrupted
    "$PYBIN" -m mrgym_verl.smokecheck --log "$W/out/$1.log" --episodes "$R/remote/$1/episodes.jsonl" --phase test > "$W/out/verdict_$1.txt" || fail "readout $1: $(cut -d'|' -f1 "$W/out/verdict_$1.txt")"
    [ "$RC" = 0 ] || fail "readout $1: the evaluation ended with exit code $RC (see $1.log)"
    if [ "$2" = auto ]; then grep -q "Resuming from" "$W/out/$1.log" || fail "readout $1: the evaluation did not load the checkpoint"; fi
    date -u +%FT%TZ > "$R/remote/$1/EVAL_DONE"
  }
  evaluate test_base disable "$R/run/ckpt_base" ""
  evaluate test_trained auto "$R/run/ckpt_eval" "$MRGYM_CKPT_DEST"
  "$PYBIN" -m mrgym_verl.readout --base "$R/remote/test_base/episodes.jsonl" --trained "$R/remote/test_trained/episodes.jsonl" --untrained "${UNTRAINED:-}" \
      --gain "${GAIN:-0.15}" --lower "${LOWER:-0.05}" --harm "${HARM:-0.05}" --min-envs "${MIN_ENVS:-2}" --out "$R/remote/readout.json" > "$W/out/readout.txt" 2>&1 || fail "readout (see readout.txt)"
  cat "$W/out/readout.txt"
  st put "$R/remote/readout.json" "$SR/readout.json"; st put "$W/out/readout.txt" "$SR/readout.txt"
fi
final_sync
st put "$R/remote/metrics.json" "$SR/metrics.json"; [ -f "$W/out/gpu_mem_peak.txt" ] && st put "$W/out/gpu_mem_peak.txt" "$SR/gpu_mem_peak.txt"
finish "TRAIN DONE: $(grep -c . "$LOG") training and evaluation episodes; $(tail -n 1 "$W/out/readout.txt" 2>/dev/null || echo 'no readout (TEST_ARMS not given)')"
