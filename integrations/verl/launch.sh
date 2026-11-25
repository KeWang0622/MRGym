#!/usr/bin/env bash
# One MR Gym run on verl: one node with 8 GPUs. Run from the verl checkout, inside its environment, with
# integrations/verl and the mrgym package importable:
#
#   TRAIN_ARMS="Scanner/NoWrap-v1:2,3 Scanner/SNRSpec-v0:1" MODEL_PATH=/models/M RUN_NAME=r1 \
#   MRGYM_CKPT_DEST=s3://<bucket>/<prefix>/r1/ckpt   bash <repo>/integrations/verl/launch.sh [extra hydra overrides]
#
# The same command resumes: on a disk without a checkpoint it restores the newest one from MRGYM_CKPT_DEST first
# (a preempted job starts on an empty disk) and verl continues from it (trainer.resume_mode=auto).
#
# Variables besides those of overrides.sh
#   TRAIN_ARMS         training cells "ID:levels ID:levels" (required unless DATA_DIR already holds the row files)
#   DEV_ARMS           cells of the in-training evaluation (default: TRAIN_ARMS); DEV_SEEDS (0-31) of the public dev split
#   TEST_ARMS          cells of the final readout on the secret split test-iid; TEST_SEEDS (0-199)
#   CURRICULUM_MODE    bandit | ladder | uniform (default bandit); CURRICULUM_DEAD "ID:levels" = cells calibrated dead
#   MRGYM_CKPT_DEST    s3://... or a directory; empty = checkpoints stay local
#   MRGYM_STUB=1       start the GPU-free stand-in (mrgym_verl.sim) instead of verl: same inputs, same files
#   MRGYM_PRINT_ONLY=1 print the command and exit
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ADAPTER_DIR=${ADAPTER_DIR:-$HERE}
source "$HERE/overrides.sh"
PY=${PYTHON:-python3}
STUB=${MRGYM_STUB:-0}
FMT=$([ "$STUB" = 1 ] && echo jsonl || echo parquet)
export PYTHONPATH="$HERE:${PYTHONPATH:-}"                       # mrgym_verl: driver and Ray workers
export MRGYM_CKPT_DEST=${MRGYM_CKPT_DEST:-} MRGYM_RECORD_DIR="$RECORD_DIR" RUN_NAME
export MRGYM_MODEL_NAME=${MRGYM_MODEL_NAME:-$(basename "$MODEL_PATH")}
export MRGYM_CURRICULUM="$CKPT_DIR/curriculum.json"
export MRGYM_ATTEMPT=${MRGYM_ATTEMPT:-$(date -u +%Y%m%dT%H%M%SZ)-$$}
export TOKENIZERS_PARALLELISM=false
export DATA_DIR CKPT_DIR RECORD_DIR TOTAL_STEPS SAVE_FREQ TEST_FREQ PROMPTS_PER_STEP GROUP_SIZE VAL_N VAL_FILE VAL_ONLY \
       VAL_BEFORE_TRAIN RESUME_MODE SIGNAL MAX_TOKENS_PER_TURN TOOL_CALL_LIMIT
mkdir -p "$CKPT_DIR" "$DATA_DIR" "$RECORD_DIR"

# ---- data: an index of instances; cheap, rebuilt when missing
if [ ! -f "$DATA_DIR/dataset.json" ]; then
    : "${TRAIN_ARMS:?TRAIN_ARMS (training cells) is required to build the row files}"
    args=(--out "$DATA_DIR" --arms "$TRAIN_ARMS" --train-rows "$((PROMPTS_PER_STEP * (TOTAL_STEPS + 8)))" --format "$FMT"
          --dev "${DEV_ARMS:-$TRAIN_ARMS}" --dev-seeds "${DEV_SEEDS:-0-31}")
    [ -n "${TEST_ARMS:-}" ] && args+=(--test "$TEST_ARMS" --test-seeds "${TEST_SEEDS:-0-199}")
    "$PY" -m mrgym_verl.instances "${args[@]}" > "$DATA_DIR/instances.log"
fi

# ---- the loop's options and the curriculum hook of this run
"$PY" - "$HERE/config/agent_loops.yaml" "$LOOP_CONFIG" <<PY
import re, sys
text = open(sys.argv[1]).read()
for key, value in (("signal", "${SIGNAL}"), ("max_tokens_per_turn", "${MAX_TOKENS_PER_TURN}"), ("tool_call_limit", "${TOOL_CALL_LIMIT:-null}")):
    text, n = re.subn(rf"(?m)^(  {key}: )\S+", rf"\g<1>{value}", text)
    assert n == 1, key
open(sys.argv[2], "w").write(text)
PY
if [ -n "${TRAIN_ARMS:-}" ] && [ ! -f "$MRGYM_CURRICULUM" ]; then
    "$PY" - <<PY
from mrgym_verl import instances
dead = instances.parse_arms("${CURRICULUM_DEAD:-}") if "${CURRICULUM_DEAD:-}" else []
instances.write_config("$MRGYM_CURRICULUM", instances.parse_arms("${TRAIN_ARMS}"), group_size=${GROUP_SIZE}, groups_per_step=${PROMPTS_PER_STEP},
                       state_path="$CKPT_DIR/curriculum_state.json", mode="${CURRICULUM_MODE:-bandit}", seed=${SEED:-0}, dead=dead)
PY
fi

# ---- restore: newest remote checkpoint -> local folder + verl's tracker file (+ the curriculum's state)
if [ -n "$MRGYM_CKPT_DEST" ] && [ "$RESUME_MODE" = auto ] && [ ! -f "$CKPT_DIR/latest_checkpointed_iteration.txt" ]; then
    latest=$("$PY" -m mrgym_verl.store restore "$MRGYM_CKPT_DEST" "$CKPT_DIR" | tail -1)
    if [[ "$latest" =~ ^[0-9]+$ ]]; then
        echo "[mrgym] restoring global_step_${latest} from $MRGYM_CKPT_DEST"
        [ -f "$CKPT_DIR/curriculum_state.json" ] || cp "$CKPT_DIR/global_step_${latest}/curriculum_state.json" "$CKPT_DIR/curriculum_state.json" 2>/dev/null || true
    else
        echo "[mrgym] no remote checkpoint: starting from the base model"
    fi
fi

if [ "$STUB" = 1 ]; then CMD=("$PY" -m mrgym_verl.sim); else CMD=("$PY" -m verl.trainer.main_ppo "${OVERRIDES[@]}" ray_kwargs.ray_init.runtime_env.py_executable=null "$@"); fi
if [ "${MRGYM_PRINT_ONLY:-0}" = 1 ]; then printf '%s \\\n' "${CMD[@]}"; exit 0; fi
printf '%s\n' "${OVERRIDES[@]}" > "$CKPT_DIR/overrides.txt"
exec "${CMD[@]}"
