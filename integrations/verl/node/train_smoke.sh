#!/bin/bash
# Batch evaluation worker for one 8-GPU node, preemptible: a short end-to-end check of the trainer path on the chosen
# model at a tiny batch. Stages: install -> the adapter on CPU with the model's tokenizer -> serve + rollouts + gradient
# steps + checkpoints (run A) -> the checkpoint left the node -> local disk lost, restore, one evaluation with the
# restored adapter and one more step (run B) -> the stored episodes are complete records.
# The last line of the log is `SMOKE PASS ...` (exit 0) or `SMOKE FAIL <stage>: <reason>` (exit 1). Causes that are
# named when they occur: out of memory on load, out of memory on the first backward, tool-call parser mismatch,
# context overflow (integrations/verl/mrgym_verl/smokecheck.py). The report (smoke_report.json) holds what only a GPU
# run can measure: seconds per step, peak memory per GPU, token lengths of real episodes.
#
# ============================== parameters: environment variables (prepend `export NAME=value` lines, or edit here) ==
#   RUN_ID, OUT_URI, REPO_SRC      as in calibrate.sh                                       (required)
#   MODEL_HF        Hugging Face id of the model to train                                   (required)
#   TRAIN_ARMS      cells "ID:levels ID:levels"; the first cell is used for the CPU gate    (required)
#   SMOKE_STEPS     gradient steps of run A (run B adds one)                                2
#   PROMPTS_PER_STEP / GROUP_SIZE   groups per step / rollouts per group                    2 / 8
#   DEV_SEEDS / VAL_N               evaluation instances per cell / episodes each           0-1 / 1
#   every variable of integrations/verl/overrides.sh (ROLLOUT_TP, GPU_MEM_UTIL, MAX_PROMPT_LEN, MAX_RESPONSE_LEN,
#   MAX_TOKENS_PER_TURN, TOKENS_PER_GPU, LORA_RANK, CHAT_KWARGS, TOOL_FORMAT, ...) is passed through when set
#   NGPUS 8, SYNC_EVERY 45, HOLD_S 600, WORK_ROOT, VERL_REF (pinned commit)
#   MRGYM_STUB=1    local simulation: no verl, no weights, no GPU; integrations/verl/launch.sh runs the stand-in trainer
#                   (mrgym_verl.sim) against a stand-in server. STUB_SKIP_VENV=1 + STUB_PYTHON as in calibrate.sh.
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
: "${MODEL_HF:?}" "${TRAIN_ARMS:?}"
SMOKE_STEPS=${SMOKE_STEPS:-2}
sfail() { echo "SMOKE FAIL $1"; fail "smoke: $1"; }
stage() { echo; log "=== SMOKE $1 ==="; mark "stage $1"; }

start_run
stage "0 node"; need_gpus
stage "1 install"; env_venv; verl_env; get_model "$MODEL_HF"
export RUN_NAME=${RUN_NAME:-smoke} MODEL_PATH=$MODEL_DIR MRGYM_MODEL_NAME=${MODEL_NAME:-$(basename "$MODEL_HF")} TRAIN_ARMS
export DATA_DIR=$R/smoke/data CKPT_DIR=$R/smoke/ckpt RECORD_DIR=$R/remote/records MRGYM_CKPT_DEST=$R/remote/ckpt
export PROMPTS_PER_STEP=${PROMPTS_PER_STEP:-2} GROUP_SIZE=${GROUP_SIZE:-8} SAVE_FREQ=1 TEST_FREQ=1 VAL_N=${VAL_N:-1} DEV_SEEDS=${DEV_SEEDS:-0-1}
rm -rf "$R/smoke" "$R/remote"; mkdir -p "$CKPT_DIR"
LOG=$RECORD_DIR/episodes.jsonl
verdict() { "$PYBIN" -m mrgym_verl.smokecheck --log "$1" --episodes "$LOG" "${@:2}" | tee "$W/out/verdict.txt"; return "${PIPESTATUS[0]}"; }

stage "2 the adapter on CPU with the model's tokenizer (verl's own code, no GPU)"
cpu_gate "$(echo "$TRAIN_ARMS" | awk '{print $1}')"

ATTEMPT=attempts/$(date -u +%Y%m%dT%H%M%SZ)          # a smoke that was preempted starts clean: every start has its own folder
mirror_start "$R/remote" "$ATTEMPT"
if [ "$STUB" = 1 ]; then
  serve_start stand-in "$TRAIN_ARMS" "train:10000-$((10000 + PROMPTS_PER_STEP * (SMOKE_STEPS + 3))),dev:$DEV_SEEDS" || sfail "1: the stand-in server did not start"
  export SIM_BASE_URLS; SIM_BASE_URLS=$(printf 'http://127.0.0.1:%s/v1,' "${PORTS[@]}"); SIM_BASE_URLS=${SIM_BASE_URLS%,}
fi
gpu_mem_watch

stage "3 run A: serve, evaluate, $SMOKE_STEPS steps of $PROMPTS_PER_STEP x $GROUP_SIZE rollouts, a checkpoint per step"
launch_trainer "$W/out/run_a.log" TOTAL_STEPS="$SMOKE_STEPS"
verdict "$W/out/run_a.log" --expect-step "$SMOKE_STEPS" || sfail "3: $(cut -d'|' -f1 "$W/out/verdict.txt")"
[ "$RC" = 0 ] || sfail "3: the trainer ended with exit code $RC (see run_a.log)"
grep -q "actor/grad_norm" "$W/out/run_a.log" || sfail "3: no gradient step was logged"
for n in $(seq "$SMOKE_STEPS"); do [ -d "$CKPT_DIR/global_step_$n" ] || sfail "3: checkpoint $n missing"; done
du -sh "$CKPT_DIR"/global_step_* | tee "$W/out/ckpt_sizes.txt"
cp "$W/out/gpu_mem_peak.txt" "$W/out/gpu_mem_peak_run_a.txt" 2>/dev/null

stage "4 the checkpoint left the node"
[ "$("$BOOT_PY" -c "import json; print(json.load(open('$MRGYM_CKPT_DEST/LATEST'))['global_step'])" 2>/dev/null)" = "$SMOKE_STEPS" ] || sfail "4: LATEST is not step $SMOKE_STEPS"
final_sync; st exists "$SR/$ATTEMPT/ckpt/LATEST" || sfail "4: LATEST did not reach $SR/$ATTEMPT/ckpt"

stage "5 preemption: local disk lost, same command again; evaluation with the restored adapter, one more step"
rm -rf "$CKPT_DIR"
launch_trainer "$W/out/run_b.log" TOTAL_STEPS="$((SMOKE_STEPS + 1))"
verdict "$W/out/run_b.log" --expect-step "$((SMOKE_STEPS + 1))" || sfail "5: $(cut -d'|' -f1 "$W/out/verdict.txt")"
[ "$RC" = 0 ] || sfail "5: the trainer ended with exit code $RC (see run_b.log)"
grep -q "restoring global_step_$SMOKE_STEPS" "$W/out/run_b.log" || sfail "5: nothing was restored"
grep -q "Resuming from .*global_step_$SMOKE_STEPS" "$W/out/run_b.log" || sfail "5: the trainer did not resume from step $SMOKE_STEPS"
grep -q "step:1 - " "$W/out/run_b.log" && sfail "5: the run started over instead of resuming"
[ -d "$CKPT_DIR/global_step_$((SMOKE_STEPS + 1))" ] || sfail "5: checkpoint $((SMOKE_STEPS + 1)) missing"

stage "6 the stored episodes: complete records, readable log, the evaluation after the restore"
"$PYBIN" - "$RECORD_DIR" "$SMOKE_STEPS" "$PROMPTS_PER_STEP" "$GROUP_SIZE" > "$W/out/smoke_report.json" 2> "$W/out/records_check.log" <<'PY' || sfail "6: $(tail -n 1 "$W/out/records_check.log")"
import glob, json, os, sys
from mrgym import record as mrecord
from mrgym_verl import stoprules
root, steps, prompts, group = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rows = stoprules.load(os.path.join(root, "episodes.jsonl"))
train = [r for r in rows if r["phase"] == "train"]
attempts = sorted({r["attempt"] for r in rows})
restored_eval = [r for r in rows if r["phase"] == "dev" and r["step"] == steps and r["attempt"] == attempts[-1]]
assert len(attempts) == 2, f"expected the episodes of two trainer runs, found {len(attempts)}"
assert len([r for r in train if r["step"] <= steps and r["attempt"] == attempts[0]]) == steps * prompts * group, "run A did not log every training episode"
assert len([r for r in train if r["step"] == steps + 1]) == prompts * group, "run B did not log the episodes of its step"
assert restored_eval, "no evaluation episode was logged after the restore"
paths = sorted(glob.glob(os.path.join(root, "*", "step_*", "episodes", "*", "*", "*", "*", "*", "record.json")))
assert len(paths) == len(rows), f"{len(paths)} record files for {len(rows)} logged episodes"
bad = []
for p in paths:
    rec = json.load(open(p))
    problems = mrecord.missing(rec) + ([mrecord.transcript_problem(rec, os.path.dirname(p))] if mrecord.transcript_problem(rec, os.path.dirname(p)) else [])
    if problems:
        bad.append((p, problems))
assert not bad, f"{len(bad)} record(s) lack fields, first: {bad[0]}"
d = stoprules.decide(rows, dict(episodes_per_step=prompts * group, group_size=group, no_reward_steps=10 ** 6, harness_steps=10 ** 6, window=10 ** 6))
print(json.dumps(dict(episodes=len(rows), records=len(paths), attempts=attempts, evaluation_after_restore=len(restored_eval), steps=d["steps"], dev=d["dev"],
                      context_tokens_max=max((r.get("context_tokens") or 0 for r in rows), default=0),
                      tokens_out_p50=sorted(r.get("tokens_out") or 0 for r in rows)[len(rows) // 2],
                      outcomes={o: sum(r["outcome"] == o for r in rows) for o in sorted({r["outcome"] for r in rows})}), indent=1))
PY
"$PYBIN" -m mrgym_verl.stoprules --episodes "$LOG" --out "$R/remote/metrics.json" > "$W/out/metrics.txt" 2>&1; cat "$W/out/metrics.txt"
grep -h "timing_s/step" "$W/out/run_a.log" "$W/out/run_b.log" | grep -o "step:[0-9]* \|timing_s/[a-z_]*:[0-9.]*" | tr '\n' ' ' > "$W/out/timings.txt"; echo; cat "$W/out/timings.txt"; echo
cat "$W/out/gpu_mem_peak.txt" 2>/dev/null
final_sync
for f in smoke_report.json metrics.txt timings.txt gpu_mem_peak.txt gpu_mem_peak_run_a.txt ckpt_sizes.txt run_a.log run_b.log dryrun.log; do [ -f "$W/out/$f" ] && st put "$W/out/$f" "$SR/$f"; done
echo "SMOKE PASS"
finish "SMOKE PASS: $((SMOKE_STEPS + 1)) steps, checkpoint restored from $SR/$ATTEMPT/ckpt, $(grep -c . "$LOG") episodes recorded"
