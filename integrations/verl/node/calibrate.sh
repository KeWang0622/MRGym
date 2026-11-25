#!/bin/bash
# Batch evaluation worker for one 8-GPU node, preemptible: base rates of one served open-weight model per
# (environment, level), in groups of K episodes per instance, with the package's own command (`mrgym calibrate`).
# Output: the episode records with full conversations, and a summary that names the cells in the 20 - 60 % band.
#
# Resumable. Every finished episode is uploaded within SYNC_EVERY seconds. A preempted job that starts again repeats
# the cheap setup, downloads the records, and `mrgym calibrate` plays only the episodes that are missing. A run id
# that finished or failed is never started again (use a new RUN_ID).
#
# Order: write access to the output -> repository -> environment package (CPU) -> self-test of the whole path against a
# stand-in server (no GPU) -> server environment -> weights -> servers -> two real episodes -> the plan -> summary.
#
# ============================== parameters: environment variables (prepend `export NAME=value` lines, or edit here) ==
#   RUN_ID          sub-folder of this run under OUT_URI/runs/                               (required)
#   OUT_URI         s3://<bucket>/<prefix> on a node; a directory in the local simulation   (required)
#   REPO_SRC        the repository: s3://.../repo.tar.gz | git+https://host/repo@ref | a .tar.gz | a directory  (required)
#   MODEL_HF        Hugging Face id of the model to serve                                   (required)
#   MODEL_NAME      served name; also the folder name of its records                        (required)
#   CAL_PLAN        cells: "ID:levels ID:levels", e.g. "Scanner/NoWrap-v1:0,1,2,3 Scanner/SNRSpec-v0:1,2,3"   (required)
#   CAL_N           instances per level                                                     32
#   CAL_K           episodes per instance (the group size of training)                      8
#   CAL_FIRST_SEED  first seed of the public train split                                    9000
#   CAL_MAX_TOKENS  cap of one reply, reasoning included (stated in the task text)          16384
#   CAL_TOOL_LIMIT  tool calls per episode; empty = each environment's own limit            (empty)
#   CAL_FROM_URI    records folder of an earlier calibration run (.../runs/<id>/records) to continue from: the same
#                   settings with a larger CAL_N or more cells play only what is missing                    (empty)
#   CAL_WORKERS     episodes in parallel per process (the runner allows at most 8)          8
#   CAL_PROCS       processes per server (seeds are split over servers x processes)         2
#   REASONING       reasoning effort the servers apply by default: medium | xhigh | low | off (no thinking)   medium
#   VLLM_TP / VLLM_MAXLEN / SERVE_PORTS   tensor parallel size, context, ports              4 / 262144 / "8000 8001"
#                   Before the plan and before every pass each port must answer with the served MODEL_NAME; if not, the
#                   job fails and plays nothing (a server of another run on the same port would give wrong episodes)
#   NGPUS           GPUs the node must have                                                 8
#   SYNC_EVERY      seconds between uploads of new records                                  45
#   SUMMARY_EVERY   seconds between interim summaries written next to the records (0 = off) 600
#   SELFTEST        1 = first run the whole path against a stand-in server on two instances (0: tests only)   1
#   HOLD_S          seconds a failed start keeps the node for inspection                    600
#   WORK_ROOT       work disk; give the node's local SSD mount. Default: the writable local disk with the most free space
#   MRGYM_STUB=1    local simulation: no installs of the server, no weights, no GPU; a stand-in server replays control
#                   policies (MODEL_NAME=mix:<p>:<q>) on free ports (STUB_SERVERS of them, default 1), each checked by a
#                   nonce of this run. STUB_SKIP_VENV=1 + STUB_PYTHON=<python> skips building the
#                   environment's venv too. SIM_DIE_AT=calibrate kills the job SIM_DIE_AFTER_S seconds into the plan.
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
: "${MODEL_HF:?}" "${MODEL_NAME:?}" "${CAL_PLAN:?}"
CAL_N=${CAL_N:-32}; CAL_K=${CAL_K:-8}; CAL_FIRST_SEED=${CAL_FIRST_SEED:-9000}; CAL_MAX_TOKENS=${CAL_MAX_TOKENS:-16384}
CAL_TOOL_LIMIT=${CAL_TOOL_LIMIT:-}; CAL_WORKERS=${CAL_WORKERS:-8}; CAL_PROCS=${CAL_PROCS:-2}

start_run
need_gpus
env_venv
REC=$W/rec; CAL=$REC/cal
LIMIT_ARGS=(--max-turns "${CAL_TOOL_LIMIT:-0}")          # 0 = each environment's own tool-call limit (`mrgym calibrate` alone defaults to 12)
LAST_SEED=$((CAL_FIRST_SEED + CAL_N - 1))

# ---- self-test without a GPU: the same command against a stand-in server on two instances of the first cell
FIRST_CELL=$(echo "$CAL_PLAN" | awk '{print $1}'); FIRST_ID=${FIRST_CELL%%:*}; FIRST_LV=${FIRST_CELL##*:}; FIRST_LV=${FIRST_LV%%,*}
if [ "${SELFTEST:-1}" = 1 ]; then
ST_PORT=$(stub_start selftest "$FIRST_ID:$FIRST_LV" "train:$CAL_FIRST_SEED-$((CAL_FIRST_SEED + 1))") || fail "self-test: the stand-in server did not start (see stub_selftest.log)"
curl -s -m 5 "http://127.0.0.1:$ST_PORT/health" | grep -q "\"nonce\": \"$RUN_NONCE\"" || fail "self-test: port $ST_PORT is not answered by this run's stand-in server"
"$PYBIN" -m mrgym.cli calibrate "$FIRST_ID" --model stub:reference --base-url "http://127.0.0.1:$ST_PORT/v1" --levels "$FIRST_LV" --n 2 --episodes 2 \
    --first-seed "$CAL_FIRST_SEED" --max-tokens "$CAL_MAX_TOKENS" "${LIMIT_ARGS[@]}" --workers 2 --out "$W/selftest" --json > "$W/out/selftest.json" 2> "$W/out/selftest.log"
kill "$(cat "$W/out/server_selftest.pid")" 2>/dev/null; rm -f "$W/out/server_selftest.pid"
"$PYBIN" - "$W/out/selftest.json" <<'PY' || fail "self-test of the calibration path against the stand-in server (see selftest.log)"
import json, sys
r = json.load(open(sys.argv[1]))
row = r["calibration"][0]
assert r["infra_errors"] == 0 and row["n_episodes"] == 4, row
print("self-test ok:", row["n_episodes"], "episodes, pass", row["pass"], "engine", r["engine"])
PY
mark "self-test ok"
fi

vllm_venv
get_model "$MODEL_HF"
# ---- records: pull what an earlier start uploaded, then upload every finished episode
if [ -n "${CAL_FROM_URI:-}" ]; then st down "$CAL_FROM_URI" "$REC" || fail "cannot read the records of the earlier run ($CAL_FROM_URI)"; fi
mirror_start "$REC" records
ALREADY=$(find "$CAL/episodes" -name record.json 2>/dev/null | wc -l); log "records already present: $ALREADY"
STUB_SEEDS="train:$CAL_FIRST_SEED-$LAST_SEED"
serve_start "$MODEL_NAME" "$CAL_PLAN" "$STUB_SEEDS" || fail "servers did not start"

if [ "$STUB" != 1 ] && [ "$ALREADY" = 0 ]; then   # two real episodes: the model must emit tool calls the server parses, without endpoint errors
  "$PYBIN" -m mrgym.cli run-agent --env "$FIRST_ID" --levels "$FIRST_LV" --seeds 8990-8991 --model "$MODEL_NAME" --base-url "http://127.0.0.1:${PORTS[0]}/v1" \
      --temperature 1.0 --max-tokens "$CAL_MAX_TOKENS" ${CAL_TOOL_LIMIT:+--max-turns "$CAL_TOOL_LIMIT"} --workers 2 --out "$W/smoke" > "$W/out/smoke.log" 2>&1
  tail -n 4 "$W/out/smoke.log"; st put "$W/out/smoke.log" "$SR/logs/smoke.log" >/dev/null 2>&1
  grep -q "finished 2 episode(s), 0 infra error(s)" "$W/out/smoke.log" || fail "the two real smoke episodes had endpoint errors (see smoke.log)"
  "$PYBIN" - "$W/smoke" <<'PY' || fail "tool-call parser mismatch: no call of the real model reached a tool in the smoke episodes"
import glob, json, sys
recs = [json.load(open(p)) for p in glob.glob(sys.argv[1] + "/episodes/*/*/*/*/*/record.json")]
executed = sum((r.get("interface") or {}).get("executed", 0) for r in recs)
print("smoke:", [(r["stop_reason"], r["tool_calls"], (r.get("interface") or {}).get("executed"), r["tokens_out"]) for r in recs])
assert executed > 0
PY
  mark "real-model smoke ok"
fi

# ---- the plan: one `mrgym calibrate` per (server, process) on its own share of the seeds; a second pass replays endpoint failures
NPROC=$(( ${#PORTS[@]} * CAL_PROCS )); [ "$NPROC" -gt "$CAL_N" ] && NPROC=$CAL_N
summary() {   # from the records; nothing is played. Exit 0 = every planned episode has a completed record, 2 = not yet
  "$PYBIN" -m mrgym_verl.calsum --out "$CAL" --model "$MODEL_NAME" --plan "$CAL_PLAN" --n "$CAL_N" --episodes "$CAL_K" --first-seed "$CAL_FIRST_SEED" \
      --gpus "$NGPUS" --concurrency "$(( NPROC * CAL_WORKERS ))" --write "$REC/summary" > "$W/out/summary.txt" 2> "$W/out/summary.log"
}
if [ "${SUMMARY_EVERY:-600}" != 0 ]; then   # an interim summary (marked INCOMPLETE) next to the records, so that a stopped job still leaves its table
  ( while :; do sleep "${SUMMARY_EVERY:-600}"; summary; done ) & bg $!
fi
for pass in 1 2 3; do
  check_servers "$MODEL_NAME" || fail "before pass $pass of the plan a server does not answer as this run's (see the log line above); no episode of this pass was played"
  PIDS=()
  for ((p = 0; p < NPROC; p++)); do
    lo=$(( CAL_FIRST_SEED + p * CAL_N / NPROC )); hi=$(( CAL_FIRST_SEED + (p + 1) * CAL_N / NPROC )); n=$(( hi - lo ))
    port=${PORTS[$(( p % ${#PORTS[@]} ))]}
    ( for cell in $CAL_PLAN; do
        "$PYBIN" -m mrgym.cli calibrate "${cell%%:*}" --model "$MODEL_NAME" --base-url "http://127.0.0.1:$port/v1" --levels "${cell##*:}" --n "$n" --episodes "$CAL_K" \
            --first-seed "$lo" --max-tokens "$CAL_MAX_TOKENS" "${LIMIT_ARGS[@]}" --reasoning "${REASONING:-medium}" --workers "$CAL_WORKERS" --out "$CAL" --json \
            > "$W/out/cal_p${p}_$(echo "${cell%%:*}" | tr '/' '-').json" 2>> "$W/out/cal_p$p.log"
        echo "pass $pass $cell rc=$? $(date -u +%FT%TZ)" >> "$W/out/cal_p$p.log"
      done ) & PIDS+=($!)
  done
  if [ "$STUB" = 1 ] && [ "${SIM_DIE_AT:-}" = calibrate ]; then sleep "${SIM_DIE_AFTER_S:-15}"; sim_die calibrate; fi
  for pid in "${PIDS[@]}"; do wait "$pid"; done
  summary; RC=$?
  mark "pass $pass of the plan: $(find "$CAL/episodes" -name record.json | wc -l) record(s), summary rc=$RC"
  [ $RC = 0 ] && break
done
serve_stop
cat "$W/out/summary.txt"
kill "$MIRROR_PID" 2>/dev/null; final_sync
for f in calibration_summary.json calibration_summary.md train_arms.json; do [ -f "$REC/summary/$f" ] && st put "$REC/summary/$f" "$SR/$f"; done
if [ $RC = 0 ]; then finish "calibration complete: $(find "$CAL/episodes" -name record.json | wc -l) episode record(s); cells in rule: $("$PYBIN" -c "import json; print(json.load(open('$REC/summary/train_arms.json'))['train_arms'] or 'none')")"; fi
[ $RC = 2 ] && fail "calibration incomplete after 3 passes: some planned episodes have no completed record (endpoint errors; see cal_p*.log). The finished records and a summary marked INCOMPLETE are uploaded"
fail "summary (see summary.log)"
