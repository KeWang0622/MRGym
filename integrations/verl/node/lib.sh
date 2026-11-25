#!/bin/bash
# Shared functions of the node entry scripts (calibrate.sh, train_smoke.sh, train.sh). Sourced after their bootstrap
# block, which has set: W (work folder), REPO (the unpacked repository), RUN_ID, OUT_URI, STUB, HOLD_S, BOOT_PY.
#
# Everything a job writes for others goes under $SR = $OUT_URI/runs/$RUN_ID (an s3:// prefix on a node, a directory in
# the local simulation; integrations/verl/mrgym_verl/store.py handles both). Nothing is ever deleted there.
#
# Lessons kept from the earlier jobs (project record, runs r01, t01, t02):
#   * virtual environments are built with uv-managed CPython only (the image's python has no Python.h);
#   * uv is called through a link to the real uv executable (uv.find_uv_bin()), never as `python -m uv`: inside an
#     activated environment `python` is that environment's interpreter, which has no uv module, and
#     `python -m uv pip install` installs into the interpreter that launched it;
#   * a run id that finished or failed is never started again; a preempted one (no marker) continues.

SR=${OUT_URI%/}/runs/$RUN_ID
ADAPTER=$REPO/integrations/verl
NGPUS=${NGPUS:-8}
SYNC_EVERY=${SYNC_EVERY:-45}
BG_PIDS=()

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
st() { "$BOOT_PY" "$ADAPTER/mrgym_verl/store.py" "$@"; }            # put | get | exists | ls | down | up | loop | restore
mark() { echo "$1 $(date -u +%FT%TZ)" | tee -a "$W/out/progress.status"; st put "$W/out/progress.status" "$SR/progress.status" >/dev/null 2>&1 || true; }
upload_logs() { local f; for f in "$W"/out/*.log "$W"/out/*.status "$W"/out/*.json "$W"/out/*.txt; do [ -f "$f" ] && st put "$f" "$SR/logs/$(basename "$f")" >/dev/null 2>&1; done; return 0; }
bg() { BG_PIDS+=("$1"); }
kill_tree() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c" "${2:-TERM}"; done; kill -"${2:-TERM}" "$1" 2>/dev/null; return 0; }   # a process and its descendants
stop_bg() { local p; for p in "${BG_PIDS[@]:-}"; do [ -n "$p" ] && kill_tree "$p"; done; serve_stop 2>/dev/null; return 0; }

fail() {  # record, upload every log, leave a marker so that a restarted job does not repeat a failing setup, hold, exit 1
  mark "FAILED $1"; upload_logs
  echo "$(date -u +%FT%TZ) $(hostname) FAILED $1" > "$W/out/FAILED.txt"; st put "$W/out/FAILED.txt" "$SR/FAILED.txt" >/dev/null 2>&1
  log "FAILED $1 - holding for ${HOLD_S} s"; sleep "$HOLD_S"
  stop_bg; exit 1
}
finish() {  # finish "<one line>"  -> DONE marker, exit 0
  mark "done: $1"; upload_logs
  echo "$(date -u +%FT%TZ) $1" > "$W/out/DONE.txt"; st put "$W/out/DONE.txt" "$SR/DONE.txt" >/dev/null 2>&1
  stop_bg; exit 0
}
interrupted() {  # a signal (preemption): no marker, so the requeued job continues from what was uploaded
  kill_trainer TERM; mark "interrupted - what was uploaded stays; the next start continues"; final_sync 2>/dev/null; upload_logs; stop_bg; exit 143
}

start_run() {
  exec > >(tee -a "$W/out/main.log") 2>&1
  log "start host=$(hostname) run=$RUN_ID stub=$STUB out=$SR work=$W"
  st put "$W/out/main.log" "$SR/logs/main.log" || { log "FATAL: cannot write to $SR (on a node: attach the data-access role of the bucket to the job)"; sleep "$HOLD_S"; exit 1; }
  local m
  for m in DONE.txt FAILED.txt; do
    if st get "$SR/$m" "$W/out/prev_$m" >/dev/null 2>&1; then
      log "$SR/$m exists: $(cat "$W/out/prev_$m"). Nothing to do; use a new RUN_ID for a new run."; exit 0
    fi
  done
  trap interrupted TERM INT
  trap stop_bg EXIT
  if [ "${HEARTBEAT_S:-300}" != 0 ]; then heartbeat & bg $!; fi
  mark "start"
}

heartbeat() {
  while :; do
    { echo "== $(date -u +%FT%TZ) $(hostname) run=$RUN_ID"; tail -n 30 "$W/out/main.log"
      echo "== gpu"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
      echo "== disk"; df -h "$W" | tail -1; echo "== load $(cut -d' ' -f1-3 /proc/loadavg)"
      echo "== metrics"; tail -n 25 "$W/out/metrics.txt" 2>/dev/null; } > "$W/out/heartbeat.txt" 2>&1
    st put "$W/out/heartbeat.txt" "$SR/heartbeat.txt" >/dev/null 2>&1
    sleep "${HEARTBEAT_S:-300}"
  done
}

need_gpus() {
  [ "$STUB" = 1 ] && { log "stub: GPU check skipped"; return 0; }
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | tee "$W/out/gpus.txt"
  [ "$(wc -l < "$W/out/gpus.txt")" -ge "$NGPUS" ] || fail "fewer than $NGPUS GPUs"
}

# ---------------------------------------------------------------- python environments
setup_uv() {
  mkdir -p "$W/bin"
  if [ -x "${UV_BIN:-}" ]; then UVBIN=$UV_BIN
  else
    "$BOOT_PY" -m uv --version >/dev/null 2>&1 || "$BOOT_PY" -m pip install -q uv 2>&1 | tail -1
    UVBIN=$("$BOOT_PY" -c "import uv; print(uv.find_uv_bin())" 2>/dev/null); [ -x "$UVBIN" ] || UVBIN=$(command -v uv)
  fi
  [ -x "$UVBIN" ] || fail "uv executable not found"
  ln -sf "$UVBIN" "$W/bin/uv"
  export PATH=$W/bin:$PATH UV_PYTHON_PREFERENCE=only-managed UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$W/uv-python} UV_CACHE_DIR=${UV_CACHE_DIR:-$W/.uv_cache}
  uv --version || fail "uv"
}

env_venv() {   # the environment package (numpy + scipy) in its own venv; PYBIN runs mrgym and mrgym_verl
  if [ "$STUB" = 1 ] && [ "${STUB_SKIP_VENV:-0}" = 1 ]; then
    PYBIN=${STUB_PYTHON:?STUB_PYTHON (an interpreter with numpy and scipy) is required with STUB_SKIP_VENV=1}
    export PYTHONPATH="$REPO/src:$ADAPTER"
  else
    setup_uv
    [ -x "$W/genv/bin/python" ] || { uv venv -q -p 3.12 "$W/genv" && uv pip install -q --python "$W/genv/bin/python" "$REPO" boto3; } || fail "environment package install"
    PYBIN=$W/genv/bin/python
    export PYTHONPATH="$ADAPTER"
  fi
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
  "$PYBIN" -m mrgym.cli --version || fail "environment package does not import"
  "$PYBIN" -c "import mrgym_verl.episode, mrgym_verl.stoprules, mrgym_verl.calsum; print('adapter', mrgym_verl.episode.adapter_sha256()[:12])" || fail "adapter does not import"
  mark "environment ok"
}

vllm_venv() {
  [ "$STUB" = 1 ] && { log "stub: no vLLM environment"; return 0; }
  [ -x "$W/vvenv/bin/vllm" ] || { uv venv -q -p 3.12 "$W/vvenv" && uv pip install -q --python "$W/vvenv/bin/python" "vllm==${VLLM_VERSION:-0.33.0}" "huggingface_hub[cli]" hf_transfer; }
  "$W/vvenv/bin/python" -c "import sysconfig, os, sys; h = os.path.join(sysconfig.get_paths()['include'], 'Python.h'); print('server python', sys.version.split()[0], 'Python.h', os.path.exists(h)); sys.exit(0 if os.path.exists(h) else 1)" || fail "server python has no Python.h"
  "$W/vvenv/bin/python" -c "import vllm, transformers, torch; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)" || fail "vllm environment"
  HF=$W/vvenv/bin/hf
  mark "server environment ok"
}

get_model() {   # get_model <hf id> -> MODEL_DIR
  MODEL_DIR=$W/models/$(basename "$1")
  [ "$STUB" = 1 ] && { log "stub: no weights"; return 0; }
  export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME=$W/hf_home
  local tries=0
  until "$HF" download "$1" --local-dir "$MODEL_DIR" > "$W/out/hf_download.log" 2>&1; do tries=$((tries+1)); [ $tries -ge 6 ] && break; sleep 30; done
  [ -s "$MODEL_DIR/config.json" ] && [ "$(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l)" -ge 1 ] || fail "model download ($1)"
  du -sh "$MODEL_DIR"; mark "model done"
}

# ---------------------------------------------------------------- model servers (calibration; verl starts its own)
PORTS=(${SERVE_PORTS:-8000 8001})          # ports of the real servers; the stand-ins of the simulation take free ports
RUN_NONCE=${RUN_NONCE:-$RUN_ID-$$-$(date +%s%N)}

stub_start() {  # stub_start <label> <cells> <seeds>: a stand-in server on a free port; prints the port; pid in $W/out/server_<label>.pid
  local pf=$W/out/server_$1.port t0
  rm -f "$pf"
  "$PYBIN" -m mrgym_verl.mixstub --port 0 --port-file "$pf" --nonce "$RUN_NONCE" --cells "$2" --seeds "$3" > "$W/out/stub_$1.log" 2>&1 &
  echo $! > "$W/out/server_$1.pid"
  t0=$(date +%s)
  until [ -s "$pf" ]; do
    kill -0 "$(cat "$W/out/server_$1.pid")" 2>/dev/null || { tail -n 5 "$W/out/stub_$1.log" >&2; return 1; }
    [ $(( $(date +%s) - t0 )) -gt 120 ] && return 1
    sleep 0.2
  done
  cat "$pf"
}

check_servers() {  # check_servers <served name>: every port answers, and with this run's server (not one left by, or started for, another run)
  local port body
  for port in "${PORTS[@]}"; do
    if [ "$STUB" = 1 ]; then
      body=$(curl -s -m 5 "http://127.0.0.1:$port/health") || { log "no answer on port $port"; return 1; }
      echo "$body" | grep -q "\"nonce\": \"$RUN_NONCE\"" || { log "port $port is answered by another run's server: $body"; return 1; }
    else
      body=$(curl -s -m 30 "http://127.0.0.1:$port/v1/models") || { log "no answer on port $port"; return 1; }
      echo "$body" | grep -q "\"id\": *\"$1\"" || { log "port $port does not serve $1: $(echo "$body" | head -c 300)"; return 1; }
    fi
  done
}

serve_start() {  # serve_start <served name> <stub cells> <stub seeds>
  local port i
  if [ "$STUB" = 1 ]; then
    PORTS=()
    for ((i = 0; i < ${STUB_SERVERS:-1}; i++)); do port=$(stub_start "$i" "$2" "$3") || return 1; PORTS+=("$port"); done
  else
    QWEN_W=$W QWEN_TP=${VLLM_TP:-4} QWEN_MAXLEN=${VLLM_MAXLEN:-262144} QWEN_PORTS="${PORTS[*]}" bash "$ADAPTER/node/serve.sh" "${REASONING:-medium}" "$MODEL_DIR" "$1" \
      || { for port in "${PORTS[@]}"; do st put "$W/out/vllm_$port.log" "$SR/logs/vllm_$port.log" >/dev/null 2>&1; done; return 1; }
  fi
  check_servers "$1" || return 1
  mark "servers ready (ports ${PORTS[*]})"
}
serve_stop() {
  local f
  for f in "$W"/out/server_*.pid "$W"/run/vllm_*.pid; do [ -f "$f" ] && { kill "$(cat "$f")" 2>/dev/null; rm -f "$f"; }; done
  return 0
}

# ---------------------------------------------------------------- mirror of a folder (records, checkpoints)
mirror_start() {  # mirror_start <local dir> <sub-prefix>: pull what a previous start uploaded, then upload continuously
  MIRROR_DIR=$1; MIRROR_URI=$SR/$2; mkdir -p "$MIRROR_DIR"
  st down "$MIRROR_URI" "$MIRROR_DIR" || fail "mirror download ($MIRROR_URI)"
  "$BOOT_PY" "$ADAPTER/mrgym_verl/store.py" loop "$MIRROR_DIR" "$MIRROR_URI" "$SYNC_EVERY" > "$W/out/sync.log" 2>&1 & MIRROR_PID=$!; bg $MIRROR_PID   # the interpreter itself, so that killing the pid ends it
}
final_sync() { [ -n "${MIRROR_DIR:-}" ] && st up "$MIRROR_DIR" "$MIRROR_URI" >> "$W/out/sync.log" 2>&1; return 0; }

sim_die() {  # sim_die <stage>: in the simulation, SIM_DIE_AT=<stage> kills this job the hard way at that point (a preemption)
  if [ "$STUB" = 1 ] && [ "${SIM_DIE_AT:-}" = "$1" ]; then
    log "SIM_DIE_AT=$1: killing the job (simulated preemption; no marker, no final upload)"
    trap - EXIT TERM INT
    trap '' PIPE; exec > /dev/null 2>&1                 # the log pipe dies with the children; nothing more is written
    serve_stop; kill_trainer KILL
    local p; for p in $(pgrep -P $$ 2>/dev/null); do kill_tree "$p" KILL; done
    kill -9 $$
  fi
}

# ---------------------------------------------------------------- the trainer's environment (train_smoke.sh, train.sh)
verl_env() {
  R=$W/rl; mkdir -p "$R"
  if [ "$STUB" = 1 ]; then log "stub: verl is not installed; launch.sh runs mrgym_verl.sim"; TRAIN_PY=$PYBIN; return 0; fi
  local ref=${VERL_REF:-8357f1e0a7ce4c06dc970c2c637d49b33815f5cc}
  [ -d "$R/verl/.git" ] || git clone -q https://github.com/verl-project/verl.git "$R/verl" || fail "verl clone"
  git -C "$R/verl" fetch -q --depth 1 origin "$ref" && git -C "$R/verl" checkout -q "$ref" || fail "verl checkout $ref"
  ( cd "$R/verl" && uv sync --frozen --extra vllm --extra fsdp > "$W/out/uv_sync.log" 2>&1 ) || fail "uv sync of verl (see uv_sync.log)"
  uv pip install -q --python "$R/verl/.venv/bin/python" "$REPO" boto3 > "$W/out/pip_env.log" 2>&1 || fail "environment package install into the trainer's environment"
  TRAIN_PY=$R/verl/.venv/bin/python
  "$TRAIN_PY" - <<PY || fail "imports / CUDA in the trainer's environment"
import torch, vllm, verl, mrgym, boto3
assert torch.cuda.is_available() and torch.cuda.device_count() >= $NGPUS, (torch.cuda.is_available(), torch.cuda.device_count())
print("torch", torch.__version__, "vllm", vllm.__version__, "mrgym", mrgym.__version__, mrgym.ENGINE_ID)
PY
  HF=$R/verl/.venv/bin/hf; [ -x "$HF" ] || { uv venv -q -p 3.12 "$W/hfenv" && uv pip install -q --python "$W/hfenv/bin/python" "huggingface_hub[cli]" hf_transfer && HF=$W/hfenv/bin/hf; } || fail "download tool"
  mark "trainer environment ok"
}

cpu_gate() {  # the adapter with verl's own code and the model's tokenizer, no GPU: wrong override keys, template and parser mismatches stop here
  [ "$STUB" = 1 ] && { log "stub: CPU gate skipped (it needs verl; see SIMULATION.md for its own run)"; return 0; }
  ( cd "$R/verl" && export PYTHONPATH="$ADAPTER" && source "$ADAPTER/overrides.sh" && printf '%s\n' "${OVERRIDES[@]}" > "$R/overrides_gate.txt" \
    && cp "$ADAPTER/config/agent_loops.yaml" "$LOOP_CONFIG" \
    && "$TRAIN_PY" -m mrgym_verl.dryrun --tokenizer "$MODEL_DIR" --overrides "$R/overrides_gate.txt" --cells "$1" --seeds 100-101 \
         --model-type "$("$TRAIN_PY" -c "import json;print(json.load(open('$MODEL_DIR/config.json'))['model_type'])")" --out "$R/dryrun" ) > "$W/out/dryrun.log" 2>&1 \
    || fail "CPU gate of the adapter (see dryrun.log)"
  grep -q "DRYRUN PASS" "$W/out/dryrun.log" || fail "CPU gate did not pass"
  mark "CPU gate ok"
}

gpu_mem_watch() {  # peak memory per GPU, sampled every 5 s -> $W/out/gpu_mem_peak.txt
  [ "$STUB" = 1 ] && return 0
  ( peak=""; while :; do
      cur=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')
      peak=$("$BOOT_PY" -c "import sys; a=[int(x) for x in sys.argv[1].split()]; b=[int(x) for x in sys.argv[2].split()] or [0]*len(a); print(' '.join(str(max(x,y)) for x,y in zip(a,b)))" "$cur" "$peak" 2>/dev/null)
      echo "peak MiB per GPU: $peak" > "$W/out/gpu_mem_peak.txt"; sleep 5
    done ) & bg $!
}

kill_trainer() { [ -f "$W/out/trainer.pgid" ] && kill -"${1:-TERM}" -- -"$(cat "$W/out/trainer.pgid")" 2>/dev/null; return 0; }

launch_trainer() {  # launch_trainer <log file> [VAR=value ...]: integrations/verl/launch.sh as the leader of its own process group; sets RC
  local logf=$1; shift
  rm -f "$W/out/trainer.pgid"
  ( [ "$STUB" = 1 ] || { cd "$R/verl" && source .venv/bin/activate; }
    export "$@" PYTHON="$TRAIN_PY" MRGYM_STUB="$STUB"
    exec setsid -w bash -c 'echo $$ > "$0"; exec bash "$1"' "$W/out/trainer.pgid" "$ADAPTER/launch.sh" ) >> "$logf" 2>&1 &
  wait $!; RC=$?
  [ "$STUB" = 1 ] || "$R/verl/.venv/bin/ray" stop --force >/dev/null 2>&1
  if [ "$STUB" = 1 ] && [ "$RC" = 137 ] && [ -n "${SIM_DIE_AT_STEP:-}" ]; then SIM_DIE_AT=trainer sim_die trainer; fi   # the stand-in was "preempted"
  return 0
}
