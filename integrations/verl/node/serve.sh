#!/bin/bash
# Start (or restart) vLLM servers for one open-weight model on one 8-GPU node: by default two servers, GPUs 0-3 -> :8000
# and GPUs 4-7 -> :8001, tensor parallel 4. Used by calibrate.sh (the training scripts let verl start its own replicas).
# This is the serving script of the earlier base-rate job (it served three models for 4.7 h without a failure), with the
# ports and GPU groups made variables.
# Usage: serve.sh <reasoning_effort: medium|xhigh|low|off> <model_dir> <served_name>      (off: the template's enable_thinking=false)
# Variables: QWEN_W (work folder), QWEN_TP (4), QWEN_MAXLEN (262144), QWEN_PORTS ("8000 8001"), QWEN_GPUS ("0,1,2,3 4,5,6,7"),
#            QWEN_MAX_SEQS (32), QWEN_LORA_MODULES / QWEN_LORA_ENABLE / QWEN_LORA_RANK (serve adapters next to the base model).
# Writes $W/out/serve.status ("ready <mode>" or "failed") and keeps server logs in $W/out/vllm_<port>.log.
# Fallback chain if a server does not become healthy: drop MTP speculative decoding, then drop --language-model-only.
set -u
W=${QWEN_W:?QWEN_W (work folder) is required}
EFFORT=${1:-medium}
MODEL_DIR=${2:?model folder}
NAME=${3:?served name}
VENV=$W/vvenv
TP=${QWEN_TP:-4}
MAXLEN=${QWEN_MAXLEN:-262144}
read -r -a PORTS <<< "${QWEN_PORTS:-8000 8001}"; read -r -a GPUS <<< "${QWEN_GPUS:-0,1,2,3 4,5,6,7}"
LORA_ON=${QWEN_LORA_MODULES:-}${QWEN_LORA_ENABLE:-}   # QWEN_LORA_ENABLE=1: LoRA on, adapters loaded at runtime (/v1/load_lora_adapter)
mkdir -p $W/out $W/run
if [ "$EFFORT" = off ]; then TEMPLATE_KW='{"enable_thinking":false}'; else TEMPLATE_KW="{\"reasoning_effort\":\"$EFFORT\"}"; fi
echo "starting $(date -u +%FT%TZ) effort=$EFFORT template kwargs=$TEMPLATE_KW" > $W/out/serve.status

stop_servers() {
  for f in $W/run/vllm_*.pid; do [ -f "$f" ] && { kill "$(cat "$f")" 2>/dev/null; rm -f "$f"; }; done
  # also any stray vllm engine processes
  for p in $(pgrep -f "[v]llm serve"); do kill "$p" 2>/dev/null; done
  sleep 5
  for p in $(pgrep -f "[v]llm serve"); do kill -9 "$p" 2>/dev/null; done
  sleep 3
}

launch() {  # launch <variant>  variant: full | nomtp | plain
  local variant=$1 i port gpus extra
  for i in "${!PORTS[@]}"; do
    port=${PORTS[$i]}; gpus=${GPUS[$i]}
    extra=()
    [ "$variant" != "plain" ] && extra+=(--language-model-only)
    [ "$variant" = "full" ] && extra+=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
    if [ -n "${QWEN_LORA_MODULES:-}${QWEN_LORA_ENABLE:-}" ]; then
      extra+=(--enable-lora --max-lora-rank "${QWEN_LORA_RANK:-32}" --max-loras 2 --max-cpu-loras 4)
      # shellcheck disable=SC2206
      [ -n "${QWEN_LORA_MODULES:-}" ] && extra+=(--lora-modules ${QWEN_LORA_MODULES})
    fi
    # Python frontend: the Rust frontend returned 501 on one route in an earlier job; the Python one served every run since.
    CUDA_VISIBLE_DEVICES=$gpus VLLM_USE_RUST_FRONTEND=0 VLLM_ALLOW_RUNTIME_LORA_UPDATING=${LORA_ON:+True} nohup "$VENV/bin/vllm" serve "$MODEL_DIR" \
      --served-model-name "$NAME" --host 127.0.0.1 --port "$port" \
      --tensor-parallel-size "$TP" --max-model-len "$MAXLEN" --max-num-seqs "${QWEN_MAX_SEQS:-32}" \
      --gpu-memory-utilization 0.90 --enable-prefix-caching \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
      --mm-encoder-tp-mode data \
      --default-chat-template-kwargs "$TEMPLATE_KW" \
      "${extra[@]}" > "$W/out/vllm_${port}_${variant}.log" 2>&1 &
    echo $! > "$W/run/vllm_${port}.pid"
    ln -sf "$W/out/vllm_${port}_${variant}.log" "$W/out/vllm_${port}.log"
  done
}

healthy() {  # healthy <timeout_s>
  local t0=$(date +%s) ok
  while :; do
    ok=1
    for port in "${PORTS[@]}"; do curl -s -m 3 -o /dev/null "http://127.0.0.1:$port/health" || ok=0; done
    [ $ok = 1 ] && return 0
    for port in "${PORTS[@]}"; do
      if ! kill -0 "$(cat "$W/run/vllm_${port}.pid" 2>/dev/null)" 2>/dev/null; then return 1; fi   # a server died
    done
    [ $(( $(date +%s) - t0 )) -gt "$1" ] && return 1
    sleep 10
  done
}

stop_servers
VARIANTS="full nomtp plain"; [ -n "${LORA_ON}" ] && VARIANTS="nomtp plain"
for variant in $VARIANTS; do
  echo "launch variant=$variant $(date -u +%FT%TZ)" >> $W/out/serve.status
  launch "$variant"
  if healthy 2400; then
    echo "ready $variant effort=$EFFORT $(date -u +%FT%TZ)" >> $W/out/serve.status
    # one completion to confirm the chat template + reasoning parser work
    curl -s -m 600 "http://127.0.0.1:${PORTS[0]}/v1/chat/completions" -H 'content-type: application/json' \
      -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word ready.\"}],\"max_tokens\":512}" \
      > $W/out/serve_probe.json 2>&1
    head -c 1500 $W/out/serve_probe.json >> $W/out/serve.status; echo >> $W/out/serve.status
    exit 0
  fi
  echo "variant=$variant failed $(date -u +%FT%TZ)" >> $W/out/serve.status
  for port in "${PORTS[@]}"; do tail -40 "$W/out/vllm_${port}_${variant}.log" >> $W/out/serve.status; done
  stop_servers
done
echo "failed $(date -u +%FT%TZ)" >> $W/out/serve.status
exit 1
