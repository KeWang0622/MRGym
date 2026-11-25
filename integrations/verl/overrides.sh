#!/usr/bin/env bash
# Hydra overrides of an MR Gym run on verl (pinned 8357f1e), as one bash array (OVERRIDES). Sourced by launch.sh and
# by the CPU dry run (which composes them against verl's own config tree to catch a wrong key before a node is used).
#
# Layout: one node, 8 GPUs, verl's colocated synchronous trainer (trainer.use_v1, trainer_mode=sync, hybrid_engine).
# Rollout and training share the GPUs and alternate: vLLM replicas generate one batch of episodes, release their
# cache (free_cache_engine), the FSDP2 actor trains, the LoRA adapter is pushed back into the replicas, next batch.
#
# Inputs (environment variables; defaults in the right column; the 27B values are in docs/experiments/27b/overrides.env)
#   MODEL_PATH           HF id or local folder                                  Qwen/Qwen3.5-9B
#   TOOL_FORMAT          verl tool parser                                       qwen3_coder (Qwen3.5 / 3.8; hermes for Qwen3)
#   CHAT_KWARGS          chat-template switches, "k=v k=v"                      enable_thinking=False
#   ROLLOUT_TP           GPUs per vLLM replica                                  1      (27B on 80 GB cards: 4)
#   GPU_MEM_UTIL         vLLM share of each GPU                                 0.45
#   PROMPTS_PER_STEP     groups (instances) per step                            16
#   GROUP_SIZE           rollouts per instance                                  8
#   MAX_TOKENS_PER_TURN  cap of one reply; written into agent_loops.yaml        16384
#   TOOL_CALL_LIMIT      empty = the environment's own limit                    (empty)
#   SIGNAL               pass | progress (see config/agent_loops.yaml)          pass
#   MAX_PROMPT_LEN       tokens of system + task + tools                        8192
#   MAX_RESPONSE_LEN     tokens after the first prompt                          49152
#   TOKENS_PER_GPU       actor tokens per GPU and micro-batch                   MAX_PROMPT_LEN + MAX_RESPONSE_LEN + 2048
#   LORA_RANK / LORA_ALPHA                                                      32 / 64
#   LR / WARMUP_STEPS / GRAD_CLIP                                               1e-5 / 5 / 1.0
#   KL_COEF              0 switches the KL term off                             0
#   TOTAL_STEPS, SAVE_FREQ, TEST_FREQ                                           100, 5, 10
#   VAL_N                episodes per evaluation instance                       2
#   VAL_FILE             dev | test   (file of DATA_DIR used for evaluation)    dev
#   VAL_ONLY             1 = evaluate and exit (final readout)                  0
#   VAL_BEFORE_TRAIN     1 = evaluate before the first step (the base rate)     1
#   RESUME_MODE          auto | disable                                         auto
#   AGENT_WORKERS        agent-loop worker processes                            8
#   EXTRA_OVERRIDES      further Hydra overrides, space separated (e.g. the sequence-parallel size of the actor)
#   DATA_DIR, CKPT_DIR, RECORD_DIR (default: under ${WORK:-$PWD}/rl/RUN_NAME), RUN_NAME, NGPUS, ADAPTER_DIR (this folder), LOOP_CONFIG

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}
TOOL_FORMAT=${TOOL_FORMAT:-qwen3_coder}
CHAT_KWARGS=${CHAT_KWARGS:-enable_thinking=False}
ROLLOUT_TP=${ROLLOUT_TP:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.45}
PROMPTS_PER_STEP=${PROMPTS_PER_STEP:-16}
GROUP_SIZE=${GROUP_SIZE:-8}
MAX_TOKENS_PER_TURN=${MAX_TOKENS_PER_TURN:-16384}
TOOL_CALL_LIMIT=${TOOL_CALL_LIMIT:-}
SIGNAL=${SIGNAL:-pass}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-8192}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-49152}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LR=${LR:-1e-5}
WARMUP_STEPS=${WARMUP_STEPS:-5}
GRAD_CLIP=${GRAD_CLIP:-1.0}
KL_COEF=${KL_COEF:-0}
TOTAL_STEPS=${TOTAL_STEPS:-100}
SAVE_FREQ=${SAVE_FREQ:-5}
TEST_FREQ=${TEST_FREQ:-10}
VAL_N=${VAL_N:-2}
VAL_FILE=${VAL_FILE:-dev}
VAL_ONLY=${VAL_ONLY:-0}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-1}
RESUME_MODE=${RESUME_MODE:-auto}
AGENT_WORKERS=${AGENT_WORKERS:-8}
NGPUS=${NGPUS:-8}
RUN_NAME=${RUN_NAME:-run}
ADAPTER_DIR=${ADAPTER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
DATA_DIR=${DATA_DIR:-${WORK:-$PWD}/rl/${RUN_NAME}/data}
CKPT_DIR=${CKPT_DIR:-${WORK:-$PWD}/rl/${RUN_NAME}/ckpt}
RECORD_DIR=${RECORD_DIR:-${WORK:-$PWD}/rl/${RUN_NAME}/records}
LOOP_CONFIG=${LOOP_CONFIG:-${CKPT_DIR}/agent_loops.yaml}
MAX_MODEL_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))
TOKENS_PER_GPU=${TOKENS_PER_GPU:-$((MAX_MODEL_LEN + 2048))}
USE_KL=$([ "${KL_COEF}" = "0" ] && echo False || echo True)
_bool() { [ "$1" = 1 ] && echo True || echo False; }

OVERRIDES=(
    # ---- algorithm: GRPO over groups of GROUP_SIZE rollouts of one instance; a group without spread gets advantage 0
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=True
    algorithm.use_kl_in_reward=False
    algorithm.rollout_correction.bypass_mode=True          # old policy = the sampler's own log-probs
    algorithm.filter_groups.enable=False                   # flat groups are counted from the episode log, not refilled
    # ---- data: one row per group; the agent loop builds the prompt
    data.train_files="['${DATA_DIR}/train.parquet']"
    data.val_files="['${DATA_DIR}/${VAL_FILE}.parquet']"
    data.train_batch_size=${PROMPTS_PER_STEP}
    data.max_prompt_length=${MAX_PROMPT_LEN}
    data.max_response_length=${MAX_RESPONSE_LEN}
    data.return_raw_chat=True
    data.filter_overlong_prompts=False
    data.truncation=error
    data.shuffle=False                                      # training rows are placeholders; the curriculum picks the instance
    # ---- model: LoRA on the language model, vision tower left alone
    actor_rollout_ref.hybrid_engine=True
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.lora_rank=${LORA_RANK}
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    actor_rollout_ref.model.target_modules=all-linear
    "actor_rollout_ref.model.exclude_modules='.*visual.*'"
    # ---- actor: one optimizer step per rollout batch
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=${LR}
    actor_rollout_ref.actor.optim.lr_warmup_steps=${WARMUP_STEPS}
    actor_rollout_ref.actor.optim.weight_decay=0.0
    actor_rollout_ref.actor.grad_clip=${GRAD_CLIP}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PROMPTS_PER_STEP}
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${TOKENS_PER_GPU}
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.use_kl_loss=${USE_KL}
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.use_torch_compile=False
    +actor_rollout_ref.actor.checkpoint.save_lora_only=True   # adapter + optimizer per checkpoint, not the base model
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${NGPUS}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    # ---- rollout: vLLM replicas on the same GPUs, token-level generate, sampler log-probs kept
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.n=${GROUP_SIZE}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${TOKENS_PER_GPU}
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0   # evaluation samples like training does: pass rates, not a greedy path
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=null   # no turn cap: the environment's tool-call limit ends an episode
    actor_rollout_ref.rollout.multi_turn.format=${TOOL_FORMAT}
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_WORKERS}
    actor_rollout_ref.rollout.agent.default_agent_loop=mrgym_agent
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${LOOP_CONFIG}"
    # ---- trainer: synchronous, colocated, resumable
    trainer.use_v1=True
    trainer.v1.trainer_mode=sync
    trainer.nnodes=1
    trainer.n_gpus_per_node=${NGPUS}
    trainer.project_name=runs
    trainer.experiment_name="${RUN_NAME}"
    "trainer.logger=['console']"
    trainer.total_epochs=1000
    trainer.total_training_steps=${TOTAL_STEPS}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.val_before_train=$(_bool "${VAL_BEFORE_TRAIN}")
    trainer.val_only=$(_bool "${VAL_ONLY}")
    trainer.resume_mode=${RESUME_MODE}
    trainer.default_local_dir="${CKPT_DIR}"
    trainer.max_actor_ckpt_to_keep=3
    trainer.checkpoint_callback_class=mrgym_verl.checkpoint.SyncCheckpoint
)
for kv in ${CHAT_KWARGS}; do OVERRIDES+=("+data.apply_chat_template_kwargs.${kv}"); done
for kv in ${EXTRA_OVERRIDES:-}; do OVERRIDES+=("${kv}"); done
