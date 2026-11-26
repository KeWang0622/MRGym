# MR Gym on verl

LoRA + GRPO training of an open-weight model on MR Gym episodes with [verl](https://github.com/verl-project/verl),
pinned at commit `8357f1e0a7ce4c06dc970c2c637d49b33815f5cc` (verl 0.10.0.dev, main of 2026-11-24). The environment
runs in the trainer's own agent-loop workers: no sandbox, no HTTP between the model and the scanner.

**State.** The first runs on 8-GPU nodes (calibration, training, readout) are reported in
[docs/experiments/RESULTS-27b.md](../../docs/experiments/RESULTS-27b.md); the steps and one settings file are in
[docs/RECIPE.md](../../docs/RECIPE.md). The rest of this paragraph and the last section of this page were written before
those runs. The agent loop is a port of an earlier adapter that trained on a
node (version 0 of the environments, a 9B model; it stopped for lack of reward); this version was run with verl's
own code and a real tokenizer on CPU and, end to end, against stand-in servers. What ran, with commands and outputs:
[SIMULATION.md](SIMULATION.md). What only a GPU can answer: the last section of this page.

## What is here

| File | What it is |
|---|---|
| `mrgym_verl/episode.py` | One episode, turn by turn, for whoever owns the model call. It is `mrgym.agents.runner.play_model` cut at the model call (a test compares the two). Writes records in the runner's schema. No trainer imported |
| `mrgym_verl/agent_loop.py` | verl `AgentLoopBase` subclass, registered as `mrgym_agent`: token-in / token-out, mask 1 on sampled tokens only |
| `mrgym_verl/instances.py` | Dataset rows (`train` placeholders, `dev`, `test`), and the curriculum hook (`mrgym.curriculum.RowHook` with a state file; a named Ray actor inside verl) |
| `mrgym_verl/checkpoint.py`, `store.py` | Checkpoint callback: every `global_step_N` leaves the node before `LATEST` is written; `s3://` or a directory |
| `mrgym_verl/stoprules.py` | The per-episode log, the per-step table and the stop rules (sequential test on the dev instances, no-reward stop, ...) |
| `mrgym_verl/readout.py` | Final readout: trained minus base, paired on the same held-out instances (`mrgym.stats`) |
| `mrgym_verl/calsum.py` | Summary of a calibration: the cells in the 20 - 60 % band, episode lengths and cost |
| `mrgym_verl/smokecheck.py` | One-line cause of a failed trainer run (out of memory on load / on the first backward, parser mismatch, context overflow) |
| `mrgym_verl/dryrun.py` | The adapter with verl's own classes and the model's tokenizer on CPU (the "CPU gate" of the node scripts) |
| `mrgym_verl/sim.py`, `mixstub.py` | GPU-free stand-ins for the trainer and the model server, for simulating the node scripts |
| `overrides.sh`, `launch.sh`, `config/agent_loops.yaml` | The Hydra overrides, the launcher (restore, dataset, hook, then verl), the loop's options |
| `node/calibrate.sh`, `node/train_smoke.sh`, `node/train.sh`, `node/lib.sh`, `node/serve.sh` | Entry scripts for one 8-GPU node. Parameters are environment variables, documented in one block at the top of each |

## Reward, limits, splits

- **Reward.** The trainer optimises the environment's paid reward: 1 when the delivered result passes, else 0
  (`signal: pass`, the default). Nothing is mixed in. `signal: progress` trains on `info["score"]["progress"]`
  instead. Progress is a diagnostic that the environment never pays; a run that uses it is a different experiment
  and every record of it says `training.signal_mode = "progress"` next to `training.paid_reward`.
- **Limits.** No turn cap. An episode ends at the terminal tool, at the environment's own tool-call or rejected-call
  limit, or after three replies without a tool call. A token stream adds two ends that a chat harness does not have:
  a reply cut at `max_tokens_per_turn` before its end-of-turn token, and a response that no longer fits
  `data.max_response_length`. Both are told to the environment (outcome `no_submission_length`), counted, and stop the
  run when they reach 20 % of the episodes. `max_tokens_per_turn` is stated in the task text.
- **Difference to `mrgym run-agent` / `mrgym calibrate`.** The chat harness continues after a cut reply (it tells the
  model); a token stream cannot. And a served endpoint with a reasoning parser drops earlier reasoning from the
  context, while the trainer's stream keeps every sampled token. Episodes are therefore longer in training than in
  calibration. The first ten steps of a run are compared against the calibrated base rate for exactly this reason
  (stop rule `harness`).
- **Splits.** Training: the public `train` split, a fresh seed per group from 10,000. In-training evaluation: the
  public `dev` split. Final readout: the secret split `test-iid`; the node reads the key from a file
  (`MRGYM_SECRET_FILE`, mode 0600, fetched by `train.sh` from `SECRET_URI`), never from a command line. Calibration:
  `train` seeds from 9,000.
- **Records.** Every episode - training, dev, test - is stored as `record.json`, `messages.json` (the whole
  conversation with the raw sampled text of every reply) and `scanner_log.jsonl` under
  `<records>/<phase>/step_<N>/episodes/...`, in the layout and schema of `mrgym run-agent`, plus a `training` block
  (signal, paid reward, outcome, per-step `info["step"]`, trainer step, group uid, attempt). One line per episode
  goes to `<records>/episodes.jsonl`; the stop rules and the readout read only that file.

## Run

```
# on a node, inside verl's environment (node/train.sh does all of this):
TRAIN_ARMS="Scanner/SNRSpec-v0:1 Scanner/NoWrap-v1:2,3" MODEL_PATH=/models/M RUN_NAME=r1 \
MRGYM_CKPT_DEST=<s3 prefix or folder>  bash integrations/verl/launch.sh

# anywhere, without a GPU: the same launcher with stand-ins
MRGYM_STUB=1 ... bash integrations/verl/launch.sh
```

The node scripts take the repository as `REPO_SRC` (an archive in the output bucket, a git URL with a ref, or a
folder) and write everything under `OUT_URI/runs/RUN_ID/`. No bucket, job or project name is written in this
repository; whoever launches passes them. `node/make_main.sh <entry> <env file> <out>` writes the parameters of one
launch into a copy of an entry script, for a job system that takes a single script.

Each script is resumable under the same `RUN_ID` until it has written `DONE.txt` or `FAILED.txt`; after that the run
id does nothing. Before it plays an episode, `calibrate.sh` checks that every port answers as its own server (the
served model name; a nonce for the stand-in), so that a server left by another run cannot answer for it. Several
`mrgym calibrate` processes share one records folder. A calibration can continue from the records of an earlier one
with a larger `CAL_N` or more cells (`CAL_FROM_URI`).

The whole of it without a GPU: `bash integrations/verl/simulate.sh <empty folder> <python>` (stand-in servers on
free ports, a stand-in trainer, a directory for the bucket).

| Script | Output under `OUT_URI/runs/RUN_ID/` |
|---|---|
| `calibrate.sh` | `records/cal/episodes/...`, `calibration_summary.json`, `calibration_summary.md`, `train_arms.json`, `logs/`, `DONE.txt` or `FAILED.txt` |
| `train_smoke.sh` | `smoke_report.json` (seconds per step, token lengths, outcomes), `gpu_mem_peak.txt`, `run_a.log`, `run_b.log`, `dryrun.log`, `attempts/<start>/{ckpt,records}`, `DONE.txt` (`SMOKE PASS`) or `FAILED.txt` (`smoke: <stage>: <reason>`) |
| `train.sh` | `remote/ckpt/{LATEST,global_step_N}`, `remote/records/`, `remote/metrics.{json,txt}`, `remote/STOP.json` (when a rule stopped it), `remote/test_base/`, `remote/test_trained/`, `readout.{json,txt}`, `DONE.txt` |

## Decisions for a 27B model on one node (on paper)

The numbers are computed by `python docs/experiments/27b/compute.py` from `docs/experiments/27b/plan.json`; the tables
are in [docs/experiments/PREREG-27b.md](../../docs/experiments/PREREG-27b.md) sections 5 and 8.

- **Layout.** verl at the pinned commit runs rollout and training colocated on the same GPUs and alternates them
  (`hybrid_engine`, `trainer_mode=sync`). Two vLLM replicas of four GPUs each generate; with a LoRA adapter verl puts
  them to sleep at level 1 for the update (weights to host memory, cache released) and wakes them with the new
  adapter. They do not have to share memory at the same moment, and they cannot be run side by side: 55.6 GB of
  weights in bf16 are 13.9 GB per GPU for a replica and 7.0 GB per GPU for the 8-way FSDP shard of the actor.
- **Adapter.** LoRA rank 32, alpha 64, on every linear layer of the language model; the vision tower is excluded.
  Checkpoints hold the adapter and the optimizer only.
- **What limits the node is the length of one episode in the update**, not the weights: per token the actor keeps
  the inputs of 64 layers (gradient checkpointing) and logits over a 248,320-entry vocabulary. On 80 GB cards the
  plan's 32,768-token episode lies between the optimistic and the pessimistic estimate; on 141 GB and 180 GB cards
  it fits on paper with a factor of at least 1.6. Sequence parallelism in the actor
  (`EXTRA_OVERRIDES=actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=N`) splits one episode over N
  GPUs; it has not been run by us.
- **Sequence lengths.** With the Qwen3.8-27B tokenizer the first prompt (session note, task, tool schemas) is 5,670
  tokens on `Scanner/NoWrap-v1` and each observation about 460 (measured on CPU, SIMULATION.md section 4; the table
  there lists every cell). What the model writes is not known for the v1 environments: the calibration job reports
  it per cell (`stream_tokens_est`), and `MAX_RESPONSE_LEN` is set from its 99th percentile.

## What only a GPU run can answer (and which script answers it)

| Question | Answered by |
|---|---|
| Pass rate of the 27B per (environment, level) on v1; which cells are in the 20 - 60 % band; tokens and GPU-seconds per episode | `calibrate.sh` |
| Does the 27B load in this layout (two TP-4 replicas at 0.60 + the actor) | `train_smoke.sh` stage 3; `FAIL OOM on load` otherwise |
| Does one update at the planned sequence length fit an 80 GB card; peak memory per GPU | stage 3, `gpu_mem_peak.txt`; `FAIL OOM on the first backward` otherwise |
| Does verl's `qwen3_coder` parser read what the served 27B writes (on CPU it read scripted turns of that format) | stage 3; `FAIL tool-call parser mismatch` when more than half of the episodes make no readable call |
| How long real episodes are as one token stream; share cut by the limits | `smoke_report.json`; `FAIL context overflow` at 20 % |
| Is the adapter pushed into the replicas before an evaluation that follows a restore | stage 5 logs the evaluation after the restore; whether its numbers reflect the adapter is not decidable in three steps |
| Seconds per step; throughput with a LoRA adapter loaded (no speculative decoding) | `timings.txt` of the smoke; the first ten steps of `train.sh` |
| Does the agent loop receive the trainer's step number and group uid (read from verl's source, not run) | stage 3; `FAIL step labels` otherwise |
