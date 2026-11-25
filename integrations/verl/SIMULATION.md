# Local simulation of the verl integration

Everything on this page ran on a CPU-only machine at commit `fe6a952` of this branch, with no GPU, no model and no
bucket. It shows that the glue works: argument handling, the adapter's episode loop and reward, records, uploads,
resume after a kill, stop rules, the readout. **It shows nothing about any model**: the "model" is a stand-in that
replays control policies of the environments, and every pass rate below is a property of that stand-in.

Versions: {"verl": "0.10.0.dev", "torch": "2.13.0+cpu", "transformers": "5.16.1", "mrgym": "0.1.0.dev0", "engine": "mrgym-engine-0.2.0"}.

## 1. What stands in for what

| On a node | In the simulation (`MRGYM_STUB=1`) |
|---|---|
| vLLM serving the model | `mrgym_verl.mixstub`: the package's stub endpoint plus model ids `mix:<p>:<q>` that follow the reference policy with probability p, the no-scan answer otherwise, and never call a tool with probability q |
| `python -m verl.trainer.main_ppo` | `mrgym_verl.sim`: the same inputs and files; a one-parameter policy (p) updated by REINFORCE with verl's GRPO advantage on the real reward of real episodes |
| The output bucket | a directory; the same code (`mrgym_verl/store.py`) with another scheme |
| Building the environments with uv, downloading weights, the GPU check | skipped (the scripts say so in their log) |
| A preemption | `SIM_DIE_AT=calibrate` / `SIM_DIE_AT_STEP=N`: SIGKILL of the job and all its children, no marker, no final upload; then the work disk is deleted |

Not skipped: the bootstrap, the parameter handling, the run-id markers, the self-test of `calibrate.sh`, the package's
own `mrgym calibrate`, the mirror to the output folder, `launch.sh` (rows, hook configuration, restore), the adapter's
`Episode` loop and records, the stop-rule watcher, the secret file for the held-out split, the readout.

## 2. The three node scripts, end to end

Command: `bash integrations/verl/simulate.sh <work> <python>` (at most two worker processes; the stand-in server
spends most of the time building its table of candidate instances). Output, with paths shortened:

```
## calibrate.sh: killed in the middle of the plan, started again on a fresh node
$ env RUN_ID=c01 MODEL_NAME=mix:0.45:0.10 CAL_PLAN=Scanner/NoWrap-v1:2,3 Scanner/SNRSpec-v0:1 CAL_N=8 CAL_K=4 CAL_WORKERS=1 CAL_PROCS=2 STUB_SERVERS=1 SIM_DIE_AT=calibrate SIM_DIE_AFTER_S=25 bash integrations/verl/node/calibrate.sh
integrations/verl/simulate.sh: line 19: 1080886 Killed                  env -i PATH="$PATH" HOME="$HOME" "${COMMON[@]}" "$@" bash "$NODE/$script" > "$logf" 2>&1
exit code 137
self-test ok: 4 episodes, pass 1.0 engine mrgym-engine-0.2.0
[09:21:22Z] records already present: 0
servers ready (ports 41413) 2026-11-25T09:21:23Z
[09:21:48Z] SIM_DIE_AT=calibrate: killing the job (simulated preemption; no marker, no final upload)
episode records in the output folder after the kill: 20  (DONE.txt: no)
$ rm -rf <work disk of the node>     # a fresh node
$ env RUN_ID=c01 MODEL_NAME=mix:0.45:0.10 CAL_PLAN=Scanner/NoWrap-v1:2,3 Scanner/SNRSpec-v0:1 CAL_N=8 CAL_K=4 CAL_WORKERS=1 CAL_PROCS=2 STUB_SERVERS=1 bash integrations/verl/node/calibrate.sh
exit code 0
[09:21:55Z] records already present: 20
pass 1 of the plan: 96 record(s), summary rc=0 2026-11-25T09:23:14Z
done: calibration complete: 96 episode record(s); cells in rule: Scanner/NoWrap-v1:2,3 Scanner/SNRSpec-v0:1 2026-11-25T09:23:14Z
--- <work>/out/runs/c01/calibration_summary.md
Calibration of `mix:0.45:0.10`: 96 episodes, seeds 9000-9007 x 4 per instance

| Environment | Level | Episodes | Pass | 95 % interval | Mixed groups | Band | In rule | Calls as written | Submit | Completion tokens p50 / p99 | Stream estimate p50 / p99 / max |
|---|---:|---:|---:|---|---:|---|---|---:|---:|---|---|
| Scanner/NoWrap-v1 | 2 | 32 | 53.1 % | 40.6 .. 65.6 | 100 % | entry | yes | 100.0 % | 94 % | 100 / 101 | 6055 / 6056 / 6056 |
| Scanner/NoWrap-v1 | 3 | 32 | 43.8 % | 25.0 .. 62.5 | 88 % | entry | yes | 100.0 % | 91 % | 45 / 102 | 5367 / 6057 / 6057 |
| Scanner/SNRSpec-v0 | 1 | 32 | 53.1 % | 37.5 .. 68.8 | 88 % | entry | yes | 100.0 % | 97 % | 992 / 998 | 13830 / 13850 / 13850 |

Rule: pass rate in [0.2, 0.6] and at least 80% of the groups of 4 mixed.
Cells that meet it (`TRAIN_ARMS`): `Scanner/NoWrap-v1:2,3 Scanner/SNRSpec-v0:1`; their mean pass rate (`BASE_RATE`): 0.5
192 episode-seconds in 2 parallel slots on 8 GPUs: 8.0 GPU-seconds per episode (slots assumed busy; idle tails make it an under-estimate).
--- train_arms.json
{'train_arms': 'Scanner/NoWrap-v1:2,3 Scanner/SNRSpec-v0:1', 'base_rate': 0.5, 'complete': True}
$ env RUN_ID=c01 MODEL_NAME=mix:0.45:0.10 CAL_PLAN=Scanner/NoWrap-v1:2,3 Scanner/SNRSpec-v0:1 CAL_N=8 CAL_K=4 CAL_WORKERS=1 CAL_PROCS=2 STUB_SERVERS=1 bash integrations/verl/node/calibrate.sh
exit code 0
[09:23:15Z] <work>/out/runs/c01/DONE.txt exists: 2026-11-25T09:23:14Z calibration complete: 96 episode record(s); cells in rule: Scanner/NoWrap-v1:2,3 Scanner/SNRSpec-v0:1. Nothing 

## train_smoke.sh
$ env RUN_ID=t01 TRAIN_ARMS=Scanner/NoWrap-v1:2,3 SMOKE_STEPS=2 PROMPTS_PER_STEP=2 GROUP_SIZE=4 bash integrations/verl/node/train_smoke.sh
exit code 0
[09:23:15Z] === SMOKE 0 node ===
[09:23:15Z] === SMOKE 1 install ===
[09:23:16Z] === SMOKE 2 the adapter on CPU with the model's tokenizer (verl's own code, no GPU) ===
[09:23:16Z] === SMOKE 3 run A: serve, evaluate, 2 steps of 2 x 4 rollouts, a checkpoint per step ===
OK ok | {"episodes": 16, "steps_logged": [0, 1, 2], "seconds_per_step": [1.4, 1.9], "pass_rate": 0.375, "no_tool_call_share": 0.0625, "interface_tax": 0.0, "length_cut_share": 0.0, "tokens_out_mean": 64, "context_tokens_max": 0, "
[09:23:47Z] === SMOKE 4 the checkpoint left the node ===
[09:23:47Z] === SMOKE 5 preemption: local disk lost, same command again; evaluation with the restored adapter, one more step ===
stage 5 preemption: local disk lost, same command again; evaluation with the restored adapter, one more step 2026-11-25T09:23:47Z
OK ok | {"episodes": 24, "steps_logged": [2, 3], "seconds_per_step": [1.7], "pass_rate": 0.3333, "no_tool_call_share": 0.0833, "interface_tax": 0.0, "length_cut_share": 0.0, "tokens_out_mean": 62, "context_tokens_max": 0, "step_so
[09:23:53Z] === SMOKE 6 the stored episodes: complete records, readable log, the evaluation after the restore ===
stage 6 the stored episodes: complete records, readable log, the evaluation after the restore 2026-11-25T09:23:53Z
step:1  timing_s/step:1.4 step:2  timing_s/step:1.9 step:3  timing_s/step:1.7 
SMOKE PASS
done: SMOKE PASS: 3 steps, checkpoint restored from <work>/out/runs/t01/attempts/20261125T092316Z/ckpt, 44 episodes recorded 2026-11-25T09:23:55Z
[mrgym] restoring global_step_2 from <work>/node/job_t01/rl/remote/ckpt
Resuming from <work>/node/job_t01/rl/smoke/ckpt/global_step_2, setting global step to 2
step:2 - val-core/mrgym/Scanner/NoWrap-v1/L2/dev/acc/mean@1:0.5000 - val-core/mrgym/Scanner/NoWrap-v1/L3/dev/acc/mean@1:0.5000
step:3 - critic/score/mean:0.2500 - mrgym/pass:0.2500 - mrgym/groups_without_spread:1 - actor/grad_norm:0.2165 - sim/p:0.6006 - timing_s/step:1.7
step:3 - val-core/mrgym/Scanner/NoWrap-v1/L2/dev/acc/mean@1:1.0000 - val-core/mrgym/Scanner/NoWrap-v1/L3/dev/acc/mean@1:0.5000
--- smoke_report.json (head)
{'episodes': 44, 'records': 44, 'attempts': ['20261125T092316Z-1086965', '20261125T092347Z-1088060'], 'evaluation_after_restore': 4, 'outcomes': {'no_measurement': 19, 'no_submission_stopped': 3, 'pass': 22}}
--- a failing smoke names its cause: the same script when every first reply is cut by the token limit (SIM_FORCE_CUT=1)
$ env RUN_ID=t02 TRAIN_ARMS=Scanner/NoWrap-v1:2,3 SMOKE_STEPS=1 PROMPTS_PER_STEP=1 GROUP_SIZE=4 SIM_FORCE_CUT=1 bash integrations/verl/node/train_smoke.sh
exit code 1
SMOKE FAIL 3: FAIL context overflow: 100 % of the episodes were cut at the reply limit or by the response length (longest stream 0 tokens); raise MAX_RESPONSE_LEN / MAX_TOKENS_PER_TURN 
FAILED smoke: 3: FAIL context overflow: 100 % of the episodes were cut at the reply limit or by the response length (longest stream 0 tokens); raise MAX_RESPONSE_LEN / MAX_TOKENS_PER_TURN  2026-11-25T09:24:00Z

## train.sh: killed after the rollouts of step 3, started again on a fresh node, read out on the secret split
$ env RUN_ID=u01 TRAIN_ARMS=Scanner/NoWrap-v1:2,3 BASE_RATE=0.45 TEST_ARMS=Scanner/NoWrap-v1:2,3 Scanner/SNRSpec-v0:1 UNTRAINED=Scanner/SNRSpec-v0 TEST_SEEDS=0-5 SECRET_URI=<work>/secret.key TOTAL_STEPS=6 SAVE_FREQ=2 TEST_FREQ=2 PROMPTS_PER_STEP=4 GROUP_SIZE=4 DEV_SEEDS=0-7 VAL_N=2 SIM_LR=3.0 SIM_IDLE_SHARE=0.05 NO_REWARD_STEPS=2 HARNESS_STEPS=3 MIXED_MIN=0 WINDOW=2 COLLAPSE_STEPS=3 MIN_ENVS=1 SIM_DIE_AT_STEP=3 bash integrations/verl/node/train.sh
integrations/verl/simulate.sh: line 19: 1091075 Killed                  env -i PATH="$PATH" HOME="$HOME" "${COMMON[@]}" "$@" bash "$NODE/$script" > "$logf" 2>&1
exit code 137
[09:24:01Z] === TRAIN 0 node ===
[09:24:01Z] === TRAIN 1 install ===
held-out split opens: Scanner/NoWrap-v1 level 2
[09:24:03Z] === TRAIN 2 the adapter on CPU with the model's tokenizer ===
[09:24:03Z] mirror: 0 episode(s) logged so far; newest checkpoint: none
[09:24:04Z] === TRAIN 3 training: 4 groups x 4 rollouts per step, 6 steps ===
[09:26:57Z] SIM_DIE_AT=trainer: killing the job (simulated preemption; no marker, no final upload)
in the output folder after the kill: LATEST = {"global_step": 2, "folder": "global_step_2", "files": 3, "utc": "2026-11-25T09:26:47Z"}; 104 episode line(s)
$ rm -rf <work disk of the node>     # a fresh node
$ env RUN_ID=u01 TRAIN_ARMS=Scanner/NoWrap-v1:2,3 BASE_RATE=0.45 TEST_ARMS=Scanner/NoWrap-v1:2,3 Scanner/SNRSpec-v0:1 UNTRAINED=Scanner/SNRSpec-v0 TEST_SEEDS=0-5 SECRET_URI=<work>/secret.key TOTAL_STEPS=6 SAVE_FREQ=2 TEST_FREQ=2 PROMPTS_PER_STEP=4 GROUP_SIZE=4 DEV_SEEDS=0-7 VAL_N=2 SIM_LR=3.0 SIM_IDLE_SHARE=0.05 NO_REWARD_STEPS=2 HARNESS_STEPS=3 MIXED_MIN=0 WINDOW=2 COLLAPSE_STEPS=3 MIN_ENVS=1 bash integrations/verl/node/train.sh
exit code 0
[09:26:57Z] === TRAIN 0 node ===
[09:26:57Z] === TRAIN 1 install ===
[09:26:59Z] === TRAIN 2 the adapter on CPU with the model's tokenizer ===
[09:26:59Z] mirror: 104 episode(s) logged so far; newest checkpoint: {"global_step": 2, "folder": "global_step_2", "files": 3, "utc": "2026-11-25T09:26:47Z"}
[09:27:00Z] === TRAIN 3 training: 4 groups x 4 rollouts per step, 6 steps ===
training reached step 6 2026-11-25T09:30:09Z
[09:30:10Z] === TRAIN 4 readout on the held-out split: the base model and the final checkpoint, same instances, same harness ===
done: TRAIN DONE: 264 training and evaluation episodes; verdict: 1 trained environment(s) meet the rule (needed 1); no harm on the untrained environment: True; it trained: True 2026-11-25T09:31:13Z
[mrgym] restoring global_step_2 from <work>/node/job_u01/rl/remote/ckpt
Resuming from <work>/node/job_u01/rl/run/ckpt/global_step_2, setting global step to 2
step:2 - val-core/mrgym/Scanner/NoWrap-v1/L2/dev/acc/mean@2:0.6875 - val-core/mrgym/Scanner/NoWrap-v1/L3/dev/acc/mean@2:0.8125
step:3 - critic/score/mean:0.6875 - mrgym/pass:0.6875 - mrgym/groups_without_spread:1 - actor/grad_norm:0.3103 - sim/p:0.9300 - timing_s/step:4.8
step:4 - critic/score/mean:0.8125 - mrgym/pass:0.8125 - mrgym/groups_without_spread:2 - actor/grad_norm:0.2020 - sim/p:0.9606 - timing_s/step:4.0
step:4 - val-core/mrgym/Scanner/NoWrap-v1/L2/dev/acc/mean@2:0.8125 - val-core/mrgym/Scanner/NoWrap-v1/L3/dev/acc/mean@2:0.9375
step:5 - critic/score/mean:1.0000 - mrgym/pass:1.0000 - mrgym/groups_without_spread:4 - actor/grad_norm:0.0000 - sim/p:0.9606 - timing_s/step:5.4
step:6 - critic/score/mean:0.6250 - mrgym/pass:0.6250 - mrgym/groups_without_spread:1 - actor/grad_norm:0.3248 - sim/p:0.9847 - timing_s/step:4.2
step:6 - val-core/mrgym/Scanner/NoWrap-v1/L2/dev/acc/mean@2:1.0000 - val-core/mrgym/Scanner/NoWrap-v1/L3/dev/acc/mean@2:0.8750
--- <work>/out/runs/u01/remote/metrics.txt
step  episodes  pass   signal  mixed  submit  cut    tokens_out  context_max
   1        16  0.312  0.312   1.00   0.88    0.00          60           0
   2        16  0.750  0.750   0.75   0.88    0.00          85           0
   3        16  0.688  0.688   0.75   1.00    0.00          83           0
   4        16  0.812  0.812   0.50   0.88    0.00          88           0
   5        16  1.000  1.000   0.00   1.00    0.00         101           0
   6        16  0.625  0.625   0.75   0.81    0.00          76           0
evaluation: step  episodes  instances  pass [95 % interval]   look against the first evaluation
               0        32         16  0.312 [0.170, 0.502]
               2        32         16  0.750 [0.562, 0.875]   diff +0.438 bounds +0.178 .. +0.697 at level 0.9833  efficacy True  futility False
               4        32         16  0.875 [0.702, 0.954]   diff +0.562 bounds +0.352 .. +0.773 at level 0.9833  efficacy True  futility False
               6        32         16  0.938 [0.781, 0.984]   diff +0.625 bounds +0.425 .. +0.825 at level 0.9833  efficacy True  futility False
decision: continue
--- <work>/out/runs/u01/readout.txt
environment / level        role       base                                   trained                                paired difference (trained - base)
Scanner/NoWrap-v1          trained    37.5 % [18.4, 61.5] (24 ep, 12 inst)   91.7 % [71.0, 98.0] (24 ep, 12 inst)   +54.2 points [+27.6, +80.8], 12 instances, p = 0.0011 (paired-t)  MEETS RULE
Scanner/SNRSpec-v0         untrained  50.0 % [20.2, 79.8] (12 ep, 6 inst)    83.3 % [46.3, 96.7] (12 ep, 6 inst)    +33.3 points [-20.9, +87.5], 6 instances, p = 0.1747 (paired-t)  MEETS RULE
Scanner/NoWrap-v1 L2                  50.0 % [15.8, 84.2] (12 ep, 6 inst)    100.0 % [47.6, 100.0] (12 ep, 6 inst)  +50.0 points [+3.1, +96.9], 6 instances, p = 0.0409 (paired-t)
Scanner/NoWrap-v1 L3                  25.0 % [6.6, 61.2] (12 ep, 6 inst)     83.3 % [46.3, 96.7] (12 ep, 6 inst)    +58.3 points [+18.8, +97.8], 6 instances, p = 0.0127 (paired-t)
Scanner/SNRSpec-v0 L1                 50.0 % [20.2, 79.8] (12 ep, 6 inst)    83.3 % [46.3, 96.7] (12 ep, 6 inst)    +33.3 points [-20.9, +87.5], 6 instances, p = 0.1747 (paired-t)
verdict: 1 trained environment(s) meet the rule (needed 1); no harm on the untrained environment: True; it trained: True
--- the secret is in no uploaded file: 0 file(s) contain it

## train.sh on a policy that never passes: the no-reward rule stops it
$ env RUN_ID=u02 TRAIN_ARMS=Scanner/NoWrap-v1:2,3 TOTAL_STEPS=8 SAVE_FREQ=2 TEST_FREQ=4 PROMPTS_PER_STEP=2 GROUP_SIZE=4 DEV_SEEDS=0-1 VAL_N=1 SIM_THETA0=-20 NO_REWARD_STEPS=3 bash integrations/verl/node/train.sh
exit code 0
[09:32:31Z] stopped by a rule: {"rule": "no_reward", "reason": "no reward: mean training pass rate of the first 3 steps is 0.0000 (< 0.02); per step [0.0, 0.0, 0.0]", "utc": "2026-11-25T09:32:30Z"}
done: STOPPED by rule no_reward: no reward: mean training pass rate of the first 3 steps is 0.0000 (< 0.02); per step [0.0, 0.0, 0.0] - the held-out split was not opened 2026-11-25T09:32:32Z
--- STOP.json
{"rule": "no_reward", "reason": "no reward: mean training pass rate of the first 3 steps is 0.0000 (< 0.02); per step [0.0, 0.0, 0.0]", "utc": "2026-11-25T09:32:30Z"}

processes left behind: 0
```

What to read in it.

- `calibrate.sh`: the first start is killed in the middle of the plan; the second, on an empty work disk, finds the
  uploaded records, plays only the missing episodes, writes the summary and the cells in the rule; a third start of
  the same run id does nothing.
- `train_smoke.sh`: `SMOKE PASS` after two steps, a checkpoint that left the node, a wiped local checkpoint folder, a
  restore, an evaluation after the restore and a third step; the second call shows the one-line cause of a failure.
- `train.sh`: killed after the rollouts of step 3; the second start restores step 2, replays step 3 (the per-step
  table counts it once), finishes, evaluates the base policy and the final checkpoint on the same instances of the
  secret split and prints the paired readout. The stand-in policy has one parameter for every environment, so the
  "untrained" environment moves with the trained one here: the lines show the arithmetic of the readout only.
- The last part: a policy that never passes is stopped by the no-reward rule, and the job says so.

## 3. The adapter with verl's own code and the Qwen3.8-27B tokenizer, on CPU

verl (the pinned commit) imports in a CPU environment; its trainer cannot run a step there, because its rollout
engine (vLLM) needs a GPU build and is not installed in that environment. **`python -m verl.trainer.main_ppo` was
therefore not run.** What ran is every piece of verl that the adapter touches, around a scripted "server" that
returns token ids of scripted assistant turns:

```
$ set -a; source docs/experiments/27b/overrides.env; set +a            # the settings of the 27B plan
$ source integrations/verl/overrides.sh; printf '%s\n' "${OVERRIDES[@]}" > overrides.txt
$ PYTHONPATH=<repo>/src:<repo>/integrations/verl <venv r4_32_verl>/bin/python -m mrgym_verl.dryrun --tokenizer <tokenizer of Qwen/Qwen3.8-27B> \
      --model-type qwen3_5 --overrides overrides.txt --cells "Scanner/NoWrap-v1:2 Scanner/SNRSpec-v0:1" --seeds 100-101 --grad-step --ray --out <work>/verl/out
```

| Check | Result |
|---|---|
| Hydra composes verl's `ppo_trainer` config with the overrides of `overrides.sh` | ok, 84 overrides |
| Token builder verl selects for the model family | QwenContinuousTokenBuilder |
| The loop's options as built by verl | {"signal": "pass", "max_tokens_per_turn": 16384, "tool_call_limit": null, "response_length": 24576, "tool_format": "qwen3_coder"} |
| Trajectories with a length, log-prob alignment or reward mismatch | {"length": 0, "logprob_alignment": 0, "reward_vs_direct": 0} |
| Masked tokens are exactly the sampled tokens | True |
| Token stream equals a from-scratch chat-template rendering of the final conversation | 18 identical, 0 different |
| Episode records written / expected; fields missing | 26 / 26; none |
| Training rows served by the curriculum hook: one instance per group, distinct seeds, state file | {"one_instance_per_group": true, "distinct_seeds": true, "sampler_steps": 1, "seeds_issued": 2, "state_file": true} |
| Groups with reward spread (verl's GRPO advantage computed on them); advantage zero off the mask | 4/4; True |
| verl's PPO policy loss on a tiny random causal LM with a LoRA adapter (sequences cropped): loss, gradient norm, largest change of a log-prob after the step, largest difference after save and reload | -0.2611, 0.0090, 0.0137, 0.0 |
| Checkpoint callback: `LATEST` after the copy; the sampler's state inside the checkpoint | step 5, 2 files; True |
| Curriculum hook as a named Ray actor asked from worker processes: distinct worker pids, same group same instance, other group other seed | 2 worker processes (ray 2.60.1); True; True |

| Scripted behaviour | Episodes | Outcome(s) | Reward | Prompt tokens | Response tokens (max) | Loss tokens (max) | All checks |
|---|---:|---|---:|---:|---:|---:|---|
| reference | 6 | pass | 1.0 | 5,670 - 5,961 | 13,199 | 3,232 | yes |
| prior | 4 | no_measurement | 0.0 | 5,670 - 5,961 | 908 | 169 | yes |
| text_only | 4 | no_submission_stopped | 0.0 | 5,670 - 5,961 | 113 | 57 | yes |
| bad_json | 4 | pass | 1.0 | 5,670 - 5,961 | 13,329 | 3,243 | yes |
| cut | 4 | no_submission_length | 0.0 | 5,670 - 5,961 | 14 | 14 | yes |
| context_cap | 4 | no_submission_length | 0.0 | 5,670 - 5,961 | 32 | 32 | yes |

Last line of the run: `DRYRUN PASS`. Peak resident memory of the largest process of this run:
2694 MiB (the test in `tests/test_verl_adapter_verl.py` plays one cell and one seed and stays near 1,300 MiB).

## 4. Token lengths with the Qwen3.8-27B tokenizer

What an episode holds besides the model's own text, per cell (3 instances each, chat template with
`reasoning_effort=medium`):

```
$ PYTHONPATH=<repo>/src:<repo>/integrations/verl <venv r4_32_verl>/bin/python -m mrgym_verl.dryrun --tokenizer <tokenizer of Qwen/Qwen3.8-27B> --model-type qwen3_5 \
      --overrides /dev/null --cells "Scanner/SNRSpec-v0:1,2,3 Scanner/NoWrap-v1:0,1,2,3 Scanner/NullTissue-v0:0,1,2,3 Scanner/PhantomQA-v0:1,2,3 Scanner/ProtocolToSpec-v0:1,2,3 Scanner/ArtefactClinic-v0:1,2,3 Scanner/T1Transmit-v1:1,2,3" \
      --seeds 100-102 --tokens-only --chat-kwargs reasoning_effort=medium --out <work>/verl/tokens
```

| Cell | First prompt | of which tool schemas | Observation median / max | Reference episode without model text | Tool-call limit | First prompt + a median observation per allowed call |
|---|---:|---:|---:|---:|---:|---:|
| Scanner/SNRSpec-v0/L1 | 5,961 | 4,769 | 494 / 730 | 18,958 | 48 | 29,673 |
| Scanner/SNRSpec-v0/L2 | 5,959 | 4,769 | 487 / 729 | 18,914 | 48 | 29,335 |
| Scanner/SNRSpec-v0/L3 | 5,961 | 4,769 | 493 / 729 | 18,952 | 48 | 29,625 |
| Scanner/NoWrap-v1/L0 | 5,970 | 4,717 | 459 / 1,228 | 8,548 | 40 | 24,330 |
| Scanner/NoWrap-v1/L1 | 5,970 | 4,717 | 459 / 1,228 | 8,547 | 40 | 24,330 |
| Scanner/NoWrap-v1/L2 | 5,670 | 4,717 | 459 / 1,228 | 8,251 | 40 | 24,030 |
| Scanner/NoWrap-v1/L3 | 5,670 | 4,717 | 459 / 1,228 | 8,251 | 40 | 24,030 |
| Scanner/NullTissue-v0/L0 | 5,870 | 4,523 | 501 / 728 | 17,061 | 48 | 29,918 |
| Scanner/NullTissue-v0/L1 | 5,481 | 4,523 | 500 / 727 | 16,665 | 48 | 29,481 |
| Scanner/NullTissue-v0/L2 | 5,557 | 4,563 | 499 / 727 | 13,359 | 48 | 29,509 |
| Scanner/NullTissue-v0/L3 | 5,556 | 4,563 | 499 / 727 | 13,338 | 48 | 29,508 |
| Scanner/PhantomQA-v0/L1 | 6,172 | 4,678 | 779 / 1,359 | 40,192 | 64 | 56,028 |
| Scanner/PhantomQA-v0/L2 | 6,172 | 4,678 | 768 / 1,360 | 39,287 | 64 | 55,324 |
| Scanner/PhantomQA-v0/L3 | 6,172 | 4,678 | 770 / 1,366 | 39,429 | 64 | 55,452 |
| Scanner/ProtocolToSpec-v0/L1 | 6,212 | 4,760 | 812 / 911 | 12,176 | 48 | 45,188 |
| Scanner/ProtocolToSpec-v0/L2 | 6,211 | 4,760 | 812 / 908 | 12,179 | 48 | 45,187 |
| Scanner/ProtocolToSpec-v0/L3 | 6,212 | 4,760 | 812 / 910 | 12,174 | 48 | 45,188 |
| Scanner/ArtefactClinic-v0/L1 | 7,856 | 6,312 | 541 / 882 | 28,863 | 160 | 94,416 |
| Scanner/ArtefactClinic-v0/L2 | 7,879 | 6,332 | 540 / 884 | 28,876 | 160 | 94,279 |
| Scanner/ArtefactClinic-v0/L3 | 7,888 | 6,332 | 533 / 834 | 23,769 | 160 | 93,168 |
| Scanner/T1Transmit-v1/L1 | 5,795 | 4,722 | 519 / 814 | 13,044 | 64 | 39,011 |
| Scanner/T1Transmit-v1/L2 | 5,719 | 4,681 | 516 / 724 | 12,622 | 64 | 38,743 |
| Scanner/T1Transmit-v1/L3 | 5,794 | 4,722 | 516 / 791 | 12,966 | 64 | 38,818 |

The first prompt is dominated by the tool schemas. The last column is what the context holds besides the model's own
text when an episode uses every tool call its environment allows. The model's text is not in this table: no model
was run. `MAX_PROMPT_LEN` of the plan (8,192) has to cover the first column of every cell that is trained on or
evaluated; `MAX_RESPONSE_LEN` is set from the calibration (PREREG-27b.md section 2).

## 5. Tests

```
$ PYTHONPATH=<repo>/src <venv rebuild>/bin/python -m pytest tests/test_verl_adapter.py tests/test_verl_adapter_verl.py -q
............................                                             [100%]
=========================== short test summary info ============================
SKIPPED [1] tests/test_verl_adapter_verl.py:22: verl is not installed (see the module docstring)
28 passed, 1 skipped in 41.69s

$ MRGYM_VERL_TOKENIZER=<tokenizer of Qwen/Qwen3.8-27B> PYTHONPATH=<repo>/src <venv r4_32_verl>/bin/python -m pytest tests/test_verl_adapter_verl.py -q
.                                                                        [100%]
1 passed in 18.04s

$ ruff check src tests integrations docs
All checks passed!

$ python docs/experiments/27b/compute.py --check; echo $?
0
```

## 6. Not simulated

- vLLM, the FSDP actor, weight sync between them, verl's trainer loop and its TransferQueue; therefore memory,
  throughput, and whether the agent loop receives `global_steps` and `uid` the way verl's source says.
- Building the Python environments with uv on a node and downloading weights (the scripts keep the commands of the
  earlier jobs that did both; in the simulation they are skipped).
- Uploads to a bucket: the simulation writes to a directory through the same class; the `s3://` branch of
  `mrgym_verl/store.py` (boto3) has not been executed.
- The real serving script `node/serve.sh` (taken from the earlier base-rate job with ports and GPU groups made
  variables; that edit has not been run).
- `node/make_main.sh` output as a job: it was generated and syntax-checked only.
