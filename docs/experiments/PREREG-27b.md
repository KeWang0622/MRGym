# Pre-registration: does Qwen3.8-27B improve on MR Gym by reinforcement learning?

First runs exist and are reported in [RESULTS-27b.md](RESULTS-27b.md). They are exploratory: they do not follow this plan (thinking
off, one trained environment) and do not replace it; the status below is that of the plan as written.

Status: **not launched.** Nothing in this plan has been run on a GPU: no pass rate of this model on any v1 environment
exists, and the trainer has never run at this model size. The plan is fixed here before the first training step; a
change after that step is a deviation and is reported as one. The fields marked *(calibration)* are blank on purpose:
they are filled from the calibration job's output by the rule written next to them, before the first training step,
in a commit of their own.

Machine-readable plan: [27b/plan.json](27b/plan.json). Settings as environment variables of the node scripts:
[27b/overrides.env](27b/overrides.env). Both tables below are computed (`python docs/experiments/27b/compute.py`;
`--check` fails when they are stale). The code: [integrations/verl](../../integrations/verl/README.md).

This plan stands beside [PREREG-learnability.md](PREREG-learnability.md), which was written for a 9B model on one
environment. That model had no reward on the v0 environments (two runs stopped by their own rule); the 27B had a base
rate inside the band there. Neither fact carries over to v1: the interface and the tasks changed.

## 1. Hypotheses and what counts as "it trained"

| # | Hypothesis | Counts as shown when |
|---|---|---|
| H1 | Training on the paid reward raises the pass rate on unseen instances of the trained environments | on the secret split, per trained environment (its trained levels as strata): trained minus base is at least **+15 points** and the lower bound of its 95 % interval is above **+5 points** |
| H2 | It does so on more than one environment | H1 holds on **at least 2** trained environments |
| H3 | It does not cost the model elsewhere | on the environment that was never trained on, trained minus base is **not below -5 points** (reported with its interval, either sign) |

**"It trained" = H1 on at least 2 environments and H3.** One environment meeting H1 is reported as "improved on one
environment" and is not the success of this plan. Reported whatever they show: per-cell pass rates of base and
trained, the outcome table (how episodes end), the share of calls taken as written, tokens per episode, and the dev
curve of every evaluation.

Not claims of this plan: that the gain is MRI skill rather than harness skill (the outcome table and
P(pass | a delivered, measured answer) are printed so that a reader can judge); that a second seed would repeat it
(arm R2 exists for that and is not part of the verdict); anything about a practice level.

## 2. Launch gate

All of these before the first training step:

1. **Calibration** (`integrations/verl/node/calibrate.sh`): the served base model, reasoning effort `medium`,
   temperature 1.0, each environment's own tool-call limit, 16,384 tokens per reply, 32 instances x 8 episodes per
   level, on `Scanner/SNRSpec-v0` 1-3, `Scanner/NoWrap-v1` 0-3, `Scanner/NullTissue-v0` 0-3, `Scanner/PhantomQA-v0` 1.
   - Training cells *(calibration)*: every (environment, level) of an environment admitted by the stored gate, outside
     its practice levels, whose pass rate is in **[0.20, 0.60]** with **at least 80 % of the groups of 8 mixed**
     (`train_arms.json` of the job). No cell is added or dropped by hand.
   - If fewer than two environments have such a cell: H2 cannot be met. The run may still start on one environment;
     the report says in its first sentence that the plan's success is out of reach.
   - Sequence length *(calibration)*: `MAX_RESPONSE_LEN` is the 99th percentile of the job's stream estimate over
     the training cells minus the first prompt, rounded up to a multiple of 4,096, and at most what the smoke showed
     to fit (item 2). If the 99th percentile does not fit, the reasoning effort is lowered to `low`, the training
     cells are calibrated again at that setting, and the rule above is applied to the new numbers.
   - In the same records: at least 95 % of the calls taken as written, at least 70 % of the episodes submit. A cell
     below either is not trained on.
2. **Training smoke** (`train_smoke.sh`) passes on the node type of the run with this model and the sequence lengths
   of item 1: three steps at a tiny batch, a checkpoint restored after the local disk was wiped, an evaluation after
   the restore, complete records. Its report gives the peak memory per GPU and the seconds per step; the compute
   table is recomputed from them before the run.
3. The stored gate (`mrgym gate <id> --n 300`) admits every environment that is trained on or read out, on the engine
   of the run.
4. The owner's word.

## 3. Environments, levels, splits

| Item | Value |
|---|---|
| Candidates for training | `Scanner/SNRSpec-v0` levels 1-3 and `Scanner/NoWrap-v1` levels 2-3 (the environments admitted at the time of writing) |
| Training cells | *(calibration)*: by the rule of section 2 |
| Never trained on, for transfer | one whole environment *(calibration)*: the first of `Scanner/NullTissue-v0`, `Scanner/PhantomQA-v0`, `Scanner/ProtocolToSpec-v0`, `Scanner/ArtefactClinic-v0` that the stored gate admits at launch and that has a level with a calibrated pass rate in [0.05, 0.80] (so that a drop and a rise are both visible). If none qualifies, H3 is not assessed and no verdict is given |
| Practice levels | never trained on as a result and never reported as one: `Scanner/NoWrap-v1` 0-1, every level of `Scanner/T1Transmit-v1` |
| Calibration | public `train` split, seeds 9000-9031 per level |
| Training | public `train` split, a fresh seed for every group from 10,000 (`mrgym.curriculum`): 16 x 60 instances per arm |
| Dev | public `dev` split, seeds 0-31 per training cell, 2 episodes each, before the first step and every 10 steps: stop rules only, never the result |
| Test | secret split `test-iid` (`MRGYM_SECRET_FILE`), seeds 0-149 per cell, 2 episodes each, temperature 1.0; cells: every training cell and every level of the untrained environment that calibrated in [0.05, 0.80]. Opened once, after training, for the last checkpoint |

## 4. Arms

| Arm | What | Seeds |
|---|---|---|
| R1 | RL from the base model on the training cells, bandit curriculum, the environment's paid reward | 1 |
| R2 | as R1 with another seed; started only after R1 has passed stop rule F1; reported next to R1, never pooled | 1 |
| B | the base model, evaluated through the same agent loop on the same held-out instances | - |
| scripted rows | the `reference` and `prior` control policies on every test cell (`mrgym run-agent --policy`) | - |

## 5. Algorithm and memory

GRPO over whole episodes in verl (commit `8357f1e`), colocated synchronous trainer on one node of 8 GPUs: groups of
8, 16 groups per step, 60 steps, learning rate 1e-5 after 5 warm-up steps, clip 0.2 / 0.28, gradient clip 1.0, no KL
term, no entropy bonus, sampling temperature 1.0, top-p 1. LoRA rank 32 / alpha 64 on every linear layer of the
language model. The reward is the environment's paid reward, 1 for a pass and 0 otherwise; a group without spread
gets zero advantage and is counted. No turn cap: an episode ends where the environment ends it. 16,384 tokens per
reply, stated in the task text. Prompt 8,192 and response 24,576 tokens *(calibration, section 2)*. Checkpoint every
5 steps (adapter and optimizer), copied off the node before it is announced. The reward form, the limits and the
curriculum are never changed inside a run.

Rollout and training alternate on the same GPUs: two rollout replicas with tensor-parallel size 4 generate, verl puts
them to sleep (weights to host memory, cache released), the 8-way sharded actor does one update, the adapter is
pushed into the replicas. What each GPU holds, from the model's dimensions:

<!-- memory-table: python docs/experiments/27b/compute.py -->
| Card | While generating: weights of a replica per GPU (TP 4) | Rollout budget per GPU (0.60) | KV cache per replica | Longest episodes (32,768 tokens) held in cache per replica | While training: actor shard per GPU | Left for activations (reserve 6 GB) | Tokens of one micro-batch that fit | The plan's 32,768-token episode |
|---|---:|---:|---:|---:|---:|---:|---|---|
| H100 80 GB | 13.9 GB | 48 GB | 136 GB = 2.07 M tokens | 63 | 7.0 GB | 67 GB | 28k - 49k (2.35 - 1.36 MB per token) | between the two estimates: the smoke decides |
| H200 141 GB | 13.9 GB | 85 GB | 283 GB = 4.28 M tokens | 131 | 7.0 GB | 128 GB | 54k - 94k (2.35 - 1.36 MB per token) | fits on paper |
| B200 180 GB | 13.9 GB | 108 GB | 376 GB = 5.70 M tokens | 174 | 7.0 GB | 167 GB | 71k - 123k (2.35 - 1.36 MB per token) | fits on paper |
<!-- /memory-table -->

Reading of the table. The weights are not the constraint on any card. The constraint is one long episode in the
update: per token the actor keeps the inputs of 64 layers and the logits over 248,320 vocabulary entries (one copy
if the in-place cross-entropy is available, up to three otherwise). The per-token figures are arithmetic, not
measurements. On 80 GB cards the planned episode length is inside the uncertainty; the smoke answers it, and its
failure line names the cause (`OOM on the first backward`). The cache of one replica holds 63 of the longest episodes
on 80 GB cards against 64 in flight per replica: requests then wait, they do not fail.

How much of an episode is left for the model's own text. The calls of a procedure and their observations are in the
stream whatever the model writes around them. Measured with the model's tokenizer on CPU
([27b/token_table.json](27b/token_table.json); no model was run), for the reference procedure of each cell:

<!-- room-table: python docs/experiments/27b/compute.py -->
| Cell | First prompt | Reference procedure: calls | Its stream without any model text | Model text per call: at 32,768 tokens (the plan) | at 54,392 tokens (H200, lower estimate) | at 70,959 tokens (B200, lower estimate) | Stream without model text if every allowed call is used |
|---|---:|---:|---:|---:|---:|---:|---:|
| Scanner/SNRSpec-v0/L1 | 5,961 | 20 | 18,958 | 690 | 1,772 | 2,600 | 29,673 (limit 48) |
| Scanner/SNRSpec-v0/L2 | 5,959 | 20 | 18,914 | 693 | 1,774 | 2,602 | 29,335 (limit 48) |
| Scanner/SNRSpec-v0/L3 | 5,961 | 20 | 18,952 | 691 | 1,772 | 2,600 | 29,625 (limit 48) |
| Scanner/NoWrap-v1/L0 | 5,970 | 4 | 8,548 | 6,055 | 11,461 | 15,603 | 24,330 (limit 40) |
| Scanner/NoWrap-v1/L1 | 5,970 | 4 | 8,547 | 6,055 | 11,461 | 15,603 | 24,330 (limit 40) |
| Scanner/NoWrap-v1/L2 | 5,670 | 4 | 8,251 | 6,129 | 11,535 | 15,677 | 24,030 (limit 40) |
| Scanner/NoWrap-v1/L3 | 5,670 | 4 | 8,251 | 6,129 | 11,535 | 15,677 | 24,030 (limit 40) |
| Scanner/NullTissue-v0/L0 | 5,870 | 19 | 17,061 | 827 | 1,965 | 2,837 | 29,918 (limit 48) |
| Scanner/NullTissue-v0/L1 | 5,481 | 19 | 16,665 | 848 | 1,986 | 2,858 | 29,481 (limit 48) |
| Scanner/NullTissue-v0/L2 | 5,557 | 13 | 13,359 | 1,493 | 3,156 | 4,431 | 29,509 (limit 48) |
| Scanner/NullTissue-v0/L3 | 5,556 | 13 | 13,338 | 1,495 | 3,158 | 4,432 | 29,508 (limit 48) |
| Scanner/PhantomQA-v0/L1 | 6,172 | 31 | 40,192 | does not fit | 458 | 992 | 56,028 (limit 64) |
| Scanner/PhantomQA-v0/L2 | 6,172 | 30 | 39,287 | does not fit | 504 | 1,056 | 55,324 (limit 64) |
| Scanner/PhantomQA-v0/L3 | 6,172 | 30 | 39,429 | does not fit | 499 | 1,051 | 55,452 (limit 64) |
| Scanner/ProtocolToSpec-v0/L1 | 6,212 | 6 | 12,176 | 3,432 | 7,036 | 9,797 | 45,188 (limit 48) |
| Scanner/ProtocolToSpec-v0/L2 | 6,211 | 6 | 12,179 | 3,432 | 7,036 | 9,797 | 45,187 (limit 48) |
| Scanner/ProtocolToSpec-v0/L3 | 6,212 | 6 | 12,174 | 3,432 | 7,036 | 9,798 | 45,188 (limit 48) |
| Scanner/ArtefactClinic-v0/L1 | 7,856 | 27 | 28,863 | 145 | 946 | 1,559 | 94,416 (limit 160) |
| Scanner/ArtefactClinic-v0/L2 | 7,879 | 27 | 28,876 | 144 | 945 | 1,559 | 94,279 (limit 160) |
| Scanner/ArtefactClinic-v0/L3 | 7,888 | 21 | 23,769 | 429 | 1,458 | 2,247 | 93,168 (limit 160) |
| Scanner/T1Transmit-v1/L1 | 5,795 | 9 | 13,044 | 2,192 | 4,594 | 6,435 | 39,011 (limit 64) |
| Scanner/T1Transmit-v1/L2 | 5,719 | 9 | 12,622 | 2,238 | 4,641 | 6,482 | 38,743 (limit 64) |
| Scanner/T1Transmit-v1/L3 | 5,794 | 9 | 12,966 | 2,200 | 4,603 | 6,444 | 38,818 (limit 64) |
<!-- /room-table -->

Reading of this table. At the plan's 32,768 tokens an episode of `Scanner/NoWrap-v1` has about 6,000 tokens of
model text per call of its four-call procedure; an episode of `Scanner/SNRSpec-v0` that makes the twenty calls of its
procedure has about 690 per call, reasoning included; `Scanner/PhantomQA-v0` does not fit at all. For scale: this
model wrote about 2,350 completion tokens per reply at reasoning effort `medium` in an earlier job on another kind of
task (job r02 of the project record); what it writes on v1 is not known. So the two environments admitted today
cannot both be trained at 32,768 tokens unless the model writes far less per reply than it did there. H2 needs two
environments. The ways out, in the order of what they change:

1. cards with more memory: at their lower estimate, 141 GB cards leave about 1,770 and 180 GB cards about 2,600
   tokens per call on `Scanner/SNRSpec-v0`;
2. sequence parallelism in the actor (one episode split over several GPUs; a setting of verl we have not run);
3. reasoning effort `low` (another model setting: it is calibrated again, section 2).

Which of them applies is decided from the calibration's measured episode lengths, written into this file before the
first training step, and is the reason the sequence lengths above are marked *(calibration)*.

## 6. Curriculum

`mrgym.curriculum.Curriculum(mode="bandit", alpha=0.5, tau=0.5, floor=0.10)` over the training cells only, one
sampler for the run (a named actor inside the trainer), its state in every checkpoint, its seeds never issued twice
to a step that is kept.

## 7. Stop rules

Applied by `integrations/verl/mrgym_verl/stoprules.py` from the per-episode log, every two minutes. A stop is a
result: the job writes the rule and its numbers, keeps the last checkpoint, and the held-out split stays closed
(except for S1).

| Rule | Stops when |
|---|---|
| R0 no reward | mean training pass rate of the first 4 steps below 2 % |
| F1 harness | over the first 10 steps: mean training pass rate below half the calibrated base rate of the training cells, or fewer than 40 % of the groups mixed. The training harness then differs from the calibration harness; fix that, do not continue |
| F3a not delivering | submit rate below 50 % over the last 5 steps |
| F3b cut replies | 20 % or more of the episodes of the last 5 steps cut by the reply limit or a full context |
| F3c collapse | training pass rate below half its earlier maximum for 10 steps in a row |
| F2 futility | sequential test on the dev instances at an evaluation from step 30 on: the one-sided upper bound of (checkpoint minus base), at level 1 - 0.05 / 6, is below +5 points. The seed stays in the table as a null |
| S1 mastered | dev pass rate at least 90 % at two consecutive evaluations: stop early, read out as planned |

The sequential test: 6 looks declared (steps 10 to 60), each a one-sided paired test on the instance means against
the evaluation before the first step, at 0.05 / 6 (`mrgym.stats.sequential.checkpoint_look`). Efficacy (lower bound
above 0) is recorded at every look and never stops a run. The dev split is public; it selects nothing.

## 8. Compute

<!-- compute-table: python docs/experiments/27b/compute.py -->
One step = 128 episodes, 4.2 M trained tokens: generation 3.4 - 6.8 GPU-h, update 1.2 - 2.3 GPU-h, total 4.6 - 9.2 GPU-h = 2060 - 4121 s on one 8-GPU node.

| Design | Part | Episodes | GPU-hours | Basis |
|---|---|---:|---:|---|
| Core (R1 + B) | Calibration (served base model, no adapter) | 3,072 | 84 | 12 levels x 32 x 8 |
| Core (R1 + B) | Training smoke | 60 | 8 - 9 | 3 steps at a tiny batch, two trainer starts, the CPU gate |
| Core (R1 + B) | RL steps | 7,680 | 275 - 549 | 60 steps of 128 episodes |
| Core (R1 + B) | Dev evaluations (in training) | 1,792 | 48 - 96 | 7 evaluations per arm, 4 cells assumed |
| Core (R1 + B) | Held-out readout | 3,600 | 100 - 196 | 2 models x 6 cells assumed x 150 x 2 |
| Core (R1 + B) | Total | 16,204 | 514 - 934 | node-hours on 8-GPU nodes: 64 - 117; with 30 % for preemption: 84 - 152 |
| With the second seed (R1 + R2 + B) | Calibration (served base model, no adapter) | 3,072 | 84 | 12 levels x 32 x 8 |
| With the second seed (R1 + R2 + B) | Training smoke | 60 | 8 - 9 | 3 steps at a tiny batch, two trainer starts, the CPU gate |
| With the second seed (R1 + R2 + B) | RL steps | 15,360 | 549 - 1099 | 120 steps of 128 episodes |
| With the second seed (R1 + R2 + B) | Dev evaluations (in training) | 3,584 | 96 - 191 | 7 evaluations per arm, 4 cells assumed |
| With the second seed (R1 + R2 + B) | Held-out readout | 5,400 | 150 - 294 | 3 models x 6 cells assumed x 150 x 2 |
| With the second seed (R1 + R2 + B) | Total | 27,476 | 887 - 1677 | node-hours on 8-GPU nodes: 111 - 210; with 30 % for preemption: 144 - 273 |
<!-- /compute-table -->

What the table rests on. One measured number: 96 GPU-seconds of generation per episode, for this model on the v0
`NoWrap` environment with a 12-call cap, served without an adapter and with speculative decoding. Episodes of the v1
environments have other limits and have not been timed. The range of the generation column is an assumed factor of
1 to 2 for serving with an adapter loaded. The update column is an estimate from another trainer on other cards.
The numbers of cells (4 trained, 6 read out) are placeholders until the calibration. The table is an order of
magnitude, not a budget: the calibration job reports GPU-seconds per episode on v1, the smoke reports seconds per
step, and the table is recomputed from both before the run.

## 9. Readout and statistics

Fixed in advance and computed by `integrations/verl/mrgym_verl/readout.py` from two episode logs (base and trained,
the same instances, the same agent loop):

- per cell and per environment: pass rate with the clustered 95 % interval of `mrgym.stats.intervals.cluster_rate`
  (the instance is the cluster, levels are strata);
- trained minus base on the instances both played: the paired difference of the instance means with its 95 %
  interval (`mrgym.stats.compare.paired_compare`), levels as strata;
- Holm-adjusted p-values over the trained environments;
- the verdict of section 1 with its three numbers (+15, +5, -5) and the count of environments.

The held-out split is opened once per arm, for the checkpoint of step 60 or of an S1 stop, never for selection. A run
stopped by any other rule is reported with its dev curve and no held-out numbers.

## 10. Not known

- Any pass rate of this model on any v1 level, and therefore the training cells and whether two environments qualify.
- How long its episodes are on v1 as one token stream, and whether that length fits an 80 GB card in the update.
- Seconds per step in this stack at this size; throughput of the rollout replicas with an adapter loaded.
- Whether the tool-call parser of the trainer reads everything the served model writes (it read scripted turns of
  the same format on CPU).
- Whether a third environment will be admitted in time to serve as the untrained one.
