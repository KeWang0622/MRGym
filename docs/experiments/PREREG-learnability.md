# Pre-registration: does an open model improve on MR Gym by reinforcement learning?

Status: **not launched.** Nothing in this plan has been run: no model was called, no job was started, no pass rate of
any model on any level of `Scanner/NullTissue-v0` exists. The plan is fixed here before the first training step; a
change after that step is a deviation and is reported as one. Launching needs the owner's word, the gate of section 2
and the adapter work of section 8.

The machine-readable plan is [learnability/plan.json](learnability/plan.json); the trainer settings are
[learnability/overrides.env](learnability/overrides.env). Design values (thresholds, sizes, seeds) are choices made
here. The two measured inputs of the compute table were measured on another environment and are marked as such.

## 1. Claim and endpoints

Claim under test: "an open model demonstrably improves by RL on MR Gym" = E1 and E2 and E3. E4 to E7 are reported
whatever they show.

| # | Endpoint | Counts as shown when |
|---|---|---|
| E1 | learns: held-out pass rate on the entry rung, trained minus base | at least +15 points and interval lower bound above +5, in each of R1, R2, R3 |
| E2 | climbs: held-out pass rate on level 2, trained minus base | at least +15 points in at least 2 of the 3 seeds |
| E3 | not the harness only: P(pass given a delivered, measured answer), and the share of outcome `some_nulled` or better among delivered answers | both rise, lower bounds above 0; the outcome table (section 2 of docs/TRAINING.md) is printed in any case |
| E4 | unseen difficulty: level 3, never trained on | lower bound of trained minus base above 0 |
| E5 | untrained environment | reported with interval, either sign; "no harm" = not below base minus 5 points |
| E6 | reward against demonstration: R minus S on level 2, level 3 and the untrained environment | reported; R at least S minus 5 on level 2 reads "teaches by reward" |
| E7 | ablations U (uniform levels) and G (`progress` as the signal) against the mean of R1 to R3 | reported |

Statistics, fixed in advance: pass rate with a Wilson 95 % interval per cell; differences by bootstrap over instances
(10,000 resamples, the instance is the cluster); exact McNemar on the first episode of each instance; Holm over E1 to
E4; seeds are never pooled. The test split is opened once, for the checkpoint of step 120 (or the early-stop step),
never for selection.

## 2. Launch gate

All of these, measured by commands of the package, before the first training step:

1. `mrgym gate Scanner/NullTissue-v0 --n 300 --write` admits levels 1, 2 and 3 on the engine of the run.
2. `mrgym check Scanner/NullTissue-v0 --learnability --level 2 --write` passes.
3. `mrgym calibrate Scanner/NullTissue-v0 --model Qwen/Qwen3.5-9B --levels 0,1,2,3 --n 64 --episodes 8 --write` finds an
   entry rung among levels 0 to 2: pass rate in [0.20, 0.60] and at least 80 % of the groups mixed. In the same
   records: at least 95 % of the calls are taken as written (`counts.shape` + `counts.vocabulary` against
   `counts.calls`), at least 70 % of the episodes submit,
   fewer than 20 % of the passes fall on the last turn.
4. If no level is in band and all are below: one more hint tier on level 0, re-measured, at most twice. If it still
   fails, the arms R1 to R3 switch to `progress` as the training signal and the report says so in its first sentence.

Whether the model lands in band on any level is not known.

## 3. Environment, levels, splits

| Item | Value |
|---|---|
| Training environment | `Scanner/NullTissue-v0`, levels 0 (practice), 1, 2 |
| Never trained on | level 3 of the same id |
| Untrained environment | a second admitted family environment with a calibrated in-band level. None exists at the time of writing; the cell is dropped from the report if none exists at launch |
| Calibration | seeds 9000 - 9063 per level, public train split |
| Dev | seeds 0 - 63 per level, public train split, never sampled in training; 2 episodes each, every 30 steps; checkpoint health only |
| Training | a fresh seed for every group, from 10,000 (`mrgym.curriculum`): 16 x 120 instances per arm |
| Test | split `heldout` (needs `MRGYM_SECRET`), 200 instances x 2 episodes per cell, temperature 1.0; cells: level 1, level 2, level 3, the untrained environment |

Level 0 is a practice level: a gain on level 0 is never reported as the result.

## 4. Arms

| Arm | What | Seeds |
|---|---|---|
| R1, R2, R3 | RL from the base model, bandit curriculum over levels 0 - 2, the environment's paid reward | 3 |
| U | as R1 with uniform level sampling | 1 |
| G | as R1 with `info["score"]["progress"]` as the training signal | 1 |
| S | supervised fine-tuning on 1,200 reference-policy trajectories of levels 0 - 2 (lr 1e-4, 2 epochs, assistant tokens only), no RL | 1 |
| B | the base model | - |
| scripted rows | `reference`, `textbook`, `fixed_recipe`, `prior` on every test cell | - |

## 5. Algorithm

GRPO over whole episodes in verl: groups of 8, 16 groups per step, 120 steps, learning rate 1e-5 after 5 warm-up
steps, clip 0.2, gradient clip 1.0, no KL term, no entropy bonus, temperature 1.0, top-p 1, LoRA rank 32 / alpha 64 on
`Qwen/Qwen3.5-9B`, a cap of 12 turns stated in the task text and the status line, 2,048 new tokens per turn without reasoning (8,192 with),
context 32k. Groups without spread get zero advantage and are counted per step. The reward is the environment's paid
reward. The reward form is never changed inside a run.

Curriculum: `mrgym.curriculum.Curriculum(mode="bandit", alpha=0.5, tau=0.5, floor=0.10)`, its state in every
checkpoint, its metrics in the training log.

## 6. Stop rules

A stop is reported as a result.

- F1, steps 1 - 10: mean training pass rate below half the calibrated base rate, or fewer than 40 % of the groups mixed:
  stop. The training harness differs from the calibration harness; fix that, do not continue.
- F2, step 60: dev pass rate on the entry rung below base plus 5 points: stop that seed and keep it in the table as a
  null.
- F3, any time: submit rate below 50 %, or 20 % or more of the replies cut by the token limit, or the training pass
  rate below half its running maximum for 10 steps: stop and keep the last checkpoint before the fall.
- S1: dev pass rate at least 90 % on levels 0 - 2 at two consecutive evaluations: stop early; test as planned.

## 7. Compute

<!-- compute-table: python docs/experiments/learnability/compute.py -->
| Design | Part | Episodes | GPU-hours |
|---|---|---:|---:|
| Full design (R1+R2+R3+U+G+S+B) | Calibration | 2,048 | 10 |
| Full design (R1+R2+R3+U+G+S+B) | SFT arm | 0 | 4 |
| Full design (R1+R2+R3+U+G+S+B) | RL arms | 76,800 | 304 - 527 |
| Full design (R1+R2+R3+U+G+S+B) | Dev and test evaluations | 18,880 | 89 |
| Full design (R1+R2+R3+U+G+S+B) | Total | 97,728 | 407 - 629 |
| Full design | Node-hours on 8-GPU nodes; with 30 % for preemption | | 51 - 79; 66 - 102 |
| Core only (R1+R2+R3+S+B) | Calibration | 2,048 | 10 |
| Core only (R1+R2+R3+S+B) | SFT arm | 0 | 4 |
| Core only (R1+R2+R3+S+B) | RL arms | 46,080 | 182 - 316 |
| Core only (R1+R2+R3+S+B) | Dev and test evaluations | 12,608 | 60 |
| Core only (R1+R2+R3+S+B) | Total | 60,736 | 256 - 389 |
| Core only | Node-hours on 8-GPU nodes; with 30 % for preemption | | 32 - 49; 42 - 63 |
<!-- /compute-table -->

The table is computed from `plan.json` (`compute.py --check` fails when it is stale). Its two measured inputs, the
GPU-seconds per episode and the seconds per training step, come from runs on `Scanner/NoWrap-v0` with 12 turns and
reasoning off. Episodes of `Scanner/NullTissue-v0` have more scans and a longer first prompt on level 0; their cost
has not been measured, so the table is an estimate of the order, not a budget. The SFT figure is an upper-bound
estimate and was never run. The first 10 steps of R1 measure the step time; the table is recomputed from it before
the other arms start.

## 8. What has to exist before launch

Built and tested in this package: the calibration command, the learnability check, the curriculum sampler and its
row hook, the outcome and progress fields of the episode records, the turn cap in the status line.

Not built, and needed:

1. The trainer adapter has to take its instances from the curriculum (`mrgym.curriculum.RowHook`) for the arms R and
   G. In verl the agent loops run in several worker processes, so the hook has to be one named Ray actor. This wiring
   is not written and has never been run. Without it only the arms U (a static dataset over levels 0 - 2) and B, and the
   calibration, can start.
2. The adapter has to offer `progress` as a reward mode for arm G, call `env.note_turn()` once per reply with
   `max_turns` passed to `mrgym.make`, and log `outcome` and `counts` per episode.
3. A trajectory writer for arm S (reference-policy episodes as chat transcripts in the trainer's format).
4. The held-out secret (`MRGYM_SECRET`) on the evaluation node, and `mrgym results` support for the readout of
   section 1.

## 9. Not known

- Any pass rate of any model on these levels.
- The cost of an episode and of a training step on this environment.
- Whether level 0 as written is passed by a model that follows its text: the scripted reference passes it; no model
  has read it.
- Whether a second family environment will be admitted in time for E5.
