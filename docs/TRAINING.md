# Training on MR Gym

What a trainer gets from an environment beyond `reset` and `step`, and how to find out whether a model can start
learning on it. Everything on this page is for environments of the family framework (`Scanner/NullTissue-v0` and
later families); `Scanner/NoWrap-v1` and `Scanner/T1Transmit-v1` follow it since their migration; their `-v0` ids are retired.

No measured number is typed on this page. The band table is computed (`mrgym.learn.band_table()`, checked by a
test); results of the commands are stored as package data and shown in the environment card.

## 1. The reward

The paid reward is 1 when the delivered result passes every check on the hidden world and the scanner's log shows a
measurement within the budget, else 0. It is paid once, by `submit`. Nothing else is ever paid: no partial credit, no
step bonus, no efficiency multiplier.

`info["score"]["progress"]` is a diagnostic in [0, 1] that the environment computes and never pays:

| Value | When |
|---|---|
| 1 | exactly when the episode passes |
| 0 | nothing valid was delivered, nothing was measured, or the log does not replay within the budget |
| in [0, 1) | a delivered, measured answer that fails: the family's own value, or by default `PROGRESS_FAIL_MAX` times the share of the task's checks that pass before the first failure, in their declared order |

A family's own value has to meet three conditions. Classes of outcome are ordered: the largest value of a worse class
lies below the smallest of the next. Every failing answer stays below 1, so 1 is reached only by passing. An answer
that ignores the object stays low: `mrgym check <id> --learnability` measures the mean progress of the best answer
that uses the task card only and fails above `mrgym.learn.check.BLIND_PROGRESS_MAX`.

A trainer may use `progress` as its own training signal (for a start below the entry band, section 6). It is the
trainer's choice and it is stated with the result; the environment's reward stays the pass.

## 2. What `info` holds

`reset` returns, besides the fields of every environment:

```
group       {id, env_id, level, split, task_id, noise_nonce}    episodes with the same id played the same hidden world
max_turns   the turn cap, or None
```

`step` returns at every step:

```
step        {turn, tool, ok, error_class, error_type, counted, cost_s, scan_time_left_s, tool_calls_left,
             rejected_calls_left, turns_left}
            error_class, error_type: the interface's own (mrgym.interface). The class is None, "shape" or
            "vocabulary" (the interface tax: the call was not taken as written), "task" (the instrument or the
            verifier said no) or "internal"; the type is one of mrgym.interface.ERRORS.
            counted: the call counts as a tool call. A call refused with class "shape" or "vocabulary" (for
            how it is written) is a rejected call instead, as is a refused submission; a call refused with class
            "task" (for what it asks) is a tool call, whatever its type
```

and at the last step of the episode, terminated or truncated:

```
outcome       one name of env.outcomes(), the closed taxonomy of the environment, worst first
ladder        {submitted, valid, measured, in_budget, passed}       each flag implies the ones before it
passed, reward, progress
distance      numbers of the family on how far the delivered answer was ({} when there are none)
counts        {turns, length_cuts, scans, scan_s_used, refused_deliverables} and the interface's own counts
              (env.interface_counts()): {calls, executed, rejected, shape, vocabulary, task_refusals, coerced,
              internal_errors}
truncated_by  None | "max_steps" | "rejected_call_limit" | "max_turns" | "stopped" | "length"
```

`mrgym.learn.hooks.validate(info, env.outcomes())` checks a dict against this schema.

Outcomes of every family, worst first: `no_submission_cap` (a limit on tool calls or turns), `no_submission_rejected`
(the interface's limit on rejected calls), `no_submission_stopped`, `no_submission_length`, `invalid_deliverable`
(nothing was delivered and at least one submission was refused), `no_measurement`, `log_or_budget`, then the family's labels for a delivered answer that fails
(`check_failed` by default; `none_nulled`, `some_nulled` for `Scanner/NullTissue-v0`), then `pass`.

An episode that the harness ends has no last step. The harness says how it ended, and gets the same final dict back:

```python
env.note_turn()                          # once per reply of the agent, before executing the calls of that reply
env.note_length_cut()                    # a reply was cut by the token limit
env.end_without_submission("stopped")    # or "length", "max_turns": the episode is over, reward 0
env.training_info()                      # the final dict of the episode, None while it runs
```

`mrgym run-agent` and `mrgym calibrate` do this, and write `outcome`, `progress`, `ladder`, `counts`, `checks` and the
group id into every episode record of a family environment.

## 3. Caps

The limit every environment has is the number of tool calls (`max_steps`). The task text, the status line, the
`status` tool and `info["limits"]` state it, and `mrgym run-agent` and `mrgym calibrate` use it as their cap.

A trainer that counts replies can add a cap on turns. A turn is one reply of the agent, whatever the number of tool
calls in it. With `mrgym.make(id, max_turns=N)` the cap is a limit like the others: the Limits block of the task
text, `info["limits"]`, the `status` tool and the status line (`turn k of N`) state it; when one turn is left the
status line says to submit in it; the call of `note_turn` that would start turn N + 1 truncates the episode with
reward 0 and outcome `no_submission_cap`. Without `max_turns` nothing about turns is said or enforced.

## 4. Groups and noise

The same (seed, level, tool calls) gives the same observations. The members of a group therefore share the hidden
world and, for identical calls, the noise: common random numbers, the default. For independent noise on the same
world give every member another nonce:

```python
env.reset(options={"noise_nonce": k})    # same world, same task, same verdict on an answer; other pixel noise
```

## 5. Step rewards that no policy can exploit

`mrgym.learn.shaping.Shaped(env)` adds potential-based step rewards: `F = phi(after) - phi(before)`, and the last
step of an episode, however it ends, pays back what was paid before it. The step rewards of an episode sum to its
terminal reward for every policy. A trainer that uses the return of the whole episode is unchanged by the wrapper; it
only matters for estimators with per-turn credit. The potential comes from the family (`task.potential`); `env.rewards`
lists what was paid. A bonus that is paid and never taken back is not offered.

## 6. Where a model can start: `mrgym calibrate`

```
mrgym calibrate Scanner/NullTissue-v0 --model MODEL --base-url http://HOST:8000/v1 --levels 0,1,2,3 --n 64 --episodes 8 --out runs/cal
```

plays 64 instances per level in groups of 8 at temperature 1.0 (`--max-turns` becomes the tool-call limit of the
environment, which the task text states), and reports per level the pass
rate with its interval over instances, the share of groups that hold both a pass and a failure, the intraclass
correlation, the outcome counts and the band:

<!-- band-table: mrgym.learn.band_table() -->
| Band | Pass rate | Groups of 8 with both a pass and a failure | Use |
|---|---|---|---|
| dead | < 2 % | < 15 % | never train on it: add an easier level or warm-start the model |
| frontier | 2 - 20 % | 15 - 83 % | only inside a curriculum next to an easier level |
| entry | 20 - 60 % | 83 - 99 % | where a run starts |
| comfortable | 60 - 80 % | 83 - 98 % | keep in the mix |
| mastered | > 80 % | < 83 % | little left to learn: promote, keep a small share |
<!-- /band-table -->

`--max-turns` defaults to 12 tool calls; `--max-turns 0` plays every episode on the environment's own limit, and the
rows then state that limit.

The entry rung is the hardest level in the entry band whose measured share of mixed groups is at least 80 %. The
command exits non-zero when no level qualifies. `--write` stores the rows; an environment card names a model under
`learnability.trainable_by` only through that stored file.

One open-weight model has been calibrated on every cell of the suite: section 5 of
[experiments/RESULTS-27b.md](experiments/RESULTS-27b.md). No calibration is stored as package data yet.

## 7. Levels

A level changes the hidden-world prior, the budget, what is already acquired at reset and the hint tier of the task
text. It never changes the tools, the deliverable or the scoring rule.

A family may mark a level as practice (`LEVELS[lv]["kind"] = "practice"`): it is registered and playable, `mrgym gate`
does not run it by default, and the admitted claim of the id does not cover it. Level 0 of `Scanner/NullTissue-v0` is
one: the world of level 1, the procedure stated in the task text, and the first probe acquired at reset through the
session (it is scan 0, in the log, charged to the budget). Because that probe is a measurement, an answer delivered
without any further scan is not labelled `no_measurement` at level 0. A result on a practice level is never a headline
number.

## 8. Does the reward teach the task: `mrgym check <id> --learnability`

No model and no GPU. A family offers a small parametric family of scripted policies around its reference
(`controls.policy_space()`); a distribution over those parameters is trained on real episodes with the
group-normalised policy gradient of a trainer (groups of 8 on one instance) and, independently, with the
cross-entropy method. It is run from a start that hardly ever passes (`weak`) and from a start in the entry band
(`in_band`), on the paid reward and on `progress`:

| Run | Must |
|---|---|
| `progress` from `weak` | T1 no plateau, T2 climbs, T3 lands on the procedure |
| paid reward from `in_band` | T1, T2, T3 |
| paid reward from `weak` | printed; expected to fail T1 (this is how the check shows that it can fail) |
| cross-entropy method from `weak`, on `progress` and on the paid reward | printed with T4; not part of the verdict |
| best answer that ignores the object | mean `progress` at or below the bound, on fresh `gate:` instances |

The thresholds are `mrgym.learn.check.THRESHOLDS`; the module docstring defines T1 to T4. T1 counts a step in which
every group is without spread only while the policy still fails; the count over every such step, converged ones
included, is stored beside it (`longest_all_dead_run_literal`).

Every distinct (instance, parameter values) pair is one real episode through `mrgym.make` and the tools, played once
and memoised. `--write` stores the report as package data; `mrgym card <id> --json` carries its summary rows under
`learnability.check`.

What the check shows: on this policy family the reward leads from the weak start to the procedure. It does not show
that a language model learns the task; that is the experiment of
[experiments/PREREG-learnability.md](experiments/PREREG-learnability.md) (a 9B model, one environment) and of
[experiments/PREREG-27b.md](experiments/PREREG-27b.md) (a 27B model, the cells its calibration puts in the band).
Neither has been launched.

## 9. Curriculum

```python
from mrgym.curriculum import Curriculum
cur = Curriculum([("Scanner/NullTissue-v0", 0), ("Scanner/NullTissue-v0", 1), ("Scanner/NullTissue-v0", 2)])
rows = cur.sample(16)                    # [{env_id, level, seed, split, group_id}]; a seed is never issued twice
cur.update(row["group_id"], rewards)     # the rewards of one finished group
log = cur.end_step()                     # once per training step
state = cur.state_dict()                 # into the trainer checkpoint
```

`mode="bandit"` samples levels by learning progress (mean absolute group-normalised advantage, with a floor so that
every level stays measured), `mode="ladder"` promotes by a pass-rate threshold, `mode="uniform"` is the ablation.
`mrgym.curriculum.RowHook` adapts it to a trainer with a static dataset. Inside a trainer whose agent loops run in
several processes the hook has to live in one place. For verl that place is a named Ray actor
(`integrations/verl/mrgym_verl/instances.py`); it was asked from two worker processes on CPU and has not run inside
verl's trainer. `Curriculum.drop_pending()` is what a trainer calls after it restored the sampler from a checkpoint:
the groups that were in flight when the process died will never report.

## 10. Training with verl

`integrations/verl/` holds a trainer integration for [verl](https://github.com/verl-project/verl) at a pinned commit:
an agent loop that plays one in-process episode per trajectory, token-in / token-out; instance rows for the `train`,
`dev` and secret `test-iid` splits; the curriculum hook; checkpoints that leave the node; stop rules computed from a
per-episode log; a paired readout; and three entry scripts for one 8-GPU node (calibration, a training smoke, the
run). What it does with the pieces of this page:

| Piece of this page | In the integration |
|---|---|
| The paid reward (section 1) | the trainer's reward, unchanged. `signal: progress` is offered, labelled in every record, and is another experiment |
| Per-step `info`, the outcome (section 2) | stored in every episode record (`training.steps`, `training.outcome`), with the whole conversation, in the schema of `mrgym run-agent` |
| Caps (section 3) | no turn cap; the environment's tool-call limit ends an episode. A reply cut by the token limit or a full context ends it as `no_submission_length` |
| `mrgym calibrate` (section 6) | `node/calibrate.sh` runs it per cell on a served model, resumably, and names the cells in the entry band |
| Curriculum (section 9) | `RowHook` behind the training rows, restricted to the calibrated cells, its state in every checkpoint |

It has been run with verl's own classes and a real tokenizer on CPU and end to end against stand-in servers
(`integrations/verl/SIMULATION.md`). Its first runs on a GPU node are reported in
[experiments/RESULTS-27b.md](experiments/RESULTS-27b.md); the steps and the settings are in [RECIPE.md](RECIPE.md).

## 11. Not built

- The turn cap, the outcome and the per-step `info` exist for family environments only. They are a mixin in front of
  the environment class; moving them into the base class would give them to every environment.
- With a cap on turns, the runner asks the model for one more reply before it learns from `note_turn` that the cap is
  used up. The reply is discarded.
- A dataset command (`mrgym dataset`) is not built; `mrgym.curriculum.rows` and `assert_disjoint` give the rows, and
  `python -m mrgym_verl.instances` (in `integrations/verl`) writes them in verl's dataset format.
- A trajectory writer for supervised fine-tuning on reference-policy episodes is not built.
