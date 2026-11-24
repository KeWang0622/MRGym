# Results

Every number on this page is generated from the episode records under [`episodes/`](episodes/) by `mrgym results`.
Nothing between the two markers is typed by hand, and `mrgym results --check` fails when the table or
[`results.json`](results.json) no longer matches the records.

<!-- RESULTS:START -->
1200 episodes, played 2026-11-24 (UTC) on engine `mrgym-engine-0.1.0`, package version `0.1.0.dev0`. Environments: `Scanner/NoWrap-v0` levels 1, 2; `Scanner/T1Transmit-v0` levels 1, 2, 3. Instances: public train split, seeds 0-23. Agents: scripted control policies only, no language model.

These rows are scripted control policies only. No language model has been run on these instances yet.

**`Scanner/NoWrap-v0`, level 1** (24 instances; the admission gate admits 13 of them)

| Agent | Kind | Episodes | Passed | 95 % interval | Mean reward | Median error | Median scans | Median budget used | Median turns | Tokens in / out | Infra errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| brute_force | scripted | 24 | 0/24 (0%) | 0-14 % | 0.528 | 49.3 % | 3 | 76 % | - | - | 0 |
| console_default | scripted | 24 | 0/24 (0%) | 0-14 % | 0.145 | 84.1 % | 1 | 34 % | - | - | 0 |
| fixed_recipe | scripted | 24 | 6/24 (25%) | 12-45 % | 0.250 | 0.0 % | 1 | 34 % | - | - | 0 |
| no_swap | scripted | 24 | 13/24 (54%) | 35-72 % | 0.852 | 7.3 % | 1 | 34 % | - | - | 0 |
| prior | scripted | 24 | 0/24 (0%) | 0-14 % | 0.000 | 98.6 % | 0 | 0 % | - | - | 0 |
| reference | scripted | 24 | 24/24 (100%) | 86-100 % | 1.000 | 0.0 % | 1 | 34 % | - | - | 0 |
| safe_max | scripted | 24 | 0/24 (0%) | 0-14 % | 0.000 | 98.6 % | 1 | 34 % | - | - | 0 |
| seed_attacker | scripted | 24 | 6/24 (25%) | 12-45 % | 0.250 | -2.9 % | 1 | 34 % | - | - | 0 |
| size_only | scripted | 24 | 10/24 (42%) | 24-61 % | 0.417 | -2.7 % | 1 | 34 % | - | - | 0 |
| wrong_units | scripted | 24 | 0/24 (0%) | 0-14 % | 0.000 | - | 1 | 34 % | - | - | 0 |

**`Scanner/NoWrap-v0`, level 2** (24 instances; the admission gate admits 15 of them)

| Agent | Kind | Episodes | Passed | 95 % interval | Mean reward | Median error | Median scans | Median budget used | Median turns | Tokens in / out | Infra errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| brute_force | scripted | 24 | 3/24 (12%) | 4-31 % | 0.649 | 22.4 % | 3 | 76 % | - | - | 0 |
| console_default | scripted | 24 | 0/24 (0%) | 0-14 % | 0.269 | 44.3 % | 1 | 34 % | - | - | 0 |
| fixed_recipe | scripted | 24 | 5/24 (21%) | 9-40 % | 0.411 | 3.4 % | 1 | 34 % | - | - | 0 |
| no_swap | scripted | 24 | 16/24 (67%) | 47-82 % | 0.932 | 2.8 % | 1 | 34 % | - | - | 0 |
| prior | scripted | 24 | 0/24 (0%) | 0-14 % | 0.000 | 55.7 % | 0 | 0 % | - | - | 0 |
| reference | scripted | 24 | 24/24 (100%) | 86-100 % | 1.000 | 0.0 % | 1 | 34 % | - | - | 0 |
| safe_max | scripted | 24 | 0/24 (0%) | 0-14 % | 0.000 | 55.7 % | 1 | 34 % | - | - | 0 |
| seed_attacker | scripted | 24 | 1/24 (4%) | 1-20 % | 0.042 | -23.9 % | 1 | 34 % | - | - | 0 |
| size_only | scripted | 24 | 0/24 (0%) | 0-14 % | 0.000 | -11.8 % | 1 | 34 % | - | - | 0 |
| wrong_units | scripted | 24 | 0/24 (0%) | 0-14 % | 0.000 | - | 1 | 34 % | - | - | 0 |

**`Scanner/T1Transmit-v0`, level 1** (24 instances; the admission gate admits 0 of them)

| Agent | Kind | Episodes | Passed | 95 % interval | Mean reward | Median error | Median scans | Median budget used | Median turns | Tokens in / out | Infra errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blind_vfa | scripted | 24 | 0/24 (0%) | 0-14 % | 0.261 | 36.8 % | 2 | 98 % | - | - | 0 |
| brute_force | scripted | 24 | 0/24 (0%) | 0-14 % | 0.201 | 36.6 % | 5.5 | 92 % | - | - | 0 |
| fixed_recipe | scripted | 24 | 24/24 (100%) | 86-100 % | 1.000 | 1.9 % | 3 | 74 % | - | - | 0 |
| ir_only | scripted | 24 | 0/24 (0%) | 0-14 % | 0.000 | 69.9 % | 1 | 65 % | - | - | 0 |
| open_loop | scripted | 24 | 24/24 (100%) | 86-100 % | 1.000 | 1.7 % | 3 | 99 % | - | - | 0 |
| prior | scripted | 24 | 0/24 (0%) | 0-14 % | 0.000 | 69.9 % | 0 | 0 % | - | - | 0 |
| reference | scripted | 24 | 24/24 (100%) | 86-100 % | 1.000 | 1.6 % | 3 | 99 % | - | - | 0 |
| seed_attacker | scripted | 24 | 0/24 (0%) | 0-14 % | 0.019 | 122.8 % | 1 | 9 % | - | - | 0 |
| short_tr_ideal | scripted | 24 | 0/24 (0%) | 0-14 % | 0.668 | 10.2 % | 3 | 99 % | - | - | 0 |
| wrong_units | scripted | 24 | 0/24 (0%) | 0-14 % | 0.000 | - | 3 | 99 % | - | - | 0 |

**`Scanner/T1Transmit-v0`, level 2** (24 instances; the admission gate admits 0 of them)

| Agent | Kind | Episodes | Passed | 95 % interval | Mean reward | Median error | Median scans | Median budget used | Median turns | Tokens in / out | Infra errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blind_vfa | scripted | 24 | 0/24 (0%) | 0-14 % | 0.270 | 29.7 % | 2 | 99 % | - | - | 0 |
| brute_force | scripted | 24 | 0/24 (0%) | 0-14 % | 0.259 | 31.6 % | 6 | 90 % | - | - | 0 |
| fixed_recipe | scripted | 24 | 23/24 (96%) | 80-99 % | 0.998 | 1.8 % | 3 | 73 % | - | - | 0 |
| ir_only | scripted | 24 | 0/24 (0%) | 0-14 % | 0.000 | 74.0 % | 1 | 64 % | - | - | 0 |
| open_loop | scripted | 24 | 23/24 (96%) | 80-99 % | 0.999 | 1.4 % | 3 | 98 % | - | - | 0 |
| prior | scripted | 24 | 0/24 (0%) | 0-14 % | 0.000 | 74.0 % | 0 | 0 % | - | - | 0 |
| reference | scripted | 24 | 24/24 (100%) | 86-100 % | 1.000 | 1.4 % | 3 | 98 % | - | - | 0 |
| seed_attacker | scripted | 24 | 0/24 (0%) | 0-14 % | 0.014 | 88.7 % | 1 | 9 % | - | - | 0 |
| short_tr_ideal | scripted | 24 | 0/24 (0%) | 0-14 % | 0.645 | 10.3 % | 3 | 99 % | - | - | 0 |
| wrong_units | scripted | 24 | 0/24 (0%) | 0-14 % | 0.000 | - | 3 | 98 % | - | - | 0 |

**`Scanner/T1Transmit-v0`, level 3** (24 instances; the admission gate admits 0 of them)

| Agent | Kind | Episodes | Passed | 95 % interval | Mean reward | Median error | Median scans | Median budget used | Median turns | Tokens in / out | Infra errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blind_vfa | scripted | 24 | 0/24 (0%) | 0-14 % | 0.256 | 39.0 % | 2 | 97 % | - | - | 0 |
| brute_force | scripted | 24 | 0/24 (0%) | 0-14 % | 0.059 | 65.4 % | 3 | 76 % | - | - | 0 |
| fixed_recipe | scripted | 24 | 24/24 (100%) | 86-100 % | 1.000 | 2.1 % | 3 | 81 % | - | - | 0 |
| ir_only | scripted | 24 | 0/24 (0%) | 0-14 % | 0.000 | 70.8 % | 1 | 87 % | - | - | 0 |
| open_loop | scripted | 24 | 24/24 (100%) | 86-100 % | 1.000 | 2.1 % | 3 | 98 % | - | - | 0 |
| prior | scripted | 24 | 0/24 (0%) | 0-14 % | 0.000 | 70.8 % | 0 | 0 % | - | - | 0 |
| reference | scripted | 24 | 24/24 (100%) | 86-100 % | 1.000 | 2.0 % | 3 | 98 % | - | - | 0 |
| seed_attacker | scripted | 24 | 0/24 (0%) | 0-14 % | 0.011 | 105.7 % | 1 | 14 % | - | - | 0 |
| short_tr_ideal | scripted | 24 | 1/24 (4%) | 1-20 % | 0.699 | 11.5 % | 3 | 99 % | - | - | 0 |
| wrong_units | scripted | 24 | 0/24 (0%) | 0-14 % | 0.000 | - | 3 | 98 % | - | - | 0 |

Passed: episodes that meet the pass rule of the environment, as `k/N (p%)`, with the Wilson 95 % interval. N counts every started episode except those that ended in an endpoint or runner failure; these are not agent results and stand under Infra errors. An episode without a delivery has reward 0. The rows cover every instance, admitted or not: on an instance the admission gate rejects, a shortcut policy can pass, and that is why the gate rejects it (`results.json` also counts the passes on admitted instances). Median error: the error measure of the environment (`mrgym card <id>`), in percent; an episode without a valid deliverable counts as infinite, and `-` means the median is infinite. Median budget used: scan seconds used over the scan-time budget. Turns and tokens are per episode and exist for models only; `(est.)` marks counts estimated from characters because the endpoint reported no usage. Agents are in alphabetical order within kind; the table does not rank.
<!-- RESULTS:END -->

## How these rows were produced

```bash
mrgym run-agent --env Scanner/NoWrap-v0,Scanner/T1Transmit-v0 --seeds 0-23 --policy all --records-only --workers 2 --out results
mrgym results
```

- **Scripted policies, no model.** Each row is a control policy of the environment (`mrgym card <id>` describes
  them), played through the same tools, budget and scoring an agent gets. `reference` is the policy that is meant to
  pass. The others are shortcuts and diagnostic variants; the card says which of them must fail for an instance to
  be admitted. No language model was called to make this folder.
- **One episode per instance.** An instance is an environment, a level and an integer seed of the public train
  split. The environment is deterministic, so a scripted policy plays each instance once. The admission run
  (`mrgym check <id> --admission`) plays the same policies over many noise draws; its numbers are on the environment
  cards and differ slightly from this single-episode table.
- **Constant recipes.** `fixed_recipe` and `open_loop` apply the recipe that the stored admission run tuned on
  other worlds for that environment and level (`"recipe_source": "admission"` in the records).
- **Error columns.** On `Scanner/NoWrap-v0` the error is the scan time of the delivered geometry over the fastest
  valid one, minus 1; a negative value means a scan shorter than any valid one, so the object does not fit. On
  `Scanner/T1Transmit-v0` it is the root-mean-square relative T1 error.
- **Not played.** `ir_unpriced` of `Scanner/T1Transmit-v0` needs 40 times the scan-time budget. It cannot be played
  through the tools and has no row here.
- **`Scanner/T1Transmit-v0` is a practice environment.** One constant protocol passes it, which is why
  `fixed_recipe` and `open_loop` stand next to `reference` there. Its pass tolerance per instance is the one stored
  by the admission run.

## Add a model

A model plays the same instances through any OpenAI-compatible chat-completions endpoint, and its rows land in the
same table:

```bash
export OPENAI_API_KEY=...
mrgym run-agent --env Scanner/NoWrap-v0,Scanner/T1Transmit-v0 --seeds 0-23 --model MODEL --base-url https://HOST/v1 --api-key-env OPENAI_API_KEY --workers 4 --out results
mrgym results
```

`--dry-run` prints exactly what the model would receive and calls nothing. Repeat the command to resume: finished
episodes are skipped, and episodes lost to an endpoint failure are played again. A model episode also saves its
conversation (`messages.json`) and the scanner log (`scanner_log.jsonl`) next to the record; add `--records-only` to
keep this folder to records. The offline self-test of the runner needs no model:
`python -m pytest tests/test_runner_stub.py`.

## Files

| File | Content |
|---|---|
| `episodes/<agent>/<env>/L<level>/s<seed>/e<k>/record.json` | one episode: agent and kind, model and parameters sent or dropped, environment, level, seed, engine and package version, status, stop reason, reward, pass, error, gates, scans, scan seconds used and budget, turns, tool calls, invalid calls, tokens, head of the scanner log, wall seconds, UTC time |
| `results.json` | provenance read from the records and one row per environment, level and agent |
| `run_config.json` | what each invocation of `mrgym run-agent` was asked to do |
