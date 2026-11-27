# Results: the pilot archive

**Nothing is published yet.** This page is the archive of the pilot runs of package 0.1.0, played on the retired ids
`Scanner/NoWrap-v0` and `Scanner/T1Transmit-v0` with engine `mrgym-engine-0.1.0`. No current environment has a row.

- **What is stored here:** the table below, [`results.json`](results.json) (the same rows as data) and
  [`run_config.json`](run_config.json) (the commands that were run).
- **What is not stored:** the raw episode records. They were of schema 1 (no conversation kept) and cannot be replayed
  by this package, which no longer serves those ids. Commit `294ae25` is the last one that holds them, under
  `results/episodes/`.
- **What the check covers:** `mrgym results --check --dir docs/results` compares the table with `results.json` and
  applies the reporting rules to both. It compares nothing with records and says so.
- **A run of your own** writes a folder with its records; there the same command rebuilds the table from them
  ("Run a model for a published row" below).

A row is one agent on one environment, with the levels as strata. Rows come in two tiers. A **published** row has
records of schema 2 (the whole conversation, the harness id, the delivered answer) on enough instances for a
comparison; it is held to the reporting rules below. A **pilot** row is everything else: kept with its interval, under
its own heading, and never compared.

<!-- RESULTS:START -->
1400 episode records, played 2026-11-24 (UTC) on engine `mrgym-engine-0.1.0`, package version `0.1.0.dev0`. Environments: `Scanner/NoWrap-v0` levels 1, 2; `Scanner/T1Transmit-v0` levels 1, 2, 3. Agents: scripted control policies and 4 language models (claude-haiku-5-5, claude-sonnet-5-5, gemini-3.8-flash, kimi-k3). Published: 0 records; pilot: 1400 records.

### Published table

No row is published yet. A published row needs records of schema 2 (stored conversation, harness id, deliverable), at least 300 instances per environment on the reporting instances of the analysis plan, and 2 episodes per instance for a model.

### Pilot rows (not published)

Kept for the record. They carry intervals, but they are too small or lack what a published row must store, they are absent from the README table, and no comparison is computed from them. Why pilot: `schema 1` = the records hold no stored conversation, harness id or deliverable; `< 100 instances` = too few instances for a comparison.

**`Scanner/NoWrap-v0`** (pilot; levels 1, 2 pooled as strata)

| Agent | Kind | Instances | Passed: k/N (rate; 95 % interval) | Mean reward | Median error | Median scans | Median turns | Tokens in / out, median | Infra errors | Why pilot |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| brute_force | scripted | 48 | 3/48 (6.2 %; 2.1-17.2) | 0.588 | 33.8 % | 3 | - | - | 0 | schema 1 |
| console_default | scripted | 48 | 0/48 (0.0 %; 0.0-7.8; one-sided bound <= 6.1 %) | 0.207 | 60.8 % | 1 | - | - | 0 | schema 1 |
| fixed_recipe | scripted | 48 | 11/48 (22.9 %; 13.0-37.2) | 0.330 | 2.2 % | 1 | - | - | 0 | schema 1 |
| no_swap | scripted | 48 | 29/48 (60.4 %; 45.8-73.4) | 0.892 | 5.6 % | 1 | - | - | 0 | schema 1 |
| prior | scripted | 48 | 0/48 (0.0 %; 0.0-7.8; one-sided bound <= 6.1 %) | 0.000 | 73.4 % | 0 | - | - | 0 | schema 1 |
| reference | scripted | 48 | 48/48 (100.0 %; 92.2-100.0; one-sided bound >= 93.9 %) | 1.000 | 0.0 % | 1 | - | - | 0 | schema 1 |
| safe_max | scripted | 48 | 0/48 (0.0 %; 0.0-7.8; one-sided bound <= 6.1 %) | 0.000 | 73.4 % | 1 | - | - | 0 | schema 1 |
| seed_attacker | scripted | 48 | 7/48 (14.6 %; 7.1-27.6) | 0.146 | -15.2 % | 1 | - | - | 0 | schema 1 |
| size_only | scripted | 48 | 10/48 (20.8 %; 11.5-34.7) | 0.208 | -5.0 % | 1 | - | - | 0 | schema 1 |
| wrong_units | scripted | 48 | 0/48 (0.0 %; 0.0-7.8; one-sided bound <= 6.1 %) | 0.000 | - | 1 | - | - | 0 | schema 1 |
| claude-haiku-5-5 | model | 20 | 11/20 (55.0 %; 32.0-76.0) | 0.769 | 5.3 % | 2 | 7 | 93.4k / 9.7k | 0 | schema 1 |
| claude-sonnet-5-5 | model | 20 | 11/20 (55.0 %; 32.9-75.3) | 0.974 | 9.1 % | 1 | 4 | 27.6k / 0.7k | 0 | schema 1 |
| gemini-3.8-flash | model | 20 | 0/20 (0.0 %; 0.0-18.1; one-sided bound <= 13.9 %) | 0.123 | - | 1 | 12 | 94.7k / 10.1k | 0 | schema 1 |
| kimi-k3 | model | 20 | 9/20 (45.0 %; 24.0-68.0) | 0.659 | 3.9 % | 3 | 12 | 120.5k / 19.6k | 0 | schema 1 |

**`Scanner/T1Transmit-v0`** (pilot; levels 1, 2, 3 pooled as strata)

| Agent | Kind | Instances | Passed: k/N (rate; 95 % interval) | Mean reward | Median error | Median scans | Median turns | Tokens in / out, median | Infra errors | Why pilot |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| blind_vfa | scripted | 72 | 0/72 (0.0 %; 0.0-5.2; one-sided bound <= 4.1 %) | 0.262 | 33.6 % | 2 | - | - | 0 | schema 1 |
| brute_force | scripted | 72 | 0/72 (0.0 %; 0.0-5.2; one-sided bound <= 4.1 %) | 0.173 | 46.5 % | 4 | - | - | 0 | schema 1 |
| fixed_recipe | scripted | 72 | 71/72 (98.6 %; 92.3-99.8) | 0.999 | 2.0 % | 3 | - | - | 0 | schema 1 |
| ir_only | scripted | 72 | 0/72 (0.0 %; 0.0-5.2; one-sided bound <= 4.1 %) | 0.000 | 71.5 % | 1 | - | - | 0 | schema 1 |
| open_loop | scripted | 72 | 71/72 (98.6 %; 92.3-99.8) | 1.000 | 1.7 % | 3 | - | - | 0 | schema 1 |
| prior | scripted | 72 | 0/72 (0.0 %; 0.0-5.2; one-sided bound <= 4.1 %) | 0.000 | 71.5 % | 0 | - | - | 0 | schema 1 |
| reference | scripted | 72 | 72/72 (100.0 %; 94.8-100.0; one-sided bound >= 95.9 %) | 1.000 | 1.7 % | 3 | - | - | 0 | schema 1 |
| seed_attacker | scripted | 72 | 0/72 (0.0 %; 0.0-5.2; one-sided bound <= 4.1 %) | 0.015 | 106.8 % | 1 | - | - | 0 | schema 1 |
| short_tr_ideal | scripted | 72 | 1/72 (1.4 %; 0.2-7.7) | 0.671 | 10.5 % | 3 | - | - | 0 | schema 1 |
| wrong_units | scripted | 72 | 0/72 (0.0 %; 0.0-5.2; one-sided bound <= 4.1 %) | 0.000 | - | 3 | - | - | 0 | schema 1 |
| claude-haiku-5-5 | model | 30 | 0/30 (0.0 %; 0.0-12.3; one-sided bound <= 9.5 %) | 0.168 | 49.7 % | 2 | 12 | 164.4k / 12.2k | 0 | schema 1 |
| claude-sonnet-5-5 | model | 30 | 1/30 (3.3 %; 0.5-18.0) | 0.498 | 15.4 % | 3 | 6 | 50.7k / 2.3k | 0 | schema 1 |
| gemini-3.8-flash | model | 30 | 0/30 (0.0 %; 0.0-12.3; one-sided bound <= 9.5 %) | 0.187 | 58.6 % | 1 | 12 | 128.2k / 8.5k | 0 | schema 1 |
| kimi-k3 | model | 30 | 4/30 (13.3 %; 5.1-30.6) | 0.536 | 14.3 % | 4 | 12 | 153.0k / 22.9k | 0 | schema 1 |

### Reliability of the published rows

No records to compute reliability from.

### Design and planned comparisons

What the designs detect (points of pass rate; p = 0.5, two-sided alpha = 0.05, power 0.80; computed by `mrgym stats power`):

| Design (instances x episodes) | 95 % half-width of one rate | Detectable paired difference, rho = 0.07, r = 0 | same, Holm over 10 | rho = 0.3, r = 0.5 | Use |
|---|---:|---:|---:|---:|---|
| 10 x 1 | 35.8 | 62.6 | 81.6 | 57.8 | none |
| 24 x 1 | 21.1 | 40.4 | 52.7 | 37.3 | none |
| 96 x 8 | 4.4 | 8.7 | 11.4 | 9.9 | pilot for rho |
| 100 x 2 | 7.3 | 14.5 | 18.9 | 14.0 | descriptive |
| 300 x 1 | 5.7 | 11.4 | 14.9 | 10.5 |  |
| 300 x 2 | 4.2 | 8.4 | 10.9 | 8.1 | published minimum |
| 600 x 2 | 2.9 | 5.9 | 7.7 | 5.7 | trained against base |
| 1500 x 2 | 1.9 | 3.7 | 4.9 | 3.6 | suite score |

No analysis plan is registered in this folder (`analysis_plan.json`): no comparison is computed.

Settings of `claude-haiku-5-5`, `claude-sonnet-5-5`, `gemini-3.8-flash`, `kimi-k3`: at most 12 turns per episode, session `calc` tool on, temperature -, max tokens - (`-` = not sent, the endpoint's default).

How to read a row. The unit is the instance: N counts episodes, the interval counts instances. A rate is printed as `k/N (rate; 95 % interval)`; the interval is the Wilson interval on the Korn-Graubard effective sample size with a t quantile on instances minus strata (`mrgym.stats.cluster_rate`); a cell with fewer than 100 instances also gets the Beta-binomial interval and the wider of the two is printed. At 0 or 100 % the one-sided 95 % bound stands beside the rate. N counts every started episode except those that ended in an endpoint or runner failure (Infra errors); an episode without a delivery has reward 0. pass^2: share of instances whose episodes all pass. Regret: distance of the delivered answer from the oracle's optimum of the same instance (`-`: no oracle values stored). Per-level cells are descriptive: no comparison is computed from them. Agents are in alphabetical order within kind; the table does not rank. A difference that is not in the comparison table is not a finding.
<!-- RESULTS:END -->

## How the pilot rows were produced

```bash
mrgym run-agent --env Scanner/NoWrap-v0,Scanner/T1Transmit-v0 --seeds 0-23 --policy all --records-only --workers 2 --out results
mrgym results
```

- **Scripted policies, no model.** Each scripted row is a control policy of the environment (`mrgym card <id>`
  describes them), played through the same tools, budget and scoring an agent gets. `reference` is the policy that
  is meant to pass. The others are shortcuts and diagnostic variants.
- **Model rows.** A language model behind a chat-completions endpoint, played by `mrgym run-agent` on a subset of the
  seeds of the scripted rows, one episode per instance, without sampling settings sent. The records are of schema 1:
  the conversations were not kept. These rows are pilot data and will be replaced.
- **One episode per instance.** The environment is deterministic, so a scripted policy plays each instance once.
- **Constant recipes.** `fixed_recipe` and `open_loop` apply the recipe that the admission of package 0.1.0 tuned on
  other worlds for that environment and level (`"recipe_source": "admission"` in the records).
- **Error columns.** On `Scanner/NoWrap-v0` the error is the scan time of the delivered geometry over the verifier's
  fastest valid one, minus 1; a negative value means a scan shorter than that. On `Scanner/T1Transmit-v0` it is the
  root-mean-square relative T1 error.
- **Not played.** `ir_unpriced` of `Scanner/T1Transmit-v0` needs a multiple of the scan-time budget. It cannot be
  played through the tools and has no row here.

## What a published row needs

The rules are checked by `mrgym results --check`; each message names the rule, the file and the line.

| Rule | A published row, or the text about it, fails when |
|---|---|
| R1 | a rate has no instance count, episode count or interval; a rate is typed by hand outside a generated block or a `<!-- STAT:key -->` span |
| R2 | episodes outnumber instances and the interval is not clustered; the effective sample size exceeds the episodes |
| R3 | a sentence compares two agents, levels or environments without a planned comparison (`<!-- CMP:id -->`) whose Holm-adjusted p is below 0.05 |
| R4 | the seeds, the split or the seed-set hash is missing; two rows of one table played different instances; the seed set is not the one of the analysis plan |
| R5 | the command, package version, engine, model id, provider, date, or the temperature, token limit and turn cap as sent are missing |
| R6 | a rate of 0 or 100 %, or "always" / "never" about a rate, stands without its one-sided bound |
| R7 | a comparison uses a pilot row |
| R8 | the environment table lacks the reference, a blind policy, an ablation, the no-scan answer, or the regret column |
| R9 | a rate is computed on a subset chosen by outcome (admitted only, valid only, best of several) without the full-set rate |
| R10 | infrastructure errors exceed 2 % of a cell, or differ by more than 2 points between two compared agents |
| R11 | no analysis plan is registered, a record is older than the plan, or the comparisons are not the planned family |
| R12 | a model row has no token totals or wall time |
| R13 | a record lacks the stored conversation, the delivered answer, the outcome or the harness id, or a conversation file does not have the hash stored in its record |
| D1 | fewer than 300 instances per environment, or fewer than 2 episodes on an instance for a model |

The statistics behind the table (the interval, the paired test, the design and its power, the stop rules) are in
[docs/ANALYSIS_PLAN.md](../ANALYSIS_PLAN.md).

## Run a model for a published row

```bash
mrgym manifest report --check --dir docs/results/manifests       # the frozen reporting instances
mrgym results --register-plan --models MODEL_A,MODEL_B --dir runs/baseline   # the plan, before the first episode
mrgym run-agent --env all --policy all --model MODEL_A --model MODEL_B --episodes 2 --temperature T --max-tokens N \
      --base-url https://HOST/v1 --api-key-env OPENAI_API_KEY --workers 4 --out runs/baseline
mrgym results --oracle --dir runs/baseline                       # the oracle's optimum per instance (regret column)
mrgym results --dir runs/baseline && mrgym results --check --dir runs/baseline
mrgym export --dir runs/baseline --out runs/baseline/episodes.jsonl
```

The instances of a published row are those of the `report` manifest
([manifests/report.json](manifests/report.json)). `mrgym run-agent` takes integer seeds today; playing the seeds of a
manifest needs the runner's suite option, which is not part of this folder's tooling yet. `--dry-run` prints exactly
what the model would receive and calls nothing. Repeat the command to resume: finished episodes are skipped, and
episodes lost to an endpoint failure are played again. Every model episode saves its conversation
(`messages.json`: prompt, every model turn with its reasoning text when the endpoint returns it, every tool call and
result, usage and latency per request) and the scanner log next to the record. A dollar figure is printed only when
`prices.json` (model, date, USD per million tokens in and out, source URL) is in the folder. The offline self-test of
the runner needs no model: `python -m pytest tests/test_runner_stub.py tests/test_records.py`.

## Files

Stored in this folder:

| File | Content |
|---|---|
| `results.json` | `mrgym-results/2`: provenance, one row per environment and agent with its tier, the planned comparisons, reliability |
| `run_config.json` | what each invocation of `mrgym run-agent` was asked to do |
| `manifests/<suite>.json` | the frozen instance lists (`mrgym manifest <suite> --check --dir docs/results/manifests`) |

Written into a run folder by `mrgym run-agent` and `mrgym results` (none of these is stored here):

| File | Content |
|---|---|
| `episodes/<agent>/<env>/L<level>/s<seed>/e<k>/record.json` | one episode (`mrgym-episode/2`): agent and kind, model, provider, sampling settings requested, sent and dropped, harness id and configuration, environment, level, seed, split, engine and package version, status, stop reason, reward, pass, error, checks, outcome, the delivered answer, scans, scan seconds used and budget, turns, tool calls, tokens in / out / reasoning, wall and model seconds, head of the scanner log, hash and counts of the stored conversation, the command and the hash of the analysis plan, the canary string. Records of `mrgym-episode/1` lack most of this and are pilot data |
| `episodes/.../messages.json` | the stored conversation of a model episode (`mrgym-transcript/1`) |
| `episodes/.../scanner_log.jsonl` | the scanner's hash-chained log of the episode |
| `analysis_plan.json` | the design, the planned comparisons and the stop rules, registered before the run |
| `oracle.json`, `prices.json` | optional: the oracle's optimum per instance; the price list used for the cost column |
