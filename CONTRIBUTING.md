# Contributing to MR Gym

MR Gym is a virtual MRI scanner you can drop an AI agent into. Three kinds of change are useful: a scanner that is
more faithful, an environment that forces a decision on the scanner, and a policy that earns reward without
measuring. Rules for coding agents are in [AGENTS.md](AGENTS.md); they apply to people too.

## Set up

```sh
git clone https://github.com/KeWang0622/MRGym && cd MRGym
python -m pip install -e .
python -m pip install "pytest>=8" "ruff>=0.15"     # dev tools; add "matplotlib>=3.7" for the figures
make test                                          # or: python -m pytest
mrgym demo                                         # one episode on the scanner, printed
```

Python 3.10 or newer. `make help` lists the other targets.

## What every pull request needs

1. Tests that pass: `make test` and `make lint`. Tests need no network, no credentials and no model.
2. Generated files regenerated, not edited: `make generate`, then `make check`.
3. An answer to the question in the pull-request template: can this change any reward or any scan data?
4. A line under "Unreleased" in [CHANGELOG.md](CHANGELOG.md) when a user could notice the change.
5. Numbers that come from a command, with the command. A physics change comes with `mrgym validate`; an environment
   or scoring change comes with the table of `mrgym check <id> --admission`.
6. No held-out seed, secret or episode config in the diff, the logs or the screenshots.

## The engine version

`mrgym.ENGINE_ID` names the numerical behaviour of the scanner: world generation, signal models, k-space formation,
console arithmetic and noise streams. It is a constant in `src/mrgym/scanner/session.py`.

- Bump it by hand when the same seed and protocol can give other data, and only then.
- `tests/test_engine_version.py` pins the id together with three reference numbers. If the numbers move, the id
  must move with them, and the changelog gets an "Engine" line.
- A recorded log replays only on the engine that wrote it. A change of the engine means stored episodes are re-run.
- A comment, a rename or a reformat of an engine file does not change the id.

## Dependencies

- Core (`src/mrgym`): numpy and scipy. Nothing else. The HTTP service uses `http.server`; the model runner speaks
  the OpenAI-compatible chat API over `urllib`.
- Figures: the optional extra `viz` (matplotlib). The episode viewer does not need it.
- Agent frameworks are not dependencies. Adapters live under [integrations/](integrations/README.md).
- `tests/test_packaging.py` fails when importing `mrgym` loads anything beyond the core.

## Add an environment

An environment is a job done on the scanner: a hidden world, a scan-time budget, a deliverable and a score. The
worked example is `src/mrgym/envs/nowrap/`. `src/mrgym/envs/t1_transmit/` is the worked example of a rejection.

1. **World and task.** Write `src/mrgym/envs/<name>/task.py` with the world spec per level, `make_instance(seed,
   level, task_id)`, `statement(cfg)` and `score(...)`. The score has hard gates (a valid deliverable, scans in the
   log, units, budget) and then a continuous part that is exactly 0 at the no-scan answer.
2. **Control policies.** Write `controls.py` against the scanner interface only (`info`, `plan`, `scan`, `view`,
   `read_regions`, `budget`). It holds a reference policy that must pass and shortcut policies that must fail: the
   no-scan prior, the best constant recipe tuned on other worlds, a blind sweep, the right numbers in the wrong
   units, a seed attack, and the environment's own confounder-blind policies. It also provides `admit` (the rule
   for one instance) and `tune_recipe` (how the constant recipe is tuned).
3. **Environment class.** Write `env.py` with a subclass of `ScannerEnv` that implements `make_instance`,
   `task_text`, `terminal_tool` and `finish`, and a `card()` function.
4. **Register the id** in `src/mrgym/envs/__init__.py` with `kind="practice"`. It stays practice until the gate
   admits instances.
5. **Run the gate and read the table.**

   ```sh
   mrgym check <id> --admission --n 24 --write
   ```

   The status is whatever the gate says. An environment that its gate rejects is listed as rejected.
6. **Run the structural check.** `mrgym check <id>` must pass.
7. **Regenerate.** `python tools/generate.py` rewrites the card under `envs/` and the table in the README.
8. **Tests.** At least: instances are reproducible from the seed; nothing hidden is served; the no-scan answer
   scores exactly 0; the reference policy passes one episode through the tools.
9. **Changelog.** Add a line under "Unreleased". If a reward of an existing id can change, register a new `-vN`
   and leave the old id as it is.

A new sequence or a new physical effect is a separate pull request against `src/mrgym/scanner/`, with its own
rows in `mrgym validate` and an engine-version bump.

## Report a shortcut

A policy that earns reward without handling what the environment is about is a result. If it works by reading the
seed or the world, or by changing the budget or the log, report it privately ([SECURITY.md](SECURITY.md)). If it
is a legitimate physical shortcut, open an issue with the policy and its pass rate. It becomes a control policy of
that environment.
