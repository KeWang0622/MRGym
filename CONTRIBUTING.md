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

1. Tests that pass: `make test` and `make lint`. Tests need no network, no credentials and no model. A change of
   scoring, gate or engine code also passes `make audit` (below).
2. Generated files regenerated, not edited: `make generate`, then `make check` (it also checks the links of the pages).
3. An answer to the question in the pull-request template: can this change any reward or any scan data?
4. A line under "Unreleased" in [CHANGELOG.md](CHANGELOG.md) when a user could notice the change.
5. Numbers that come from a command, with the command. A physics change comes with `mrgym validate`; an environment
   or scoring change comes with the table of `mrgym gate <id>` and with its scoring table.
6. No held-out seed, secret or episode config in the diff, the logs or the screenshots.

## Four version numbers, and when each moves

| Number | Where | Bump it when |
|---|---|---|
| package | `[project] version` in `pyproject.toml` and `mrgym.__version__` (the same string) | every release: `0.y.z`, `y` for a change of the API, `z` for fixes |
| engine | `ENGINE_VERSION` in `src/mrgym/scanner/session.py` | the same seed and protocol can give other scan data, a scan can cost another time, or the log changes |
| environment id | the `-vN` suffix | a reward, the pass rule, the observation text or the instance distribution of a released id can change; the old id stays registered |
| log | `LOG_VERSION` in `src/mrgym/scanner/session.py` | the format of a log record changes |

A deprecation warns with `DeprecationWarning` naming the release that removes it, for at least two minor releases,
and is listed in the changelog. The tests turn a `DeprecationWarning` raised by `mrgym` into an error.

## The engine version and the golden data

`mrgym.ENGINE_ID` names the numerical behaviour of the scanner: world generation, signal models, k-space formation,
console arithmetic and noise streams.

- Bump it by hand when the same seed and protocol can give other data, and only then. A comment, a rename or a
  reformat of an engine file does not change it.
- `tests/golden/engine_digests.json` pins the engine: a noise-free image of every sequence on every phantom at both
  field strengths, a session of noisy scans with its log chain, and a price table. `tests/test_golden.py` compares
  bit for bit on the stack the file was recorded on (numpy, scipy, machine, SIMD extensions) and within the replay
  tolerance elsewhere; the header of every test run says which.
- If the numbers move while `ENGINE_VERSION` stays, the test fails and `tools/audit/golden.py --write` refuses to
  record them. Bump the version first; then one command regenerates everything:

  ```sh
  python tools/audit/golden.py --write
  ```

- `tests/golden/task_snapshots.json` pins what each task says and how it is parameterised: the instance of seed 0
  and the statement per level, the submit schema, the card, every module-level constant of the task module, the
  constants of the gate, and one episode of the reference policy. The same command rewrites it and lists the ids it
  touched. Review that diff: a changed snapshot of a released id needs a new `-vN`.
- A recorded log replays only on the engine that wrote it. A change of the engine means stored episodes are re-run.
- `tests/test_engine_version.py` still pins the version string and three numbers of one image; the golden file is the
  wider net.

## Dependencies

- Core (`src/mrgym`): numpy and scipy. Nothing else. The HTTP service uses `http.server`; the model runner speaks
  the OpenAI-compatible chat API over `urllib`.
- Figures: the optional extra `viz` (matplotlib). The episode viewer does not need it.
- Agent frameworks are not dependencies. Adapters live under [integrations/](integrations/README.md).
- `tests/test_packaging.py` fails when importing `mrgym` loads anything beyond the core.

## Add an environment

An environment is a job done on the scanner: a hidden world, a scan-time budget, a deliverable and a pass rule.
**Start with [docs/ADDING_A_TASK.md](docs/ADDING_A_TASK.md)**: it walks through a small family that runs
(`examples/new_family/`). A task family is a sub-package `src/mrgym/envs/<family>/`; the full-size example is
`src/mrgym/envs/null_tissue/`, and `src/mrgym/envs/family.py` states the contract. (`nowrap` and `t1_transmit` predate
it and are kept as they are.)

1. **`task.py`.** The world spec per level, `make_instance(seed, level, task_id)`, `statement(cfg)`,
   `submit_schema(cfg)`, `check(cfg, answer, exam)` (why a deliverable cannot be evaluated, or `None`),
   `score(cfg, answer, world, engine)` (a dict with `passed`, `checks`, `diagnostic`) and `card()`. The reward is 1 for
   a pass and 0 otherwise: there is no partial credit, and continuous quantities go to `info["score"]`.
2. **`controls.py`.** Policies written against the scanner interface only (`info`, `plan`, `scan`, `view`,
   `read_regions`, `budget`): a reference that must pass, the policies that must fail (the no-scan answer, the best
   answer that uses only what the task states, a sweep without a model, the wrong unit), and the ablations with the
   levels at which each must lose.
3. **`spec.py`.** `SPECS`, a list of keyword dicts for `register`, with `kind="candidate"`. The sub-package is found
   by listing the directory. One line in one shared file places the family on the map: `FAMILIES` in
   `src/mrgym/physics_map.py` (the kind of task, the one item it isolates, the items it uses, the families it builds
   on; a family listed under `PLANNED` moves its line). `mrgym map --assert` checks it.
4. **A scoring table** (next section) and the snapshot of the task: `python tools/audit/golden.py --write`.
5. **Run the gate and read the table.** The status is whatever the gate says.

   ```sh
   mrgym gate <id> --n 300 --write
   ```
6. **Run the structural check.** `mrgym check <id>` must pass.
7. **Regenerate.** `python tools/generate.py` rewrites the card under `envs/`, the task table of the README and the
   map (`docs/tasks.md`).
8. **Changelog.** Add a line under "Unreleased". If a reward of a released id can change, register a new `-vN` and
   leave the old id as it is.

A new sequence or a new physical effect is a separate pull request against `src/mrgym/scanner/`, with its own
rows in `mrgym validate`, an engine-version bump and regenerated golden data.

## Scoring table

The reward of a family is tested as a table: `tests/scoring/<family>.py`, named after the sub-package. The file
defines three names and nothing else is needed; `tests/test_scoring_table.py` finds it.

```python
from scoring import Row

ENV_ID = "Scanner/NullTissue-v0"
INSTANCES = {"L1": dict(level=1, seed=0), "L2": dict(level=2, seed=0)}       # public training seeds

ROWS = [
    Row("L1", "every region at its null", at(), passed=True, expect=agrees_with_the_chain),
    Row("L1", "one region 10 % early", at(R1=0.90), passed=False, failing=("nulled_R1",), reason="R1 keeps"),
    Row("L1", "the right numbers in seconds", in_seconds, refused=True, reason="the console rejects the target scan"),
]
```

| Field | Meaning |
|---|---|
| `instance` | key of `INSTANCES` |
| `name` | what the row shows; it becomes the test id |
| `deliverable` | the arguments of `submit`, or `callable(case) -> arguments` when it depends on the hidden world. `case` holds `cfg`, `world`, `engine`, `exam`, `truth` (the true tissue values of the regions) |
| `passed` | `True` or `False`: the verdict of `task.score` on the hidden world |
| `failing` | the checks that must be false, exactly these (empty for a pass) |
| `reason` | text that must appear in the diagnostic of a fail, or in the refusal |
| `refused` | `True`: `task.check` cannot evaluate the deliverable; the episode continues |
| `expect` | optional `callable(case, score)` with further asserts; `score` is the score dict plus `answer` |

Rules:

- **The expected value is derived in the fixture file** from the documented pass rule and the hidden truth, with
  arithmetic written there (the NullTissue table walks the magnetisation through the pulse chain). It never comes
  from `task.score`, `task.check`, a helper of the task module, or a recorded output of them.
- A table needs at least one passing, one failing and one refused row; one failing row must sit on the same
  instance as a passing row. Put the rows a careful and a careless agent would produce: the right answer, the answer
  just inside and just outside the tolerance, each requirement failing alone, the wrong unit, the textbook shortcut
  the levels are built to defeat.
- Keep it fast: a few instances, and deliverables that share scans (the verifier's engine caches them).

What the harness adds for every family, so no table has to: the two gates of `family.py` (the log replays within
budget, at least one accepted scan) in every combination; a pass without a scan earns 0; a refusal is an error
observation and does not end the episode; the reward is exactly 0.0 or 1.0; the agent is not shown the score; the
verifier gives the same verdict from the episode config alone; the task and controls modules define the contract.

Edges that a table cannot reach (a threshold hit exactly, a search routine) go into a small test file of the family;
`tests/test_null_tissue.py` is the example.

## What `make audit` checks

```sh
python -m pip install -r tools/ci/requirements-audit.txt
make audit                                   # about 25 minutes on 2 cores; writes audit_report.json
python tools/audit/report.py --list          # the rows, their commands, their pass criteria
```

| Row | Passes when |
|---|---|
| lint, types | `ruff check` is clean; no file of the typed scope has more `mypy` errors than `tools/audit/mypy_baseline.json` |
| suite, coverage | 0 failed with deprecations as errors; every skip and expected failure is listed in `tools/audit/allowed_*.txt`; total coverage not below the ratchet in `tools/audit/coverage_baseline.json` (it only rises: `python tools/audit/report.py --only suite,coverage --write-coverage`) and 100 % on the scoring code; the row lists the three least-covered modules |
| contract | the `reset` / `step` / `submit` state machine of `tests/test_contract_stateful.py` holds over 200 examples of 20 steps |
| metamorphic | the 20 relations of `tests/test_metamorphic.py` hold within their tolerances |
| golden, tables | the golden files are current; every family id has a scoring table and a snapshot (a family without one is named in the row) |
| isolation | the tests of keys and splits, the MAC'd log, the red-team probes, epochs and the service pass |
| mutation | at least 85 % of all mutants of the scoring code are killed, in under 10 minutes (`tools/audit/mutation.py`) |
| handmutants | the 16 defects of the engineering review of 2026-11-24 and 6 further edits (constants, the slice-selective path) are each noticed (`tools/audit/hand_mutants.py`) |
| wheel, generated | the wheel holds exactly the tracked package files; generated files are current; every README command exits 0 |

Scope of the mutation gate (`[tool.mutmut]` in `pyproject.toml`): `envs/family.py`, `gate.py` and every
`envs/*/task.py`. A new family is in scope the day it is added. Three things to know:

- The tool mutates code inside functions only. Module-level constants (thresholds, level tables) are pinned by the
  task snapshot and by the constant mutants of `tools/audit/hand_mutants.py`.
- A surviving mutant is either killed by a new row or test, or matched by a rule in
  `tools/audit/mutation_classes.txt` that says why it cannot change a result. Do not weaken an assert to move the
  score, and do not add `# pragma: no mutate` (the run fails on a pragma that is not in
  `tools/audit/no_mutate_allowlist.txt`).
- The table of `tests/test_metamorphic.py` says for every relation whether it is run on slice-selective scans, on
  hard-pulse scans (`excitation: nonselective`) or both; `pytest -m hard_pulse` lists the ones that are statements
  about hard pulses.
- Every command of the audit runs under an address-space limit (`MRGYM_AUDIT_MEM_GB`, default 4) and the report gives
  the peak resident memory of every row. A test file that needs more than 1.5 GB is a defect: an episode holds a
  hidden world, an engine and its scans, so a test that plays many episodes in one process closes them, drops the
  references and keeps generated scans small (`tests/test_contract_stateful.py` is the example).

## Report a shortcut

A policy that earns reward without handling what the environment is about is a result. If it works by reading the
seed or the world, or by changing the budget or the log, report it privately ([SECURITY.md](SECURITY.md)). If it
is a legitimate physical shortcut, open an issue with the policy and its pass rate. It becomes a control policy of
that environment.
