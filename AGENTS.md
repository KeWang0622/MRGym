# AGENTS.md

> [!IMPORTANT]
> This file is for developing MR Gym, not for acting in it. An agent inside an episode sees the tools of one
> environment and nothing else. It never runs inside a clone of this repository: the repository holds the world
> generator and the score functions.

## Hard rule: no MRI Agent Bench material

[MRI Agent Bench](https://github.com/KeWang0622/MRIAgentBench) is the held-out evaluation. MR Gym is for practice
and training. No task, data, grader or canary from that repository may enter this one, in any form: not copied, not
paraphrased, not as a test fixture. `tests/test_no_benchmark_content.py` enforces what can be checked by machine.
If a change needs something from the benchmark, stop and ask the maintainer.

## Setup

```sh
python -m pip install -e .
python -m pip install "pytest>=8" "ruff>=0.15" "matplotlib>=3.7"
```

Python 3.10 or newer. The core imports numpy, scipy and the standard library only. Tests run without network
access, credentials or a model.

## Commands

| Command | Does |
|---|---|
| `make test` | the test suite (`python -m pytest`) |
| `make lint` | `ruff check` |
| `make generate` | regenerates the environment cards under `envs/` and the generated tables of the README and `docs/TOUR.md` (`python tools/generate.py`) |
| `make check` | fails when a generated file is stale, when a link of a page is broken, when the results archive does not match its table, or when `mrgym check` fails on a registered id |
| `make figures` | redraws the figures (`python tools/figures/make_all.py`; needs matplotlib) |
| `mrgym validate` | physics checks against analytic formulas, as a table |
| `mrgym check <id>` | what must hold before an id is published; compares its kind with the stored gate result |
| `mrgym gate <id>` | the control policies on fresh instances, one verdict per level; `--write` stores the table |

## Rules for changes

- **Ids are permanent.** An environment id is never renamed, renumbered or reused. A change that can move a reward
  registers the next `-vN`; the old id keeps its meaning.
- **Generated files are regenerated, not edited.** Cards under `envs/`, the tables between markers in the README and in `docs/`
  and the results table come from `make generate`. Edit the source and run it again.
- **Every number in the README and in `docs/` comes from a command.** State the command. Do not type a number
  from a terminal into a document, and do not round one by hand.
- **The engine version is bumped by hand when scan data can change.** `ENGINE_VERSION` in
  `src/mrgym/scanner/session.py`, together with the reference numbers in `tests/test_engine_version.py` and an
  "Engine" line in the changelog. Not for comments, renames or formatting.
- **The status of an environment is what its gate says.** Do not set `kind="admitted"` by hand ahead of
  `mrgym gate <id> --n 300 --write`. An environment whose gate rejects a level is listed as practice.
- **The core stays numpy and scipy.** No new dependency in `src/mrgym` without the maintainer's agreement.
- **No internal references.** No working notes, local paths or private file names in code, comments or documents.
  `tests/test_hygiene.py` checks the patterns it knows.
- **Say what was not run.** A check that was skipped, or a result that was not verified, is stated in the pull
  request.

## Pull requests

- Work on a branch and open a pull request against `main`. Fill in the template, including the question about
  rewards and scan data.
- Agents open pull requests. They do not merge them, tag releases or publish packages.
- One change per pull request. A physics change, an environment change and a documentation change are three.
