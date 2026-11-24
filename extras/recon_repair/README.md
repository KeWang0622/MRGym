# Recon-repair (extra)

A set of generated code-repair tasks in an MR setting. Each task hands an agent a reconstruction script `recon.py`,
a note `FORMAT.md` that describes one scanner's raw-data format, and two example scans. The script runs but
disagrees with the note in one to three places. The agent repairs the script; the repaired script is then run on
scans it has not seen and each image is compared with the simulated object.

This is code repair against a written specification. Nothing is acquired: the agent never operates the scanner.
For that reason it is not part of `mrgym`, is not installed with it, does not import it, and is not counted in any
MR Gym result. The tasks are unchanged since they left the main package; only names, imports and the network
setting of the generated `task.toml` were edited.

## Read these two warnings first

1. **Run it with the network closed.** The reference solution is a function of `FORMAT.md`, which the agent sees,
   and of the generator in this folder, which is public. An agent that can install or import this code can search
   the generator's options until its format note equals the task's, then print the reference reconstruction. No
   MR knowledge and no data are needed. The generated `task.toml` therefore sets `network_mode = "no-network"`
   under `[environment]`. If the agent harness needs a model endpoint, allow that host only, and keep this
   repository out of the agent's image.
2. **The local verifier is not a security boundary.** `python -m recon_repair verify` runs the submission as the
   same user as the verifier. A submission can read the verifier's process arguments, find the hidden folder and
   return the stored ground truth. The separation by user (verifier as root, submission as an unprivileged user,
   ground truth readable by root only) exists only inside the task container, and that path has not been run.

Also: seeds are plain integers and appear in task names and in `task.toml`. They are not secrets. `answer_key.json`
is written at the root of each task directory, next to `environment/`; do not mount a task directory into an
agent's workspace.

## Run

From this folder, with Python 3.10 or newer, numpy and scipy:

```sh
python -m pytest -q                                                  # 7 checks, about 13 s on one core
python -m recon_repair --help
python -m recon_repair make --seed 0 --level 1 --split train --out tasks/
python -m recon_repair selfcheck --n 12                              # build instances, run the control submissions
```

`make` writes one task directory in the Harbor task layout: `instruction.md`, `task.toml`, `environment/` (what the
agent sees), `tests/` (verifier and hidden scans), `solution/` and `answer_key.json`.

| Command | Does |
|---|---|
| `make` | writes one task directory from `(split, level, seed)` |
| `batch` | writes many and prints which fault variants were used and why draws were rejected |
| `workspace` | copies what the agent sees to a folder, for a local run |
| `verify` | grades a folder that holds a `recon.py` against the hidden scans of a task (see warning 2) |
| `selfcheck` | builds instances and checks the control submissions through the verifier |

## What a task contains

| Item | Value |
|---|---|
| Scenario | 2D multi-coil Cartesian scan, raw records in a scanner-specific format, SENSE reconstruction |
| Levels | 1, 2, 3 = number of faults left in `recon.py`; level 3 has no localizer image |
| Splits | `train`, `dev`, `heldout_mechanism`; three fault mechanisms are never injected in `train` |
| Scans | 2 visible, 3 hidden; every hidden scan must pass |
| Reward | 1 if every hidden image is within the threshold of that scan, else 0 |
| Agent may use | Python standard library, numpy, scipy |

A draw becomes a task only if four things hold on every scan: the reference reconstruction is close to the truth;
a second, separately written correct reconstruction passes; the script as given fails with margin; the script with
any single fault left fails with margin.

## What has and has not been run

| Check | State |
|---|---|
| `python -m pytest -q` in this folder | passes |
| `python -m recon_repair selfcheck` (local verifier) | passes: reference 1.0; script as given, zeros, a crash and one fault left 0.0 |
| A task under Harbor or in a container | not run |
| The `no-network` setting enforced by a runner | not run |
| A language model on these tasks | not run |

Licence: Apache-2.0, as the repository.
