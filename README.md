# MR Gym

Physics-application RL environments for LLM agents in MRI. **Research preview, 2026-11-24.** Environments, verifiers and
admission controls are built and measured. No model has been trained on them yet.

## What a physics-application episode is

The agent faces a hidden **system + tissue** world through a protocol-level virtual scanner. It has a scan budget too small
for search. What it delivers — a quantity with units, or a protocol — is wrong unless a specific physical confounder is
recognised and handled. The reward is the error of the delivered result, measured on the hidden world.

Five design rules follow from that definition, each from a failure we or others measured (see [docs/DESIGN.md](docs/DESIGN.md)):

| Rule | Why |
|---|---|
| The world lives in a scanner service behind a process boundary, with a secret per-episode seed | With an open generator, a guessable seed is a stored answer |
| The interface is protocol level: a free `plan` (timing, limits; never signal) and a priced `scan` | Waveform-level tools turn the task back into code repair |
| The world that serves and scores data is of higher fidelity than the formulas the agent is given | A closed-form world only tests inverting the generator's own formula |
| Scans are priced: overhead, dummy repetitions, bandwidth-dependent noise | A scan-time-only budget makes small or slow-but-safe scans a free escape |
| An instance is admitted only if a reference policy passes and every shortcut policy fails | A hidden world and a budget do not, by themselves, force physics |

## What is new

To our knowledge (search limited to arXiv, GitHub, PyPI and Hugging Face, November 2026):

- **A trainable environment, not an evaluation.** Published MRI agent work (sequence-design and scanner-development agents)
  runs frozen models and releases no environment. Here instances are generated without limit from seeds, with train and
  held-out splits by fault mechanism.
- **Admission by control policies, with the negative results kept.** Every family ships the shortcut policies that must fail:
  no-scan prior answer, confounder-blind textbook protocol, best fixed recipe, brute force within budget, wrong units, seed
  attack. Our first physics family failed this gate and is reported as failed (below).
- **A scoring world that disagrees with the textbook formula where real scanners do.** Spoiled gradient echo is served from an
  EPG steady state with finite T2 and spoiler moment; the ideal Ernst formula is given to the agent as a helper and is wrong in
  the cheap short-TR regime by a measured amount.
- **Verification by execution on hidden data.** Reconstruction tasks are graded by running the submitted code on scans the
  agent never saw and comparing with the simulated object; an independently written second correct implementation must pass.

## Families

| Family | Agent's job | Status |
|---|---|---|
| `recon_repair` | Repair a faulty multi-coil Cartesian reconstruction against the scanner's raw-data format; graded on hidden scans | Built and verified. Data-and-encoding layer: it is code repair in an MR setting, not physics application |
| `t1_transmit` | Measure T1 in marked regions with an uncalibrated transmit field, within a scan budget | Built. **Rejected by its own admission rule** — one fixed recipe solves every instance. Kept as a curriculum rung and as the worked example of the gate |
| `diagnose` | Identify which deviation from the nominal system is active, its magnitude, and deliver a remedy protocol that is re-scanned | In progress |

## Measured so far

**`recon_repair`** — 25 fault variants in 13 mechanism groups; each instance draws a scanner "dialect" (array layout, index
conventions, units, field names) so that a memorised file does not help.

| Check | Result |
|---|---|
| Reference solution, through the verifier subprocess | 12 / 12 instances pass |
| Reconstruction as given, all-zero image, crash | 0 / 12 each |
| Multi-fault instance with one fault left unrepaired | 0 / 20 |
| Independently written second correct implementation | error at most 0.77 × pass threshold |

Model baselines are not published yet.

**Scanner core** — closed forms against an independent matrix Bloch steady state: 3.7e-16 over 21,276 points. EPG against a
time-stepping EPG: 3.5e-12. Served spoiled-GRE signal relative to the Ernst formula: 0.895–1.192 at TR 4 ms with a weak
spoiler, 0.965–1.072 at TR 20 ms.

**`t1_transmit`** — 50 instances per level, pass rate of each policy:

| Policy | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| Reference (maps B1, chooses angles, corrects, fits) | 0.988 | 0.983 | 0.991 |
| **Best fixed recipe** | **0.963** | **0.960** | **0.998** |
| No-scan prior, confounder-blind VFA, brute force, inversion recovery only, wrong units, seed attack | 0 | 0 | 0 |

The fixed recipe passing is the finding: the family forces B1 mapping, a spoiling-aware protocol and correct units, but no
per-instance decision. Admitted instances: 0 / 150.

## Run it

Python ≥ 3.10, numpy, scipy.

```sh
python3 tests/test_recon_repair.py            # 6 checks
python3 tests/test_scanner_core.py            # 13 checks
python3 -m mrfrl selfcheck --n 12             # recon_repair: build instances, run the controls through the verifier
python3 -m mrfrl make --seed 0 --level 2 --split train --out tasks/   # one task directory (Harbor layout)
python3 -m mrfrl.t1_transmit selfcheck --n 10 # scanner family: admission table per level
```

A task directory holds `instruction.md`, `environment/` (what the agent sees), `tests/` (verifier and hidden scans),
`solution/` and `task.toml` in the [Harbor](https://github.com/harbor-framework/harbor) task format with a separate verifier.
The scanner is documented for the agent in [docs/scanner.md](docs/scanner.md).

## Not done

- No model has been trained; there is no evidence yet that training here improves a model.
- Task directories were not run under Harbor / Docker.
- The engine uses hard pulses without slice profile, serves the steady state (dummy repetitions are charged, the transient is
  not simulated), one receive channel with white noise, and has no readout T2* blur, off-resonance shift or chemical shift yet.
- Noise level, budgets and tolerance multipliers in `t1_transmit` are hand-set.
- The generator is public, so the held-out splits in this repository are not secret.

## Related

[MRFoundry](https://github.com/KeWang0622/MRFoundry) (agent skills for MRI) ·
[MRI Agent Bench](https://github.com/KeWang0622/MRIAgentBench) (held-out evaluation; none of its tasks, generators or
reference solutions are used here).

Licence: Apache-2.0.
