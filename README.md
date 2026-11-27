<h1 align="center">MR Gym</h1>

<p align="center">
  <b>A virtual MRI scanner for AI agents.</b><br>
  The agent scans an object it cannot see, and the scanner scores what it delivers.<br>
  <a href="LICENSE"><img alt="Code: Apache-2.0" src="https://img.shields.io/badge/code-Apache--2.0-0e7490"></a>
</p>

<p align="center">
  <img src="assets/hero.png" width="100%"
       alt="A scripted session on the virtual scanner: five requests, the brain images that came back, the scan-time budget being spent">
</p>

## What it is

MR Gym is a simulated MRI scanner that an agent operates through tool calls. It is a reinforcement-learning
environment: you can train a model on it and test one on it.

- You send a protocol (sequence, timing, geometry) and get back the image a scanner would give, formed in k-space, with
  noise, wrap-around, fat shift and banding.
- Each task hides an object in the scanner. The agent gets a task text, the tools (`plan`, `scan`, `view`, `profile`,
  `read_regions`, `calc`, `submit`) and a scan-time budget.
- At the end the agent submits a protocol or a measurement. The scanner checks it on the hidden object: reward 1 or 0.
- Every seed gives a new object, so the same task serves training and testing.

## Install and use

```bash
git clone https://github.com/KeWang0622/MRGym.git
cd MRGym
pip install -e .
mrgym demo
```

It needs numpy and scipy, no GPU and no network. `mrgym demo` plays one episode with a scripted solution and prints
every call ([what it prints](docs/TOUR.md#what-mrgym-demo-prints)). The same from Python:

```python
import mrgym

env = mrgym.make("Scanner/NoWrap-v1", level=2, seed=0)
obs, info = env.reset()                      # obs.text is the task; the object is hidden and new for every seed
tools = env.tools("openai")                  # the tool schemas for a model; also "anthropic" and "mcp"
calls = [                                    # an agent chooses these; here one localiser, then the answer of `mrgym demo`
    {"name": "scan", "arguments": {"sequence": "spgr", "tr_ms": 11, "flip_deg": 10, "fov_mm": 288, "matrix": 160, "averages": 2}},
    {"name": "submit", "arguments": {"pe_dir": "y", "fov_mm": 168, "phase_fov": 0.595238, "matrix": 84,
                                     "centre_y_mm": -1.0, "centre_x_mm": 16.3, "rotation_deg": 90}},
]
for call in calls:
    obs, reward, terminated, truncated, info = env.step(call)
print(reward, terminated)                    # 1.0 True: the delivered geometry passed on the hidden object
```

The agent sees the task text and the results of its own tool calls, nothing else. To go on:
[getting started](docs/GETTING-STARTED.md), [a scripted agent](examples/first_episode.py),
[a language model as the agent](examples/run_llm_agent.py).

## Tasks

<!-- TASKS:START -->
| Task | What you do | Kind | Status |
|---|---|---|---|
| [`Scanner/Explore-v0`](envs/scanner-explore-v0/) | scan freely; nothing is asked and nothing is scored | - | no score |
| **A · Signal and contrast**: what signal a tissue gives | | | |
| [`Scanner/T2Decay-v0`](envs/scanner-t2decay-v0/) | measure the T2 of each vial | measure | experimental |
| [`Scanner/FatFraction-v0`](envs/scanner-fatfraction-v0/) | measure the fat fraction of each mixture | measure | experimental |
| [`Scanner/T1Vial-v0`](envs/scanner-t1vial-v0/) | measure the T1 of each vial | measure | experimental |
| [`Scanner/T1Transmit-v1`](envs/scanner-t1transmit-v1/) | measure T1 when the transmit field is miscalibrated | measure | experimental |
| [`Scanner/NullTissue-v0`](envs/scanner-nulltissue-v0/) | find the inversion time that makes each vial dark | set | ready |
| [`Scanner/FatSuppress-v0`](envs/scanner-fatsuppress-v0/) | choose the fat suppression that works in each region | set | ready |
| [`Scanner/ErnstFlip-v0`](envs/scanner-ernstflip-v0/) | find the flip angle that gives each vial most signal | set | experimental |
| [`Scanner/ProtocolToSpec-v0`](envs/scanner-protocoltospec-v0/) | turn written requirements into a protocol that meets them | trade | experimental |
| **B · Spatial encoding and reconstruction**: where the signal lands in the image | | | |
| [`Scanner/NoWrap-v1`](envs/scanner-nowrap-v1/) | find the smallest, fastest field of view around the object | set | ready |
| [`Scanner/Resolve-v0`](envs/scanner-resolve-v0/) | find the fastest scan that still separates two fine features | set | experimental |
| [`Scanner/FoldAway-v0`](envs/scanner-foldaway-v0/) | let the object wrap, but not onto your region | set | experimental |
| [`Scanner/ShiftFree-v0`](envs/scanner-shiftfree-v0/) | choose the bandwidth that keeps shifted fat off the structure | trade | experimental |
| **C · The system**: what the machine adds: field, RF, gradients, noise, limits | | | |
| [`Scanner/PhantomQA-v0`](envs/scanner-phantomqa-v0/) | run the phantom quality tests and name what is broken | measure | ready |
| [`Scanner/FrequencyAdjust-v0`](envs/scanner-frequencyadjust-v0/) | find how far off resonance the water is | measure | experimental |
| [`Scanner/TransmitAdjust-v0`](envs/scanner-transmitadjust-v0/) | measure the real flip angle against the nominal one | measure | experimental |
| [`Scanner/NoiseSNR-v0`](envs/scanner-noisesnr-v0/) | measure the signal-to-noise ratio the way the standards do | measure | experimental |
| [`Scanner/GradientCal-v0`](envs/scanner-gradientcal-v0/) | measure the gradient scale error with a known phantom | measure | experimental |
| **D · Image quality and artefacts**: how good the image is, and what went wrong | | | |
| [`Scanner/SNRSpec-v0`](envs/scanner-snrspec-v0/) | reach a required signal-to-noise ratio in the least scan time | set | ready |
| [`Scanner/BandFree-v0`](envs/scanner-bandfree-v0/) | keep the dark bands of balanced SSFP off the object | set | ready |
| [`Scanner/GhostAside-v0`](envs/scanner-ghostaside-v0/) | steer motion ghosts away from the region you need | set | experimental |
| [`Scanner/ArtefactClinic-v0`](envs/scanner-artefactclinic-v0/) | say what causes each artefact and fix the protocol | trade | ready |
| [`Scanner/EPISetup-v0`](envs/scanner-episetup-v0/) | set up echo-planar imaging within distortion and ghost limits | trade | experimental |

A task is ready when it has been checked on 300 fresh objects: a scripted solution passes, and shortcuts that ignore the
object fail. Each task has levels of difficulty (`mrgym list`); [envs/](envs/README.md) has the full table.
[The map](docs/tasks.md) lists 115 pieces of MR physics: the scanner forms 61, and 22 have a task of their own.
<!-- TASKS:END -->

## Add a task

A task is three Python files in one folder: what is asked and how it is scored, a scripted solution next to the
shortcuts that must not work, and one line that registers it. [Adding a task](docs/ADDING_A_TASK.md) walks through a
small example that runs: [examples/new_family/](examples/new_family/).

## Does training work?

<!-- TRAINING:START -->
<p align="center">
  <img src="assets/training_27b.png" width="900"
       alt="Pass rate of Qwen3.8-27B before and after 60 steps of reinforcement learning, on test objects it had never seen">
</p>

Qwen3.8-27B was trained for 60 steps by reinforcement learning, on the reward alone. On 96 test episodes with objects it
had never seen, the level it trained on went from 47 passes to 79, and an easy level of a task it never trained on from
32 to 74. These are first runs on one task, and training on a second task did not work: [the recipe](docs/RECIPE.md) and
[the full report](docs/experiments/RESULTS-27b.md) have the details.
<!-- TRAINING:END -->

## Docs

- [Getting started](docs/GETTING-STARTED.md): command line, Python, a language model
- [Scanner manual](docs/scanner.md): what the agent is told
- [A tour](docs/TOUR.md): the demo, the scanner, speed, repository map
- [Design](docs/DESIGN.md): why it is built this way
- [Physics validation](docs/physics/validation.md): the scanner against textbook formulas
- [Limits](docs/TOUR.md#limits): what a pass does not show
- [Training](docs/RECIPE.md): the recipe, its cost, its pitfalls
- [Versioning](docs/VERSIONING.md): package, engine, task ids
- [Everything else](docs/README.md)

## Cite

[CITATION.cff](CITATION.cff). The code is under [Apache-2.0](LICENSE). MR Gym is a research preview; see the
[changelog](CHANGELOG.md) and [how to contribute](CONTRIBUTING.md).
