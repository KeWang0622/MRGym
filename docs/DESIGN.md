# Design

Why MR Gym is built the way it is. The instrument itself is described in [scanner.md](scanner.md).

## An episode on the scanner

An episode is one session on the virtual scanner. The agent faces a hidden world: an object, and the state of the
system that images it. It can plan a protocol for free and run it for scan time, within a budget. At the end it
delivers one thing: a quantity with units, a protocol or a decision. The reward is computed from what was
delivered, on the hidden world. It is paid once, at the end, and only if the scanner's log shows a measurement.

The world differs between episodes in a way that changes the right answer. An episode that can be solved without
looking at the object is not an episode in this sense, whatever it is called.

## Five rules

| Rule | Reason |
|---|---|
| 1. The world stays with whoever runs the episode. The agent gets tools, and no tool resets the world, reads it, or returns anything derived from the seed except measurements. | The generator is public, so a known seed is a stored answer. |
| 2. The interface is the console: a free `plan` that returns timing, cost and limits and never signal, and a priced `scan`. A design task adds design actions as parameters: a generator call with a handful of numbers, or a bounded table of at most 64 control points. Raw sample arrays and free code are not actions. | Tools at the level of samples or code turn the task into programming. |
| 3. The engine that serves and scores data models more than the formulas the agent is given. | If both sides use one closed form, the task is to invert that formula. |
| 4. Every scan has a price: overhead, dummy cycles, lines, averages, and noise that depends on the protocol. | With a loose budget, a sweep or a slow safe scan replaces the decision. |
| 5. A level of an environment is admitted by control policies, not by its description. | A hidden world and a budget do not, by themselves, force a measurement. |

### Rule 2 and design tasks (amended 2026-11-27)

Until 2026-11-27 rule 2 excluded every tool below the console ("tools at the level of waveforms or code turn the task
into programming"). The destination of the project is pulse-sequence design, and a console parameter cannot express a
design: at console depth the verifier computes the optimum by exhaustive search, so nothing better than the reference
can be delivered. The rule now allows design actions, in three forms only:

| Allowed | Not allowed | Why the line is there |
|---|---|---|
| a generator and its parameters (`design`: `slr`, `sinc`, `sech`, `verse`) | an array of samples | 512 samples are 1,800 to 8,600 tokens per call, more than a reply holds; a generator call is 50 to 70 |
| a bounded table of control points (`points`: at most 64 numbers, a stated interpolation) | code that returns a waveform | code makes the task programming, and its reward a property of the sandbox |
| the number of a stored, immutable design (`simulate`, `scan`, `submit`) | editing a design in place | an old answer must keep describing the same waveform |

What does not change: the hidden world stays with whoever runs the episode (rule 1); the simulator an agent is
given is the public model and the scanner models more (rule 3: the transmit chain of `mrgym.design.hardware`); playing
a design on the scanner is a priced scan (rule 4); a design level is admitted by control policies, among them
`ideal_tuned`, the best design of the simulator delivered without a measurement, which must fail (rule 5).
A family opts in; the tools, schemas and answers of every other family are byte for byte what they were
(`python -m mrgym.compat.fingerprint --check`).

How a family opts in (the template is `src/mrgym/envs/rf_pulse/`):

- `spec.py` names the class with the design tools: `entry_point="mrgym.envs.design_tools:DesignFamilyEnv"`.
- `task.py` declares the generators it offers: `DESIGN_TOOLS = dict(kinds=("sinc", "slr", "points"))`, and puts
  `design = dict(specification=..., limits=dict(b1_peak_ut=..., g_max_mt_m=..., slew_t_m_s=..., b1rms_ut=..., tr_ms=...))`
  into the task card of every instance.
- The world spec lists the scan that plays a design, `console_sequences=["rfscan"]`, and gives the transmit chain,
  `rf_hardware=dict(tx_scale=(low, high), amp_sat_ut=(low, high))`.
- When `score` runs, the designs of the episode are in `world["console"]["designs"]` (number → call and facts; the
  pulse is `mrgym.design.tool.build(call)`).
- `controls.py` includes `ideal_tuned`. A design task whose `ideal_tuned` passes hides nothing that matters.

## Admission by control policies

Each scored environment ships a `controls` module with scripted policies. They play through the same interface
as an agent and never read the world.

| Policy | Must | What it stands for |
|---|---|---|
| reference | pass | the procedure a physicist would use |
| no-scan prior | fail | answering without measuring; it defines reward 0 |
| best constant recipe | fail | one protocol and answer rule, tuned on other worlds, applied blindly |
| blind sweep | fail | spending the budget without a decision |
| wrong units | fail | the right numbers in the wrong unit |
| seed enumeration | not a control policy | `mrgym redteam`, row RT17: with the documented derivation the world of a public split is recovered (the positive control), that of a salted or secret split is not |
| the environment's own | fail | procedures that ignore the one thing the environment hides |

The verdict is one per level, never one per instance. On fresh instances of the public split `gate` a level is
admitted when the one-sided 95 % lower bound of the reference's pass rate is at least 90 %, the upper bound of
every must-fail policy is at most 10 %, and every ablation the family lists for the level (the reference with one
decision removed) loses at least 25 points. `mrgym gate <id> --n 300 --write` measures this and stores the table. The
environment card and `mrgym list` read that file. Every id carries a kind: `candidate` (the gate is not stored on
this engine yet), `admitted` (the stored gate admits every level of the claim) or `practice` (scored, outside any
claim: its gate rejects a level, or a fixed protocol passes). `mrgym check <id>` fails when an id is registered as
admitted without a stored gate on the current engine that admits every claim level. A family may mark levels as
practice levels (`kind="practice"` in its level table): they are playable and outside the claim.

What the gate shows: the listed shortcuts fail on fresh instances, and the task can be done by the reference.

What it does not show:

- That no other shortcut exists. A new one becomes a control policy.
- That an agent which passes understands the physics. It shows the listed ways of not measuring do not pay.
- That the scanner agrees with a real one. `mrgym validate` compares the engine with analytic formulas.
  [physics/measured.md](physics/measured.md) compares statistics of its images and signals with public phantom
  measurements and lists the differences; no image of its sequences has been compared pixel by pixel with a scan of
  the same object, and nothing here has been reviewed by an MR physicist.
- Anything about a seed outside the stored instances. Admission is per instance.

## Threat model

Three kinds of agent, three boundaries. [SECURITY.md](../SECURITY.md) has the full statement; `mrgym redteam <id>`
measures it, row by row, and exits non-zero when an expected outcome is not the observed one.

| Tier | The agent can | What stops it | What does not |
|---|---|---|---|
| Tool-calling model, in process (`mrgym.make`) | send tool calls and read the replies | no reply carries the seed, an id derived from it or a hidden value; a deliverable without a measurement, in the wrong unit or malformed earns 0 | nothing stops code in the same process: the seed, the world and the scorer are Python objects there |
| Code in another process (`mrgym serve`) | all of the above, install the package, enumerate seeds, read what its user can read | worlds of a secret or salted split cannot be rebuilt; ids are random or keyed; tokens on every route; no secret in arguments or environment; a log that does not match its head is refused | on a public unsalted split the enumeration recovers the world, by design; a process of the same user can read the service's files |
| Code in a container (`integrations/container/`) | all of the above inside its own container | separate container, user, volumes, internal network; the result comes from the operator plane | not run on the development machine |

- Public splits (`train`, `dev`, `tune`, `gate`) are reproducible by anyone and never touch the secret. Secret
  splits (`test-*`, `ood`, `heldout`, `semi`) refuse to run without it. A held-out epoch is committed to in advance
  and verified after its reveal (`mrgym epoch`).
- The scanner log carries a MAC per record under a key of the episode. Its head (length, last MAC, scan seconds)
  is kept outside the log directory and handed to the trainer. The verifier replays the log against that head, so a
  shortened, edited or foreign log does not verify, and a restart on a log that does not match its receipt serves
  nothing. A budget cannot be refilled given the head; without a head held by someone else, a chain proves order,
  not completeness.
- The episode id is random. The task id is a trainer-side label; no agent-visible payload carries it.

## Three versions

| Version | Example | Changes when |
|---|---|---|
| Package | `0.1.0.dev0` | see the rule at the top of [CHANGELOG.md](../CHANGELOG.md) |
| Engine | `mrgym-engine-0.1.0` | the same seed and protocol can give other data |
| Environment | `Scanner/NoWrap-v1` | a reward of that environment can change |

The engine version is a constant, bumped by hand, and pinned by a test together with reference numbers. A recorded
log replays only on the engine that wrote it: bit-exact on the same numerical stack, within a stated tolerance on
another, and the replay report says which. Every reported number names all three versions.
