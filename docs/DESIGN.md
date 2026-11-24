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
| 2. The interface is the console: a free `plan` that returns timing, cost and limits and never signal, and a priced `scan`. | Tools at the level of waveforms or code turn the task into programming. |
| 3. The engine that serves and scores data models more than the formulas the agent is given. | If both sides use one closed form, the task is to invert that formula. |
| 4. Every scan has a price: overhead, dummy cycles, lines, averages, and noise that depends on the protocol. | With a loose budget, a sweep or a slow safe scan replaces the decision. |
| 5. An environment is admitted by control policies, not by its description. | A hidden world and a budget do not, by themselves, force a measurement. |

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
| seed attack | fail | rebuilding the world from guessable seeds with the public generator |
| the environment's own | fail | procedures that ignore the one thing the environment hides |

An instance is admitted when the reference passes on fresh noise draws and every control fails on every draw.
`mrgym check <id> --admission --n 24 --write` runs the rule over public training seeds and stores the table. The
environment card and `mrgym list` read that file, and `mrgym check <id>` fails when an id is registered as admitted
without admitted instances on the current engine. An environment that its gate rejects stays listed, as rejected.

What the gate shows: the listed shortcuts fail on the listed instances, and the task can be done by the reference.

What it does not show:

- That no other shortcut exists. A new one becomes a control policy.
- That an agent which passes understands the physics. It shows the listed ways of not measuring do not pay.
- That the scanner agrees with a real one. `mrgym validate` compares the engine with analytic formulas, and
  nothing here has been compared with a physical scanner or reviewed by an MR physicist.
- Anything about a seed outside the stored instances. Admission is per instance.

## Trust model

| Surface | The world is hidden from | The world is not hidden from |
|---|---|---|
| In process: `mrgym.make`, `env.step` | a model that only sends tool calls | any code in the same process |
| HTTP service: `mrgym serve`, one process per episode | an agent in another process that reaches only the port | whoever can read the service's files, memory or episode config |

- In process is the surface for tool-calling models and for training loops that own the episode. The seed, the
  world and the score function are ordinary Python objects there.
- An agent that executes code needs the HTTP service behind a process boundary. The service holds the episode
  config and the log. It has no route to reset, to re-seed or to the truth; a fresh state is a fresh process.
- The scanner log is append-only and hash-chained. The verifier replays it from the episode config and checks
  every record, so a budget cannot be refilled and a scan cannot be invented after the fact.
- Training seeds are public: anyone can rebuild those worlds. A held-out split needs `MRGYM_SECRET` and refuses to
  run without it.
- The task id is derived from the seed one way. Nothing else that the agent sees depends on the seed except
  measurements.

## Three versions

| Version | Example | Changes when |
|---|---|---|
| Package | `0.1.0.dev0` | see the rule at the top of [CHANGELOG.md](../CHANGELOG.md) |
| Engine | `mrgym-engine-0.1.0` | the same seed and protocol can give other data |
| Environment | `Scanner/NoWrap-v0` | a reward of that environment can change |

The engine version is a constant, bumped by hand, and pinned by a test together with reference numbers. A recorded
log replays only on the engine that wrote it: bit-exact on the same numerical stack, within a stated tolerance on
another, and the replay report says which. Every reported number names all three versions.
