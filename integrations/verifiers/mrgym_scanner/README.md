# mrgym-scanner

MR Gym as a [verifiers](https://github.com/PrimeIntellect-ai/verifiers) v1 taskset. The model operates a virtual MRI
scanner through priced tool calls and delivers a measurement or a protocol. Each task is one seeded instance of one
registered MR Gym environment id; the taskset is procedural and unbounded.

The package adds no physics and no scoring. The reward of a rollout is the reward `mrgym` returned when the episode
ended: 1 when the deliverable passes every check on the hidden world, else 0. The environment card of each id
(`mrgym card <id>`) states what is given, what is delivered and how it is scored.

## Use

```sh
pip install -e . -e integrations/verifiers/mrgym_scanner        # from a checkout of MR Gym
vf-validate mrgym-scanner -n 5 --runtime.type subprocess        # gold and no-op check, no model
vf-validate mrgym-scanner -n 5 --runtime.type subprocess --taskset.env-id <id> --taskset.level <level>
vf-run mrgym-scanner -n 5 -r 3 -m <model>                       # needs a model endpoint
```

The command names are those of verifiers 0.4.1 (`vf-validate`, `vf-run`, `vf-debug`); `vf-eval` no longer exists.

| Taskset option | Meaning |
|---|---|
| `--taskset.env-id` | one registered id; empty (default) = every scored id of the registry in turn |
| `--taskset.level` | level of the id; default = the id's default level |
| `--taskset.first-seed` | first instance seed (integer) |
| `--taskset.split` | `train` (public seeds) or `heldout` (needs `MRGYM_SECRET` on the evaluator) |

## What the checks mean

- Gold: the id's scripted reference policy is replayed call by call through the same function the tool server runs
  (`Episode.call`). Valid when the episode ends with a reward at the id's threshold.
- No-op: the task is scored untouched. Valid when the reward stays below the threshold (it is the floor, 0).

## Limits

- The tool server is a custom MCP toolset. Prime's guide prefers tasksets without custom tools, driven by a plain
  shell harness; that form is the Harbor dataset written by `mrgym export harbor`, which `HarborTaskset` can load.
- The tools are served with MR Gym's own JSON schemas (the catalogue is read from `env.tools("mcp")` per episode, not
  derived from Python signatures). A harness that rewrites schemas changes what the model sees.
- `vf-validate` does not start the tool server: it runs the gold replay in the evaluator process. `vf-run` against a
  model starts it; that path needs a model endpoint and is listed under "not run" in `integrations/README.md` until
  someone runs it.
- Three versions travel with every number: `mrgym.__version__`, `mrgym.ENGINE_ID` and the environment id.
