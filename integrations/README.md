# Integrations

The core package `mrgym` depends on numpy and scipy only. Anything that needs an agent framework lives here, outside
the package, and is not installed with it.

| Target | Artifact | State |
|---|---|---|
| Python users, trainers with their own loop | wheel of `mrgym` | Not on PyPI. Install from a checkout: `pip install -e .`. The test workflow builds the wheel and runs the tests against it; the workflow has not run yet. |
| OpenEnv | [`openenv/openenv_server.py`](openenv/openenv_server.py), [`openenv/demo.py`](openenv/demo.py) | Written against openenv 0.14.0. Needs `pip install openenv uvicorn`. Not run in this tree. Not published as a Space. |
| verifiers | none | Not written. |
| Harbor | none | Not written. |

## Rules for an adapter

1. It adds no physics and no scoring. The world, the pricing, the log, the replay and the score come from `mrgym`.
2. It states the three versions it was run with: package, engine (`mrgym --version`) and environment id.
3. The seed and the episode config stay on the side that runs the episode. They never enter a tool description, a
   task name, a command line or anything else the agent can read.
4. It is listed here with its real state. An adapter that has not been run says so.

## OpenEnv

`openenv_server.py` wraps any registered environment id as an OpenEnv `Environment`. The tools are the ones from
`env.tools("mcp")`, a `CallToolAction` is one `env.step`, and reward and `done` travel on the observation.

```sh
pip install openenv uvicorn
python integrations/openenv/demo.py
```

The demo plays `Scanner/NoWrap-v0` in process, then over HTTP and WebSocket, then runs the environment's reference
control policy over the wire through `mrgym.policies.ToolScanner`.

State: not run. `openenv` is not installed in the development environment of this repository, and nothing was
installed to try it. The call shapes the demo relies on (tool list, scan, refused scan, submit, public state) were
checked against the environment without the OpenEnv transport.

Trust model: the environment runs inside the server process, so the world is hidden from a client that sends tool
calls and not from code that runs in that process. See [SECURITY.md](../SECURITY.md).

## Not written

- A verifiers environment package.
- Harbor task directories for the scanner environments, and the container images they would need.
- A dataset of recorded episodes.

These are plans. [docs/ROADMAP.md](../docs/ROADMAP.md) lists them with the rest of the remaining work.
