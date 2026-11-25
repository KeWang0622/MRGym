---
title: MR Gym scanner
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
tags:
  - openenv
---

# mrgym_scanner_env

MR Gym as an [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment package: a virtual MRI scanner operated
through priced tool calls. The package adds no physics and no scoring. Tools are the MCP tool list of the episode
(`ListToolsAction`), a `CallToolAction` is one tool call, reward and `done` travel on the observation.

```sh
pip install openenv uvicorn "mrgym==<the version in server/requirements.txt>"
uvicorn server.app:app --port 8000                      # from this directory
openenv validate . --level static --skip-build          # manifest check; needs no Docker
```

`reset(seed=..., env_id=..., level=...)` starts an episode of any registered MR Gym id. Without `env_id` the server
uses `$MRGYM_ENV_ID`, else the id this package was exported for (`default_env.json`).

Generated files (`mrgym export openenv --id <id>`; `--check` fails when they are stale): `openenv.yaml` (the declared
tools of the id), `validation/runtime.json` (the id's scripted reference policy as a replay plan), `default_env.json`,
`pyproject.toml` and `server/requirements.txt` (the exact `mrgym` pin). Everything else is the same for every id.

What has and has not been run is listed in [`integrations/README.md`](../../README.md). `openenv validate` exits 0 on
a WARN verdict; read the per-check lines, not the exit code.
