# Container tier: scanner and agent in separate containers

For an agent that executes code. The scanner service runs in one container with the secret, the episode, the logs
and their heads; the agent runs in another with a URL, one token and the single-file client. They share one
internal network and nothing else: no volume, no secret but the agent token, no process namespace.

**State: written and linted, not run.** The development machine has a Docker client without a daemon and without
the compose plugin, so no image was built and `probe.py` has not run inside a container. `tests/test_container_files.py`
parses the files and checks the properties listed below; the same service, tokens and probes are exercised between
processes of one machine by `tests/test_service.py` and `mrgym redteam`.

| File | What it is |
|---|---|
| `Dockerfile` | image of the scanner service: `python:3.12-slim`, the package, a user 10001 |
| `compose.yaml` | services `scanner`, `agent`, and `operator` (profile `operator`); written in the JSON subset of YAML |
| `probe.py` | run in the agent container: every forbidden access is tried and must fail |
| `operator_result.py` | run on the operator side: reads the result from the scanner's operator plane |

## What the compose file fixes

- `scanner`: `mrgym serve --episode-file /run/secrets/episode ...`. No seed on the command line, no secret in the
  environment (`MRGYM_SECRET_FILE` names a file). Read-only root file system, all capabilities dropped,
  `no-new-privileges`, user 10001. Volumes `scanlog`, `scanstate`, `scanadmin`.
- `agent`: user 10002, the secret `agent_token` only, no volume in common with the scanner, read-only root file
  system. Its network `episode` is `internal`: it reaches `scanner:8000` and nothing outside.
- `operator`: no network; it mounts `scanadmin` (the unix socket of the operator plane) and holds the operator
  token. The reward is read there, never from a file the agent's container could write.

## Secrets

Four files under `secrets/` (ignored by git):

    mrgym epoch keygen --out secrets/mrgym_secret.key
    mrgym epoch keygen --out secrets/operator_token
    mrgym epoch keygen --out secrets/agent_token
    echo '{"env_id": "Scanner/NullTissue-v0", "level": 2, "split": "test-iid", "seed": 0}' > secrets/episode.json
    chmod 600 secrets/episode.json

Compose mounts file secrets with the ownership they have on the host. The service refuses secret files that group
or others can read, so `episode.json`, `mrgym_secret.key` and `operator_token` must be owned by uid 10001 with mode
0600, and `agent_token` must be readable by uids 10001 and 10002 (for example group-readable with a shared group;
the service does not check the mode of a token file).

## Run (where Docker is available)

    docker compose -f integrations/container/compose.yaml build scanner
    docker compose -f integrations/container/compose.yaml run --rm agent          # probe.py: must exit 0
    docker compose -f integrations/container/compose.yaml --profile operator run --rm operator

A non-root user inside one container is not a boundary; the boundary here is the separate container, user, volume
set and network. A Harbor task that wraps these two services with a separate verifier is not written.
