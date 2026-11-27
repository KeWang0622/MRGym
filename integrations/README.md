# Integrations

The core package `mrgym` depends on numpy and scipy only. Adapters to other frameworks either ship inside the package
without importing their framework at package import (`mrgym.compat`), or live here, outside the package.

Every adapter is generic over the registry: it is written against `env.tools(fmt)`, `env.step(tool_call)` and
`Observation`, names no tool and no family, and so serves every id of `mrgym.list_envs()`, including ids registered
later. `tests/test_adapters_parity.py` plays the same seed and the same scripted policy through the native API and
through each adapter and compares the reward, the observation digest and the head of the scanner's log.

## Rules for an adapter

1. It adds no physics and no scoring. The world, the pricing, the log, the replay and the score come from `mrgym`.
2. It states the three versions it was run with: package, engine (`mrgym --version`) and environment id.
3. The seed and the episode config stay on the side that runs the episode. They never enter a tool description, a
   task name, a command line or anything else the agent can read.
4. It is listed here with its real state, written by `make conformance` from the validators' own output. A line
   that says PASS names the check that passed; a validator that exits 0 on a warning is reported with its skips.

## State

<!-- BEGIN GENERATED: conformance -->
Measured by `make conformance` on 2026-11-25T04:16Z at commit `63548ea+uncommitted` (mrgym 0.1.0.dev0, engine `mrgym-engine-0.2.0`, 5 registered ids). Conformance: 56.9 % of 108 items (49 pass, 25 partial, 34 fail).

| Target | Artifact | Validator | Result of the last run |
|---|---|---|---|
| Gymnasium | `src/mrgym/compat/gymnasium.py` (extra `mrgym[gymnasium]`) | `python -m mrgym.compat.gymnasium --check` | gymnasium 1.4.0: check_env PASS with 0 warnings on 5 of 5 ids |
| verifiers | `integrations/verifiers/mrgym_scanner/` | `vf-validate mrgym-scanner -n N --runtime.type subprocess` | verifiers 0.4.1 vf-validate, 3 instances per scored id: gold valid Scanner/NoWrap-v0 3/3, Scanner/NullTissue-v0 3/3, Scanner/SNRSpec-v0 3/3, Scanner/T1Transmit-v0 0/3; no-op valid with reward 0.0 on 12 of 12; gold is valid exactly where the native reference passes; it does not pass natively on every instance of: Scanner/T1Transmit-v0 (0 of 3) |
| Inspect AI | `integrations/inspect/mrgym_inspect.py` | `python integrations/inspect/mock_run.py` | inspect-ai 0.3.292 mock model: 15 of 15 episodes equal the native API; reference reward 1.0 on 9 of 9 episodes whose native reference passes; no-op reward 0.0 on 15 of 15; the reference does not pass natively for: Scanner/T1Transmit-v0 |
| OpenEnv | `integrations/openenv/mrgym_scanner_env/` | `openenv validate <dir> --level static --skip-build`; `python integrations/openenv/demo.py` | openenv validate --level static --skip-build: static.manifest PASS, 9 SKIP (graders not implemented in that build), 0 FAIL, exit 0; demo in process and over HTTP + WebSocket: same reward and digest as the native API; runtime level: not run (needs Docker) |
| Harbor | `mrgym export harbor` -> `datasets/mrgym-<family>-v<N>/` | `python -m mrgym.compat.harbor lint` and `local`; `harbor lint` | 4 exported task(s): `python -m mrgym.compat.harbor lint` 0 finding(s); the scanner service, the reference over HTTP and the verifier without Docker give the native reward and scanner-log head on 4 of 4 (oracle passes on 4, as natively on 4; no-op 0 on all); `harbor lint --extend-select ALL` on a copy without the Compose file: 0 finding(s), exit 0; with the Compose file: exit 2 (the Compose rules need the Docker CLI with the Compose plugin); `harbor run -a oracle|nop`: not run (needs a Docker daemon) |

Adapter parity (`tests/test_adapters_parity.py`, 3 seeds per id, in each framework's environment): harbor equal to the native API; gymnasium equal to the native API; openenv equal to the native API; verifiers equal to the native API; inspect equal to the native API.
<!-- END GENERATED: conformance -->

The Harbor tasks are generated, not committed: `mrgym export harbor` writes them to `datasets/` in about half a
minute. Every export draws new instances; `mrgym export harbor --check` re-renders a folder from its stored seeds.

## What needs Docker and was not run where the state above was measured

- `harbor lint` with the Compose rules (it needs the Docker CLI with the Compose plugin; no daemon). Without them
  the lint runs on a copy of each task without `environment/docker-compose.yaml`.
- `harbor run -p <task> -a oracle` and `-a nop`, and every image build. The images install `mrgym==<version>` from a
  package index, so they also need a release. Until then `python -m mrgym.compat.harbor local <task>` runs the same
  three commands (sidecar, agent side, verifier) in one process.
- `openenv validate <dir> --level runtime` and `--level semantic`.

`make conformance DOCKER=1` runs these.

## What needs a model or an account

- `vf-run mrgym-scanner` and `inspect eval ... --model <provider/model>` against a real model. The verifiers tool
  server (`ScannerToolset`) is exercised in process by the parity test; `vf-run` starts it as a server.
- `harbor check` (a rubric judged by a model), `harbor publish`, `openenv push`, `prime env push`, an entry in the
  inspect_evals register.

## Trust model

In the Gymnasium, verifiers, Inspect and OpenEnv adapters the environment runs inside the evaluator's process. That
hides the world from a model that only sends tool calls. It does not hide it from code running in that process. The
Harbor form separates them: the world lives in a sidecar container, the agent has no episode file, and the verifier
is a third container without network. See [SECURITY.md](../SECURITY.md).
