# Changelog

MR Gym versions anything that can move a reward. A change that cannot alter any reward is a patch. A change that
needs stored episodes to be re-scored is a minor version. A change that needs episodes to be re-run is a major
version.

Three versions travel with every number:

| Version | Where | It changes when |
|---|---|---|
| Package | `mrgym.__version__`, `pyproject.toml` | the rule above |
| Engine | `mrgym.ENGINE_ID`, printed by `mrgym --version` | the same seed and protocol can give other scan data. Bumped by hand; `tests/test_engine_version.py` pins it with reference numbers. Recorded logs replay only on the engine that wrote them. |
| Environment | the `-vN` suffix of the id | a reward of that environment can change. The old id is never reused. |

## Unreleased

## 0.1.0.dev0

Research preview. Not released: no tag, no upload.

Engine: `mrgym-engine-0.1.0`.

The repository was rebuilt around the scanner. Nothing before this entry was released, so there is no upgrade
path to describe.

- Package renamed to `mrgym`: one distribution, one import name, one command.
- Scanner engine that forms images in k-space, with six sequences: `spgr`, `afi`, `se`, `ir`, `bssfp`, `epi`.
- Three phantoms: brain, abdomen, QA.
- Console: every plan and scan returns the realised protocol, timing, scan time, limits and a header line.
- Scan log: append-only and hash-chained. Replay is bit-exact on the same numerical stack and within a stated
  tolerance on another one; the replay report says which.
- Explicit engine version in place of a hash of source files.
- Environment surface: `mrgym.make`, `reset`, `tools`, `step`, with tool schemas in OpenAI, Anthropic and MCP form.
- Registry of environment ids of the form `Namespace/Name-vN`, each with a kind: sandbox, practice or admitted.
- Three environments:
  - `Scanner/Explore-v0`: sandbox, no reward.
  - `Scanner/NoWrap-v0`: scored; listed as admitted by its control-policy gate. Counts are on its card.
  - `Scanner/T1Transmit-v0`: scored practice; its own admission gate rejects it.
- `mrgym check <id>`: the gate for publishing an id. With `--admission` it reruns the control policies.
- `mrgym validate`: physics checks against analytic formulas, printed as a table.
- Recorder and viewer: `mrgym record` saves an episode as one bundle, `mrgym view` writes a one-file HTML viewer.
- Model runner: `mrgym run-agent` plays an environment with a language model through an OpenAI-compatible endpoint.
- The code-repair task set `recon_repair` moved to `extras/recon_repair/`. It is not part of the package and is
  not counted in any result.
- Tests that keep MRI Agent Bench material out of the tree (`tests/test_no_benchmark_content.py`).
