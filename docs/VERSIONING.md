# Versioning

Three versions travel with every number: the environment id, the scanner engine and the package.

| Version | It changes when | Enforced by |
|---|---|---|
| Environment id `-vN` | anything that can change the reward a fixed policy earns: the scoring rule or a tolerance, the pass threshold, the task statement, a tool description or schema, the observation text, the budget, the pricing, the tool-call limit, the levels, the world prior, the engine | `tests/test_id_fingerprint.py`: the fingerprint of the id (`python -m mrgym.compat.fingerprint`) must equal the stored one |
| Engine id | the same seed and protocol can give other data | `tests/test_engine_version.py` |
| Package `X.Y.Z` | Z: no reward and no interface change. Y: new ids, new adapters, additive interface; while the package is 0.y also removals, each listed under "Removed" in the changelog. X >= 1: any removal or signature change of the public interface | `tests/test_versioning.py`: the public names of the first snapshot are still exported |
| Adapter packages | they pin `mrgym==X.Y.Z` exactly | `mrgym export ... --check` |

## An id is permanent

An id is in exactly one of three states and is never reused.

- **Served.** `mrgym.make(id)` works.
- **Deprecated.** Still served. `mrgym.registry.deprecate(id, since, successor)` makes the first `make(id)` of a
  process warn: `<id> is out of date since mrgym <since>; use <successor>`.
- **Retired.** Not runnable in this package. `mrgym.registry.retire(id, last_version, successor=None)` makes
  `make(id)` and `spec(id)` raise `mrgym.RetiredEnv` (a subclass of `UnknownEnv`): `<id> is retired: mrgym
  <last_version> is the last package version that served it (pip install mrgym==<last_version> reproduces it).
  Successor: <successor>.` Registering a retired id again is an error.

`mrgym.make("Scanner/NoWrap")` without a version suffix is an error. Picking the latest version silently would make
stored results ambiguous.

Which ids retire, and when, is decided at a release. Retired so far: `Scanner/NoWrap-v0` and
`Scanner/T1Transmit-v0` (last served by mrgym 0.1.0; successors `Scanner/NoWrap-v1` and `Scanner/T1Transmit-v1`).
Their reward was graded and their admission was decided per instance on engine 0.1.0; the successors follow the
pass-only contract of `envs/family.py`. `mrgym list` prints the retired ids, and `envs/index.json` lists them.

## The fingerprint of an id

`python -m mrgym.compat.fingerprint --check` recomputes, for every registered id, a hash over: the registration
(levels, default level, tool-call limit, reward range and threshold), the engine id, the rule fields of the
environment card, the code of the task and environment modules (as a syntax tree, so comments and docstrings do not
count), and per level the reset text, the tool schemas and one scripted episode (calls, observation digest,
scanner-log head, reward). The stored values are in `src/mrgym/compat/fingerprints.json`.

A fingerprint that changed under an unchanged id fails the test and names the parts that moved. Then:

- the id has never been released (`"frozen": false`): decide whether the change is wanted, and store the new
  fingerprint with `python -m mrgym.compat.fingerprint --write`;
- the id is released (`"frozen": true`, set with `--freeze <id>` at the release): undo the change, or register
  the next id (`-vN+1`) and deprecate or retire the old one. `--write` refuses to rewrite a frozen fingerprint.

A stored id that is neither registered nor retired also fails: an id never disappears silently.

## Releases

A released package version is not modified. Results name the package version, the engine id and the environment id
they were measured on. A retired id's results stay valid for the package version named in its `RetiredEnv` message.
