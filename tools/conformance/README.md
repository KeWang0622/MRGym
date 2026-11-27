# Conformance

`make conformance` scores MR Gym against 108 checkable items in 11 themes, merged from the standards of the field
(Gymnasium, OpenEnv, verifiers, Harbor, Terminal-Bench-Science, BetterBench and others; the `source` of each item is
in [`items.json`](items.json)). The score is `(PASS + 0.5 PARTIAL) / items`. The command exits 1 below 90 % or when an
item listed in [`baseline.json`](baseline.json) no longer passes.

| File | What it is |
|---|---|
| `items.json` | the registry: id, item, source, tier, checker, and for an item that does not pass the action that flips it |
| `checks.py` | the checkers; they read the registry of environments, stored results and the validators' output |
| `run.py` | runs the checkers, prints the scorecard, writes `report.json` and the generated block of `integrations/README.md` |
| `report.json` | the last full run: date, commit, every item with its status and the reason, every command with its exit code |
| `baseline.json` | ids that passed at the last accepted run; a run in which one of them does not pass exits 1 |
| `evidence/<ID>.json` | evidence for an item that a command cannot produce |

Tiers say what it takes to flip an item: `T1` engineering without Docker or a model, `T2` CI with Docker, `OWN` an
action of the owner, `MOD` model runs, `TRN` a training run or outside use.

A checker reads the tree as it is. It never assumes a planned change: an item about gate results reads the stored
gate JSON, an item about a validator runs the validator. A checker that cannot run is a FAIL with the reason.

## Evidence files

```json
{"claim": "what was shown", "n": 50, "result": "pass", "who": "name", "who_is": "person", "date": "YYYY-MM-DD",
 "artefact": "path to the stored material", "artefact_sha256": "..."}
```

`result` is `pass`, `partial` or `fail`. Items B12, B13, D9, F8 and I5 are about human judgement and are accepted
only with `"who_is": "person"` and a person's name; an agent or a model cannot sign them.

## Environments of the validators

Each framework's validator runs in its own environment, because their pins collide. The defaults are throw-away
`uv run --isolated` environments with pinned versions. To use existing environments:

```sh
make conformance CONF_GYM="env PATH=/path/gym-venv/bin:$PATH" CONF_VF="env PATH=/path/vf-venv/bin:$PATH" \
    CONF_INSPECT="env PATH=..." CONF_OPENENV="env PATH=..." CONF_HARBOR="env PATH=..."
```
