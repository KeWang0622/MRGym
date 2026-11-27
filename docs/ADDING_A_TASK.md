# Adding a task

For someone who knows MRI and wants agents to practise a job on the scanner. A task (the code says *family*) is one
folder with three Python files. The example on this page asks for T2 of three vials. It is complete, and
`python examples/new_family/play.py` runs it: the task text, one scripted episode, then the check on 60 fresh objects.

## What a task is

- A hidden world: an object and a scanner state, drawn from a seed. A new seed must change the right answer.
- What the agent is told: a task text, the tools and a scan-time budget. Nothing about the object.
- A deliverable, handed in with `submit`. Ask for several numbers (one per region), so that a guess almost never passes.
- A pass rule, checked on the hidden world. The reward is 1 when every check passes and the scanner's log shows a scan
  within the budget, otherwise 0. No partial credit.

## The three files

Copy [examples/new_family/](../examples/new_family/) to `src/mrgym/envs/<your_task>/` and leave `play.py` behind. The
package finds the folder by itself: no other file is edited.

`spec.py` registers the id. A new id starts as `candidate`.

```python
SPECS = [dict(id="Example/T2Vials-v0", entry_point="mrgym.envs.family:FamilyEnv", card=f"{__package__}.task:card",
              kind="candidate", family=__package__, levels=(1,), default_level=1, max_steps=24)]
```

`task.py` draws an instance and scores an answer. It also holds the task text (`statement`), the JSON schema of
`submit`, `check` (an answer that cannot be read is refused and the episode goes on) and the card: `card()` returns a
dict that `python tools/generate.py` renders to `envs/<id>/README.md`, with the stored gate result.

```python
LEVELS = {1: dict(world_spec=dict(kind="qa", b0_t=1.5, qa=dict(vial_t1_ms=(300.0, 1500.0), n_ring=(4, 5))),
                  n_regions=3, tolerance=0.15, budget_s=400.0, summary="three vials of unknown T2; two long scans fit the budget")}


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    """The episode config. Hidden from the agent: ``seed`` and ``world_spec``. Told: ``task`` and the budget."""
    lv = LEVELS[level]
    regions = W.make_world(seed, lv["world_spec"], layout_only=True)["regions"][:lv["n_regions"]]
    task = dict(family=FAMILY, regions=F.public_regions(regions), tolerance=lv["tolerance"])
    return dict(task_id=task_id, family=FAMILY, level=level, seed=seed, world_spec=lv["world_spec"],
                budget_s=lv["budget_s"], max_requests=24, task=task)


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    """The pass rule, on the hidden world: every delivered T2 within the tolerance of the true one."""
    world = world or W.make_world(cfg["seed"], cfg["world_spec"], layout_only=True)
    problem = check(cfg, answer, world.get("console") or {})
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    truth, limit = W.truth(world), math.log(1 + cfg["task"]["tolerance"])
    miss = {i: abs(math.log(v / truth[i]["t2_ms"])) for i, v in answer["t2_ms"].items()}
    checks = dict(deliverable_valid=True, **{f"t2_within_tolerance_{i}": m <= limit for i, m in miss.items()})
    return F.result(checks, "; ".join(f"{i} is outside the tolerance" for i, m in miss.items() if m > limit), error=max(miss.values()))
```

`controls.py` holds scripted players. They use the same tools as an agent and never read the world.

```python
REFERENCE = "reference"                                 # must pass: shows the task can be solved through the tools
MUST_FAIL = ("prior", "fixed_recipe", "wrong_units")    # shortcuts that ignore the object: each must fail
ABLATIONS = {}                                          # {name: levels}: the reference minus one decision, must lose there
REPORTED = ("open_loop",)                               # reported, no requirement
PROBE = dict(sequence="se", tr_ms=2500.0, matrix=64)
ECHOES_MS = (20.0, 100.0)


def reference(sc):
    """``sc`` is the agent's side of the scanner: info, budget, plan, scan, read_regions, view. Returns the deliverable."""
    regions = sc.info()["task"]["regions"]
    signal = [{r["id"]: r["complex_mean_magnitude"][0] for r in sc.scan(dict(PROBE, te_ms=te), rois=regions)["roi_readouts"]}
              for te in ECHOES_MS]
    return {"t2_ms": {i: (ECHOES_MS[1] - ECHOES_MS[0]) / math.log(max(signal[0][i] / signal[1][i], 1.001)) for i in signal[0]}}
```

## The players you write, and why

| Player | Must | Why |
|---|---|---|
| `reference` | pass | shows that the task can be solved through the tools, within the budget |
| `prior` | fail | answers without scanning; if it passes, nothing has to be measured |
| `fixed_recipe` | fail | the best constant answer, tuned on other worlds by `tune`; if it passes, the object does not matter |
| `wrong_units` | fail | the right numbers in the wrong unit |
| `open_loop` | reported | a fixed list of scans and a fixed formula; it tells what kind of task this is (below) |

- Add `brute_force` (a sweep without a model) to the must-fail list when the budget could pay for one.
- An ablation is the reference with one decision removed. List it with the levels where it must lose 25 points: that
  shows the level needs the decision.
- If `open_loop` passes within 25 points of the reference, the task tests a measurement done right. If not, the agent
  has to choose its next scan from the last result: a decision task. The card says which.

## The check

```bash
mrgym gate <id> --n 60 --workers 2      # while you develop: about a minute
mrgym gate <id> --n 300 --write         # for admission: stores the result next to the package
```

The gate plays every player on fresh objects, level by level. The reference must pass (95 % lower bound of its pass
rate at least 90 %). Every must-fail player must fail (upper bound at most 10 %: with 300 objects, at most 21 passes).
Every ablation must lose 25 points. The stored result decides the `kind` of the id:

- `candidate`: scored, no stored result yet.
- `admitted`: the stored result passes at every level. Set `kind="admitted"` in `spec.py` only then.
- `practice`: scored, but a level failed the gate, or the task is kept as an exercise. No claim rests on it.

The front page says "ready" for admitted and "experimental" for the other two.

## Checklist

1. The right answer changes from seed to seed and cannot be read off the task text.
2. The deliverable has several numbers; each tolerance is wide enough for a careful measurement.
3. `mrgym demo --id <id>` plays the reference, and it passes.
4. `prior`, `fixed_recipe` and `wrong_units` fail; `tune` looks at other worlds only.
5. `open_loop` is reported, and the card says measurement task or decision task.
6. `mrgym gate <id> --n 60 --workers 2` gives the verdict you expect; then `--n 300 --write`.
7. `mrgym check <id>` passes every line; one line in `FAMILIES` of `src/mrgym/physics_map.py` puts the task on [the map](tasks.md).
8. `tests/scoring/<your_task>.py` holds a scoring table ([CONTRIBUTING.md](../CONTRIBUTING.md#scoring-table)).
9. `python tools/audit/golden.py --write`, `python tools/generate.py`, then `make test` and `make lint`.
10. [CHANGELOG.md](../CHANGELOG.md) has a line, and `kind` in `spec.py` is what the stored result says.
