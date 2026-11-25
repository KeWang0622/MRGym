"""Scoring tables: one fixture module per task family, read by ``tests/test_scoring_table.py``.

A family built on ``mrgym/envs/family.py`` contributes ``tests/scoring/<family>.py`` (the name of its sub-package under
``src/mrgym/envs/``) with three names:

    ENV_ID     = "Scanner/NullTissue-v0"
    INSTANCES  = {"L1": dict(level=1, seed=0), ...}       public training seeds; the key is used in the rows
    ROWS       = [Row(instance, name, deliverable, passed=..., failing=..., reason=..., refused=..., expect=...), ...]

A row says what the verifier must do with one deliverable on one hidden world:

    instance     key of INSTANCES
    name         what the row shows (it becomes the test id)
    deliverable  the arguments of ``submit``, or ``callable(case) -> arguments`` when it depends on the hidden world
    passed       True / False: the verdict of ``task.score`` on the hidden world
    failing      names of the checks that must be False, exactly these (empty for a pass)
    reason       text that must appear in the diagnostic (a fail) or in the refusal (``refused=True``)
    refused      True: ``task.check`` cannot evaluate the deliverable; the environment answers with an error
                 observation and the episode continues
    expect       optional ``callable(case, score)`` with further asserts; ``score`` is the dict of ``task.score`` plus
                 ``answer`` (the deliverable of the row)

Rule for the expected values: they are derived in the fixture module from the documented pass rule and the hidden truth
(``case.truth``, ``case.world["system"]``), with arithmetic written there. They never come from ``task.score``,
``task.check`` or a helper of the task module, and never from a recorded output of them.

The harness adds, for every family, the rows no family should have to write: the two gates of ``family.py`` (the log
replays within budget; at least one accepted scan) for every combination, a pass without a scan, a refusal that does not
end the episode, and the reward being exactly 0.0 or 1.0. It needs at least one passing, one failing and one refused row.
"""
from __future__ import annotations

import dataclasses
import importlib
import pkgutil
from collections.abc import Callable
from typing import Any

import mrgym
from mrgym.env import derive_seed
from mrgym.scanner import kspace
from mrgym.scanner import world as W


@dataclasses.dataclass(frozen=True)
class Row:
    instance: str
    name: str
    deliverable: Any
    passed: bool | None = None
    failing: tuple = ()
    reason: str = ""
    refused: bool = False
    expect: Callable | None = None

    def __post_init__(self):
        if self.refused == (self.passed is not None):
            raise ValueError(f"row {self.name!r}: give either passed=True/False or refused=True")
        if self.passed and self.failing:
            raise ValueError(f"row {self.name!r}: a passing row has no failing check")
        if self.passed is False and not self.failing:
            raise ValueError(f"row {self.name!r}: a failing row names the checks that fail")
        if self.refused and not self.reason:
            raise ValueError(f"row {self.name!r}: a refused row gives the text of the refusal")


class Case:
    """One instance of a family, as the verifier holds it: config, hidden world, engine, truth."""

    def __init__(self, env_id: str, level: int, seed):
        self.env_id, self.level, self.seed = env_id, level, seed
        self.spec = mrgym.spec(env_id)
        self.task = importlib.import_module(self.spec.family + ".task")
        self.seed_hex = derive_seed(env_id, level, "train", seed, environ={})    # the derived seed; cfg["seed"] is the world's
                                                                                # (the same, or a later attempt: family.reseed)
        self.cfg = self.task.make_instance(self.seed_hex, level, task_id="scoring-table")
        self.world = W.make_world(self.cfg["seed"], self.cfg["world_spec"])
        self.engine = kspace.Engine(self.world)
        self.exam = self.world["console"]
        self.truth = W.truth(self.world)

    def answer(self, row: Row):
        return row.deliverable(self) if callable(row.deliverable) else row.deliverable


def modules() -> dict:
    """{family name: fixture module} of every ``tests/scoring/<family>.py``."""
    return {m.name: importlib.import_module(f"{__name__}.{m.name}") for m in pkgutil.iter_modules(__path__)
            if not m.name.startswith("_")}


def family_ids() -> list[str]:
    """Registered ids whose environment is built on ``family.FamilyEnv`` (the ids that owe a table)."""
    return sorted(i for i, s in mrgym.registry.items() if s.entry_point == "mrgym.envs.family:FamilyEnv")


_CASES: dict = {}


def case(module, key: str) -> Case:
    k = (module.ENV_ID, key)
    if k not in _CASES:
        _CASES[k] = Case(module.ENV_ID, **module.INSTANCES[key])
    return _CASES[k]


def registered_modules() -> dict:
    """The fixture modules whose id is registered on this tree, by family name."""
    return {name: m for name, m in sorted(modules().items()) if m.ENV_ID in mrgym.registry}


def failing(score: dict) -> list:
    """Names of the checks of a score dict that are False, sorted."""
    return sorted(k for k, ok in score["checks"].items() if not ok)


def first_row(module, **want):
    """(row, case) of the first row with the given attributes, preferring the instance of the first refused row."""
    home = next(r.instance for r in module.ROWS if r.refused)
    rows = sorted(module.ROWS, key=lambda r: r.instance != home)
    row = next(r for r in rows if all(getattr(r, k) == v for k, v in want.items()))
    return row, case(module, row.instance)
