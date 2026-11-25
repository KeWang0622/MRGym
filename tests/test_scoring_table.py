"""The reward as a table, for every family built on ``mrgym/envs/family.py``.

Each family contributes rows through ``tests/scoring/<family>.py`` (format and rule for the expected values:
``tests/scoring/__init__.py``; how to add one: CONTRIBUTING.md). This file holds what is the same for all of them:

  rows      every row against ``task.check`` and ``task.score`` on the hidden world; the same verdict from the episode
            config alone
  contract  the names a family module must define, and that an instance is a function of the seed

The same deliverables through ``reset`` / ``step`` and the two gates of ``family.py`` are in
``tests/test_scoring_episodes.py`` (it plays episodes, so it is kept out of the mutation run, where every second counts).
"""
import importlib
import json

import pytest

import mrgym
import scoring
from mrgym import tools
from mrgym.envs import family as F
from mrgym.scanner import world as W

MODULES = scoring.registered_modules()
ROWS = [pytest.param(m, r, id=f"{name}:{r.instance}:{r.name}") for name, m in MODULES.items() for r in m.ROWS]
FAMILIES = [pytest.param(m, id=name) for name, m in MODULES.items()]


# ---------------------------------------------------------------------------------------------- the table
def test_every_family_table_is_complete():
    assert MODULES, "no scoring table found under tests/scoring/"
    for name, m in MODULES.items():
        spec = mrgym.spec(m.ENV_ID)
        assert spec.family.rsplit(".", 1)[-1] == name, f"tests/scoring/{name}.py must be named after the family package"
        assert all(r.instance in m.INSTANCES for r in m.ROWS)
        assert {i["level"] for i in m.INSTANCES.values()} <= set(spec.levels)
        kinds = {"pass": sum(r.passed is True for r in m.ROWS), "fail": sum(r.passed is False for r in m.ROWS),
                 "refused": sum(r.refused for r in m.ROWS)}
        assert all(kinds.values()), f"{name}: the table needs passing, failing and refused rows, has {kinds}"
        assert len({(r.instance, r.name) for r in m.ROWS}) == len(m.ROWS), f"{name}: row names repeat"


@pytest.mark.parametrize("env_id", scoring.family_ids())
def test_every_family_has_a_scoring_table(env_id):
    name = mrgym.spec(env_id).family.rsplit(".", 1)[-1]
    if name not in MODULES:
        pytest.xfail(f"{env_id} has no scoring table: add tests/scoring/{name}.py (CONTRIBUTING.md, 'Scoring table')")
    assert MODULES[name].ENV_ID == env_id


@pytest.mark.parametrize("module,row", ROWS)
def test_row(module, row):
    case = scoring.case(module, row.instance)
    answer = case.answer(row)
    problem = case.task.check(case.cfg, answer, case.exam)
    score = case.task.score(case.cfg, answer, world=case.world, engine=case.engine)
    json.dumps(score)                                                    # the score dict goes to info["score"]
    assert type(score["passed"]) is bool and all(type(v) is bool for v in score["checks"].values())
    assert score["passed"] is all(score["checks"].values())
    assert isinstance(score["diagnostic"], str)
    if row.refused:
        assert isinstance(problem, str) and row.reason in problem, problem
        assert score["passed"] is False and scoring.failing(score) and score["diagnostic"] == problem
        assert F.reward_of(score["passed"]) == 0.0
        if row.expect:
            row.expect(case, dict(score, answer=answer))
        return
    assert problem is None
    assert score["passed"] is row.passed
    assert scoring.failing(score) == sorted(row.failing)
    assert row.reason in score["diagnostic"]
    if row.expect:
        row.expect(case, dict(score, answer=answer))


@pytest.mark.parametrize("module", FAMILIES)
def test_the_verifier_rebuilds_the_hidden_world_from_the_config_when_none_is_given(module):
    """``mrgym verify`` scores from the episode config alone: the same verdict as with the live world and engine."""
    for want in (True, False):
        row, case = scoring.first_row(module, passed=want)
        answer = case.answer(row)
        alone = case.task.score(case.cfg, answer)
        assert alone == case.task.score(case.cfg, answer, world=case.world, engine=case.engine) and alone["passed"] is want


# ---------------------------------------------------------------------------------------------- what a family defines
def _keys(obj) -> set:
    """Every dict key at any depth of a JSON-like value."""
    if isinstance(obj, dict):
        return set(obj) | {k for v in obj.values() for k in _keys(v)}
    return {k for v in obj for k in _keys(v)} if isinstance(obj, (list, tuple)) else set()


@pytest.mark.parametrize("module", FAMILIES)
def test_family_modules_define_the_contract(module):
    spec = mrgym.spec(module.ENV_ID)
    task = importlib.import_module(spec.family + ".task")
    controls = importlib.import_module(spec.family + ".controls")
    for name in ("FAMILY", "LEVELS", "SUBMIT_DOC", "make_instance", "statement", "submit_schema", "check", "score", "card"):
        assert hasattr(task, name), f"{spec.family}.task.{name}"
    for name in ("REFERENCE", "MUST_FAIL", "ABLATIONS", "REPORTED", "DESCRIPTIONS", "policies", "tune"):
        assert hasattr(controls, name), f"{spec.family}.controls.{name}"
    assert set(task.LEVELS) == set(spec.levels) and task.card()["id"] == module.ENV_ID
    names = [controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED]
    assert set(names) <= set(controls.policies()) and set(names) <= set(controls.DESCRIPTIONS)
    assert all(set(levels) <= set(spec.levels) for levels in controls.ABLATIONS.values())
    key, inst = next(iter(module.INSTANCES.items()))
    case = scoring.case(module, key)
    again = task.make_instance(case.seed_hex, inst["level"], task_id="scoring-table")
    assert json.dumps(again, sort_keys=True) == json.dumps(case.cfg, sort_keys=True)      # a function of the seed
    other = task.make_instance(mrgym.env.derive_seed(module.ENV_ID, inst["level"], "train", "another", environ={}), inst["level"])
    hidden = lambda cfg: json.dumps(W.make_world(cfg["seed"], cfg["world_spec"], layout_only=True)["tissues"], sort_keys=True)   # noqa: E731
    # another seed is another instance: in what the task card says, in the budget, or (a family whose card is the same for
    # every instance of a level, such as NoWrap) only in the hidden world
    assert other["task"] != case.cfg["task"] or other["budget_s"] != case.cfg["budget_s"] or hidden(other) != hidden(case.cfg)
    assert other["task_id"] != task.make_instance(other["seed"], inst["level"])["task_id"]     # a random token by default
    cfg = case.cfg
    # a generator may reject a world (the task would have no valid answer on it) and move to the next attempt: the world
    # seed is then ``family.reseed(derived seed, attempt)``. The attempt is the verifier's, never the agent's.
    attempt = cfg.get("attempt", 0)
    assert type(attempt) is int and 0 <= attempt <= 64, f"attempt = {attempt!r}: a small non-negative integer"
    assert cfg["seed"] == F.reseed(case.seed_hex, attempt) and cfg["level"] == inst["level"] and cfg["task_id"] == "scoring-table"
    assert {"task_id", "family", "level", "seed", "world_spec", "budget_s", "max_requests", "task"} <= set(cfg)
    assert cfg["family"] == task.FAMILY == cfg["task"]["family"] and type(cfg["budget_s"]) is float
    public = json.dumps(cfg["task"]) + task.statement(cfg) + json.dumps(task.submit_schema(cfg))
    assert "attempt" not in _keys(cfg["task"]) and "attempt" not in _keys(task.submit_schema(cfg))
    for hidden in (case.seed_hex, cfg["seed"]):
        assert hidden not in public and hidden[:8] not in public
    assert tools.check_schema(task.submit_schema(cfg)) == []
