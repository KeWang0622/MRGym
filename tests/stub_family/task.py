"""``Test/Stub``: deliver the number printed on the task card. Level 1 draws it from 1000 values (no constant answer
passes), level 2 from 2 values (a constant answer passes half of the instances: the gate must reject the level)."""
from mrgym.envs import family as F

FAMILY = "stub"
SUBMIT_DOC = "Deliver the number on the task card."
WORLD = dict(kind="qa", n=96)
LEVELS = {1: dict(summary="one of 1000 numbers", numbers=1000, budget_s=20.0),
          2: dict(summary="one of 2 numbers", numbers=2, budget_s=20.0)}
CHEAP_SCAN = dict(sequence="epi", matrix=32, tr_ms=50.0)


def number(seed: str, level: int) -> int:
    return int(seed[:8], 16) % LEVELS[level]["numbers"]


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    task = dict(family=FAMILY, regions=[], number=number(seed, level), deliverable={"value": "<integer>"})
    return dict(task_id=task_id or "stub-token", family=FAMILY, level=level, seed=seed, world_spec=WORLD,
                budget_s=LEVELS[level]["budget_s"], max_requests=8, task=task)


def statement(cfg: dict) -> str:
    return f"# Deliver the number {cfg['task']['number']}"


def submit_schema(cfg: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": ["value"],
            "properties": {"value": {"type": "number", "description": f"the number on the task card (level {cfg['level']})"}}}


def check(cfg: dict, answer, exam: dict):
    if not isinstance(answer, dict) or set(answer) != {"value"} or isinstance(answer["value"], bool) \
            or not isinstance(answer["value"], (int, float)):
        return 'the deliverable is {"value": <number>}'
    if answer["value"] < 0:
        return "the number is not negative"
    return None


def score(cfg: dict, answer, world=None, engine=None) -> dict:
    problem = check(cfg, answer, None)
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    off = abs(answer["value"] - cfg["task"]["number"])
    return F.result(dict(deliverable_valid=True, right_number=off == 0), "" if off == 0 else f"off by {off:g}", error=float(off))


def card() -> dict:
    return dict(id="Test/Stub-v0", title="Deliver the number on the task card", kind="candidate")
