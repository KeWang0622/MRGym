"""What is asked and how it is scored: the hidden world, the task text, the deliverable, the pass rule, the card."""
import math

from mrgym.envs import family as F
from mrgym.scanner import world as W

FAMILY = "t2_vials"
SUBMIT_DOC = "Deliver the transverse relaxation time T2 of every marked region, in milliseconds."
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


def statement(cfg: dict) -> str:
    """Everything the agent is told about this instance."""
    t = cfg["task"]
    return (f"# Measure T2 in the marked regions\nA test object with vials of unknown materials lies in the field of view.\n"
            f"Regions (circles; centre and radius in mm from the isocentre):\n{F.region_table(t['regions'])}\n"
            f"Deliver T2 of every region in milliseconds with `submit`. Passing: every value lies within a factor "
            f"{1 + t['tolerance']:g} of the true T2. The scanner's log must show that you measured.")


def submit_schema(cfg: dict) -> dict:
    ids = [r["id"] for r in cfg["task"]["regions"]]
    values = {"type": "object", "additionalProperties": False, "required": ids, "description": "T2 of every marked region, milliseconds",
              "properties": {i: {"type": "number", "minimum": 0.001} for i in ids}}
    return {"type": "object", "additionalProperties": False, "required": ["t2_ms"], "properties": {"t2_ms": values}}


def check(cfg: dict, answer, exam: dict):
    """None, or why the deliverable cannot be evaluated (the episode then continues)."""
    ids = {r["id"] for r in cfg["task"]["regions"]}
    shaped = isinstance(answer, dict) and isinstance(answer.get("t2_ms"), dict) and set(answer["t2_ms"]) == ids
    if not shaped or not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in answer["t2_ms"].values()):
        return 'the deliverable is {"t2_ms": {<region id>: <positive number, milliseconds>}} for exactly the marked regions'
    return None


def card() -> dict:
    """The environment card: ``python tools/generate.py`` renders it to envs/<id>/README.md and adds the gate result."""
    from . import controls
    must = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "reported") for n in controls.REPORTED]
    return dict(
        id="Example/T2Vials-v0", title="T2 of three marked vials", kind="candidate",
        given="A QA phantom with vials of hidden materials; the scanner; a scan-time budget",
        deliver="T2 of every marked vial, in ms", scored_by="The hidden T2 of every vial: each delivered value within a factor 1.15 of it",
        summary="The smallest complete family: a hidden object, a measurement the agent must plan, a multi-coordinate deliverable.",
        world=dict(phantom="QA phantom (a disc of gel with vials), 1.5 T", sequences_useful=["se"],
                   agent_side_formulas="S(TE) = S0 exp(-TE / T2) at a fixed, long repetition time"),
        hidden=["T1, T2 and proton density of every material", "noise level", "the 128-bit world seed"],
        told=["the regions", "the pass tolerance", "the instrument datasheet", "realised protocol and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        limits=dict(max_scanner_requests=24, max_tool_calls=24),
        deliverable={"tool": "submit", "format": {"t2_ms": {"<region id>": "<milliseconds>"}}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan"],
                    error="largest over the regions of |ln(T2_delivered / T2_true)|",
                    score="none: the reward is 1 for a pass and 0 otherwise", passing="all gates and, for every region, error <= ln(1.15)"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in must],
        determinism="same (seed, level, tool calls) -> same observations, log and reward; the verifier reads the hidden T2 and scans nothing")
