"""One environment class for every task family on the scanner, and what their verifiers share.

A family is a package with two modules:

``<family>.task``      what is asked and how it is scored
    FAMILY             short name
    LEVELS             {level: dict(summary=..., ...)}
    make_instance(seed_hex, level, task_id) -> episode config (hidden: seed, world_spec; public: task, budget)
                       optional fourth parameter ``split="train"``: a family that declares it is told the split and
                       draws its held-out ranges there (``mrgym.keys.SPLITS``); one that does not serves the train
                       distribution on every split
    statement(cfg)     -> the task text
    submit_schema(cfg) -> JSON Schema of the deliverable (the arguments of ``submit``)
    SUBMIT_DOC         one paragraph: what ``submit`` delivers
    check(cfg, answer, exam) -> None, or why the deliverable cannot be evaluated (the episode then continues)
    score(cfg, answer, world, engine) -> dict with at least ``passed`` (bool), ``checks`` ({name: bool}), ``diagnostic``
    card()             -> the environment card
  optional, for trainers (see "Progress and outcome" below and docs/TRAINING.md)
    score(...)["progress"], ["distance"]   the family's own progress of a failing answer; numbers on how far it was
    OUTCOMES, outcome(score)               outcome labels of a delivered, measured answer that fails, worst first
    LEVELS[lv]["kind"] = "practice"        a level the gate does not run by default and the admitted claim does not cover
    LEVELS[lv]["hints"], prefill(cfg, scanner), statement(cfg, given)
                                           hint tiers of the task text; scans a level acquires at reset (logged, charged)
    potential(cfg, events)                 the potential of ``mrgym.learn.shaping`` (trainer side)
  optional, for a design family (registered with the class ``mrgym.envs.design_tools:DesignFamilyEnv``)
    DESIGN_TOOLS = dict(kinds=(...))       the generators the ``design`` tool offers; the family then gets ``design`` and
                                           ``simulate``, and its task card holds ``design = dict(limits, specification)``.
                                           The designs of an episode are in ``world["console"]["designs"]`` when ``score`` runs
  optional, for ``mrgym coverage`` (``mrgym.coverage``)
    COMPETENCIES                           ids of the examined competencies the family exercises (rows of the map)

``<family>.controls``  scripted policies that play through the tools only
    REFERENCE          name of the policy that must pass
    MUST_FAIL          names of the shortcut policies that must fail
    REPORTED           names of further policies that are reported
    DESCRIPTIONS       {name: one line}
    policies(recipe=None) -> {name: callable(scanner) -> answer}
    tune(cfgs)         -> the strongest constant answer on other worlds (the ``recipe`` of ``fixed_recipe``)
  optional, for ``mrgym check <id> --learnability`` (``mrgym.learn.check``)
    policy_space()     -> a small parametric family of policies around the reference
    blind_answers(task) -> {label: answer}: answers that use the task card only (bounds ``progress`` from below the object)

Reward. One rule for every family: ``reward = 1`` when the delivered result passes every check on the hidden world and
the scanner's log shows a measurement within the budget, else ``0``. There is no partial credit: a graded score pays
answers that ignore the object (a constant plateau) and moves the optimum away from the pass rule. Everything
continuous (errors, margins, scan seconds) is in ``info["score"]`` for analysis and is never paid.

A deliverable the console cannot evaluate (unknown field, value outside the console's range) is refused like any other
malformed call: an error observation, nothing charged, the episode continues.

Progress and outcome. ``info["score"]["progress"]`` is a diagnostic in [0, 1], never paid:
    1                   exactly when the episode passes
    0                   nothing valid was delivered, nothing was measured, or the log does not replay within the budget
    in [0, 1) otherwise the family's own value (``score(...)["progress"]``) or, by default, ``PROGRESS_FAIL_MAX`` times
                        the share of the task's checks that pass before the first failure, in their declared order
A family's own value must be class-ordered (a worse class of outcome never scores more than a better one), must stay
below 1 for every failing answer, and must not be reachable by an answer that ignores the object:
``mrgym check <id> --learnability`` measures the best blind answer's mean progress against ``mrgym.learn.check.BLIND_PROGRESS_MAX``.
``info["score"]["outcome"]`` is one label of the closed taxonomy ``outcomes(task)``: the first failed stage, worst first.
"""
from __future__ import annotations

import hashlib
import importlib

import numpy as np

from .. import keys
from ..learn import hooks
from ..scanner import console
from ..scanner import session as core
from ..tools import ToolSpec
from .scanner_env import ScannerEnv

GATES = ("log_replays_within_budget", "measured")
PROGRESS_FAIL_MAX = 0.5                 # the default progress of a failing answer stays at or below this: a step at the pass line
PROGRESS_SUP = 0.999                    # no failing answer scores more, whatever a family returns
GATE_OUTCOMES = ("no_measurement", "log_or_budget")
# The rules of the game every family with a target scan states, in the same words (a family whose deliverable is a
# measurement on the agent's own scan does not: there the scan is charged). ``OWN_SCANS``: what binds the agent's own
# scans. ``NOT_CHARGED``: true wherever a verifier re-plans the deliverable itself (``console.plan`` + a noise-free
# acquisition), so that the budget gate (``log_replays_within_budget``) sees the agent's scans only.
OWN_SCANS = ("Your own scans may use any sequence and any parameters the console accepts; only the deliverable is evaluated, "
             "on the scan the task names.")
NOT_CHARGED = "That scan is not charged to the scan-time budget and not limited by it."


def reward_of(passed: bool) -> float:
    """The reward rule of every family (see the module docstring)."""
    return 1.0 if passed else 0.0


def progress_of(score: dict) -> float:
    """The progress of a scored episode (see "Progress and outcome" in the module docstring). ``score`` holds the final
    ``passed`` and ``checks`` (gates included) and, optionally, the family's own ``progress``."""
    checks = score["checks"]
    if score["passed"]:
        return 1.0
    if not checks.get("deliverable_valid", True) or not all(checks.get(g, True) for g in GATES):
        return 0.0
    if score.get("progress") is not None:
        return float(min(max(score["progress"], 0.0), PROGRESS_SUP))
    own = [bool(v) for k, v in checks.items() if k not in GATES]
    lead = next((i for i, ok in enumerate(own) if not ok), len(own))
    return PROGRESS_FAIL_MAX * lead / max(len(own), 1)


def outcomes(task=None) -> tuple:
    """The closed outcome taxonomy of a family, worst first."""
    return hooks.NO_SUBMISSION + GATE_OUTCOMES + tuple(getattr(task, "OUTCOMES", ("check_failed",))) + ("pass",)


def outcome_of(score: dict, task=None) -> str:
    """The outcome label of a delivered answer: the first failed stage."""
    checks = score["checks"]
    if score["passed"]:
        return "pass"
    if not checks.get("deliverable_valid", True):
        return "invalid_deliverable"
    if not checks.get("measured", True):
        return "no_measurement"
    if not checks.get("log_replays_within_budget", True):
        return "log_or_budget"
    return task.outcome(score) if hasattr(task, "outcome") else "check_failed"


class FamilyEnv(hooks.TrainingHooks, ScannerEnv):
    """``Scanner/<Family>-vN``: everything specific to the family lives in ``<spec.family>.task``. The training-facing
    side (per-step ``info``, outcome, turn cap, groups) is ``mrgym.learn.hooks.TrainingHooks``."""

    def __init__(self, spec, **kw):
        super().__init__(spec, **kw)
        self.task = importlib.import_module(spec.family + ".task")
        self._given = (None, None)

    def outcomes(self):
        return outcomes(self.task)

    def make_instance(self, seed_hex, task_id):
        cfg = keys.make_instance(self.task, seed_hex, self.level, task_id, self.split)   # the split only if the family takes it
        return dict(cfg, noise_draw=self._noise_nonce) if self._noise_nonce else cfg     # an independent noise stream, same world

    def task_text(self):
        if not hasattr(self.task, "prefill"):
            return self.task.statement(self._cfg)
        if self._given[0] is not self._session:                         # once per episode: the scans are logged and charged
            self._given = (self._session, self.task.prefill(self._cfg, self._scanner))
        return self.task.statement(self._cfg, self._given[1])

    def terminal_tool(self):
        return ToolSpec("submit", self.task.SUBMIT_DOC + " Ends the episode; a deliverable the console cannot evaluate is refused and "
                        "the episode continues.", self.task.submit_schema(self._cfg), terminal=True)

    def check_deliverable(self, a):
        """``task.check``: None, or why the deliverable cannot be evaluated: a sentence, or a dict with ``message`` and
        optionally ``parameter`` / ``nearest_valid`` / ``fix``, or ``violations`` + ``protocol`` (see ``ScannerEnv``).
        The refusal is an ``invalid_deliverable`` error observation and the episode continues.

        A family whose deliverable is the geometry or timing of one scan may offer ``task.protocol_of(answer)``, the
        protocol that deliverable stands for. When its check says no in a sentence and the console refuses that
        protocol, the refusal carries the console's violations: the parameter, the nearest valid value and a fix."""
        why = self.task.check(self._cfg, a, self._session.exam)
        protocol_of = getattr(self.task, "protocol_of", None)
        if isinstance(why, str) and callable(protocol_of) and isinstance(a, dict):
            try:
                protocol = protocol_of(a)
                violations = console.plan(protocol, self._session.exam).get("violations")
            except (TypeError, KeyError, ValueError):
                return why
            if violations and all(v.get("parameter") in a for v in violations):
                return dict(message=why, protocol=protocol, violations=violations)
        return why

    def finish(self, a, rep):
        s = dict(self.task.score(self._cfg, a, world=self._session.world, engine=self._session.engine))
        s["checks"] = dict(s.get("checks", {}), log_replays_within_budget=bool(rep["ok"] and not rep["over_budget"]),
                           measured=bool(rep["scans"] > 0))
        s["passed"] = bool(s["passed"] and all(s["checks"][g] for g in GATES))
        s["reward"] = reward_of(s["passed"])
        s["progress"], s["outcome"] = progress_of(s), outcome_of(s, self.task)   # diagnostics for trainers; never paid
        failed = [k for k, ok in s["checks"].items() if not ok]
        text = "deliverable received; the episode is over"
        if self.reveal_score:
            text += f"\nreward {s['reward']:.0f} · " + ("every check passed" if s["passed"] else "failed: " + ", ".join(failed))
            if s.get("diagnostic"):
                text += f"\n{s['diagnostic']}"
        return s["reward"], s, text


# ------------------------------------------------------------------------------------------------ for task modules
def reseed(seed_hex: str, attempt: int) -> str:
    """World seed of the ``attempt``-th draw of an instance (attempt 0 is the seed itself). A generator that must
    reject a world (the task would have no valid answer on it) moves to the next attempt, deterministically."""
    return seed_hex if attempt == 0 else hashlib.sha256(f"{seed_hex}|attempt|{attempt}".encode()).hexdigest()[:32]


def public_regions(regions) -> list:
    return [dict(id=r["id"], y_mm=r["y_mm"], x_mm=r["x_mm"], radius_mm=r["radius_mm"]) for r in regions]


def region_table(regions) -> str:
    mm = lambda v: f"{v:.2f}".rstrip("0").rstrip(".")                    # noqa: E731
    return "\n".join(f"  {r['id']}: y {mm(r['y_mm'])}, x {mm(r['x_mm'])}, radius {mm(r['radius_mm'])}" for r in regions)


def noise_free(engine, realised, regions):
    """Verifier side: the noise-free images of a realised protocol and the read-out of ``regions`` on them.
    Returns (complex images [n_images, ny, nx], {region id: read-out row})."""
    img = engine.acquire(realised, None)
    rows = core.roi_readouts(img, regions, core.fov_yx(realised), core.centre_yx(realised), core.rotation_of(realised))
    return img, {r["id"]: r for r in rows}


def region_complex(engine, realised, regions, image=0) -> dict:
    """Verifier side: noise-free complex mean of each region, {id: complex}."""
    _, rows = noise_free(engine, realised, regions)
    return {k: r["complex_mean_magnitude"][image] * np.exp(1j * r["complex_mean_phase_rad"][image]) for k, r in rows.items()}


def result(checks: dict, diagnostic: str = "", **details) -> dict:
    """The dict a ``score`` function returns: ``passed`` is the conjunction of ``checks``."""
    checks = {k: bool(v) for k, v in checks.items()}
    return dict(passed=all(checks.values()), checks=checks, diagnostic=diagnostic, **details)


def kind_by_open_loop(stored: dict | None, margin: float = 25.0) -> dict:
    """Measurement task or decision task, per level, from a gate result (``gate.load(id)``; never typed by hand).

    A family reports a control named ``open_loop``: a fixed list of scans followed by a fixed estimator. A level on
    which it passes within ``margin`` points of the reference does not ask for a choice of the next scan from the
    previous result: a *measurement* task; otherwise a *decision* task. Returns dict(kind: one word when every level
    agrees, else "mixed"; levels: {level: kind}; evidence; sentence: one sentence for the card)."""
    if not stored:
        return dict(kind="undetermined", levels={}, evidence="no stored gate result",
                    sentence="Measurement task or decision task (does a fixed list of scans pass within 25 points of the "
                             "reference?): not determined, the gate result is not stored yet.")
    import mrgym
    if stored.get("engine") != mrgym.ENGINE_ID:                         # a result of another engine says nothing about this one
        return dict(kind="undetermined", levels={}, evidence=f"the stored gate result is from {stored.get('engine')}, the engine is {mrgym.ENGINE_ID}",
                    sentence="Measurement task or decision task (does a fixed list of scans pass within 25 points of the "
                             f"reference?): not determined on this engine, the stored gate result is from {stored.get('engine')}.")
    gaps = {}
    for lv, s in stored["levels"].items():
        rate = {r["policy"]: r["rate"] for r in s["reported"] + s["ablations"]}.get("open_loop")
        if rate is None:
            return dict(kind="undetermined", levels={}, evidence="the stored gate result has no open_loop row",
                        sentence="Measurement task or decision task: not determined, the gate result has no open-loop control.")
        gaps[lv] = round(100 * (s["reference"]["rate"] - rate), 1)
    kinds = {lv: ("decision" if g >= margin else "measurement") for lv, g in gaps.items()}
    n = min(s["n"] for s in stored["levels"].values())
    points = ", ".join(f"{v:g} at level {lv}" for lv, v in sorted(gaps.items()))
    return dict(kind=next(iter(kinds.values())) if len(set(kinds.values())) == 1 else "mixed", levels=kinds,
                evidence=f"reference minus open_loop, points of pass rate: {points} (n = {n} per level, engine {stored['engine']})",
                sentence="By level: " + ", ".join(f"level {lv} is a {k} task" for lv, k in sorted(kinds.items()))
                         + f" (a fixed list of scans with a fixed estimator passes this many points below the reference: {points}; "
                           f"{n} instances per level; {margin:g} points or more make a decision task).")
