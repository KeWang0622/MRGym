"""The training-facing side of an environment: what whoever runs the episodes (a trainer, a harness) reads and sets.

``TrainingHooks`` is a mixin placed in front of an ``mrgym.Env`` subclass (``FamilyEnv`` uses it). The agent sees one
thing of it, and only when it is set: a cap on turns, stated with the other limits. Everything else is in ``info``,
which only the runner of the episode receives.

``reset`` info adds
    group       {id, env_id, level, split, task_id, noise_nonce}: episodes with the same ``id`` played the same hidden world
    max_turns   the turn cap (None: no cap besides ``max_steps``)

``step`` info adds, at every step
    step        {turn, tool, ok, error_class, error_type, counted, cost_s, scan_time_left_s, tool_calls_left,
                 rejected_calls_left, turns_left}
                error_class and error_type are the interface's own (``mrgym.interface``): the class is None | "shape" |
                "vocabulary" (the two are the interface tax: the call was not taken as written) | "task" (the instrument
                or the verifier said no) | "internal"; the type is one of ``mrgym.interface.ERRORS``. ``counted``: the
                call counts as a tool call. A call refused with class "shape" or "vocabulary" (for how it is written)
                is a rejected call instead, as is a refused submission; a call refused with class "task" (for what it
                asks) is a tool call, whatever its type

and at the last step of the episode (terminated or truncated)
    outcome       exactly one name of the environment's closed taxonomy (``env.outcomes()``), worst first
    ladder        {submitted, valid, measured, in_budget, passed}: each flag implies the ones before it
    passed, reward, progress
    distance      family-specific numbers on how far the delivered answer was (``score["distance"]``; {} when none)
    counts        {turns, length_cuts, scans, scan_s_used, refused_deliverables} and the interface's own counts
                  (``env.interface_counts()``): {calls, executed, rejected, shape, vocabulary, task_refusals, coerced,
                  internal_errors}. One set of numbers: nothing is counted a second time here
    truncated_by  None | "max_steps" | "rejected_call_limit" | "max_turns" | "stopped" | "length"

Turn cap. The limit every environment has is the number of tool calls (``max_steps``); ``mrgym run-agent`` and
``mrgym calibrate`` use that one. A trainer that counts replies instead can add a cap on turns: a turn is one reply of
the agent, whatever the number of tool calls in it, and the harness says so with ``env.note_turn()`` once per reply,
before it executes the calls of that reply. With ``make(..., max_turns=N)`` the cap is a limit like the others: the
Limits block of the task text, ``info["limits"]``, the ``status`` tool and the status line of every observation state
it (``turn k of N``; when one turn is left the line says to submit in it), and the call of ``note_turn`` that would
start turn N + 1 truncates the episode with reward 0. A harness that stops for a reason of its own (the agent stopped
calling tools, a reply was cut by the token limit) ends the episode with ``env.end_without_submission(reason)`` so
that the episode still has exactly one outcome.

Groups and noise. The same (seed, level, tool calls) gives the same observations: the members of a group share the
hidden world and, for identical calls, the noise (common random numbers; the default). For independent noise on the
same world pass ``reset(options={"noise_nonce": k})`` with a different ``k`` per member; the world, the task and the
verifier's noise-free scans do not depend on it.
"""
from __future__ import annotations

from .. import interface as I
from ..env import ResetNeeded, StepResult
from ..registry import Error

NO_SUBMISSION = ("no_submission_cap", "no_submission_rejected", "no_submission_stopped", "no_submission_length", "invalid_deliverable")
LADDER = ("submitted", "valid", "measured", "in_budget", "passed")
STEP_KEYS = ("turn", "tool", "ok", "error_class", "error_type", "counted", "cost_s", "scan_time_left_s", "tool_calls_left",
             "rejected_calls_left", "turns_left")
FINAL_KEYS = ("outcome", "ladder", "passed", "reward", "progress", "distance", "counts", "truncated_by", "group")
INTERFACE_COUNTS = ("calls", "executed", "rejected", "shape", "vocabulary", "task_refusals", "coerced", "internal_errors")
COUNT_KEYS = ("turns", "length_cuts", "scans", "scan_s_used", "refused_deliverables") + INTERFACE_COUNTS
END_REASONS = {"max_steps": "no_submission_cap", "max_turns": "no_submission_cap", "rejected_call_limit": "no_submission_rejected",
               "stopped": "no_submission_stopped", "length": "no_submission_length"}
HARNESS_REASONS = ("stopped", "length", "max_turns")


def error_class(observation) -> str | None:
    """None for an answered call; else the interface's class of the refusal (``mrgym.interface.CLASSES``)."""
    return (I.error_class(observation.data) or "task") if observation.is_error else None


def ladder_of(score: dict | None) -> dict:
    """The five flags of an episode from its score dict (None: nothing was delivered). Each implies the ones before."""
    checks = (score or {}).get("checks", {})
    raw = [score is not None, checks.get("deliverable_valid", True), checks.get("measured", False),
           checks.get("log_replays_within_budget", False), bool((score or {}).get("passed"))]
    out, ok = {}, True
    for name, flag in zip(LADDER, raw):
        ok = ok and bool(flag)
        out[name] = ok
    return out


def validate(info: dict, outcomes=None, final: bool | None = None) -> list[str]:
    """Problems of a ``step`` info dict against the schema of the module docstring ([] when it is valid)."""
    bad = []
    step = info.get("step")
    if not isinstance(step, dict) or tuple(step) != STEP_KEYS:
        return ["info['step'] is missing or has other keys"]
    if step["error_class"] not in (None,) + I.CLASSES or step["ok"] != (step["error_class"] is None) \
            or (step["error_type"] is None) != step["ok"] or (step["error_type"] or "invalid_value") not in I.ERRORS:
        bad.append("step.ok, step.error_class and step.error_type disagree")
    is_final = "outcome" in info
    if final is not None and final != is_final:
        bad.append("the outcome is given at the last step and only there")
    if not is_final:
        return bad
    bad += [f"missing final key {k}" for k in FINAL_KEYS if k not in info]
    if bad:
        return bad
    if outcomes is not None and info["outcome"] not in outcomes:
        bad.append(f"outcome {info['outcome']!r} is not in the taxonomy")
    flags = [info["ladder"][k] for k in LADDER]
    if tuple(info["ladder"]) != LADDER or any(b and not a for a, b in zip(flags, flags[1:])):
        bad.append("the ladder flags are not a prefix")
    if info["passed"] != flags[-1] or (info["outcome"] == "pass") != info["passed"] or (info["progress"] == 1.0) != info["passed"]:
        bad.append("pass, outcome, ladder and progress disagree")
    if info["reward"] != (1.0 if info["passed"] else 0.0) or not 0.0 <= info["progress"] <= 1.0:
        bad.append("the reward is not the pass, or progress is outside [0, 1]")
    if tuple(info["counts"]) != COUNT_KEYS:
        bad.append("counts has other keys")
    if (info["truncated_by"] is None) != info["ladder"]["submitted"]:
        bad.append("truncated_by and ladder.submitted disagree")
    return bad


class TrainingHooks:
    """Mixin: put it before the environment class (``class FamilyEnv(TrainingHooks, ScannerEnv)``)."""

    def __init__(self, spec, max_turns: int | None = None, **kw):
        super().__init__(spec, **kw)
        if max_turns is not None and int(max_turns) < 1:
            raise Error("max_turns must be at least 1")
        self.max_turns = None if max_turns is None else int(max_turns)
        self._noise_nonce = 0
        self._tr = None
        self._group = None

    # ------------------------------------------------------------------ taxonomy
    def outcomes(self) -> tuple:
        """The closed outcome taxonomy of this environment, worst first. A subclass extends it."""
        return NO_SUBMISSION + ("check_failed", "pass")

    # ------------------------------------------------------------------ lifecycle
    def reset(self, *, seed=None, options: dict | None = None):
        options = dict(options or {})
        nonce = options.pop("noise_nonce", 0)
        if isinstance(nonce, bool) or not isinstance(nonce, int) or not 0 <= nonce < 2 ** 31:
            raise Error("noise_nonce must be an integer in [0, 2^31)")
        self._noise_nonce = nonce
        self._tr = dict(turn=0, length_cuts=0, final=None)
        obs, info = super().reset(seed=seed, options=options or None)
        task_id = info.get("task_id")
        self._group = dict(id=f"{self.spec.id}|L{self.level}|{self.split}|{task_id or self._episode['id']}", env_id=self.spec.id,
                           level=self.level, split=self.split, task_id=task_id, noise_nonce=nonce)
        info.update(group=dict(self._group), max_turns=self.max_turns)
        return obs, info

    def step(self, tool_call: dict) -> StepResult:
        r = super().step(tool_call)
        obs, tr, state = r.observation, self._tr, self.state()
        cls = error_class(obs)
        info = dict(r.info)
        info["step"] = dict(turn=tr["turn"], tool=obs.tool, ok=cls is None, error_class=cls,
                            error_type=(obs.data.get("error") or {}).get("type") if cls else None,
                            counted=bool(I.counts_as_tool_call(obs.data)) if cls else True, cost_s=float(obs.data.get("cost_s") or 0.0),
                            scan_time_left_s=state.get("scan_time_left_s"), tool_calls_left=state["tool_calls_left"],
                            rejected_calls_left=max(0, state["max_rejected_calls"] - state["rejected_calls"]),
                            turns_left=None if self.max_turns is None else self.max_turns - tr["turn"])
        if r.done:
            info.update(self._final(r.info, r.info.get("reason", "max_steps") if r.truncated else None))
        return StepResult((obs, r.reward, r.terminated, r.truncated, info))

    def note_turn(self) -> dict:
        """The harness calls this once per reply of the agent, before executing the calls of that reply.

        Returns ``{turn, max_turns, turns_left, truncated}``. When the cap is already used up the episode is truncated
        with reward 0 and the returned dict also carries the final info (``outcome`` = ``no_submission_cap`` ...)."""
        ep = self._episode
        if ep is None:
            raise ResetNeeded("call reset() before note_turn()")
        if ep["done"]:
            raise ResetNeeded("the episode is over; call reset() to start another")
        tr = self._tr
        if self.max_turns is not None and tr["turn"] >= self.max_turns:
            return dict(turn=tr["turn"], max_turns=self.max_turns, turns_left=0, truncated=True, **self._end("max_turns"))
        tr["turn"] += 1
        return dict(turn=tr["turn"], max_turns=self.max_turns, truncated=False,
                    turns_left=None if self.max_turns is None else self.max_turns - tr["turn"])

    def note_length_cut(self) -> None:
        """The harness reports a reply that the token limit cut (counted in ``counts.length_cuts``)."""
        if self._tr is not None:
            self._tr["length_cuts"] += 1

    def end_without_submission(self, reason: str = "stopped") -> dict:
        """The harness ends an episode the agent did not end: ``reason`` is "stopped" (it stopped calling tools),
        "length" (a reply was cut by the token limit) or "max_turns" (the harness's own turn cap). Reward 0. Returns
        the final info. Calling it on an episode that is already over returns that episode's final info."""
        if self._episode is None:
            raise ResetNeeded("call reset() first")
        if self._episode["done"]:
            ep = self._episode
            return dict(self._tr["final"] or self._final(ep["info"], ep["info"].get("reason", "max_steps") if ep["truncated"] else None))
        if reason not in HARNESS_REASONS:
            raise Error(f"reason must be one of {HARNESS_REASONS}")
        return self._end(reason)

    def training_info(self) -> dict | None:
        """The final info of the episode (None while it runs)."""
        return dict(self._tr["final"]) if self._tr and self._tr["final"] else None

    # ------------------------------------------------------------------ internals
    def _end(self, reason: str) -> dict:
        self._episode.update(done=True, truncated=True, reward=0.0, info={"reason": reason, "submitted": False})
        return self._final(self._episode["info"], reason)

    def _final(self, info: dict, truncated_by: str | None) -> dict:
        tr, state, refused = self._tr, self.state(), int(self._episode.get("invalid_deliverables", 0))
        score = info.get("score") if info.get("submitted") else None
        if score is not None:
            outcome = score.get("outcome") or ("pass" if score.get("passed") else "check_failed")
        else:                                                            # nothing was delivered: a refused submission says more than the limit
            outcome = "invalid_deliverable" if refused else END_REASONS[truncated_by]
        ladder = ladder_of(score)
        final = dict(
            outcome=outcome, ladder=ladder, passed=ladder["passed"], reward=1.0 if ladder["passed"] else 0.0,
            progress=float((score or {}).get("progress", 1.0 if ladder["passed"] else 0.0)), distance=dict((score or {}).get("distance") or {}),
            counts=dict(turns=tr["turn"], length_cuts=tr["length_cuts"], scans=state.get("scans", 0),
                        scan_s_used=state.get("scan_time_used_s", 0.0), refused_deliverables=refused, **self.interface_counts()),
            truncated_by=truncated_by, group=dict(self._group))
        tr["final"] = final
        return dict(final)

    # ------------------------------------------------------------------ the turn cap is a limit like the others
    def _limits(self):
        out = super()._limits()
        return out if self.max_turns is None else dict(out, max_turns=self.max_turns)

    def _limit_sentences(self):
        out = super()._limit_sentences()
        if self.max_turns is not None:
            out = out + [f"Turns: {self.max_turns} (a turn is one reply of yours, whatever the number of tool calls in it)."]
        return out

    def _status(self):
        s = super()._status()
        if self.max_turns is not None and self._tr is not None:
            s = dict(s, turn=self._tr["turn"], max_turns=self.max_turns)
        return s

    def _status_parts(self, s):
        parts = super()._status_parts(s)
        if "max_turns" not in s:
            return parts
        left, terminal = s["max_turns"] - s["turn"], next((t.name for t in self._tool_specs() if t.terminal), "the terminal tool")
        hint = "" if left > 1 or self._episode["done"] else f", {terminal} in your next reply or the episode scores 0" if left == 1 \
            else ", no turn left"
        return parts + [f"turn {s['turn']} of {s['max_turns']}{hint}"]
