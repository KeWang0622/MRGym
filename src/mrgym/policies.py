"""Scripted policies that play through the public tool surface only.

``ToolScanner`` gives the interface the control policies are written against (``info`` / ``plan`` / ``scan`` /
``view`` / ``read_regions`` / ``budget``) on top of ``env.step(tool_call)``. If a control policy can run through it
unchanged, the tool surface is complete. ``run`` plays one episode and submits whatever the policy returns.
"""
from __future__ import annotations

import importlib

from .env import Env, StepResult
from .registry import Error
from .scanner import session

ERRORS = {"budget": session.BudgetExhausted}                           # every other refusal is an InvalidRequest
PLAN_REFUSALS = ("console_limit", "invalid_value", "unknown_parameter", "not_applicable")


class ToolScanner:
    """The scanner interface of the control policies (``info, budget, plan, scan(protocol, rois), read_regions, view,
    profile``) on top of ``env.step``. Every call is a tool call of the episode: it is counted, logged and charged
    like an agent's.

    One thing is not a tool. ``view`` hands a scripted policy the image array and the extent of its bright pixels
    (``bounding_box``), as ``LocalScanner.view`` does in process. Both are read on the trainer's side from the stored
    scan after the ``view`` tool was called; the tool itself returns neither. A language model has ``profile``.
    """

    def __init__(self, env: Env):
        self.env = env

    def _readouts(self, rois) -> dict:
        """The circles of a policy as tool arguments: these circles and none of the task's marked regions."""
        marked = any("regions" in t.parameters.get("properties", {}) for t in self.env._tool_specs() if t.name == "scan")
        return dict(regions=[], circles=list(rois or [])) if marked else dict(circles=list(rois or []))

    def _step(self, name, **arguments):
        obs = self.env.step({"name": name, "arguments": arguments})[0]
        if obs.is_error:
            err = obs.data.get("error", {})
            if name == "plan" and err.get("type") in PLAN_REFUSALS:
                return obs
            raise ERRORS.get(err.get("type"), session.InvalidRequest)(err.get("message", obs.text), error=err)
        return obs

    def info(self) -> dict:
        return self._step("scanner_info").data

    def budget(self) -> dict:
        s = self.env.state()
        return {"budget_s": s["scan_time_budget_s"], "used_s": s["scan_time_used_s"], "remaining_s": s["scan_time_left_s"],
                "scans": s["scans"], "requests": s["scanner_requests"], "max_requests": s["max_scanner_requests"]}

    @staticmethod
    def _flat(protocol) -> dict:
        return dict(protocol) if isinstance(protocol, dict) and all(isinstance(k, str) for k in protocol) else {"protocol": protocol}

    def plan(self, protocol) -> dict:
        obs = self._step("plan", **self._flat(protocol))
        if obs.is_error:                                                 # the console's answer to a protocol it cannot realise
            problems = obs.data["error"]["problems"]
            return dict(accept=False, sequence=protocol.get("sequence") if isinstance(protocol, dict) else None, error=obs.data["error"],
                        violations=[dict(code=p.get("code", p["type"].upper()), parameter=p.get("parameter"), message=p["message"]) for p in problems])
        return obs.data

    def scan(self, protocol, rois=None, out=None) -> dict:
        return self._step("scan", **self._flat(protocol), **self._readouts(rois)).data

    def design(self, call) -> dict:
        """A family with the design tools: store a design; ``dict(design=<number>, call, facts)``."""
        return self._step("design", **call).data

    def simulate(self, design, **model) -> dict:
        """A family with the design tools: the figures of a stored design on the public model."""
        return self._step("simulate", design=design, **{k: v for k, v in model.items() if v is not None}).data

    def read_regions(self, scan, rois) -> list:
        return self._step("read_regions", scan_index=scan, **self._readouts(rois)).data["roi_readouts"]

    def view(self, scan, image=0, kind="magnitude") -> dict:
        data = self._step("view", scan_index=scan, image_index=image, kind=kind).data
        stored = self.env.stored_scans()[scan]                           # trainer side, in process
        return dict(data, scan=scan, image=image, rotation_deg=stored.get("rotation_deg", 0.0), array=stored[kind][image],
                    bounding_box=session.image_extent(stored, image))

    def profile(self, scan, axis, image=0, kind="magnitude", **options) -> dict:
        return self._step("profile", scan_index=scan, axis=axis, image_index=image, kind=kind,
                          **{k: v for k, v in options.items() if v is not None}).data


def controls_of(env_or_spec):
    """The control-policy module of an environment (``<family>.controls``), or None for a sandbox."""
    spec = getattr(env_or_spec, "spec", env_or_spec)
    return importlib.import_module(spec.family + ".controls") if spec.family else None


def policy_names(env_or_spec) -> list[str]:
    """Control policies of an environment (the ones ``run`` accepts by name)."""
    c = controls_of(env_or_spec)
    return list(c.policies()) if c else []


def run(env: Env, policy, seed=None, recipe=None) -> StepResult:
    """Play one episode: reset, let ``policy(ToolScanner)`` measure and return an answer, submit it.

    ``policy`` is a callable or the name of one of the environment's control policies. A policy that raises is a
    failed episode: the no-scan prior answer is submitted and the exception text is put in ``info["policy_error"]``
    (never dropped)."""
    c = controls_of(env)
    if isinstance(policy, str):
        table = c.policies(recipe)
        if policy not in table:
            raise Error(f"{env.spec.id} has no control policy {policy!r}; policies: {', '.join(table)}")
        policy = table[policy]
    env.reset(seed=seed)
    sc, error = ToolScanner(env), None
    terminal = next(t.name for t in env._tool_specs() if t.terminal)
    try:
        answer = policy(sc)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
        if env.state()["done"]:
            raise
        answer = c.prior(sc)
    result = env.step({"name": terminal, "arguments": answer})
    if not result.done:                                                  # the deliverable was refused: a failed episode
        refusal = result.observation.text.splitlines()[0]
        budget = sc.budget()
        result = StepResult((result.observation, 0.0, False, True, dict(
            submitted=False, refused=refusal, budget=budget, reason="deliverable_refused",
            score=dict(passed=False, reward=0.0, error=None, diagnostic=refusal,
                       checks=dict(deliverable_valid=False, measured=budget["scans"] > 0)))))
    result.info["policy_error"] = error
    return result
