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

ERRORS = {c.type: c for c in (session.InvalidRequest, session.BudgetExhausted, session.RequestLimit)}


class ToolScanner:
    def __init__(self, env: Env):
        self.env = env

    def _step(self, name, **arguments):
        obs = self.env.step({"name": name, "arguments": arguments})[0]
        if obs.is_error:
            if name == "plan" and "violations" in obs.data:
                return obs
            err = obs.data.get("error", {})
            raise ERRORS.get(err.get("type"), session.ScannerError)(err.get("message", obs.text))
        return obs

    def info(self) -> dict:
        return self._step("scanner_info").data

    def budget(self) -> dict:
        s = self.env.state()
        return {"budget_s": s["scan_time_budget_s"], "used_s": s["scan_time_used_s"], "remaining_s": s["scan_time_left_s"],
                "scans": s["scans"], "requests": s["scanner_requests"], "max_requests": s["max_scanner_requests"]}

    def plan(self, protocol) -> dict:
        return self._step("plan", protocol=protocol).data

    def scan(self, protocol, rois=None, out=None) -> dict:
        return self._step("scan", protocol=protocol, regions=rois or []).data

    def read_regions(self, scan, rois) -> list:
        return self._step("read_regions", scan=scan, regions=rois).data["roi_readouts"]

    def view(self, scan, image=0, kind="magnitude") -> dict:
        return self._step("view", scan=scan, image=image, kind=kind).data


def controls_of(env_or_spec):
    """The control-policy module of an environment (``<family>.controls``), or None for a sandbox."""
    spec = getattr(env_or_spec, "spec", env_or_spec)
    return importlib.import_module(spec.family + ".controls") if spec.family else None


def policy_names(env_or_spec) -> list[str]:
    """Control policies of an environment that play on its own budget (the ones ``run`` accepts by name)."""
    c = controls_of(env_or_spec)
    return [name for name, (_, scale) in c.policies().items() if scale == 1.0] if c else []


def run(env: Env, policy, seed=None, recipe=None) -> StepResult:
    """Play one episode: reset, let ``policy(ToolScanner)`` measure and return an answer, submit it.

    ``policy`` is a callable or the name of one of the environment's control policies. A policy that raises is a
    failed episode: the no-scan prior answer is submitted and the exception text is put in ``info["policy_error"]``
    (never dropped)."""
    c = controls_of(env)
    if isinstance(policy, str):
        fn, budget_scale = c.policies(recipe)[policy]
        if budget_scale != 1.0:
            raise Error(f"{policy} plays with {budget_scale:g} times the budget; an environment cannot grant that. "
                        "It runs only inside the admission gate.")
        policy = fn
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
    result.info["policy_error"] = error
    return result
