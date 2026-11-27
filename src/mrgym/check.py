"""``mrgym check <id>``: what must hold before an environment id is published.

Structure (every id): reset and step follow the contract and are deterministic; tool schemas are portable; errors
are observations and cost nothing; a repeated scan is a new measurement; nothing hidden is served.
Scored ids: the family defines the contract of ``envs/family.py``; the reference policy passes through the tools; the
no-scan answer and the wrong-unit answer earn the floor; the card is complete; and the claim in the registry
(``kind``) agrees with the evidence, the stored gate result (``envs/results/<id>.gate.json``, schema ``mrgym-gate/2``):

    sandbox     no reward, no gate
    candidate   the gate has not been stored on this engine yet: nothing is required of it
    practice    scored and outside any claim. A stored gate, if there is one, must say why: it rejects a level the id
                would claim, or its reported ``open_loop`` row (a fixed scan list with a fixed estimator) passes within
                ``OPEN_LOOP_POINTS`` of the reference at every level: a measurement task, where no decision is tested
    admitted    the stored gate exists, was measured on the current engine and admits every level of the claim
                (``gate.claim_levels``)

There is no verdict per instance: every instance of a level is served.
"""
from __future__ import annotations

import importlib
import json
import math

from . import cards as card_mod
from . import gate, policies
from . import tools as T

REFERENCE_SEEDS, REFERENCE_MIN = 3, 2           # the reference is played on this many instances and must pass this many
OPEN_LOOP_POINTS = 0.25                         # a fixed protocol this close to the reference makes the task a measurement task


def open_loop_gap(stored: dict, levels) -> float | None:
    """Largest gap over ``levels`` between the pass rate of the reference and of the reported ``open_loop`` policy in a
    stored gate result; None when a level has no such row."""
    gaps = []
    for lv in levels:
        level = (stored.get("levels") or {}).get(str(lv)) or {}
        row = next((r for r in level.get("reported", ()) if r["policy"] == "open_loop"), None)
        if row is None or "reference" not in level:
            return None
        gaps.append(level["reference"]["rate"] - row["rate"])
    return max(gaps) if gaps else None


def claim_evidence(spec, stored: dict | None, engine_id: str) -> tuple[bool, str]:
    """(the registered ``kind`` agrees with the stored gate result, why). ``stored``: ``gate.load(id)``."""
    levels = [str(lv) for lv in gate.claim_levels(spec)]
    if spec.kind == "sandbox":
        return True, "no reward"
    here = bool(stored) and stored.get("schema") == gate.SCHEMA and stored.get("engine") == engine_id
    verdicts = {lv: (stored["levels"].get(lv) or {}).get("verdict") for lv in levels} if here else {}
    all_admitted = here and all(v == "admitted" for v in verdicts.values())
    said = ", ".join(f"level {lv} {v or 'not run'}" for lv, v in verdicts.items())
    if spec.kind == "candidate":
        return True, ("no gate stored on this engine" if not here else said)
    if spec.kind == "practice":
        if all_admitted:
            gap = open_loop_gap(stored, levels)
            if gap is not None and gap <= OPEN_LOOP_POINTS:
                return True, f"{said}; a fixed protocol (open_loop) passes within {100 * gap:.0f} points of the reference: a measurement task"
            return False, f"the stored gate admits every claim level ({said}) and reports no fixed protocol near the reference: the id is not a practice id"
        return True, ("no gate stored on this engine" if not here else said)
    if spec.kind == "admitted":
        if not stored:
            return False, "no stored gate result"
        if stored.get("schema") != gate.SCHEMA:
            return False, f"the stored result has schema {stored.get('schema')!r}, not {gate.SCHEMA!r}"
        if stored.get("engine") != engine_id:
            return False, f"the stored gate was measured on {stored.get('engine')}, the engine is {engine_id}"
        return all_admitted, said
    return False, f"unknown kind {spec.kind!r}"


def contract_problems(spec) -> list[str]:
    """What a scored family lacks of the contract of ``envs/family.py``."""
    task, controls = (importlib.import_module(f"{spec.family}.{m}") for m in ("task", "controls"))
    bad = [f"task.{n}" for n in ("FAMILY", "LEVELS", "SUBMIT_DOC", "make_instance", "statement", "submit_schema", "check", "score", "card")
           if not hasattr(task, n)]
    bad += [f"controls.{n}" for n in ("REFERENCE", "MUST_FAIL", "ABLATIONS", "REPORTED", "DESCRIPTIONS", "policies", "tune") if not hasattr(controls, n)]
    if bad:
        return ["missing: " + ", ".join(bad)]
    names = [controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED]
    table = controls.policies()
    bad += [f"policy {n} is listed and not in policies()" for n in names if n not in table]
    bad += [f"policy {n} has no description" for n in names if n not in controls.DESCRIPTIONS]
    bad += [f"policy {n} is not a callable of the scanner (the (fn, budget_scale) table is gone)" for n, v in table.items() if not callable(v)]
    bad += [f"ablation {n} names level {lv}, which is not registered" for n, lvs in controls.ABLATIONS.items() for lv in lvs if lv not in spec.levels]
    if set(task.LEVELS) != set(spec.levels):
        bad.append(f"task.LEVELS {sorted(task.LEVELS)} and the registered levels {sorted(spec.levels)} differ")
    if not gate.claim_levels(spec):
        bad.append("every level is a practice level")
    return bad


def check_env(id: str, level: int | None = None, seed: int = 7, verbose: bool = False) -> list[tuple[str, bool, str]]:
    """Returns [(check name, ok, detail)]. All must be ok."""
    import mrgym
    spec = mrgym.spec(id)
    level = spec.default_level if level is None else level
    scored = spec.kind != "sandbox"
    res = []

    def add(name, ok, detail=""):
        res.append((name, bool(ok), detail))

    env = mrgym.make(id, level=level, seed=seed)
    # ---- API
    try:
        env.step({"name": "scanner_info", "arguments": {}})
        add("step before reset raises ResetNeeded", False)
    except mrgym.ResetNeeded:
        add("step before reset raises ResetNeeded", True)
    obs, info = env.reset()
    add("reset returns (Observation, info) with text and tools", isinstance(obs, mrgym.Observation) and obs.text and info["tools"])
    add("info names the instance and carries no verdict per instance", "task_id" in info and "admitted" not in info)
    specs = env._tool_specs()
    terminal = [t.name for t in specs if t.terminal]
    lint = [p for t in specs for p in T.check_schema(t.parameters, t.name, described=not t.see)]     # scan: the parameters plan documents
    add("tool schemas stay inside the portable JSON-Schema subset", not lint, "; ".join(lint[:3]))
    add("no tool uses a reserved name; exactly one terminal tool", not {t.name for t in specs} & T.RESERVED and len(terminal) == 1)
    for fmt in ("openai", "mcp", "anthropic"):
        json.dumps(env.tools(fmt))
    add("tools serialise as openai / mcp / anthropic", True)
    # ---- errors are observations and cost nothing
    before = env.state()["scan_time_used_s"]
    bad = [env.step({"name": "nope", "arguments": {}}),
           env.step({"name": "scan", "arguments": {"sequence": "spgr", "seed": 1}}),
           env.step({"name": "scan", "arguments": "{not json"}),
           env.step({"name": "scan", "arguments": {"sequence": "spgr", "tr_ms": 3.0, "flip_deg": 10}}),
           env.step({"name": "plan", "arguments": {"sequence": "ir", "tr_ms": 20000, "ti_ms": 500, "averages": 64}})]
    add("malformed and refused calls are error observations, not exceptions", all(s.observation.is_error for s in bad[:4]) and not any(s.done for s in bad))
    add("refusals and plans cost no scan time", env.state()["scan_time_used_s"] == before)
    # ---- determinism and noise
    prot = {"sequence": "spgr", "matrix": 48, "tr_ms": 12.0, "flip_deg": 8.0}
    first = env.state()["scans"]                                         # a level may acquire scans at reset
    a1 = env.step({"name": "scan", "arguments": prot}).observation
    a2 = env.step({"name": "scan", "arguments": prot}).observation
    add("a repeated scan is a new noise draw", not a1.is_error and not a2.is_error and a1.data["data_sha256"] != a2.data["data_sha256"])
    env.step({"name": "view", "arguments": {"scan_index": first}})
    env_b = mrgym.make(id, level=level, seed=seed)
    env_b.reset()
    for call in [e["call"] for e in env.transcript()[1:]]:
        env_b.step(call if "raw" not in call else call["raw"])
    add("same seed and calls -> identical transcript and log head",
        [e["text"] for e in env.transcript()] == [e["text"] for e in env_b.transcript()] and env._session.chain == env_b._session.chain)
    env_c = mrgym.make(id, level=level, seed=seed + 1)
    env_c.reset()
    c1 = env_c.step({"name": "scan", "arguments": prot}).observation
    add("another seed -> another world", not c1.is_error and not a1.is_error and c1.data["data_sha256"] != a1.data["data_sha256"])
    # ---- nothing hidden is served
    world, cfg = env._session.world, env._cfg
    # a transmit gain of exactly 1 is a calibrated transmitter, which a task states: it is not a hidden value, and its
    # text ("1") is in every observation
    secret = [cfg["seed"]] + [f"{c['t1_ms']:.6g}" for c in world["tissues"].values()] \
        + [f"{world['system'][k]:.6g}" for k in ("tx_gain", "sigma0", "cf_offset_hz") if not (k == "tx_gain" and world["system"][k] == 1.0)]
    blob = json.dumps([e["text"] for e in env.transcript()]) + json.dumps([e["observation"].data for e in env._log], default=str) + json.dumps(env.state())
    leaks = [s for s in secret if s in blob]
    add("no seed, true T1, transmit gain, noise level or frequency offset in any observation or in state()", not leaks, str(leaks))
    # ---- end of episode
    answer = policies.controls_of(spec).prior(policies.ToolScanner(env)) if scored else {}
    end = env.step({"name": terminal[0], "arguments": answer})
    add("the terminal tool ends the episode; reward is a finite float in the declared range",
        end.terminated and not end.truncated and math.isfinite(end.reward) and spec.reward_range[0] <= end.reward <= spec.reward_range[1])
    add("the scanner log of the episode replays", end.info["replay"]["ok"], str(end.info["replay"]["problems"]))
    try:
        env.step({"name": "scanner_info", "arguments": {}})
        add("step after the end raises ResetNeeded", False)
    except mrgym.ResetNeeded:
        add("step after the end raises ResetNeeded", True)
    short = mrgym.make(id, level=level, seed=seed, max_steps=2)
    short.reset()
    short.step({"name": "scanner_info", "arguments": {}})
    tr = short.step({"name": "scanner_info", "arguments": {}})
    add("max_steps truncates with reward 0", tr.truncated and not tr.terminated and tr.reward == 0.0)
    # ---- card
    card = mrgym.card(id)
    problems = card_mod.lint(card)
    add("environment card is complete", not problems, "; ".join(problems[:3]))
    if scored:
        # ---- the family contract
        problems = contract_problems(spec)
        add("the family defines the contract of envs/family.py (task, controls with callables, levels)", not problems, "; ".join(problems[:3]))
        from .envs import oracle
        if oracle.load(spec) is not None:
            problems = oracle.independence_problems(spec.family)
            add("the oracle does not import the policies and does not name the reference", not problems, "; ".join(problems[:3]))
        # ---- reference and floor, through the tools
        runs = [policies.run(mrgym.make(id, level=level, seed=seed + k), "reference") for k in range(REFERENCE_SEEDS)]
        passed = [r.reward >= spec.reward_threshold and not r.info["policy_error"] for r in runs]
        errors = sorted({r.info["policy_error"] for r in runs if r.info["policy_error"]})
        add(f"reference policy passes through the tools on at least {REFERENCE_MIN} of {REFERENCE_SEEDS} instances", sum(passed) >= REFERENCE_MIN,
            f"passed {sum(passed)} of {REFERENCE_SEEDS} (seeds {seed} to {seed + REFERENCE_SEEDS - 1})" + (f", {errors[0]}" if errors else ""))
        floor = policies.run(mrgym.make(id, level=level, seed=seed), "prior")
        add("no-scan prior answer earns the floor", floor.reward == spec.reward_range[0], f"reward {floor.reward:.0f}")
        unit = policies.run(mrgym.make(id, level=level, seed=seed), "wrong_units")
        add("the reference's answer in the wrong unit earns the floor", unit.reward == spec.reward_range[0], f"reward {unit.reward:.0f}")
        add("the reward is 1 for a pass and 0 otherwise", all(r.reward in (0.0, 1.0) for r in runs + [floor, unit]))
    # ---- the claim against the evidence
    ok, why = claim_evidence(spec, card["gate"], mrgym.ENGINE_ID)
    add(f"listed as {spec.kind}: the claim agrees with the stored gate result", ok, why)
    if verbose:
        for name, ok, detail in res:
            print(f"{'ok  ' if ok else 'FAIL'} {name}{'  (' + detail + ')' if detail else ''}")
    return res
