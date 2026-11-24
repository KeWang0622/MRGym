"""``mrgym check <id>``: what must hold before an environment id is published.

Structure (every id): reset and step follow the contract and are deterministic; tool schemas are portable; errors
are observations and cost nothing; a repeated scan is a new measurement; nothing hidden is served.
Scored ids: the reference policy reaches the reward threshold through the tools, the no-scan answer and the
wrong-unit answer earn the floor, the card is complete, and the claim in the registry (``kind``) agrees with the
stored control-policy results: an id listed as admitted must have admitted instances on the current engine.
"""
from __future__ import annotations

import json
import math

from . import admission, policies
from . import cards as card_mod
from . import tools as T


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
    specs = env._tool_specs()
    terminal = [t.name for t in specs if t.terminal]
    lint = [p for t in specs for p in T.check_schema(t.parameters, t.name)]
    add("tool schemas stay inside the portable JSON-Schema subset", not lint, "; ".join(lint[:3]))
    add("no tool uses a reserved name; exactly one terminal tool", not {t.name for t in specs} & T.RESERVED and len(terminal) == 1)
    for fmt in ("openai", "mcp", "anthropic"):
        json.dumps(env.tools(fmt))
    add("tools serialise as openai / mcp / anthropic", True)
    # ---- errors are observations and cost nothing
    before = env.state()["scan_time_used_s"]
    bad = [env.step({"name": "nope", "arguments": {}}),
           env.step({"name": "scan", "arguments": {"protocol": {"sequence": "spgr"}, "seed": 1}}),
           env.step({"name": "scan", "arguments": "{not json"}),
           env.step({"name": "scan", "arguments": {"protocol": {"sequence": "spgr", "tr_ms": 3.0, "flip_deg": 10}}}),
           env.step({"name": "plan", "arguments": {"protocol": {"sequence": "ir", "tr_ms": 20000, "ti_ms": 500, "averages": 64}}})]
    add("malformed and refused calls are error observations, not exceptions", all(s.observation.is_error for s in bad[:4]) and not any(s.done for s in bad))
    add("refusals and plans cost no scan time", env.state()["scan_time_used_s"] == before)
    # ---- determinism and noise
    prot = {"sequence": "spgr", "matrix": 48, "tr_ms": 12.0, "flip_deg": 8.0}
    a1 = env.step({"name": "scan", "arguments": {"protocol": prot}}).observation
    a2 = env.step({"name": "scan", "arguments": {"protocol": prot}}).observation
    add("a repeated scan is a new noise draw", not a1.is_error and a1.data["data_sha256"] != a2.data["data_sha256"])
    env.step({"name": "view", "arguments": {"scan": 0}})
    env_b = mrgym.make(id, level=level, seed=seed)
    env_b.reset()
    for call in [e["call"] for e in env.transcript()[1:]]:
        env_b.step(call if "raw" not in call else call["raw"])
    add("same seed and calls -> identical transcript and log head",
        [e["text"] for e in env.transcript()] == [e["text"] for e in env_b.transcript()] and env._session.chain == env_b._session.chain)
    env_c = mrgym.make(id, level=level, seed=seed + 1)
    env_c.reset()
    env_c.step({"name": "scan", "arguments": {"protocol": prot}})
    add("another seed -> another world", env_c._session.events[-1].get("data_sha256") != env._session.events[2].get("data_sha256"))
    # ---- nothing hidden is served
    world, cfg = env._session.world, env._cfg
    secret = [cfg["seed"]] + [f"{c['t1_ms']:.6g}" for c in world["tissues"].values()] \
        + [f"{world['system']['tx_gain']:.6g}", f"{world['system']['sigma0']:.6g}", f"{world['system']['cf_offset_hz']:.6g}"]
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
    add("environment card is complete" + (" and carries measured controls" if scored else ""), not problems, "; ".join(problems[:3]))
    if scored:
        # ---- oracle and floor, through the tools
        measured = card["measured"]
        k, n = admission.counts(measured)
        ref_seed = seed
        if measured and k:                                               # an admitted instance of this level, if there is one
            adm = [i["seed"] for i in measured["levels"].get(str(level), {}).get("instance_list", []) if i["admitted"]]
            ref_seed = adm[0] if adm else seed
        ref = policies.run(mrgym.make(id, level=level, seed=ref_seed), "reference")
        add("reference policy reaches the reward threshold through the tools", ref.reward >= spec.reward_threshold and not ref.info["policy_error"],
            f"reward {ref.reward:.3f}, seed {ref_seed}" + (f", {ref.info['policy_error']}" if ref.info["policy_error"] else ""))
        floor = policies.run(mrgym.make(id, level=level, seed=ref_seed), "prior")
        add("no-scan prior answer earns the floor", floor.reward == spec.reward_range[0], f"reward {floor.reward:.3f}")
        unit = policies.run(mrgym.make(id, level=level, seed=ref_seed), "wrong_units")
        add("the reference's answer in the wrong unit earns the floor", unit.reward == spec.reward_range[0], f"reward {unit.reward:.3f}")
        # ---- the claim against the evidence
        if spec.kind == "admitted":
            add("listed as admitted: the stored control-policy run admits instances on this engine",
                bool(measured) and k > 0 and measured["engine"] == mrgym.ENGINE_ID, f"{k} of {n} admitted" if measured else "no stored result")
        else:
            add("listed as practice: a stored control-policy run exists on this engine", bool(measured) and measured["engine"] == mrgym.ENGINE_ID,
                f"{k} of {n} admitted" if measured else "no stored result")
    if verbose:
        for name, ok, detail in res:
            print(f"{'ok  ' if ok else 'FAIL'} {name}{'  (' + detail + ')' if detail else ''}")
    return res
