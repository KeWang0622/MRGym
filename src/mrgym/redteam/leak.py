"""The leak walk: does anything an agent can receive carry the seed, an identifier derived from it, or a hidden value?

Needles of one episode
    the world seed            every window of 8 hex digits
    its v0 identifiers        every window of 8 hex digits of SHA-256("task-id:" + seed) and SHA-256("episode:" + seed)
    the trainer-side label    the task id as a whole and its 12-character suffix
    the canary                it marks exported records and must never be in text a model reads
    hidden values             every number of the hidden system state and of the region truth: a number in a payload that
                              agrees with one of them to 6 significant digits
plus, in structured payloads, any key that matches ``seed|task_id|admitted|tolerance``.

Chance. Payloads are full of measured numbers, and one of them agrees with one of some thirty hidden numbers to 6 digits
about once in a hundred episodes. A leak is systematic, chance is not: a 6-digit agreement counts as a hit when it
also holds to 8 digits, or when the same hidden quantity shows up in the same payload in two or more episodes.

Haystack: the text and the JSON of the reset observation and of every tool result (also in the MCP and OpenAI wire
shapes), the status, ``env.state()``, the tool schemas, refusals and error text. ``info`` is the trainer's and is not
walked. The service's bodies are walked by the same functions (``hits`` on the raw response bytes).
"""
from __future__ import annotations

import hashlib
import json
import re

from .. import keys
from . import common

KEY_RE = re.compile(r"seed|task_id|admitted|tolerance", re.I)
NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])")
WINDOW = 8


def _windows(text: str, n: int = WINDOW) -> set:
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _numbers(obj, path, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _numbers(v, f"{path}.{k}", out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _numbers(v, f"{path}[{i}]", out)
    elif isinstance(obj, float) and obj != 0.0 and obj == obj and abs(obj) != float("inf"):
        out[path] = obj


def needles(seed_hex: str, task_id: str | None, world: dict | None) -> dict:
    """``dict(text={needle: what}, hidden={name: value})``."""
    text = {}
    for w in _windows(seed_hex):
        text[w] = "world seed"
    for label in ("task-id:", "episode:"):
        for w in _windows(hashlib.sha256((label + seed_hex).encode()).hexdigest()):
            text[w] = f"SHA-256({label!r} + seed)"
    if task_id:
        text[task_id] = "task id"
        text[task_id.rsplit("-", 1)[-1]] = "task id suffix"
    text[keys.CANARY] = "canary"
    hidden = {}
    if world:
        _numbers(world.get("system"), "system", hidden)
        try:
            from ..scanner import world as W
            _numbers(W.truth(world), "truth", hidden)
        except Exception:  # noqa: BLE001  a world without region truth
            pass
    return dict(text=text, hidden=hidden)


def _keys(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "fix" and isinstance(v, dict):                      # a fix is a patch of the caller's OWN arguments: its keys
                continue                                                # are what the agent wrote (e.g. an unknown "seed"), not a leak
            if KEY_RE.search(str(k)):
                out.add(str(k))
            _keys(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _keys(v, out)


def _digits(token: str) -> int:
    return len(token.split("e")[0].split("E")[0].replace("-", "").replace(".", "").lstrip("0"))


def hits(haystack: list, needle_map: dict, structured: list = ()) -> tuple:
    """(hits, candidates). A hit is ``(what, needle, where)``: a text needle found, a forbidden key, or a hidden value
    matched to 8 digits. A candidate is ``(name, where)``: a hidden value matched to 6 digits only (see the module text)."""
    found, candidates = [], []
    hidden = list(needle_map.get("hidden", {}).items())
    for where, text in haystack:
        for needle, what in needle_map["text"].items():
            if needle in text:
                found.append((what, needle, where))
        if not hidden:
            continue
        for tok in set(NUMBER_RE.findall(text)):
            if len(tok) > 24 or _digits(tok) < 6:
                continue
            x = float(tok)
            for name, v in hidden:
                if abs(x - v) <= 5e-7 * abs(v):
                    if _digits(tok) >= 8 and abs(x - v) <= 5e-9 * abs(v):
                        found.append((f"hidden value {name}", tok, where))
                    else:
                        candidates.append((name, where))
    bad = set()
    for obj in structured:
        _keys(obj, bad)
    return found + [("forbidden key", k, "structured payload") for k in sorted(bad)], candidates


def _obs(obs, where, haystack, structured):
    haystack.append((where + ".text", obs.text))
    haystack.append((where + ".data", json.dumps(obs.data, default=str)))
    haystack.append((where + ".status", json.dumps(obs.status, default=str)))
    mcp = obs.to_mcp()
    haystack.append((where + ".mcp", json.dumps({k: v for k, v in mcp.items() if k != "content"}, default=str)
                     + "".join(b.get("text", "") for b in mcp["content"])))
    haystack.append((where + ".openai", json.dumps([m for m in obs.to_openai("c0") if m["role"] == "tool"])))
    structured.append(obs.data)
    structured.append(obs.status)


def episode_payloads(env) -> tuple:
    """Play a fixed script that touches every tool and every kind of refusal. Returns (haystack, structured)."""
    from .. import policies
    haystack, structured = [], []
    obs, _ = env.reset()
    _obs(obs, "reset", haystack, structured)
    for fmt in ("openai", "mcp", "anthropic"):
        haystack.append((f"tools.{fmt}", json.dumps(env.tools(fmt))))
    sc = policies.ToolScanner(env)
    terminal = common.terminal_name(env)
    junk = next((p for p in common.JUNK_PROTOCOLS if sc.plan(p).get("accept")), common.JUNK_PROTOCOLS[-1])
    script = [("scanner_info", {}), ("scanner_info", {"topic": "sequences"}), ("scanner_info", {"topic": "physics"}),
              ("scanner_info", {"topic": "task"}), ("plan", {"protocol": junk}), ("plan", {"protocol": dict(junk, tr_ms=0.001)}),
              ("scan", {"protocol": junk}), ("scan", {"protocol": common.HUGE_PROTOCOL}), ("view", {"scan": 0}),
              ("view", {"scan": 0, "kind": "phase"}), ("view", {"scan": 7}),
              ("read_regions", {"scan": 0, "regions": [dict(id="a", y_mm=0.0, x_mm=0.0, radius_mm=12.0)]}),
              ("no_such_tool", {}), ("scan", {"protocol": "not a protocol"}), (terminal, {}), (terminal, {"seed": 1})]
    for i, (name, args) in enumerate(script):
        step = env.step({"name": name, "arguments": args})
        _obs(step.observation, f"step{i}.{name}", haystack, structured)
        if step.done:
            break
    state = env.state()
    haystack.append(("state", json.dumps(state, default=str)))
    structured.append({k: v for k, v in state.items() if k != "episode_id"})
    controls = policies.controls_of(env)
    if not env.state().get("done"):
        try:
            answer = controls.prior(sc)
        except Exception:  # noqa: BLE001
            answer = {}
        step = env.step({"name": terminal, "arguments": answer})
        _obs(step.observation, "submit", haystack, structured)
        haystack.append(("state.end", json.dumps({k: v for k, v in env.state().items() if k != "reward"}, default=str)))
    return haystack, structured


def confirm(candidates: list, episodes: int) -> list:
    """Hidden-value candidates ``[(episode, name, where)]`` that recur: the same quantity in the same payload in two or
    more episodes. Returns them as hits."""
    count = {}
    for ep, name, where in candidates:
        count.setdefault((name, where), set()).add(ep)
    return [(sorted(eps)[0], f"hidden value {name}", f"in {len(eps)} of {episodes} episodes", where)
            for (name, where), eps in sorted(count.items()) if len(eps) >= 2]


def walk(ctx, n: int) -> dict:
    """The leak walk over ``n`` in-process episodes. Returns ``dict(episodes, payloads, hits, chance_candidates)``."""
    found, cands, payloads = [], [], 0
    for seed in range(n):
        env = ctx.make(seed)
        haystack, structured = episode_payloads(env)
        cfg = env.episode_config()
        h, c = hits(haystack, needles(cfg["seed"], cfg.get("task_id"), env.hidden_world()), structured)
        found += [(seed,) + x for x in h]
        cands += [(seed, name, where.split(".", 1)[-1] if where.startswith("step") else where) for name, where in c]
        payloads += len(haystack)
        env.close()
    confirmed = confirm(cands, n)
    return dict(episodes=n, payloads=payloads, hits=found + confirmed, chance_candidates=len(cands) - sum(1 for _ in confirmed))
