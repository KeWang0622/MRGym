"""Run agents on MR Gym environments and save one small record per episode.

    mrgym run-agent --env Scanner/NoWrap-v0 --seeds 0-9 --policy all --out runs/first
    mrgym run-agent --env Scanner/NoWrap-v0 --seeds 0-9 --model MODEL --base-url https://HOST/v1 --out runs/first
    mrgym results --dir runs/first

Two kinds of agent go through the same environment, the same scoring and the same record format, so their rows land
in one table:

  scripted   a control policy of the environment (``--policy reference,prior`` or ``--policy all``), played through
             the tool surface by ``mrgym.policies.run``. No model is called. Policies that apply one constant recipe
             use the recipe the stored admission run tuned for that environment and level (``recipe_source``).
  model      a language model behind an OpenAI-compatible chat-completions endpoint (``--model``, ``--base-url``).
             The environment runs in this process: the model has no code execution, it only writes tool calls, and
             the hidden world never enters a message. This is the intended trust model for tool-calling models.

What a model receives (``--dry-run`` prints it and calls nothing): a short session note, the task statement returned
by ``env.reset()`` as the first user message, and the tools of ``env.tools("openai")``. By default one more tool is
added, ``calc``: an arithmetic helper of the session, not of the scanner (``--no-calc`` removes it). The session note
tells the turn cap, and one reminder is sent before the last turn.

Instances. An instance is (environment, level, integer seed of the public train split). Every agent plays the same
instances. ``--episodes`` repeats an instance for a model; repeats differ by the model's sampling only (the
environment is deterministic). A scripted policy is deterministic too, so it always plays one episode per instance.

Failures. An agent failure is a scored episode, never a crash: invalid JSON arguments, an unknown tool, more than
12 calls in one turn (answered with an error), three turns in a row without a tool call (``no_tool_calls``), the turn
cap (``turn_cap``), the environment's own tool-call limit (``truncated``). All of them end with reward 0 unless the
agent submitted. An endpoint failure is not an agent result: after the retries of the chat client the episode is
written as ``status = infra_error``, left out of every rate and run again when the same command is repeated.
Every started episode has a record: it is written as ``infra_error / interrupted`` at the start and replaced at the
end, so a killed run leaves no silent gaps.

Files, under ``<out>``:

    run_config.json                                         what was asked for, per invocation
    episodes/<agent>/<env>/L<level>/s<seed>/e<k>/
        record.json          the result of the episode (see ``RECORD_FIELDS``)
        messages.json        the tools and the conversation as sent to the endpoint (models only)
        scanner_log.jsonl    the scanner's hash-chained log
        episode_bundle.json  a replayable recording, with ``--record-bundles`` and when ``mrgym.record`` offers it

``--records-only`` writes ``record.json`` alone. Resume: repeat the command; finished episodes are skipped.
"""
from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import re
import shlex
import shutil
import sys
import threading
import time

import numpy as np

from . import chat

RECORD_SCHEMA = "mrgym-episode/1"
RUN_SCHEMA = "mrgym-run/1"
MAX_CALLS_PER_TURN = 12
IDLE_TURNS = 3                              # turns in a row without a tool call that end the episode
CHARS_PER_TOKEN = 3.5                       # used only when the endpoint reports no usage; such counts are flagged
MAX_WORKERS = 8
DEFAULT_MAX_TURNS = 20
STOP_REASONS = ("submitted", "turn_cap", "no_tool_calls", "truncated")
# record.json, in this order. Who played; on what; how it ended; the score; the effort; provenance.
RECORD_FIELDS = (
    "schema", "agent", "agent_kind", "model", "params_sent", "params_dropped", "settings",
    "env_id", "level", "seed", "split", "episode", "task_id", "admitted", "engine", "package_version",
    "status", "stop_reason", "infra_error", "recipe_source", "policy_error",
    "reward", "passed", "error", "gates", "diagnostic", "tolerance",
    "scans", "scan_s_used", "scan_s_budget", "scanner_requests", "turns", "tool_calls", "env_tool_calls", "invalid_calls", "by_tool",
    "tokens_in", "tokens_out", "tokens_estimated", "log", "bundle", "wall_s", "utc")
_PRINT_LOCK = threading.Lock()

SESSION_NOTE = """You operate a virtual MR scanner through function tools. The task is in the first user message. Everything about \
the object and about the state of the system has to be measured. The datasheet of the instrument is behind \
`scanner_info` (free).

Rules of this session
- The tools of the scanner are {scanner_tools}. The scanner charges scan time. A refused or invalid call costs no \
scan time, but it counts as a tool call.
{calc_rule}- Deliver by calling `{terminal}` once. The episode ends there and nothing about correctness is returned.
- You have at most {max_turns} turns; one turn is one reply of yours. You may issue up to {max_calls} tool calls in \
one turn. Without `{terminal}` by the last turn the episode scores 0.
"""
CALC_RULE = ("- `calc` is not a scanner tool. It is an arithmetic helper of this session and knows nothing about the "
             "scanner or the object.\n")
CALC_TOOL = {"type": "function", "function": {
    "name": "calc",
    "description": "Arithmetic helper of the session, not a scanner tool: it knows nothing about the scanner or the "
                   "object. Statements separated by ';' or newlines; assignments allowed; lists are arrays and "
                   "operations are element-wise. Functions: sin cos tan arcsin arccos arctan arctan2 exp log log10 "
                   "sqrt abs degrees radians mean median sum min max std; constant pi. Returns the value of every "
                   "statement. Free; not counted by the scanner.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["expressions"],
                   "properties": {"expressions": {"type": "string"}}}}}


# ------------------------------------------------------------------ small helpers
def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def slug(text: str) -> str:
    """A folder name for an agent label or an environment id."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "agent"


def env_slug(env_id: str) -> str:
    return env_id.replace("/", "-")


def agent_kind(model: str) -> str:
    return "stub" if model.startswith("stub:") else "model"


def write_json(path: str, obj, **kw) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, **kw)
        f.write("\n")
    os.replace(tmp, path)


def parse_seeds(text: str) -> list[int]:
    """``"0-9"``, ``"0,3,7"`` or ``"0-3,8"`` -> sorted integers without repeats."""
    out = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        if m:
            lo, hi = int(m[1]), int(m[2])
            if hi < lo:
                raise ValueError(f"seed range {part!r} runs backwards")
            out.update(range(lo, hi + 1))
        elif part.isdigit():
            out.add(int(part))
        else:
            raise ValueError(f"seeds are non-negative integers or ranges like 0-9, got {part!r}")
    if not out:
        raise ValueError("no seeds given")
    return sorted(out)


def seeds_text(seeds) -> str:
    """Integers -> ``"0-9"`` / ``"0-3,8"``."""
    seeds, parts, i = sorted(set(int(s) for s in seeds)), [], 0
    while i < len(seeds):
        j = i
        while j + 1 < len(seeds) and seeds[j + 1] == seeds[j] + 1:
            j += 1
        parts.append(str(seeds[i]) if i == j else f"{seeds[i]}-{seeds[j]}")
        i = j + 1
    return ",".join(parts)


def _split(values) -> list[str]:
    """Repeatable options and comma lists -> a flat list without repeats, order kept."""
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    out = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip()
            if part and part not in out:
                out.append(part)
    return out


# ------------------------------------------------------------------ calc: arithmetic for the agent, nothing else
MAX_CALC_CHARS, MAX_CALC_STATEMENTS, MAX_CALC_ARRAY = 4000, 80, 256


def _reducer(fn):
    """``mean([1, 2, 3])`` and ``max(1, 2, 3)`` both work."""
    return lambda *a: fn(np.asarray(a[0], float)) if len(a) == 1 else fn(np.array([np.asarray(x, float) for x in a]))


_FUNCS = dict(sin=np.sin, cos=np.cos, tan=np.tan, arcsin=np.arcsin, arccos=np.arccos, arctan=np.arctan,
              asin=np.arcsin, acos=np.arccos, atan=np.arctan, arctan2=np.arctan2, atan2=np.arctan2, exp=np.exp,
              log=np.log, ln=np.log, log10=np.log10, sqrt=np.sqrt, abs=np.abs, degrees=np.degrees, radians=np.radians,
              mean=_reducer(np.mean), median=_reducer(np.median), sum=_reducer(np.sum), min=_reducer(np.min),
              max=_reducer(np.max), std=_reducer(np.std))
_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}
_CALC_ERRORS = (ValueError, SyntaxError, TypeError, IndexError, ZeroDivisionError, OverflowError, RecursionError, MemoryError)


def _ev(node, names):
    """Evaluate one whitelisted node. Anything else (attributes, imports, strings, comprehensions) is refused."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numbers are allowed")
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        if node.id in names:
            return names[node.id]
        raise ValueError(f"unknown name {node.id!r}")
    if isinstance(node, (ast.List, ast.Tuple)):
        if len(node.elts) > MAX_CALC_ARRAY:
            raise ValueError("list too long")
        return np.array([_ev(e, names) for e in node.elts], float)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_ev(node.left, names), _ev(node.right, names))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _ev(node.operand, names)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_ev(a, names) for a in node.args])
    if isinstance(node, ast.Subscript):
        index = _ev(node.slice, names)
        if not isinstance(index, float) or int(index) != index:
            raise ValueError("an index must be one integer")
        return np.asarray(_ev(node.value, names), float)[int(index)]
    raise ValueError(f"not allowed in calc: {type(node).__name__}")


def _num(v):
    a = np.asarray(v, float)
    if a.size > MAX_CALC_ARRAY:
        raise ValueError("result too large")
    f = lambda x: float(f"{x:.7g}") if math.isfinite(x) else str(x)    # noqa: E731
    return f(float(a)) if a.ndim == 0 else [f(float(x)) for x in a.ravel()]


def calc(text: str) -> list[tuple[str | None, object]]:
    """Evaluate arithmetic statements. Returns ``[(name or None, value)]``, one entry per statement.

    The text is parsed to a syntax tree and only numbers, lists, ``+ - * / **``, the listed functions, integer
    indexing and assignment to plain names are evaluated. Raises ``ValueError`` (or ``SyntaxError``) otherwise.
    """
    if not isinstance(text, str) or len(text) > MAX_CALC_CHARS:
        raise ValueError(f"expressions must be a string of at most {MAX_CALC_CHARS} characters")
    lines = [ln.strip() for ln in text.replace("^", "**").replace(";", "\n").split("\n")]
    tree = ast.parse("\n".join(ln for ln in lines if ln), mode="exec")
    if len(tree.body) > MAX_CALC_STATEMENTS:
        raise ValueError("too many statements")
    names, out = {}, []
    with np.errstate(all="ignore"):
        for st in tree.body:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                name = st.targets[0].id
                if name in _FUNCS or name == "pi":
                    raise ValueError(f"{name} is reserved")
                names[name] = _ev(st.value, names)
                out.append((name, _num(names[name])))
            elif isinstance(st, ast.Expr):
                out.append((None, _num(_ev(st.value, names))))
            else:
                raise ValueError("only assignments and expressions are allowed")
    return out


def calc_call(raw_arguments: str) -> tuple[str, bool]:
    """The ``calc`` tool: raw argument string -> (text for the agent, whether the call was valid)."""
    try:
        args = json.loads(raw_arguments) if (raw_arguments or "").strip() else {}
        if not isinstance(args, dict) or set(args) != {"expressions"}:
            raise ValueError('arguments must be {"expressions": "<statements>"}')
        rows = calc(args["expressions"])
    except _CALC_ERRORS as e:
        return f"error: invalid_tool_call: calc: {e}\ncalc is the arithmetic helper of the session, not a scanner tool", False
    if not rows:
        return "calc: nothing to evaluate", True
    return "calc\n" + "\n".join(f"{name} = {json.dumps(v)}" if name else json.dumps(v) for name, v in rows), True


# ------------------------------------------------------------------ what the agent receives
def world_seed(env) -> str:
    """The 32-hex world seed of the running episode. Runner side only: it is compared against every outgoing
    message and never written to a file."""
    config = getattr(env, "episode_config", None)
    if callable(config):
        return config()["seed"]
    from ..env import derive_seed
    return derive_seed(env.spec.id, env.level, env.split, env._seed)


def agent_tools(env, calc_tool: bool = True) -> tuple[list[dict], str]:
    """(tools in the OpenAI wire format, name of the terminal tool). Call after ``env.reset()``."""
    neutral = env.tools("neutral")
    tools = env.tools("openai")
    terminal = next(t["name"] for t in neutral if t["terminal"])
    if calc_tool and all(t["name"] != "calc" for t in neutral):
        tools = tools + [CALC_TOOL]
    return tools, terminal


def first_messages(env, task_text: str, max_turns: int, calc_tool: bool = True) -> list[dict]:
    """The session note (system) and the task statement of ``env.reset()`` (user)."""
    neutral = env.tools("neutral")
    terminal = next(t["name"] for t in neutral if t["terminal"])
    has_calc = calc_tool and all(t["name"] != "calc" for t in neutral)
    note = SESSION_NOTE.format(scanner_tools=", ".join(f"`{t['name']}`" for t in neutral), terminal=terminal,
                               calc_rule=CALC_RULE if has_calc else "", max_turns=max_turns, max_calls=MAX_CALLS_PER_TURN)
    return [{"role": "system", "content": note}, {"role": "user", "content": task_text}]


def model_settings(max_turns: int, calc_tool: bool, temperature, max_tokens) -> dict:
    """Everything of the harness that can move a model's result. Episodes of one agent in one folder must agree."""
    return dict(max_turns=int(max_turns), calc=bool(calc_tool), temperature=temperature, max_tokens=max_tokens,
                max_calls_per_turn=MAX_CALLS_PER_TURN,
                session_note_sha256=hashlib.sha256((SESSION_NOTE + CALC_RULE).encode()).hexdigest()[:12])


# ------------------------------------------------------------------ one model episode
def play_model(env, client: chat.ChatClient, max_turns: int = DEFAULT_MAX_TURNS, calc_tool: bool = True,
               sink: dict | None = None) -> dict:
    """Drive one episode of ``env`` with the model behind ``client``. Resets the environment.

    Returns ``dict(stop_reason, final, turns, tool_calls, invalid_calls, by_tool, tokens_in, tokens_out,
    tokens_estimated, tools, messages, events)``. ``final`` is the last ``StepResult`` when the episode ended inside
    the environment (``submitted`` or ``truncated``), else None. ``events`` are the assistant turns with the number
    of the environment step each led to (the form ``mrgym.record`` takes as ``agent_events``).

    ``sink``, if given, receives ``tools`` and ``messages`` before the first request, so that a caller can save the
    conversation of an episode that an endpoint failure cut short. Raises ``chat.GatewayError`` on an endpoint
    failure and ``AssertionError`` if the world seed is about to be sent.
    """
    obs, _ = env.reset()
    seed_hex = world_seed(env)
    tools, _ = agent_tools(env, calc_tool)
    has_calc = any(t is CALC_TOOL for t in tools)                           # an environment's own `calc` is left to it
    messages = first_messages(env, obs.text, max_turns, calc_tool)
    if sink is not None:
        sink.update(tools=tools, messages=messages)
    tools_json = json.dumps(tools)
    known = {t["function"]["name"] for t in tools}
    out = dict(stop_reason="turn_cap", final=None, turns=0, tool_calls=0, invalid_calls=0, by_tool={}, tokens_in=0,
               tokens_out=0, tokens_estimated=False, tools=tools, messages=messages, events=[])
    idle = 0
    while out["turns"] < max_turns:
        sent = json.dumps(messages)
        if seed_hex in sent or seed_hex in tools_json:                      # checked on every request
            raise AssertionError("the world seed must never reach the agent")
        resp = client.complete(messages, tools)                             # GatewayError: the caller records it
        out["turns"] += 1
        usage = resp["usage"]
        if isinstance(usage.get("prompt_tokens"), int) and isinstance(usage.get("completion_tokens"), int):
            out["tokens_in"] += usage["prompt_tokens"]
            out["tokens_out"] += usage["completion_tokens"]
        else:
            written = len(resp["text"]) + sum(len(c["arguments"]) + len(str(c["name"])) for c in resp["tool_calls"])
            out["tokens_estimated"] = True
            out["tokens_in"] += round((len(sent) + len(tools_json)) / CHARS_PER_TOKEN)
            out["tokens_out"] += round(written / CHARS_PER_TOKEN)
        messages.append(chat.assistant_message(resp["text"], resp["tool_calls"]))
        event = dict(step=None, role="assistant", content=resp["text"],
                     tool_calls=[dict(name=c["name"], arguments=c["arguments"]) for c in resp["tool_calls"]])
        out["events"].append(event)
        for k, c in enumerate(resp["tool_calls"]):
            out["tool_calls"] += 1
            key = c["name"] if c["name"] in known else "(unknown)"
            out["by_tool"][key] = out["by_tool"].get(key, 0) + 1
            if out["final"] is not None:
                messages.append(chat.tool_message(c["id"], "error: episode_over: the episode has ended; this call was not executed"))
            elif k >= MAX_CALLS_PER_TURN:
                out["invalid_calls"] += 1
                messages.append(chat.tool_message(c["id"], f"error: invalid_tool_call: at most {MAX_CALLS_PER_TURN} tool calls "
                                                           "per turn; this call was not executed"))
            elif has_calc and c["name"] == "calc":
                text, ok = calc_call(c["arguments"])
                out["invalid_calls"] += not ok
                messages.append(chat.tool_message(c["id"], text))
            else:                                                           # the environment answers everything else,
                step = env.step({"id": c["id"], "type": "function",         # malformed and unknown calls included
                                 "function": {"name": c["name"], "arguments": c["arguments"]}})
                messages += step.observation.to_openai(c["id"])
                if event["step"] is None:
                    event["step"] = env.state().get("tool_calls")            # the first environment step of this turn
                if step.observation.is_error and (step.observation.data.get("error") or {}).get("type") == "invalid_tool_call":
                    out["invalid_calls"] += 1
                if step.done:
                    out.update(final=step, stop_reason="truncated" if step.truncated else "submitted")
        if out["final"] is not None:
            break
        notes = []
        if not resp["tool_calls"]:
            idle += 1
            if idle >= IDLE_TURNS:
                out["stop_reason"] = "no_tool_calls"
                break
            notes.append("No tool call was received. Continue with the tools; deliver with the terminal tool.")
        else:
            idle = 0
        if out["turns"] == max_turns - 1:
            notes.append("One turn is left. Deliver now.")
        if notes:                                                           # one user message, never two in a row
            messages.append({"role": "user", "content": " ".join(notes)})
    return out


# ------------------------------------------------------------------ one episode, any agent
def _outcome(env, final, stop_reason: str) -> dict:
    """Result fields of a played episode, read from the last step and the public state of the environment."""
    state = env.state()
    info = final.info if final is not None else {}
    score = info.get("score") or {}
    replay = info.get("replay") or {}
    events = env.scanner_log() if callable(getattr(env, "scanner_log", None)) else []
    return dict(
        stop_reason=stop_reason, reward=float(final.reward) if final is not None else 0.0, passed=bool(score.get("passed", False)),
        error=score.get("error"), gates={k[5:]: bool(v) for k, v in score.items() if k.startswith("gate_")},
        diagnostic=score.get("diagnostic", "no submission"), tolerance=score.get("tolerance"),
        scans=state.get("scans"), scan_s_used=state.get("scan_time_used_s"), scan_s_budget=state.get("scan_time_budget_s"),
        scanner_requests=state.get("scanner_requests"), env_tool_calls=state.get("tool_calls"),
        log=dict(events=len(events), head=events[-1]["chain"] if events else None, replays=replay.get("ok"), mode=replay.get("mode")))


def _save_log(env, directory: str) -> None:
    try:
        events = env.scanner_log()
    except Exception:  # noqa: BLE001  an environment without a scanner, or an episode that never started
        return
    with open(os.path.join(directory, "scanner_log.jsonl"), "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")


def _save_bundle(env, final, agent: dict, events, directory: str) -> bool:
    """Write ``episode_bundle.json`` if ``mrgym.record`` can make one of the episode ``env`` has just played (it needs an
    episode that ended inside the environment). Returns whether it did."""
    try:
        from .. import record
        bundle = record.bundle_from_env(env, final, agent, agent_events=events)
        with open(os.path.join(directory, "episode_bundle.json"), "w", encoding="utf-8") as f:
            json.dump(bundle, f, separators=(",", ":"), allow_nan=False)
            f.write("\n")
        return True
    except Exception:  # noqa: BLE001  recordings are optional; an episode result never depends on them
        return False


def run_episode(job: dict) -> dict:
    """Play one episode and write its files. Never raises for an agent or endpoint failure: the former is a scored
    record, the latter an ``infra_error`` record. Returns the record."""
    import mrgym

    from .. import admission, policies
    opts, directory, t0 = job["opts"], job["dir"], time.time()
    shutil.rmtree(directory, ignore_errors=True)                # an episode is never continued by another conversation
    os.makedirs(directory)
    scripted = job["kind"] == "scripted"
    rec = dict(schema=RECORD_SCHEMA, agent=job["agent"], agent_kind=job["kind"], model=job.get("model"),
               params_sent={}, params_dropped=[], settings=None if scripted else opts["settings"],
               env_id=job["env_id"], level=job["level"], seed=job["seed"], split="train", episode=job["episode"],
               engine=mrgym.ENGINE_ID, package_version=mrgym.__version__,
               status="infra_error", stop_reason="interrupted", infra_error="the runner stopped before this episode ended",
               utc=now_utc(), wall_s=0.0)
    path = os.path.join(directory, "record.json")
    write_json(path, rec, indent=1)                             # every started episode has a record
    env, sink, played, client, final, events = None, {}, None, None, None, None
    try:
        env = mrgym.make(job["env_id"], level=job["level"], seed=job["seed"])
        if scripted:
            final = policies.run(env, job["policy"], recipe=job["recipe"])
            played = dict(_outcome(env, final, "submitted"), recipe_source="admission" if job["recipe"] else "default",
                          policy_error=final.info.get("policy_error"), turns=None, tool_calls=env.state().get("tool_calls"),
                          invalid_calls=0, tokens_in=None, tokens_out=None, tokens_estimated=False)
        else:
            client = chat.ChatClient(opts["base_url"], job["model"], opts["api_key"],
                                     dict(temperature=opts["settings"]["temperature"], max_tokens=opts["settings"]["max_tokens"]),
                                     opts["timeout"], opts["max_retries"], opts["backoff_s"])
            p = play_model(env, client, opts["settings"]["max_turns"], opts["settings"]["calc"], sink)
            final, events = p["final"], p["events"]
            played = dict(_outcome(env, p["final"], p["stop_reason"]),
                          **{k: p[k] for k in ("turns", "tool_calls", "invalid_calls", "by_tool", "tokens_in", "tokens_out", "tokens_estimated")})
        if env.spec.family:                                     # the public task id, and what the stored gate says of it
            task_id = getattr(env, "episode_config", lambda: {})().get("task_id")
            stored = admission.stored_instance(job["env_id"], task_id) if task_id else None
            played.update(task_id=task_id, admitted=bool(stored["admitted"]) if stored else None)
        rec.update(status="completed", infra_error=None, **played)
    except chat.GatewayError as e:
        rec.update(stop_reason="endpoint_error", infra_error=str(e)[:500])
    except Exception as e:  # noqa: BLE001  a fault of the runner must not take the other episodes down
        rec.update(stop_reason="runner_error", infra_error=f"{type(e).__name__}: {e}"[:500])
    if client is not None:
        rec.update(params_sent=dict(client.params), params_dropped=list(client.dropped))
    if rec["infra_error"] is None:
        del rec["infra_error"]
    if env is not None:
        try:
            if not opts["records_only"]:
                _save_log(env, directory)
                if sink and world_seed(env) not in json.dumps(sink):        # a conversation with the seed is not kept
                    write_json(os.path.join(directory, "messages.json"), sink, indent=1)
            if opts["record_bundles"] and rec["status"] == "completed":
                agent = dict(kind=job["kind"], name=job["agent"], language_model=job["kind"] == "model")
                rec["bundle"] = _save_bundle(env, final, agent, events, directory)
        except Exception as e:  # noqa: BLE001
            rec.update(status="infra_error", stop_reason="runner_error", infra_error=f"{type(e).__name__}: {e}"[:500])
        finally:
            env.close()
    rec.update(utc=now_utc(), wall_s=round(time.time() - t0, 2))
    rec = {k: rec[k] for k in RECORD_FIELDS if k in rec} | {k: v for k, v in rec.items() if k not in RECORD_FIELDS}
    write_json(path, rec, indent=1)
    return rec


def _line(rec: dict) -> str:
    head = f"[{rec['utc']}] {rec['agent']:<24} {rec['env_id']} L{rec['level']} s{rec['seed']} e{rec['episode']}  "
    if rec["status"] != "completed":
        return head + f"INFRA {rec['stop_reason']}: {rec.get('infra_error', '')[:160]}"
    err = "-" if rec["error"] is None else f"{100 * rec['error']:.1f}%"
    tail = "" if rec["turns"] is None else f" turns {rec['turns']} tokens {rec['tokens_in']}/{rec['tokens_out']}"
    return head + (f"{rec['stop_reason']:<13} reward {rec['reward']:.3f} pass {int(rec['passed'])} error {err:>7} "
                   f"scans {rec['scans']}{tail}")


# ------------------------------------------------------------------ a run: instances x agents
def policy_recipe(spec, level: int) -> dict | None:
    """The constant recipe of the ``fixed_recipe`` family of control policies for one environment and level: the one
    the stored admission run tuned on other worlds, or None (the package default) when no result is stored."""
    from .. import admission
    stored = admission.load(spec.id) or {}
    return (stored.get("levels", {}).get(str(level)) or {}).get("fixed_recipe")


def _policies_for(spec, wanted: list[str]) -> tuple[list[str], dict]:
    """(policies of ``spec`` to play, {name: why it is skipped}). ``all`` = every control policy that plays on the
    budget the environment grants; one that needs a multiple of it cannot be played through the tool surface."""
    from .. import policies
    module = policies.controls_of(spec)
    table = module.policies() if module else {}
    scaled = {n: f"needs {v[1]:g} times the budget of the environment" for n, v in table.items() if v[1] != 1.0}
    if "all" in wanted:
        return [n for n in table if n not in scaled], scaled
    return [n for n in wanted if n in table and n not in scaled], {n: scaled[n] for n in wanted if n in scaled}


def run(out: str, envs, *, levels=None, seeds=range(10), episodes: int = 1, models=(), policies=(),
        base_url: str | None = None, api_key: str | None = None, max_turns: int = DEFAULT_MAX_TURNS,
        temperature: float | None = None, max_tokens: int | None = None, workers: int = 1, calc: bool = True,
        record_bundles: bool = False, records_only: bool = False, timeout: float = 600.0, max_retries: int = 5,
        backoff_s: float = 2.0, command: str | None = None, quiet: bool = False) -> dict:
    """Play every agent on every instance and write the episode folders under ``out``. Resumable.

    ``envs``      environment ids.            ``levels``   levels to play (None: all levels of each environment;
    ``seeds``     integers of the public                   a level an environment does not have is skipped for it).
                  train split.                ``episodes`` episodes per instance for models (scripted: always 1).
    ``models``    model ids of the endpoint   ``policies`` control-policy names, or ``["all"]``.
                  at ``base_url``.

    Returns ``dict(started, already_finished, infra_errors, records)``; ``records`` are those of the episodes this
    call played. Raises ``mrgym.Error`` for a request that cannot be run as asked.
    """
    import mrgym
    envs, models, policy_names = _split(envs), _split(models), _split(policies)
    seeds = sorted(set(int(s) for s in seeds))
    workers = max(1, min(int(workers), MAX_WORKERS))
    if not envs or not seeds:
        raise mrgym.Error("give at least one environment and one seed")
    if not models and not policy_names:
        raise mrgym.Error("give at least one model (--model) or one scripted policy (--policy)")
    if models and not base_url:
        raise mrgym.Error("a model needs the base URL of an OpenAI-compatible endpoint (--base-url)")
    if max_turns < 2 or episodes < 1:
        raise mrgym.Error("max_turns must be at least 2 and episodes at least 1")
    folders = {}
    for label, folder in [(m, slug(m)) for m in models]:
        if folders.setdefault(folder, label) != label:
            raise mrgym.Error(f"models {folders[folder]!r} and {label!r} would share the folder {folder!r}")
        if folder.startswith("policy-"):
            raise mrgym.Error(f"model id {label!r} collides with the folders of scripted policies")
    settings = model_settings(max_turns, calc, temperature, max_tokens)
    opts = dict(base_url=base_url, api_key=api_key, settings=settings, timeout=timeout, max_retries=max_retries,
                backoff_s=backoff_s, record_bundles=record_bundles, records_only=records_only)
    plan, skipped, unknown = {}, {}, set(policy_names) - {"all"}
    for env_id in envs:
        spec = mrgym.spec(env_id)
        lv = [x for x in (levels or spec.levels) if x in spec.levels]
        names, skip = _policies_for(spec, policy_names)
        unknown -= set(names) | set(skip)
        plan[env_id] = dict(levels=lv, policies=names, recipes={str(x): policy_recipe(spec, x) for x in lv} if names else {})
        if skip:
            skipped[env_id] = skip
    if unknown:
        raise mrgym.Error(f"no environment of this run has a control policy named {sorted(unknown)}")
    if not any(p["levels"] for p in plan.values()):
        raise mrgym.Error(f"none of the environments has a level in {list(levels)}")

    jobs, finished = [], 0
    for e in range(episodes):                                   # interleaved: a partial run is balanced over agents
        for env_id in envs:
            for lv in plan[env_id]["levels"]:
                for seed in seeds:
                    agents = [dict(agent=n, kind="scripted", policy=n, folder="policy-" + slug(n), recipe=plan[env_id]["recipes"][str(lv)])
                              for n in plan[env_id]["policies"] if e == 0]
                    agents += [dict(agent=m, kind=agent_kind(m), model=m, folder=slug(m)) for m in models]
                    for ag in agents:
                        d = os.path.join(out, "episodes", ag.pop("folder"), env_slug(env_id), f"L{lv}", f"s{seed}", f"e{e}")
                        old = _read(os.path.join(d, "record.json"))
                        if old and old.get("agent") != ag["agent"]:
                            raise mrgym.Error(f"{d} holds episodes of another agent ({old.get('agent')!r})")
                        if old and old.get("status") == "completed":
                            if ag["kind"] != "scripted" and old.get("settings") != settings:
                                raise mrgym.Error(f"{out} holds episodes of {ag['agent']!r} played with other settings "
                                                  f"({old.get('settings')}); they would not be comparable. Use a new --out.")
                            finished += 1
                            continue
                        jobs.append(dict(ag, env_id=env_id, level=lv, seed=seed, episode=e, dir=d, opts=opts))
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    say(f"{len(jobs)} episode(s) to run, {finished} already finished; {workers} worker(s); out = {out}")
    for env_id, skip in skipped.items():
        for name, why in skip.items():
            say(f"not played: {name} on {env_id} ({why})")
    if policy_names and episodes > 1:
        say("scripted policies are deterministic: one episode per instance")
    if not jobs:
        return dict(started=0, already_finished=finished, infra_errors=0, records=[])
    os.makedirs(out, exist_ok=True)
    _note_invocation(out, dict(utc=now_utc(), command=command, package_version=mrgym.__version__, engine=mrgym.ENGINE_ID,
                               python=sys.version.split()[0], numpy=np.__version__, split="train", seeds=seeds_text(seeds),
                               episodes_per_instance=episodes, environments=plan, policies_not_played=skipped, models=models,
                               base_url=base_url if models else None, settings=settings if models else None,
                               records_only=records_only, workers=workers, episodes_started=len(jobs),
                               episodes_already_finished=finished))
    t0, records = time.time(), []

    def done(rec):
        records.append(rec)
        with _PRINT_LOCK:
            say(_line(rec))

    scripted = [j for j in jobs if j["kind"] == "scripted"]
    others = [j for j in jobs if j["kind"] != "scripted"]
    # Scripted episodes are computation: processes. Model episodes wait on the endpoint: threads.
    if workers > 1 and len(scripted) > 1 and threading.active_count() == 1 and "fork" in multiprocessing.get_all_start_methods():
        with multiprocessing.get_context("fork").Pool(min(workers, len(scripted))) as pool:
            for rec in pool.imap_unordered(run_episode, scripted, chunksize=1):
                done(rec)
    else:
        others = scripted + others
    if workers > 1 and len(others) > 1:
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            for fut in concurrent.futures.as_completed([pool.submit(run_episode, j) for j in others]):
                done(fut.result())
    else:
        for j in others:
            done(run_episode(j))
    bad = sum(r["status"] != "completed" for r in records)
    say(f"finished {len(records) - bad} episode(s), {bad} infra error(s), {time.time() - t0:.0f} s")
    return dict(started=len(jobs), already_finished=finished, infra_errors=bad, records=records)


def _read(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _note_invocation(out: str, entry: dict) -> None:
    path = os.path.join(out, "run_config.json")
    cfg = _read(path) or dict(schema=RUN_SCHEMA, created_utc=entry["utc"], invocations=[])
    cfg["invocations"].append(entry)
    write_json(path, cfg, indent=1)


# ------------------------------------------------------------------ command line
def dry_run(envs, levels=None, seeds=(0,), model: str | None = None, max_turns: int = DEFAULT_MAX_TURNS, calc: bool = True) -> list[dict]:
    """What a model would receive on the first instance of each environment: ``[{env_id, level, seed, request}]``,
    ``request`` being the body of the first chat-completions call. Nothing is called and nothing is written."""
    import mrgym
    out = []
    for env_id in _split(envs):
        spec = mrgym.spec(env_id)
        lv = [x for x in (levels or spec.levels) if x in spec.levels]
        if not lv:
            continue
        env = mrgym.make(env_id, level=lv[0], seed=int(seeds[0]))
        obs, _ = env.reset()
        tools, _ = agent_tools(env, calc)
        messages = first_messages(env, obs.text, max_turns, calc)
        if world_seed(env) in json.dumps([messages, tools]):
            raise AssertionError("the world seed must never reach the agent")
        out.append(dict(env_id=env_id, level=lv[0], seed=int(seeds[0]), request=dict(model=model or "<model>", messages=messages, tools=tools)))
        env.close()
    return out


def _command(a) -> str:
    """The command as typed, rebuilt from the parsed options (defaults left out)."""
    parts = ["mrgym", "run-agent", "--env", ",".join(_split(a.env))]
    if a.levels != "all":
        parts += ["--levels", a.levels]
    parts += ["--seeds", a.seeds]
    if a.episodes != 1:
        parts += ["--episodes", str(a.episodes)]
    if a.policy:
        parts += ["--policy", ",".join(_split(a.policy))]
    for m in _split(a.model):
        parts += ["--model", m]
    if a.model:
        parts += ["--base-url", a.base_url or "", "--api-key-env", a.api_key_env, "--max-turns", str(a.max_turns)]
        for flag, v in (("--temperature", a.temperature), ("--max-tokens", a.max_tokens)):
            if v is not None:
                parts += [flag, str(v)]
        if a.no_calc:
            parts.append("--no-calc")
    for flag, on in (("--record-bundles", a.record_bundles), ("--records-only", a.records_only)):
        if on:
            parts.append(flag)
    parts += ["--workers", str(a.workers), "--out", a.out]
    return shlex.join(parts)


def cmd_run_agent(a) -> int:
    import mrgym
    try:
        envs = _split(a.env)
        if envs == ["all"]:
            envs = [i for i in mrgym.list_envs() if mrgym.spec(i).family]
        levels = None if a.levels == "all" else [int(x) for x in _split(a.levels)]
        seeds = parse_seeds(a.seeds)
        if a.dry_run:
            model = (_split(a.model) or [None])[0]
            for item in dry_run(envs, levels, seeds, model, a.max_turns, not a.no_calc):
                req = item["request"]
                print(f"# {item['env_id']} level {item['level']} seed {item['seed']}: the first request, as it would be sent")
                print(json.dumps(req, indent=1))
                sizes = [len(json.dumps(req["messages"])), len(json.dumps(req["tools"]))]
                print(f"# {sum(sizes)} characters (messages {sizes[0]}, tool schemas {sizes[1]}); nothing was called\n")
            return 0
        result = run(a.out, envs, levels=levels, seeds=seeds, episodes=a.episodes, models=_split(a.model),
                     policies=_split(a.policy), base_url=a.base_url, api_key=os.environ.get(a.api_key_env),
                     max_turns=a.max_turns, temperature=a.temperature, max_tokens=a.max_tokens, workers=a.workers,
                     calc=not a.no_calc, record_bundles=a.record_bundles, records_only=a.records_only,
                     timeout=a.timeout, max_retries=a.max_retries, backoff_s=a.backoff_s, command=_command(a))
    except (mrgym.Error, ValueError) as e:
        raise SystemExit(f"mrgym run-agent: {e}") from None
    print(f"table: mrgym results --dir {a.out}")
    return 1 if result["infra_errors"] else 0


def add_parser(sub) -> None:
    """Register ``mrgym run-agent``."""
    import argparse
    p = sub.add_parser("run-agent", help="run a language model or scripted policies on environments; one record per episode",
                       description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--env", action="append", required=True, metavar="ID",
                   help="environment id; repeat the option or give a comma list; 'all' = every scored environment")
    p.add_argument("--levels", default="all", help="comma list, e.g. 1,2 (default: all levels of each environment)")
    p.add_argument("--seeds", default="0-9", help="integer seeds of the public train split, e.g. 0-9 or 0,3,7 (default 0-9)")
    p.add_argument("--episodes", type=int, default=1,
                   help="episodes per instance for a model; repeats differ by the model's sampling only (default 1).\n"
                        "A scripted policy is deterministic and always plays one.")
    p.add_argument("--model", action="append", metavar="NAME", help="model id of the endpoint; repeatable")
    p.add_argument("--base-url", default=None, metavar="URL", help="OpenAI-compatible endpoint, e.g. https://host/v1")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY", metavar="VAR", help="environment variable that holds the key (default OPENAI_API_KEY)")
    p.add_argument("--policy", "--policies", dest="policy", action="append", metavar="NAMES",
                   help="scripted control policies, comma list, or 'all'; see mrgym card <id>")
    p.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS, help=f"assistant turns per episode; told to the model (default {DEFAULT_MAX_TURNS})")
    p.add_argument("--temperature", type=float, default=None, help="not sent unless given")
    p.add_argument("--max-tokens", type=int, default=None, help="not sent unless given")
    p.add_argument("--workers", type=int, default=1, help=f"episodes in parallel, at most {MAX_WORKERS} (default 1)")
    p.add_argument("--out", default="runs/agent", metavar="DIR", help="run folder; repeat the command to resume (default runs/agent)")
    p.add_argument("--dry-run", action="store_true", help="print exactly what the model would receive (messages, tool schemas); call nothing")
    p.add_argument("--no-calc", action="store_true", help="do not add the session's arithmetic tool")
    p.add_argument("--record-bundles", action="store_true", help="also write episode_bundle.json per episode (for mrgym view)")
    p.add_argument("--records-only", action="store_true", help="write record.json only: no scanner log, no messages")
    p.add_argument("--timeout", type=float, default=600.0, help="seconds per request (default 600)")
    p.add_argument("--max-retries", type=int, default=5, help="retries per request on 429 / 5xx / timeouts (default 5)")
    p.add_argument("--backoff-s", type=float, default=2.0, help="first back-off in seconds; doubles per retry (default 2)")
    p.set_defaults(fn=cmd_run_agent)
