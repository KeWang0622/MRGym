"""Run agents on MR Gym environments and save one small record per episode.

    mrgym run-agent --env Scanner/NoWrap-v1 --seeds 0-9 --policy all --out runs/first
    mrgym run-agent --env Scanner/NoWrap-v1 --seeds 0-9 --model MODEL --base-url https://HOST/v1 --out runs/first
    mrgym results --dir runs/first

Two kinds of agent go through the same environment, the same scoring and the same record format, so their rows land
in one table:

  scripted   a control policy of the environment (``--policy reference,prior`` or ``--policy all``), played through
             the tool surface by ``mrgym.policies.run``. No model is called. Policies that apply one constant recipe
             use the recipe the stored gate run tuned for that environment and level (``recipe_source``).
  model      a language model behind an OpenAI-compatible chat-completions endpoint (``--model``, ``--base-url``).
             The environment runs in this process: the model has no code execution, it only writes tool calls, and
             the hidden world never enters a message. This is the intended trust model for tool-calling models.

What a model receives (``--dry-run`` prints it and calls nothing): a short session note, the task statement returned
by ``env.reset()`` as the first user message, and the tools of ``env.tools("openai")`` (``--no-calc`` makes the
environment without its arithmetic tool).

Limits. The runner has no cap of its own. An episode ends where the environment ends it (the terminal tool, the
tool-call limit, the rejected-call limit) or after three replies in a row without a tool call. ``--max-turns N`` makes
the environment with ``max_steps=N``: the number the model reads in the task text and in every status line is the
number that applies, and the run prints it. ``--max-tokens`` is the one limit of the harness; it is passed to
``env.reset(options={"harness": {"max_reply_tokens": N}})`` so that the task text names it too, and a reply that was
cut at it is told to the model and counted (``interface.cut_replies``).

Instances. An instance is (environment, level, integer seed of the public train split). Every agent plays the same
instances. ``--episodes`` repeats an instance for a model; repeats differ by the model's sampling only (the
environment is deterministic). A scripted policy is deterministic too, so it always plays one episode per instance.

Failures. An agent failure is a scored episode, never a crash: invalid JSON arguments, an unknown tool, more than
12 calls in one turn (answered with an error), three turns in a row without a tool call (``no_tool_calls``), the
environment's tool-call limit (``truncated``), its rejected-call limit (``rejected_call_limit``). All of them end with
reward 0 unless the agent submitted. An endpoint failure is not an agent result: after the retries of the chat client the episode is
written as ``status = infra_error``, left out of every rate and run again when the same command is repeated.
Every started episode has a record: it is written as ``infra_error / interrupted`` at the start and replaced at the
end, so a killed run leaves no silent gaps.

Files, under ``<out>``:

    run_config.json                                         what was asked for, per invocation
    episodes/<agent>/<env>/L<level>/s<seed>/e<k>/
        record.json          the result of the episode (see ``RECORD_FIELDS`` and ``mrgym.record.finish_episode``)
        messages.json        the whole conversation (models only): tools, every message as sent, every model turn with
                             its reasoning text when the endpoint returns it, usage and latency per request
        scanner_log.jsonl    the scanner's hash-chained log
        episode_bundle.json  a replayable recording, with ``--record-bundles`` and when ``mrgym.record`` offers it

``--records-only`` writes ``record.json`` alone for a scripted policy; a model episode always keeps its conversation
and its log. Resume: repeat the command; finished episodes are skipped.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
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

RECORD_SCHEMA = "mrgym-episode/2"                # schema 2: mrgym.record.finish_episode adds transcript, harness, deliverable
RUN_SCHEMA = "mrgym-run/1"
MAX_CALLS_PER_TURN = 12
IDLE_TURNS = 3                              # turns in a row without a tool call that end the episode
CHARS_PER_TOKEN = 3.5                       # used only when the endpoint reports no usage; such counts are flagged
MAX_WORKERS = 8
# How an episode ended. The first is a delivery; "deliverable_refused" is a scripted policy whose only delivery was refused.
STOP_REASONS = ("submitted", "no_tool_calls", "truncated", "rejected_call_limit", "deliverable_refused")
# record.json, in this order. Who played; on what; how it ended; the score; the effort; provenance.
RECORD_FIELDS = (
    "schema", "agent", "agent_kind", "model", "params_sent", "params_dropped", "settings",
    "env_id", "level", "seed", "split", "episode", "task_id", "engine", "package_version",
    "status", "stop_reason", "infra_error", "recipe_source", "policy_error",
    "reward", "passed", "error", "gates", "diagnostic", "tolerance",
    "scans", "scan_s_used", "scan_s_budget", "scanner_requests", "turns", "tool_calls", "env_tool_calls", "invalid_calls", "by_tool",
    "interface",
    "tokens_in", "tokens_out", "tokens_estimated", "log", "bundle", "wall_s", "utc")
_PRINT_LOCK = threading.Lock()

SESSION_NOTE = """You operate a virtual MR scanner through function tools. The task and every limit of the episode are in the \
first user message. Everything about the object and about the state of the system has to be measured: use numbers from \
tool results only. The datasheet of the instrument is behind `scanner_info`.

Rules of this session
- The tools are {scanner_tools}. Only `scan` spends scan time.
- A call that cannot be executed is answered with an error and a `fix` line. It costs no scan time: read the fix and \
send the corrected call.
- Deliver by calling `{terminal}` once. The episode ends there and nothing about correctness is returned.
- You may issue up to {max_calls} tool calls in one reply. Every reply must contain at least one tool call until you \
call `{terminal}`. Call `status` if you are unsure what is left.
"""
CUT_NOTE = ("Your last reply was cut at the reply limit{limit} before it ended, so a tool call in it may be missing or "
            "incomplete. Send the call again, with less text around it.")


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
    """Write ``obj`` to ``path`` through a temporary file of this process and an atomic rename. The temporary name
    holds the process id and a random suffix: several runner processes may write into one run folder (two used to
    share ``<path>.tmp``, and the second rename failed with FileNotFoundError)."""
    tmp = f"{path}.{os.getpid()}.{os.urandom(4).hex()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, **kw)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):                                          # only when the rename did not happen
            os.unlink(tmp)


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
    """(tools in the OpenAI wire format, name of the terminal tool). Call after ``env.reset()``. The tools are the
    environment's and nothing else; ``calc`` is one of them unless the environment was made with ``calc=False``."""
    neutral = env.tools("neutral")
    return env.tools("openai"), next(t["name"] for t in neutral if t["terminal"])


def first_messages(env, task_text: str, *_unused) -> list[dict]:
    """The session note (system) and the task statement of ``env.reset()`` (user). The note states no limit: the
    limits are in the task text, generated by the environment."""
    neutral = env.tools("neutral")
    terminal = next(t["name"] for t in neutral if t["terminal"])
    note = SESSION_NOTE.format(scanner_tools=", ".join(f"`{t['name']}`" for t in neutral), terminal=terminal, max_calls=MAX_CALLS_PER_TURN)
    return [{"role": "system", "content": note}, {"role": "user", "content": task_text}]


def model_settings(max_turns: int | None, calc_tool: bool, temperature, max_tokens) -> dict:
    """Everything of the harness that can move a model's result. Episodes of one agent in one folder must agree.
    ``max_turns``: the tool-call limit the environment is made with (None: the environment's own)."""
    return dict(max_turns=None if max_turns is None else int(max_turns), calc=bool(calc_tool), temperature=temperature, max_tokens=max_tokens,
                max_calls_per_turn=MAX_CALLS_PER_TURN, idle_turns=IDLE_TURNS,
                session_note_sha256=hashlib.sha256((SESSION_NOTE + CUT_NOTE).encode()).hexdigest()[:12])


def make_env(env_id: str, level: int, seed, settings: dict | None = None, **env_kwargs):
    """The environment of an episode. For a model the harness settings become the environment's own limits, so
    that nothing ends an episode that the task text does not state."""
    import mrgym
    kw = dict(env_kwargs)
    if settings:
        if settings.get("max_turns") is not None:
            kw["max_steps"] = int(settings["max_turns"])
        if not settings.get("calc", True):
            kw["calc"] = False
    return mrgym.make(env_id, level=level, seed=seed, **kw)


def reset_options(settings: dict | None) -> dict | None:
    """``env.reset(options=...)`` for a model episode: the harness states its reply limit, if it has one."""
    if settings and settings.get("max_tokens"):
        return {"harness": {"max_reply_tokens": int(settings["max_tokens"])}}
    return None


# ------------------------------------------------------------------ one model episode
def play_model(env, client: chat.ChatClient, sink: dict | None = None, options: dict | None = None) -> dict:
    """Drive one episode of ``env`` with the model behind ``client``. Resets the environment (``options`` go to it).

    The loop has two ways to end and no other: the environment ends the episode (the terminal tool or one of its
    limits), or the model sends ``IDLE_TURNS`` replies in a row without a tool call.

    Returns ``dict(stop_reason, final, turns, tool_calls, invalid_calls, by_tool, interface, tokens_in, tokens_out,
    tokens_estimated, tools, messages, events)``. ``final`` is the last ``StepResult`` when the episode ended inside
    the environment, else None. ``invalid_calls`` are the calls that were answered without being executed.
    ``interface`` holds the environment's own counts (calls, shape, vocabulary, task_refusals, coerced) plus what only
    the harness sees: ``lost_turns`` (replies in which every call was refused for its shape or vocabulary),
    ``idle_turns`` (replies without a call) and ``cut_replies`` (replies cut at the reply limit). ``events`` are the
    assistant turns with the number of the environment step each led to (the form ``mrgym.record`` takes as
    ``agent_events``).

    ``sink``, if given, receives ``tools`` and ``messages`` before the first request, so that a caller can save the
    conversation of an episode that an endpoint failure cut short. Raises ``chat.GatewayError`` on an endpoint
    failure and ``AssertionError`` if the world seed is about to be sent.
    """
    from .. import interface as I
    obs, info = env.reset(options=options)
    seed_hex = world_seed(env)
    tools, _ = agent_tools(env)
    messages = first_messages(env, obs.text)
    if sink is not None:
        sink.update(tools=tools, messages=messages)
    tools_json = json.dumps(tools)
    known = {t["function"]["name"] for t in tools}
    reply_limit = (info.get("limits", {}).get("harness") or {}).get("max_reply_tokens")
    out = dict(stop_reason="no_tool_calls", final=None, turns=0, tool_calls=0, invalid_calls=0, by_tool={}, tokens_in=0,
               tokens_out=0, tokens_estimated=False, tools=tools, messages=messages, events=[])
    idle = lost_turns = idle_turns = cut_replies = 0
    while out["final"] is None:                                             # ends with the environment, or on idle replies
        sent = json.dumps(messages)
        if seed_hex in sent or seed_hex in tools_json:                      # checked on every request
            raise AssertionError("the world seed must never reach the agent")
        resp = client.complete(messages, tools)                             # GatewayError: the caller records it
        out["turns"] += 1
        if callable(getattr(env, "note_turn", None)) and env.note_turn().get("truncated"):
            out["stop_reason"] = "truncated"                                # the environment's own cap on turns, if one was set
            break
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
        lost = []
        for k, c in enumerate(resp["tool_calls"]):
            out["tool_calls"] += 1
            key = c["name"] if c["name"] in known else "(unknown)"
            out["by_tool"][key] = out["by_tool"].get(key, 0) + 1
            if out["final"] is not None:
                messages.append(chat.tool_message(c["id"], "error episode_over · the episode has ended; this call was not executed"))
            elif k >= MAX_CALLS_PER_TURN:
                out["invalid_calls"] += 1
                messages.append(chat.tool_message(c["id"], f"error malformed_call · at most {MAX_CALLS_PER_TURN} tool calls per reply; "
                                                           "this call was not executed and nothing was charged"))
            else:                                                           # the environment answers everything else,
                step = env.step({"id": c["id"], "type": "function",         # malformed and unknown calls included
                                 "function": {"name": c["name"], "arguments": c["arguments"]}})
                messages += step.observation.to_openai(c["id"])
                if event["step"] is None:
                    event["step"] = env.state().get("tool_calls")            # the first environment step of this turn
                refused = step.observation.is_error and not I.counts_as_tool_call(step.observation.data)
                out["invalid_calls"] += refused
                lost.append(step.observation.is_error and I.error_class(step.observation.data) in I.INTERFACE_CLASSES)
                if step.done:
                    reason = step.info.get("reason")
                    out.update(final=step, stop_reason=("rejected_call_limit" if reason == "rejected_call_limit" else "truncated")
                               if step.truncated else "submitted")
        lost_turns += bool(lost) and all(lost)
        if out["final"] is not None:
            break
        notes = []
        if "length" in (resp.get("finish") or []):
            cut_replies += 1
            notes.append(CUT_NOTE.format(limit=f" ({reply_limit} tokens)" if reply_limit else ""))
        if not resp["tool_calls"]:
            idle += 1
            idle_turns += 1
            if idle >= IDLE_TURNS:
                break
            if not notes:
                notes.append("No tool call was received. Continue with the tools; deliver with the terminal tool.")
        else:
            idle = 0
        if notes:                                                           # one user message, never two in a row
            messages.append({"role": "user", "content": " ".join(notes)})
    out["interface"] = dict(env.interface_counts(), lost_turns=lost_turns, idle_turns=idle_turns, cut_replies=cut_replies)
    return out


# ------------------------------------------------------------------ one episode, any agent
def _outcome(env, final, stop_reason: str) -> dict:
    """Result fields of a played episode, read from the last step and the public state of the environment."""
    state = env.state()
    info = final.info if final is not None else {}
    score = info.get("score") or {}
    replay = info.get("replay") or {}
    events = env.scanner_log() if callable(getattr(env, "scanner_log", None)) else []
    training = {}
    if callable(getattr(env, "end_without_submission", None)):              # environments with training hooks: one outcome per episode
        t = env.training_info() or env.end_without_submission("stopped")
        training = dict(outcome=t["outcome"], progress=t["progress"], ladder=t["ladder"], counts=t["counts"], group=t["group"]["id"],
                        checks=score.get("checks"))
    return dict(
        **training,
        stop_reason=stop_reason, reward=float(final.reward) if final is not None else 0.0, passed=bool(score.get("passed", False)),
        error=score.get("error"), gates={**{k: bool(v) for k, v in (score.get("checks") or {}).items()},
                                         **{k[5:]: bool(v) for k, v in score.items() if k.startswith("gate_")}},
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

    from .. import policies, record
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
        env = make_env(job["env_id"], job["level"], job["seed"], None if scripted else opts["settings"], **(opts.get("env_kwargs") or {}))
        if scripted:
            final = policies.run(env, job["policy"], recipe=job["recipe"])
            played = dict(_outcome(env, final, "submitted" if final.info.get("submitted", True) else "deliverable_refused"),
                          recipe_source="gate" if job["recipe"] else "default",
                          policy_error=final.info.get("policy_error"), turns=None, tool_calls=env.state().get("tool_calls"),
                          invalid_calls=env.state().get("rejected_calls", 0), interface=env.interface_counts(),
                          tokens_in=None, tokens_out=None, tokens_estimated=False)
        else:
            client = chat.ChatClient(opts["base_url"], job["model"], opts["api_key"],
                                     dict(temperature=opts["settings"]["temperature"], max_tokens=opts["settings"]["max_tokens"]),
                                     opts["timeout"], opts["max_retries"], opts["backoff_s"])
            client = record.Recorder(client)                    # keeps reasoning text, usage and latency of every request
            p = play_model(env, client, sink, reset_options(opts["settings"]))
            final, events = p["final"], p["events"]
            played = dict(_outcome(env, p["final"], p["stop_reason"]),
                          **{k: p[k] for k in ("turns", "tool_calls", "invalid_calls", "by_tool", "interface", "tokens_in", "tokens_out", "tokens_estimated")})
        if env.spec.family:                                     # the trainer-side label of the instance
            played.update(task_id=getattr(env, "episode_config", lambda: {})().get("task_id"))
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
            rec = record.finish_episode(rec, job, env, final, client, sink, directory)       # transcript, harness id, deliverable
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
    """The constant recipe of the ``fixed_recipe`` control policy for one environment and level: the first of the
    recipes the stored gate run tuned on other worlds, or None (the package default) when no result is stored."""
    from .. import gate
    stored = gate.load(spec.id) or {}
    return next(iter((stored.get("levels", {}).get(str(level)) or {}).get("fixed_recipes") or []), None)


def _policies_for(spec, wanted: list[str]) -> tuple[list[str], dict]:
    """(policies of ``spec`` to play, {name: why it is skipped}). ``all`` = every control policy of the family; each
    plays on the budget the environment grants."""
    from .. import policies
    module = policies.controls_of(spec)
    table = module.policies() if module else {}
    if "all" in wanted:
        return list(table), {}
    return [n for n in wanted if n in table], {}


def run(out: str, envs, *, levels=None, seeds=range(10), episodes: int = 1, models=(), policies=(),
        base_url: str | None = None, api_key: str | None = None, max_turns: int | None = None,
        temperature: float | None = None, max_tokens: int | None = None, workers: int = 1, calc: bool = True,
        record_bundles: bool = False, records_only: bool = False, timeout: float = 600.0, max_retries: int = 5,
        backoff_s: float = 2.0, command: str | None = None, quiet: bool = False, env_kwargs: dict | None = None) -> dict:
    """Play every agent on every instance and write the episode folders under ``out``. Resumable.

    ``envs``      environment ids.            ``levels``   levels to play (None: all levels of each environment;
    ``seeds``     integers of the public                   a level an environment does not have is skipped for it).
                  train split.                ``episodes`` episodes per instance for models (scripted: always 1).
    ``models``    model ids of the endpoint   ``policies`` control-policy names, or ``["all"]``.
                  at ``base_url``.            ``max_turns`` tool calls per model episode: the environment is made
                                                           with ``max_steps=max_turns`` (None: its own limit). The
                                                           runner adds no cap. Scripted policies play on the
                                                           environment's own limit.

    ``env_kwargs`` keywords passed to ``mrgym.make`` for every episode (``mrgym calibrate`` passes ``max_turns`` so
                  that the agent sees the turn cap in the status line).

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
    if (max_turns is not None and max_turns < 2) or episodes < 1:
        raise mrgym.Error("max_turns must be at least 2 and episodes at least 1")
    folders = {}
    for label, folder in [(m, slug(m)) for m in models]:
        if folders.setdefault(folder, label) != label:
            raise mrgym.Error(f"models {folders[folder]!r} and {label!r} would share the folder {folder!r}")
        if folder.startswith("policy-"):
            raise mrgym.Error(f"model id {label!r} collides with the folders of scripted policies")
    settings = model_settings(max_turns, calc, temperature, max_tokens)
    opts = dict(base_url=base_url, api_key=api_key, settings=settings, timeout=timeout, max_retries=max_retries,
                backoff_s=backoff_s, record_bundles=record_bundles, records_only=records_only, env_kwargs=dict(env_kwargs or {}))
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
    if models:                                                  # the cap that applies is the environment's, and it is said here
        for env_id in envs:
            say(f"tool-call limit of a model episode on {env_id}: {mrgym.spec(env_id).max_steps if max_turns is None else max_turns} "
                f"(the environment's limit, stated in its task text; the runner adds none)")
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
                               records_only=records_only, env_kwargs=dict(env_kwargs or {}), workers=workers, episodes_started=len(jobs),
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
    """Append ``entry`` to ``<out>/run_config.json``. Read, append and write happen under a lock file, so that runner
    processes that share a run folder (seeds split over processes) each leave their invocation."""
    path = os.path.join(out, "run_config.json")
    lock = open(path + ".lock", "a", encoding="utf-8")                  # noqa: SIM115  held until the write is done
    try:
        try:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        except (ImportError, OSError):                                   # no advisory locks here: the atomic write still holds
            pass
        cfg = _read(path) or dict(schema=RUN_SCHEMA, created_utc=entry["utc"], invocations=[])
        cfg["invocations"].append(entry)
        write_json(path, cfg, indent=1)
    finally:
        lock.close()


# ------------------------------------------------------------------ command line
def dry_run(envs, levels=None, seeds=(0,), model: str | None = None, max_turns: int | None = None, calc: bool = True,
            max_tokens: int | None = None) -> list[dict]:
    """What a model would receive on the first instance of each environment: ``[{env_id, level, seed, limits, request}]``,
    ``request`` being the body of the first chat-completions call and ``limits`` the limits of that episode (the
    numbers the task text states). Nothing is called and nothing is written."""
    import mrgym
    settings = model_settings(max_turns, calc, None, max_tokens)
    out = []
    for env_id in _split(envs):
        spec = mrgym.spec(env_id)
        lv = [x for x in (levels or spec.levels) if x in spec.levels]
        if not lv:
            continue
        env = make_env(env_id, lv[0], int(seeds[0]), settings)
        obs, info = env.reset(options=reset_options(settings))
        tools, _ = agent_tools(env)
        messages = first_messages(env, obs.text)
        if world_seed(env) in json.dumps([messages, tools]):
            raise AssertionError("the world seed must never reach the agent")
        out.append(dict(env_id=env_id, level=lv[0], seed=int(seeds[0]), limits=info["limits"],
                        request=dict(model=model or "<model>", messages=messages, tools=tools)))
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
        parts += ["--base-url", a.base_url or "", "--api-key-env", a.api_key_env]
        if a.max_turns is not None:
            parts += ["--max-turns", str(a.max_turns)]
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
            for item in dry_run(envs, levels, seeds, model, a.max_turns, not a.no_calc, a.max_tokens):
                req = item["request"]
                print(f"# {item['env_id']} level {item['level']} seed {item['seed']}: the first request, as it would be sent")
                print(f"# limits of the episode (the environment's; the runner adds none): {json.dumps(item['limits'])}")
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
    p.add_argument("--max-turns", type=int, default=None, metavar="N",
                   help="tool calls per model episode: the environment is made with max_steps=N, so the task text and every\n"
                        "status line state it (default: the environment's own limit). The runner adds no cap of its own.")
    p.add_argument("--temperature", type=float, default=None, help="not sent unless given")
    p.add_argument("--max-tokens", type=int, default=None, help="not sent unless given")
    p.add_argument("--workers", type=int, default=1, help=f"episodes in parallel, at most {MAX_WORKERS} (default 1)")
    p.add_argument("--out", default="runs/agent", metavar="DIR", help="run folder; repeat the command to resume (default runs/agent)")
    p.add_argument("--dry-run", action="store_true", help="print exactly what the model would receive (messages, tool schemas); call nothing")
    p.add_argument("--no-calc", action="store_true", help="make the environment without its arithmetic tool (calc)")
    p.add_argument("--record-bundles", action="store_true", help="also write episode_bundle.json per episode (for mrgym view)")
    p.add_argument("--records-only", action="store_true", help="scripted policies: write record.json only; a model episode always keeps its conversation and log")
    p.add_argument("--timeout", type=float, default=600.0, help="seconds per request (default 600)")
    p.add_argument("--max-retries", type=int, default=5, help="retries per request on 429 / 5xx / timeouts (default 5)")
    p.add_argument("--backoff-s", type=float, default=2.0, help="first back-off in seconds; doubles per retry (default 2)")
    p.set_defaults(fn=cmd_run_agent)
    from . import iface_tax
    iface_tax.add_parser(sub)
