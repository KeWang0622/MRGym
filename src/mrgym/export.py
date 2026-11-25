"""Trajectory export: the saved episodes of a run folder as JSON lines.

    mrgym export --dir results --out episodes.jsonl                  one line per episode with a stored conversation
    mrgym export --dir results --format sft --out sft.jsonl          {"messages", "tools"} of the passing episodes

``--format rl`` (default). One object per episode:

    id, env_id, level, split, seed, episode, task_id, agent, model, harness_id, harness, messages, tools, reward,
    passed, terminated, truncated, stop_reason, deliverable, gates, outcome, tokens {in, out, reasoning, estimated},
    wall_s, engine, package_version, canary

``messages`` is the conversation as stored: the system note, the task, every assistant turn (with ``reasoning`` when
the endpoint returned it), every tool call and every tool result. ``gates`` are the verifier's checks. Every line
carries the canary string, so a corpus filter can drop evaluation content.

``--format sft``. Passing episodes of the public training split only, as ``{"messages", "tools"}`` with standard
chat keys; episodes of any other split are refused, since they are evaluation content.

Scripted policies have no conversation and are skipped (``--include-scripted`` exports them with ``messages: null``).
The world seed is in no stored file, so it is in no export.
"""
from __future__ import annotations

import json
import os
import sys

from . import record as _record
from .agents import summarize
from .canary import CANARY

FORMATS = ("rl", "sft")
STANDARD_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")


def _conversation(directory: str, rec: dict) -> dict | None:
    path = os.path.join(directory, rec["_dir"], _record.TRANSCRIPT_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def episode_line(rec: dict, conversation: dict | None) -> dict:
    """The exported object of one episode (``--format rl``)."""
    stop = rec.get("stop_reason")
    return dict(
        id=rec["_dir"].split("episodes/", 1)[-1], env_id=rec["env_id"], level=rec["level"], split=rec.get("split"), seed=rec["seed"],
        episode=rec["episode"], task_id=rec.get("task_id"), agent=rec["agent"], model=rec.get("model"),
        harness_id=rec.get("harness_id"), harness=rec.get("harness"),
        messages=None if conversation is None else conversation.get("messages"),
        tools=None if conversation is None else conversation.get("tools"),
        reward=rec.get("reward"), passed=bool(rec.get("passed")), terminated=stop == "submitted", truncated=stop != "submitted",
        stop_reason=stop, deliverable=rec.get("deliverable"), gates=rec.get("checks") or rec.get("gates"), outcome=rec.get("outcome"),
        tokens=dict(**{"in": rec.get("tokens_in")}, out=rec.get("tokens_out"), reasoning=rec.get("tokens_reasoning"),
                    estimated=bool(rec.get("tokens_estimated"))),
        wall_s=rec.get("wall_s"), engine=rec.get("engine"), package_version=rec.get("package_version"), canary=CANARY)


def lines(directory: str, fmt: str = "rl", agents=None, envs=None, include_scripted: bool = False):
    """Yield the exported objects of the completed episodes under ``directory``."""
    import mrgym
    if fmt not in FORMATS:
        raise mrgym.Error(f"format is one of {', '.join(FORMATS)}")
    for rec in summarize.load(directory):
        if rec["status"] != "completed" or (agents and rec["agent"] not in agents) or (envs and rec["env_id"] not in envs):
            continue
        conversation = None if rec["agent_kind"] == "scripted" else _conversation(directory, rec)
        if fmt == "sft":
            if conversation is None or not rec.get("passed"):
                continue
            if rec.get("split") != "train":
                raise mrgym.Error(f"{rec['_dir']}: split {rec.get('split')!r} is evaluation content; --format sft exports the train split only")
            yield dict(messages=[{k: m[k] for k in STANDARD_KEYS if k in m} for m in conversation["messages"]], tools=conversation.get("tools"))
        elif conversation is not None or (include_scripted and rec["agent_kind"] == "scripted"):
            yield episode_line(rec, conversation)


def cmd_export(a) -> int:
    import mrgym
    split = lambda values: [v for part in values or [] for v in part.split(",") if v] or None       # noqa: E731
    out = sys.stdout if a.out in (None, "-") else open(a.out, "w", encoding="utf-8")
    n = 0
    try:
        for obj in lines(a.dir, a.format, split(a.agent), split(a.env), a.include_scripted):
            out.write(json.dumps(obj, separators=(",", ":")) + "\n")
            n += 1
    except mrgym.Error as e:
        raise SystemExit(f"mrgym export: {e}") from None
    finally:
        if out is not sys.stdout:
            out.close()
    if out is not sys.stdout:
        print(f"written: {a.out} ({n} episodes, format {a.format})")
    return 0


def add_parser(sub) -> None:
    """Register ``mrgym export``."""
    p = sub.add_parser("export", help="saved episodes as JSON lines: messages, deliverable, harness id, canary")
    p.add_argument("--dir", default="results", help="run folder with episodes/ (default results)")
    p.add_argument("--format", default="rl", choices=FORMATS, help="rl: one object per episode; sft: messages and tools of passing episodes")
    p.add_argument("--out", default=None, metavar="FILE", help="output file (default: standard output)")
    p.add_argument("--agent", action="append", metavar="NAMES", help="only these agents (comma list)")
    p.add_argument("--env", action="append", metavar="IDS", help="only these environments (comma list)")
    p.add_argument("--include-scripted", action="store_true", help="also export scripted policies (no conversation)")
    p.set_defaults(fn=cmd_export)
