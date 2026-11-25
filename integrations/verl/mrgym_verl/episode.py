"""One MR Gym episode, driven turn by turn by whoever owns the model call. No trainer is imported here.

    ep = Episode("Scanner/NoWrap-v1", level=2, seed=10017, split="train")
    messages, tools = ep.start()                          # system note + task statement; env.tools("openai")
    appended, done = ep.assistant_turn(text, calls)       # calls = [{"id", "name", "arguments": "<json string>"}]
    ...
    result = ep.result()                                  # paid reward, pass, outcome, progress, the trainer's signal
    record = ep.write_record(directory, agent="...", model="...", settings={...})

It is ``mrgym.agents.runner.play_model`` cut at the model call: the same session note, the same task statement, the
same tool schemas (``env.tools("openai")``, flat arguments), the same harness replies (the per-reply call cap, the
idle note, the cut note) and the same two ways to end: the environment ends the episode (the terminal tool, its
tool-call limit, its rejected-call limit) or the agent sends ``IDLE_TURNS`` replies without a tool call. There is no
turn cap here. ``tests/test_verl_adapter.py`` plays the same stub behaviours through both and compares the
conversations and the results.

Two things a token-level trainer adds, both said to the environment so that the episode still has exactly one outcome:
``abort("length")`` when a reply was cut before its end-of-turn token or the context is full (the runner continues
after a cut reply; a token stream cannot), and ``abort("stopped")`` for any other trainer-side end.

Reward. ``result()["reward"]`` is the environment's paid reward: 1 for a pass, else 0. ``result()["signal"]`` is what
the trainer optimises: the same number with ``signal="pass"`` (the default), or ``info["score"]["progress"]`` with
``signal="progress"``. Progress is a diagnostic of the environment that it never pays; a run trained on it must say
so (docs/TRAINING.md section 1), and every record carries ``training.signal_mode``.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import mrgym
from mrgym import interface as I
from mrgym import record as mrecord
from mrgym.agents import chat as mchat
from mrgym.agents import runner

from . import SIGNALS

LOOP_NAME = "mrgym_verl.episode.Episode"
ABORT_REASONS = ("length", "stopped")
_HERE = os.path.dirname(os.path.abspath(__file__))


def adapter_sha256() -> str:
    """Hash of the adapter's source files (stored in every record: which adapter played the episode)."""
    h = hashlib.sha256()
    for name in sorted(os.listdir(_HERE)):
        if name.endswith(".py"):
            with open(os.path.join(_HERE, name), "rb") as f:
                h.update(name.encode() + b"\0" + f.read())
    return h.hexdigest()


class Episode:
    def __init__(self, env_id: str, level: int, seed: int, split: str = "train", *, tool_call_limit: int | None = None,
                 calc: bool = True, max_reply_tokens: int | None = None, signal: str = "pass", noise_nonce: int = 0):
        if signal not in SIGNALS:
            raise ValueError(f"signal must be one of {SIGNALS}")
        kw = dict(split=split)
        if tool_call_limit is not None:                                  # else the environment's own limit applies
            kw["max_steps"] = int(tool_call_limit)
        if not calc:
            kw["calc"] = False
        self.env = mrgym.make(env_id, level=int(level), seed=int(seed), **kw)
        self.env_id, self.level, self.seed, self.split = env_id, int(level), int(seed), split
        self.signal_mode, self.noise_nonce = signal, int(noise_nonce)
        self.max_reply_tokens = None if max_reply_tokens is None else int(max_reply_tokens)
        self.settings = runner.model_settings(tool_call_limit, calc, None, self.max_reply_tokens)
        self.final, self.stop_reason, self.aborted = None, "no_tool_calls", None
        self.turns = self.tool_calls = self.invalid_calls = 0
        self.by_tool: dict = {}
        self.idle = self.lost_turns = self.idle_turns = self.cut_replies = 0
        self.tokens_in = self.tokens_out = 0
        self.tools: list = []
        self.messages: list = []
        self.events: list = []
        self.turn_log: list = []                 # one entry per reply of the model: text, calls, finish, usage, seconds
        self.step_infos: list = []               # info["step"] of every environment step (mrgym.learn.hooks.STEP_KEYS)
        self.reset_info: dict = {}
        self.cut_reply: dict | None = None
        self._seed_hex = None
        self._known: set = set()
        self._reply_limit = None
        self._t0 = time.time()

    # ------------------------------------------------------------------ start
    def start(self) -> tuple[list, list]:
        options = dict(runner.reset_options(self.settings) or {})
        if self.noise_nonce:
            options["noise_nonce"] = self.noise_nonce
        obs, info = self.env.reset(options=options or None)
        self.reset_info = info
        self._seed_hex = runner.world_seed(self.env)
        self.tools, _ = runner.agent_tools(self.env)
        self.messages = runner.first_messages(self.env, obs.text)
        self._known = {t["function"]["name"] for t in self.tools}
        self._reply_limit = (info.get("limits", {}).get("harness") or {}).get("max_reply_tokens")
        self._check_seed(self.messages + [{"tools": self.tools}])
        return list(self.messages), list(self.tools)

    @property
    def done(self) -> bool:
        return self.final is not None or self.aborted is not None or self.idle >= runner.IDLE_TURNS

    def _check_seed(self, payload) -> None:
        if self._seed_hex and self._seed_hex in json.dumps(payload):
            raise AssertionError("the world seed must never reach the agent")

    # ------------------------------------------------------------------ one reply of the model
    def assistant_turn(self, text: str, calls: list, *, finish: str | None = None, prompt_tokens: int | None = None,
                       completion_tokens: int | None = None, seconds: float | None = None, raw_text: str | None = None,
                       reasoning: str | None = None) -> tuple[list, bool]:
        """Apply one reply. Returns the messages the harness appends after it (tool results, then at most one user
        note) and whether the episode is over. Never raises for anything the model did."""
        if self.done:
            raise RuntimeError("the episode is over")
        self.turns += 1
        self.tokens_in += int(prompt_tokens or 0)
        self.tokens_out += int(completion_tokens or 0)
        self.turn_log.append(dict(turn=self.turns, reasoning=reasoning, text=text, raw_text=raw_text,
                                  tool_calls=[dict(id=c.get("id"), name=c.get("name"), arguments=c.get("arguments")) for c in calls],
                                  finish=[finish] if finish else [], model=None, sent_params=None, attempts=1, seconds=seconds,
                                  usage=dict(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
                                  if prompt_tokens is not None and completion_tokens is not None else {}))
        if callable(getattr(self.env, "note_turn", None)) and self.env.note_turn().get("truncated"):
            self.stop_reason, self.aborted = "truncated", "max_turns"    # only when the environment was made with a turn cap
            return [], True
        self.messages.append(mchat.assistant_message(text, calls))
        event = dict(step=None, role="assistant", content=text,
                     tool_calls=[dict(name=c["name"], arguments=c["arguments"]) for c in calls])
        self.events.append(event)
        appended, lost = [], []
        for k, c in enumerate(calls):
            self.tool_calls += 1
            key = c["name"] if c["name"] in self._known else "(unknown)"
            self.by_tool[key] = self.by_tool.get(key, 0) + 1
            if self.final is not None:
                appended.append(mchat.tool_message(c["id"], "error episode_over · the episode has ended; this call was not executed"))
            elif k >= runner.MAX_CALLS_PER_TURN:
                self.invalid_calls += 1
                appended.append(mchat.tool_message(c["id"], f"error malformed_call · at most {runner.MAX_CALLS_PER_TURN} tool calls per "
                                                            "reply; this call was not executed and nothing was charged"))
            else:
                step = self.env.step({"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}})
                appended += step.observation.to_openai(c["id"])
                if event["step"] is None:
                    event["step"] = self.env.state().get("tool_calls")
                if isinstance(step.info.get("step"), dict):
                    self.step_infos.append(dict(step.info["step"]))
                refused = step.observation.is_error and not I.counts_as_tool_call(step.observation.data)
                self.invalid_calls += bool(refused)
                lost.append(step.observation.is_error and I.error_class(step.observation.data) in I.INTERFACE_CLASSES)
                if step.done:
                    reason = step.info.get("reason")
                    self.final = step
                    self.stop_reason = ("rejected_call_limit" if reason == "rejected_call_limit" else "truncated") if step.truncated else "submitted"
        self.lost_turns += bool(lost) and all(lost)
        if self.final is None:
            notes = []
            if finish == "length":
                self.cut_replies += 1
                if callable(getattr(self.env, "note_length_cut", None)):
                    self.env.note_length_cut()
                notes.append(runner.CUT_NOTE.format(limit=f" ({self._reply_limit} tokens)" if self._reply_limit else ""))
            if not calls:
                self.idle += 1
                self.idle_turns += 1
                if self.idle < runner.IDLE_TURNS and not notes:
                    notes.append("No tool call was received. Continue with the tools; deliver with the terminal tool.")
            else:
                self.idle = 0
            if notes and self.idle < runner.IDLE_TURNS:                  # one user message, never two in a row
                appended.append({"role": "user", "content": " ".join(notes)})
        self.messages += appended
        self._check_seed(appended)
        return appended, self.done

    def abort(self, reason: str, reply: dict | None = None) -> None:
        """The trainer ends the episode: ``length`` (a reply cut before its end, or no room left in the context) or
        ``stopped``. Reward 0; the environment records the outcome (``no_submission_length`` / ``..._stopped``).
        ``reply``: the cut reply (``raw_text``, ``prompt_tokens``, ``completion_tokens``). It is no turn of the
        conversation (nothing answers it); it is kept in the record under ``training.cut_reply``."""
        if reason not in ABORT_REASONS:
            raise ValueError(f"reason must be one of {ABORT_REASONS}")
        if self.done:
            return
        self.aborted = reason
        self.stop_reason = "length_cut" if reason == "length" else "no_tool_calls"
        if reply:
            self.cut_reply = dict(reply)
            self.tokens_in += int(reply.get("prompt_tokens") or 0)
            self.tokens_out += int(reply.get("completion_tokens") or 0)
        if reason == "length":
            self.cut_replies += 1
            if callable(getattr(self.env, "note_length_cut", None)):
                self.env.note_length_cut()
        if callable(getattr(self.env, "end_without_submission", None)):
            self.env.end_without_submission(reason)

    # ------------------------------------------------------------------ result
    def result(self) -> dict:
        """Everything the trainer logs for the episode. JSON-safe. Call once the episode is over."""
        if not self.done:
            raise RuntimeError("the episode is still running")
        o = runner._outcome(self.env, self.final, self.stop_reason)
        reward, progress = float(o["reward"]), float(o.get("progress", 0.0) or 0.0)
        final = (self.env.training_info() if callable(getattr(self.env, "training_info", None)) else None) or {}
        counts = dict(final.get("counts") or o.get("counts") or {})
        calls = counts.get("calls", self.tool_calls) or 0
        return dict(
            reward=reward, passed=bool(o["passed"]), signal_mode=self.signal_mode,
            signal=reward if self.signal_mode == "pass" else progress, progress=progress,
            outcome=o.get("outcome"), ladder=o.get("ladder"), distance=final.get("distance") or {}, counts=counts,
            truncated_by=final.get("truncated_by"), group=o.get("group"), stop_reason=self.stop_reason,
            submitted=bool((o.get("ladder") or {}).get("submitted", self.stop_reason == "submitted")),
            turns=self.turns, tool_calls=self.tool_calls, invalid_calls=int(self.invalid_calls), cut_replies=self.cut_replies,
            interface_tax=(counts.get("shape", 0) + counts.get("vocabulary", 0)) / calls if calls else 0.0,
            scans=o.get("scans"), scan_s_used=o.get("scan_s_used"), tokens_in=self.tokens_in, tokens_out=self.tokens_out,
            env_id=self.env_id, level=self.level, seed=self.seed, split=self.split, wall_s=round(time.time() - self._t0, 3))

    def write_record(self, directory: str, *, agent: str, model: str, sampler: dict | None = None, episode: int = 0,
                     base_url: str | None = None, training: dict | None = None, command: str | None = None) -> dict:
        """Write ``record.json``, ``messages.json`` (the whole conversation), ``scanner_log.jsonl`` under ``directory``
        in the schema of ``mrgym run-agent`` (``mrgym.record.finish_episode``), so that ``mrgym results --dir`` and
        the transcript checks read a training episode like any other. Adds ``training``: the signal mode and value,
        the per-step ``info["step"]`` rows and whatever the trainer passes (step, group uid, session, phase)."""
        os.makedirs(directory, exist_ok=True)
        _note_run(directory, command or os.environ.get("MRGYM_COMMAND") or "integrations/verl (mrgym_verl)", model)
        result = self.result()
        settings = dict(self.settings, temperature=(sampler or {}).get("temperature"))
        rec = dict(schema=runner.RECORD_SCHEMA, agent=agent, agent_kind=runner.agent_kind(model), model=model,
                   params_sent=dict(sampler or {}), params_dropped=[], settings=settings,
                   env_id=self.env_id, level=self.level, seed=self.seed, split=self.split, episode=int(episode),
                   engine=mrgym.ENGINE_ID, package_version=mrgym.__version__, status="completed", utc=runner.now_utc(), wall_s=result["wall_s"])
        played = dict(runner._outcome(self.env, self.final, self.stop_reason), turns=self.turns, tool_calls=self.tool_calls,
                      invalid_calls=int(self.invalid_calls), by_tool=dict(self.by_tool),
                      interface=dict(self.env.interface_counts(), lost_turns=self.lost_turns, idle_turns=self.idle_turns,
                                     cut_replies=self.cut_replies),
                      tokens_in=self.tokens_in, tokens_out=self.tokens_out, tokens_estimated=False)
        if self.env.spec.family:
            played["task_id"] = getattr(self.env, "episode_config", lambda: {})().get("task_id")
        rec.update(played)
        sink = dict(tools=self.tools, messages=self.messages)
        job = dict(kind="model", agent=agent, model=model,
                   opts=dict(settings=settings, base_url=base_url, records_only=False, adapter_sha256=adapter_sha256()))
        client = _Turns(self.turn_log, dict(sampler or {}))
        runner._save_log(self.env, directory)
        if self._seed_hex not in json.dumps(sink):
            runner.write_json(os.path.join(directory, "messages.json"), sink, indent=1)
        rec = mrecord.finish_episode(rec, job, self.env, self.final, client, sink, directory)
        rec["harness"] = dict(rec["harness"], loop=LOOP_NAME)             # the loop that played it is this one, not the runner's
        rec["harness_id"] = mrecord.harness_id(rec["harness"])
        _restamp_transcript(rec, directory)
        rec["training"] = dict(signal_mode=self.signal_mode, signal=result["signal"], paid_reward=result["reward"],
                               progress=result["progress"], outcome=result["outcome"], aborted=self.aborted,
                               truncated_by=result["truncated_by"], cut_reply=self.cut_reply,
                               steps=self.step_infos, **(training or {}))
        rec = {k: rec[k] for k in runner.RECORD_FIELDS if k in rec} | {k: v for k, v in rec.items() if k not in runner.RECORD_FIELDS}
        runner.write_json(os.path.join(directory, "record.json"), rec, indent=1)
        return rec

    def close(self) -> None:
        self.env.close()


class _Turns:
    """What ``mrgym.record.finish_episode`` reads from a chat client: the turns and the sampling parameters."""

    def __init__(self, turns: list, params: dict):
        self.turns, self.params, self.dropped = turns, params, []


def _note_run(directory: str, command: str, model: str) -> None:
    """``run_config.json`` of the run folder an episode folder belongs to, written once (records name their command)."""
    out = mrecord._run_folder(directory)
    if not out or os.path.exists(os.path.join(out, "run_config.json")):
        return
    entry = dict(utc=runner.now_utc(), command=command, package_version=mrgym.__version__, engine=mrgym.ENGINE_ID, models=[model],
                 adapter_sha256=adapter_sha256(), loop=LOOP_NAME)
    tmp = os.path.join(out, f"run_config.json.{os.getpid()}.{time.time_ns()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(schema=runner.RUN_SCHEMA, created_utc=entry["utc"], invocations=[entry]), f, indent=1)
        f.write("\n")
    os.replace(tmp, os.path.join(out, "run_config.json"))


def _restamp_transcript(rec: dict, directory: str) -> None:
    """The stored conversation names the harness id; rewrite it after the loop name was corrected, and re-hash."""
    t = rec.get("transcript")
    if not t:
        return
    path = os.path.join(directory, t["file"])
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    doc["harness_id"] = rec["harness_id"]
    text = json.dumps(doc, indent=1) + "\n"
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(path + ".tmp", path)
    t.update(sha256=hashlib.sha256(text.encode()).hexdigest(), bytes=len(text.encode()))


def episode_dir(out: str, agent: str, env_id: str, level: int, seed: int, episode: int) -> str:
    """``<out>/episodes/<agent>/<env>/L<level>/s<seed>/e<k>``: the layout of ``mrgym run-agent``."""
    return os.path.join(out, "episodes", runner.slug(agent), runner.env_slug(env_id), f"L{level}", f"s{seed}", f"e{episode}")


def free_dir(directory: str, uid: str) -> str:
    """``directory``, or ``directory-<uid prefix>`` when an episode is already stored there: the same instance and
    session number played again (an evaluation repeated after a restore) is kept, never overwritten."""
    return directory if not os.path.exists(os.path.join(directory, "record.json")) else f"{directory}-{str(uid)[:8]}"


def play_chat(ep: Episode, client) -> dict:
    """Drive ``ep`` with a chat-completions client (``mrgym.agents.chat.ChatClient``): the HTTP form of the loop,
    used by the simulation and wherever a served endpoint is evaluated through this adapter. Returns ``ep.result()``.
    Raises ``chat.GatewayError`` on an endpoint failure."""
    messages, tools = ep.start()
    done = False
    while not done:
        resp = client.complete(ep.messages, tools)
        usage = resp.get("usage") or {}
        _, done = ep.assistant_turn(resp["text"], resp["tool_calls"], finish="length" if "length" in (resp.get("finish") or []) else None,
                                    prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                                    seconds=resp.get("seconds"), reasoning=mrecord.reasoning_of(resp.get("raw")))
    return ep.result()
