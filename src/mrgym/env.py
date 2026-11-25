"""``mrgym.Env``: the episode lifecycle every environment shares.

    env = mrgym.make("Scanner/NoWrap-v1", level=1, seed=0)
    obs, info = env.reset()                      # task statement + tools; a new hidden world, a full budget
    tools = env.tools("openai")                  # or "mcp" / "anthropic": JSON-Schema function tools
    obs, reward, terminated, truncated, info = env.step({"name": "plan", "arguments": {...}})
    ...                                          # reward is 0 until the terminal tool (``submit``) is called
    env.close()

Two boundaries (the split OpenEnv makes): ``reset`` / ``step`` / ``state`` / ``close`` belong to whoever runs the
episode (a trainer, a harness, a test). The agent only ever sees the tools. No tool can reset the world or read it.

``step`` is total: for any input it returns ``(Observation, float, bool, bool, dict)``. A call the interface cannot
execute is answered with a typed error observation (``mrgym.interface``) and the episode continues; a fault of the
environment itself is answered the same way (``internal_error``) and counted for the trainer.

Limits. An episode ends at the terminal tool or at one of the limits, and every limit is told to the agent in the
task text (``Limits``), in the status line of every result, and in ``info["limits"]``: the same numbers.

    max_tool_calls         calls that reached a tool (``max_steps``), whatever the tool answered
    max_rejected_calls     calls the interface refused before any tool saw them (8); they cost nothing else
    max_invalid_deliverables   refused submissions (3), within the rejected calls
    plus whatever the environment charges (the scanner: scan time)

An environment implements three hooks: ``_reset(seed_hex)``, ``_tool_specs()``, ``_call(name, arguments)``.
"""
from __future__ import annotations

import base64
import copy
import dataclasses
import json

from . import interface as I
from . import keys
from . import tools as T
from .registry import Error

PUBLIC_KEY = keys.PUBLIC_KEY            # the key of the public splits: their worlds can be rebuilt by anyone
OBS_MODES = ("text", "vision")


class ResetNeeded(Error):
    pass


@dataclasses.dataclass
class Observation:
    """What comes back from ``reset`` and from every tool call.

    text     what a text-only model reads: one header line, key=value lines, aligned tables. Always complete: a
             model that never sees ``images`` loses nothing it needs for a number.
    data     the same content as JSON with full precision, for code, logs and structured-output channels.
    images   [{"png": bytes, "caption": str}] only in ``obs="vision"`` and only from the ``view`` tool.
    is_error a refused or malformed call. Refusals are observations, never exceptions, and cost nothing.
    status   budget and counters: data the agent may plan with, never a reward.
    """
    tool: str | None = None
    text: str = ""
    data: dict = dataclasses.field(default_factory=dict)
    images: list = dataclasses.field(default_factory=list)
    is_error: bool = False
    status: dict = dataclasses.field(default_factory=dict)

    def content(self) -> list:
        """MCP / Anthropic style content blocks."""
        blocks = [{"type": "text", "text": self.text}]
        for im in self.images:
            blocks.append({"type": "image", "data": base64.b64encode(im["png"]).decode(), "mimeType": "image/png"})
        return blocks

    def to_mcp(self) -> dict:
        return {"content": self.content(), "structuredContent": self.data, "isError": self.is_error}

    def to_openai(self, tool_call_id: str | None) -> list:
        """Chat Completions messages: the tool message is text; images follow as a user message (tool messages
        cannot carry images there)."""
        msgs = [{"role": "tool", "tool_call_id": tool_call_id, "content": self.text}]
        if self.images:
            msgs.append({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(im["png"]).decode()}}
                for im in self.images]})
        return msgs

    def to_anthropic(self, tool_use_id: str | None) -> dict:
        blocks = [{"type": "text", "text": self.text}] + [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(im["png"]).decode()}} for im in self.images]
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": blocks, "is_error": self.is_error}


class StepResult(tuple):
    """``(observation, reward, terminated, truncated, info)`` — unpacks like a Gymnasium step, with names."""
    __slots__ = ()
    observation = property(lambda s: s[0])
    reward = property(lambda s: s[1])
    terminated = property(lambda s: s[2])
    truncated = property(lambda s: s[3])
    info = property(lambda s: s[4])
    done = property(lambda s: s[2] or s[3])


def derive_seed(env_id: str, level: int, split: str, seed, environ=None) -> str:
    """128-bit world seed (32 hex digits) from a user seed; the derivations are in ``mrgym.keys``.

    public split      ``train``, ``dev``, ``tune``, ``gate``: HMAC-SHA256(public key, "<id>|<level>|<split>|<seed>"). Anyone
                      can rebuild the world, which is what a training split is. A secret in the environment does not
                      change these worlds; a trainer's salt (MRGYM_SALT_FILE) does, on purpose.
    secret split      ``test-*``, ``ood``, ``heldout``, ``semi``: keyed by the secret (MRGYM_SECRET_FILE); refuses without.
    32 hex digits     used as is (an operator's own seed).
    None              fresh entropy from the operating system: not reproducible, on purpose.
    A seed is an int (not a bool) or a short printable string; anything else, and any unknown split, raises ``Error``.
    """
    return keys.world_seed(env_id, level, split, seed, environ=environ)


class Env:
    """Base class. Subclasses set the hooks; everything else (ordering, validation, limits, transcript) is here."""

    def __init__(self, spec, level=None, seed=None, obs="text", split="train", max_steps=None, reveal_score=False,
                 max_rejected_calls=None):
        if obs not in OBS_MODES:
            raise Error(f"obs must be one of {OBS_MODES}")
        self.spec = spec
        self.level = spec.default_level if level is None else level
        self.obs_mode, self.split, self.reveal_score = obs, split, reveal_score
        self.max_steps = spec.max_steps if max_steps is None else max_steps
        self.max_rejected_calls = I.MAX_REJECTED_CALLS if max_rejected_calls is None else max_rejected_calls
        self.max_invalid_deliverables = I.MAX_INVALID_DELIVERABLES
        for name, value in (("max_steps", self.max_steps), ("max_rejected_calls", self.max_rejected_calls)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise Error(f"{name} must be a positive integer, not {value!r}")
        self._seed = seed
        self._episode = None             # None until reset
        self._harness = {}
        self._log = []

    # ------------------------------------------------------------------ hooks
    def _reset(self, seed_hex):          # -> Observation (the task statement)
        raise NotImplementedError

    def _tool_specs(self):               # -> [ToolSpec]; may depend on the episode (e.g. region ids in ``submit``)
        raise NotImplementedError

    def _call(self, name, arguments):    # -> Observation; a terminal tool sets self._finish(reward, info)
        raise NotImplementedError

    def _check(self, name, arguments):   # -> problems (mrgym.interface.problem) of a call that fits its schema
        return []

    def _status(self):                   # -> dict shown with every observation
        return {}

    def _info(self):                     # -> extra keys of the info dict returned by reset (trainer side)
        return {}

    def _limits(self):                   # -> the environment's own limits, {name: number} (the scanner: scan time)
        return {}

    def _limit_sentences(self):          # -> the sentences of the Limits block for the environment's own limits
        return []

    def _spent(self):                    # -> what the environment has charged so far (the scanner: scan seconds)
        return 0.0

    # ------------------------------------------------------------------ limits: one set of numbers, said three times
    def limits(self) -> dict:
        """Every limit that can end the episode. The task text, the status line and ``info`` state these numbers."""
        out = dict(self._limits(), max_tool_calls=self.max_steps, max_rejected_calls=self.max_rejected_calls,
                   max_invalid_deliverables=self.max_invalid_deliverables)
        if self._harness:
            out["harness"] = dict(self._harness)
        return out

    def limits_text(self) -> str:
        """The Limits block that ends the task text; the same in every harness."""
        terminal = next((t.name for t in self._tool_specs() if t.terminal), "the terminal tool")
        parts = self._limit_sentences() + [
            f"Tool calls: {self.max_steps} (every call that reaches a tool counts once).",
            f"Rejected calls: at most {self.max_rejected_calls} (a malformed call, such as an unknown tool or parameter or a value of the "
            f"wrong type or outside its fixed range, reaches no tool: it is answered with a fix and costs nothing else), of them at most "
            f"{self.max_invalid_deliverables} refused `{terminal}` calls."]
        if "max_reply_tokens" in self._harness:
            parts.append(f"Reply length: {self._harness['max_reply_tokens']} tokens (harness).")
        parts.append(f"The episode ends at `{terminal}` or at a limit; without `{terminal}` the score is 0.")
        return "Limits. " + " ".join(parts)

    # ------------------------------------------------------------------ lifecycle
    def reset(self, *, seed=None, options: dict | None = None) -> tuple[Observation, dict]:
        """Start a new episode: new hidden world, full budget, empty log. Returns (observation, info).

        ``options``: ``{"harness": {"max_reply_tokens": N}}`` lets the harness state its own limit, so that the task
        text names every limit the agent meets. Nothing else is accepted."""
        options = dict(options or {})
        harness = options.pop("harness", None) or {}
        if options or not isinstance(harness, dict) or set(harness) - {"max_reply_tokens"}:
            raise Error(f"unknown reset options {sorted(options) or sorted(set(harness) - {'max_reply_tokens'})}; "
                        "accepted: {\"harness\": {\"max_reply_tokens\": N}}")
        if "max_reply_tokens" in harness and (isinstance(harness["max_reply_tokens"], bool)
                                              or not isinstance(harness["max_reply_tokens"], int) or harness["max_reply_tokens"] < 1):
            raise Error("harness.max_reply_tokens must be a positive integer")
        self._harness = harness
        if seed is not None:
            self._seed = seed
        seed_hex = derive_seed(self.spec.id, self.level, self.split, self._seed)
        # the episode id is random: two episodes of one world differ, and the id says nothing about the seed
        self._episode = dict(id=keys.episode_handle(), steps=0, rejected=0, invalid_deliverables=0, internal_errors=0, coerced=0,
                             done=False, reward=None, info={}, truncated=False, by_class={c: 0 for c in I.CLASSES}, faults=[])
        self._log = []
        obs = self._reset(seed_hex)
        obs.text = obs.text.rstrip() + "\n\n" + self.limits_text()
        obs.data = dict(obs.data, limits=self.limits())
        obs.status = self._full_status()
        obs.text = obs.text.rstrip() + "\n" + self._status_line(obs.status)
        obs.data = copy.deepcopy(obs.data)
        info = dict(env_id=self.spec.id, level=self.level, split=self.split, episode_id=self._episode["id"],
                    obs=self.obs_mode, max_steps=self.max_steps, limits=self.limits(), tools=[t.name for t in self._tool_specs()])
        info.update(self._info())
        self._log.append(dict(step=0, call=None, observation=obs, reward=0.0))
        return obs, info

    def tools(self, format: str = "openai") -> list:
        """The tool list the agent sees, in the named wire format. After ``reset`` it is specific to the episode."""
        return [t.to(format) for t in self._tool_specs()]

    def admit(self, tool_call) -> tuple:
        """The front half of ``step``, without executing anything: ``(name, canonical arguments, coercions, problems)``.

        ``problems`` is empty when the call would be handed to the tool. Read-only: a harness may use it to check a
        call before sending it, and the interface-tax metric uses it to class recorded calls."""
        name, _, args, coercions, problems = self._admit(tool_call)
        return name, args, coercions, problems

    def _admit(self, tool_call) -> tuple:
        """``(name, arguments as written, canonical arguments, coercions, problems)``."""
        specs = {t.name: t for t in self._tool_specs()}
        try:
            name, written, _ = T.parse_tool_call(tool_call)
        except ValueError as e:
            return None, None, None, [], [I.problem("malformed_call", None, str(e))]
        if name not in specs:
            guess = I.did_you_mean(name, list(specs))
            return name, None, None, [], [I.problem("unknown_tool", None, f"no tool named {T._show(name, 40)}"
                                                    + (f"; did you mean {guess}?" if guess else ".") + f" Tools: {', '.join(specs)}",
                                                    did_you_mean=guess, allowed={"values": list(specs)})]
        unsafe = T.unsafe_values(written)
        if unsafe:
            return name, None, None, [], [I.problem("malformed_call", None, "; ".join(unsafe[:4]))]
        spec = specs[name]
        written = copy.deepcopy(written)                                 # bounded by the check above; detached from the caller's object
        args, coercions, problems = I.canonicalise(spec, written)
        top = lambda p: str(p["parameter"]).split(".")[0].split("[")[0]      # noqa: E731
        named = {top(p) for p in problems}
        problems = problems + [p for p in I.schema_problems(spec, args) if top(p) not in named]
        if not problems:
            problems = list(self._check(name, args))
        return name, written, args, coercions, problems

    def step(self, tool_call) -> StepResult:
        """Execute one tool call. Never raises for anything the agent did: errors come back as observations."""
        ep = self._episode
        if ep is None:
            raise ResetNeeded("call reset() before step()")
        if ep["done"]:
            raise ResetNeeded("the episode is over; call reset() to start another")
        name = args = written = None
        coercions, before, refused = [], 0.0, False
        try:
            before = float(self._spent())
            name, written, args, coercions, problems = self._admit(tool_call)
            refused = bool(problems)
            if problems:
                data = I.error_data(problems)
                obs = Observation(is_error=True, data=data, text=I.error_text(data))
            else:
                obs = self._call(name, args)
        except ResetNeeded:
            raise
        except Exception as e:  # noqa: BLE001  a fault of the environment is not the agent's: answer, count, keep the episode
            ep["internal_errors"] += 1
            ep["faults"].append(f"{type(e).__name__}: {e}"[:300])        # trainer side; the text never reaches the agent
            data = I.error_data([I.problem("internal_error", None, "the environment could not execute this call (a fault on its side, "
                                           "not yours); send another call")])
            obs = Observation(is_error=True, data=data, text=I.error_text(data))
        obs.tool = name
        # ---- what the call is charged
        counted = I.counts_as_tool_call(obs.data) if obs.is_error else True
        if counted:
            ep["steps"] += 1
        else:
            ep["rejected"] += 1
            ep["invalid_deliverables"] += (obs.data.get("error") or {}).get("type") == "invalid_deliverable"
        if obs.is_error:
            ep["by_class"][I.error_class(obs.data) or "task"] += 1
        if coercions:
            ep["coerced"] += 1
            obs.data = dict(obs.data, coercions=coercions)                # in data only: the text is a function of the canonical call
        truncated = False
        if not ep["done"]:
            if ep["steps"] >= self.max_steps:
                reason, why = "max_steps", f"{self.max_steps} tool calls used without a submission"
            elif ep["rejected"] >= self.max_rejected_calls:
                reason, why = "rejected_call_limit", f"{self.max_rejected_calls} calls rejected"
            elif ep["invalid_deliverables"] >= self.max_invalid_deliverables:
                reason, why = "rejected_call_limit", f"{self.max_invalid_deliverables} submissions refused"
            else:
                reason = None
            if reason:
                ep.update(done=True, truncated=True, reward=0.0, info={"reason": reason, "submitted": False})
                truncated = True
        if obs.is_error:
            spent = max(0.0, float(self._spent()) - before)
            obs.data = dict(obs.data, charged=dict(scan_time_s=round(spent, 6), tool_calls=int(counted), rejected_calls=int(not counted)),
                            episode_continues=not ep["done"])
            cost = self._charged_line(counted)
            if cost:
                obs.text = obs.text.rstrip() + f"\n{cost}" + ("; the episode continues" if not ep["done"] else "")
        if truncated:
            obs.text += f"\nepisode truncated: {why}"
        obs.status = self._full_status()
        obs.text = obs.text.rstrip() + "\n" + self._status_line(obs.status)
        obs.data = copy.deepcopy(obs.data)                               # nothing returned shares an object with the episode's state
        reward = float(ep["reward"]) if ep["done"] else 0.0
        if ep["done"]:
            ep["info"] = dict(ep["info"], limits=self.limits(), interface=self.interface_counts())
        info = copy.deepcopy(ep["info"]) if ep["done"] else {}
        # the transcript keeps what was executed in its canonical form, and a call that was refused as it was written:
        # replaying the transcript's calls then reproduces the episode
        call = {"raw": _bounded(tool_call)} if written is None else {"name": name, "arguments": written if refused else args}
        self._log.append(dict(step=ep["steps"], call=call, observation=obs, reward=reward))
        return StepResult((obs, reward, ep["done"] and not truncated, truncated, info))

    def _finish(self, reward, info):
        self._episode.update(done=True, reward=reward, info=info)

    def interface_counts(self) -> dict:
        """Trainer side: how the calls of this episode fared at the interface. ``shape`` + ``vocabulary`` is the
        interface tax; ``task_refusals`` are the instrument's own refusals; ``coerced`` counts calls the canonicaliser
        rewrote (format skill the interface absorbed); ``internal_errors`` are faults of the environment."""
        ep = self._episode
        return dict(calls=ep["steps"] + ep["rejected"], executed=ep["steps"], rejected=ep["rejected"], shape=ep["by_class"]["shape"],
                    vocabulary=ep["by_class"]["vocabulary"], task_refusals=ep["by_class"]["task"], coerced=ep["coerced"],
                    internal_errors=ep["internal_errors"])

    def _charged_line(self, counted: bool):
        """The line under a refusal that says what it cost; None when the status line of the environment says it."""
        ep = self._episode
        return (f"charged: no scan time; counts as tool call {ep['steps']} of {self.max_steps}" if counted else
                f"charged: nothing; rejected call {ep['rejected']} of {self.max_rejected_calls}")

    def _full_status(self):
        ep = self._episode
        return dict(self._status(), tool_calls=ep["steps"], max_tool_calls=self.max_steps, tool_calls_left=max(0, self.max_steps - ep["steps"]),
                    rejected_calls=ep["rejected"], max_rejected_calls=self.max_rejected_calls)

    def _status_line(self, s):
        return "[" + " · ".join(self._status_parts(s)) + "]"

    def _status_parts(self, s):
        parts = [f"tool calls {s['tool_calls']} of {s['max_tool_calls']} used, {s['tool_calls_left']} left",
                 f"rejected {s['rejected_calls']} of {s['max_rejected_calls']}"]
        if s["tool_calls_left"] <= 3 and not self._episode["done"]:
            terminal = next((t.name for t in self._tool_specs() if t.terminal), None)
            if terminal:
                parts.append(f"{terminal} now or the episode scores 0")
        return parts

    def state(self) -> dict:
        """Public episode state for the trainer / UI. Holds nothing hidden (it is what an OpenEnv ``/state`` serves)."""
        if self._episode is None:
            return dict(env_id=self.spec.id, level=self.level, episode_id=None)
        ep = self._episode
        return dict(env_id=self.spec.id, level=self.level, episode_id=ep["id"], done=ep["done"],
                    truncated=ep["truncated"], reward=ep["reward"] if ep["done"] else None, internal_errors=ep["internal_errors"],
                    **self._full_status())

    def faults(self) -> list:
        """Trainer side: the exception texts behind this episode's ``internal_error`` answers."""
        return list(self._episode["faults"]) if self._episode else []

    def close(self) -> None:
        self._episode = None

    # ------------------------------------------------------------------ transcript
    def transcript(self) -> list:
        """[{step, call, text, is_error, reward}] of the episode so far (what a viewer or a trajectory file needs)."""
        return [dict(step=e["step"], call=e["call"], text=e["observation"].text, is_error=e["observation"].is_error,
                     images=[im["caption"] for im in e["observation"].images], reward=e["reward"]) for e in self._log]

    def render(self, width: int = 118) -> str:
        out = [f"=== {self.spec.id} · level {self.level} · episode {self._episode['id'] if self._episode else '-'} · obs={self.obs_mode} ==="]
        for e in self.transcript():
            if e["call"] is None:
                out.append("\n[reset]")
            else:
                call = json.dumps(e["call"].get("arguments", e["call"].get("raw")), separators=(", ", ": "), default=str)
                name = e["call"].get("name", "<malformed>")
                out.append(f"\n[{e['step']}] agent -> {name}({call if len(call) <= width - 20 else call[:width - 23] + '...'})")
            for line in e["text"].splitlines():
                out.append("    " + line)
            for cap in e["images"]:
                out.append(f"    <image/png: {cap}>")
        if self._episode and self._episode["done"]:
            out.append(f"\n=== episode over · reward {self._episode['reward']:.3f} · "
                       f"{'truncated' if self._episode['truncated'] else 'terminated'} ===")
        return "\n".join(out)


def _bounded(tool_call):
    """A refused call as the transcript keeps it: the call itself when it is small, else a note of what it was. A
    transcript must stay serialisable whatever was sent."""
    try:
        if T.unsafe_values(tool_call):
            return f"<omitted: a call the interface refused as too large, too deep or not JSON ({type(tool_call).__name__})>"
        json.dumps(tool_call)
    except Exception:  # noqa: BLE001
        return f"<omitted: not JSON ({type(tool_call).__name__})>"
    return tool_call
