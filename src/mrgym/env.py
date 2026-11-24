"""``mrgym.Env``: the episode lifecycle every environment shares.

    env = mrgym.make("Scanner/NoWrap-v0", level=1, seed=0)
    obs, info = env.reset()                      # task statement + tools; a new hidden world, a full budget
    tools = env.tools("openai")                  # or "mcp" / "anthropic": JSON-Schema function tools
    obs, reward, terminated, truncated, info = env.step({"name": "plan", "arguments": {...}})
    ...                                          # reward is 0 until the terminal tool (``submit``) is called
    env.close()

Two boundaries (the split OpenEnv makes): ``reset`` / ``step`` / ``state`` / ``close`` belong to whoever runs the
episode (a trainer, a harness, a test). The agent only ever sees the tools. No tool can reset the world or read it.

An environment implements three hooks: ``_reset(seed_hex)``, ``_tool_specs()``, ``_call(name, arguments)``.
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import hmac
import json
import os

from . import tools as T
from .registry import Error

PUBLIC_KEY = "mrgym-public-v0"          # without MRGYM_SECRET every seed is public: fine for train, useless for held-out
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
    """128-bit world seed (32 hex digits) from a user seed.

    int / short str   HMAC-SHA256(key, "<id>|<level>|<split>|<seed>"), key = MRGYM_SECRET or the public key. With the
                      public key anyone can rebuild the world, which is what a training split is. The held-out split
                      refuses to run without a secret.
    32 hex digits     used as is (an operator's own secret seed).
    None              fresh entropy from the operating system: not reproducible, on purpose.
    """
    environ = os.environ if environ is None else environ
    if seed is None:
        return os.urandom(16).hex()
    if isinstance(seed, str) and len(seed) == 32 and all(c in "0123456789abcdef" for c in seed):
        return seed
    secret = environ.get("MRGYM_SECRET")
    if split != "train" and not secret:
        raise Error(f"split {split!r} needs MRGYM_SECRET in the environment: without it the worlds are public")
    msg = f"{env_id}|{level}|{split}|{seed}".encode()
    return hmac.new((secret or PUBLIC_KEY).encode(), msg, hashlib.sha256).hexdigest()[:32]


class Env:
    """Base class. Subclasses set the hooks; everything else (ordering, validation, limits, transcript) is here."""

    def __init__(self, spec, level=None, seed=None, obs="text", split="train", max_steps=None, reveal_score=False):
        if obs not in OBS_MODES:
            raise Error(f"obs must be one of {OBS_MODES}")
        self.spec = spec
        self.level = spec.default_level if level is None else level
        self.obs_mode, self.split, self.reveal_score = obs, split, reveal_score
        self.max_steps = spec.max_steps if max_steps is None else max_steps
        self._seed = seed
        self._episode = None             # None until reset
        self._log = []

    # ------------------------------------------------------------------ hooks
    def _reset(self, seed_hex):          # -> Observation (the task statement)
        raise NotImplementedError

    def _tool_specs(self):               # -> [ToolSpec]; may depend on the episode (e.g. region ids in ``submit``)
        raise NotImplementedError

    def _call(self, name, arguments):    # -> Observation; a terminal tool sets self._finish(reward, info)
        raise NotImplementedError

    def _status(self):                   # -> dict shown with every observation
        return {}

    def _info(self):                     # -> extra keys of the info dict returned by reset (trainer side)
        return {}

    # ------------------------------------------------------------------ lifecycle
    def reset(self, *, seed=None, options: dict | None = None) -> tuple[Observation, dict]:
        """Start a new episode: new hidden world, full budget, empty log. Returns (observation, info)."""
        if options:
            raise Error(f"unknown reset options {sorted(options)}")
        if seed is not None:
            self._seed = seed
        seed_hex = derive_seed(self.spec.id, self.level, self.split, self._seed)
        self._episode = dict(id=hashlib.sha256(("episode:" + seed_hex).encode()).hexdigest()[:12], steps=0, done=False,
                             reward=None, info={}, truncated=False)
        self._log = []
        obs = self._reset(seed_hex)
        obs.status = self._full_status()
        obs.text = obs.text.rstrip() + "\n" + self._status_line(obs.status)
        info = dict(env_id=self.spec.id, level=self.level, split=self.split, episode_id=self._episode["id"],
                    obs=self.obs_mode, max_steps=self.max_steps, tools=[t.name for t in self._tool_specs()])
        info.update(self._info())
        self._log.append(dict(step=0, call=None, observation=obs, reward=0.0))
        return obs, info

    def tools(self, format: str = "openai") -> list:
        """The tool list the agent sees, in the named wire format. After ``reset`` it is specific to the episode."""
        return [t.to(format) for t in self._tool_specs()]

    def step(self, tool_call: dict) -> StepResult:
        """Execute one tool call. Never raises for anything the agent did: errors come back as observations."""
        ep = self._episode
        if ep is None:
            raise ResetNeeded("call reset() before step()")
        if ep["done"]:
            raise ResetNeeded("the episode is over; call reset() to start another")
        ep["steps"] += 1
        specs = {t.name: t for t in self._tool_specs()}
        name = args = None
        try:
            name, args, _ = T.parse_tool_call(tool_call)
            if name not in specs:
                raise ValueError(f"unknown tool {name!r}; available: {sorted(specs)}")
            schema = getattr(self, "_shallow_schemas", {}).get(name, specs[name].parameters)
            args = T.normalise(args, specs[name].parameters)     # the full schema knows the type of every field
            unsafe = T.unsafe_values(args, specs[name].parameters)
            if unsafe:
                raise ValueError("; ".join(unsafe[:4]))
            problems = T.validate(args, schema)
            if problems:
                raise ValueError("; ".join(problems[:4]))
        except ValueError as e:
            obs = Observation(tool=name, is_error=True, text=f"error: invalid_tool_call: {e}\nnothing was charged",
                              data={"error": {"type": "invalid_tool_call", "message": str(e)}})
        else:
            obs = self._call(name, args)
            obs.tool = name
        truncated = False
        if not ep["done"] and ep["steps"] >= self.max_steps:
            ep.update(done=True, truncated=True, reward=0.0, info={"reason": "max_steps", "submitted": False})
            truncated = True
            obs.text += f"\nepisode truncated: {self.max_steps} tool calls used without a submission"
        obs.status = self._full_status()
        obs.text = obs.text.rstrip() + "\n" + self._status_line(obs.status)
        reward = float(ep["reward"]) if ep["done"] else 0.0
        info = dict(ep["info"]) if ep["done"] else {}
        self._log.append(dict(step=ep["steps"], call={"name": name, "arguments": args} if name else {"raw": tool_call},
                              observation=obs, reward=reward))
        return StepResult((obs, reward, ep["done"] and not truncated, truncated, info))

    def _finish(self, reward, info):
        self._episode.update(done=True, reward=reward, info=info)

    def _full_status(self):
        ep = self._episode
        return dict(self._status(), tool_calls=ep["steps"], max_tool_calls=self.max_steps)

    def _status_line(self, s):
        return "[" + " · ".join(f"{k} {v}" for k, v in s.items()) + "]"

    def state(self) -> dict:
        """Public episode state for the trainer / UI. Holds nothing hidden (it is what an OpenEnv ``/state`` serves)."""
        if self._episode is None:
            return dict(env_id=self.spec.id, level=self.level, episode_id=None)
        ep = self._episode
        return dict(env_id=self.spec.id, level=self.level, episode_id=ep["id"], done=ep["done"],
                    truncated=ep["truncated"], reward=ep["reward"] if ep["done"] else None, **self._full_status())

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
