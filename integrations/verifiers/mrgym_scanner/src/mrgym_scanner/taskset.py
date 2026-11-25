"""verifiers v1 taskset over the MR Gym registry.

The adapter adds no physics and no scoring, and names no tool. The tool server (``ScannerToolset``) holds one
``mrgym.Env`` per rollout; its MCP tool list is ``env.tools("mcp")`` of that episode, with MR Gym's own JSON schemas,
and a tool call is one ``env.step``. The reward is the one ``env.step`` returned when the episode ended; scoring reads
typed state, never tool text. The seed is task data on the evaluator side and never enters the prompt.

    vf-validate mrgym-scanner -n 5 --runtime.type subprocess                           gold + no-op, every scored id
    vf-validate mrgym-scanner -n 5 --runtime.type subprocess --taskset.env-id <id>     one id
"""
from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import verifiers.v1 as vf

import mrgym
from mrgym import compat


class ScannerData(vf.TaskData):
    env_id: str
    level: int
    seed: int                      # evaluator side only: TaskData extras are never injected into the prompt
    split: str = "train"


class ScannerState(vf.State):
    done: bool = False
    truncated: bool = False
    reward: float = 0.0
    tool_calls: int = 0


class Episode:
    """One live episode. The only place in this package where a tool call is executed."""

    def __init__(self, env_id: str, level: int, seed, split: str = "train"):
        self.env = mrgym.make(env_id, level=level, seed=seed, split=split)
        obs, self.info = self.env.reset()
        self.texts = [obs.text]
        self.done, self.truncated, self.reward, self.tool_calls = False, False, 0.0, 0

    def tools(self) -> list[dict]:
        return self.env.tools("mcp")

    def call(self, name: str, arguments: dict | None) -> tuple[str, bool]:
        """One ``env.step``. Returns (observation text, is_error)."""
        if self.done:
            return "error: the episode is over", True
        r = self.env.step({"name": name, "arguments": arguments or {}})
        self.tool_calls += 1
        self.texts.append(r.observation.text)
        if r.done:
            self.done, self.truncated, self.reward = True, bool(r.truncated), float(r.reward)
        return r.observation.text, bool(r.observation.is_error)

    def trace(self) -> dict:
        return compat.trace(self.reward, self.done, self.texts, self.env)

    def sync(self, state: ScannerState) -> None:
        state.done, state.truncated, state.reward, state.tool_calls = self.done, self.truncated, self.reward, self.tool_calls


class ScannerToolset(vf.Toolset[vf.ToolsetConfig, ScannerState]):
    """Per-rollout MCP tool server. The catalogue is dynamic: whatever ``env.tools("mcp")`` lists for the episode."""

    TOOL_PREFIX = None             # the model sees MR Gym's tool names, not "<server>_<tool>"

    async def setup_task(self, task) -> None:
        self.episode = Episode(task.env_id, task.level, task.seed, getattr(task, "split", "train"))

    def register(self, mcp) -> None:
        # MCPServer's protocol handlers call these public methods; replacing them serves the episode's own tool list
        # with its exact input schemas instead of schemas derived from Python signatures.
        server: Any = mcp
        server.list_tools = self._with_state(self.list_tools)
        server.call_tool = self._with_state(self.call_tool)

    async def list_tools(self) -> list:
        from mcp.types import Tool
        return [Tool(name=t["name"], description=t["description"], input_schema=t["inputSchema"]) for t in self.episode.tools()]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None, context=None):
        from mcp.types import CallToolResult, TextContent
        text, is_error = self.episode.call(name, arguments)
        self.episode.sync(self.state)
        return CallToolResult(content=[TextContent(type="text", text=text)], is_error=is_error)


class ScannerTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class ScannerTask(vf.Task[ScannerData, ScannerState, ScannerTaskConfig]):
    @classmethod
    def toolsets(cls, config: ScannerTaskConfig) -> list[vf.Toolset]:
        return [ScannerToolset(config.tools)]

    @vf.stop
    async def submitted(self, trace: vf.Trace) -> bool:
        return trace.state.done

    @vf.reward(weight=1.0)
    async def mrgym_reward(self, trace: vf.Trace) -> float:
        return float(trace.state.reward)

    async def validate(self, runtime: vf.Runtime, trace: vf.Trace) -> bool | None:
        """Gold check: the id's scripted reference policy, replayed call by call through ``Episode.call`` (the function
        the tool server runs). None for an id without control policies: it has no model-free check."""
        d = self.data
        spec = mrgym.spec(d.env_id)
        if compat.reference_name(spec) is None:
            return None
        calls = compat.scripted_episode(d.env_id, d.level, d.seed)["calls"]
        episode = Episode(d.env_id, d.level, d.seed, d.split)
        for call in calls:
            episode.call(call["name"], call.get("arguments"))
        episode.sync(trace.state)
        episode.env.close()
        return episode.done and not episode.truncated and episode.reward >= spec.reward_threshold


class ScannerConfig(vf.TasksetConfig):
    env_id: str = ""               # "" = every scored id of the registry, in turn
    level: int | None = None       # None = the default level of each id
    first_seed: int = 0
    split: str = "train"           # "heldout" needs MRGYM_SECRET on the evaluator
    task: ScannerTaskConfig = ScannerTaskConfig()


class MRGymScannerTaskset(vf.Taskset[ScannerTask, ScannerConfig]):
    INFINITE = True                # procedural: every seed is an instance

    def load(self) -> Iterator[ScannerTask]:
        c = self.config
        ids = [c.env_id] if c.env_id else [i for i in mrgym.list_envs() if mrgym.spec(i).kind != "sandbox"]
        for i in itertools.count():
            env_id, seed = ids[i % len(ids)], c.first_seed + i // len(ids)
            level = mrgym.spec(env_id).default_level if c.level is None else c.level
            env = mrgym.make(env_id, level=level, seed=seed, split=c.split)
            obs, info = env.reset()
            env.close()
            yield ScannerTask(ScannerData(idx=i, name=f"{env_id}:L{level}:{compat.instance_label(info)}", prompt=obs.text,
                                          env_id=env_id, level=level, seed=seed, split=c.split), self.config.task)
