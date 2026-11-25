"""Any registered ``mrgym.Env`` as an OpenEnv ``Environment`` (written against openenv 0.14).

The class adds no physics and no scoring, and names no tool. Tools come from ``env.tools("mcp")`` of the episode; a
``CallToolAction`` is one ``env.step``; reward and ``done`` travel on the observation; ``state`` serves
``env.state()``, which holds nothing hidden.

    reset(seed=..., env_id=..., level=...)      the seed is a reset argument, i.e. a dataset column for a trainer

Trust model: the environment runs inside the server process. That hides the world from a client that only sends tool
calls. It does not hide it from code running in the server process.
"""
from __future__ import annotations

from typing import Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    Tool,
)
from openenv.core.env_server.types import Observation, State

import mrgym


class MRGymState(State):
    env_id: str = ""
    level: int | None = None
    done: bool = False
    reward: float | None = None
    public: dict = {}


def make_environment_class(default_env_id: str, **make_kwargs):
    """An ``Environment`` class whose episodes default to ``default_env_id``; ``reset(env_id=...)`` selects another
    registered id for that episode."""

    class MRGymOpenEnv(Environment):
        SUPPORTS_CONCURRENT_SESSIONS = True               # one mrgym.Env per session, no shared mutable state

        def __init__(self):
            super().__init__()
            self._env, self._episode_id, self._env_id = None, None, default_env_id

        @property
        def mrgym_env(self):
            """Trainer side: the wrapped ``mrgym.Env`` of the current episode."""
            return self._env

        def reset(self, seed: int | None = None, episode_id: str | None = None, env_id: str | None = None,
                  level: int | None = None, options: dict | None = None, **kwargs: Any) -> Observation:
            options = options or {}
            self._env_id = env_id or options.get("env_id") or default_env_id
            level = level if level is not None else options.get("level")
            kw = dict(make_kwargs, **({} if level is None else {"level": level}))
            if self._env is not None:
                self._env.close()
            self._env = mrgym.make(self._env_id, seed=seed, **kw)
            obs, info = self._env.reset()
            self._episode_id = episode_id or info["episode_id"]
            # the observation travels to the client: no trainer-side label of the instance in it (see ``mrgym.keys``)
            public = {k: v for k, v in info.items() if k not in ("task_id", "admitted", "gate", "log_head")}
            return Observation(done=False, reward=0.0, metadata={"text": obs.text, "task": obs.data, "status": obs.status, "info": public})

        def step(self, action, timeout_s: float | None = None, **kwargs: Any) -> Observation:
            if self._env is None:
                return Observation(done=False, reward=0.0, metadata={"error": "call reset() first"})
            if isinstance(action, ListToolsAction):
                return ListToolsObservation(tools=[Tool(name=t["name"], description=t["description"], input_schema=t["inputSchema"])
                                                   for t in self._env.tools("mcp")])
            if not isinstance(action, CallToolAction):
                return Observation(done=False, reward=0.0, metadata={"error": "use ListToolsAction or CallToolAction"})
            try:
                s = self._env.step({"name": action.tool_name, "arguments": action.arguments})
            except mrgym.ResetNeeded as e:
                return CallToolObservation(tool_name=action.tool_name, result={"text": f"error: {e}", "isError": True}, done=True, reward=0.0)
            return CallToolObservation(tool_name=action.tool_name, result=s.observation.to_mcp(), done=s.done, reward=s.reward,
                                       metadata={"status": s.observation.status, "truncated": s.truncated})

        @property
        def state(self) -> MRGymState:
            st = self._env.state() if self._env else {}
            return MRGymState(episode_id=self._episode_id, step_count=st.get("tool_calls", 0), env_id=self._env_id, level=st.get("level"),
                              done=st.get("done", False), reward=st.get("reward"), public=st)

    return MRGymOpenEnv


def make_app(default_env_id: str, max_concurrent_envs: int = 8, **make_kwargs):
    from openenv.core.env_server.http_server import create_app
    return create_app(make_environment_class(default_env_id, **make_kwargs), CallToolAction, CallToolObservation,
                      env_name="mrgym_scanner_env", max_concurrent_envs=max_concurrent_envs, state_cls=MRGymState)
