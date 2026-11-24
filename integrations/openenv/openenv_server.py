"""OpenEnv face: any ``mrgym.Env`` as an OpenEnv ``Environment`` (needs ``pip install openenv``; tested on 0.14.0).

Generated, not hand-written per family: tools come from ``env.tools("mcp")``; a ``CallToolAction`` is one
``env.step``; reward and ``done`` travel on the observation; ``state`` serves ``env.state()`` (nothing hidden).

    app = make_app("Scanner/T1Transmit-v0")           # FastAPI app: /ws, /reset, /step, /state, /schema
    uvicorn.run(app, port=8000)
    reset(seed=..., level=...)                        # the seed is a reset argument, i.e. a dataset column for TRL
"""
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation, ListToolsAction, ListToolsObservation, Tool
from openenv.core.env_server.types import Observation, State

import mrgym


class MRGymState(State):
    env_id: str = ""
    level: Optional[int] = None
    done: bool = False
    reward: Optional[float] = None
    public: dict = {}


def make_environment_class(env_id, **make_kwargs):
    class MRGymOpenEnv(Environment):
        SUPPORTS_CONCURRENT_SESSIONS = True               # one mrgym.Env per session, no shared mutable state

        def __init__(self):
            super().__init__()
            self._env, self._episode_id = None, None

        def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, level: Optional[int] = None, **kwargs: Any) -> Observation:
            kw = dict(make_kwargs, **({} if level is None else {"level": level}))
            self._env = mrgym.make(env_id, seed=seed, **kw)
            obs, info = self._env.reset()
            self._episode_id = episode_id or info["episode_id"]
            return Observation(done=False, reward=0.0, metadata={"text": obs.text, "task": obs.data, "status": obs.status, "info": info})

        def step(self, action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
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
            return MRGymState(episode_id=self._episode_id, step_count=st.get("tool_calls", 0), env_id=env_id, level=st.get("level"),
                              done=st.get("done", False), reward=st.get("reward"), public=st)

    return MRGymOpenEnv


def make_app(env_id, max_concurrent_envs=8, **make_kwargs):
    from openenv.core.env_server.http_server import create_app
    return create_app(make_environment_class(env_id, **make_kwargs), CallToolAction, CallToolObservation,
                      env_name=env_id.replace("/", "_").lower(), max_concurrent_envs=max_concurrent_envs, state_cls=MRGymState)
