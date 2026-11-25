"""Any registered MR Gym id as an Inspect AI task.  Needs ``pip install mrgym[inspect]`` (written against inspect-ai 0.3.292).

    inspect eval integrations/inspect/mrgym_inspect.py@mrgym_scanner -T env_id=Scanner/NoWrap-v1 -T level=1 -T seeds=0-9 \
        --model <provider/model>
    python integrations/inspect/mock_run.py          # scripted mock model: reference = native reward, no tool call = floor

The adapter adds no physics and no scoring, and names no tool: the tools are ``env.tools("openai")`` of the episode,
with MR Gym's own JSON schemas, and a tool call is one ``env.step``. The hidden world lives in the evaluator process;
the model sees the task statement and tool results only. The seed stays in the sample metadata and is never part of
the sample id or the prompt.

``env_id`` omitted: every scored id of the registry at its default level. ``seeds`` is ``"lo-hi"`` (inclusive) or one
integer. ``split="heldout"`` needs ``MRGYM_SECRET`` in the evaluator's environment.
"""
from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, accuracy, mean, scorer, stderr
from inspect_ai.solver import TaskState, solver
from inspect_ai.tool import ToolDef, ToolParams

import mrgym
from mrgym import compat

TASK_VERSION = 1                 # Inspect task version: bump when the adapter changes what a model sees or how it is scored


class Session:
    """One live episode on the evaluator side. (A plain class: Inspect loads this file as an unregistered module, where
    ``dataclasses`` cannot resolve annotations.)"""

    def __init__(self, env, first_text):
        self.env, self.texts = env, [first_text]
        self.reward, self.done, self.truncated, self.tool_calls = 0.0, False, False, 0
        self.undescribed = 0     # tool parameters that arrived without a description (Inspect requires one)


_SESSIONS: dict[str, Session] = {}


def _seeds(text) -> range:
    lo, _, hi = str(text).partition("-")
    return range(int(lo), int(hi or lo) + 1)


def _tool(session: Session, spec: dict):
    fn = spec["function"]
    name, schema = fn["name"], fn["parameters"]

    async def call(**arguments):
        if session.done:
            return "error: the episode is over"
        r = session.env.step({"name": name, "arguments": arguments})
        session.tool_calls += 1
        session.texts.append(r.observation.text)
        if r.done:
            session.done, session.reward, session.truncated = True, float(r.reward), bool(r.truncated)
        return r.observation.text

    properties = {}
    for key, sub in schema.get("properties", {}).items():
        if not sub.get("description"):                        # Inspect refuses a parameter without a description
            session.undescribed += 1
            sub = {"description": key, **sub}
        properties[key] = sub
    params = ToolParams.model_validate({"type": "object", "properties": properties, "required": schema.get("required", [])})
    return ToolDef(call, name=name, description=fn["description"], parameters=params, parallel=False).as_tool()


@solver
def scanner_session():
    """Start the episode of this sample and give the model its tools."""
    async def solve(state: TaskState, generate):
        m = state.metadata
        env = mrgym.make(m["env_id"], level=m["level"], seed=m["seed"], split=m.get("split", "train"))
        obs, _ = env.reset()
        session = _SESSIONS[state.uuid] = Session(env, obs.text)
        state.tools = [_tool(session, t) for t in env.tools("openai")]
        return state

    return solve


@solver
def tool_loop(max_turns: int):
    """Generate, execute the tool calls of that turn, repeat until the episode ends or the model stops calling tools."""
    async def solve(state: TaskState, generate):
        for _ in range(max_turns):
            state = await generate(state, tool_calls="single")
            if _SESSIONS[state.uuid].done or not state.output.message.tool_calls:
                break
        return state

    return solve


@scorer(metrics={"passed": [accuracy(), stderr()], "reward": [mean(), stderr()]})
def mrgym_reward():
    """The reward ``env.step`` returned when the episode ended; the floor of the id when it never ended."""
    async def score(state: TaskState, target):
        session = _SESSIONS.pop(state.uuid)
        spec = mrgym.spec(state.metadata["env_id"])
        reward = session.reward if session.done else float(spec.reward_range[0])
        trace = compat.trace(reward, session.done, session.texts, session.env)
        session.env.close()
        passed = session.done and not session.truncated and spec.kind != "sandbox" and reward >= spec.reward_threshold
        return Score(value={"passed": float(passed), "reward": reward},
                     metadata=dict(trace, tool_calls=session.tool_calls, truncated=session.truncated,
                                   undescribed_parameters=session.undescribed, mrgym=mrgym.__version__, engine=mrgym.ENGINE_ID))

    return score


@task
def mrgym_scanner(env_id: str | None = None, level: int | None = None, seeds: str = "0-9", split: str = "train"):
    ids = [env_id] if env_id else [i for i in mrgym.list_envs() if mrgym.spec(i).kind != "sandbox"]
    samples, turns = [], 1
    for i in ids:
        spec = mrgym.spec(i)
        lv = spec.default_level if level is None else int(level)
        turns = max(turns, spec.max_steps)
        for s in _seeds(seeds):
            env = mrgym.make(i, level=lv, seed=s, split=split)
            obs, info = env.reset()
            env.close()
            samples.append(Sample(id=f"{i}:L{lv}:{compat.instance_label(info)}", input=obs.text,
                                  metadata={"env_id": i, "level": lv, "seed": s, "split": split}))
    return Task(dataset=samples, solver=[scanner_session(), tool_loop(turns)], scorer=mrgym_reward(), version=TASK_VERSION,
                metadata={"env_ids": ids, "split": split, "mrgym": mrgym.__version__, "engine": mrgym.ENGINE_ID})
