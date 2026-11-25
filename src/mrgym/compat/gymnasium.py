"""Every registered MR Gym id as a ``gymnasium.Env``, under the same id.  Needs ``pip install mrgym[gymnasium]``.

    import gymnasium as gym
    import mrgym.compat.gymnasium                       # registers every id of ``mrgym.list_envs()``
    env = gym.make("Scanner/NoWrap-v1", level=1)        # or gym.make("mrgym.compat.gymnasium:Scanner/NoWrap-v1")
    text, info = env.reset(seed=3)                      # info["tools"]: the tool schemas of this episode
    text, reward, terminated, truncated, info = env.step('{"name": "plan", "arguments": {...}}')

    python -m mrgym.compat.gymnasium --check            # gymnasium's check_env on every registered id

Action = one tool call as a JSON string (space ``ToolCall``). Observation = the text the agent reads (space
``Message``). Both spaces accept any string: an observation may echo agent input, so a fixed character set would
reject valid observations, and a malformed action must reach the environment, which answers it with an error
observation. The adapter adds no physics and no scoring, and names no tool: the ids come from the registry and the tool
list from ``env.tools``.

``info`` is the trainer's side, as in Gymnasium: at reset it holds the instance label and the tool schemas, at the end
the score and the head of the scanner's log. Show the model the observation and the tools, never the rest of ``info``.

Seeding follows Gymnasium: ``reset(seed=k)`` starts the world of seed ``k`` (the same world as
``mrgym.make(id, seed=k)``); ``reset()`` without a seed draws the next world from the generator that the last seeded
reset set up, so ``reset(seed=k), reset(), reset()`` is a reproducible stream of distinct episodes.
"""
from __future__ import annotations

import json
import warnings

import gymnasium as gym
from gymnasium import spaces

import mrgym

from . import plain


class Message(spaces.Space):
    """Any ``str`` of at most ``max_length`` characters."""

    def __init__(self, max_length: int = 1_048_576, seed=None):
        self.max_length = int(max_length)
        super().__init__(shape=None, dtype=str, seed=seed)

    @property
    def is_np_flattenable(self):
        return False

    def sample(self, mask=None, probability=None):
        return ""

    def contains(self, x):
        return isinstance(x, str) and len(x) <= self.max_length

    def __eq__(self, other):
        return type(other) is type(self) and other.max_length == self.max_length

    def __repr__(self):
        return f"Message(max_length={self.max_length})"


class ToolCall(Message):
    """A tool call ``{"name": <tool>, "arguments": {...}}`` as a JSON string. ``sample()`` is a well-formed call of a
    random tool with empty arguments: enough for an API check, not a policy."""

    def __init__(self, tool_names, max_length: int = 65_536, seed=None):
        self.tool_names = tuple(tool_names)
        super().__init__(max_length=max_length, seed=seed)

    def sample(self, mask=None, probability=None):
        name = self.tool_names[int(self.np_random.integers(len(self.tool_names)))]
        return json.dumps({"name": name, "arguments": {}})

    def __eq__(self, other):
        return type(other) is type(self) and other.tool_names == self.tool_names and other.max_length == self.max_length

    def __repr__(self):
        return f"ToolCall({list(self.tool_names)})"


class MRGymEnv(gym.Env):
    """One ``mrgym.Env`` behind the Gymnasium API. ``make_kwargs`` go to ``mrgym.make`` (``level``, ``obs``, ...)."""

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(self, env_id: str, render_mode: str | None = None, **make_kwargs):
        if render_mode not in (None, "ansi"):
            raise ValueError(f"render_mode must be None or 'ansi', not {render_mode!r}")
        if "seed" in make_kwargs:
            raise ValueError("pass the seed to reset(seed=...), not to gym.make")
        self.env_id, self.render_mode = env_id, render_mode
        self._env = mrgym.make(env_id, **make_kwargs)
        probe = mrgym.make(env_id, **dict(make_kwargs, seed=0))          # tool names of the id, read from a throw-away episode
        probe.reset()
        self.action_space = ToolCall([t["name"] for t in probe.tools("neutral")])
        probe.close()
        self.observation_space = Message()
        self._over = True

    @property
    def mrgym_env(self):
        """Trainer side: the wrapped ``mrgym.Env`` (state, transcript, scanner log)."""
        return self._env

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        world = int(seed) if seed is not None else int(self.np_random.integers(2**31 - 1))
        obs, info = self._env.reset(seed=world, options=options or None)
        self._over = False
        return obs.text, dict(plain(info), seed=world, tools=plain(self._env.tools("openai")))

    def step(self, action):
        if self._over:
            raise gym.error.ResetNeeded("the episode is over or was never started; call reset()")
        call = action
        if isinstance(action, str):
            try:
                call = json.loads(action)
            except ValueError:
                pass                                         # not JSON: mrgym answers with an error observation
        r = self._env.step(call)
        self._over = r.done
        obs = r.observation
        info = dict(plain(dict(r.info)), is_error=bool(obs.is_error), status=plain(obs.status))
        if obs.images:
            info["content"] = obs.content()
        return obs.text, float(r.reward), bool(r.terminated), bool(r.truncated), info

    def render(self):
        return self._env.render() if self.render_mode == "ansi" else None

    def close(self):
        self._env.close()
        self._over = True


def register_all() -> list[str]:
    """Register every id of ``mrgym.list_envs()`` with Gymnasium under the same id. Safe to call again after more ids
    were registered with mrgym. Returns the ids registered by this call."""
    new = []
    for i in mrgym.list_envs():
        if i in gym.registry:
            continue
        s = mrgym.spec(i)
        gym.register(id=i, entry_point=MRGymEnv, kwargs={"env_id": i}, max_episode_steps=None,
                     reward_threshold=s.reward_threshold, nondeterministic=s.nondeterministic,
                     disable_env_checker=True, order_enforce=False)
        new.append(i)
    return new


register_all()


def check(ids=None, verbose: bool = True) -> list[dict]:
    """``gymnasium.utils.env_checker.check_env`` (render and close checks included) on each id, plus the seeded-reset
    stream. Returns one row per id: ``dict(id, ok, warnings, error, distinct_worlds, reproducible)``."""
    from gymnasium.utils.env_checker import check_env
    register_all()
    rows = []
    for i in ids or mrgym.list_envs():
        caught, error = [], None
        original = gym.logger.warn
        gym.logger.warn = lambda msg, *a, **k: caught.append(str(msg))            # noqa: B023
        try:
            with warnings.catch_warnings(record=True) as py:
                warnings.simplefilter("always")
                env = gym.make(i)
                try:
                    check_env(env.unwrapped)
                except Exception as e:  # noqa: BLE001  the checker reports through exceptions
                    error = f"{type(e).__name__}: {e}"
                # the worlds of a seeded stream (``info["seed"]``: the world this adapter started; an episode id is random)
                a = [env.reset(seed=3)[1]["seed"]] + [env.reset()[1]["seed"] for _ in range(2)]
                b = [env.reset(seed=3)[1]["seed"]] + [env.reset()[1]["seed"] for _ in range(2)]
                env.close()
            caught += [f"{w.category.__name__}: {w.message}" for w in py]
        finally:
            gym.logger.warn = original
        row = dict(id=i, ok=error is None and not caught and len(set(a)) == 3 and a == b, warnings=caught, error=error,
                   distinct_worlds=len(set(a)), reproducible=a == b)
        rows.append(row)
        if verbose:
            print(f"{i} -> {'PASS' if error is None else 'FAIL ' + error[:300]} | warnings: {len(caught)} | "
                  f"reset(seed=3), reset(), reset(): {row['distinct_worlds']} distinct worlds, reproducible: {row['reproducible']}")
            for w in caught:
                print("    warning:", w[:300])
    return rows


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m mrgym.compat.gymnasium", description="Gymnasium check_env on every registered MR Gym id.")
    ap.add_argument("--check", action="store_true", help="run check_env on every registered id (or on --id)")
    ap.add_argument("--id", action="append", help="check only this id; repeatable")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if not a.check:
        print("\n".join(sorted(i for i in gym.registry if i in mrgym.registry)))
        return 0
    rows = check(a.id, verbose=not a.json)
    ok = sum(r["ok"] for r in rows)
    if a.json:
        print(json.dumps(dict(gymnasium=gym.__version__, mrgym=mrgym.__version__, engine=mrgym.ENGINE_ID, rows=rows), indent=1))
    else:
        print(f"gymnasium {gym.__version__} check_env: {ok} of {len(rows)} ids PASS with 0 warnings")
    return 0 if rows and ok == len(rows) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
