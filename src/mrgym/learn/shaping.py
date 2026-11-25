"""Optional step rewards for trainers with per-turn credit, built so that no policy can earn anything from them.

    env = Shaped(mrgym.make("Scanner/NullTissue-v0", level=2, seed=0))
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(call)       # reward = environment reward + F

F is potential-based (Ng, Harada and Russell 1999): ``F = phi(state after) - phi(state before)``, and the last step of
an episode, however it ends, pays back whatever shaping was paid before it. The step rewards of an episode therefore
sum to the environment's terminal reward for every policy: scanning, viewing or repeating calls earns nothing that
is not taken back. Consequences:

  * A trainer that uses the return of the whole episode (group-normalised GRPO over episodes) is unchanged by this
    wrapper. It only matters for estimators with per-turn credit (returns-to-go, GAE with discount < 1).
  * A step bonus that is not of this form (paid and never taken back) is paid without the outcome. This package ships
    none, and ``tests/test_training_api.py`` checks the sum for every way an episode can end.

The potential comes from the family (``task.potential(cfg, scanner log events) -> float``, trainer side: it may read
the log, never the hidden world); without one, it is 0.1 once a scan has been accepted. ``info["shaping"]`` carries
``{potential, step_reward}`` at every step, and ``env.rewards`` lists what was paid in the episode, step by step.

An episode that the harness ends without a terminal step (``note_turn`` at the cap, ``end_without_submission``) has no
step to carry the pay-back: the returned info holds it as ``shaping_reward``, to be added to that episode's rewards.
"""
from __future__ import annotations

from ..env import StepResult


def default_potential(cfg: dict, events: list) -> float:
    return 0.1 if any(e["type"] == "scan" for e in events) else 0.0


class Shaped:
    """Wraps an environment; everything not listed here is the environment's own attribute."""

    def __init__(self, env, potential=None):
        self.env = env
        task = getattr(env, "task", None)
        self._phi = potential or getattr(task, "potential", None) or default_potential
        self._last = self._paid = 0.0
        self.rewards = []                                                # everything paid in the running episode, in order

    def __getattr__(self, name):
        return getattr(self.env, name)

    def potential(self) -> float:
        return float(self._phi(self.env.episode_config(), self.env.scanner_log()))

    def reset(self, **kw):
        out = self.env.reset(**kw)
        self._last, self._paid, self.rewards = self.potential(), 0.0, []  # scans a level acquires at reset earn nothing
        return out

    def _settle(self, reward: float = 0.0) -> float:
        back, self._paid = -self._paid, 0.0
        self.rewards.append(reward + back)
        return back

    def step(self, tool_call: dict) -> StepResult:
        r = self.env.step(tool_call)
        if r.done:
            f, phi = self._settle(r.reward), self._last
        else:
            phi = self.potential()
            f, self._last = phi - self._last, phi
            self._paid += f
            self.rewards.append(r.reward + f)
        info = dict(r.info, shaping=dict(potential=phi, step_reward=f))
        return StepResult((r.observation, r.reward + f, r.terminated, r.truncated, info))

    def note_turn(self) -> dict:
        out = self.env.note_turn()
        return dict(out, shaping_reward=self._settle()) if out.get("truncated") else out

    def end_without_submission(self, reason: str = "stopped") -> dict:
        return dict(self.env.end_without_submission(reason), shaping_reward=self._settle())
