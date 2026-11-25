"""A curriculum sampler over (environment, level) cells. Numpy only; it knows no trainer and calls no model.

    cur = Curriculum([("Scanner/NullTissue-v0", 0), ("Scanner/NullTissue-v0", 1), ("Scanner/NullTissue-v0", 2)])
    rows = cur.sample(16)                      # 16 groups: [{env_id, level, seed, split, group_id}], a fresh seed each
    ...                                        # the trainer plays G episodes per row
    cur.update(row["group_id"], rewards)       # once per finished group: the G rewards (and, optionally, the passes)
    log = cur.end_step()                       # once per training step: moves the sampler, returns what to log
    ckpt["curriculum"] = cur.state_dict()      # goes into the trainer checkpoint; load_state_dict resumes exactly

Modes
    bandit    (default) learning progress as a non-stationary bandit. The reward of an arm in a step is the mean absolute
              group-normalised advantage of its groups (0 for a group without spread; 2 sqrt(p (1 - p)) in expectation
              for a pass / fail reward with pass rate p: largest at p = 0.5, zero for a level that is never or always
              passed). Q <- alpha * reward + (1 - alpha) * Q once per step for the arms that were played; arms are drawn
              with probability (1 - floor) * softmax(Q / tau) + floor / n_arms. The floor keeps every arm measured.
    ladder    promotion by threshold. Per environment the active level is the lowest one whose mean pass rate over its
              last ``window`` played steps is below ``promote`` (a level played in fewer than ``window`` steps counts as
              below); 80 % of that environment's groups come from it, 20 % are spread over the levels below it.
    uniform   every arm with the same probability: the ablation.

Seeds. Every group gets a seed that this sampler has never issued (a counter from ``first_seed``), so no instance is
trained on twice and none of them collides with evaluation seeds below ``first_seed``.

The sampler reads rewards only. It works the same with the environment's paid reward (pass / fail) and with a
trainer's own signal such as ``info["score"]["progress"]``; pass ``passed=`` as well so that the pass rate per arm is
logged from the paid reward in either case.
"""
from __future__ import annotations

import threading

import numpy as np

MODES = ("bandit", "ladder", "uniform")
SPREAD_MIN = 1e-3                       # a group whose rewards have a smaller standard deviation carries no gradient
DEAD_Q = -4.0                           # starting value of an arm that a calibration found dead: weight floor / n until it shows spread


def learning_progress(rewards) -> float:
    """Mean absolute group-normalised advantage of one group; 0 when the group has no spread."""
    r = np.asarray(rewards, float)
    sd = float(r.std())
    return 0.0 if sd < SPREAD_MIN else float(np.mean(np.abs(r - r.mean())) / sd)


class Curriculum:
    def __init__(self, arms, mode: str = "bandit", alpha: float = 0.5, tau: float = 0.5, floor: float = 0.10,
                 promote: float = 0.70, window: int = 20, seed: int = 0, first_seed: int = 10_000, split: str = "train",
                 dead=()):
        self.arms = [(str(e), int(lv)) for e, lv in arms]
        if not self.arms or len(set(self.arms)) != len(self.arms):
            raise ValueError("arms must be a non-empty list of distinct (env_id, level) pairs")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not (0.0 < alpha <= 1.0 and tau > 0.0 and 0.0 <= floor <= 1.0 and 0.0 < promote <= 1.0 and window >= 1):
            raise ValueError("alpha in (0, 1], tau > 0, floor in [0, 1], promote in (0, 1], window >= 1")
        self.mode, self.alpha, self.tau, self.floor, self.promote, self.window = mode, float(alpha), float(tau), float(floor), float(promote), int(window)
        self.split, self.first_seed = split, int(first_seed)
        dead = {(str(e), int(lv)) for e, lv in dead}
        self.Q = np.array([DEAD_Q if a in dead else 0.0 for a in self.arms])
        self.pass_hist = [[] for _ in self.arms]                         # mean pass (or reward) of the arm, one entry per played step
        self.issued, self.steps = 0, 0
        self._rng = np.random.default_rng(seed)
        self._pending = {}                                               # group_id -> arm index, until its rewards arrive
        self._acc = [[] for _ in self.arms]                              # this step: (learning progress, mean pass, mixed) per group
        self._last = {}

    # ------------------------------------------------------------------ sampling
    def _ladder_probs(self) -> np.ndarray:
        p = np.zeros(len(self.arms))
        envs = sorted({e for e, _ in self.arms})
        for env in envs:
            idx = sorted((i for i, a in enumerate(self.arms) if a[0] == env), key=lambda i: self.arms[i][1])
            active = next((k for k, i in enumerate(idx) if len(self.pass_hist[i]) < self.window
                           or float(np.mean(self.pass_hist[i][-self.window:])) < self.promote), len(idx) - 1)
            share = 1.0 / len(envs)
            if active == 0:
                p[idx[0]] += share
            else:
                p[idx[active]] += 0.8 * share
                for i in idx[:active]:
                    p[i] += 0.2 * share / active
        return p

    def probs(self) -> np.ndarray:
        """Probability of each arm for the next group, in the order of ``arms``."""
        n = len(self.arms)
        if self.mode == "uniform":
            return np.full(n, 1.0 / n)
        if self.mode == "ladder":
            return self._ladder_probs()
        z = (self.Q - self.Q.max()) / self.tau
        q = np.exp(z) / np.exp(z).sum()
        return (1.0 - self.floor) * q + self.floor / n

    def sample(self, n_groups: int) -> list[dict]:
        """Rows of ``n_groups`` groups: ``{env_id, level, seed, split, group_id}``. A seed is never issued twice."""
        picks = self._rng.choice(len(self.arms), size=int(n_groups), p=self.probs())
        rows = []
        for i in picks:
            env_id, level = self.arms[int(i)]
            seed = self.first_seed + self.issued
            self.issued += 1
            gid = f"{env_id}|L{level}|{self.split}|s{seed}"
            self._pending[gid] = int(i)
            rows.append(dict(env_id=env_id, level=level, seed=seed, split=self.split, group_id=gid))
        return rows

    # ------------------------------------------------------------------ feedback
    def update(self, group_id: str, rewards, passed=None) -> None:
        """The rewards of the members of one finished group (``passed``: their pass flags, when the rewards are not
        the paid reward). A group that never reports (an infrastructure failure) is simply never counted."""
        arm = self._pending.pop(group_id)                                # KeyError: unknown group or reported twice
        r = np.asarray(rewards, float)
        p = r if passed is None else np.asarray(passed, float)
        self._acc[arm].append((learning_progress(r), float(p.mean()), float(r.std()) >= SPREAD_MIN))

    def drop_pending(self) -> int:
        """Forget the groups that were issued and never reported, and the reports of the unfinished step: what a
        trainer calls after it restored this sampler from a checkpoint (the rollouts that were in flight when the
        process died will never report). Their seeds stay issued: no instance is ever given out twice. Returns the
        number of groups dropped."""
        n = len(self._pending) + sum(len(a) for a in self._acc)
        self._pending = {}
        self._acc = [[] for _ in self.arms]
        return n

    def end_step(self) -> dict:
        """Apply the groups reported since the last call (one training step) and return ``metrics()``."""
        self._last = {}
        for i, acc in enumerate(self._acc):
            if not acc:
                continue
            lp, ps, mixed = (np.array(x, float) for x in zip(*acc))
            self.Q[i] = self.alpha * float(lp.mean()) + (1.0 - self.alpha) * self.Q[i]
            self.pass_hist[i].append(float(ps.mean()))
            self._last[i] = dict(groups=len(acc), mixed_share=float(mixed.mean()), dropped=int(len(acc) - mixed.sum()), pass_rate=float(ps.mean()))
        self._acc = [[] for _ in self.arms]
        self.steps += 1
        return self.metrics()

    def metrics(self) -> dict:
        """Flat values for the trainer's log: per arm the sampling probability, Q, and for the last step the number of
        groups, their pass rate, the share with spread and the number dropped."""
        out, p = {"curriculum/step": self.steps, "curriculum/seeds_issued": self.issued}, self.probs()
        for i, (env_id, level) in enumerate(self.arms):
            name = f"{env_id}/L{level}"
            out[f"curriculum/p[{name}]"] = float(p[i])
            out[f"curriculum/Q[{name}]"] = float(self.Q[i])
            last = self._last.get(i, {})
            for k in ("groups", "pass_rate", "mixed_share", "dropped"):
                out[f"curriculum/{k}[{name}]"] = last.get(k, 0 if k in ("groups", "dropped") else None)
        return out

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict:
        """Everything needed to resume, as JSON-serialisable values."""
        return dict(schema="mrgym-curriculum/1", arms=[list(a) for a in self.arms], mode=self.mode, alpha=self.alpha, tau=self.tau,
                    floor=self.floor, promote=self.promote, window=self.window, split=self.split, first_seed=self.first_seed,
                    Q=self.Q.tolist(), pass_hist=[list(h) for h in self.pass_hist], issued=self.issued, steps=self.steps,
                    rng=self._rng.bit_generator.state, pending=dict(self._pending), acc=[list(map(list, a)) for a in self._acc])

    def load_state_dict(self, d: dict) -> None:
        if [tuple(a) for a in d["arms"]] != self.arms:
            raise ValueError("the checkpoint was written for other arms")
        for k in ("mode", "alpha", "tau", "floor", "promote", "window", "split", "first_seed", "issued", "steps"):
            setattr(self, k, d[k])
        self.Q = np.array(d["Q"], float)
        self.pass_hist = [list(h) for h in d["pass_hist"]]
        self._rng.bit_generator.state = d["rng"]
        self._pending = {k: int(v) for k, v in d["pending"].items()}
        self._acc = [[(float(a), float(b), bool(c)) for a, b, c in arm] for arm in d["acc"]]
        self._last = {}


class RowHook:
    """Adapter for a trainer that owns a static dataset (verl): a dataset row is a placeholder, and the agent loop asks
    which instance it stands for.

        env_id, level, seed = hook.assign(row_uid)        # cached: the G rollouts of one row share the instance
        hook.report(row_uid, reward, passed)              # once per finished rollout

    After ``group_size`` reports the row's group is passed to the sampler; after ``groups_per_step`` groups the step is
    closed (``Curriculum.end_step``) and its metrics are kept in ``last_metrics``. Thread-safe within one process. In a
    trainer whose agent loops run in several processes this object must live in one place (for verl: a single named
    Ray actor); that wiring is not part of this module.
    """

    def __init__(self, curriculum: Curriculum, group_size: int = 8, groups_per_step: int = 16):
        self.curriculum, self.group_size, self.groups_per_step = curriculum, int(group_size), int(groups_per_step)
        self._rows, self._reports, self._closed = {}, {}, 0
        self._lock = threading.Lock()
        self.last_metrics = {}

    def assign(self, row_uid) -> tuple[str, int, int]:
        with self._lock:
            if row_uid not in self._rows:
                self._rows[row_uid] = self.curriculum.sample(1)[0]
                self._reports[row_uid] = []
            r = self._rows[row_uid]
            return r["env_id"], r["level"], r["seed"]

    def report(self, row_uid, reward: float, passed: bool) -> None:
        with self._lock:
            got = self._reports[row_uid]
            got.append((float(reward), bool(passed)))
            if len(got) < self.group_size:
                return
            row = self._rows.pop(row_uid)
            del self._reports[row_uid]
            self.curriculum.update(row["group_id"], [g[0] for g in got], [g[1] for g in got])
            self._closed += 1
            if self._closed >= self.groups_per_step:
                self._closed = 0
                self.last_metrics = self.curriculum.end_step()


def rows(env_id: str, levels, seeds, split: str = "train") -> list[dict]:
    """Dataset rows of fixed instances (evaluation sets, a static training set): ``{env_id, level, seed, split, group_id}``."""
    return [dict(env_id=env_id, level=int(lv), seed=int(s), split=split, group_id=f"{env_id}|L{int(lv)}|{split}|s{int(s)}")
            for lv in levels for s in seeds]


def assert_disjoint(*row_sets) -> None:
    """Refuse row sets that share an instance (the same environment, level, split and seed)."""
    seen = {}
    for k, rs in enumerate(row_sets):
        for r in rs:
            key = (r["env_id"], r["level"], r["split"], r["seed"])
            if seen.setdefault(key, k) != k:
                raise ValueError(f"row sets {seen[key]} and {k} share the instance {key}")
