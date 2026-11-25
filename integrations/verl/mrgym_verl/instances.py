"""Which instance a rollout plays: dataset rows for verl, and the curriculum hook behind the training rows.

    python -m mrgym_verl.instances --out DIR --arms "Scanner/NoWrap-v1:2,3 Scanner/SNRSpec-v0:1" \
        --train-rows 2048 --dev "Scanner/NoWrap-v1:2,3 Scanner/SNRSpec-v0:1" --dev-seeds 0-31 \
        --test "Scanner/NoWrap-v1:2,3 Scanner/PhantomQA-v0:1" --test-seeds 0-199

writes ``train.parquet``, ``dev.parquet``, ``test.parquet`` (``--format jsonl`` for the simulation) and
``dataset.json``. A row is an index, never task content: the task statement and the tools come from ``env.reset()``
inside the agent loop, so a file never goes stale against the package.

Rows
    train   ``extra_info = {phase: "train", curriculum: true}``: a placeholder. The agent loop asks the hook which
            instance the group plays (``hook.assign(uid)``; verl gives the ``n`` rollouts of one prompt one ``uid``
            per step). Seeds of the public ``train`` split, issued by ``mrgym.curriculum`` from 10,000, never twice.
            With ``--static-train-seeds`` the rows name fixed instances instead (the uniform-sampling ablation).
    dev     ``{phase: "dev", env_id, level, seed, split: "dev"}``: public, never sampled in training (another split).
    test    ``{phase: "test", env_id, level, seed, split: "test-iid"}``: worlds from the secret. Building the rows
            needs no secret; playing them does (``MRGYM_SECRET_FILE`` on the node).

The hook is ``mrgym.curriculum.RowHook`` restricted to the given (environment, level) list, with its state written
to a file after every step so that a restarted job continues the sampler (and never issues a seed twice). verl's
agent loops run in several worker processes; there the hook lives in one named Ray actor (``client()`` creates or
finds it). Outside Ray (the simulation, the CPU dry run, tests) it is an object of this process.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import threading

import mrgym
from mrgym.agents.runner import parse_seeds
from mrgym.curriculum import Curriculum, RowHook, assert_disjoint, rows

from . import AGENT_NAME

ACTOR_NAME, ACTOR_NAMESPACE = "mrgym_curriculum", "mrgym"
STATE_SCHEMA = "mrgym-verl-curriculum/1"
ENV_VAR = "MRGYM_CURRICULUM"                 # path of the hook's configuration file (written by ``write_config``)


def parse_arms(text: str) -> list[tuple[str, int]]:
    """``"Scanner/NoWrap-v1:2,3 Scanner/SNRSpec-v0:1"`` -> [(id, level), ...]; ids and levels are checked against the registry."""
    arms = []
    for part in str(text).replace(";", " ").split():
        env_id, _, levels = part.partition(":")
        spec = mrgym.spec(env_id)
        for lv in ([int(x) for x in levels.split(",")] if levels else list(spec.levels)):
            if lv not in spec.levels:
                raise ValueError(f"{env_id} has no level {lv} (levels {list(spec.levels)})")
            if (env_id, lv) in arms:
                raise ValueError(f"{env_id} level {lv} is listed twice")
            arms.append((env_id, lv))
    if not arms:
        raise ValueError("no (environment, level) given")
    return arms


def arms_text(arms) -> str:
    by_env: dict = {}
    for env_id, lv in arms:
        by_env.setdefault(env_id, []).append(int(lv))
    return " ".join(f"{e}:{','.join(map(str, lvs))}" for e, lvs in by_env.items())


# ------------------------------------------------------------------ dataset rows
def _row(extra: dict, source: str, text: str) -> dict:
    return {"data_source": source, "agent_name": AGENT_NAME, "prompt": [{"role": "user", "content": text}], "ability": "mrgym",
            "reward_model": {"style": "rule", "ground_truth": ""}, "extra_info": extra}


def train_rows(n: int) -> list[dict]:
    return [_row(dict(phase="train", split="train", index=i, curriculum=True), "mrgym/train", f"training group {i}") for i in range(n)]


def static_rows(arms, seeds, split: str, phase: str) -> list[dict]:
    out = []
    for env_id, level in arms:
        for r in rows(env_id, [level], seeds, split):
            out.append(_row(dict(phase=phase, split=split, index=len(out), curriculum=False, env_id=env_id, level=int(level), seed=int(r["seed"])),
                            f"mrgym/{env_id}/L{level}/{split}", f"{phase} instance {len(out)}"))
    return out


def write_rows(data: list[dict], path: str) -> None:
    if path.endswith(".parquet"):
        import pandas as pd
        pd.DataFrame(data).to_parquet(path)
    else:
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")


def read_rows(path: str) -> list[dict]:
    if path.endswith(".parquet"):
        import pandas as pd
        return json.loads(pd.read_parquet(path).to_json(orient="records"))
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------ the hook
def write_config(path: str, arms, *, group_size: int, groups_per_step: int, state_path: str, mode: str = "bandit",
                 alpha: float = 0.5, tau: float = 0.5, floor: float = 0.10, seed: int = 0, first_seed: int = 10_000, dead=()) -> dict:
    cfg = dict(schema=STATE_SCHEMA, arms=[list(a) for a in arms], mode=mode, alpha=alpha, tau=tau, floor=floor, seed=int(seed),
               first_seed=int(first_seed), dead=[list(a) for a in dead], group_size=int(group_size),
               groups_per_step=int(groups_per_step), state_path=state_path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    return cfg


class Hook:
    """``RowHook`` plus persistence. ``assign(uid)`` -> ``[env_id, level, seed, step]`` (``step``: the training step the
    sampler is in, counted from 1); ``report(uid, signal, passed)`` once per finished rollout."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.curriculum = Curriculum([tuple(a) for a in cfg["arms"]], mode=cfg["mode"], alpha=cfg["alpha"], tau=cfg["tau"],
                                     floor=cfg["floor"], seed=cfg["seed"], first_seed=cfg["first_seed"], split="train",
                                     dead=[tuple(a) for a in cfg.get("dead") or []])
        self.restored = None
        path = cfg["state_path"]
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                state = json.load(f)
            self.curriculum.load_state_dict(state["curriculum"])
            self.restored = dict(steps=self.curriculum.steps, issued=self.curriculum.issued, dropped=self.curriculum.drop_pending())
        self.rows = RowHook(self.curriculum, cfg["group_size"], cfg["groups_per_step"])
        self._lock = threading.Lock()
        self._steps_saved = self.curriculum.steps

    def assign(self, uid: str) -> list:
        with self._lock:
            env_id, level, seed = self.rows.assign(uid)
            self._save(force=True)                                       # an issued seed is on disk before it is played
            return [env_id, int(level), int(seed), self.curriculum.steps + 1]

    def report(self, uid: str, signal: float, passed: bool) -> None:
        with self._lock:
            try:
                self.rows.report(uid, float(signal), bool(passed))
            except KeyError:                                              # a group of before a restart: never counted
                return
            self._save()

    def info(self) -> dict:
        with self._lock:
            return dict(steps=self.curriculum.steps, issued=self.curriculum.issued, restored=self.restored,
                        metrics=dict(self.rows.last_metrics), probs=[float(p) for p in self.curriculum.probs()])

    def _save(self, force: bool = False) -> None:
        path = self.cfg["state_path"]
        if not path or (not force and self.curriculum.steps == self._steps_saved):
            return
        self._steps_saved = self.curriculum.steps
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dict(schema=STATE_SCHEMA, curriculum=self.curriculum.state_dict(), metrics=self.rows.last_metrics), f)
        os.replace(tmp, path)


def snapshot_state(checkpoint_dir: str, config_path: str | None = None) -> bool:
    """Copy the hook's state file into a checkpoint folder (``curriculum_state.json``): a checkpoint carries its sampler."""
    path = config_path or os.environ.get(ENV_VAR)
    if not path or not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        state = json.load(f).get("state_path")
    if not state or not os.path.exists(state):
        return False
    shutil.copyfile(state, os.path.join(checkpoint_dir, "curriculum_state.json"))
    return True


_LOCAL: dict = {}


class _Client:
    """One calling convention for the two homes of the hook: ``await client.assign(uid)``, ``await client.report(...)``."""

    def __init__(self, target, remote: bool):
        self._t, self.remote = target, remote

    async def assign(self, uid: str) -> list:
        return list(await self._t.assign.remote(uid)) if self.remote else self._t.assign(uid)

    async def report(self, uid: str, signal: float, passed: bool) -> None:
        if self.remote:
            await self._t.report.remote(uid, signal, passed)
        else:
            self._t.report(uid, signal, passed)

    async def info(self) -> dict:
        return dict(await self._t.info.remote()) if self.remote else self._t.info()


def client(config_path: str | None = None) -> _Client:
    """The hook of this run. Inside an initialised Ray session: the named actor ``mrgym_curriculum`` (created by the
    first caller, detached, so that every agent-loop worker talks to the same sampler). Otherwise: one object per
    process and configuration file."""
    path = config_path or os.environ.get(ENV_VAR)
    if not path:
        raise RuntimeError(f"a curriculum row needs the hook's configuration ({ENV_VAR}=<file>)")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    try:
        import ray
        in_ray = ray.is_initialized()
    except ImportError:
        in_ray = False
    key = (os.path.abspath(path), in_ray)
    if key not in _LOCAL:                                                # one handle per process: asked once, used for every episode
        if in_ray:
            actor = ray.remote(Hook).options(name=ACTOR_NAME, namespace=ACTOR_NAMESPACE, get_if_exists=True, lifetime="detached",
                                             num_cpus=0, max_restarts=-1).remote(cfg)
            _LOCAL[key] = _Client(actor, True)
        else:
            _LOCAL[key] = _Client(Hook(cfg), False)
    return _LOCAL[key]


# ------------------------------------------------------------------ command line
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", required=True, help='training cells: "ID:levels ID:levels"')
    ap.add_argument("--train-rows", type=int, default=4096, help="placeholder rows (at least prompts per step x steps)")
    ap.add_argument("--static-train-seeds", default="", help="fixed training instances per cell instead of the curriculum")
    ap.add_argument("--dev", default="", help="cells of the in-training evaluation (default: the training cells)")
    ap.add_argument("--dev-seeds", default="0-31")
    ap.add_argument("--test", default="", help="cells of the final readout, split test-iid; may name environments never trained on")
    ap.add_argument("--test-seeds", default="0-199")
    ap.add_argument("--format", choices=("parquet", "jsonl"), default="parquet")
    a = ap.parse_args(argv)
    arms = parse_arms(a.arms)
    os.makedirs(a.out, exist_ok=True)
    if a.static_train_seeds:
        train = static_rows(arms, parse_seeds(a.static_train_seeds), "train", "train")
    else:
        train = train_rows(a.train_rows)
    dev = static_rows(parse_arms(a.dev) if a.dev else arms, parse_seeds(a.dev_seeds), "dev", "dev")
    test = static_rows(parse_arms(a.test), parse_seeds(a.test_seeds), "test-iid", "test") if a.test else []
    key = lambda r: dict(env_id=r["extra_info"]["env_id"], level=r["extra_info"]["level"], split=r["extra_info"]["split"],   # noqa: E731
                         seed=r["extra_info"]["seed"])
    assert_disjoint(*[[key(r) for r in rs if not r["extra_info"]["curriculum"]] for rs in (train, dev, test)])
    summary = dict(engine=mrgym.ENGINE_ID, package=mrgym.__version__, arms=arms_text(arms), files={})
    for name, data in (("train", train), ("dev", dev), ("test", test)):
        if data:
            path = os.path.join(a.out, f"{name}.{a.format}")
            write_rows(data, path)
            summary["files"][name] = dict(path=path, rows=len(data), cells=sorted({r["data_source"] for r in data}))
    with open(os.path.join(a.out, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
