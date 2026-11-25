"""A GPU-free stand-in for ``python -m verl.trainer.main_ppo``: same inputs, same files, a one-parameter policy.

    MRGYM_STUB=1 bash integrations/verl/launch.sh          # launch.sh starts this module instead of verl

It exists to exercise everything around the trainer on a machine without a GPU or a model: dataset rows, the
curriculum hook, the adapter's episode loop and reward, the per-episode records and log, checkpoints leaving the
node, restore after a kill, in-training evaluation, evaluation-only runs, the stop rules. It reads the environment
variables of ``overrides.sh`` and prints verl-like lines (``step:N - ...``, ``Resuming from ...``).

What stands in for what
    model server      ``mrgym_verl.mixstub`` (model id ``mix:<p>:<q>``) behind ``SIM_BASE_URLS``, over chat completions.
                      verl talks to vLLM in token ids instead; that path is covered by ``mrgym_verl.dryrun``.
    policy            one logit ``theta``; p = sigmoid(theta) is the probability of following the reference procedure
    policy update     REINFORCE on that logit with verl's GRPO advantage (group mean and standard deviation; a group
                      without spread gets 0): ``theta += lr * mean(advantage * (a - p))``, ``a`` = 1 when the episode
                      followed the reference. A real gradient step of a real (tiny) policy on the real reward.
    checkpoint        ``global_step_N/sim_policy.json`` + ``latest_checkpointed_iteration.txt``, published with the
                      same function the verl checkpoint callback uses (``store.publish_checkpoint``)

``SIM_DIE_AT_STEP=N`` kills this process with SIGKILL after the rollouts of step N and before its update: a
preemption. ``SIM_FORCE_CUT=1`` ends every episode at a first reply cut by the token limit (for showing that the
checks name that cause). Nothing printed here is a result about any model.
"""
from __future__ import annotations

import concurrent.futures
import json
import math
import os
import signal
import sys
import time
import uuid

import numpy as np

from mrgym.agents import chat

from . import instances, stoprules, store
from .episode import Episode, episode_dir, free_dir, play_chat


def env(name: str, default=None, cast=str):
    v = os.environ.get(name)
    return default if v in (None, "") else cast(v)


def grpo_advantage(signals: list[float], eps: float = 1e-6) -> list[float]:
    """verl's ``compute_grpo_outcome_advantage`` for one group (``norm_adv_by_std_in_grpo=True``)."""
    r = np.asarray(signals, float)
    if len(r) < 2:
        return [0.0] * len(r)
    return list((r - r.mean()) / (r.std(ddof=1) + eps))


class Sim:
    def __init__(self):
        self.total, self.save_freq, self.test_freq = env("TOTAL_STEPS", 4, int), env("SAVE_FREQ", 2, int), env("TEST_FREQ", 2, int)
        self.prompts, self.group, self.val_n = env("PROMPTS_PER_STEP", 4, int), env("GROUP_SIZE", 8, int), env("VAL_N", 2, int)
        self.signal, self.lr = env("SIGNAL", "pass"), env("SIM_LR", 1.0, float)
        self.max_tokens, self.limit = env("MAX_TOKENS_PER_TURN", 4096, int), env("TOOL_CALL_LIMIT", None, int)
        self.data, self.ckpt, self.records = env("DATA_DIR"), env("CKPT_DIR"), env("RECORD_DIR")
        self.dest, self.val_only = env("MRGYM_CKPT_DEST"), env("VAL_ONLY", "0") == "1"
        self.val_file, self.val_phase = env("VAL_FILE", "dev"), env("VAL_PHASE", env("VAL_FILE", "dev"))
        self.urls = env("SIM_BASE_URLS", "http://127.0.0.1:8000/v1").split(",")
        self.q, self.workers = env("SIM_IDLE_SHARE", 0.1, float), min(2, env("SIM_WORKERS", 2, int))
        self.resume = env("RESUME_MODE", "auto")
        self.die_at = env("SIM_DIE_AT_STEP", None, int)
        self.force_cut = env("SIM_FORCE_CUT", "0") == "1"
        self.theta, self.step = env("SIM_THETA0", -0.4, float), 0
        self.log = os.path.join(self.records, "episodes.jsonl")
        self.train_rows = instances.read_rows(os.path.join(self.data, "train.jsonl")) if not self.val_only else []
        self.val_rows = instances.read_rows(os.path.join(self.data, f"{self.val_file}.jsonl"))
        self.hook = None
        self._n = 0

    @property
    def p(self) -> float:
        return 1.0 / (1.0 + math.exp(-self.theta))

    # ------------------------------------------------------------------ checkpoint
    def load(self) -> None:
        tracker = os.path.join(self.ckpt, "latest_checkpointed_iteration.txt")
        if self.resume == "disable" or not os.path.exists(tracker):
            return
        with open(tracker, encoding="utf-8") as f:
            step = int(f.read().strip())
        folder = os.path.join(self.ckpt, f"global_step_{step}")
        with open(os.path.join(folder, "sim_policy.json"), encoding="utf-8") as f:
            state = json.load(f)
        self.theta, self.step = float(state["theta"]), int(state["global_step"])
        print(f"Resuming from {folder}, setting global step to {self.step}", flush=True)

    def save(self) -> None:
        folder = os.path.join(self.ckpt, f"global_step_{self.step}")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "sim_policy.json"), "w", encoding="utf-8") as f:
            json.dump(dict(theta=self.theta, global_step=self.step, p=self.p), f)
        with open(os.path.join(folder, "data.pt"), "wb") as f:
            f.write(b"sim")
        with open(os.path.join(self.ckpt, "latest_checkpointed_iteration.txt"), "w", encoding="utf-8") as f:
            f.write(str(self.step))
        instances.snapshot_state(folder)
        if self.dest:
            latest = store.publish_checkpoint(folder, self.dest, self.step)
            print(f"[mrgym] step {self.step}: checkpoint copied to {self.dest} ({latest['files']} file(s))", flush=True)

    # ------------------------------------------------------------------ rollouts
    def _play(self, job: dict) -> dict:
        self._n += 1
        ep = Episode(job["env_id"], job["level"], job["seed"], job["split"], tool_call_limit=self.limit,
                     max_reply_tokens=self.max_tokens, signal=self.signal)
        model = f"mix:{job['p']:.4f}:{self.q:.4f}"
        client = chat.ChatClient(self.urls[job["k"] % len(self.urls)], model, None, dict(temperature=1.0, max_tokens=self.max_tokens),
                                 backoff_s=0.05, max_retries=3)
        try:
            if self.force_cut:                                           # every first reply is cut by the token limit
                ep.start()
                ep.abort("length", reply=dict(raw_text="(cut)", prompt_tokens=0, completion_tokens=self.max_tokens))
                result = ep.result()
            else:
                result = play_chat(ep, client)
            agent = env("RUN_NAME", "sim")
            out = os.path.join(self.records, job["phase"], f"step_{job['step']:06d}")
            training = dict(phase=job["phase"], step=job["step"], step_source="trainer", uid=job["uid"], session=job["session"],
                            attempt=os.environ.get("MRGYM_ATTEMPT", "0"), policy_p=job["p"])
            ep.write_record(free_dir(episode_dir(out, agent, job["env_id"], job["level"], job["seed"], job["session"]), job["uid"]), agent=agent, model=model,
                            sampler=dict(temperature=1.0, max_tokens=self.max_tokens), episode=job["session"],
                            base_url=client.url, training=training, command="python -m mrgym_verl.sim (integrations/verl/launch.sh, MRGYM_STUB=1)")
            followed = [t["text"] for t in ep.turn_log if (t["text"] or "").startswith("policy=")]
            result["action"] = None if not followed or followed[0] == "policy=idle" else float(followed[0] == "policy=reference")
            stoprules.append(self.log, stoprules.line(result, phase=job["phase"], step=job["step"], step_source="trainer", uid=job["uid"],
                                                      session=job["session"], context_tokens=None))
            return result
        finally:
            ep.close()

    def _run(self, jobs: list[dict]) -> list[dict]:
        with concurrent.futures.ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self._play, jobs))

    def validate(self) -> dict:
        jobs, k = [], 0
        for row in self.val_rows:
            x, uid = row["extra_info"], uuid.uuid4().hex
            for s in range(self.val_n):
                jobs.append(dict(env_id=x["env_id"], level=x["level"], seed=x["seed"], split=x["split"], phase=x.get("phase") or self.val_phase,
                                 step=self.step, uid=uid, session=s, p=self.p, k=k))
                k += 1
        results = self._run(jobs)
        cells: dict = {}
        for j, r in zip(jobs, results):
            cells.setdefault(f"mrgym/{j['env_id']}/L{j['level']}/{j['split']}", []).append(float(r["passed"]))
        metrics = {f"val-core/{c}/acc/mean@{self.val_n}": float(np.mean(v)) for c, v in sorted(cells.items())}
        print(f"step:{self.step} - " + " - ".join(f"{k}:{v:.4f}" for k, v in metrics.items()), flush=True)
        return metrics

    async def _assign(self, uid):
        return await self.hook.assign(uid)

    def train_step(self) -> None:
        import asyncio
        t0 = time.time()
        step = self.step + 1
        jobs, groups, k = [], [], 0
        for i in range(self.prompts):
            row = self.train_rows[((step - 1) * self.prompts + i) % len(self.train_rows)]["extra_info"]
            uid = uuid.uuid4().hex
            if row.get("curriculum"):
                if self.hook is None:
                    self.hook = instances.client()
                env_id, level, seed, _ = asyncio.run(self._assign(uid))
            else:
                env_id, level, seed = row["env_id"], row["level"], row["seed"]
            groups.append(uid)
            for s in range(self.group):
                jobs.append(dict(env_id=env_id, level=level, seed=seed, split="train", phase="train", step=step, uid=uid, session=s,
                                 p=self.p, k=k))
                k += 1
        results = self._run(jobs)
        if self.die_at == step:
            print(f"[sim] SIM_DIE_AT_STEP={step}: killed after the rollouts of step {step}, before the update", flush=True)
            sys.stdout.flush()
            os.kill(os.getpid(), signal.SIGKILL)
        grad, n, flat = 0.0, 0, 0
        for uid in groups:
            members = [(j, r) for j, r in zip(jobs, results) if j["uid"] == uid]
            adv = grpo_advantage([r["signal"] for _, r in members])
            flat += all(a == 0.0 for a in adv) or len({r["signal"] for _, r in members}) == 1
            for (j, r), a in zip(members, adv):
                if self.hook is not None:
                    asyncio.run(self.hook.report(uid, r["signal"], r["passed"]))
                if r["action"] is not None:
                    grad += a * (r["action"] - j["p"])
                    n += 1
        update = self.lr * grad / max(1, n)
        self.theta += update
        self.step = step
        print(f"step:{step} - critic/score/mean:{np.mean([r['signal'] for r in results]):.4f} - mrgym/pass:{np.mean([r['passed'] for r in results]):.4f}"
              f" - mrgym/groups_without_spread:{flat} - actor/grad_norm:{abs(grad / max(1, n)):.4f} - sim/p:{self.p:.4f}"
              f" - timing_s/step:{time.time() - t0:.1f}", flush=True)

    def fit(self) -> int:
        os.makedirs(self.ckpt, exist_ok=True)
        self.load()
        if env("VAL_BEFORE_TRAIN", "1") == "1" or self.val_only:
            self.validate()
        if self.val_only:
            return 0
        while self.step < self.total:
            self.train_step()
            if self.step % self.save_freq == 0 or self.step == self.total:
                self.save()
            if self.step % self.test_freq == 0 or self.step == self.total:
                self.validate()
        return 0


if __name__ == "__main__":
    raise SystemExit(Sim().fit())
