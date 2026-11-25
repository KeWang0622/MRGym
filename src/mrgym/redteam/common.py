"""What the rows of the red-team suite share: the family adapter, the generic attack policies, the worker.

Everything here plays through ``mrgym.make`` / ``env.step`` and ``policies.ToolScanner``: the same surface as an agent.
The only family hooks used are the ones of ``envs/family.py`` (``controls.policies``, ``REFERENCE``, ``MUST_FAIL``,
``prior``, ``tune``; ``task.make_instance``).
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import json
import multiprocessing
import os

import numpy as np

JUNK_PROTOCOLS = (dict(sequence="spgr", tr_ms=10.0, flip_deg=8.0, matrix=32), dict(sequence="spgr", tr_ms=10.0, flip_deg=8.0, matrix=48),
                  dict(sequence="spgr", tr_ms=10.0, flip_deg=8.0, matrix=64), dict(sequence="spgr", tr_ms=20.0, flip_deg=10.0))
HUGE_PROTOCOL = dict(sequence="se", tr_ms=5000.0, matrix=256, averages=16)      # far beyond any budget


class Context:
    def __init__(self, env_id: str, level: int | None = None, n: int = 50, split: str = "dev", workers: int = 2):
        import mrgym
        self.env_id, self.spec = env_id, mrgym.spec(env_id)
        if not self.spec.family:
            raise mrgym.Error(f"{env_id} is a sandbox: it pays no reward, so there is nothing to attack")
        self.level = self.spec.default_level if level is None else level
        self.n, self.split, self.workers = int(n), split, max(1, min(int(workers), 2))
        self.controls = importlib.import_module(self.spec.family + ".controls")
        self.task = importlib.import_module(self.spec.family + ".task")
        self.reference = getattr(self.controls, "REFERENCE", "reference")
        self.must_fail = tuple(getattr(self.controls, "MUST_FAIL", ()))
        self.family_style = hasattr(self.controls, "REFERENCE") and hasattr(self.task, "check")
        self._ref, self._recipe, self.episodes = {}, None, []

    # ------------------------------------------------------------------ episodes
    def make(self, seed, split=None, **kw):
        import mrgym
        return mrgym.make(self.env_id, level=self.level, seed=seed, split=split or self.split, **kw)

    def jobs(self, kind, seeds, split=None, **payload):
        return [(self.env_id, self.level, split or self.split, s, kind, payload) for s in seeds]

    def map(self, jobs):
        """Run worker jobs, in at most two processes. Returns a list of ``dict(passed, reward, answer, used_s, error)``."""
        if self.workers == 1 or len(jobs) < 4:
            out = [job(j) for j in jobs]
        else:
            with multiprocessing.get_context("fork").Pool(self.workers) as pool:
                out = pool.map(job, jobs, chunksize=1)
        self.episodes += [dict(passed=r["passed"], reward=r["reward"]) for r in out]
        return out

    def reference_runs(self, seeds, split=None):
        """The reference policy on the instances ``seeds`` (cached): {seed: dict(passed, answer, ...)}."""
        split = split or self.split
        todo = [s for s in seeds if (split, s) not in self._ref]
        for s, r in zip(todo, self.map(self.jobs("policy", todo, split, name=self.reference))):
            self._ref[(split, s)] = r
        return {s: self._ref[(split, s)] for s in seeds}

    def recipe(self, worlds: int = 16):
        """The best answer that ignores the object, tuned on worlds of the ``tune`` split that no row is measured on."""
        if self._recipe is None:
            from .. import keys
            from ..env import derive_seed
            cfgs = []
            for i in range(worlds):
                seed_hex = derive_seed(self.env_id, self.level, "tune", f"rt:{i}", environ={})
                cfgs.append(keys.make_instance(self.task, seed_hex, self.level, keys.task_id_for(self.spec, seed_hex, "tune"), "tune"))
            self._recipe = self.controls.tune(cfgs)
        return self._recipe


def passed_of(spec, step) -> bool:
    s = step.info.get("score") or {}
    return bool(s["passed"]) if isinstance(s.get("passed"), bool) else bool(step.reward >= spec.reward_threshold)


def terminal_name(env) -> str:
    return next(t.name for t in env._tool_specs() if t.terminal)


def submit_schema(env) -> dict:
    return next(t.parameters for t in env._tool_specs() if t.terminal)


PROBE_ROI = [dict(id="probe", y_mm=0.0, x_mm=0.0, radius_mm=30.0)]


def junk_scan(sc, rois=None):
    """The cheapest scan the console accepts: it satisfies "the log shows a measurement" and measures nothing useful.
    Returns the scan's response, or None when no candidate protocol fits."""
    for p in JUNK_PROTOCOLS:
        pl = sc.plan(p)
        if pl.get("accept") and pl["scan_s"] <= sc.budget()["remaining_s"]:
            return sc.scan(p, rois or [])
    return None


def fingerprint(response) -> str | None:
    """What a scan tells about which world it came from: its read-out of a fixed circle (the same world, protocol and
    scan index give the same numbers, noise included)."""
    return json.dumps(response["roi_readouts"], sort_keys=True) if response and response.get("roi_readouts") else None


def leaves(obj, path=()):
    """[(path, value)] of the scalar leaves of a JSON value, in order."""
    if isinstance(obj, dict):
        return [x for k, v in obj.items() for x in leaves(v, path + (k,))]
    if isinstance(obj, list):
        return [x for i, v in enumerate(obj) for x in leaves(v, path + (i,))]
    return [(path, obj)]


def with_leaf(obj, path, value):
    out = copy.deepcopy(obj)
    node = out
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return out


def transplant(answer, template):
    """``answer`` (of another instance) in the shape of ``template`` (this instance): numeric leaves in order, cycled."""
    if isinstance(answer, dict) and isinstance(template, dict) and [p for p, _ in leaves(answer)] == [p for p, _ in leaves(template)]:
        return copy.deepcopy(answer)
    nums = [v for _, v in leaves(answer) if isinstance(v, (int, float)) and not isinstance(v, bool)] or [1.0]
    out, k = copy.deepcopy(template), 0
    for path, v in leaves(template):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out = with_leaf(out, path, nums[k % len(nums)])
            k += 1
    return out


def random_answer(schema: dict, rng):
    """A random value of a JSON Schema (the subset the deliverables use)."""
    if "enum" in schema:
        return schema["enum"][int(rng.integers(len(schema["enum"])))]
    t = schema.get("type")
    if t == "object":
        req = set(schema.get("required", ()))
        return {k: random_answer(v, rng) for k, v in schema.get("properties", {}).items() if k in req or rng.random() < 0.5}
    if t in ("number", "integer"):
        lo, hi = float(schema.get("minimum", 1.0)), float(schema.get("maximum", 1000.0))
        v = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if lo > 0 else float(rng.uniform(lo, hi))
        return int(round(v)) if t == "integer" else round(v, 3)
    if t == "boolean":
        return bool(rng.integers(2))
    if t == "array":
        return []
    return "x"


@contextlib.contextmanager
def environ(**values):
    """Set (or, with None, remove) environment variables for a block."""
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ------------------------------------------------------------------------------------------------ the worker
def job(args) -> dict:
    """One episode of one attack. ``kind``:
    policy          a named control policy (``name``, optional ``recipe``)
    junk_prior      the cheapest scan, then the family's no-scan answer
    noscan_answer   ``answer`` delivered without any scan
    junk_answer     the cheapest scan, then ``answer`` put into this instance's shape
    random          the cheapest scan, then a random value of the deliverable's schema
    lookup          the seed-enumeration attack: match the public task card and, where cards do not tell worlds apart,
                    the read-out of one junk scan against ``table`` ({card: [[world seed, fingerprint or None], ...]});
                    on a match rebuild the world, run the reference on the copy and deliver its answer
    """
    import mrgym

    from .. import policies
    env_id, level, split, seed, kind, payload = args
    env = mrgym.make(env_id, level=level, seed=seed, split=split)
    spec, controls = env.spec, policies.controls_of(env)
    extra = {}
    if kind == "policy":
        policy = payload["name"]
    elif kind == "junk_prior":
        def policy(sc):
            junk_scan(sc)
            return controls.prior(sc)
    elif kind == "noscan_answer":
        def policy(sc):
            return transplant(payload["answer"], controls.prior(sc))
    elif kind == "junk_answer":
        def policy(sc):
            junk_scan(sc)
            return transplant(payload["answer"], controls.prior(sc))
    elif kind == "random":
        def policy(sc):
            junk_scan(sc)
            return random_answer(submit_schema(env), np.random.default_rng([payload.get("stream", 0), int(hashlib.sha256(str(seed).encode()).hexdigest()[:8], 16)]))
    elif kind == "lookup":
        def policy(sc):
            card = json.dumps(sc.info()["task"], sort_keys=True)
            seen = fingerprint(junk_scan(sc, PROBE_ROI))
            found = next((seed_hex for seed_hex, fp in payload["table"].get(card, ()) if fp is None or fp == seen), None)
            extra["claimed"] = found
            if not found:
                return controls.prior(sc)
            from ..scanner import session as core
            task = importlib.import_module(spec.family + ".task")
            copy_cfg = task.make_instance(found, level)
            ref = controls.policies()[getattr(controls, "REFERENCE", "reference")]
            ref = ref[0] if isinstance(ref, tuple) else ref
            return ref(core.LocalScanner(core.ScannerSession(copy_cfg)))
    else:
        raise ValueError(kind)
    step = policies.run(env, policy, recipe=payload.get("recipe"))
    call = env.transcript()[-1]["call"] or {}
    out = dict(seed=seed, passed=passed_of(spec, step), reward=float(step.reward), answer=call.get("arguments"),
               used_s=(step.info.get("budget") or {}).get("used_s"), error=step.info.get("policy_error"),
               task_id=env.episode_config().get("task_id"), world_seed=env.episode_config()["seed"], **extra)
    env.close()
    return out
