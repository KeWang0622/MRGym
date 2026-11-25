"""Frozen instance lists: which instances a suite holds, written once and committed.

    mrgym manifest list                       the suites
    mrgym manifest report                     the rows of a suite, one JSON object per line
    mrgym manifest report --write             write results/manifests/report.json
    mrgym manifest report --check             exit 1 if the stored file differs from a fresh build

A suite is a split, a seed prefix and a number of instances per environment, divided equally over the levels (the
strata of a results row). A row is ``{env_id, level, split, seed, task_id}``; the seed is the user seed of
``mrgym.make`` (``report:17``), never the world seed. ``seed_set_sha256`` per environment is the hash a results row
carries, so "this row played exactly the reporting instances" is one string comparison.

    report        300 instances per environment on the public split, seeds ``report:<i>``: the published baseline
                  table. Public seeds make the table reproducible by anyone; they are burned for tuning, training and
                  checkpoint selection.
    validation    200 instances per environment, seeds ``val:<i>``: checkpoint selection, never reported.
    public-test   100 instances per environment on a secret split (``test-iid``): needs the secret to build the worlds;
                  without it the manifest lists the seeds and omits the task ids.
    smoke         12 instances per environment, seeds ``smoke:<i>``: a quick end-to-end run.

The instance lists used by the gate (``gate:<i>``) and for tuning (``tune:<set>:<i>``) are disjoint from all of these
by their prefix.
"""
from __future__ import annotations

import json
import os

from .agents.summarize import seed_set_sha256
from .canary import CANARY

SCHEMA = "mrgym-manifest/1"
DEFAULT_DIR = os.path.join("results", "manifests")
SUITES = {
    "report": dict(split="train", prefix="report", per_env=300, use="the published baseline table"),
    "validation": dict(split="train", prefix="val", per_env=200, use="checkpoint selection; never reported"),
    "public-test": dict(split="test-iid", prefix="test", per_env=100, use="scored runs on a secret split"),
    "smoke": dict(split="train", prefix="smoke", per_env=12, use="a quick end-to-end run"),
}


def default_envs() -> list[str]:
    """Scored environments that are admitted or candidates (a practice id is not reported)."""
    import mrgym
    return [i for i in mrgym.list_envs() if mrgym.spec(i).family and mrgym.spec(i).kind in ("admitted", "candidate")]


def _allocation(per_env: int, levels) -> dict:
    """Instances per level: equal, the remainder to the lowest levels."""
    base, extra = divmod(per_env, len(levels))
    return {lv: base + (1 if i < extra else 0) for i, lv in enumerate(sorted(levels))}


def build(suite: str, env_ids=None, environ=None) -> dict:
    """The manifest of ``suite`` over ``env_ids`` (default: ``default_envs()``). Deterministic: two builds are equal."""
    import mrgym

    from . import gate, keys
    from .env import derive_seed
    if suite not in SUITES:
        raise mrgym.Error(f"unknown suite {suite!r}; suites: {', '.join(SUITES)}")
    s = SUITES[suite]
    env_ids = list(default_envs() if env_ids is None else env_ids)
    rows, envs, keyed = [], {}, True
    for env_id in env_ids:
        spec = mrgym.spec(env_id)
        alloc = _allocation(s["per_env"], gate.claim_levels(spec))
        part = []
        for level, n in alloc.items():
            for i in range(n):
                seed = f"{s['prefix']}:{i}"
                try:
                    task_id = keys.task_id_for(spec, derive_seed(env_id, level, s["split"], seed, environ=environ))
                except mrgym.Error:                                      # a secret split without the secret
                    task_id, keyed = None, False
                part.append(dict(env_id=env_id, level=level, split=s["split"], seed=seed, task_id=task_id))
        envs[env_id] = dict(levels={str(lv): n for lv, n in alloc.items()}, instances=len(part),
                            seed_set_sha256=seed_set_sha256(env_id, {(r["level"], r["split"], r["seed"]) for r in part}))
        rows += part
    return dict(schema=SCHEMA, canary=CANARY, suite=suite, use=s["use"], split=s["split"], seed_prefix=s["prefix"],
                task_ids="listed" if keyed else "omitted: the split needs the secret", instances=len(rows), environments=envs, rows=rows)


def text(manifest: dict) -> str:
    return json.dumps(manifest, indent=1) + "\n"


def path_of(suite: str, directory: str = DEFAULT_DIR) -> str:
    return os.path.join(directory, suite + ".json")


def load(suite: str, directory: str = DEFAULT_DIR) -> dict | None:
    try:
        with open(path_of(suite, directory), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def drift(suite: str, directory: str = DEFAULT_DIR) -> str | None:
    """Why the stored manifest of ``suite`` is not frozen: None when a fresh build over the environments it lists
    gives the same file."""
    stored = load(suite, directory)
    if stored is None:
        return f"{path_of(suite, directory)} does not exist"
    fresh = build(suite, list(stored["environments"]))
    return None if text(fresh) == text(stored) else f"{path_of(suite, directory)} differs from a fresh build"


def cmd_manifest(a) -> int:
    import mrgym
    if a.suite == "list":
        for name, s in SUITES.items():
            print(f"{name:<12} {s['per_env']:>4} instances per environment, split {s['split']}, seeds {s['prefix']}:<i>   {s['use']}")
        return 0
    try:
        envs = [e for part in a.env or [] for e in part.split(",") if e] or None
        if a.check:
            problem = drift(a.suite, a.dir)
            print(problem or f"{path_of(a.suite, a.dir)} is frozen: a fresh build gives the same file")
            return 1 if problem else 0
        m = build(a.suite, envs)
    except mrgym.Error as e:
        raise SystemExit(f"mrgym manifest: {e}") from None
    if a.write:
        os.makedirs(a.dir, exist_ok=True)
        with open(path_of(a.suite, a.dir), "w", encoding="utf-8") as f:
            f.write(text(m))
        print("written:", path_of(a.suite, a.dir), f"({m['instances']} instances)")
        return 0
    for row in m["rows"]:
        print(json.dumps(row))
    return 0


def add_parser(sub) -> None:
    """Register ``mrgym manifest``."""
    p = sub.add_parser("manifest", help="frozen instance lists of the suites (report, validation, public-test, smoke)")
    p.add_argument("suite", help="a suite name, or 'list'")
    p.add_argument("--env", action="append", metavar="IDS", help="environments (default: admitted and candidate ids)")
    p.add_argument("--dir", default=DEFAULT_DIR, help=f"folder of the stored manifests (default {DEFAULT_DIR})")
    p.add_argument("--write", action="store_true", help="write <dir>/<suite>.json")
    p.add_argument("--check", action="store_true", help="exit 1 if the stored manifest differs from a fresh build")
    p.set_defaults(fn=cmd_manifest)
