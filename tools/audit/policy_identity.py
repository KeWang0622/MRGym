"""Does a change move what a scripted policy earns? Play the control policies on two trees and compare episode by episode.

    PYTHONPATH=<tree>/src python tools/audit/policy_identity.py play OUT.jsonl [--ids ID ...] [--seeds 20] [--max-rss-gb 0.9]
    python tools/audit/policy_identity.py compare BASE.jsonl BRANCH.jsonl [--relabelled check_failed]
    python tools/audit/policy_identity.py relabel EPISODES_DIR [--ids ID ...]

``play`` runs the reference and every control policy of ``controls.policies()`` on ``--seeds`` integer seeds of every
level of the ids, through the tools, and writes one line per episode: reward, pass, progress, outcome, the checks in
their declared order, the delivered answer, and a hash of everything else the verifier returned (diagnostic, errors,
margins). It imports nothing but ``mrgym``, so the same file plays either tree: the tree is the one on ``PYTHONPATH``.
The tuned recipe of ``fixed_recipe`` is the family's default one (no tuning run): the same on both trees. A run
resumes: episodes already in ``OUT.jsonl`` are not played again. With ``--max-rss-gb`` the process stops when its
memory passes that size and exits 75; run it again until it exits 0 (``until play ...; do :; done`` bounds the memory).

``compare`` exits 1 unless both files hold the same episodes with the same reward, pass, progress, checks (names, order
and values), deliverable and verifier details. The outcome label may differ only where the base label is one of
``--relabelled`` (default ``check_failed``); such changes are counted per id and listed as ``from -> to``.

``relabel`` reads stored episode records (``<id>/L*/s*/e*/record.json`` with ``checks`` and ``outcome``, as
``mrgym calibrate`` writes them for one model) under a folder, recomputes the outcome of every delivered answer from
its stored checks with the tree on ``PYTHONPATH`` and prints, per id and level, how many of each stored label became
which label. Nothing is written.
"""
from __future__ import annotations

import argparse
import collections
import gc
import glob
import hashlib
import json
import os
import resource
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

EXIT_AGAIN = 75                               # play: stopped at the memory limit with episodes left; run again
DEFAULT_IDS = ("Scanner/SNRSpec-v0", "Scanner/BandFree-v0", "Scanner/ArtefactClinic-v0", "Scanner/ProtocolToSpec-v0")
COMPARED = ("reward", "passed", "progress", "checks", "deliverable", "details_sha256", "submitted", "policy_error")


def _plain(obj):
    """JSON-able copy: numpy scalars and arrays become Python numbers and lists, tuples lists, keys strings."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(_plain(obj), sort_keys=True).encode()).hexdigest()


def play_one(env_id: str, level: int, seed: int, policy: str) -> dict:
    import mrgym
    from mrgym import policies
    t0 = time.time()
    env = mrgym.make(env_id, level=level, seed=seed)
    delivered, step = [], env.step

    def recording_step(call):
        if call.get("name") == "submit":
            delivered.append(call.get("arguments"))
        return step(call)
    env.step = recording_step
    res = policies.run(env, policy)
    info = res.info
    if "outcome" not in info:                                            # the deliverable was refused: the harness ends the episode
        info = dict(info, **env.end_without_submission("stopped"))
    score = dict(info.get("score") or {})
    row = dict(id=env_id, level=level, seed=seed, policy=policy, reward=float(res.reward), passed=bool(score.get("passed")),
               progress=info.get("progress"), outcome=info.get("outcome"), submitted=bool(info.get("submitted")),
               checks=[[k, bool(v)] for k, v in (score.get("checks") or {}).items()],          # a list: the declared order is compared
               deliverable=_plain(delivered[-1] if delivered else None),
               details_sha256=_sha({k: v for k, v in score.items() if k not in ("outcome",)}),
               policy_error=info.get("policy_error"), wall_s=round(time.time() - t0, 2))             # the time is not compared
    env.close()
    del env.step                                                         # the recorder holds the environment: break the cycle
    return row


def _rss_gb() -> float:
    with open("/proc/self/statm", encoding="ascii") as f:               # resident pages now (the peak never falls)
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9


def _rows(path: str) -> list:
    """The episodes of a file a run wrote. A last line cut by a stopped run is dropped, and removed from the file."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    if lines[-1]:                                                        # no newline at the end: the line is not whole
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(line + "\n" for line in lines[:-1]))
    return [json.loads(line) for line in lines[:-1]]


def play(a) -> int:
    import mrgym
    from mrgym import policies
    k, n = (int(x) for x in a.shard.split("/"))
    done = {(r["id"], r["level"], r["seed"], r["policy"]) for r in _rows(a.out)}   # resume: these are not played again
    jobs = [(env_id, level, seed, policy) for env_id in a.ids for level in mrgym.spec(env_id).levels
            for seed in range(a.seeds) for policy in policies.policy_names(mrgym.spec(env_id))]
    jobs = [j for i, j in enumerate(jobs) if i % n == k and j not in done]
    t0, played = time.time(), 0
    with open(a.out, "a", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(play_one(*job)) + "\n")
            f.flush()
            played += 1
            gc.collect()
            if played % 100 == 0:
                print(f"{played} / {len(jobs)} episodes, {time.time() - t0:.0f} s, {_rss_gb():.2f} GB", file=sys.stderr, flush=True)
            if a.max_rss_gb and _rss_gb() > a.max_rss_gb and played < len(jobs):
                break
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6      # kilobytes on Linux
    print(f"{played} of {len(jobs)} episodes played ({len(done)} were there) in {time.time() - t0:.0f} s, peak memory {peak:.2f} GB, "
          f"mrgym from {os.path.dirname(mrgym.__file__)}")
    return 0 if played == len(jobs) else EXIT_AGAIN


def _load(paths) -> dict:
    return {(r["id"], r["level"], r["seed"], r["policy"]): r for path in paths.split(",") for r in _rows(path)}


def compare(a) -> int:
    base, new = _load(a.base), _load(a.branch)
    only = sorted(set(base) ^ set(new))
    differ, moved, kept = [], collections.Counter(), collections.Counter()
    for key in sorted(set(base) & set(new)):
        b, n = base[key], new[key]
        fields = [f for f in COMPARED if b[f] != n[f]]
        if b["outcome"] != n["outcome"]:
            if b["outcome"] in a.relabelled:
                moved[(key[0], b["outcome"], n["outcome"])] += 1
            else:
                fields.append("outcome")
        else:
            kept[(key[0], b["outcome"])] += 1
        if fields:
            differ.append((key, fields))
    n_both = len(set(base) & set(new))
    print(f"episodes compared: {n_both} ({len(base)} base, {len(new)} branch); in one file only: {len(only)}")
    print(f"fields compared per episode: {', '.join(COMPARED)}; outcome (may change only from {', '.join(a.relabelled)})")
    print(f"episodes that differ: {len(differ)}")
    for key, fields in differ[:40]:
        print("  ", key, fields)
    left = sum(c for (_, label), c in kept.items() if label in a.relabelled)
    print(f"outcome labels: {sum(moved.values())} relabelled, {left} still {' / '.join(a.relabelled)}")
    for (env_id, was, now), c in sorted(moved.items()):
        print(f"  {env_id}: {was} -> {now}: {c}")
    for (env_id, label), c in sorted(kept.items()):
        print(f"  {env_id}: {label} (unchanged): {c}")
    return 1 if differ or only or not n_both else 0


def relabel(a) -> int:
    import importlib

    import mrgym
    from mrgym.envs import family as F
    for env_id in a.ids:
        task = importlib.import_module(mrgym.spec(env_id).family + ".task")
        table, n = collections.defaultdict(collections.Counter), 0
        for path in sorted(glob.glob(os.path.join(a.folder, env_id.replace("/", "-"), "L*", "s*", "e*", "record.json"))):
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
            now = rec["outcome"]
            if rec.get("ladder", {}).get("submitted") and rec.get("checks"):       # a delivered answer: the label is a function of its checks
                now = F.outcome_of(dict(passed=bool(rec["passed"]), checks=rec["checks"]), task)
            table[rec["level"]][(rec["outcome"], now)] += 1
            n += 1
        print(f"{env_id}: {n} records; labels of a failed delivery: {', '.join(getattr(task, 'OUTCOMES', ('check_failed',)))}")
        for level in sorted(table):
            for (was, now), c in sorted(table[level].items(), key=lambda kv: (kv[0][0], -kv[1])):
                print(f"  L{level}  {was:24s} -> {now:28s} {c}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("play")
    p.add_argument("out")
    p.add_argument("--ids", nargs="+", default=list(DEFAULT_IDS))
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--shard", default="0/1", help="k/n: play every n-th episode starting at k (one process per shard)")
    p.add_argument("--max-rss-gb", type=float, default=0.0, help="stop when the resident memory passes this and exit 75 (0: no limit)")
    c = sub.add_parser("compare")
    c.add_argument("base", help="file, or files separated by commas")
    c.add_argument("branch")
    c.add_argument("--relabelled", nargs="+", default=["check_failed"])
    r = sub.add_parser("relabel")
    r.add_argument("folder")
    r.add_argument("--ids", nargs="+", default=list(DEFAULT_IDS))
    a = ap.parse_args(argv)
    return dict(play=play, compare=compare, relabel=relabel)[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
