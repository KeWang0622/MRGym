"""Held-out epochs: commit before the first evaluation, reveal at rotation, verify afterwards without trusting the host.

A held-out number is only worth something if nobody could have chosen the instances after seeing the results. The
generator is procedural, so the instances of an epoch are fixed by (secret, epoch name, index). This module publishes
that choice in advance without revealing it:

    mrgym epoch keygen --out secret.key                         32 random bytes, hex, mode 0600
    mrgym epoch new    --id ID --split test-iid --epoch 2026q4 --n 512 --out e.commit.json      publish this file first
    mrgym epoch run    --commit e.commit.json --policy reference --n 8 --out e.result.json      a scripted policy; models
                                                                 go through a harness that writes the same result file
    mrgym epoch reveal --commit e.commit.json --out e.reveal.json                               at rotation
    mrgym epoch verify --commit e.commit.json --reveal e.reveal.json [--result e.result.json]   anyone, later

Commitment of index i of a level:  c_i = SHA-256("mrgym-commit-v1" + world_seed_i + r_i),  r_i = HMAC(K_commit, i).
It binds by collision resistance and hides by the secret r_i. The file carries every c_i, a Merkle root per level
(RFC 6962 hashing) and one over the levels, the engine id, and the SHA-256 of the generator sources and of the level
specs: the ranges of a held-out level need not be in the public repository, their digest is in the commitment.

A result names the epoch, the root and, per level, the indices it played. They must be 0 .. m-1 without a gap: every
started index is in the denominator. ``verify`` recomputes every c_i and the roots from the reveal, rebuilds the worlds,
replays every published log against its head and recomputes every reward with the code that scores in process. It fails
when one seed, one log line, one reported index or one reward is changed.

Rotation. An evaluation never needs an instance twice: a new epoch per release (or per 90 days); inside an epoch every
agent plays the same leading indices, and only aggregate scores leave the host until the reveal.
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import importlib
import importlib.resources
import json
import os

from . import keys

COMMIT_SCHEMA, REVEAL_SCHEMA, RESULT_SCHEMA = "mrgym-epoch-commit/1", "mrgym-epoch-reveal/1", "mrgym-epoch-result/1"
_TAG = b"mrgym-commit-v1"


class EpochError(keys.Error):
    pass


# ------------------------------------------------------------------------------------------------ hashing
def merkle_root(leaves: list) -> str:
    """Merkle tree hash of RFC 6962 section 2.1 over hex leaves (leaf prefix 0x00, node prefix 0x01)."""
    def mth(d):
        if not d:
            return hashlib.sha256(b"").digest()
        if len(d) == 1:
            return hashlib.sha256(b"\x00" + d[0]).digest()
        k = 1
        while k * 2 < len(d):
            k *= 2
        return hashlib.sha256(b"\x01" + mth(d[:k]) + mth(d[k:])).digest()
    return mth([bytes.fromhex(x) for x in leaves]).hex()


def commitment(world_seed: str, blind: str) -> str:
    return hashlib.sha256(_TAG + bytes.fromhex(world_seed) + bytes.fromhex(blind)).hexdigest()


def _blind(master: bytes, epoch: str, env_id: str, level: int, split: str, index: int) -> str:
    return hmac.new(keys.sub(master, "commit"), f"{epoch}|{env_id}|{level}|{split}|{index}".encode(), hashlib.sha256).hexdigest()


def _tree_digest(package: str) -> str:
    """SHA-256 over the Python and Markdown sources of a package directory (names and bytes, sorted)."""
    root = importlib.resources.files(package)
    h = hashlib.sha256()
    for f in sorted((p for p in root.iterdir() if p.is_file() and p.name.endswith((".py", ".md"))), key=lambda p: p.name):
        h.update(f.name.encode() + b"\x00" + f.read_bytes() + b"\x00")
    return h.hexdigest()


def source_digests(spec) -> dict:
    """What generates and scores the worlds: the scanner package and the family package."""
    return dict(scanner=_tree_digest("mrgym.scanner"), family=_tree_digest(spec.family), family_package=spec.family)


def level_digest(spec, level: int) -> str:
    task = importlib.import_module(spec.family + ".task")
    return hashlib.sha256(json.dumps(task.LEVELS[level], sort_keys=True, default=str).encode()).hexdigest()


def _master(environ=None) -> bytes:
    master = keys.load_master(environ)
    if not master:
        raise EpochError("this needs the secret: set MRGYM_SECRET_FILE to a key file (mrgym epoch keygen --out FILE)")
    return master


def _pairs(master, env_id, level, split, epoch, n):
    seeds = [keys.world_seed(env_id, level, split, i, environ={}, epoch=epoch, master=master) for i in range(n)]
    return seeds, [_blind(master, epoch, env_id, level, split, i) for i in range(n)]


# ------------------------------------------------------------------------------------------------ commit / reveal
def new(env_id: str, levels=None, split: str = "test-iid", epoch: str = "0", n: int = 512, environ=None) -> dict:
    """The commitment file of an epoch. Publish it before the first evaluation."""
    import mrgym
    spec = mrgym.spec(env_id)
    if not spec.family:
        raise EpochError(f"{env_id} is a sandbox: nothing is scored on it")
    if keys.is_public(split):
        raise EpochError(f"split {split!r} is public: there is nothing to commit to. Secret splits: {', '.join(keys.SECRET_SPLITS)}")
    epoch, master = keys.epoch_of({}, epoch), _master(environ)
    out = dict(schema=COMMIT_SCHEMA, canary=keys.CANARY, env_id=env_id, split=split, epoch=epoch, n=int(n),
               engine=mrgym.ENGINE_ID, package_version=mrgym.__version__, sources=source_digests(spec),
               date_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
               scheme='c_i = SHA-256("mrgym-commit-v1" + world_seed_i + r_i); Merkle root as in RFC 6962 section 2.1', levels={})
    for lv in (levels or spec.levels):
        seeds, blinds = _pairs(master, env_id, lv, split, epoch, n)
        cs = [commitment(s, b) for s, b in zip(seeds, blinds)]
        out["levels"][str(lv)] = dict(level_spec_sha256=level_digest(spec, lv), root=merkle_root(cs), commitments=cs)
    out["root"] = merkle_root([v["root"] for _, v in sorted(out["levels"].items(), key=lambda kv: int(kv[0]))])
    return out


def reveal(commit: dict, environ=None) -> dict:
    """The opening of a commitment: every world seed and blind. Publishing it ends the epoch."""
    master = _master(environ)
    out = dict(schema=REVEAL_SCHEMA, canary=keys.CANARY, env_id=commit["env_id"], split=commit["split"], epoch=commit["epoch"],
               root=commit["root"], levels={})
    for lv in commit["levels"]:
        seeds, blinds = _pairs(master, commit["env_id"], int(lv), commit["split"], commit["epoch"], commit["n"])
        out["levels"][lv] = dict(world_seeds=seeds, blinds=blinds)
    check = check_reveal(commit, out)
    if check:
        raise EpochError("this secret does not open the commitment: " + check[0])
    return out


def check_reveal(commit: dict, rev: dict) -> list:
    """Problems of a reveal against its commitment (empty = it opens every commitment and reproduces the roots)."""
    problems = []
    if commit.get("schema") != COMMIT_SCHEMA or rev.get("schema") != REVEAL_SCHEMA:
        return ["not a commitment / reveal file of this schema"]
    for k in ("env_id", "split", "epoch", "root"):
        if commit.get(k) != rev.get(k):
            problems.append(f"{k} differs between the commitment and the reveal")
    roots = []
    for lv, c in sorted(commit["levels"].items(), key=lambda kv: int(kv[0])):
        r = rev.get("levels", {}).get(lv)
        cs = c["commitments"]
        if not r or len(r.get("world_seeds", [])) != len(cs) or len(r.get("blinds", [])) != len(cs) or len(cs) != commit["n"]:
            problems.append(f"level {lv}: the reveal does not hold {commit['n']} seeds and blinds")
            continue
        try:
            bad = [i for i, (s, b) in enumerate(zip(r["world_seeds"], r["blinds"])) if commitment(s, b) != cs[i]]
        except (ValueError, TypeError):
            bad = [-1]
        if bad:
            problems.append(f"level {lv}: {len(bad)} revealed seed(s) do not open their commitment (first index {bad[0]})")
        if merkle_root(cs) != c["root"]:
            problems.append(f"level {lv}: the commitments do not hash to the level root")
        roots.append(c["root"])
    if merkle_root(roots) != commit.get("root"):
        problems.append("the level roots do not hash to the epoch root")
    return problems


# ------------------------------------------------------------------------------------------------ playing and scoring
def episode_env(spec: dict, reveal_score: bool = True):
    """The environment of an episode spec ``{"env_id", "level", "split", "seed"}``, reset (it holds the hidden config)."""
    import mrgym
    kw = {k: spec[k] for k in ("level",) if spec.get(k) is not None}
    env = mrgym.make(spec["env_id"], seed=spec.get("seed"), split=spec.get("split", "train"), reveal_score=reveal_score, **kw)
    env.reset()
    return env


def verify_episode(spec: dict, events: list, head, answer, log_key=None) -> dict:
    """Score a deliverable against a recorded log and its head, with the code that scores in process.

    Returns ``dict(reward, passed, score, replay, refused)``. ``refused``: the reason when the deliverable is not one the
    environment evaluates (the episode would have continued); the reward is then 0."""
    from .scanner import session as core
    env = episode_env(spec)
    cfg = env.episode_config()
    env.use_session(core.RecordedSession(cfg, events, head, engine=env._session.engine, log_key=log_key))
    terminal = next(t.name for t in env._tool_specs() if t.terminal)
    step = env.step({"name": terminal, "arguments": answer})
    if not step.done:
        rep = core.replay(cfg, events, engine=env._session.engine, head=env._session.receipt(with_key=True))
        return dict(reward=0.0, passed=False, score=None, replay=rep, refused=step.observation.text.splitlines()[0])
    score = step.info.get("score") or {}
    return dict(reward=float(step.reward), passed=score.get("passed"), score=score, replay=step.info["replay"], refused=None)


def run(commit: dict, policy: str = "reference", n: int = 8, levels=None) -> dict:
    """Play the leading ``n`` indices of every level with a scripted control policy; returns the result to publish.
    (A model is run by a harness; what it must write is this structure: per index the answer, the log and its head.)"""
    import mrgym

    from . import policies
    _master()
    old = os.environ.get("MRGYM_EPOCH")
    os.environ["MRGYM_EPOCH"] = commit["epoch"]
    try:
        out = dict(schema=RESULT_SCHEMA, canary=keys.CANARY, env_id=commit["env_id"], split=commit["split"], epoch=commit["epoch"],
                   root=commit["root"], engine=mrgym.ENGINE_ID, agent=dict(kind="scripted", name=policy), levels={})
        for lv in [str(x) for x in (levels or sorted(int(k) for k in commit["levels"]))]:
            if n > commit["n"]:
                raise EpochError(f"the epoch commits to {commit['n']} indices per level; {n} were asked for")
            episodes = []
            for i in range(n):
                env = mrgym.make(commit["env_id"], level=int(lv), seed=i, split=commit["split"])
                step = policies.run(env, policy)
                call = env.transcript()[-1]["call"] or {}
                episodes.append(dict(index=i, answer=call.get("arguments"), reward=float(step.reward),
                                     submitted=bool(step.info.get("submitted")), head=env._session.receipt(with_key=True),
                                     log=env.scanner_log()))
                env.close()
            out["levels"][lv] = dict(indices=list(range(n)), passed=int(sum(e["reward"] >= 1.0 for e in episodes)), episodes=episodes)
        return out
    finally:
        if old is None:
            os.environ.pop("MRGYM_EPOCH", None)
        else:
            os.environ["MRGYM_EPOCH"] = old


def verify(commit: dict, rev: dict, result: dict | None = None, allow_source_drift: bool = False) -> dict:
    """``dict(ok, problems, checked)``. See the module docstring for what is recomputed."""
    import mrgym
    problems = check_reveal(commit, rev)
    checked = dict(commitments=sum(len(c["commitments"]) for c in commit.get("levels", {}).values()), episodes=0, rewards_equal=0)
    spec = mrgym.spec(commit["env_id"])
    drift = [k for k, v in source_digests(spec).items() if commit.get("sources", {}).get(k) != v]
    drift += [f"level {lv} spec" for lv, c in commit["levels"].items() if c.get("level_spec_sha256") != level_digest(spec, int(lv))]
    if commit.get("engine") != mrgym.ENGINE_ID:
        drift.append(f"engine ({commit.get('engine')} committed, {mrgym.ENGINE_ID} installed)")
    if drift and not allow_source_drift:
        problems.append("the installed generator is not the committed one (" + ", ".join(drift) + "): the worlds cannot be "
                        "rebuilt with it. Install the committed version, or pass --allow-source-drift to try anyway")
    if result is not None and not problems:
        if result.get("schema") != RESULT_SCHEMA or any(result.get(k) != commit.get(k) for k in ("env_id", "split", "epoch", "root")):
            problems.append("the result does not name this epoch (env_id, split, epoch and root must equal the commitment's)")
        for lv, block in (result.get("levels") or {}).items() if not problems else ():
            seeds = rev["levels"].get(lv, {}).get("world_seeds")
            eps = block.get("episodes") or []
            idx = [e.get("index") for e in eps]
            if seeds is None:
                problems.append(f"level {lv}: not in the commitment")
                continue
            if block.get("indices") != list(range(len(eps))) or idx != list(range(len(eps))) or len(eps) > len(seeds):
                problems.append(f"level {lv}: the reported indices are not 0..{len(eps) - 1} without a gap "
                                "(every started index must be in the denominator)")
                continue
            for e in eps:
                checked["episodes"] += 1
                spec_i = dict(env_id=commit["env_id"], level=int(lv), split=commit["split"], seed=seeds[e["index"]])
                try:
                    v = verify_episode(spec_i, e.get("log") or [], e.get("head"), e.get("answer"))
                except Exception as ex:  # noqa: BLE001  a published file is untrusted input
                    problems.append(f"level {lv} index {e['index']}: the episode cannot be re-scored ({type(ex).__name__})")
                    continue
                rep = v["replay"]
                if not rep["ok"]:
                    problems.append(f"level {lv} index {e['index']}: the log does not replay: {rep['problems'][0]}")
                elif not rep["sealed"]:
                    problems.append(f"level {lv} index {e['index']}: the log is not sealed")
                elif abs(v["reward"] - float(e.get("reward", -1))) > 1e-12:
                    problems.append(f"level {lv} index {e['index']}: reported reward {e.get('reward')}, recomputed {v['reward']}")
                else:
                    checked["rewards_equal"] += 1
            if block.get("passed") != sum(float(e.get("reward", 0)) >= 1.0 for e in eps):
                problems.append(f"level {lv}: the reported pass count is not the sum over the episodes")
    return dict(ok=not problems, problems=problems, checked=checked, root=commit.get("root"), epoch=commit.get("epoch"))


# ------------------------------------------------------------------------------------------------ command line
def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def add_parser(sub):
    p = sub.add_parser("epoch", help="secrets, and the commitment of a held-out epoch: keygen, new, run, reveal, verify",
                       description=__doc__.split("\n\n")[0])
    s = p.add_subparsers(dest="epoch_cmd", required=True)
    q = s.add_parser("keygen", help="write 32 random bytes as hex to a new file of mode 0600 (a secret, or a salt)")
    q.add_argument("--out", required=True)
    q = s.add_parser("new", help="write the commitment of an epoch (needs MRGYM_SECRET_FILE); publish it before evaluating")
    q.add_argument("--id", required=True)
    q.add_argument("--levels", default=None, help="comma-separated (default: every level)")
    q.add_argument("--split", default="test-iid")
    q.add_argument("--epoch", default="0", help="name of the epoch, e.g. 2026q4")
    q.add_argument("--n", type=int, default=512, help="indices per level")
    q.add_argument("--out", required=True)
    q = s.add_parser("run", help="play the leading indices with a scripted policy and write the result file")
    q.add_argument("--commit", required=True)
    q.add_argument("--policy", default="reference")
    q.add_argument("--n", type=int, default=8)
    q.add_argument("--levels", default=None)
    q.add_argument("--out", required=True)
    q = s.add_parser("reveal", help="write the opening of a commitment (needs the secret); publishing it ends the epoch")
    q.add_argument("--commit", required=True)
    q.add_argument("--out", required=True)
    q = s.add_parser("verify", help="check a reveal against its commitment and re-score a published result")
    q.add_argument("--commit", required=True)
    q.add_argument("--reveal", required=True)
    q.add_argument("--result", default=None)
    q.add_argument("--allow-source-drift", action="store_true")
    return p


def main(a) -> int:
    levels = [int(x) for x in a.levels.split(",")] if getattr(a, "levels", None) else None
    try:
        if a.epoch_cmd == "keygen":
            print("written:", keys.keygen(a.out), "(mode 0600; use it as MRGYM_SECRET_FILE, or as MRGYM_SALT_FILE)")
        elif a.epoch_cmd == "new":
            c = new(a.id, levels, a.split, a.epoch, a.n)
            _dump(c, a.out)
            print(f"written: {a.out}\nepoch {c['epoch']} · {a.id} · split {c['split']} · {c['n']} indices x {len(c['levels'])} level(s) · root {c['root']}")
        elif a.epoch_cmd == "run":
            r = run(_load(a.commit), a.policy, a.n, levels)
            _dump(r, a.out)
            for lv, b in r["levels"].items():
                print(f"level {lv}: {a.policy} passes {b['passed']} of {len(b['episodes'])} (indices 0..{len(b['episodes']) - 1})")
            print("written:", a.out)
        elif a.epoch_cmd == "reveal":
            _dump(reveal(_load(a.commit)), a.out)
            print("written:", a.out, "(the epoch is over once this file is published)")
        else:
            v = verify(_load(a.commit), _load(a.reveal), _load(a.result) if a.result else None, a.allow_source_drift)
            print(json.dumps(v, indent=1))
            return 0 if v["ok"] else 1
    except (keys.Error, FileExistsError) as e:
        print(f"mrgym epoch: {e}")
        return 2
    return 0
