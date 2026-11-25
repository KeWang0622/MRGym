"""The per-id fingerprint: a hash of everything that can move a reward under an environment id.

    python -m mrgym.compat.fingerprint --check             every registered id against the stored fingerprints (exit 1 on a change)
    python -m mrgym.compat.fingerprint --write             store the fingerprints of the ids that are not frozen
    python -m mrgym.compat.fingerprint --freeze ID ...     mark ids as released: their fingerprint can no longer be rewritten

The rule (docs/VERSIONING.md): rewards are comparable only within one environment id. Anything that can change the
reward a fixed policy earns therefore needs a new id (``-vN+1``). The fingerprint makes that rule mechanical. Its parts:

    registration      levels, default level, tool-call limit, reward range and threshold, constructor keywords
    engine            the scanner engine id (same seed and protocol give the same data only on the same engine)
    rules             the authored rule fields of the environment card: deliverable, reward, limits, levels
    scoring_source    the code of the modules that define the task and the environment class, as a syntax tree
                      without docstrings (comments, formatting and docstrings do not count; any other edit does)
    L<level>.statement   the text the agent reads at reset (fixed seed)
    L<level>.tools       the tool schemas and descriptions (``env.tools("openai")``)
    L<level>.transcript  one scripted episode (the reference policy; the probe policy for an id without controls):
                         calls, observation digest, scanner-log head and reward

A fingerprint that differs from the stored one under an unchanged id fails ``tests/test_id_fingerprint.py`` with the
names of the parts that moved. While an id has never been released (``frozen: false``) the answer may be
``--write``; once it is frozen the only answers are to undo the change or to register a new id and retire or
deprecate the old one. Freezing is the release manager's decision; nothing here freezes or retires an id on its own.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.resources
import inspect
import json
import os
import sys

SCHEMA = "mrgym-fingerprint/1"
SEED = "fingerprint:0"                        # a string seed: its own world, away from the integer training seeds
RULE_FIELDS = ("deliverable", "reward", "limits", "levels")


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


def _tree(node):
    """A syntax tree as nested lists that read the same on every supported Python: empty and absent fields are left
    out (newer versions add optional fields), positions are not part of it."""
    if isinstance(node, ast.AST):
        return [type(node).__name__] + [[name, _tree(value)] for name, value in ast.iter_fields(node)
                                        if value is not None and value != [] and name not in ("type_comment", "kind")]
    if isinstance(node, list):
        return [_tree(x) for x in node]
    return repr(node)


def source_digest(module_names) -> str:
    """sha256 over the syntax trees (docstrings removed) of the named modules."""
    h = hashlib.sha256()
    for name in sorted(set(module_names)):
        src = inspect.getsource(importlib.import_module(name))
        h.update(name.encode() + b"\x00" + json.dumps(_tree(_strip_docstrings(ast.parse(src)))).encode() + b"\x00")
    return h.hexdigest()


def scoring_modules(spec) -> list[str]:
    """Modules whose code decides the reward of an id: the task module of its family and the module of its class."""
    names = [spec.entry_point.partition(":")[0], spec.card.partition(":")[0]]
    if spec.family:
        names.append(spec.family + ".task")
    return names


def static_parts(env_id: str) -> dict:
    """The parts that need no episode: registration, engine, rules, scoring_source."""
    import mrgym
    spec = mrgym.spec(env_id)
    module, _, attr = spec.card.partition(":")
    card = getattr(importlib.import_module(module), attr)()              # the authored card: no stored results, no status
    return {
        "registration": _sha(dict(levels=list(spec.levels), default_level=spec.default_level, max_steps=spec.max_steps,
                                  reward_range=list(spec.reward_range), reward_threshold=spec.reward_threshold,
                                  nondeterministic=spec.nondeterministic, kwargs=spec.kwargs)),
        "engine": _sha(mrgym.ENGINE_ID),
        "rules": _sha({k: card.get(k) for k in RULE_FIELDS}),
        "scoring_source": source_digest(scoring_modules(spec)),
    }


def level_parts(env_id: str, level: int) -> dict:
    """The parts of one level: what the agent reads at reset, its tools, and one scripted episode."""
    from . import scripted_episode
    ep = scripted_episode(env_id, level, SEED)
    return {f"L{level}.statement": _sha(ep["statement"]), f"L{level}.tools": _sha(ep["tools"]),
            f"L{level}.transcript": _sha({k: ep[k] for k in ("calls", "reward", "done", "text_sha256", "log_head", "policy")})}


def parts(env_id: str) -> dict:
    """{part name: sha256} of an id. See the module docstring."""
    import mrgym
    out = static_parts(env_id)
    for level in mrgym.spec(env_id).levels:
        out.update(level_parts(env_id, level))
    return out


def fingerprint(part_hashes: dict) -> str:
    return _sha(part_hashes)


def path() -> str:
    return str(importlib.resources.files("mrgym.compat").joinpath("fingerprints.json"))


def load(file: str | None = None) -> dict:
    try:
        with open(file or path(), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return dict(schema=SCHEMA, ids={})


def check(ids=None, file: str | None = None) -> list[str]:
    """Problems (empty = every id still has the fingerprint that is stored for it)."""
    import mrgym
    from mrgym.registry import retired
    stored = load(file)["ids"]
    bad = []
    for env_id in ids or mrgym.list_envs():
        if env_id not in stored:
            bad.append(f"{env_id}: no stored fingerprint. A new id: run `python -m mrgym.compat.fingerprint --write`.")
            continue
        now = parts(env_id)
        if fingerprint(now) != stored[env_id]["fingerprint"]:
            was = stored[env_id].get("parts", {})
            moved = sorted(k for k in set(now) | set(was) if now.get(k) != was.get(k))
            state = "FROZEN (released)" if stored[env_id].get("frozen") else "not frozen (never released)"
            bad.append(f"{env_id}: the fingerprint changed under an unchanged id; parts that moved: {', '.join(moved)}. The id is {state}. "
                       "A change that can move a reward needs a new id (-vN+1). "
                       + ("Undo the change, or register the new id and retire or deprecate this one."
                          if stored[env_id].get("frozen") else
                          "If this id has never been released, `python -m mrgym.compat.fingerprint --write` stores the new fingerprint."))
    if ids is None:
        for env_id in sorted(set(stored) - set(mrgym.registry) - set(retired)):
            bad.append(f"{env_id}: has a stored fingerprint but is neither registered nor retired. An id never disappears: "
                       "retire it with `mrgym.registry.retire(id, last_version, successor)`.")
    return bad


def write(ids=None, file: str | None = None) -> tuple[list[str], list[str]]:
    """Store the fingerprints of ``ids`` (default: every registered id). A frozen id whose fingerprint changed is
    refused. Returns (written ids, refusals)."""
    import mrgym
    data = load(file)
    written, refused = [], []
    for env_id in ids or mrgym.list_envs():
        now = parts(env_id)
        old = data["ids"].get(env_id)
        if old and old.get("frozen"):
            if old["fingerprint"] != fingerprint(now):
                refused.append(f"{env_id}: frozen; its fingerprint cannot be rewritten. Register a new id.")
            continue
        data["ids"][env_id] = dict(fingerprint=fingerprint(now), frozen=False, package=mrgym.__version__, engine=mrgym.ENGINE_ID, parts=now)
        written.append(env_id)
    data["ids"] = dict(sorted(data["ids"].items()))
    _dump(data, file)
    return written, refused


def freeze(ids, file: str | None = None) -> list[str]:
    """Mark ids as released. Refuses an id whose stored fingerprint is not the current one."""
    data, bad = load(file), check(ids, file)
    if bad:
        return bad
    for env_id in ids:
        data["ids"][env_id]["frozen"] = True
    _dump(data, file)
    return []


def _dump(data, file):
    with open(file or path(), "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(schema=SCHEMA, ids=data["ids"]), f, indent=1, sort_keys=True)
        f.write("\n")


def main(argv=None) -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ap = argparse.ArgumentParser(prog="python -m mrgym.compat.fingerprint", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--freeze", nargs="+", metavar="ID")
    ap.add_argument("--id", action="append")
    ap.add_argument("--file", default=None, help="fingerprint file (default: the one shipped in the package)")
    a = ap.parse_args(argv)
    if a.freeze:
        bad = freeze(a.freeze, a.file)
        print("\n".join(bad) if bad else f"frozen: {', '.join(a.freeze)}")
        return 1 if bad else 0
    if a.write:
        written, refused = write(a.id, a.file)
        print(f"written: {', '.join(written) or 'nothing'}")
        print("\n".join(refused), end="\n" if refused else "")
        return 1 if refused else 0
    bad = check(a.id, a.file)
    if bad:
        print("\n".join(bad))
        return 1
    stored = load(a.file)["ids"]
    for env_id, row in stored.items():
        print(f"{env_id}  {row['fingerprint'][:16]}  {'frozen' if row.get('frozen') else 'not frozen'}  (mrgym {row.get('package')}, {row.get('engine')})")
    print(f"{len(stored)} id(s): every fingerprint is current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
