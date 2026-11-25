"""Environment ids, specs and ``make``.

Id grammar (the Gymnasium one):  ``<Namespace>/<Name>-v<int>``, e.g. ``Scanner/NoWrap-v1``.

    Namespace   the instrument the agent is dropped into (``Scanner`` = the protocol-level virtual MR scanner)
    Name        what is asked of the agent on that instrument (a task family)
    -vN         bumped whenever a change can move a reward: physics, noise, pricing, tool schema, scoring.
                Old versions stay registered; a result always names the id it was measured on.
Levels are a keyword (``level=2``), not part of the id: a level changes the hidden-world prior, the budget, what is
already acquired at reset and the hint tier of the task text, never the tools, the deliverable or the scoring rule.

Every id carries a ``kind``:
    sandbox    no reward; for demos and for people
    practice   scored, but its own gate rejects a level it would claim (for example: a fixed recipe passes); for
               learning the instrument
    admitted   scored, and the stored gate admits every level of the claim on the current engine: the reference policy
               passes, every policy that ignores the object fails, every listed ablation loses
    candidate  scored; the gate has not been run and stored on the current engine yet (development state)
``kind`` is a claim; the stored gate result (``mrgym gate <id> --n 300 --write``) is the evidence, and ``mrgym check
<id>`` fails when an id is listed as admitted without it.
"""
from __future__ import annotations

import dataclasses
import importlib
import re

ENV_ID_RE = re.compile(r"^(?P<namespace>[A-Z][A-Za-z0-9]*)/(?P<name>[A-Z][A-Za-z0-9]*)-v(?P<version>\d+)$")


class Error(Exception):
    pass


class UnknownEnv(Error):
    pass


class RetiredEnv(UnknownEnv):
    """The id was served by an earlier package version and cannot be run in this one."""


@dataclasses.dataclass(frozen=True)
class EnvSpec:
    id: str
    entry_point: str                    # "module:Class"; imported lazily, so listing ids costs nothing
    card: str                           # "module:CARD" -> the environment card (a dict, see mrgym/cards.py)
    kind: str = "sandbox"               # sandbox | practice | admitted | candidate
    family: str = ""                    # package with ``task`` and ``controls`` modules ("" for a sandbox)
    levels: tuple = (1,)
    default_level: int = 1
    max_steps: int = 64                 # tool calls per episode before truncation
    reward_range: tuple = (0.0, 1.0)
    reward_threshold: float = 1.0       # reward of a passing episode
    nondeterministic: bool = False      # same (seed, level, tool calls) -> same observations and reward
    kwargs: dict = dataclasses.field(default_factory=dict)

    @property
    def namespace(self):
        return ENV_ID_RE.match(self.id)["namespace"]

    @property
    def name(self):
        return ENV_ID_RE.match(self.id)["name"]

    @property
    def version(self):
        return int(ENV_ID_RE.match(self.id)["version"])


registry: dict[str, EnvSpec] = {}


# Versioning (docs/VERSIONING.md). An id is permanent: it is served, deprecated (served, with a warning that names its
# successor) or retired (not runnable here; the error names the last package version that served it). It is never reused.
retired: dict[str, tuple[str, str | None]] = {}       # id -> (last package version that served it, successor id or None)
deprecated: dict[str, tuple[str, str]] = {}           # registered id -> (package version that deprecated it, successor id)
_warned: set[str] = set()


def retire(id: str, last_version: str, successor: str | None = None) -> None:
    """Record that ``id`` is no longer served. ``last_version``: the last package version whose ``make(id)`` works."""
    if not ENV_ID_RE.match(id):
        raise Error(f"malformed environment id {id!r}; ids look like 'Scanner/NoWrap-v1'")
    if id in registry:
        raise Error(f"{id} is registered; an id is served or retired, not both")
    if not last_version or successor == id:
        raise Error(f"retiring {id} needs the last package version that served it, and a successor other than itself")
    retired[id] = (str(last_version), successor)


def deprecate(id: str, since: str, successor: str) -> None:
    """Keep serving ``id`` but warn once per process that ``successor`` replaces it."""
    if id not in registry:
        raise Error(f"{id} is not registered; only a served id can be deprecated")
    if not successor or successor == id:
        raise Error(f"deprecating {id} needs a successor id other than itself")
    deprecated[id] = (str(since), successor)


def _retired_message(id: str) -> str:
    last, successor = retired[id]
    return (f"{id} is retired: mrgym {last} is the last package version that served it (`pip install mrgym=={last}` "
            f"reproduces it). " + (f"Successor: {successor}." if successor else "It has no successor."))


def register(id: str, entry_point: str, card: str, **kw) -> EnvSpec:
    if not ENV_ID_RE.match(id):
        raise Error(f"malformed environment id {id!r}; ids look like 'Scanner/NoWrap-v1'")
    if id in registry:
        raise Error(f"{id} is already registered")
    if id in retired:
        raise Error(f"{id} is retired; an id is never reused")
    registry[id] = EnvSpec(id=id, entry_point=entry_point, card=card, **kw)
    return registry[id]


def spec(id: str) -> EnvSpec:
    if id in registry:
        return registry[id]
    if isinstance(id, str) and id in retired:
        raise RetiredEnv(_retired_message(id))
    m = ENV_ID_RE.match(id) if isinstance(id, str) else None
    if not m:
        raise UnknownEnv(f"malformed environment id {id!r}; ids look like 'Scanner/NoWrap-v1'")
    same = sorted(s.version for s in registry.values() if (s.namespace, s.name) == (m["namespace"], m["name"]))
    if same:
        raise UnknownEnv(f"{id} is not registered; registered versions of {m['namespace']}/{m['name']}: "
                         f"{', '.join(f'v{v}' for v in same)}")
    raise UnknownEnv(f"{id} is not registered; registered ids: {sorted(registry)}")


def _load(path):
    module, _, attr = path.partition(":")
    return getattr(importlib.import_module(module), attr)


def make(id: str, **kwargs):
    """``mrgym.make("Scanner/NoWrap-v1", level=1, seed=0, obs="text")`` -> an ``mrgym.Env`` (call ``reset`` next).

    Keywords: ``level`` (one of ``spec(id).levels``), ``seed`` (int, str, 32 hex digits, or None for fresh entropy),
    ``obs`` ("text" or "vision"), ``split`` ("train"; "heldout" needs MRGYM_SECRET), ``max_steps``, ``reveal_score``,
    plus the environment's own keywords."""
    s = spec(id)
    if id in deprecated and id not in _warned:
        import warnings
        _warned.add(id)
        warnings.warn(f"{id} is out of date since mrgym {deprecated[id][0]}; use {deprecated[id][1]}", DeprecationWarning, stacklevel=2)
    level = kwargs.get("level", s.default_level)
    if level not in s.levels:
        raise Error(f"{id} has levels {list(s.levels)}, not {level!r}")
    return _load(s.entry_point)(spec=s, **{**s.kwargs, **kwargs})


def list_envs(namespace: str | None = None) -> list[str]:
    return sorted(i for i, s in registry.items() if namespace in (None, s.namespace))


def card(id: str) -> dict:
    """The environment card: authored fields, the registered claim (``kind``) and the stored evidence, if any: the
    gate result (``mrgym gate <id> --write``) and the validity table (``mrgym validate --perturb <id> --write``)."""
    from . import gate, perturb
    s = spec(id)
    c = dict(_load(s.card)(), kind=s.kind, levels_registered=list(s.levels), default_level=s.default_level)
    c["gate"] = gate.load(id)
    c["validity"] = perturb.load(id)
    c["status"] = gate.status_line(s, c["gate"])
    return c
