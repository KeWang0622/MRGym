"""Environment ids, specs and ``make``.

Id grammar (the Gymnasium one):  ``<Namespace>/<Name>-v<int>``, e.g. ``Scanner/NoWrap-v0``.

    Namespace   the instrument the agent is dropped into (``Scanner`` = the protocol-level virtual MR scanner)
    Name        what is asked of the agent on that instrument (a task family)
    -vN         bumped whenever a change can move a reward: physics, noise, pricing, tool schema, scoring.
                Old versions stay registered; a result always names the id it was measured on.
Levels are a keyword (``level=2``), not part of the id: a level changes the hidden-world prior and the budget, never
the tools, the deliverable or the scoring rule.

Every id carries a ``kind``:
    sandbox    no reward; for demos and for people
    practice   scored, but its own admission gate rejects it (a fixed recipe passes); for learning the instrument
    admitted   scored, and the gate admits instances: the reference policy passes and every shortcut policy fails
``kind`` is a claim; the stored control-policy results (``mrgym check <id> --admission``) are the evidence, and a
test fails when an id is listed as admitted without them.
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


@dataclasses.dataclass(frozen=True)
class EnvSpec:
    id: str
    entry_point: str                    # "module:Class"; imported lazily, so listing ids costs nothing
    card: str                           # "module:CARD" -> the environment card (a dict, see mrgym/cards.py)
    kind: str = "sandbox"               # sandbox | practice | admitted
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


def register(id: str, entry_point: str, card: str, **kw) -> EnvSpec:
    if not ENV_ID_RE.match(id):
        raise Error(f"malformed environment id {id!r}; ids look like 'Scanner/NoWrap-v0'")
    if id in registry:
        raise Error(f"{id} is already registered")
    registry[id] = EnvSpec(id=id, entry_point=entry_point, card=card, **kw)
    return registry[id]


def spec(id: str) -> EnvSpec:
    if id in registry:
        return registry[id]
    m = ENV_ID_RE.match(id) if isinstance(id, str) else None
    if not m:
        raise UnknownEnv(f"malformed environment id {id!r}; ids look like 'Scanner/NoWrap-v0'")
    same = sorted(s.version for s in registry.values() if (s.namespace, s.name) == (m["namespace"], m["name"]))
    if same:
        raise UnknownEnv(f"{id} is not registered; registered versions of {m['namespace']}/{m['name']}: "
                         f"{', '.join(f'v{v}' for v in same)}")
    raise UnknownEnv(f"{id} is not registered; registered ids: {sorted(registry)}")


def _load(path):
    module, _, attr = path.partition(":")
    return getattr(importlib.import_module(module), attr)


def make(id: str, **kwargs):
    """``mrgym.make("Scanner/NoWrap-v0", level=1, seed=0, obs="text")`` -> an ``mrgym.Env`` (call ``reset`` next).

    Keywords: ``level`` (one of ``spec(id).levels``), ``seed`` (int, str, 32 hex digits, or None for fresh entropy),
    ``obs`` ("text" or "vision"), ``split`` ("train"; "heldout" needs MRGYM_SECRET), ``max_steps``, ``reveal_score``,
    plus the environment's own keywords."""
    s = spec(id)
    level = kwargs.get("level", s.default_level)
    if level not in s.levels:
        raise Error(f"{id} has levels {list(s.levels)}, not {level!r}")
    return _load(s.entry_point)(spec=s, **{**s.kwargs, **kwargs})


def list_envs(namespace: str | None = None) -> list[str]:
    return sorted(i for i, s in registry.items() if namespace in (None, s.namespace))


def card(id: str) -> dict:
    """The environment card: authored fields plus the measured control-policy results, if any are stored."""
    from . import admission
    s = spec(id)
    c = dict(_load(s.card)(), levels_registered=list(s.levels), default_level=s.default_level)
    c["measured"] = admission.load(id)
    c["status"] = admission.status_line(s, c["measured"])
    return c
