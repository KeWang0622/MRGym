"""The instance bank: generated instances stored next to the package, so that an episode does not pay for generation.

Generating an instance costs seconds (candidate sites are classified on the twins, the remedy is verified), and it is
deterministic in (seed, level). ``tools/build_clinic_bank.py`` runs the generator offline and writes one JSON line per
instance to ``bank/level<k>.jsonl``; ``get`` returns a stored instance or None, and the generator is the fallback.
A line is keyed by a one-way function of the world seed and is ignored unless it was written by this engine and
this generator version, so a stale bank can never change an instance.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.resources
import json

from ...scanner import session

GENERATOR_VERSION = 2


def stamp() -> str:
    return f"{session.ENGINE_ID}/clinic-generator-{GENERATOR_VERSION}"


def key(seed: str, level: int) -> str:
    return hashlib.sha256(f"clinic-bank|{level}|{seed}".encode()).hexdigest()[:24]


@functools.lru_cache(maxsize=8)
def _load(level: int) -> dict:
    try:
        text = importlib.resources.files("mrgym.envs.artefact_clinic").joinpath("bank", f"level{level}.jsonl").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return {}
    rows = (json.loads(line) for line in text.splitlines() if line.strip())
    return {r["key"]: r["instance"] for r in rows if r.get("stamp") == stamp()}


def get(seed: str, level: int):
    return _load(level).get(key(seed, level))


def line(seed: str, level: int, instance: dict) -> str:
    return json.dumps(dict(key=key(seed, level), stamp=stamp(), instance=instance), sort_keys=True, separators=(",", ":"))


def budget_range(level: int):
    """Smallest and largest scan-time budget among the stored instances of a level, or None without a bank."""
    b = [inst["budget_s"] for inst in _load(level).values()]
    return [min(b), max(b)] if b else None


def battery_over_budget(level: int):
    """(median, number of instances) of the stored ratio "cost of the minimal battery / budget" of a level whose
    instances carry it, or None."""
    r = sorted(inst["hidden"]["battery_over_budget"] for inst in _load(level).values() if "battery_over_budget" in inst["hidden"])
    return (r[len(r) // 2], len(r)) if r else None
