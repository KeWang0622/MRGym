"""What a trainer needs from MR Gym beyond ``reset`` / ``step``. Numpy only; nothing here calls a model.

    mrgym.learn.hooks      per-step ``info``, one outcome label per episode, a turn cap the agent can see, group identity
                           and independent noise for the members of a group (the mixin behind every ``FamilyEnv``)
    mrgym.learn.shaping    optional potential-based step rewards that sum to the terminal reward for every policy
    mrgym.learn.check      ``mrgym check <id> --learnability``: a small parametric policy family trained on the real
                           reward on CPU; says whether the reward leads from a weak policy to the intended procedure
    mrgym.curriculum       a sampler over (environment, level) by learning progress
    mrgym.calibrate        ``mrgym calibrate``: pass rate of a model endpoint per level, and which levels are in band

The paid reward of every family is 1 for a pass and 0 otherwise (``mrgym.envs.family``). ``info["score"]["progress"]``
is a class-ordered diagnostic in [0, 1] that a trainer may use as its own signal; the environment never pays it.
See docs/TRAINING.md.
"""
from __future__ import annotations

import importlib.resources
import json
import os

BANDS = (("dead", 0.0, 0.02), ("frontier", 0.02, 0.20), ("entry", 0.20, 0.60), ("comfortable", 0.60, 0.80), ("mastered", 0.80, 1.0 + 1e-9))
ENTRY_BAND = (0.20, 0.60)
MIXED_MIN = 0.80                        # share of groups with reward spread that an entry rung needs
GROUP_SIZE = 8


def band(pass_rate: float) -> str:
    """Name of the band a pass rate falls in (see docs/TRAINING.md): dead, frontier, entry, comfortable, mastered."""
    return next(name for name, lo, hi in BANDS if lo <= pass_rate < hi)


def mixed_share(pass_rate: float, k: int = GROUP_SIZE) -> float:
    """Probability that a group of ``k`` independent episodes at ``pass_rate`` holds both a pass and a failure."""
    return 1.0 - pass_rate ** k - (1.0 - pass_rate) ** k


BAND_USE = dict(dead="never train on it: add an easier level or warm-start the model",
                frontier="only inside a curriculum next to an easier level",
                entry="where a run starts",
                comfortable="keep in the mix",
                mastered="little left to learn: promote, keep a small share")


def band_table(k: int = GROUP_SIZE) -> str:
    """The band table of docs/TRAINING.md, computed (a test compares the document with this)."""
    out = [f"| Band | Pass rate | Groups of {k} with both a pass and a failure | Use |", "|---|---|---|---|"]
    for name, lo, hi in BANDS:
        hi = min(hi, 1.0)
        shares = [mixed_share(lo + (hi - lo) * i / 400.0, k) for i in range(401)]
        rate = f"< {100 * hi:g} %" if lo == 0.0 else f"> {100 * lo:g} %" if hi == 1.0 else f"{100 * lo:g} - {100 * hi:g} %"
        mixed = f"< {100 * max(shares):.0f} %" if lo == 0.0 or hi == 1.0 else f"{100 * min(shares):.0f} - {100 * max(shares):.0f} %"
        out.append(f"| {name} | {rate} | {mixed} | {BAND_USE[name]} |")
    return "\n".join(out)


def results_path(env_id: str, kind: str, directory: str | None = None) -> str:
    """Where the stored ``kind`` ("learnability" or "calibration") result of an environment lives (package data)."""
    d = directory or str(importlib.resources.files("mrgym.envs").joinpath("results"))
    return os.path.join(d, env_id.replace("/", "-") + f".{kind}.json")


def load(env_id: str, kind: str, directory: str | None = None) -> dict | None:
    try:
        with open(results_path(env_id, kind, directory), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, OSError):
        return None


def write(result: dict, kind: str, directory: str | None = None) -> str:
    path = results_path(result["env_id"], kind, directory)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    return path


def stored(env_id: str) -> dict:
    """The ``learnability`` field of an environment card: nothing typed, only what the stored results hold.

    ``check``        verdict and summary rows of the stored ``mrgym check <id> --learnability --write`` (None: not run)
    ``calibration``  the stored rows of ``mrgym calibrate <id> ... --write`` (empty: no model was calibrated)
    ``trainable_by`` the models with a stored entry rung (a level in the entry band with enough mixed groups)
    """
    chk, cal = load(env_id, "learnability"), load(env_id, "calibration")
    rows = (cal or {}).get("calibration", [])
    return dict(
        check=None if not chk else dict(verdict=chk["verdict"], level=chk["level"], engine=chk["engine"], date_utc=chk["date_utc"],
                                        command=chk["command"], rows=chk["summary"]),
        calibration=rows,
        trainable_by=sorted({r["model"] for r in rows if r.get("entry_rung")}))
