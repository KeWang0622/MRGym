"""Oracle of ``Scanner/T1Transmit``: the deliverable with the smallest error, read from the hidden world.

The error of this family is the largest ``|ln(T1_delivered / T1_true)|`` over the marked regions. The true T1 of
every material is in the hidden world, so the optimum needs no search: the deliverable that holds the true values has
error 0 (``method = "exact"``). ``window_ms`` is the pass window of each region (T1 / (1 + tolerance) to T1 x (1 +
tolerance)), and ``certificate`` the two values just outside each window with the check they fail.

What this oracle is not: a bound on what a measurement can reach within the budget. The smallest error any protocol
can deliver at the hidden noise level (a Cramer-Rao bound over the protocol space) is not computed here; the regret
of a policy against this oracle is its plain error.

Imports ``task`` and the scanner only; a test reads the syntax tree and fails if this file imports the policies.
"""
from __future__ import annotations

import math

from ...scanner import world as W
from .. import oracle as O
from . import task

REGRET_KIND = "difference"          # E* = 0: the regret of an answer is its error
OUTSIDE = 1.001                     # the certificate steps this factor beyond each end of a window


@O.timed
def solve(cfg: dict, world: dict | None = None) -> dict:
    """The optimum of this instance (see ``mrgym.envs.oracle`` for the contract). Extra keys: ``window_ms`` per region."""
    world = world or W.make_world(cfg["seed"], cfg["world_spec"], layout_only=True)
    truth, f = task.true_t1_ms(cfg, world), 1.0 + cfg["task"]["tolerance"]
    best = {"t1_ms": dict(truth)}
    window = {i: [t / f, t * f] for i, t in truth.items()}
    certificate = [dict(deliverable={"t1_ms": {i: v}}, fails=f"t1_within_tolerance_{i}")
                   for i, (lo, hi) in window.items() for v in (lo / OUTSIDE, hi * OUTSIDE)]
    return dict(value=float(task.score(cfg, best, world)["error"]), argmin=best, method="exact",
                grid=dict(axes={}, size=1, evaluated=1), certificate=certificate, window_ms=window)


def miss(cfg: dict, answer: dict, solution: dict) -> dict:
    """How far a deliverable is from the optimum, per region, in units of the tolerance: at most 1 inside the pass
    window, above 1 outside. ``{region: miss}``; None for a region the answer does not give."""
    limit, out = math.log(1.0 + cfg["task"]["tolerance"]), {}
    for i, t in solution["argmin"]["t1_ms"].items():
        v = (answer.get("t1_ms") or {}).get(i)
        out[i] = abs(math.log(float(v) / t)) / limit if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else None
    return out
