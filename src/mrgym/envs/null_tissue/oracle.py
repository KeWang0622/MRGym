"""Oracle of ``Scanner/NullTissue``: the inversion time that nulls each marked region, found on the hidden world.

It knows what no agent knows (the world) and uses nothing a policy uses: no probe scans, no fit, no model of the
recovery. It evaluates the verifier's own target scan on the noise-free engine and searches the deliverable:

    1. one pass over a coarse grid of inversion times between the console minimum and the end of the repetition
       time; each scan gives the signed signal of every marked region, relative to the scan without the inversion
    2. per region, the sign change is located to the console raster (``TI_STEP_MS``)
    3. per region, the two inversion times at which the residual signal reaches the pass threshold: the pass window

``value`` is E*, the residual signal of the best deliverable (largest over the regions), ``argmin`` that deliverable,
``window_ms`` the pass window of each region (the tolerance of the task in milliseconds, which the task text states
only as a signal ratio), and ``certificate`` the raster neighbours just outside each window with the check they fail.

Imports ``task`` and the scanner only; a test reads the syntax tree and fails if this file imports the policies.
"""
from __future__ import annotations

import numpy as np

from ...scanner import console, kspace
from ...scanner import world as W
from .. import family as F
from .. import oracle as O
from . import task

TI_STEP_MS = 0.01                   # the console's time raster: a finer deliverable is rounded up to it
COARSE_POINTS = 32
REGRET_KIND = "difference"          # E* is close to 0: the regret of an answer is its residual signal minus E*


class _Target:
    """The target scan of an instance on the noise-free engine: signed residual signal of every marked region."""

    def __init__(self, cfg: dict, world: dict, engine):
        self.cfg, self.world, self.engine, self.evaluated = cfg, world, engine, 0
        self.regions = cfg["task"]["regions"]
        self.ids = [r["id"] for r in self.regions]
        plan = console.plan(task.target_protocol(cfg["task"], 100.0), world["console"])
        self.lo = plan["timing"]["ti_min_ms"] + TI_STEP_MS
        self.hi = cfg["task"]["target_scan"]["tr_ms"] - plan["realised"]["te_ms"] - 20.0
        se = console.plan(dict(task.GEOMETRY, sequence="se", tr_ms=plan["realised"]["tr_ms"], te_ms=plan["realised"]["te_ms"]),
                          world["console"])["realised"]
        self.without = F.region_complex(engine, se, self.regions)
        self._memo = {}

    def signed(self, ti_ms: float) -> dict:
        """{region: residual signal with its sign} of the target scan at ``ti_ms`` (rounded up to the raster)."""
        key = round(float(ti_ms), 6)
        if key not in self._memo:
            realised = console.plan(task.target_protocol(self.cfg["task"], key), self.world["console"])["realised"]
            with_inversion = F.region_complex(self.engine, realised, self.regions)
            self._memo[key] = {i: float((with_inversion[i] / self.without[i]).real) for i in self.ids}
            self.evaluated += 1
        return self._memo[key]


def _raster(ti_ms: float) -> float:
    return round(round(ti_ms / TI_STEP_MS) * TI_STEP_MS, 2)


@O.timed
def solve(cfg: dict, world: dict | None = None) -> dict:
    """The optimum of this instance (see ``mrgym.envs.oracle`` for the contract). Extra keys: ``null_ms`` and
    ``window_ms`` per region, ``unique`` (one sign change per region on the coarse grid)."""
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    target = _Target(cfg, world, kspace.Engine(world))
    eps = cfg["task"]["null_max"]
    grid = np.linspace(target.lo, target.hi, COARSE_POINTS).tolist()
    coarse = [target.signed(t) for t in grid]
    null, window, unique, certificate = {}, {}, True, []
    for i in target.ids:
        s = [c[i] for c in coarse]
        changes = O.sign_changes(s)
        unique = unique and len(changes) == 1
        if not changes:
            best = O.exhaustive_min(grid, lambda t, i=i: abs(target.signed(t)[i]))      # no null inside the range
            null[i], window[i] = _raster(best["argmin"]), None
            continue
        k = changes[0]
        rising = s[k + 1] > s[k]
        root, _ = O.refine_root(lambda t, i=i: target.signed(t)[i], grid[k], grid[k + 1], TI_STEP_MS / 2)
        null[i] = _raster(root)
        edges = []
        for side, level in ((-1, -eps if rising else eps), (1, eps if rising else -eps)):
            j = k if side < 0 else k + 1                                              # walk outwards to bracket the threshold
            while 0 <= j < len(grid) and abs(s[j]) <= eps:
                j += side
            far = grid[j] if 0 <= j < len(grid) else (target.lo if side < 0 else target.hi)
            f = lambda t, i=i, level=level: target.signed(t)[i] - level                  # noqa: E731
            if (f(far) > 0) == (f(null[i]) > 0):
                edges.append(far)                                                     # the window runs to the end of the range
            else:
                lo, hi = (far, null[i]) if side < 0 else (null[i], far)
                edges.append(O.refine_root(f, lo, hi, TI_STEP_MS / 2)[0])
        inside = [_raster(edges[0] + TI_STEP_MS), _raster(edges[1] - TI_STEP_MS)]
        window[i] = inside
        for ti in (_raster(inside[0] - 2 * TI_STEP_MS), _raster(inside[1] + 2 * TI_STEP_MS)):
            if target.lo <= ti <= target.hi and abs(target.signed(ti)[i]) > eps:
                certificate.append(dict(deliverable={"ti_ms": {i: ti}}, fails=f"nulled_{i}"))
    argmin = {"ti_ms": dict(null)}
    value = task.score(cfg, argmin, world, target.engine)["error"]            # E*, by the verifier's own definition of the error
    return dict(value=float(value), argmin=argmin, method="exhaustive",
                grid=dict(axes=dict(ti_ms=dict(start=round(target.lo, 2), stop=round(target.hi, 2), coarse_points=COARSE_POINTS,
                                               refined_to_ms=TI_STEP_MS)),
                          size=int(round((target.hi - target.lo) / TI_STEP_MS)) + 1, evaluated=target.evaluated),
                certificate=certificate, null_ms=null, window_ms=window, unique=unique)


def miss(cfg: dict, answer: dict, solution: dict) -> dict:
    """How far a deliverable is from the optimum, per region, in units of the half pass window on its side: at most 1
    inside the window, above 1 outside. ``{region: miss}``; None for a region the answer does not give."""
    out = {}
    for i, t0 in solution["null_ms"].items():
        ti, w = (answer.get("ti_ms") or {}).get(i), solution["window_ms"].get(i)
        if not isinstance(ti, (int, float)) or w is None:
            out[i] = None
            continue
        half = (t0 - w[0]) if ti < t0 else (w[1] - t0)
        out[i] = abs(float(ti) - t0) / half if half > 0 else None
    return out
