"""Oracle of ``Scanner/Resolve``: per pair, the fastest geometry that shows the whole object and separates the pair,
found on the hidden world.

It knows what no agent knows (the outline of the object on the master grid, where the two features of a pair are) and
uses nothing a policy uses: no localiser, no image, no ruler. The scan time of the target scan depends on the number
of phase-encode lines only, so the search is over the lines:

    1. per phase-encode axis and per even number of lines n, the geometry with the SMALLEST pixel whose phase field
       of view still holds the object (``geometry``): a coarser pixel with the same lines separates worse and a finer
       one cuts the object off, so if this geometry fails, n lines along this axis cannot pass
    2. the candidates of both axes in ascending scan time, each put through the verifier's own checks, the noise-free
       re-scan included; the first that passes is the optimum
    3. the search starts where the pixel is ``START_PIXEL`` (0.7) times the spacing of the pair: coarser pixels show
       no dip (two points 1 / 0.7 = 1.43 pixels apart have a dip of 0.14, ``task.point_pair_dip``; a test checks the
       engine there). ``certificate`` holds every candidate from there to the optimum with the check it fails

``value`` is 0: the error of this family is the relative excess scan time ``T / T* - 1`` and the optimum has none.
``pairs[id]`` holds T* (``scan_s``), its lines and pixel and the geometry; ``argmin`` is the deliverable.

Imports ``task`` and the scanner only; a test reads the syntax tree and fails if this file imports the policies.
"""
from __future__ import annotations

import math

import numpy as np

from ...scanner import console, kspace
from ...scanner import world as W
from .. import oracle as O
from ..nowrap import task as N
from . import task

REGRET_KIND = "difference"          # error = T / T* - 1 and E* = 0: the regret of an answer is its relative excess scan time
START_PIXEL = 0.7                   # the search starts at pixel = this share of the spacing (see the module docstring)
_SEQ = console.SEQUENCES[task.TARGET["sequence"]]
MIN_FOV_MM, MAX_FOV_MM = _SEQ["fov_mm"][:2]
MIN_MATRIX, MAX_MATRIX = _SEQ["matrix"][:2]
MIN_PHASE_FOV = _SEQ["phase_fov"][0]


def _even_up(x: float) -> int:
    return 2 * int(math.ceil(x / 2.0 - 1e-9))


def extent(outline: np.ndarray, grid_mm: float) -> dict:
    """Low and high end of the object along y and along x (mm from the isocentre), as the verifier's coverage check
    counts them: the outermost outline points, half a grid cell added, the slack of the check taken off."""
    out = {}
    for axis, col in (("y", 0), ("x", 1)):
        out[axis] = (float(outline[:, col].min()) - grid_mm / 2 + task.COVER_SLACK_MM, float(outline[:, col].max()) + grid_mm / 2 - task.COVER_SLACK_MM)
    return out


def smallest_pixel(lo: float, hi: float, n: int) -> float:
    """The smallest pixel p for which n pixels about the isocentre hold [lo, hi]: the image reaches from
    -n p / 2 - p / 2 to n p / 2 - p / 2."""
    return max(2.0 * hi / (n - 1), -2.0 * lo / (n + 1))


def geometry(box: dict, pe_dir: str, lines: int) -> dict | None:
    """The geometry with ``lines`` phase-encode lines along ``pe_dir`` and the smallest pixel that holds the object
    (``box``: ``extent``). None when the console cannot realise it."""
    pe, ro = (box["y"], box["x"]) if pe_dir == "y" else (box["x"], box["y"])
    p = smallest_pixel(*pe, lines) * (1.0 + 2e-6)                       # the field of view is written with six decimals
    matrix = None
    for m in range(max(lines, MIN_MATRIX, _even_up(MIN_FOV_MM / p)), MAX_MATRIX + 1, 2):   # the readout holds the object as well
        if m * p >= max(2.0 * ro[1] + p, -2.0 * ro[0] - p):
            matrix = m
            break
    if matrix is None or matrix * p > MAX_FOV_MM or lines / matrix < MIN_PHASE_FOV:
        return None
    return dict(pe_dir=pe_dir, fov_mm=round(matrix * p, 6), phase_fov=round(lines / matrix, 6), matrix=int(matrix))


def first_failure(cfg: dict, pair: str, g: dict, world: dict, engine, outline: np.ndarray) -> str | None:
    """Name of the first of requirements 1 to 4 that the geometry ``g`` fails for ``pair`` on the noise-free engine, or None."""
    checks = task.pair_checks(cfg, pair, g, world, engine, outline, first_failure=True)["checks"]
    return next((k for k, ok in checks.items() if not ok), None)


def solve_pair(cfg: dict, pair: str, world: dict, engine, outline: np.ndarray) -> dict:
    """The optimum of one pair: dict(scan_s, lines, pixel_mm, geometry, certificate, evaluated)."""
    box = extent(outline, world["extent_mm"] / world["n"])
    spacing = cfg["hidden"]["pairs"][pair]["spacing_mm"]
    rows = []
    for pe_dir in ("y", "x"):
        lo, hi = box[pe_dir]
        n = max(_even_up((hi - lo) / (START_PIXEL * spacing)) - 2, 4)
        while smallest_pixel(lo, hi, n) > START_PIXEL * spacing:
            n += 2
        for lines in range(n, MAX_MATRIX + 1, 2):
            g = geometry(box, pe_dir, lines)
            pl = console.plan(task.protocol_of(g), world["console"]) if g else None
            if pl and pl["accept"] and pl["realised"]["lines"] == lines:
                rows.append((pl["scan_s"], pe_dir, g))
            if pl and pl["scan_s"] > task.MAX_SCAN_S:
                break
    rows.sort(key=lambda r: (r[0], r[1] != "y"))
    found = O.exhaustive_first(((g, t) for t, _, g in rows), lambda g: first_failure(cfg, pair, g, world, engine, outline), keep=len(rows))
    if found["value"] is None:
        raise ValueError(f"no geometry of the search meets the requirements for pair {pair} on this world")
    best = found["argmin"]
    plan = console.plan(task.protocol_of(best), world["console"])
    return dict(scan_s=float(found["value"]), lines=int(plan["realised"]["lines"]), pixel_mm=round(best["fov_mm"] / best["matrix"], 4),
                geometry=best, certificate=found["certificate"], evaluated=found["grid"]["evaluated"], size=len(rows))


@O.timed
def solve(cfg: dict, world: dict | None = None, engine=None) -> dict:
    """The optimum of this instance (see ``mrgym.envs.oracle`` for the contract). Extra key: ``pairs`` (per pair: T* as
    ``scan_s``, ``lines``, ``pixel_mm``, ``geometry``)."""
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    engine = engine or kspace.Engine(world)
    outline = N.outline(world)
    pairs = {i: solve_pair(cfg, i, world, engine, outline) for i in cfg["task"]["pairs"]}
    return dict(value=0.0, argmin={i: p["geometry"] for i, p in pairs.items()}, method="exhaustive",
                grid=dict(axes=dict(pe_dir=["y", "x"], lines=f"every even number from pixel = {START_PIXEL:g} x spacing upward, "
                                                              "with the smallest pixel that holds the object"),
                          size=sum(p["size"] for p in pairs.values()), evaluated=sum(p["evaluated"] for p in pairs.values())),
                certificate=[dict(c, pair=i) for i, p in pairs.items() for c in p["certificate"]],
                pairs={i: dict(scan_s=p["scan_s"], lines=p["lines"], pixel_mm=p["pixel_mm"], geometry=p["geometry"]) for i, p in pairs.items()})
