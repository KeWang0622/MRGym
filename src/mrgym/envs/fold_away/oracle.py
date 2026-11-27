"""Oracle of ``Scanner/FoldAway``: per region, the fastest geometry in which no aliased signal lands on the region,
found on the hidden world.

It knows what no agent knows (the outline of the object on the master grid) and uses nothing a policy uses: no
localiser, no image, no ruler. The scan time of the target scan depends on the number of phase-encode lines only, so
the search is over the direction of the phase-encode axis and the lines:

    1. directions: the two axes of an upright field of view; at level 3 every ``ANGLE_STEP_DEG`` of rotation (a
       geometry with ``pe_dir = "x"`` and rotation r is the geometry with ``pe_dir = "y"`` and rotation r + 90 degrees
       with rows and columns exchanged; the search writes every turned direction with ``pe_dir = "y"``)
    2. per direction: the fold-free extent of the region from the outline, line by line (``extent``:
       ``task.fold_free_extent`` on every line through the region, with the length of the circle on that line), and
       from it the smallest number of lines worth trying: the extent minus ``-START_MARGIN_PX`` pixels. With less,
       a copy of the object covers a strip of the region two pixels deep and with it several pixel centres, each
       changed by half of the signal or more (a test checks that such a geometry fails)
    3. the candidates (direction, lines) in ascending scan time, each put through the verifier's own checks, the
       noise-free re-scan and its twin included; the first that passes is the optimum. The field of view is centred
       on the region: the aliasing depends on the phase field of view alone, the centre only has to keep the region
       inside the image

``value`` is 0: the error of this family is the relative excess scan time ``T / T* - 1`` and the optimum has none.
``regions[id]`` holds T* (``scan_s``), its lines and the geometry; ``argmin`` is the deliverable.

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
ANGLE_STEP_DEG = 3.0
START_MARGIN_PX = -2.0              # the search starts at the fold-free extent plus this many pixels (see the module docstring)
PASS_MARGIN_PX = 2.0                # a geometry this many pixels above the fold-free extent is taken to pass (tested); it bounds
TURNED_WITHIN_PX = PASS_MARGIN_PX - START_MARGIN_PX     # the directions of level 3 that can hold the optimum: within this of the smallest extent
MIN_PHASE_FOV = console.SEQUENCES[task.TARGET["sequence"]]["phase_fov"][0]
MIN_LINES = max(2, int(round(task.TARGET["matrix"] * MIN_PHASE_FOV / 2) * 2))     # as the console rounds the smallest phase field of view


def _even_up(x: float) -> int:
    return 2 * int(math.ceil(x / 2.0 - 1e-9))


def extent(points: np.ndarray, region: dict, pe_dir: str, rotation_deg: float, grid_mm: float) -> float:
    """The fold-free extent (mm) of the circle ``region`` along the phase-encode axis of a field of view turned by
    ``rotation_deg``: the largest, over the lines through the region, of ``task.fold_free_extent`` with the ends of
    the object and of the circle on that line. ``points``: outline of the object on the master grid (a point stands
    for a grid cell). A point at (a along the axis, c across it) from the centre of the region, with the circle
    reaching +-h = sqrt(radius^2 - c^2) on its line, asks for a + h when it is the high end and h - a when the low end."""
    v, u = N.in_frame(points, (region["y_mm"], region["x_mm"]), rotation_deg)
    along, across = (v, u) if pe_dir == "y" else (u, v)
    half = 0.5 * grid_mm * (abs(math.cos(math.radians(rotation_deg))) + abs(math.sin(math.radians(rotation_deg))))
    on = np.abs(across) <= region["radius_mm"]
    h = np.sqrt(region["radius_mm"] ** 2 - across[on] ** 2)
    return float(np.max(np.abs(along[on]) + h)) + half


def directions(cfg: dict) -> list:
    """(pe_dir, rotation_deg) of every direction of the search."""
    if not cfg["task"]["rotation"]:
        return [("y", 0.0), ("x", 0.0)]
    return [("y", float(a)) for a in np.arange(-90.0 + ANGLE_STEP_DEG, 90.0 + 1e-9, ANGLE_STEP_DEG)]


def geometry(cfg: dict, region: dict, pe_dir: str, rotation_deg: float, lines: int) -> dict:
    """The geometry with ``lines`` phase-encode lines along a direction, centred on the region."""
    g = dict(pe_dir=pe_dir, phase_fov=round(max(lines / task.TARGET["matrix"], MIN_PHASE_FOV), 6),
             centre_y_mm=round(float(region["y_mm"]), 3), centre_x_mm=round(float(region["x_mm"]), 3))
    if cfg["task"]["rotation"]:
        g["rotation_deg"] = float(rotation_deg)
    return g


def first_failure(cfg: dict, region: dict, g: dict, world: dict, engine, support: np.ndarray) -> str | None:
    """Name of the first of requirements 1 and 2 that the geometry ``g`` fails for ``region`` on the noise-free engine, or None."""
    checks = task.region_checks(cfg, region, g, world, engine, support, first_failure=True)["checks"]
    return next((k for k, ok in checks.items() if not ok), None)


def candidates(cfg: dict, region: dict, outline: np.ndarray, grid_mm: float, exam: dict) -> list:
    """(scan_s, fold-free extent, geometry) of every candidate of one region, in ascending scan time."""
    px, top = task.PIXEL_MM, task.TARGET["matrix"]
    dirs = [(d, r, extent(outline, region, d, r, grid_mm)) for d, r in directions(cfg)]
    if cfg["task"]["rotation"]:
        least = min(e for _, _, e in dirs)
        dirs = [x for x in dirs if x[2] <= least + TURNED_WITHIN_PX * px]
    rows = []
    for d, r, e in dirs:
        for lines in range(max(MIN_LINES, _even_up((e + START_MARGIN_PX * px) / px)), top + 1, 2):
            g = geometry(cfg, region, d, r, lines)
            pl = console.plan(task.protocol_of(g), exam)
            if pl["accept"] and pl["realised"]["lines"] == lines:
                rows.append((pl["scan_s"], e, g))
    return sorted(rows, key=lambda x: x[:2])


def solve_region(cfg: dict, region: dict, world: dict, engine, outline: np.ndarray, support: np.ndarray) -> dict:
    """The optimum of one region: dict(scan_s, lines, geometry, certificate, evaluated, size)."""
    rows = candidates(cfg, region, outline, world["extent_mm"] / world["n"], world["console"])
    found = O.exhaustive_first(((g, t) for t, _, g in rows), lambda g: first_failure(cfg, region, g, world, engine, support), keep=len(rows))
    if found["value"] is None:
        raise ValueError(f"no geometry of the search meets the requirements for region {region['id']} on this world")
    best = found["argmin"]
    plan = console.plan(task.protocol_of(best), world["console"])
    return dict(scan_s=float(found["value"]), lines=int(plan["realised"]["lines"]), geometry=best, certificate=found["certificate"],
                evaluated=found["grid"]["evaluated"], size=len(rows))


@O.timed
def solve(cfg: dict, world: dict | None = None, engine=None) -> dict:
    """The optimum of this instance (see ``mrgym.envs.oracle`` for the contract). Extra key: ``regions`` (per region: T*
    as ``scan_s``, ``lines``, ``geometry``)."""
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    engine = engine or kspace.Engine(world)
    outline, support = N.outline(world), N.outline(world, N.SUPPORT_THRESHOLD)
    out = {r["id"]: solve_region(cfg, r, world, engine, outline, support) for r in cfg["task"]["regions"]}
    turned = cfg["task"]["rotation"]
    return dict(value=0.0, argmin={i: p["geometry"] for i, p in out.items()}, method="exhaustive",
                grid=dict(axes=dict(direction=(dict(start=-90.0 + ANGLE_STEP_DEG, stop=90.0, step=ANGLE_STEP_DEG,
                                                    kept=f"fold-free extent within {TURNED_WITHIN_PX:g} pixels of the smallest") if turned
                                               else ["y", "x"]),
                                    lines=f"every even number from the fold-free extent - {-START_MARGIN_PX:g} pixels upward"),
                          size=sum(p["size"] for p in out.values()), evaluated=sum(p["evaluated"] for p in out.values())),
                certificate=[dict(c, region=i) for i, p in out.items() for c in p["certificate"]],
                regions={i: dict(scan_s=p["scan_s"], lines=p["lines"], geometry=p["geometry"]) for i, p in out.items()})
