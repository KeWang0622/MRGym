"""Oracle of ``Scanner/NoWrap``: the fastest geometry that shows the whole object without wrap-around, found on the
hidden world.

It knows what no agent knows (the outline of the object on the master grid) and uses nothing a policy uses: no
localiser, no image, no ruler. The scan time of the target scan depends on the number of phase-encode lines only, so
the search is over the direction of the phase-encode axis:

    1. directions: every whole degree, plus the direction across every edge of the convex hull of the object. The
       narrowest view of a convex shape has an edge of its hull flush with the field of view, so the hull edges make
       the search exact for a rotation that is a real number, not only on the degree raster
    2. per direction: the smallest even number of lines whose phase field of view holds the object when it is centred
       on it (fewer lines cannot hold it wherever the centre is), and the smallest readout matrix that holds it
    3. the candidates (two lines fewer, the smallest, two and four lines more, per direction) in ascending scan time,
       each put through the verifier's own checks, the noise-free re-scan included; the first that passes is the optimum

``value`` is 0: the error of this family is the relative excess scan time ``T / T* - 1`` and the optimum has none.
``scan_s`` is T* in seconds and ``lines`` its phase-encode lines; ``argmin`` is the geometry; ``certificate`` holds one
candidate per (number of lines, failed check) of those tried before it: the cheaper ones, and those of the same cost
at other rotations.

A geometry with ``pe_dir = "x"`` and rotation r is the geometry with ``pe_dir = "y"`` and rotation r + 90 degrees with
rows and columns exchanged; the search writes every direction with ``pe_dir = "y"``.

Imports ``task`` and the scanner only; a test reads the syntax tree and fails if this file imports the policies.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.spatial

from ...scanner import console, kspace
from ...scanner import world as W
from .. import oracle as O
from . import task

REGRET_KIND = "difference"          # error = T / T* - 1 and E* = 0: the regret of an answer is its relative excess scan time
ANGLE_STEP_DEG = 1.0
MIN_FOV_MM = console.SEQUENCES[task.TARGET["sequence"]]["fov_mm"][0]
MIN_MATRIX = console.SEQUENCES[task.TARGET["sequence"]]["matrix"][0]
MIN_PHASE_FOV = console.SEQUENCES[task.TARGET["sequence"]]["phase_fov"][0]


def _even_up(x: float) -> int:
    return 2 * int(math.ceil(x / 2.0 - 1e-9))


def hull(points: np.ndarray) -> np.ndarray:
    """Vertices of the convex hull of ``points`` [n, 2], in order around it."""
    return points[scipy.spatial.ConvexHull(points).vertices]


def directions(vertices: np.ndarray) -> list:
    """Rotations (degrees, in (-90, 90]) to try: the degree raster and the direction across every edge of the hull
    ``vertices``. An edge (dy, dx) is flush with the phase field of view when the image rows run along its normal."""
    edges = np.roll(vertices, -1, axis=0) - vertices
    across = np.degrees(np.arctan2(edges[:, 0], -edges[:, 1]))
    angles = list(np.arange(-90.0 + ANGLE_STEP_DEG, 90.0 + 1e-9, ANGLE_STEP_DEG)) + [float(a) for a in across]
    wrapped = {round(90.0 - (90.0 - a) % 180.0, 4) for a in angles}     # into (-90, 90]
    return sorted(wrapped)


def geometry(points: np.ndarray, rotation_deg: float, pixel_mm: float, grid_mm: float, extra_lines: int = 0) -> dict | None:
    """The smallest field of view turned by ``rotation_deg`` that holds ``points`` with the phase-encode axis along
    the image rows, plus ``extra_lines`` phase-encode lines. None when the console cannot realise it."""
    v, u = task.in_frame(points, (0.0, 0.0), rotation_deg)
    half = 0.5 * grid_mm * (abs(math.cos(math.radians(rotation_deg))) + abs(math.sin(math.radians(rotation_deg))))
    need_pe = float(v.max() - v.min()) + 2 * half - 2 * task.COVER_SLACK_MM
    need_ro = float(u.max() - u.min()) + 2 * half - 2 * task.COVER_SLACK_MM
    n_ro = max(_even_up(need_ro / pixel_mm), MIN_MATRIX, _even_up(MIN_FOV_MM / pixel_mm))
    n_pe = max(_even_up(need_pe / pixel_mm), 2) + extra_lines
    n_pe = max(n_pe, _even_up(MIN_PHASE_FOV * n_ro))
    n_ro = max(n_ro, n_pe)
    if n_pe < 2:
        return None
    mv, mu = 0.5 * float(v.max() + v.min()) + pixel_mm / 2, 0.5 * float(u.max() + u.min()) + pixel_mm / 2     # the image is half a pixel off-centre
    c, s = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
    return dict(pe_dir="y", fov_mm=round(n_ro * pixel_mm, 6), phase_fov=round(n_pe / n_ro, 6), matrix=int(n_ro),
                centre_y_mm=round(mv * c - mu * s, 3), centre_x_mm=round(mv * s + mu * c, 3), rotation_deg=float(rotation_deg),
                phase_oversampling=0.0)


def first_failure(cfg: dict, deliverable: dict, world: dict, engine, shapes: dict) -> str | None:
    """Name of the first check of the verifier that ``deliverable`` fails on the noise-free engine, or None."""
    if task.check(cfg, deliverable, world["console"]):
        return "deliverable_valid"
    checks = task.geometry_checks(cfg, deliverable, world, engine, shapes, first_failure=True)["checks"]
    return next((k for k, ok in checks.items() if not ok), None)


@O.timed
def solve(cfg: dict, world: dict | None = None, engine=None) -> dict:
    """The optimum of this instance (see ``mrgym.envs.oracle`` for the contract). Extra keys: ``scan_s`` (T*), ``lines``,
    ``directions`` (how many rotations were tried)."""
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    engine = engine or kspace.Engine(world)
    pixel, grid_mm = cfg["task"]["max_pixel_mm"], world["extent_mm"] / world["n"]
    shapes = dict(object=task.outline(world), support=task.outline(world, task.SUPPORT_THRESHOLD))
    vertices = hull(shapes["object"])                                    # the extremes of the object along any direction are hull vertices
    angles = directions(vertices)
    rows = []
    for a in angles:
        for extra in (-2, 0, 2, 4):
            g = geometry(vertices, a, pixel, grid_mm, extra)
            pl = console.plan(task.protocol_of(g), world["console"]) if g else None
            if pl and pl["accept"]:
                rows.append((pl["scan_s"], abs(a), extra, g))
    rows.sort(key=lambda r: r[:3])
    found = O.exhaustive_first(((g, t) for t, _, _, g in rows), lambda d: first_failure(cfg, d, world, engine, shapes), keep=len(rows))
    if found["value"] is None:
        raise ValueError("no geometry of the search passes the verifier's checks on this world")
    best = found["argmin"]
    plan = console.plan(task.protocol_of(best), world["console"])
    seen = {}
    for c in found["certificate"]:                                       # one entry per (lines, failed check): what was tried before, and why not
        lines = console.plan(task.protocol_of(c["deliverable"]), world["console"])["realised"]["lines"]
        seen.setdefault((lines, c["fails"]), dict(c, lines=lines))
    return dict(value=0.0, argmin=best, method="exhaustive", scan_s=float(found["value"]), lines=int(plan["realised"]["lines"]),
                grid=dict(axes=dict(rotation_deg=dict(start=-90.0 + ANGLE_STEP_DEG, stop=90.0, step=ANGLE_STEP_DEG, plus="the direction across every hull edge"),
                                    lines="the smallest that holds the object centred, and -2, +2, +4"),
                          size=len(rows), evaluated=found["grid"]["evaluated"]),
                certificate=list(seen.values()), directions=len(angles))
