"""Control policies of ``Scanner/NoWrap``. Every policy plays through the agent's interface only (``info`` / ``plan`` /
``scan`` / ``view`` / ``budget``) and returns the deliverable. None reads the hidden world.

The ruler. A language model measures the object with ``profile``. A scripted policy reads the stored image in process
(``view(...)["array"]`` and ``["bounding_box"]``, see ``mrgym.policies.ToolScanner``): the pixels brighter than a
fifth of the 99.5th percentile, isolated ones dropped.

Must pass
  reference        one localiser; the bright pixels are projected onto every direction in half-degree steps; the
                   field of view is turned so that the phase-encode axis lies along the narrowest one, centred on the
                   object, with ``MARGIN_MM`` on each side
Must fail
  prior            no scan; the geometry that is safe for any object
  console_default  one token scan, then the console's default field of view, centred, phase along y
  fixed_recipe     one token scan, then the best geometry that ignores the object: one constant phase axis, rotation
                   and number of lines, centred on the isocentre, tuned on other worlds
  brute_force      shrinks a centred square field of view scan by scan until the object touches its border; never
                   moves, turns or shortens it
  wrong_units      the reference's geometry with every length in centimetres
Ablations (the reference with one decision removed; each must lose at the levels where that decision matters)
  size_only        the right size and direction, but the field of view stays on the isocentre         (levels 2, 3)
  centred_only     the field of view stays on the isocentre and is enlarged until it holds the object: a valid image,
                   too slow; the answer of ``Scanner/NoWrap-v0``                                       (levels 2, 3)
  no_swap          moves the field of view but always phase-encodes along y and never turns it         (levels 2, 3)
  no_turn          moves the field of view but never turns it: the shorter side of the upright bounding box (level 3)
Reported
  open_loop        a fixed scan list (the one localiser) and a fixed estimator: this is the reference itself, because
                   the reference chooses no scan from a measurement. By the rule of this package the family is a
                   measurement task, not a decision task
  safe_max         one scan, then the safe geometry: valid and slow
"""
from __future__ import annotations

import math

import numpy as np

from . import task

REFERENCE = "reference"
MUST_FAIL = ("prior", "console_default", "fixed_recipe", "brute_force", "wrong_units")
ABLATIONS = {"size_only": (2, 3), "centred_only": (2, 3), "no_swap": (2, 3), "no_turn": (3,)}
REPORTED = ("open_loop", "safe_max")
DESCRIPTIONS = dict(
    reference="one localiser; phase-encode axis along the narrowest direction of the bright pixels, field of view centred on them",
    prior="no scan; the geometry that is safe for any object",
    console_default="one token scan, then the console's default field of view, centred, phase along y",
    fixed_recipe="one token scan, then one constant geometry (axis, rotation, lines; centred) tuned on other worlds",
    brute_force="shrinks a centred square field of view scan by scan until the object touches its border",
    wrong_units="the reference's geometry with every length in centimetres",
    size_only="the right size and direction, with the field of view left on the isocentre",
    centred_only="the field of view left on the isocentre and enlarged until it holds the object (the v0 answer)",
    no_turn="moves the field of view but never turns it",
    no_swap="moves the field of view but always phase-encodes along y and never turns it",
    open_loop="a fixed scan list and a fixed estimator: the reference itself (no scan depends on a measurement)",
    safe_max="one scan, then the safe geometry",
)
LOCALISER = dict(task.LOCALISER)
MARGIN_MM = 1.5                             # kept on each side between the measured edge and the edge of the image
SAFE_FOV_MM = 260.0                         # holds any object the scanner's bore can hold
ANGLE_STEP_DEG = 0.5
BRIGHT = 0.2                                # of the 99.5th percentile of the localiser


def _px(card):
    return card["task"]["max_pixel_mm"]


def _even_up(x):
    return 2 * int(math.ceil(x / 2.0 - 1e-9))


def _safe(card):
    n = _even_up(SAFE_FOV_MM / _px(card))
    return dict(pe_dir="y", fov_mm=round(n * _px(card), 6), phase_fov=1.0, matrix=n, centre_y_mm=0.0, centre_x_mm=0.0, rotation_deg=0.0)


def _look(sc, protocol=LOCALISER):
    """One localiser. Returns (coordinates [n, 2] (y, x) in mm of its bright pixels, (pixel_y, pixel_x) in mm, the
    upright bounding box (y_lo, y_hi, x_lo, x_hi)), or None when nothing is bright or the object touches the border."""
    v = sc.view(sc.scan(protocol)["scan_index"])
    box = v["bounding_box"]
    if not box or box["touches_edge"]:
        return None
    img = np.asarray(v["array"], float)
    ny, nx = img.shape
    m = img > BRIGHT * float(np.percentile(img, 99.5))
    pad = np.pad(m.astype(np.int8), 1)
    m = sum(pad[i:i + ny, j:j + nx] for i in range(3) for j in range(3)) >= 6        # drop isolated noise pixels
    (fy, fx), (cy, cx) = v["fov_yx_mm"], v["centre_yx_mm"]
    iy, ix = np.nonzero(m)
    pts = np.stack([cy + (iy - ny // 2) * (fy / ny), cx + (ix - nx // 2) * (fx / nx)], 1)
    return pts, (fy / ny, fx / nx), (*box["y_mm"], *box["x_mm"])


def _extent(pts, pixel, rotation_deg):
    """Edges of the bright pixels along the rows and columns of a field of view turned by ``rotation_deg`` about the
    isocentre: ((v_lo, v_hi), (u_lo, u_hi)), outer pixel edges."""
    c, s = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
    v, u = pts[:, 0] * c + pts[:, 1] * s, -pts[:, 0] * s + pts[:, 1] * c
    hv, hu = 0.5 * (pixel[0] * abs(c) + pixel[1] * abs(s)), 0.5 * (pixel[0] * abs(s) + pixel[1] * abs(c))
    return (float(v.min()) - hv, float(v.max()) + hv), (float(u.min()) - hu, float(u.max()) + hu)


def _geometry(rows, cols, rotation_deg, px, pe_dir="y", margin=MARGIN_MM, centred=True):
    """The smallest field of view turned by ``rotation_deg`` that reaches from ``rows[0]`` to ``rows[1]`` along the image
    rows and ``cols[0]`` to ``cols[1]`` along the columns (mm, in the turned frame about the isocentre), ``margin``
    to spare on each side. ``centred`` False: the field of view stays on the isocentre and keeps that size."""
    pe, ro = (rows, cols) if pe_dir == "y" else (cols, rows)
    n_pe = _even_up((pe[1] - pe[0] + 2 * margin) / px)
    n_ro = max(_even_up((ro[1] - ro[0] + 2 * margin) / px), n_pe, 40)
    n_pe = max(n_pe, _even_up(0.25 * n_ro))
    mv, mu = (0.5 * (rows[0] + rows[1]) + px / 2, 0.5 * (cols[0] + cols[1]) + px / 2) if centred else (0.0, 0.0)
    c, s = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
    return dict(pe_dir=pe_dir, fov_mm=round(n_ro * px, 6), phase_fov=round(n_pe / n_ro, 6), matrix=int(n_ro),
                centre_y_mm=round(mv * c - mu * s, 2), centre_x_mm=round(mv * s + mu * c, 2), rotation_deg=float(rotation_deg))


def _narrowest(pts, pixel):
    """(rotation, rows, cols) of the direction along which the bright pixels are narrowest."""
    best = None
    for a in np.arange(-90.0 + ANGLE_STEP_DEG, 90.0 + 1e-9, ANGLE_STEP_DEG):
        rows, cols = _extent(pts, pixel, float(a))
        if best is None or rows[1] - rows[0] < best[1][1] - best[1][0]:
            best = (float(a), rows, cols)
    return best


def reference(sc):
    card = sc.info()
    seen = _look(sc)
    if seen is None:
        return _safe(card)
    rotation, rows, cols = _narrowest(seen[0], seen[1])
    return _geometry(rows, cols, rotation, _px(card))


open_loop = reference                       # the reference is a fixed scan list and a fixed estimator


def prior(sc):
    return _safe(sc.info())


def safe_max(sc):
    card = sc.info()
    sc.scan(LOCALISER)
    return _safe(card)


def console_default(sc):
    card = sc.info()
    sc.scan(LOCALISER)
    n = _even_up(card["default_fov_mm"] / _px(card))
    return dict(_safe(card), fov_mm=round(n * _px(card), 6), matrix=n)


def fixed_recipe(recipe):
    def policy(sc):
        card = sc.info()
        sc.scan(LOCALISER)
        n = _even_up(SAFE_FOV_MM / _px(card))
        return dict(pe_dir=recipe["pe_dir"], fov_mm=round(n * _px(card), 6), phase_fov=round(recipe["lines"] / n, 6), matrix=n,
                    centre_y_mm=0.0, centre_x_mm=0.0, rotation_deg=float(recipe["rotation_deg"]))
    return policy


DEFAULT_RECIPE = dict(pe_dir="y", lines=64, rotation_deg=0.0)


def brute_force(sc):
    """Centred square fields of view from large to small with a cheap scan each; keeps the last one the object did
    not touch."""
    card = sc.info()
    px, best = _px(card), SAFE_FOV_MM
    for fov in (240.0, 216.0, 192.0, 168.0, 144.0, 120.0):
        prot = dict(sequence="spgr", tr_ms=11.0, flip_deg=15.0, fov_mm=fov, matrix=64)
        p = sc.plan(prot)
        if not p["accept"] or p["scan_s"] > sc.budget()["remaining_s"]:
            break
        box = sc.view(sc.scan(prot)["scan_index"])["bounding_box"]
        if not box or box["touches_edge"]:
            break
        best = fov
    n = _even_up(best / px)
    return dict(_safe(card), fov_mm=round(n * px, 6), matrix=n)


def wrong_units(sc):
    a = reference(sc)
    return dict(a, fov_mm=a["fov_mm"] / 10.0, centre_y_mm=a["centre_y_mm"] / 10.0, centre_x_mm=a["centre_x_mm"] / 10.0)


def _ablation(build):
    def policy(sc):
        card = sc.info()
        seen = _look(sc)
        return _safe(card) if seen is None else build(seen, _px(card))
    return policy


def _size_only(seen, px):
    rotation, rows, cols = _narrowest(seen[0], seen[1])
    return _geometry(rows, cols, rotation, px, centred=False)


def _centred_only(seen, px):
    y_lo, y_hi, x_lo, x_hi = seen[2]
    hy, hx = max(abs(y_lo), abs(y_hi)), max(abs(x_lo), abs(x_hi))
    return _geometry((-hy, hy), (-hx, hx), 0.0, px, "y" if hy <= hx else "x", centred=False)


def _no_turn(seen, px):
    y_lo, y_hi, x_lo, x_hi = seen[2]
    return _geometry((y_lo, y_hi), (x_lo, x_hi), 0.0, px, "y" if y_hi - y_lo <= x_hi - x_lo else "x")


def _no_swap(seen, px):
    y_lo, y_hi, x_lo, x_hi = seen[2]
    return _geometry((y_lo, y_hi), (x_lo, x_hi), 0.0, px, "y")


size_only, centred_only, no_turn, no_swap = (_ablation(f) for f in (_size_only, _centred_only, _no_turn, _no_swap))


def policies(recipe=None):
    return dict(reference=reference, prior=prior, console_default=console_default, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE),
                brute_force=brute_force, wrong_units=wrong_units, size_only=size_only, centred_only=centred_only, no_swap=no_swap,
                no_turn=no_turn, open_loop=open_loop, safe_max=safe_max)


def recipe_candidates():
    """The constant geometries a recipe is chosen from: phase axis along one of twelve directions 15 degrees apart,
    every even number of lines from 44 to 128, centred on the isocentre."""
    return [dict(pe_dir=d, lines=n, rotation_deg=r) for d in ("y", "x") for r in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0) for n in range(44, 130, 2)]


def tune(cfgs):
    """The strongest answer that ignores the object, on other worlds: the constant geometry that passes the most of
    them under the verifier's rule (pixel, whole object inside, scan time against the optimum of each world; a geometry
    that holds the object does not wrap)."""
    from ...scanner import console, kspace
    from ...scanner import world as W
    worlds = []
    for c in cfgs:
        w = W.make_world(c["seed"], c["world_spec"])
        worlds.append((w, task.outline(w), task.optimum(c, w, kspace.Engine(w))["scan_s"]))
    best = None
    for cand in recipe_candidates():
        n = _even_up(SAFE_FOV_MM / cfgs[0]["task"]["max_pixel_mm"])
        geo = dict(pe_dir=cand["pe_dir"], fov_mm=n * cfgs[0]["task"]["max_pixel_mm"], phase_fov=round(cand["lines"] / n, 6), matrix=n,
                   centre_y_mm=0.0, centre_x_mm=0.0, rotation_deg=cand["rotation_deg"])
        passed = 0
        for w, pts, t_opt in worlds:
            pl = console.plan(task.protocol_of(geo), w["console"])
            passed += bool(pl["accept"] and pl["scan_s"] <= task.TOL * t_opt + 1e-9
                           and task.overhang_mm(pts, pl["realised"], w["extent_mm"] / w["n"]) <= task.COVER_SLACK_MM)
        if best is None or passed > best[0]:
            best = (passed, cand)
    return dict(best[1], tuned_pass_rate=round(best[0] / len(worlds), 3))
