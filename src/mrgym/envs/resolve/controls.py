"""Control policies of ``Scanner/Resolve``. Every policy plays through the agent's interface only (``info`` / ``plan`` /
``scan`` / ``view`` / ``budget``) and returns the deliverable. None reads the hidden world.

The ruler. A language model measures with ``profile``. A scripted policy reads the stored image in process
(``view(...)["array"]``, see ``mrgym.policies.ToolScanner``) and takes the same profiles from it: the mean of a band
of rows or columns.

Must pass
  reference        one localiser with a pixel of 1 mm; the edges of the object where a profile through the isocentre
                   crosses half of its plateau; the spacing of each pair from the two peaks of a profile through its
                   region (centre of mass of the samples above half of the highest value); pixel = ``RULE_PIXEL`` x spacing; phase-encode axis along the
                   shorter extent; the smallest field of view that holds the object with ``RULE_MARGIN_MM`` to spare
Must fail
  prior            no scan; a geometry that holds any object, with a pixel of 2 mm
  console_default  one token scan, then the console's default field of view and matrix, phase along y
  fixed_recipe     one token scan, then the best geometry that ignores the object: one constant phase axis, pixel and
                   number of lines for every pair, tuned on other worlds
  sweep            scans the object at a fixed list of pixels, coarse to fine, as far as the budget goes, and delivers
                   for each pair the coarsest pixel at which its own image shows two peaks; no spacing is measured.
                   (The budget pays one probe of the list: see "Why the budget pays one localiser" in ``task``.)
  wrong_units      the reference's geometry with the field of view in centimetres
Ablations (the reference with one decision changed; each must lose at the levels where that decision matters)
  pixel_is_spacing the pixel set to the measured spacing: one pixel per feature, none between them       (all levels)
  fixed_axis       always phase-encodes along y                                                          (level 3)
Reported
  half_spacing     the pixel set to half the measured spacing (two pixels per spacing): separated, and slower
  open_loop        a fixed scan list (the one localiser) and a fixed estimator: this is the reference itself, because
                   the reference chooses no scan from a measurement. By the rule of this package the family is a
                   measurement task, not a decision task
"""
from __future__ import annotations

import math

import numpy as np

from . import task

REFERENCE = "reference"
MUST_FAIL = ("prior", "console_default", "fixed_recipe", "sweep", "wrong_units")
ABLATIONS = {"pixel_is_spacing": (1, 2, 3), "fixed_axis": (3,)}
REPORTED = ("half_spacing", "open_loop")
DESCRIPTIONS = dict(
    reference="one 1 mm localiser; spacing of each pair from the two peaks of a profile; pixel = 0.55 x spacing; smallest field of view that holds the object",
    prior="no scan; a geometry that holds any object, with a pixel of 2 mm",
    console_default="one token scan, then the console's default field of view and matrix",
    fixed_recipe="one token scan, then one constant geometry (axis, pixel, lines) for every pair, tuned on other worlds",
    sweep="scans at a fixed list of pixels and delivers the coarsest at which its own image shows two peaks",
    wrong_units="the reference's geometry with the field of view in centimetres",
    pixel_is_spacing="the reference with the pixel set to the measured spacing",
    fixed_axis="the reference with the phase-encode axis always along y",
    half_spacing="the reference with the pixel set to half the measured spacing",
    open_loop="a fixed scan list and a fixed estimator: the reference itself (no scan depends on a measurement)",
)
LOCALISER = dict(task.LOCALISER)
TOKEN = dict(sequence="spgr", tr_ms=11.0, flip_deg=20.0, fov_mm=256.0, matrix=64)
SAFE_HALF_MM = 125.0                        # half of a field of view that holds any object of this family
BAND_MM = task.EDGE_BAND_MM                 # width of the profile through the isocentre that gives the edges
PAIR_BAND_MM = 4.0                          # width of the profile through a pair
SWEEP_MATRIX = (64, 96, 128)                # pixels of 4, 2.67 and 2 mm at a field of view of 256 mm


def _even_up(x):
    return 2 * int(math.ceil(x / 2.0 - 1e-9))


def _task(sc):
    return sc.info()["task"]


def _image(sc, protocol):
    """One scan: (magnitude image, positions of its rows along y, of its columns along x), mm from the isocentre."""
    v = sc.view(sc.scan(protocol)["scan_index"])
    img = np.asarray(v["array"], float)
    ny, nx = img.shape
    (fy, fx), (cy, cx) = v["fov_yx_mm"], v["centre_yx_mm"]
    return img, cy + (np.arange(ny) - ny // 2) * (fy / ny), cx + (np.arange(nx) - nx // 2) * (fx / nx)


def _profile(seen, axis, position_mm, width_mm, start_mm=None, stop_mm=None):
    """The mean profile along ``axis`` on a band ``width_mm`` wide at ``position_mm`` on the other axis: (positions, values)."""
    img, ypos, xpos = seen
    along, across = (ypos, xpos) if axis == "y" else (xpos, ypos)
    img = img if axis == "y" else img.T
    cols = np.abs(across - position_mm) <= width_mm / 2 + 1e-9
    rows = np.ones(len(along), bool) if start_mm is None else (along >= start_mm - 1e-9) & (along <= stop_mm + 1e-9)
    return along[rows], img[np.ix_(rows, cols)].mean(1)


def _edges(seen, axis):
    """Low and high edge of the object along ``axis``: where the profile through the isocentre crosses half of its
    plateau (the median of its upper half), interpolated between the two samples."""
    pos, val = _profile(seen, axis, 0.0, BAND_MM)
    level = 0.5 * float(np.median(val[val > 0.5 * val.max()]))
    above = np.flatnonzero(val > level)
    i, j = int(above[0]), int(above[-1])
    lo = pos[i] if i == 0 else pos[i - 1] + (level - val[i - 1]) / (val[i] - val[i - 1]) * (pos[i] - pos[i - 1])
    hi = pos[j] if j == len(pos) - 1 else pos[j] + (val[j] - level) / (val[j] - val[j + 1]) * (pos[j + 1] - pos[j])
    return float(lo), float(hi)


def _groups(val):
    """Runs of neighbouring samples above half of the highest value: [(first, last), ...]."""
    above = val > 0.5 * float(val.max())
    edges = np.flatnonzero(np.diff(np.concatenate([[False], above, [False]]).astype(int)))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2])]


def _spacing(seen, region, axis):
    """Centre-to-centre spacing of the two peaks of the profile along ``axis`` through ``region``; None without two peaks.
    The samples above half of the highest value form two groups; the position of a group is the mean of its positions
    weighted with (value - half of the highest value)."""
    centre, other = (region["x_mm"], region["y_mm"]) if axis == "x" else (region["y_mm"], region["x_mm"])
    pos, val = _profile(seen, axis, other, PAIR_BAND_MM, centre - region["radius_mm"], centre + region["radius_mm"])
    weight = val - 0.5 * float(val.max())
    runs = sorted(_groups(val), key=lambda r: -float(weight[r[0]:r[1] + 1].sum()))[:2]
    if len(runs) < 2:
        return None
    where = sorted(float(np.sum(pos[a:b + 1] * weight[a:b + 1]) / np.sum(weight[a:b + 1])) for a, b in runs)
    return where[1] - where[0]


def _half(lo, hi, pixel, margin=task.RULE_MARGIN_MM):
    """Half of the field of view about the isocentre that holds [lo, hi]: the image reaches half a pixel less far on
    the high side than on the low side."""
    return max(-lo - pixel / 2, hi + pixel / 2) + margin


def _geometry(edges, pe_dir, pixel):
    """The smallest field of view about the isocentre that holds the object (``edges``: {axis: (low, high)})."""
    pe, ro = (edges["y"], edges["x"]) if pe_dir == "y" else (edges["x"], edges["y"])
    lines = 2 * int(math.ceil(_half(*pe, pixel) / pixel - 1e-9))
    matrix = max(2 * int(math.ceil(_half(*ro, pixel) / pixel - 1e-9)), lines, 32)
    return dict(pe_dir=pe_dir, fov_mm=round(pixel * matrix, 4), phase_fov=round(lines / matrix, 6), matrix=int(matrix))


def _safe(t, pixel=2.0):
    edges = dict(y=(-SAFE_HALF_MM, SAFE_HALF_MM), x=(-SAFE_HALF_MM, SAFE_HALF_MM))
    return {i: _geometry(edges, "y", pixel) for i in t["pairs"]}


def _measure(sc, t):
    """The localiser, read: (edges {axis: (low, high)}, the long axis, {pair: spacing or None})."""
    seen = _image(sc, LOCALISER)
    edges = dict(y=_edges(seen, "y"), x=_edges(seen, "x"))
    long_axis = t.get("long_axis") or ("x" if edges["x"][1] - edges["x"][0] >= edges["y"][1] - edges["y"][0] else "y")
    regions = {r["id"]: r for r in t["regions"]}
    return edges, long_axis, {i: _spacing(seen, regions[i], long_axis) for i in t["pairs"]}


def _from_spacing(pixel_per_spacing, fixed_axis=None):
    def policy(sc):
        t = _task(sc)
        edges, long_axis, spacing = _measure(sc, t)
        pe_dir = fixed_axis or ("y" if long_axis == "x" else "x")
        return {i: _geometry(edges, pe_dir, pixel_per_spacing * (spacing[i] or task.SPACING_MM[0])) for i in t["pairs"]}
    return policy


reference = _from_spacing(task.RULE_PIXEL)
open_loop = reference                       # the reference is a fixed scan list and a fixed estimator
pixel_is_spacing = _from_spacing(1.0)
half_spacing = _from_spacing(0.5)
fixed_axis = _from_spacing(task.RULE_PIXEL, fixed_axis="y")


def prior(sc):
    return _safe(_task(sc))


def console_default(sc):
    t, card = _task(sc), sc.info()
    sc.scan(TOKEN)
    return {i: dict(pe_dir="y", fov_mm=float(card["default_fov_mm"]), phase_fov=1.0, matrix=128) for i in t["pairs"]}


def _constant(recipe):
    pixel, lines = float(recipe["pixel_mm"]), int(recipe["lines"])
    matrix = max(2 * int(math.ceil(SAFE_HALF_MM / pixel)), lines, 32)
    return dict(pe_dir=recipe["pe_dir"], fov_mm=round(pixel * matrix, 4), phase_fov=round(lines / matrix, 6), matrix=int(matrix))


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        sc.scan(TOKEN)
        return {i: _constant(recipe) for i in t["pairs"]}
    return policy


DEFAULT_RECIPE = dict(pe_dir="y", pixel_mm=3.0, lines=48)


def _two_peaks(seen, region, axis):
    """Whether the profile through ``region`` shows two groups of samples above half of its highest value."""
    centre, other = (region["x_mm"], region["y_mm"]) if axis == "x" else (region["y_mm"], region["x_mm"])
    _, val = _profile(seen, axis, other, PAIR_BAND_MM, centre - region["radius_mm"], centre + region["radius_mm"])
    return len(_groups(val)) >= 2


def sweep(sc):
    """Pixels from coarse to fine with one scan each, as far as the budget goes; per pair the coarsest pixel whose own
    image shows two peaks (the finest one scanned when none does). The extent of the object is read on the last scan."""
    t = _task(sc)
    regions = {r["id"]: r for r in t["regions"]}
    found, seen, pixel = {}, None, None
    for m in SWEEP_MATRIX:
        prot = dict(TOKEN, matrix=m)
        p = sc.plan(prot)
        if not p["accept"] or p["scan_s"] > sc.budget()["remaining_s"]:
            break
        seen, pixel = _image(sc, prot), TOKEN["fov_mm"] / m
        edges = dict(y=_edges(seen, "y"), x=_edges(seen, "x"))
        long_axis = t.get("long_axis") or ("x" if edges["x"][1] - edges["x"][0] >= edges["y"][1] - edges["y"][0] else "y")
        for i in t["pairs"]:
            if i not in found and _two_peaks(seen, regions[i], long_axis):
                found[i] = pixel
    if seen is None:
        return _safe(t)
    return {i: _geometry(edges, "y" if long_axis == "x" else "x", found.get(i, pixel)) for i in t["pairs"]}


def wrong_units(sc):
    return {i: dict(g, fov_mm=g["fov_mm"] / 10.0) for i, g in reference(sc).items()}


def policies(recipe=None):
    return dict(reference=reference, prior=prior, console_default=console_default, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE),
                sweep=sweep, wrong_units=wrong_units, pixel_is_spacing=pixel_is_spacing, fixed_axis=fixed_axis,
                half_spacing=half_spacing, open_loop=open_loop)


def recipe_candidates():
    """The constant geometries a recipe is chosen from: either phase axis, a pixel from 1.8 to 6 mm in steps of 0.1 mm,
    every even number of lines from 20 to 110."""
    return [dict(pe_dir=d, pixel_mm=round(0.1 * k, 1), lines=n) for d in ("y", "x") for k in range(18, 61) for n in range(20, 112, 2)]


def tune(cfgs):
    """The strongest answer that ignores the object, on other worlds: the constant geometry (the same for every pair)
    that passes the most of them. A candidate is judged per pair by the verifier's rule from the optimum of that pair
    (``task.optimum``): its pixel is at most the optimum's (then the pair is separated and the SNR floor holds, both
    checked there), its field of view holds the object, and its scan time is within the factor and the cap. Nothing
    the task text states says anything about the spacings or the size of the object: the regions lie where the
    inserts are, at distances drawn independently of both."""
    from ...scanner import console, kspace
    from ...scanner import world as W
    worlds = []
    for c in cfgs:
        w = W.make_world(c["seed"], c["world_spec"])
        worlds.append((w, task.N.outline(w), task.optimum(c, w, kspace.Engine(w))))
    plans, best = {}, None
    for cand in recipe_candidates():
        geo = _constant(cand)
        key = (geo["fov_mm"], geo["phase_fov"], geo["matrix"])
        if key not in plans:
            plans[key] = console.plan(task.protocol_of(geo), worlds[0][0]["console"])
        pl = plans[key]
        if not pl["accept"] or pl["scan_s"] > task.MAX_SCAN_S:
            continue
        every = pairs = 0
        for w, outline, opt in worlds:
            inside = task.N.overhang_mm(outline, dict(pl["realised"], pe_dir=cand["pe_dir"]), w["extent_mm"] / w["n"]) <= task.COVER_SLACK_MM
            ok = [inside and cand["pixel_mm"] <= o["pixel_mm"] + 1e-9 and pl["scan_s"] <= task.TOL * o["scan_s"] + 1e-9 for o in opt.values()]
            every, pairs = every + all(ok), pairs + sum(ok)
        if best is None or (every, pairs) > best[0]:
            best = ((every, pairs), cand)
    return dict(best[1], tuned_pass_rate=round(best[0][0] / len(worlds), 3),
                tuned_pair_pass_rate=round(best[0][1] / (len(task.PAIRS) * len(worlds)), 3))
