"""Control policies of ``Scanner/FoldAway``. Every policy plays through the agent's interface only (``info`` / ``plan`` /
``scan`` / ``view`` / ``budget``) and returns the deliverable. None reads the hidden world.

The ruler. A language model measures the object with ``profile``: where a profile crosses half of its plateau. A
scripted policy reads the stored image in process (``view(...)["array"]``, see ``mrgym.policies.ToolScanner``) and
takes the same crossings from it: between every pixel brighter than half of the plateau and every neighbour that is
not, the point where the straight line between their two values crosses half of the plateau (``_look``).

Must pass
  reference        one localiser; per region and direction of the phase-encode axis, the ends of the object on the
                   lines through the region; the fold-free extent plus ``MARGIN_PX`` pixels; the direction with the
                   fewest lines (level 3: every degree of rotation); the field of view centred on the region.
                   At a level that states the method it follows the statement: one extent per region, from the
                   outermost ends of the object over the width of the region (``stated_method``; on the safe side
                   where the edge of the object runs obliquely to the lines). Elsewhere it applies the rule line
                   by line, with the length of the circle on each line, which is exact
Must fail
  prior            no scan; the full phase field of view, centred on each region
  fixed_recipe     one token scan, then the best geometry that uses only what the task states: a phase axis that is
                   constant or lies across the line through the regions, a number of lines that is constant or grows
                   with the region's distance from the centre of the regions, tuned on other worlds
  whole_object     the answer of ``Scanner/NoWrap``: one localiser, then the phase field of view that holds the whole
                   object along its narrowest direction, centred on the object. Free of wrap-around, and too slow
  region_only      one token scan, then the smallest phase field of view the console has, centred on the region: it
                   holds the region and nothing of the fold is kept away from it
  sweep            one scan with the full phase field of view, then one probe per region with a small one; keeps the
                   small one where the region's signal did not change, else a large one. No edge is measured
  wrong_units      the reference's geometry with the phase field of view in millimetres
Ablations (the reference with one decision changed; each must lose at the levels where that decision matters)
  fixed_axis       always phase-encodes along y, never turned                                             (all levels)
  whole_extent     the ends of the whole object instead of its ends on the lines through the region: the rule
                   with the bounding box of the object                                                    (all levels)
  no_turn          the better of the two upright axes, never turned                                       (level 3)
  no_margin        the fold-free extent without the pixels for the ringing of the folded edge. Where the rule is
                   applied line by line it has nothing to spare and the margin decides (levels 2, 3)
Reported
  stated_method    the method level 1 states, at every level (at level 1 this is the reference)
  open_loop        a fixed scan list (the one localiser) and a fixed estimator: this is the reference itself, because
                   the reference chooses no scan from a measurement. By the rule of this package the family is a
                   measurement task, not a decision task
"""
from __future__ import annotations

import math

import numpy as np

from . import task

REFERENCE = "reference"
MUST_FAIL = ("prior", "fixed_recipe", "whole_object", "region_only", "sweep", "wrong_units")
ABLATIONS = {"fixed_axis": (1, 2, 3), "whole_extent": (1, 2, 3), "no_turn": (3,), "no_margin": (2, 3)}
REPORTED = ("stated_method", "open_loop")
DESCRIPTIONS = dict(
    reference="one localiser; per region the ends of the object on the lines through it; fold-free phase field of view plus 2 pixels, along the better direction",
    stated_method="the method level 1 states: one fold-free extent per region from the outermost ends of the object over its width, plus 2 pixels",
    prior="no scan; the full phase field of view, centred on each region",
    fixed_recipe="one token scan, then one geometry from the task text alone (axis and lines), tuned on other worlds",
    whole_object="the NoWrap answer: the phase field of view holds the whole object along its narrowest direction",
    region_only="one token scan, then the smallest phase field of view of the console, centred on the region",
    sweep="one full scan and one small-field-of-view probe per region; keeps the small one where the region's signal did not change",
    wrong_units="the reference's geometry with the phase field of view in millimetres",
    no_margin="the reference without the pixels kept for the ringing of the folded edge",
    fixed_axis="the reference with the phase-encode axis always along y, never turned",
    no_turn="the reference with the better of the two upright axes, never turned",
    whole_extent="the reference with the ends of the whole object instead of its ends on the lines through the region",
    open_loop="a fixed scan list and a fixed estimator: the reference itself (no scan depends on a measurement)",
)
LOCALISER = dict(task.LOCALISER)
TOKEN = dict(sequence="spgr", tr_ms=11.0, flip_deg=10.0, fov_mm=288.0, matrix=64)
MARGIN_PX = task.RULE_MARGIN_PX
BAND_MM = 1.0                               # the lines through the region: its diameter and this much on each side (the stated method)
ANGLE_STEP_DEG = 1.0
MIN_LINES = int(round(task.TARGET["matrix"] * 0.25 / 2) * 2)      # of the smallest phase field of view of the console
SWEEP_SMALL, SWEEP_LARGE = 0.45, 0.75       # phase fields of view of the sweep


def _even_up(x):
    return 2 * int(math.ceil(x / 2.0 - 1e-9))


def _task(sc):
    return sc.info()["task"]


def _look(sc, protocol=LOCALISER):
    """One localiser: patient coordinates [n, 2] (y, x) in mm of points on the edge of the object, or None when nothing
    is bright. A point lies between a pixel above half of the plateau and a neighbour (up, down, left, right) at or
    below it, where the straight line between the two values crosses half of the plateau."""
    v = sc.view(sc.scan(protocol)["scan_index"])
    img = np.asarray(v["array"], float)
    ny, nx = img.shape
    bright = img > 0.2 * float(np.percentile(img, 99.5))
    if not bright.any():
        return None
    level = 0.5 * float(np.median(img[bright]))                         # half of the plateau
    pad = np.pad((img > level).astype(np.int8), 1)
    m = (img > level) & (sum(pad[i:i + ny, j:j + nx] for i in range(3) for j in range(3)) >= 6)     # drop isolated noise pixels
    (fy, fx), (cy, cx) = v["fov_yx_mm"], v["centre_yx_mm"]
    ypos, xpos = cy + (np.arange(ny) - ny // 2) * (fy / ny), cx + (np.arange(nx) - nx // 2) * (fx / nx)
    out = []
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        iy, ix = np.nonzero(m)
        jy, jx = iy + di, ix + dj
        ok = (jy >= 0) & (jy < ny) & (jx >= 0) & (jx < nx)
        iy, ix, jy, jx = iy[ok], ix[ok], jy[ok], jx[ok]
        edge = ~m[jy, jx]
        iy, ix, jy, jx = iy[edge], ix[edge], jy[edge], jx[edge]
        t = np.clip((img[iy, ix] - level) / np.maximum(img[iy, ix] - img[jy, jx], 1e-12), 0.0, 1.0)
        out.append(np.stack([ypos[iy] + t * (ypos[jy] - ypos[iy]), xpos[ix] + t * (xpos[jx] - xpos[ix])], 1))
    pts = np.concatenate(out)
    return pts if len(pts) else None


def _frame(pts, origin, rotation_deg):
    """Coordinates (along the image rows, along the columns) of patient points in a field of view turned by
    ``rotation_deg``, from ``origin``."""
    c, s = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
    dy, dx = pts[:, 0] - origin[0], pts[:, 1] - origin[1]
    return dy * c + dx * s, -dy * s + dx * c


def _extent(pts, region, pe_dir, rotation_deg, how="lines"):
    """The fold-free extent (mm) of ``region`` along a direction of the phase-encode axis, from the edge points of the
    localiser. ``how``: "lines" = the rule on every line through the region, with the length of the circle on that
    line (an edge point at ``a`` along the axis and ``c`` across it asks for |a| + sqrt(radius^2 - c^2)); "band" = the
    rule once, with the outermost ends of the object over the width of the region (what level 1 states); "whole" =
    the rule once, with the ends of the whole object."""
    v, u = _frame(pts, (region["y_mm"], region["x_mm"]), rotation_deg)
    along, across = (v, u) if pe_dir == "y" else (u, v)
    r = region["radius_mm"]
    if how == "lines":
        on = np.abs(across) <= r
        return float(np.max(np.abs(along[on]) + np.sqrt(r * r - across[on] ** 2))) if on.any() else None
    band = np.ones(len(along), bool) if how == "whole" else np.abs(across) <= r + BAND_MM
    if not band.any():
        return None
    return task.fold_free_extent(float(along[band].min()), float(along[band].max()), -r, r)


def _geometry(t, region, pe_dir, rotation_deg, lines):
    g = dict(pe_dir=pe_dir, phase_fov=round(max(lines / t["target_scan"]["matrix"], 0.25), 6),
             centre_y_mm=float(region["y_mm"]), centre_x_mm=float(region["x_mm"]))
    if t["rotation"]:
        g["rotation_deg"] = float(rotation_deg)
    return g


def _full(t):
    return {r["id"]: _geometry(t, r, "y", 0.0, t["target_scan"]["matrix"]) for r in t["regions"]}


def _directions(t, turn=True, axes=("y", "x")):
    if t["rotation"] and turn:
        return [("y", float(a)) for a in np.arange(-90.0 + ANGLE_STEP_DEG, 90.0 + 1e-9, ANGLE_STEP_DEG)]
    return [(d, 0.0) for d in axes]


def _fold_away(margin_px=MARGIN_PX, turn=True, axes=("y", "x"), how=None):
    def policy(sc):
        t = _task(sc)
        pts = _look(sc)
        if pts is None:
            return _full(t)
        rule = how or ("band" if "rule" in t.get("hints", ()) else "lines")      # a stated method is followed
        px, out = t["pixel_mm"], {}
        for region in t["regions"]:
            best = None
            for d, rot in _directions(t, turn, axes):
                e = _extent(pts, region, d, rot, rule)
                if e is not None and (best is None or e < best[0]):
                    best = (e, d, rot)
            lines = min(max(_even_up((best[0] + margin_px * px) / px), MIN_LINES), t["target_scan"]["matrix"])
            out[region["id"]] = _geometry(t, region, best[1], best[2], lines)
        return out
    return policy


reference = _fold_away()
open_loop = reference                       # the reference is a fixed scan list and a fixed estimator
stated_method = _fold_away(how="band")
no_margin = _fold_away(margin_px=0.0)
fixed_axis = _fold_away(turn=False, axes=("y",))
no_turn = _fold_away(turn=False)
whole_extent = _fold_away(how="whole")


def prior(sc):
    return _full(_task(sc))


def region_only(sc):
    t = _task(sc)
    sc.scan(TOKEN)
    return {r["id"]: _geometry(t, r, "y", 0.0, MIN_LINES) for r in t["regions"]}


def whole_object(sc):
    """The narrowest direction of the whole object (level 3: turned), the phase field of view that holds it with
    1.5 mm to spare on each side, centred on the object along the phase axis and on the region along the readout."""
    t = _task(sc)
    pts = _look(sc)
    if pts is None:
        return _full(t)
    best = None
    for d, rot in _directions(t):
        v, u = _frame(pts, (0.0, 0.0), rot)
        along = v if d == "y" else u
        lo, hi = float(along.min()), float(along.max())
        if best is None or hi - lo < best[0]:
            best = (hi - lo, d, rot, 0.5 * (lo + hi))
    width, d, rot, mid = best
    lines = min(max(_even_up((width + 3.0) / t["pixel_mm"]), MIN_LINES), t["target_scan"]["matrix"])
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    out = {}
    for region in t["regions"]:
        rv, ru = region["y_mm"] * c + region["x_mm"] * s, -region["y_mm"] * s + region["x_mm"] * c     # the region in the turned frame
        fv, fu = (mid + t["pixel_mm"] / 2, ru) if d == "y" else (rv, mid + t["pixel_mm"] / 2)         # centre of the field of view there
        g = _geometry(t, region, d, rot, lines)
        out[region["id"]] = dict(g, centre_y_mm=round(fv * c - fu * s, 2), centre_x_mm=round(fv * s + fu * c, 2))
    return out


def _mean(sc, protocol, region):
    return sc.scan(protocol, rois=[region])["roi_readouts"][0]["mean_magnitude"][0]


def sweep(sc):
    """A full scan, then per region one probe with a small phase field of view centred on it; the small one is kept
    where the mean of the region changed by less than 5 %, a large one otherwise (the full one when the budget ends)."""
    t = _task(sc)
    probe = dict(sequence="spgr", tr_ms=11.0, flip_deg=10.0, fov_mm=t["target_scan"]["fov_mm"], matrix=t["target_scan"]["matrix"])
    full = sc.scan(dict(probe), rois=t["regions"])
    before = {row["id"]: row["mean_magnitude"][0] for row in full["roi_readouts"]}
    out = _full(t)
    for region in t["regions"]:
        small = dict(probe, phase_fov=SWEEP_SMALL, centre_y_mm=float(region["y_mm"]), centre_x_mm=float(region["x_mm"]))
        p = sc.plan(small)
        if not p["accept"] or p["scan_s"] > sc.budget()["remaining_s"]:
            break
        after = _mean(sc, small, region)
        keep = before[region["id"]] and abs(after / before[region["id"]] - 1.0) < 0.05
        out[region["id"]] = _geometry(t, region, "y", 0.0, _even_up(t["target_scan"]["matrix"] * (SWEEP_SMALL if keep else SWEEP_LARGE)))
    return out


def wrong_units(sc):
    t = _task(sc)
    return {i: dict(g, phase_fov=round(g["phase_fov"] * t["target_scan"]["fov_mm"], 3)) for i, g in reference(sc).items()}


# ------------------------------------------------------------------------------------------------ the answer from the task text alone
def _across(t):
    """The direction across the line through the regions (what the task text says about the long axis of the object:
    the regions lie along it more often than across it): (the upright axis, the rotation in degrees) across the
    principal axis of the region centres."""
    yx = np.array([[r["y_mm"], r["x_mm"]] for r in t["regions"]], float)
    yx = yx - yx.mean(0)
    dy, dx = np.linalg.eigh(yx.T @ yx)[1][:, -1]                         # the direction of the largest spread
    rot = math.degrees(math.atan2(dy, -dx))                              # image rows (cos r, sin r) across the line (dy, dx)
    return ("y" if abs(dy) <= abs(dx) else "x"), 90.0 - (90.0 - rot) % 180.0


def recipe_geometry(recipe, t, region):
    """The geometry a recipe gives a region. ``direction``: "y", "x", "across" (upright axis across the line through the
    regions), a rotation in degrees, or "turned_across". ``lines`` + ``per_mm`` x the region's distance from the
    centre of the regions along the phase-encode axis."""
    direction = recipe["direction"]
    upright, turned = _across(t)
    if direction in ("y", "x"):
        d, rot = direction, 0.0
    elif direction == "across":
        d, rot = upright, 0.0
    else:
        d, rot = "y", (turned if direction == "turned_across" else float(direction))
    if not t["rotation"]:
        rot = 0.0
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    my, mx = (float(np.mean([r[k] for r in t["regions"]])) for k in ("y_mm", "x_mm"))      # the best guess of the centre of the object
    v, u = (region["y_mm"] - my) * c + (region["x_mm"] - mx) * s, -(region["y_mm"] - my) * s + (region["x_mm"] - mx) * c
    lines = _even_up(recipe["lines"] + recipe.get("per_mm", 0.0) * abs(v if d == "y" else u))
    return _geometry(t, region, d, rot, min(max(lines, MIN_LINES), t["target_scan"]["matrix"]))


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        sc.scan(TOKEN)
        return {r["id"]: recipe_geometry(recipe, t, r) for r in t["regions"]}
    return policy


DEFAULT_RECIPE = dict(direction="across", lines=88, per_mm=0.6)


def policies(recipe=None):
    return dict(reference=reference, prior=prior, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE), whole_object=whole_object,
                region_only=region_only, sweep=sweep, wrong_units=wrong_units, fixed_axis=fixed_axis, whole_extent=whole_extent,
                no_turn=no_turn, no_margin=no_margin, stated_method=stated_method, open_loop=open_loop)


def recipe_candidates(turned: bool):
    """The recipes a tuning set chooses from: the direction constant (an axis; turned levels: every 15 degrees) or across
    the line through the regions; every fourth number of lines from 48 to 180; 0 to 1.2 lines more per mm of the
    region's distance from the centre of the regions along the phase-encode axis."""
    dirs = ["y", "x", "across"] + ([float(a) for a in range(-75, 91, 15)] + ["turned_across"] if turned else [])
    return [dict(direction=d, lines=n, per_mm=k) for d in dirs for n in range(48, 182, 4) for k in (0.0, 0.3, 0.6, 0.9, 1.2)]


TUNE_MARGIN_PX = 2.0                        # the margin a candidate is granted in ``tune``


def tune(cfgs):
    """The strongest answer from the task text alone, on other worlds: the recipe that passes the most of them (every
    region). A candidate is judged per region from the outline of the object: its phase field of view must exceed
    the fold-free extent along its direction by ``TUNE_MARGIN_PX`` pixels (about what the verifier's 5 % asks for), and
    its scan time must be within the factor of the optimum of that region (``task.optimum``, the verifier's own)."""
    from ...scanner import console, kspace
    from ...scanner import world as W
    from . import oracle
    worlds = []
    for c in cfgs:
        w = W.make_world(c["seed"], c["world_spec"])
        worlds.append((c["task"], w, task.N.outline(w), task.optimum(c, w, kspace.Engine(w))))
    px, best, need = task.PIXEL_MM, None, {}
    for cand in recipe_candidates(cfgs[0]["task"]["rotation"]):
        both = regions = 0
        for k, (t, w, outline, opt) in enumerate(worlds):
            ok = []
            for region in t["regions"]:
                g = recipe_geometry(cand, t, region)
                key = (k, region["id"], g["pe_dir"], g.get("rotation_deg", 0.0))
                if key not in need:
                    need[key] = oracle.extent(outline, region, g["pe_dir"], g.get("rotation_deg", 0.0), w["extent_mm"] / w["n"]) + TUNE_MARGIN_PX * px
                pl = console.plan(task.protocol_of(g), w["console"])
                ok.append(pl["accept"] and pl["realised"]["fov_pe_mm"] >= need[key] and pl["scan_s"] <= task.TOL * opt[region["id"]]["scan_s"] + 1e-9)
            both, regions = both + all(ok), regions + sum(ok)
        if best is None or (both, regions) > best[0]:
            best = ((both, regions), cand)
    return dict(best[1], tuned_pass_rate=round(best[0][0] / len(worlds), 3),
                tuned_region_pass_rate=round(best[0][1] / (len(task.REGIONS) * len(worlds)), 3))
