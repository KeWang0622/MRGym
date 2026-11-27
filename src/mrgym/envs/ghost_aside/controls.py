"""Control policies of ``Scanner/GhostAside``. Every policy plays through the agent's interface only.

Must pass
  reference       the target scan at the shortest repetition time with either phase-encode axis; what differs between
                  the two images along a tube's column and row is its ghosts; their distance gives the rate; then, per
                  axis, the first repetition time at which the law puts the ghost clear of that axis's region, the
                  faster of the two axes, plus 5 %. A tube without a ghost on its regions gets the shortest time
Must fail
  prior           no scan; the shortest repetition time, pe_dir y
  shortest_tr     one token scan, then the shortest repetition time, pe_dir y
  fixed_recipe    one token scan, then the best answer that ignores the object, tuned on other worlds: one constant
                  repetition time and axis; or one assumed rate put into the law with the told distances; or the ghost
                  assumed at one fixed place of the interval in which the statement says it lies at the shortest time
  sweep           the target scan at two repetition times with pe_dir y; per tube the one at which its two regions
                  look more alike; no model
  wrong_units     the reference's repetition time in seconds
Ablations (must lose where the step they leave out matters)
  fixed_axis      the reference with pe_dir y only (levels 1 to 3)
  assume_ghost    the reference without the question "does a ghost lie on this tube's regions at all" (level 3)
Reported
  open_loop       the reference: its scan list and its estimator are fixed, no scan depends on an earlier one
"""
from __future__ import annotations

import math

import numpy as np

from . import task

REFERENCE = "reference"
MUST_FAIL = ("prior", "shortest_tr", "fixed_recipe", "sweep", "wrong_units")
ABLATIONS = {"fixed_axis": (1, 2, 3), "assume_ghost": (3,)}
REPORTED = ("open_loop",)
DESCRIPTIONS = dict(
    reference="the target scan at the shortest TR with either axis; ghost distance -> rate; per axis the first TR that puts the ghost clear of the region; the faster axis, plus 5 %",
    prior="no scan; the shortest repetition time, pe_dir y",
    shortest_tr="one token scan, then the shortest repetition time, pe_dir y",
    fixed_recipe="one token scan, then a constant repetition time and axis, or an assumed rate or ghost position in the law with the told distances, tuned on other worlds",
    sweep="the target scan at two repetition times with pe_dir y; per tube the one at which its two regions look more alike",
    wrong_units="the reference's repetition time in seconds",
    fixed_axis="the reference with pe_dir y only",
    assume_ghost="the reference without asking whether a ghost lies on the tube's regions at all",
    open_loop="the reference: a fixed scan list and a fixed estimator",
)
TOKEN = dict(sequence="spgr", tr_ms=12.0, flip_deg=10.0, matrix=64)
CLEAR_MM = task.CLEAR_MM                # the stated distance to clear (the verifier's lies 14.3 to 15.9 mm beyond the region's centre)
SAFETY = 1.05                           # the delivered repetition time over the first clear one by the law
STRIP_MM = 3.5                          # half-width of the strip along a tube's column or row
SEARCH_MM = (9.0, 118.0)                # where a ghost is looked for, from the tube's centre
GHOST_MIN = 0.06                        # a ghost is one whose strip signal reaches this share of the tube's own signal
SWEEP_TR_MS = (300.0, 600.0)


def _task(sc):
    return sc.info()["task"]


def _complex(sc, scan_index):
    """A stored scan as a complex image with the positions of its pixels (y, x) in mm."""
    mag, phase = sc.view(scan_index, 0, "magnitude"), sc.view(scan_index, 0, "phase")
    img = np.asarray(mag["array"], float) * np.exp(1j * np.asarray(phase["array"], float))
    axes = [c + (np.arange(n) - n // 2) * (f / n) for n, f, c in zip(img.shape, mag["fov_yx_mm"], mag["centre_yx_mm"])]
    return img, axes[0], axes[1]


def _ghost_distance(diff, along, across, tube_along, tube_across, own):
    """Distance from a tube to its ghost along one axis, from the difference image of the two scouts: the centroid of
    the strip signal around its maximum, on both sides, averaged. None when nothing reaches ``GHOST_MIN`` of ``own``."""
    strip = np.abs(diff[:, np.abs(across - tube_across) <= STRIP_MM].mean(axis=1))
    found = []
    for sign in (1.0, -1.0):
        u = sign * (along - tube_along)
        sel = np.flatnonzero((u >= SEARCH_MM[0]) & (u <= SEARCH_MM[1]))
        if not len(sel):
            continue
        k = sel[int(np.argmax(strip[sel]))]
        if strip[k] < GHOST_MIN * own:
            continue
        near = np.flatnonzero(np.abs(u - u[k]) <= 7.0)
        w = strip[near] ** 2
        found.append(float(np.sum(w * u[near]) / np.sum(w)))
    return float(np.mean(found)) if found else None


def _scouts(sc, t):
    """Per tube: the distance of its ghost at the shortest repetition time, from each axis (None: no ghost seen)."""
    tr0 = t["tr_window_ms"][0]
    img = {}
    for pe in ("y", "x"):
        r = sc.scan(dict(task.TARGET, tr_ms=tr0, pe_dir=pe))
        img[pe] = _complex(sc, r["scan_index"])
    (iy, ypos, xpos), (ix, _, _) = img["y"], img["x"]
    diff = iy - ix
    out = {}
    for u in t["tubes"]:
        core = (np.abs(ypos[:, None] - u["y_mm"]) <= 2.5) & (np.abs(xpos[None, :] - u["x_mm"]) <= 2.5)
        own = float(np.abs(0.5 * (iy + ix))[core].mean())
        out[u["id"]] = dict(y=_ghost_distance(diff, ypos, xpos, u["y_mm"], u["x_mm"], own),
                            x=_ghost_distance(diff.T, xpos, ypos, u["x_mm"], u["y_mm"], own))
    return out


def _region_distance(u, pe):
    r = u["regions"][0 if pe == "y" else 1]
    return math.hypot(r["y_mm"] - u["y_mm"], r["x_mm"] - u["x_mm"])


def _first_clear(t, rate, distance):
    """The first repetition time of the window (1 ms steps) at which the law puts the ghost clear of a region at ``distance``."""
    lo, hi = t["tr_window_ms"]
    for tr in np.arange(lo, hi + 0.5, 1.0):
        if abs(task.ghost_distance_mm(rate, float(tr)) - distance) >= CLEAR_MM:
            return float(tr)
    return None


def _answer(t, seen, axes=("y", "x"), assume_ghost=False):
    lo, hi = t["tr_window_ms"]
    rates = {i: [d / (task.FOV_MM * lo * 1e-3) for d in s.values() if d is not None] for i, s in seen.items()}
    typical = [float(np.mean(v)) for v in rates.values() if v]
    scans = {}
    for u in t["tubes"]:
        r = rates[u["id"]]
        if not r and not assume_ghost:                                   # nothing pulsates here: any axis, the shortest time
            scans[u["id"]] = dict(pe_dir=axes[0], tr_ms=lo)
            continue
        rate = float(np.mean(r)) if r else (float(np.mean(typical)) if typical else 0.3)
        cand = []
        for pe in axes:
            d = _region_distance(u, pe)
            tr = _first_clear(t, rate, d)
            if assume_ghost and tr == lo:                                # it insists that the ghost has to be moved off the region
                tr = 1e3 * (d + CLEAR_MM) / (task.FOV_MM * rate)
            if tr is not None:
                cand.append((tr, pe))
        tr, pe = min(cand) if cand else (hi, axes[0])
        scans[u["id"]] = dict(pe_dir=pe, tr_ms=round(float(np.clip(tr if tr == lo else tr * SAFETY, lo, hi)), 1))
    return dict(scans=scans)


def reference(sc):
    t = _task(sc)
    return _answer(t, _scouts(sc, t))


def fixed_axis(sc):
    t = _task(sc)
    return _answer(t, _scouts(sc, t), axes=("y",))


def assume_ghost(sc):
    t = _task(sc)
    return _answer(t, _scouts(sc, t), assume_ghost=True)


def _constant(t, tr_ms, pe_dir="y"):
    lo, hi = t["tr_window_ms"]
    return dict(scans={u["id"]: dict(pe_dir=pe_dir, tr_ms=round(float(np.clip(tr_ms, lo, hi)), 1)) for u in t["tubes"]})


def prior(sc):
    t = _task(sc)
    return _constant(t, t["tr_window_ms"][0])


def shortest_tr(sc):
    t = _task(sc)
    sc.scan(dict(TOKEN))
    return _constant(t, t["tr_window_ms"][0])


def _by_rate(t, rate):
    """The law with one assumed rate and the told distances: the nearer region's axis, its clearing time plus the safety."""
    lo, hi = t["tr_window_ms"]
    scans = {}
    for u in t["tubes"]:
        pe = min(("y", "x"), key=lambda p: _region_distance(u, p))
        tr = 1e3 * (_region_distance(u, pe) + CLEAR_MM) / (task.FOV_MM * rate) * SAFETY
        scans[u["id"]] = dict(pe_dir=pe, tr_ms=round(float(np.clip(tr, lo, hi)), 1))
    return dict(scans=scans)


def _by_share(t, share):
    """What the statement of levels 1 and 2 gives away, used: at the shortest repetition time the ghost lies on both
    regions of its tube, so between (further distance - reach) and (nearer distance + reach). The ghost is assumed at
    the fraction ``share`` of that interval; the law then gives the time at which it clears the nearer region."""
    lo, hi = t["tr_window_ms"]
    scans = {}
    for u in t["tubes"]:
        pe = min(("y", "x"), key=lambda p: _region_distance(u, p))
        near, far = sorted(_region_distance(u, p) for p in ("y", "x"))
        ghost = (far - CLEAR_MM) + share * ((near + CLEAR_MM) - (far - CLEAR_MM))
        scans[u["id"]] = dict(pe_dir=pe, tr_ms=round(float(np.clip(lo * (near + CLEAR_MM) / ghost * SAFETY, lo, hi)), 1))
    return dict(scans=scans)


def _recipe_answer(t, recipe):
    if recipe.get("rate_hz"):
        return _by_rate(t, recipe["rate_hz"])
    if recipe.get("ghost_share") is not None:
        return _by_share(t, recipe["ghost_share"])
    return _constant(t, recipe["tr_ms"], recipe.get("pe_dir", "y"))


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        sc.scan(dict(TOKEN))
        return _recipe_answer(t, recipe)
    return policy


DEFAULT_RECIPE = dict(rate_hz=0.3)


def sweep(sc):
    """Two repetition times, no model: per tube the one at which its two regions read more alike."""
    t = _task(sc)
    seen = []
    for tr in SWEEP_TR_MS:
        r = sc.scan(dict(task.TARGET, tr_ms=tr, pe_dir="y"), rois=t["regions"])
        seen.append({row["id"]: row["mean_magnitude"][0] for row in r["roi_readouts"]})
    scans = {}
    for u in t["tubes"]:
        a, b = (r["id"] for r in u["regions"])
        k = int(np.argmin([abs(math.log(s[a] / s[b])) for s in seen]))
        scans[u["id"]] = dict(pe_dir="y", tr_ms=SWEEP_TR_MS[k])
    return dict(scans=scans)


def wrong_units(sc):
    a = reference(sc)
    return dict(scans={i: dict(s, tr_ms=s["tr_ms"] / 1000.0) for i, s in a["scans"].items()})


def policies(recipe=None):
    return dict(reference=reference, prior=prior, shortest_tr=shortest_tr, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE),
                sweep=sweep, wrong_units=wrong_units, fixed_axis=fixed_axis, assume_ghost=assume_ghost, open_loop=reference)


def _scan_s(tr_ms):
    """Scan time of the target scan (console arithmetic: overhead + one dummy cycle + one line per repetition)."""
    from ...scanner import console
    return console.SYSTEM["overhead_s"] + tr_ms * 1e-3 * (console.SYSTEM["prepared_dummy_cycles"] + task.TARGET["matrix"])


def tune(cfgs):
    """The strongest answer that ignores the object, on other worlds: the best constant repetition time with a constant
    axis, the best assumed rate in the law with the told distances and the best assumed place of the ghost in the
    interval the statement implies, by the per-tube pass rate that the drawn rates predict (a slow tube clears its
    nearer region at (distance + ``task.CLEAR_MM``) / (field of view x rate); a tube that is still or fast is clear at
    the shortest time on either axis)."""
    lo = task.TR_WINDOW_MS[0]
    rows = []                                                            # (first clear repetition time, its axis or None for either, tube)
    for c in cfgs:
        for u in c["truth"]["tubes"]:
            if u["kind"] != "slow":
                rows.append((lo, None, u))
                continue
            pe = min(("y", "x"), key=lambda p: _region_distance(u, p))
            rows.append((1e3 * (_region_distance(u, pe) + task.CLEAR_MM) / (task.FOV_MM * u["rate_hz"]), pe, u))
    passes = lambda tr, pe, row: (row[1] is None or row[1] == pe) and row[0] <= tr and _scan_s(tr) <= task.TIME_TOL * _scan_s(row[0])   # noqa: E731
    best = None
    for pe in ("y", "x"):
        for cand in sorted({round(r[0] * 1.03, 1) for r in rows}):
            rate = float(np.mean([passes(cand, pe, r) for r in rows]))
            if best is None or rate > best[0]:
                best = (rate, dict(tr_ms=cand, pe_dir=pe))
    told = dict(tr_window_ms=list(task.TR_WINDOW_MS))
    family = [dict(rate_hz=round(float(f), 4)) for f in np.exp(np.linspace(math.log(0.15), math.log(1.5), 240))]
    family += [dict(ghost_share=round(float(q), 3)) for q in np.linspace(0.0, 1.0, 81)]
    for recipe in family:
        hits = []
        for r in rows:
            scan = _recipe_answer(dict(told, tubes=[r[2]]), recipe)["scans"][r[2]["id"]]
            hits.append(passes(scan["tr_ms"], scan["pe_dir"], r))
        if float(np.mean(hits)) > best[0]:
            best = (float(np.mean(hits)), recipe)
    return dict(best[1], tuned_tube_pass_rate=round(best[0], 3))
