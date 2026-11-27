"""Control policies of ``Scanner/PhantomQA``. Every policy plays through the agent's tools only: ``scan``, ``plan``,
``profile`` (lines, and with ``samples = 1`` the mean of a rectangle) and ``read_regions`` (means of circles). None
reads an image array.

Must pass
  reference          the QA scan; edges from six line profiles, PIU from circles on a lattice and two refinements, the
                     ghosting ratio from one circle and four rectangles; verdicts by the limits; then, for each
                     failed test, what tells its faults apart: a short scan with the readout along the other axis
                     (gradient or shim), a flip-angle map read at the brightest and the darkest spot (transmit or
                     receive), the height of the ghost in the background (a copy of the phantom or its edges)
Must fail
  prior              no scan; nominal values, every test passes, no fault
  fixed_recipe       one token scan, then the most frequent answer on other worlds with their median values
  brute_force        measures like the reference, then names every fault of the list that moves a failed test
  wrong_units        the reference's answer with the diameters in cm, the uniformity as a fraction, the ghosting in percent
Ablations
  no_discrimination  the reference without its second scan and without the look at the ghost: names the first fault of
                     each failed test (must lose at every level)
  single_fault       the reference that stops at the first failed test (must lose at level 3, where there are two)
Reported
  open_loop          a fixed scan list (the QA scan, then the two discriminating scans in a fixed order, as far as the
                     budget goes) and the reference's estimator on whatever that list acquired. It decides the label
                     of a level: within 25 points of the reference = a measurement task, otherwise a decision task
"""
from __future__ import annotations

import importlib.resources
import json
import math
import statistics

from ... import qa
from . import task

REFERENCE = "reference"
MUST_FAIL = ("prior", "fixed_recipe", "brute_force", "wrong_units")
ABLATIONS = {"no_discrimination": (1, 2, 3), "single_fault": (3,)}
REPORTED = ("open_loop",)
DESCRIPTIONS = dict(
    reference="QA scan, the four quantities by profile and read_regions, then one short scan per failed test that needs it",
    prior="no scan; nominal values, every test passes, no fault",
    fixed_recipe="one token scan, then the most frequent answer on other worlds",
    brute_force="measures like the reference, then names every fault that moves a failed test",
    wrong_units="the reference's answer in cm, as a fraction and in percent",
    no_discrimination="the reference without the discriminating step: the first fault of each failed test",
    single_fault="the reference that stops at the first failed test",
    open_loop="a fixed scan list (QA scan, readout-swapped scan, flip-angle map, as far as the budget goes) and the reference's estimator",
)
R_LARGE = math.sqrt(qa.LARGE_ROI_CM2 * 100 / math.pi)                    # 79.8 mm
R_SMALL = math.sqrt(qa.SMALL_ROI_CM2 * 100 / math.pi)                    # 5.64 mm
SWAP = dict(sequence="se", tr_ms=60.0, fov_mm=250.0, matrix=64, bandwidth_hz_per_px=500.0)                 # + pe_dir: 6.9 s
FLIP_MAP = dict(sequence="afi", tr_ms=10.0, tr_ratio=4, flip_deg=60.0, fov_mm=250.0, matrix=32,
                bandwidth_hz_per_px=1000.0, spoiler_mt_ms_per_m=50.0)                                      # 8.6 s
TOKEN = dict(sequence="se", tr_ms=60.0, fov_mm=250.0, matrix=32)
LATTICE_MM = 12.0            # pitch of the lattice of 1 cm^2 circles the uniformity search starts from
RESTORED_MM = 2.2            # a diameter this close to 190 mm on the readout-swapped scan: the distortion was the field's
FLIP_SPREAD = 0.18           # flip angle at the darkest spot / at the brightest spot differs from 1 by more: the transmit field
GHOST_MEDIAN = 0.35          # median / highest sample of the ghost in the background bands: at least this = a copy of the phantom
NOMINAL = dict(diameter_y_mm=190.0, diameter_x_mm=190.0, piu_percent=95.0, ghosting_ratio=0.001)


def _task(sc):
    return sc.info()["task"]


def _rows(readouts) -> dict:
    return {r["id"]: r["mean_magnitude"][0] for r in readouts}


def _edges(sc, scan, axis, level, at=0.0):
    """(low edge, high edge) in mm of the object along ``axis`` on the line at ``at``: the outermost crossings of
    ``level``, from two full-resolution profiles."""
    out = []
    for lo, hi, first in ((-1e9, -55.0, True), (55.0, 1e9, False)):
        p = sc.profile(scan, axis, position_mm=at, width_mm=3.0, start_mm=None if first else lo, stop_mm=hi if first else None)
        pos, val = p["positions_mm"], p["values"]
        idx = [i for i, v in enumerate(val) if v >= level]
        if not idx:
            out.append(float("nan"))
            continue
        i = idx[0] if first else idx[-1]
        j = i - 1 if first else i + 1
        if 0 <= j < len(val) and val[i] != val[j]:
            out.append(pos[j] + (level - val[j]) / (val[i] - val[j]) * (pos[i] - pos[j]))
        else:
            out.append(pos[i])
    return out[0], out[1]


def _circles(points, prefix):
    return [dict(id=f"{prefix}{k}", y_mm=round(y, 2), x_mm=round(x, 2), radius_mm=round(R_SMALL, 3)) for k, (y, x) in enumerate(points)]


def _inside(points, centre):
    """Keep the 1 cm^2 circles that lie inside the 200 cm^2 circle."""
    lim = R_LARGE - R_SMALL - 0.3
    out = []
    for y, x in points:
        d = math.hypot(y - centre[0], x - centre[1])
        if d > lim:
            y, x = centre[0] + (y - centre[0]) * lim / d, centre[1] + (x - centre[1]) * lim / d
        out.append((y, x))
    return out


def measure(sc, t) -> dict:
    """The QA scan and the four quantities through the tools. Returns the quantities and what the diagnosis needs:
    the scan index, the centre, the signal of the large region, where the image is darkest and brightest."""
    proto = task.standard_protocol(t)
    r = sc.scan(proto, rois=[dict(id="large0", y_mm=0.0, x_mm=0.0, radius_mm=round(R_LARGE, 2))])
    k = r["scan_index"]
    level = 0.5 * _rows(r["roi_readouts"])["large0"]
    x0, x1 = _edges(sc, k, "x", level)
    y0, y1 = _edges(sc, k, "y", level, at=0.5 * (x0 + x1))
    x0, x1 = _edges(sc, k, "x", level, at=0.5 * (y0 + y1))
    c = (0.5 * (y0 + y1), 0.5 * (x0 + x1))
    # ---- uniformity: 1 cm^2 circles on a hexagonal lattice, then two refinements round the darkest and the brightest
    lim, rows = R_LARGE - R_SMALL - 0.3, int(R_LARGE // (LATTICE_MM * 0.866)) + 1
    pts = [(c[0] + j * LATTICE_MM * 0.866, c[1] + (i + 0.5 * (j % 2)) * LATTICE_MM) for j in range(-rows, rows + 1) for i in range(-rows, rows + 1)]
    pts = [p for p in pts if math.hypot(p[0] - c[0], p[1] - c[1]) <= lim]
    means, large = {}, None
    for i in range(0, len(pts), 15):
        batch = pts[i:i + 15]
        got = _rows(sc.read_regions(k, _circles(batch, f"u{i}_") + ([dict(id="large", y_mm=round(c[0], 2), x_mm=round(c[1], 2), radius_mm=round(R_LARGE, 2))] if i == 0 else [])))
        large = got.pop("large", None) if i == 0 else large
        means.update({p: got[f"u{i}_{j}"] for j, p in enumerate(batch)})
    for step in (0.5 * LATTICE_MM, 0.25 * LATTICE_MM):
        order = sorted(means, key=means.get)
        for n, group in enumerate((order[:2], order[-2:])):                  # round the two darkest, round the two brightest
            ring = _inside([(q[0] + step * math.sin(a), q[1] + step * math.cos(a)) for q in group for a in (i * math.pi / 4 for i in range(8))], c)
            got = _rows(sc.read_regions(k, _circles(ring, f"r{int(step * 10)}_{n}_")))
            means.update({p: got[f"r{int(step * 10)}_{n}_{j}"] for j, p in enumerate(ring)})
    lo, hi = min(means, key=means.get), max(means, key=means.get)
    piu = 100.0 * (1.0 - (means[hi] - means[lo]) / (means[hi] + means[lo]))
    # ---- ghosting: four background rectangles, each the mean of one profile sample
    w, ln, at = qa.BACKGROUND_ROI_MM[0], qa.BACKGROUND_ROI_MM[1] / 2, qa.BACKGROUND_AT_MM
    bg = {name: sc.profile(k, axis, position_mm=pos, width_mm=w, start_mm=-ln, stop_mm=ln, samples=1)["values"][0]
          for name, axis, pos in (("top", "x", -at), ("bottom", "x", at), ("left", "y", -at), ("right", "y", at))}
    ghost = abs((bg["top"] + bg["bottom"]) - (bg["left"] + bg["right"])) / (2.0 * large)
    return dict(values=dict(diameter_y_mm=round(y1 - y0, 2), diameter_x_mm=round(x1 - x0, 2), piu_percent=round(piu, 2), ghosting_ratio=round(ghost, 5)),
                scan=k, centre=c, large=large, level=level, low_at=lo, high_at=hi)


def _failed(values, t) -> list:
    return sorted(k for k, bad in task.verdicts(values, t["field_strength_t"]).items() if bad)


def _affordable(sc, proto) -> bool:
    p = sc.plan(proto)
    return bool(p["accept"] and p["scan_s"] <= sc.budget()["remaining_s"])


def _geometry_fault(sc, t, m, scan=None) -> str | None:
    """Which fault lengthens or shortens the phantom. Along the phase-encode axis only a gradient can. Along the
    readout axis: a scan with the readout along the other axis (``scan``: its index if it was already acquired);
    a field gradient no longer acts on this length, a gradient error still does. None: the scan could not be paid."""
    pe = t["standard_scan"]["pe_dir"]
    ro = "x" if pe == "y" else "y"
    off = {a: abs(m["values"][f"diameter_{a}_mm"] - qa.A["diameter_mm"]) for a in ("y", "x")}
    if off[pe] > off[ro]:
        return f"gradient_{pe}"
    if scan is None:
        proto = dict(SWAP, pe_dir=ro)
        if not _affordable(sc, proto):
            return None
        r = sc.scan(proto, rois=[dict(id="large0", y_mm=0.0, x_mm=0.0, radius_mm=round(R_LARGE, 2))])
        scan = (r["scan_index"], 0.5 * _rows(r["roi_readouts"])["large0"])
    a0, a1 = _edges(sc, scan[0], ro, scan[1], at=m["centre"][0 if ro == "x" else 1])
    return "shim" if abs((a1 - a0) - qa.A["diameter_mm"]) <= RESTORED_MM else f"gradient_{ro}"


def _uniformity_fault(sc, t, m, scan=None) -> str | None:
    """Transmit or receive: a flip-angle map (dual-TR) at the darkest and the brightest spot of the QA image. The
    receive profile cancels in the ratio of its two images, the flip angle does not."""
    if scan is None:
        if not _affordable(sc, FLIP_MAP):
            return None
        scan = sc.scan(FLIP_MAP)["scan_index"]
    rois = [dict(id=name, y_mm=round(p[0], 1), x_mm=round(p[1], 1), radius_mm=10.0) for name, p in (("low", m["low_at"]), ("high", m["high_at"]))]
    n = FLIP_MAP["tr_ratio"]
    flip = {}
    for row in sc.read_regions(scan, rois):
        r = row["mean_magnitude"][1] / max(row["mean_magnitude"][0], 1e-30)
        flip[row["id"]] = math.acos(min(max((r * n - 1.0) / (n - r), -1.0), 1.0))
    return "b1_transmit" if abs(flip["low"] / max(flip["high"], 1e-9) - 1.0) > FLIP_SPREAD else "receive_coil"


def _ghost_fault(sc, t, m) -> str:
    """A copy of the phantom or the ghost of its edges, from the QA image (no scan): a 60 mm wide profile through the
    two background bands along the phase-encode axis. A copy fills the bands (the median sample is at least a
    third of the highest; where two copies overlap at the edge of the field of view it is half); the edges of a
    rocking phantom are arcs with nothing between them."""
    pe = t["standard_scan"]["pe_dir"]
    values = []
    for lo, hi in ((None, -99.0), (99.0, None)):
        values += sc.profile(m["scan"], pe, position_mm=0.0, width_mm=60.0, start_mm=lo, stop_mm=hi)["values"]
    return "signal_instability" if statistics.median(values) >= GHOST_MEDIAN * max(values) else "phantom_motion"


def _first(test, t, m) -> str:
    """The fault named for a failed test without telling its faults apart."""
    if test == "geometric_accuracy":
        off = {a: abs(m["values"][f"diameter_{a}_mm"] - qa.A["diameter_mm"]) for a in ("y", "x")}
        return "gradient_" + max(off, key=off.get)
    return dict(intensity_uniformity="receive_coil", ghosting="signal_instability")[test]


def _diagnose(sc, t, m, failed, scans=None, stop_after=None) -> list:
    """The fault of each failed test. ``scans``: discriminating scans already acquired (open loop)."""
    scans = scans or {}
    faults = []
    for test in failed[:stop_after]:
        if test == "geometric_accuracy":
            f = _geometry_fault(sc, t, m, scans.get("swap"))
        elif test == "intensity_uniformity":
            f = _uniformity_fault(sc, t, m, scans.get("flip_map"))
        else:
            f = _ghost_fault(sc, t, m)
        faults.append(f or _first(test, t, m))
    return sorted(faults)


def reference(sc):
    t = _task(sc)
    m = measure(sc, t)
    failed = _failed(m["values"], t)
    return dict(measured=m["values"], failed_tests=failed, faults=_diagnose(sc, t, m, failed))


def prior(sc):
    return dict(measured=dict(NOMINAL), failed_tests=[], faults=[])


DEFAULT_RECIPE = dict(measured=dict(NOMINAL, piu_percent=72.0), failed_tests=["intensity_uniformity"], faults=["receive_coil"])


def fixed_recipe(recipe):
    def policy(sc):
        sc.scan(dict(TOKEN))
        return dict(measured=dict(recipe["measured"]), failed_tests=list(recipe["failed_tests"]), faults=list(recipe["faults"]))
    return policy


def brute_force(sc):
    t = _task(sc)
    m = measure(sc, t)
    failed = _failed(m["values"], t)
    return dict(measured=m["values"], failed_tests=failed, faults=sorted(f for test in failed for f in task.GROUPS[test]))


def wrong_units(sc):
    a = reference(sc)
    v = a["measured"]
    return dict(a, measured=dict(diameter_y_mm=round(v["diameter_y_mm"] / 10.0, 3), diameter_x_mm=round(v["diameter_x_mm"] / 10.0, 3),
                                 piu_percent=round(v["piu_percent"] / 100.0, 4), ghosting_ratio=round(v["ghosting_ratio"] * 100.0, 3)))


def no_discrimination(sc):
    t = _task(sc)
    m = measure(sc, t)
    failed = _failed(m["values"], t)
    return dict(measured=m["values"], failed_tests=failed, faults=sorted(_first(test, t, m) for test in failed))


def single_fault(sc):
    t = _task(sc)
    m = measure(sc, t)
    failed = _failed(m["values"], t)
    return dict(measured=m["values"], failed_tests=failed, faults=_diagnose(sc, t, m, failed, stop_after=1))


def open_loop(recipe=None):
    """The fixed scan list: the QA scan, then the discriminating scans of ``recipe["order"]`` whatever the QA scan
    showed, as far as the budget goes; then the reference's estimator on what was acquired."""
    order = (recipe or {}).get("order") or ("swap", "flip_map")

    def policy(sc):
        t = _task(sc)
        m = measure(sc, t)
        ro = "x" if t["standard_scan"]["pe_dir"] == "y" else "y"
        scans = {}
        for name in order:
            proto = dict(SWAP, pe_dir=ro) if name == "swap" else dict(FLIP_MAP)
            if not _affordable(sc, proto):
                continue
            if name == "swap":
                r = sc.scan(proto, rois=[dict(id="large0", y_mm=0.0, x_mm=0.0, radius_mm=round(R_LARGE, 2))])
                scans["swap"] = (r["scan_index"], 0.5 * _rows(r["roi_readouts"])["large0"])
            else:
                scans["flip_map"] = sc.scan(proto)["scan_index"]
        failed = _failed(m["values"], t)
        faults = []
        for test in failed:                                              # the estimator may not buy a scan: use what the list acquired
            if test == "geometric_accuracy":
                pe = t["standard_scan"]["pe_dir"]
                off = {a: abs(m["values"][f"diameter_{a}_mm"] - qa.A["diameter_mm"]) for a in ("y", "x")}
                f = f"gradient_{pe}" if off[pe] > off[ro] else (_geometry_fault(sc, t, m, scans["swap"]) if "swap" in scans else _first(test, t, m))
            elif test == "intensity_uniformity":
                f = _uniformity_fault(sc, t, m, scans["flip_map"]) if "flip_map" in scans else _first(test, t, m)
            else:
                f = _ghost_fault(sc, t, m)
            faults.append(f)
        return dict(measured=m["values"], failed_tests=failed, faults=sorted(faults))
    return policy


def policies(recipe=None):
    return dict(reference=reference, prior=prior, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE), brute_force=brute_force,
                wrong_units=wrong_units, no_discrimination=no_discrimination, single_fault=single_fault, open_loop=open_loop(recipe))


def tune(cfgs):
    """The strongest answer that ignores the object, on other worlds: the most frequent fault list with the median of
    the measured values of its worlds; and, for ``open_loop``, the order of the two discriminating scans by how often
    each would have been needed."""
    by = {}
    for c in cfgs:
        by.setdefault(tuple(c["hidden"]["faults"]), []).append(c["hidden"]["measured"])
    faults, rows = max(sorted(by.items()), key=lambda kv: len(kv[1]))
    measured = {q: round(statistics.median(r[q] for r in rows), 4) for q in task.QUANTITIES}
    need = dict(swap=0, flip_map=0)
    for c in cfgs:
        pe = c["task"]["standard_scan"]["pe_dir"]
        ro = "x" if pe == "y" else "y"
        need["swap"] += any(f in (f"gradient_{ro}", "shim") for f in c["hidden"]["faults"])
        need["flip_map"] += any(f in task.GROUPS["intensity_uniformity"] for f in c["hidden"]["faults"])
    return dict(measured=measured, failed_tests=sorted({task.FAULTS[f]["test"] for f in faults}), faults=list(faults),
                order=sorted(need, key=lambda k: (-need[k], k)), tuned_share=round(len(rows) / len(cfgs), 3))


# ------------------------------------------------------------------------------------------------ the label of a level
def stored_gate() -> dict | None:
    """The stored gate result of the id (``mrgym gate <id> --write``) when it was measured on the installed engine,
    else the development run shipped with the family (``gate_dev.json``; its ``n`` is in the file), else whatever is
    stored (a result of another engine: ``task_kind`` names the engine it reads)."""
    import mrgym

    from ... import gate
    g = gate.load(task.ENV_ID)
    if g and g.get("engine") == mrgym.ENGINE_ID:
        return g
    try:
        dev = json.loads(importlib.resources.files(__package__).joinpath("gate_dev.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        dev = None
    return dev if dev and (dev.get("engine") == mrgym.ENGINE_ID or not g) else g or dev


def task_kind(g: dict | None = None) -> dict:
    """Measurement or decision, per level, by the open-loop rule: a fixed scan list with a fixed estimator that passes
    within 25 points of the reference makes the level a measurement task. Read from a gate result, never typed."""
    g = stored_gate() if g is None else g
    if not g:
        return dict(rule="open_loop within 25 points of the reference = measurement task, else decision task", levels={}, source="no gate result stored")
    out = {}
    for lv, s in g["levels"].items():
        ol = next((r for r in s["reported"] if r["policy"] == "open_loop"), None)
        if ol is None:
            continue
        gap = s["reference"]["rate"] - ol["rate"]
        out[lv] = dict(kind="measurement" if gap <= 0.25 else "decision", reference=s["reference"]["rate"], open_loop=ol["rate"],
                       gap_points=round(100 * gap, 1), n=s["n"])
    return dict(rule="open_loop within 25 points of the reference = measurement task, else decision task", levels=out,
                source=f"{g.get('command')} on {g.get('engine')}, {g.get('date_utc')}")
