"""Control policies of ``Scanner/GradientCal``. Every policy plays through the agent's tools only: ``scan``, ``plan``,
``profile`` (the values along a line) and ``read_regions``. None reads an image array.

Must pass
  reference      two scans with the phase-encode axis along y and along x (one scan where the main field is told to be
                 uniform); on each, a profile along its phase-encode axis through the centre of the phantom; edges at
                 half of the plateau just inside them, interpolated; scale = diameter / 190 mm. Where the gradients
                 are told to be non-linear: the shadows of the four pairs of bars on the same line as well, and
                 true / imaged separation of the five pairs against the square of their imaged half separation,
                 extrapolated to the isocentre
Must fail
  prior          no scan; both gradients calibrated (1.000, 1.000)
  fixed_recipe   one token scan, then the pair of scales of another world (tuned on other worlds)
  wrong_units    the reference's answer as the error in percent
  brute_force    a sweep of the receiver bandwidth with one orientation; both diameters on every scan, averaged over
                 the sweep. No model of what the bandwidth changes
  pixel_count    one coarse scan (matrix 64); diameter = number of pixels above half the maximum x the pixel size
  single_scan    the reference's first scan alone, both diameters read on it: no second orientation, no polarity pair
Ablation (must lose at level 3, where the gradients are non-linear)
  diameter_only  the reference without the bars: scale = diameter / 190 mm on the two scans
Reported
  open_loop      the reference played as what it is: a fixed scan list and a fixed estimator
  max_bandwidth  one scan at the strongest readout gradient the console accepts, both diameters read on it: how far
                 "raise the bandwidth" goes without a second scan
Further policies (in ``policies()``, playable with ``mrgym demo --policy``; the gate does not run them)
  global_level   the reference with every edge at half of the mean signal of a large central circle (the edge level
                 of the phantom guidance) instead of half of the plateau next to the edge
  coarse_pixels  the reference on scans with 1.95 mm pixels (matrix 128): what the pixel size costs when the edge is
                 interpolated
"""
from __future__ import annotations

import statistics

import numpy as np

from . import task

REFERENCE = "reference"
MUST_FAIL = ("prior", "fixed_recipe", "wrong_units", "brute_force", "pixel_count", "single_scan")
ABLATIONS = {"diameter_only": (3,)}
REPORTED = ("open_loop", "max_bandwidth")
DESCRIPTIONS = dict(
    reference="two scans with swapped axes; each length on the phase-encode axis; edges at half of the local plateau, interpolated; "
              "with non-linear gradients the bar pairs as well, extrapolated to the isocentre",
    prior="no scan; both scales 1.000",
    fixed_recipe="one token scan, then the pair of scales of another world",
    wrong_units="the reference's answer as the error in percent",
    brute_force="a bandwidth sweep in one orientation; both diameters averaged over the sweep",
    pixel_count="one coarse scan; diameter = pixels above half the maximum x the pixel size",
    single_scan="the reference's first scan alone, both diameters read on it",
    diameter_only="the reference without the bars: scale = diameter / 190 mm",
    open_loop="a fixed scan list (two scans with swapped axes) and the reference's estimator",
    max_bandwidth="one scan at the strongest readout gradient the console accepts, both diameters read on it",
    global_level="the reference with every edge at half of the mean signal of a large central circle",
    coarse_pixels="the reference on scans with 1.95 mm pixels",
)
SCAN = dict(sequence="se", tr_ms=60.0, fov_mm=250.0, matrix=384, bandwidth_hz_per_px=300.0)       # + pe_dir: 26.1 s
COARSE = dict(sequence="se", tr_ms=60.0, fov_mm=250.0, matrix=64)
TOKEN = dict(sequence="se", tr_ms=60.0, fov_mm=250.0, matrix=32)
SWEEP = dict(SCAN, matrix=256)
SWEEP_BANDWIDTH = (150.0, 300.0, 600.0)
MAX_BANDWIDTH_FROM, MAX_BANDWIDTH_STEP = 1200.0, 25.0     # max_bandwidth: the first bandwidth, going down, that the console accepts
LINE_WIDTH_MM = 6.0          # width of the line a diameter is read on
BAR_LINE_WIDTH_MM = 8.0      # width of the line the bar shadows are read on (it stays between the bars that run along it)
EDGE_FROM_MM = 60.0          # an edge profile starts this far from the centre: outside the grid, inside the disc
PLATEAU_MM = (4.0, 12.0)     # the plateau of an edge: the samples this far inside it
BARS_REACH_MM = 64.0         # the bar shadows are read out to this distance from the centre
DIP_HALF_MM, DIP_SHOULDER_MM, DIP_SEARCH_MM = 3.5, 2.5, 2.5      # shadow of a bar: window, shoulders beyond it, search round the guess
LARGE_RADIUS_MM = 75.0       # global_level: the circle whose mean gives the edge level
MAX_SAMPLES = 120            # a profile returns one value per pixel up to 128 samples: lines are kept shorter than this


def _task(sc):
    return sc.info()["task"]


def _edge(pos, val, level=None) -> float:
    """Position where ``val`` (ordered from outside the object inwards) first crosses ``level``: by default half of the
    plateau ``PLATEAU_MM`` inside the first sample above half the maximum. Linear interpolation between two samples."""
    if level is None:
        i = next(j for j, v in enumerate(val) if v >= 0.5 * max(val))
        inside = [v for p, v in zip(pos[i + 1:], val[i + 1:]) if PLATEAU_MM[0] <= abs(p - pos[i]) <= PLATEAU_MM[1]]
        level = 0.5 * (statistics.fmean(inside) if inside else max(val))
    i = next((j for j, v in enumerate(val) if v >= level), len(val) - 1)
    if i == 0 or val[i] == val[i - 1]:
        return pos[i]
    return pos[i - 1] + (level - val[i - 1]) / (val[i] - val[i - 1]) * (pos[i] - pos[i - 1])


def _edges(sc, k, proto, axis, at, level=None):
    """(low edge, high edge) in mm of the disc along ``axis`` on the line at ``at`` (mm on the other axis), on stored
    scan ``k`` of protocol ``proto``: one profile per edge, from the background to ``EDGE_FROM_MM`` from the centre."""
    reach = MAX_SAMPLES * proto["fov_mm"] / proto["matrix"]             # one value per pixel
    whole = EDGE_FROM_MM + reach >= proto["fov_mm"] / 2.0               # the line reaches the edge of the image
    out = []
    for side in (-1, 1):
        if side < 0:
            p = sc.profile(k, axis, position_mm=at, width_mm=LINE_WIDTH_MM, start_mm=None if whole else -EDGE_FROM_MM - reach, stop_mm=-EDGE_FROM_MM)
            pos, val = p["positions_mm"], p["values"]
        else:
            p = sc.profile(k, axis, position_mm=at, width_mm=LINE_WIDTH_MM, start_mm=EDGE_FROM_MM, stop_mm=None if whole else EDGE_FROM_MM + reach)
            pos, val = p["positions_mm"][::-1], p["values"][::-1]
        out.append(_edge(pos, val, level))
    return out[0], out[1]


def _span(sc, k, proto, axis, level=None):
    """(low edge, high edge, position of the line) of the disc along ``axis`` through its centre."""
    other = "x" if axis == "y" else "y"
    a0, a1 = _edges(sc, k, proto, other, 0.0, level)
    at = 0.5 * (a0 + a1)
    b0, b1 = _edges(sc, k, proto, axis, at, level)
    return b0, b1, at


def _bars(sc, k, axis, at, centre, scale) -> dict:
    """{signed nominal position: imaged position} of the shadows of the bars that cross the line along ``axis`` at
    ``at``: the centroid of each dip below the straight line through its two shoulders. ``scale``: a first guess of
    the imaged / true length, to find each shadow."""
    out = {}
    for side in (-1, 1):
        lo, hi = (centre - BARS_REACH_MM, centre - 1.0) if side < 0 else (centre + 1.0, centre + BARS_REACH_MM)
        p = sc.profile(k, axis, position_mm=at, width_mm=BAR_LINE_WIDTH_MM, start_mm=lo, stop_mm=hi)
        pos, val = np.array(p["positions_mm"]), np.array(p["values"])
        for a in task.BARS_MM:
            guess = centre + side * a * scale
            j = int(np.argmin(np.where(np.abs(pos - guess) <= DIP_SEARCH_MM, val, np.inf)))
            far = np.abs(pos - pos[j])
            dip, left, right = far <= DIP_HALF_MM, (far > DIP_HALF_MM) & (far <= DIP_HALF_MM + DIP_SHOULDER_MM) & (pos < pos[j]), \
                (far > DIP_HALF_MM) & (far <= DIP_HALF_MM + DIP_SHOULDER_MM) & (pos > pos[j])
            base = val[left].mean() + (val[right].mean() - val[left].mean()) * (pos[dip] - pos[left].mean()) / (pos[right].mean() - pos[left].mean())
            depth = np.clip(base - val[dip], 0.0, None)
            out[side * a] = float((depth * pos[dip]).sum() / depth.sum())
    return out


def _isocentre_scale(sc, k, axis, at, lo, hi) -> float:
    """Scale at the isocentre under a non-linearity of the form position x (1 - g r^2): for a pair of marks at true
    half separation a and imaged half separation p, true / imaged = (1 - g' p^2) / s exactly, so a straight line
    through (p^2, a / p) of the four bar pairs and the two edges has the intercept 1 / s. Weights: the half separation
    (the error of a ratio falls with it)."""
    bars = _bars(sc, k, axis, at, 0.5 * (lo + hi), (hi - lo) / task.DIAMETER_MM)
    true = np.array([*task.BARS_MM, task.DIAMETER_MM / 2.0])
    imaged = np.array([*(0.5 * (bars[a] - bars[-a]) for a in task.BARS_MM), 0.5 * (hi - lo)])
    design = np.stack([np.ones(len(true)), imaged ** 2], 1)
    intercept = np.linalg.lstsq(design * true[:, None], (true / imaged) * true, rcond=None)[0][0]
    return float(1.0 / intercept)


def _answer(scale: dict) -> dict:
    return dict(scale_y=round(float(scale["y"]), 5), scale_x=round(float(scale["x"]), 5))


def _measure(sc, proto=SCAN, bars=None, level_radius=None, swap=None):
    """The reference's measurement. ``bars``: use the bar pairs (default: where the task says the gradients are not
    linear). ``swap``: a scan per axis (default: unless the task says the main field is uniform). ``level_radius``:
    edges at half of the mean of a centred circle of this radius instead of half of the local plateau."""
    t = _task(sc)
    bars = (not t["told"]["gradients_linear"]) if bars is None else bars
    swap = (not t["told"]["main_field_uniform"]) if swap is None else swap
    scans, levels, scale = {}, {}, {}
    for a in task.AXES:
        if swap or not scans:
            r = sc.scan(dict(proto, pe_dir=a), rois=[dict(id="large", y_mm=0.0, x_mm=0.0, radius_mm=level_radius)] if level_radius else None)
            scans[a] = r["scan_index"]
            levels[a] = 0.5 * r["roi_readouts"][0]["mean_magnitude"][0] if level_radius else None
        else:
            scans[a], levels[a] = scans["y"], levels["y"]
        lo, hi, at = _span(sc, scans[a], proto, a, levels[a])
        scale[a] = _isocentre_scale(sc, scans[a], a, at, lo, hi) if bars else (hi - lo) / task.DIAMETER_MM
    return _answer(scale)


def reference(sc):
    return _measure(sc)


def open_loop(sc):
    return _measure(sc)


def diameter_only(sc):
    return _measure(sc, bars=False)


def global_level(sc):
    return _measure(sc, level_radius=LARGE_RADIUS_MM)


def coarse_pixels(sc):
    return _measure(sc, proto=dict(SCAN, matrix=128))


def single_scan(sc):
    return _measure(sc, bars=False, swap=False)


def max_bandwidth(sc):
    bw = MAX_BANDWIDTH_FROM
    while bw > SCAN["bandwidth_hz_per_px"] and not sc.plan(dict(SCAN, bandwidth_hz_per_px=bw, pe_dir="y"))["accept"]:
        bw -= MAX_BANDWIDTH_STEP
    return _measure(sc, proto=dict(SCAN, bandwidth_hz_per_px=bw), bars=False, swap=False)


def brute_force(sc):
    got = {a: [] for a in task.AXES}
    for bw in SWEEP_BANDWIDTH:
        proto = dict(SWEEP, bandwidth_hz_per_px=bw)
        if sc.plan(dict(proto, pe_dir="y"))["scan_s"] > sc.budget()["remaining_s"]:
            break
        k = sc.scan(dict(proto, pe_dir="y"))["scan_index"]
        for a in task.AXES:
            lo, hi, _ = _span(sc, k, proto, a)
            got[a].append((hi - lo) / task.DIAMETER_MM)
    return _answer({a: statistics.fmean(v) for a, v in got.items()})


def pixel_count(sc):
    k = sc.scan(dict(COARSE))["scan_index"]
    scale = {}
    for a in task.AXES:
        p = sc.profile(k, a, position_mm=0.0)
        scale[a] = sum(v >= 0.5 * max(p["values"]) for v in p["values"]) * p["sample_mm"] / task.DIAMETER_MM
    return _answer(scale)


def prior(sc):
    return dict(scale_y=1.0, scale_x=1.0)


def wrong_units(sc):
    a = reference(sc)
    return {k: round(100.0 * (v - 1.0), 3) for k, v in a.items()}


DEFAULT_RECIPE = dict(scale_y=1.03, scale_x=0.97)


def fixed_recipe(recipe):
    def policy(sc):
        sc.scan(dict(TOKEN))
        return dict(scale_y=float(recipe["scale_y"]), scale_x=float(recipe["scale_x"]))
    return policy


def policies(recipe=None):
    return dict(reference=reference, prior=prior, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE), wrong_units=wrong_units,
                brute_force=brute_force, pixel_count=pixel_count, single_scan=single_scan, diameter_only=diameter_only, open_loop=open_loop,
                max_bandwidth=max_bandwidth, global_level=global_level, coarse_pixels=coarse_pixels)


def tune(cfgs):
    """The strongest answer that ignores the object, on other worlds: the pair of scales of the tuning world that
    passes most of the other tuning worlds under the verifier's rule. (The task text states nothing the scales are a
    function of.)"""
    truth = [c["hidden"]["scale"] for c in cfgs]
    best = None
    for cand in truth:
        passed = sum(all(abs(cand[a] / t[a] - 1.0) <= task.TOLERANCE for a in task.AXES) for t in truth)
        if best is None or passed > best[0]:
            best = (passed, cand)
    return dict(scale_y=best[1]["y"], scale_x=best[1]["x"], tuned_pass_rate=round((best[0] - 1) / max(len(truth) - 1, 1), 3))
