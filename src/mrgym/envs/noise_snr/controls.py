"""Control policies of ``Scanner/NoiseSNR``. Every policy plays through the agent's tools only: ``scan``, ``plan``,
``read_regions`` (means and standard deviations of circles) and ``profile`` (the values of the pixels along a line).
None reads an image array.

Must pass
  reference      the stated protocol twice; signal = mean of each region; noise = SD of the pixel-by-pixel difference
                 of the two scans on eight lines inside the phantom, divided by sqrt(2)
Must fail
  prior          no scan; a typical SNR for every region
  fixed_recipe   one token scan, then the answer of another world: its three SNR values, or its three ratios of SNR
                 to the console's relative-SNR figure of the protocol (tuned on other worlds)
  wrong_units    the reference's answer in decibels (20 log10)
  brute_force    a sweep of the receiver bandwidth round the stated protocol; per region the largest mean / SD of
                 its pixels over the sweep. No model of what that SD holds
  signal_sd      one scan; per region mean / SD of its pixels (the coil shading is in that SD)
  no_rayleigh    one scan; noise = SD of the four background corners, taken as it is (0.655 x the noise at best)
  header_snr     the relative-SNR figure of the console's header line, delivered as the answer
Ablations
  no_sqrt2       the reference without the factor sqrt(2): noise = SD of the difference image (must lose at every level)
  background     the reference with the background method in place of the second scan: noise = SD of the four
                 background corners / 0.655. A correct method wherever the corners are clean air (levels 1 and 2);
                 must lose at level 3, where a ghost covers the background
Reported
  open_loop      the reference played as what it is: a fixed scan list (the protocol twice) and a fixed estimator
  neighbours     one scan; noise = SD of the differences of neighbouring pixels (disjoint pairs) on the same eight lines
                 / sqrt(2). A slow change of the signal cancels between neighbours as the object cancels between two
                 scans; it is right where the noise of neighbouring pixels is independent, which holds for the stated
                 protocols (full k-space, no filter). It shows whether a level can be passed without the second scan
"""
from __future__ import annotations

import math
import statistics

from . import task

REFERENCE = "reference"
MUST_FAIL = ("prior", "fixed_recipe", "wrong_units", "brute_force", "signal_sd", "no_rayleigh", "header_snr")
ABLATIONS = {"no_sqrt2": (1, 2, 3), "background": (3,)}
REPORTED = ("open_loop", "neighbours")
DESCRIPTIONS = dict(
    reference="the protocol twice; noise = SD of the pixel-by-pixel difference on eight lines / sqrt(2); signal = region mean",
    prior="no scan; a typical SNR for every region",
    fixed_recipe="one token scan, then the SNR values of another world (or its ratios to the console's relative-SNR figure)",
    wrong_units="the reference's answer in decibels",
    brute_force="a bandwidth sweep; per region the largest mean / SD of its pixels",
    signal_sd="one scan; mean / SD of the pixels of each region",
    no_rayleigh="one scan; noise = SD of the four background corners without the factor 0.655",
    header_snr="the console's relative-SNR figure as the answer",
    no_sqrt2="the reference with noise = SD of the difference image, not divided by sqrt(2)",
    background="one scan; noise = SD of the four background corners / 0.655",
    open_loop="a fixed scan list (the protocol twice) and the reference's estimator",
    neighbours="one scan; noise = SD of the differences of neighbouring pixels on eight lines / sqrt(2)",
)
RAYLEIGH_SD = math.sqrt(2.0 - math.pi / 2.0)        # SD of the magnitude of complex Gaussian noise of unit SD per component: 0.6551
LINES_MM = tuple(-42.0 + 12.0 * k for k in range(8))     # the noise lines: along x at these y, inside the phantom
LINE_HALF_MM, LINE_HALF_PX = 60.0, 62                # half length of a line: at most this, and at most this many pixels (<= 128 samples)
CORNER_INSET_MM, CORNER_RADIUS_MM = 12.0, 10.0       # background circles: this far inside each corner of the field of view
TOKEN = dict(sequence="se", tr_ms=60.0, matrix=32)
PRIOR_SNR = 150.0
SWEEP_BANDWIDTH = (0.5, 1.0, 2.0)                    # brute_force: multiples of the stated bandwidth


def _task(sc):
    return sc.info()["task"]


def _ids(t):
    return [r["id"] for r in t["regions"]]


def _means(readouts, key="mean_magnitude") -> dict:
    return {r["id"]: r[key][0] for r in readouts}


def _lines(sc, t, k: int) -> list:
    """The pixel values of stored scan ``k`` on the noise lines: one list per line, one value per pixel."""
    px = t["protocol"]["fov_mm"] / t["protocol"]["matrix"]
    half = min(LINE_HALF_MM, LINE_HALF_PX * px)
    out = []
    for pos in LINES_MM:
        p = sc.profile(k, "x", position_mm=pos, start_mm=-half, stop_mm=half)
        if p["pixels_per_sample"] != 1:
            raise RuntimeError("a sample of the profile is not one pixel")
        out.append(p["values"])
    return out


def _difference(sc, t, a: int, b: int) -> list:
    """Pixel-by-pixel differences of two stored scans of the same protocol on the noise lines."""
    return [u - v for la, lb in zip(_lines(sc, t, a), _lines(sc, t, b)) for u, v in zip(la, lb)]


def reference(sc):
    t = _task(sc)
    proto = task.protocol_of(t)
    a, b = sc.scan(proto, rois=t["regions"]), sc.scan(proto, rois=t["regions"])
    sa, sb = _means(a["roi_readouts"]), _means(b["roi_readouts"])
    noise = statistics.stdev(_difference(sc, t, a["scan_index"], b["scan_index"])) / math.sqrt(2.0)
    return dict(snr={i: round(0.5 * (sa[i] + sb[i]) / noise, 3) for i in _ids(t)})


def no_sqrt2(sc):
    return dict(snr={i: round(v / math.sqrt(2.0), 3) for i, v in reference(sc)["snr"].items()})


def _corners(t) -> list:
    c = t["protocol"]["fov_mm"] / 2.0 - CORNER_INSET_MM
    return [dict(id=f"bg{k}", y_mm=sy * c, x_mm=sx * c, radius_mm=CORNER_RADIUS_MM) for k, (sy, sx) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1)))]


def _one_scan(sc, t):
    """The stated protocol once: (signal per region, SD of the pixels per region, mean SD of the four background corners)."""
    r = sc.scan(task.protocol_of(t), rois=t["regions"] + _corners(t))
    mean, sd = _means(r["roi_readouts"]), _means(r["roi_readouts"], "sd_magnitude")
    return ({i: mean[i] for i in _ids(t)}, {i: sd[i] for i in _ids(t)}, statistics.fmean(sd[f"bg{k}"] for k in range(4)))


def background(sc):
    t = _task(sc)
    signal, _, bg = _one_scan(sc, t)
    return dict(snr={i: round(signal[i] / (bg / RAYLEIGH_SD), 3) for i in _ids(t)})


def no_rayleigh(sc):
    t = _task(sc)
    signal, _, bg = _one_scan(sc, t)
    return dict(snr={i: round(signal[i] / bg, 3) for i in _ids(t)})


def signal_sd(sc):
    t = _task(sc)
    signal, sd, _ = _one_scan(sc, t)
    return dict(snr={i: round(signal[i] / sd[i], 3) for i in _ids(t)})


def header_snr(sc):
    t = _task(sc)
    proto = task.protocol_of(t)
    sc.scan(proto)
    rel = sc.plan(proto)["derived"]["relative_snr"]
    return dict(snr={i: max(round(rel, 4), 0.01) for i in _ids(t)})


def brute_force(sc):
    t = _task(sc)
    proto = task.protocol_of(t)
    best = {i: 0.0 for i in _ids(t)}
    for f in SWEEP_BANDWIDTH:
        p = dict(proto, bandwidth_hz_per_px=proto["bandwidth_hz_per_px"] * f)
        pl = sc.plan(p)
        if not (pl["accept"] and pl["scan_s"] <= sc.budget()["remaining_s"]):
            continue
        r = sc.scan(p, rois=t["regions"])
        mean, sd = _means(r["roi_readouts"]), _means(r["roi_readouts"], "sd_magnitude")
        best = {i: max(best[i], mean[i] / sd[i]) for i in best}
    return dict(snr={i: round(max(v, 0.01), 3) for i, v in best.items()})


def prior(sc):
    return dict(snr={i: PRIOR_SNR for i in _ids(_task(sc))})


def wrong_units(sc):
    return dict(snr={i: round(20.0 * math.log10(v), 3) for i, v in reference(sc)["snr"].items()})


DEFAULT_RECIPE = dict(kind="snr", values=[150.0, 190.0, 240.0])


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        sc.scan(dict(TOKEN))
        scale = sc.plan(task.protocol_of(t))["derived"]["relative_snr"] if recipe["kind"] == "snr_over_relative_snr" else 1.0
        return dict(snr={i: min(max(round(v * scale, 3), 0.01), 1e6) for i, v in zip(_ids(t), recipe["values"])})
    return policy


def open_loop(sc):
    return reference(sc)


def neighbours(sc):
    t = _task(sc)
    r = sc.scan(task.protocol_of(t), rois=t["regions"])
    signal = _means(r["roi_readouts"])
    diffs = [line[i] - line[i + 1] for line in _lines(sc, t, r["scan_index"]) for i in range(0, len(line) - 1, 2)]
    noise = statistics.stdev(diffs) / math.sqrt(2.0)
    return dict(snr={i: round(signal[i] / noise, 3) for i in _ids(t)})


def policies(recipe=None):
    return dict(reference=reference, prior=prior, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE), wrong_units=wrong_units,
                brute_force=brute_force, signal_sd=signal_sd, no_rayleigh=no_rayleigh, header_snr=header_snr, background=background,
                open_loop=open_loop, neighbours=neighbours, no_sqrt2=no_sqrt2)


def tune(cfgs):
    """The strongest answer that ignores the object, on other worlds. Candidates: the three SNR values of every tuning
    world, and its three ratios of SNR to the relative-SNR figure the console prints for the protocol (a function of
    what the task states). The one that passes most tuning worlds under the verifier's rule is delivered."""
    from ...scanner import console
    rows = []
    for c in cfgs:
        rel = console.plan(task.protocol_of(c["task"]), dict(b0_t=c["task"]["field_strength_t"], min_pixel_mm=0.5))["derived"]["relative_snr"]
        rows.append(([c["hidden"]["snr"][r["id"]] for r in c["task"]["regions"]], rel))
    best = None
    for kind in ("snr", "snr_over_relative_snr"):
        for values, rel in rows:
            cand = [v / rel for v in values] if kind == "snr_over_relative_snr" else list(values)
            passed = sum(all(abs(c * (r if kind == "snr_over_relative_snr" else 1.0) / v - 1.0) <= task.TOLERANCE for c, v in zip(cand, vals))
                         for vals, r in rows)
            if best is None or passed > best[0]:
                best = (passed, kind, cand)
    return dict(kind=best[1], values=[round(v, 4) for v in best[2]], tuned_pass_rate=round((best[0] - 1) / max(len(rows) - 1, 1), 3))
