"""Scoring table of ``Scanner/Resolve-v0`` (format: ``tests/scoring/__init__.py``).

Pass rule (task statement), for each of the three pairs on the scan delivered for it: the whole object lies inside the
image; the two features are separated (the image read between its pixels falls, between its two peaks, to at most
half of the lower peak); the SNR of the reference region is at least 15; the scan takes at most 60 s; and at most
20 % longer than the fastest geometry that meets the first four.

Every expected value is derived here from the hidden object and the datasheet, not from the task module.

The object. The body of the QA phantom is an ellipse (centre, long and short semi-axis: the layout of the hidden
world; at level 2 the long axis lies along x) with a soft edge, ``weight = (1 + tanh(-d / 0.3 mm)) / 2`` at ``d`` mm
outside the nominal edge, ``d`` measured in units of the shorter semi-axis (the ellipse is scaled, not offset). The
pass rule calls "the object" everything above 5 % of the largest proton-density-weighted signal (the features, 1.0),
so its outline is the ellipse scaled by ``1 + d0 / short``, ``d0 = 0.3 mm x atanh(1 - 2 x 0.05 / body density)``.

The image. Pixel ``i`` of ``n`` is centred at ``(i - n / 2) x pixel`` (datasheet; the field of view is centred on
the isocentre), so ``n`` pixels reach from ``-(n + 1) x pixel / 2`` to ``(n - 1) x pixel / 2``.

Separation. The image of a point is sinc(x / pixel): two points merge at a spacing of one pixel and show a dip of one
half at 1.63 pixels (a pixel of 0.61 spacings). The features are discs half as wide as their spacing, which lowers
that to 0.57 to 0.60 spacings. So a pixel of 0.55 spacings and less separates a pair, a pixel of 0.75 spacings and
more does not, and the fastest geometry has a pixel between 0.50 and 0.61 spacings (0.50: two lines fewer at 28 lines
and more). The scan has ``lines = matrix x phase_fov`` lines and takes 3.5 + 0.5 x lines seconds (datasheet).
"""
import math

from mrgym.scanner import world as W
from scoring import Row

ENV_ID = "Scanner/Resolve-v0"
INSTANCES = {"L2": dict(level=2, seed=0)}
FACTOR, CAP_S, EDGE_MM, OBJECT_SHARE = 1.20, 60.0, 0.3, 0.05
IDS = ("A", "B", "C")


def scan_s(lines):
    return 3.0 + 0.5 + 0.5 * lines


def even_up(x):
    return 2 * math.ceil(x / 2 - 1e-9)


def box(case):
    """Low and high end of the object's outline along y and along x, mm from the isocentre."""
    b = W.make_world(case.cfg["seed"], case.cfg["world_spec"], layout_only=True)["body"]
    assert b["rot"] == 90.0 and b["a"] > b["b"]                          # level 2: the long semi-axis a lies along x
    d0 = EDGE_MM * math.atanh(1.0 - 2.0 * OBJECT_SHARE / case.world["tissues"]["body"]["pd"])
    scale = 1.0 + d0 / b["b"]
    return dict(y=(b["y"] - b["b"] * scale, b["y"] + b["b"] * scale), x=(b["x"] - b["a"] * scale, b["x"] + b["a"] * scale))


def spacing(case, pair):
    return case.cfg["hidden"]["pairs"][pair]["spacing_mm"]


def pixels_for(lo, hi, pixel, spare_mm):
    """The smallest even number of pixels about the isocentre that holds [lo, hi] with ``spare_mm`` on each side."""
    return even_up(2 * (max(-lo - pixel / 2, hi + pixel / 2) + spare_mm) / pixel)


def geometry(case, pixel, pe_dir="y", spare_mm=1.0, lines=0):
    b = box(case)
    pe, ro = (b["y"], b["x"]) if pe_dir == "y" else (b["x"], b["y"])
    n_pe = pixels_for(*pe, pixel, spare_mm) + lines
    n_ro = max(pixels_for(*ro, pixel, spare_mm), n_pe, 32)
    return dict(pe_dir=pe_dir, fov_mm=round(pixel * n_ro, 6), phase_fov=round(n_pe / n_ro, 6), matrix=n_ro)


def field(pair=None, share=0.55, **how):
    """Deliverable: for every pair a pixel of 0.55 spacings, the phase axis along y (the short side) and the smallest
    field of view that holds the object with 1 mm to spare. ``pair``: only this pair differs, by ``share`` and ``how``
    (arguments of ``geometry``)."""
    def build(case):
        return {i: geometry(case, (share if pair in (None, i) else 0.55) * spacing(case, i), **(how if pair in (None, i) else {})) for i in IDS}
    return build


def lines_of(geo):
    return round(geo["matrix"] * geo["phase_fov"] / 2) * 2


def agrees_with_the_arithmetic(case, score):
    """Lines, scan seconds and pixel are the hand values; the checks come requirement by requirement; the optimum of
    every pair lies in the window the sinc leaves it."""
    names = ("whole_object_inside", "features_separated", "snr_floor", "within_time_cap", "scan_time_within_factor")
    assert list(score["checks"]) == ["deliverable_valid"] + [f"{name}_{i}" for name in names for i in IDS]
    lo, hi = box(case)["y"]
    for i in IDS:
        row, geo = score["pairs"][i], score["answer"][i]
        lines = lines_of(geo)
        assert (row["lines"], row["scan_s"], row["pixel_mm"]) == (lines, scan_s(lines), round(geo["fov_mm"] / geo["matrix"], 4))
        assert 0.50 * spacing(case, i) <= row["pixel_mm_fastest"] <= 0.61 * spacing(case, i)
        p = row["pixel_mm_fastest"]                                      # ... and its phase field of view just holds the object
        assert row["scan_s_fastest"] == scan_s(row["lines_fastest"]) and abs(row["lines_fastest"] * p - max(2 * hi + p, -2 * lo - p)) <= 1.5
        assert score["checks"][f"scan_time_within_factor_{i}"] is (scan_s(lines) <= FACTOR * row["scan_s_fastest"] + 1e-9)
        assert score["checks"][f"within_time_cap_{i}"] is (scan_s(lines) <= CAP_S)
        assert score["checks"][f"features_separated_{i}"] is (row["dip"] >= 0.5) and score["checks"][f"snr_floor_{i}"] is (row["snr"] >= 15.0)
    assert abs(score["error"] - max(v["scan_s"] / v["scan_s_fastest"] - 1.0 for v in score["pairs"].values())) < 1e-12
    assert abs(score["tolerance"] - (FACTOR - 1.0)) < 1e-12


def separated_with_room(case, score):
    agrees_with_the_arithmetic(case, score)
    assert all(v["dip"] >= 0.5 and v["overhang_mm"] <= 0.5 and v["snr"] >= 15.0 for v in score["pairs"].values())


def merged(case, score):
    agrees_with_the_arithmetic(case, score)
    assert score["pairs"]["A"]["dip"] < 0.1


def not_evaluated(case, score):
    assert score["checks"] == dict(deliverable_valid=False) and score["error"] is None and "pairs" not in score


def changed(pair, **change):
    return lambda case: dict(field()(case), **{pair: dict(field()(case)[pair], **change)})


SLOW, CUT, MERGED, CAP, SNR = "scan_time_within_factor_", "whole_object_inside_", "features_separated_", "within_time_cap_", "snr_floor_"
ONE = '{"pe_dir", "fov_mm", "phase_fov", "matrix"}'
SHAPE = "the deliverable is {" + ", ".join(f'"{i}": {ONE}' for i in IDS) + "}"
ROWS = [
    Row("L2", "a pixel of 0.55 spacings for every pair, phase along the short side, 1 mm to spare", field(), passed=True, expect=separated_with_room),
    Row("L2", "the same with 3 mm to spare", field(spare_mm=3.0), passed=True, expect=separated_with_room),
    Row("L2", "pair A with a pixel as large as its spacing: the two features merge", field("A", share=1.0), passed=False, failing=(MERGED + "A",),
        reason="pair A: the two features are not separated", expect=merged),
    Row("L2", "pair A with a pixel of 0.75 spacings", field("A", share=0.75), passed=False, failing=(MERGED + "A",),
        reason="pair A: the two features are not separated", expect=agrees_with_the_arithmetic),
    Row("L2", "pair B with a pixel of 0.4 spacings: separated, and half as many lines again", field("B", share=0.4), passed=False, failing=(SLOW + "B",),
        reason="pair B: the scan is slower than allowed", expect=agrees_with_the_arithmetic),
    Row("L2", "pair C with the phase axis along the long side: separated and too slow", field("C", pe_dir="x"), passed=False, failing=(SLOW + "C",),
        reason="pair C: the scan is slower than allowed", expect=agrees_with_the_arithmetic),
    Row("L2", "pair B with six lines fewer than hold the object", field("B", lines=-6), passed=False, failing=(CUT + "B",),
        reason="pair B: part of the object lies outside the image", expect=agrees_with_the_arithmetic),
    Row("L2", "every pair with the console's default: 240 mm at 128 x 128, 67.5 s",
        lambda c: {i: dict(pe_dir="y", fov_mm=240.0, phase_fov=1.0, matrix=128) for i in IDS}, passed=False,
        failing=tuple(CAP + i for i in IDS) + tuple(SLOW + i for i in IDS), reason="pair A: the scan is longer than the examination affords"),
    # ---- deliverables the console cannot evaluate: refused, the episode continues
    Row("L2", "the field of view of pair B in centimetres", lambda c: dict(field()(c), B=dict(field()(c)["B"], fov_mm=field()(c)["B"]["fov_mm"] / 10.0)),
        refused=True, reason="(a field of view of this size looks like centimetres)", expect=not_evaluated),
    Row("L2", "an odd matrix", changed("A", matrix=61), refused=True, reason="matrix must be even"),
    Row("L2", "a phase axis that does not exist", changed("C", pe_dir="z"), refused=True, reason="pe_dir must be one of ['y', 'x']"),
    Row("L2", "a pixel below the scanner's minimum", changed("A", fov_mm=240.0, matrix=400), refused=True, reason="pixel size 0.600 mm is below"),
    Row("L2", "a field too many", changed("B", averages=2), refused=True, reason=SHAPE),
    Row("L2", "a pair is missing", lambda c: {k: v for k, v in field()(c).items() if k != "C"}, refused=True, reason=SHAPE),
    Row("L2", "one geometry instead of one per pair", lambda c: field()(c)["A"], refused=True, reason=SHAPE),
    Row("L2", "nothing", None, refused=True, reason=SHAPE, expect=not_evaluated),
]
