"""Scoring table of ``Scanner/FoldAway-v0`` (format: ``tests/scoring/__init__.py``).

Pass rule (task statement), for each of the four regions on the scan delivered for it: the region lies inside the
image; no pixel of the region differs by more than 5 % of the region's mean from the same scan without wrap-around;
the scan takes at most 10 % longer than the fastest geometry that meets the first two.

Every expected value is derived here from the hidden object and the datasheet, not from the task module.

The object. The body of the QA phantom is an ellipse (centre, long and short semi-axis, rotation: the layout of the
hidden world). At level 2 it is upright: the long axis lies along y (rotation 0) or along x (rotation 90).

Where a fold lands. A phase field of view of L mm shows the object again at +L and -L along the phase-encode axis.
On the line at distance c from the centre of the region (across the phase axis) the object ends at e_lo(c) and
e_hi(c) and the circle reaches +-h(c) = sqrt(R^2 - c^2) about the region's centre. The copies miss the circle on
that line when L > e_hi(c) - r + h(c) and L > r - e_lo(c) + h(c) (r: the centre of the region along the axis), and
miss the region when this holds on every line: ``extent`` is the largest of these numbers.

The rows. The scan has 192 x phase_fov lines of 1 mm and takes 3.5 + 0.5 x lines seconds (datasheet). A phase
field of view two pixels and more above ``extent`` keeps the folded edge and the first lobes of its ringing off
the region (1 / (pi^2 d) of the step at d pixels: 5 % at d = 2); one that is 12 mm below it puts 12 mm of a copy on
the region. The fastest geometry lies between these two, so a row that must pass the time factor stays within 5
lines of ``extent`` (at the 50 lines and more of this family: within 9 %), and a row that must fail it adds 40 % and more.
"""
import math

import numpy as np

from mrgym.scanner import world as W
from scoring import Row

ENV_ID = "Scanner/FoldAway-v0"
INSTANCES = {"L2": dict(level=2, seed=0)}
PIXEL, MATRIX, FACTOR, LIMIT = 1.0, 192, 1.10, 0.05
IDS = ("A", "B", "C", "D")


def scan_s(lines):
    return 3.0 + 0.5 + 0.5 * lines


def even_up(x):
    return 2 * math.ceil(x / 2 - 1e-9)


def ends(case, axis, at):
    """The two ends of the hidden ellipse along ``axis`` ("y" or "x") on the line at ``at`` on the other axis (nan outside)."""
    b = W.make_world(case.cfg["seed"], case.cfg["world_spec"], layout_only=True)["body"]
    assert b["rot"] in (0.0, 90.0) and b["a"] > b["b"]                  # level 2: upright; a is the long semi-axis
    half_y, half_x = (b["a"], b["b"]) if b["rot"] == 0.0 else (b["b"], b["a"])
    centre, other, half, half_other = (b["y"], b["x"], half_y, half_x) if axis == "y" else (b["x"], b["y"], half_x, half_y)
    reach = half * np.sqrt(np.maximum(1.0 - ((at - other) / half_other) ** 2, 0.0))
    return centre - reach, centre + reach


def extent(case, region, axis):
    """The smallest phase field of view along ``axis`` whose copies of the object miss the circle (module docstring)."""
    r, other = (region["y_mm"], region["x_mm"]) if axis == "y" else (region["x_mm"], region["y_mm"])
    c = np.linspace(-region["radius_mm"], region["radius_mm"], 2001)
    h = np.sqrt(region["radius_mm"] ** 2 - c ** 2)
    lo, hi = ends(case, axis, other + c)
    return float(np.max(np.maximum(hi - r, r - lo) + h))


def better_axis(case, region):
    return min(("y", "x"), key=lambda axis: extent(case, region, axis))


def field(region_id=None, spare_px=2.0, axis=None, lines=None, **change):
    """Deliverable: every region with the phase axis of its smaller extent, ``spare_px`` pixels above it, centred on the
    region. ``region_id``: only this region differs from that, by ``axis`` / ``lines`` / changed fields."""
    def build(case):
        out = {}
        for region in case.cfg["task"]["regions"]:
            mine = region_id in (None, region["id"])
            ax = (axis if mine and axis else better_axis(case, region))
            n = even_up((extent(case, region, ax) + (spare_px if mine else 2.0) * PIXEL) / PIXEL)
            n = lines if mine and lines else n
            geo = dict(pe_dir=ax, phase_fov=n / MATRIX, centre_y_mm=region["y_mm"], centre_x_mm=region["x_mm"])
            out[region["id"]] = dict(geo, **change) if mine and region_id else geo
        return out
    return build


def lines_of(geo):
    return round(MATRIX * geo["phase_fov"] / 2) * 2


def agrees_with_the_arithmetic(case, score):
    """Lines and scan seconds are the hand values; the checks come requirement by requirement; the optimum of every
    region lies in the window the fold rule leaves it."""
    for region in case.cfg["task"]["regions"]:
        i, row = region["id"], score["regions"][region["id"]]
        lines = lines_of(score["answer"][i])
        assert (row["lines"], row["scan_s"], row["phase_fov_mm"]) == (lines, scan_s(lines), lines * PIXEL)
        least = min(extent(case, region, axis) for axis in ("y", "x"))
        assert least - 2 * PIXEL - 0.5 <= row["lines_fastest"] * PIXEL <= least + 5 * PIXEL
        assert row["scan_s_fastest"] == scan_s(row["lines_fastest"])
        assert score["checks"][f"scan_time_within_factor_{i}"] is (scan_s(lines) <= FACTOR * row["scan_s_fastest"] + 1e-9)
        assert score["checks"][f"no_wrap_on_region_{i}"] is (row["aliased"] <= LIMIT)
    assert list(score["checks"]) == ["deliverable_valid"] + [f"{name}_{i}" for name in ("region_inside_image", "no_wrap_on_region", "scan_time_within_factor") for i in IDS]
    assert abs(score["error"] - max(v["scan_s"] / v["scan_s_fastest"] - 1.0 for v in score["regions"].values())) < 1e-12
    assert abs(score["tolerance"] - (FACTOR - 1.0)) < 1e-12


def clean(case, score):
    agrees_with_the_arithmetic(case, score)
    assert all(v["aliased"] <= LIMIT and v["overhang_mm"] < 0 for v in score["regions"].values())


def not_evaluated(case, score):
    assert score["checks"] == dict(deliverable_valid=False) and score["error"] is None and "regions" not in score


def other_axis(case):
    """Region B with the phase axis of its LARGER extent, two pixels above that extent: clean, and slower."""
    region = next(r for r in case.cfg["task"]["regions"] if r["id"] == "B")
    worse = max(("y", "x"), key=lambda axis: extent(case, region, axis))
    return field("B", axis=worse)(case)


def below(mm):
    """Region A with a phase field of view ``mm`` below its fold-free extent."""
    def build(case):
        region = next(r for r in case.cfg["task"]["regions"] if r["id"] == "A")
        axis = better_axis(case, region)
        return field("A", lines=2 * math.floor((extent(case, region, axis) - mm) / PIXEL / 2))(case)
    return build


def moved_along_readout(case):
    """Region C with the field of view moved 110 mm along the readout, towards the isocentre: the image is 192 mm wide,
    so no pixel of it lies in the region."""
    geo = field()(case)["C"]
    key = "centre_x_mm" if geo["pe_dir"] == "y" else "centre_y_mm"
    return field("C", **{key: geo[key] - math.copysign(110.0, geo[key])})(case)


def one_missing(case):
    return {k: v for k, v in field()(case).items() if k != "D"}


SLOW, WRAP, CUT = "scan_time_within_factor_", "no_wrap_on_region_", "region_inside_image_"
ONE = '{"pe_dir", "phase_fov", "centre_y_mm", "centre_x_mm"}'
SHAPE = "the deliverable is {" + ", ".join(f'"{i}": {ONE}' for i in IDS) + "}"
ROWS = [
    Row("L2", "two pixels above the fold-free extent of every region, along its better axis", field(), passed=True, expect=clean),
    Row("L2", "three pixels above it", field(spare_px=3.0), passed=True, expect=clean),
    Row("L2", "the full phase field of view for region B: nothing folds, 192 lines", field("B", lines=192), passed=False, failing=(SLOW + "B",),
        reason="region B: the scan is slower than allowed", expect=agrees_with_the_arithmetic),
    Row("L2", "the full phase field of view for every region", lambda c: {i: dict(g, phase_fov=1.0) for i, g in field()(c).items()}, passed=False,
        failing=tuple(SLOW + i for i in IDS), reason="region A: the scan is slower than allowed; region B: the scan is slower than allowed",
        expect=agrees_with_the_arithmetic),
    Row("L2", "region B along the axis of its larger extent: clean and too slow", other_axis, passed=False, failing=(SLOW + "B",),
        reason="region B: the scan is slower than allowed", expect=agrees_with_the_arithmetic),
    Row("L2", "region A 12 mm below its fold-free extent: a copy of the object lies on it", below(12.0), passed=False, failing=(WRAP + "A",),
        reason="region A: aliased signal lands on the region", expect=agrees_with_the_arithmetic),
    Row("L2", "the smallest phase field of view of the console for every region: fast and folded", lambda c: {i: dict(g, phase_fov=0.25) for i, g in field()(c).items()},
        passed=False, failing=tuple(WRAP + i for i in IDS), reason="region A: aliased signal lands on the region", expect=agrees_with_the_arithmetic),
    Row("L2", "region C moved out of the image along the readout", moved_along_readout, passed=False, failing=(CUT + "C", WRAP + "C"),
        reason="region C: the region does not lie inside the image"),
    # ---- deliverables the console cannot evaluate: refused, the episode continues
    Row("L2", "the phase field of view in millimetres", lambda c: {i: dict(g, phase_fov=g["phase_fov"] * 192.0) for i, g in field()(c).items()},
        refused=True, reason="(phase_fov is a fraction of fov_mm, not millimetres or percent)", expect=not_evaluated),
    Row("L2", "a phase field of view below the console's range", field("A", phase_fov=0.2), refused=True, reason="phase_fov=0.2 outside [0.25, 1.0]"),
    Row("L2", "a centre outside the console's range", field("B", centre_y_mm=180.0), refused=True, reason="centre_y_mm=180.0 outside [-150.0, 150.0]"),
    Row("L2", "a phase axis that does not exist", field("C", pe_dir="z"), refused=True, reason="pe_dir must be one of ['y', 'x']"),
    Row("L2", "a rotation at a level without one", field("A", rotation_deg=20.0), refused=True, reason=SHAPE),
    Row("L2", "a region is missing", one_missing, refused=True, reason=SHAPE),
    Row("L2", "one geometry instead of one per region", lambda c: field()(c)["A"], refused=True, reason=SHAPE),
    Row("L2", "nothing", None, refused=True, reason=SHAPE, expect=not_evaluated),
]
