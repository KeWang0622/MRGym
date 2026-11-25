"""Scoring table of ``Scanner/NoWrap-v1`` (format: ``tests/scoring/__init__.py``).

Pass rule (task statement): the console accepts the geometry; the pixel is at most 2 mm; the whole object lies inside
the image; the image shows no wrap-around; the scan takes at most 10 % longer than the fastest geometry that meets
these requirements, where the field of view may be moved and turned.

Every expected value is derived here from the hidden object and the datasheet, not from the task module.

The object. The body of the QA phantom is an ellipse (centre, two semi-axes, rotation: the layout of the hidden
world) with a soft edge, ``weight = (1 + tanh(-d / 0.5 mm)) / 2`` at ``d`` mm outside the nominal edge, where the
generator measures ``d`` in units of the shorter semi-axis (the ellipse is scaled, not offset). The pass rule calls
"the object" everything above 5 % of the largest proton-density-weighted signal, so its outline is the ellipse
scaled by ``1 + d0 / short``, ``d0 = 0.5 mm x atanh(1 - 2 x 0.05 x largest signal / body density)``.

The fastest geometry. The scan time of the target scan is ``3.0 + 0.5 + 0.5 x lines`` seconds (datasheet: overhead,
one dummy cycle, TR = 500 ms per line). An ellipse is narrowest across its short axis, so the fewest lines are the
smallest even number whose phase field of view, 2 mm per line, holds ``2 x short`` with the 0.5 mm slack of the pass
rule on each side. The master grid has 0.4 mm cells, so the verifier's outline can differ from the ellipse by a
fraction of a cell: the instances of this table are chosen so that this cannot move that even number
(``test_nowrap.py`` asserts the margin), and every row keeps at least 2 mm from the edge it tests.

The image. Pixel ``i`` of ``n`` is centred at ``centre + (i - n / 2) x pixel`` (datasheet), so the image reaches half
a pixel further on the low side: a field of view is centred on a point when its ``centre`` is that point plus half a
pixel along both image axes.
"""
import math

from mrgym.scanner import world as W
from scoring import Row

ENV_ID = "Scanner/NoWrap-v1"
INSTANCES = {"L2": dict(level=2, seed=1), "L3": dict(level=3, seed=0)}
PIXEL, SLACK, FACTOR = 2.0, 0.5, 1.10
EDGE_MM, OBJECT_SHARE = 0.5, 0.05


def scan_s(lines):
    return 3.0 + 0.5 + 0.5 * lines


def even_up(x):
    return 2 * math.ceil(x / 2 - 1e-9)


def body(case):
    """The hidden ellipse of the instance: centre (y, x), long and short semi-axis of the object's outline in mm, and
    ``turn``: the rotation of a field of view whose image rows run along the short axis, in (-90, 90]."""
    b = W.make_world(case.cfg["seed"], case.cfg["world_spec"], layout_only=True)["body"]
    tissues = case.world["tissues"]
    largest = max(t["pd"] for t in tissues.values())
    d0 = EDGE_MM * math.atanh(1.0 - 2.0 * OBJECT_SHARE * largest / tissues["body"]["pd"])
    scale = 1.0 + d0 / min(b["a"], b["b"])
    assert b["a"] > b["b"]                                               # the generator of this family: a is the long semi-axis
    # the short semi-axis lies along (dy, dx) = (sin rot, cos rot); image rows run along (cos turn, sin turn)
    turn = 90.0 - (90.0 - (90.0 - b["rot"])) % 180.0
    return dict(y=b["y"], x=b["x"], long=b["a"] * scale, short=b["b"] * scale, turn=turn)


def fewest_lines(case):
    return even_up((2 * body(case)["short"] - 2 * SLACK) / PIXEL)


def most_lines_allowed(case):
    """The largest even number of lines whose scan time is within the factor of the fastest."""
    limit = (FACTOR * scan_s(fewest_lines(case)) - 3.5) / 0.5
    return 2 * math.floor(limit / 2 + 1e-9)


def field(lines=0, readout_spare_mm=6.0, shift_rows_mm=0.0, shift_cols_mm=0.0, quarter_turn=False, pixel=PIXEL, oversampling=None,
          long_axis=False, lines_spare_mm=None):
    """Deliverable: a field of view centred on the object with its image rows along the short axis of the object.

    lines              phase-encode lines relative to the fewest that hold the object (0: the fastest geometry)
    lines_spare_mm     instead: as many lines as hold the object with this much to spare on each side
    readout_spare_mm   what the readout field of view has to spare on each side of the long axis
    shift_*_mm         the centre moved along the image rows / columns
    quarter_turn       the same field of view written with the phase axis along x and the rotation a quarter turn back
    long_axis          the phase axis along the long axis of the object instead
    """
    def build(case):
        b = body(case)
        across, along = (b["long"], b["short"]) if long_axis else (b["short"], b["long"])
        turn = b["turn"] if not long_axis else 90.0 - (90.0 - (b["turn"] + 90.0)) % 180.0
        n_pe = even_up((2 * across - 2 * SLACK) / pixel) + lines if lines_spare_mm is None else even_up((2 * across + 2 * lines_spare_mm) / pixel)
        n_ro = max(even_up((2 * along + 2 * readout_spare_mm) / pixel), n_pe, 40)
        c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
        rows, cols = pixel / 2 + shift_rows_mm, pixel / 2 + shift_cols_mm                # from the object's centre, along rows and columns
        geo = dict(pe_dir="y", fov_mm=n_ro * pixel, phase_fov=n_pe / n_ro, matrix=n_ro, centre_y_mm=b["y"] + rows * c - cols * s,
                   centre_x_mm=b["x"] + rows * s + cols * c, rotation_deg=turn)
        if quarter_turn:                                                 # rows and columns exchange; columns of the new frame run along the old rows
            t2 = turn - 90.0 if turn > 0 else turn + 90.0
            c2, s2 = math.cos(math.radians(t2)), math.sin(math.radians(t2))
            geo.update(pe_dir="x", rotation_deg=t2, centre_y_mm=b["y"] + (pixel / 2) * (c2 - s2), centre_x_mm=b["x"] + (pixel / 2) * (s2 + c2))
        if oversampling is not None:
            geo["phase_oversampling"] = oversampling
        return geo
    return build


def lines_of(answer):
    n_pe = round(answer["matrix"] * answer["phase_fov"] / 2) * 2
    return round(n_pe * (1 + answer.get("phase_oversampling", 0.0)) / 2) * 2


def agrees_with_the_arithmetic(case, score):
    """Lines, scan seconds, the optimum and the relative excess are the hand values; the checks come in the declared order."""
    lines, best = lines_of(score["answer"]), fewest_lines(case)
    assert (score["lines"], score["lines_fastest"]) == (lines, best)
    assert (score["scan_s"], score["scan_s_fastest"]) == (scan_s(lines), scan_s(best))
    assert abs(score["error"] - (scan_s(lines) / scan_s(best) - 1.0)) < 1e-12 and abs(score["tolerance"] - (FACTOR - 1.0)) < 1e-12
    assert list(score["checks"]) == ["deliverable_valid", "pixel_as_required", "whole_object_inside", "no_wrap_around", "scan_time_within_factor"]
    assert score["checks"]["scan_time_within_factor"] is (scan_s(lines) <= FACTOR * scan_s(best) + 1e-9)
    assert score["pixel_mm"] == round(score["answer"]["fov_mm"] / score["answer"]["matrix"], 4)
    assert score["distance"] == dict(excess_scan_time=score["error"], overhang_mm=score["overhang_mm"], wrap_fraction=score["wrap_fraction"])


def inside_with_room(case, score):
    agrees_with_the_arithmetic(case, score)
    assert score["overhang_mm"] <= 0.5 and score["wrap_fraction"] <= 2e-3


def not_evaluated(case, score):
    assert score["checks"] == dict(deliverable_valid=False) and score["error"] is None and "scan_s" not in score


def changed(**change):
    return lambda case: dict(field()(case), **change)


def without(key):
    return lambda case: {k: v for k, v in field()(case).items() if k != key}


def at_the_allowance(extra=0):
    """The field of view with the most lines the time factor allows, plus ``extra``."""
    return lambda case: field(most_lines_allowed(case) - fewest_lines(case) + extra)(case)


SLOW, CUT, WRAP, PIXEL_CHECK = "scan_time_within_factor", "whole_object_inside", "no_wrap_around", "pixel_as_required"
SHAPE = 'the deliverable is {"pe_dir", "fov_mm", "phase_fov", "matrix", "centre_y_mm", "centre_x_mm", "rotation_deg"} and optionally "phase_oversampling"'
CONSOLE = ""          # the refusal names the violation; its preamble differs between the interface and task.check
ROWS = [
    # ---- level 3: the object is off-centre and turned
    Row("L3", "the fastest geometry: the fewest lines across the short axis, centred on the object", field(), passed=True, expect=inside_with_room),
    Row("L3", "the same field of view written with the phase axis along x and a quarter turn", field(quarter_turn=True), passed=True,
        expect=inside_with_room),
    Row("L3", "two lines more", field(2), passed=True, expect=inside_with_room),
    Row("L3", "the most lines the 10 % allow", at_the_allowance(), passed=True, expect=inside_with_room),
    Row("L3", "two lines beyond the 10 %", at_the_allowance(2), passed=False, failing=(SLOW,), reason="the scan is slower than allowed",
        expect=inside_with_room),
    Row("L3", "ten lines fewer than the object needs: cut off on both sides, and it wraps", field(-10), passed=False, failing=(WRAP, CUT),
        reason="part of the object lies outside the image; the image shows wrap-around", expect=agrees_with_the_arithmetic),
    Row("L3", "the same with phase oversampling: nothing wraps, the object is still cut off, and the scan is slow", field(-10, oversampling=0.5),
        passed=False, failing=(SLOW, CUT), reason="part of the object lies outside the image; the scan is slower than allowed",
        expect=agrees_with_the_arithmetic),
    Row("L3", "the field of view 12 mm off the object along the phase axis", field(shift_rows_mm=12.0), passed=False, failing=(WRAP, CUT),
        reason="part of the object lies outside the image; the image shows wrap-around", expect=agrees_with_the_arithmetic),
    Row("L3", "12 mm off along the readout with 2 mm to spare there: cut off, and the readout does not wrap",
        field(readout_spare_mm=2.0, shift_cols_mm=12.0), passed=False, failing=(CUT,), reason="part of the object lies outside the image",
        expect=agrees_with_the_arithmetic),
    Row("L3", "the phase axis along the long axis: a valid image, too slow", field(long_axis=True, lines_spare_mm=2.0), passed=False, failing=(SLOW,),
        reason="the scan is slower than allowed"),
    Row("L3", "2.5 mm pixels with 3 mm to spare", field(pixel=2.5, lines_spare_mm=3.0), passed=False, failing=(PIXEL_CHECK,),
        reason="the pixel is larger than required"),
    Row("L3", "1.6 mm pixels: allowed, and more lines than the 10 % allow", field(pixel=1.6, lines_spare_mm=1.0), passed=False, failing=(SLOW,),
        reason="the scan is slower than allowed"),
    Row("L3", "the geometry that is safe for any object",
        dict(pe_dir="y", fov_mm=260.0, phase_fov=1.0, matrix=130, centre_y_mm=0.0, centre_x_mm=0.0, rotation_deg=0.0), passed=False, failing=(SLOW,),
        reason="the scan is slower than allowed", expect=agrees_with_the_arithmetic),
    # ---- level 2: off-centre, long axis along x or y
    Row("L2", "the fastest geometry", field(), passed=True, expect=inside_with_room),
    Row("L2", "four lines more: inside the 10 %", field(4), passed=True, expect=inside_with_room),
    Row("L2", "the right size left on the isocentre", lambda c: dict(field()(c), centre_y_mm=1.0, centre_x_mm=1.0), passed=False,
        failing=(WRAP, CUT), reason="part of the object lies outside the image"),
    # ---- deliverables the console cannot evaluate: refused, the episode continues
    Row("L3", "the lengths in centimetres", lambda c: {k: v / 10.0 if k in ("fov_mm", "centre_y_mm", "centre_x_mm") else v for k, v in field()(c).items()},
        refused=True, reason="(a field of view of this size looks like centimetres)", expect=not_evaluated),
    Row("L3", "a field of view below the console's range", changed(fov_mm=60.0), refused=True, reason=CONSOLE + "fov_mm=60.0 outside [80.0, 500.0]"),
    Row("L3", "an odd matrix", changed(matrix=91), refused=True, reason=CONSOLE + "matrix must be even"),
    Row("L3", "a rotation beyond a quarter turn", changed(rotation_deg=120.0), refused=True, reason=CONSOLE + "rotation_deg=120.0 outside [-90.0, 90.0]"),
    Row("L3", "a phase axis that does not exist", changed(pe_dir="z"), refused=True, reason=CONSOLE + "pe_dir must be one of ['y', 'x']"),
    Row("L3", "a pixel below the scanner's minimum", changed(fov_mm=80.0, matrix=512), refused=True, reason=CONSOLE + "pixel size 0.156 mm is below"),
    Row("L3", "the centre is missing (the v0 deliverable)", lambda c: {k: v for k, v in field()(c).items() if not k.startswith(("centre", "rotation"))},
        refused=True, reason=SHAPE),
    Row("L3", "the rotation is missing", without("rotation_deg"), refused=True, reason=SHAPE),
    Row("L3", "a field too many", changed(averages=2), refused=True, reason=SHAPE),
    Row("L3", "a list", [180.0, 0.5, 90], refused=True, reason=SHAPE),
    Row("L3", "nothing", None, refused=True, reason=SHAPE, expect=not_evaluated),
]
