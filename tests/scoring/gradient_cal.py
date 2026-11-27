"""Scoring table of ``Scanner/GradientCal-v0`` (format: ``tests/scoring/__init__.py``).

Pass rule (task statement): scale_y and scale_x are each within 0.3 % of the true scale of that gradient.

Nothing here comes from the task module. The expected values are read from the hidden world: the calibration error of
the session, ``world["system"]["grad_scale"]`` = (strength of the y gradient, of the x gradient) / nominal, which is
the factor by which the scanner draws a length along that axis (``scanner/world.py::_warp``; checked on images in
``tests/test_gradient_cal.py``).
"""
from scoring import Row

ENV_ID = "Scanner/GradientCal-v0"
# The first instance must differ from the instance of the seed "another" in what the task states (the field strength).
INSTANCES = {"L1": dict(level=1, seed=1), "L0": dict(level=0, seed=0), "L2": dict(level=2, seed=0), "L3": dict(level=3, seed=2)}
TOLERANCE = 0.003               # the pass window of the task statement


def truth(case) -> dict:
    sy, sx = case.world["system"]["grad_scale"]
    return dict(scale_y=float(sy), scale_x=float(sx))


def at(y=1.0, x=1.0):
    """Deliverable: each true scale times a factor."""
    return lambda case: dict(scale_y=truth(case)["scale_y"] * y, scale_x=truth(case)["scale_x"] * x)


def agrees(case, score):
    want, got = truth(case), score["answer"]
    assert score["scale"] == dict(y=round(want["scale_y"], 6), x=round(want["scale_x"], 6)) and score["tolerance"] == TOLERANCE
    worst = max(abs(got[k] / want[k] - 1.0) for k in want)
    assert abs(score["error"] - worst) <= 1e-12
    assert all(0.01 <= abs(v - 1.0) <= 0.05 for v in want.values())     # the hidden error is 1 to 5 % of either sign


ROWS = [
    Row("L1", "the true scales", at(), passed=True, expect=agrees),
    Row("L1", "0.29 % long along y, 0.29 % short along x", at(1.0029, 0.9971), passed=True, expect=agrees),
    Row("L1", "0.31 % long along y", at(y=1.0031), passed=False, failing=("scale_y",), reason="scale_y: delivered", expect=agrees),
    Row("L1", "0.31 % short along x", at(x=0.9969), passed=False, failing=("scale_x",), reason="scale_x: delivered", expect=agrees),
    Row("L1", "the two scales swapped", lambda case: dict(scale_y=truth(case)["scale_x"], scale_x=truth(case)["scale_y"]), passed=False,
        failing=("scale_x", "scale_y"), reason="scale_y: delivered"),
    Row("L1", "both gradients taken as calibrated", dict(scale_y=1.0, scale_x=1.0), passed=False, failing=("scale_x", "scale_y"), reason="the gradient's is"),
    Row("L1", "the inverse: true / imaged length", lambda case: {k: 1.0 / v for k, v in truth(case).items()}, passed=False,
        failing=("scale_x", "scale_y"), reason="scale_x: delivered"),
    Row("L0", "the true scales", at(), passed=True, expect=agrees),
    Row("L2", "the true scales", at(), passed=True, expect=agrees),
    Row("L2", "a field gradient of 8 Hz/mm read as a gradient error at 300 Hz per 0.65 mm pixel (1.7 %)", at(x=1.0 - 8.0 * 0.651 / 300.0), passed=False,
        failing=("scale_x",), reason="(-1.74 %)", expect=agrees),
    Row("L3", "the true scales", at(), passed=True, expect=agrees),
    Row("L3", "the scale at the edge of the disc under the smallest non-linearity (1.2 %)", at(1.0 + 0.08 * (95.0 / 250.0) ** 2, 1.0 + 0.08 * (95.0 / 250.0) ** 2),
        passed=False, failing=("scale_x", "scale_y"), reason="(+1.16 %)", expect=agrees),
    Row("L1", "one scale is missing", dict(scale_y=1.02), refused=True, reason='the deliverable is {"scale_y": <number>, "scale_x": <number>}'),
    Row("L1", "a field too many", dict(scale_y=1.02, scale_x=0.98, scale_z=1.0), refused=True, reason='the deliverable is {"scale_y"'),
    Row("L1", "the error in percent", dict(scale_y=2.3, scale_x=-1.5), refused=True, reason="scale_y must be a number from 0.5 to 2"),
    Row("L1", "a length in mm", dict(scale_y=1.02, scale_x=193.8), refused=True, reason="scale_x must be a number from 0.5 to 2"),
    Row("L1", "a text", dict(scale_y="too long", scale_x=0.98), refused=True, reason="scale_y must be a number"),
    Row("L1", "true", dict(scale_y=True, scale_x=0.98), refused=True, reason="scale_y must be a number"),
    Row("L1", "not a list", [1.02, 0.98], refused=True, reason="the deliverable is"),
]
