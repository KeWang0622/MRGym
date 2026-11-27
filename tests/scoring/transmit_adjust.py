"""Scoring table of ``Scanner/TransmitAdjust-v0`` (format: ``tests/scoring/__init__.py``).

Pass rule (task statement): for every marked region the delivered factor (actual / nominal flip angle) lies within
4 % of the true factor, between 0.96 x and 1.04 x the true factor.

The expected verdicts are derived here from the hidden world: the mean of the transmit pattern over the grid points
inside a region's circle times the transmit gain of the hidden system. Nothing comes from the task module. The
verifier scans nothing.
"""
import numpy as np

from mrgym.scanner import world as W
from scoring import Row

ENV_ID = "Scanner/TransmitAdjust-v0"
INSTANCES = {"L1": dict(level=1, seed=0), "L2": dict(level=2, seed=0), "L3": dict(level=3, seed=0)}
TOLERANCE = 0.04                # the tolerance of the task statement


def ids(case):
    return [r["id"] for r in case.cfg["task"]["regions"]]


def hidden(case) -> dict:
    """{region id: actual / nominal flip angle, averaged over the region} from the hidden world."""
    w = case.world
    y, x = W.grid(w)
    out = {}
    for r in case.cfg["task"]["regions"]:
        inside = (y - r["y_mm"]) ** 2 + (x - r["x_mm"]) ** 2 <= r["radius_mm"] ** 2
        out[r["id"]] = float(np.mean(w["fields"]["b1"][inside])) * w["system"]["tx_gain"]
    return out


def at(factor=1.0, **per_region):
    """Deliverable: every region at ``factor`` times its true ratio; ``R1=1.05`` overrides the factor of one region."""
    return lambda case: {"flip_ratio": {i: v * per_region.get(i, factor) for i, v in hidden(case).items()}}


def with_value(case, **change):
    a = at()(case)
    a["flip_ratio"].update(change)
    return a


def agrees_with_the_rule(case, score):
    """One check per region and the validity of the deliverable; a region passes exactly when delivered / true lies in
    [0.96, 1.04]; the error is the largest |delivered / true - 1|; the diagnostic names every failing region."""
    answer, truth = score["answer"]["flip_ratio"], hidden(case)
    miss = {i: answer[i] / t - 1.0 for i, t in truth.items()}
    inside = {i: abs(m) <= TOLERANCE for i, m in miss.items()}
    assert score["checks"] == dict(deliverable_valid=True, **{f"ratio_within_tolerance_{i}": ok for i, ok in inside.items()})
    assert abs(score["error"] - max(abs(m) for m in miss.values())) < 1e-9 and score["tolerance"] == TOLERANCE
    assert score["per_region"] == {i: round(m, 5) for i, m in miss.items()}
    assert score["diagnostic"] == "; ".join(f"{i} is off by {100 * m:+.1f} %" for i, m in miss.items() if not inside[i])
    assert score["distance"] == dict(regions=len(miss), within=sum(inside.values()), worst_relative=score["error"])


def far_from_one(case, score):
    """The instance is a miscalibrated one: no region is within the tolerance of 1.0 (so the rows about 1.0 mean something)."""
    assert all(abs(1.0 / t - 1.0) > TOLERANCE for t in hidden(case).values())
    agrees_with_the_rule(case, score)


def not_evaluated(case, score):
    """A deliverable the verifier cannot evaluate: one failed check, no error value."""
    assert score["checks"] == dict(deliverable_valid=False) and score["error"] is None and "per_region" not in score


SHAPE = 'the deliverable is {"flip_ratio": {'
ALL6 = tuple(f"ratio_within_tolerance_R{k}" for k in range(1, 7))        # the level-1 instance has six vials
ROWS = [
    # ---- level 1: six regions
    Row("L1", "every region at its true factor", at(), passed=True, expect=agrees_with_the_rule),
    Row("L1", "every region 3.9 % high", at(1.039), passed=True, expect=agrees_with_the_rule),
    Row("L1", "every region 3.9 % low", at(0.961), passed=True, expect=agrees_with_the_rule),
    Row("L1", "one region 4.1 % high", at(R1=1.041), passed=False, failing=("ratio_within_tolerance_R1",), reason="R1 is off by +4.1 %",
        expect=agrees_with_the_rule),
    Row("L1", "one region 4.1 % low", at(R3=0.959), passed=False, failing=("ratio_within_tolerance_R3",), reason="R3 is off by -4.1 %",
        expect=agrees_with_the_rule),
    Row("L1", "two regions off, in both directions", at(R2=1.2, R4=0.7), passed=False,
        failing=("ratio_within_tolerance_R2", "ratio_within_tolerance_R4"), reason="R2 is off by +20.0 %; R4 is off by -30.0 %", expect=agrees_with_the_rule),
    Row("L1", "a calibrated transmitter: 1.0 everywhere", lambda c: {"flip_ratio": {i: 1.0 for i in ids(c)}}, passed=False, failing=ALL6,
        reason="R1 is off by -", expect=far_from_one),
    Row("L1", "the inverse ratio: nominal / actual", lambda c: {"flip_ratio": {i: 1.0 / v for i, v in hidden(c).items()}}, passed=False,
        failing=ALL6, reason="R1 is off by -", expect=agrees_with_the_rule),
    Row("L1", "every region 30 % low: what the 2D scan reads", at(0.70), passed=False, failing=ALL6, reason="R1 is off by -30.0 %",
        expect=agrees_with_the_rule),
    Row("L1", "the value of one region for all of them", lambda c: {"flip_ratio": {i: hidden(c)["R4"] for i in ids(c)}}, passed=False,
        failing=tuple(f"ratio_within_tolerance_R{k}" for k in (2, 3, 5, 6)), reason="R2 is off by -", expect=agrees_with_the_rule),
    # ---- levels 2 and 3: the same rule
    Row("L2", "every region at its true factor", at(), passed=True, expect=agrees_with_the_rule),
    Row("L2", "one region 5 % high", at(R2=1.05), passed=False, failing=("ratio_within_tolerance_R2",), reason="R2 is off by +5.0 %",
        expect=agrees_with_the_rule),
    Row("L3", "every region 2 % low", at(0.98), passed=True, expect=agrees_with_the_rule),
    Row("L3", "the fifth region 6 % high", at(R5=1.06), passed=False, failing=("ratio_within_tolerance_R5",), reason="R5 is off by +6.0 %",
        expect=agrees_with_the_rule),
    # ---- deliverables the verifier cannot evaluate: refused, the episode continues
    Row("L1", "the factors in percent", lambda c: {"flip_ratio": {i: 100.0 * v for i, v in hidden(c).items()}}, refused=True,
        reason="not a ratio of actual to nominal flip angle, which lies between 0.2 and 3 (a value of this size looks like percent or degrees)",
        expect=not_evaluated),
    Row("L1", "the actual flip angle in degrees", lambda c: with_value(c, R2=52.0), refused=True, reason="the factor of R2 is 52: not a ratio"),
    Row("L1", "zero", lambda c: with_value(c, R1=0.0), refused=True, reason="the factor of R1 is 0: not a ratio of actual to nominal flip angle, which lies between 0.2 and 3"),
    Row("L1", "a negative value", lambda c: with_value(c, R1=-0.9), refused=True, reason="the factor of R1 is -0.9: not a ratio"),
    Row("L1", "a value that is not a number", lambda c: with_value(c, R2="low"), refused=True, reason="the factor of R2 is not a finite number"),
    Row("L1", "a boolean", lambda c: with_value(c, R1=True), refused=True, reason="the factor of R1 is not a finite number"),
    Row("L1", "not finite", lambda c: with_value(c, R1=float("inf")), refused=True, reason="the factor of R1 is not a finite number"),
    Row("L1", "a region is missing", lambda c: {"flip_ratio": {"R1": 0.9}}, refused=True,
        reason=SHAPE + ", ".join(f'"R{k}": <number>' for k in range(1, 7)) + "}}"),
    Row("L1", "a region too many", lambda c: with_value(c, R9=0.9), refused=True, reason=SHAPE),
    Row("L1", "a units field", lambda c: dict(at()(c), units="ratio"), refused=True, reason=SHAPE),
    Row("L1", "the numbers without the field", lambda c: at()(c)["flip_ratio"], refused=True, reason=SHAPE),
    Row("L1", "one number for the whole object", 0.87, refused=True, reason=SHAPE),
    Row("L1", "nothing", None, refused=True, reason=SHAPE, expect=not_evaluated),
]
