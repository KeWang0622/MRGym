"""Scoring table of ``Scanner/T1Transmit-v1`` (format: ``tests/scoring/__init__.py``).

Pass rule (task statement): for every marked region the delivered T1 lies within a factor 1.1 of the true T1 of its
material, between T1 / 1.1 and 1.1 x T1.

The expected verdicts are derived here from the hidden T1 of each region (``case.truth``) and the factor of the
statement; nothing comes from the task module. The verifier scans nothing: a region is one homogeneous material and
its T1 is a number of the hidden world.
"""
import math

from scoring import Row

ENV_ID = "Scanner/T1Transmit-v1"
INSTANCES = {"L1": dict(level=1, seed=0), "L2": dict(level=2, seed=0), "L3": dict(level=3, seed=1)}
FACTOR = 1.10                   # the pass factor of the task statement


def ids(case):
    return [r["id"] for r in case.cfg["task"]["regions"]]


def at(factor=1.0, **per_region):
    """Deliverable: every region at ``factor`` times its true T1; ``R1=1.2`` overrides the factor of one region."""
    return lambda case: {"t1_ms": {i: case.truth[i]["t1_ms"] * per_region.get(i, factor) for i in ids(case)}}


def with_t1(case, **change):
    a = at()(case)
    a["t1_ms"].update(change)
    return a


def agrees_with_the_rule(case, score):
    """One check per region and the validity of the deliverable; a region passes exactly when its ratio to the truth is
    inside [1 / 1.1, 1.1]; the error is the largest |ln(ratio)|; the diagnostic names every failing region with its
    signed relative error."""
    answer = score["answer"]["t1_ms"]
    ratio = {i: answer[i] / case.truth[i]["t1_ms"] for i in ids(case)}
    inside = {i: 1.0 / FACTOR <= r <= FACTOR for i, r in ratio.items()}
    assert score["checks"] == dict(deliverable_valid=True, **{f"t1_within_tolerance_{i}": ok for i, ok in inside.items()})
    assert abs(score["error"] - max(abs(math.log(r)) for r in ratio.values())) < 1e-12
    assert abs(score["tolerance"] - math.log(FACTOR)) < 1e-15
    assert score["per_region"] == {i: round(r - 1.0, 5) for i, r in ratio.items()}
    assert score["diagnostic"] == "; ".join(f"{i} is off by {100 * (r - 1.0):+.1f} %" for i, r in ratio.items() if not inside[i])
    assert score["distance"] == dict(regions=len(ratio), within=sum(inside.values()), worst_log_ratio=score["error"])


def not_evaluated(case, score):
    """A deliverable the verifier cannot evaluate: one failed check, no error value."""
    assert score["checks"] == dict(deliverable_valid=False) and score["error"] is None and "per_region" not in score


SHAPE = 'the deliverable is {"t1_ms": {'
ROWS = [
    # ---- level 1
    Row("L1", "every region at its true T1", at(), passed=True, expect=agrees_with_the_rule),
    Row("L1", "every region 9 % long", at(1.09), passed=True, expect=agrees_with_the_rule),
    Row("L1", "every region 8 % short (1 / 1.087)", at(0.92), passed=True, expect=agrees_with_the_rule),
    Row("L1", "one region 11 % long", at(R1=1.11), passed=False, failing=("t1_within_tolerance_R1",), reason="R1 is off by +11.0 %",
        expect=agrees_with_the_rule),
    Row("L1", "one region 10 % short: outside, the window is a factor and not a difference", at(R2=0.90), passed=False,
        failing=("t1_within_tolerance_R2",), reason="R2 is off by -10.0 %", expect=agrees_with_the_rule),
    Row("L1", "two regions off, in both directions", at(R1=1.25, R3=0.7), passed=False,
        failing=("t1_within_tolerance_R1", "t1_within_tolerance_R3"), reason="R1 is off by +25.0 %; R3 is off by -30.0 %", expect=agrees_with_the_rule),
    Row("L1", "10.3 % long: outside, the window ends at the factor 1.1", at(R1=1.103), passed=False, failing=("t1_within_tolerance_R1",),
        reason="R1 is off by +10.3 %", expect=agrees_with_the_rule),
    Row("L1", "9.5 % short: outside, the window ends at 1 / 1.1 = 0.909", at(R2=0.905), passed=False, failing=("t1_within_tolerance_R2",),
        reason="R2 is off by -9.5 %", expect=agrees_with_the_rule),
    Row("L1", "9 % short: inside", at(R2=0.91), passed=True, expect=agrees_with_the_rule),
    Row("L1", "integers are numbers", lambda c: {"t1_ms": {i: int(round(v)) for i, v in at()(c)["t1_ms"].items()}}, passed=True),
    # ---- levels 2 and 3: the same rule
    Row("L2", "every region at its true T1", at(), passed=True, expect=agrees_with_the_rule),
    Row("L2", "one region 12 % long", at(R2=1.12), passed=False, failing=("t1_within_tolerance_R2",), reason="R2 is off by +12.0 %",
        expect=agrees_with_the_rule),
    Row("L3", "every region 5 % short", at(0.95), passed=True, expect=agrees_with_the_rule),
    Row("L3", "four regions 15 % short", at(R1=0.85, R2=0.85, R3=0.85, R4=0.85), passed=False,
        failing=tuple(f"t1_within_tolerance_R{k}" for k in (1, 2, 3, 4)), reason="R1 is off by -15.0 %; R2 is off by -15.0 %",
        expect=agrees_with_the_rule),
    # ---- deliverables the verifier cannot evaluate: refused, the episode continues
    Row("L1", "the right numbers in seconds", lambda c: {"t1_ms": {i: v / 1000.0 for i, v in at()(c)["t1_ms"].items()}},
        refused=True, reason="not a T1 in milliseconds, which lies between 30 and 20000 (a value of this size looks like seconds)", expect=not_evaluated),
    Row("L1", "a value above the range", lambda c: with_t1(c, R2=25000.0), refused=True, reason="the T1 of R2 is 25000: not a T1 in milliseconds"),
    Row("L1", "a negative value", lambda c: with_t1(c, R1=-800.0), refused=True, reason="the T1 of R1 is -800: not a T1 in milliseconds, which lies between 30 and 20000"),
    Row("L1", "a value that is not a number", lambda c: with_t1(c, R2="long"), refused=True, reason="the T1 of R2 is not a finite number"),
    Row("L1", "a boolean", lambda c: with_t1(c, R1=True), refused=True, reason="the T1 of R1 is not a finite number"),
    Row("L1", "not finite", lambda c: with_t1(c, R1=float("inf")), refused=True, reason="the T1 of R1 is not a finite number"),
    Row("L1", "a region is missing", lambda c: {"t1_ms": {"R1": 900.0}}, refused=True, reason=SHAPE + '"R1": <number>, "R2": <number>'),
    Row("L1", "a region too many", lambda c: with_t1(c, R99=900.0), refused=True, reason=SHAPE),
    Row("L1", "the v0 deliverable with its units field", lambda c: dict(at()(c), units="ms"), refused=True, reason=SHAPE),
    Row("L1", "the numbers without the field", lambda c: at()(c)["t1_ms"], refused=True, reason=SHAPE),
    Row("L1", "a list", [900.0, 1200.0], refused=True, reason=SHAPE),
    Row("L1", "nothing", None, refused=True, reason=SHAPE, expect=not_evaluated),
    Row("L1", "the field is not an object", {"t1_ms": 900.0}, refused=True, reason=SHAPE),
]
