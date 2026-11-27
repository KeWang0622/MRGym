"""Scoring table of ``Scanner/FrequencyAdjust-v0`` (format: ``tests/scoring/__init__.py``).

Pass rule (task statement): for every marked region of water the delivered offset lies within 3 Hz of the mean offset
of the water over the region; at level 3 every marked region of oil is delivered as the word "oil".

The expected verdicts are derived here from the hidden world: the mean of the field over the grid points inside a
region's circle plus the centre-frequency error of the hidden system, and the ``fat`` flag of the vial the region lies
in. Nothing comes from the task module. The verifier scans nothing.
"""
import numpy as np

from mrgym.scanner import world as W
from scoring import Row

ENV_ID = "Scanner/FrequencyAdjust-v0"
INSTANCES = {"L1": dict(level=1, seed=0), "L2": dict(level=2, seed=0), "L3": dict(level=3, seed=0), "L3b": dict(level=3, seed=1)}
TOLERANCE_HZ = 3.0              # the tolerance of the task statement
OIL_IDS = {"L3": ("R2", "R3"), "L3b": ("R1", "R3")}      # the marked regions of oil of the two level-3 instances (asserted below)


def ids(case):
    return [r["id"] for r in case.cfg["task"]["regions"]]


def hidden(case) -> dict:
    """{region id: (off-resonance of water there in Hz, the region is oil)} from the hidden world."""
    w = case.world
    y, x = W.grid(w)
    out = {}
    for r in case.cfg["task"]["regions"]:
        inside = (y - r["y_mm"]) ** 2 + (x - r["x_mm"]) ** 2 <= r["radius_mm"] ** 2
        vial = min(w["regions"], key=lambda v: (v["y_mm"] - r["y_mm"]) ** 2 + (v["x_mm"] - r["x_mm"]) ** 2)
        out[r["id"]] = (float(np.mean(w["fields"]["b0_hz"][inside])) + w["system"]["cf_offset_hz"], bool(w["tissues"][vial["cls"]]["fat"]))
    return out


def at(shift=0.0, **per_region):
    """Deliverable: every region of water at its true offset + ``shift`` Hz and every region of oil as "oil";
    ``R1=3.1`` overrides the shift of one region, ``R2="oil"`` or ``R2=("hz", -410.0)`` sets what is delivered for it."""
    def build(case):
        out = {}
        for i, (offset, oil) in hidden(case).items():
            v = per_region.get(i, shift)
            if isinstance(v, tuple):
                out[i] = v[1]
            elif isinstance(v, str):
                out[i] = v
            else:
                out[i] = "oil" if oil and i not in per_region else offset + v
        return {"offset_hz": out}
    return build


def with_value(case, **change):
    a = at()(case)
    a["offset_hz"].update(change)
    return a


def agrees_with_the_rule(case, score):
    """One check per region: a region of water passes exactly when a number within 3 Hz of the hidden offset was
    delivered; a region of oil exactly when "oil" was. The error is the largest miss of the numbers delivered for water."""
    answer, truth = score["answer"]["offset_hz"], hidden(case)
    if case.level == 3:
        key = next(k for k, v in INSTANCES.items() if v == dict(level=case.level, seed=case.seed))
        assert tuple(i for i, (_, oil) in truth.items() if oil) == OIL_IDS[key]            # the rows below name these regions
    else:
        assert not any(oil for _, oil in truth.values())
    checks, notes, miss = dict(deliverable_valid=True), [], {}
    for i, (offset, oil) in truth.items():
        v = answer[i]
        if oil:
            checks[f"oil_named_{i}"] = v == "oil"
            if v != "oil":
                notes.append(f"{i} holds oil: {float(v):g} Hz is not a water frequency ({float(v) - offset:+.0f} Hz from the water offset there)")
        elif v == "oil":
            checks[f"offset_within_tolerance_{i}"] = False
            notes.append(f"{i} holds water")
        else:
            miss[i] = v - offset
            checks[f"offset_within_tolerance_{i}"] = abs(miss[i]) <= TOLERANCE_HZ
            if abs(miss[i]) > TOLERANCE_HZ:
                notes.append(f"{i} is off by {miss[i]:+.1f} Hz")
    assert score["checks"] == checks and score["diagnostic"] == "; ".join(notes)
    worst = max(abs(m) for m in miss.values()) if miss else None
    assert (score["error"] is None and worst is None) or abs(score["error"] - worst) < 1e-9
    assert score["tolerance_hz"] == TOLERANCE_HZ
    assert score["per_region"] == {i: round(m, 3) for i, m in miss.items()}
    assert score["distance"] == dict(regions=len(truth), right=sum(v for k, v in checks.items() if k != "deliverable_valid"), worst_hz=score["error"])
    assert score["species_wrong"] == [i for i, (_, oil) in truth.items() if (answer[i] == "oil") != oil]


def not_evaluated(case, score):
    """A deliverable the verifier cannot evaluate: one failed check, no error value."""
    assert score["checks"] == dict(deliverable_valid=False) and score["error"] is None and "per_region" not in score


SHAPE = 'the deliverable is {"offset_hz": {'
ROWS = [
    # ---- level 1: two regions of water
    Row("L1", "every region at its true offset", at(), passed=True, expect=agrees_with_the_rule),
    Row("L1", "every region 2.9 Hz high", at(2.9), passed=True, expect=agrees_with_the_rule),
    Row("L1", "every region 2.9 Hz low", at(-2.9), passed=True, expect=agrees_with_the_rule),
    Row("L1", "one region 3.1 Hz high", at(R1=3.1), passed=False, failing=("offset_within_tolerance_R1",), reason="R1 is off by +3.1 Hz",
        expect=agrees_with_the_rule),
    Row("L1", "one region 3.1 Hz low", at(R2=-3.1), passed=False, failing=("offset_within_tolerance_R2",), reason="R2 is off by -3.1 Hz",
        expect=agrees_with_the_rule),
    Row("L1", "both regions off, in both directions", at(R1=40.0, R2=-7.5), passed=False,
        failing=("offset_within_tolerance_R1", "offset_within_tolerance_R2"), reason="R1 is off by +40.0 Hz; R2 is off by -7.5 Hz", expect=agrees_with_the_rule),
    Row("L1", "the sign reversed", lambda c: {"offset_hz": {i: -v for i, v in at()(c)["offset_hz"].items()}}, passed=False,
        failing=("offset_within_tolerance_R1", "offset_within_tolerance_R2"), reason="R1 is off by -", expect=agrees_with_the_rule),
    Row("L1", "the phase slope in rad/s", lambda c: {"offset_hz": {i: 2 * np.pi * v for i, v in at()(c)["offset_hz"].items()}}, passed=False,
        failing=("offset_within_tolerance_R1", "offset_within_tolerance_R2"), reason="R1 is off by +", expect=agrees_with_the_rule),
    Row("L1", "zero everywhere", lambda c: {"offset_hz": {i: 0.0 for i in ids(c)}}, passed=False,
        failing=("offset_within_tolerance_R1", "offset_within_tolerance_R2"), reason="R2 is off by -", expect=agrees_with_the_rule),
    Row("L1", "integers are numbers", lambda c: {"offset_hz": {i: int(round(v)) for i, v in at()(c)["offset_hz"].items()}}, passed=True,
        expect=agrees_with_the_rule),
    # ---- level 2: three regions, the same rule
    Row("L2", "every region at its true offset", at(), passed=True, expect=agrees_with_the_rule),
    Row("L2", "one region wrapped by 1 / (2 ms)", at(R2=500.0), passed=False, failing=("offset_within_tolerance_R2",),
        reason="R2 is off by +500.0 Hz", expect=agrees_with_the_rule),
    Row("L2", "one value for every region: the offset of R1", lambda c: {"offset_hz": {i: at()(c)["offset_hz"]["R1"] for i in ids(c)}},
        passed=False, failing=("offset_within_tolerance_R2",), reason="R2 is off by -5.4 Hz", expect=agrees_with_the_rule),
    # ---- level 3: one region of water and two of oil; two of oil around one of water
    Row("L3", "water at its true offset, oil named", at(), passed=True, expect=agrees_with_the_rule),
    Row("L3", "water 2 Hz low, oil named", at(-2.0), passed=True, expect=agrees_with_the_rule),
    Row("L3", "the frequency read on one region of oil", at(R2=-434.0), passed=False, failing=("oil_named_R2",),
        reason="R2 holds oil", expect=agrees_with_the_rule),
    Row("L3", "the water offset delivered for a region of oil: still not a reference", at(R3=0.0), passed=False, failing=("oil_named_R3",),
        reason="(+0 Hz from the water offset there)", expect=agrees_with_the_rule),
    Row("L3", "a number for every region", at(R2=-430.0, R3=-441.0), passed=False, failing=("oil_named_R2", "oil_named_R3"),
        reason="R2 holds oil", expect=agrees_with_the_rule),
    Row("L3", "the region of water named as oil", at(R1="oil"), passed=False, failing=("offset_within_tolerance_R1",), reason="R1 holds water",
        expect=agrees_with_the_rule),
    Row("L3", "oil named, water 4 Hz high", at(R1=4.0), passed=False, failing=("offset_within_tolerance_R1",), reason="R1 is off by +4.0 Hz",
        expect=agrees_with_the_rule),
    Row("L3b", "water at its true offset, oil named", at(), passed=True, expect=agrees_with_the_rule),
    Row("L3b", "species exchanged", at(R1=0.0, R2="oil", R3=0.0), passed=False, failing=("oil_named_R1", "offset_within_tolerance_R2", "oil_named_R3"),
        reason="R2 holds water", expect=agrees_with_the_rule),
    # ---- deliverables the verifier cannot evaluate: refused, the episode continues
    Row("L1", "the word oil where every region holds water", lambda c: with_value(c, R1="oil"), refused=True,
        reason='the offset of R1 is "oil": every marked region of this task holds water; deliver a number in Hz', expect=not_evaluated),
    Row("L1", "a value far outside any off-resonance", lambda c: with_value(c, R2=64.0e6), refused=True,
        reason="the offset of R2 is 6.4e+07: not an off-resonance in Hz, which lies between -5000 and 5000"),
    Row("L1", "a value that is not a number", lambda c: with_value(c, R2="high"), refused=True, reason="the offset of R2 is not a finite number"),
    Row("L1", "a boolean", lambda c: with_value(c, R1=True), refused=True, reason="the offset of R1 is not a finite number"),
    Row("L1", "not finite", lambda c: with_value(c, R1=float("nan")), refused=True, reason="the offset of R1 is not a finite number"),
    Row("L1", "null", lambda c: with_value(c, R1=None), refused=True, reason="the offset of R1 is not a finite number"),
    Row("L1", "a region is missing", lambda c: {"offset_hz": {"R1": 10.0}}, refused=True, reason=SHAPE + '"R1": <number>, "R2": <number>}}'),
    Row("L1", "a region too many", lambda c: with_value(c, R9=10.0), refused=True, reason=SHAPE),
    Row("L1", "a units field", lambda c: dict(at()(c), units="Hz"), refused=True, reason=SHAPE),
    Row("L1", "the numbers without the field", lambda c: at()(c)["offset_hz"], refused=True, reason=SHAPE),
    Row("L1", "one number", 12.5, refused=True, reason=SHAPE),
    Row("L1", "nothing", None, refused=True, reason=SHAPE, expect=not_evaluated),
    Row("L3", "another word than oil", lambda c: with_value(c, R2="fat"), refused=True, reason='the offset of R2 is not a finite number (or the word "oil")'),
    Row("L3", "null for a region of oil", lambda c: with_value(c, R2=None), refused=True, reason='the offset of R2 is not a finite number (or the word "oil")'),
    Row("L3", "a region is missing", lambda c: {"offset_hz": {"R1": 10.0, "R2": "oil"}}, refused=True,
        reason=SHAPE + '"R1": <number, or "oil">, "R2": <number, or "oil">, "R3": <number, or "oil">}}'),
]
