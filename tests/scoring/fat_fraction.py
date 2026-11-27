"""Scoring table of ``Scanner/FatFraction-v0`` (format: ``tests/scoring/__init__.py``).

Pass rule (task statement): for every marked disc the delivered fat fraction lies within 3 percentage points of the
true one, the proton-density fat fraction of the hidden mixture.

The expected verdicts are derived here from the hidden mixture: every disc is two materials of the world spec on one
circle, and the true fraction is fat share x fat proton density / (that + water share x water proton density). Nothing
comes from the task module. The verifier scans nothing.
"""
from scoring import Row

ENV_ID = "Scanner/FatFraction-v0"
INSTANCES = {"L1": dict(level=1, seed=0), "L2": dict(level=2, seed=0), "L3": dict(level=3, seed=0)}
POINTS = 3.0                    # the pass window of the task statement, percentage points


def ids(case):
    return [r["id"] for r in case.cfg["task"]["regions"]]


def truth(case) -> dict:
    """{disc id: percent}: from the classes of the hidden world (``insert0`` water and ``insert1`` fat of the first disc, ...)."""
    out = {}
    inserts = case.cfg["world_spec"]["qa"]["inserts"]
    for k, i in enumerate(ids(case)):
        amount = {}
        for j in (2 * k, 2 * k + 1):
            cls = case.world["tissues"][f"insert{j}"]
            amount[cls["fat"]] = inserts[j]["weight"] * cls["pd"]
        out[i] = 100.0 * amount[True] / (amount[True] + amount[False])
    return out


def at(shift=0.0, **per_disc):
    """Deliverable: every disc at its true fraction plus ``shift`` points; ``D1=4.0`` overrides the shift of one disc."""
    return lambda case: {"pdff_percent": {i: v + per_disc.get(i, shift) for i, v in truth(case).items()}}


def with_value(case, **change):
    a = at()(case)
    a["pdff_percent"].update(change)
    return a


def agrees_with_the_rule(case, score):
    """One check per disc and the validity of the deliverable; a disc passes exactly when it is within 3 points; the
    error is the largest distance in points; the diagnostic names every failing disc with its signed distance."""
    answer, true = score["answer"]["pdff_percent"], truth(case)
    miss = {i: answer[i] - true[i] for i in ids(case)}
    inside = {i: abs(m) <= POINTS for i, m in miss.items()}
    assert ids(case) == ["D1", "D2", "D3", "D4"]
    assert score["checks"] == dict(deliverable_valid=True, **{f"pdff_within_tolerance_{i}": ok for i, ok in inside.items()})
    assert abs(score["error"] - max(abs(m) for m in miss.values())) < 1e-9 and score["tolerance"] == POINTS
    assert all(abs(score["per_region"][i] - m) < 1e-4 for i, m in miss.items())
    assert score["diagnostic"] == "; ".join(f"{i} is off by {m:+.1f} points" for i, m in miss.items() if not inside[i])
    assert score["distance"] == dict(regions=4, within=sum(inside.values()), worst_points=score["error"])


def the_hidden_fractions_are_in_the_told_range(case, score):
    lo, hi = case.cfg["task"]["pdff_range_percent"]
    assert all(lo < v < hi for v in truth(case).values())


def not_evaluated(case, score):
    """A deliverable the verifier cannot evaluate: one failed check, no error value."""
    assert score["checks"] == dict(deliverable_valid=False) and score["error"] is None and "per_region" not in score


SHAPE = 'the deliverable is {"pdff_percent": {"D1": <number>, "D2": <number>, "D3": <number>, "D4": <number>}}'
ALL = tuple(f"pdff_within_tolerance_D{k}" for k in (1, 2, 3, 4))
ROWS = [
    Row("L1", "every disc at its true fraction", at(), passed=True, expect=agrees_with_the_rule),
    Row("L1", "the told range holds", at(), passed=True, expect=the_hidden_fractions_are_in_the_told_range),
    Row("L1", "every disc 2.9 points high", at(2.9), passed=True, expect=agrees_with_the_rule),
    Row("L1", "every disc 2.9 points low", at(-2.9), passed=True, expect=agrees_with_the_rule),
    Row("L1", "one disc 3.1 points high", at(D1=3.1), passed=False, failing=("pdff_within_tolerance_D1",), reason="D1 is off by +3.1 points",
        expect=agrees_with_the_rule),
    Row("L1", "the textbook formula: every disc 5.5 points low", at(-5.5), passed=False,
        failing=ALL, reason="D1 is off by -5.5 points; D2 is off by -5.5 points; D3 is off by -5.5 points; D4 is off by -5.5 points",
        expect=agrees_with_the_rule),
    Row("L1", "the right numbers as fractions between 0 and 1", lambda c: {"pdff_percent": {i: v / 100.0 for i, v in truth(c).items()}},
        passed=False, failing=ALL, reason="D1 is off by", expect=agrees_with_the_rule),
    Row("L2", "every disc at its true fraction", at(), passed=True, expect=agrees_with_the_rule),
    Row("L2", "two discs off, in both directions", at(D1=4.0, D3=-10.0), passed=False,
        failing=("pdff_within_tolerance_D1", "pdff_within_tolerance_D3"), reason="D1 is off by +4.0 points; D3 is off by -10.0 points", expect=agrees_with_the_rule),
    Row("L3", "every disc at its true fraction", at(), passed=True, expect=agrees_with_the_rule),
    Row("L3", "the told range holds", at(), passed=True, expect=the_hidden_fractions_are_in_the_told_range),
    Row("L3", "every disc on the other side of one half (100 - fraction)", lambda c: {"pdff_percent": {i: 100.0 - v for i, v in truth(c).items()}},
        passed=False, failing=ALL, reason="D1 is off by", expect=agrees_with_the_rule),
    # ---- deliverables the verifier cannot evaluate: refused, the episode continues
    Row("L1", "a value above 100", lambda c: with_value(c, D2=120.0), refused=True,
        reason="the fat fraction of D2 is 120: a fraction in percent lies between 0 and 100", expect=not_evaluated),
    Row("L1", "a negative value", lambda c: with_value(c, D1=-2.0), refused=True, reason="the fat fraction of D1 is -2: a fraction in percent"),
    Row("L1", "a value that is not a number", lambda c: with_value(c, D2="a third"), refused=True, reason="the fat fraction of D2 is not a finite number"),
    Row("L1", "a boolean", lambda c: with_value(c, D1=True), refused=True, reason="the fat fraction of D1 is not a finite number"),
    Row("L1", "not finite", lambda c: with_value(c, D1=float("nan")), refused=True, reason="the fat fraction of D1 is not a finite number"),
    Row("L1", "a disc is missing", lambda c: {"pdff_percent": {"D1": 20.0}}, refused=True, reason=SHAPE),
    Row("L1", "a disc too many", lambda c: with_value(c, D9=20.0), refused=True, reason=SHAPE),
    Row("L1", "the numbers without the field", lambda c: at()(c)["pdff_percent"], refused=True, reason=SHAPE),
    Row("L1", "a list", [10.0, 20.0, 30.0, 40.0], refused=True, reason=SHAPE),
    Row("L1", "nothing", None, refused=True, reason=SHAPE, expect=not_evaluated),
    Row("L1", "the field is not an object", {"pdff_percent": 20.0}, refused=True, reason=SHAPE),
]
