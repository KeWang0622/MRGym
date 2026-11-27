"""Scoring table of ``Scanner/ErnstFlip-v0`` (format: ``tests/scoring/__init__.py``).

Pass rule (task statement): for every marked region, the target scan (2D spoiled gradient echo at the stated
repetition time) at the delivered nominal flip angle gives at least 99.5 % of the largest signal the region can give
at any flip angle from 1 to 90 degrees.

The expected fraction is derived here, not taken from the task module, from ``ernst_flip/model.py`` or from the
engine's signal code:

  pulse    128 hard sub-pulses with the amplitudes of a sinc of time-bandwidth product 4 under a Hann window, summing
           to the flip angle at the slice centre (nominal angle x B1 of the region); precession about z between them
           in proportion to the height in the slice; a rephasing lobe of half the selection area. Simulated as a
           cascade of 3 x 3 rotations of the magnetisation vector (the engine and the family's public model use
           Cayley-Klein parameters).
  column   a column at height z sees the flip angle theta(z) = arccos(Mz after the pulse) every repetition. With
           ideal spoiling its longitudinal magnetisation before a pulse is (1 - E) / (1 - E cos(theta)), E =
           exp(-TR / T1) (the Ernst steady state), and the pulse turns it into transverse magnetisation along the
           direction the cascade gives.
  signal   the sum over heights (40 per slice thickness out to 3.5 thicknesses; the engine uses 24) of the transverse
           component along the direction of the slice centre.

The scanner serves the steady state with RF spoiling, finite T2 and diffusion (extended phase graphs), so the two
differ, more the further an angle is from the best one, and most for a short T1 with a long T2 at a short repetition
time (spoiling is least complete there). ``atol(fraction)`` is the allowance: 3.5e-3 at the maximum, growing by 0.5
per unit of signal lost (measured on these instances on engine 0.2.0: at most 1.3e-3 at the chain's best angle,
2.4e-3 at 4 % off, 1.3e-2 at 30 to 40 % off, where the allowance is 1.6e-2 or more). Every row is placed further than
its allowance from the 99.5 % line, and the table asserts that: a row the chain cannot decide is an error of the table.
"""
import functools
import math

import numpy as np

from scoring import Row

ENV_ID = "Scanner/ErnstFlip-v0"
# L3: a transmitter 28 % too weak (the nominal maximum moves up); L3hi: 28 % too strong (it moves down)
INSTANCES = {"L1": dict(level=1, seed=0), "L2": dict(level=2, seed=0), "L3": dict(level=3, seed=7), "L3hi": dict(level=3, seed=2)}
NEED = 0.995                    # the pass fraction of the task statement
BEST_RTOL = 0.08                # the verifier's best angle against the chain's (the maximum is flat; measured: 6.7 % at most)


def atol(fraction):
    """Allowance for |verifier - chain| at a fraction of the maximum (module docstring)."""
    return 3.5e-3 + 0.5 * (1.0 - fraction)
TBW, SUB_PULSES = 4.0, 128      # the console's default pulse ("normal"), asserted in tests/test_ernst_flip.py
HEIGHTS = np.arange(-140, 141) / 40.0
FLIPS = np.arange(1.0, 90.0001, 0.25)


def _pulse(alpha, zeta):
    """(mx, my, mz) [angle, height] that the windowed-sinc pulse of centre angles ``alpha`` (rad, 1D) makes of Mz = 1 at
    heights ``zeta`` (slice thicknesses), after the rephasing lobe."""
    t = (np.arange(SUB_PULSES) + 0.5) / SUB_PULSES - 0.5
    amp = np.sinc(TBW * t) * 0.5 * (1.0 + np.cos(2 * np.pi * t))
    amp = amp / amp.sum()
    step = (2 * np.pi * TBW * zeta / SUB_PULSES)[None, :]                # precession per sub-pulse at each height
    shape = (len(alpha), len(zeta))
    mx, my, mz = np.zeros(shape), np.zeros(shape), np.ones(shape)

    def about_z(mx, my, phi):
        return mx * np.cos(phi) + my * np.sin(phi), my * np.cos(phi) - mx * np.sin(phi)
    for w in amp:
        mx, my = about_z(mx, my, step / 2)
        c, sn = np.cos(w * alpha)[:, None], np.sin(w * alpha)[:, None]
        my, mz = my * c + mz * sn, mz * c - my * sn                      # the sub-pulse turns the vector about x
        mx, my = about_z(mx, my, step / 2)
    mx, my = about_z(mx, my, -0.5 * step * SUB_PULSES)
    return mx, my, mz


@functools.lru_cache(maxsize=16)
def _curve(b1, tr, t1):
    """Signal of the slice per unit thickness at the nominal flip angles ``FLIPS`` for a transmit factor ``b1``."""
    mx, my, mz = _pulse(np.deg2rad(FLIPS) * b1, HEIGHTS)
    centre = len(HEIGHTS) // 2
    norm = np.hypot(mx[:, centre], my[:, centre])[:, None]
    along = (mx * mx[:, centre:centre + 1] + my * my[:, centre:centre + 1]) / norm
    e1 = math.exp(-tr / t1)
    columns = (1.0 - e1) / (1.0 - e1 * mz) * along
    return (columns[:, :-1] + columns[:, 1:]).sum(1) / 2.0 * (HEIGHTS[1] - HEIGHTS[0])


def curve(case, region, b1=None):
    t = case.truth[region]
    return _curve(round(float(t["b1"] if b1 is None else b1), 9), case.cfg["task"]["target_scan"]["tr_ms"], t["t1_ms"])


def best_deg(case, region, b1=None):
    """The nominal flip angle of the largest signal of the chain (0.25 degree grid)."""
    return float(FLIPS[int(np.argmax(curve(case, region, b1)))])


def fraction(case, region, flip_deg):
    """Signal of ``region`` at ``flip_deg`` over its largest signal, by the chain (hidden truth)."""
    c = curve(case, region)
    return float(np.interp(flip_deg, FLIPS, c) / c.max())


def ids(case):
    return [r["id"] for r in case.cfg["task"]["regions"]]


def at(factor=1.0, **per_region):
    """Deliverable: every region at ``factor`` times its best angle; ``R1=1.3`` overrides the factor of one region."""
    return lambda case: {"flip_deg": {i: round(min(best_deg(case, i) * per_region.get(i, factor), 90.0), 2) for i in ids(case)}}


def textbook(case):
    """arccos(exp(-TR / T1)) of the TRUE T1: the best angle when every spin sees one flip angle."""
    tr = case.cfg["task"]["target_scan"]["tr_ms"]
    return {"flip_deg": {i: round(math.degrees(math.acos(math.exp(-tr / case.truth[i]["t1_ms"]))), 2) for i in ids(case)}}


def calibrated(case):
    """The best 2D angle of the TRUE T1 for a transmitter that is taken to be calibrated (factor 1)."""
    return {"flip_deg": {i: round(best_deg(case, i, b1=1.0), 2) for i in ids(case)}}


def agrees_with_the_chain(case, score):
    """The verifier's fraction of every region is the chain's within ``atol`` and on the same side of the pass
    line; its best angle is the chain's within ``BEST_RTOL``; the largest loss is the reported error; one check per
    region plus the validity of the deliverable; the diagnostic names what each failing region gives."""
    answer = score["answer"]["flip_deg"]
    assert list(score["fraction"]) == list(answer) and score["pass_fraction"] == NEED
    for i in ids(case):
        want = fraction(case, i, answer[i])
        assert abs(score["fraction"][i] - want) <= atol(want), (i, score["fraction"][i], want)
        assert abs(want - NEED) > atol(want), f"row too close to the pass line for {i}: the chain gives {want:.4f}"
        assert (score["fraction"][i] >= NEED) == (want >= NEED)
        assert abs(math.log(score["best_flip_deg"][i] / best_deg(case, i))) <= BEST_RTOL, (i, score["best_flip_deg"][i], best_deg(case, i))
        assert 0.0 < score["fraction"][i] <= 1.0
    low = min(score["fraction"].values())
    assert score["error"] == 1.0 - low
    assert score["checks"] == dict(deliverable_valid=True, **{f"near_maximum_{i}": v >= NEED for i, v in score["fraction"].items()})
    assert score["diagnostic"] == "; ".join(f"{i} gives {100 * v:.1f} % of its maximum" for i, v in score["fraction"].items() if v < NEED)
    n, bad = len(score["fraction"]), sum(v < NEED for v in score["fraction"].values())
    assert score["distance"] == dict(regions=n, near_maximum=n - bad, worst_fraction=low)


def textbook_loses_four_to_eight_percent(case, score):
    """Module docstring of the task: at the Ernst angle of the true T1 a vial gives 92 to 96 % of its maximum (the
    slice profile), with a calibrated transmitter."""
    agrees_with_the_chain(case, score)
    assert all(0.92 <= v <= 0.96 for v in score["fraction"].values()), score["fraction"]


def not_evaluated(case, score):
    """A deliverable the console cannot evaluate: one failed check, no error value, nothing computed."""
    assert score["checks"] == dict(deliverable_valid=False) and score["error"] is None and "fraction" not in score


def with_flip(case, **change):
    a = at()(case)
    a["flip_deg"].update(change)
    return a


ALL = tuple(f"near_maximum_R{k}" for k in (1, 2, 3, 4, 5))
CONSOLE = "the console rejects the target scan with the flip angle of "
SHAPE = 'the deliverable is {"flip_deg": {'
ROWS = [
    # ---- level 1: transmit calibrated
    Row("L1", "every region at its best angle", at(), passed=True, expect=agrees_with_the_chain),
    Row("L1", "every region 4 % high: the maximum is flat", at(1.04), passed=True, expect=agrees_with_the_chain),
    Row("L1", "every region 4 % low", at(0.96), passed=True, expect=agrees_with_the_chain),
    Row("L1", "one region 40 % high", at(R1=1.40), passed=False, failing=("near_maximum_R1",), reason="R1 gives", expect=agrees_with_the_chain),
    Row("L1", "another region 30 % low", at(R2=0.70), passed=False, failing=("near_maximum_R2",), reason="R2 gives", expect=agrees_with_the_chain),
    Row("L1", "two regions off, in both directions", at(R3=1.45, R5=0.7), passed=False, failing=("near_maximum_R3", "near_maximum_R5"),
        reason="; R5 gives", expect=agrees_with_the_chain),
    Row("L1", "the textbook Ernst angle of the true T1: wrong in a 2D scan, for every region", textbook, passed=False, failing=ALL,
        reason="R1 gives 9", expect=textbook_loses_four_to_eight_percent),
    Row("L1", "the best 2D angle computed from the true T1", calibrated, passed=True, expect=agrees_with_the_chain),
    Row("L1", "integers are numbers", lambda c: {"flip_deg": {i: int(round(v)) for i, v in at()(c)["flip_deg"].items()}}, passed=True),
    # ---- level 2: the same world without the stated method
    Row("L2", "every region at its best angle", at(), passed=True, expect=agrees_with_the_chain),
    Row("L2", "the textbook Ernst angle of the true T1", textbook, passed=False, failing=ALL, reason="R2 gives 9",
        expect=textbook_loses_four_to_eight_percent),
    Row("L2", "one region 40 % high", at(R4=1.40), passed=False, failing=("near_maximum_R4",), reason="R4 gives", expect=agrees_with_the_chain),
    # ---- level 3: a hidden transmit scale
    Row("L3", "every region at its best nominal angle", at(), passed=True, expect=agrees_with_the_chain),
    Row("L3", "the best 2D angle computed from the true T1 for a calibrated transmitter: too low for every region", calibrated,
        passed=False, failing=ALL, reason="R1 gives", expect=agrees_with_the_chain),
    Row("L3", "the textbook Ernst angle of the true T1: lower still", textbook, passed=False, failing=ALL, reason="R1 gives",
        expect=agrees_with_the_chain),
    Row("L3hi", "every region at its best nominal angle", at(), passed=True, expect=agrees_with_the_chain),
    Row("L3hi", "the best 2D angle computed from the true T1 for a calibrated transmitter: too high for every region", calibrated,
        passed=False, failing=ALL, reason="R1 gives", expect=agrees_with_the_chain),
    # ---- deliverables the console cannot evaluate: refused, the episode continues
    Row("L1", "the right angles in radians", lambda c: {"flip_deg": {i: round(math.radians(v), 4) for i, v in at()(c)["flip_deg"].items()}},
        refused=True, reason=CONSOLE + "R1: flip_deg=", expect=not_evaluated),
    Row("L1", "an angle above the console's range", lambda c: with_flip(c, R2=120.0), refused=True,
        reason=CONSOLE + "R2: flip_deg=120.0 outside [1.0, 90.0]"),
    Row("L1", "an angle of zero", lambda c: with_flip(c, R1=0.0), refused=True, reason=CONSOLE + "R1: flip_deg=0.0 outside [1.0, 90.0]"),
    Row("L1", "a value that is not a number", lambda c: with_flip(c, R3="large"), refused=True, reason=CONSOLE + "R3: flip_deg must be a finite number"),
    Row("L1", "a region is missing", lambda c: {"flip_deg": {"R1": 20.0}}, refused=True, reason=SHAPE + '"R1": <number>, "R2": <number>'),
    Row("L1", "a region too many", lambda c: with_flip(c, R9=20.0), refused=True, reason=SHAPE),
    Row("L1", "a second field", lambda c: dict(at()(c), units="deg"), refused=True, reason=SHAPE),
    Row("L1", "the numbers without the field", lambda c: at()(c)["flip_deg"], refused=True, reason=SHAPE),
    Row("L1", "a list", [20.0, 30.0], refused=True, reason=SHAPE),
    Row("L1", "nothing", None, refused=True, reason=SHAPE, expect=not_evaluated),
    Row("L1", "the field is not an object", {"flip_deg": 20.0}, refused=True, reason=SHAPE),
]
