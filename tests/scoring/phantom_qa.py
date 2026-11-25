"""Scoring table of ``Scanner/PhantomQA-v0`` (format: ``tests/scoring/__init__.py``).

Pass rule (task statement): the four measured values agree with the scanner's own noise-free measurement on the site's
ACR T1 scan (0.6 mm, 3.0 PIU points, the larger of 0.006 and 15 % of the ghosting ratio); the failed tests are exactly
the tests that fail by the limits of the ACR guidance; the faults are exactly those of the hidden system.

Nothing here comes from the task module or from ``mrgym.qa``. The expected values are derived in this file:

  the four quantities   from the noise-free image of the engine with plain numpy written here: edges by linear
                        interpolation along the pixel row / column through the centre, every 1 cm^2 circle by summing
                        shifted copies of the image, the ghosting ratio from five masks
  the limits            typed from the guidance: 190 +- 3.0 mm, PIU 85 % below 3 T and 80 % at 3 T, ghosting 0.030
  the faults            read from the hidden world: the calibration error of the gradients, the field gradient of the
                        shim, the transmit and the receive fault of the fields, the motion list
"""
import math

import numpy as np

from mrgym.scanner import console
from scoring import Row

ENV_ID = "Scanner/PhantomQA-v0"
# The first instance must differ from the instance of the seed "another" in what the task states (field, phase axis).
INSTANCES = {"L1": dict(level=1, seed=2), "L1G": dict(level=1, seed=0), "L2": dict(level=2, seed=5), "L3": dict(level=3, seed=3)}
DIAMETER_MM, GEOMETRY_MM, PIU_MIN, GHOST_MAX = 190.0, 3.0, {1.5: 85.0, 3.0: 80.0}, 0.030
TOL_MM, TOL_PIU, TOL_GHOST = 0.6, 3.0, (0.006, 0.15)
R_LARGE, R_SMALL = math.sqrt(20000.0 / math.pi), math.sqrt(100.0 / math.pi)        # 200 cm^2 and 1 cm^2 circles, mm
AGREE = dict(diameter_y_mm=0.15, diameter_x_mm=0.15, piu_percent=0.3, ghosting_ratio=0.002)   # this file against the verifier
_CACHE = {}


def _edges(profile, pos, level):
    """Outermost crossings of ``level`` along one line of pixels, by linear interpolation."""
    idx = np.flatnonzero(profile >= level)
    i, j = idx[0], idx[-1]
    lo = pos[i - 1] + (level - profile[i - 1]) / (profile[i] - profile[i - 1]) * (pos[i] - pos[i - 1])
    hi = pos[j] + (profile[j] - level) / (profile[j] - profile[j + 1]) * (pos[j + 1] - pos[j])
    return lo, hi


def truth(case):
    """The four quantities on the noise-free ACR T1 scan of the hidden system, from first principles."""
    if case.seed_hex in _CACHE:
        return _CACHE[case.seed_hex]
    rp = console.plan(dict(case.cfg["task"]["standard_scan"]), case.exam)["realised"]
    img = np.abs(case.engine.acquire(rp, None)[0])
    n = img.shape[0]
    px = 250.0 / n
    pos = (np.arange(n) - n // 2) * px
    yy, xx = np.meshgrid(pos, pos, indexing="ij")
    level = 0.5 * img[np.hypot(yy, xx) <= R_LARGE].mean()
    cy = cx = 0.0
    for _ in range(3):                                                   # centre from the chords, the chords through the centre
        row, col = int(round(cy / px)) + n // 2, int(round(cx / px)) + n // 2
        x0, x1 = _edges(img[row], pos, level)
        y0, y1 = _edges(img[:, col], pos, level)
        cy, cx = 0.5 * (y0 + y1), 0.5 * (x0 + x1)
    # every 1 cm^2 circle centred on a pixel: the sum of the image shifted over the pixels of the circle
    k = int(math.ceil(R_SMALL / px))
    acc, count = np.zeros_like(img), 0
    pad = np.pad(img, k)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            if dy * dy + dx * dx <= (R_SMALL / px) ** 2:
                acc += pad[k + dy:k + dy + n, k + dx:k + dx + n]
                count += 1
    means = acc / count
    r = np.hypot(yy - cy, xx - cx)
    inside = r <= R_LARGE - R_SMALL
    hi, lo = means[inside].max(), means[inside].min()
    large = img[r <= R_LARGE].mean()
    band = lambda a, b: (a >= b - 8.0) & (a <= b + 8.0)                  # noqa: E731  16 mm wide, centred 110 mm out
    along = np.abs(pos) <= 31.5
    top, bottom = img[np.ix_(band(pos, -110.0), along)].mean(), img[np.ix_(band(pos, 110.0), along)].mean()
    left, right = img[np.ix_(along, band(pos, -110.0))].mean(), img[np.ix_(along, band(pos, 110.0))].mean()
    out = dict(diameter_y_mm=float(y1 - y0), diameter_x_mm=float(x1 - x0), piu_percent=float(100.0 * (1.0 - (hi - lo) / (hi + lo))),
               ghosting_ratio=float(abs((top + bottom) - (left + right)) / (2.0 * large)))
    _CACHE[case.seed_hex] = out
    return out


def failed_tests(case, values=None):
    v = values or truth(case)
    out = []
    if max(abs(v["diameter_y_mm"] - DIAMETER_MM), abs(v["diameter_x_mm"] - DIAMETER_MM)) > GEOMETRY_MM:
        out.append("geometric_accuracy")
    if v["piu_percent"] < PIU_MIN[case.cfg["task"]["field_strength_t"]]:
        out.append("intensity_uniformity")
    if v["ghosting_ratio"] > GHOST_MAX:
        out.append("ghosting")
    return out


def faults(case):
    """The faults of the hidden system, read from its state: a size at or beyond the smallest size of a fault."""
    w, spec = case.world, case.world["source"]["spec"]
    ro = "x" if case.cfg["task"]["standard_scan"]["pe_dir"] == "y" else "y"
    out = []
    sy, sx = w["system"]["grad_scale"]
    out += ["gradient_y"] if abs(sy - 1.0) >= 0.02 else []
    out += ["gradient_x"] if abs(sx - 1.0) >= 0.02 else []
    fields = spec["acr_fields"]
    out += ["shim"] if abs(fields["shim_hz_per_mm"][0 if ro == "y" else 1]) >= 2.5 else []
    out += ["b1_transmit"] if abs(fields["b1_fault"] or 0.0) >= 0.3 else []
    out += ["receive_coil"] if (fields["rx_fault"] or 0.0) >= 0.5 else []
    for m in w.get("motion") or []:
        if m["kind"] == "pulse" and m["depth"] >= 0.05:
            out.append("signal_instability")
        if m["kind"] == "shift" and m["amp_mm"] >= 3.0:
            out.append("phantom_motion")
    return sorted(out)


def answer(case, **change):
    """The right deliverable, with ``change`` applied: a number is added to a measured value, a list replaces a list."""
    m = {k: round(v, 4) for k, v in truth(case).items()}
    a = dict(measured=m, failed_tests=failed_tests(case), faults=faults(case))
    for k, v in change.items():
        if k in m:
            m[k] = round(m[k] + v, 4)
        else:
            a[k] = v
    return a


def right(**change):
    return lambda case: answer(case, **change)


LOOKALIKE = dict(gradient_x="shim", gradient_y="shim", shim="gradient_x", b1_transmit="receive_coil", receive_coil="b1_transmit",
                 signal_instability="phantom_motion", phantom_motion="signal_instability")
GROUP = dict(geometric_accuracy=["gradient_x", "gradient_y", "shim"], intensity_uniformity=["b1_transmit", "receive_coil"],
             ghosting=["phantom_motion", "signal_instability"])


def lookalike(case):
    return answer(case, faults=sorted(LOOKALIKE[f] for f in faults(case)))


def every_candidate(case):
    return answer(case, faults=sorted(f for t in failed_tests(case) for f in GROUP[t]))


def wrong_units(case):
    a = answer(case)
    m = a["measured"]
    a["measured"] = dict(diameter_y_mm=m["diameter_y_mm"] / 10.0, diameter_x_mm=m["diameter_x_mm"] / 10.0, piu_percent=m["piu_percent"] / 100.0,
                         ghosting_ratio=m["ghosting_ratio"] * 100.0)
    return a


def first_fault_only(case):
    return answer(case, faults=faults(case)[:1])


def agrees(case, score):
    """The verifier's own measurement is the one derived here; the verdicts and the error follow from it."""
    want = truth(case)
    for q, tol in AGREE.items():
        assert abs(score["measured"][q] - want[q]) <= tol, (q, score["measured"][q], want[q])
    assert score["failed_tests"] == sorted(failed_tests(case)) == sorted(failed_tests(case, score["measured"]))
    got = score["answer"]["measured"]
    tol = dict(diameter_y_mm=TOL_MM, diameter_x_mm=TOL_MM, piu_percent=TOL_PIU,
               ghosting_ratio=max(TOL_GHOST[0], TOL_GHOST[1] * score["measured"]["ghosting_ratio"]))
    error = max(abs(got[q] - score["measured"][q]) / tol[q] for q in tol)
    assert abs(score["error"] - error) < 0.02 * max(1.0, error) and (error <= 1.0) == all(score["checks"][f"measured_{q}"] for q in tol)
    assert list(score["checks"]) == ["deliverable_valid"] + [f"measured_{q}" for q in tol] + ["failed_tests", "faults"]


def the_instance_has_one_answer(case, score):
    """What the instance generator promises, checked from the hidden world: each fault fails its own test, the other
    tests pass, and every quantity keeps clear of its limit by more than its tolerance."""
    v, field = truth(case), case.cfg["task"]["field_strength_t"]
    test_of = {f: t for t, fs in GROUP.items() for f in fs}
    assert sorted({test_of[f] for f in faults(case)}) == sorted(failed_tests(case))
    assert sorted(faults(case)) == sorted(case.cfg["hidden"]["faults"])
    for q in ("diameter_y_mm", "diameter_x_mm"):
        assert abs(abs(v[q] - DIAMETER_MM) - GEOMETRY_MM) > TOL_MM + 0.2
    assert abs(v["piu_percent"] - PIU_MIN[field]) > TOL_PIU + 0.3
    assert not GHOST_MAX - 0.008 < v["ghosting_ratio"] < GHOST_MAX * 1.15 + 0.004


def not_evaluated(case, score):
    assert score["checks"] == dict(deliverable_valid=False) and score["error"] is None and "measured" not in score


SHAPE = 'the deliverable is {"measured": {"diameter_y_mm": <number>'
ROWS = [
    # ---- level 1: one fault (transmit field), 1.5 T
    Row("L1", "the right answer", right(), passed=True, expect=agrees),
    Row("L1", "the instance has one answer", right(), passed=True, expect=the_instance_has_one_answer),
    Row("L1", "every value inside its tolerance", right(diameter_y_mm=0.4, diameter_x_mm=-0.4, piu_percent=2.5, ghosting_ratio=0.004),
        passed=True, expect=agrees),
    Row("L1", "a diameter 0.9 mm off", right(diameter_x_mm=0.9), passed=False, failing=("measured_diameter_x_mm",),
        reason="diameter_x_mm: delivered", expect=agrees),
    Row("L1", "the uniformity 4 points off", right(piu_percent=-4.0), passed=False, failing=("measured_piu_percent",),
        reason="piu_percent: delivered", expect=agrees),
    Row("L1", "a ghosting ratio of one percent where there is none", right(ghosting_ratio=0.01), passed=False,
        failing=("measured_ghosting_ratio",), reason="ghosting_ratio: delivered", expect=agrees),
    Row("L1", "right values, every test called a pass", right(failed_tests=[]), passed=False, failing=("failed_tests",),
        reason="the failed tests are intensity_uniformity"),
    Row("L1", "right values, a test too many", right(failed_tests=["ghosting", "intensity_uniformity"]), passed=False, failing=("failed_tests",),
        reason="the failed tests are intensity_uniformity"),
    Row("L1", "the look-alike fault", lookalike, passed=False, failing=("faults",), reason="the fault list is not the system's"),
    Row("L1", "every fault that moves the failed test", every_candidate, passed=False, failing=("faults",), reason="the fault list"),
    Row("L1", "no fault named", right(faults=[]), passed=False, failing=("faults",), reason="the fault list"),
    Row("L1", "integers are numbers", lambda c: dict(answer(c), measured={k: (int(round(v)) if k.startswith("diameter") else v)
                                                                             for k, v in answer(c)["measured"].items()}), passed=True),
    # ---- level 1: a gradient fault along the phase-encode axis, 3 T
    Row("L1G", "the right answer", right(), passed=True, expect=agrees),
    Row("L1G", "the instance has one answer", right(), passed=True, expect=the_instance_has_one_answer),
    Row("L1G", "the look-alike fault", lookalike, passed=False, failing=("faults",), reason="the fault list is not the system's"),
    # ---- level 2: a dead coil element beside sub-threshold imperfections, 3 T
    Row("L2", "the right answer", right(), passed=True, expect=agrees),
    Row("L2", "the instance has one answer", right(), passed=True, expect=the_instance_has_one_answer),
    Row("L2", "the look-alike fault", lookalike, passed=False, failing=("faults",), reason="the fault list is not the system's"),
    # ---- level 3: two faults that move different tests
    Row("L3", "the right answer", right(), passed=True, expect=agrees),
    Row("L3", "the instance has one answer", right(), passed=True, expect=the_instance_has_one_answer),
    Row("L3", "one of the two faults", first_fault_only, passed=False, failing=("faults",), reason="the fault list is not the system's"),
    Row("L3", "one of the two failed tests", lambda c: answer(c, failed_tests=failed_tests(c)[:1]), passed=False, failing=("failed_tests",),
        reason="the failed tests are geometric_accuracy, ghosting"),
    Row("L3", "the right answer in cm, as a fraction and in percent", wrong_units, passed=False,
        failing=("measured_diameter_y_mm", "measured_diameter_x_mm", "measured_piu_percent", "measured_ghosting_ratio"),
        reason="diameter_y_mm: delivered", expect=agrees),
    # ---- deliverables the console cannot evaluate: refused, the episode continues
    Row("L1", "a quantity is missing", lambda c: dict(answer(c), measured={k: v for k, v in answer(c)["measured"].items() if k != "piu_percent"}),
        refused=True, reason=SHAPE, expect=not_evaluated),
    Row("L1", "a quantity too many", lambda c: dict(answer(c), measured=dict(answer(c)["measured"], snr=300.0)), refused=True, reason=SHAPE),
    Row("L1", "the fault list is missing", lambda c: {k: v for k, v in answer(c).items() if k != "faults"}, refused=True, reason=SHAPE),
    Row("L1", "a fourth field", lambda c: dict(answer(c), units="mm"), refused=True, reason=SHAPE),
    Row("L1", "a value that is not a number", lambda c: dict(answer(c), measured=dict(answer(c)["measured"], piu_percent="high")),
        refused=True, reason="measured.piu_percent must be a number >= 0"),
    Row("L1", "a negative value", right(ghosting_ratio=-1.0), refused=True, reason="measured.ghosting_ratio must be a number >= 0"),
    Row("L1", "a fault that is not in the list", right(faults=["eddy_currents"]), refused=True, reason="faults: unknown name 'eddy_currents'"),
    Row("L1", "a test that is not in the list", right(failed_tests=["slice_thickness"]), refused=True, reason="failed_tests: unknown name"),
    Row("L1", "one fault as a word, not a list", right(faults="b1_transmit"), refused=True, reason="faults must be a list of names"),
    Row("L1", "a fault named twice", right(faults=["b1_transmit", "b1_transmit"]), refused=True, reason="faults: a name is listed twice"),
    Row("L1", "nothing", None, refused=True, reason=SHAPE, expect=not_evaluated),
    Row("L1", "a list", [190.0, 190.0, 95.0, 0.0], refused=True, reason=SHAPE),
]
