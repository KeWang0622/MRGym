"""``Scanner/PhantomQA``: instances and the verifier.

The job is the physicist's part of phantom QA. The uniform section (slice 7) of the ACR large phantom lies in a
scanner that has one hidden fault, two, or none. The agent runs the site's ACR T1 scan, measures the test quantities
the guidance defines on that section, says which tests fail, and names what is wrong with the system from a closed
list. Several faults move the same test, so a failed test is not a diagnosis: a second, cheap scan chosen for the
test that failed tells them apart.

Deliverable    {"measured": {"diameter_y_mm": 190.2, "diameter_x_mm": 194.6, "piu_percent": 96.1, "ghosting_ratio": 0.002},
                "failed_tests": ["geometric_accuracy"], "faults": ["shim"]}
Pass rule      every measured value agrees with the verifier's own measurement on a noise-free scan of the hidden
               system (tolerances ``TOLERANCE``); the failed tests are exactly the tests that fail there; the faults
               are exactly the hidden ones.

Tests on the uniform section and the limits of the guidance (``mrgym.qa.LIMITS``):
    geometric_accuracy     both diameters within 190 +- 3.0 mm           (the guidance reads them on slices 1 and 5; the
                                                                          inner diameter is the same on every section)
    intensity_uniformity   PIU >= 85 % (1.5 T), >= 80 % (3 T)
    ghosting               ratio <= 0.030
Not in this family: resolution and low contrast (other sections; a session images one section), slice thickness and
slice position (not simulated, ``mrgym.qa.NOT_SIMULATED``).

Faults (``FAULTS``), by the test they move, with what tells them apart:
    geometric_accuracy     gradient_x, gradient_y   a gradient a few percent off: the length along its own axis, whatever the protocol
                           shim                     a linear field gradient: the length along the readout axis, in
                                                    proportion to 1 / bandwidth; gone when the readout runs along the other axis
    intensity_uniformity   b1_transmit              the flip angle near the centre is off: a flip-angle map is not flat
                           receive_coil             the receive sensitivity falls across the phantom: a flip-angle map is flat
    ghosting               signal_instability       the amplitude varies from pulse to pulse: a copy of the phantom, flat
                           phantom_motion           the phantom rocks along the phase-encode axis: the ghost of its edges
Every instance is built so that the question has one answer: each hidden fault fails its test with a margin, every
other test passes with a margin, and the test passes again on the twin system with that fault switched off
(``make_instance`` draws the sizes again otherwise; the faults and the world seed stay). The margins are wider than
the tolerances, so a value inside its tolerance gives the right verdict.

What the family does not make hard, stated plainly: the two discriminating scans are short (under 10 s after a
131.5 s QA scan), so a fixed list that always runs both solves levels 1 and 2 as well as a policy that chooses
(``controls.open_loop``); by the open-loop rule those levels are measurement tasks. Level 3 leaves time for one.

Levels
    1   one fault beyond the limit; everything else at its healthy size; time for both discriminating scans
    2   one fault next to a sub-threshold amount of its look-alike (a small gradient error beside a shim fault, a
        little transmit non-uniformity beside a dead coil element, ...); less time
    3   two faults that move different tests, and scan time for one discriminating scan after the QA scan
About one instance in seventy has no fault at all (``P_NONE``).
"""
from __future__ import annotations

import functools
import os

import numpy as np

from ... import qa
from ...scanner import console
from ...scanner import world as W
from .. import family as F

FAMILY = "phantom_qa"
ENV_ID = "Scanner/PhantomQA-v0"
SUBMIT_DOC = ("Deliver the four measured test quantities of the ACR T1 scan, the list of tests that fail, and the list of system "
              "faults (empty lists when nothing fails and nothing is wrong).")
# Examined competencies this family exercises (ids of ``mrgym.coverage.COMPETENCIES``).
COMPETENCIES = ("qc.geometric_accuracy", "qc.signal_uniformity", "qc.ghosting", "qc.fault_diagnosis",
                "artefact.motion_ghost", "artefact.b0_distortion", "artefact.b1_shading",
                "parameter.receive_bandwidth", "parameter.phase_frequency_direction", "parameter.matrix")
TESTS = ("geometric_accuracy", "intensity_uniformity", "ghosting")
QUANTITIES = ("diameter_y_mm", "diameter_x_mm", "piu_percent", "ghosting_ratio")
FAULTS = dict(
    gradient_x=dict(test="geometric_accuracy", what="the X gradient (left-right, image columns) is miscalibrated: lengths along x are wrong by the same percentage in every protocol"),
    gradient_y=dict(test="geometric_accuracy", what="the Y gradient (anterior-posterior, image rows) is miscalibrated: lengths along y are wrong by the same percentage in every protocol"),
    shim=dict(test="geometric_accuracy", what="the shim left a linear B0 gradient across the phantom: the image is stretched or compressed along the readout axis, in proportion to 1 / bandwidth"),
    b1_transmit=dict(test="intensity_uniformity", what="the transmit field is too high or too low near the centre of the phantom (the flip angle is off there)"),
    receive_coil=dict(test="intensity_uniformity", what="a receive coil element is down: the receive sensitivity falls off across the phantom"),
    signal_instability=dict(test="ghosting", what="the signal amplitude varies from pulse to pulse (transmitter, receiver or gradient amplifier)"),
    phantom_motion=dict(test="ghosting", what="the phantom is not fixed: it rocks along the phase-encode axis during the scan"),
)
GROUPS = dict(geometric_accuracy=("gradient_x", "gradient_y", "shim"), intensity_uniformity=("b1_transmit", "receive_coil"),
              ghosting=("signal_instability", "phantom_motion"))
TOLERANCE = dict(diameter_mm=0.6, piu_percent=3.0, ghosting_abs=0.006, ghosting_rel=0.15)
# How far a quantity stays from its limit on every instance (wider than the tolerance on either side).
MARGIN = dict(diameter_mm=1.3, piu_percent=3.6, ghosting=(0.020, 0.042))
P_NONE = 0.015
LARGE_RADIUS_MM = float(np.sqrt(qa.LARGE_ROI_CM2 * 100 / np.pi))        # the 200 cm^2 circle: 79.8 mm
STANDARD_S = 131.5                       # console time of the ACR T1 scan (asserted in the tests)
# Sizes of a fault (drawn uniformly, sign drawn where it has one) and of what a healthy system carries.
SIZE = dict(gradient=(0.024, 0.040), shim_hz_per_mm=(3.2, 5.2), shim_other_hz_per_mm=2.0, b1=(0.33, 0.50), rx=(0.60, 0.95),
            instability_depth=(0.09, 0.20), motion_mm=(3.4, 5.0), hz=(0.20, 0.36), motion_hz=(0.31, 0.36))
# A healthy gradient is calibrated to within 0.8 % (1.5 mm on the phantom: inside the limit, two and a half times the
# tolerance of a measured diameter, so that the nominal 190 mm is not a measurement).
HEALTHY = dict(gradient=0.008, shim_hz_per_mm=0.1, b1=0.0, rx=0.0, instability_depth=0.0)
NUISANCE = dict(gradient=0.006, shim_hz_per_mm=0.5, b1=0.08, rx=0.12, instability_depth=0.012)   # sub-threshold look-alikes (level 2, 3)
LEVELS = {
    1: dict(n_faults=1, nuisance=False, budget_s=STANDARD_S + 60.0,
            summary="one fault beyond its limit on an otherwise healthy system; scan time for both discriminating scans"),
    2: dict(n_faults=1, nuisance=True, budget_s=STANDARD_S + 25.0,
            summary="one fault beside a sub-threshold amount of its look-alike; less scan time"),
    3: dict(n_faults=2, nuisance=True, budget_s=STANDARD_S + 10.0,
            summary="two faults that move different tests; scan time for one discriminating scan after the QA scan"),
}
# Level 3: the two faults are drawn so that at most one of them needs a scan to be told from its look-alike.
PAIRS = ((0.5, ("geometry_readout", "ghosting")), (0.25, ("intensity_uniformity", "ghosting")), (0.25, ("geometry_phase", "intensity_uniformity")))
MAX_ATTEMPTS, SAME_QUESTION = 120, 20
GRID_N = 640                             # master grid of the uniform section: 0.4 mm (the 0.25 mm grid of the acr kind is for the resolution insert)


def standard_protocol(task: dict) -> dict:
    """The site's ACR T1 scan: the series of the guidance with the site's phase-encode axis."""
    return dict(qa.SERIES["t1"], pe_dir=task["standard_scan"]["pe_dir"])


def limits(field_t: float) -> dict:
    return dict(diameter_mm=(qa.A["diameter_mm"], qa.LIMITS["geometry_mm"]), piu_percent=qa.LIMITS["piu_percent"][float(field_t)],
                ghosting_ratio=qa.LIMITS["ghosting_ratio"])


# ------------------------------------------------------------------------------------------------ measurement
def measure(engine, task: dict) -> dict:
    """Verifier side: the four test quantities on a noise-free scan of the standard protocol, as ``mrgym.qa`` defines
    them (every position of the 1 cm^2 region; the four background rectangles), with the edge level the statement
    gives: half the mean signal of the 200 cm^2 circle at the isocentre, a number the tools return."""
    rp = console.plan(standard_protocol(task), engine.w["console"])["realised"]
    img = np.abs(engine.acquire(rp, None)[0])
    fov = (rp["fov_pe_mm"], rp["fov_mm"]) if rp["pe_dir"] == "y" else (rp["fov_mm"], rp["fov_pe_mm"])
    y, x = qa.positions(img.shape, fov)
    level = 0.5 * float(img[np.hypot(y[:, None], x[None, :]) <= LARGE_RADIUS_MM].mean())      # half the mean of the 200 cm^2 circle at the isocentre
    c = qa.find_centre(img, fov, level)
    u, g = qa.uniformity(img, fov, c), qa.ghosting(img, fov, c)
    return dict(diameter_y_mm=float(qa.chord(img, fov, c, 90.0, level)[0]), diameter_x_mm=float(qa.chord(img, fov, c, 0.0, level)[0]),
                piu_percent=float(u["piu_percent"]), ghosting_ratio=float(g["ratio"]))


def verdicts(values: dict, field_t: float) -> dict:
    """{test: True when it FAILS} from measured values and the limits of the guidance."""
    lim = limits(field_t)
    return dict(geometric_accuracy=bool(max(abs(values["diameter_y_mm"] - lim["diameter_mm"][0]), abs(values["diameter_x_mm"] - lim["diameter_mm"][0])) > lim["diameter_mm"][1]),
                intensity_uniformity=bool(values["piu_percent"] < lim["piu_percent"]),
                ghosting=bool(values["ghosting_ratio"] > lim["ghosting_ratio"]))


def clear_of_limits(values: dict, field_t: float) -> bool:
    """Whether every quantity keeps ``MARGIN`` from its limit (so that a value inside its tolerance gives the verdict)."""
    lim = limits(field_t)
    for k in ("diameter_y_mm", "diameter_x_mm"):
        if abs(abs(values[k] - lim["diameter_mm"][0]) - lim["diameter_mm"][1]) < MARGIN["diameter_mm"]:
            return False
    if abs(values["piu_percent"] - lim["piu_percent"]) < MARGIN["piu_percent"]:
        return False
    return not MARGIN["ghosting"][0] < values["ghosting_ratio"] < MARGIN["ghosting"][1]


def within(name: str, got: float, want: float) -> bool:
    """Whether a delivered value agrees with the verifier's (``TOLERANCE``)."""
    if name.startswith("diameter"):
        return abs(got - want) <= TOLERANCE["diameter_mm"]
    if name == "piu_percent":
        return abs(got - want) <= TOLERANCE["piu_percent"]
    return abs(got - want) <= max(TOLERANCE["ghosting_abs"], TOLERANCE["ghosting_rel"] * want)


# ------------------------------------------------------------------------------------------------ instances
def _signed(rng, lo, hi):
    return float(rng.uniform(lo, hi) * (1 if rng.uniform() < 0.5 else -1))


def _draw(seed: str, level: int, attempt: int = 0) -> dict:
    """One draw of an instance: field, phase-encode axis and the faults from the seed; the sizes of the faults and of
    the small imperfections from a sub-stream per ``attempt``. A draw that has to be repeated keeps its question and
    its world seed, so the mix of faults of a level is the one written here and not the mix that is easiest to accept."""
    lv = LEVELS[level]
    rng = W.rng_for(seed, "task")
    field = (1.5, 3.0)[int(rng.integers(2))]
    pe = ("y", "x")[int(rng.integers(2))]
    ro = "x" if pe == "y" else "y"
    amb, clear = (f"gradient_{ro}", "shim"), (f"gradient_{pe}",)        # geometry faults along the readout axis look alike
    pick = lambda names: names[int(rng.integers(len(names)))]            # noqa: E731
    by_group = dict(geometry_readout=amb, geometry_phase=clear, intensity_uniformity=GROUPS["intensity_uniformity"], ghosting=GROUPS["ghosting"])
    if rng.uniform() < P_NONE:
        faults = []
    elif lv["n_faults"] == 1:
        g = ("geometry_readout", "geometry_readout", "geometry_phase", "intensity_uniformity", "intensity_uniformity", "intensity_uniformity",
             "ghosting", "ghosting", "ghosting")[int(rng.integers(9))]
        faults = [pick(by_group[g])]
    else:
        u, acc, pair = rng.uniform(), 0.0, PAIRS[-1][1]
        for p, candidate in PAIRS:
            acc += p
            if u < acc:
                pair = candidate
                break
        faults = [pick(by_group[g]) for g in pair]
    rng = W.rng_for(seed, "task", 1 + attempt)
    small = NUISANCE if lv["nuisance"] else HEALTHY
    k = dict(grad_y=_signed(rng, 0.0, small["gradient"]), grad_x=_signed(rng, 0.0, small["gradient"]),
             shim_y=_signed(rng, 0.0, small["shim_hz_per_mm"]), shim_x=_signed(rng, 0.0, small["shim_hz_per_mm"]),
             b1=_signed(rng, 0.0, small["b1"]), rx=float(rng.uniform(0.0, small["rx"])),
             depth=float(rng.uniform(0.0, small["instability_depth"])), hz=float(rng.uniform(*SIZE["hz"])), motion_mm=0.0,
             motion_hz=float(rng.uniform(*SIZE["motion_hz"])))
    size = {f: _signed(rng, *SIZE["gradient"]) for f in ("gradient_x", "gradient_y")}
    size.update(shim=_signed(rng, *SIZE["shim_hz_per_mm"]), shim_other=_signed(rng, 0.0, SIZE["shim_other_hz_per_mm"]),
                b1_transmit=_signed(rng, *SIZE["b1"]), receive_coil=float(rng.uniform(*SIZE["rx"])),
                signal_instability=float(rng.uniform(*SIZE["instability_depth"])), phantom_motion=float(rng.uniform(*SIZE["motion_mm"])))
    return dict(field_t=field, pe_dir=pe, faults=faults, knobs=k, size=size)


def _spec(draw: dict, without: str | None = None) -> dict:
    """World spec of a draw; ``without``: the twin with that fault switched off (its knob back at the small size)."""
    k, size = dict(draw["knobs"]), draw["size"]
    ro = "x" if draw["pe_dir"] == "y" else "y"
    for f in draw["faults"]:
        if f == without:
            continue
        if f in ("gradient_x", "gradient_y"):
            k["grad_" + f[-1]] = size[f]
        elif f == "shim":
            k["shim_" + ro], k["shim_" + draw["pe_dir"]] = size["shim"], size["shim_other"]
        elif f == "b1_transmit":
            k["b1"] = size["b1_transmit"]
        elif f == "receive_coil":
            k["rx"] = size["receive_coil"]
        elif f == "signal_instability":
            k["depth"] = size["signal_instability"]
        elif f == "phantom_motion":
            k["motion_mm"] = size["phantom_motion"]
    motion = []
    if k["depth"] > 0:
        motion.append(dict(cls="*", kind="pulse", f_hz=k["hz"], depth=k["depth"]))
    if k["motion_mm"] > 0:
        motion.append(dict(cls="*", kind="shift", f_hz=k["motion_hz"], amp_mm=k["motion_mm"], dir=draw["pe_dir"]))
    return dict(qa.HEALTHY, kind="acr", n=GRID_N, b0_t=draw["field_t"], acr=dict(slice=7), grad_scale=dict(y=k["grad_y"], x=k["grad_x"]),
                acr_fields=dict(shim_hz_per_mm=(k["shim_y"], k["shim_x"]), b1_fault=k["b1"] or None, rx_fault=k["rx"] or None),
                motion=motion or None)


@functools.lru_cache(maxsize=256)
def _instance(seed: str, level: int) -> tuple:
    """(attempt, draw, measured values, values on the twin without each fault): the first attempt whose question
    has one answer (see the module docstring). Cached: every policy of a gate run asks for the same instance."""
    from ...scanner import kspace
    for attempt in range(MAX_ATTEMPTS):
        # the sizes are drawn again for the same question SAME_QUESTION times; a question that never gets one answer on
        # this world (the reason is the world, e.g. a fault whose test another imperfection already fails) is then drawn again
        draw = _draw(F.reseed(seed, attempt // SAME_QUESTION), level, attempt)
        task = dict(standard_scan=dict(pe_dir=draw["pe_dir"]))
        values = measure(kspace.Engine(W.make_world(seed, _spec(draw))), task)
        fails = verdicts(values, draw["field_t"])
        if not clear_of_limits(values, draw["field_t"]) or {t for t, bad in fails.items() if bad} != {FAULTS[f]["test"] for f in draw["faults"]}:
            continue
        twins, ok = {}, True
        for f in draw["faults"]:
            twins[f] = measure(kspace.Engine(W.make_world(seed, _spec(draw, without=f))), task)
            ok = ok and not verdicts(twins[f], draw["field_t"])[FAULTS[f]["test"]] and clear_of_limits(twins[f], draw["field_t"])
        if ok:
            return attempt, draw, values, twins
    raise RuntimeError(f"no instance with one answer in {MAX_ATTEMPTS} draws (seed {seed[:8]}..., level {level})")


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    attempt, draw, values, twins = _instance(seed, level)
    lv = LEVELS[level]
    lim = limits(draw["field_t"])
    task = dict(family=FAMILY, phantom="ACR large phantom, uniform section (slice 7)", field_strength_t=draw["field_t"],
                standard_scan=dict(qa.SERIES["t1"], pe_dir=draw["pe_dir"]),
                quantities=dict(diameter_y_mm="edge to edge through the centre of the phantom along y, edges at half the mean signal of the 200 cm^2 circle "
                                              "(radius 79.8 mm) centred on the isocentre",
                                diameter_x_mm="the same along x",
                                piu_percent="100 (1 - (high - low) / (high + low)): highest and lowest mean of a 1 cm^2 circle (radius 5.64 mm) "
                                            "lying entirely inside the 200 cm^2 circle (radius 79.8 mm) centred on the phantom",
                                ghosting_ratio="|((top + bottom) - (left + right)) / (2 large)|: large = mean of the 200 cm^2 circle; the others = mean of "
                                               "16 x 63 mm rectangles in the background, centred 110 mm from the isocentre, the 63 mm side tangential "
                                               "(top, bottom: along x; left, right: along y)"),
                limits=dict(diameter_mm=f"{lim['diameter_mm'][0]:g} +- {lim['diameter_mm'][1]:g}", piu_percent_min=lim["piu_percent"],
                            ghosting_ratio_max=lim["ghosting_ratio"]),
                tolerances=dict(TOLERANCE), tests=list(TESTS), faults={k: v["what"] for k, v in FAULTS.items()},
                deliverable=dict(measured={q: "<number>" for q in QUANTITIES}, failed_tests="<list of tests>", faults="<list of faults>"))
    return dict(task_id=task_id or "phantomqa-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed, world_spec=_spec(draw),
                budget_s=float(lv["budget_s"]), max_requests=64, task=task,
                hidden=dict(faults=sorted(draw["faults"]), attempt=attempt, measured={k: round(v, 5) for k, v in values.items()},
                            twins={f: {k: round(v, 5) for k, v in t.items()} for f, t in twins.items()}))


def statement(cfg: dict) -> str:
    t = cfg["task"]
    s, lim, tol = t["standard_scan"], t["limits"], t["tolerances"]
    ro = "x" if s["pe_dir"] == "y" else "y"
    return f"""# Phantom QA: measure, judge and diagnose
The scanner is a {t['field_strength_t']:g} T system. The uniform section (slice 7) of the ACR large
phantom lies at the isocentre: a disc of solution, {qa.A['diameter_mm']:g} mm inner diameter. The scanner may have a fault, two, or none.

1. Run the site's ACR T1 scan and measure on its image:
    sequence {s['sequence']}, tr_ms {s['tr_ms']:g}, te_ms {s['te_ms']:g}, fov_mm {s['fov_mm']:g}, matrix {s['matrix']}, slice_mm {s['slice_mm']:g}, averages {s['averages']},
    bandwidth_hz_per_px {s['bandwidth_hz_per_px']:g}, pe_dir {s['pe_dir']} (phase encoding along {s['pe_dir']}, readout along {ro}).
  diameter_y_mm, diameter_x_mm   {t['quantities']['diameter_y_mm']}; {t['quantities']['diameter_x_mm']}
  piu_percent                    {t['quantities']['piu_percent']}
  ghosting_ratio                 {t['quantities']['ghosting_ratio']}
  (profile gives the numbers along a line, and with samples = 1 the mean of a rectangle; read_regions the mean of circles.)
2. Judge the tests of the ACR guidance:
  geometric_accuracy     fails when a diameter is outside {lim['diameter_mm']} mm
  intensity_uniformity   fails when piu_percent is below {lim['piu_percent_min']:g}
  ghosting               fails when ghosting_ratio is above {lim['ghosting_ratio_max']:g}
3. Name every fault of the system. The closed list (a fault moves the test named in brackets):
{chr(10).join(f"  {k:<19}[{v['test']}] {v['what']}" for k, v in FAULTS.items())}
  A failed test is not a diagnosis: more than one fault moves each test. A failed test has exactly one fault. The
  scan-time budget leaves room after the QA scan for a short scan of your own design that tells them apart. A fault
  in the list is either present (its test fails) or absent; small imperfections that fail no test are not faults.

Deliver with `submit`: the four measured values, the failed tests, the faults. Passing: each value within
{tol['diameter_mm']:g} mm, {tol['piu_percent']:g} PIU points, and {tol['ghosting_abs']:g} or {100 * tol['ghosting_rel']:g} % of the ratio (whichever is larger) of the scanner's own
noise-free measurement on the scan above; the failed tests exactly those that fail; the faults exactly those present.
The scanner's log must show that you measured. An answer in the wrong unit is wrong."""


def submit_schema(cfg: dict) -> dict:
    number = lambda what, hi: {"type": "number", "minimum": 0, "maximum": hi, "description": what}      # noqa: E731
    return {"type": "object", "additionalProperties": False, "required": ["measured", "failed_tests", "faults"],
            "properties": {
                "measured": {"type": "object", "additionalProperties": False, "required": list(QUANTITIES),
                             "description": "the test quantities measured on the ACR T1 scan",
                             "properties": {"diameter_y_mm": number("diameter along y, mm", 1000), "diameter_x_mm": number("diameter along x, mm", 1000),
                                            "piu_percent": number("percent integral uniformity, percent", 100),
                                            "ghosting_ratio": number("ghosting ratio, a fraction (0.03 = 3 %)", 10)}},
                "failed_tests": {"type": "array", "maxItems": len(TESTS), "items": {"type": "string", "enum": list(TESTS)},
                                 "description": "the tests that fail; [] when all pass"},
                "faults": {"type": "array", "maxItems": len(FAULTS), "items": {"type": "string", "enum": list(FAULTS)},
                           "description": "the faults of the system; [] when there is none"}}}


SHAPE = ('the deliverable is {"measured": {' + ", ".join(f'"{q}": <number>' for q in QUANTITIES)
         + '}, "failed_tests": [<test>, ...], "faults": [<fault>, ...]}')


def check(cfg: dict, answer, exam: dict):
    if not isinstance(answer, dict) or set(answer) != {"measured", "failed_tests", "faults"}:
        return SHAPE
    m = answer["measured"]
    if not isinstance(m, dict) or set(m) != set(QUANTITIES):
        return SHAPE
    for q in QUANTITIES:
        v = m[q]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
            return f"measured.{q} must be a number >= 0"
    for key, allowed in (("failed_tests", TESTS), ("faults", tuple(FAULTS))):
        v = answer[key]
        if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
            return f"{key} must be a list of names ([] for none)"
        unknown = [x for x in v if x not in allowed]
        if unknown:
            return f"{key}: unknown name {unknown[0]!r:.40}; allowed: {', '.join(allowed)}"
        if len(set(v)) != len(v):
            return f"{key}: a name is listed twice"
    return None


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    from ...scanner import kspace
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    engine = engine or kspace.Engine(world)
    problem = check(cfg, answer, world["console"])
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    t = cfg["task"]
    truth = measure(engine, t)                                           # re-measured on the hidden system, noise-free
    failed = sorted(k for k, bad in verdicts(truth, t["field_strength_t"]).items() if bad)
    got = answer["measured"]
    ok = {q: within(q, float(got[q]), truth[q]) for q in QUANTITIES}
    checks = dict(deliverable_valid=True, **{f"measured_{q}": ok[q] for q in QUANTITIES},
                  failed_tests=sorted(answer["failed_tests"]) == failed, faults=sorted(answer["faults"]) == sorted(cfg["hidden"]["faults"]))
    notes = [f"{q}: delivered {float(got[q]):.4g}, measured {truth[q]:.4g}" for q in QUANTITIES if not ok[q]]
    if not checks["failed_tests"]:
        notes.append("the failed tests are " + (", ".join(failed) or "none"))
    if not checks["faults"]:
        notes.append("the fault list is not the system's")
    tol = dict(diameter_y_mm=TOLERANCE["diameter_mm"], diameter_x_mm=TOLERANCE["diameter_mm"], piu_percent=TOLERANCE["piu_percent"],
               ghosting_ratio=max(TOLERANCE["ghosting_abs"], TOLERANCE["ghosting_rel"] * truth["ghosting_ratio"]))
    error = max(abs(float(got[q]) - truth[q]) / tol[q] for q in QUANTITIES)
    return F.result(checks, "; ".join(notes), error=float(error), measured={q: round(truth[q], 5) for q in QUANTITIES}, failed_tests=failed,
                    distance=dict(values_off=sum(not v for v in ok.values()), worst_value_error_in_tolerances=round(float(error), 3)))


OUTCOMES = ("measurement_off", "verdict_wrong", "diagnosis_wrong")       # of a delivered, measured answer that fails (worst first)


def outcome(score: dict) -> str:
    c = score["checks"]
    if not all(v for k, v in c.items() if k.startswith("measured_")):
        return "measurement_off"
    return "verdict_wrong" if not c.get("failed_tests", True) else "diagnosis_wrong"


def card() -> dict:
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    return dict(
        id=ENV_ID, title="Phantom QA: measure, judge and diagnose", kind="candidate",
        given="The uniform section of the ACR large phantom in a scanner with zero, one or two hidden faults; the site's ACR T1 protocol; "
              "the limits of the ACR guidance; a closed list of seven faults; a scan-time budget",
        deliver="The four test quantities measured on the ACR T1 scan, the tests that fail, and the faults of the system",
        scored_by="A noise-free scan of the same protocol on the hidden system: the values must agree with the verifier's measurement, "
                  "the verdicts with the guidance's limits, the faults with the hidden state",
        summary="The physicist's part of phantom QA. Each test of the guidance can be failed by more than one fault (a gradient "
                "miscalibration or a shim error; the transmit field or a receive element; signal instability or a loose phantom), "
                "so the QA scan says which test fails and a second scan, chosen for that test, says why.",
        world=dict(phantom="ACR large phantom, slice 7 (world kind acr), 1.5 T or 3 T", sequences_useful=["se", "afi", "spgr"],
                   agent_side_formulas="edges at half the water signal; PIU; ghosting ratio; a distortion from a field gradient "
                                       "scales with 1 / bandwidth and follows the readout axis; a ratio of two images acquired with "
                                       "the same coil cancels the receive profile"),
        hidden=["which faults are present and their sizes", "gradient calibration, shim, transmit and receive profiles",
                "position of the phantom (within 1 mm, 0.3 degrees)", "noise level", "the 128-bit world seed"],
        told=["the phantom and its diameter", "the ACR T1 protocol of the site", "the definitions of the four quantities", "the limits and the tolerances",
              "the closed fault list with the test each fault moves", "the instrument datasheet", "realised protocol, timing and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        limits=dict(max_scanner_requests=64, max_tool_calls=64),
        deliverable={"tool": "submit", "format": {"measured": {q: "<number>" for q in QUANTITIES}, "failed_tests": ["<test>"], "faults": ["<fault>"]}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan"],
                    error="largest over the four quantities of |delivered - measured| / tolerance",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing="all gates; every quantity within its tolerance; the failed tests and the faults exactly right"),
        competencies=list(COMPETENCIES),
        task_kind=controls.task_kind(),
        not_simulated=dict(qa.NOT_SIMULATED, other_sections="resolution (slice 1) and low contrast (slices 8 to 11) are measured by "
                           "`mrgym qa acr`, not here: a session images one section"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier's scan is noise-free",
    )
