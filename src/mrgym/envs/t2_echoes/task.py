"""``Scanner/T2Echoes``: instances and the verifier.

The job: deliver the transverse relaxation time T2 of three marked vials from a multi-echo spin echo (``mese``), on a
scanner whose transmit scale is hidden. The scan-time budget pays one scan of useful length: on this console every
scan starts with ``ADJUST_S`` seconds of adjustments (``console_system``: its overhead per scan). One echo train
fits; the series of single-echo spin echoes by which ``Scanner/T2Decay`` measures T2 (slowly, and without the effect
below) does not, and two scans of any kind leave 10 s of scanning between them. That is what the adjustment time is
for: a ``mese`` scan costs what an ``se`` scan costs, and with the 3 s of the default console two single echoes at
half the repetition time fit the scan time of one train and pass as often (59 of 60 instances at every level,
measured when the family was built). The budget is not set to exactly one scan: ``mrgym check`` plays two short
scans on every environment (a repeated scan must draw new noise), and they have to fit. What that leaves open, also
measured when the family was built: two single echoes of 10 s of scanning together (a 32 matrix, repetition times
near 170 ms) still pass 48 and 50 of 60 instances at levels 1 and 2 and 14 of 60 at level 3. The echo train is the
measurement the budget is made for; it is not the only one that fits. More noise closes that route only as fast as
it costs the train its own margin (at three times the noise: 9 and 17 of 60 against 57 and 59 for the train).

What stands between the echoes and T2. A refocusing pulse that is not 180 degrees turns part of the transverse
magnetisation onto the longitudinal axis, where it decays with T1 instead of T2, and later pulses bring it back as
stimulated echoes. The first echo is low and the later ones are high, so the train decays more slowly than
exp(-TE / T2). On this scanner the angle is never 180 degrees everywhere: it falls to zero across the edge of the slice
profile of the refocusing pulse, and it scales with the hidden transmit factor. A straight line through ln(signal)
of the echoes reads T2 long (``BIAS``; tests/test_t2_echoes.py measures every number of that table on the engine, for
T1 of 400 to 2000 ms and T2 of 40 to 200 ms): 10 to 35 % with the console's default pulse and 4 to 24 % with the
``sharp`` pulse when the transmit scale is off by 8 to 22 %, 9 to 76 % when it is off by 25 to 45 %; leaving out the
first echo or the odd echoes takes a few points off and no more. The echoes are those of a known recursion
(``model``): fitted with the refocusing angle as a free parameter it gives T2 within -2 to +6 % on the ``sharp``
pulse, whose angle is nearly the same through the slice (-8 to +11 % with the transmit scale 25 to 45 % off), and
within -1 to +10 % on the default pulse. (The recursion is one train from rest. The scanner serves the periodic
state: what a pulse stored along z in one repetition reaches the next, and spins diffuse across the crushers. Both
are in these numbers; for hard pulses they move an echo by 0.2 to 0.9 % of the first.)

Deliverable    {"t2_ms": {"R1": 61.0, "R2": 142.0, "R3": 88.0}}        T2 of every marked region, milliseconds
Pass rule      for every marked region the delivered T2 lies within a factor ``1 + TOLERANCE`` of the true T2 of its
               material. The tolerance is a constant of the family; it is not derived from what any policy achieves.

Levels. 1 states the method (``METHOD``: the scan, the model as a text for ``calc``, the fit). 2 withholds it; the
transmit scale is off by 8 to 22 % either way, and a line through the echoes without the first one, or through the
even ones, still passes in a third to a half of the instances. 3 puts the transmit scale 25 to 45 % off: every
straight-line rule then fails and only the model fit is left.

What the family does not reach: T2 below about 35 ms. With the echo spacing of the sharp pulse (13.7 ms) four to six
echoes carry such a vial, the first echoes are where the angles of the slice profile differ most from one angle, and
the one-angle fit reads 5 to 17 % long (measured when the family was built: a level with T2 from 15 to 400 ms passed
the reference in 46 of 60 instances and was withdrawn). That needs the echo-train model summed over the slice profile
of both pulses, which this family does not ask for.

A measurement task by construction: the reference is one scan and a fixed estimator at every level.

What the pass rule depends on, and what it does not. The true T2 is a number of the hidden world: the verifier scans
nothing. What a policy measures are the echoes the engine serves: the periodic state of the repetition, in which what
an imperfect refocusing pulse stored along z reaches the next excitation, and what is still transverse at the end of
a repetition is taken as dephased (docs/physics/limits.md). The materials here have T2 of at most 200 ms and T1 of at
most 2000 ms, no fluid. The numbers of ``BIAS`` and the gate are measured on that engine, at the repetition time the
budget pays (1150 ms).
"""
from __future__ import annotations

import math
import os

from ...scanner import world as W
from .. import family as F
from . import model

FAMILY = "t2_echoes"
ENV_ID = "Scanner/T2Echoes-v0"
SUBJECT = "A"                               # signal and contrast
KIND = "measure"
ISOLATES = "stimulated echoes of an echo train: T2 from a multi-echo spin echo with the echo-train model"
SUBMIT_DOC = "Deliver T2 of every marked region, in milliseconds."
T1_RANGE_MS = (400.0, 2000.0)               # told to the agent
T2_VALID_MS = (2.0, 5000.0)                 # anything outside is not a T2 in milliseconds
TOLERANCE = 0.10                            # a delivered T2 may be off by this factor: T2 / 1.1 to 1.1 x T2
N_REGIONS = 3
OUTCOMES = ("no_region_within", "some_regions_within")      # of a delivered, measured answer that fails (worst first)
# Examined competencies this family exercises (rows of ``mrgym.coverage``): the echo time, the echo train and its
# refocusing angle, the transmit calibration it depends on.
COMPETENCIES = ("parameter.te", "parameter.etl", "parameter.flip_angle", "qc.transmit_gain")
SEQUENCES = ("mese",)                       # what the console of this family offers beyond the six
ADJUST_S = 40.0                             # adjustments before every scan of this console (its overhead per scan), seconds
BUDGET_S = 90.0                             # one echo train of the method (82.55 s); what is left pays no second scan of more than a few seconds
BASE = dict(kind="qa", b0_t=1.5, console_sequences=list(SEQUENCES), console_system=dict(overhead_s=ADJUST_S), tx_gain_dev=(0.08, 0.22), sigma0=0.10)
TX_BANDS = {1: "8-22", 2: "8-22", 3: "25-45"}       # percent the transmit scale is off, either way, by level: the key into ``BIAS``
# Bias of T2 in percent, (lowest, highest) over T1 of 400 to 2000 ms and T2 of 40 to 200 ms, of the estimators on the
# train of the method (16 echoes), by pulse and by how far the transmit scale is off. Measured on the engine; a test
# measures them again and holds every number of this table and of the texts written from it.
BIAS = dict(
    line={("sharp", "8-22"): (4, 24), ("normal", "8-22"): (10, 35), ("sharp", "25-45"): (9, 76)},
    line_without_first={("sharp", "8-22"): (2, 20), ("sharp", "25-45"): (5, 66)},
    model={("sharp", "8-22"): (-2, 6), ("normal", "8-22"): (-1, 10), ("sharp", "25-45"): (-8, 11)},
)
QA = dict(n_ring=(4, 5), vial_t1_ms=T1_RANGE_MS, t2_over_t1_max=0.4)
# The method level 1 states, as data: the statement prints it and the reference policy plays it.
METHOD = dict(scan=dict(sequence="mese", rf_pulse="sharp", n_echoes=16, tr_ms=1150.0, matrix=48, partial_fourier=0.75),
              min_share=0.08, min_echoes=4, start=(80.0, 140.0), t2_bounds_ms=(5.0, 2000.0), theta_bounds_deg=(60.0, 180.0))
LEVELS = {
    1: dict(world_spec=dict(BASE, qa=dict(QA)), t2_range_ms=(40.0, 200.0), budget_s=BUDGET_S, hints=("method",),
            summary="the method is stated: one echo train with the sharp pulse, the echo-train model fitted with the refocusing angle free"),
    2: dict(world_spec=dict(BASE, qa=dict(QA)), t2_range_ms=(40.0, 200.0), budget_s=BUDGET_S,
            summary="no method: a line through ln(signal) of the echoes reads T2 4 to 35 % long, by an amount that depends on the pulse "
                    "and on the hidden transmit scale"),
    3: dict(world_spec=dict(BASE, tx_gain_dev=(0.25, 0.45), qa=dict(QA)), t2_range_ms=(40.0, 200.0), budget_s=BUDGET_S,
            summary="the transmit scale is off by 25 to 45 %: no straight line through the echoes comes close (5 to 76 % long, with or "
                    "without the first echo); the echo-train model still fits"),
}


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    """Episode config. Nothing public is derived from the seed except through the world it generates (the positions
    of the marked regions)."""
    lv = LEVELS[level]
    regions = W.make_world(seed, lv["world_spec"], layout_only=True)["regions"]
    targets = [r for r in regions if int(r["id"][1:]) <= N_REGIONS]     # region ids are a random permutation of the vials
    task = dict(family=FAMILY, regions=F.public_regions(targets), t2_range_ms=list(lv["t2_range_ms"]), t1_range_ms=list(T1_RANGE_MS),
                transmit_off_percent=[round(100 * v) for v in lv["world_spec"]["tx_gain_dev"]], tolerance=TOLERANCE, deliverable={"t2_ms": {"<region id>": "<number, milliseconds>"}})
    if lv.get("hints"):
        task["hints"] = list(lv["hints"])
    return dict(task_id=task_id or "t2echoes-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed,
                world_spec=lv["world_spec"], budget_s=float(lv["budget_s"]), max_requests=40, task=task)


def fit_text() -> str:
    """The fit of level 1 as a text for ``calc`` (after ``model.CALC_TEXT``): ``echoes`` holds the kept echoes of one
    region, ``esp`` the echo spacing; the answer is ``p[1]``."""
    m = METHOD
    (t2_lo, t2_hi), (th_lo, th_hi) = m["t2_bounds_ms"], m["theta_bounds_deg"]
    return (f"p, cov = curve_fit(lambda k, a, t2, theta: a * train(len(k), esp, t2, theta), arange(len(echoes)), echoes, "
            f"[echoes[0], {m['start'][0]:g}, {m['start'][1]:g}], bounds=([0, {t2_lo:g}, {th_lo:g}], [inf, {t2_hi:g}, {th_hi:g}]))")


def method_text() -> str:
    """The method of level 1, written from ``METHOD`` (the reference policy plays the same numbers)."""
    m, s = METHOD, METHOD["scan"]
    recursion = "\n".join("     " + line for line in model.CALC_TEXT.split("\n"))
    return f"""How to do it (this level states the method).
  1. Run one `mese` scan: rf_pulse "{s['rf_pulse']}", n_echoes {s['n_echoes']}, tr_ms {s['tr_ms']:g}, matrix {s['matrix']}, partial_fourier {s['partial_fourier']:g},
     echo_spacing_ms left at its minimum. Image k is the echo at k x echo spacing. (With the sharp pulse the refocusing angle
     is nearly the same through the slice; with the default pulse the model of step 3 leaves T2 up to {BIAS['model'][('normal', '8-22')][1]} % long.)
  2. For each region take `|complex_mean|` of the images in order. Keep the echoes from the first one down to the last
     that is at least {100 * m['min_share']:g} % of the largest (at least {m['min_echoes']}): `echoes`.
  3. Do not fit a line to ln(echoes): the refocusing angle is not 180 degrees, the first echo is low and the later ones
     are raised by stimulated echoes, and the line reads T2 {BIAS['line'][('sharp', '8-22')][0]} to {BIAS['line'][('sharp', '8-22')][1]} % long. Fit the echoes of a train with one
     refocusing angle theta instead, with `calc` (esp = the echo spacing in ms; T1 is set to {model.T1_ASSUMED_MS:g} ms, the echoes
     depend on it weakly):
{recursion}
     {fit_text()}
     T2 = p[1]. (p[2] is an effective angle of the slice, not the transmit scale.)"""


def statement(cfg: dict) -> str:
    t = cfg["task"]
    f = 1.0 + t["tolerance"]
    (lo, hi), (t1_lo, t1_hi) = t["t2_range_ms"], t["t1_range_ms"]
    hints = "\n\n" + method_text() if "method" in t.get("hints", ()) else ""
    return f"""# Measure T2 of each marked vial from an echo train
A test object with vials of unknown materials lies in the field of view.
Deliver the transverse relaxation time T2 of each marked region, in milliseconds, with `submit`.
Regions (circles; centre and radius in mm from the isocentre; each lies inside one homogeneous material):
{F.region_table(t['regions'])}
T2 values lie between {lo:g} and {hi:g} ms, T1 values between {t1_lo:g} and {t1_hi:g} ms.
The transmit scale of this scanner is off by {t['transmit_off_percent'][0]:g} to {t['transmit_off_percent'][1]:g} %, up or down: every flip angle is that much larger or
smaller than nominal (`tx_scale` multiplies it for one scan).
Nothing else about the object or the system is known.
This console has a multi-echo spin echo, `mese`: up to 32 echoes per scan, one image each (scanner_info(topic="mese")).
Every scan of this console starts with {ADJUST_S:g} s of adjustments, which are part of its scan time: the scan time pays one scan of useful length.

Passing: for every region the delivered T2 lies within a factor {f:g} of the true T2 (between T2 / {f:g} and {f:g} x T2).
The scanner's log must show that you measured. An answer in the wrong unit is wrong.""" + hints


def submit_schema(cfg: dict) -> dict:
    ids = [r["id"] for r in cfg["task"]["regions"]]
    return {"type": "object", "additionalProperties": False, "required": ["t2_ms"],
            "properties": {"t2_ms": {"type": "object", "additionalProperties": False, "required": ids,
                                     "description": "T2 of every marked region in milliseconds, by region id",
                                     "properties": {i: {"type": "number", "minimum": T2_VALID_MS[0], "maximum": T2_VALID_MS[1]} for i in ids}}}}


def check(cfg: dict, answer, exam: dict):
    ids = [r["id"] for r in cfg["task"]["regions"]]
    if not isinstance(answer, dict) or set(answer) != {"t2_ms"} or not isinstance(answer["t2_ms"], dict) or set(answer["t2_ms"]) != set(ids):
        return 'the deliverable is {"t2_ms": {' + ", ".join(f'"{i}": <number>' for i in ids) + "}}"
    for i in ids:
        v = answer["t2_ms"][i]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return f"the T2 of {i} is not a finite number"
        if not T2_VALID_MS[0] <= v <= T2_VALID_MS[1]:
            hint = " (a value of this size looks like seconds)" if 0 < v < T2_VALID_MS[0] else ""
            return f"the T2 of {i} is {v:g}: not a T2 in milliseconds, which lies between {T2_VALID_MS[0]:g} and {T2_VALID_MS[1]:g}{hint}"
    return None


def true_t2_ms(cfg: dict, world: dict | None = None) -> dict:
    """Verifier side: {region id: T2 of its material in ms}."""
    world = world or W.make_world(cfg["seed"], cfg["world_spec"], layout_only=True)
    truth = W.truth(world)
    return {r["id"]: float(truth[r["id"]]["t2_ms"]) for r in cfg["task"]["regions"]}


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    world = world or W.make_world(cfg["seed"], cfg["world_spec"], layout_only=True)
    problem = check(cfg, answer, world.get("console") or {})
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    truth, limit = true_t2_ms(cfg, world), math.log(1.0 + cfg["task"]["tolerance"])
    miss = {i: abs(math.log(float(answer["t2_ms"][i]) / t)) for i, t in truth.items()}       # |ln(delivered / true)| per region
    checks = dict(deliverable_valid=True, **{f"t2_within_tolerance_{i}": m <= limit for i, m in miss.items()})
    bad = [f"{i} is off by {100 * (float(answer['t2_ms'][i]) / truth[i] - 1.0):+.1f} %" for i, m in miss.items() if m > limit]
    return F.result(checks, "; ".join(bad), error=max(miss.values()), tolerance=limit,
                    per_region={i: round(float(answer["t2_ms"][i]) / truth[i] - 1.0, 5) for i in truth},
                    distance=dict(regions=len(miss), within=len(miss) - len(bad), worst_log_ratio=max(miss.values())))


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured answer that fails a check: one of ``OUTCOMES``."""
    return "some_regions_within" if any(v for k, v in score["checks"].items() if k.startswith("t2_within_tolerance_")) else "no_region_within"


def card() -> dict:
    from ... import gate, learn
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    kind = F.kind_by_open_loop(gate.load(ENV_ID))
    off = "; ".join(f"level {lv}: {TX_BANDS[lv].replace('-', ' to ')} %" for lv in LEVELS)
    return dict(
        id=ENV_ID, title="T2 of marked vials from a multi-echo spin echo", kind="candidate",
        given="A QA phantom with 4 to 6 vials, three of them marked; a scanner with a multi-echo spin echo and a hidden transmit scale; "
              "a scan-time budget that pays one scan",
        deliver="T2 of every marked vial, in ms",
        scored_by=f"The hidden T2 of every marked vial: each delivered value within a factor {1 + TOLERANCE:g} of it",
        summary="Subject A (signal and contrast), a single-skill measurement: the stimulated echoes of an echo train. The refocusing "
                "angle is not 180 degrees (slice profile, transmit scale), so the echoes do not decay as exp(-TE / T2) and a line "
                "through their logarithm reads T2 long. The echoes follow a known recursion; fitted with the refocusing angle free it "
                "gives T2. Scanner/T2Decay measures the same quantity with one spin echo per echo time; here the budget pays one scan.",
        task_kind=kind["kind"], task_kind_why=kind["evidence"],
        world=dict(phantom="QA phantom (a disc of gel with vials), 1.5 T", sequences_useful=["mese"],
                   console=f"offers `mese` beyond the six sequences of every console; adjustments take {ADJUST_S:g} s before every scan",
                   agent_side_formulas="the recursion of an echo train with one refocusing angle (level 1 prints it as a text for `calc`; the "
                                       "datasheet gives the first two echoes in closed form); the served echoes are that train "
                                       "integrated over the profiles of both pulses of the 2D slice"),
        hidden=["T2, T1, proton density of every material", "geometry of body and vials",
                f"transmit gain (off either way; {off}) and B1 pattern", "receive profile", "noise level", "the 128-bit world seed"],
        told=["region centres and radii", "the range of T2 (" + "; ".join(f"level {lv}: {d['t2_range_ms'][0]:g} to {d['t2_range_ms'][1]:g} ms" for lv, d in LEVELS.items())
              + f") and of T1 ({T1_RANGE_MS[0]:g} to {T1_RANGE_MS[1]:g} ms)", "by how much the transmit scale is off at most and at least",
              "the pass tolerance", "level 1: the method", "the instrument datasheet with its supplement for `mese`", "budget and request limits",
              "realised protocol, timing and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        practice_levels=[lv for lv, d in LEVELS.items() if d.get("kind") == "practice"],
        learnability=learn.stored(ENV_ID),
        limits=dict(max_scanner_requests=40, max_tool_calls=40),
        deliverable={"tool": "submit", "format": {"t2_ms": {"<region id>": "<milliseconds>"}}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan",
                           f"deliverable valid: exactly the region ids, values in {T2_VALID_MS[0]:g} to {T2_VALID_MS[1]:g} ms"],
                    error="largest over the regions of |ln(T2_delivered / T2_true)|",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing=f"all gates and, for every region, error <= ln({1 + TOLERANCE:g})"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier reads the hidden T2 and scans nothing",
    )
