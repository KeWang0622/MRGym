"""``Scanner/T1Vial``: instances and the verifier.

The job: deliver the longitudinal relaxation time T1 of several marked vials, on a system whose transmit calibration
is correct. This is the single-skill version of ``Scanner/T1Transmit`` (there the flip angle is off by a hidden,
position-dependent factor and has to be mapped; here it is not, and the task says so).

What is hidden: T1 of every vial, and the noise level of the receiver. What decides the answer is how the
measurement is made:

  the method        inversion recovery. At a FIXED repetition time the signal of a vial is exactly
                        S(TI) = a - b exp(-TI / T1)
                    with the T1 of the material, whatever the repetition time, the completeness of the inversion and
                    the slice profile are: each of them changes a and b and none of them changes the exponent (a
                    column of spins at any height in the slice has this form with its own a and b, and so has their
                    sum). Three inversion times give T1; the read-outs before the null are negative and have to be
                    signed with the phase image. The textbook two-parameter form S = M (1 - 2 exp(-TI / T1)) holds
                    only for a repetition time of many T1 and a complete inversion.
  the timing        an inversion-recovery scan costs its repetition time per phase-encode line, so the budget decides
                    the repetition time, and the repetition time bounds the longest inversion time. Level 2 pays for
                    three scans at a repetition time of 900 ms, shorter than the longest T1: the three inversion times
                    have to span the recovery from its start to its end, and the fit must not assume full recovery.
  the range         level 3: vials from 60 to 2000 ms that differ by a factor of ten or more. Three inversion times
                    that serve a narrow range leave a short T1 with one point on its recovery and two on its plateau,
                    and four are needed.
  the read-out      a region mean also holds a few percent of what surrounds the vial (the point-spread function of a
                    coarse matrix), and the gel around a vial recovers with another T1. The fit turns that leak into an
                    error of T1 of a few percent, more when two inversion times sit on the plateau of the vial and
                    most for the shortest T1; a finer matrix halves it (the reference of level 3 uses matrix 48). With
                    partial Fourier the console synthesises the missing lines (homodyne detection, engine 0.3.0): the
                    image takes its phase from a low-resolution image, in which the gel reaches into a vial, so the
                    signed read-out of a vial near its null is biased and the error grows (more than twice on the
                    instance ``tests/test_t1_vial.py`` compares, at ``partial_fourier`` 0.55; on engine 0.2.0, which
                    filled the missing lines with zeros, it was larger), and the noise of a pixel is higher. The
                    statement says so at every level, and the scans of the stated method and of the reference are
                    full Fourier.

Deliverable    {"t1_ms": {"R1": 812.0, "R2": 1530.0, "R3": 96.0}}      T1 of every marked region, milliseconds
Pass rule      for every marked region the delivered T1 lies within a factor 1.10 of the true T1 of its material
               (between T1 / 1.1 and 1.1 x T1). The tolerance is a constant of the family in physical units.

Three regions make the deliverable multi-coordinate: one constant T1 for every region passes only when all three
hidden materials lie within 10 % of it.
"""
from __future__ import annotations

import math
import os

from ...scanner import console
from ...scanner import world as W
from .. import family as F

FAMILY = "t1_vial"
ENV_ID = "Scanner/T1Vial-v0"
SUBMIT_DOC = "Deliver T1 of every marked region, in milliseconds."
T1_VALID_MS = (10.0, 20000.0)               # anything outside is not a T1 in milliseconds
TOLERANCE = 0.10                            # a delivered T1 may be off by this factor: T1 / 1.1 to 1.1 x T1
OUTCOMES = ("no_region_within", "some_regions_within")      # of a delivered, measured answer that fails (worst first)
# Examined competencies this family exercises (rows of ``mrgym.coverage``): choosing inversion times and a repetition time.
COMPETENCIES = ("parameter.ti", "parameter.tr")
# The scan of the stated method: 32 lines, one repetition each, and one dummy repetition. Full Fourier on purpose
# (module docstring, "the read-out").
PROBE = dict(sequence="ir", matrix=32)
NARROW_MS, WIDE_MS = (300.0, 1500.0), (60.0, 2000.0)
MIN_SPREAD = 10.0                           # level 3: the longest marked T1 is at least this many times the shortest
# QA phantom at 1.5 T. The transmitter is calibrated (no gain error, no B1 pattern): the flip angle at the slice centre
# is the nominal one everywhere. The noise density is hidden: 5.3 times the default of this phantom, times a factor
# drawn log-normally with a standard deviation of 15 %.
BASE = dict(kind="qa", b0_t=1.5, tx_gain_dev=(0.0, 0.0), b1_shape_amp=(0.0, 0.0), sigma0=0.16, sigma0_log_sd=0.15)
STATED = dict(tr_ms=1500.0, ti_ms=(30.0, 400.0, 1450.0))        # the three scans level 1 states
LEVELS = {
    1: dict(world_spec=dict(BASE, qa=dict(vial_t1_ms=NARROW_MS, n_ring=(4, 5))), t1_range_ms=NARROW_MS, n_regions=3, budget_s=165.0,
            method=True, summary="T1 from 300 to 1500 ms; the method is stated: three inversion times and a signed fit"),
    2: dict(world_spec=dict(BASE, qa=dict(vial_t1_ms=NARROW_MS, n_ring=(4, 5))), t1_range_ms=NARROW_MS, n_regions=3, budget_s=100.0,
            method=False, summary="the same vials on 60 % of the scan time: a shorter repetition time, inversion times that still span the recovery"),
    3: dict(world_spec=dict(BASE, qa=dict(vial_t1_ms=WIDE_MS, n_ring=(4, 5))), t1_range_ms=WIDE_MS, n_regions=3, budget_s=705.0,
            min_spread=MIN_SPREAD, method=False,
            summary="T1 from 60 to 2000 ms and very different between the vials: no three inversion times serve all"),
}


def _targets(regions, n):
    return [r for r in regions if int(r["id"][1:]) <= n]             # region ids are a random permutation of the vials


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    """Episode config. Level 3 wants vials that differ: a drawn world whose marked T1 values span less than
    ``min_spread`` is passed over for the next attempt of the seed (``family.reseed``)."""
    lv = LEVELS[level]
    for attempt in range(64):
        world_seed = F.reseed(seed, attempt)
        layout = W.make_world(world_seed, lv["world_spec"], layout_only=True)
        targets = _targets(layout["regions"], lv["n_regions"])
        t1 = [layout["tissues"][r["cls"]]["t1_ms"] for r in targets]
        if max(t1) / min(t1) >= lv.get("min_spread", 1.0):
            break
    else:
        raise RuntimeError("no world with the required spread of T1 in 64 attempts")
    task = dict(family=FAMILY, regions=F.public_regions(targets), t1_range_ms=list(lv["t1_range_ms"]), tolerance=TOLERANCE,
                deliverable={"t1_ms": {"<region id>": "<number, milliseconds>"}})
    if lv["method"]:
        task["method"] = dict(protocol=dict(PROBE, tr_ms=STATED["tr_ms"]), ti_ms=list(STATED["ti_ms"]))
    if lv.get("min_spread"):
        task["min_spread"] = lv["min_spread"]
    return dict(task_id=task_id or "t1vial-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=world_seed,
                world_spec=lv["world_spec"], budget_s=float(lv["budget_s"]), max_requests=32, task=task)


def method_text(task: dict) -> str:
    """The stated method of level 1 as the paragraph appended to the statement ("" when the level states none)."""
    m = task.get("method")
    if not m:
        return ""
    p, (t1, t2, t3) = m["protocol"], m["ti_ms"]
    cost = console.plan(dict(p, ti_ms=t1))["scan_s"]
    return f"""How to do it (this level states the procedure).
  1. Run three inversion-recovery scans: sequence ir, tr_ms {p['tr_ms']:g}, matrix {p['matrix']} (no partial Fourier), with
     ti_ms {t1:g}, {t2:g} and {t3:g}. Each costs {cost:.1f} s. Read out the marked regions.
  2. Sign the read-outs. At ti_ms {t3:g} every material is past its null, so its signal is positive: S3 =
     `complex_mean_magnitude`. A read-out of the same region whose `complex_mean_phase_rad` differs from that of the
     ti_ms {t3:g} scan by more than pi / 2 (after wrapping the difference into -pi .. pi) is negative: S = -magnitude.
  3. At a fixed repetition time the signal is exactly S(TI) = a - b exp(-TI / T1). a and b drop out of
         q = (S3 - S2) / (S2 - S1) = (exp(-{t2:g} / T1) - exp(-{t3:g} / T1)) / (exp(-{t1:g} / T1) - exp(-{t2:g} / T1)).
     The right side grows with T1. Find the T1 at which it equals the measured q, by bisection between 50 and 5000 ms.
  4. Deliver that T1 in milliseconds for the region.
Do not use S = M (1 - 2 exp(-TI / T1)): it assumes a repetition time of many T1 and a complete inversion."""


def statement(cfg: dict) -> str:
    t = cfg["task"]
    f = 1.0 + t["tolerance"]
    lo, hi = t["t1_range_ms"]
    spread = (f" The longest of the marked T1 values is at least {t['min_spread']:g} times the shortest."
              if t.get("min_spread") else "")
    method = method_text(t)
    return f"""# Measure T1 of each marked vial
A test object with vials of unknown materials lies in the field of view.
Deliver the longitudinal relaxation time T1 of each marked region, in milliseconds, with `submit`.
Regions (circles; centre and radius in mm from the isocentre; each lies inside one homogeneous material):
{F.region_table(t['regions'])}
T1 values lie between {lo:.0f} and {hi:.0f} ms.{spread} Nothing else about the object is known, and the noise level is not given.
The transmit calibration is correct: at the slice centre the flip angle is the nominal one, everywhere in the object.
{F.OWN_SCANS.split(';')[0]}; they are charged to the scan-time budget.
A region read-out is a mean over the image as reconstructed: with partial Fourier the image takes its phase from a
low-resolution image, which mixes the material around a vial into the read-out, and its noise is higher.

Passing: for every region the delivered T1 lies within a factor {f:g} of the true T1 (between T1 / {f:g} and {f:g} x T1).
The scanner's log must show that you measured. An answer in the wrong unit is wrong.""" + ("\n\n" + method if method else "")


def submit_schema(cfg: dict) -> dict:
    ids = [r["id"] for r in cfg["task"]["regions"]]
    return {"type": "object", "additionalProperties": False, "required": ["t1_ms"],
            "properties": {"t1_ms": {"type": "object", "additionalProperties": False, "required": ids,
                                     "description": "T1 of every marked region in milliseconds, by region id",
                                     "properties": {i: {"type": "number", "minimum": T1_VALID_MS[0], "maximum": T1_VALID_MS[1]} for i in ids}}}}


def check(cfg: dict, answer, exam: dict):
    ids = [r["id"] for r in cfg["task"]["regions"]]
    if not isinstance(answer, dict) or set(answer) != {"t1_ms"} or not isinstance(answer["t1_ms"], dict) or set(answer["t1_ms"]) != set(ids):
        return 'the deliverable is {"t1_ms": {' + ", ".join(f'"{i}": <number>' for i in ids) + "}}"
    for i in ids:
        v = answer["t1_ms"][i]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return f"the T1 of {i} is not a finite number"
        if not T1_VALID_MS[0] <= v <= T1_VALID_MS[1]:
            hint = " (a value of this size looks like seconds)" if 0 < v < T1_VALID_MS[0] else ""
            return f"the T1 of {i} is {v:g}: not a T1 in milliseconds, which lies between {T1_VALID_MS[0]:g} and {T1_VALID_MS[1]:g}{hint}"
    return None


def true_t1_ms(cfg: dict, world: dict | None = None) -> dict:
    """Verifier side: {region id: T1 of its material in ms}."""
    world = world or W.make_world(cfg["seed"], cfg["world_spec"], layout_only=True)
    truth = W.truth(world)
    return {r["id"]: float(truth[r["id"]]["t1_ms"]) for r in cfg["task"]["regions"]}


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    world = world or W.make_world(cfg["seed"], cfg["world_spec"], layout_only=True)
    problem = check(cfg, answer, world.get("console") or {})
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    truth, limit = true_t1_ms(cfg, world), math.log(1.0 + cfg["task"]["tolerance"])
    miss = {i: abs(math.log(float(answer["t1_ms"][i]) / t)) for i, t in truth.items()}       # |ln(delivered / true)| per region
    checks = dict(deliverable_valid=True, **{f"t1_within_tolerance_{i}": m <= limit for i, m in miss.items()})
    bad = [f"{i} is off by {100 * (float(answer['t1_ms'][i]) / truth[i] - 1.0):+.1f} %" for i, m in miss.items() if m > limit]
    return F.result(checks, "; ".join(bad), error=max(miss.values()), tolerance=limit,
                    per_region={i: round(float(answer["t1_ms"][i]) / truth[i] - 1.0, 5) for i in truth},
                    distance=dict(regions=len(miss), within=len(miss) - len(bad), worst_log_ratio=max(miss.values())))


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured answer that fails a check: one of ``OUTCOMES``."""
    return "some_regions_within" if any(v for k, v in score["checks"].items() if k.startswith("t1_within_tolerance_")) else "no_region_within"


def task_kind(stored: dict | None = None) -> dict:
    """Measurement task or decision task, per level, from a gate result (default: the stored one), never typed."""
    from ... import gate
    return F.kind_by_open_loop(stored or gate.load(ENV_ID))


def card() -> dict:
    from ... import learn
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    kind = task_kind()
    return dict(
        id=ENV_ID, title="Measure T1 of each marked vial",
        given="A test object with vials of hidden T1 on a system with a calibrated transmitter; a scan-time budget",
        deliver="T1 of every marked vial, in ms",
        scored_by=f"The hidden T1 of every vial: each delivered value within a factor {1 + TOLERANCE:g} of it",
        summary="T1 by inversion recovery, the reference method of relaxometry, as one skill. At a fixed repetition time the "
                "signal of a vial is a - b exp(-TI / T1) with the true T1, whatever the repetition time, the inversion "
                "efficiency and the slice profile are, so three signed inversion times give T1. What has to be decided is "
                "the timing: the budget sets the repetition time, and the inversion times have to span the recovery of "
                "every vial. The transmitter is calibrated; Scanner/T1Transmit is the task with a transmit error. " + kind["sentence"],
        task_kind=kind["kind"], task_kind_why=kind["evidence"], task_kind_levels=kind["levels"], competencies=list(COMPETENCIES),
        world=dict(phantom="QA phantom (a disc of gel with 4 to 6 vials), 1.5 T, calibrated transmitter", sequences_useful=["ir"],
                   agent_side_formulas="S(TI) = a - b exp(-TI / T1) at a fixed repetition time; the sign of the signal is in "
                                       "the phase image"),
        hidden=["T1, T2 and proton density of every material", "noise level", "inversion efficiency", "receive profile",
                "off-resonance field", "the 128-bit world seed"],
        told=["the regions", "the T1 range of the level", "level 3: the least spread of the marked T1 values",
              "that the transmit calibration is correct", "the pass tolerance", "level 1: three scans and the equation for T1",
              "the instrument datasheet", "realised protocol, timing and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        practice_levels=[lv for lv, d in LEVELS.items() if d.get("kind") == "practice"],
        learnability=learn.stored(ENV_ID),
        limits=dict(max_scanner_requests=32, max_tool_calls=40),
        deliverable={"tool": "submit", "format": {"t1_ms": {"<region id>": "<milliseconds>"}}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan",
                           "deliverable valid: exactly the region ids, values in 10 to 20000 ms"],
                    error="largest over the regions of |ln(T1_delivered / T1_true)|",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing=f"all gates and, for every region, error <= ln({1 + TOLERANCE:g})"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier reads the hidden T1 and scans nothing",
    )
