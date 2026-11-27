"""``Scanner/T2Decay``: instances and the verifier.

The job: deliver the transverse relaxation time T2 of several marked vials of a test object. A spin echo carries
exp(-TE / T2), so T2 is the slope of ln(signal) against the echo time. The scanner has no multi-echo spin echo: every
echo time is a scan of its own, and two things stand between those scans and T2.

  recovery     the longitudinal magnetisation recovers from the refocusing pulse, not from the excitation, so at one
               repetition time a longer echo time leaves less time to recover: the later echo is smaller by more
               than T2 decay and T2 reads short (by 2 to 20 % at TR 600 ms; by more the longer T1 and the larger
               T2 / T1). With ideal pulses the signal is 1 - 2 E(TR - TE / 2) + E(TR), which is 1 - E(TR - TE) to
               first order in TE / T1: holding TR - TE fixed removes the effect. On the 2D slice of this scanner the
               refocusing pulse inverts the slice only in part, and the time to hold fixed is TR - ``RECOVERY_SHARE``
               x TE (``RECOVERY_SHARE`` = 0.75; tests/test_t2_decay.py derives the effect and the share from the
               pulses of the datasheet with a rotation cascade written there). T1 is hidden, so the effect cannot be
               divided out. Holding TR - TE fixed (the rule for ideal pulses) leaves T2 up to 7 % long at TR 600 ms.
  noise floor  the mean of a magnitude does not fall below the noise (and not below what the surrounding gel leaks
               into the region): an echo that has decayed to that level reads too high and T2 reads long.

Deliverable    {"t2_ms": {"R1": 61.0, "R2": 142.0, "R3": 88.0}}        T2 of every marked region, milliseconds
Pass rule      for every marked region the delivered T2 lies within a factor ``1 + tolerance`` of the true T2 of its
               material (between T2 / 1.1 and 1.1 x T2). The tolerance is a constant of the family in physical
               units; it is not derived from what any policy achieves.

Three regions make the deliverable multi-coordinate: one constant T2 for every region passes only when all three
hidden materials lie within 10 % of it.

Levels. 1 states the method (``METHOD``: the scans and the fit). 2 withholds it and gives a quarter of the scan time:
the repetition times that fit are short, where the recovery effect is large, and three echo times have to be chosen.
3 widens the range of T2 from a factor 5 to a factor 27: an echo time long enough for the longest T2 is at the noise
floor for the shortest, so which echoes enter the fit decides.

A measurement task at every level: the reference is a fixed list of scans with a fixed estimator. At level 3 a list
of four echo times spread over the told range serves every vial; choosing the later echo times from the earlier ones
(``controls.adaptive``) is reported next to it.
"""
from __future__ import annotations

import math
import os

from ...scanner import world as W
from .. import family as F

FAMILY = "t2_decay"
ENV_ID = "Scanner/T2Decay-v0"
SUBJECT = "A"                               # signal and contrast
KIND = "measure"
ISOLATES = "T2 decay of the spin echo"
SUBMIT_DOC = "Deliver T2 of every marked region, in milliseconds."
T1_RANGE_MS = (300.0, 2500.0)               # told to the agent
T2_VALID_MS = (2.0, 5000.0)                 # anything outside is not a T2 in milliseconds
TOLERANCE = 0.10                            # a delivered T2 may be off by this factor: T2 / 1.1 to 1.1 x T2
N_REGIONS = 3
RECOVERY_SHARE = 0.75                       # the recovery before an excitation of this scanner's 2D spin echo lasts TR - 0.75 TE
OUTCOMES = ("no_region_within", "some_regions_within")      # of a delivered, measured answer that fails (worst first)
# Examined competencies this family exercises (rows of ``mrgym.coverage``): the echo time and the repetition time of a
# spin echo, the noise level (the floor of a magnitude).
COMPETENCIES = ("parameter.te", "parameter.tr", "qc.snr")
BASE = dict(kind="qa", b0_t=1.5)
QA = dict(n_ring=(4, 5))
# The method level 1 states, as data: the statement prints it and the reference policy plays it.
METHOD = dict(te_ms=(12.0, 40.0, 80.0, 160.0), tr0_ms=1500.0, matrix=64, partial_fourier=0.75,
              background=dict(id="background", y_mm=-105.0, x_mm=-105.0, radius_mm=10.0), floor_factor=5.0, min_share=0.10)
LEVELS = {
    1: dict(world_spec=dict(BASE, qa=dict(QA)), t2_range_ms=(40.0, 200.0), budget_s=340.0, hints=("method",),
            summary="the method is stated: four echo times with the recovery held fixed, a line through ln(signal), "
                    "the echoes at the noise floor left out"),
    2: dict(world_spec=dict(BASE, qa=dict(QA), sigma0=0.10), t2_range_ms=(40.0, 200.0), budget_s=85.0,
            summary="no method and a quarter of the scan time: short repetition times, where the recovery differs between "
                    "echo times, and three echo times to choose"),
    3: dict(world_spec=dict(BASE, qa=dict(QA, vial_t2_ms=(15.0, 400.0)), sigma0=0.06), t2_range_ms=(15.0, 400.0), budget_s=165.0,
            summary="T2 from 15 to 400 ms: an echo time long enough for the longest T2 is at the noise floor for the shortest"),
}


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    """Episode config. Nothing public is derived from the seed except through the world it generates (the positions
    of the marked regions)."""
    lv = LEVELS[level]
    regions = W.make_world(seed, lv["world_spec"], layout_only=True)["regions"]
    targets = [r for r in regions if int(r["id"][1:]) <= N_REGIONS]     # region ids are a random permutation of the vials
    task = dict(family=FAMILY, regions=F.public_regions(targets), t2_range_ms=list(lv["t2_range_ms"]), t1_range_ms=list(T1_RANGE_MS),
                tolerance=TOLERANCE, deliverable={"t2_ms": {"<region id>": "<number, milliseconds>"}})
    if lv.get("hints"):
        task["hints"] = list(lv["hints"])
    return dict(task_id=task_id or "t2decay-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed,
                world_spec=lv["world_spec"], budget_s=float(lv["budget_s"]), max_requests=40, task=task)


def method_text() -> str:
    """The method of level 1, written from ``METHOD`` (the reference policy plays the same numbers)."""
    m, bg = METHOD, METHOD["background"]
    listed = lambda values: ", ".join(f"{v:g}" for v in values[:-1]) + f" and {values[-1]:g}"      # noqa: E731
    tes, trs = listed(m["te_ms"]), listed([m["tr0_ms"] + RECOVERY_SHARE * te for te in m["te_ms"]])
    return f"""How to do it (this level states the method).
  1. Run {len(m['te_ms'])} `se` scans with te_ms {tes}. Give each tr_ms = {m['tr0_ms']:g} + {RECOVERY_SHARE:g} x te_ms ({trs}),
     matrix {m['matrix']} and partial_fourier {m['partial_fourier']:g}. (With one tr_ms for all, a longer echo leaves less time to recover
     and T2 reads short.)
  2. In every scan read `mean_magnitude` of each marked region and of the circle y_mm {bg['y_mm']:g}, x_mm {bg['x_mm']:g}, radius_mm {bg['radius_mm']:g}.
     That circle holds no object: the mean of its values over the scans is the noise floor of a magnitude.
  3. For each region keep the two shortest echoes, and of the others those whose signal is at least {m['floor_factor']:g} times the
     noise floor and at least {100 * m['min_share']:g} % of the region's signal at te_ms {m['te_ms'][0]:g}.
  4. Fit a straight line to ln(signal) against te_ms through the kept echoes (`calc`: polyfit(te, log(s), 1)).
     T2 = -1 / slope."""


def statement(cfg: dict) -> str:
    t = cfg["task"]
    f = 1.0 + t["tolerance"]
    (lo, hi), (t1_lo, t1_hi) = t["t2_range_ms"], t["t1_range_ms"]
    hints = "\n\n" + method_text() if "method" in t.get("hints", ()) else ""
    return f"""# Measure T2 of each marked vial
A test object with vials of unknown materials lies in the field of view.
Deliver the transverse relaxation time T2 of each marked region, in milliseconds, with `submit`.
Regions (circles; centre and radius in mm from the isocentre; each lies inside one homogeneous material):
{F.region_table(t['regions'])}
T2 values lie between {lo:g} and {hi:g} ms, T1 values between {t1_lo:g} and {t1_hi:g} ms.
Nothing else about the object or the system is known.
One `se` scan gives one echo time: the scanner has no multi-echo spin echo.

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
    return dict(
        id=ENV_ID, title="T2 of marked vials from spin echoes", kind="candidate",
        given="A QA phantom with 4 to 6 vials, three of them marked; the scanner; a scan-time budget",
        deliver="T2 of every marked vial, in ms",
        scored_by=f"The hidden T2 of every marked vial: each delivered value within a factor {1 + TOLERANCE:g} of it",
        summary="Subject A (signal and contrast), a single-skill measurement: T2 decay of the spin echo. Every echo time is "
                "a scan of its own. The slope of ln(signal) against echo time is -1 / T2 only if the recovery before the "
                "excitation is the same in every scan and the echoes at the noise floor are left out.",
        task_kind=kind["kind"], task_kind_why=kind["evidence"],
        world=dict(phantom="QA phantom (a disc of gel with vials), 1.5 T", sequences_useful=["se"],
                   agent_side_formulas="S = M0 (1 - 2 E(TR - TE/2) + E(TR)) exp(-TE / T2) for ideal pulses (datasheet section 5); the "
                                       "served signal is the pulse chain integrated over the profiles of both pulses of the 2D slice"),
        hidden=["T2, T1, proton density and T2' of every material", "geometry of body and vials", "transmit gain and B1 pattern",
                "receive profile", "noise level", "the 128-bit world seed"],
        told=["region centres and radii", "the range of T2 (" + "; ".join(f"level {lv}: {d['t2_range_ms'][0]:g} to {d['t2_range_ms'][1]:g} ms" for lv, d in LEVELS.items())
              + f") and of T1 ({T1_RANGE_MS[0]:g} to {T1_RANGE_MS[1]:g} ms)",
              "the pass tolerance", "level 1: the method", "the instrument datasheet", "budget and request limits",
              "realised protocol, timing and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        practice_levels=[lv for lv, d in LEVELS.items() if d.get("kind") == "practice"],
        learnability=learn.stored(ENV_ID),
        limits=dict(max_scanner_requests=40, max_tool_calls=48),
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
