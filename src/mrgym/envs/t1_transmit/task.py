"""``Scanner/T1Transmit``: instances and the verifier.

The job: deliver the longitudinal relaxation time T1 of several marked vials of a test object. Two things stand
between a gradient-echo signal and T1. The transmit field is not calibrated, so the flip angle is the nominal angle
times a hidden, position-dependent factor. And a slice-selective pulse turns the spins by less and less away from
the slice centre (the slice profile), so a 2D signal is not the signal at one angle. Both have to be measured or
modelled, or paid for: a non-selective 3D acquisition has one flip angle per voxel and costs the scan time of every
partition.

Deliverable    {"t1_ms": {"R1": 812.0, "R2": 1530.0, ...}}        T1 of every marked region, milliseconds
Pass rule      for every marked region the delivered T1 lies within a factor ``1 + tolerance`` of the true T1 of its
               material (between T1 / 1.1 and 1.1 x T1 at the tolerance of 10 %). The tolerance is a constant of the
               level in physical units; it is not derived from what any policy achieves.

Several regions make the deliverable multi-coordinate: one constant T1 for every region passes only when all the
hidden materials lie within 10 % of it.

Levels. The budget decides how the slice profile can be handled: level 1 has room for the non-selective 3D
acquisition (no profile to model); levels 2 and 3 have not, and the profile of the 2D slice has to be modelled.
"""
from __future__ import annotations

import math
import os

import numpy as np

from ...scanner import world as W
from .. import family as F

FAMILY = "t1_transmit"
ENV_ID = "Scanner/T1Transmit-v1"
SUBMIT_DOC = "Deliver T1 of every marked region, in milliseconds."
T1_RANGE_MS = (300.0, 2500.0)               # told to the agent
T1_VALID_MS = (30.0, 20000.0)               # anything outside is not a T1 in milliseconds
TOLERANCE = 0.10                            # a delivered T1 may be off by this factor: T1 / 1.1 to 1.1 x T1
OUTCOMES = ("no_region_within", "some_regions_within")      # of a delivered, measured answer that fails (worst first)
# QA phantom at 3 T, received with the body coil: a noise density about six times that of the head exam. The
# transmit adjustment is off by 8 to 25 % (the first confounder of this task; the slice profile is the second).
BASE = dict(kind="qa", b0_t=3.0, sigma0=0.173, sigma0_log_sd=0.12, tx_gain_dev=(0.08, 0.25), cf_offset_hz=25.0)
STRONG = dict(b1_shape_amp=(0.14, 0.24))
LEVELS = {
    1: dict(world_spec=dict(BASE, b1_shape_amp=(0.01, 0.04)), budget_s=(260.0, 320.0), tolerance=TOLERANCE,
            summary="mild B1 non-uniformity; the budget has room for a non-selective 3D acquisition"),
    2: dict(world_spec=dict(BASE, **STRONG), budget_s=(60.0, 100.0), tolerance=TOLERANCE,
            summary="strong B1 non-uniformity; the budget only fits 2D scans, whose slice profile has to be modelled"),
    3: dict(world_spec=dict(BASE, **STRONG, qa=dict(vial_t2_ms=(80.0, 250.0), vial_adc_um2_ms=(0.6, 1.2))), budget_s=(45.0, 60.0),
            tolerance=TOLERANCE,
            summary="as level 2 with a tighter budget, long T2 and slow diffusion (spoiling is hard)"),
}


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    """Episode config. Nothing public is derived from the seed except through the world it generates (region
    positions) and the budget."""
    lv = LEVELS[level]
    regions = W.make_world(seed, lv["world_spec"], layout_only=True)["regions"]
    budget = float(np.round(W.rng_for(seed, "task").uniform(*lv["budget_s"])))
    task = dict(family=FAMILY, regions=F.public_regions(regions), t1_range_ms=list(T1_RANGE_MS), tolerance=lv["tolerance"],
                deliverable={"t1_ms": {"<region id>": "<number, milliseconds>"}})
    return dict(task_id=task_id or "t1transmit-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed,
                world_spec=lv["world_spec"], budget_s=budget, max_requests=48, task=task)


def statement(cfg: dict) -> str:
    t = cfg["task"]
    f = 1.0 + t["tolerance"]
    return f"""# Measure T1 in marked regions with an uncalibrated transmit field
A test object of unknown composition lies in the field of view.
The transmit field is not calibrated: the flip angle a spin experiences at the centre of the slice is the nominal
flip angle times an unknown, position-dependent factor. A slice-selective pulse turns the spins by less away from
the slice centre (the slice profile; `scanner_info` with topic slice_profile describes it).

Deliver the longitudinal relaxation time T1 of each marked region, in milliseconds, with `submit`.
Regions (circles; centre and radius in mm from the isocentre; each lies inside one homogeneous material):
{F.region_table(t['regions'])}
T1 values lie between {T1_RANGE_MS[0]:.0f} and {T1_RANGE_MS[1]:.0f} ms. Nothing else about the object or the system is known.

Passing: for every region the delivered T1 lies within a factor {f:g} of the true T1 (between T1 / {f:g} and {f:g} x T1).
The scanner's log must show that you measured. An answer in the wrong unit is wrong."""


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


def card() -> dict:
    from ... import learn
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    return dict(
        id=ENV_ID, title="T1 in marked regions with an uncalibrated transmit field",
        given="A QA phantom with 4 to 7 vials, the scanner, a scan-time budget",
        deliver="T1 of every marked vial, in ms",
        scored_by=f"The hidden T1 of every vial: each delivered value within a factor {1 + TOLERANCE:g} of it",
        summary="A hidden QA phantom (gel body and 4 to 7 vials) on the 3 T scanner. The agent must deliver T1 per marked "
                "region within a scan-time budget. The answer is wrong unless the transmit-field error is mapped and "
                "corrected, the slice profile of a 2D scan is modelled (or a non-selective 3D acquisition is paid for) "
                "and the gradient-echo protocol keeps spoiling errors small.",
        task_kind="measurement",
        task_kind_why="a fixed protocol passes: the `open_loop` control plays one fixed scan list with a fixed estimator",
        why_practice="Registered as practice: every instance needs the same procedure, so the family tests whether the "
                     "measurement is done right, not a decision. The gate table reports the fixed protocol (`open_loop`) "
                     "next to the reference.",
        world=dict(phantom="QA phantom (disc of gel with vials), 3 T, body-coil receive",
                   sequences_useful=["afi", "spgr", "ir"],
                   agent_side_formulas="ideal-spoiling Ernst / AFI closed forms and the slice profile of the windowed-sinc pulse "
                                       "(datasheet sections 1 and 5); the served signal is the EPG steady state integrated over "
                                       "the slice and deviates from the closed forms at short TR and weak spoilers"),
        hidden=["geometry of body and vials", "PD, T1, T2, T2', diffusion coefficient per material",
                "B1+ pattern and transmit-gain error (8 to 25 %)", "B0 pattern and centre-frequency offset",
                "receive profile", "inversion efficiency", "noise level", "the 128-bit world seed"],
        told=["region centres and radii", "T1 prior range 300 to 2500 ms", "the pass tolerance", "the instrument datasheet, with the "
              "excitation pulse", "budget and request limits", "realised protocol, timing, limits and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"], budget_s=list(d["budget_s"])) for lv, d in LEVELS.items()},
        practice_levels=[],
        learnability=learn.stored(ENV_ID),
        limits=dict(max_scanner_requests=48, max_tool_calls=64),
        deliverable={"tool": "submit", "format": {"t1_ms": {"<region id>": "<milliseconds>"}}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan",
                           "deliverable valid: exactly the region ids, values in 30 to 20000 ms"],
                    error="largest over the regions of |ln(T1_delivered / T1_true)|",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing=f"all gates and, for every region, error <= ln({1 + TOLERANCE:g})"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier reads the hidden T1 and scans nothing",
    )
