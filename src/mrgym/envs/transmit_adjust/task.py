"""``Scanner/TransmitAdjust``: instances and the verifier.

The job: the second adjustment of every examination. The transmitter is not calibrated, so the flip angle a spin
experiences is the nominal flip angle times a factor, and the factor changes over the object (the transmit field is
not uniform at 3 T). Deliver that factor, actual / nominal flip angle, for each marked region.

A flip angle is not read off an image; it is inferred from how the signal depends on it, through a signal formula.
The content of the task is which measurement makes its formula true on this scanner:
  * the dual-TR formula (actual flip-angle imaging) holds for ONE flip angle per voxel. A slice-selective pulse turns
    the spins by less and less away from the slice centre, a 2D signal is the sum over that profile, and the formula
    then reads low. A non-selective (3D) acquisition has one angle per voxel and costs the scan time of every
    partition;
  * the formula assumes that no transverse magnetisation survives a repetition. With the console's default spoiler
    and RF phase increment that is not so for materials with a long T2, and the factor reads high.
Both are stated in the datasheet (``scanner_info``: slice_profile, afi); neither is visible in a single result. The
sizes are measured by the controls ``afi_2d`` and ``default_spoiling`` (gate table) and in ``tests/test_transmit_adjust.py``.

Deliverable    {"flip_ratio": {"R1": 0.87, "R2": 0.81, ...}}       actual / nominal flip angle per marked region
Pass rule      for every marked region the delivered factor lies within ``tolerance`` (relative) of the mean of the
               hidden factor over the region: |delivered / true - 1| <= 0.04. The tolerance is a constant of the
               family; it is not derived from what any policy achieves.

Why several regions. The transmit gain is one number per session: with one region, or with a uniform transmit field,
one constant answer passes far too often (the window of 8 % is a large part of the range of the gain). The transmit
pattern makes the regions differ, so the five to seven vials of the object make the best constant rare
(``fixed_recipe`` in the gate table) while each tolerance stays wide enough for one scan.

Levels
  1   the method is stated, protocol and formula; the budget pays it once
  2   the method is withheld
  3   the transmit pattern is stronger and the budget no longer pays the level-1 protocol: the measurement has to
      be made cheaper without making it invalid
"""
from __future__ import annotations

import math
import os

import numpy as np

from ...scanner import world as W
from .. import family as F

FAMILY = "transmit_adjust"
ENV_ID = "Scanner/TransmitAdjust-v0"
SUBMIT_DOC = "Deliver, for each marked region, the ratio of the actual to the nominal flip angle (1.0 = calibrated)."
TOLERANCE = 0.04                            # a delivered factor may be off by this fraction of the true one
RATIO_VALID = (0.2, 3.0)                    # anything outside is not a flip-angle ratio
RATIO_BOUND = (0.4, 1.7)                    # told: the factor of every marked region of every level lies inside (tests check it on instances)
OUTCOMES = ("no_region_within", "some_regions_within")      # of a delivered, measured answer that fails (worst first)
# Examined competencies this family exercises (rows of ``mrgym.coverage``).
COMPETENCIES = ("qc.transmit_gain", "parameter.flip_angle", "option.2d_3d", "artefact.b1_shading")
# QA phantom at 3 T with five or six vials on the ring (and a centre vial in half of the objects), received with the
# body coil (the noise density of ``Scanner/T1Transmit``). The transmit adjustment is off by 8 to 30 % either way; the
# transmit pattern is strong (a bowl of 14 to 24 % of the nominal field, more at level 3).
BASE = dict(kind="qa", b0_t=3.0, qa=dict(n_ring=(5, 6)), sigma0=0.173, sigma0_log_sd=0.12, tx_gain_dev=(0.08, 0.30), cf_offset_hz=25.0)
# The protocol the first level states. Non-selective: one flip angle per voxel. Spoiler and RF phase increment: the
# strongest spoiler the console plays at this repetition time and the increment that goes with it (datasheet, afi).
METHOD_SCAN = dict(sequence="afi", excitation="nonselective", tr_ms=25.0, tr_ratio=5, flip_deg=60.0, spoiler_mt_ms_per_m=585.0,
                   rf_phase_increment_deg=45.0, matrix=32, slice_mm=10.0, bandwidth_hz_per_px=500.0)
LEVELS = {
    1: dict(world_spec=dict(BASE, b1_shape_amp=(0.14, 0.24)), budget_s=100.0, hints=("method",),
            summary="the method is stated: a non-selective dual-TR scan with strong spoiling and its formula"),
    2: dict(world_spec=dict(BASE, b1_shape_amp=(0.14, 0.24)), budget_s=100.0,
            summary="the method is withheld: a 2D scan gives a factor that is far too low without a sign of it"),
    3: dict(world_spec=dict(BASE, b1_shape_amp=(0.20, 0.30)), budget_s=65.0,
            summary="the transmit pattern is stronger and the budget does not pay the level-1 protocol"),
}


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    """Episode config. Public: the marked regions (a circle inside every vial: five to seven), the tolerance and the
    bound of the factor."""
    lv = LEVELS[level]
    regions = sorted(W.make_world(seed, lv["world_spec"], layout_only=True)["regions"], key=lambda r: int(r["id"][1:]))
    task = dict(family=FAMILY, regions=F.public_regions(regions), tolerance=TOLERANCE, ratio_bound=list(RATIO_BOUND),
                deliverable={"flip_ratio": {"<region id>": "<number: actual / nominal flip angle>"}})
    if lv.get("hints"):
        task["hints"] = list(lv["hints"])
    return dict(task_id=task_id or "transmitadjust-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed,
                world_spec=lv["world_spec"], budget_s=float(lv["budget_s"]), max_requests=32, task=task)


def hint_text(cfg: dict) -> str:
    """The method, as the paragraph appended to the statement of a level that states it ("" otherwise)."""
    if "method" not in cfg["task"].get("hints", ()):
        return ""
    p = METHOD_SCAN
    return f"""How to do it (this level states the method).
  1. Scan `afi` with excitation {p['excitation']}, tr_ms {p['tr_ms']:g}, tr_ratio {p['tr_ratio']}, flip_deg {p['flip_deg']:g}, spoiler_mt_ms_per_m {p['spoiler_mt_ms_per_m']:g},
     rf_phase_increment_deg {p['rf_phase_increment_deg']:g}, matrix {p['matrix']}, slice_mm {p['slice_mm']:g}, bandwidth_hz_per_px {p['bandwidth_hz_per_px']:g}, and read out the marked regions.
     The scan returns two images.
  2. Per region: r = (complex-mean magnitude in image 2) / (complex-mean magnitude in image 1). n = {p['tr_ratio']} (the tr_ratio).
  3. actual flip angle = arccos((r x n - 1) / (n - r)). Factor = actual flip angle / {p['flip_deg']:g} degrees.
  The formula needs one flip angle per voxel: with a slice-selective excitation it reads far too low.
  It also needs strong spoiling: with the console's default spoiler it can read several percent too high."""


def statement(cfg: dict) -> str:
    t = cfg["task"]
    ids = ", ".join(r["id"] for r in t["regions"])
    hints = hint_text(cfg)
    lo, hi = t["ratio_bound"]
    return f"""# Calibrate the transmitter: the flip-angle factor in each marked region
A test object with vials lies in the field of view. The transmit adjustment of the scanner has not been run.
The flip angle a spin experiences is therefore the nominal flip angle times an unknown factor,
and the factor is not the same everywhere in the object.
Deliver that factor for each marked region ({ids}) with `submit`:
    factor = actual flip angle / nominal flip angle, as the mean over the region (1.0 = calibrated).
Regions (circles; centre and radius in mm from the isocentre; each lies inside one homogeneous material):
{F.region_table(t['regions'])}
In every marked region the factor is between {lo:g} and {hi:g}.
You may use any sequence and any parameters the console accepts; only the delivered values are evaluated.
Nothing else about the object or the state of the system is known in advance.

Passing: for every region the delivered factor is within {100 * t['tolerance']:g} % of the true factor
(between {1 - t['tolerance']:g} x and {1 + t['tolerance']:g} x the true factor).
The scanner's log must show that you measured. An answer in the wrong unit is wrong.""" + ("\n\n" + hints if hints else "")


def submit_schema(cfg: dict) -> dict:
    ids = [r["id"] for r in cfg["task"]["regions"]]
    return {"type": "object", "additionalProperties": False, "required": ["flip_ratio"],
            "properties": {"flip_ratio": {"type": "object", "additionalProperties": False, "required": ids,
                                          "description": "actual / nominal flip angle of every marked region (a ratio, 1.0 = calibrated), by region id",
                                          "properties": {i: {"type": "number", "minimum": RATIO_VALID[0], "maximum": RATIO_VALID[1]} for i in ids}}}}


def check(cfg: dict, answer, exam: dict):
    ids = [r["id"] for r in cfg["task"]["regions"]]
    if not isinstance(answer, dict) or set(answer) != {"flip_ratio"} or not isinstance(answer["flip_ratio"], dict) or set(answer["flip_ratio"]) != set(ids):
        return 'the deliverable is {"flip_ratio": {' + ", ".join(f'"{i}": <number>' for i in ids) + "}}"
    for i in ids:
        v = answer["flip_ratio"][i]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return f"the factor of {i} is not a finite number"
        if not RATIO_VALID[0] <= v <= RATIO_VALID[1]:
            hint = " (a value of this size looks like percent or degrees)" if v > RATIO_VALID[1] else ""
            return (f"the factor of {i} is {v:g}: not a ratio of actual to nominal flip angle, which lies between "
                    f"{RATIO_VALID[0]:g} and {RATIO_VALID[1]:g}{hint}")
    return None


def region_mean(world: dict, field: np.ndarray, region: dict) -> float:
    """Verifier side: mean of a master-grid array over the grid points inside a region's circle."""
    n, d = world["n"], world["extent_mm"] / world["n"]
    pos = (np.arange(n) - n // 2) * d
    mask = (pos[:, None] - region["y_mm"]) ** 2 + (pos[None, :] - region["x_mm"]) ** 2 <= region["radius_mm"] ** 2
    return float(field[mask].mean())


def truth(cfg: dict, world: dict | None = None) -> dict:
    """Verifier side: {region id: actual / nominal flip angle}: the mean over the region of the hidden transmit
    pattern times the hidden transmit gain (``world.b1_at`` is the same product at one grid point)."""
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    gain = world["system"]["tx_gain"]
    return {r["id"]: region_mean(world, world["fields"]["b1"], r) * gain for r in cfg["task"]["regions"]}


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    problem = check(cfg, answer, world.get("console") or {})
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    tol, tr = cfg["task"]["tolerance"], truth(cfg, world)
    miss = {i: float(answer["flip_ratio"][i]) / t - 1.0 for i, t in tr.items()}           # relative error per region
    checks = dict(deliverable_valid=True, **{f"ratio_within_tolerance_{i}": abs(m) <= tol for i, m in miss.items()})
    bad = [f"{i} is off by {100 * m:+.1f} %" for i, m in miss.items() if abs(m) > tol]
    worst = max(abs(m) for m in miss.values())
    return F.result(checks, "; ".join(bad), error=worst, tolerance=tol, per_region={i: round(m, 5) for i, m in miss.items()},
                    distance=dict(regions=len(miss), within=len(miss) - len(bad), worst_relative=worst))


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured answer that fails a check: one of ``OUTCOMES``."""
    return "some_regions_within" if any(v for k, v in score["checks"].items() if k.startswith("ratio_within_tolerance_")) else "no_region_within"


def task_kind(stored: dict | None = None) -> dict:
    """Measurement task or decision task, per level, read from a gate result (default: the stored one), never typed."""
    from ... import gate
    return F.kind_by_open_loop(stored or gate.load(ENV_ID))


def card() -> dict:
    from ... import learn
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    kind = task_kind()
    return dict(
        id=ENV_ID, title="Flip-angle factor in marked regions (the transmit adjustment)", kind="candidate",
        given="A test object with vials; a scanner whose transmitter is not calibrated and whose transmit field is not uniform; "
              "a scan-time budget",
        deliver="For each marked region, the ratio of the actual to the nominal flip angle",
        scored_by=f"The hidden transmit gain and transmit pattern: each delivered factor within {100 * TOLERANCE:g} % of their product, "
                  "averaged over the region. The verifier scans nothing",
        summary="The second adjustment of every examination, as a number per region. The textbook dual-TR formula returns the "
                "flip angle only where every voxel has one flip angle and the spoiling is strong: on this scanner the 2D "
                "(slice-selective) version reads low and the 3D version with the default spoiling reads high in materials "
                "with a long T2 (controls `afi_2d` and `default_spoiling`). Which measurement is valid is stated in the "
                "datasheet and is the content of the task. " + kind["sentence"],
        task_kind=kind["kind"], task_kind_why=kind["evidence"], task_kind_levels=kind["levels"], competencies=list(COMPETENCIES),
        world=dict(phantom="QA phantom (a disc of gel with 5 to 7 vials), 3 T, body-coil receive", sequences_useful=["afi"],
                   agent_side_formulas="r = S2 / S1 = (1 + n cos a) / (n + cos a) for TR much shorter than T1 and ideal spoiling, "
                                       "n = TR2 / TR1; factor = a / nominal angle"),
        hidden=[f"the transmit-gain error ({100 * BASE['tx_gain_dev'][0]:g} to {100 * BASE['tx_gain_dev'][1]:g} % either way)",
                "the transmit pattern (its size is drawn per instance; larger at level 3)",
                "T1, T2 and diffusion coefficient of every material (they set how well a spoiler works)", "the receive profile, the field",
                "the noise level", "the 128-bit world seed"],
        told=["region centres and radii", "a bound of the factor", "the tolerance", "the method (level 1)",
              "the instrument datasheet (slice profile, spoiling)", "realised protocol, timing and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        practice_levels=[],
        learnability=learn.stored(ENV_ID),
        limits=dict(max_scanner_requests=32, max_tool_calls=40),
        deliverable={"tool": "submit", "format": {"flip_ratio": {"<region id>": "<actual / nominal flip angle>"}}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan",
                           f"deliverable valid: exactly the region ids, values in {RATIO_VALID[0]:g} to {RATIO_VALID[1]:g}"],
                    error="largest over the regions of |delivered / true - 1|",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing=f"all gates and, for every region, error <= {TOLERANCE:g}"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier reads the hidden transmit field and scans nothing",
    )
