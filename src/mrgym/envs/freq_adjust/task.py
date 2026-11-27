"""``Scanner/FrequencyAdjust``: instances and the verifier.

The job: the first adjustment of every examination. The scanner's centre frequency is not on the water of the
object, and the static field is not uniform, so the water of each marked region is off resonance by its own amount.
Deliver that amount per region, in Hz: the frequency of the water there minus the scanner's centre frequency.

Nothing the console prints depends on it. It is in the data: a gradient echo carries the phase 2 pi x offset x TE on
top of the phase of the receive coil, which is unknown and differs from place to place, so one echo says nothing and
the difference of two echoes gives the offset, up to a whole number of turns: an echo-time difference dTE measures
the offset modulo 1 / dTE. Which difference is safe depends on how far off the scanner can be.

Deliverable    {"offset_hz": {"R1": -41.5, "R2": 12.0}}          one offset per marked region, Hz, signed
               level 3: {"offset_hz": {"R1": -41.5, "R2": "oil", "R3": 12.0}}: a region of oil is named, not measured
Pass rule      for every marked region that holds water the delivered offset lies within ``tolerance_hz`` of the mean
               of (field + centre-frequency error) over the region on the hidden world; every marked region that
               holds oil is delivered as the word "oil". The tolerance is a constant of the family in Hz.

Why several regions. One offset in a known range is one number: the best constant passes (2 x tolerance) / range of
the instances. The field differs between the regions by a few to tens of Hz, so two or three regions make that
product smaller while each tolerance stays wide for a careful measurement; the same scans serve every region.

Levels
  1   the method is stated; the offset is small enough that the stated echo-time difference does not wrap
  2   the method is withheld and the centre frequency can be off by more than half of 1 / (that difference)
  3   as level 2, and some of the marked vials hold oil. Oil resonates 3.4 ppm below water: a frequency read there is
      not the water frequency, and such a region has to be recognised and named
"""
from __future__ import annotations

import math
import os

import numpy as np

from ...scanner import world as W
from .. import family as F

FAMILY = "freq_adjust"
ENV_ID = "Scanner/FrequencyAdjust-v0"
SUBMIT_DOC = "Deliver, for each marked region, the off-resonance of its water in Hz (water frequency minus the scanner's centre frequency)."
TOLERANCE_HZ = 3.0                          # a delivered offset may be off by this much
OFFSET_VALID_HZ = (-5000.0, 5000.0)         # anything outside is not an offset of this scanner in Hz
OIL = "oil"                                 # what is delivered for a region that holds oil (level 3)
FAT_PPM = 3.4                               # told (datasheet): the main fat line lies this far below water
OUTCOMES = ("species_wrong", "no_offset_within", "some_offsets_within")   # of a delivered, measured answer that fails (worst first)
# Examined competencies this family exercises (rows of ``mrgym.coverage``).
COMPETENCIES = ("qc.centre_frequency", "parameter.te", "artefact.chemical_shift")
# QA phantom at 3 T. ``b0_hz``: size of the linear and saddle terms of the field over 240 mm (an unshimmed object);
# ``sigma0``: twice the noise density of the default exam; ``cf_offset_hz``: the centre frequency is off by up to this
# much either way. ``offset_bound_hz`` is told: it is larger
# than the offset of any marked region the level can draw (``tests/test_freq_adjust.py`` checks it on instances).
BASE = dict(kind="qa", b0_t=3.0, b0_hz=40.0, sigma0=0.06)
METHOD_SCAN = dict(sequence="spgr", tr_ms=30.0, flip_deg=15.0, matrix=64, bandwidth_hz_per_px=1000.0, n_echoes=2, echo_spacing_ms=2.0)
LEVELS = {
    1: dict(world_spec=dict(BASE, cf_offset_hz=100.0), n_regions=2, offset_bound_hz=200.0, budget_s=30.0, hints=("method",),
            summary="the method is stated; offsets up to 200 Hz: two echoes 2 ms apart do not wrap"),
    2: dict(world_spec=dict(BASE, cf_offset_hz=500.0), n_regions=3, offset_bound_hz=600.0, budget_s=40.0,
            summary="the method is withheld; offsets up to 600 Hz: the echo-time difference of level 1 wraps"),
    3: dict(world_spec=dict(BASE, cf_offset_hz=500.0, qa=dict(n_oil=2)), n_regions=3, n_oil=(1, 2), offset_bound_hz=600.0, budget_s=40.0,
            summary="as level 2, and one or two of the three marked vials hold oil, 3.4 ppm below water: they have to be named"),
}


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    """Episode config. Public: the marked regions (circles inside vials), the bound of the offset, the tolerance.
    Level 3 marks one or two vials of oil and at least one of water and gives the marked regions new ids in a drawn
    order, so neither an id nor the set of ids says anything about the content."""
    lv = LEVELS[level]
    layout = W.make_world(seed, lv["world_spec"], layout_only=True)
    regions = sorted(layout["regions"], key=lambda r: int(r["id"][1:]))
    if lv.get("n_oil"):
        rng = W.rng_for(seed, "task")
        k = int(rng.integers(lv["n_oil"][0], lv["n_oil"][1] + 1))
        oil = [r for r in regions if layout["tissues"][r["cls"]]["fat"]]
        water = [r for r in regions if not layout["tissues"][r["cls"]]["fat"]]
        picked = oil[:k] + water[:lv["n_regions"] - k]
        marked = [dict(picked[int(j)], id=f"R{n + 1}") for n, j in enumerate(rng.permutation(len(picked)))]
    else:
        marked = regions[:lv["n_regions"]]
    task = dict(family=FAMILY, regions=F.public_regions(marked), tolerance_hz=TOLERANCE_HZ, offset_bound_hz=lv["offset_bound_hz"],
                oil_possible=bool(lv.get("n_oil")),
                deliverable={"offset_hz": {"<region id>": "<number, Hz>" + (f', or "{OIL}"' if lv.get("n_oil") else "")}})
    if lv.get("hints"):
        task["hints"] = list(lv["hints"])
    return dict(task_id=task_id or "freqadjust-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed,
                world_spec=lv["world_spec"], budget_s=float(lv["budget_s"]), max_requests=32, task=task)


def hint_text(cfg: dict) -> str:
    """The method, as the paragraph appended to the statement of a level that states it ("" otherwise)."""
    if "method" not in cfg["task"].get("hints", ()):
        return ""
    p = METHOD_SCAN
    wrap = 0.5e3 / p["echo_spacing_ms"]
    return f"""How to do it (this level states the method).
  1. Scan `spgr` with two echoes {p['echo_spacing_ms']:g} ms apart (for example tr_ms {p['tr_ms']:g}, flip_deg {p['flip_deg']:g}, matrix {p['matrix']},
     bandwidth_hz_per_px {p['bandwidth_hz_per_px']:g}, n_echoes {p['n_echoes']}, echo_spacing_ms {p['echo_spacing_ms']:g}) and read out the marked regions.
  2. Per region: d = phase_rad of echo 2 - phase_rad of echo 1. Add or subtract 2 pi until d lies between -pi and pi.
  3. offset_hz = d / (2 pi x {p['echo_spacing_ms'] / 1000:g} s).
  With {p['echo_spacing_ms']:g} ms between the echoes this is right for offsets up to {wrap:g} Hz either way, which is enough here."""


def statement(cfg: dict) -> str:
    t = cfg["task"]
    ids = ", ".join(r["id"] for r in t["regions"])
    hints = hint_text(cfg)
    if t["oil_possible"]:
        content = (f'Some of the marked regions hold oil instead of water; at least one holds water.\n'
                   f'A region of oil cannot give the water frequency: deliver the word "{OIL}" for it instead of a number.')
        passing = (f'Passing: for every region of water the delivered offset is within {t["tolerance_hz"]:g} Hz of the mean offset of the water over the region,\n'
                   f'and every region of oil is delivered as "{OIL}".')
    else:
        content = "Every marked region holds water."
        passing = f'Passing: for every region the delivered offset is within {t["tolerance_hz"]:g} Hz of the mean offset of the water over the region.'
    return f"""# Find the off-resonance of the water in each marked region
A test object with vials lies in the field of view. The frequency adjustment of the scanner has not been run.
The scanner's centre frequency is therefore not the frequency of the water, and the static field is not uniform:
the water of each region is off resonance by its own amount.
Deliver that amount for each marked region ({ids}) with `submit`, in Hz:
    offset = frequency of the water in the region - centre frequency of the scanner.
A positive offset means the water precesses faster than the scanner's reference.
Regions (circles; centre and radius in mm from the isocentre; each lies inside one vial):
{F.region_table(t['regions'])}
{content}
At every marked region the offset of water is between -{t['offset_bound_hz']:g} and +{t['offset_bound_hz']:g} Hz.
You may use any sequence and any parameters the console accepts; only the delivered values are evaluated.
Nothing else about the object or the state of the system is known in advance.

{passing}
The scanner's log must show that you measured. An answer in the wrong unit is wrong.""" + ("\n\n" + hints if hints else "")


def submit_schema(cfg: dict) -> dict:
    t = cfg["task"]
    ids = [r["id"] for r in t["regions"]]
    number = {"type": "number", "minimum": OFFSET_VALID_HZ[0], "maximum": OFFSET_VALID_HZ[1]}
    value = {"anyOf": [number, {"type": "string", "enum": [OIL]}]} if t["oil_possible"] else number
    doc = "off-resonance of the water of every marked region in Hz (water minus scanner centre frequency), by region id"
    if t["oil_possible"]:
        doc += f'; the word "{OIL}" for a region that holds oil'
    return {"type": "object", "additionalProperties": False, "required": ["offset_hz"],
            "properties": {"offset_hz": {"type": "object", "additionalProperties": False, "required": ids, "description": doc,
                                         "properties": {i: dict(value) for i in ids}}}}


def check(cfg: dict, answer, exam: dict):
    t = cfg["task"]
    ids = [r["id"] for r in t["regions"]]
    what = '<number, or "oil">' if t["oil_possible"] else "<number>"
    if not isinstance(answer, dict) or set(answer) != {"offset_hz"} or not isinstance(answer["offset_hz"], dict) or set(answer["offset_hz"]) != set(ids):
        return 'the deliverable is {"offset_hz": {' + ", ".join(f'"{i}": {what}' for i in ids) + "}}"
    for i in ids:
        v = answer["offset_hz"][i]
        if isinstance(v, str):
            if v == OIL and t["oil_possible"]:
                continue
            if v == OIL:
                return f'the offset of {i} is "{OIL}": every marked region of this task holds water; deliver a number in Hz'
            return f"the offset of {i} is not a finite number" + (f' (or the word "{OIL}")' if t["oil_possible"] else "")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return f"the offset of {i} is not a finite number" + (f' (or the word "{OIL}")' if t["oil_possible"] else "")
        if not OFFSET_VALID_HZ[0] <= v <= OFFSET_VALID_HZ[1]:
            return (f"the offset of {i} is {v:g}: not an off-resonance in Hz, which lies between {OFFSET_VALID_HZ[0]:g} and "
                    f"{OFFSET_VALID_HZ[1]:g} (a value of this size looks like another unit)")
    return None


def region_mean(world: dict, field: np.ndarray, region: dict) -> float:
    """Verifier side: mean of a master-grid array over the grid points inside a region's circle."""
    n, d = world["n"], world["extent_mm"] / world["n"]
    pos = (np.arange(n) - n // 2) * d
    mask = (pos[:, None] - region["y_mm"]) ** 2 + (pos[None, :] - region["x_mm"]) ** 2 <= region["radius_mm"] ** 2
    return float(field[mask].mean())


def truth(cfg: dict, world: dict | None = None) -> dict:
    """Verifier side: {region id: dict(offset_hz, oil)}. ``offset_hz`` is the mean over the region of the hidden field
    plus the hidden centre-frequency error: the off-resonance of water at that place (for a region of oil, of the
    water that would be there). ``oil``: the vial the region lies in holds fat (the vial is found by the region's
    centre: a marked region is the circle of one vial of the world, under an id of the task's own)."""
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    cls = {(r["y_mm"], r["x_mm"]): r["cls"] for r in world["regions"]}
    off = world["system"]["cf_offset_hz"]
    return {r["id"]: dict(offset_hz=region_mean(world, world["fields"]["b0_hz"], r) + off,
                          oil=bool(world["tissues"][cls[(r["y_mm"], r["x_mm"])]]["fat"]))
            for r in cfg["task"]["regions"]}


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    problem = check(cfg, answer, world.get("console") or {})
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    tol, tr = cfg["task"]["tolerance_hz"], truth(cfg, world)
    checks, notes, miss, species_wrong = dict(deliverable_valid=True), [], {}, []
    for i, t in tr.items():
        v = answer["offset_hz"][i]
        if t["oil"]:                                                     # a region of oil must be named
            checks[f"oil_named_{i}"] = v == OIL
            if v != OIL:
                species_wrong.append(i)
                notes.append(f"{i} holds oil: {float(v):g} Hz is not a water frequency ({float(v) - t['offset_hz']:+.0f} Hz from the water offset there)")
        elif v == OIL:                                                   # a region of water named as oil
            checks[f"offset_within_tolerance_{i}"] = False
            species_wrong.append(i)
            notes.append(f"{i} holds water")
        else:
            miss[i] = float(v) - t["offset_hz"]
            checks[f"offset_within_tolerance_{i}"] = abs(miss[i]) <= tol
            if abs(miss[i]) > tol:
                notes.append(f"{i} is off by {miss[i]:+.1f} Hz")
    worst = max((abs(m) for m in miss.values()), default=None)
    return F.result(checks, "; ".join(notes), error=worst, tolerance_hz=tol, per_region={i: round(m, 3) for i, m in miss.items()},
                    species_wrong=species_wrong,
                    distance=dict(regions=len(tr), right=sum(v for k, v in checks.items() if k != "deliverable_valid"), worst_hz=worst))


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured answer that fails a check: one of ``OUTCOMES``. ``species_wrong``: a
    region of oil was delivered as a number or a region of water as oil; otherwise by the offsets."""
    checks = score["checks"]
    if score.get("species_wrong") or any(not v for k, v in checks.items() if k.startswith("oil_named_")):
        return "species_wrong"
    return "some_offsets_within" if any(v for k, v in checks.items() if k.startswith("offset_within_tolerance_")) else "no_offset_within"


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
        id=ENV_ID, title="Off-resonance of the water in marked regions (the frequency adjustment)", kind="candidate",
        given="A test object with vials in a hidden, non-uniform field; a scanner whose centre frequency is off; a scan-time budget",
        deliver="For each marked region, the off-resonance of its water in Hz; at level 3 the word \"oil\" for a region that holds oil",
        scored_by=f"The hidden field and centre-frequency error: each delivered offset within {TOLERANCE_HZ:g} Hz of their mean over the "
                  "region; every region of oil named. The verifier scans nothing",
        summary="The first adjustment of every examination, as a number. A gradient echo carries the phase 2 pi x offset x TE "
                "on top of the unknown phase of the receive coil, so the offset is in the phase difference of two echoes, "
                "modulo 1 / (the echo-time difference). A short difference covers a wide range and is less precise, a long one "
                "is more precise and wraps; fat lies 3.4 ppm below water and a frequency read on it is not the water frequency. "
                + kind["sentence"],
        task_kind=kind["kind"], task_kind_why=kind["evidence"], task_kind_levels=kind["levels"], competencies=list(COMPETENCIES),
        world=dict(phantom="QA phantom (a disc of gel with vials; at level 3 two vials of oil), 3 T",
                   sequences_useful=["spgr"],
                   agent_side_formulas="offset = (phase of echo 2 - phase of echo 1) / (2 pi x echo-time difference), modulo "
                                       "1 / (echo-time difference); fat main line 3.4 ppm x 42.58 MHz/T x B0 below water"),
        hidden=["the error of the centre frequency (up to " + ", ".join(f"{d['world_spec']['cf_offset_hz']:g} Hz at level {lv}" for lv, d in LEVELS.items()) + ")",
                "the field over the object",
                "which marked vials hold oil (level 3)", "the phase of the receive coil", "tissue values, transmit gain, noise level",
                "the 128-bit world seed"],
        told=["region centres and radii", "the sign convention", "a bound of the offset", "that some regions hold oil (level 3)",
              "the tolerance", "the method (level 1)", "the instrument datasheet", "realised protocol, timing and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        practice_levels=[],
        learnability=learn.stored(ENV_ID),
        limits=dict(max_scanner_requests=32, max_tool_calls=40),
        deliverable={"tool": "submit", "format": {"offset_hz": {"<region id>": "<Hz>, level 3: or \"oil\""}}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan",
                           f"deliverable valid: exactly the region ids, numbers in {OFFSET_VALID_HZ[0]:g} to {OFFSET_VALID_HZ[1]:g} Hz (level 3: or \"oil\")"],
                    error="largest over the regions of water of |delivered - true offset|, Hz",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing=f"all gates; every region of water within {TOLERANCE_HZ:g} Hz; every region of oil delivered as \"oil\""),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier reads the hidden field and scans nothing",
    )
