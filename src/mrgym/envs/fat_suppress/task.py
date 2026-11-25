"""``Scanner/FatSuppress``: instances and the verifier.

The job: a fat-suppressed spin-echo image of each marked, fat-containing region. Two methods are on the console.
Spectral fat saturation costs no water signal, but it works only where the fat resonance lies inside the band of the
saturation pulse (the field must be homogeneous there and the frequency adjustment right) and only as well as the
pulse is efficient. Inversion recovery with the inversion time at the null of the fat (STIR) does not care about
the field, but it costs a large part of the water signal, and the null depends on the T1 of that fat, on the repetition time
and on how complete the inversion is. Which method is right is decided region by region by a measurement; where it
is inversion recovery, the inversion time has to be measured too.

Deliverable    {"R2": {"method": "fat_sat"}, "R5": {"method": "stir", "ti_ms": 231.0}, ...}       one entry per region
Pass rule      for every marked region, on noise-free scans of the target image with the delivered method:
               1  suppressed: the region's signal is at most ``fat_max`` of its signal without suppression
               2  no water signal given away: inversion recovery passes only where fat saturation is not enough, that
                  is where fat saturation leaves more than ``fat_sat_sufficient`` (< ``fat_max``) of the signal. Between
                  the two numbers both methods pass, so a region on the edge never decides an episode.

Why several regions. With one region the answer is one of two methods, and the commoner one passes without a scan.
Each marked region holds its own fat (its own T1) at its own place in the field, so the methods and the inversion
times of the regions are separate coordinates of the deliverable; the same probe scans serve all of them.

The hidden world is ``W.make_world(seed, world_spec)`` and nothing else: the verifier, ``mrgym verify``, the HTTP
service and an exported task rebuild exactly the world the episode was played on. Two keys of the level specs do the
work of this family (both are drawn by the world generator of engine 0.2.0):
  ``qa.oil_t1_jitter``  j: the T1 of every fat-containing vial is the table value times an independent factor drawn
                        uniformly from 1 - j to 1 + j, so no constant inversion time nulls every fat
  ``fatsat_bw_hz``      (low, high): the bandwidth of the saturation pulse at 3 T is drawn uniformly from this range
                        (absent: 1.3 ppm, 166 Hz at 3 T), so a field map alone does not say where the pulse works
"""
from __future__ import annotations

import math
import os

import numpy as np

from ...scanner import console
from ...scanner import world as W
from .. import family as F

FAMILY = "fat_suppress"
SUBMIT_DOC = ("Deliver, for each marked region, the method of the fat-suppressed target image of that region: fat_sat, or "
              "stir with the inversion time in milliseconds.")
GEOMETRY = dict(matrix=128, bandwidth_hz_per_px=300.0)
METHODS = ("fat_sat", "stir")
FAT_MAX, FAT_SAT_SUFFICIENT = 0.05, 0.03
# examined competencies this family exercises (ids of ``mrgym.coverage.COMPETENCIES``)
COMPETENCIES = ("option.suppression", "parameter.ti", "parameter.tr", "artefact.chemical_shift", "qc.centre_frequency")
QA = dict(n_ring=(5, 6), n_oil=5, roi_radius_mm=9.0, oil_t1_jitter=0.35)
BASE = dict(kind="qa", b0_t=3.0, b0_hz=500.0, cf_offset_hz=150.0)
LEVELS = {
    1: dict(world_spec=dict(BASE, qa=dict(QA), fatsat_eff=(0.975, 0.998)), tr_ms=(1500.0, 3000.0), budget_s=120.0,
            summary="an efficient saturation pulse and a nearly complete inversion: the field at a region decides whether "
                    "fat saturation works there; the null follows from the fat's T1 and the repetition time"),
    2: dict(world_spec=dict(BASE, qa=dict(QA), fatsat_eff=(0.975, 0.998), fatsat_bw_hz=(60.0, 330.0)), tr_ms=(1500.0, 3000.0), budget_s=120.0,
            summary="the band of the saturation pulse is not the nominal one, which no field map shows: fat saturation "
                    "has to be tried, not predicted"),
    3: dict(world_spec=dict(BASE, qa=dict(QA), fatsat_eff=(0.975, 0.998), fatsat_bw_hz=(60.0, 330.0), inv_eff=(0.70, 0.92),
                            tx_gain_dev=(0.08, 0.20), sigma0=0.06),
            tr_ms=(1500.0, 3000.0), budget_s=120.0,
            summary="as level 2 with an incomplete inversion, a miscalibrated transmitter and more noise: the null moves "
                    "with the inversion efficiency"),
}


def target_protocol(task: dict, entry: dict) -> dict:
    tr = task["target_scan"]["tr_ms"]
    if entry["method"] == "fat_sat":
        return dict(GEOMETRY, sequence="se", tr_ms=tr, fat_sat=True)
    return dict(GEOMETRY, sequence="ir", tr_ms=tr, ti_ms=entry["ti_ms"])


# ------------------------------------------------------------------------------------------------ instances
def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    lv = LEVELS[level]
    rng = W.rng_for(seed, "task")
    tr = float(np.round(rng.uniform(*lv["tr_ms"]) / 10.0) * 10.0)
    layout = W.make_world(seed, lv["world_spec"], layout_only=True)
    targets = sorted((r for r in layout["regions"] if layout["tissues"][r["cls"]]["fat"]), key=lambda r: int(r["id"][1:]))
    task = dict(family=FAMILY, level=level, regions=F.public_regions(targets), fat_max=FAT_MAX, fat_sat_sufficient=FAT_SAT_SUFFICIENT,
                target_scan=dict(GEOMETRY, sequence="se", tr_ms=tr, te_ms="min", averages=1, slice_mm=5.0),
                methods=dict(fat_sat="the target scan with fat_sat true", stir="sequence ir with the delivered ti_ms, otherwise the target scan"),
                deliverable={"<region id>": {"method": "fat_sat | stir", "ti_ms": "<number, milliseconds; stir only>"}})
    return dict(task_id=task_id or "fatsuppress-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed,
                world_spec=lv["world_spec"], budget_s=float(lv["budget_s"]), max_requests=48, task=task)


def statement(cfg: dict) -> str:
    t = cfg["task"]
    s = t["target_scan"]
    ids = ", ".join(r["id"] for r in t["regions"])
    return f"""# Suppress the fat in each marked region and keep the water signal
A test object with vials of unknown materials lies in the field of
view. Each marked region lies inside a vial that contains fat, a different fat in every vial. For each of them the
examination needs a spin-echo image in which the fat of that region is suppressed:
    tr_ms {s['tr_ms']:g}, te_ms at the console minimum, matrix {s['matrix']}, bandwidth_hz_per_px {s['bandwidth_hz_per_px']:g}, field of view at the
    console default, 1 average, slice 5 mm,
    either  sequence se with fat_sat true                (method fat_sat: spectral fat saturation)
    or      sequence ir with an inversion time ti_ms     (method stir: inversion recovery at the null of the fat).
Regions (circles; centre and radius in mm from the isocentre; each lies inside one homogeneous material):
{F.region_table(t['regions'])}
Deliver one method per region ({ids}) with `submit`, and the inversion time where the method is stir. The target scans
are not run in this episode: the scanner evaluates what you deliver. {F.OWN_SCANS} {F.NOT_CHARGED}
Nothing about the materials, the field or the state of the system is known in advance.

Passing, for every region:
1. Suppressed. In the target scan with the method you delivered for it, its signal is at most {100 * t['fat_max']:g} % of its signal in
   the same spin echo without any suppression.
2. No water signal given away. Fat saturation leaves the water signal alone; inversion recovery removes a large part of it.
   stir therefore passes for a region only where fat saturation is not enough: where fat saturation leaves more than
   {100 * t['fat_sat_sufficient']:g} % of the region's signal. (Between {100 * t['fat_sat_sufficient']:g} % and {100 * t['fat_max']:g} % both methods pass.)
The scanner's log must show that you measured. An answer in the wrong unit is wrong."""


def submit_schema(cfg: dict) -> dict:
    lo, hi = console.SEQUENCES["ir"]["ti_ms"][:2]
    entry = {"type": "object", "additionalProperties": False, "required": ["method"],
             "properties": {"method": {"type": "string", "enum": list(METHODS)},
                            "ti_ms": {"type": "number", "minimum": lo, "maximum": hi,
                                      "description": "inversion time of the target scan, milliseconds (method stir only)"}}}
    ids = [r["id"] for r in cfg["task"]["regions"]]
    return {"type": "object", "additionalProperties": False, "required": ids,
            "properties": {i: dict(entry, description=f"fat suppression of the target image of region {i}") for i in ids}}


def check(cfg: dict, answer, exam: dict):
    ids = [r["id"] for r in cfg["task"]["regions"]]
    form = ('the deliverable is {' + ", ".join(f'"{i}": ...' for i in ids) + '} with {"method": "fat_sat"} or '
            '{"method": "stir", "ti_ms": <number>} for every region')
    if not isinstance(answer, dict) or set(answer) != set(ids):
        return form
    for i in ids:
        e = answer[i]
        if not isinstance(e, dict) or e.get("method") not in METHODS or not set(e) <= {"method", "ti_ms"}:
            return form
        if e["method"] == "fat_sat":
            if e.get("ti_ms") is not None:
                return f"{i}: fat_sat has no inversion time"
            continue
        ti = e.get("ti_ms")
        if isinstance(ti, bool) or not isinstance(ti, (int, float)) or not math.isfinite(ti):
            return f"{i}: stir needs ti_ms, a number in milliseconds"
        pl = console.plan(target_protocol(cfg["task"], e), exam)
        if not pl["accept"]:
            return f"the console rejects the target scan of {i}: " + "; ".join(v["message"] for v in pl["violations"])
    return None


# ------------------------------------------------------------------------------------------------ the verifier
def residuals(cfg: dict, answer: dict, world: dict, engine) -> dict:
    """Verifier side, noise-free: per region the signal with fat saturation and with the delivered method, both
    relative to the plain spin echo of the target geometry. {region id: dict(fat_sat, delivered)}."""
    t, exam = cfg["task"], world["console"]
    regions = t["regions"]
    scan = lambda protocol: {k: abs(v) for k, v in F.region_complex(engine, console.plan(protocol, exam)["realised"], regions).items()}   # noqa: E731
    plain = scan(dict(GEOMETRY, sequence="se", tr_ms=t["target_scan"]["tr_ms"]))
    sat = scan(target_protocol(t, dict(method="fat_sat")))
    out = {}
    for i, e in answer.items():
        fs = sat[i] / max(plain[i], 1e-30)
        got = fs if e["method"] == "fat_sat" else scan(target_protocol(t, e))[i] / max(plain[i], 1e-30)
        out[i] = dict(fat_sat=float(fs), delivered=float(got))
    return out


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    from ...scanner import kspace
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    engine = engine or kspace.Engine(world)
    problem = check(cfg, answer, world["console"])
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    t = cfg["task"]
    res = residuals(cfg, answer, world, engine)
    checks, notes = dict(deliverable_valid=True), []
    for i, r in res.items():
        checks[f"suppressed_{i}"] = r["delivered"] <= t["fat_max"]
        checks[f"method_{i}"] = answer[i]["method"] == "fat_sat" or r["fat_sat"] > t["fat_sat_sufficient"]
        if not checks[f"suppressed_{i}"]:
            notes.append(f"{i} keeps {100 * r['delivered']:.1f} % with {answer[i]['method']}")
        if not checks[f"method_{i}"]:
            notes.append(f"{i}: fat saturation is enough ({100 * r['fat_sat']:.1f} % left)")
    return F.result(checks, "; ".join(notes), error=round(max(r["delivered"] for r in res.values()), 5), residual=res,
                    fat_max=t["fat_max"], methods={i: e["method"] for i, e in answer.items()})


def task_kind(stored: dict | None = None) -> dict:
    """Measurement task or decision task, per level, by the rule of this package: a level on which a fixed list of
    scans with a fixed estimator (the ``open_loop`` control) passes within 25 points of the reference is a measurement
    task, otherwise a decision task. Read from a gate result (default: the stored one), never typed."""
    from ... import gate
    return F.kind_by_open_loop(stored or gate.load("Scanner/FatSuppress-v0"))


def card() -> dict:
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    kind = task_kind()
    return dict(
        id="Scanner/FatSuppress-v0", title="Suppress the fat in each marked region and keep the water signal", kind="candidate",
        given="A test object with several fat-containing vials (a different fat in each) in a strongly inhomogeneous field; "
              "the scanner; a scan-time budget for probing",
        deliver="For each marked region the method of its fat-suppressed spin-echo image: spectral fat saturation, or "
                "inversion recovery with the inversion time",
        scored_by="Noise-free scans of the target image on the hidden object: every region must be suppressed by its "
                  "method, and inversion recovery passes only where fat saturation is not enough",
        summary="The daily choice between spectral fat saturation and STIR. Fat saturation keeps the water signal but "
                "fails where the field is inhomogeneous, where the frequency adjustment is off and where the pulse is "
                "weak; STIR works everywhere but removes a large part of the water signal, and its inversion time depends on "
                "the T1 of that fat, on the repetition time and on how complete the inversion is. Neither can be "
                "looked up: fat saturation has to be tried on this object, and the null has to be measured. " + kind["sentence"],
        task_kind=kind["kind"], task_kind_why=kind["evidence"], task_kind_levels=kind["levels"], competencies=list(COMPETENCIES),
        world=dict(phantom="QA phantom (a disc of gel with vials, four of them fat with their own T1), 3 T",
                   sequences_useful=["se", "ir", "spgr"],
                   agent_side_formulas="S(TI) = a - b exp(-TI / T1) at a fixed repetition time; the sign of the signal is in "
                                       "the phase image; fat saturation acts where the fat resonance lies inside the pulse band"),
        hidden=["T1 of the fat in every vial", "the off-resonance field and the error of the frequency adjustment",
                "efficiency and bandwidth of the saturation pulse", "inversion efficiency", "transmit gain and B1 pattern",
                "noise level", "the 128-bit world seed"],
        told=["the target scan and its repetition time", "the regions, and that each contains fat", "both pass thresholds",
              "the instrument datasheet", "realised protocol, timing and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        limits=dict(max_scanner_requests=48, max_tool_calls=64),
        deliverable={"tool": "submit", "format": {"<region id>": {"method": "fat_sat | stir", "ti_ms": "<milliseconds, stir only>"}}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan",
                           "the console accepts the target scan of every region"],
                    error="largest over the regions of: signal in the target scan with the delivered method / signal without suppression (noise-free)",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing=f"all gates and, for every region, error <= {100 * FAT_MAX:g} %, and stir only where fat saturation "
                            f"leaves more than {100 * FAT_SAT_SUFFICIENT:g} %"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier's scans are noise-free",
    )
