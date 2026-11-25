"""``Scanner/ArtefactClinic``: instances and the verifier.

The job. An examination of a test object was run with the protocol on the card, and a reader marked findings on its
image. Each finding is either made by the acquisition (one or two of the causes in ``catalogue``) or is the object
itself. The agent runs priced differential experiments (other bandwidth, echo time, matrix, phase field of view,
directions ...), names the cause of every finding and delivers one protocol that images all of them cleanly inside
the examination's requirements (scan time, pixel size, SNR, locked weighting and directions).

Deliverable    {"causes": {"F1": ["chemical_shift_displacement"], "F2": [], "F3": [...]}, "protocol": {...}}
               one list of causes per finding (an empty list: the finding is the object; nothing to correct)
Pass rule      every finding is diagnosed exactly; and on a noise-free re-scan of the delivered protocol on the hidden
               object every finding is clean against its counterfactual twins (``metrics``); and the protocol keeps
               the requirements: scan time, pixel size, SNR in the reference region, locked parameters, TE window.

Why it cannot be answered from the card. The complaint protocol arms several mechanisms at once; which finding
carries which is decided by the hidden object (what the vial next to it holds, what lies beside the object one
phase field of view away, how strong the edge is). The requirements leave room for the remedies that are needed and
not for all of them, so the diagnosis decides the protocol.

Level 3 (``generator.build_constructed``). The cause of every finding is drawn first and the object is built to
realise it; where a finding lies and what the sheet says are drawn independently of the causes. The findings of an
episode look alike (all darker or all brighter than the gel) and hold both artefact causes of that look. Two
single-change experiments tell the three possibilities of a look apart, the two looks need different ones, and the
budget is below four times the cost that every scan has (overhead and dummy cycles). Two remedies have a dose that
an experiment measures, and the SNR floor is a share of the best clean protocol's, so a dose that is too strong
fails as one that is too weak does.
What level 3 is not: a decision task. The experiments of the two looks do not disturb each other, so one scan can
carry a change for each (half the slice, phase oversampling and one average at once); a fixed list of three such
scans with a fixed estimator diagnoses both looks (``controls._several``, the open-loop control).
"""
from __future__ import annotations

import copy
import functools
import json
import os

from ...scanner import console
from ...scanner import world as W
from .. import family as F
from .. import scanner_env
from . import bank
from . import catalogue as C
from . import generator as G
from . import metrics as M

FAMILY = "artefact_clinic"
SUBMIT_DOC = ("Deliver the diagnosis and the corrected protocol: for each marked finding the list of its causes (an empty "
              "list when the finding is the object itself), and one protocol for the examination.")
MAX_ATTEMPTS, ATTEMPTS_PER_DRAW = 64, 4
MAX_CAUSES = 2
_SLOW = dict(tr_ms=(100.0, 220.0), flip_deg=(60.0, 85.0), averages=(6, 8), scan_s=(100.0, 170.0), grid_n=640, matrix=(96, 104, 112))
_FAST = dict(tr_ms=(13.6, 15.4), flip_deg=(25.0, 40.0), averages=(7,), scan_s=(8.0, 13.0), grid_n=320, matrix=(64, 68, 72), te_min_ms=4.0,
             phase_fov=(0.49, 0.52), slice_mm=8.0)
_COMMON = dict(field_t=(1.5, 3.0), sigma0=0.0004, time_slack=(1.0, 1.12), snr_slack=(0.6, 0.85), causes=C.BASE, protocol=_SLOW, n_findings=4)
LEVELS = {
    1: dict(_COMMON, arm_all=False, pairs=False, benign=True, never=("wrap",), budget_factor=1.5,
            summary="four findings, each with at most one cause from a small set: the one fat mechanism the protocol "
                    "arms, or truncation, or none (the finding is the object)"),
    2: dict(_COMMON, arm_all=False, pairs=False, benign=True, budget_factor=1.3,
            summary="four findings from the look-alike set of four causes: the armed fat mechanism, truncation, wrap-around, "
                    "or the object itself (a real edge or another material next to the artefacts); less scan time"),
    3: dict(_COMMON, arm_all=False, pairs=False, benign=True, causes=C.LABELS, never=("chemshift", "truncation"), protocol=_FAST, n_findings=4,
            unlocked=("slice_mm",), same_look=True, constructed=True, snr_slack=(0.85, 0.93),
            summary="four findings that look alike, all darker or all brighter than the gel. Dark: susceptibility signal "
                    "loss, opposed-phase cancellation in a material that holds water and fat, or a material that is dark. "
                    "Bright: a folded copy of a tube beside the object, a ghost copy of a vial that pulses, or a vial that "
                    "is bright. Every episode holds both artefact causes of its look at least once; position and sheet are "
                    "drawn independently of the causes. Two single-change experiments tell the three apart, and which "
                    "two depends on the look (one scan can carry changes for both looks, so a fixed list of three scans "
                    "serves either: see the open-loop control); the remedies have doses that the experiments measure (the slice from the measured loss at "
                    "the echo time of the remedy, the averages from the measured displacement of the copy), and the "
                    "delivered protocol needs a stated share of the SNR of the best clean protocol, so a dose stronger "
                    "than needed fails as well. Every scan "
                    "costs at least the console's 7 s of overhead and dummy cycles, and the budget is below four times "
                    "that: the first look and two experiments. Not served here: chemical-shift displacement (it needs "
                    "an in-phase echo and a low bandwidth, the cancellation an opposed echo) and truncation (at this "
                    "pixel size doubling the matrix does not clean it)"),
}


@functools.lru_cache(maxsize=512)
def _instance(seed: str, level: int) -> dict:
    stored = bank.get(seed, level)
    if stored is not None:
        return stored
    rejected, constructed = [], bool(LEVELS[level].get("constructed"))
    for attempt in range(MAX_ATTEMPTS):
        draw = G.draw_wanted(seed, LEVELS[level])
        # a constructed level never changes the drawn causes; a searched one takes what a world offers after a few worlds
        fallback = not constructed and attempt >= LEVELS[level].get("attempts_per_draw", ATTEMPTS_PER_DRAW)
        try:
            world_seed = F.reseed(seed, attempt)                        # the world is that of family.reseed(seed, attempt)
            inst = G.build_constructed(world_seed, LEVELS[level], draw) if constructed else G.build(world_seed, LEVELS[level], draw, fallback)
            inst = dict(inst, rejected=rejected, redraws=int(fallback), attempt=attempt)
            return json.loads(json.dumps(inst))                         # exactly what a bank line holds
        except G.Reject as e:
            rejected.append(str(e))
    raise RuntimeError(f"no instance within {MAX_ATTEMPTS} attempts")


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    inst = copy.deepcopy(_instance(seed, level))
    t = inst["task"]
    task = dict(family=FAMILY, **t, regions=t["findings"] + [t["reference_region"]],
                deliverable={"causes": {"<finding id>": ["<cause>", "..."]}, "protocol": {"<parameter>": "<value>"}})
    return dict(task_id=task_id or "clinic-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=inst["seed"],
                attempt=inst["attempt"], world_spec=inst["world_spec"], budget_s=inst["budget_s"], max_requests=64, task=task,
                hidden=inst["hidden"])


def _protocol_text(p: dict) -> str:
    return ", ".join(f"{k} {str(v).lower() if isinstance(v, bool) else v}" for k, v in p.items())


WINDOWED = ("te_ms", "tr_ms", "slice_mm")


def _budget_text(cfg: dict) -> str:
    if G.adaptive(cfg["task"]):
        return ("The budget holds the current protocol once and two more scans, fewer than there are single experiments that "
                "tell the possible causes apart: decide what each scan changes.")
    return ("The budget is smaller than one run of the current protocol: experiments have to be cheaper (shorter TR, one "
            "average).")


def _windows_text(req: dict) -> str:
    return "; ".join(f"{k} between {req[k][0]:g} and {req[k][1]:g}" for k in WINDOWED if k in req)


def _snr_text(req: dict) -> str:
    """The SNR requirement: a number where the card states one, else a share of the best clean protocol's (the
    number would name the doses of the remedies)."""
    how = ("(mean magnitude in the region divided\n  by the noise SD of one pixel; in an empty region the noise SD is the SD of the "
           "magnitude / 0.655)")
    if "min_snr_reference" in req:
        return f"signal-to-noise ratio in the reference region at least {req['min_snr_reference']:g} {how}"
    return (f"signal-to-noise ratio in the reference region at least {100 * req['snr_share']:g} % of that of the protocol with the "
            f"highest\n  signal-to-noise ratio that meets everything else here (as thick a slice and as many averages as keep every "
            f"finding clean) {how}")


def min_snr(cfg: dict) -> float:
    """The SNR a delivered protocol needs in the reference region: on the card, or (a share of the verified
    remedy's) with the hidden part of the instance."""
    req = cfg["task"]["requirements"]
    return req["min_snr_reference"] if "min_snr_reference" in req else cfg["hidden"]["min_snr_reference"]


def statement(cfg: dict) -> str:
    t = cfg["task"]
    req, p0 = t["requirements"], t["current_protocol"]
    ids = ", ".join(f["id"] for f in t["findings"])
    ref = t["reference_region"]
    locked = ", ".join(req["locked"])
    return f"""# Artefact clinic: say what each marked finding is, and correct the protocol
The scanner is a {t['exam']['field_t']:g} T system. A test object (a body of gel with vials
of unknown materials) lies in the scanner. The examination below was run on it, and a reader marked {len(t['findings'])}
findings on its image. Each finding is one of two things: produced by the acquisition (it would not be there with
another protocol), or the object itself (it is there whatever the protocol).

Current protocol ({t['current_scan_s']:.1f} s):
    {_protocol_text(p0)}
    {t['current_header']}
Findings (circles; centre and radius in mm from the isocentre):
{F.region_table(t['findings'])}
Reference region for the signal-to-noise ratio (homogeneous gel):
{F.region_table([ref])}

Deliver with `submit`:
  causes     for each finding ({ids}) the list of its causes, from
             {', '.join(t['causes_vocabulary'])}
             (at most {MAX_CAUSES} per finding; an empty list means the finding is the object itself).
  protocol   one protocol for the examination, on which every finding is free of acquisition artefacts. A field
             left out takes the console default, not the current protocol's value.
Requirements for the delivered protocol:
  scan time at most {req['max_scan_s']:g} s; pixel size (fov_mm / matrix) at most {req['max_pixel_mm']:g} mm;
  {_snr_text(req)};
  {_windows_text(req)}; unchanged from the current protocol: {locked}.
The delivered protocol is not run in this episode: the scanner evaluates it. {F.OWN_SCANS} {F.NOT_CHARGED}
{_budget_text(cfg)} Nothing about the materials is known in advance.

Passing: every finding is given exactly its causes; on a noise-free scan of the delivered protocol every finding
agrees with the same scan without the artefact mechanisms (mean absolute difference at most {100 * G.CLEAN_MECH:g} % of the
signal; at most {100 * G.CLEAN_TRUNC:g} % against the object itself averaged over each pixel); every requirement is met.
The scanner's log must show that you measured."""


def submit_schema(cfg: dict) -> dict:
    ids = [f["id"] for f in cfg["task"]["findings"]]
    cause = {"type": "array", "maxItems": MAX_CAUSES, "items": {"type": "string", "enum": list(cfg["task"]["causes_vocabulary"])}}
    return {"type": "object", "additionalProperties": False, "required": ["causes", "protocol"],
            "properties": {"causes": {"type": "object", "additionalProperties": False, "required": ids,
                                      "description": "the diagnosis: per marked finding the list of its causes; [] = the finding is the object itself",
                                      "properties": {i: cause for i in ids}},
                           "protocol": dict(scanner_env.protocol_object(), description="the corrected protocol of the examination "
                                            "(the parameters of plan)")}}


def check(cfg: dict, answer, exam: dict):
    ids = [f["id"] for f in cfg["task"]["findings"]]
    shape = '{"causes": {' + ", ".join(f'"{i}": [...]' for i in ids) + '}, "protocol": {...}}'
    if not isinstance(answer, dict) or set(answer) != {"causes", "protocol"} or not isinstance(answer["causes"], dict) \
            or set(answer["causes"]) != set(ids):
        return "the deliverable is " + shape
    for i in ids:
        v = answer["causes"][i]
        vocabulary = cfg["task"]["causes_vocabulary"]
        if not isinstance(v, list) or len(v) > MAX_CAUSES or len(set(map(str, v))) != len(v) or any(c not in vocabulary for c in v):
            return f"causes of {i}: a list of at most {MAX_CAUSES} different names from {list(vocabulary)}"
    pl = console.plan(answer["protocol"], exam)
    if not pl["accept"]:
        return "the console rejects the delivered protocol: " + "; ".join(v["message"] for v in pl["violations"])
    return None


# ------------------------------------------------------------------------------------------------ the pass rule
def diagnosis_ok(delivered, truth) -> bool:
    return sorted(set(delivered)) == sorted(truth)


def locked_ok(rp: dict, rp0: dict, locked) -> bool:
    return all(rp.get(k) == rp0.get(k) for k in locked)


def window_ok(value: float, window) -> bool:
    return window[0] - 1e-9 <= value <= window[1] + 1e-9


def pixel_ok(rp: dict, max_pixel_mm: float) -> bool:
    return rp["fov_mm"] / rp["matrix"] <= max_pixel_mm + 1e-9


def time_ok(scan_s: float, max_scan_s: float) -> bool:
    return scan_s <= max_scan_s + 1e-9


def snr_ok(snr: float, min_snr: float) -> bool:
    return snr >= min_snr


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    problem = check(cfg, answer, world["console"])
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    t, truth = cfg["task"], cfg["hidden"]["causes"]
    req = t["requirements"]
    pl = console.plan(answer["protocol"], world["console"])
    rp, rp0 = pl["realised"], console.plan(t["current_protocol"], world["console"])["realised"]
    tw = M.Twins(world, cfg["hidden"]["mechanisms"])
    errors = {f["id"]: tw.errors(rp, f) for f in t["findings"]}
    snr, floor = M.snr(tw.engine(), rp, t["reference_region"]), min_snr(cfg)
    checks = dict(deliverable_valid=True)
    checks.update({f"diagnosis_{i}": diagnosis_ok(answer["causes"][i], truth[i]) for i in truth})
    checks.update({f"clean_{i}": G.is_clean(e) for i, e in errors.items()})
    checks.update(locked_parameters=locked_ok(rp, rp0, req["locked"]),
                  **{f"{k.split('_')[0]}_window": window_ok(rp[k], req[k]) for k in WINDOWED if k in req}, pixel_size=pixel_ok(rp, req["max_pixel_mm"]), scan_time=time_ok(pl["scan_s"], req["max_scan_s"]),
                  snr_reference=snr_ok(snr, floor))
    dirty = [f"{i} differs from its twin by {100 * e['mech']:.1f} % and from the object by {100 * e['trunc']:.1f} %"
             for i, e in errors.items() if not G.is_clean(e)]
    wrong = [i for i in truth if not checks[f"diagnosis_{i}"]]
    notes = dirty + ([f"wrong diagnosis of {', '.join(wrong)}"] if wrong else [])
    if not checks["snr_reference"]:
        notes.append(f"SNR {snr:.0f} below {floor:g}")
    if not checks["scan_time"]:
        notes.append(f"scan time {pl['scan_s']:.1f} s above {req['max_scan_s']:g} s")
    worst = max(max(e["mech"] / G.CLEAN_MECH, e["trunc"] / G.CLEAN_TRUNC) for e in errors.values())
    return F.result(checks, "; ".join(notes), error=round(worst, 4), errors={i: {k: round(v, 4) for k, v in e.items()} for i, e in errors.items()},
                    snr_reference=round(snr, 1), scan_s=pl["scan_s"], pixel_mm=round(rp["fov_mm"] / rp["matrix"], 4),
                    water_fat_shift_mm=round(pl["derived"]["water_fat_shift_px"] * rp["fov_mm"] / rp["matrix"], 3),
                    echo=C.echo_class(rp["te_ms"], world["b0_t"]), diagnosis_correct=len(truth) - len(wrong), findings=len(truth))


def _task_kind() -> dict:
    """Per level, from the stored gate result: "measurement" when the open-loop policy (a fixed scan list and a
    fixed estimator) passes within 25 points of the reference, else "decision"; None before the gate is stored."""
    from ... import gate
    stored = gate.load("Scanner/ArtefactClinic-v0")
    if not stored:
        return {lv: None for lv in LEVELS}
    out = {}
    for lv, r in stored["levels"].items():
        open_loop = next((x["rate"] for x in r["reported"] if x["policy"] == "open_loop"), None)
        out[int(lv)] = None if open_loop is None else ("measurement" if r["reference"]["rate"] - open_loop < 0.25 else "decision")
    return out


def _budget_rule(level: int, lv: dict) -> str:
    if lv.get("same_look"):
        ratio = bank.battery_over_budget(level)
        stored = f"{ratio[0]:.2f} (median of {ratio[1]} stored instances)" if ratio else "stored with every instance"
        return ("budget: the first look and the two experiments of the costlier look, never more than four times the "
                f"{G.SCAN_FLOOR_S:g} s every scan costs before its first line; cost of the minimal battery that diagnoses without "
                f"looking / budget = {stored}")
    return f"budget {lv['budget_factor']:g} x the scan time of the reference's experiments, always less than one run of the complaint protocol"


READING = ("How images are read: the reference policy reads single pixels of stored images inside each finding. A text-only "
           "agent gets the same numbers from `read_regions`: a circle of radius 1 mm on a pixel centre (centre + (i - n // 2) x "
           "field of view / n) returns that pixel alone when pixels are larger than 1 mm, which holds for every experiment of "
           "the reference (matrix at most 224 on a 240 mm field of view); 16 circles per call, free of charge. At level 3 the "
           "reference also reads two phase images (`view` with kind phase; `read_regions` returns the complex mean of a circle): "
           "the complex difference of the complaint and the one-average scan along the phase-encode line of a ghost finding "
           "shows where its copies went. ")


def _kind_sentence() -> str:
    kinds = {lv: k for lv, k in _task_kind().items() if k}
    if not kinds:
        return "Task kind (measurement or decision): not stated, the gate result is not stored."
    return "Task kind by the stored gate: " + ", ".join(f"level {lv} a {k} task" for lv, k in sorted(kinds.items())) + \
        " (measurement: a fixed scan list with a fixed estimator passes within 25 points of the reference)."


def card() -> dict:
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    return dict(
        id="Scanner/ArtefactClinic-v0", title="Artefact clinic: diagnose the marked findings and correct the protocol", kind="candidate",
        given="A test object with hidden materials, the protocol of an examination, findings a reader marked on its image, "
              "the requirements of the examination; scan time for experiments",
        deliver="For each finding its causes (or none: it is the object), and one protocol that images every finding cleanly "
                "inside the requirements",
        scored_by="Noise-free re-scan of the delivered protocol on the hidden object against its counterfactual twins (the same "
                  "protocol with the artefact mechanisms switched off, and the object averaged over each pixel); the diagnosis "
                  "against the mechanisms that corrupt each finding in the complaint protocol",
        summary="An examination with look-alike findings: chemical-shift displacement, opposed-phase cancellation, truncation "
                "ringing, wrap-around, a motion ghost, susceptibility signal loss, or the object itself. At levels 1 and 2 one "
                "experiment separates each pair (other bandwidth, other echo time, finer matrix, more phase field of view). At "
                "level 3 the findings of an episode look alike, two experiments tell the three possibilities of a look apart "
                "(half the slice and the in-phase echo; phase oversampling and one average), the budget holds the first look "
                "and two experiments, and two remedies have a dose that an experiment measures. Every remedy costs scan time "
                "or signal-to-noise ratio, and the requirements do not leave room for all of them. " + READING + _kind_sentence(),
        world=dict(phantom="QA phantom, 1.5 T or 3 T. Levels 1 and 2: a gel body with a ring of vials, two of them oil. Level 3: "
                           "an elongated gel body with placed vials (dark, bright, a mixture of water and fat, one that "
                           "pulses), tubes beside it, patches of through-slice field gradient",
                   sequences_useful=["spgr"],
                   agent_side_formulas="; ".join(f"{c.label}: {c.formula}" for c in C.CATALOGUE.values())),
        hidden=["which vials hold oil; T1, T2, proton density of every material", "the extent of the object and what lies beside it",
                "which mechanism reaches which finding", "level 3: which vial pulses and how fast, where the through-slice gradient is and how strong",
                "noise level", "the 128-bit world seed"],
        told=["the complaint protocol and its header line", "the findings and the reference region (circles)",
              "the requirements: scan time, pixel size, SNR, TE window, locked parameters", "the vocabulary of causes",
              "the instrument datasheet; realised protocol, timing and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"] + "; " + _budget_rule(lv, d), budget_s=bank.budget_range(lv) or [0.0, 0.0]) for lv, d in LEVELS.items()},
        limits=dict(max_scanner_requests=64, max_tool_calls=160),
        deliverable={"tool": "submit", "format": {"causes": {"<finding id>": ["<cause>"]}, "protocol": {"<parameter>": "<value>"}}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan",
                           "the console accepts the delivered protocol"],
                    error="largest over the findings of E / threshold on the delivered protocol; E = mean absolute difference "
                          "to the twin, relative to the twin, inside the finding",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing=f"all gates; every finding diagnosed exactly; every finding clean (E <= {G.CLEAN_MECH} against the "
                            f"mechanism-free twin, <= {G.CLEAN_TRUNC} against the pixel-averaged object); scan time, pixel "
                            "size, SNR, TE window and locked parameters as required (level 3: the SNR floor is a stated "
                            "share of the SNR of the verified remedy, a number the card does not print)"),
        catalogue=[dict(label=c.label, twin=c.mechanism, formula=c.formula, remedies=[list(r) for r in c.remedies],
                        experiment=c.experiment, present=c.present, absent=c.absent, clean=c.clean) for c in C.CATALOGUE.values()],
        not_yet=[dict(p) for p in C.PENDING],
        reading_images=READING, task_kind=_task_kind(),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier's scans are noise-free",
    )
