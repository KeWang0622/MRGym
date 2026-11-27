"""``Scanner/FatFraction``: instances and the verifier.

The job: deliver the proton-density fat fraction (PDFF) of several discs that hold a mixture of water and fat. In a
gradient echo the fat signal turns against the water signal with the echo time, so two echoes, one with fat opposed
to water and one in phase, give the two amounts: the textbook formula is PDFF = (IP - OP) / (2 IP). On a scanner that
formula is wrong three times over, and the task is to know why and to remove each error.

  spectrum     fat is not one line. With the six peaks the engine gives its fat (``tissue.FAT_PEAKS``; the datasheet
               says "a multi-peak spectrum" and names no peak, so the task statement lists them) the fat vector
               ``spectrum(t)`` is about -0.79 at the opposed-phase time, not -1, and about +0.82 at the in-phase
               time: the formula reads low.
  T2*          the in-phase echo comes later than the opposed-phase echo and has decayed more: reads low again.
  T1           fat recovers faster than water, so at a flip angle that saturates water the fat share of the SIGNAL is
               larger than its share of the protons: reads high. A low flip angle (or a long repetition time) keeps
               T1 out. The T1 of the water in each disc is hidden, so the error cannot be divided out afterwards.
At a flip angle near 20 degrees the third error offsets part of the first two; how much depends on the hidden T1 and
T2* of each disc (``controls.two_point_high_flip`` plays that recipe; it must fail).

Deliverable    {"pdff_percent": {"D1": 12.5, "D2": 31.0, "D3": 22.4, "D4": 8.1}}       fat fraction of every disc, percent
Pass rule      for every disc the delivered value lies within ``TOLERANCE_POINTS`` percentage points of the true
               fraction: fat proton density x fat share / (the same + water proton density x water share) of the
               hidden mixture. The tolerance is a constant of the family; it is not derived from what a policy achieves.

Four discs make the deliverable multi-coordinate: one constant fraction passes only when all four hidden mixtures
lie within 3 points of it.

The world. A disc of gel without vials (``qa.n_ring = (0, 0)``) with four mixture discs placed by this module:
every disc is two ``inserts`` of the world builder on one circle, a water material and a fat material, with the
weights of the mixture (``anatomy.qa``: "two inserts on one disc are a mixture"). The world is
``W.make_world(seed, world_spec)`` and nothing else; the fractions are drawn here from the ``task`` stream of the
seed and written into the world spec, which is hidden.

Levels. 1 states the method (one multi-echo scan at a low flip angle and arithmetic on six magnitudes; fat is the
smaller part of every disc). 2 withholds it. 3 lets fat be the larger part: the opposed-phase magnitude is
|W - 0.79 F| and does not say which of the two is larger (the two readings cross where W = 0.79 F, near 56 %, not at
50 %); the phase of the echoes says it.
"""
from __future__ import annotations

import math
import os

import numpy as np

from ...scanner import tissue
from ...scanner import world as W
from .. import family as F

FAMILY = "fat_fraction"
ENV_ID = "Scanner/FatFraction-v0"
SUBJECT = "A"                               # signal and contrast
KIND = "measure"
ISOLATES = "water-fat chemical shift: fat fraction from in-phase and opposed-phase echoes"
SUBMIT_DOC = "Deliver the proton-density fat fraction of every marked disc, in percent."
TOLERANCE_POINTS = 3.0                      # a delivered fraction may be off by this many percentage points
N_DISCS = 4
DISC_MM, ROI_MM = 18.0, 10.0                # radius of a mixture disc and of the marked circle inside it
RING_MM = (44.0, 54.0)                      # distance of the disc centres from the isocentre
B0_T = 1.5
OFF_RESONANCE_MAX_HZ = 80.0                 # told: the field at a disc is off resonance by less than this
GAMMA_MHZ_PER_T = 42.577478518
OUTCOMES = ("no_disc_within", "some_discs_within")          # of a delivered, measured answer that fails (worst first)
# Examined competencies this family exercises (rows of ``mrgym.coverage``): in-phase and opposed-phase echo times, the
# cancellation of water and fat at the opposed-phase time, the Dixon idea, the echo time and the flip angle of a gradient echo.
COMPETENCIES = ("option.in_out_of_phase", "artefact.chemical_misregistration", "option.suppression", "parameter.te", "parameter.flip_angle")
BASE = dict(kind="qa", b0_t=B0_T, b0_hz=20.0, cf_offset_hz=50.0)
QA = dict(n_ring=(0, 0), p_centre=0.0)
WATER = dict(pd=(0.75, 1.0), t1_ms=(150.0, 3000.0), t2_ms=(40.0, 150.0))        # the water part of a disc (T1, T2 log-uniform)
T2_OVER_T1_MAX = 0.6                        # of the water: T2 is capped at this share of T1
FAT = dict(pd=(0.85, 0.95), t1_scale=(0.85, 1.15), t2_scale=(0.85, 1.15))       # the fat part: scales of the table's fat
T2P_MS = (10.0, 200.0)                      # T2' of a disc, log-uniform: the field spread inside a voxel, shared by its water and its fat
EDGE = 0.002                                # the fractions are drawn this far inside the told range (0.2 points)
# The method level 1 states, as data: the statement prints it and the reference policy plays it.
METHOD = dict(sequence="spgr", flip_deg=3.0, tr_ms=60.0, n_echoes=6, te_ms=2.3, echo_spacing_ms=2.3, matrix=64, partial_fourier=0.75,
              bandwidth_hz_per_px=800.0, averages=8)
LEVELS = {
    1: dict(pdff=(0.05, 0.45), budget_s=50.0, hints=("method",),
            summary="the method is stated: six echoes at a low flip angle, the decay from the in-phase echoes, the fat vector "
                    "of the six-peak spectrum; fat is the smaller part of every disc"),
    2: dict(pdff=(0.05, 0.45), budget_s=50.0,
            summary="no method: the flip angle, the echo times and the signal model are the agent's"),
    3: dict(pdff=(0.05, 0.95), budget_s=50.0,
            summary="fat may be the larger part: the opposed-phase magnitude does not say which of water and fat is larger"),
}


def spectrum(t_ms, b0_t: float = B0_T):
    """The fat vector at time ``t_ms`` after the excitation: sum of the six peaks of the scanner's fat, each turning
    at its own frequency, normalised to 1 at t = 0. Complex; water is 1 at every time."""
    f0 = GAMMA_MHZ_PER_T * b0_t                                           # Hz per ppm
    total = sum(a for _, a in tissue.FAT_PEAKS)
    return sum(a * np.exp(2j * np.pi * ppm * f0 * np.asarray(t_ms, float) * 1e-3) for ppm, a in tissue.FAT_PEAKS) / total


def _log_uniform(rng, lo, hi) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    """Episode config. The mixtures are drawn from the ``task`` stream of the seed and written into the hidden world
    spec; the public task holds the circles, the told bounds and the spectrum."""
    lv = LEVELS[level]
    rng = W.rng_for(seed, "task")
    fat = tissue.table(B0_T)["subcut_fat"]
    ring, start = float(rng.uniform(*RING_MM)), float(rng.uniform(0.0, 2 * math.pi))
    angles = start + 2 * math.pi * np.arange(N_DISCS) / N_DISCS + rng.uniform(-0.15, 0.15, N_DISCS)
    inserts, regions = [], []
    for k, a in enumerate(angles):
        y, x = round(ring * math.sin(a), 2), round(ring * math.cos(a), 2)
        t2p = round(_log_uniform(rng, *T2P_MS), 2)
        t1 = round(_log_uniform(rng, *WATER["t1_ms"]), 1)
        water = dict(pd=round(float(rng.uniform(*WATER["pd"])), 4), t1_ms=t1,
                     t2_ms=round(min(_log_uniform(rng, *WATER["t2_ms"]), T2_OVER_T1_MAX * t1), 2), t2p_ms=t2p, fat=False)
        oil = dict(pd=round(float(rng.uniform(*FAT["pd"])), 4), t1_ms=round(fat["t1_ms"] * float(rng.uniform(*FAT["t1_scale"])), 1),
                   t2_ms=round(fat["t2_ms"] * float(rng.uniform(*FAT["t2_scale"])), 2), t2p_ms=t2p, adc_um2_ms=fat["adc_um2_ms"], fat=True)
        pdff = float(rng.uniform(lv["pdff"][0] + EDGE, lv["pdff"][1] - EDGE))            # inside the told range after the rounding below
        share = pdff * water["pd"] / (pdff * water["pd"] + (1.0 - pdff) * oil["pd"])      # share of the disc the fat fills
        share = round(share, 4)
        inserts += [dict(y_mm=y, x_mm=x, radius_mm=DISC_MM, tissue=water, weight=round(1.0 - share, 4)),
                    dict(y_mm=y, x_mm=x, radius_mm=DISC_MM, tissue=oil, weight=share)]
        regions.append(dict(id=f"D{k + 1}", y_mm=y, x_mm=x, radius_mm=ROI_MM))
    world_spec = dict(BASE, qa=dict(QA, inserts=inserts))
    task = dict(family=FAMILY, regions=regions, pdff_range_percent=[100 * lv["pdff"][0], 100 * lv["pdff"][1]],
                tolerance_points=TOLERANCE_POINTS, field_t=B0_T, off_resonance_max_hz=OFF_RESONANCE_MAX_HZ,
                fat_peaks=[dict(ppm=ppm, share=a) for ppm, a in tissue.FAT_PEAKS],
                deliverable={"pdff_percent": {"<disc id>": "<number, percent>"}})
    if lv.get("hints"):
        task["hints"] = list(lv["hints"])
    return dict(task_id=task_id or "fatfraction-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed,
                world_spec=world_spec, budget_s=float(lv["budget_s"]), max_requests=40, task=task)


def method_text() -> str:
    """The method of level 1, written from ``METHOD`` and the spectrum (the reference policy plays the same numbers)."""
    m = METHOD
    te = [m["te_ms"] + k * m["echo_spacing_ms"] for k in range(m["n_echoes"])]
    c1, c2, c4 = (float(spectrum(te[k]).real) for k in (0, 1, 3))
    return f"""How to do it (this level states the method; fat is the smaller part of every disc).
  1. Run one `spgr` scan: flip_deg {m['flip_deg']:g}, tr_ms {m['tr_ms']:g}, n_echoes {m['n_echoes']}, te_ms {m['te_ms']:g}, echo_spacing_ms {m['echo_spacing_ms']:g}, matrix {m['matrix']},
     partial_fourier {m['partial_fourier']:g}, bandwidth_hz_per_px {m['bandwidth_hz_per_px']:g}, averages {m['averages']}. Fat is opposed to water in echoes 1, 3, 5 and
     in phase in echoes 2, 4, 6. (The low flip angle keeps T1 out: at a high one fat, which recovers faster, is over-weighted.)
  2. Read `mean_magnitude` of each disc: m1 ... m6 at the echo times t1 ... t6 (ms). The fat vector of the six peaks is
     c1 = {c1:+.3f} at t1, c2 = {c2:+.3f} at t2 and c4 = {c4:+.3f} at t4 (not -1, +1, +1).
  3. Decay rate from two in-phase echoes, first guess: R = ln(m2 / m4) / (t4 - t2).
  4. The in-phase signal carried back to the first echo time: I = m2 x exp(R x (t2 - t1)).
     F = (I - m1) / (c2 - c1), W = I - c2 x F.
  5. The fat vector shrinks between t2 and t4; take that out of the decay:
     R = ln(m2 x (W + c4 x F) / (m4 x (W + c2 x F))) / (t4 - t2). Do step 4 again with this R.
  6. Fat fraction = 100 x F / (W + F) percent."""


def statement(cfg: dict) -> str:
    t = cfg["task"]
    lo, hi = t["pdff_range_percent"]
    peaks = ", ".join(f"{p['ppm']:+.2f} ppm {100 * p['share']:.1f} %" for p in t["fat_peaks"])
    hints = "\n\n" + method_text() if "method" in t.get("hints", ()) else ""
    return f"""# Measure the fat fraction of each marked disc
A test object lies in the field of view: a disc of gel with {len(t['regions'])} smaller discs, each a uniform mixture of water and fat.
Deliver the proton-density fat fraction of each marked disc, in percent, with `submit`:
    fat fraction = 100 x fat protons / (fat protons + water protons),
the share of the signal that is fat when neither T1 nor T2* weights it.
Regions (circles; centre and radius in mm from the isocentre; each lies inside one disc):
{F.region_table(t['regions'])}
The fat fractions lie between {lo:g} and {hi:g} percent.
The fat of this scanner has six peaks (frequency relative to water, share of the fat signal):
    {peaks}
At {t['field_t']:g} T, 1 ppm is {GAMMA_MHZ_PER_T * t['field_t']:.2f} Hz. The field at a disc is off resonance by an unknown amount below {t['off_resonance_max_hz']:g} Hz.
T1 and T2* of the water and of the fat differ from disc to disc and are not known.

Passing: for every disc the delivered fat fraction lies within {t['tolerance_points']:g} percentage points of the true one.
The scanner's log must show that you measured. An answer in the wrong unit is wrong.""" + hints


def submit_schema(cfg: dict) -> dict:
    ids = [r["id"] for r in cfg["task"]["regions"]]
    return {"type": "object", "additionalProperties": False, "required": ["pdff_percent"],
            "properties": {"pdff_percent": {"type": "object", "additionalProperties": False, "required": ids,
                                            "description": "proton-density fat fraction of every marked disc in percent, by region id",
                                            "properties": {i: {"type": "number", "minimum": 0.0, "maximum": 100.0} for i in ids}}}}


def check(cfg: dict, answer, exam: dict):
    ids = [r["id"] for r in cfg["task"]["regions"]]
    if not isinstance(answer, dict) or set(answer) != {"pdff_percent"} or not isinstance(answer["pdff_percent"], dict) \
            or set(answer["pdff_percent"]) != set(ids):
        return 'the deliverable is {"pdff_percent": {' + ", ".join(f'"{i}": <number>' for i in ids) + "}}"
    for i in ids:
        v = answer["pdff_percent"][i]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return f"the fat fraction of {i} is not a finite number"
        if not 0.0 <= v <= 100.0:
            return f"the fat fraction of {i} is {v:g}: a fraction in percent lies between 0 and 100"
    return None


def true_pdff_percent(cfg: dict) -> dict:
    """Verifier side: {disc id: proton-density fat fraction of its mixture in percent}, from the hidden world spec."""
    inserts = cfg["world_spec"]["qa"]["inserts"]
    out = {}
    for k, r in enumerate(cfg["task"]["regions"]):
        water, fat = inserts[2 * k], inserts[2 * k + 1]
        assert fat["tissue"]["fat"] and not water["tissue"]["fat"] and (water["y_mm"], water["x_mm"]) == (r["y_mm"], r["x_mm"])
        f, w = fat["weight"] * fat["tissue"]["pd"], water["weight"] * water["tissue"]["pd"]
        out[r["id"]] = 100.0 * f / (f + w)
    return out


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    problem = check(cfg, answer, (world or {}).get("console") or {})
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    truth, limit = true_pdff_percent(cfg), float(cfg["task"]["tolerance_points"])
    miss = {i: float(answer["pdff_percent"][i]) - t for i, t in truth.items()}              # delivered - true, points
    checks = dict(deliverable_valid=True, **{f"pdff_within_tolerance_{i}": abs(m) <= limit for i, m in miss.items()})
    bad = [f"{i} is off by {m:+.1f} points" for i, m in miss.items() if abs(m) > limit]
    worst = max(abs(m) for m in miss.values())
    return F.result(checks, "; ".join(bad), error=worst, tolerance=limit, per_region={i: round(m, 4) for i, m in miss.items()},
                    distance=dict(regions=len(miss), within=len(miss) - len(bad), worst_points=worst))


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured answer that fails a check: one of ``OUTCOMES``."""
    return "some_discs_within" if any(v for k, v in score["checks"].items() if k.startswith("pdff_within_tolerance_")) else "no_disc_within"


def card() -> dict:
    from ... import gate, learn
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    kind = F.kind_by_open_loop(gate.load(ENV_ID))
    return dict(
        id=ENV_ID, title="Fat fraction of water-fat mixtures from in-phase and opposed-phase echoes", kind="candidate",
        given="A gel phantom with four discs of a water-fat mixture; the fat spectrum of the scanner; a scan-time budget",
        deliver="The proton-density fat fraction of every disc, in percent",
        scored_by=f"The hidden mixture of every disc: each delivered fraction within {TOLERANCE_POINTS:g} percentage points of it",
        summary="Subject A (signal and contrast), a single-skill measurement: the water-fat chemical shift as a quantity. "
                "The textbook formula (IP - OP) / (2 IP) reads low on a real spectrum with T2* decay and high where T1 "
                "weights the echoes. The task is to remove each error: a low flip angle, the fat vector of the six "
                "peaks, the decay between the echoes; and, where fat may be the larger part, the phase.",
        task_kind=kind["kind"], task_kind_why=kind["evidence"],
        world=dict(phantom="a disc of gel with four water-fat mixture discs, 1.5 T", sequences_useful=["spgr"],
                   agent_side_formulas="signal of a mixture at echo time t: (W + F c(t)) exp(-t / T2*) with c(t) the sum of the six "
                                       "peaks (task statement); the Ernst formula for the T1 weighting (datasheet section 5)"),
        hidden=["the fat fraction of every disc", "proton density, T1, T2 and T2' of the water and of the fat of every disc",
                "off-resonance field and centre-frequency error", "transmit gain and B1 pattern", "receive profile", "noise level",
                "the 128-bit world seed"],
        told=["region centres and radii", "the six-peak fat spectrum and the field strength", "the range of the fat fractions",
              "a bound of the off-resonance", "the pass tolerance", "level 1: the method", "the instrument datasheet",
              "budget and request limits", "realised protocol, timing and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        practice_levels=[lv for lv, d in LEVELS.items() if d.get("kind") == "practice"],
        learnability=learn.stored(ENV_ID),
        limits=dict(max_scanner_requests=40, max_tool_calls=48),
        deliverable={"tool": "submit", "format": {"pdff_percent": {"<disc id>": "<percent>"}}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan",
                           "deliverable valid: exactly the disc ids, values in 0 to 100 percent"],
                    error="largest over the discs of |delivered - true| in percentage points",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing=f"all gates and, for every disc, error <= {TOLERANCE_POINTS:g} points"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier reads the hidden mixture and scans nothing",
    )
