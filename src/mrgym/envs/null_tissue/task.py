"""``Scanner/NullTissue``: instances and the verifier.

The job: choose the inversion time of an inversion-recovery scan so that one material gives no signal (what FLAIR does
to fluid and STIR does to fat). The inversion time that nulls a material depends on its T1, on the repetition time of
the scan (the magnetisation has not recovered when the next inversion arrives) and on how complete the inversion is.
All three are hidden or instance-specific, so the answer has to be measured.

Deliverable    {"ti_ms": {"R1": 412.0, "R2": 655.0}}        one inversion time per marked region
Pass rule      for every marked region: on a noise-free scan of the target protocol with the inversion time delivered
               for it, its signal is at most ``null_max`` times its signal in the same scan without the inversion.

Why several regions. With one region the deliverable is one number in a known range, and the best answer that ignores
the object passes one instance in ten to twenty. Two or three independent materials make that product small while
each tolerance stays wide enough for a careful measurement; the same probe scans serve every region.

Progress (``score["progress"]``, a diagnostic for trainers and for ``mrgym check --learnability``; never paid). With
``k`` of ``n`` regions nulled and ``c`` the mean closeness of the others (``ln(0.5 / ratio) / ln(0.5 / null_max)``
clipped to [0, 1): 0 when a region keeps half its signal or more), ``progress = 0.5 (k + c) / n``: classes are ordered
by the number of regions nulled, the largest value of a class lies below the smallest of the next, and 1 is reached
only by a pass. An answer that ignores the object stays low (``mrgym check <id> --learnability`` measures the bound).

Level 0 is a practice level (``kind="practice"``: the gate does not claim it). It is the world of level 1 with two
hints rendered into the statement: the procedure, and a first probe that is acquired at reset (in the log, charged).
"""
from __future__ import annotations

import math
import os

import numpy as np

from ...scanner import console
from ...scanner import world as W
from .. import family as F

FAMILY = "null_tissue"
SUBMIT_DOC = "Deliver, for each marked region, the inversion time (milliseconds) of the target scan that nulls it."
GEOMETRY = dict(matrix=128, bandwidth_hz_per_px=300.0)
BASE = dict(kind="qa", b0_t=1.5)
# The probe the practice level proposes and acquires: fully sampled at matrix 32, the probe of the reference policy
# (``controls.PROBE``). Until engine 0.2.0 it was matrix 64 with partial_fourier 0.6, the cheapest probe of that engine;
# on engine 0.3.0 a partial Fourier image has more noise and a biased signed mean where a region is near its null
# (docs/physics/limits.md, block 5a), so the level no longer proposes it.
FIRST_PROBE = dict(sequence="ir", matrix=32, bandwidth_hz_per_px=300.0, tr_ms=800.0, ti_ms=740.0)
FAR_RATIO = 0.5                         # a region that keeps this share of its signal, or more, has closeness 0
OUTCOMES = ("none_nulled", "some_nulled")       # outcome labels of a delivered, measured answer that fails (worst first)
# Examined competencies this family exercises (rows of ``mrgym.coverage``): choosing TI for a given TR to null a material.
COMPETENCIES = ("parameter.ti", "parameter.tr", "option.suppression")
LEVELS = {
    0: dict(kind="practice", hints=("rule", "first_measurement"),
            world_spec=dict(BASE, qa=dict(vial_t1_ms=(200.0, 2000.0), n_ring=(4, 5))), tr_ms=(6000.0, 6000.0), n_regions=2,
            null_max=0.05, budget_s=185.0,
            summary="practice: the world of level 1 with the procedure stated and the first probe already acquired"),
    1: dict(world_spec=dict(BASE, qa=dict(vial_t1_ms=(200.0, 2000.0), n_ring=(4, 5))), tr_ms=(6000.0, 6000.0), n_regions=2,
            null_max=0.05, budget_s=150.0,
            summary="long repetition time: the textbook rule TI = ln 2 x T1 is nearly right, once T1 is measured"),
    2: dict(world_spec=dict(BASE, qa=dict(vial_t1_ms=(200.0, 2000.0), n_ring=(4, 5))), tr_ms=(1200.0, 3000.0), n_regions=3,
            null_max=0.05, budget_s=200.0,
            summary="repetition time comparable to T1: the null moves with the repetition time"),
    3: dict(world_spec=dict(BASE, qa=dict(vial_t1_ms=(200.0, 2000.0), n_ring=(4, 5)), inv_eff=(0.78, 0.95),
                            tx_gain_dev=(0.08, 0.20), sigma0=0.12),
            tr_ms=(1200.0, 3000.0), n_regions=3, null_max=0.05, budget_s=200.0,
            summary="as level 2 with an incomplete inversion, a miscalibrated transmitter and more noise"),
}


def target_protocol(task: dict, ti_ms) -> dict:
    return dict(GEOMETRY, sequence="ir", tr_ms=task["target_scan"]["tr_ms"], ti_ms=ti_ms)


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    lv = LEVELS[level]
    rng = W.rng_for(seed, "task")
    tr = float(np.round(rng.uniform(*lv["tr_ms"]) / 10.0) * 10.0)
    regions = W.make_world(seed, lv["world_spec"], layout_only=True)["regions"]
    targets = [r for r in regions if int(r["id"][1:]) <= lv["n_regions"]]   # region ids are a random permutation of the vials
    task = dict(family=FAMILY, regions=F.public_regions(targets), null_max=lv["null_max"],
                target_scan=dict(GEOMETRY, sequence="ir", tr_ms=tr, te_ms="console minimum", averages=1, slice_mm=5.0),
                deliverable={"ti_ms": {"<region id>": "<number, milliseconds>"}})
    if lv.get("hints"):
        task["hints"] = list(lv["hints"])
    return dict(task_id=task_id or "nulltissue-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed,
                world_spec=lv["world_spec"], budget_s=float(lv["budget_s"]), max_requests=40, task=task)


def prefill(cfg: dict, scanner) -> list | None:
    """What a level acquires at reset, through the session (so it is in the log, charged to the budget and counts as a
    measurement): the responses of those scans, or None. Called once per episode by ``FamilyEnv``."""
    if "first_measurement" not in cfg["task"].get("hints", ()):
        return None
    return [scanner.scan(dict(FIRST_PROBE), rois=cfg["task"]["regions"])]


def hint_text(cfg: dict, given: list | None = None) -> str:
    """The hints of the level, as the paragraphs appended to the statement ("" when the level has none)."""
    hints, out = cfg["task"].get("hints", ()), []
    if "rule" in hints:
        p = FIRST_PROBE
        out.append(f"""How to do it (this level states the procedure).
  1. Probe with `ir` at a repetition time short enough for several scans (for example tr_ms {p['tr_ms']:g}, matrix {p['matrix']},
     bandwidth_hz_per_px {p['bandwidth_hz_per_px']:g}, fully sampled): three or more inversion times spread from 60 ms to
     tr_ms - 50 ms, read out on the marked regions.
  2. Sign the signal. At the longest inversion time every material is past its null, so its signal is positive. A
     read-out whose phase differs from the region's phase in that scan by more than pi / 2 is negative.
  3. Per region, fit S(TI) = a - b exp(-TI / T1) to the signed signals. Let q = b / a.
  4. The probes ran at another repetition time than the target scan, so carry the fit over. With E(TR) = exp(-(TR - TE / 2) / T1):
     k = (q - 1) / (1 - q E(TR_probe)), then TI_null = T1 ln((1 + k) / (1 + k E(TR_target))).""")
    if "first_measurement" in hints and given:
        rows = "\n".join(f"  {r['id']:4s} |complex mean| {r['complex_mean_magnitude'][0]:.6g}   phase_rad {r['complex_mean_phase_rad'][0]:.4f}"
                         for r in given[0]["roi_readouts"])
        rp = given[0]["realised"]
        out.append(f"""The first probe is already acquired: scan {given[0]['scan_index']} (`ir`, tr_ms {rp['tr_ms']:g}, ti_ms {rp['ti_ms']:g}, te_ms {rp['te_ms']:g}, matrix {rp['matrix']}),
{given[0]['cost_s']:.2f} s of the budget. Its read-out (the longest inversion time of that repetition time: every signal is positive):
{rows}""")
    return "\n\n".join(out)


def statement(cfg: dict, given: list | None = None) -> str:
    t = cfg["task"]
    s = t["target_scan"]
    ids = ", ".join(r["id"] for r in t["regions"])
    hints = hint_text(cfg, given)
    return f"""# Null each marked material with inversion recovery
A test object with vials of unknown materials lies in the field of
view. For each marked region the examination needs an inversion-recovery image in which the material of that region
gives no signal. The scan is the same for all of them except for its inversion time:
    sequence ir, tr_ms {s['tr_ms']:g}, te_ms at the console minimum, matrix {s['matrix']}, bandwidth_hz_per_px {s['bandwidth_hz_per_px']:g},
    field of view at the console default, 1 average, slice 5 mm.
Regions (circles; centre and radius in mm from the isocentre; each lies inside one homogeneous material):
{F.region_table(t['regions'])}
Deliver one inversion time per region ({ids}) with `submit`. The target scans are not run in this episode: the scanner
evaluates the inversion times you deliver. {F.OWN_SCANS} {F.NOT_CHARGED}
Nothing about the materials or the state of the system is known in advance.

Passing: for every region, in the target scan with the inversion time you delivered for it, its signal is at most
{100 * t['null_max']:g} % of its signal in the same scan without the inversion pulse. The scanner's log must show that you
measured. An answer in the wrong unit is wrong.""" + ("\n\n" + hints if hints else "")


def submit_schema(cfg: dict) -> dict:
    lo, hi = console.SEQUENCES["ir"]["ti_ms"][:2]
    ids = [r["id"] for r in cfg["task"]["regions"]]
    return {"type": "object", "additionalProperties": False, "required": ["ti_ms"],
            "properties": {"ti_ms": {"type": "object", "additionalProperties": False, "required": ids,
                                     "description": "inversion time of the target scan per region, milliseconds",
                                     "properties": {i: {"type": "number", "minimum": lo, "maximum": hi} for i in ids}}}}


def check(cfg: dict, answer, exam: dict):
    ids = [r["id"] for r in cfg["task"]["regions"]]
    if not isinstance(answer, dict) or set(answer) != {"ti_ms"} or not isinstance(answer["ti_ms"], dict) or set(answer["ti_ms"]) != set(ids):
        return 'the deliverable is {"ti_ms": {' + ", ".join(f'"{i}": <number>' for i in ids) + "}}"
    for i in ids:
        pl = console.plan(target_protocol(cfg["task"], answer["ti_ms"][i]), exam)
        if not pl["accept"]:
            return f"the console rejects the target scan with the inversion time of {i}: " + "; ".join(v["message"] for v in pl["violations"])
    return None


def null_ratio(cfg: dict, region: str, ti_ms: float, world: dict, engine) -> dict:
    """Verifier side: signal of ``region`` in the target scan relative to the same scan without the inversion."""
    t = cfg["task"]
    rp = console.plan(target_protocol(t, ti_ms), world["console"])["realised"]
    se = console.plan(dict(GEOMETRY, sequence="se", tr_ms=rp["tr_ms"], te_ms=rp["te_ms"]), world["console"])["realised"]
    s_ir, s_se = F.region_complex(engine, rp, t["regions"])[region], F.region_complex(engine, se, t["regions"])[region]
    return dict(ratio=float(abs(s_ir) / max(abs(s_se), 1e-30)), sign=float(np.sign((s_ir / s_se).real)))


def true_null_ms(cfg: dict, region: str, world: dict, engine) -> float:
    """Verifier side: the inversion time at which the region's signal changes sign (bisection on the engine)."""
    rp0 = console.plan(target_protocol(cfg["task"], 100.0), world["console"])
    lo, hi = rp0["timing"]["ti_min_ms"] + 0.01, cfg["task"]["target_scan"]["tr_ms"] - rp0["realised"]["te_ms"] - 20.0
    f = lambda ti: null_ratio(cfg, region, ti, world, engine)["sign"]    # noqa: E731
    a = f(lo)
    if a == f(hi):
        return float("nan")
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) == a else (lo, mid)
    return round(0.5 * (lo + hi), 2)


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    from ...scanner import kspace
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    engine = engine or kspace.Engine(world)
    problem = check(cfg, answer, world["console"])
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    eps = cfg["task"]["null_max"]
    ratios = {i: null_ratio(cfg, i, float(answer["ti_ms"][i]), world, engine)["ratio"] for i in answer["ti_ms"]}
    bad = [f"{i} keeps {100 * v:.1f} %" for i, v in ratios.items() if v > eps]
    checks = dict(deliverable_valid=True, **{f"nulled_{i}": v <= eps for i, v in ratios.items()})
    return F.result(checks, "; ".join(bad), error=max(ratios.values()), null_ratio=ratios, null_max=eps,
                    progress=progress(ratios, eps),
                    distance=dict(regions=len(ratios), nulled=len(ratios) - len(bad), worst_ratio=max(ratios.values())))


def progress(ratios: dict, eps: float) -> float:
    """The class-ordered progress of a delivered answer from its residual ratios (see the module docstring)."""
    left = [v for v in ratios.values() if v > eps]
    if not left:
        return 1.0
    close = [min(max(math.log(FAR_RATIO / v) / math.log(FAR_RATIO / eps), 0.0), 1.0) for v in left]
    return F.PROGRESS_FAIL_MAX * (len(ratios) - len(left) + min(sum(close) / len(close), 0.999)) / len(ratios)


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured answer that fails a check: one of ``OUTCOMES``."""
    return "some_nulled" if any(v for k, v in score["checks"].items() if k.startswith("nulled_")) else "none_nulled"


def potential(cfg: dict, events: list) -> float:
    """Trainer side: the potential of ``mrgym.learn.shaping`` after the logged ``events``, in [0, 0.3]: 0.1 per distinct
    inversion time probed with ``ir`` and read out on regions, up to three (what a fit of a - b exp(-TI / T1) needs)."""
    tis = {round(e["realised"]["ti_ms"], 1) for e in events if e["type"] == "scan" and e["realised"]["sequence"] == "ir" and e["rois"]}
    return 0.1 * min(len(tis), 3)


def card() -> dict:
    from ... import learn
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    return dict(
        id="Scanner/NullTissue-v0", title="Null each marked material with inversion recovery", kind="admitted",
        given="A test object with vials of hidden materials; the scanner; a scan-time budget for probing",
        deliver="For each marked region, the inversion time that nulls it in an inversion-recovery scan with a given repetition time",
        scored_by="Noise-free scans of the target protocol on the hidden object: every marked region must be nulled by its inversion time",
        summary="What FLAIR does to fluid and STIR does to fat. The inversion time that nulls a material depends on its "
                "T1, on the repetition time (recovery is incomplete when the next inversion arrives) and on how complete "
                "the inversion is, so it cannot be looked up: it has to be measured on this object and this system.",
        world=dict(phantom="QA phantom (a disc of gel with vials), 1.5 T", sequences_useful=["ir", "se", "spgr"],
                   agent_side_formulas="S(TI) = a - b exp(-TI / T1) at a fixed repetition time; the sign of the signal is in "
                                       "the phase image"),
        hidden=["T1, T2 and proton density of every material", "inversion efficiency", "transmit gain and B1 pattern",
                "noise level", "the 128-bit world seed"],
        told=["the target scan and its repetition time", "the regions", "the pass threshold", "the instrument datasheet",
              "realised protocol, timing and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        practice_levels=[lv for lv, d in LEVELS.items() if d.get("kind") == "practice"],
        learnability=learn.stored("Scanner/NullTissue-v0"),
        limits=dict(max_scanner_requests=40, max_tool_calls=48),
        deliverable={"tool": "submit", "format": {"ti_ms": {"<region id>": "<milliseconds>"}}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan",
                           "the console accepts the target scan with the delivered inversion time"],
                    error="largest over the regions of: signal in the target scan / signal without the inversion pulse (noise-free)",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing="all gates and, for every region, error <= 5 %"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier's scan is noise-free",
    )


LN2 = math.log(2.0)
