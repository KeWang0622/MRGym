"""``Scanner/ErnstFlip``: instances and the verifier.

The job: choose the flip angle of a spoiled gradient echo so that each marked vial gives its highest signal. In a
spoiled steady state a small angle tips little magnetisation and a large one saturates it; in between lies a maximum
whose position depends on TR / T1. T1 is hidden, so the angle has to be found on the object.

Two things make the textbook answer arccos(exp(-TR / T1)) wrong here even with a perfectly measured T1:
  slice profile   the target scan is a 2D scan. Its selective pulse turns the spins in the wings of the slice by less
                  than the nominal angle, so the whole slice is at its best at a LARGER nominal angle, about 1.5 times
                  the Ernst angle with the console's default pulse. At the Ernst angle itself a vial gives 92 to 96 %
                  of its maximum.
  transmit scale  (level 3) the actual flip angle is the nominal one times a hidden factor; the nominal angle of the
                  maximum moves by the inverse of that factor.
A sweep of the nominal angle on the target scan itself sees both.

Deliverable    {"flip_deg": {"R1": 14.5, "R2": 31.0, ...}}      one nominal flip angle per marked region, degrees
Pass rule      for every marked region: a noise-free target scan at the delivered angle gives the region at least
               ``pass_fraction`` (99.5 %) of the largest signal it can give in that scan at any flip angle the console
               accepts (1 to 90 degrees).

Why the tolerance is a signal fraction, and why it is this tight. The maximum is flat: 99.5 % of it is held within
about 12 % of the best angle, and the best angle differs between materials by a factor of 4.5 only (T1 from 150 to
3000 ms). So an answer that does not locate the maximum is right for many vials: one angle for the told repetition
time for one vial in five, the better of two scanned angles for nearly two in five, the best of three for more than
half (measured on 50 tuning worlds when the levels were designed; the rows ``fixed_recipe`` and ``brute_force`` of
``mrgym gate`` are the current numbers). Five regions and a budget of two scans bring the better of two scanned angles
to about 1 instance in 100; three regions, or a third scan, would leave it near 1 in 20.

Why the budget is two scans. Every gradient-echo scan costs at least 7 s (3 s of adjustments, 4 s of dummy cycles), so
the budget is a number of scans, and with two the position of the maximum has to be computed from a signal model. The
repetition time of the target stays below 50 ms, the shortest of the single-shot sequence, whose scans cost 3 s: a
sweep of those has another steady state than the target.

What the verifier computes. A region lies inside one homogeneous material, so its signal in the target scan at a
flip angle is the engine's steady state of that material, integrated over the slice profile of the pulse the scan
plays, at the transmit factor of the region (``signal_curves``). The fraction is that signal at the delivered angle
over its largest value on a grid of 0.25 degrees from 1 to 90. No image is formed: the ratio of two noise-free region
means of the same scan at two flip angles is the same number up to what neighbouring materials leak into the region
through the point-spread function, and forming them costs two scans per region. ``image_fractions`` computes the
image version; ``tests/test_ernst_flip.py`` holds the two against each other (they differ by less than 5e-4).
"""
from __future__ import annotations

import math
import os

import numpy as np

from ...scanner import console
from ...scanner import world as W
from .. import family as F
from . import model

FAMILY = "ernst_flip"
ENV_ID = "Scanner/ErnstFlip-v0"
SUBMIT_DOC = "Deliver, for each marked region, the nominal flip angle (degrees) at which the target scan gives it the highest signal."
TARGET = dict(sequence="spgr", matrix=64)                # everything else at the console default: 2D, the normal pulse
FLIP_MIN_DEG, FLIP_MAX_DEG = console.SEQUENCES["spgr"]["flip_deg"][:2]
PASS_FRACTION = 0.995                                    # of the largest signal of the region in the target scan
T1_RANGE_MS = (150.0, 3000.0)                            # told to the agent
TR_RANGE_MS = (12.0, 30.0)                               # of the target scan; below the shortest TR of ``epi`` (50 ms)
OUTCOMES = ("no_region_near_maximum", "some_regions_near_maximum")      # of a delivered, measured answer that fails (worst first)
# Examined competencies this family exercises (rows of ``mrgym.coverage``): choosing the flip angle; at level 3 the
# transmit-gain error decides where the nominal maximum lies.
COMPETENCIES = ("parameter.flip_angle", "qc.transmit_gain")
# The stated method of level 1: two scans of the target at these offsets (in ln of the flip angle) from the best angle
# of a material in the middle of the T1 range, and a table for T1 values on this grid.
PROBE_OFFSETS = (-0.9, 0.8)
TABLE_T1_MS = tuple(float(v) for v in np.round(np.geomspace(60.0, 7500.0, 15), 0))
BASE = dict(kind="qa", b0_t=1.5, tx_gain_dev=(0.0, 0.0), b1_shape_amp=(0.0, 0.0), qa=dict(vial_t1_ms=T1_RANGE_MS, n_ring=(5, 6)))
LEVELS = {
    1: dict(world_spec=BASE, n_regions=5, budget_s=20.0, method=True,
            summary="transmit calibrated; the method is stated: two scans and a table of the 2D signal ratio"),
    2: dict(world_spec=BASE, n_regions=5, budget_s=20.0, method=False,
            summary="transmit calibrated; the method is withheld: the 2D signal curve has to be modelled"),
    3: dict(world_spec=dict(BASE, tx_gain_dev=(0.20, 0.30), b1_shape_amp=(0.01, 0.04)), n_regions=5, budget_s=20.0, method=False,
            summary="as level 2 with a hidden transmit scale (20 to 30 % off): an angle computed from T1 is wrong"),
}


def target_protocol(task: dict, flip_deg) -> dict:
    return dict(TARGET, tr_ms=task["target_scan"]["tr_ms"], flip_deg=flip_deg)


def probe_flips(tr_ms: float) -> list:
    """The two flip angles of the stated method for a repetition time (whole degrees): ``PROBE_OFFSETS`` around the
    best angle of a material in the middle of the told T1 range, from the public model."""
    centre = model.optimum_deg(tr_ms, math.sqrt(T1_RANGE_MS[0] * T1_RANGE_MS[1]))
    return [float(np.clip(round(centre * math.exp(d)), FLIP_MIN_DEG + 1, FLIP_MAX_DEG)) for d in PROBE_OFFSETS]


def method_table(tr_ms: float) -> dict:
    """What level 1 states: the two flip angles and the rows (signal ratio, best flip angle) of the public model."""
    lo, hi = probe_flips(tr_ms)
    return dict(flips_deg=[lo, hi], table=[[round(r, 4), round(a, 1)] for r, a in model.table(tr_ms, lo, hi, TABLE_T1_MS)])


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    lv = LEVELS[level]
    rng = W.rng_for(seed, "task")
    tr = float(np.round(np.exp(rng.uniform(math.log(TR_RANGE_MS[0]), math.log(TR_RANGE_MS[1])))))
    regions = W.make_world(seed, lv["world_spec"], layout_only=True)["regions"]
    targets = [r for r in regions if int(r["id"][1:]) <= lv["n_regions"]]   # region ids are a random permutation of the vials
    task = dict(family=FAMILY, regions=F.public_regions(targets), pass_fraction=PASS_FRACTION, t1_range_ms=list(T1_RANGE_MS),
                target_scan=dict(TARGET, tr_ms=tr, te_ms="console minimum", averages=1, slice_mm=5.0, excitation="selective",
                                 rf_pulse="normal"),
                deliverable={"flip_deg": {"<region id>": "<number, degrees>"}})
    if lv["method"]:
        task["method"] = method_table(tr)
    return dict(task_id=task_id or "ernstflip-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed,
                world_spec=lv["world_spec"], budget_s=float(lv["budget_s"]), max_requests=24, task=task)


def method_text(task: dict) -> str:
    """The stated method of level 1 as the paragraph appended to the statement ("" when the level states none)."""
    m = task.get("method")
    if not m:
        return ""
    lo, hi = m["flips_deg"]
    rows = "\n".join(f"      {r:.4f}    {a:5.1f}" for r, a in m["table"])
    cost = console.plan(dict(sequence="spgr", matrix=32, tr_ms=task["target_scan"]["tr_ms"], flip_deg=hi))["scan_s"]
    return f"""How to do it (this level states the procedure).
  1. Run the target scan twice, with flip_deg {lo:g} and flip_deg {hi:g}. With matrix 32 each scan costs {cost:.2f} s; the
     matrix of your own scans does not change the answer.
  2. For each region take `complex_mean_magnitude` of both scans and compute r = S(flip {hi:g}) / S(flip {lo:g}).
  3. Read the flip angle of the largest signal for that r from the table (interpolate linearly between two rows).
     The table is the steady state S = sin(a) (1 - E) / (1 - E cos(a)), E = exp(-TR / T1), summed over the slice
     profile of the pulse, for this repetition time and these two angles; each row is one T1.
          r      flip_deg of the largest signal
{rows}
  4. Deliver that flip angle for the region.
Do not deliver arccos(exp(-TR / T1)): that is the best angle when every spin sees the same flip angle. In a 2D scan
the best nominal angle is about 1.5 times larger (the slice profile)."""


def statement(cfg: dict) -> str:
    t = cfg["task"]
    s = t["target_scan"]
    ids = ", ".join(r["id"] for r in t["regions"])
    transmit = ("The transmit calibration is correct: at the slice centre the flip angle is the nominal one, everywhere in the object."
                if cfg["world_spec"]["tx_gain_dev"][1] == 0 else
                "The transmit field is not calibrated: at the slice centre the flip angle is the nominal one times an unknown "
                "factor, about the same everywhere in the object.")
    method = method_text(t)
    return f"""# Find the flip angle of the highest signal for each marked vial
A test object with vials of unknown materials lies in the field of view. The examination needs, for each marked region,
a spoiled gradient-echo image in which that region is as bright as it can be. The scan is the same for all of them
except for its flip angle:
    sequence spgr, tr_ms {s['tr_ms']:g}, matrix {s['matrix']}, everything else at the console default
    (a 2D scan with the slice-selective pulse `normal`, te_ms at the console minimum, 1 average, slice 5 mm).
Regions (circles; centre and radius in mm from the isocentre; each lies inside one homogeneous material):
{F.region_table(t['regions'])}
Deliver one nominal flip angle in degrees per region ({ids}) with `submit`. The target scans are not run in this
episode: the scanner evaluates the flip angles you deliver. {F.OWN_SCANS} {F.NOT_CHARGED}
T1 of the materials lies between {t['t1_range_ms'][0]:.0f} and {t['t1_range_ms'][1]:.0f} ms. Nothing else about them is known in advance.
{transmit}
The scan-time budget pays for two gradient-echo scans (each costs at least 7 s), not for three.

Passing: for every region, the target scan at the flip angle you delivered for it gives at least {100 * t['pass_fraction']:g} % of the
largest signal that region can give in this scan at any flip angle from {FLIP_MIN_DEG:g} to {FLIP_MAX_DEG:g} degrees (noise-free). The
maximum is flat: {100 * t['pass_fraction']:g} % of it is held within about 12 % of the best angle. The scanner's log must show that
you measured. An answer in the wrong unit is wrong.""" + ("\n\n" + method if method else "")


def submit_schema(cfg: dict) -> dict:
    ids = [r["id"] for r in cfg["task"]["regions"]]
    return {"type": "object", "additionalProperties": False, "required": ["flip_deg"],
            "properties": {"flip_deg": {"type": "object", "additionalProperties": False, "required": ids,
                                        "description": "nominal flip angle of the target scan per region, degrees",
                                        "properties": {i: {"type": "number", "minimum": FLIP_MIN_DEG, "maximum": FLIP_MAX_DEG} for i in ids}}}}


def check(cfg: dict, answer, exam: dict):
    ids = [r["id"] for r in cfg["task"]["regions"]]
    if not isinstance(answer, dict) or set(answer) != {"flip_deg"} or not isinstance(answer["flip_deg"], dict) \
            or set(answer["flip_deg"]) != set(ids):
        return 'the deliverable is {"flip_deg": {' + ", ".join(f'"{i}": <number>' for i in ids) + "}}"
    for i in ids:
        pl = console.plan(target_protocol(cfg["task"], answer["flip_deg"][i]), exam)
        if not pl["accept"]:
            return f"the console rejects the target scan with the flip angle of {i}: " + "; ".join(v["message"] for v in pl["violations"])
    return None


# ------------------------------------------------------------------------------------------------ verifier side
def signal_curves(cfg: dict, world: dict, engine, flips_deg=None) -> dict:
    """Verifier side: {region: noise-free signal of its material in the target scan, per unit proton density, at the
    nominal flip angles ``flips_deg``} (default ``model.FLIP_GRID_DEG``). The engine's own steady state: integrated over
    the slice profile of the pulse the scan plays, at the transmit factor of the region's centre. One profile table
    serves every region (it is tabulated over the flip angle at the slice centre and read at nominal angle x factor)."""
    from ...scanner import kspace
    flips = model.FLIP_GRID_DEG if flips_deg is None else np.asarray(flips_deg, float)
    rp = console.plan(target_protocol(cfg["task"], 10.0), world["console"])["realised"]
    hidden = {r["id"]: r for r in world["regions"]}
    regions = [hidden[r["id"]] for r in cfg["task"]["regions"]]
    b1 = {r["id"]: W.b1_at(world, r["y_mm"], r["x_mm"]) for r in regions}
    tbw, window = engine.pulse(rp)
    if tbw is None:                                                      # an engine without a slice profile (perturbation studies)
        return {r["id"]: np.abs(kspace.gre_steady(rp, world["tissues"][r["cls"]], np.deg2rad(flips) * b1[r["id"]], engine.tier)[0])
                for r in regions}
    centre = np.deg2rad(np.linspace(0.0, 1.001 * FLIP_MAX_DEG * max(b1.values()), 721))    # at most 0.22 degrees apart
    prof = kspace.slice_profile(centre, tbw, window)
    out = {}
    for r in regions:
        table = np.abs(kspace.slice_table(rp, world["tissues"][r["cls"]], prof, engine.tier)[0])
        out[r["id"]] = np.interp(np.deg2rad(flips) * b1[r["id"]], centre, table)
    return out


def fractions(cfg: dict, flips: dict, world: dict, engine) -> dict:
    """Verifier side: {region: dict(fraction, best_flip_deg)}: the signal of the region's material at ``flips[region]``
    as a fraction of the largest signal it gives at any nominal flip angle from 1 to 90 degrees (0.25 degree grid),
    and the angle of that largest signal (refined by a parabola, two decimals)."""
    grid = model.FLIP_GRID_DEG
    ids = list(flips)
    curves = signal_curves(cfg, world, engine, np.concatenate([grid, [flips[i] for i in ids]]))
    out = {}
    for k, i in enumerate(ids):
        on_grid, got = curves[i][:len(grid)], float(curves[i][len(grid) + k])
        best = round(float(np.clip(model._peak(grid, on_grid), FLIP_MIN_DEG, FLIP_MAX_DEG)), 2)
        out[i] = dict(fraction=got / max(float(on_grid.max()), got, 1e-30), best_flip_deg=best)
    return out


def image_fractions(cfg: dict, flips: dict, world: dict, engine) -> dict:
    """The same fractions read from noise-free IMAGES of the target scan (the mean of each region at the delivered
    angle over its mean at the best angle). Not used for the verdict: it costs two scans per region. A region lies
    inside one homogeneous material, so the two agree up to what neighbouring materials leak into the region through
    the point-spread function (``tests/test_ernst_flip.py`` holds the comparison)."""
    t = cfg["task"]
    best = {i: v["best_flip_deg"] for i, v in fractions(cfg, flips, world, engine).items()}

    def read(angles):
        scans, out = {}, {}
        for i, flip in angles.items():
            if flip not in scans:
                scans[flip] = F.region_complex(engine, console.plan(target_protocol(t, flip), world["console"])["realised"], t["regions"])
            out[i] = float(abs(scans[flip][i]))
        return out
    s_best, s_got = read(best), read(flips)
    return {i: s_got[i] / s_best[i] for i in flips}


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    from ...scanner import kspace
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    engine = engine or kspace.Engine(world)
    problem = check(cfg, answer, world["console"])
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    need = cfg["task"]["pass_fraction"]
    got = fractions(cfg, {i: float(v) for i, v in answer["flip_deg"].items()}, world, engine)
    frac = {i: g["fraction"] for i, g in got.items()}
    bad = [f"{i} gives {100 * v:.1f} % of its maximum" for i, v in frac.items() if v < need]
    checks = dict(deliverable_valid=True, **{f"near_maximum_{i}": v >= need for i, v in frac.items()})
    return F.result(checks, "; ".join(bad), error=1.0 - min(frac.values()), fraction=frac, pass_fraction=need,
                    best_flip_deg={i: g["best_flip_deg"] for i, g in got.items()},
                    distance=dict(regions=len(frac), near_maximum=len(frac) - len(bad), worst_fraction=min(frac.values())))


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured answer that fails a check: one of ``OUTCOMES``."""
    return "some_regions_near_maximum" if any(v for k, v in score["checks"].items() if k.startswith("near_maximum_")) \
        else "no_region_near_maximum"


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
        id=ENV_ID, title="Find the flip angle of the highest signal for each marked vial",
        given="A test object with vials of hidden T1; a spoiled gradient echo at a stated repetition time; a scan-time "
              "budget of two scans",
        deliver="For each marked region, the nominal flip angle at which the 2D target scan gives it the highest signal",
        scored_by=f"Noise-free target scans on the hidden object: at the delivered angle every region must give at least "
                  f"{100 * PASS_FRACTION:g} % of the largest signal it can give at any flip angle",
        summary="The Ernst angle, on a scanner. The flip angle of the highest steady-state signal depends on TR / T1, and "
                "T1 is hidden. The textbook angle arccos(exp(-TR / T1)) is wrong for the 2D target scan even with the "
                "true T1: the slice profile moves the best nominal angle up by about a half, and at level 3 a hidden "
                "transmit scale moves it again. Two scans of the target and a signal model find it. " + kind["sentence"],
        task_kind=kind["kind"], task_kind_why=kind["evidence"], task_kind_levels=kind["levels"], competencies=list(COMPETENCIES),
        world=dict(phantom="QA phantom (a disc of gel with 5 to 7 vials), 1.5 T", sequences_useful=["spgr"],
                   agent_side_formulas="S = sin(a) (1 - E) / (1 - E cos(a)), E = exp(-TR / T1), summed over the slice profile "
                                       "of the selective pulse (datasheet sections 1 and 5); the served signal is the EPG "
                                       "steady state integrated over the slice"),
        hidden=["T1, T2, diffusion coefficient and proton density of every material", "level 3: the transmit gain (20 to 30 % "
                "off) and a mild B1 pattern", "receive profile", "noise level", "the 128-bit world seed"],
        told=["the target scan and its repetition time", "the regions", "the T1 range 150 to 3000 ms", "the pass fraction",
              "whether the transmit calibration is correct", "level 1: two flip angles and the table of the 2D signal ratio",
              "the instrument datasheet", "realised protocol, timing and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        practice_levels=[lv for lv, d in LEVELS.items() if d.get("kind") == "practice"],
        learnability=learn.stored(ENV_ID),
        limits=dict(max_scanner_requests=24, max_tool_calls=32),
        deliverable={"tool": "submit", "format": {"flip_deg": {"<region id>": "<degrees>"}}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan",
                           "the console accepts the target scan with the delivered flip angle (1 to 90 degrees)"],
                    error="largest over the regions of: 1 - signal at the delivered angle / largest signal of the region (noise-free target scan)",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing=f"all gates and, for every region, error <= {100 * (1 - PASS_FRACTION):g} %"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier's scans are noise-free",
    )
