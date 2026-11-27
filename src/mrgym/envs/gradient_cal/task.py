"""``Scanner/GradientCal``: instances and the verifier.

The job: say by how much each in-plane gradient is off, from images of a phantom whose dimensions are known. A
gradient that is s times its nominal strength draws every length along its axis s times too long. Three things stand
between a length read off an image and that number:

    the edge        the edge of the disc lies between two pixels: it is where the profile crosses half of the plateau
                    just inside it, interpolated. Counting the pixels above a threshold is good to one pixel
    the main field  a field error displaces everything along the readout axis by (error in Hz) / (bandwidth per
                    pixel) pixels; a field gradient therefore stretches or compresses the image along the readout,
                    exactly like a gradient error, and never along the phase-encode axis. A length measured along
                    the phase-encode axis, or the mean of two scans with opposite readout polarity, is free of it
    non-linearity   away from the isocentre a gradient is not linear: a length further out is drawn with another
                    scale. The scale at the isocentre is found on a structure near it, or by extrapolating inwards

Deliverable    {"scale_y": 1.023, "scale_x": 0.981}       imaged length / true length along y (rows) and x (columns)
Pass rule      each within ``TOLERANCE`` (0.3 %) of the hidden calibration error of that gradient
               (``world["system"]["grad_scale"]``; the truth is a constant of the hidden system, no scan is needed).

What is hidden and decides the answer: the two scale errors (1 to 5 %, either sign, independent), from level 1 a
linear field gradient along both axes left by a wrong shim (8 to 12 Hz/mm, either sign: even at the console's
strongest readout gradient, 40 mT/m = 1703 Hz/mm, it distorts the readout axis by 0.47 % or more), at level 3 the
non-linearity of the gradients (positions scaled by 1 - g (r / 250 mm)^2 with |g| from 0.08 to 0.14: the edge of the
disc is 1.2 to 2 % off the scale at the isocentre).

Why 0.3 % and two numbers. A pair of constants passes when both hidden errors fall in its window: with errors spread
over 8 % (1 to 5 %, two signs) and a window of 0.6 % that is (0.6 / 8)^2, about 0.6 % of instances; one number alone
would pass 7.5 %. The limit of the phantom guidance is 3 mm on 190 mm (1.6 %): a calibration has to be several times
finer than the limit it serves. What the gate measures for the tuned pair is in the stored gate result, not here.

The phantom is the grid section of the ACR-type phantom (``anatomy.ACR``, slice 5): a disc of 190.0 mm inner
diameter and, inside it, bars of plastic 1.2 mm wide whose centres lie 7.5, 22.5, 37.5 and 52.5 mm either side of the
centre. The diameter is the guidance's; the layout of the bars is declared by this package.

Levels
    0   practice: uniform main field, linear gradients, the procedure stated. One scan is enough
    1   the field gradient is there; the procedure is stated (two scans, each length on its phase-encode axis)
    2   the same worlds; the procedure is withheld
    3   as level 2 with non-linear gradients (told; their size is not): the scale at the isocentre is asked for
"""
from __future__ import annotations

import math
import os

from ...scanner import anatomy
from ...scanner import world as W
from .. import family as F

FAMILY = "gradient_cal"
ENV_ID = "Scanner/GradientCal-v0"
SUBJECT = "C"                            # the system: characterising the instrument
KIND = "measure"
ISOLATES = "gradient calibration: the scale of the image"
SUBMIT_DOC = "Deliver the scale of each in-plane gradient at the isocentre: imaged length / true length along y and along x (ratios near 1)."
COMPETENCIES = ("qc.geometric_accuracy", "artefact.b0_distortion", "parameter.phase_frequency_direction", "parameter.pixel",
                "parameter.receive_bandwidth")
OUTCOMES = ("both_scales_off", "scale_y_off", "scale_x_off")     # of a delivered, measured answer that fails (worst first)
AXES = ("y", "x")
TOLERANCE = 0.003                        # relative: 0.57 mm on the diameter
DIAMETER_MM = float(anatomy.ACR["diameter_mm"])
BAR_MM = float(anatomy.ACR["grid_bar_mm"])
BARS_MM = tuple(float(anatomy.ACR["grid_pitch_mm"] * (k + 0.5)) for k in range(int(anatomy.ACR["grid_half_mm"] // anatomy.ACR["grid_pitch_mm"])))
GRID_N = 1024                            # master grid 0.25 mm: the bars are 1.2 mm wide; smallest pixel 0.5 mm
SCALE_ERROR = (0.010, 0.050)             # |s - 1| of each gradient, the sign drawn
FIELD_GRADIENT_HZ_PER_MM = (8.0, 12.0)   # |gradient| of the main field along each axis, the sign drawn (levels 1 to 3)
NONLINEARITY = (0.08, 0.14)              # |g| of the non-linearity, the sign drawn (level 3)
SYSTEM = dict(tx_gain_dev=(0.0, 0.03), cf_offset_hz=5.0, sigma0=0.0065, sigma0_log_sd=0.05, grad_delay_us=0.4)   # a healthy system otherwise
LEVELS = {
    0: dict(kind="practice", field=False, nonlinear=False, hints=("method",), budget_s=45.0,
            summary="practice: uniform main field, linear gradients, the procedure stated; one scan is enough"),
    1: dict(field=True, nonlinear=False, hints=("method",), budget_s=75.0,
            summary="a field gradient distorts the readout axis; the procedure is stated (two scans, each length on its phase-encode axis)"),
    2: dict(field=True, nonlinear=False, budget_s=75.0,
            summary="the same worlds; the procedure is withheld"),
    3: dict(field=True, nonlinear=True, budget_s=75.0,
            summary="as level 2 with non-linear gradients: the scale at the isocentre is asked for"),
}


def _signed(rng, lo, hi) -> float:
    return float(rng.uniform(lo, hi) * (1 if rng.uniform() < 0.5 else -1))


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    lv = LEVELS[level]
    rng = W.rng_for(seed, "task")
    field = (1.5, 3.0)[int(rng.integers(2))]
    error = {a: _signed(rng, *SCALE_ERROR) for a in AXES}
    shim = [_signed(rng, *FIELD_GRADIENT_HZ_PER_MM) for _ in AXES]      # drawn at every level, used from level 1
    gnl = _signed(rng, *NONLINEARITY)                                    # drawn at every level, used at level 3
    spec = dict(SYSTEM, kind="acr", n=GRID_N, b0_t=field, acr=dict(slice=5), grad_scale=dict(error))
    if lv["field"]:
        spec["acr_fields"] = dict(shim_hz_per_mm=shim)
    if lv["nonlinear"]:
        spec["gnl"] = gnl
    task = dict(family=FAMILY, field_strength_t=field,
                phantom=dict(inner_diameter_mm=DIAMETER_MM, bar_width_mm=BAR_MM, bar_centres_mm=list(BARS_MM),
                             position="centred on the isocentre within 1 mm, turned by at most 0.3 degrees"),
                told=dict(main_field_uniform=not lv["field"], gradients_linear=not lv["nonlinear"]),
                tolerance_percent=100 * TOLERANCE,
                deliverable=dict(scale_y="<imaged / true length along y>", scale_x="<imaged / true length along x>"))
    if lv.get("hints"):
        task["hints"] = list(lv["hints"])
    return dict(task_id=task_id or "gradientcal-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed, world_spec=spec,
                budget_s=float(lv["budget_s"]), max_requests=40, task=task,
                hidden=dict(scale={a: round(1.0 + error[a], 6) for a in AXES}))


def hint_text(cfg: dict) -> str:
    """The hints of the level, as the paragraph appended to the statement ("" when the level has none)."""
    t = cfg["task"]
    if "method" not in t.get("hints", ()):
        return ""
    edge = """Edge: where the profile crosses half of the plateau value just inside that edge (the mean of the samples 4 to 12 mm
     inside it); interpolate linearly between the two samples either side of the crossing."""
    if t["told"]["main_field_uniform"]:
        return f"""How to do it (this level states the procedure).
  1. One scan with pixels of 1 mm or smaller (for example se, tr_ms 60, fov_mm 250, matrix 256).
  2. `profile` along y through the centre of the phantom (axis y, position_mm = the x of its centre, width_mm 6), and
     along x through its centre.
  3. {edge}
  4. scale_y = (distance between the two edges along y) / {DIAMETER_MM:g} mm; scale_x the same along x."""
    return f"""How to do it (this level states the procedure).
  1. A field error displaces everything along the readout axis, by (error in Hz) / bandwidth_hz_per_px pixels, and never
     along the phase-encode axis. So measure each length along the phase-encode axis: one scan with pe_dir y for scale_y,
     one with pe_dir x for scale_x (for example se, tr_ms 60, fov_mm 250, matrix 384, bandwidth_hz_per_px 300).
  2. In the pe_dir y scan: `profile` along y through the centre of the phantom (axis y, position_mm = the x of its
     centre, width_mm 6). In the pe_dir x scan: along x through its centre.
  3. {edge}
  4. scale_y = (distance between the two edges along y) / {DIAMETER_MM:g} mm; scale_x the same along x."""


def statement(cfg: dict) -> str:
    t = cfg["task"]
    ph, told = t["phantom"], t["told"]
    bars = ", ".join(f"{v:g}" for v in ph["bar_centres_mm"])
    known = []
    if told["main_field_uniform"]:
        known.append("The main field is uniform over the phantom.")
    known.append("The gradients are linear over the phantom." if told["gradients_linear"] else
                 "The gradients of this scanner are not linear: away from the isocentre lengths are drawn at another scale, and the change "
                 "of scale grows, to leading order, with the square of the distance from the isocentre. The scale asked for is the one "
                 "at the isocentre.")
    hints = hint_text(cfg)
    return f"""# Calibrate the two in-plane gradients
The grid section of a test phantom lies at the isocentre of a {t['field_strength_t']:g} T scanner ({ph['position']}): a disc of
solution, {ph['inner_diameter_mm']:.1f} mm inner diameter. Inside it lies a grid of plastic bars without signal, {ph['bar_width_mm']:g} mm wide: bars along x and
bars along y, their centres {bars} mm either side of the centre of the phantom.
The in-plane gradients are miscalibrated: a gradient that is 2 % too strong draws every length along its axis 2 % too
long, in every protocol.
Deliver with `submit` the scale of each gradient at the isocentre: scale_y = imaged length / true length along y
(image rows), scale_x the same along x (image columns). A calibrated gradient has 1.000.
Your scans may use any sequence and any parameters the console accepts.
{' '.join(known)} Nothing else about the state of the system is known in advance.

Passing: scale_y and scale_x each within {t['tolerance_percent']:g} % of the true scale of that gradient ({TOLERANCE * DIAMETER_MM:.2f} mm on the diameter).
The scanner's log must show that you measured. An answer in the wrong unit is wrong (a ratio near 1; not a percentage, not a length).""" \
        + ("\n\n" + hints if hints else "")


def submit_schema(cfg: dict) -> dict:
    # no range in the schema: a value in another unit reaches ``check``, whose refusal says what the number is
    number = lambda axis: {"type": "number", "description": f"imaged length / true length along {axis} at the isocentre, a ratio near 1"}   # noqa: E731
    return {"type": "object", "additionalProperties": False, "required": ["scale_y", "scale_x"],
            "properties": {"scale_y": number("y (image rows)"), "scale_x": number("x (image columns)")}}


SHAPE = 'the deliverable is {"scale_y": <number>, "scale_x": <number>}: imaged length / true length, ratios near 1'


def check(cfg: dict, answer, exam: dict):
    if not isinstance(answer, dict) or set(answer) != {"scale_y", "scale_x"}:
        return SHAPE
    for k in ("scale_y", "scale_x"):
        v = answer[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0.5 <= v <= 2.0:
            return f"{k} must be a number from 0.5 to 2 (imaged length / true length; a calibrated gradient has 1.000)"
    return None


def true_scale(world: dict) -> dict:
    """Verifier side: the calibration error of the session, {axis: gradient strength / nominal}."""
    return dict(zip(AXES, (float(v) for v in world["system"]["grad_scale"])))


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    problem = check(cfg, answer, world["console"])
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    tol = cfg["task"]["tolerance_percent"] / 100.0
    want = true_scale(world)
    off = {a: float(answer[f"scale_{a}"]) / want[a] - 1.0 for a in AXES}
    ok = {a: abs(off[a]) <= tol for a in AXES}
    notes = [f"scale_{a}: delivered {float(answer[f'scale_{a}']):.5f}, the gradient's is {want[a]:.5f} ({100 * off[a]:+.2f} %)" for a in AXES if not ok[a]]
    worst = float(max(abs(v) for v in off.values()))
    return F.result(dict(deliverable_valid=True, **{f"scale_{a}": ok[a] for a in AXES}), "; ".join(notes), error=worst,
                    scale={a: round(want[a], 6) for a in AXES}, tolerance=tol,
                    distance=dict(axes=len(AXES), within=sum(ok.values()), worst_relative_error=round(worst, 6)))


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured answer that fails a check: one of ``OUTCOMES``."""
    c = score["checks"]
    if not c.get("scale_y", True) and not c.get("scale_x", True):
        return "both_scales_off"
    return "scale_y_off" if not c.get("scale_y", True) else "scale_x_off"


def card() -> dict:
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    return dict(
        id=ENV_ID, title="Calibrate the two in-plane gradients", kind="candidate",
        given="The grid section of a phantom of known dimensions in a scanner whose in-plane gradients are miscalibrated, with a "
              "hidden field gradient and (level 3) non-linear gradients; a scan-time budget for three short scans",
        deliver="The scale of each in-plane gradient at the isocentre: imaged length / true length along y and along x",
        scored_by="The hidden calibration error of each gradient: each delivered scale must agree with it",
        summary="The geometry test of quality control as a calibration. A length read off an image is the true length times "
                "the gradient's scale only where the edge is found between the pixels, where the main field does not "
                "displace it (the phase-encode axis, or a pair of opposite readout polarities), and where the gradient "
                "is linear (near the isocentre).",
        task_kind="measurement",
        task_kind_why="the reference is a fixed scan list (two scans with swapped axes) with a fixed estimator; `open_loop` plays it: no scan depends on an earlier one",
        competencies=list(COMPETENCIES),
        world=dict(phantom="ACR-type phantom, grid section (world kind acr, slice 5), 1.5 T or 3 T", sequences_useful=["se", "spgr"],
                   agent_side_formulas="displacement along the readout = field error / bandwidth per pixel, in pixels; "
                                       "edge at half of the local plateau, interpolated; scale = imaged / true length "
                                       "(levels 0 and 1 state the procedure)"),
        hidden=["the calibration error of each gradient (1 to 5 %, either sign)", "from level 1: a field gradient along both axes (8 to 12 Hz/mm)",
                "level 3: the size and the sign of the gradient non-linearity", "position of the phantom (within 1 mm, 0.3 degrees)",
                "transmit and receive profiles, noise level", "the 128-bit world seed"],
        told=["the inner diameter of the phantom and where its bars are", "whether the gradients are linear over the phantom",
              "level 0: that the main field is uniform", "the tolerance", "levels 0 and 1: the procedure",
              "the instrument datasheet", "realised protocol, timing and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        practice_levels=[lv for lv, d in LEVELS.items() if d.get("kind") == "practice"],
        limits=dict(max_scanner_requests=40, max_tool_calls=64),
        deliverable={"tool": "submit", "format": dict(scale_y="<ratio near 1>", scale_x="<ratio near 1>")},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan"],
                    error="larger of |delivered / true - 1| over the two gradients",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing=f"all gates and error <= {100 * TOLERANCE:g} % for both gradients"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier reads the hidden calibration error and scans nothing",
    )
