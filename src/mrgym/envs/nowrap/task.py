"""``Scanner/NoWrap``: instances and the verifier.

The job is the first decision of any examination: look at a localiser, then prescribe the geometry of the real scan.
The real scan is a spin echo with a fixed repetition time, so its scan time grows with the number of phase-encode
lines. The field of view can be moved and turned, so the fastest geometry puts the phase-encode axis along the
narrowest direction of the object and centres the field of view on it. Size, position and orientation of the object
are hidden: the answer has to be measured.

Deliverable    {"pe_dir": "y", "fov_mm": 184.0, "phase_fov": 0.652174, "matrix": 92, "centre_y_mm": 13.0,
                "centre_x_mm": -7.5, "rotation_deg": 31.0}          and optionally "phase_oversampling"
Pass rule      the console accepts the geometry; the pixel is not larger than required; the whole object lies inside
               the image; a noise-free re-scan of the delivered geometry on the hidden world shows no wrap-around; and
               its scan time is at most ``TOL`` times the scan time of the fastest geometry that meets the first four
               (the optimum of ``oracle.solve``: an exhaustive search on the hidden world, see ``oracle.py``).

Why several coordinates. With a centred, upright field of view the deliverable is one number (the phase extent) and
one bit (the axis), and one constant geometry passes about one instance in seven. A hidden position and a hidden
orientation make the product small: the constant has to guess the size, the offset along the phase axis and the
angle at once.

What "inside the image" means. Pixel ``i`` of an ``n``-pixel axis is centred at ``centre + (i - n // 2) * pixel``, so
along each image axis the image reaches from ``centre - FOV / 2 - pixel / 2`` to ``centre + FOV / 2 - pixel / 2``. The
object is the set of points whose proton-density-weighted signal exceeds ``OBJECT_THRESHOLD`` of the maximum; it is
inside when no point of it lies more than ``COVER_SLACK_MM`` beyond that range.

Levels. 0 and 1 are practice levels (``kind="practice"``): the object is centred and upright, the deliverable has two
effective coordinates and a constant passes too often for a claim. Level 0 states the procedure and has the localiser
already acquired; level 1 states the procedure only. Levels 2 (off-centre) and 3 (off-centre and turned) carry the claim.
"""
from __future__ import annotations

import collections
import json
import math
import os

import numpy as np

from ...scanner import console
from ...scanner import session as core
from ...scanner import world as W
from .. import family as F

FAMILY = "nowrap"
ENV_ID = "Scanner/NoWrap-v1"
SUBMIT_DOC = ("Deliver the geometry of the target spin-echo scan: phase-encode axis, field of view, phase field of view, matrix, "
              "centre and in-plane rotation of the field of view. Lengths are in millimetres, the rotation in degrees.")
TOL = 1.10                                  # a delivered scan may take 10 % longer than the fastest valid one
TARGET = dict(sequence="se", tr_ms=500.0, bandwidth_hz_per_px=300.0)
OBJECT_THRESHOLD = 0.05                     # the object: proton-density-weighted signal above this share of its maximum
SUPPORT_THRESHOLD = 1e-4                    # every spin the engine images (its own support threshold)
COVER_SLACK_MM = 0.5
WRAP_LIMIT = 2e-3                           # energy of (delivered - unwrapped re-scan) / energy of the unwrapped re-scan
REQUIRED = ("pe_dir", "fov_mm", "phase_fov", "matrix", "centre_y_mm", "centre_x_mm", "rotation_deg")
FIELDS = REQUIRED + ("phase_oversampling",)
LOCALISER = dict(sequence="spgr", tr_ms=11.0, flip_deg=10.0, fov_mm=288.0, matrix=160, averages=2)
OUTCOMES = ("object_cut_off", "wrap_around", "pixel_too_large", "too_slow")     # of a delivered, measured answer that fails
BASE = dict(kind="qa", b0_t=1.5, tx_gain_dev=(0.0, 0.05))
# An elongated disc: long semi-axis 74 to 90 mm, short semi-axis = long / (1.6 to 2.0) = 37 to 56 mm. With the largest
# offset the object stays inside the 256 mm master grid at every rotation of the field of view.
_QA = dict(body_radius_mm=(74.0, 90.0), body_aspect=(1.6, 2.0), ring_radius_mm=(16.0, 20.0), vial_radius_mm=(9.0, 11.0), n_ring=(3, 4),
           roi_radius_mm=6.0)
_CENTRED = dict(BASE, qa=dict(_QA, offset_mm=3.0, body_rotation_deg=(0.0, 90.0)))
LEVELS = {
    0: dict(kind="practice", hints=("rule", "first_measurement"), world_spec=_CENTRED, budget_s=41.0, pixel_mm=2.0,
            summary="practice: object centred, long axis along x or y; the procedure is stated and the localiser is already acquired"),
    1: dict(kind="practice", hints=("rule",), world_spec=_CENTRED, budget_s=30.0, pixel_mm=2.0,
            summary="practice: object centred, long axis along x or y (size and orientation hidden); the procedure is stated"),
    2: dict(world_spec=dict(BASE, qa=dict(_QA, offset_mm=25.0, body_rotation_deg=(0.0, 90.0))), budget_s=30.0, pixel_mm=2.0,
            summary="object off-centre by up to 25 mm along each axis, long axis along x or y"),
    3: dict(world_spec=dict(BASE, qa=dict(_QA, offset_mm=25.0)), budget_s=30.0, pixel_mm=2.0,
            summary="object off-centre by up to 25 mm along each axis and turned by any angle"),
}


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    lv = LEVELS[level]
    task = dict(family=FAMILY, target_scan=dict(TARGET, te_ms="console minimum", averages=1, slice_mm=5.0),
                max_pixel_mm=lv["pixel_mm"], max_time_factor=TOL, regions=[],
                deliverable={"pe_dir": "y or x", "fov_mm": "<mm>", "phase_fov": "<fraction, 0.25 to 1>", "matrix": "<even integer>",
                             "centre_y_mm": "<mm>", "centre_x_mm": "<mm>", "rotation_deg": "<degrees, -90 to 90>",
                             "phase_oversampling": "<fraction, 0 to 1; optional>"})
    if lv.get("hints"):
        task["hints"] = list(lv["hints"])
    return dict(task_id=task_id or "nowrap-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed,
                world_spec=lv["world_spec"], budget_s=float(lv["budget_s"]), max_requests=32, task=task)


def prefill(cfg: dict, scanner) -> dict | None:
    """What a level acquires at reset, through the session (in the log, charged to the budget, counts as a
    measurement): the localiser and the extent of the object on it. Called once per episode by ``FamilyEnv``."""
    if "first_measurement" not in cfg["task"].get("hints", ()):
        return None
    r = scanner.scan(dict(LOCALISER))
    return dict(scan_index=r["scan_index"], cost_s=r["cost_s"], box=scanner.view(r["scan_index"])["bounding_box"])


def hint_text(cfg: dict, given: dict | None = None) -> str:
    """The hints of the level, as the paragraphs appended to the statement ("" when the level has none)."""
    hints, px, out = cfg["task"].get("hints", ()), cfg["task"]["max_pixel_mm"], []
    loc = LOCALISER
    if "first_measurement" in hints and given and given.get("box"):
        b = given["box"]
        out.append(f"""A localiser is already acquired: scan {given['scan_index']} (`spgr`, field of view {loc['fov_mm']:g} mm, matrix {loc['matrix']}), {given['cost_s']:.2f} s of
the budget. The object's bright pixels on it reach from y {b['y_mm'][0]:.1f} to {b['y_mm'][1]:.1f} mm and from x {b['x_mm'][0]:.1f} to {b['x_mm'][1]:.1f} mm.""")
    if "rule" in hints:
        first = ("Use the edges given above (or measure them yourself with `profile`)." if "first_measurement" in hints else
                 f"Take a localiser that shows the whole object, for example `spgr` with tr_ms {loc['tr_ms']:g}, flip_deg {loc['flip_deg']:g}, "
                 f"fov_mm {loc['fov_mm']:g}, matrix {loc['matrix']}, averages {loc['averages']}.\n"
                 "     Find the four edges of the object with `profile`: low and high along y, low and high along x (a profile with\n"
                 "     reduce max and a width of the whole image is a projection of the object onto that axis).")
        out.append(f"""How to do it (this level states the procedure; here the long axis of the object lies along x or y).
  1. {first}
  2. Along each axis: middle M = (low + high) / 2 and extent E = high - low + 4 mm (2 mm of margin on each side).
  3. Phase-encode along the axis with the smaller E. With the pixel P = {px:g} mm:
     lines = 2 * ceil(E_phase / (2 * P)), matrix = max(2 * ceil(E_readout / (2 * P)), lines, 40),
     fov_mm = P * matrix, phase_fov = lines / matrix.
  4. centre_y_mm = M_y + P / 2 and centre_x_mm = M_x + P / 2 (the image reaches half a pixel less far on the high side
     than on the low side), rotation_deg = 0.""")
    return "\n\n".join(out)


def statement(cfg: dict, given: dict | None = None) -> str:
    t = cfg["task"]
    hints = hint_text(cfg, given)
    return f"""# Prescribe the fastest scan that shows the whole object without wrap-around
An object of unknown size, position and orientation lies in the
scanner. The examination needs one spin-echo image of the whole object:
    sequence se, tr_ms {TARGET['tr_ms']:g}, te_ms at the console minimum, bandwidth_hz_per_px {TARGET['bandwidth_hz_per_px']:g}, 1 average, slice 5 mm,
    in-plane pixel size at most {t['max_pixel_mm']:g} mm along both axes.
Deliver the geometry of that scan with `submit`: the phase-encode axis (`pe_dir`), the readout field of view (`fov_mm`),
`phase_fov` (the phase field of view as a fraction of `fov_mm`), `matrix`, the centre of the field of view
(`centre_y_mm`, `centre_x_mm`, in mm from the isocentre), its in-plane rotation (`rotation_deg`: image rows turned
from +y towards +x) and, optionally, `phase_oversampling`. The target scan is not run in this episode: the scanner
evaluates the geometry you deliver. {F.OWN_SCANS} {F.NOT_CHARGED}
Nothing about the object is known in advance.

Passing: the console accepts the geometry; the pixel is at most {t['max_pixel_mm']:g} mm; the whole object lies inside the image; the
image shows no wrap-around; and the scan takes at most {round(100 * (t['max_time_factor'] - 1))} % longer than the fastest geometry that meets these
requirements (the field of view may be moved and turned to reach it). The scanner's log must show that you measured.
A length in the wrong unit is wrong.""" + ("\n\n" + hints if hints else "")


def submit_schema(cfg: dict) -> dict:
    g = console.SEQUENCES[TARGET["sequence"]]
    num = lambda k, what: {"type": "integer" if g[k][2] == "i" else "number", "minimum": g[k][0], "maximum": g[k][1], "description": what}   # noqa: E731
    return {"type": "object", "additionalProperties": False, "required": list(REQUIRED),
            "properties": {"pe_dir": {"type": "string", "enum": ["y", "x"], "description": "phase-encode axis: y = image rows, x = image columns"},
                           "fov_mm": num("fov_mm", "readout field of view, mm"),
                           "phase_fov": num("phase_fov", "phase field of view as a fraction of fov_mm"),
                           "matrix": num("matrix", "pixels along the readout, an even number"),
                           "centre_y_mm": num("centre_y_mm", "centre of the field of view along y, mm from the isocentre"),
                           "centre_x_mm": num("centre_x_mm", "centre of the field of view along x, mm from the isocentre"),
                           "rotation_deg": num("rotation_deg", "in-plane rotation of the field of view, degrees: image rows turned from +y towards +x"),
                           "phase_oversampling": num("phase_oversampling", "extra phase field of view as a fraction, acquired and discarded; default 0")}}


def protocol_of(geometry: dict) -> dict:
    """The target scan with a delivered geometry."""
    return dict(TARGET, **{k: geometry[k] for k in FIELDS if k in geometry})


def check(cfg: dict, answer, exam: dict):
    if not isinstance(answer, dict) or not set(REQUIRED) <= set(answer) or set(answer) - set(FIELDS):
        return "the deliverable is {" + ", ".join(f'"{k}"' for k in REQUIRED) + '} and optionally "phase_oversampling"'
    pl = console.plan(protocol_of(answer), exam)
    if not pl["accept"]:
        fov = answer.get("fov_mm")
        hint = " (a field of view of this size looks like centimetres)" if isinstance(fov, (int, float)) and not isinstance(fov, bool) \
            and 0 < fov < console.SEQUENCES[TARGET["sequence"]]["fov_mm"][0] else ""
        return "the console rejects this geometry: " + "; ".join(v["message"] for v in pl["violations"]) + hint
    return None


# ------------------------------------------------------------------------------------------------ the object, verifier side
def outline(world: dict, threshold: float = OBJECT_THRESHOLD) -> np.ndarray:
    """Verifier side: patient coordinates [n, 2] (y, x) in mm of the boundary points of the object on the master grid.
    ``threshold`` = ``OBJECT_THRESHOLD``: the object of the pass rule (proton-density weighted, relative to its maximum);
    ``SUPPORT_THRESHOLD``: every spin the engine images (sum of the class weights)."""
    if threshold == SUPPORT_THRESHOLD:
        m = world["weights"].sum(0) > threshold
    else:
        pd = np.array([world["tissues"][k]["pd"] for k in world["names"]], np.float32)
        m = np.tensordot(pd, world["weights"], 1)
        m = m > threshold * m.max()
    pad = np.pad(m, 1)
    edge = m & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    y, x = W.grid(world)
    return np.stack([np.asarray(y)[edge], np.asarray(x)[edge]], 1).astype(float)


def in_frame(points: np.ndarray, centre_yx, rotation_deg: float) -> tuple:
    """Coordinates (v, u) of patient points along the image rows and columns of a field of view, from its centre."""
    c, s = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
    dy, dx = points[:, 0] - centre_yx[0], points[:, 1] - centre_yx[1]
    return dy * c + dx * s, -dy * s + dx * c


def overhang_mm(points: np.ndarray, realised: dict, grid_mm: float, phase_only: bool = False, oversampled: bool = False) -> float:
    """How far ``points`` reach beyond the image of a realised protocol, in mm (at most 0: all inside). ``grid_mm``:
    spacing of the grid the points live on (half of it is added: a point stands for a grid cell). ``phase_only``:
    along the phase-encode axis only; ``oversampled``: against the encoded phase field of view instead of the image."""
    rot = core.rotation_of(realised)
    v, u = in_frame(points, core.centre_yx(realised), rot)
    half = 0.5 * grid_mm * (abs(math.cos(math.radians(rot))) + abs(math.sin(math.radians(rot))))
    pixel = realised["fov_mm"] / realised["matrix"]
    fov_pe = realised["fov_pe_mm"] * (realised["lines_full"] / realised["matrix_pe"] if oversampled else 1.0)
    along_pe, along_ro = (v, u) if realised["pe_dir"] == "y" else (u, v)
    out = []
    for p, fov in ((along_pe, fov_pe),) + (() if phase_only else ((along_ro, realised["fov_mm"]),)):
        lo, hi = -fov / 2 - pixel / 2, fov / 2 - pixel / 2                # the image along this axis, about the centre
        out += [lo - (float(p.min()) - half), (float(p.max()) + half) - hi]
    return max(out)


def wrap_fraction(engine, realised: dict, support: np.ndarray | None = None) -> float:
    """Energy of the wrap-around in a noise-free scan of ``realised``, relative to the same scan with enough phase
    oversampling to encode every spin. 0 without a scan when the encoded phase field of view already holds every spin."""
    world = engine.w
    support = outline(world, SUPPORT_THRESHOLD) if support is None else support
    grid_mm = world["extent_mm"] / world["n"]
    if overhang_mm(support, realised, grid_mm, phase_only=True, oversampled=True) <= 0:
        return 0.0
    v, u = in_frame(support, core.centre_yx(realised), core.rotation_of(realised))
    reach = float(np.abs(v if realised["pe_dir"] == "y" else u).max()) + realised["fov_mm"] / realised["matrix"] + grid_mm
    need = min(1.0, max(2.0 * reach / realised["fov_pe_mm"] - 1.0, realised["phase_oversampling"]))
    ref_p = console.plan(dict(console.request_of(realised), phase_oversampling=need), world["console"])["realised"]
    img = engine.recon(engine.clean_kspace(realised)[0], realised)
    ref = engine.recon(engine.clean_kspace(ref_p)[0], ref_p)
    return float(np.sum(np.abs(img - ref) ** 2) / max(np.sum(np.abs(ref) ** 2), 1e-30))


def geometry_checks(cfg: dict, answer: dict, world: dict, engine, shapes: dict | None = None, first_failure: bool = False) -> dict:
    """Verifier side: the checks of a deliverable that do not need the optimum, in their declared order, with the
    numbers behind them. ``shapes``: ``dict(object=outline(world), support=outline(world, SUPPORT_THRESHOLD))``.
    ``first_failure``: stop at the first check that fails (the re-scan is then skipped; ``wrap_fraction`` is None)."""
    pl = console.plan(protocol_of(answer), world["console"])
    rp, grid_mm = pl["realised"], world["extent_mm"] / world["n"]
    shapes = shapes or dict(object=outline(world), support=outline(world, SUPPORT_THRESHOLD))
    pixel = rp["fov_mm"] / rp["matrix"]
    over = overhang_mm(shapes["object"], rp, grid_mm)
    checks = dict(deliverable_valid=True, pixel_as_required=pixel <= cfg["task"]["max_pixel_mm"] + 1e-6, whole_object_inside=over <= COVER_SLACK_MM)
    wrap = None
    if not first_failure or all(checks.values()):
        wrap = wrap_fraction(engine, rp, shapes["support"])
        checks["no_wrap_around"] = wrap <= WRAP_LIMIT
    return dict(checks=checks, scan_s=pl["scan_s"], lines=rp["lines"], pixel_mm=round(pixel, 4), overhang_mm=round(over, 3),
                wrap_fraction=wrap)


_OPTIMA: collections.OrderedDict = collections.OrderedDict()


def optimum(cfg: dict, world: dict, engine) -> dict:
    """Verifier side: the fastest geometry that passes ``geometry_checks`` on this world, from the oracle. Kept for
    the last instances scored in this process (the optimum of an instance is a function of its seed and level)."""
    from . import oracle
    key = (cfg["seed"], json.dumps(cfg["world_spec"], sort_keys=True), cfg["task"]["max_pixel_mm"])
    if key not in _OPTIMA:
        if len(_OPTIMA) >= 32:
            _OPTIMA.popitem(last=False)
        s = oracle.solve(cfg, world, engine)
        _OPTIMA[key] = dict(scan_s=s["scan_s"], lines=s["lines"], geometry=s["argmin"])
    return _OPTIMA[key]


DIAGNOSTICS = dict(pixel_as_required="the pixel is larger than required", whole_object_inside="part of the object lies outside the image",
                   no_wrap_around="the image shows wrap-around", scan_time_within_factor="the scan is slower than allowed")


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    from ...scanner import kspace
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    engine = engine or kspace.Engine(world)
    problem = check(cfg, answer, world["console"])
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    g = geometry_checks(cfg, answer, world, engine)
    best = optimum(cfg, world, engine)
    excess = g["scan_s"] / best["scan_s"] - 1.0                           # relative excess scan time
    checks = dict(g["checks"], scan_time_within_factor=g["scan_s"] <= TOL * best["scan_s"] + 1e-9)
    return F.result(checks, "; ".join(DIAGNOSTICS[k] for k, ok in checks.items() if not ok), error=float(excess), tolerance=TOL - 1.0,
                    scan_s=g["scan_s"], scan_s_fastest=best["scan_s"], lines=g["lines"], lines_fastest=best["lines"],
                    pixel_mm=g["pixel_mm"], overhang_mm=g["overhang_mm"], wrap_fraction=g["wrap_fraction"],
                    distance=dict(excess_scan_time=float(excess), overhang_mm=g["overhang_mm"], wrap_fraction=g["wrap_fraction"]))


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured answer that fails a check: one of ``OUTCOMES``."""
    c = score["checks"]
    return ("pixel_too_large" if not c.get("pixel_as_required", True) else "object_cut_off" if not c.get("whole_object_inside", True)
            else "wrap_around" if not c.get("no_wrap_around", True) else "too_slow")


def card() -> dict:
    from ... import learn
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    return dict(
        id=ENV_ID, title="The fastest geometry without wrap-around",
        given="An object of hidden size, position and orientation; the scanner; 30 s of scan time for localisers",
        deliver="Geometry of a spin-echo scan: phase-encode axis, field of view, phase field of view, matrix, centre, rotation",
        scored_by="Re-scan on the hidden object: whole object inside, no wrap-around, pixel as required, scan time within "
                  f"{round(100 * (TOL - 1))} % of the fastest such geometry",
        summary="The first decision of an examination. A hidden test object lies in the 1.5 T scanner. The agent looks (a "
                "localiser costs scan time), then prescribes the geometry of a spin-echo scan whose time grows with the number "
                "of phase-encode lines. The field of view can be moved and turned, so the fastest geometry depends on where "
                "the object lies, how large it is and which way it points.",
        task_kind="measurement",
        task_kind_why="one fixed localiser and a fixed estimator (`open_loop`) is the reference policy itself: no scan depends on an earlier one",
        world=dict(phantom="QA phantom (an elliptical disc of gel with vials), 1.5 T",
                   sequences_useful=["spgr", "se"],
                   agent_side_formulas="scan time = overhead + dummy cycle + TR x phase-encode lines; the readout is oversampled "
                                       "and does not wrap; the phase-encode direction wraps unless the encoded field of view "
                                       "covers the object; pixel i of n is centred at centre + (i - n / 2) x pixel"),
        hidden=["size, position and orientation of the object", "its materials", "fields and system state", "noise level",
                "the 128-bit world seed"],
        told=["the target sequence and the required pixel size", "the pass rule and its time factor", "the instrument datasheet",
              "realised protocol, timing, limits and cost of every plan and scan",
              "any stored image, and the numbers along any line through it (the extent of the object has to be read from them)"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        practice_levels=[lv for lv, d in LEVELS.items() if d.get("kind") == "practice"],
        learnability=learn.stored(ENV_ID),
        limits=dict(max_scanner_requests=32, max_tool_calls=40),
        deliverable={"tool": "submit", "format": {"pe_dir": "y | x", "fov_mm": "<mm>", "phase_fov": "<fraction>", "matrix": "<even integer>",
                                                  "centre_y_mm": "<mm>", "centre_x_mm": "<mm>", "rotation_deg": "<degrees>",
                                                  "phase_oversampling": "<fraction, optional>"}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan", "the console accepts the geometry",
                           "pixel <= the required size",
                           f"the whole object lies inside the image (hidden outline, {COVER_SLACK_MM:g} mm slack)",
                           f"wrap-around energy of a noise-free re-scan <= {WRAP_LIMIT:g} of the image",
                           f"scan time <= {TOL:g} x the scan time of the fastest geometry that passes the gates above (exhaustive search on the hidden world)"],
                    error="T / T_opt - 1: scan time of the delivered geometry over the oracle's optimum",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing="all gates"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier's re-scan is noise-free",
    )
