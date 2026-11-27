"""``Scanner/FoldAway``: instances and the verifier.

The job: a field of view smaller than the object, placed so that the wrap-around misses what the examination needs.
The scan time of the target spin echo grows with the number of phase-encode lines, lines = phase field of view /
pixel, and the pixel is prescribed. A phase field of view that holds the whole object (the answer of
``Scanner/NoWrap``) is safe and slow. A phase field of view of L mm shows the object repeated every L mm along the
phase-encode axis; the copies may land anywhere except on the marked region. With the object reaching from o_lo to
o_hi along that axis ON THE LINES THROUGH THE REGION, and the region from r_lo to r_hi, the copy shifted by +L begins
at o_lo + L and the copy shifted by -L ends at o_hi - L, so the region is free of them exactly when

    L > r_hi - o_lo   and   L > o_hi - r_lo                                  (``fold_free_extent``).

Where the object ends relative to the region is hidden: it has to be measured. Four regions (A to D) are examined,
each with its own scan.

Deliverable    {"A": {"pe_dir": "x", "phase_fov": 0.4375, "centre_y_mm": 12.0, "centre_x_mm": -31.5}, "B": {...}, "C": {...}, "D": {...}}
               and at level 3 "rotation_deg" in each
Pass rule      for each region, on a noise-free re-scan of the delivered geometry on the hidden world:
                 1. the region lies inside the image
                 2. no aliased signal on the region: every pixel of the region differs from the same pixel of the twin
                    scan without wrap-around by at most ``WRAP_MAX`` of the region's mean signal in the twin
                 3. the scan takes at most ``TOL`` times the scan time of the fastest geometry that meets 1 and 2
                    (``oracle.solve``: a search of the console's geometries on the hidden world)

The twin without wrap-around (``unwrapped``). The same k-space extent sampled more densely: phase-encode lines are
added between the delivered ones until the encoded field of view holds every spin, and the reconstruction discards
what lies outside the image (the console's ``phase_oversampling`` does this up to a factor of 2; the twin is built
directly). The number of lines of the twin is a multiple of the delivered number, so the delivered image is exactly
the twin's uncropped image added to itself at shifts of L (a test does the sum): what the check measures is the
aliased signal and nothing else. That signal includes the truncation ringing of a folded edge, which reaches a few
pixels beyond the edge (about 1 / (pi^2 d) of the step at d pixels): a copy that ends at the rim of the region still
rings into it. This is why a margin is part of the method.

The rule holds on every line through the region, with the ends of the object and the length of the region on that
line: ``oracle.extent`` is the largest of these, the exact threshold for a circle. Taken once, with the outermost
ends of the object over the width of the region and its diameter, it is the method level 1 states: exact where the
edge of the object runs across the lines, and on the safe side by radius x (1 + k - sqrt(1 + k^2)) where it runs
obliquely with slope k (the circle is shorter on the line on which the object is longer). (With the ends on the one
line through the centre of the region the rule is short by radius x (sqrt(1 + k^2) - 1), which is unsafe for a
region beside a steep edge.)

Why four regions. One region gives one hidden number per axis and one bit; an answer from the task text alone (the
regions tell roughly where the object is) then passes one region in five to ten, and the regions of one object are
not independent: they share its size. Four regions in one object make the product small, and the localiser that
serves one serves the others.

Not delivered, on purpose. The readout field of view and the matrix: the readout does not wrap (it is oversampled
and filtered) and costs no scan time, so 192 mm at 192 pixels is part of the stated scan and holds any phase field
of view a region needs.

Levels. 1: the long axis of the object lies along x or y, the method is stated. 2: the method is withheld. 3: the
object is turned by any angle, and the field of view may be turned as well (``rotation_deg``): the fastest geometry
phase-encodes along the direction in which the region is nearest to both ends of the object.
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
from ..nowrap import task as N

FAMILY = "fold_away"
ENV_ID = "Scanner/FoldAway-v0"
SUBMIT_DOC = ("Deliver, for each marked region, the geometry of the target spin-echo scan: phase-encode axis, phase field of view (a "
              "fraction of the readout field of view) and the centre of the field of view in millimetres (at level 3 also its rotation in degrees).")
TARGET = dict(sequence="se", tr_ms=500.0, bandwidth_hz_per_px=300.0, fov_mm=192.0, matrix=192)
PIXEL_MM = TARGET["fov_mm"] / TARGET["matrix"]
BASE_FIELDS = ("pe_dir", "phase_fov", "centre_y_mm", "centre_x_mm")
REGIONS = ("A", "B", "C", "D")
REGION_RADIUS_MM = 5.0
WRAP_MAX = 0.05                             # largest |delivered - twin| of a pixel of the region, as a share of the region's mean in the twin
TOL = 1.10                                  # a delivered scan may take 10 % longer than the fastest valid one
OUTCOMES = ("region_cut_off", "wrap_on_region", "too_slow")       # of a delivered, measured answer that fails
REQUIREMENTS = ("region_inside_image", "no_wrap_on_region", "scan_time_within_factor")
# examined competencies this family exercises (ids of ``mrgym.coverage.COMPETENCIES``)
COMPETENCIES = ("parameter.fov", "parameter.phase_frequency_direction", "option.rectangular_fov", "option.anti_aliasing",
                "artefact.aliasing", "artefact.gibbs")
BASE = dict(kind="qa", b0_t=1.5, tx_gain_dev=(0.0, 0.05))
# A homogeneous elongated disc of gel: long semi-axis 88 to 100 mm, short semi-axis = long / (1.2 to 1.9) = 46 to 83 mm,
# its centre up to 20 mm from the isocentre along each axis: inside the 256 mm master grid at every rotation. (The
# wide range of the short axis is what keeps an answer from the task text alone below one instance in a hundred.)
_QA = dict(body_radius_mm=(88.0, 100.0), body_aspect=(1.2, 1.9), n_ring=(0, 0), p_centre=0.0, offset_mm=20.0)
_UPRIGHT = dict(BASE, qa=dict(_QA, body_rotation_deg=(0.0, 90.0)))
REGION_ALONG = 0.4                          # a region lies within this share of the long semi-axis from the centre of the object
REGION_INSET_MM = 5.0                       # ... and at least this far inside its edge
REGION_APART_MM = 25.0                      # smallest distance between the centres of two regions
LEVELS = {
    1: dict(hints=("rule",), world_spec=_UPRIGHT, budget_s=30.0, rotation=False,
            summary="the long axis of the object lies along x or y (hidden); the method is stated"),
    2: dict(world_spec=_UPRIGHT, budget_s=30.0, rotation=False,
            summary="the long axis of the object lies along x or y (hidden); the method is withheld"),
    3: dict(world_spec=dict(BASE, qa=dict(_QA)), budget_s=30.0, rotation=True,
            summary="the object is turned by any angle; the field of view may be turned as well"),
}
LOCALISER = dict(sequence="spgr", tr_ms=11.0, flip_deg=10.0, fov_mm=288.0, matrix=160, averages=2)
NO_PIXEL = 1e9                              # ``aliased_share`` of a region that holds no pixel centre of the image
RULE_MARGIN_PX = 2.0                        # pixels of the target scan added to the fold-free extent (the method level 1 states)


def fields(cfg_or_task: dict) -> tuple:
    """The fields of one delivered geometry of this instance."""
    t = cfg_or_task.get("task", cfg_or_task)
    return BASE_FIELDS + (("rotation_deg",) if t.get("rotation") else ())


def fold_free_extent(o_lo: float, o_hi: float, r_lo: float, r_hi: float) -> float:
    """The phase field of view above which no copy of an object that reaches from ``o_lo`` to ``o_hi`` along the
    phase-encode axis lands on a region from ``r_lo`` to ``r_hi`` (see the module docstring), in the unit of its arguments."""
    return max(o_hi - r_lo, r_hi - o_lo)


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    lv = LEVELS[level]
    body = W.make_world(seed, lv["world_spec"], layout_only=True)["body"]
    rng = W.rng_for(seed, "task")
    c, s = math.cos(math.radians(body["rot"])), math.sin(math.radians(body["rot"]))
    a, b = body["a"], body["b"]                                          # long and short semi-axis
    ring = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
    reach = REGION_RADIUS_MM + REGION_INSET_MM
    placed = []
    while len(placed) < len(REGIONS):                                    # (u along the long axis, v along the short one), from the centre of the object
        u, v = float(rng.uniform(-REGION_ALONG, REGION_ALONG) * a), float(rng.uniform(-1.0, 1.0) * b)
        inside = np.all(((u + reach * np.cos(ring)) / a) ** 2 + ((v + reach * np.sin(ring)) / b) ** 2 < 1.0)
        if inside and all(math.hypot(u - p[0], v - p[1]) >= REGION_APART_MM for p in placed):
            placed.append((u, v))
    regions = [dict(id=name, y_mm=round(body["y"] + u * c + v * s, 1), x_mm=round(body["x"] - u * s + v * c, 1), radius_mm=REGION_RADIUS_MM)
               for name, (u, v) in zip(REGIONS, placed)]
    one = {"pe_dir": "y or x", "phase_fov": "<fraction, 0.25 to 1>", "centre_y_mm": "<mm>", "centre_x_mm": "<mm>"}
    if lv["rotation"]:
        one["rotation_deg"] = "<degrees, -90 to 90>"
    task = dict(family=FAMILY, target_scan=dict(TARGET, te_ms="console minimum", averages=1, slice_mm=5.0), pixel_mm=PIXEL_MM,
                regions=regions, rotation=bool(lv["rotation"]), wrap_max=WRAP_MAX, max_time_factor=TOL, deliverable={"<region id>": one})
    if lv.get("hints"):
        task["hints"] = list(lv["hints"])
    return dict(task_id=task_id or "foldaway-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed,
                world_spec=lv["world_spec"], budget_s=float(lv["budget_s"]), max_requests=32, task=task)


def hint_text(cfg: dict) -> str:
    """The method, as the paragraph appended to the statement of a level that states it ("" otherwise)."""
    if "rule" not in cfg["task"].get("hints", ()):
        return ""
    loc, px = LOCALISER, PIXEL_MM
    return f"""How to do it (this level states the method).
  1. Take one localiser that shows the whole object: `spgr` with tr_ms {loc['tr_ms']:g}, flip_deg {loc['flip_deg']:g}, fov_mm {loc['fov_mm']:g},
     matrix {loc['matrix']}, averages {loc['averages']}.
  2. For each region and each axis (y, then x): the ends of the object on the lines through the region. `profile`
     along that axis, position_mm = the region's coordinate on the other axis, width_mm = the region's diameter + 2,
     reduce max. o_lo and o_hi are where it crosses half of its plateau. The region reaches from r_lo = centre - radius
     to r_hi = centre + radius along that axis.
  3. A phase field of view of L mm repeats the object every L mm along the phase-encode axis. The copy shifted by +L
     begins at o_lo + L, the copy shifted by -L ends at o_hi - L. Both miss the region when L > r_hi - o_lo and
     L > o_hi - r_lo. A folded edge rings into its neighbourhood, so keep {RULE_MARGIN_PX:g} pixels:
     L = max(r_hi - o_lo, o_hi - r_lo) + {RULE_MARGIN_PX * px:g} mm.
  4. Phase-encode along the axis with the smaller L. lines = 2 x ceil(L / {2 * px:g}), phase_fov = lines / {TARGET['matrix']} (at least 0.25).
     Centre the field of view on the region: centre_y_mm, centre_x_mm = the region's centre."""


def statement(cfg: dict) -> str:
    t = cfg["task"]
    turned = t["rotation"]
    where = ("The object is turned by an unknown angle." if turned else "The long axis of the object lies along x or along y.")
    extra = (",\nand its in-plane rotation (`rotation_deg`: image rows turned from +y towards +x, about the centre of the field of view)"
             if turned else "")
    fixed = "" if turned else ", not rotated"
    hints = hint_text(cfg)
    return f"""# Prescribe the fastest scan in which no wrap-around lands on the marked region
A homogeneous test object lies in the scanner. It is larger than the field of view the examination can afford.
{where}
Four regions inside it are examined, each with its own spin-echo scan:
    sequence se, tr_ms {TARGET['tr_ms']:g}, te_ms at the console minimum, bandwidth_hz_per_px {TARGET['bandwidth_hz_per_px']:g}, fov_mm {TARGET['fov_mm']:g}, matrix {TARGET['matrix']}
    (pixel {t['pixel_mm']:g} mm), 1 average, slice 5 mm, all of k-space sampled, no phase oversampling{fixed}.
Regions (circles; centre and radius in mm from the isocentre):
{F.region_table(t['regions'])}
Deliver one geometry per region ({', '.join(r['id'] for r in t['regions'])}) with `submit`: the phase-encode axis (`pe_dir`), `phase_fov` (the phase field
of view as a fraction of fov_mm; the scan has {TARGET['matrix']} x phase_fov phase-encode lines), the centre of the field of view
(`centre_y_mm`, `centre_x_mm`, in mm from the isocentre){extra}.
Wrap-around is allowed anywhere except on the region. The target scans are not run in this episode: the scanner
evaluates the geometries you deliver.
{F.OWN_SCANS}
{F.NOT_CHARGED}
Where the object ends is not known in advance.

Passing, for each region, on the scan with the geometry you delivered for it:
  the region lies inside the image;
  no aliased signal lands on the region: every pixel of the region differs by at most {100 * t['wrap_max']:g} % of the region's mean
  signal from the same pixel of the same scan without wrap-around;
  the scan takes at most {round(100 * (t['max_time_factor'] - 1))} % longer than the fastest geometry that meets the two requirements above.
The scanner's log must show that you measured. A length in the wrong unit is wrong.""" + ("\n\n" + hints if hints else "")


def _geometry_schema(names: tuple) -> dict:
    g = console.SEQUENCES[TARGET["sequence"]]
    num = lambda k, what: {"type": "number", "minimum": g[k][0], "maximum": g[k][1], "description": what}   # noqa: E731
    props = {"pe_dir": {"type": "string", "enum": ["y", "x"], "description": "phase-encode axis"},       # (the unit is in the name;
             "phase_fov": num("phase_fov", "fraction of fov_mm"),                                            # the task text says what each is)
             "centre_y_mm": num("centre_y_mm", "from the isocentre"), "centre_x_mm": num("centre_x_mm", "from the isocentre"),
             "rotation_deg": num("rotation_deg", "rows from +y towards +x")}
    return {"type": "object", "additionalProperties": False, "required": list(names), "properties": {k: props[k] for k in names}}


def submit_schema(cfg: dict) -> dict:
    ids = [r["id"] for r in cfg["task"]["regions"]]
    return {"type": "object", "additionalProperties": False, "required": ids,
            "properties": {i: dict(_geometry_schema(fields(cfg)), description=f"geometry of the target scan for region {i}") for i in ids}}


def protocol_of(geometry: dict) -> dict:
    """The target scan with one delivered geometry."""
    return dict(TARGET, **geometry)


def check(cfg: dict, answer, exam: dict):
    ids, names = [r["id"] for r in cfg["task"]["regions"]], fields(cfg)
    shape = "the deliverable is {" + ", ".join(f'"{i}": {{' + ", ".join(f'"{k}"' for k in names) + "}" for i in ids) + "}"
    if not isinstance(answer, dict) or set(answer) != set(ids):
        return shape
    for i in ids:
        g = answer[i]
        if not isinstance(g, dict) or set(g) != set(names):
            return shape
        pl = console.plan(protocol_of(g), exam)
        if not pl["accept"]:
            pf = g.get("phase_fov")
            hint = " (phase_fov is a fraction of fov_mm, not millimetres or percent)" if isinstance(pf, (int, float)) and not isinstance(pf, bool) and pf > 1 else ""
            return f"the console rejects the geometry of region {i}: " + "; ".join(v["message"] for v in pl["violations"]) + hint
    return None


# ------------------------------------------------------------------------------------------------ the measures, verifier side
def region_overhang_mm(region: dict, realised: dict) -> float:
    """How far the circle ``region`` reaches beyond the image of a realised protocol, in mm (at most 0: inside). Pixel i
    of an n-pixel axis is centred at centre + (i - n // 2) * pixel, so the image reaches from -FOV / 2 - pixel / 2 to
    FOV / 2 - pixel / 2 about the centre of the field of view, along its own rows and columns."""
    v, u = N.in_frame(np.array([[region["y_mm"], region["x_mm"]]], float), core.centre_yx(realised), core.rotation_of(realised))
    pixel = realised["fov_mm"] / realised["matrix"]
    along_pe, along_ro = (float(v[0]), float(u[0])) if realised["pe_dir"] == "y" else (float(u[0]), float(v[0]))
    out = []
    for p, fov in ((along_pe, realised["fov_pe_mm"]), (along_ro, realised["fov_mm"])):
        out += [(-fov / 2 - pixel / 2) - (p - region["radius_mm"]), (p + region["radius_mm"]) - (fov / 2 - pixel / 2)]
    return max(out)


def unwrapped(realised: dict, world: dict, support: np.ndarray | None = None) -> dict:
    """The twin of a realised protocol without wrap-around (see the module docstring): phase-encode lines added, in a
    multiple of the delivered number, until the encoded field of view holds every spin from the centre of this field
    of view. ``support``: ``nowrap.task.outline(world, SUPPORT_THRESHOLD)``. The protocol itself when it already does."""
    support = N.outline(world, N.SUPPORT_THRESHOLD) if support is None else support
    v, u = N.in_frame(support, core.centre_yx(realised), core.rotation_of(realised))
    pixel = realised["fov_mm"] / realised["matrix"]
    reach = float(np.abs(v if realised["pe_dir"] == "y" else u).max()) + pixel + world["extent_mm"] / world["n"]
    factor = int(math.ceil(2.0 * reach / (pixel * realised["lines_full"]) - 1e-9))
    if factor <= 1:
        return realised
    full = factor * realised["lines_full"]
    return dict(realised, lines_full=full, lines=full)


def aliased_share(engine, realised: dict, region: dict, support: np.ndarray | None = None) -> float:
    """Verifier side: the largest |delivered - twin| over the pixels of ``region`` in a noise-free scan of ``realised``,
    as a share of the mean |twin| over those pixels. 0 without a scan when the encoded field of view holds every spin;
    ``NO_PIXEL`` when no pixel centre lies in the region."""
    twin = unwrapped(realised, engine.w, support)
    if twin is realised:
        return 0.0
    img = engine.acquire(realised)[0]
    ref = engine.acquire(twin)[0]
    y, x = core.pixel_grid(img.shape, core.fov_yx(realised), core.centre_yx(realised), core.rotation_of(realised))
    mask = (y - region["y_mm"]) ** 2 + (x - region["x_mm"]) ** 2 <= region["radius_mm"] ** 2
    if not mask.any():
        return NO_PIXEL
    return float(np.abs(img - ref)[mask].max() / max(float(np.abs(ref)[mask].mean()), 1e-30))


def region_checks(cfg: dict, region: dict, geometry: dict, world: dict, engine, support: np.ndarray | None = None, first_failure: bool = False) -> dict:
    """Verifier side: requirements 1 and 2 for one region, in their declared order, with the numbers behind them.
    ``first_failure``: stop at the first requirement that fails (the re-scan is then skipped; ``aliased`` is None)."""
    pl = console.plan(protocol_of(geometry), world["console"])
    rp = pl["realised"]
    over = region_overhang_mm(region, rp)
    checks, aliased = dict(region_inside_image=over <= 1e-9), None
    if not first_failure or checks["region_inside_image"]:
        aliased = aliased_share(engine, rp, region, support)
        checks["no_wrap_on_region"] = aliased <= cfg["task"]["wrap_max"]
    return dict(checks=checks, scan_s=pl["scan_s"], lines=rp["lines"], phase_fov_mm=rp["fov_pe_mm"], overhang_mm=round(over, 3), aliased=aliased)


_OPTIMA: collections.OrderedDict = collections.OrderedDict()


def optimum(cfg: dict, world: dict, engine) -> dict:
    """Verifier side: per region, the fastest geometry that meets requirements 1 and 2 on this world, from the oracle.
    Kept for the last instances scored in this process (the optimum of an instance is a function of its seed and level)."""
    from . import oracle
    key = (cfg["seed"], json.dumps(cfg["world_spec"], sort_keys=True), json.dumps(cfg["task"]["regions"], sort_keys=True), cfg["task"]["rotation"])
    if key not in _OPTIMA:
        if len(_OPTIMA) >= 32:
            _OPTIMA.popitem(last=False)
        _OPTIMA[key] = oracle.solve(cfg, world, engine)["regions"]
    return _OPTIMA[key]


DIAGNOSTICS = dict(region_inside_image="the region does not lie inside the image", no_wrap_on_region="aliased signal lands on the region",
                   scan_time_within_factor="the scan is slower than allowed")


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    from ...scanner import kspace
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    engine = engine or kspace.Engine(world)
    problem = check(cfg, answer, world["console"])
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    support, best = N.outline(world, N.SUPPORT_THRESHOLD), optimum(cfg, world, engine)
    regions = {r["id"]: r for r in cfg["task"]["regions"]}
    rows = {i: region_checks(cfg, r, answer[i], world, engine, support) for i, r in regions.items()}
    for i in regions:
        rows[i]["checks"]["scan_time_within_factor"] = rows[i]["scan_s"] <= TOL * best[i]["scan_s"] + 1e-9
        rows[i]["excess"] = rows[i]["scan_s"] / best[i]["scan_s"] - 1.0
    checks = dict(deliverable_valid=True, **{f"{name}_{i}": rows[i]["checks"][name] for name in REQUIREMENTS for i in regions})   # requirement by requirement
    first = {i: next((name for name in REQUIREMENTS if not rows[i]["checks"][name]), None) for i in regions}    # per region: the first that fails
    bad = [f"region {i}: {DIAGNOSTICS[name]}" for i, name in first.items() if name]
    excess = max(r["excess"] for r in rows.values())
    return F.result(checks, "; ".join(bad), error=float(excess), tolerance=TOL - 1.0,
                    regions={i: dict(scan_s=r["scan_s"], scan_s_fastest=best[i]["scan_s"], lines=r["lines"], lines_fastest=best[i]["lines"],
                                     phase_fov_mm=r["phase_fov_mm"], aliased=r["aliased"], overhang_mm=r["overhang_mm"]) for i, r in rows.items()},
                    distance=dict(excess_scan_time=float(excess), largest_aliased=max(r["aliased"] for r in rows.values()),
                                  largest_overhang_mm=max(r["overhang_mm"] for r in rows.values())))


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured answer that fails a check: the first requirement that fails for a region."""
    labels = dict(zip(REQUIREMENTS, OUTCOMES))
    return next((labels[k.rsplit("_", 1)[0]] for k, ok in score["checks"].items() if not ok and k.rsplit("_", 1)[0] in labels), OUTCOMES[-1])


def card() -> dict:
    from ... import gate, learn
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    kind = F.kind_by_open_loop(gate.load(ENV_ID))
    return dict(
        id=ENV_ID, title="A small field of view whose wrap-around misses the region",
        given="A homogeneous object of hidden size, position and orientation; four marked regions inside it; the scanner; 30 s of scan time for looking",
        deliver="Per region, the geometry of a spin-echo scan: phase-encode axis, phase field of view, centre (level 3: rotation)",
        scored_by="Re-scan on the hidden object against a twin without wrap-around: no aliased signal on the region, the region "
                  f"inside the image, scan time within {round(100 * (TOL - 1))} % of the fastest such geometry",
        summary="An object larger than the field of view folds over. The examination needs four regions free of wrap-around and "
                "does not care where else the image folds. The agent looks (a localiser costs scan time), finds where the object "
                "ends on the lines through each region, and prescribes the smallest phase field of view whose copies of the "
                "object miss the region. A field of view that holds the whole object is valid and too slow.",
        task_kind=kind["kind"], task_kind_why=kind["evidence"], task_kind_levels=kind["levels"], competencies=list(COMPETENCIES),
        world=dict(phantom="QA phantom (a homogeneous elliptical disc of gel), 1.5 T", sequences_useful=["spgr", "se"],
                   agent_side_formulas="lines = matrix x phase_fov; scan time = overhead + TR x (1 + lines); a phase field of view of "
                                       "L repeats the object every L along the phase-encode axis: the region (r_lo, r_hi) is free of "
                                       "the copies of an object (o_lo, o_hi) when L > r_hi - o_lo and L > o_hi - r_lo; the readout "
                                       "does not wrap"),
        hidden=["size, position and orientation of the object", "its material", "fields and system state", "noise level", "the 128-bit world seed"],
        told=["the target sequence with its readout field of view and matrix", "the four regions", "the pass rule and its time factor",
              "the instrument datasheet", "realised protocol, timing, limits and cost of every plan and scan",
              "any stored image, and the numbers along any line through it (the edges of the object have to be read from them)"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        practice_levels=[lv for lv, d in LEVELS.items() if d.get("kind") == "practice"],
        learnability=learn.stored(ENV_ID),
        limits=dict(max_scanner_requests=32, max_tool_calls=40),
        deliverable={"tool": "submit", "format": {"<region id>": {"pe_dir": "y | x", "phase_fov": "<fraction>", "centre_y_mm": "<mm>",
                                                               "centre_x_mm": "<mm>", "rotation_deg": "<degrees, level 3 only>"}}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan", "the console accepts each geometry",
                           "the region lies inside the image",
                           f"largest difference of a pixel of the region between the noise-free re-scan and its twin without wrap-around <= {WRAP_MAX:g} of the region's mean",
                           f"scan time <= {TOL:g} x the scan time of the fastest geometry that passes the gates above (search on the hidden world)"],
                    error="largest over the regions of T / T_opt - 1: scan time of the delivered geometry over the oracle's optimum",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing="all gates, for every region"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier's re-scan and its twin are noise-free",
    )
