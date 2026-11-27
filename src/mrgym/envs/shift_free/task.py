"""``Scanner/ShiftFree``: instances and the verifier.

The physics item: the readout is a time axis. Fat precesses a few hundred hertz below water, so along the
frequency-encode axis it appears displaced by (fat-water frequency difference) / (bandwidth per pixel) pixels, towards
the side the readout direction sets. A higher bandwidth shrinks the displacement and costs signal-to-noise ratio
(noise grows with the square root of the bandwidth): the bandwidth should be enough and no more.

The job. A gel object holds discs of fat. Each marked region lies in the gel with a fat disc on either side along x
(level 3: also along y), at hidden distances. For each region one spin-echo scan is delivered: its bandwidth, its
readout direction and (level 3) its phase-encode axis, so that the displaced fat stays off the region at the highest
signal-to-noise ratio that allows.

Deliverable    {"scans": {"R1": {"bandwidth_hz_per_px": 120.0, "ro_polarity": -1}, "R2": {...}, "R3": {...}}}
               (level 3: each scan also carries "pe_dir")
Pass rule      for every region, on a noise-free scan of the target protocol with the delivered settings:
                 fat_off     |mean fat signal in the region| <= ``COVER_MAX`` x |mean water signal in the region|
                 snr_floor   the region's signal-to-noise ratio >= the stated floor
                 near_best   that ratio >= ``NEAR_BEST`` x the highest ratio of any setting that satisfies fat_off

What is hidden and why it decides. The frequency difference is no secret (field strength and fat shift are on the
datasheet, and every ``plan`` prints the shift of the main fat line in pixels; the hidden field at a fat border moves
it by a few percent, which a measured displacement includes). What decides is where the fat borders lie: on which
side of a region the fat has more room, and how much. The displacement that just reaches the region sets the
bandwidth, the side sets the readout direction, and at level 3 the axis with the most room sets the frequency-encode
axis. The borders have to be measured on images in which the fat itself is displaced, so the displacement has to be
measured or computed and taken out. The noise level is hidden too; the stated floor is each region's ratio at
``FLOOR_BW``, a statement about signal and noise that says nothing of the borders.

How the verifier forms its images. Image formation is linear in the per-class echo amplitudes, so the engine is asked
for the k-space of the fat classes alone and of the other classes alone (``Engine.amplitudes`` / ``Engine.kspace`` /
``Engine.recon``): the fat image and the water image of the same scan, the twin pair of this family. Each is averaged
over the area of the region's circle (``area_weights``: a pixel on the edge counts with the share of it inside), as a
complex mean: the truncation ripples of a fat edge beside the region alternate in sign and do not count as fat on it.
Noise is analytic (``Engine.noise_sd``: the standard deviation of one component of a pixel).

The best setting. ``best`` searches, per region, every readout direction (and axis) over a bandwidth grid of 2 % steps
from the console minimum: the fat share of a region falls as the bandwidth rises (asserted on a coarse grid across
the threshold in ``tests/test_shift_free.py``), so the lowest bandwidth that satisfies fat_off is found by bisection,
and the best signal-to-noise ratio is the one at the lowest such bandwidth over the settings. The result depends on
the instance only and is kept per instance. Where the threshold lies is checked against a model of its own (lens
areas of two circles, six fat lines, the spin-echo contrast) in ``tests/scoring/shift_free.py``.

Why several regions. One region's answer is a bandwidth inside a window of 29 % (``NEAR_BEST`` squared) and one of
two (level 3: four) directions, while the bandwidth the widest gap needs varies by a factor of 2.7 between regions: a
constant answer hits it about once in eight tries. Three regions with independent borders make that product small
while each window stays wide enough for a careful measurement.
"""
from __future__ import annotations

import collections
import math
import os
import threading

import numpy as np

from ...scanner import console
from ...scanner import session as core
from ...scanner import tissue as T
from ...scanner import world as W
from .. import family as F

FAMILY = "shift_free"
ENV_ID = "Scanner/ShiftFree-v0"
SUBJECT = "B"                           # spatial encoding and reconstruction
KIND = "set"                            # the deliverable is a setting of the scanner
ISOLATES = "chemical-shift displacement along the frequency-encode axis: (fat-water frequency difference) / (bandwidth per pixel) pixels"
SUBMIT_DOC = ("Deliver, for each marked region, the readout settings of its scan: the bandwidth (Hz per pixel), the readout "
              "direction and, where the task leaves it free, the phase-encode axis.")
TARGET = dict(sequence="se", tr_ms=500.0, te_ms=30.0, fov_mm=240.0, matrix=192)
PIXEL_MM = TARGET["fov_mm"] / TARGET["matrix"]
GRID_N = 512                            # master grid of the worlds: 0.5 mm, 2.5 points per pixel of the target scan
COVER_MAX = 0.05                        # fat_off: the mean fat signal in the region is at most this share of the mean water signal
NEAR_BEST = 0.88                        # near_best: the signal-to-noise ratio is at least this share of the best one under fat_off
FLOOR_BW = 600.0                        # the stated signal-to-noise floor is the region's ratio at this bandwidth (it says nothing of the borders)
BW_STEP, BW_TOP = 1.02, 2000.0          # the bandwidth grid of ``best``: 2 % steps from the console minimum
BW_GRID = tuple(round(console.SEQUENCES["se"]["bandwidth_hz_per_px"][0] * BW_STEP ** k, 2)
                for k in range(int(math.log(BW_TOP / console.SEQUENCES["se"]["bandwidth_hz_per_px"][0]) / math.log(BW_STEP)) + 1))
SUBSAMPLE = 8                           # points per pixel and axis with which the verifier weighs a pixel on the edge of a region
REGION_MM, FAT_MM = 5.0, 9.0            # radius of a marked region and of a fat disc
SITES = ((-27.0, -40.0), (-27.0, 40.0), (36.0, 0.0))     # where the regions lie (y, x), each moved by up to JITTER_MM
JITTER_MM = 3.0
ROOM_MM = (3.2, 9.0)                    # the larger gap between a region and a fat disc (log-uniform)
OTHER_MM, OTHER_SHARE = 1.6, 0.6        # every other gap: from OTHER_MM to OTHER_SHARE x the larger one
SIDES = {"x-": (0.0, -1.0), "x+": (0.0, 1.0), "y-": (-1.0, 0.0), "y+": (1.0, 0.0)}   # unit vector (y, x) from a region to a disc
REACH_MM = 0.25                         # the stated allowance: rule fat_off is reached 0 to 0.75 mm after the main fat line has crossed the gap
SIGMA0 = (0.02, 0.045)                  # receiver noise density of a world (log-uniform): the noise level is hidden
FIELD_T = 3.0
FAT_T1_SPREAD = 0.15                    # every fat disc is a fat of its own: T1 of subcutaneous fat x (1 +- this), drawn per disc
OUTCOMES = ("fat_on_region", "snr_below_floor", "bandwidth_too_high")     # of a delivered, measured answer that fails, worst first
# Examined competencies this family exercises (rows of ``mrgym.coverage``).
COMPETENCIES = ("parameter.receive_bandwidth", "parameter.phase_frequency_direction", "artefact.chemical_shift")
LEVELS = {
    1: dict(sides=("x-", "x+"), free=("bandwidth_hz_per_px", "ro_polarity"), hints=("method",), budget_s=110.0,
            summary="fat on both sides of each region along x, the method stated: the room on each side decides direction and bandwidth"),
    2: dict(sides=("x-", "x+"), free=("bandwidth_hz_per_px", "ro_polarity"), budget_s=110.0,
            summary="the worlds of level 1, the method withheld"),
    3: dict(sides=("x-", "x+", "y-", "y+"), free=("bandwidth_hz_per_px", "ro_polarity", "pe_dir"), budget_s=210.0,
            summary="fat on four sides: the frequency-encode axis is free and matters, along one axis the fat has more room"),
}
_kept = collections.OrderedDict()       # per instance: what ``measure`` and ``best`` found (functions of the instance alone)
_kept_lock = threading.Lock()
KEPT, MEASURED_KEPT = 32, 512


def approaching(pe_dir: str, ro_polarity: int) -> str:
    """The side of a region from which fat moves towards it: the readout runs along x when the phase is encoded
    along y; with ``ro_polarity`` 1 fat (below the water frequency) is displaced towards positive positions."""
    axis = "x" if pe_dir == "y" else "y"
    return axis + ("-" if ro_polarity == 1 else "+")


def target_protocol(task: dict, scan: dict) -> dict:
    """The target scan with the delivered settings of one region."""
    p = dict(TARGET, pe_dir=task["target_scan"].get("pe_dir") or scan.get("pe_dir"),
             bandwidth_hz_per_px=scan.get("bandwidth_hz_per_px"), ro_polarity=scan.get("ro_polarity"))
    return p


def world_spec(sites: list, sigma0: float) -> dict:
    """The world of an instance: a gel disc with a fat disc at every drawn gap of every site."""
    fat = T.table(FIELD_T)["subcut_fat"]
    oil = lambda t1: dict(pd=fat["pd"], t1_ms=t1, t2_ms=fat["t2_ms"], t2p_ms=round(fat["t2p_ms"], 3), adc_um2_ms=fat["adc_um2_ms"], fat=True)   # noqa: E731
    inserts = [dict(y_mm=round(s["y_mm"] + SIDES[side][0] * (REGION_MM + g + FAT_MM), 3),
                    x_mm=round(s["x_mm"] + SIDES[side][1] * (REGION_MM + g + FAT_MM), 3), radius_mm=FAT_MM, tissue=oil(s["fat_t1_ms"][side]))
               for s in sites for side, g in s["gap_mm"].items()]
    return dict(kind="qa", n=GRID_N, b0_t=FIELD_T, sigma0=float(sigma0),
                qa=dict(n_ring=(0, 0), p_centre=0.0, body_radius_mm=(93.0, 98.0), offset_mm=2.0, inserts=inserts))


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    from ...scanner import kspace
    lv = LEVELS[level]
    rng = W.rng_for(seed, "task")
    sites = []
    for k, (y0, x0) in enumerate(SITES):
        y, x = (round(float(v + rng.uniform(-JITTER_MM, JITTER_MM)), 1) for v in (y0, x0))
        room = float(np.exp(rng.uniform(*np.log(ROOM_MM))))
        wide = lv["sides"][int(rng.integers(0, len(lv["sides"])))]
        gaps = {side: round(room if side == wide else float(rng.uniform(OTHER_MM, max(OTHER_SHARE * room, OTHER_MM + 0.05))), 2)
                for side in lv["sides"]}
        t1 = {side: round(float(T.table(FIELD_T)["subcut_fat"]["t1_ms"] * (1.0 + FAT_T1_SPREAD * rng.uniform(-1.0, 1.0))), 2) for side in lv["sides"]}
        sites.append(dict(id=f"R{k + 1}", y_mm=y, x_mm=x, radius_mm=REGION_MM, gap_mm=gaps, fat_t1_ms=t1))
    spec = world_spec(sites, float(np.exp(rng.uniform(*np.log(SIGMA0)))))
    regions = F.public_regions(sites)
    # the stated floor: each region's signal-to-noise ratio at FLOOR_BW (a statement about signal and noise, not about the borders)
    world = W.make_world(seed, spec)
    engine = kspace.Engine(world, cache=1)
    rp = console.plan(dict(TARGET, pe_dir="y", bandwidth_hz_per_px=FLOOR_BW, ro_polarity=1), world["console"])["realised"]
    water = _means(_part(engine, rp, fat=False), rp, regions)
    floor = {r["id"]: round(float(math.floor(10.0 * abs(water[r["id"]]) / engine.noise_sd(rp)) / 10.0), 1) for r in regions}
    scan = {"bandwidth_hz_per_px": "<number, Hz per pixel>", "ro_polarity": "<1 or -1>"}
    if "pe_dir" in lv["free"]:
        scan["pe_dir"] = "<y or x>"
    task = dict(family=FAMILY, regions=regions, cover_max=COVER_MAX, near_best=NEAR_BEST, snr_floor=floor, free=list(lv["free"]),
                target_scan=dict(TARGET, averages=1, slice_mm=5.0, **({} if "pe_dir" in lv["free"] else dict(pe_dir="y"))),
                deliverable={"scans": {"<region id>": scan}})
    if lv.get("hints"):
        task["hints"] = list(lv["hints"])
    return dict(task_id=task_id or "shiftfree-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed, world_spec=spec,
                budget_s=float(lv["budget_s"]), max_requests=48, task=task, truth=dict(sites=sites))


# ------------------------------------------------------------------------------------------------ the task text
PROBE_TEXT = dict(sequence="se", tr_ms=150.0, matrix=TARGET["matrix"], bandwidth_hz_per_px=150.0)


def hint_text(cfg: dict) -> str:
    if "method" not in cfg["task"].get("hints", ()):
        return ""
    p = PROBE_TEXT
    return f"""How to do it (this level states the method).
  1. Scan twice at one low bandwidth, once with ro_polarity 1 and once with -1 (for example se, tr_ms {p['tr_ms']:g}, matrix {p['matrix']},
     bandwidth_hz_per_px {p['bandwidth_hz_per_px']:g}; fov_mm and pe_dir as in the target scan).
  2. On both scans take a `profile` along x through the centre of a region. Read where the fat on each side begins: where
     the signal crosses the level half a fat signal above what lies in front of the fat (gel, or the dark gap the fat left).
  3. The two readout directions move a fat border by the same distance d in opposite directions. Its true position is the
     mean of the two readings and d is half their difference.
  4. The room on a side is the distance from the true fat border to the edge of the region. With ro_polarity 1 fat moves
     towards +x: the fat on the -x side comes closer. With -1 the fat on the +x side does.
  5. Take the direction whose fat has more room. The displacement is d x (probe bandwidth) / bandwidth. Rule 1 is reached
     when it is about the room plus {REACH_MM:g} mm: deliver the bandwidth whose displacement is 0.9 x (room + {REACH_MM:g} mm)."""


def statement(cfg: dict) -> str:
    t = cfg["task"]
    s = t["target_scan"]
    axis_free = "pe_dir" in t["free"]
    where = "one on each of its four sides, along x and along y" if axis_free else "one on each side along x"
    chosen = "bandwidth_hz_per_px, ro_polarity and pe_dir" if axis_free else "bandwidth_hz_per_px and ro_polarity"
    fixed = "" if axis_free else f", pe_dir {s['pe_dir']}"
    rows = "\n".join(f"  {r['id']}: y {r['y_mm']:g}, x {r['x_mm']:g}, radius {r['radius_mm']:g}, signal-to-noise floor {t['snr_floor'][r['id']]:g}"
                     for r in t["regions"])
    any_setting = "bandwidth, readout direction and phase-encode axis" if axis_free else "bandwidth and readout direction"
    hints = hint_text(cfg)
    return f"""# Keep displaced fat off each marked region
A disc of gel holds discs of fat. Each marked region lies in the gel between fat discs: {where}.
In an image fat is displaced along the frequency-encode axis. Where it lands on a region it adds to the region's signal.
For each region the examination needs a spin-echo scan of its own in which the fat stays off that region, at the
highest signal-to-noise ratio this allows:
    sequence se, tr_ms {s['tr_ms']:g}, te_ms {s['te_ms']:g}, fov_mm {s['fov_mm']:g}, matrix {s['matrix']}, 1 average, slice 5 mm{fixed};
    you choose {chosen}.
Regions (circles; centre and radius in mm from the isocentre):
{rows}
Deliver the settings of every region's scan with `submit`. The target scans are not run in this episode: the scanner
evaluates the settings you deliver, without noise.
{F.OWN_SCANS}
{F.NOT_CHARGED}
Where the fat discs lie is not known in advance.

Passing, for every region, in the scan delivered for it:
  1. the mean fat signal over the area of the region is at most {100 * t['cover_max']:g} % of its mean water signal;
  2. the region's signal-to-noise ratio (its mean water signal / standard deviation of the noise in the real part of a pixel) is at least its floor;
  3. that ratio is at least {100 * t['near_best']:g} % of the highest one that any {any_setting} reaches under rule 1.
The scanner's log must show that you measured. A value in another unit is wrong.""" + ("\n\n" + hints if hints else "")


def submit_schema(cfg: dict) -> dict:
    t = cfg["task"]
    lo, hi = console.SEQUENCES["se"]["bandwidth_hz_per_px"][:2]
    props = {"bandwidth_hz_per_px": {"type": "number", "minimum": lo, "maximum": hi, "description": "receiver bandwidth, Hz per pixel"},
             "ro_polarity": {"type": "integer", "enum": [1, -1], "description": "direction of the readout"}}
    if "pe_dir" in t["free"]:
        props["pe_dir"] = {"type": "string", "enum": ["y", "x"], "description": "phase-encode axis (the readout runs along the other one)"}
    scan = {"type": "object", "additionalProperties": False, "required": list(t["free"]), "properties": props}
    ids = [r["id"] for r in t["regions"]]
    return {"type": "object", "additionalProperties": False, "required": ["scans"],
            "properties": {"scans": {"type": "object", "additionalProperties": False, "required": ids,
                                     "description": "the readout settings of each region's scan",
                                     "properties": {i: scan for i in ids}}}}


def check(cfg: dict, answer, exam: dict):
    t = cfg["task"]
    ids, free = [r["id"] for r in t["regions"]], list(t["free"])
    shape = '{"scans": {' + ", ".join(f'"{i}": {{' + ", ".join(f'"{k}": ...' for k in free) + "}" for i in ids) + "}}"
    if not isinstance(answer, dict) or set(answer) != {"scans"} or not isinstance(answer["scans"], dict) or set(answer["scans"]) != set(ids):
        return "the deliverable is " + shape
    for i in ids:
        scan = answer["scans"][i]
        if not isinstance(scan, dict) or set(scan) != set(free):
            return f"the scan of {i} needs exactly " + ", ".join(free) + ": the deliverable is " + shape
        pl = console.plan(target_protocol(t, scan), exam)
        if not pl["accept"]:
            return f"the console rejects the scan of {i}: " + "; ".join(v["message"] for v in pl["violations"])
    return None


# ------------------------------------------------------------------------------------------------ the verifier
def _part(engine, rp, fat: bool):
    """Verifier side: the complex image of the fat classes alone (``fat`` True) or of every other class alone, for one
    realised protocol (image formation is linear in the per-class amplitudes; the two add up to the noise-free scan)."""
    w = engine.w
    amps = [engine.amplitudes(rp, c)[0] for c in range(len(w["names"]))]
    keep = [bool(w["tissues"][name]["fat"]) == fat for name in w["names"]]
    return engine.recon(engine.kspace(rp, rp["echo_times_ms"][0], [a if k else np.zeros_like(a) for a, k in zip(amps, keep)]), rp)


def area_weights(rp, region, sub: int = SUBSAMPLE):
    """Share of every pixel of the image of ``rp`` that lies inside the circle ``region`` (``sub`` x ``sub`` points per
    pixel). A region read by pixel centres has an edge that jumps by up to half a pixel with its position on the pixel
    grid; the verifier's region is the circle itself."""
    shape = (rp["matrix_pe"], rp["matrix"]) if rp["pe_dir"] == "y" else (rp["matrix"], rp["matrix_pe"])
    y, x = core.pixel_positions(shape, core.fov_yx(rp), core.centre_yx(rp))
    o = (np.arange(sub) + 0.5) / sub - 0.5
    yy = (y[:, None] + o[None, :] * (y[1] - y[0])).ravel()
    xx = (x[:, None] + o[None, :] * (x[1] - x[0])).ravel()
    inside = (yy[:, None] - region["y_mm"]) ** 2 + (xx[None, :] - region["x_mm"]) ** 2 <= region["radius_mm"] ** 2
    return inside.reshape(len(y), sub, len(x), sub).mean((1, 3))


def _means(image, rp, regions) -> dict:
    """Complex mean of ``image`` over the area of every region."""
    out = {}
    for r in regions:
        m = area_weights(rp, r)
        out[r["id"]] = complex(np.sum(m * image) / np.sum(m))
    return out


def _instance_store(cfg: dict) -> dict:
    """What is kept per instance: the measurements made on it and, once searched, the best settings."""
    key = (cfg["seed"], cfg["level"])
    with _kept_lock:
        if key not in _kept:
            _kept[key] = dict(measured=collections.OrderedDict())
            while len(_kept) > KEPT:
                _kept.popitem(last=False)
        return _kept[key]


def measure(cfg: dict, region: dict, scan: dict, world: dict, engine) -> dict:
    """Verifier side: fat share and signal-to-noise ratio of ``region`` in the target scan with the settings ``scan``
    (kept per instance and setting: the images are noise-free, so a repeated question has the same answer)."""
    rp = console.plan(target_protocol(cfg["task"], scan), world["console"])["realised"]
    kept = _instance_store(cfg)["measured"]
    key = (region["id"], rp["pe_dir"], rp["ro_polarity"], rp["bandwidth_hz_per_px"])
    with _kept_lock:
        if key in kept:
            return dict(kept[key], realised=rp)
    f = abs(_means(_part(engine, rp, fat=True), rp, [region])[region["id"]])
    w = abs(_means(_part(engine, rp, fat=False), rp, [region])[region["id"]])
    out = dict(cover=float(f / max(w, 1e-30)), snr=float(w / engine.noise_sd(rp)))
    with _kept_lock:
        kept[key] = out
        while len(kept) > MEASURED_KEPT:
            kept.popitem(last=False)
    return dict(out, realised=rp)


def lowest_clean(cfg: dict, region: dict, pe_dir: str, ro_polarity: int, world: dict, engine, top: int | None = None):
    """Verifier side: index in ``BW_GRID`` of the lowest bandwidth at which ``region`` satisfies fat_off with this
    axis and readout direction, among the indices up to ``top`` (default: the whole grid); None when there is none.
    Bisection: the fat share falls as the bandwidth rises."""
    clean = lambda k: measure(cfg, region, dict(bandwidth_hz_per_px=BW_GRID[k], ro_polarity=ro_polarity, pe_dir=pe_dir),   # noqa: E731
                              world, engine)["cover"] <= COVER_MAX
    hi = len(BW_GRID) - 1 if top is None else top
    if hi < 0 or not clean(hi):
        return None
    if clean(0):
        return 0
    lo = 0                                                               # not clean at lo, clean at hi
    site = next(s for s in cfg["truth"]["sites"] if s["id"] == region["id"])
    room = site["gap_mm"].get(approaching(pe_dir, ro_polarity))
    if room is not None:                                                 # a first guess from the drawn gap narrows the bracket; the bisection decides
        f_main = abs(console.FAT_PPM) * 1e-6 * console.GAMMA_HZ_PER_T * world["b0_t"]
        guess = f_main * PIXEL_MM / (room + REACH_MM)
        for k in (int(np.searchsorted(BW_GRID, guess / 1.12)), int(np.searchsorted(BW_GRID, guess * 1.12))):
            if lo < k < hi:
                lo, hi = (lo, k) if clean(k) else (k, hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if clean(mid) else (mid, hi)
    return hi


def best(cfg: dict, world: dict, engine) -> dict:
    """Verifier side: per region, the setting with the highest signal-to-noise ratio under fat_off, searched over
    the readout directions, the axes the task leaves free and ``BW_GRID``: {id: dict(bandwidth_hz_per_px, ro_polarity,
    pe_dir, snr)}. Kept per instance."""
    store = _instance_store(cfg)
    if "best" in store:
        return store["best"]
    t = cfg["task"]
    axes = ("y", "x") if "pe_dir" in t["free"] else (t["target_scan"]["pe_dir"],)
    out = {}
    for region, site in zip(t["regions"], cfg["truth"]["sites"]):
        combos = sorted(((pe, pol) for pe in axes for pol in (1, -1)), key=lambda c: -site["gap_mm"][approaching(*c)])
        found = None                                                     # (grid index, pe_dir, ro_polarity)
        for pe, pol in combos:                                           # the widest gap first; the others only have to be beaten
            k = lowest_clean(cfg, region, pe, pol, world, engine, top=None if found is None else found[0] - 1)
            if k is not None:
                found = (k, pe, pol)
        if found is None:
            out[region["id"]] = None
            continue
        scan = dict(bandwidth_hz_per_px=BW_GRID[found[0]], ro_polarity=found[2], pe_dir=found[1])
        out[region["id"]] = dict(scan, snr=measure(cfg, region, scan, world, engine)["snr"])
    with _kept_lock:
        store["best"] = out
    return out


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    from ...scanner import kspace
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    engine = engine or kspace.Engine(world)
    problem = check(cfg, answer, world["console"])
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    t = cfg["task"]
    top = best(cfg, world, engine)
    checks, rows, bad = dict(deliverable_valid=True), {}, []
    for region in t["regions"]:
        i = region["id"]
        m = measure(cfg, region, answer["scans"][i], world, engine)
        ref = top[i]["snr"] if top[i] else 0.0
        rows[i] = dict(cover=m["cover"], snr=m["snr"], snr_floor=t["snr_floor"][i], snr_best=ref,
                       best=None if not top[i] else {k: top[i][k] for k in ("bandwidth_hz_per_px", "ro_polarity", "pe_dir")},
                       displacement_mm=abs(console.FAT_PPM) * 1e-6 * console.GAMMA_HZ_PER_T * world["b0_t"]
                       / m["realised"]["bandwidth_hz_per_px"] * PIXEL_MM)
        checks[f"fat_off_{i}"] = m["cover"] <= COVER_MAX
        checks[f"snr_floor_{i}"] = m["snr"] >= t["snr_floor"][i]
        checks[f"near_best_{i}"] = m["snr"] >= NEAR_BEST * ref
        if not checks[f"fat_off_{i}"]:
            bad.append(f"{i}: fat is {100 * m['cover']:.1f} % of the water signal")
        elif not checks[f"snr_floor_{i}"]:
            bad.append(f"{i}: signal-to-noise ratio {m['snr']:.1f} is below the floor {t['snr_floor'][i]:g}")
        elif not checks[f"near_best_{i}"]:
            bad.append(f"{i}: signal-to-noise ratio {m['snr']:.1f} is {100 * m['snr'] / ref:.0f} % of the best under rule 1")
    return F.result(checks, "; ".join(bad), error=max(r["cover"] for r in rows.values()), regions=rows,
                    distance=dict(regions=len(rows), passed=sum(all(v for k, v in checks.items() if k.endswith("_" + i)) for i in rows)))


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured answer that fails a check: one of ``OUTCOMES``."""
    c = score["checks"]
    if not all(v for k, v in c.items() if k.startswith("fat_off_")):
        return "fat_on_region"
    if not all(v for k, v in c.items() if k.startswith("snr_floor_")):
        return "snr_below_floor"
    return "bandwidth_too_high"


def card() -> dict:
    from ... import gate, learn
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    kind = F.kind_by_open_loop(gate.load(ENV_ID))
    return dict(
        id=ENV_ID, title="Keep displaced fat off each marked region", kind="candidate",
        given="A gel object with fat discs at hidden places; three marked regions between them; the scanner; a scan-time budget",
        deliver="For each region the readout settings of its spin-echo scan: bandwidth, readout direction and, at level 3, the phase-encode axis",
        scored_by="Noise-free fat and water images of the delivered scans on the hidden object: fat off the region, a "
                  f"signal-to-noise floor, and a signal-to-noise ratio within {round(100 * NEAR_BEST)} % of the best setting that keeps the fat off",
        summary="The readout as a time axis. Fat is displaced along the frequency-encode axis by its frequency difference "
                "divided by the bandwidth per pixel, to the side the readout direction sets. A higher bandwidth shrinks the "
                "displacement and costs signal-to-noise ratio. How much room the fat has, and on which side, is hidden: it is "
                "measured on images in which the fat is already displaced.",
        task_kind=kind["kind"], task_kind_why=kind["sentence"],
        world=dict(phantom="QA phantom (a disc of gel) with placed fat discs, 3 T", sequences_useful=["se", "spgr"],
                   agent_side_formulas="displacement = (fat-water frequency difference) / bandwidth_hz_per_px pixels along the "
                                       "readout, towards positive positions with ro_polarity 1; noise grows with the square root "
                                       "of the bandwidth"),
        hidden=["where the fat discs lie: the room between each region and the fat on every side", "the materials (every fat disc has a T1 of its own)",
                "fields and system state", "noise level", "the 128-bit world seed"],
        told=["the target scan", "the regions and their signal-to-noise floors", "the pass rule and its two thresholds",
              "the instrument datasheet (field strength, fat shift)", "realised protocol, timing and cost of every plan and scan",
              "any stored image, and the numbers along any line through it"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        practice_levels=[lv for lv, d in LEVELS.items() if d.get("kind") == "practice"],
        learnability=learn.stored(ENV_ID),
        limits=dict(max_scanner_requests=48, max_tool_calls=64),
        deliverable={"tool": "submit", "format": {"scans": {"<region id>": {"bandwidth_hz_per_px": "<Hz per pixel>", "ro_polarity": "1 | -1",
                                                                            "pe_dir": "y | x (level 3 only)"}}}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan",
                           "the console accepts every delivered scan"],
                    error="largest over the regions of: |mean fat signal| / |mean water signal| in the region (noise-free)",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing=f"all gates and, for every region: error <= {100 * COVER_MAX:g} %, signal-to-noise ratio >= its floor and "
                            f">= {round(100 * NEAR_BEST)} % of the best setting that keeps the fat off"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier's images are noise-free",
    )
