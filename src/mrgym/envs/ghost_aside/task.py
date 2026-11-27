"""``Scanner/GhostAside``: instances and the verifier.

The physics item: the phase-encode lines of a scan are acquired one repetition time apart, so the phase-encode axis
is a slow time axis. Material whose signal pulsates with the rate f is modulated from line to line, and the image
shows two copies of it (ghosts) displaced along the phase-encode axis, whatever the direction of the motion, by

    g = (phase field of view) x dist(f x TR x averages)          dist(u) = distance of u from the nearest integer

on either side (``mrgym validate`` rows K25, K26; re-derived with a discrete Fourier transform in
``tests/test_ghost_aside.py``). The engine's motion model is a declared one: the k-space lines of one tissue class
are multiplied by 1 + depth x cos(2 pi f t + phase) with t the time of the line; no flow, no spin history.

The job. A gel object holds three thin tubes whose signal pulsates, each at a hidden rate of its own. Each tube has
two marked regions: A in its column (reached by its ghosts when the phase is encoded along y) and B in its row (reached
when it is encoded along x). For each tube one spin-echo scan is delivered, its phase-encode axis and its repetition
time: the fastest scan that keeps the tube's ghost off both of its regions.

Deliverable    {"scans": {"T1": {"pe_dir": "y", "tr_ms": 465.0}, "T2": {...}, "T3": {...}}}
Pass rule      for every tube, on a noise-free scan of the target protocol with the delivered axis and repetition time,
               against the same scan with the motion switched off (the twin, ``Engine(effects=ALL - {"motion"})``):
                 ghost_off   in each of the tube's regions |mean of (scan - twin)| <= ``GHOST_MAX`` x |mean of the twin|
                 fastest     scan time <= ``TIME_TOL`` x the scan time of the fastest scan that satisfies ghost_off

What is hidden and why it decides. Tubes and regions are told. The rate of each tube is hidden, and with it where its
ghosts lie at any repetition time. At the shortest repetition time the ghosts of a slow tube lie on its regions with
either axis (levels 1 and 2: by construction, and the statement says so). A longer repetition time moves them outward
at (field of view) x f millimetres per second, so the repetition time at which they clear a region is
(distance to clear) / (field of view x f): it cannot be known without f, and f is read from the ghost distance of a
scan. Level 3 adds two things an agent has to find out per tube: it may not pulsate at all, or pulsate so fast that
its ghosts already lie far beyond its regions at the shortest repetition time (the distance law folds: dist()).

Averages. The target scan has one average: with n averages the scan takes n times as long, and inside the repetition
time window a one-average scan that satisfies ghost_off always exists on these worlds, so no scan with more averages
can be the fastest. The deliverable does not carry the number of averages.

The fastest scan. ``fastest`` searches both axes over a repetition-time grid of ``TR_STEP`` ms. Along one axis the
ghost share of a region falls once and stays down inside the window (a slow tube's ghost only moves outward there,
and no other tube's ghost can reach the region: both asserted in the tests, the first on a coarse grid of the engine,
the second on the drawn geometry), so the first repetition time that satisfies ghost_off is found by bisection. Kept
per instance.

Why three tubes. One tube's answer is a repetition time inside a window of 10 %, and the hidden rate moves it over a
range of a factor of 1.7; a rate assumed in the law with the told distances hits it about once in six tries (the
axis follows from the told distances: the nearer region). Three independent rates make that product small.

No signal-to-noise floor. The repetition time has a told lower bound and the averages are fixed, so every scan of the
window has at least the signal-to-noise ratio of the shortest one: a floor could never bind and is not stated.
"""
from __future__ import annotations

import collections
import math
import os
import threading

import numpy as np

from ...scanner import console
from ...scanner import session as core
from ...scanner import world as W
from .. import family as F

FAMILY = "ghost_aside"
ENV_ID = "Scanner/GhostAside-v0"
SUBJECT = "B"                           # spatial encoding and reconstruction
KIND = "set"                            # the deliverable is a setting of the scanner
ISOLATES = "ghosts of periodic motion along the phase-encode axis: distance = phase field of view x dist(rate x TR x averages)"
SUBMIT_DOC = "Deliver, for each tube, the phase-encode axis and the repetition time (milliseconds) of its scan."
TARGET = dict(sequence="se", te_ms=15.0, fov_mm=240.0, matrix=128, bandwidth_hz_per_px=300.0)
FOV_MM = TARGET["fov_mm"]
GRID_N = 384                            # master grid of the worlds: 0.67 mm, 2.8 points per pixel of the target scan
TR_WINDOW_MS = (300.0, 1000.0)          # the repetition times the examination allows
TR_STEP = 2.0                           # grid of ``fastest``, ms
GHOST_MAX = 0.02                        # ghost_off: the mean ghost signal in a region is at most this share of the region's mean signal
TIME_TOL = 1.10                         # fastest: scan time at most this multiple of the fastest scan under ghost_off
SUBSAMPLE = 8                           # points per pixel and axis with which the verifier weighs a pixel on the edge of a region
TUBE_MM, REGION_MM = 5.0, 12.0          # radius of a tube and of a marked region
TUBES = ((-54.0, -54.0), (0.0, 0.0), (54.0, 54.0))      # (y, x) of the tubes before the mirror and the jitter
SIDE = (1.0, 1.0, -1.0)                 # on which side of its tube the two regions lie (towards the middle of the object)
JITTER_MM = 2.0
NEAR_MM = (20.0, 23.5)                  # centre distance of the nearer region from its tube
FURTHER_MM = (6.0, 7.0)                 # the other region lies this much further away
CLEAR_MM = TUBE_MM + REGION_MM - 2.0    # a ghost this far beyond a region's centre is about off it (the verifier decides: 14.3 to 15.9 mm)
SLOW_MARGIN_MM = (6.0, 5.0)             # at the shortest TR a slow tube's ghost lies this far inside both regions' reach (entering, leaving)
FAST_HZ = (1.1, 1.45)                   # level 3: a fast tube; its ghosts lie 79 to 104 mm away at the shortest TR
DEPTH = (0.6, 0.85)                     # modulation depth of a pulsating tube
TUBE_T1_MS = (400.0, 520.0)
SIGMA0 = (0.008, 0.016)
OUTCOMES = ("ghost_on_region", "too_slow")      # of a delivered, measured answer that fails, worst first
# Examined competencies this family exercises (rows of ``mrgym.coverage``).
COMPETENCIES = ("parameter.phase_frequency_direction", "parameter.tr", "artefact.motion_ghost")
LEVELS = {
    1: dict(still=0, hints=("method",), budget_s=130.0,
            summary="three slow tubes, the method stated: at the shortest TR every ghost lies on a region with either axis"),
    2: dict(still=0, budget_s=130.0, summary="the worlds of level 1, the method withheld"),
    3: dict(still=1, budget_s=130.0,
            summary="one of the tubes does not pulsate, or pulsates fast: its regions are already clear at the shortest TR, and which tube it is is hidden"),
}
_kept = collections.OrderedDict()       # per instance: the twin engine and the result of ``fastest``
_kept_lock = threading.Lock()
KEPT, MEASURED_KEPT = 32, 512


def ghost_distance_mm(rate_hz: float, tr_ms: float, averages: int = 1, fov_pe_mm: float = FOV_MM) -> float:
    """The law of the family: distance of the two ghost copies from the pulsating material along the phase-encode axis."""
    u = rate_hz * tr_ms * 1e-3 * averages
    return fov_pe_mm * abs(u - round(u))


def target_protocol(scan: dict) -> dict:
    """The target scan with the delivered settings of one tube."""
    return dict(TARGET, pe_dir=scan.get("pe_dir"), tr_ms=scan.get("tr_ms"))


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    lv = LEVELS[level]
    rng = W.rng_for(seed, "task")
    mirror = 1.0 if rng.uniform() < 0.5 else -1.0
    still = set(int(i) for i in rng.permutation(len(TUBES))[:lv["still"]])
    tr0 = TR_WINDOW_MS[0] * 1e-3
    tubes, inserts, motion = [], [], []
    for k, (y0, x0) in enumerate(TUBES):
        y, x = round(float(y0 + rng.uniform(-JITTER_MM, JITTER_MM)), 1), round(float(mirror * x0 + rng.uniform(-JITTER_MM, JITTER_MM)), 1)
        near = float(rng.uniform(*NEAR_MM))
        far = near + float(rng.uniform(*FURTHER_MM))
        d_a, d_b = (near, far) if rng.uniform() < 0.5 else (far, near)
        # a slow tube: at the shortest TR its ghost lies on both regions, well inside their reach
        lo, hi = (far - CLEAR_MM + SLOW_MARGIN_MM[0]) / (FOV_MM * tr0), (near + CLEAR_MM - SLOW_MARGIN_MM[1]) / (FOV_MM * tr0)
        rate = float(np.exp(rng.uniform(math.log(lo), math.log(hi))))
        fast = float(rng.uniform(*FAST_HZ))
        kind = "slow" if k not in still else ("none" if rng.uniform() < 0.5 else "fast")
        depth, t1 = float(rng.uniform(*DEPTH)), float(rng.uniform(*TUBE_T1_MS))
        sy, sx = SIDE[k], SIDE[k] * mirror
        regions = [dict(id=f"T{k + 1}A", y_mm=round(y + sy * d_a, 1), x_mm=x, radius_mm=REGION_MM),
                   dict(id=f"T{k + 1}B", y_mm=y, x_mm=round(x + sx * d_b, 1), radius_mm=REGION_MM)]
        tubes.append(dict(id=f"T{k + 1}", y_mm=y, x_mm=x, radius_mm=TUBE_MM, regions=regions, kind=kind,
                          rate_hz=None if kind == "none" else round(rate if kind == "slow" else fast, 4)))
        inserts.append(dict(y_mm=y, x_mm=x, radius_mm=TUBE_MM, tissue=dict(pd=1.0, t1_ms=round(t1, 1), t2_ms=180.0, t2p_ms=200.0)))
        if kind != "none":
            motion.append(dict(cls=f"insert{k}", kind="pulse", f_hz=tubes[-1]["rate_hz"], depth=round(depth, 3)))
    spec = dict(kind="qa", n=GRID_N, b0_t=1.5, b0_hz=10.0, cf_offset_hz=5.0, sigma0=float(np.exp(rng.uniform(*np.log(SIGMA0)))),
                qa=dict(n_ring=(0, 0), p_centre=0.0, body_radius_mm=(93.0, 98.0), offset_mm=2.0, inserts=inserts), motion=motion)
    public = [dict(id=t["id"], y_mm=t["y_mm"], x_mm=t["x_mm"], radius_mm=t["radius_mm"], regions=F.public_regions(t["regions"])) for t in tubes]
    task = dict(family=FAMILY, tubes=public, regions=[r for t in public for r in t["regions"]], ghost_max=GHOST_MAX, time_tol=TIME_TOL,
                tr_window_ms=list(TR_WINDOW_MS), complaint=lv["still"] == 0,
                target_scan=dict(TARGET, averages=1, slice_mm=5.0),
                deliverable={"scans": {"<tube id>": {"pe_dir": "<y or x>", "tr_ms": "<number, milliseconds>"}}})
    if lv.get("hints"):
        task["hints"] = list(lv["hints"])
    return dict(task_id=task_id or "ghostaside-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed, world_spec=spec,
                budget_s=float(lv["budget_s"]), max_requests=48, task=task, truth=dict(tubes=tubes))


# ------------------------------------------------------------------------------------------------ the task text
def hint_text(cfg: dict) -> str:
    if "method" not in cfg["task"].get("hints", ()):
        return ""
    lo = TR_WINDOW_MS[0]
    return f"""How to do it (this level states the method).
  1. Run the target scan at tr_ms {lo:g} twice, with pe_dir y and with pe_dir x. With pe_dir y the ghosts of a tube lie in
     its column, with pe_dir x in its row: what differs between the two images along those lines is the ghosts. A ghost
     can be brighter or darker than the gel under it; the phase image shows it as well.
  2. Read the distance g from the centre of a tube to the centre of its ghost. The law is
     g = (phase field of view) x dist(rate x TR x averages), dist(u) = distance of u from the nearest whole number.
     Here g = {FOV_MM:g} mm x rate x {lo / 1000:g} s, so rate = g / ({FOV_MM * lo / 1000:g} mm s).
  3. A ghost is a copy of the tube. It is off a region whose centre lies D mm from the tube once g is at least
     D + {CLEAR_MM:g} mm, that is from tr = (D + {CLEAR_MM:g} mm) / ({FOV_MM:g} mm x rate) on.
  4. Take the region that is nearer to the tube and encode the phase along the line that joins it to the tube (pe_dir y
     for region A, x for region B). Deliver that tr plus 5 %."""


def statement(cfg: dict) -> str:
    t = cfg["task"]
    s = t["target_scan"]
    lo, hi = t["tr_window_ms"]
    rows = "\n".join(f"  {u['id']}: tube at y {u['y_mm']:g}, x {u['x_mm']:g}; "
                     + "; ".join(f"region {r['id']} at y {r['y_mm']:g}, x {r['x_mm']:g}" for r in u["regions"]) for u in t["tubes"])
    known = (f"Every tube pulsates, at a few tenths of a hertz. At tr_ms {lo:g} a ghost of every tube lies on one of its regions,\n"
             "whichever phase-encode axis is chosen." if t["complaint"] else
             "Most tubes pulsate at a few tenths of a hertz. One may pulsate faster (up to 1.5 Hz) or not at all: its regions can be\n"
             f"free of ghosts already at tr_ms {lo:g}.")
    hints = hint_text(cfg)
    return f"""# The fastest scan that keeps each tube's ghost off its regions
A disc of gel holds three thin tubes of fluid whose signal can pulsate, each at a rate of its own. In an image a
pulsating tube leaves displaced copies of itself (ghosts) along the phase-encode axis; how far they lie from the tube
depends on the timing of the scan.
Each tube has two marked regions: A in its column (same x) and B in its row (same y).
For each tube the examination needs a spin-echo scan of its own, as fast as possible, in which both regions of that
tube are free of its ghost:
    sequence se, te_ms {s['te_ms']:g}, fov_mm {s['fov_mm']:g}, matrix {s['matrix']}, bandwidth_hz_per_px {s['bandwidth_hz_per_px']:g}, 1 average, slice 5 mm;
    you choose pe_dir and tr_ms, with tr_ms from {lo:g} to {hi:g}.
{known}
Tubes (radius {TUBE_MM:g} mm) and regions (circles of radius {REGION_MM:g} mm); positions in mm from the isocentre:
{rows}
Deliver the two settings of every tube's scan with `submit`. The target scans are not run in this episode: the scanner
evaluates the settings you deliver, without noise.
{F.OWN_SCANS}
{F.NOT_CHARGED}
The rates are not known in advance.

Passing, for every tube, in the scan delivered for it, compared with the same scan of the object at rest:
  1. in each of the tube's two regions the mean ghost signal over the area of the region is at most {100 * t['ghost_max']:g} % of its mean signal;
  2. the scan time is at most {round(100 * (t['time_tol'] - 1))} % above that of the fastest scan that satisfies rule 1 (either pe_dir, any tr_ms from {lo:g} to {hi:g}).
The scanner's log must show that you measured. A value in another unit is wrong.""" + ("\n\n" + hints if hints else "")


def submit_schema(cfg: dict) -> dict:
    t = cfg["task"]
    lo, hi = t["tr_window_ms"]
    scan = {"type": "object", "additionalProperties": False, "required": ["pe_dir", "tr_ms"],
            "properties": {"pe_dir": {"type": "string", "enum": ["y", "x"], "description": "phase-encode axis"},
                           "tr_ms": {"type": "number", "description": f"repetition time, milliseconds, from {lo:g} to {hi:g}"}}}
    ids = [u["id"] for u in t["tubes"]]
    return {"type": "object", "additionalProperties": False, "required": ["scans"],
            "properties": {"scans": {"type": "object", "additionalProperties": False, "required": ids,
                                     "description": "phase-encode axis and repetition time of each tube's scan",
                                     "properties": {i: scan for i in ids}}}}


def check(cfg: dict, answer, exam: dict):
    t = cfg["task"]
    ids = [u["id"] for u in t["tubes"]]
    lo, hi = t["tr_window_ms"]
    shape = '{"scans": {' + ", ".join(f'"{i}": {{"pe_dir": ..., "tr_ms": ...}}' for i in ids) + "}}"
    if not isinstance(answer, dict) or set(answer) != {"scans"} or not isinstance(answer["scans"], dict) or set(answer["scans"]) != set(ids):
        return "the deliverable is " + shape
    for i in ids:
        scan = answer["scans"][i]
        if not isinstance(scan, dict) or set(scan) != {"pe_dir", "tr_ms"}:
            return f"the scan of {i} needs exactly pe_dir and tr_ms: the deliverable is " + shape
        pl = console.plan(target_protocol(scan), exam)
        if not pl["accept"]:
            return f"the console rejects the scan of {i}: " + "; ".join(v["message"] for v in pl["violations"])
        if not lo <= pl["realised"]["tr_ms"] <= hi:
            return f"tr_ms of {i} is {pl['realised']['tr_ms']:g}: the examination allows {lo:g} to {hi:g} ms"
    return None


# ------------------------------------------------------------------------------------------------ the verifier
def area_weights(rp, region, sub: int = SUBSAMPLE):
    """Share of every pixel of the image of ``rp`` that lies inside the circle ``region`` (``sub`` x ``sub`` points per
    pixel): the verifier's region is the circle itself, not the pixels whose centres fall inside it."""
    shape = (rp["matrix_pe"], rp["matrix"]) if rp["pe_dir"] == "y" else (rp["matrix"], rp["matrix_pe"])
    y, x = core.pixel_positions(shape, core.fov_yx(rp), core.centre_yx(rp))
    o = (np.arange(sub) + 0.5) / sub - 0.5
    yy = (y[:, None] + o[None, :] * (y[1] - y[0])).ravel()
    xx = (x[:, None] + o[None, :] * (x[1] - x[0])).ravel()
    inside = (yy[:, None] - region["y_mm"]) ** 2 + (xx[None, :] - region["x_mm"]) ** 2 <= region["radius_mm"] ** 2
    return inside.reshape(len(y), sub, len(x), sub).mean((1, 3))


def _instance_store(cfg: dict, world: dict) -> dict:
    """What is kept per instance: the twin engine (the world at rest) and, once computed, the fastest scans."""
    from ...scanner import kspace
    key = (cfg["seed"], cfg["level"])
    with _kept_lock:
        if key not in _kept:
            _kept[key] = dict(twin=kspace.Engine(world, effects=kspace.ALL - {"motion"}, cache=8), measured=collections.OrderedDict(),
                              lock=threading.Lock())                     # the twin engine is shared by every episode of the instance
            while len(_kept) > KEPT:
                _kept.popitem(last=False)
        return _kept[key]


def measure(cfg: dict, tube: dict, scan: dict, world: dict, engine) -> dict:
    """Verifier side: ghost share of the two regions of ``tube`` in the target scan with the settings ``scan``, and its
    scan time (kept per instance and setting: the images are noise-free, so a repeated question has the same answer)."""
    pl = console.plan(target_protocol(scan), world["console"])
    rp = pl["realised"]
    store = _instance_store(cfg, world)
    key = (tube["id"], rp["pe_dir"], rp["tr_ms"])
    with _kept_lock:
        share = store["measured"].get(key)
    if share is None:
        moving = engine.acquire(rp, None)[0]
        with store["lock"]:
            rest = store["twin"].acquire(rp, None)[0]
        share = {}
        for r in tube["regions"]:
            m = area_weights(rp, r)
            share[r["id"]] = float(abs(np.sum(m * (moving - rest))) / max(abs(np.sum(m * rest)), 1e-30))
        with _kept_lock:
            store["measured"][key] = share
            while len(store["measured"]) > MEASURED_KEPT:
                store["measured"].popitem(last=False)
    return dict(ghost=dict(share), worst=max(share.values()), scan_s=float(pl["scan_s"]), realised=rp)


def tr_grid() -> list:
    return [TR_WINDOW_MS[0] + TR_STEP * k for k in range(int(round((TR_WINDOW_MS[1] - TR_WINDOW_MS[0]) / TR_STEP)) + 1)]


def first_clean(cfg: dict, tube: dict, pe_dir: str, world: dict, engine, top: int | None = None):
    """Verifier side: index in ``tr_grid()`` of the shortest repetition time at which both regions of ``tube`` satisfy
    ghost_off with this phase-encode axis, among the indices up to ``top``; None when there is none. Bisection: along
    one axis the ghost share falls once inside the window."""
    grid = tr_grid()
    clean = lambda k: measure(cfg, tube, dict(pe_dir=pe_dir, tr_ms=grid[k]), world, engine)["worst"] <= GHOST_MAX   # noqa: E731
    if clean(0):
        return 0
    hi = len(grid) - 1 if top is None else top
    if hi <= 0 or not clean(hi):
        return None
    lo = 0                                                               # not clean at lo, clean at hi
    truth = next(u for u in cfg["truth"]["tubes"] if u["id"] == tube["id"])
    if truth["rate_hz"]:                                                 # a first guess from the drawn rate narrows the bracket; the bisection decides
        r = truth["regions"][0 if pe_dir == "y" else 1]
        d = math.hypot(r["y_mm"] - truth["y_mm"], r["x_mm"] - truth["x_mm"])
        guess = 1e3 * (d + CLEAR_MM) / (FOV_MM * truth["rate_hz"])
        for k in (int((guess * 0.95 - grid[0]) / TR_STEP), int((guess * 1.05 - grid[0]) / TR_STEP) + 1):
            if lo < k < hi:
                lo, hi = (lo, k) if clean(k) else (k, hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if clean(mid) else (mid, hi)
    return hi


def fastest(cfg: dict, world: dict, engine) -> dict:
    """Verifier side: per tube, the fastest scan under ghost_off over both axes and ``tr_grid()``:
    {id: dict(pe_dir, tr_ms, scan_s)}. Kept per instance."""
    store = _instance_store(cfg, world)
    if "fastest" in store:
        return store["fastest"]
    grid, out = tr_grid(), {}
    for tube in cfg["task"]["tubes"]:
        near = min(tube["regions"], key=lambda r: math.hypot(r["y_mm"] - tube["y_mm"], r["x_mm"] - tube["x_mm"]))
        axes = ("y", "x") if near["id"].endswith("A") else ("x", "y")    # the axis of the nearer region first; the other only has to be beaten
        found = None
        for pe in axes:
            k = first_clean(cfg, tube, pe, world, engine, top=None if found is None else found[0] - 1)
            if k is not None:
                found = (k, pe)
        if found is None:
            out[tube["id"]] = None
            continue
        scan = dict(pe_dir=found[1], tr_ms=grid[found[0]])
        out[tube["id"]] = dict(scan, scan_s=measure(cfg, tube, scan, world, engine)["scan_s"])
    with _kept_lock:
        store["fastest"] = out
    return out


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    from ...scanner import kspace
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    engine = engine or kspace.Engine(world)
    problem = check(cfg, answer, world["console"])
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    t = cfg["task"]
    top = fastest(cfg, world, engine)
    checks, rows, bad = dict(deliverable_valid=True), {}, []
    for tube in t["tubes"]:
        i = tube["id"]
        m = measure(cfg, tube, answer["scans"][i], world, engine)
        ref = top[i]["scan_s"] if top[i] else float("inf")
        rows[i] = dict(ghost=m["worst"], by_region=m["ghost"], scan_s=m["scan_s"], scan_s_fastest=None if not top[i] else ref,
                       time_ratio=m["scan_s"] / ref if top[i] else None,
                       fastest=None if not top[i] else {k: top[i][k] for k in ("pe_dir", "tr_ms")})
        checks[f"ghost_off_{i}"] = m["worst"] <= GHOST_MAX
        checks[f"fastest_{i}"] = bool(top[i]) and m["scan_s"] <= TIME_TOL * ref
        if not checks[f"ghost_off_{i}"]:
            worst = max(m["ghost"], key=m["ghost"].get)
            bad.append(f"{i}: the ghost in {worst} is {100 * m['worst']:.1f} % of its signal")
        elif not top[i]:
            bad.append(f"{i}: no scan of the window keeps both regions free of the ghost")
        elif not checks[f"fastest_{i}"]:
            bad.append(f"{i}: {m['scan_s']:.1f} s is {100 * (m['scan_s'] / ref - 1):.0f} % above the fastest scan without a ghost")
    return F.result(checks, "; ".join(bad), error=max(r["ghost"] for r in rows.values()), sites=rows,
                    distance=dict(tubes=len(rows), passed=sum(checks[f"ghost_off_{i}"] and checks[f"fastest_{i}"] for i in rows)))


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured answer that fails a check: one of ``OUTCOMES``."""
    return "ghost_on_region" if not all(v for k, v in score["checks"].items() if k.startswith("ghost_off_")) else "too_slow"


def card() -> dict:
    from ... import gate, learn
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    kind = F.kind_by_open_loop(gate.load(ENV_ID))
    return dict(
        id=ENV_ID, title="The fastest scan that keeps each tube's ghost off its regions", kind="candidate",
        given="A gel object with three tubes whose signal pulsates at hidden rates; two marked regions per tube; the scanner; a scan-time budget",
        deliver="For each tube the phase-encode axis and the repetition time of its spin-echo scan",
        scored_by="Noise-free scans of the delivered protocols against the same scans of the object at rest: no ghost on the "
                  f"tube's regions, and a scan time within {round(100 * (TIME_TOL - 1))} % of the fastest scan without one",
        summary="The line-to-line time axis. Periodic change over the phase-encode lines shows as copies of the changing "
                "material along the phase-encode axis, at a distance set by its rate, the repetition time and the averages. "
                "The rate is hidden: it is read from the ghost distance of a scan, and it decides how long the repetition time "
                "must be before the ghosts have left the regions.",
        task_kind=kind["kind"], task_kind_why=kind["sentence"],
        world=dict(phantom="QA phantom (a disc of gel) with three placed tubes, 1.5 T; a declared motion model (limits.md, section 7)",
                   sequences_useful=["se", "spgr"],
                   agent_side_formulas="ghost distance = phase field of view x dist(rate x TR x averages) along the phase-encode "
                                       "axis, on both sides of the pulsating material; scan time = overhead + TR x (1 + lines)"),
        hidden=["the pulsation rate and depth of every tube (level 3: whether it pulsates)", "the materials", "fields and system state",
                "noise level", "the 128-bit world seed"],
        told=["the target scan and its repetition-time window", "tubes and regions", "the pass rule and its two thresholds",
              "the instrument datasheet", "realised protocol, timing and cost of every plan and scan",
              "any stored image, and the numbers along any line through it"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        practice_levels=[lv for lv, d in LEVELS.items() if d.get("kind") == "practice"],
        learnability=learn.stored(ENV_ID),
        limits=dict(max_scanner_requests=48, max_tool_calls=64),
        deliverable={"tool": "submit", "format": {"scans": {"<tube id>": {"pe_dir": "y | x", "tr_ms": "<milliseconds>"}}}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan",
                           "the console accepts every delivered scan and its repetition time lies in the window"],
                    error="largest over the regions of: |mean of (scan - scan at rest)| / |mean of the scan at rest| (noise-free)",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing=f"all gates and, for every tube: error <= {100 * GHOST_MAX:g} % in both of its regions and scan time <= "
                            f"{TIME_TOL:g} x the fastest scan that keeps them clear"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier's scans are noise-free",
    )
