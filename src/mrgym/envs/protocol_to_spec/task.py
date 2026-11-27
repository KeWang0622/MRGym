"""``Scanner/ProtocolToSpec``: instances and the verifier.

The job: an examination of two inversion-recovery spin-echo images runs with a standard protocol. Each image has a specification of four
requirements on marked regions of the object: a T1 contrast (one region brighter than another by a ratio), a T2
contrast, a signal-to-noise ratio and a scan time. On this object the standard protocol misses the specification.
Three fields of each image are free: the inversion time, the echo time and the number of averages. Deliver, for
each image, a protocol that meets all four requirements.

Why the requirements are coupled. The T1 contrast asks for one region to be much brighter than another: that holds
only near the inversion time that nulls the other, which depends on its hidden T1, and moving the inversion time moves
the signal of every other region with it. A longer echo time raises the T2 contrast and lowers the signal, and the
contrast of the other pair with it; averages buy the signal back with scan time, which is limited. Which protocols
pass is a narrow range of one field whose two ends are set by hidden quantities: the relaxation times of the pair that
binds, the noise of the receiver, the relaxation of the SNR region.

Why it is a decision. On this console every scan starts with its adjustments (``PRICE_S`` seconds), and the budget
pays for two scans. One scan of the standard protocol at low resolution shows which requirement fails. What has to be
measured next depends on that: how the signals change with the inversion time (a second scan at another TI) or with
the echo time (a second scan at another TE). Measuring both, and the standard protocol, costs three scans.

Deliverable    {"A": {"ti_ms": 95.0, "te_ms": 12.0, "averages": 3}, "B": {...}}
Pass rule      for each image, on a noise-free scan of the delivered protocol with the analytic noise of this scanner
               (``Engine.noise_sd``): both contrast ratios and the SNR reach their minimum and the scan time is within
               its limit. ``oracle.solve`` finds the protocols with the largest margin by exhaustive search.

Instances. The specification is the same for every instance of every level (``T1_RATIO``, ``T2_RATIO``, ``SNR_MIN``);
only the scan-time limits and the regions differ, and those are drawn without regard to the hidden world. What varies
is the object: which pairs of vials the contrasts are asked for, and the noise of the receiver, are chosen so that
the feasible range of the field that has to move is ``WINDOW`` wide and the other field cannot repair the image. The
levels differ in which field that is: the inversion time (1), the echo time (2), or either, not told (3).
"""
from __future__ import annotations

import copy
import functools
import math
import os

import numpy as np

from ...scanner import console
from ...scanner import world as W
from .. import family as F
from . import oracle

FAMILY = "protocol_to_spec"
SUBMIT_DOC = ("Deliver, for each image of the examination, the inversion time (milliseconds), the echo time (milliseconds) "
              "and the number of averages of its scan.")
IMAGES = ("A", "B")
OPEN_LOOP = ("always_ti", "always_te", "open_loop_card", "open_loop_all", "open_loop_short")     # the fixed lists of scans among the policies
FIELDS = oracle.FIELDS
T1_RATIO, T2_RATIO, SNR_MIN = 4.0, 1.5, 20.0                            # the specification, the same in every instance
# Outcome labels of a delivered, measured answer that fails (worst first) and the checks behind each: (label, stems),
# a check belongs to a requirement when its name is the stem or ``<stem>_<image>``. The label is the first of the four
# requirements of the specification that the delivery misses, in the order in which ``score`` declares the checks of
# one image (the scan time of the examination last); it names the requirement, never the image.
REQUIREMENTS = (("t1_contrast_too_low", ("t1_contrast",)), ("t2_contrast_too_low", ("t2_contrast",)), ("snr_too_low", ("snr",)),
                ("scan_too_long", ("time",)))
OUTCOMES = tuple(label for label, _ in REQUIREMENTS)
CONSOLE = dict(overhead_s=30.0, repeat_overhead_s=3.0)                  # this console: adjustments before a scan, and before one that reuses them
PRICE_S = CONSOLE["overhead_s"]
BUDGET_S = 63.0                                                         # the standard protocol at low resolution and one repeat of it
AVERAGES_NEEDED = (2, 7)                                                # each image is made to need a number of averages drawn from here
WINDOW = (1.2, 1.3)                                                     # upper / lower end of the feasible range of the echo time
WINDOW_OK = (1.12, 1.45)                                                # ... of the field that moves, as found on the grid
EDGE = dict(ti=(38.0, 160.0), te=(20.0, 80.0))                          # where the binding contrast is met (TI: centre; TE: from), ms
SNR_ROOM = (0.15, 0.35)                                                 # TI: ln(SNR / required) at the centre with all the averages allowed
FAILS_BY, HOLDS_BY, REACH, PAD, NEAR = 1.06, 1.2, 1.9, 15, 1.25           # see ``_cut``, ``_holds`` and ``_image``
BEND = 0.03                                                             # see ``_decays_exponentially``
MARGIN_MIN = 0.10                                                       # the best protocol clears every requirement by 10 %
MAX_ATTEMPTS, PATIENCE = 400, 40
SEARCH_GRID = 64                    # grid points per axis of the world on which ``_draw`` searches (the episode's has 640)
BASE = dict(kind="qa", b0_t=1.5, qa=dict(n_ring=(6, 6), vial_t1_ms=(50.0, 800.0)), console_system=CONSOLE,
            sigma0_log_sd=0.0)
LEVELS = {
    1: dict(knobs=("ti",), budget_s=BUDGET_S,
            summary="the T1 contrast fails at the standard protocol: the inversion time has to be found"),
    2: dict(knobs=("te",), budget_s=BUDGET_S,
            summary="the T2 contrast fails at the standard protocol: the echo time has to be found"),
    3: dict(knobs=("ti", "te"), budget_s=BUDGET_S,
            summary="one of the two contrasts fails, which one is not told: the second scan has to be chosen after the first"),
}


# ------------------------------------------------------------------------------------------------ instances
def _spec(sigma0: float) -> dict:
    """World spec of an instance whose receiver noise density is ``sigma0`` (the spec value is stated at 3 T)."""
    return dict(BASE, sigma0=float(sigma0) * BASE["b0_t"] / 3.0)


def _along(model, knob: str, region: str) -> np.ndarray:
    """Magnitude of the signal of a region along the field that moves, the other field at its standard value."""
    return np.maximum(np.abs(model.sig[region][oracle.I_TE0, :] if knob == "ti" else model.sig[region][:, oracle.I_TI0]), 1e-12)


def _cut(model, knob: str, a: str, b: str):
    """Where the binding contrast requirement on the ordered pair (a brighter than b) is met along the field that moves:
    (first grid index, last grid index or None, a value in ms), or None when the pair does not make the task.

    Inversion time: the requirement holds on one range around the null of b; it has to fail at the standard inversion
    time by ``FAILS_BY``, lie inside the grid with its centre in ``EDGE`` and be ``WINDOW_OK`` wide. Echo time: the
    ratio has to fail at the standard echo time by ``FAILS_BY``, rise with the echo time and reach the requirement at
    an echo time inside ``EDGE``; it holds from there on."""
    r = _along(model, knob, a) / _along(model, knob, b)
    if knob == "ti":
        grid = oracle.TI_GRID
        ok, k = r >= T1_RATIO, int(np.argmax(r))
        if r[oracle.I_TI0] * FAILS_BY > T1_RATIO or not ok[k]:
            return None
        lo, hi = k, k
        while lo > 0 and ok[lo - 1]:
            lo -= 1
        while hi < len(r) - 1 and ok[hi + 1]:
            hi += 1
        centre = math.sqrt(grid[lo] * grid[hi])
        if lo == 0 or hi == len(r) - 1 or ok.sum() != hi - lo + 1 or not EDGE[knob][0] <= centre <= EDGE[knob][1] \
                or not WINDOW_OK[0] <= grid[hi] / grid[lo] <= WINDOW_OK[1]:
            return None
        return lo, hi, float(centre)
    grid = oracle.TE_GRID
    if r[oracle.I_TE0] * FAILS_BY > T2_RATIO:
        return None
    lo, hi = (int(np.searchsorted(grid, v)) for v in EDGE[knob])
    ok = np.flatnonzero(r[lo:hi + 1] >= T2_RATIO)
    if np.any(np.diff(r[:hi + 1]) < 0) or not len(ok) or not lo < lo + ok[0] < hi:
        return None
    return lo + int(ok[0]), None, float(grid[lo + ok[0]])


def _holds(model, knob: str, a: str, b: str, cut: tuple) -> bool:
    """Whether the other contrast requirement holds on the ordered pair (a, b) by ``HOLDS_BY`` wherever the field that
    moves may end up: ``PAD`` grid steps around the range of the inversion time and at the standard one, or from the standard echo time to
    ``REACH`` times the echo time at which the binding contrast is met."""
    r = _along(model, knob, a) / _along(model, knob, b)
    if knob == "ti":
        return bool(np.all(r[max(cut[0] - PAD, 0):cut[1] + PAD + 1] >= HOLDS_BY * T2_RATIO) and r[oracle.I_TI0] >= HOLDS_BY * T2_RATIO)
    k = int(np.searchsorted(oracle.TE_GRID, min(cut[2] * REACH, oracle.TE_GRID[-1])))
    return bool(np.all(r[:k + 1] >= HOLDS_BY * T1_RATIO))


def _decays_exponentially(model, region: str, last: int) -> bool:
    """Whether the signal of a region at the standard inversion time falls with the echo time like one exponential, up
    to grid index ``last``: its logarithm stays within ``BEND`` of the straight line between the two ends. It does not
    for a region near its null, whose null moves with the echo time."""
    y = np.log(_along(model, "te", region)[:last + 1])
    x = oracle.TE_GRID[:last + 1]
    return bool(np.max(np.abs(y - (y[0] + (y[-1] - y[0]) * (x - x[0]) / (x[-1] - x[0])))) <= BEND)


def _fits(model, knob: str, req: dict, averages: int, margin=None) -> bool:
    """Whether an image is the task this family poses: with ``averages`` its requirements hold on a range of the field
    that moves whose ends are ``WINDOW_OK`` apart, nowhere with that field at its standard value, and the best protocol
    clears ``MARGIN_MIN``."""
    w = model.windows(req, averages)
    along, across = (w["ti_ms"], w["te_ms"]) if knob == "ti" else (w["te_ms"], w["ti_ms"])
    if along is None or across is not None or not WINDOW_OK[0] <= along[1] / along[0] <= WINDOW_OK[1]:
        return False
    return model.best(req, averages, margin)["margin"] >= MARGIN_MIN


def _image(model, knob: str, binding: list, other: list, used: list, drawn: tuple, unit: float, sigma0, near: float):
    """The specification of one image on this object, ``(requirements, noise density, binding pair, averages)``, or None.

    ``drawn``: the number of averages the image is to need, the target for the binding contrast, a number in ``WINDOW``
    and an integer for the arbitrary choices. The binding contrast is asked for an ordered pair of vials whose range
    lies within a factor ``near`` of the target (the four nearest are tried in turn; so the ranges are spread like the
    targets), the other contrast for a pair on which it holds by ``HOLDS_BY`` over the reach of the field, the SNR for
    a vial. Without ``sigma0`` (the first image) the receiver noise is set from the drawn number: when the echo time
    moves, so that with the drawn averages the SNR requirement closes the feasible range that many times above the echo
    time at which the contrast is met; when the inversion time moves, so that the SNR at the centre of the range exceeds
    the requirement by a share in ``SNR_ROOM``. With ``sigma0`` (the second image) the vials whose SNR comes nearest to
    that are tried. The last element returned is the smallest number of averages with which the image still clears
    ``MARGIN_MIN`` on a range of the right width."""
    n, target, width, pick = drawn
    cuts = sorted(((p, c) for p in binding if frozenset(p) not in used for c in [_cut(model, knob, *p)]
                   if c and abs(math.log(c[2] / target)) <= math.log(near)), key=lambda pc: abs(math.log(pc[1][2] / target)))
    room = SNR_ROOM[0] + (width - WINDOW[0]) / (WINDOW[1] - WINDOW[0]) * (SNR_ROOM[1] - SNR_ROOM[0])
    for pair, cut in cuts[:4]:
        slack = [p for p in other if frozenset(p) != frozenset(pair) and _holds(model, knob, *p, cut)]
        if not slack:
            continue
        second = slack[pick % len(slack)]
        if knob == "ti":                                                 # SNR of every vial at the centre of the range, per unit noise
            k, want = (cut[0] + cut[1]) // 2, math.exp(room)
        else:                                                            # ... at ``width`` times the echo time of the cut
            k, want = min(int(np.searchsorted(oracle.TE_GRID, cut[2] * width)), len(oracle.TE_GRID) - 1), 1.0
        level = {i: float(_along(model, knob, i)[k] * math.sqrt(n) / (SNR_MIN * unit)) for i in model.ids}
        if sigma0 is None:
            order = model.ids[pick % len(model.ids):] + model.ids[:pick % len(model.ids)]
            tries = [(i, level[i] / want) for i in order[:3]]
        else:
            tries = [(i, sigma0) for i in sorted(model.ids, key=lambda i: abs(math.log(level[i] / (want * sigma0))))[:3]]
        t1, t2 = (pair, second) if knob == "ti" else (second, pair)
        if knob == "te":                                                 # the regions the echo time is chosen for decay like exponentials
            last = min(int(np.searchsorted(oracle.TE_GRID, cut[2] * WINDOW_OK[1])), len(oracle.TE_GRID) - 1)
            tries = [(i, v) for i, v in tries if all(_decays_exponentially(model, r, last) for r in (*pair, i))]
        for region, noise in tries:
            req = dict(t1_contrast=dict(brighter=t1[0], darker=t1[1], min_ratio=T1_RATIO),
                       t2_contrast=dict(brighter=t2[0], darker=t2[1], min_ratio=T2_RATIO), snr=dict(region=region, min=SNR_MIN))
            model.sd1 = unit * noise
            m = model.margin(req)
            if _fits(model, knob, req, n, m):
                while n > 1 and _fits(model, knob, req, n - 1, m):       # the fewest averages with which it is still this task
                    n -= 1
                return req, noise, pair, n
    return None


def _draw(seed: str, knob: str, targets: list, near: float = NEAR):
    """One draw of the task on the world of ``seed``, or None when the object has no pairs of vials for it.

    ``knob`` is the field that has to move and ``targets`` where, for each image, the binding contrast should be met:
    both drawn before any world is looked at (``_instance``). The number of averages each image is to need and a
    number for the room the SNR requirement leaves are drawn here, without regard to the object. ``_image`` then looks
    for the pairs of vials and the receiver noise that make it so, and returns the fewest averages with which the image
    clears every requirement by ``MARGIN_MIN``. The scan-time limit of the examination pays for those averages of both
    images and not for one more, so how the averages are shared out between the images is part of the answer. The T2 contrast is always
    asked with the longer T2 brighter, as the task states. Nothing here looks at a policy: the conditions are on the
    signal model of the hidden world (``oracle.Model``)."""
    rng = W.rng_for(seed, "task")
    drawn = [(int(rng.integers(AVERAGES_NEEDED[0], AVERAGES_NEEDED[1] + 1)), target, float(rng.uniform(*WINDOW)), int(rng.integers(1 << 30)))
             for target in targets]
    spare = float(rng.uniform(0.05, 0.6))                                # of the time of one average
    world = W.make_world(seed, dict(_spec(1.0), n=SEARCH_GRID))          # the same object and fields on a coarse grid
    world["console"]["min_pixel_mm"] = round(2.0 * W.GRID["qa"][1] / W.GRID["qa"][0], 3)     # ... with the console of the episode
    model = oracle.Model(world)                                          # ``sd1`` is the noise per unit noise density
    unit, tis = model.sd1, {r["id"]: world["tissues"][r["cls"]] for r in world["regions"]}
    t1_pairs = [(a, b) for a in model.ids for b in model.ids if a != b]                                 # (brighter, nulled)
    t2_pairs = [(a, b) for a in model.ids for b in model.ids if tis[a]["t2_ms"] > tis[b]["t2_ms"]]      # (longer T2, shorter T2)
    binding, other = (t1_pairs, t2_pairs) if knob == "ti" else (t2_pairs, t1_pairs)
    images, used, sigma0, total = {}, [], None, 0
    for name, values in zip(IMAGES, drawn):
        found = _image(model, knob, binding, other, used, values, unit, sigma0, near)
        if not found:
            return None
        images[name], sigma0, total = found[0], found[1], total + found[3]
        used.append(frozenset(found[2]))
    time_max = float(math.floor(len(IMAGES) * float(model.scan_s(0)) + float(model.scan_s(total + spare) - model.scan_s(0))))
    if model.averages_max(time_max, len(IMAGES)) != total:
        return None
    return images, sigma0, F.public_regions(sorted(world["regions"], key=lambda r: int(r["id"][1:]))), time_max


@functools.lru_cache(maxsize=128)
def _instance(seed: str, level: int) -> tuple:
    """(world seed, images, noise density, regions, scan-time limit, field that moves) of the instance of ``seed``: the
    first world, in the fixed sequence of re-draws of the seed, on which ``_draw`` finds the task. The field that has to
    move and the targets of the binding contrasts are drawn before any world is looked at, so which field moves and
    where the feasible ranges lie do not depend on which worlds have the task (every ``PATIENCE`` worlds without one the
    tolerance on the targets is raised to the next power of ``NEAR``). A pure function of its arguments (cached: every
    episode of an instance in one process shares the search)."""
    rng, knobs = W.rng_for(seed, "task", 1), LEVELS[level]["knobs"]
    knob = knobs[int(rng.integers(len(knobs)))]                         # drawn once per instance: re-draws of the world keep it
    values = [math.exp(rng.uniform(*np.log(EDGE[knob]))) for _ in IMAGES]
    for attempt in range(MAX_ATTEMPTS):
        world_seed = F.reseed(seed, attempt)
        drawn = _draw(world_seed, knob, values, NEAR ** (1 + attempt // PATIENCE))
        if drawn:
            return (world_seed, *drawn, knob)
    raise RuntimeError(f"no instance with an answer in {MAX_ATTEMPTS} draws")


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    world_seed, images, sigma0, regions, time_max, _ = copy.deepcopy(_instance(seed, level))
    lv = LEVELS[level]
    task = dict(family=FAMILY, level=level, regions=regions,
                scan=dict(oracle.SCAN, partial_fourier=1.0, phase_fov=1.0), standard_protocol=dict(oracle.STANDARD),
                images=images, scan_time_max_s=time_max,
                adjustments=dict(new_scan_s=CONSOLE.get("overhead_s", console.SYSTEM["overhead_s"]),
                                 repeat_scan_s=CONSOLE.get("repeat_overhead_s", CONSOLE.get("overhead_s", console.SYSTEM["overhead_s"])),
                                 repeated_when_changed=list(console.ADJUSTMENT_FIELDS)),
                snr_definition="magnitude of the mean complex signal of the region's pixels / standard deviation of the noise "
                               "in one pixel, one channel (real or imaginary part)",
                deliverable={k: {"ti_ms": "<number, milliseconds>", "te_ms": "<number, milliseconds>", "averages": "<integer>"}
                             for k in images})
    return dict(task_id=task_id or "protocoltospec-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=world_seed,
                world_spec=_spec(sigma0), budget_s=float(lv["budget_s"]), max_requests=40, task=task)


def statement(cfg: dict) -> str:
    t = cfg["task"]
    s, p, adj = t["scan"], t["standard_protocol"], t["adjustments"]
    lines = []
    for name, q in t["images"].items():
        a, b = q["t1_contrast"], q["t2_contrast"]
        lines.append(f"""    image {name}
      T1 contrast   {a['brighter']} at least {a['min_ratio']:g} times as bright as {a['darker']} (the inversion can null {a['darker']})
      T2 contrast   {b['brighter']} (the longer T2) at least {b['min_ratio']:g} times as bright as {b['darker']}
      SNR           at least {q['snr']['min']:g} in {q['snr']['region']}""")
    names = " and ".join(t["images"])
    return f"""# Adapt the standard protocol until the examination meets its specification
A test object with vials of unknown materials lies in the field of
view. The examination has {len(t['images'])} inversion-recovery spin-echo images, {names}, with the same geometry and repetition time:
    sequence {s['sequence']}, tr_ms {s['tr_ms']:g}, fov_mm {s['fov_mm']:g}, matrix {s['matrix']}, slice_mm {s['slice_mm']:g}, bandwidth_hz_per_px {s['bandwidth_hz_per_px']:g}, full k-space.
Both run with the standard protocol ti_ms {p['ti_ms']:g}, te_ms {p['te_ms']:g}, averages {p['averages']}. Each image has its own specification:
{chr(10).join(lines)}
and the examination has one more: the scan times of the images together are at most {t['scan_time_max_s']:g} s.
"As bright as" compares the signals of two regions in the image. SNR of a region = {t['snr_definition']}.
On this object the standard protocol does not meet the specification. For each image you choose ti_ms, te_ms and
averages; everything else stays as above.
Regions (circles; centre and radius in mm from the isocentre; each lies inside one homogeneous material):
{F.region_table(t['regions'])}
Deliver the three numbers for each image with `submit`. The images are not run in this episode: the scanner evaluates
the protocols you deliver. {F.OWN_SCANS} {F.NOT_CHARGED}
This console adjusts itself before a scan (centre frequency, transmit gain, shim): {adj['new_scan_s']:g} s. A scan with the same sequence
and the same field of view, slice and orientation as the scan before it reuses those adjustments and starts after
{adj['repeat_scan_s']:g} s; changing only timing, matrix or averages does not repeat them. `plan` shows the cost of the next scan. Nothing
about the materials, the noise or the state of the system is known in advance.

Passing: every requirement holds, evaluated on noise-free scans of your protocols with the noise level of this scanner;
the scan time of each image is what the console plans for it as a scan of its own, adjustments included. The scanner's
log must show that you measured. An answer in the wrong unit is wrong."""


def submit_schema(cfg: dict) -> dict:
    spec = console.SEQUENCES["ir"]
    one = {"type": "object", "additionalProperties": False, "required": list(FIELDS), "properties": {
        "ti_ms": {"type": "number", "minimum": spec["ti_ms"][0], "maximum": spec["ti_ms"][1], "description": "inversion time, milliseconds"},
        "te_ms": {"type": "number", "minimum": spec["te_ms"][0], "maximum": spec["te_ms"][1], "description": "echo time, milliseconds"},
        "averages": {"type": "integer", "minimum": spec["averages"][0], "maximum": spec["averages"][1], "description": "number of averages"}}}
    names = list(cfg["task"]["images"])
    return {"type": "object", "additionalProperties": False, "required": names,
            "properties": {k: dict(one, description=f"the scan of image {k}") for k in names}}


# ------------------------------------------------------------------------------------------------ the verifier
def check(cfg: dict, answer, exam: dict):
    names = list(cfg["task"]["images"])
    if not isinstance(answer, dict) or set(answer) != set(names) \
            or any(not isinstance(answer[k], dict) or set(answer[k]) != set(FIELDS) for k in names):
        return "the deliverable is {" + ", ".join(f'"{k}": {{"ti_ms": <number>, "te_ms": <number>, "averages": <integer>}}' for k in names) + "}"
    for k in names:
        pl = console.plan(dict(oracle.SCAN, **answer[k]), exam)
        if not pl["accept"]:
            return f"the console rejects the scan of image {k}: " + "; ".join(v["message"] for v in pl["violations"])
    return None


def meets(value: float, minimum: float) -> bool:
    return bool(value >= minimum)


def in_time(scan_s: float, limit_s: float) -> bool:
    return bool(scan_s <= limit_s)


def measure(cfg: dict, name: str, fields: dict, world: dict, engine) -> dict:
    """Verifier side: what the delivered protocol of image ``name`` gives on the hidden object: the two signal ratios
    and the SNR on a noise-free scan with the analytic noise of the realised protocol, and the planned scan time."""
    q = cfg["task"]["images"][name]
    pl = console.plan(dict(oracle.SCAN, **fields), world["console"])
    rp = pl["realised"]
    ids = {q[k][side] for k in ("t1_contrast", "t2_contrast") for side in ("brighter", "darker")} | {q["snr"]["region"]}
    sig = {i: abs(v) for i, v in F.region_complex(engine, rp, [r for r in cfg["task"]["regions"] if r["id"] in ids]).items()}
    return dict(scan_s=pl["scan_s"], realised=rp, snr=float(sig[q["snr"]["region"]] / engine.noise_sd(rp)),
                t1_ratio=float(sig[q["t1_contrast"]["brighter"]] / sig[q["t1_contrast"]["darker"]]),
                t2_ratio=float(sig[q["t2_contrast"]["brighter"]] / sig[q["t2_contrast"]["darker"]]))


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    from ...scanner import kspace
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    engine = engine or kspace.Engine(world)
    problem = check(cfg, answer, world["console"])
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    checks, bad, detail, slacks = dict(deliverable_valid=True), [], {}, []
    for name, q in cfg["task"]["images"].items():
        m = measure(cfg, name, answer[name], world, engine)
        rows = (("t1_contrast", m["t1_ratio"], q["t1_contrast"]["min_ratio"], "T1 contrast"),
                ("t2_contrast", m["t2_ratio"], q["t2_contrast"]["min_ratio"], "T2 contrast"),
                ("snr", m["snr"], q["snr"]["min"], "SNR"))
        slack = {}
        for key, value, need, label in rows:
            checks[f"{key}_{name}"] = meets(value, need)
            slack[key] = math.log(value / need)
            if not checks[f"{key}_{name}"]:
                bad.append(f"image {name}: {label} {value:.3g}, required {need:g}")
        detail[name] = dict(scan_s=m["scan_s"], t1_ratio=m["t1_ratio"], t2_ratio=m["t2_ratio"], snr=m["snr"], slack=slack)
        slacks += list(slack.values())
    total, limit = float(sum(d["scan_s"] for d in detail.values())), cfg["task"]["scan_time_max_s"]
    checks["time"] = in_time(total, limit)
    if not checks["time"]:
        bad.append(f"scan time of the examination {total:.1f} s, allowed {limit:g} s")
    # ``error`` is minus the margin of the examination (the smallest logarithmic slack of any requirement): negative
    # exactly when it passes; the oracle's value is the smallest attainable
    return F.result(checks, "; ".join(bad), error=-min(slacks + [math.log(limit / total)]), images=detail, scan_s=total)


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured answer that fails a check: one of ``OUTCOMES``, the first requirement
    of ``REQUIREMENTS`` with a failed check. A function of the names and values of ``score["checks"]`` and nothing else."""
    failed = [k for k, ok in score["checks"].items() if not ok]
    missed = lambda stems: any(k == s or k.startswith(s + "_") for k in failed for s in stems)   # noqa: E731
    return next((label for label, stems in REQUIREMENTS if missed(stems)), OUTCOMES[-1])


def task_kind(stored: dict | None = None) -> dict:
    """Measurement task or decision task, per level, by the rule of this package: a level on which a fixed list of
    scans with a fixed solver (the ``open_loop`` control) passes within 25 points of the reference is a measurement
    task, otherwise a decision task; the control is the best of the fixed lists ``OPEN_LOOP``. Read from a gate result (default: the stored one), never typed."""
    from ... import gate
    stored = stored or gate.load("Scanner/ProtocolToSpec-v0")
    if not stored:
        return dict(kind="undetermined", levels={}, evidence="no stored gate result",
                    sentence="Measurement task or decision task (does a fixed list of scans pass within 25 points of the "
                             "reference?): not determined, the gate result is not stored yet.")
    gaps = {}
    for lv, s in stored["levels"].items():
        rates = {r["policy"]: r["rate"] for r in s["ablations"] + s["reported"] if r["policy"] in OPEN_LOOP}
        if not rates:
            return dict(kind="undetermined", levels={}, evidence="the stored gate result has no open-loop control",
                        sentence="Measurement task or decision task: not determined, the gate result has no open-loop control.")
        gaps[lv] = round(100 * (s["reference"]["rate"] - max(rates.values())), 1)
    kinds = {lv: ("decision" if g >= 25.0 else "measurement") for lv, g in gaps.items()}
    top = max(gaps, key=int)
    n = min(s["n"] for s in stored["levels"].values())
    points = ", ".join(f"{v:g} at level {lv}" for lv, v in sorted(gaps.items()))
    return dict(kind=kinds[top], levels=kinds,
                evidence=f"reference minus open_loop, points of pass rate: {points} (n = {n} per level, engine {stored['engine']})",
                sentence="By level: " + ", ".join(f"level {lv} is a {k} task" for lv, k in sorted(kinds.items()))
                         + f" (a fixed list of scans with a fixed solver passes this many points below the reference: {points}; "
                           f"{n} instances per level; 25 points or more make a decision task).")


def card() -> dict:
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    kind = task_kind()
    return dict(
        id="Scanner/ProtocolToSpec-v0", title="Adapt the standard protocol until the examination meets its specification", kind="candidate",
        given="A test object with vials of hidden materials; the scanner; the standard protocol of a two-image inversion-recovery "
              "examination; a scan-time budget for the standard protocol at low resolution and one repeat of it",
        deliver="For each of the two images: inversion time, echo time and number of averages (the averages share one scan-time limit)",
        scored_by="Noise-free scans of the delivered protocols on the hidden object with the analytic noise of the hidden "
                  "receiver: a T1 contrast ratio, a T2 contrast ratio and an SNR must hold for each image, and the scan times of the "
                  "two images together must stay within a limit",
        summary="The everyday protocol question: the standard protocol misses the specification on this patient, what has "
                "to change? One scan of the standard protocol shows which requirement fails; the budget then pays for one "
                "more scan, and which one is worth making (another inversion time or another echo time) depends on the "
                "first. " + kind["sentence"],
        task_kind=dict(kind=kind["kind"], levels=kind["levels"], evidence=kind["evidence"]),
        world=dict(phantom="QA phantom (a disc of gel with vials), 1.5 T", sequences_useful=["ir", "se"],
                   console=f"adjustments (centre frequency, transmit gain, shim) take {CONSOLE.get('overhead_s', 3):g} s before a scan and "
                           f"{CONSOLE.get('repeat_overhead_s', CONSOLE.get('overhead_s', 3)):g} s before one with the same sequence, field of view, slice and "
                           "orientation as the scan before it (scanner_info, plan). The kind of task above is measured on this "
                           "console; other pricings: python -m mrgym.envs.protocol_to_spec.report",
                   agent_side_formulas="from the datasheet: the inversion-recovery spin-echo signal mechanism (signed recovery "
                                       "with the inversion time, exp(-TE / T2), the slice profile), the scan-time rule, noise variance proportional to "
                                       "the receiver bandwidth. Not given: how SNR scales with voxel size, lines and averages, "
                                       "and the statistics of the magnitude of noise"),
        hidden=["T1, T2 and proton density of every material", "receiver noise level", "transmit gain and B1 pattern",
                "receive profile", "off-resonance field", "which requirement the standard protocol misses", "the 128-bit world seed"],
        told=["the geometry and the standard protocol of the images", "the four requirements of each image and their regions",
              "which region of the T1 pair is to be nulled and which region of the T2 pair has the longer T2", "the definition of SNR",
              "the instrument datasheet", "realised protocol, timing and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        limits=dict(max_scanner_requests=40, max_tool_calls=48),
        deliverable={"tool": "submit", "format": {k: {"ti_ms": "<milliseconds>", "te_ms": "<milliseconds>", "averages": "<integer>"}
                                                  for k in IMAGES}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan",
                           "the console accepts both delivered scans"],
                    error="minus the margin of the examination: the smallest of ln(achieved / required) over the requirements of "
                          "both images (for the scan time ln(allowed / planned)); negative exactly when every requirement holds",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing=f"all gates and, for each image, T1 contrast >= {T1_RATIO:g}, T2 contrast >= {T2_RATIO:g}, SNR >= {SNR_MIN:g} "
                            "; and the scan times of the two images together <= the stated limit"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier's scans are noise-free and its noise is analytic",
    )
