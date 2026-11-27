"""``Scanner/EPISetup``: instances and the verifier.

The job: an examination needs a single-shot gradient-echo EPI of two marked regions. It has an echo-time window, a pixel
ceiling, a limit on how far the off-resonance field may displace each region, a limit on the Nyquist ghost of each
region, no wrap-around, a signal-to-noise floor and, where the object holds fat, a limit on what displaced fat adds to
a region. The repetition time is fixed, so the scan time is the number of averages. Deliver one protocol that meets
every requirement with the fewest averages.

What it integrates. A long echo spacing (low bandwidth) has the least noise and the smallest ghost, and the largest
displacement: off-resonance x echo spacing x phase field of view. A short phase field of view shortens the echo train
and the displacement, and wraps when the object does not fit; the object is elongated and lies off the isocentre, so the
two phase-encode axes differ and a short field of view has to be centred on it. The ghost of a region grows with the
bandwidth and with its distance from the isocentre along the readout, so the axis also decides which regions ghost.
Fat is displaced by tens of millimetres along the phase-encode axis, towards the side the blip polarity sets. Hidden:
the field, the gradient delay, the size, orientation and position of the object, where the fat lies, the noise and
the T2* of the regions.

Deliverable    {"pe_dir": "x", "pe_polarity": 1, "bandwidth_hz_per_px": 1210.0, "matrix": 64, "phase_fov": 0.75,
                "centre_pe_mm": -11.5, "te_ms": 31.2, "averages": 2}
Pass rule      on noise-free scans of the delivered protocol and its twins on the hidden world (``metrics``): echo time
               inside the window; pixel within the ceiling; every region displaced by at most the limit; the ghost of
               every region within the limit; wrap-around energy within ``oracle.WRAP_LIMIT``; displaced fat changes no
               region by more than ``oracle.FAT_LIMIT``; every region reaches the SNR floor with the analytic noise;
               AND the scan time is not longer than the shortest with which any allowed protocol meets all of this
               (``oracle.least``: an exhaustive search on the hidden world).

Why partial Fourier and fat saturation are fixed (full k-space, no fat saturation). The family was built on engine
0.2.0, whose noise for partial Fourier and whose fat-saturation pulse were wrong and were corrected in engine 0.3.0
(docs/physics/limits.md, block 0); its instances, its search and its limits were made without either, so that the
answer did not change with that correction. Opening them is a new task (a new id), not a change of this one.
The field of view moves along the phase-encode axis only: with a shifted readout the engine takes the phase ramp of the
gradient delay from the centre of the field of view, a real scanner from the isocentre.

Instances. A world is kept when
  (1) the task has an answer that a careful measurement reaches: the search with every limit tightened
      (``oracle.STRICT``) needs the same number of averages as the search at the stated limits, so nothing hinges on a
      boundary;
  (2) the hidden field decides the bandwidth: at the best strict protocol the displacement of a region lies at its
      limit, and at the lowest bandwidth that the echo-time window and the console allow for that geometry it exceeds
      the limit by 20 %;
  (3) the SNR floor is set so that the three best strict protocols clear it by 7 to 10 % with a number of averages
      drawn beforehand (1, 2 or 3);
  (4) displacement and ghost conflict: the field asks for a short echo spacing, the gradient delay for a low bandwidth,
      and of the geometries the console offers (two axes, about seventeen phase fields of view each) only two to five
      meet both limits with the strict margins. A trap drawn beforehand is there or is not: in half of the instances
      the geometry with the largest SNR when the delay is ignored misses the ghost limit by 15 % or more, in the other
      half it meets it;
  (5) at level 3 a disc of oil is placed in the gel (an insert of the hidden world spec), and a second trap drawn
      beforehand is there or is not: in half of the instances the oil lies where the protocol that would be best
      without it puts the main line of fat on a marked region, in the other half that protocol clears the regions by
      8 mm or more.
Decided on the hidden world with the search of ``oracle``, on a coarse master grid, never with a policy. Two numbers of
an instance's hidden world spec are set, not drawn (``_sizes``): the size of the off-resonance field, so that the
bandwidth it asks for is the one drawn for the instance (log-uniform, 650 to 2600 Hz/px: the answer is spread over the
range of the console), and the size of the gradient delay, so that (4) holds. The directions and shapes of the field
and the sign and share of the delay are those of the seed; a world whose field would exceed 1 ppm anywhere in the
object is not used.

Levels: 1 states the method, 2 withholds it, 3 adds a disc of oil (where is hidden) whose displaced image must stay
off the marked regions.
"""
from __future__ import annotations

import collections
import functools
import math
import os

import numpy as np

from ...scanner import console
from ...scanner import world as W
from .. import family as F
from . import metrics as M
from . import oracle

FAMILY = "epi_setup"
ENV_ID = "Scanner/EPISetup-v0"
SUBMIT_DOC = ("Deliver the protocol of the echo-planar scan: phase-encode axis and blip polarity, receiver bandwidth (hertz per pixel), "
              "matrix, phase field of view (fraction) and its centre along the phase-encode axis (millimetres from the isocentre), echo time "
              "(milliseconds) and number of averages.")
# The simpler skills this task draws on (documentation; a later module reads it).
NEEDS = ("map the off-resonance field with two echoes", "off-resonance displacement", "bandwidth and readout time", "echo spacing and echo time of an echo train",
         "phase-encode direction", "field of view and wrap", "centre the field of view on the object", "Nyquist ghost and gradient delay", "noise and averaging",
         "pixel size and SNR", "chemical-shift displacement")
OPEN_LOOP_NOTE = ("At the development gate (60 instances per level; its result is not stored) a fixed list of scans passed within three "
                  "instances of the reference at each level, which makes this a measurement task: a signal-free scan for the noise, a "
                  "two-echo gradient echo for the field and the shape of the object (at level 3 one more for the oil), an echo-planar scan "
                  "at the highest bandwidth on each phase-encode axis for the gradient delay, and one echo-planar scan of a fixed protocol "
                  "at each end of the echo-time window, then the reference's solver.")
DISPLACEMENT_MAX_MM, GHOST_MAX = 3.0, 0.05
PIXELS_MM = (3.0, 3.4, 3.75)                                            # the pixel ceiling is drawn from here (matrix 80, 72, 64)
TE_HIGH_MS, TE_SPAN_MS = (40.0, 45.0, 50.0, 55.0), 15.0                 # the echo-time window: upper end drawn, lower end = upper - span
AVERAGES_TARGET = (1, 2, 3)                                             # the number of averages the best protocol is made to need
SNR_MARGIN = (0.07, 0.10)                                               # by how much the strict protocols clear the SNR floor
SNR_RANK = 3                                                            # ... counted on the third best of them: the best three all clear it
BANDWIDTH_TARGET = (650.0, 2600.0)                                      # the bandwidth the field is to ask for is drawn log-uniformly in here
FIELD_SCALE, DELAY_US = (0.15, 6.0), (0.05, 12.0)                       # where the size of the field (a factor) and of the delay are looked for
FIELD_MAX_HZ = 64.0                                                     # the off-resonance anywhere in the object stays below this (1 ppm at 1.5 T)
BINDS = 0.93                                                            # the displacement of the best strict protocol reaches this share of its limit
FIELD_DECIDES = 1.2                                                     # ... and at the lowest bandwidth the window allows it exceeds the limit by this factor
FEW = (2, 5)                                                            # how many geometries (axis, polarity, phase field of view) meet displacement and ghost
# drawn traps: the protocol that looks best when the delay is ignored misses the ghost limit by this factor; the one that
# looks best when the oil is ignored overlaps a region by this much (trap), or clears it by that (no trap)
TRAP = dict(ghost=1.15, fat_mm=-3.0, clear_mm=8.0)
OIL = dict(radius_mm=(9.0, 11.0), distance_mm=(24.0, 46.0), jitter_mm=3.0, gap_mm=2.5, tries=24)     # the disc of oil of level 3
SNR_RANGE = (8.0, 120.0)
MAX_ATTEMPTS, PATIENCE = 600, 40                                        # worlds tried in all, and per drawn task
COARSE = 160                                                            # master grid of the worlds the generator searches (the episode's has 640)
N_REGIONS = 2
VIAL_REACH_MM = 15.75                                                   # the largest semi-axis of a vial (radius 15 mm, 5 % ellipticity)
OUTCOMES = ("te_outside_window", "pixel_too_large", "displaced", "ghost", "wrap_around", "fat_on_region", "snr_too_low", "too_many_averages")
COMPETENCIES = ("parameter.te", "parameter.fov", "parameter.matrix", "parameter.pixel", "parameter.averages", "parameter.phase_frequency_direction",
                "parameter.receive_bandwidth", "option.rectangular_fov", "artefact.aliasing", "artefact.chemical_shift", "artefact.susceptibility",
                "artefact.eddy_currents", "artefact.b0_distortion", "qc.snr", "qc.centre_frequency")
BASE = dict(kind="qa", b0_t=1.5, b0_hz=50.0, cf_offset_hz=34.0, grad_delay_us=1.5, sigma0=0.12, sigma0_log_sd=0.7, tx_gain_dev=(0.0, 0.05))
QA = dict(body_radius_mm=(80.0, 90.0), body_aspect=(1.15, 1.45), body_rotation_deg=(0.0, 90.0), ring_radius_mm=(34.0, 42.0),
          vial_radius_mm=(13.0, 15.0), n_ring=(3, 4), roi_radius_mm=6.0, p_centre=0.0, offset_mm=22.0,
          vial_t1_ms=(300.0, 1500.0), vial_t2_ms=(60.0, 200.0), vial_t2p_ms=(60.0, 300.0))
LEVELS = {
    1: dict(world_spec=dict(BASE, qa=QA), hints=("method",), budget_s=85.0,
            summary="the method is stated: map the field, predict displacement and ghost of candidate protocols, fit the phase field of view, count the averages"),
    2: dict(world_spec=dict(BASE, qa=QA), budget_s=85.0, summary="the method is withheld"),
    3: dict(world_spec=dict(BASE, qa=QA), oil=True, budget_s=95.0,
            summary="as level 2 with a disc of oil near a marked region: its image is displaced along the phase-encode axis and must stay off the regions"),
}


# ------------------------------------------------------------------------------------------------ instances
def _sig3(x: float) -> float:
    return float(f"{x:.3g}")


def _coarse(seed: str, spec: dict) -> dict:
    world = W.make_world(seed, dict(spec, n=COARSE))
    world["console"]["min_pixel_mm"] = round(2.0 * W.GRID["qa"][1] / W.GRID["qa"][0], 3)      # the console of the episode
    return world


def _largest(ok, lo: float, hi: float, steps: int = 12):
    """The largest x in [lo, hi] (in equal ratios, to 2 ** (-steps) of the range) with ``ok(x)``, for an ``ok`` that
    holds up to some x and not beyond; None when it does not hold at ``lo``."""
    if not ok(lo):
        return None
    if ok(hi):
        return hi
    for _ in range(steps):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def _sizes(seed: str, level: int, task: dict, drawn: dict):
    """The size of the off-resonance field and of the gradient delay of this world: (field scale, delay amplitude in
    microseconds) of its world spec, or None. The field is scaled so that the lowest bandwidth within the strict
    displacement limit is the drawn target; the delay is then set between the largest that leaves ``FEW[0]`` geometries
    and the largest that leaves more than ``FEW[1]``, and so that the drawn ghost trap is there (the geometry with the
    largest SNR when the delay is ignored misses the ghost limit by ``TRAP["ghost"]``) or is not (it meets the strict
    one). Both sizes scale the world linearly, so the closed forms of the oracle are evaluated on one build."""
    unit = _coarse(seed, dict(LEVELS[level]["world_spec"], grad_delay_us=1.0))
    f = oracle.facts(unit, task)
    sign = unit["system"]["grad_delay_us"]                             # the delay of this world per unit of its amplitude
    if abs(sign) < 0.1:
        return None

    def geometries(scale, delay_us, how):
        return oracle.geometries(unit, task, how, oracle.scaled(f, scale, delay_us * 1e-6))

    no_ghost = dict(oracle.STRICT, ghost=oracle.IGNORE["ghost"], fat_mm=oracle.IGNORE["fat"])
    lowest = lambda scale: min((g["bandwidth_hz_per_px"] for g in geometries(scale, 0.0, no_ghost)), default=float("inf"))     # noqa: E731
    scale = _largest(lambda v: lowest(v) <= drawn["bandwidth_hz_per_px"], *FIELD_SCALE)
    if scale is None or scale >= FIELD_SCALE[1] or scale * f["body"]["df_max"] > FIELD_MAX_HZ:
        return None                                                      # no such field, or one that is far off resonance at the edge of the object
    count = lambda delay: len(geometries(scale, delay, oracle.STRICT))   # noqa: E731
    few, many = _largest(lambda v: count(v) >= FEW[0], *DELAY_US), _largest(lambda v: count(v) > FEW[1], *DELAY_US)
    if few is None or few >= DELAY_US[1]:
        return None
    lo, hi = (many if many is not None else DELAY_US[0]), few
    # the ghost trap: the geometry that looks best when the delay is ignored, and its ghost as the delay grows
    blind = max(geometries(scale, 0.0, no_ghost), key=lambda v: oracle.relative_snr(f, v))
    ghost = lambda delay: max(oracle.ghost_of(r, delay * 1e-6, blind) for r in f["regions"])        # noqa: E731
    if drawn["ghost_trap"]:
        clean = _largest(lambda v: ghost(v) < TRAP["ghost"] * GHOST_MAX, *DELAY_US)
        lo = max(lo, clean if clean is not None else DELAY_US[0])
    else:
        clean = _largest(lambda v: ghost(v) <= oracle.STRICT["ghost"] * GHOST_MAX, *DELAY_US)
        hi = min(hi, clean if clean is not None else 0.0)
    if not hi > lo * 1.02:
        return None                                                      # this world has no delay that sets (or spares) the trap
    return scale, math.sqrt(lo * hi) / abs(sign)


def _oil(seed: str, layout: dict, regions: list, best: dict, f: dict, trap: bool):
    """Where the disc of oil of level 3 lies: an insert of the world spec, or None when it finds no room. With ``trap`` it
    is placed so that the protocol ``best`` (the best one when the oil is ignored) puts the main line of fat on a marked
    region; without, at a drawn distance and direction from a marked region. Inside the gel, clear of every vial."""
    rng = W.rng_for(seed, "task", 2)
    body = layout["body"]
    co, si = math.cos(math.radians(body["rot"])), math.sin(math.radians(body["rot"]))
    hz = oracle.tissue.FAT_PPM * oracle.kspace.GAMMA_MHZ * layout["b0_t"]
    for _ in range(OIL["tries"]):
        radius = float(rng.uniform(*OIL["radius_mm"]))
        r = regions[int(rng.integers(len(regions)))]
        if trap:
            jitter = rng.uniform(-OIL["jitter_mm"], OIL["jitter_mm"], 2)
            move = float(oracle.displacement_mm(hz, best))
            y, x = r["y_mm"] + jitter[0] - (move if best["pe_dir"] == "y" else 0.0), r["x_mm"] + jitter[1] - (move if best["pe_dir"] == "x" else 0.0)
        else:
            dist, ang = float(rng.uniform(*OIL["distance_mm"])), float(rng.uniform(0, 2 * math.pi))
            y, x = r["y_mm"] + dist * math.sin(ang), r["x_mm"] + dist * math.cos(ang)
        u, v = (x - body["x"]) * co + (y - body["y"]) * si, -(x - body["x"]) * si + (y - body["y"]) * co      # along the axes of the gel (``anatomy.ell``)
        inside = (u / (body["b"] - radius - OIL["gap_mm"])) ** 2 + (v / (body["a"] - radius - OIL["gap_mm"])) ** 2 <= 1.0
        clear = all(math.hypot(y - q["y_mm"], x - q["x_mm"]) >= VIAL_REACH_MM + radius + OIL["gap_mm"] for q in layout["regions"])
        if inside and clear:
            return dict(y_mm=round(float(y), 2), x_mm=round(float(x), 2), radius_mm=round(radius, 2), tissue="oil")
    return None


def _draw(seed: str, level: int, drawn: dict):
    """The task on the world of ``seed``: (regions, SNR floor, world spec), or None when the world does not carry it (see
    the module docstring). ``drawn``: pixel ceiling, echo-time window, target bandwidth, target averages and SNR margin,
    drawn before any world."""
    base = LEVELS[level]["world_spec"]
    layout = W.make_world(seed, base, layout_only=True)
    marked = sorted(layout["regions"], key=lambda r: int(r["id"][1:]))[:N_REGIONS]             # ids are a random permutation of the vials
    task = dict(regions=F.public_regions(marked), pixel_max_mm=drawn["pixel_max_mm"], te_window_ms=drawn["te_window_ms"],
                displacement_max_mm=DISPLACEMENT_MAX_MM, ghost_max=GHOST_MAX)
    sizes = _sizes(seed, level, task, drawn)
    if sizes is None:
        return None
    spec = dict(base, b0_hz=round(base["b0_hz"] * sizes[0], 3), cf_offset_hz=round(base["cf_offset_hz"] * sizes[0], 3), grad_delay_us=round(sizes[1], 4))
    world = _coarse(seed, spec)
    scans, f = M.Scans(world), oracle.facts(world, task)
    strict = oracle.search(world, task, scans, strict=True, f=f)
    if strict["candidates"] and LEVELS[level].get("oil"):                # the best protocol without oil decides where the oil goes
        blind = strict["candidates"][0]["realised"]
        oil = _oil(seed, layout, task["regions"], blind, f, drawn["trap"])
        if oil is None:
            return None
        spec = dict(spec, qa=dict(spec["qa"], inserts=[oil]))
        world = _coarse(seed, spec)
        scans, f = M.Scans(world), oracle.facts(world, task)
        gap = oracle.fat_clearance_mm(f, blind)
        if (gap > TRAP["fat_mm"]) if drawn["trap"] else (gap < TRAP["clear_mm"]):
            return None                                                  # the trap is neither clearly there nor clearly absent
        strict = oracle.search(world, task, scans, strict=True, f=f)
    if not FEW[0] <= len(strict["candidates"]) <= FEW[1] * (2 if f["fat"] else 1):
        return None                                                      # displacement and ghost leave too many geometries, or too few to be found safely
    best = strict["candidates"][0]["realised"]
    worst = lambda rp: max(abs(float(oracle.displacement_mm(r["df_hz"], rp))) for r in f["regions"])        # noqa: E731
    lowest = next((rp for rp in oracle.ladder(world["console"], best, task["te_window_ms"])), None)
    if worst(best) < BINDS * oracle.STRICT["displacement"] * DISPLACEMENT_MAX_MM or lowest is None or worst(lowest) < FIELD_DECIDES * DISPLACEMENT_MAX_MM:
        return None                                                      # the echo-time window or the console's range set the bandwidth, not the field
    third = strict["candidates"][min(SNR_RANK, len(strict["candidates"])) - 1]["snr1"]
    snr_min = _sig3(third * oracle.STRICT["snr"] * math.sqrt(drawn["averages"]) / (1.0 + drawn["margin"]))
    if not SNR_RANGE[0] <= snr_min <= SNR_RANGE[1] or oracle.averages_for(strict["snr1"], snr_min) != drawn["averages"]:
        return None
    loose = oracle.search(world, task, scans, f=f, model=strict["model"])
    if oracle.averages_for(loose["snr1"] * oracle.MODEL_TOL, snr_min) != drawn["averages"]:
        return None                                                      # the stated limits allow fewer averages than a careful measurement reaches
    return task["regions"], snr_min, spec


@functools.lru_cache(maxsize=256)
def _instance(seed: str, level: int) -> tuple:
    """(world seed, regions, SNR floor, pixel ceiling, echo-time window, world spec) of the instance of ``seed``: the first
    world in the fixed sequence of re-draws on which ``_draw`` finds the task. What the task states apart from the SNR
    floor, and the bandwidth the field is to ask for, are drawn before any world is looked at, and drawn again after
    ``PATIENCE`` worlds without the task. A pure function of its arguments (cached: the episodes of an instance share it)."""
    rng = W.rng_for(seed, "task")
    for attempt in range(MAX_ATTEMPTS):
        if attempt % PATIENCE == 0:                                      # a drawn task that no world carries is drawn again
            high = float(TE_HIGH_MS[int(rng.integers(len(TE_HIGH_MS)))])
            drawn = dict(pixel_max_mm=float(PIXELS_MM[int(rng.integers(len(PIXELS_MM)))]), te_window_ms=(high - TE_SPAN_MS, high),
                         averages=int(AVERAGES_TARGET[int(rng.integers(len(AVERAGES_TARGET)))]), margin=float(rng.uniform(*SNR_MARGIN)),
                         bandwidth_hz_per_px=float(np.exp(rng.uniform(*np.log(BANDWIDTH_TARGET)))), ghost_trap=bool(rng.integers(2)),
                         trap=bool(rng.integers(2)) and bool(LEVELS[level].get("oil")))
        world_seed = F.reseed(seed, attempt)
        found = _draw(world_seed, level, drawn)
        if found:
            return world_seed, found[0], found[1], drawn["pixel_max_mm"], drawn["te_window_ms"], found[2]
    raise RuntimeError(f"no instance with an answer in {MAX_ATTEMPTS} draws")


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    world_seed, regions, snr_min, pixel, window, spec = _instance(seed, level)
    lv = LEVELS[level]
    task = dict(family=FAMILY, level=level, regions=[dict(r) for r in regions],
                scan=dict(oracle.SCAN, flip_deg=90.0, partial_fourier=1.0, echo_spacing_ms="console minimum"),
                te_window_ms=[float(window[0]), float(window[1])], pixel_max_mm=pixel, displacement_max_mm=DISPLACEMENT_MAX_MM,
                ghost_max=GHOST_MAX, wrap_max=oracle.WRAP_LIMIT, fat_max=oracle.FAT_LIMIT, snr_min=snr_min, has_fat=bool(lv.get("oil")),
                snr_definition="mean of the magnitude image over the pixels of a region / standard deviation of the noise in one pixel, one "
                               "channel (real or imaginary part)",
                deliverable={"pe_dir": "y or x", "pe_polarity": "1 or -1", "bandwidth_hz_per_px": "<number, hertz per pixel>", "matrix": "<even integer>",
                             "phase_fov": "<fraction, 0.25 to 1>", "centre_pe_mm": "<number, millimetres from the isocentre along the phase-encode axis>",
                             "te_ms": "<number, milliseconds>", "averages": "<integer>"})
    if lv.get("hints"):
        task["hints"] = list(lv["hints"])
    return dict(task_id=task_id or "episetup-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=world_seed,
                world_spec=dict(spec, qa=dict(spec["qa"])), budget_s=float(lv["budget_s"]), max_requests=64, task=task)


def hint_text(cfg: dict) -> str:
    """The method, as the paragraph a level with the hint appends to the statement ("" otherwise)."""
    if "method" not in cfg["task"].get("hints", ()):
        return ""
    return """How to do it (this level states the method).
  1. Field. Run a gradient echo of your own with two echoes (spgr, n_echoes 2). The off-resonance of a region in Hz is the
     difference of its two phase read-outs / (2 pi x the echo spacing in seconds). The same image shows how far the
     object reaches along y and along x.
  2. Displacement of a region in the EPI = its off-resonance x echo spacing x phase field of view (Hz x s x mm), along
     the phase-encode axis. `plan` prints the echo spacing of a bandwidth. Find, for each phase-encode axis, the lowest
     bandwidth whose largest displacement stays below the limit; a shorter phase field of view lowers it further.
  3. Wrap. The phase field of view must hold the object along the phase-encode axis with about 6 mm to spare on each
     side: centre it on the object with centre_pe_mm (the object does not lie on the isocentre, and the off-resonance
     field moves its edges by the displacement of step 2).
  4. Ghost. In an EPI the ghost of anything lies half a phase field of view away along the phase-encode axis. Relative
     to its source its size is |tan(phi)|, with phi = 2 pi x bandwidth_hz_per_px x matrix x delay x readout coordinate
     of the source / fov_mm (delay in seconds; the readout runs along x when pe_dir is y and along y when pe_dir is x).
     Measure it once in an EPI of your own at a high bandwidth, where a ghost falls on empty space, and solve for the
     delay. The ghost of a marked region in a candidate protocol is then |sin(phi)| at its readout coordinate.
  5. SNR. Run the best candidate once and read the regions. The noise of one pixel is the mean magnitude of a
     signal-free area / 1.253; it scales as matrix x sqrt(lines x bandwidth / averages) / (fov x phase field of view
     in mm), so measure it on any scan with many signal-free pixels and scale it to the candidate.
  6. Deliver the candidate that meets every limit with the fewest averages: the smallest matrix that meets the pixel
     ceiling, the shortest echo time inside the window."""


def statement(cfg: dict) -> str:
    t = cfg["task"]
    s = t["scan"]
    ids = " and ".join(r["id"] for r in t["regions"])
    fat = (f"\n  - fat: the object holds a disc of oil; its displaced image may change the signal of {ids} by at most {100 * t['fat_max']:g} %"
           if t["has_fat"] else "")
    hints = hint_text(cfg)
    return f"""# Set up a single-shot EPI that meets its limits in the least scan time
A test object with vials of unknown materials lies in the field of
view. The examination needs a single-shot gradient-echo EPI of the regions {ids}:
    sequence epi, fov_mm {s['fov_mm']:g}, slice_mm {s['slice_mm']:g}, tr_ms {s['tr_ms']:g}, flip_deg {s['flip_deg']:g}, full k-space, no fat saturation,
    echo spacing at the console minimum of the bandwidth.
You choose pe_dir, pe_polarity, bandwidth_hz_per_px, matrix, phase_fov, centre_pe_mm, te_ms and averages. centre_pe_mm is the
centre of the field of view along the phase-encode axis, in mm from the isocentre; along the readout it stays on the isocentre.
Convention of this console: with pe_polarity 1 a spin above resonance is displaced towards the negative end of the
phase-encode axis (-y for pe_dir y, -x for pe_dir x), a spin below resonance (fat) towards the positive end; pe_polarity -1 reverses both.
Regions (circles; centre and radius in mm from the isocentre; each lies inside one homogeneous material):
{F.region_table(t['regions'])}
Requirements:
  - echo time: te_ms between {t['te_window_ms'][0]:g} and {t['te_window_ms'][1]:g} ms
  - pixel: fov_mm / matrix at most {t['pixel_max_mm']:g} mm
  - displacement: the off-resonance field shifts the material of each region by at most {t['displacement_max_mm']:g} mm along the phase-encode axis
  - ghost: the Nyquist ghost of each region (its copy half a phase field of view away) is at most {100 * t['ghost_max']:g} % of the region's signal
  - wrap: the object does not wrap around in the phase-encode direction{" (where the displaced image of the oil lands is judged by the next line only)" if t["has_fat"] else ""}{fat}
  - SNR: at least {t['snr_min']:g} in each region; SNR = {t['snr_definition']}
  - scan time: not longer than the shortest scan time with which any protocol meets the requirements above
Deliver the protocol with `submit`. The scan is not run in this episode: the scanner evaluates the protocol you
deliver. {F.OWN_SCANS} {F.NOT_CHARGED}
Nothing about the object, the field, the noise or the state of the system is known in advance.

Passing: every requirement holds, evaluated on noise-free scans of your protocol on the hidden object: displacement,
ghost{", wrap and fat" if t["has_fat"] else " and wrap"} against the same scan with that one mechanism switched off, the SNR with the noise level of this
scanner. The scanner's log must show that you measured. An answer in the wrong unit is wrong.""" + ("\n\n" + hints if hints else "")


def submit_schema(cfg: dict) -> dict:
    spec = console.SEQUENCES["epi"]
    num = lambda name, what, kind="number": {"type": kind, "minimum": spec[name][0], "maximum": spec[name][1], "description": what}   # noqa: E731
    return {"type": "object", "additionalProperties": False, "required": list(oracle.FIELDS), "properties": {
        "pe_dir": {"type": "string", "enum": ["y", "x"], "description": "phase-encode axis"},
        "pe_polarity": {"type": "integer", "enum": [1, -1], "description": "sign of the phase-encode blips"},
        "bandwidth_hz_per_px": num("bandwidth_hz_per_px", "receiver bandwidth, hertz per pixel"),
        "matrix": num("matrix", "readout matrix (even)", "integer"),
        "phase_fov": num("phase_fov", "phase field of view as a fraction of fov_mm"),
        "centre_pe_mm": {"type": "number", "minimum": spec["centre_y_mm"][0], "maximum": spec["centre_y_mm"][1],
                         "description": "centre of the field of view along the phase-encode axis, millimetres from the isocentre"},
        "te_ms": num("te_ms", "echo time, milliseconds"),
        "averages": num("averages", "number of averages", "integer")}}


# ------------------------------------------------------------------------------------------------ the verifier
protocol_of = oracle.protocol


def check(cfg: dict, answer, exam: dict):
    if not isinstance(answer, dict) or set(answer) != set(oracle.FIELDS):
        return "the deliverable is one protocol with exactly the fields " + ", ".join(oracle.FIELDS)
    if isinstance(answer["te_ms"], str):
        return "te_ms is a number of milliseconds"
    pl = console.plan(protocol_of(answer), exam)
    if not pl["accept"]:
        return "the console rejects the delivered scan: " + "; ".join(v["message"] for v in pl["violations"])
    return None


_optima = collections.OrderedDict()


def optimum(cfg: dict, world: dict, scans: M.Scans) -> dict:
    """The fewest averages of the instance on the hidden world (``oracle.least``). Cached per instance: the episodes of
    an instance in this process share one search."""
    t = cfg["task"]
    key = (cfg["seed"], cfg["level"], tuple(r["id"] for r in t["regions"]), t["snr_min"], t["pixel_max_mm"], tuple(t["te_window_ms"]))
    if key not in _optima:
        found = oracle.least(cfg, world, scans)
        _optima[key] = dict(averages=found["averages"], fields=found["fields"])
        while len(_optima) > 16:
            _optima.popitem(last=False)
    return _optima[key]


def within(value: float, limit: float) -> bool:
    return bool(value <= limit)


def reaches(value: float, floor: float) -> bool:
    return bool(value >= floor)


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    problem = check(cfg, answer, world["console"])
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    t = cfg["task"]
    scans = M.Scans(world, engine)
    m = oracle.evaluate(t, answer, world, scans)
    ids = [r["id"] for r in t["regions"]]
    checks = dict(deliverable_valid=True, echo_time=t["te_window_ms"][0] - 1e-9 <= m["te_ms"] <= t["te_window_ms"][1] + 1e-9,
                  pixel=within(m["pixel_mm"], t["pixel_max_mm"] + 1e-9))
    checks.update({f"displacement_{i}": within(m["displacement_mm"][i], t["displacement_max_mm"]) for i in ids})
    checks.update({f"ghost_{i}": within(m["ghost"][i], t["ghost_max"]) for i in ids})
    checks["wrap"] = within(m["wrap"], t["wrap_max"])
    checks.update({f"fat_{i}": within(m["fat"][i], t["fat_max"]) for i in ids})
    checks.update({f"snr_{i}": reaches(m["snr"][i], t["snr_min"]) for i in ids})
    rest_ok = all(checks.values())
    best = optimum(cfg, world, scans)
    fewest = best["averages"] if best["averages"] is not None else (int(answer["averages"]) if rest_ok else None)
    checks["time"] = fewest is not None and int(answer["averages"]) <= fewest
    bad = []
    if not checks["echo_time"]:
        bad.append(f"te_ms {m['te_ms']:g} outside {t['te_window_ms'][0]:g} to {t['te_window_ms'][1]:g}")
    if not checks["pixel"]:
        bad.append(f"pixel {m['pixel_mm']:.3g} mm, allowed {t['pixel_max_mm']:g}")
    for i in ids:
        if not checks[f"displacement_{i}"]:
            bad.append(f"{i} displaced by {m['displacement_mm'][i]:.2f} mm, allowed {t['displacement_max_mm']:g}")
        if not checks[f"ghost_{i}"]:
            bad.append(f"ghost of {i} {100 * m['ghost'][i]:.1f} %, allowed {100 * t['ghost_max']:g}")
        if not checks[f"fat_{i}"]:
            bad.append(f"displaced fat changes {i} by {100 * m['fat'][i]:.0f} %, allowed {100 * t['fat_max']:g}")
        if not checks[f"snr_{i}"]:
            bad.append(f"SNR {m['snr'][i]:.3g} in {i}, required {t['snr_min']:g}")
    if not checks["wrap"]:
        bad.append(f"wrap-around: {100 * m['wrap']:.2f} % of the image energy, allowed {100 * t['wrap_max']:g}")
    if not checks["time"]:
        bad.append(f"{answer['averages']} averages; the requirements can be met with fewer")
    # ``error`` is the regret of the delivered scan: its averages minus the fewest that suffice (0 at the optimum)
    return F.result(checks, "; ".join(bad), error=None if fewest is None else float(int(answer["averages"]) - fewest), measured=m,
                    fewest_averages=fewest, scan_s=m["scan_s"])


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured answer that fails a check: the first failed requirement."""
    c = score["checks"]
    for label, prefix in zip(OUTCOMES, ("echo_time", "pixel", "displacement", "ghost", "wrap", "fat", "snr", "time")):
        if any(not ok for k, ok in c.items() if k == prefix or k.startswith(prefix + "_")):
            return label
    return OUTCOMES[-1]


def task_kind(stored: dict | None = None) -> dict:
    """Measurement task or decision task, per level, from a gate result (default: the stored one), never typed."""
    from ... import gate
    return F.kind_by_open_loop(stored or gate.load(ENV_ID))


def card() -> dict:
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    kind = task_kind()
    return dict(
        id=ENV_ID, title="Set up a single-shot EPI that meets its limits in the least scan time", kind="candidate",
        given="An elongated, off-centre test object with vials of hidden materials, two of them marked; the scanner; the requirements of an "
              "echo-planar scan; a scan-time budget for probing",
        deliver="One echo-planar protocol: phase-encode axis and blip polarity, receiver bandwidth, matrix, phase field of view and its centre, echo time, averages",
        scored_by="Noise-free scans of the delivered protocol on the hidden object and their twins with one mechanism switched off: "
                  "displacement of each region against the scan without the off-resonance field, its ghost against the scan without the "
                  "gradient delay, wrap-around against a scan whose phase field of view holds the object, displaced fat against the scan "
                  "without chemical shift, the SNR with the analytic noise of the hidden receiver; and no more averages than the fewest "
                  "with which any allowed protocol meets all of it (exhaustive search on the hidden world)",
        summary="Echo-planar imaging trades everything against everything: a long echo spacing has little noise and little ghost and "
                "a large displacement; a short phase field of view shortens the train and wraps; the phase-encode axis decides "
                "which regions ghost and where fat lands. The field, the gradient delay, the shape of the object and the noise are "
                "hidden, so the protocol has to be measured for. " + OPEN_LOOP_NOTE + " " + kind["sentence"],
        task_kind=dict(kind=kind["kind"], levels=kind["levels"], evidence=kind["evidence"]),
        needs=list(NEEDS),
        world=dict(phantom="QA phantom (an elongated disc of gel with vials; at level 3 also a disc of oil), 1.5 T", sequences_useful=["spgr", "epi"],
                   agent_side_formulas="stated at level 1 only: displacement = off-resonance x echo spacing x phase field of view; ghost of "
                                       "a source = |tan| (in the image) or |sin| (against the scan without delay) of 2 pi x bandwidth x matrix x "
                                       "delay x readout coordinate / fov; the SNR laws. Not given: the fat spectrum has several lines",
                   fixed="full k-space and no fat saturation: the family was built before engine 0.3.0 corrected the noise of partial Fourier "
                         "and the fat-saturation pulse, and its instances and search were made without either"),
        hidden=["the off-resonance field and the centre-frequency error", "the gradient delay", "size, orientation and position of the object (up to 22 mm off the isocentre along each axis)",
                "where the oil lies (level 3)", "T1, T2* and proton density of every material", "receiver noise level",
                "transmit gain and B1 pattern", "receive profile", "the 128-bit world seed"],
        told=["the fixed fields of the scan", "the regions", "every limit: echo-time window, pixel ceiling, displacement, ghost, wrap, fat, SNR floor",
              "the definition of SNR", "the instrument datasheet", "realised protocol, timing and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        limits=dict(max_scanner_requests=64, max_tool_calls=72),
        deliverable={"tool": "submit", "format": {"pe_dir": "y | x", "pe_polarity": "1 | -1", "bandwidth_hz_per_px": "<hertz per pixel>",
                                                  "matrix": "<even integer>", "phase_fov": "<fraction>", "centre_pe_mm": "<millimetres>",
                                                  "te_ms": "<milliseconds>", "averages": "<integer>"}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan", "the console accepts the delivered scan"],
                    error="averages of the delivered scan minus the fewest with which any allowed protocol meets the requirements",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing=f"all gates; echo time in the window; pixel within the ceiling; displacement <= {DISPLACEMENT_MAX_MM:g} mm and ghost <= "
                            f"{100 * GHOST_MAX:g} % for each region; wrap-around energy <= {100 * oracle.WRAP_LIMIT:g} %; fat <= {100 * oracle.FAT_LIMIT:g} % "
                            "per region; SNR >= the floor in each region; averages <= the fewest that suffice"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier's scans are noise-free and its noise is analytic",
    )


_ = np
