"""``Scanner/BandFree``: instances and the verifier.

The job: two balanced SSFP images, each with no dark band across its marked regions. The bands sit where the phase by
which a spin runs ahead of the RF pulse per repetition, 2 pi x off-resonance x TR - phase cycle, is a multiple of 2 pi; they repeat every
1 / TR in frequency and are as wide as the material and the excitation make them. Where the regions lie in that
pattern depends on the hidden field, on the hidden error of the frequency adjustment, on the material (oil sits
3.4 ppm below water) and on the repetition time, which follows from the receiver bandwidth. A low bandwidth gives
the least noise and the longest repetition time, so the narrowest pattern.

Deliverable    {"A": {"bandwidth_hz_per_px": 240.0, "phase_cycle_deg": [112.0]},             one acquisition
                "B": {"bandwidth_hz_per_px": 130.0, "phase_cycle_deg": [40.0, 205.0]}}       two acquisitions; a region is read from one of them
Pass rule      for each of the two images, all of
               1  band-free with a margin: every marked region of the image keeps >= ``keep_min`` of its signal at its
                  best phase cycle (same bandwidth and repetition time), at the adjusted centre frequency and at every
                  drift of the centre frequency of up to ``drift_hz`` either way. With two acquisitions: in one and
                  the same of the two (the larger of the two signals at each drift would let two opposite phase
                  cycles pass on any object whose bands are narrower than half their spacing).
               2  two acquisitions only when no single acquisition at a bandwidth <= ``single_up_to`` is band-free
                  (below the top of the allowed range: a single acquisition that exists only at the very top never
                  decides an episode; there both answers pass)
               3  bandwidth <= ``bandwidth_slack`` x the lowest bandwidth at which a band-free protocol with the same
                  number of acquisitions exists
               The comparison protocols of rules 2 and 3 are those that stay band-free up to ``reference_drift_hz``
               (larger than ``drift_hz``): a protocol that only just exists never decides an episode. That matters
               most where band-free bandwidths are islands and not a half-line: with oil among the regions a single
               acquisition exists only near repetition times at which fat and water are in step, and the lowest such
               island can be a fraction of a hertz wide in drift.

Why two images. With one image the deliverable has one hard coordinate, the bandwidth, with a tolerance of 15 %
inside a range that the physics makes narrow: the target image at the best constant bandwidth with a sweep of the
phase cycle and no model then passes far more often than the gate allows. Two images with their own regions make
that a product while each tolerance stays as wide as it was; the same probe scans serve both.

The target is a 2D acquisition (slice-selective excitation, the console default), and that decides the shape of a
band. The spins in the transition of the slice profile see a smaller flip angle than the slice centre, and the
steady state of a small flip angle peaks right next to the band. Integrated through the slice, a region keeps 80 %
of its best signal over a larger share of the band spacing than with a hard pulse (3D) at the same nominal angle
(``tests/test_band_free.py`` measures both for T1 / T2 from 40 to 2): the band of a 2D image is narrower than the
textbook formula says, and a band model has to carry the slice profile. ``excitation: nonselective`` is on the
console at a multiple of the scan time; a scout made with it shows the hard-pulse band, not the band of the target.

How it is evaluated. The delivered protocol is scanned noise-free on the hidden object at the adjusted centre
frequency (and so is the best phase cycle of each region); rule 1 at the present frequency is read from those scans.
A drift of the centre frequency by d Hz is the same thing as a change of the phase cycle by 360 x d x TR degrees
(the other way) for this sequence; the drift part of rule 1 and the set of all band-free protocols of rules 2 and 3 are computed from the
echo amplitudes on each region's support (``Evaluator``: no k-space, every whole degree of the phase cycle, a grid
over the bandwidth), with the readout displacement of off-resonant spins included.
"""
from __future__ import annotations

import collections
import os

import numpy as np

from ...scanner import console, tissue
from ...scanner import world as W
from .. import family as F

FAMILY = "band_free"
SUBMIT_DOC = ("Deliver, for each of the two images, the receiver bandwidth of the balanced SSFP image and its phase cycle: one "
              "phase cycle for a single acquisition, two for a pair of acquisitions of which every region uses the one that "
              "shows it better.")
TARGET = dict(sequence="bssfp", flip_deg=60.0, matrix=120)
BANDWIDTH = (60.0, 600.0)                            # allowed bandwidths (level 3: from OIL_BANDWIDTH_MIN)
OIL_BANDWIDTH_MIN = 90.0                             # below it the image of an oil region is displaced out of its vial (3.4 ppm / bandwidth)
KEEP_MIN, DRIFT_HZ, REFERENCE_DRIFT_HZ, BANDWIDTH_SLACK = 0.8, 20.0, 26.0, 1.15
SINGLE_UP_TO = 400.0                                 # rule 2 compares with single acquisitions up to this bandwidth
GRID_STEP = 0.02                                     # the verifier's bandwidth grid: steps of 2 %
IMAGES = dict(A=(1, 2, 3), B=(4, 5))                 # image: the numbers of its marked regions (R1 ... R5)
MAX_ATTEMPTS = 64
# examined competencies this family exercises (ids of ``mrgym.coverage.COMPETENCIES``)
COMPETENCIES = ("parameter.tr", "parameter.receive_bandwidth", "option.2d_3d", "artefact.chemical_shift", "artefact.b0_distortion",
                "qc.centre_frequency")
# Outcome labels of a delivered, measured answer that fails (worst first) and the checks behind each: (label, stems),
# a check belongs to a requirement when its name is ``<stem>_<image>``. The label is the first of the three pass rules
# that the delivery misses, in the order of the rules (the order in which ``score`` declares the checks of one image);
# it names the rule, never the image or the region.
REQUIREMENTS = (("band_on_region", ("band_free",)),                            # rule 1
                ("second_acquisition_not_needed", ("acquisitions_needed",)),   # rule 2
                ("bandwidth_too_high", ("bandwidth_low",)))                    # rule 3
OUTCOMES = tuple(label for label, _ in REQUIREMENTS)
QA = dict(n_ring=(5, 6), roi_radius_mm=6.0)
HIGH_RATIO = dict(vial_t1_ms=(1500.0, 2500.0), vial_t2_ms=(40.0, 60.0))            # T1 / T2 of 25 to 62
BASE = dict(kind="qa", b0_t=3.0, cf_offset_hz=150.0)
LEVELS = {
    1: dict(world_spec=dict(BASE, qa=dict(QA, **HIGH_RATIO)), b0_hz=(60.0, 160.0), oil=False, bandwidth=BANDWIDTH, budget_s=43.0,
            summary="materials with T1 much longer than T2: one band shape for all, the widest band a 2D image has. The "
                    "field spread sets the bandwidth and, where it exceeds the pass band, the number of acquisitions; the "
                    "frequency error sets the phase cycle"),
    2: dict(world_spec=dict(BASE, qa=dict(QA)), b0_hz=(60.0, 160.0), oil=False, bandwidth=BANDWIDTH, budget_s=43.0,
            summary="arbitrary materials: the width of the band differs between the regions and has to be measured"),
    3: dict(world_spec=dict(BASE, qa=dict(QA, n_oil=1)), b0_hz=(120.0, 280.0), oil=True, bandwidth=(OIL_BANDWIDTH_MIN, BANDWIDTH[1]), budget_s=43.0,
            summary="one marked region holds oil (not told which, nor that): its bands sit 3.4 ppm away, so the repetition "
                    "time has to suit oil and water at once, or the image needs two acquisitions; and a larger field spread"),
}


# ------------------------------------------------------------------------------------------------ the target protocol
def tr_min_ms(bandwidth: float, exam: dict):
    """Console minimum of the repetition time of the target image at ``bandwidth`` (None when the console refuses)."""
    pl = console.plan(dict(TARGET, bandwidth_hz_per_px=float(bandwidth), tr_ms=console.SEQUENCES["bssfp"]["tr_ms"][1]), exam)
    return pl["timing"]["tr_min_ms"] if pl["accept"] else None


def target_plan(bandwidth: float, phase_cycle_deg: float, exam: dict) -> dict:
    tr = tr_min_ms(bandwidth, exam)
    if tr is None:
        return console.plan(dict(TARGET, bandwidth_hz_per_px=float(bandwidth), tr_ms=50.0), exam)
    return console.plan(dict(TARGET, bandwidth_hz_per_px=float(bandwidth), tr_ms=tr, phase_cycle_deg=float(phase_cycle_deg) % 360.0), exam)


def drift_deg(drift_hz: float, tr_ms: float) -> float:
    """Change of the phase per repetition by a drift of the centre frequency: the same as a change of the phase cycle."""
    return 360.0 * drift_hz * tr_ms * 1e-3


def image_regions(task: dict, image: str) -> list:
    """The marked regions of one image, in the order of the task card."""
    ids = set(task["images"][image])
    return [r for r in task["regions"] if r["id"] in ids]


# ------------------------------------------------------------------------------------------------ instances
def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    """The world of an instance is that of ``family.reseed(seed, attempt)``: at a level with an oil vial, the first
    attempt on which the oil vial is one of the marked regions (so the ids R1 ... R5 say nothing about it)."""
    lv = LEVELS[level]
    lo, hi = lv["b0_hz"]
    n_marked = max(max(v) for v in IMAGES.values())
    for attempt in range(MAX_ATTEMPTS):
        world_seed = F.reseed(seed, attempt)
        spec = dict(lv["world_spec"], b0_hz=round(float(W.rng_for(world_seed, "task").uniform(lo, hi)), 2))   # hidden, per instance
        layout = W.make_world(world_seed, spec, layout_only=True)
        marked = sorted((r for r in layout["regions"] if int(r["id"][1:]) <= n_marked), key=lambda r: int(r["id"][1:]))
        if len(marked) == n_marked and (not lv["oil"] or any(layout["tissues"][r["cls"]]["fat"] for r in marked)):
            break
    else:
        raise RuntimeError(f"no instance within {MAX_ATTEMPTS} attempts")
    task = dict(family=FAMILY, level=level, regions=F.public_regions(marked),
                images={name: [f"R{k}" for k in numbers] for name, numbers in IMAGES.items()},
                target_scan=dict(TARGET, averages=1, slice_mm=5.0), target_tr_ms="the console minimum for the delivered bandwidth",
                bandwidth_hz_per_px=list(lv["bandwidth"]), keep_min=KEEP_MIN, drift_hz=DRIFT_HZ, reference_drift_hz=REFERENCE_DRIFT_HZ,
                bandwidth_slack=BANDWIDTH_SLACK, single_up_to=SINGLE_UP_TO,
                deliverable={name: {"bandwidth_hz_per_px": "<number>", "phase_cycle_deg": "<list of one or two numbers, degrees>"}
                             for name in IMAGES})
    return dict(task_id=task_id or "bandfree-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=world_seed, attempt=attempt,
                world_spec=spec, budget_s=float(lv["budget_s"]), max_requests=60, task=task)


def statement(cfg: dict) -> str:
    t = cfg["task"]
    s = t["target_scan"]
    lo, hi = t["bandwidth_hz_per_px"]
    names = list(t["images"])
    tables = "\n".join(f"  image {name}\n{F.region_table(image_regions(t, name))}" for name in names)
    return f"""# Keep the dark bands of two balanced SSFP images off their marked regions
A test object with vials of unknown materials lies in the field of
view. The examination needs two balanced SSFP images of it, image {names[0]} and image {names[1]}, each with a protocol of its own:
    sequence bssfp, flip_deg {s['flip_deg']:g}, matrix {s['matrix']}, field of view at the console default, slice 5 mm with the console's
    default slice-selective pulse, 1 average, bandwidth_hz_per_px of your choice between {lo:g} and {hi:g}, tr_ms at the console
    minimum for that bandwidth (a plan shows it), phase_cycle_deg of your choice
No dark band may cross the marked regions of an image (circles; centre and radius in mm from the isocentre; each lies
inside one homogeneous material):
{tables}
Deliver with `submit`, for each image, the bandwidth and one phase cycle. If no single acquisition can keep every
marked region of an image out of the bands, deliver for that image the bandwidth and two phase cycles: two
acquisitions that differ only in the phase cycle. Every region is then read from one of the two, the one that shows
it better, and rule 1 must hold for it in that one.

Passing, for each image, all of:
1. Band-free, with a margin. Every marked region of the image keeps at least {100 * t['keep_min']:g} % of the largest signal it can have
   at this bandwidth and repetition time (its signal at its best phase cycle). This must hold at the present centre
   frequency and at every drift of the centre frequency of up to {t['drift_hz']:g} Hz either way (the same drift everywhere in the
   object).
2. No more acquisitions than needed. Two acquisitions pass only when no single acquisition at a bandwidth of {t['single_up_to']:g} or
   less is band-free for that image.
3. Low bandwidth (the noise of the image grows with the square root of the bandwidth). The delivered bandwidth is at
   most {t['bandwidth_slack']:g} times the lowest bandwidth at which a band-free protocol with the same number of acquisitions exists
   for that image.
For rules 2 and 3 the scanner compares with protocols that stay band-free up to a drift of {t['reference_drift_hz']:g} Hz, so a protocol
that only just exists does not count against you.

The target images are not acquired in this episode: the scanner evaluates the protocols you deliver, noise-free.
{F.OWN_SCANS} {F.NOT_CHARGED}
Nothing about the materials, the field or the frequency adjustment is known in advance. The scanner's log must show that you measured. An answer in the wrong unit is wrong."""


def submit_schema(cfg: dict) -> dict:
    lo, hi = cfg["task"]["bandwidth_hz_per_px"]
    image = {"type": "object", "additionalProperties": False, "required": ["bandwidth_hz_per_px", "phase_cycle_deg"],
             "properties": {"bandwidth_hz_per_px": {"type": "number", "minimum": lo, "maximum": hi,
                                                    "description": "receiver bandwidth of this image, Hz per pixel"},
                            "phase_cycle_deg": {"type": "array", "items": {"type": "number", "minimum": 0.0, "maximum": 360.0},
                                                "description": "RF phase advance per TR in degrees: a list of one value for a "
                                                               "single acquisition, of two values for two acquisitions"}}}
    names = list(cfg["task"]["images"])
    return {"type": "object", "additionalProperties": False, "required": names,
            "properties": {n: dict(image, description=f"protocol of image {n} (regions {', '.join(cfg['task']['images'][n])})") for n in names}}


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def check(cfg: dict, answer, exam: dict):
    lo, hi = cfg["task"]["bandwidth_hz_per_px"]
    names = list(cfg["task"]["images"])
    form = ("the deliverable is {" + ", ".join(f'"{n}": ...' for n in names)
            + '} with {"bandwidth_hz_per_px": <number>, "phase_cycle_deg": [<one or two numbers>]} for each image')
    if not isinstance(answer, dict) or set(answer) != set(names):
        return form
    for n in names:
        a = answer[n]
        if not isinstance(a, dict) or set(a) != {"bandwidth_hz_per_px", "phase_cycle_deg"}:
            return form
        bw, phis = a["bandwidth_hz_per_px"], a["phase_cycle_deg"]
        if not _number(bw) or not isinstance(phis, (list, tuple)) or len(phis) not in (1, 2) or not all(_number(p) for p in phis):
            return form
        if not lo <= bw <= hi:
            return f"image {n}: bandwidth_hz_per_px={bw:g} is outside the allowed range {lo:g} to {hi:g} Hz per pixel"
        if not all(0.0 <= p <= 360.0 for p in phis):
            return f"image {n}: phase_cycle_deg must lie between 0 and 360 degrees"
        pl = target_plan(bw, phis[0], exam)
        if not pl["accept"]:
            return f"the console rejects the target scan of image {n}: " + "; ".join(v["message"] for v in pl["violations"])
    return None


# ------------------------------------------------------------------------------------------------ the set of band-free protocols
class Evaluator:
    """Verifier side: the signal of every marked region as a function of bandwidth and phase cycle, from the echo
    amplitudes on the region's support (no k-space). ``keep(bandwidth)`` -> array [region, 360]: signal at each whole
    degree of the phase cycle relative to the region's best phase cycle.

    The steady state itself is never written down here: for each region it is read from the engine
    (``Engine.amplitudes`` on a small auxiliary world that holds the region's material at the region's transmit
    field over one period of off-resonance), so the table follows whatever the engine computes for this sequence:
    the slice profile of the excitation and, for oil, the chemical shift."""

    def __init__(self, world: dict, regions: list):
        from scipy import ndimage
        self.world, self.regions, self._nd = world, regions, ndimage
        self.exam = world["console"]
        self.d, self.n = world["extent_mm"] / world["n"], world["n"]
        self.df = world["fields"]["b0_hz"] + world["system"]["cf_offset_hz"]
        self.rx = world["fields"]["rx"]
        cls = {r["id"]: r["cls"] for r in world["regions"]}
        fat_hz = tissue.FAT_PPM * 1e-6 * console.GAMMA_HZ_PER_T * world["b0_t"]
        m = TARGET["matrix"]
        grid = (np.arange(m) - m // 2) * self.exam["fov_mm"] / m          # pixel centres of the target image, mm
        yy, xx = np.meshgrid(grid, grid, indexing="ij")
        self.points, self._cache = [], {}
        for r in regions:                                                # a region is read from the pixels whose centre lies in the circle
            inside = (yy - r["y_mm"]) ** 2 + (xx - r["x_mm"]) ** 2 <= r["radius_mm"] ** 2
            centre = (int(round(r["y_mm"] / self.d)) + self.n // 2, int(round(r["x_mm"] / self.d)) + self.n // 2)
            self.points.append(dict(y=yy[inside], x=xx[inside], cls=cls[r["id"]], centre=centre,
                                    shift_hz=fat_hz if world["tissues"][cls[r["id"]]]["fat"] else 0.0))

    def _at(self, field, y_mm, x_mm):
        idx = np.stack([y_mm / self.d + self.n // 2, x_mm / self.d + self.n // 2])
        if np.iscomplexobj(field):
            return self._nd.map_coordinates(field.real, idx, order=1) + 1j * self._nd.map_coordinates(field.imag, idx, order=1)
        return self._nd.map_coordinates(field, idx, order=1)

    def steady_state(self, point: dict, realised: dict, side: int = 20) -> np.ndarray:
        """Complex steady state of the region's material at every whole degree of the phase per TR that the field
        gives it (0 .. 359; the chemical shift of oil comes on top, inside the engine), as the engine computes it for
        ``realised`` (phase cycle 0), without the phase the echo gains from the precession."""
        from ...scanner import kspace
        w, (i, j) = self.world, point["centre"]
        theta = 360.0 * np.arange(side * side) / (side * side)
        fields = {k: (np.full((side, side), v[i, j], v.dtype) if isinstance(v, np.ndarray) and v.shape == (self.n, self.n) else v)
                  for k, v in w["fields"].items()}
        fields["b0_hz"] = (theta / (360.0 * realised["tr_ms"] * 1e-3) - w["system"]["cf_offset_hz"]).reshape(side, side)
        aux = dict(w, n=side, extent_mm=w["extent_mm"] * side / self.n, names=[point["cls"]], tissues={point["cls"]: w["tissues"][point["cls"]]},
                   weights=np.ones((1, side, side), np.float32), fields=fields, regions=[])
        amp = np.asarray(kspace.Engine(aux).amplitudes(realised, 0)[0]) * np.exp(-0.5j * np.deg2rad(theta))
        whole = np.arange(360.0)
        return np.interp(whole, theta, amp.real, period=360.0) + 1j * np.interp(whole, theta, amp.imag, period=360.0)

    def amplitude(self, bandwidth: float) -> tuple:
        """(|complex mean| of every region at phase cycles 0..359 degrees [region, 360], repetition time in ms)."""
        key = round(float(bandwidth), 4)
        if key in self._cache:
            return self._cache[key]
        realised = target_plan(bandwidth, 0.0, self.exam)["realised"]
        tr = realised["tr_ms"]
        px = self.exam["fov_mm"] / TARGET["matrix"]
        phi = np.arange(360.0)
        out = []
        for p in self.points:
            x = p["x"]
            for _ in range(2):                                           # where the spins seen at this pixel really are
                x = p["x"] + (self._at(self.df, p["y"], x) + p["shift_hz"]) / bandwidth * px
            theta = 360.0 * self._at(self.df, p["y"], x) * tr * 1e-3     # phase per TR of every point from the field, degrees
            table = self.steady_state(p, realised)
            pos = (theta[:, None] - phi[None, :]) % 360.0              # the spin runs ahead of the pulse by its phase per TR minus the phase cycle
            lo = np.floor(pos).astype(int) % 360
            s = table[lo] * (1 - (pos - lo)) + table[(lo + 1) % 360] * (pos - lo)
            out.append(np.abs((s * (np.exp(0.5j * np.deg2rad(theta)) * self._at(self.rx, p["y"], x))[:, None]).mean(0)))
        if len(self._cache) >= 8:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (np.array(out), tr)
        return self._cache[key]

    def keep(self, bandwidth: float) -> tuple:
        amp, tr = self.amplitude(bandwidth)
        return amp / amp.max(1, keepdims=True), tr


def bandwidth_grid(allowed=BANDWIDTH) -> np.ndarray:
    lo, hi = allowed
    return np.round(np.exp(np.arange(np.log(lo), np.log(hi) + 1e-9, GRID_STEP)), 2)


def robust(keep: np.ndarray, tr_ms: float, drift_hz: float, keep_min: float = KEEP_MIN) -> np.ndarray:
    """[region, 360] bool: the region is band-free at this phase cycle at every drift from -``drift_hz`` to +``drift_hz``
    (every whole degree of the equivalent change of the phase cycle, rounded outwards)."""
    m = int(np.ceil(drift_deg(drift_hz, tr_ms) - 1e-9))
    ok = keep >= keep_min
    if 2 * m + 1 >= 360:
        return np.zeros_like(ok)
    run = np.cumsum(np.concatenate([np.zeros((ok.shape[0], 1), int), np.tile(ok, (1, 3)).astype(int)], axis=1), axis=1)
    i = np.arange(360, 720)
    return run[:, i + m + 1] - run[:, i - m] == 2 * m + 1


def window_min(keep_row: np.ndarray, phase_cycle_deg: float, half_deg: float) -> float:
    """Smallest kept fraction of one region over the phase cycles ``phase_cycle_deg`` -/+ ``half_deg`` (the table holds
    whole degrees; between them it is linear, so the ends of the window and the whole degrees inside it suffice)."""
    if half_deg >= 180.0:
        return float(keep_row.min())
    a, b = phase_cycle_deg - half_deg, phase_cycle_deg + half_deg
    pos = np.concatenate([[a, b], np.arange(np.ceil(a), np.floor(b) + 1)]) % 360.0
    lo = np.floor(pos).astype(int) % 360
    return float((keep_row[lo] * (1 - (pos - lo)) + keep_row[(lo + 1) % 360] * (pos - lo)).min())


def pair_exists(ok: np.ndarray) -> bool:
    """Two phase cycles such that every region is band-free (with drift) in at least one of the two acquisitions."""
    miss = np.unique(((~ok) * (1 << np.arange(ok.shape[0]))[:, None]).sum(0))
    return bool(np.any((miss[:, None] & miss[None, :]) == 0))


_REFERENCE = collections.OrderedDict()


def lowest_bandwidths(ev: Evaluator, rows: list, drift_hz: float, keep_min: float, allowed=BANDWIDTH) -> dict:
    """The lowest bandwidth of the grid at which one acquisition, and at which a pair, keeps the regions ``rows``
    (indices into the evaluator's regions) band-free up to ``drift_hz`` (None where there is none)."""
    single = pair = None
    for bw in bandwidth_grid(allowed):
        keep, tr = ev.keep(float(bw))
        ok = robust(keep[rows], tr, drift_hz, keep_min)
        if single is None and ok.all(0).any():
            single = float(bw)
        if pair is None and pair_exists(ok):
            pair = float(bw)
        if single is not None and pair is not None:
            break
    return dict(single_bandwidth=single, pair_bandwidth=pair)


def reference_protocols(cfg: dict, world: dict) -> dict:
    """Verifier side, per image: the lowest bandwidth at which one acquisition, and at which a pair, stays band-free up
    to the reference drift (None where there is none within the allowed bandwidths). Cached per instance."""
    key = (cfg["seed"], cfg["level"])
    if key not in _REFERENCE:
        t = cfg["task"]
        ev = Evaluator(world, t["regions"])
        index = {r["id"]: k for k, r in enumerate(t["regions"])}
        _REFERENCE[key] = {name: lowest_bandwidths(ev, [index[i] for i in ids], t["reference_drift_hz"], t["keep_min"], t["bandwidth_hz_per_px"])
                           for name, ids in t["images"].items()}
        while len(_REFERENCE) > 16:
            _REFERENCE.popitem(last=False)
    return _REFERENCE[key]


def kept(cfg: dict, image: str, bandwidth: float, phase_cycles: list, world: dict, engine, ev: Evaluator | None = None) -> dict:
    """Verifier side, rule 1 for one image: {region id: dict(now, drift, acquisition)}.

    ``now``    kept fraction at the adjusted centre frequency, from noise-free scans: the region's signal in the
               delivered acquisition over its signal at its best phase cycle (the best phase cycle is taken from the
               amplitudes, its signal from a scan)
    ``drift``  smallest kept fraction over every drift of the centre frequency up to ``drift_hz`` either way, from the
               echo amplitudes on the region's support
    With two acquisitions a region is read from one of them, the same at every drift: the one whose smaller number
    of the two is larger (``acquisition``: its index)."""
    t = cfg["task"]
    regions, exam = image_regions(t, image), world["console"]
    ev = ev or Evaluator(world, t["regions"])
    rows = [k for k, r in enumerate(t["regions"]) if r["id"] in set(t["images"][image])]
    amp, tr = ev.amplitude(bandwidth)
    amp = amp[rows]
    table = amp / amp.max(1, keepdims=True)
    scan = lambda phi: {k: abs(v) for k, v in F.region_complex(engine, target_plan(bandwidth, phi, exam)["realised"], regions).items()}   # noqa: E731
    best = {r["id"]: 0.0 for r in regions}
    done = []
    for phi in sorted(int(a) for a in amp.argmax(1)):                    # the top of a profile is flat: best phase cycles
        if all(min(abs(phi - p), 360 - abs(phi - p)) > 8 for p in done):  # within 8 degrees of each other share a scan
            done.append(phi)
            best = {k: max(best[k], v) for k, v in scan(float(phi)).items()}
    sig = [scan(phi) for phi in phase_cycles]                            # [acquisition][region]
    best = {k: max([v] + [s[k] for s in sig]) for k, v in best.items()}
    half = drift_deg(t["drift_hz"], tr)
    out = {}
    for row, r in zip(table, regions):
        k = r["id"]
        per = [(float(s[k] / max(best[k], 1e-30)), window_min(row, phi, half)) for s, phi in zip(sig, phase_cycles)]
        use = max(range(len(per)), key=lambda a: min(per[a]))
        out[k] = dict(now=round(per[use][0], 5), drift=round(per[use][1], 5), acquisition=use)
    return out


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    from ...scanner import kspace
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    engine = engine or kspace.Engine(world)
    problem = check(cfg, answer, world["console"])
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    t = cfg["task"]
    ref = reference_protocols(cfg, world)
    ev = Evaluator(world, t["regions"])
    checks, notes, images = dict(deliverable_valid=True), [], {}
    for name in t["images"]:
        bw, phis = float(answer[name]["bandwidth_hz_per_px"]), [float(p) for p in answer[name]["phase_cycle_deg"]]
        keep = kept(cfg, name, bw, phis, world, engine, ev)
        worst_id = min(keep, key=lambda k: min(keep[k]["now"], keep[k]["drift"]))
        worst = min(keep[worst_id]["now"], keep[worst_id]["drift"])
        lowest = ref[name]["single_bandwidth"] if len(phis) == 1 else ref[name]["pair_bandwidth"]
        single = ref[name]["single_bandwidth"]
        checks[f"band_free_{name}"] = worst >= t["keep_min"]
        checks[f"acquisitions_needed_{name}"] = len(phis) == 1 or single is None or single > t["single_up_to"]
        checks[f"bandwidth_low_{name}"] = lowest is None or bw <= t["bandwidth_slack"] * lowest * (1 + 1e-9)
        if not checks[f"band_free_{name}"]:
            where = "at the present centre frequency" if keep[worst_id]["now"] <= keep[worst_id]["drift"] else f"within a drift of {t['drift_hz']:g} Hz"
            notes.append(f"image {name}: {worst_id} keeps {100 * worst:.0f} % {where}")
        if not checks[f"acquisitions_needed_{name}"]:
            notes.append(f"image {name}: one acquisition is band-free at {single:g} Hz per pixel")
        if not checks[f"bandwidth_low_{name}"]:
            notes.append(f"image {name}: a band-free protocol exists at {lowest:g} Hz per pixel")
        images[name] = dict(keep=keep, keep_worst=round(float(worst), 5), acquisitions=len(phis), bandwidth_hz_per_px=bw,
                            lowest_bandwidth=dict(ref[name]), bandwidth_ratio=None if lowest is None else round(bw / lowest, 4))
    worst = min(d["keep_worst"] for d in images.values())
    return F.result(checks, "; ".join(notes), error=round(1.0 - worst, 5), keep_worst=worst, images=images)


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured answer that fails a check: one of ``OUTCOMES``, the first requirement
    of ``REQUIREMENTS`` with a failed check. A function of the names and values of ``score["checks"]`` and nothing else."""
    failed = [k for k, ok in score["checks"].items() if not ok]
    missed = lambda stems: any(k == s or k.startswith(s + "_") for k in failed for s in stems)   # noqa: E731
    return next((label for label, stems in REQUIREMENTS if missed(stems)), OUTCOMES[-1])


def task_kind(stored: dict | None = None) -> dict:
    """Measurement task or decision task, per level, by the rule of this package: a level on which a fixed list of
    scans with a fixed estimator (the ``open_loop`` control) passes within 25 points of the reference is a measurement
    task, otherwise a decision task. Read from a gate result (default: the stored one), never typed."""
    from ... import gate
    return F.kind_by_open_loop(stored or gate.load("Scanner/BandFree-v0"))


def card() -> dict:
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    lo, hi = BANDWIDTH
    kind = task_kind()
    return dict(
        id="Scanner/BandFree-v0", title="Keep the dark bands of two balanced SSFP images off their marked regions", kind="candidate",
        given="A test object with vials of hidden materials in a hidden, inhomogeneous field; the scanner; a scan-time budget for probing",
        deliver="For each of two balanced SSFP images: the receiver bandwidth (the repetition time is the console minimum "
                "for that bandwidth) and the phase cycle; two phase cycles when one acquisition cannot be band-free",
        scored_by="Noise-free scans of the delivered protocols on the hidden object and the echo amplitudes on the marked "
                  "regions over a drift of the centre frequency either way: every marked region of an image must keep "
                  "its signal; the number of acquisitions and the bandwidth are compared with the best band-free "
                  "protocol of the hidden object",
        summary="What a frequency scout is for. The dark bands of balanced SSFP repeat every 1 / TR in frequency; where "
                "the marked regions fall in that pattern depends on the field, on the error of the frequency adjustment, "
                "on the material (oil sits 3.4 ppm below water) and on the repetition time, and how wide a band is "
                "depends on the material and on the excitation. The target is a 2D image: the slice profile of the "
                "selective pulse adds spins at smaller flip angles, whose signal peaks next to the band, so the band is "
                "narrower than the hard-pulse formula says and a band model has to carry the slice profile (a 3D "
                "acquisition with a hard pulse costs a multiple of the scan time and shows another band). A "
                "low bandwidth gives the least noise and the longest repetition time. The answer, for each of the two "
                "images, is the lowest bandwidth at which one phase cycle keeps every region clear of the bands with a "
                "margin for frequency drift, or the recognition that no single acquisition can, and then the pair of "
                "phase cycles that does. " + kind["sentence"],
        task_kind=kind["kind"], task_kind_why=kind["evidence"], task_kind_levels=kind["levels"], competencies=list(COMPETENCIES),
        world=dict(phantom="QA phantom (a disc of gel with vials; at level 3 one vial of oil), 3 T, field spread drawn per instance",
                   sequences_useful=["bssfp", "spgr"],
                   agent_side_formulas="phase per TR = 360 x off-resonance x TR - phase cycle; steady-state signal of bssfp as "
                                       "a function of that phase, integrated over the slice profile of the documented pulse "
                                       "(datasheet); off-resonance = phase slope of a multi-echo spgr"),
        hidden=["the off-resonance field and its spread", "the error of the frequency adjustment (up to 150 Hz)",
                "T1 and T2 of every material (they set the width of its band)", "which marked region holds oil, and that one does (level 3)",
                "transmit gain and B1 pattern", "whether one acquisition can be band-free", "the 128-bit world seed"],
        told=["the target images and the allowed bandwidths", "the regions of each image", "the three pass rules with every number",
              "the instrument datasheet (it gives the selective pulse)", "realised protocol, timing and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        limits=dict(max_scanner_requests=60, max_tool_calls=64),
        deliverable={"tool": "submit", "format": {name: {"bandwidth_hz_per_px": f"<{lo:g} to {hi:g}; level 3: from {OIL_BANDWIDTH_MIN:g}>",
                                                         "phase_cycle_deg": "[<one or two angles>]"}
                                                  for name in IMAGES}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan",
                           "the console accepts the target scan of each image at the delivered bandwidth"],
                    error="1 - the smallest kept fraction over images, regions and drifts (signal / signal at the region's best phase cycle)",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing=f"all gates and, for each image: every region keeps >= {100 * KEEP_MIN:g} % at every drift up to -/+ {DRIFT_HZ:g} Hz; two "
                            f"acquisitions only if no single acquisition up to {SINGLE_UP_TO:g} Hz per pixel is band-free; bandwidth <= "
                            f"{BANDWIDTH_SLACK:g} x the lowest band-free bandwidth (comparison protocols: band-free up to {REFERENCE_DRIFT_HZ:g} Hz of drift)"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier's scans are noise-free",
    )
