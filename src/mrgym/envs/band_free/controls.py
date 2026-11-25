"""Control policies of ``Scanner/BandFree``. Every policy plays through the agent's interface only.

Must pass
  reference          a three-echo field map, read pixel by pixel, gives the off-resonance over every vial (a quadratic
                     surface per region) and shows which one is oil (the multi-peak spectrum bends its phase and dips
                     its second echo); four balanced SSFP scans of the target geometry at the bandwidth that map
                     suggests, at phase cycles that put the flanks of the predicted bands on the regions, are read
                     pixel by pixel as well: the field changes across a vial, so every scan shows the band of its
                     material at many phases per TR at once, and a fit of the band of a 2D slice (the steady state
                     integrated over the slice profile of the documented pulse) gives the width of the band (one
                     number per material) and corrects the map; then, for each image, the lowest bandwidth at which
                     one phase cycle keeps every region band-free with a drift margin, or, when there is none, the
                     lowest bandwidth at which two phase cycles do
Must fail
  prior              no scan; a middle bandwidth and the default phase cycle for both images
  wrong_units        one token scan, then the tuned constant protocols with the bandwidths in kHz per pixel and the
                     phase cycles in radians
  fixed_recipe       one token scan, then the best answer that ignores the object, tuned on other worlds by the gate:
                     per image a constant protocol, or a bandwidth that follows the distance between its regions
  constant           the same, tuned once on 300 other worlds per level (``python -m mrgym.envs.band_free.controls``)
  brute_force        the target image at one bandwidth (the best constant, tuned on the same 300 worlds) with as many
                     phase cycles as the budget holds; per image the phase cycle with the best worst region, or the
                     best pair of them when none keeps every region; no model
Ablations (the reference with one decision removed; each must lose where the levels say so)
  universal_profile  assumes one band shape for every region: that of T1 / T2 = 40
  water_only         every region is taken for water: the field-map frequency of the oil is not moved to the fat line
  pair_only          always delivers two acquisitions
  default_phase      the reference's bandwidths with the default phase cycle of 180 degrees
Reported
  open_loop          a fixed scan list (field map, four scans at a fixed bandwidth and fixed phase cycles) and the
                     reference's estimator
Further policies (``FURTHER``: playable by name, not run by the gate, which plays every listed policy on every instance)
  hard_pulse_band    the textbook band of the nominal flip angle (no slice profile) in the fit and in the decision
  quarter_turns      the four scout scans at phase cycles 0 / 90 / 180 / 270 degrees, whatever the field map says
  single_only        never delivers two acquisitions

For ``mrgym check <id> --learnability``
  blind_answers      constant protocols for both images on a grid: answers that use the task card only
"""
from __future__ import annotations

import functools
import itertools
import math

import numpy as np

from . import task

REFERENCE = "reference"
MUST_FAIL = ("prior", "wrong_units", "fixed_recipe", "constant", "brute_force")
ABLATIONS = {"universal_profile": (2, 3), "water_only": (3,), "pair_only": (1, 2, 3), "default_phase": (1, 2, 3)}
REPORTED = ("open_loop",)
FURTHER = ("hard_pulse_band", "quarter_turns", "single_only")
LADDER = {1: "universal_profile", 2: "water_only"}       # the policy that suffices at a level and must not at the next
DESCRIPTIONS = dict(
    reference="field map read pixel by pixel (off-resonance over every vial, which region is oil), four scans at the suggested "
              "bandwidth with the flanks of the predicted bands on the regions, read pixel by pixel: band width per material "
              "with the band of a 2D slice; then per image the lowest bandwidth with a band-free phase cycle or, if there is "
              "none, a band-free pair",
    prior="no scan; a middle bandwidth and the default phase cycle for both images",
    wrong_units="one token scan, then the tuned constant protocols with the bandwidths in kHz per pixel and the phase cycles in radians",
    fixed_recipe="one token scan, then the best protocols that ignore the object (constant, or bandwidth from the distance "
                 "between the regions of the image), tuned on other worlds by the gate",
    constant="one token scan, then the best protocols that ignore the object, tuned once on 300 other worlds",
    brute_force="the target image at the best constant bandwidth with as many phase cycles as the budget holds; per image "
                "the best of them, or the best pair, no model",
    universal_profile="the reference with one band shape (that of T1 / T2 = 40) assumed for every region",
    hard_pulse_band="the reference with the textbook band of the nominal flip angle instead of the band of a 2D slice",
    water_only="the reference without the recognition of oil: every region's field-map frequency is taken as it is",
    quarter_turns="the reference with its scout scans at 0 / 90 / 180 / 270 degrees instead of the phase cycles chosen from the field map",
    single_only="the reference, but never two acquisitions",
    pair_only="the reference, but always two acquisitions",
    default_phase="the reference's bandwidths with the default phase cycle of 180 degrees",
    open_loop="a fixed scan list (field map, four scans at 300 Hz per pixel, phase cycles 0 / 90 / 180 / 270 degrees) and the "
              "reference's estimator",
)
FIELD_MAP = dict(sequence="spgr", flip_deg=10.0, tr_ms=12.0, matrix=64, bandwidth_hz_per_px=3000.0, n_echoes=3, averages=3)
SCOUT_PHASES = (0.0, 90.0, 180.0, 270.0)             # the fixed phase cycles of ``quarter_turns`` and ``open_loop``
FLANK_DEG, FLANK_WIDTH_DEG = 45.0, 15.0              # where a scout scan says most about the width of a band: this far from its middle
MAP_ERROR_HZ, OIL_MAP_ERROR_HZ = 4.0, 12.0           # the band lies within this of where the field map puts it (the lines of oil bias the map)
OIL_RATIOS = np.exp(np.linspace(math.log(4.0), math.log(8.0), 8))          # T1 / T2 of the oil of the test object (tissue table: 382 / 68 ms at 3 T, T1 +- 20 %)
VIAL_MM = 11.0                                       # pixels within this of a region's centre lie inside its vial (the smallest vial: 17 mm)
UNIVERSAL_RATIO, UNIVERSAL_T2_MS = 40.0, 70.0        # the band of T1 / T2 = 40 (``universal_profile``, and before the scouts are read)
RATIOS = np.exp(np.linspace(math.log(1.0), math.log(120.0), 30))           # the grid of the fit: T1 / T2 ...
T2S = (35.0, 45.0, 57.0, 72.0, 92.0, 120.0, 160.0, 220.0)                  # ... and T2 in ms
REFINE = np.array([0.85, 0.9, 0.95, 1.0, 1.05, 1.11, 1.17])                # ... then both again on this grid around the best pair
OIL_T2_MS = 68.0                                     # T2 of the oil of the test object (tissue table, 3 T)
SCOUT_BANDWIDTH = (130.0, 450.0)                     # the four scans stay affordable, and near the bandwidths that are asked for
ABOVE_LOWEST, ABOVE_LOWEST_PAIR = 1.03, 1.03         # the answer may sit this far above the lowest band-free bandwidth
TOKEN = dict(sequence="bssfp", flip_deg=60.0, matrix=32, tr_ms=5.0, bandwidth_hz_per_px=1000.0)
# oil in a three-echo map: its spectrum has several lines, so the phase is not linear in the echo time (water: a
# straight line) and the second echo is weaker than T2* decay makes it
OIL_CURVATURE_DEG, OIL_SECOND_ECHO = 6.0, 0.86
OIL_MAP_BIAS_HZ = -4.7                               # the three-echo map reads oil this far from its main line (measured on training worlds)
FAT_PPM, GAMMA_HZ_PER_T = -3.4, 42.577478518e6       # datasheet: fat resonates 3.4 ppm below water


# ------------------------------------------------------------------------------------------------ the band of a 2D slice
PULSE_SAMPLES, PULSE_TBW = 128, 4.0                  # datasheet, "RF, selective": the console's default pulse (normal)
HEIGHTS = np.arange(0, 85) / 24.0                    # slice thicknesses from the centre, as far as the pulse reaches (0.5 + 12 / time-bandwidth)


@functools.lru_cache(maxsize=8)
def slice_columns(flip_deg: float) -> tuple:
    """What the documented pulse does through the slice, for a flip angle ``flip_deg`` at its centre: per height
    (cos of the flip angle there, transverse magnetisation along the direction of the slice centre, weight of the
    height in the integral). The datasheet's recipe: 128 small rotations about the pulse axis with the amplitudes of a
    Hann-windowed sinc, each between two half-steps of precession proportional to the height, then the rephasing lobe."""
    t = (np.arange(PULSE_SAMPLES) + 0.5) / PULSE_SAMPLES - 0.5
    amp = np.sinc(PULSE_TBW * t) * 0.5 * (1.0 + np.cos(2 * np.pi * t))
    amp = amp / amp.sum() * math.radians(flip_deg)
    step = 2 * np.pi * PULSE_TBW * HEIGHTS / PULSE_SAMPLES
    mx, my, mz = np.zeros_like(HEIGHTS), np.zeros_like(HEIGHTS), np.ones_like(HEIGHTS)

    def about_z(mx, my, phi):
        return mx * np.cos(phi) + my * np.sin(phi), my * np.cos(phi) - mx * np.sin(phi)
    for a in amp:
        mx, my = about_z(mx, my, step / 2)
        my, mz = my * math.cos(a) + mz * math.sin(a), mz * math.cos(a) - my * math.sin(a)
        mx, my = about_z(mx, my, step / 2)
    mx, my = about_z(mx, my, -0.5 * step * PULSE_SAMPLES)
    along = (mx * mx[0] + my * my[0]) / math.hypot(mx[0], my[0])
    weight = np.full(len(HEIGHTS), 2.0 * (HEIGHTS[1] - HEIGHTS[0]))      # both sides of the slice
    weight[0] *= 0.5
    weight[-1] *= 0.5
    return mz, along, weight


@functools.lru_cache(maxsize=8192)
def band_table(ratio: float, tr_ms: float, t2_ms: float = 70.0, hard: bool = False, flip_deg: float = 60.0) -> np.ndarray:
    """Complex echo of a material with T1 / T2 = ``ratio`` and T2 = ``t2_ms`` at every whole degree of its phase per TR
    over two turns (0 .. 719; the echo at TR / 2 changes sign from one pass band to the next), in arbitrary units.

    The datasheet's steady state S = (1 - E1) sin a (1 - E2 exp(-i theta)) / (C cos theta + D) at the flip angle of
    every height of the slice, summed over the slice (``hard``: at the nominal angle only, the band of a hard pulse)."""
    cos_a, sin_a, weight = (np.array([math.cos(math.radians(flip_deg))]), np.array([math.sin(math.radians(flip_deg))]), np.ones(1)) if hard \
        else slice_columns(flip_deg)
    e1, e2 = math.exp(-tr_ms / (ratio * t2_ms)), math.exp(-tr_ms / t2_ms)
    th = np.deg2rad(np.arange(720.0))
    c = e2 * (e1 - 1.0) * np.cos(th)
    den = (c + 1.0 - e1 * e2 * e2)[:, None] + (c - e1 + e2 * e2)[:, None] * cos_a[None, :]      # C cos theta + D, per height
    return ((weight * sin_a)[None, :] / den).sum(1) * (1.0 - e1) * (1.0 - e2 * np.exp(-1j * th)) * np.exp(0.5j * th)


def region_keep(ratio: float, tr_ms: float, theta_deg, hard: bool = False, t2_ms: float = 70.0) -> np.ndarray:
    """Signal of a region at every whole degree of the phase cycle (0 .. 359), relative to its best. ``theta_deg``:
    the phase per TR of each of its pixels at phase cycle 0 (the console reads a region as the complex mean over the
    pixels whose centre lies in its circle; the field, and with it the phase per TR, changes across them)."""
    table = band_table(round(float(ratio), 4), round(float(tr_ms), 4), round(float(t2_ms), 3), hard)
    pos = (np.arange(360.0)[:, None] + np.asarray(theta_deg, float)[None, :]) % 720.0
    lo = np.floor(pos).astype(int)
    a = np.abs((table[lo] * (1 - (pos - lo)) + table[(lo + 1) % 720] * (pos - lo)).mean(1))
    return a / a.max()


def fit_band(theta_deg, phases_deg, readouts, tr_ms: float, shifts_deg=(0.0,), ratios=None, t2s=None, hard: bool = False):
    """Width and position of the band of one material from the complex values of its pixels in scans at several phase
    cycles of one repetition time. ``theta_deg`` [pixel]: the phase per TR of every pixel as the field map has it;
    ``readouts`` [pixel, scan]. Least squares of

        read-out x exp(i phase cycle / 2) = c x band_table(ratio, T2)[theta + shift + phase cycle]

    over a grid of ratios T1 / T2 and of T2, the candidate ``shifts_deg`` (the field map is off by a little
    everywhere in a vial) and one complex factor c (receive gain and phase). The field changes across a vial, so the
    pixels of one scan sample the band of the material at many phases per TR at once: its flanks are in the data
    whenever a band crosses the vial, and the phase of the echo, which turns by half a turn across a band, fixes its
    position. Two numbers per material, not one: at these repetition times T2 is not long against TR, and a model
    with the wrong T2 fits a wrong ratio (the data then fix T1, and the band comes out too narrow or too wide).
    Returns (ratio, T2 in ms, shift in degrees, relative misfit)."""
    theta, ph = np.asarray(theta_deg, float), np.asarray(phases_deg, float)
    d = np.asarray(readouts, complex) * np.exp(0.5j * np.deg2rad(ph))[None, :]
    shifts = np.asarray(shifts_deg, float)
    pos = (theta[None, :, None] + ph[None, None, :] + shifts[:, None, None]) % 720.0     # [shift, pixel, scan]
    lo = np.floor(pos).astype(int)
    frac = pos - lo
    energy = max(float((np.abs(d) ** 2).sum()), 1e-30)
    best = None

    def search(ratio_grid, t2_grid):
        nonlocal best
        for t2 in t2_grid:
            for ratio in ratio_grid:
                table = band_table(round(float(ratio), 4), round(float(tr_ms), 4), round(float(t2), 3), hard)
                m = table[lo] * (1 - frac) + table[(lo + 1) % 720] * frac
                c = (np.conj(m) * d[None]).sum((1, 2)) / np.maximum((np.abs(m) ** 2).sum((1, 2)), 1e-30)
                err = (np.abs(d[None] - c[:, None, None] * m) ** 2).sum((1, 2))
                k = int(np.argmin(err))
                if best is None or err[k] < best[0]:
                    best = (float(err[k]), float(ratio), float(t2), float(shifts[k]))

    search(RATIOS if ratios is None else ratios, T2S if t2s is None else t2s)
    if ratios is None and t2s is None:                                   # a finer grid around the best point of the coarse one
        search(best[1] * REFINE, best[2] * REFINE)
    return best[1], best[2], best[3], math.sqrt(best[0] / energy)


def _tolerance(ok: np.ndarray) -> np.ndarray:
    """Per degree: by how many degrees the phase may move either way and stay where ``ok`` holds (-1 where it does not)."""
    n = len(ok)
    if ok.all():
        return np.full(n, n // 2)
    bad, i = np.flatnonzero(~np.tile(ok, 3)), np.arange(n, 2 * n)
    k = np.searchsorted(bad, i)
    return np.minimum(i - bad[np.maximum(k - 1, 0)], np.abs(bad[np.minimum(k, len(bad) - 1)] - i)) - 1


def _centre(mask):
    """Index at the centre of the longest run of True on a circular array."""
    n = len(mask)
    if mask.all():
        return 0
    best, start = (0, 0), None
    for i in range(2 * n):
        if mask[i % n] and start is None:
            start = i
        if not mask[i % n] and start is not None:
            if i - start > best[0]:
                best = (i - start, start)
            start = None
    return (best[1] + best[0] // 2) % n


def working_drift_hz(t) -> float:
    """The drift a protocol of the reference has to survive in its own model: halfway between the drift the pass rule
    asks for and the drift of the scanner's comparison protocols. Whatever the scanner counts as band-free is then
    band-free in the model with room to spare, and whatever the model calls band-free passes rule 1 with room to spare."""
    return 0.5 * (t["drift_hz"] + t["reference_drift_hz"])


def margins(vials: list, tr_of, bandwidths, t, pixel_mm=2.0, hard=False) -> list:
    """Per bandwidth: (bandwidth, [region, 360]) the drift margin of every region at every whole degree of the phase
    cycle: by how many degrees its phase per TR may move either way without the region losing more than allowed, minus
    the working drift (``working_drift_hz``). ``vials``: per region dict(coef: its field, ``field_at``; ratio, t2:
    T1 / T2 and T2 of its material, which set the width of its band; roi: the offsets of the pixels the console reads
    the region from)."""
    rows = []
    for bw in bandwidths:
        tr = tr_of(bw)
        m = []
        for v in vials:
            theta = 360.0 * tr * 1e-3 * seen_field(v["coef"], v["roi"][0], v["roi"][1], bw, pixel_mm)
            tol = _tolerance(region_keep(v["ratio"], tr, theta, hard, v["t2"]) >= t["keep_min"])
            m.append(tol - task.drift_deg(working_drift_hz(t), tr))
        rows.append((float(bw), np.array(m)))
    return rows


def decide(rows: list, members: list, t: dict, mode: str = "auto") -> dict:
    """The decision for one image from the margins of its regions (``members``: their row indices).

    One acquisition: the phase cycle with the largest smallest margin. Two: the pair with the largest smallest margin
    when each region takes the better of the two. The answer is at the lowest bandwidth with a margin >= 0 or a little
    above it (``ABOVE_LOWEST``), where the margin is largest. ``mode``: 'auto' (one acquisition if a bandwidth below
    the top of the range allows it, else two), 'single', 'pair'. Returns dict(bandwidth, phases)."""
    phi = np.arange(360)
    rows = [(bw, m[members]) for bw, m in rows]

    def pick(score, above):
        """(bandwidth, margins, lowest bandwidth with a margin >= 0): the bandwidth where the margin is largest, from
        the lowest one with a margin >= 0 to ``above`` times that."""
        best = [(bw, score(m)) for bw, m in rows]
        ok = [i for i, (_, sc_) in enumerate(best) if sc_.max() >= 0.0]
        if not ok:
            return None
        i = max((i for i in ok if best[i][0] <= above * best[ok[0]][0]), key=lambda i: best[i][1].max())
        return best[i][0], best[i][1], best[ok[0]][0]

    one = None if mode == "pair" else pick(lambda m: m.min(0), ABOVE_LOWEST)
    top = math.sqrt(t["single_up_to"] * t["bandwidth_hz_per_px"][1])       # above it, rule 2 lets two acquisitions pass
    if one is not None and (mode == "single" or one[2] <= top):
        return dict(bandwidth=one[0], phases=[float(phi[_centre(one[1] >= one[1].max() - 1)])])
    two = None if mode == "single" else pick(lambda m: np.maximum(m[:, :, None], m[:, None, :]).min(0), ABOVE_LOWEST_PAIR)
    if two is not None:
        a, b = np.unravel_index(int(np.argmax(two[1])), two[1].shape)
        return dict(bandwidth=two[0], phases=[float(a), float(b)])
    if one is not None:
        return dict(bandwidth=one[0], phases=[float(phi[_centre(one[1] >= one[1].max() - 1)])])
    bw, m = max(rows, key=lambda r: r[1].min(0).max())                   # nothing is band-free: the least bad single acquisition
    return dict(bandwidth=bw, phases=[float(phi[int(np.argmax(m.min(0)))])])


# ------------------------------------------------------------------------------------------------ measurements
def _task(sc):
    return sc.info()["task"]


def _ids(t):
    return [r["id"] for r in t["regions"]]


def _members(t):
    ids = _ids(t)
    return {name: [ids.index(i) for i in group] for name, group in t["images"].items()}


def _grid(t):
    lo, hi = t["bandwidth_hz_per_px"]
    return np.exp(np.arange(math.log(lo), math.log(hi) + 1e-9, 0.02))


def _target(t, bandwidth, tr_ms, phase):
    s = t["target_scan"]
    return dict(sequence="bssfp", flip_deg=s["flip_deg"], matrix=s["matrix"], bandwidth_hz_per_px=float(bandwidth), tr_ms=float(tr_ms),
                phase_cycle_deg=float(phase) % 360.0, excitation=s.get("excitation", "selective"))


def _tr_model(sc, t):
    """Console minimum of the repetition time of the target image as a function of the bandwidth, from three plans:
    TR = a + b x bandwidth + 1000 / bandwidth (pulse and prephasers, readout ramps, readout)."""
    lo, hi = t["bandwidth_hz_per_px"]
    bws = np.array([lo, math.sqrt(lo * hi), hi])
    trs = np.array([sc.plan(_target(t, bw, 50.0, 180.0))["timing"]["tr_min_ms"] for bw in bws])
    a, b = np.linalg.lstsq(np.stack([np.ones(3), bws], 1), trs - 1000.0 / bws, rcond=None)[0]
    return lambda bw: math.ceil((a + b * bw + 1000.0 / bw) * 100.0 - 1e-6) / 100.0


def pixel_positions(fov_mm: float, matrix: int) -> np.ndarray:
    """Centres of the pixels of an image along one axis, mm (scanner_info: pixel i of an n-point axis is centred at
    (i - n // 2) x field of view / n)."""
    return (np.arange(matrix) - matrix // 2) * fov_mm / matrix


def pixels_near(region: dict, fov_mm: float, matrix: int, radius_mm: float, moved_mm: float = 0.0) -> tuple:
    """(row indices, column indices, dy, dx) of the pixels within ``radius_mm`` of the centre of ``region`` moved by
    ``moved_mm`` along x; dy, dx in mm from the (unmoved) centre."""
    pos = pixel_positions(fov_mm, matrix)
    dy, dx = np.meshgrid(pos - region["y_mm"], pos - region["x_mm"], indexing="ij")
    iy, ix = np.nonzero(dy * dy + (dx - moved_mm) ** 2 <= radius_mm ** 2)
    return iy, ix, dy[iy, ix], dx[iy, ix]


def field_at(coef, dy, dx):
    """Off-resonance in Hz at offsets (mm) from the centre of a region: a quadratic surface c0 + c1 dy + c2 dx +
    c3 dy^2 + c4 dy dx + c5 dx^2."""
    return coef[0] + coef[1] * dy + coef[2] * dx + coef[3] * dy * dy + coef[4] * dy * dx + coef[5] * dx * dx


def seen_field(coef, dy, dx, bandwidth: float, pixel_mm: float):
    """Off-resonance of the spins that an image at ``bandwidth`` shows at offsets (dy, dx) from the centre of a region:
    off-resonant spins appear displaced along the readout (x) by off-resonance / bandwidth pixels towards -x, so the
    pixel shows the spins that far towards +x."""
    f = field_at(coef, dy, dx)
    for _ in range(3):
        f = field_at(coef, dy, dx + f / bandwidth * pixel_mm)
    return f


def complex_image(sc, scan_index: int, image: int = 0) -> np.ndarray:
    return sc.view(scan_index, image, "magnitude")["array"] * np.exp(1j * sc.view(scan_index, image, "phase")["array"])


def _three_echoes(z, te):
    """Off-resonance in Hz from the phase of three gradient echoes (the first pair resolves the wrap); ``z`` [echo, ...]."""
    u = z / np.maximum(np.abs(z), 1e-30)
    f12 = np.angle(u[1] * np.conj(u[0])) / (2 * np.pi * (te[1] - te[0]) * 1e-3)
    dt = (te[2] - te[0]) * 1e-3
    return f12 + np.angle(u[2] * np.conj(u[0]) * np.exp(-2j * np.pi * f12 * dt)) / (2 * np.pi * dt)


def _is_oil(z) -> bool:
    """A vial of oil in the three-echo map (``z`` [echo, pixel]): the second echo dips, or the phase bends between
    the echoes. Both are taken pixel by pixel and then summed over the vial, so the field across the vial and the
    noise of single pixels do not count."""
    bend = abs(math.degrees(float(np.angle((z[2] * z[0] * np.conj(z[1] * z[1])).sum()))))
    return bend > OIL_CURVATURE_DEG or float(np.abs(z[1]).sum()) < OIL_SECOND_ECHO * float(np.abs(z[0]).sum())


def _field_map(sc, t, find_oil=True) -> list:
    """One three-echo gradient-echo scan, read pixel by pixel: per region its off-resonance as a quadratic surface
    over its vial (``coef``, Hz and mm from the region's centre; least squares over the map pixels within
    ``VIAL_MM`` of the centre, weighted with their signal), whether it holds oil, and the pixels the console reads
    the region from in the target image (``roi``). The map's frequency is known modulo 1 / echo spacing; water lies
    within half of that around 0, oil within half of that around the fat line (3.4 ppm below water), where it is
    moved."""
    info = sc.info()
    r = sc.scan(dict(FIELD_MAP), rois=t["regions"])
    te, fov = r["realised"]["echo_times_ms"], info["default_fov_mm"]
    period = 1e3 / (te[1] - te[0])
    fat_hz = FAT_PPM * 1e-6 * GAMMA_HZ_PER_T * info["field_strength_t"]
    z = np.array([complex_image(sc, r["scan_index"], e) for e in range(3)])
    f = _three_echoes(z, te)
    vials = []
    for reg, row in zip(t["regions"], r["roi_readouts"]):
        mean = np.asarray(row["complex_mean_magnitude"]) * np.exp(1j * np.asarray(row["complex_mean_phase_rad"]))
        raw = float(_three_echoes(mean, te))
        iy, ix, dy, dx = pixels_near(reg, fov, z.shape[-1], VIAL_MM)
        oil = bool(find_oil and _is_oil(z[:, iy, ix]))
        centre = raw + period * round((fat_hz - raw) / period) - OIL_MAP_BIAS_HZ if oil else raw
        value = centre + (f[iy, ix] - raw + period / 2) % period - period / 2            # every pixel on the branch of the centre
        basis = np.stack([np.ones_like(dy), dy, dx, dy * dy, dy * dx, dx * dx], 1)
        w = np.abs(z[0][iy, ix])
        coef = np.linalg.lstsq(basis * w[:, None], value * w, rcond=None)[0]
        _, _, ry, rx = pixels_near(reg, fov, t["target_scan"]["matrix"], reg["radius_mm"])
        vials.append(dict(id=reg["id"], region=reg, oil=oil, coef=coef, roi=(ry, rx), ratio=UNIVERSAL_RATIO, t2=UNIVERSAL_T2_MS))
    return vials


def scout_phases(vials: list, tr_ms: float, n=4) -> list:
    """The phase cycles of the scout scans, chosen from the field map: on the flanks of the bands.

    The field map says where the band of every region lies at this repetition time. A scan in which a region sits on
    the flat top of a pass band says nothing about the width of the band next to it (the band of a 2D slice is
    narrow: four phase cycles a quarter turn apart can all miss it). On the flank, ``FLANK_DEG`` from the middle of
    the band, the signal runs from nearly all (a narrow band) to about half (a wide one). Among all sets of ``n``
    phase cycles on a 10 degree grid, the set is taken in which the worst-served region is served best: full marks
    for a scan with the region's centre on a flank, half as much again for one on the other flank."""
    cand = np.arange(0, 360, 10)
    combos = np.array(list(itertools.combinations(range(len(cand)), n)))
    worst, total = np.full(len(combos), np.inf), np.zeros(len(combos))
    for v in vials:
        delta = (360.0 * v["coef"][0] * tr_ms * 1e-3 + cand + 180.0) % 360.0 - 180.0     # phase per TR from the middle of the band
        up, down = (np.exp(-((delta * sign - FLANK_DEG) / FLANK_WIDTH_DEG) ** 2)[combos].max(1) for sign in (1.0, -1.0))
        served = np.maximum(up, down) + 0.5 * np.minimum(up, down)
        worst, total = np.minimum(worst, served), total + served
    return [float(c) for c in cand[combos[int(np.argmax(worst + 1e-3 * total))]]]


def _bands(sc, t, vials: list, bandwidth: float, tr_ms: float, pixel_mm: float, universal=False, hard=False, phases=None) -> list:
    """Four phase-cycled scans of the target geometry (``phases``: fixed phase cycles; None: chosen from the field
    map by ``scout_phases``), read pixel by pixel over every vial where it appears in the image: per region the
    T1 / T2 and T2 of its material (the width of its band) and a correction of the field map (``fit_band``)."""
    phases = list(phases) if phases is not None else scout_phases(vials, tr_ms)
    fov = sc.info()["default_fov_mm"]
    images = [complex_image(sc, sc.scan(_target(t, bandwidth, tr_ms, phase), rois=t["regions"])["scan_index"]) for phase in phases]
    out = []
    for v in vials:
        moved = v["coef"][0] / bandwidth * pixel_mm                      # the vial appears this far towards -x
        iy, ix, dy, dx = pixels_near(v["region"], fov, images[0].shape[-1], VIAL_MM, -moved)
        theta = 360.0 * tr_ms * 1e-3 * seen_field(v["coef"], dy, dx, bandwidth, pixel_mm)
        half = 360.0 * (OIL_MAP_ERROR_HZ if v["oil"] else MAP_ERROR_HZ) * tr_ms * 1e-3
        ratios, t2s = ([UNIVERSAL_RATIO], [UNIVERSAL_T2_MS]) if universal else (OIL_RATIOS, [OIL_T2_MS]) if v["oil"] else (None, None)
        ratio, t2, shift, _ = fit_band(theta, phases, np.stack([img[iy, ix] for img in images], 1), tr_ms,
                                       np.arange(-half, half + 1e-9, 1.5), ratios, t2s, hard)
        coef = np.array(v["coef"], float)
        coef[0] += shift / (360.0 * tr_ms * 1e-3)
        out.append(dict(v, coef=coef, ratio=ratio, t2=t2))
    return out


def _answer(sols: dict) -> dict:
    return {name: dict(bandwidth_hz_per_px=round(s["bandwidth"], 1), phase_cycle_deg=[round(p % 360.0, 1) for p in s["phases"]])
            for name, s in sols.items()}


def _solve(t, vials, pixel_mm, tr_of, mode="auto", hard=False):
    rows = margins(vials, tr_of, _grid(t), t, pixel_mm, hard)
    return {name: decide(rows, members, t, mode) for name, members in _members(t).items()}


def reference(sc, mode="auto", scout_bandwidth=None, universal=False, hard=False, find_oil=True, phases=None):
    t = _task(sc)
    tr_of = _tr_model(sc, t)
    pixel_mm = sc.info()["default_fov_mm"] / t["target_scan"]["matrix"]
    vials = _field_map(sc, t, find_oil)
    if scout_bandwidth is None:                                          # where the field map alone would put the two protocols
        first = _solve(t, vials, pixel_mm, tr_of, mode="single", hard=hard)
        lo, hi = SCOUT_BANDWIDTH
        scout_bandwidth = float(np.clip(math.sqrt(math.prod(s["bandwidth"] for s in first.values())), lo, hi))
    vials = _bands(sc, t, vials, scout_bandwidth, tr_of(scout_bandwidth), pixel_mm, universal, hard, phases)
    return _answer(_solve(t, vials, pixel_mm, tr_of, mode, hard))


def universal_profile(sc):
    return reference(sc, universal=True)


def hard_pulse_band(sc):
    return reference(sc, hard=True)


def water_only(sc):
    return reference(sc, find_oil=False)


def single_only(sc):
    return reference(sc, mode="single")


def pair_only(sc):
    return reference(sc, mode="pair")


def default_phase(sc):
    return {name: dict(a, phase_cycle_deg=[180.0]) for name, a in reference(sc, mode="single").items()}


def quarter_turns(sc):
    return reference(sc, phases=SCOUT_PHASES)


def open_loop(sc):
    return reference(sc, scout_bandwidth=300.0, phases=SCOUT_PHASES)


def prior(sc):
    t = _task(sc)
    lo, hi = t["bandwidth_hz_per_px"]
    return {name: dict(bandwidth_hz_per_px=round(math.sqrt(lo * hi), 1), phase_cycle_deg=[180.0]) for name in t["images"]}


# ------------------------------------------------------------------------------------------------ answers that ignore the object
def spread_mm(t, image) -> float:
    """What the task card says about the geometry of an image: the largest distance between the centres of two of
    its marked regions."""
    p = np.array([[r["y_mm"], r["x_mm"]] for r in task.image_regions(t, image)])
    return float(np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1)).max())


def recipe_answer(recipe, t):
    lo, hi = t["bandwidth_hz_per_px"]
    out = {}
    for name in t["images"]:
        r = recipe[name]
        bw = r["bandwidth_hz_per_px"] * (spread_mm(t, name) / 100.0) ** r.get("spread_power", 0.0)
        out[name] = dict(bandwidth_hz_per_px=round(float(np.clip(bw, lo, hi)), 1), phase_cycle_deg=list(r["phase_cycle_deg"]))
    return out


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        sc.scan(dict(TOKEN), rois=t["regions"])
        return recipe_answer(recipe, t)
    return policy


def sweep_answer(rel: np.ndarray, phases, keep_min: float) -> list:
    """The phase cycle(s) a sweep without a model delivers for one image. ``rel`` [phase, region]: the signal of each
    region at each sampled phase cycle relative to its largest over the samples. The sample with the best worst
    region; when even that one loses a region, the pair of samples with the best worst region, each region taking
    the better of the two."""
    worst = rel.min(1)
    if worst.max() >= keep_min or len(phases) < 2:
        return [float(phases[int(np.argmax(worst))])]
    both = np.maximum(rel[:, None, :], rel[None, :, :]).min(2)
    i, j = np.unravel_index(int(np.argmax(both)), both.shape)
    return [float(phases[i]), float(phases[j])]


def brute_force(sc):
    t = _task(sc)
    bw = SWEEP_BANDWIDTH[t["level"]]
    tr = sc.plan(_target(t, bw, 50.0, 180.0))["timing"]["tr_min_ms"]
    cost = sc.plan(_target(t, bw, tr, 0.0))["scan_s"]
    n = max(1, int(sc.budget()["remaining_s"] // cost))
    phases = [360.0 * k / n for k in range(n)]
    sig = np.array([[row["complex_mean_magnitude"][0] for row in sc.scan(_target(t, bw, tr, p), rois=t["regions"])["roi_readouts"]]
                    for p in phases])
    rel = sig / sig.max(0)
    return {name: dict(bandwidth_hz_per_px=bw, phase_cycle_deg=sweep_answer(rel[:, members], phases, t["keep_min"]))
            for name, members in _members(t).items()}


# tuned by ``python -m mrgym.envs.band_free.controls --n 150`` on 150 worlds per level of the tune split (seeds constant:<i>),
# which the gate never plays; engine 0.2.0. ``tuned_pass_rate``: of the recipe on those worlds, under the verifier's rules
CONSTANTS = {1: {"A": {"bandwidth_hz_per_px": 144.7, "spread_power": 0.5, "phase_cycle_deg": [275.0]},
                 "B": {"bandwidth_hz_per_px": 139.0, "spread_power": 0.0, "phase_cycle_deg": [265.0]}, "tuned_pass_rate": 0.027},
             2: {"A": {"bandwidth_hz_per_px": 335.1, "spread_power": 0.0, "phase_cycle_deg": [300.0]},
                 "B": {"bandwidth_hz_per_px": 150.6, "spread_power": 0.0, "phase_cycle_deg": [130.0]}, "tuned_pass_rate": 0.027},
             3: {"A": {"bandwidth_hz_per_px": 119.1, "spread_power": 1.0, "phase_cycle_deg": [0.0, 75.0]},
                 "B": {"bandwidth_hz_per_px": 119.1, "spread_power": 0.0, "phase_cycle_deg": [215.0]}, "tuned_pass_rate": 0.027}}
# the bandwidth of ``brute_force`` (its pass rate on the same worlds, the sweep played on the echo amplitudes: level 1 3.3 %, level 2 3.3 %, level 3 2.7 %)
SWEEP_BANDWIDTH = {1: 128.3, 2: 191.4, 3: 134.3}
SWEEP_SCANS = 5                                      # what the budget holds at those bandwidths (the tuner's assumption)


def constant(sc):
    t = _task(sc)
    sc.scan(dict(TOKEN), rois=t["regions"])
    return recipe_answer(CONSTANTS[t["level"]], t)


def wrong_units(sc):
    return {name: dict(bandwidth_hz_per_px=a["bandwidth_hz_per_px"] / 1000.0, phase_cycle_deg=[round(math.radians(p), 4) for p in a["phase_cycle_deg"]])
            for name, a in constant(sc).items()}


def blind_answers(t: dict) -> dict:
    """Answers that use the task card only: the same constant protocol for both images, on a grid. {label: answer};
    the same labels on every instance."""
    return {f"bandwidth={bw:g},phase={ph:g}": {name: dict(bandwidth_hz_per_px=float(bw), phase_cycle_deg=[float(ph)]) for name in t["images"]}
            for bw in (80.0, 110.0, 150.0, 200.0, 280.0, 400.0) for ph in (0.0, 90.0, 180.0, 270.0)}


def policies(recipe=None):
    return dict(reference=reference, prior=prior, wrong_units=wrong_units, fixed_recipe=fixed_recipe(recipe or CONSTANTS[2]),
                constant=constant, brute_force=brute_force, universal_profile=universal_profile, hard_pulse_band=hard_pulse_band,
                water_only=water_only, quarter_turns=quarter_turns, single_only=single_only, pair_only=pair_only,
                default_phase=default_phase, open_loop=open_loop)


# ------------------------------------------------------------------------------------------------ tuning on other worlds
def _tables(cfgs, every=1):
    """Per world, under the verifier's own rules (rule 1 from the echo amplitudes): for each image the lowest
    band-free bandwidths, and per bandwidth of the grid the kept fractions [region, 360] and the band-free mask at the
    task's drift."""
    from ...scanner import world as W
    bws = task.bandwidth_grid(cfgs[0]["task"]["bandwidth_hz_per_px"])[::every]
    worlds = []
    for c in cfgs:
        w = W.make_world(c["seed"], c["world_spec"])
        t = c["task"]
        ev = task.Evaluator(w, t["regions"])
        ref = task.reference_protocols(c, w)
        keep, ok = [], []
        for bw in bws:
            k, tr = ev.keep(float(bw))
            keep.append(k.astype(np.float32))
            ok.append(task.robust(k, tr, t["drift_hz"], t["keep_min"]))
        worlds.append(dict(task=t, ref=ref, members=_members(t), keep=np.array(keep), ok=np.array(ok),
                           spread={name: spread_mm(t, name) for name in t["images"]}))
    return bws, worlds


def _passes(w, name, bw, ok_rows, n_acq) -> bool:
    """Rules 2 and 3 for a protocol of image ``name`` at ``bw`` whose rule 1 is ``ok_rows`` (already decided)."""
    ref, t = w["ref"][name], w["task"]
    single = ref["single_bandwidth"]
    if n_acq == 1:
        return bool(ok_rows) and (single is None or bw <= t["bandwidth_slack"] * single)
    return bool(ok_rows) and (single is None or single > t["single_up_to"]) \
        and (ref["pair_bandwidth"] is None or bw <= t["bandwidth_slack"] * ref["pair_bandwidth"])


def tune(cfgs, top=40):
    """The strongest protocols that ignore the object, on other worlds, under the verifier's own rules. Candidates per
    image: one or two constant phase cycles (steps of 5 and 15 degrees) with a bandwidth that is constant or follows
    the distance between the marked regions of the image (power 0, 0.5 or 1). The ``top`` candidates of each image by
    their own pass rate are combined; the pair of candidates that passes both images most often is the recipe."""
    return _recipe(*_tables(cfgs, every=3), top)


def _recipe(bws, worlds, top=40) -> dict:
    lo, hi = worlds[0]["task"]["bandwidth_hz_per_px"]
    names = list(worlds[0]["task"]["images"])
    cands = {name: [] for name in names}
    for name in names:
        for power in (0.0, 0.5, 1.0):
            for base in bws:
                one = np.zeros((len(worlds), 72), bool)
                two = np.zeros((len(worlds), 24, 24), bool)
                for k, w in enumerate(worlds):
                    bw = float(np.clip(base * (w["spread"][name] / 100.0) ** power, lo, hi))
                    ok = w["ok"][int(np.argmin(np.abs(np.log(bws / bw))))][w["members"][name]]
                    if _passes(w, name, bw, True, 1):
                        one[k] = ok.all(0)[::5]
                    if _passes(w, name, bw, True, 2):
                        miss = ((~ok[:, ::15]) * (1 << np.arange(ok.shape[0]))[:, None]).sum(0)
                        two[k] = (miss[:, None] & miss[None, :]) == 0
                for j in np.argsort(-one.sum(0))[:3]:
                    cands[name].append((one[:, j], dict(bandwidth_hz_per_px=round(float(base), 1), spread_power=power, phase_cycle_deg=[float(5 * j)])))
                flat = two.reshape(len(worlds), -1)
                for j in np.argsort(-flat.sum(0))[:2]:
                    a, b = divmod(int(j), 24)
                    if a < b:
                        cands[name].append((flat[:, j], dict(bandwidth_hz_per_px=round(float(base), 1), spread_power=power,
                                                             phase_cycle_deg=[float(15 * a), float(15 * b)])))
        cands[name] = sorted(cands[name], key=lambda c: -int(c[0].sum()))[:top]
    a, b = names
    joint = np.array([[int((ca[0] & cb[0]).sum()) for cb in cands[b]] for ca in cands[a]])
    i, j = np.unravel_index(int(np.argmax(joint)), joint.shape)
    return {a: cands[a][i][1], b: cands[b][j][1], "tuned_pass_rate": round(float(joint[i, j]) / len(worlds), 3),
            "tuned_pass_rate_per_image": [round(float(cands[a][i][0].mean()), 3), round(float(cands[b][j][0].mean()), 3)]}


def tune_sweep(cfgs, n_scans=SWEEP_SCANS) -> dict:
    """The best constant bandwidth of ``brute_force`` on other worlds: the sweep is played on the echo amplitudes
    (noise-free read-outs at ``n_scans`` equally spaced phase cycles, the answer of ``sweep_answer``) and scored under
    the verifier's own rules."""
    return _sweep(*_tables(cfgs, every=2), n_scans)


def _sweep(bws, worlds, n_scans) -> dict:
    phases = [360.0 * k / n_scans for k in range(n_scans)]
    at = [int(round(p)) % 360 for p in phases]
    best = (-1, None)
    for b, bw in enumerate(bws):
        n = 0
        for w in worlds:
            good = True
            for name, members in w["members"].items():
                sig = w["keep"][b][members][:, at].T                      # [phase, region]
                pick = sweep_answer(sig / sig.max(0), phases, w["task"]["keep_min"])
                ok = w["ok"][b][members][:, [int(round(p)) % 360 for p in pick]]
                good &= _passes(w, name, float(bw), ok.any(1).all() if len(pick) == 2 else ok.all(), len(pick))
            n += good
        if n > best[0]:
            best = (n, float(bw))
    return dict(bandwidth_hz_per_px=round(best[1], 1), tuned_pass_rate=round(best[0] / len(worlds), 3))


if __name__ == "__main__":
    import argparse

    import mrgym
    from mrgym import gate
    ap = argparse.ArgumentParser(description="tune the constants of Scanner/BandFree on worlds of the tune split that the gate never plays")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--level", type=int, nargs="*", default=None)
    a = ap.parse_args()
    spec = mrgym.spec("Scanner/BandFree-v0")
    for lv in a.level or spec.levels:
        tables = _tables([gate._instance(spec, lv, f"constant:{i}") for i in range(a.n)], every=2)
        print(f"CONSTANTS[{lv}] = {_recipe(*tables)}", flush=True)
        print(f"SWEEP_BANDWIDTH[{lv}] = {_sweep(*tables, SWEEP_SCANS)}", flush=True)
