"""Phantom QA: the tests of the ACR large-phantom guidance, measured on images of the virtual scanner.

    mrgym qa acr [--field 1.5|3] [--seed N] [--fault NAME] [--json]
    mrgym qa acr --matrix [--write]          every fault alone, against every test

What runs. The ACR series (sagittal localiser, axial T1, axial T2; ``SERIES``) on the ``acr`` world, one section per
scan because the engine images one 2D slice: the localiser, slices 1, 5, 7, 8, 9, 10, 11 of the T1 series and slices
1, 7, 8, 9, 10, 11 of the T2 series (the slices the tests read). Every scan has receiver noise. The scan time printed
is the console's time of one such scan, which is the time of the whole 11-slice series on a scanner (the slices of a
conventional spin echo are interleaved inside one TR): compared with the guidance's time, never charged per slice.

What is measured (``LIMITS`` carries the guidance's pass limits with their section):

    geometric accuracy      length on the localiser, two diameters on slice 1, four on slice 5; edges at half the
                            mean signal of the water (section 1.2)
    high-contrast           smallest hole size of which one row (one column) of four holes is distinguishable, on
    resolution              slice 1 of both series (2.2). The reader is replaced by a stated rule (``RESOLVED_DEPTH``)
    intensity uniformity    PIU = 100 (1 - (high - low) / (high + low)), 1 cm^2 regions inside the 200 cm^2 region of
                            slice 7, both series (5.2, 5.3). Every position of the small region is tried
    percent-signal ghosting |((top + bottom) - (left + right)) / (2 x large region)|, slice 7 of the T1 series (6.3)
    low-contrast            complete spokes summed over slices 8 to 11, both series (7.3). The reader is replaced by a
    detectability           stated rule (``VISIBLE_Z``): a disk counts when its mean stands out from the ring around it
    artifact assessment     the guidance gives no number (section 8); a stated detector of a line of interference in
                            the background, with a declared limit

    slice thickness         NOT SIMULATED: the two signal ramps (section 3) are structure through the slice, which a
    slice position          2D world does not have; the 45 degree wedges (section 4) likewise

Faults (``FAULTS``) are states of the hidden system or of its fields, set through the world spec: the scanner images
them like any other world. None is drawn onto an image.

Declared, not published: the receiver noise of the healthy system (``HEALTHY``; chosen so that it scores above the
guidance's low-contrast limits at both fields), the bandwidth of the series, the two reader rules, the artifact
detector and its limit, and everything tagged [D] in ``mrgym.scanner.anatomy.ACR``.
"""
from __future__ import annotations

import hashlib
import importlib.resources
import json
import os
import time

import numpy as np
import scipy.ndimage
import scipy.signal

from .scanner import anatomy, console, kspace
from .scanner import session as core
from .scanner import world as W

SCHEMA = "mrgym-qa-acr/1"
A = anatomy.ACR

# ------------------------------------------------------------------------------------------------ the guidance
# [G] = "Large and Medium Phantom Test Guidance for the ACR MRI Accreditation Program" (10.19.2022), section given.
# [D] = declared by this package.
SERIES = dict(                                                           # [G] Table 2; the bandwidth is [D]
    localizer=dict(sequence="se", tr_ms=200.0, te_ms=20.0, fov_mm=250.0, matrix=256, slice_mm=10.0, averages=1, bandwidth_hz_per_px=130.0),
    t1=dict(sequence="se", tr_ms=500.0, te_ms=20.0, fov_mm=250.0, matrix=256, slice_mm=5.0, averages=1, bandwidth_hz_per_px=130.0),
    t2=dict(sequence="se", tr_ms=2000.0, te_ms=80.0, fov_mm=250.0, matrix=256, slice_mm=5.0, averages=1, bandwidth_hz_per_px=130.0),
)
GUIDANCE_SCAN_S = dict(localizer=56.0, t1=136.0, t2=536.0)              # [G] Table 2: 0:56, 2:16, 8:56
SLICES = dict(localizer=("sagittal",), t1=(1, 5, 7, 8, 9, 10, 11), t2=(1, 7, 8, 9, 10, 11))
LIMITS = dict(
    geometry_mm=3.0,                         # [G] 1.4: fails if any measurement differs more than +-3.0 mm from its true value
    resolution_mm=1.0,                       # [G] 2.4: 1.0 mm or smaller in both directions, on both series
    piu_percent={1.5: 85.0, 3.0: 80.0},      # [G] 5.4, Table 4: fails below 85 % under 3 T, below 80 % at 3 T
    piu_should_percent={1.5: 87.5, 3.0: 82.0},   # [G] 5.4: "should be greater than or equal to"
    ghosting_ratio=0.030,                    # [G] 6.4: fails if the ratio exceeds 0.030
    lcd_spokes={1.5: dict(t1=30, t2=25), 3.0: dict(t1=37, t2=37)},   # [G] 7.4, Table 5
    slice_thickness_mm=(5.0, 0.7),           # [G] 3.4 (not simulated)
    slice_position_mm=5.0,                   # [G] 4.4: bar length difference (not simulated)
    artifact_z=15.0,                         # [D] the guidance gives no number (section 8)
)
LARGE_ROI_CM2, SMALL_ROI_CM2 = 200.0, 1.0   # [G] Table 4
BACKGROUND_ROI_MM = (16.0, 63.0)            # [G] 6.2: "length-to-width ratio of about 4:1 and a total area of about 10 cm^2"
BACKGROUND_AT_MM = 110.0                    # [G] 6.2: centred between the edge of the phantom (95 mm) and of the field of view (125 mm)
BACKGROUND_KEEP_MM = 3.5                    # [D] the background regions keep their standard places while the phantom is found within this distance
                                            #     of the isocentre (half of the 7 mm between a region and the edge of the phantom); beyond it they
                                            #     go with the phantom (``follow``)
RESOLVED_DEPTH = 0.5                        # [D] a gap between two holes is darker than both by this share of the brightest hole
VISIBLE_Z = 4.0                             # [D] a disk is seen when (disk - ring) / standard error >= this
TESTS = ("geometric_accuracy", "spatial_resolution", "slice_thickness", "slice_position", "intensity_uniformity",
         "ghosting", "low_contrast", "artifacts")
NOT_SIMULATED = dict(
    slice_thickness="the two signal ramps are slots inclined through the slice ([G] 3.2); the world is one 2D section",
    slice_position="the crossed 45 degree wedges are structure through the slice ([G] 4.2); the world is one 2D section")
TITLES = dict(geometric_accuracy="1 geometric accuracy", spatial_resolution="2 high-contrast spatial resolution",
              slice_thickness="3 slice thickness accuracy", slice_position="4 slice position accuracy",
              intensity_uniformity="5 image intensity uniformity", ghosting="6 percent-signal ghosting",
              low_contrast="7 low-contrast object detectability", artifacts="8 artifact assessment")

# The healthy system (declared). ``sigma0`` is the noise density at 3 T; the engine scales it with 3 T / B0.
HEALTHY = dict(tx_gain_dev=(0.0, 0.03), cf_offset_hz=5.0, sigma0=0.0065, sigma0_log_sd=0.05, grad_delay_us=0.4)
HEALTHY_GRADIENT = 0.0015                   # calibration error of each gradient within +- this (0.3 mm on 190 mm)

# name: what is wrong, the section of the guidance that names the cause, and the tests that fail on this engine with
# the fault alone at the size ``system_spec`` gives it (asserted at both fields by the stored fault x test matrix).
# Two rows are findings rather than confirmations. A centre-frequency error alone fails nothing: the image moves by
# offset / bandwidth pixels and every test is relative to the phantom. A rocking phantom blurs the image along the
# motion and puts its ghost on the edges of the phantom, so it fails resolution and low contrast, while the ghosting
# ratio of the guidance (four background regions) stays under its limit; the guidance names motion as a cause of
# ghosting (6.5) and ghosting as a cause of resolution and low-contrast failures (2.5, 7.5).
FAULTS = dict(
    gradient_x=dict(what="X gradient (left-right) 2.5 % too strong", guidance="1.5", fails=("geometric_accuracy",)),
    gradient_y=dict(what="Y gradient (anterior-posterior) 2.5 % too weak", guidance="1.5", fails=("geometric_accuracy",)),
    gradient_z=dict(what="Z gradient (head-foot) 2.5 % too strong", guidance="1.5", fails=("geometric_accuracy",)),
    shim=dict(what="a linear field gradient of 3.2 Hz/mm along X left by the shim", guidance="1.5", fails=("geometric_accuracy",)),
    centre_frequency=dict(what="centre frequency off by up to 2 ppm (drawn)", guidance="-", fails=()),
    transmit_gain=dict(what="transmitter gain 60 % off (sign drawn)", guidance="5.1, 7.5", fails=("intensity_uniformity", "low_contrast")),
    b1_nonuniformity=dict(what="transmit field 36 % too high near the centre", guidance="5.1", fails=("intensity_uniformity", "low_contrast")),
    receive_coil=dict(what="receive sensitivity falls across the phantom (a dead coil element)", guidance="5.5", fails=("intensity_uniformity",)),
    signal_instability=dict(what="signal amplitude varies by +-15 % from pulse to pulse (0.31 Hz)", guidance="6.5", fails=("ghosting",)),
    phantom_motion=dict(what="the phantom rocks by +-1.2 mm along Y (0.31 Hz)", guidance="2.5, 6.5, 7.5", fails=("spatial_resolution", "low_contrast")),
    rf_interference=dict(what="a narrow-band interference line in the receiver band", guidance="8.0", fails=("artifacts",)),
    receiver_noise=dict(what="receiver noise 5 times the healthy level", guidance="7.5", fails=("low_contrast",)),
)
# The seven faults of the fault x test matrix, one per mechanism; the other entries are variants on another axis.
MATRIX_FAULTS = ("signal_instability", "rf_interference", "transmit_gain", "shim", "gradient_x", "receiver_noise", "b1_nonuniformity")


def seed_hex(seed) -> str:
    """The 32 hex digits of a command-line seed."""
    return hashlib.sha256(f"mrgym qa acr|{seed}".encode()).hexdigest()[:32]


def system_spec(field: float, seed, fault: str | None = None) -> tuple:
    """World spec of the system under test (no ``acr`` section yet) and the calibration error of the three gradients,
    {"X", "Y", "Z"}: the healthy system of ``seed``, with one fault if named."""
    if fault is not None and fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; faults: {', '.join(FAULTS)}")
    rng = np.random.default_rng(int(seed_hex(seed), 16) % 2 ** 63)
    grads = dict(zip("XYZ", (float(v) for v in rng.uniform(-1, 1, 3) * HEALTHY_GRADIENT)))
    line_hz = float(rng.uniform(2000.0, 9000.0) * (1 if rng.uniform() < 0.5 else -1))
    sp = dict(HEALTHY, kind="acr", b0_t=float(field))
    if fault == "gradient_x":
        grads["X"] += 0.025
    elif fault == "gradient_y":
        grads["Y"] -= 0.025
    elif fault == "gradient_z":
        grads["Z"] += 0.025
    elif fault == "shim":
        sp["acr_fields"] = dict(shim_hz_per_mm=(0.0, 3.2))
    elif fault == "centre_frequency":
        sp["cf_offset_hz"] = 2e-6 * W.GAMMA_HZ_PER_T * 3.0                 # the engine scales the offset with B0 / 3 T
    elif fault == "transmit_gain":
        sp["tx_gain_dev"] = (0.6, 0.6)
    elif fault == "b1_nonuniformity":
        sp["acr_fields"] = dict(b1_fault=0.36)
    elif fault == "receive_coil":
        sp["acr_fields"] = dict(rx_fault=0.7)
    elif fault == "signal_instability":
        sp["motion"] = [dict(cls="*", kind="pulse", f_hz=0.31, depth=0.15)]
    elif fault == "phantom_motion":
        sp["motion"] = [dict(cls="*", kind="shift", f_hz=0.31, amp_mm=1.2, dir="y")]
    elif fault == "rf_interference":
        sp["rfi"] = dict(freq_hz=line_hz, amp=400.0 * HEALTHY["sigma0"] * 3.0 / field)
    elif fault == "receiver_noise":
        sp["sigma0"] = 5.0 * HEALTHY["sigma0"]
    return sp, grads


def world_spec(sp: dict, grads: dict, section) -> dict:
    """The spec of one section of the phantom on the system ``sp``: axial sections lie in the (Y, X) plane, the
    localiser in the (Z, Y) plane, and each in-plane axis carries the calibration error of its physical gradient."""
    plane = ("Z", "Y") if section == "sagittal" else ("Y", "X")
    return dict(sp, acr=dict(slice=section), grad_scale=dict(y=grads[plane[0]], x=grads[plane[1]]))


# ------------------------------------------------------------------------------------------------ image geometry
def positions(shape, fov_yx):
    """Pixel centres (y, x) in mm: pixel i of an n-point axis at (i - n // 2) * field of view / n."""
    return core.pixel_positions(shape, fov_yx)


def sample(img, fov_yx, y, x, order=1):
    """Image values at positions (mm), interpolated (order 1: bilinear, 3: cubic spline)."""
    ny, nx = img.shape
    return scipy.ndimage.map_coordinates(img, [np.asarray(y, float) / (fov_yx[0] / ny) + ny // 2,
                                               np.asarray(x, float) / (fov_yx[1] / nx) + nx // 2], order=order, mode="nearest")


def water_level(img) -> float:
    """Mean signal of the water-only regions, as the guidance's windowing finds it ([G] 1.2): the median of the
    pixels brighter than half the brightest."""
    hi = float(np.percentile(img, 99.0))
    return float(np.median(img[img > 0.5 * hi]))


def chord(img, fov_yx, centre_yx, angle_deg, level, half_mm=140.0, step_mm=0.05):
    """Length of the object along a line through ``centre_yx`` at ``angle_deg`` (0: left-right, 90: top-bottom), from
    the outermost crossings of ``level``, and the position of its midpoint along the line. (length mm, midpoint mm)."""
    t = np.arange(-half_mm, half_mm + step_mm / 2, step_mm)
    a = np.deg2rad(angle_deg)
    v = sample(img, fov_yx, centre_yx[0] + t * np.sin(a), centre_yx[1] + t * np.cos(a))
    inside = np.flatnonzero((np.abs(centre_yx[0] + t * np.sin(a)) <= fov_yx[0] / 2) & (np.abs(centre_yx[1] + t * np.cos(a)) <= fov_yx[1] / 2))
    t, v = t[inside], v[inside]
    above = np.flatnonzero(v >= level)
    if len(above) < 2:
        return 0.0, 0.0
    i, j = above[0], above[-1]
    lo = t[i] if i == 0 else t[i - 1] + (level - v[i - 1]) / (v[i] - v[i - 1]) * step_mm
    hi = t[j] if j == len(t) - 1 else t[j] + (v[j] - level) / (v[j] - v[j + 1]) * step_mm
    return float(hi - lo), float(0.5 * (hi + lo))


def find_centre(img, fov_yx, level):
    """Centre of the object from the midpoints of its horizontal and vertical chords (two passes)."""
    cy = cx = 0.0
    for _ in range(2):
        cx = cx + chord(img, fov_yx, (cy, cx), 0.0, level)[1]
        cy = cy + chord(img, fov_yx, (cy, cx), 90.0, level)[1]
    return cy, cx


def diameters(img, fov_yx, angles=(90.0, 0.0)) -> dict:
    """Edge-to-edge lengths through the centre of the object at the given angles, edges at half the water signal."""
    level = 0.5 * water_level(img)
    c = find_centre(img, fov_yx, level)
    return dict(lengths_mm=[chord(img, fov_yx, c, a, level)[0] for a in angles], centre_mm=c, level=level)


def register(img, fov_yx) -> dict:
    """Where the phantom is in the image and how long it appears along y and x relative to its true diameter: what a
    reader does without thinking when he looks for a hole or a disk where the phantom shows it. ``place(v, u)`` maps a
    position of the nominal layout (mm from the centre of the phantom) to the image."""
    d = diameters(img, fov_yx, angles=(90.0, 0.0))
    (cy, cx), (sy, sx) = d["centre_mm"], (v / A["diameter_mm"] if v > 0 else 1.0 for v in d["lengths_mm"])
    return dict(centre_mm=(cy, cx), scale=(sy, sx), place=lambda v, u: (cy + np.asarray(v) * sy, cx + np.asarray(u) * sx))


def noise_sd(img, fov_yx) -> float:
    """Noise standard deviation of one component, from the magnitude of the four corners of the image (Rayleigh:
    mean = 1.2533 sd). The median of the four, so that one corner with a ghost or a line does not move it."""
    ny, nx = img.shape
    k = max(4, int(round(0.07 * min(ny, nx))))
    corners = [img[:k, :k], img[:k, -k:], img[-k:, :k], img[-k:, -k:]]
    return float(np.median([c.mean() for c in corners]) / 1.2533)


def _disc_kernel(radius_px):
    r = int(np.ceil(radius_px))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = (yy * yy + xx * xx <= radius_px ** 2).astype(float)
    return k / k.sum()


def uniformity(img, fov_yx, centre_yx=None) -> dict:
    """Percent integral uniformity ([G] 5.2, 5.3): the lowest and the highest mean of a 1 cm^2 circular region lying
    inside the 200 cm^2 circular region centred on the phantom; every position on the pixel grid is tried."""
    if centre_yx is None:
        centre_yx = find_centre(img, fov_yx, 0.5 * water_level(img))
    y, x = positions(img.shape, fov_yx)
    px = fov_yx[0] / img.shape[0]
    big, small = np.sqrt(LARGE_ROI_CM2 * 100 / np.pi), np.sqrt(SMALL_ROI_CM2 * 100 / np.pi)
    means = scipy.signal.fftconvolve(img, _disc_kernel(small / px), mode="same")
    dist = np.hypot(y[:, None] - centre_yx[0], x[None, :] - centre_yx[1])
    ok = dist <= big - small
    lo, hi = float(means[ok].min()), float(means[ok].max())
    i_lo, i_hi = np.argwhere(ok & (means == means[ok].min()))[0], np.argwhere(ok & (means == means[ok].max()))[0]
    return dict(piu_percent=100.0 * (1.0 - (hi - lo) / (hi + lo)), low=lo, high=hi, large_mean=float(img[dist <= big].mean()),
                low_at_mm=(float(y[i_lo[0]]), float(x[i_lo[1]])), high_at_mm=(float(y[i_hi[0]]), float(x[i_hi[1]])))


def follow(shape, fov_yx, centre_yx) -> tuple:
    """How far (y, x, in mm) the regions that the guidance places in the background are moved from their standard
    places, for a phantom found at ``centre_yx``: nothing while the phantom lies within ``BACKGROUND_KEEP_MM`` of the
    isocentre (the regions are then still in the background, 3.5 mm or more from the phantom, and every result of a
    phantom positioned within that distance is the one it was), otherwise the offset of the phantom in whole pixels. A
    technologist puts these regions between the phantom he sees and the edge of the image; a region placed from the
    isocentre lies ON a phantom that is 20 mm off (measured on a real series: ratio 0.18 instead of 0.0016)."""
    if centre_yx is None or float(np.hypot(centre_yx[0], centre_yx[1])) <= BACKGROUND_KEEP_MM:
        return 0.0, 0.0
    return tuple(float(round(c / (f / n)) * (f / n)) for c, f, n in zip(centre_yx, fov_yx, shape))


def background_rois(fov_yx, shift_yx=(0.0, 0.0)) -> dict:
    """The four background regions of the ghosting test as rectangles (y0, y1, x0, x1) in mm from the isocentre:
    the standard places of the guidance (round a phantom at the isocentre), moved by ``shift_yx`` (``follow``). A
    moved rectangle may reach past the edge of the field of view; ``rect_mean`` reads it on the periodic image."""
    w, ln = BACKGROUND_ROI_MM[0] / 2, BACKGROUND_ROI_MM[1] / 2
    c = BACKGROUND_AT_MM
    rois = dict(top=(-c - w, -c + w, -ln, ln), bottom=(c - w, c + w, -ln, ln), left=(-ln, ln, -c - w, -c + w), right=(-ln, ln, c - w, c + w))
    if not (shift_yx[0] or shift_yx[1]):
        return rois
    return {k: (r[0] + shift_yx[0], r[1] + shift_yx[0], r[2] + shift_yx[1], r[3] + shift_yx[1]) for k, r in rois.items()}


def _within(p, lo, hi, fov):
    """Pixels of an axis inside [lo, hi] mm. The image of a Fourier-encoded scan is periodic with the field of view,
    so an interval that reaches past an edge continues at the opposite one (for an interval inside the field of view
    the two extra terms select nothing)."""
    return ((p >= lo) & (p <= hi)) | ((p + fov >= lo) & (p + fov <= hi)) | ((p - fov >= lo) & (p - fov <= hi))


def rect_mean(img, fov_yx, rect) -> float:
    y, x = positions(img.shape, fov_yx)
    rows, cols = _within(y, rect[0], rect[1], fov_yx[0]), _within(x, rect[2], rect[3], fov_yx[1])
    return float(img[np.ix_(rows, cols)].mean())


def ghosting(img, fov_yx, centre_yx=None) -> dict:
    """Ghosting ratio ([G] 6.3) = |((top + bottom) - (left + right)) / (2 x large region)|. The large region is
    centred on the phantom as found; the four background regions go with the phantom (``follow``)."""
    if centre_yx is None:
        centre_yx = find_centre(img, fov_yx, 0.5 * water_level(img))
    y, x = positions(img.shape, fov_yx)
    big = np.sqrt(LARGE_ROI_CM2 * 100 / np.pi)
    large = float(img[np.hypot(y[:, None] - centre_yx[0], x[None, :] - centre_yx[1]) <= big].mean())
    m = {k: rect_mean(img, fov_yx, r) for k, r in background_rois(fov_yx, follow(img.shape, fov_yx, centre_yx)).items()}
    return dict(ratio=abs((m["top"] + m["bottom"]) - (m["left"] + m["right"])) / (2.0 * large), large_mean=large, **m)


def resolution(img, fov_yx) -> dict:
    """High-contrast spatial resolution ([G] 2.2): per hole size and direction, whether the four holes of at least one
    row (upper-left array: left-right) or column (lower-right array: top-bottom) are distinguishable.

    The rule that stands for the reader (declared): sample the image at the four hole centres and at the three gaps
    between them (cubic interpolation); the line is resolved when every gap is darker than both of its holes by
    ``RESOLVED_DEPTH`` of the brightest hole. The pattern is looked for where the phantom shows it (``register``) and
    within +-1 pixel of that place, as an eye follows it. Returns the smallest resolved hole size per direction
    (None: not even the largest)."""
    place = register(img, fov_yx)["place"]
    lay = anatomy.acr_layout(dict(slice=1))
    px = fov_yx[0] / img.shape[0]
    coef = scipy.ndimage.spline_filter(img, order=3, mode="nearest")     # cubic-spline coefficients, once
    at = lambda yy, xx: scipy.ndimage.map_coordinates(coef, [yy / px + img.shape[0] // 2, xx / (fov_yx[1] / img.shape[1]) + img.shape[1] // 2],  # noqa: E731
                                                      order=3, mode="nearest", prefilter=False)
    shifts = [(dy * px, dx * px) for dy in np.arange(-1.0, 1.01, 0.25) for dx in np.arange(-1.0, 1.01, 0.25)]
    out = {}
    for array, name in (("UL", "left_right"), ("LR", "top_bottom")):
        best, depths = None, {}
        for d in A["hole_mm"]:
            depth = -1.0
            for line in range(A["holes"]):
                h = sorted((q for q in lay["holes"] if q["d_mm"] == d and q["array"] == array and q["line"] == line), key=lambda q: q["k"])
                hy, hx = place([q["y_mm"] for q in h], [q["x_mm"] for q in h])
                gy, gx = 0.5 * (hy[:-1] + hy[1:]), 0.5 * (hx[:-1] + hx[1:])
                for sy, sx in shifts:
                    hv, gv = at(hy + sy, hx + sx), at(gy + sy, gx + sx)
                    depth = max(depth, float(np.min(np.minimum(hv[:-1], hv[1:]) - gv) / max(hv.max(), 1e-30)))
            depths[d] = depth
            if depth >= RESOLVED_DEPTH:
                best = d
        out[name] = dict(resolved_mm=best, depth=depths)
    return out


def low_contrast(img, fov_yx, section, sd=None) -> dict:
    """Complete spokes of one low-contrast section ([G] 7.2): counted clockwise from the largest disks, stopping at
    the first spoke with a disk that is not seen.

    The rule that stands for the reader (declared): a disk is seen when its mean exceeds the mean of the ring around
    it by ``VISIBLE_Z`` standard errors of that difference (noise from the corners of the image). Disks are looked
    for where the phantom shows them (``register``)."""
    place = register(img, fov_yx)["place"]
    sd = noise_sd(img, fov_yx) if sd is None else sd
    lay = anatomy.acr_layout(dict(slice=section))
    y, x = positions(img.shape, fov_yx)
    px = fov_yx[0] / img.shape[0]
    z = {}
    for dk in lay["disks"]:
        cy, cx = (float(c) for c in place(dk["y_mm"], dk["x_mm"]))
        rows, cols = np.abs(y - cy) < dk["d_mm"] / 2 + 4.0, np.abs(x - cx) < dk["d_mm"] / 2 + 4.0
        sub = img[np.ix_(rows, cols)]
        dist = np.hypot(y[rows][:, None] - cy, x[cols][None, :] - cx)
        disk = dist <= max(dk["d_mm"] / 2 - 0.25 * px, 0.75 * px)
        ring = (dist >= dk["d_mm"] / 2 + 1.0) & (dist <= dk["d_mm"] / 2 + 2.6)
        se = sd * np.sqrt(1.0 / disk.sum() + 1.0 / ring.sum())
        z[(dk["spoke"], dk["disk"])] = float((sub[disk].mean() - sub[ring].mean()) / max(se, 1e-30))
    spokes = 0
    for k in range(1, A["lcd_spokes"] + 1):
        if min(z[(k, j)] for j in range(A["lcd_disks"])) < VISIBLE_Z:
            break
        spokes = k
    return dict(spokes=spokes, z_min_per_spoke=[round(min(z[(k, j)] for j in range(A["lcd_disks"])), 2) for k in range(1, A["lcd_spokes"] + 1)])


def artifact_line(img, fov_yx, centre_yx=None) -> dict:
    """A line of interference in the background (declared detector; the guidance's artifact assessment is a reader's
    judgement, [G] 8). The mean magnitude of every column inside the top and bottom background bands, and of every
    row inside the left and right bands, is compared with its two neighbours three pixels away; a line stands above
    both (the edge of a ghost stands above one only). ``z`` is the largest such excess in units of the spread of the
    excesses (robust). The bands lie round the phantom as found (``follow``): where the phantom is off centre, the
    periodic image is first turned round by whole pixels so that the phantom is central."""
    if centre_yx is None:
        centre_yx = find_centre(img, fov_yx, 0.5 * water_level(img))
    sy, sx = follow(img.shape, fov_yx, centre_yx)
    if sy or sx:
        img = np.roll(img, (-int(round(sy / (fov_yx[0] / img.shape[0]))), -int(round(sx / (fov_yx[1] / img.shape[1])))), axis=(0, 1))
    y, x = positions(img.shape, fov_yx)
    edge = A["diameter_mm"] / 2 + 12.0
    best = 0.0
    for prof in (img[np.abs(y) > edge].mean(0), img[:, np.abs(x) > edge].mean(1)):
        d = np.minimum(prof[3:-3] - prof[:-6], prof[3:-3] - prof[6:])
        spread = 1.4826 * np.median(np.abs(d - np.median(d)))
        best = max(best, float((d.max() - np.median(d)) / max(spread, 1e-30)))
    return dict(z=best)


# ------------------------------------------------------------------------------------------------ the series
def engine_of(sp: dict, grads: dict, section, seed):
    """The scanner around one section of the phantom on the system ``sp``."""
    return kspace.Engine(W.make_world(seed_hex(seed), world_spec(sp, grads, section)))


def scan(engine, series: str, section, seed, noise=True):
    """One scan of the series on one section: (magnitude image, field of view (y, x), plan)."""
    pl = console.plan(SERIES[series], engine.w["console"])
    if not pl["accept"]:
        raise RuntimeError("; ".join(v["message"] for v in pl["violations"]))
    rng = np.random.default_rng([int(seed_hex(seed), 16) % 2 ** 63, list(SERIES).index(series), ACR_SECTION_INDEX[section]]) if noise else None
    img = np.abs(engine.acquire(pl["realised"], rng)[0])
    return img, core.fov_yx(pl["realised"]), pl


ACR_SECTION_INDEX = {s: i for i, s in enumerate(anatomy.ACR_SLICES)}


def run_acr(field: float = 1.5, seed=0, fault: str | None = None, series=("localizer", "t1", "t2")) -> dict:
    """The ACR series on the system (``seed``, ``fault``) and every test of the guidance. Returns the report dict:
    ``tests`` (one row per test: measured, limit, passed or None when not simulated), ``series`` (scan times),
    ``passed`` / ``simulated`` counts. ``series``: which series to acquire (a test needs all of its series)."""
    field = float(field)
    t0 = time.time()
    sp, grads = system_spec(field, seed, fault)
    img, times = {}, {}
    for sl in anatomy.ACR_SLICES:                                        # one world per section, shared by the series that read it
        names = [name for name in series if sl in SLICES[name]]
        if not names:
            continue
        engine = engine_of(sp, grads, sl, seed)
        for name in names:
            im, fov, pl = scan(engine, name, sl, seed)
            img[(name, sl)] = (im, fov)
            times[name] = pl["scan_s"]
    rows = []

    def row(test, measured, limit, passed, detail=None):
        rows.append(dict(test=test, title=TITLES[test], measured=measured, limit=limit, passed=passed, detail=detail or {}))

    axial = [s for s in ("t1", "t2") if s in series]
    # 1 geometric accuracy
    if "localizer" in series and "t1" in series:
        g = {"localizer length": diameters(*img[("localizer", "sagittal")], angles=(90.0,))["lengths_mm"][0]}
        d1 = diameters(*img[("t1", 1)], angles=(90.0, 0.0))["lengths_mm"]
        d5 = diameters(*img[("t1", 5)], angles=(90.0, 0.0, 45.0, 135.0))["lengths_mm"]
        g.update({"slice 1 top-bottom": d1[0], "slice 1 left-right": d1[1], "slice 5 top-bottom": d5[0], "slice 5 left-right": d5[1],
                  "slice 5 diagonal 1": d5[2], "slice 5 diagonal 2": d5[3]})
        err = {k: v - (A["length_mm"] if k.startswith("localizer") else A["diameter_mm"]) for k, v in g.items()}
        worst = max(err, key=lambda k: abs(err[k]))
        row("geometric_accuracy", f"largest error {err[worst]:+.2f} mm ({worst})", f"within +-{LIMITS['geometry_mm']:.1f} mm of 148 / 190 mm",
            bool(max(abs(v) for v in err.values()) <= LIMITS["geometry_mm"]), dict(lengths_mm={k: round(v, 2) for k, v in g.items()}))
    # 2 resolution
    if axial:
        res = {s: resolution(*img[(s, 1)]) for s in axial}
        worst = [(s, k, r[k]["resolved_mm"]) for s, r in res.items() for k in r]
        ok = all(v is not None and v <= LIMITS["resolution_mm"] for _, _, v in worst)
        text = "; ".join(f"{s.upper()} " + " / ".join("unresolved" if r[k]["resolved_mm"] is None else f"{r[k]['resolved_mm']:.1f}" for k in ("left_right", "top_bottom"))
                         for s, r in res.items())
        row("spatial_resolution", text + " mm (left-right / top-bottom)", f"<= {LIMITS['resolution_mm']:.1f} mm in both directions", ok,
            {s: {k: dict(resolved_mm=r[k]["resolved_mm"], depth={str(d): round(v, 3) for d, v in r[k]["depth"].items()}) for k in r} for s, r in res.items()})
    row("slice_thickness", "not simulated", f"{LIMITS['slice_thickness_mm'][0]:.1f} +- {LIMITS['slice_thickness_mm'][1]:.1f} mm", None,
        dict(why=NOT_SIMULATED["slice_thickness"]))
    row("slice_position", "not simulated", f"bar length difference <= {LIMITS['slice_position_mm']:.0f} mm", None, dict(why=NOT_SIMULATED["slice_position"]))
    # 5 uniformity
    if axial:
        u = {s: uniformity(*img[(s, 7)]) for s in axial}
        lim = LIMITS["piu_percent"][field]
        row("intensity_uniformity", "; ".join(f"{s.upper()} PIU {v['piu_percent']:.1f} %" for s, v in u.items()), f">= {lim:.0f} %",
            bool(all(v["piu_percent"] >= lim for v in u.values())), {s: {k: (round(v[k], 4) if isinstance(v[k], float) else v[k]) for k in v} for s, v in u.items()})
    # 6 ghosting
    if "t1" in series:
        gh = ghosting(*img[("t1", 7)])
        row("ghosting", f"ratio {gh['ratio']:.4f}", f"<= {LIMITS['ghosting_ratio']:.3f}", bool(gh["ratio"] <= LIMITS["ghosting_ratio"]),
            {k: round(v, 5) for k, v in gh.items()})
    # 7 low-contrast detectability
    if axial:
        lc = {s: {sl: low_contrast(*img[(s, sl)], section=sl) for sl in (8, 9, 10, 11)} for s in axial}
        tot = {s: sum(v["spokes"] for v in d.values()) for s, d in lc.items()}
        lim = LIMITS["lcd_spokes"][field]
        row("low_contrast", "; ".join(f"{s.upper()} {tot[s]} spokes ({' + '.join(str(lc[s][sl]['spokes']) for sl in (8, 9, 10, 11))})" for s in axial),
            " and ".join(f"{s.upper()} >= {lim[s]}" for s in axial), bool(all(tot[s] >= lim[s] for s in axial)),
            {s: {str(sl): v for sl, v in d.items()} for s, d in lc.items()})
    # 8 artifacts
    if axial:
        art = {s: artifact_line(*img[(s, 7)]) for s in axial}
        row("artifacts", "; ".join(f"{s.upper()} line z {v['z']:.1f}" for s, v in art.items()), f"line z <= {LIMITS['artifact_z']:.0f} (declared)",
            bool(all(v["z"] <= LIMITS["artifact_z"] for v in art.values())), {s: dict(z=round(v["z"], 2)) for s, v in art.items()})
    rows.sort(key=lambda r: TESTS.index(r["test"]))
    snr = None
    if "t1" in series:
        im, fov = img[("t1", 7)]
        snr = ghosting(im, fov)["large_mean"] / max(noise_sd(im, fov), 1e-30)
    from . import ENGINE_ID
    return dict(schema=SCHEMA, engine=ENGINE_ID, field_t=field, seed=seed, fault=fault, what=FAULTS[fault]["what"] if fault else "healthy system",
                tests=rows, failed=[r["test"] for r in rows if r["passed"] is False],
                passed=sum(r["passed"] is True for r in rows), simulated=sum(r["passed"] is not None for r in rows),
                series={k: dict(scan_s=times[k], guidance_s=GUIDANCE_SCAN_S[k], ratio=round(times[k] / GUIDANCE_SCAN_S[k], 4)) for k in times},
                snr_t1_slice7=None if snr is None else round(float(snr), 1), seconds=round(time.time() - t0, 1))


def _clock(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:04.1f}"


def text_table(rep: dict) -> str:
    out = [f"ACR large phantom QA · {rep['field_t']:g} T · seed {rep['seed']} · {rep['what']} · {rep['engine']}",
           "series (console time of one scan = the time of the 11-slice series; guidance time in brackets): "
           + ", ".join(f"{k} {_clock(v['scan_s'])} ({_clock(v['guidance_s'])}, x{v['ratio']:.3f})" for k, v in rep["series"].items()), ""]
    w = max(len(r["title"]) for r in rep["tests"])
    for r in rep["tests"]:
        verdict = "not simulated" if r["passed"] is None else "PASS" if r["passed"] else "FAIL"
        out.append(f"  {r['title']:<{w}}  {verdict:<13}  {r['measured']}   [limit: {r['limit']}]")
        if r["passed"] is None:
            out.append(f"  {'':<{w}}  {'':<13}  {r['detail']['why']}")
    out += ["", f"{'PASS' if rep['passed'] == rep['simulated'] else 'FAIL'} {rep['passed']} / {rep['simulated']} simulated tests"
            + (f"; failed: {', '.join(rep['failed'])}" if rep["failed"] else "") + f" · 2 of {len(TESTS)} tests not simulated"
            + (f" · SNR of T1 slice 7: {rep['snr_t1_slice7']:.0f}" if rep.get("snr_t1_slice7") else "") + f" · {rep['seconds']} s"]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ fault x test matrix
def _matrix_path(directory=None) -> str:
    return os.path.join(directory or str(importlib.resources.files("mrgym.envs").joinpath("phantom_qa")), "acr_fault_matrix.json")


def matrix(fields=(1.5, 3.0), seed=0, faults=None) -> dict:
    """Every fault alone, at both fields: which tests fail, against the tests ``FAULTS`` expects."""
    from . import ENGINE_ID
    rows = []
    for field in fields:
        for name in (None, *(faults or FAULTS)):
            rep = run_acr(field, seed, name)
            want = sorted(FAULTS[name]["fails"]) if name else []
            rows.append(dict(field_t=field, fault=name, what=rep["what"], guidance=FAULTS[name]["guidance"] if name else "-",
                             failed=sorted(rep["failed"]), expected=want, as_expected=sorted(rep["failed"]) == want,
                             measured={r["test"]: r["measured"] for r in rep["tests"] if r["passed"] is not None},
                             limits={r["test"]: r["limit"] for r in rep["tests"]}, series=rep["series"], snr_t1_slice7=rep["snr_t1_slice7"],
                             seconds=rep["seconds"]))
    return dict(schema=SCHEMA, engine=ENGINE_ID, seed=seed, command="mrgym qa acr --matrix --write", rows=rows,
                all_as_expected=all(r["as_expected"] for r in rows))


def matrix_table(m: dict) -> str:
    tests = [t for t in TESTS if t not in NOT_SIMULATED]
    short = dict(geometric_accuracy="geom", spatial_resolution="resol", intensity_uniformity="unif", ghosting="ghost", low_contrast="lcd", artifacts="artif")
    out = [f"fault x test matrix · seed {m['seed']} · {m['engine']} · X = the test fails; every row as expected: {m['all_as_expected']}",
           f"{'field':<6}{'fault':<20}" + "".join(f"{short[t]:<7}" for t in tests) + "expected            guidance"]
    for r in m["rows"]:
        out.append(f"{r['field_t']:<6g}{(r['fault'] or 'none'):<20}" + "".join(f"{'X' if t in r['failed'] else '.':<7}" for t in tests)
                   + f"{(', '.join(short[t] for t in r['expected']) or 'none'):<20}{r['guidance']}" + ("" if r["as_expected"] else "   <-- NOT AS EXPECTED"))
    return "\n".join(out)


def stored_matrix(directory=None) -> dict | None:
    try:
        with open(_matrix_path(directory), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, OSError):
        return None


# ------------------------------------------------------------------------------------------------ the page under docs/
def doc_page(m: dict) -> str:
    """``docs/physics/phantoms.md``: the phantom cards and the QA results. Every number is printed from the constants
    of this package or from the stored fault x test matrix ``m``; nothing in the page is typed."""
    a, n = anatomy.ACR, anatomy.NIST_T1
    healthy = {r["field_t"]: r for r in m["rows"] if r["fault"] is None}
    out = ["# Reference objects: the ACR large phantom and the NIST T1 plate", "",
           f"Generated by `mrgym qa acr --doc docs/physics/phantoms.md` from the constants of `mrgym.scanner.anatomy` and `mrgym.qa` and from the stored "
           f"fault x test matrix (`{m['command']}`, seed {m['seed']}, `{m['engine']}`). Do not edit: regenerate.", "",
           "Two world kinds carry objects a reader of MR knows, with numbers taken from documents that were opened and read for this "
           "package. Each number in the source carries a tag: taken from the document, or declared by this package because the document "
           "does not give it. Records of the documents: [references.md](references.md), section \"Reference objects and the examined outline\".", "",
           "## 1. ACR large phantom (world kind `acr`)", "",
           "Source: *Large and Medium Phantom Test Guidance for the ACR MRI Accreditation Program* (American College of Radiology, the version "
           "dated 10.19.2022), read in full. A world is one section of the phantom on one hidden system: `acr=dict(slice=...)` with "
           f"`slice` in {', '.join(str(x) for x in anatomy.ACR_SLICES)}.", "",
           "### Taken from the guidance", "",
           "| Quantity | Value | Guidance |", "|---|---|---|",
           f"| internal diameter, internal length | {a['diameter_mm']:g} mm, {a['length_mm']:g} mm | Table 1 |",
           f"| axial slices | {a['slices']} of {a['slice_mm']:g} mm with a {a['gap_mm']:g} mm gap | section 0.3 |",
           f"| resolution insert | hole diameters {', '.join(f'{d:g}' for d in a['hole_mm'])} mm; {a['holes']} rows of {a['holes']} holes (upper left) and "
           f"{a['holes']} columns of {a['holes']} holes (lower right) per size, one hole shared; centre-to-centre {a['hole_pitch']:g} x the diameter | Table 1, section 2.2 |",
           f"| low-contrast sections | slices {', '.join(str(k) for k in a['lcd_contrast'])} with contrast {', '.join(f'{100 * v:g}' for v in a['lcd_contrast'].values())} %; "
           f"{a['lcd_spokes']} spokes of {a['lcd_disks']} disks, {a['lcd_disk_mm'][0]:g} mm (first spoke) to {a['lcd_disk_mm'][1]:g} mm (tenth) | section 7.2 |",
           "| series | " + "; ".join(f"{k}: SE {v['tr_ms']:g} / {v['te_ms']:g} ms, FOV {v['fov_mm']:g} mm, {v['matrix']} x {v['matrix']}, {v['slice_mm']:g} mm, "
                                     f"{v['averages']} average, {_clock(GUIDANCE_SCAN_S[k])} min:s" for k, v in SERIES.items()) + " | Table 2 |",
           f"| geometric accuracy | every length within +-{LIMITS['geometry_mm']:g} mm | section 1.4 |",
           f"| high-contrast resolution | {LIMITS['resolution_mm']:g} mm or smaller in both directions, both series | section 2.4 |",
           f"| slice thickness (not simulated) | {LIMITS['slice_thickness_mm'][0]:g} +- {LIMITS['slice_thickness_mm'][1]:g} mm | section 3.4 |",
           f"| slice position (not simulated) | bar length difference <= {LIMITS['slice_position_mm']:g} mm | section 4.4 |",
           "| image intensity uniformity | PIU >= " + ", ".join(f"{v:g} % at {k:g} T" for k, v in LIMITS["piu_percent"].items())
           + " (should be >= " + ", ".join(f"{v:g} %" for v in LIMITS["piu_should_percent"].values()) + f"); regions of {SMALL_ROI_CM2:g} cm^2 inside {LARGE_ROI_CM2:g} cm^2 | section 5.4, Table 4 |",
           f"| percent-signal ghosting | ratio <= {LIMITS['ghosting_ratio']:g} | section 6.4 |",
           "| low-contrast detectability | total spokes " + "; ".join(f"{k:g} T: T1 >= {v['t1']}, T2 >= {v['t2']}" for k, v in LIMITS["lcd_spokes"].items()) + " | section 7.4, Table 5 |",
           "", "### Declared by this package (the guidance does not give them)", "",
           f"- The solution (10 mM NiCl2, 75 mM NaCl, guidance 0.2) has T1 {a['fill']['t1_ms']:g} ms and T2 {a['fill']['t2_ms']:g} ms at both fields: "
           "constants chosen by the authors, the same for every seed. They set the signal ratio of the T1 and T2 series, nothing else.",
           f"- Layout inside a section: the block of the resolution insert (y {a['insert_y_mm'][0]:g} to {a['insert_y_mm'][1]:g} mm, x {a['insert_x_mm'][0]:g} to "
           f"{a['insert_x_mm'][1]:g} mm), the stagger of the hole rows ({a['hole_stagger']:g} x the diameter per row), the grid of slice 5 (pitch {a['grid_pitch_mm']:g} mm, "
           f"bars {a['grid_bar_mm']:g} mm), the place of the disks on a spoke ({', '.join(f'{r:g}' for r in a['lcd_radius_mm'])} mm from the centre), the first spoke "
           f"{a['lcd_first_deg']:g} degrees clockwise of 12 o'clock, and the eight disk diameters between the two the guidance states (a geometric progression: "
           + ", ".join(f"{d:.2f}" for d in anatomy.acr_lcd_disk_mm()) + " mm).",
           f"- The bandwidth of the series ({SERIES['t1']['bandwidth_hz_per_px']:g} Hz per pixel) and the healthy system: receiver noise density {HEALTHY['sigma0']:g} "
           f"(3 T; twice that at 1.5 T), transmit gain within {100 * HEALTHY['tx_gain_dev'][1]:g} %, gradients within {100 * HEALTHY_GRADIENT:g} %. The noise level is a "
           "calibration, not a measurement: it was chosen so that a healthy system scores above the guidance's low-contrast limits at both fields.",
           f"- Two readers are replaced by stated rules. Resolution: a row of four holes is resolved when every gap is darker than both of its holes by "
           f"{RESOLVED_DEPTH:g} of the brightest hole. Low contrast: a disk is seen when its mean exceeds the mean of the ring around it by {VISIBLE_Z:g} standard errors.",
           f"- The artifact assessment of the guidance (section 8) is a reader's judgement with no number; here it is a detector of a line of interference in the "
           f"background with the limit z <= {LIMITS['artifact_z']:g}.",
           f"- The four background regions of the ghosting test and the bands of the line detector lie round the phantom as it is found in the image: at the places "
           f"of the guidance ({BACKGROUND_AT_MM:g} mm from the isocentre) while the phantom is within {BACKGROUND_KEEP_MM:g} mm of the isocentre, and moved with it in "
           "whole pixels beyond that, read on the periodic image. On a real image whose phantom lay well off centre, regions placed from the isocentre had "
           "given a ghosting ratio far too large ([measured.md](measured.md), row M6.13).",
           "- A plastic part displaces solution, so the low-contrast sheet lowers the weight of the solution by the contrast and its holes restore it. The contrast "
           "therefore does not change with the slice thickness of the protocol as it does in the phantom.",
           "", "### Not simulated", ""]
    out += [f"- **{TITLES[k][2:]}**: {v}." for k, v in NOT_SIMULATED.items()]
    out += ["- The localiser plane shows the outline of the phantom (a rectangle) and none of its inner structures.",
            "- The series is acquired one section per scan; the console time of one scan is compared with the guidance's time of the 11-slice series "
            "(slices of a conventional spin echo are interleaved inside one TR). There is no multi-slice acquisition and no cross-talk.",
            "- A gradient calibration error is a scaling of the encoded positions (`grad_scale` of the hidden system). The image intensity is not divided "
            "by the scale as the k-space scaling would.",
            "", "## 2. `mrgym qa acr`: a healthy system against the limits", ""]
    for field, r in healthy.items():
        out += [f"`mrgym qa acr --field {field:g} --seed {m['seed']}`: " + ("every simulated test passes" if not r["failed"] else "FAILED: " + ", ".join(r["failed"]))
                + f"; SNR of the T1 scan of slice 7: {r['snr_t1_slice7']:.0f}; console time against the guidance: "
                + ", ".join(f"{k} {_clock(v['scan_s'])} / {_clock(v['guidance_s'])} (x{v['ratio']:.3f})" for k, v in r["series"].items()) + ".", "",
                "| Test | Measured | Limit |", "|---|---|---|"]
        out += [f"| {TITLES[t]} | {r['measured'].get(t, 'not simulated')} | {r['limits'][t]} |" for t in TESTS]
        out.append("")
    out += ["## 3. Fault x test matrix", "",
            "Each fault alone, at the size given, on the same system; X = the test fails. A fault is a state of the hidden system or of its fields "
            "(world spec), imaged by the engine like anything else. \"Guidance\" is the section that names the cause.", "",
            "| Field | Fault | What is wrong | " + " | ".join(TITLES[t][2:] for t in TESTS if t not in NOT_SIMULATED) + " | Guidance |",
            "|---|---|---|" + "---|" * (len(TESTS) - len(NOT_SIMULATED)) + "---|"]
    for r in m["rows"]:
        out.append(f"| {r['field_t']:g} T | {r['fault'] or 'none'} | {r['what']} | " + " | ".join("X" if t in r["failed"] else "" for t in TESTS if t not in NOT_SIMULATED)
                   + f" | {r['guidance']} |")
    out += ["", f"Every row is the row `mrgym.qa.FAULTS` expects: {m['all_as_expected']}. The seven mechanisms of the matrix test: {', '.join(MATRIX_FAULTS)}.", "",
            "Three rows are findings rather than confirmations:", "",
            "- **A centre-frequency error alone fails nothing.** The image moves by offset / bandwidth pixels and every test is relative to the phantom.",
            "- **A rocking phantom fails resolution and low contrast, not the ghosting ratio.** It blurs the image along the motion and puts its ghost on the "
            "edges of the phantom; the four background regions of the ghosting test hardly see it at this amplitude. The guidance names motion as a cause "
            "of ghosting and ghosting as a cause of resolution and low-contrast failures.",
            "- **A transmit fault fails low contrast as well as uniformity.** A wrong flip angle costs signal; for the local transmit fault the stated reader "
            "rule also loses the lowest-contrast disks on the slope of the intensity, where an eye would not.",
            "", "## 4. NIST T1 plate (world kind `nist_t1`)", "",
            "The NiCl2 array of the ISMRM / NIST system phantom as a 2D world. The object is the same for every seed; its position in the scanner "
            "(within 1 mm and 0.3 degrees, as for the ACR phantom) and the system round it are drawn.", "",
            "Opened and used:", "",
            f"- Stupic et al. 2021 (article body): a spherical shell of {n['shell_mm']:g} mm inner diameter; array spheres of {n['sphere_mm']:g} mm inner and "
            f"{n['sphere_outer_mm']:g} mm outer diameter; {n['ring_n']} spheres on a {n['ring_mm']:g} mm radius and 4 inside on a {n['grid_mm']:g} mm grid; the Ni-12 "
            f"solution at 3 T and 20.0 degrees C: T1 {n['t1_ms'][11]:g} ms, T2 {n['t2_ms_sphere_12']:g} ms; regions of 10 mm diameter.",
            "- Table S1 of the supporting information of the article (NiCl2 array), column \"NMR measured T1 @ 3.0 T\", 20 degrees C: "
            + ", ".join(f"{v:g}" for v in n["t1_ms"]) + " ms. The list of the ISMRM reproducibility challenge (`rrsg2020/analysis`, `src/nist.py`, phantoms "
            "with serial number below 0043), from which engine 0.2.0 took the values, is the same except that it writes the fifth as 496.",
            "", "In the package, and not in this world yet:", "",
            "- The other columns of Tables S1, S2 and S4 of the article (sphere positions, T1 and T2 of every sphere of both plates at 1.5 and 3 T, the water). They "
            "are in the package as `anatomy.NIST_S1`, `NIST_S2` and the T2 plate `anatomy.nist_t2`. The world `nist_t1` uses the 3 T column of T1 only: it exists "
            "at 3 T only, its order of the spheres (1 to 10 round the ring, 11 to 14 on the inner grid) turned out to be the published one, and T2 of a sphere is "
            f"still T1 x ({n['t2_ms_sphere_12']:g} / {n['t1_ms'][11]:g}), the ratio of sphere 12, a declared rule and not the published T2. The differences, with "
            "the outline of the plate, are rows M0.3, M0.4 and M0.8 of [measured.md](measured.md); row M0.2 holds the T1 list equal to the table.",
            f"- The relaxation times of the water round the spheres (\"long-T1 deionized water\" in the article): declared, T1 {n['fill']['t1_ms']:g} ms, T2 {n['fill']['t2_ms']:g} ms.",
            "- The second list of the same file (serial numbers from 0043) was opened and is not used; the temperature correction of the file was not read.",
            "- The inversion-recovery protocol of the challenge has been run on this world and compared with the maps the challenge collected on real scanners, and "
            "the variable-flip-angle protocol of the article with its published rule: [measured.md](measured.md), rungs M1 and M2.", ""]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ command line
def add_parser(sub):
    p = sub.add_parser("qa", help="phantom QA: the ACR large-phantom tests on the virtual scanner, with or without a system fault")
    p.add_argument("phantom", choices=("acr",))
    p.add_argument("--field", type=float, default=1.5, choices=(1.5, 3.0), help="field strength in tesla (default 1.5)")
    p.add_argument("--seed", default="0", help="the system under test (default 0)")
    p.add_argument("--fault", default=None, choices=tuple(FAULTS), help="inject one system fault")
    p.add_argument("--matrix", action="store_true", help="every fault alone at both fields against every test (minutes)")
    p.add_argument("--write", action="store_true", help="with --matrix: store the result as package data")
    p.add_argument("--doc", metavar="PATH", help="write the page of docs/physics/phantoms.md from the stored matrix and exit")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_qa)
    return p


def cmd_qa(a) -> int:
    if a.doc:
        m = stored_matrix()
        if not m:
            print("no stored fault matrix: run `mrgym qa acr --matrix --write` first")
            return 1
        with open(a.doc, "w", encoding="utf-8") as f:
            f.write(doc_page(m))
        print(f"wrote {a.doc}")
        return 0
    if a.matrix:
        m = matrix(seed=a.seed)
        if a.write:
            path = _matrix_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(m, f, indent=1)
                f.write("\n")
        print(json.dumps(m, indent=1) if a.json else matrix_table(m))
        return 0 if m["all_as_expected"] else 1
    rep = run_acr(a.field, a.seed, a.fault)
    print(json.dumps(rep, indent=1) if a.json else text_table(rep))
    return 0 if rep["passed"] == rep["simulated"] else 1
