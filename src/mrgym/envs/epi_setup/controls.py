"""Control policies of ``Scanner/EPISetup``. Every policy plays through the agent's interface only.

Must pass
  reference      a signal-free scan for the noise; where the object holds oil, a quick two-echo scan at opposed phase
                 that shows where; a two-echo gradient echo (the off-resonance as a smooth surface over the object,
                 the extent of the object along both axes, a first value of T2*); one EPI at the highest bandwidth
                 for the gradient delay (the ghost of the object in the rows it leaves empty). For each phase-encode axis
                 (and blip polarity) the shortest phase field of view that holds the displaced object with a margin,
                 the lowest bandwidth whose displacement stays under the limit, the predicted ghost and the gap to the
                 displaced fat, the field of view centred on the displaced object; the candidate with the largest
                 predicted SNR is run once and its averages counted.
Must fail
  prior          no scan; the console's default EPI (matrix 96, 1500 Hz/px, phase-encode axis y, full phase field of view)
  default_epi    one token scan, then the same default EPI
  fastest_epi    one token scan, then the EPI with the shortest echo train: the highest bandwidth, one average
  fixed_recipe   one token scan, then the best constant protocol, tuned on other worlds (``tune``): polarity, bandwidth
                 and phase field of view constant; the matrix, the echo time, the axis (the one on which the stated
                 regions ghost least), the centre (the middle of the stated regions) and the averages (from the stated
                 SNR floor) may be functions of the task text
  wrong_units    the reference with the off-resonance taken as the phase difference per second (rad/s read as Hz)
  sweep          EPIs at five bandwidths, no field map and no model: the lowest bandwidth whose measured ghost is within
                 the limit, the averages from the SNR measured on it
Ablations (the reference with one decision removed)
  ignore_field   no field map: the lowest bandwidth the echo-time window allows
  ignore_ghost   no delay measurement: the candidate with the largest predicted SNR, whatever its ghost
  square_fov     the full phase field of view on the isocentre: no use of the object's shape and position
  ignore_fat     level 3: the oil is not looked for
Reported
  open_loop      a fixed list of scans, written down before anything is seen: noise, field, the delay EPI on both axes,
                 and one EPI of a fixed protocol at each end of the stated echo-time window; the SNR of the chosen
                 protocol is scaled from those two with the SNR laws instead of being measured on it. At the
                 development gate (60 instances) it passed 59, 60 and 56 at levels 1 to 3, the reference 59, 60 and 59:
                 a measurement task
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from . import oracle, task

REFERENCE = "reference"
MUST_FAIL = ("prior", "default_epi", "fastest_epi", "fixed_recipe", "wrong_units", "sweep")
ABLATIONS = {"ignore_field": (1, 2, 3), "ignore_ghost": (1, 2), "square_fov": (1, 2, 3), "ignore_fat": (3,)}
REPORTED = ("open_loop",)
DESCRIPTIONS = dict(
    reference="field map, object extent, noise and gradient delay from four scans of its own; per phase-encode axis the shortest phase "
              "field of view and the lowest bandwidth within the displacement limit; the best candidate is run once for its averages",
    prior="no scan; the console's default EPI",
    default_epi="one token scan, then the console's default EPI",
    fastest_epi="one token scan, then the EPI with the shortest echo train (highest bandwidth, one average)",
    fixed_recipe="one token scan, then the best constant protocol tuned on other worlds",
    wrong_units="the reference with the off-resonance in rad/s read as Hz",
    sweep="EPIs at five bandwidths, the lowest whose measured ghost is within the limit; no field map, no model",
    ignore_field="the reference without the field map: the lowest bandwidth the echo-time window allows",
    ignore_ghost="the reference without the delay measurement: the largest predicted SNR whatever the ghost",
    square_fov="the reference with the full phase field of view on the isocentre",
    ignore_fat="the reference without looking for the oil (level 3)",
    open_loop="a fixed list of scans; the SNR of the chosen protocol is scaled from two EPIs of a fixed protocol with the SNR laws",
)
RAYLEIGH_MEAN = math.sqrt(math.pi / 2.0)                                # mean of the magnitude of unit complex noise
FIELD_SCAN = dict(sequence="spgr", n_echoes=2, echo_spacing_ms=9.2, matrix=96, bandwidth_hz_per_px=200.0, tr_ms=40.0, flip_deg=20.0, averages=2)
FAT_SCAN = dict(sequence="spgr", n_echoes=2, echo_spacing_ms=2.3, matrix=48, bandwidth_hz_per_px=800.0, tr_ms=15.0, flip_deg=15.0)
NOISE_SCAN = dict(sequence="spgr", fov_mm=480.0, matrix=192, tr_ms=12.0, flip_deg=1.0)
NOISE_DISC_MM = 60.0
GHOST_BW, GHOST_AVERAGES, GHOST_CLEAR_MM, GHOST_FOV_MM = 3000.0, 2, 16.0, 320.0      # ghost rows: this far beyond the edges of the object
PROBE_AVERAGES = 8
FIXED_PROBE = dict(bandwidth_hz_per_px=1500.0, phase_fov=1.0, pe_dir="y", pe_polarity=1)       # the calibration EPI of ``open_loop``
MARGIN = dict(displacement=1.035, ghost=1.05, wrap_mm=3.0, fat_mm=3.0, snr=1.0)
EDGE_MM = 0.5                                                           # the object reaches this far beyond the outer edge of its last bright pixel
FAT_HZ_PER_T = (-110.7, -144.8, -161.8)                                 # the strong lines of fat, -2.6, -3.4 and -3.8 ppm: Hz per tesla
FIELD_REACH_MM = (30.0, 42.0, 55.0, 68.0, 82.0, 96.0, 110.0, 1e3)        # the field is fitted outward from the middle of the object in these steps
OBJECT_SHARE = 0.2                                                      # object = pixels above this share of the 99.5th percentile


# ------------------------------------------------------------------------------------------------ measurements
def _task(sc):
    return sc.info()["task"]


def _fixed(t):
    return {k: t["scan"][k] for k in ("sequence", "fov_mm", "slice_mm", "tr_ms", "fat_sat")}


def noise_factor(rp) -> float:
    """SD of the noise in one pixel per unit receiver noise density, from the realised protocol (the SNR laws)."""
    return rp["matrix"] * math.sqrt(rp["lines"] * rp["bandwidth_hz_per_px"] / (rp["averages"] * rp.get("partitions", 1))) \
        / (rp["fov_mm"] * rp["fov_pe_mm"] * rp["lines_full"] / rp["matrix_pe"])


def _noise(sc):
    """Receiver noise density from four large signal-free discs of a wide scan with a 1 degree flip angle."""
    c = NOISE_SCAN["fov_mm"] / 2.0 - NOISE_DISC_MM - 3.0
    discs = [dict(id=f"bg{k}", y_mm=sy * c, x_mm=sx * c, radius_mm=NOISE_DISC_MM) for k, (sy, sx) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1)))]
    r = sc.scan(dict(NOISE_SCAN), discs)
    n = np.array([row["n_pixels"] for row in r["roi_readouts"]], float)
    mean = float(np.sum(n * [row["mean_magnitude"][0] for row in r["roi_readouts"]]) / n.sum())
    return mean / RAYLEIGH_MEAN / noise_factor(r["realised"])


def _object(mag, sd):
    """Pixels of the object in a magnitude image: above ``OBJECT_SHARE`` of the 99.5th percentile and above five noise
    SDs, with at least two of their four neighbours (a single noise pixel is not the object)."""
    on = (mag > OBJECT_SHARE * np.percentile(mag, 99.5)) & (mag > 5.0 * sd)
    pad = np.pad(on, 1)
    near = pad[:-2, 1:-1].astype(int) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    return on & (near >= 2)


def _axes(r, shape):
    fy, fx = r["fov_yx_mm"]
    return (np.arange(shape[0]) - shape[0] // 2) * fy / shape[0], (np.arange(shape[1]) - shape[1] // 2) * fx / shape[1]


def _field(sc, t, sigma0, unit=1.0, fat=()):
    """The two-echo gradient echo: the off-resonance (Hz) as a quadratic surface fitted to the phase difference of the
    two echoes over the object (outward from its middle in steps, the phase of each step unwrapped against the fit of
    the step before; without the discs ``fat``, whose phase also turns with the chemical shift), its value at every marked region, the
    T2* of the regions, and the object as points (y, x, off-resonance) on the pixels of that scan.
    ``unit``: 1, or 2 pi for the unit error of ``wrong_units``."""
    r = sc.scan(dict(FIELD_SCAN, fov_mm=t["scan"]["fov_mm"]), t["regions"])
    te = r["realised"]["echo_times_ms"]
    dte = (te[1] - te[0]) * 1e-3
    out = dict(df={}, t2s={})
    for row in r["roi_readouts"]:
        ratio = row["mean_magnitude"][1] / max(row["mean_magnitude"][0], 1e-12)
        out["t2s"][row["id"]] = float(np.clip((te[1] - te[0]) / max(-math.log(min(ratio, 0.999)), 1e-3), 15.0, 400.0))
    mag = sc.view(r["scan_index"], image=0, kind="magnitude")["array"]
    ph = [sc.view(r["scan_index"], image=k, kind="phase")["array"] for k in (0, 1)]
    y, x = _axes(r, mag.shape)
    iy, ix = np.nonzero(_object(mag, sigma0 * noise_factor(r["realised"])))
    surface = lambda yy, xx: np.stack([np.ones(len(yy)), yy, xx, yy ** 2, xx ** 2, yy * xx], 1)      # noqa: E731
    a, w = surface(y[iy], x[ix]), mag[iy, ix]
    use = np.ones(len(iy), bool)
    for v in fat:
        use &= np.hypot(y[iy] - v["y_mm"], x[ix] - v["x_mm"]) > v["radius_mm"] + 4.0
    dphi = np.angle(np.exp(1j * (ph[1] - ph[0])))[iy, ix]                # wrapped into (-pi, pi]
    far = np.hypot(y[iy] - y[iy].mean(), x[ix] - x[ix].mean())           # distance from the middle of the object
    coef, full = np.zeros(a.shape[1]), dphi
    for reach in FIELD_REACH_MM:                                         # outward: unwrap against the fit so far, fit what lies inside
        full = dphi + 2 * np.pi * np.round((a @ coef - dphi) / (2 * np.pi))
        near = use & (far <= reach)
        coef = np.linalg.lstsq(a[near] * w[near, None], full[near] * w[near], rcond=None)[0]
    hz = unit / (2 * np.pi * dte)
    for reg in t["regions"]:
        out["df"][reg["id"]] = float((surface(np.array([reg["y_mm"]]), np.array([reg["x_mm"]])) @ coef)[0]) * hz
    out["body"] = dict(y=y[iy], x=x[ix], df=(a @ coef) * hz, half_px=0.5 * (y[1] - y[0]) + EDGE_MM)
    return out


def _fat(sc, t, sigma0):
    """Where the oil lies: the pixels whose phase turns by about pi between two echoes 2.3 ms apart (opposed phase at
    1.5 T; water turns far less). Returns a list of discs dict(y_mm, x_mm, radius_mm), empty when there is none."""
    r = sc.scan(dict(FAT_SCAN, fov_mm=t["scan"]["fov_mm"]), t["regions"])
    mag = sc.view(r["scan_index"], image=0, kind="magnitude")["array"]
    ph = [sc.view(r["scan_index"], image=k, kind="phase")["array"] for k in (0, 1)]
    turn = np.abs(np.angle(np.exp(1j * (ph[1] - ph[0]))))
    fat = _object(mag, sigma0 * noise_factor(r["realised"])) & (turn > 2.0)
    if fat.sum() < 6:
        return []
    y, x = _axes(r, mag.shape)
    iy, ix = np.nonzero(fat)
    area = fat.sum() * (y[1] - y[0]) * (x[1] - x[0])
    return [dict(y_mm=float(y[iy].mean()), x_mm=float(x[ix].mean()), radius_mm=math.sqrt(area / math.pi) + 0.5 * (y[1] - y[0]))]      # its edge pixels are not counted


def _request(t, fields):
    """A protocol request from delivered-style fields: ``centre_pe_mm`` is the console's centre along the phase-encode axis."""
    out = dict(_fixed(t), **{k: v for k, v in fields.items() if k != "centre_pe_mm"})
    if fields.get("centre_pe_mm"):
        out["centre_y_mm" if fields.get("pe_dir", "y") == "y" else "centre_x_mm"] = fields["centre_pe_mm"]
    return out


def _epi(sc, t, rois, **fields):
    return sc.scan(_request(t, fields), rois)


def _delay(sc, t, matrix, sigma0, body, pe):
    """(|gradient delay| in seconds, rows it was read on) from one EPI at the highest bandwidth on a field of view of
    ``GHOST_FOV_MM``, phase-encoded along ``pe``. The ghost of the object lies half a field of view away along that
    axis; the rows of the image that the
    object leaves empty (``GHOST_CLEAR_MM`` beyond its edges) hold the ghost of the rows half a field of view from
    them. With phi = k c at readout coordinate c the image holds cos(phi) of the object in place and sin(phi) of it
    in the ghost, so over those rows  sum |ghost|^2 = sum tan^2(k c) |source|^2  (both without the noise power): solved
    for k by bisection."""
    r = _epi(sc, t, [], fov_mm=GHOST_FOV_MM, matrix=matrix, bandwidth_hz_per_px=GHOST_BW, pe_dir=pe, averages=GHOST_AVERAGES)
    rp = r["realised"]
    fov = rp["fov_mm"]
    mag = sc.view(r["scan_index"], image=0, kind="magnitude")["array"].astype(float)
    y, x = _axes(r, mag.shape)
    if pe == "x":                                                        # rows = phase-encode axis
        mag, y, x = mag.T, x, y
    along, across = (body["y"], body["x"]) if pe == "y" else (body["x"], body["y"])
    empty = np.flatnonzero((y < along.min() - GHOST_CLEAR_MM) | (y > along.max() + GHOST_CLEAR_MM))
    cols = np.flatnonzero((np.abs(x) >= 15.0) & (np.abs(x) <= 0.85 * np.abs(across).max()))
    if not len(empty) or not len(cols):
        return 0.0, 0
    noise = 2.0 * (sigma0 * noise_factor(rp)) ** 2                       # power of the noise in a magnitude pixel
    source = np.maximum(mag[np.ix_((empty + mag.shape[0] // 2) % mag.shape[0], cols)] ** 2 - noise, 0.0)
    ghost = float(np.sum(mag[np.ix_(empty, cols)] ** 2 - noise))
    if ghost <= 0.0:
        return 0.0, len(empty)
    scale = 2 * np.pi * rp["bandwidth_hz_per_px"] * rp["matrix"] / fov      # phi = scale x delay x coordinate
    lo, hi = 0.0, 0.45 * np.pi / (scale * np.abs(x[cols]).max())
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if float(np.sum(np.tan(scale * mid * x[cols]) ** 2 * source)) < ghost else (lo, mid)
    return 0.5 * (lo + hi), len(empty)


# ------------------------------------------------------------------------------------------------ candidates
def _spacing(sc, t, matrix):
    """Echo spacing of the EPI as a function of the bandwidth for this matrix, from three plans: readout plus two ramps,
    the ramp proportional to the bandwidth."""
    plans = [sc.plan(dict(_fixed(t), matrix=matrix, bandwidth_hz_per_px=b)) for b in (600.0, 1500.0, 2800.0)]
    slope = float(np.mean([(p["realised"]["echo_spacing_ms"] - 1000.0 / p["realised"]["bandwidth_hz_per_px"]) / p["realised"]["bandwidth_hz_per_px"] for p in plans]))
    return lambda bw: 1000.0 / bw + slope * bw + 0.01


def _bandwidth_for(spacing, esp_max, lo=500.0, hi=3000.0):
    """The lowest bandwidth (Hz/px, rounded up to 1) whose echo spacing is at most ``esp_max`` ms; None above the range."""
    if spacing(lo) <= esp_max:
        return lo
    if spacing(hi) > esp_max:
        return None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if spacing(mid) <= esp_max else (mid, hi)
    return float(math.ceil(hi))


def _shift(df_hz, pol, esp_ms, fov_pe):
    return -pol * np.asarray(df_hz) * esp_ms * 1e-3 * fov_pe


def _candidates(sc, t, m, use=("field", "ghost", "shape", "fat")):
    """Per phase-encode axis and blip polarity: the shortest phase field of view that holds the displaced object with a
    margin, the lowest bandwidth within the displacement limit, and what it predicts. Returns the list of candidates
    that meet every limit with the margins of ``MARGIN``, best predicted SNR first; each is a dict with the protocol
    fields, ``esp_ms`` and ``relative_snr``. ``use``: which measurements enter (the ablations leave one out)."""
    fov, n = t["scan"]["fov_mm"], oracle.matrix_of(t["pixel_max_mm"])
    spacing = _spacing(sc, t, n)
    ids = [r["id"] for r in t["regions"]]
    df_max = max(abs(m["df"][i]) for i in ids) if "field" in use else 0.0
    t2s = min(m["t2s"][i] for i in ids)
    b0_t = sc.info()["field_strength_t"] if (m.get("fat") and "fat" in use) else None       # one tool call, not one per candidate
    out = []
    for pe_dir in ("y", "x"):
        axis, ro = ("y", "x") if pe_dir == "y" else ("x", "y")
        for pol in ((1, -1) if (m.get("fat") and "fat" in use) else (1,)):
            for lines in range(int(math.ceil(oracle.PHASE_FOV_MIN * n / 2.0)) * 2 if "shape" in use else n, n + 1, 2):
                fov_pe = fov * lines / n
                # displacement = off-resonance x echo spacing x phase field of view
                esp_max = t["displacement_max_mm"] / MARGIN["displacement"] / (max(df_max, 1e-6) * 1e-3 * fov_pe)
                bw = _bandwidth_for(spacing, esp_max)
                if bw is None:
                    continue
                esp = spacing(bw)
                pos = m["body"][axis] + (_shift(m["body"]["df"], pol, esp, fov_pe) if "field" in use else 0.0)
                lo, hi, px = pos.min() - m["body"]["half_px"], pos.max() + m["body"]["half_px"], fov_pe / lines
                centre = round(float(0.5 * (lo + hi) + 0.5 * px), 1) if "shape" in use else 0.0      # the object in the middle of the field of view
                if min(centre + fov_pe / 2 - px / 2 - hi, lo - (centre - fov_pe / 2 - px / 2)) < MARGIN["wrap_mm"] and "shape" in use:
                    continue                                             # the object would wrap: a longer phase field of view
                geometry = dict(matrix=n, pe_dir=pe_dir, pe_polarity=pol, phase_fov=lines / n, centre_pe_mm=centre)
                pl = sc.plan(_request(t, dict(geometry, bandwidth_hz_per_px=bw)))
                if not pl["accept"]:
                    break
                if pl["timing"]["te_min_ms"] > t["te_window_ms"][1] and bw < 3000.0:         # the echo train must fit the window
                    pl = _fit_window(sc, t, geometry, bw)
                if pl["timing"]["te_min_ms"] > t["te_window_ms"][1]:
                    break
                rp = pl["realised"]
                te = max(pl["timing"]["te_min_ms"], t["te_window_ms"][0])
                ghost = max(abs(math.sin(2 * np.pi * rp["bandwidth_hz_per_px"] * n * m["delay"] * r[ro + "_mm"] / fov)) for r in t["regions"]) \
                    if "ghost" in use else 0.0
                gap = float("inf")
                if m.get("fat") and "fat" in use:
                    for v in m["fat"]:
                        for hz in FAT_HZ_PER_T:
                            moved = v[axis + "_mm"] + float(_shift(hz * b0_t, pol, rp["echo_spacing_ms"], fov_pe))
                            for r in t["regions"]:
                                along = (moved - r[axis + "_mm"] + fov_pe / 2) % fov_pe - fov_pe / 2
                                gap = min(gap, math.hypot(along, v[ro + "_mm"] - r[ro + "_mm"]) - v["radius_mm"] - r["radius_mm"])
                if ghost * MARGIN["ghost"] <= t["ghost_max"] and gap >= MARGIN["fat_mm"]:
                    out.append(dict(geometry, bandwidth_hz_per_px=rp["bandwidth_hz_per_px"], te_ms=te, esp_ms=rp["echo_spacing_ms"], ghost=ghost,
                                    relative_snr=math.sqrt(lines / rp["bandwidth_hz_per_px"]) * math.exp(-te / t2s)))
                break                                                    # the shortest phase field of view that holds the object
    return sorted(out, key=lambda c: -c["relative_snr"])


def _fit_window(sc, t, geometry, bw):
    """The plan at the first bandwidth of the ladder bw x 1.04, x 1.04, ... (rounded up, at most 3000) whose shortest
    echo time is inside the window, or the plan at 3000. The shortest echo time falls with the bandwidth, so the
    ladder is searched by bisection: a handful of plans (every plan is a tool call of the episode)."""
    ladder = [bw]
    while ladder[-1] < 3000.0:
        ladder.append(min(float(math.ceil(ladder[-1] * 1.04)), 3000.0))
    plan = lambda k: sc.plan(_request(t, dict(geometry, bandwidth_hz_per_px=ladder[k])))      # noqa: E731
    lo, hi = 0, len(ladder) - 1                                          # ladder[lo] is outside the window
    top = plan(hi)
    if top["timing"]["te_min_ms"] > t["te_window_ms"][1]:
        return top
    while hi - lo > 1:
        mid = (lo + hi) // 2
        pl = plan(mid)
        if pl["timing"]["te_min_ms"] > t["te_window_ms"][1]:
            lo = mid
        else:
            hi, top = mid, pl
    return top


def _fields(c, averages):
    return dict({k: c[k] for k in oracle.FIELDS if k != "averages"}, averages=int(averages))


def _averages(snr1, snr_min):
    return int(np.clip(math.ceil((snr_min * MARGIN["snr"] / max(snr1, 1e-9)) ** 2), 1, oracle.AVERAGES_MAX))


def _magnitude(row, sd):
    """Mean magnitude of a region without the noise floor."""
    return math.sqrt(max(row["mean_magnitude"][0] ** 2 - 2 * sd * sd * 0.5, 1e-30))


def _run(sc, t, c, sigma0, averages=PROBE_AVERAGES):
    """Run a candidate once: its one-average SNR in the weakest marked region."""
    r = _epi(sc, t, t["regions"], **_fields(c, averages))
    rp = r["realised"]
    sd = sigma0 * noise_factor(rp)
    return min(_magnitude(row, sd) for row in r["roi_readouts"]) / (sd * math.sqrt(averages))


def _fallback(sc, t):
    """What is delivered when no candidate meets the limits with the margins: the default axis at the highest bandwidth
    whose echo train fits the window."""
    n = oracle.matrix_of(t["pixel_max_mm"])
    pl = sc.plan(dict(_fixed(t), matrix=n, bandwidth_hz_per_px=3000.0))
    return dict(pe_dir="y", pe_polarity=1, bandwidth_hz_per_px=3000.0, matrix=n, phase_fov=1.0, centre_pe_mm=0.0,
                te_ms=max(pl["timing"]["te_min_ms"], t["te_window_ms"][0]), esp_ms=pl["realised"]["echo_spacing_ms"], relative_snr=0.0)


def _play(sc, use=("field", "ghost", "shape", "fat"), unit=1.0, adaptive=True):
    """The reference and its variants. ``adaptive``: the delay is read on the shorter axis of the object and the chosen
    candidate is run for its SNR. Otherwise (the open loop) the list of scans is fixed before anything is seen: the
    delay EPI on both axes (the one with more empty rows is used) and one EPI of a fixed protocol at each end of the
    stated echo-time window, from which the SNR of the chosen candidate is scaled with the SNR laws."""
    t = _task(sc)
    sigma0 = _noise(sc)
    fat = _fat(sc, t, sigma0) if (t.get("has_fat") and "fat" in use) else []
    m = _field(sc, t, sigma0, unit, fat)
    n = oracle.matrix_of(t["pixel_max_mm"])
    m["fat"], m["delay"] = fat, 0.0
    if "ghost" in use:
        axes = ("y" if np.ptp(m["body"]["y"]) <= np.ptp(m["body"]["x"]) else "x",) if adaptive else ("y", "x")
        m["delay"] = max((_delay(sc, t, n, sigma0, m["body"], pe) for pe in axes), key=lambda v: v[1])[0]
    found = _candidates(sc, t, m, use)
    best = found[0] if found else _fallback(sc, t)
    if adaptive:
        return _fields(best, _averages(_run(sc, t, best, sigma0), t["snr_min"]))
    plan = sc.plan(_request(t, _fields(best, 1)))["realised"]
    te_min = sc.plan(dict(_fixed(t), matrix=n, **FIXED_PROBE))["timing"]["te_min_ms"]
    probes = []
    for te in (max(te_min, t["te_window_ms"][0]), max(te_min + 5.0, t["te_window_ms"][1])):
        r = _epi(sc, t, t["regions"], matrix=n, averages=PROBE_AVERAGES // 2, te_ms=te, **FIXED_PROBE)
        sd = sigma0 * noise_factor(r["realised"])
        probes.append((r["realised"]["te_ms"], {row["id"]: _magnitude(row, sd) for row in r["roi_readouts"]}))
    (te1, s1), (te2, s2) = probes
    signal = min(s1[i] * (s2[i] / s1[i]) ** ((plan["te_ms"] - te1) / (te2 - te1)) for i in s1)          # each region decays exponentially
    return _fields(best, _averages(signal / (sigma0 * noise_factor(plan)), t["snr_min"]))


# ------------------------------------------------------------------------------------------------ the policies
def reference(sc):
    return _play(sc)


def open_loop(sc):
    return _play(sc, adaptive=False)


def wrong_units(sc):
    return _play(sc, unit=2 * np.pi)


def ignore_field(sc):
    return _play(sc, use=("ghost", "shape", "fat"))


def ignore_ghost(sc):
    return _play(sc, use=("field", "shape", "fat"))


def square_fov(sc):
    return _play(sc, use=("field", "ghost", "fat"))


def ignore_fat(sc):
    return _play(sc, use=("field", "ghost", "shape"))


def _default(sc, t, **change):
    """The console's default EPI with the fixed fields of the task: its echo time is the shortest inside the window."""
    fields = dict(dict(pe_dir="y", pe_polarity=1, bandwidth_hz_per_px=1500.0, matrix=96, phase_fov=1.0, centre_pe_mm=0.0), **change)
    pl = sc.plan(_request(t, fields))
    if not pl["accept"]:
        return dict(fields, te_ms=t["te_window_ms"][1], averages=1)
    # the shortest echo time inside the window; the shortest the console has when the echo train is longer than the window
    return dict(fields, te_ms=max(pl["timing"]["te_min_ms"], t["te_window_ms"][0]), averages=1)


def prior(sc):
    t = _task(sc)
    return _default(sc, t)


def _token(sc, t):
    _epi(sc, t, t["regions"], matrix=64)


def default_epi(sc):
    t = _task(sc)
    _token(sc, t)
    return _default(sc, t)


def fastest_epi(sc):
    t = _task(sc)
    _token(sc, t)
    return _default(sc, t, bandwidth_hz_per_px=3000.0, matrix=oracle.matrix_of(t["pixel_max_mm"]))


def recipe_axis(recipe, t) -> str:
    """The phase-encode axis of a recipe: a constant, or ("ghost_free") the axis whose readout runs along the direction
    in which the marked regions lie closer to the isocentre: a function of the stated region positions."""
    if recipe["pe_dir"] != "ghost_free":
        return recipe["pe_dir"]
    return "y" if max(abs(r["x_mm"]) for r in t["regions"]) <= max(abs(r["y_mm"]) for r in t["regions"]) else "x"


def recipe_centre(recipe, t) -> float:
    """The centre of the field of view of a recipe along its phase-encode axis: the isocentre, or ("regions") the mean
    position of the stated regions along that axis (the vials lie around the middle of the object)."""
    if recipe.get("centre") != "regions":
        return 0.0
    return round(float(np.mean([r[recipe_axis(recipe, t) + "_mm"] for r in t["regions"]])), 1)


def recipe_averages(recipe, t) -> int:
    """The averages of a recipe: a constant, or (``snr_ref``) the number the stated SNR floor asks for when one average
    is taken to give ``snr_ref`` at a 3.75 mm pixel (SNR falls with the pixel area and the square root of the lines)."""
    if recipe.get("snr_ref") is None:
        return int(recipe["averages"])
    one = recipe["snr_ref"] * (t["pixel_max_mm"] / 3.75) ** 1.5
    return int(np.clip(math.ceil((t["snr_min"] / one) ** 2), 1, 8))


def from_recipe(recipe, t, te_min_ms) -> dict:
    """The protocol a recipe stands for on a task: constants, the smallest matrix that meets the stated pixel ceiling and
    the shortest echo time inside the stated window."""
    return dict(pe_dir=recipe_axis(recipe, t), pe_polarity=int(recipe["pe_polarity"]), bandwidth_hz_per_px=float(recipe["bandwidth_hz_per_px"]),
                matrix=oracle.matrix_of(t["pixel_max_mm"]), phase_fov=float(recipe["phase_fov"]), centre_pe_mm=recipe_centre(recipe, t),
                te_ms=max(te_min_ms, t["te_window_ms"][0]), averages=recipe_averages(recipe, t))


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        _token(sc, t)
        n = oracle.matrix_of(t["pixel_max_mm"])
        lines = int(round(n * recipe["phase_fov"] / 2.0) * 2)
        pl = sc.plan(dict(_fixed(t), matrix=n, pe_dir=recipe_axis(recipe, t), phase_fov=lines / n, bandwidth_hz_per_px=recipe["bandwidth_hz_per_px"]))
        return from_recipe(dict(recipe, phase_fov=lines / n), t, pl["timing"]["te_min_ms"] if pl["accept"] else t["te_window_ms"][1])
    return policy


DEFAULT_RECIPE = dict(pe_dir="ghost_free", pe_polarity=1, bandwidth_hz_per_px=1350.0, phase_fov=0.8, centre="regions", averages=2, snr_ref=None)
SWEEP_BW = (700.0, 1000.0, 1400.0, 2000.0, 2800.0)


def sweep(sc):
    """EPIs at five bandwidths on the default axis and the full phase field of view. For each: the ghost of every marked
    region read half a field of view away, and the SNR of the regions against the noise of four corner discs of the
    same image. Delivers the lowest bandwidth whose measured ghost is within the limit, with the averages its measured
    SNR asks for. No field map, no delay, no SNR law."""
    t = _task(sc)
    fov, n = t["scan"]["fov_mm"], oracle.matrix_of(t["pixel_max_mm"])
    corners = [dict(id=f"bg{k}", y_mm=sy * (fov / 2 - 11.0), x_mm=sx * (fov / 2 - 11.0), radius_mm=9.0) for k, (sy, sx) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1)))]
    ghosts = [dict(id="g" + r["id"], y_mm=r["y_mm"] - math.copysign(fov / 2, r["y_mm"]), x_mm=r["x_mm"], radius_mm=r["radius_mm"]) for r in t["regions"]]
    best = None
    for bw in SWEEP_BW:
        pl = sc.plan(dict(_fixed(t), matrix=n, bandwidth_hz_per_px=bw))
        if not pl["accept"] or pl["timing"]["te_min_ms"] > t["te_window_ms"][1] or pl["scan_s"] > sc.budget()["remaining_s"]:
            continue
        te = max(pl["timing"]["te_min_ms"], t["te_window_ms"][0])
        r = _epi(sc, t, t["regions"] + ghosts + corners, matrix=n, bandwidth_hz_per_px=bw, te_ms=te)
        rows = {row["id"]: row for row in r["roi_readouts"]}
        sd = float(np.mean([rows[f"bg{k}"]["mean_magnitude"][0] for k in range(4)])) / RAYLEIGH_MEAN
        ghost = max(rows["g" + i["id"]]["mean_magnitude"][0] / max(rows[i["id"]]["mean_magnitude"][0], 1e-12) for i in t["regions"])
        snr1 = min(rows[i["id"]]["mean_magnitude"][0] for i in t["regions"]) / sd
        if ghost <= t["ghost_max"] or best is None:
            cand = dict(pe_dir="y", pe_polarity=1, bandwidth_hz_per_px=bw, matrix=n, phase_fov=1.0, centre_pe_mm=0.0, te_ms=te, averages=_averages(snr1, t["snr_min"]))
            if best is None or ghost <= t["ghost_max"]:
                best = cand
            if ghost <= t["ghost_max"]:
                break
    return best or _default(sc, t, matrix=n)


def policies(recipe=None):
    return dict(reference=reference, prior=prior, default_epi=default_epi, fastest_epi=fastest_epi, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE),
                wrong_units=wrong_units, sweep=sweep, ignore_field=ignore_field, ignore_ghost=ignore_ghost, square_fov=square_fov,
                ignore_fat=ignore_fat, open_loop=open_loop)


# ------------------------------------------------------------------------------------------------ tuning on other worlds
TUNE_EXTRA = 1              # further worlds drawn per tuning instance (``tune`` gets 16 instances from the gate)
RECIPE_BW = (600.0, 750.0, 900.0, 1100.0, 1350.0, 1650.0, 2000.0, 2450.0)
RECIPE_PF = (0.6, 0.7, 0.8, 0.9, 1.0)
RECIPE_SNR = (12.0, 18.0, 27.0, 40.0)                                   # assumed one-average SNR at a 3.75 mm pixel


def recipes(has_fat: bool) -> list:
    """The constant protocols ``tune`` chooses from. Axis: y, x, or the axis on which the stated regions ghost least;
    averages: 1, 2, 3, or the number the stated SNR floor asks for at an assumed one-average SNR."""
    counts = [dict(averages=n, snr_ref=None) for n in task.AVERAGES_TARGET] + [dict(averages=None, snr_ref=v) for v in RECIPE_SNR]
    return [dict(pe_dir=d, pe_polarity=p, bandwidth_hz_per_px=bw, phase_fov=pf, centre=centre, **c)
            for d in ("y", "x", "ghost_free") for p in ((1, -1) if has_fat else (1,)) for bw in RECIPE_BW for pf in RECIPE_PF
            for centre in (0.0, "regions") for c in counts]


def _other(seed: str, k: int) -> str:
    return hashlib.sha256(f"{seed}|episetup-tuning|{k}".encode()).hexdigest()[:32]


def recipe_passes(cfg: dict, table: list) -> np.ndarray:
    """Which recipes of ``table`` pass an instance, by the task's own rule evaluated with the closed forms of the
    oracle on a coarse grid of the instance's world (displacement, ghost, wrap, fat, echo-time window, SNR, averages)."""
    from ...scanner import console
    from ...scanner import world as W
    from . import metrics as M
    world = W.make_world(cfg["seed"], dict(cfg["world_spec"], n=task.COARSE))
    world["console"]["min_pixel_mm"] = round(2.0 * W.GRID["qa"][1] / W.GRID["qa"][0], 3)
    t = cfg["task"]
    scans, f = M.Scans(world), oracle.facts(world, t)
    loose = oracle.search(world, t, scans, f=f)
    fewest = oracle.averages_for(loose["snr1"], t["snr_min"]) if loose["candidates"] else 0
    noise = scans.engine().noise_sd
    out = np.zeros(len(table), bool)
    n = oracle.matrix_of(t["pixel_max_mm"])
    for k, recipe in enumerate(table):
        lines = int(round(n * recipe["phase_fov"] / 2.0) * 2)
        averages = recipe_averages(recipe, t)
        pl = console.plan(oracle.protocol(dict(matrix=n, pe_dir=recipe_axis(recipe, t), pe_polarity=recipe["pe_polarity"], phase_fov=lines / n,
                                               centre_pe_mm=recipe_centre(recipe, t), bandwidth_hz_per_px=recipe["bandwidth_hz_per_px"])), world["console"])
        if not pl["accept"] or pl["timing"]["te_min_ms"] > t["te_window_ms"][1] or averages > fewest:
            continue
        rp = dict(pl["realised"], te_ms=max(pl["timing"]["te_min_ms"], t["te_window_ms"][0]))
        if max(abs(float(oracle.displacement_mm(r["df_hz"], rp))) for r in f["regions"]) > t["displacement_max_mm"] \
                or max(oracle.ghost_of(r, f["delay_s"], rp) for r in f["regions"]) > t["ghost_max"] \
                or oracle.wrap_clearance_mm(f, rp) < 0.0 or (f["fat"] and oracle.fat_clearance_mm(f, rp) < 0.0):
            continue
        out[k] = loose["model"](rp) / noise(dict(rp, averages=1)) * math.sqrt(averages) >= t["snr_min"]
    return out


def tune(cfgs):
    """The strongest constant protocol on other worlds: the recipe of ``recipes`` that passes the most of them."""
    level = cfgs[0]["level"]
    table = recipes(bool(cfgs[0]["task"].get("has_fat")))
    more = list(cfgs) + [task.make_instance(_other(c["seed"], k), level) for c in cfgs for k in range(TUNE_EXTRA)]
    passes = np.array([recipe_passes(c, table) for c in more])           # [world, recipe]
    rate = passes.mean(0)
    k = int(np.argmax(rate))
    return dict(table[k], tuned_on_worlds=len(more), tuned_pass_rate=round(float(rate[k]), 4))
