"""The optimum of ``Scanner/EPISetup``: the fewest averages with which any allowed protocol meets every requirement, found
on the hidden world.

Verifier side. Independent of the control policies (it must not import ``controls``): it reads the hidden world.

The scan is a single-shot gradient-echo EPI with a fixed repetition time, so its scan time is the number of averages.
Free fields and how the search covers them:

    matrix          the smallest even matrix whose pixel meets the ceiling. A larger one has smaller pixels (less signal
                    per pixel), more lines (a longer echo train) and a larger ghost at the same bandwidth: it is
                    dominated in every requirement and is not searched.
    pe_dir          both axes
    pe_polarity     both signs where the object holds fat (it mirrors the displacement of fat); +1 otherwise
    phase_fov       every value of the console raster (two lines apart) from 0.5 to 1
    centre_pe_mm    the centre that leaves the displaced object the same room on both sides (to 0.1 mm): no other
                    centre holds an object that this one does not
    bandwidth       500 to 3000 Hz/px in steps of 1.5 % (112 values), echo spacing at the console minimum
    te_ms           the shortest the console realises inside the window: the signal only falls with the echo time
    averages        the smallest number that reaches the SNR floor

Is the smallest matrix really enough? Its dominance holds for displacement at a given echo spacing, ghost, echo time and
SNR; ``tests/test_epi_setup.py`` scores one larger matrix on an instance.

For every geometry (pe_dir, pe_polarity, phase_fov) the bandwidths are walked upward and the first one is kept that
meets the four requirements which do not depend on the averages. They are evaluated with closed forms on the hidden
world, each checked against the engine in ``tests/test_epi_setup.py``:

    displacement of a region   |off-resonance averaged over its material| x echo spacing x phase field of view
    ghost of a region          mean of |sin(2 pi x bandwidth x matrix x delay x readout coordinate / fov)| over its pixels
    wrap                       every point of the object (density above 5 % of the maximum), displaced by its own
                               off-resonance, lies ``wrap_mm`` inside the phase field of view, which is centred on it
    fat on a region            a fat disc displaced by each strong line of the fat spectrum keeps ``fat_mm`` from the
                               region's circle

The lowest admissible bandwidth of a geometry has its least noise. The SNR of each such candidate is predicted from one
noise-free scan (region signal carried along the echo time with the region's T2*, corrected for the cos factor of the
gradient delay) and the analytic noise; the candidates are then scored by the verifier's own measures (``metrics``),
best prediction first, and the averages of each are counted on its measured SNR. A candidate is skipped only when even
``MODEL_TOL`` more SNR than predicted would not beat the best protocol found; a candidate the closed forms accept and
the measures reject is dropped. So ``least`` never returns a protocol that fails, and misses a better one only if the
SNR prediction is off by more than ``MODEL_TOL``.

Margins. ``LOOSE`` (the limits as stated, 1 % inside so that the closed forms and the measures agree on a candidate)
gives the optimum the verifier's time check uses. ``STRICT`` (every limit
tightened) is what a careful measurement can reach; the instance generator keeps a world only when both need the same
number of averages, so no instance hinges on a boundary.
Cost: one plan per bandwidth, a few hundred closed-form evaluations, and 10 to 30 noise-free scans; 1 to 4 s.
"""
from __future__ import annotations

import ast
import functools
import math

import numpy as np
import scipy.spatial

from ...scanner import console, kspace, tissue
from ...scanner import world as W
from .. import oracle as O
from . import metrics as M

SCAN = dict(sequence="epi", fov_mm=240.0, slice_mm=5.0, tr_ms=2000.0, fat_sat=False)     # fixed; flip angle, partial Fourier, echo spacing at their defaults
FIELDS = ("pe_dir", "pe_polarity", "bandwidth_hz_per_px", "matrix", "phase_fov", "centre_pe_mm", "te_ms", "averages")
BW = np.round(500.0 * 1.015 ** np.arange(121), 1)                       # 500 .. 2985 Hz/px
PHASE_FOV_MIN = 0.5
AVERAGES_MAX = 64
WRAP_LIMIT = 2e-3                                                       # energy of what wrap-around changes / energy of the image
FAT_LIMIT = 0.05                                                        # what displaced fat may change in a marked region
OBJECT_THRESHOLD = 0.05
# what the strict search asks beyond the stated limits: displacement and ghost as shares of their limits, clearances in mm, SNR
STRICT = dict(displacement=1 / 1.05, ghost=1 / 1.15, wrap_mm=5.5, fat_mm=6.0, snr=1.05)
MODEL_TOL = 1.03                                                        # the SNR model of the search is trusted to 3 %
FAT_LINE_MIN = 0.05                                                     # lines of the fat spectrum with at least this share are placed
LOOSE = dict(displacement=1 / 1.01, ghost=1 / 1.01, wrap_mm=0.5, fat_mm=1.0, snr=1.0)      # the stated limits, a hair inside: the closed forms are good to 1 %
IGNORE = dict(ghost=float("inf"), fat=float("-inf"))                    # a limit that is not looked at (the generator's traps)
REGRET_KIND = "difference"          # the value is a number of averages: the regret of an answer is its averages minus the fewest


def protocol(fields: dict) -> dict:
    """The scan a deliverable stands for: the fixed fields plus the delivered ones. ``centre_pe_mm`` is the centre of the
    field of view along the phase-encode axis (the console's ``centre_y_mm`` or ``centre_x_mm``); along the readout
    the field of view stays on the isocentre."""
    out = dict(SCAN, **{k: fields[k] for k in FIELDS if k in fields and k != "centre_pe_mm"})
    if "centre_pe_mm" in fields:
        out["centre_y_mm" if fields.get("pe_dir") == "y" else "centre_x_mm"] = fields["centre_pe_mm"]
    return out


def centre_pe(rp: dict) -> float:
    """Centre of the field of view of a (realised) protocol along its phase-encode axis, mm."""
    return float(rp.get("centre_y_mm", 0.0) if rp["pe_dir"] == "y" else rp.get("centre_x_mm", 0.0))


def matrix_of(pixel_max_mm: float) -> int:
    """The smallest even matrix whose pixel is not larger than the ceiling."""
    return int(math.ceil(SCAN["fov_mm"] / pixel_max_mm / 2.0 - 1e-9) * 2)


def phase_fovs(matrix: int) -> list:
    """The phase fields of view the console realises for a matrix (two lines apart), from ``PHASE_FOV_MIN`` to 1."""
    return [m / matrix for m in range(int(math.ceil(PHASE_FOV_MIN * matrix / 2.0)) * 2, matrix + 1, 2)]


# ------------------------------------------------------------------------------------------------ the hidden world
def facts(world: dict, task: dict) -> dict:
    """What the closed forms need from the hidden world: per marked region the off-resonance averaged over its material
    and its T2*, the gradient delay, the points of the object with their off-resonance, the fat vials and fat lines."""
    df = world["fields"]["b0_hz"] + world["system"]["cf_offset_hz"]
    by_id = {r["id"]: r for r in world["regions"]}
    regions = []
    for r in task["regions"]:
        cls = by_id[r["id"]]["cls"]
        wt = world["weights"][world["names"].index(cls)]
        regions.append(dict(r, cls=cls, df_hz=float((df * wt).sum() / wt.sum()), t2s_ms=world["tissues"][cls]["t2s_ms"]))
    pd = np.array([world["tissues"][k]["pd"] for k in world["names"]], np.float32)
    density = np.tensordot(pd, world["weights"], 1)
    body = density > OBJECT_THRESHOLD * density.max()
    n, d = world["n"], world["extent_mm"] / world["n"]
    ax = (np.arange(n) - n // 2) * d
    iy, ix = np.nonzero(body)
    f0 = tissue.FAT_PPM * kspace.GAMMA_MHZ * world["b0_t"] / tissue.FAT_PPM          # Hz per ppm
    fat = [_disc(world, name, ax) for name in world["names"] if world["tissues"][name]["fat"]]
    return dict(regions=regions, delay_s=world["system"]["grad_delay_us"] * 1e-6,
                body=dict(y=_hull(ax[iy], df[iy, ix]), x=_hull(ax[ix], df[iy, ix]), half_px=d / 2, df_max=float(np.abs(df[iy, ix]).max())), coords={},
                fat=fat, fat_hz=[ppm * f0 for ppm, amp in tissue.FAT_PEAKS if amp >= FAT_LINE_MIN])


def _hull(pos, df) -> dict:
    """The points (position along an axis, off-resonance) of the object that can be its ends along that axis after any
    displacement proportional to the off-resonance: the corners of their convex hull, dict(p=, df=)."""
    pts = np.stack([np.asarray(pos, float), np.asarray(df, float)], 1)
    if np.ptp(pts[:, 1]) < 1e-9:                                         # a flat field: the two ends
        keep = [int(np.argmin(pts[:, 0])), int(np.argmax(pts[:, 0]))]
    else:
        keep = scipy.spatial.ConvexHull(pts).vertices
    return dict(p=pts[keep, 0], df=pts[keep, 1])


def scaled(f: dict, field: float = 1.0, delay_s: float | None = None) -> dict:
    """``facts`` of the same world with its off-resonance field multiplied by ``field`` and another gradient delay
    (both enter the world linearly; the instance generator sets them)."""
    body = dict(f["body"], df_max=f["body"]["df_max"] * field, **{k: dict(p=f["body"][k]["p"], df=f["body"][k]["df"] * field) for k in ("y", "x")})
    return dict(f, delay_s=f["delay_s"] if delay_s is None else delay_s, body=body, regions=[dict(r, df_hz=r["df_hz"] * field) for r in f["regions"]])


def _disc(world: dict, cls: str, ax) -> dict:
    """Centre and radius (from its area) of the disc of one material on the master grid."""
    wt = world["weights"][world["names"].index(cls)].astype(float)
    d = world["extent_mm"] / world["n"]
    return dict(y_mm=float((wt.sum(1) * ax).sum() / wt.sum()), x_mm=float((wt.sum(0) * ax).sum() / wt.sum()), radius_mm=math.sqrt(float(wt.sum()) * d * d / math.pi))


# ------------------------------------------------------------------------------------------------ closed forms
def displacement_mm(df_hz, rp: dict):
    """Signed displacement along the phase-encode axis of a spin ``df_hz`` off resonance (the engine's sign convention)."""
    return -rp["pe_polarity"] * np.asarray(df_hz) * rp["echo_spacing_ms"] * 1e-3 * rp["fov_pe_mm"]


def ghost_of(region: dict, delay_s: float, rp: dict) -> float:
    """Ghost of a region: mean of |sin(phi)| over the pixels of its circle, phi the phase the delay gives a spin at
    their readout coordinate."""
    m = M.mask(rp, region)
    y, x = M.core.pixel_positions(m.shape, M.core.fov_yx(rp), M.core.centre_yx(rp))
    c = (x[None, :] if rp["pe_dir"] == "y" else y[:, None]) + np.zeros(m.shape)
    return float(np.abs(np.sin(ghost_phase(c[m], delay_s, rp))).mean())


def _ghost_fast(f: dict, k: int, rp: dict) -> float:
    """``ghost_of`` for region ``k`` of ``f`` with the pixels of the circle taken for a field of view on the isocentre
    (kept per axis and matrix in ``f["coords"]``): the readout coordinates of the pixels do not move with the centre
    along the phase-encode axis, only which rows the circle takes changes a little. Good to 1 % of the ghost."""
    key = (k, rp["pe_dir"], rp["matrix"], rp["matrix_pe"])
    if key not in f["coords"]:
        on = dict(rp, centre_y_mm=0.0, centre_x_mm=0.0)
        m = M.mask(on, f["regions"][k])
        y, x = M.core.pixel_positions(m.shape, M.core.fov_yx(on), M.core.centre_yx(on))
        f["coords"][key] = ((x[None, :] if rp["pe_dir"] == "y" else y[:, None]) + np.zeros(m.shape))[m]
    return float(np.abs(np.sin(ghost_phase(f["coords"][key], f["delay_s"], rp))).mean())


def ghost_phase(readout_mm, delay_s: float, rp: dict):
    """Phase (rad) that a gradient delay gives a spin at a readout coordinate: odd and even lines differ by twice this."""
    return 2 * np.pi * rp["bandwidth_hz_per_px"] * rp["matrix"] * delay_s * np.asarray(readout_mm) / rp["fov_mm"]


def displaced_extent(f: dict, rp: dict) -> tuple:
    """(low, high) end of the object along the phase-encode axis, every point displaced by its own off-resonance, mm."""
    b = f["body"]["y" if rp["pe_dir"] == "y" else "x"]
    pos = b["p"] + displacement_mm(b["df"], rp)
    return float(pos.min() - f["body"]["half_px"]), float(pos.max() + f["body"]["half_px"])


def best_centre_mm(f: dict, rp: dict) -> float:
    """The centre of the field of view along the phase-encode axis that leaves the displaced object the same room on
    both sides, on a raster of 0.1 mm. Pixel i of an n-pixel axis is centred at centre + (i - n // 2) x pixel."""
    lo, hi = displaced_extent(f, rp)
    return round(0.5 * (lo + hi) + 0.5 * rp["fov_pe_mm"] / rp["matrix_pe"], 1)


def wrap_clearance_mm(f: dict, rp: dict) -> float:
    """How far the displaced object stays inside the phase field of view (mm; negative: it wraps)."""
    lo, hi = displaced_extent(f, rp)
    c, px = centre_pe(rp), rp["fov_pe_mm"] / rp["matrix_pe"]
    return float(min(c + rp["fov_pe_mm"] / 2 - px / 2 - hi, lo - (c - rp["fov_pe_mm"] / 2 - px / 2)))


def fat_clearance_mm(f: dict, rp: dict) -> float:
    """Smallest gap (mm; negative: overlap) between a fat disc, displaced by one of its strong lines, and the circle of
    a marked region. The image is periodic along the phase axis with the phase field of view."""
    axis = "y_mm" if rp["pe_dir"] == "y" else "x_mm"
    other = "x_mm" if axis == "y_mm" else "y_mm"
    out = float("inf")
    for v in f["fat"]:
        for hz in f["fat_hz"]:
            pos = v[axis] + float(displacement_mm(hz, rp))
            for r in f["regions"]:
                along = (pos - r[axis] + rp["fov_pe_mm"] / 2) % rp["fov_pe_mm"] - rp["fov_pe_mm"] / 2
                out = min(out, math.hypot(along, v[other] - r[other]) - v["radius_mm"] - r["radius_mm"])
    return out


# ------------------------------------------------------------------------------------------------ the search
def _train(exam: dict, matrix: int) -> tuple:
    """(bandwidth, echo spacing, what the echo time needs beyond the lines before the echo) at every bandwidth of ``BW``
    that the console accepts for this matrix, from one plan each (a function of the console card and the matrix: kept)."""
    return _train_of(matrix, tuple(sorted((k, repr(v)) for k, v in exam.items())))


@functools.lru_cache(maxsize=32)
def _train_of(matrix: int, card: tuple) -> tuple:
    exam = {k: ast.literal_eval(v) for k, v in card}
    out = []
    for bw in BW:
        pl = console.plan(dict(SCAN, matrix=matrix, bandwidth_hz_per_px=float(bw)), exam)
        if pl["accept"]:
            rp = pl["realised"]
            out.append((float(bw), rp["echo_spacing_ms"], pl["timing"]["te_min_ms"] - (rp["lines_full"] // 2) * rp["echo_spacing_ms"]))
    return tuple(out)


def geometries(world: dict, task: dict, how: dict, f: dict | None = None) -> list:
    """For every geometry of the search the lowest bandwidth that meets displacement, ghost, fat and wrap under the
    margins ``how`` (the lowest bandwidth has the least noise): dicts of the fields a closed form needs (no plan)."""
    f, t = f or facts(world, task), task
    n = matrix_of(t["pixel_max_mm"])
    train = _train(world["console"], n)
    bw, esp, te0 = (np.array(v) for v in zip(*train))                    # the echo spacing falls with the bandwidth
    df_max = max(abs(r["df_hz"]) for r in f["regions"])
    out = []
    for pe_dir in ("y", "x"):
        for pol in ((1, -1) if f["fat"] else (1,)):
            for pf in phase_fovs(n):
                lines = int(round(n * pf))
                fov_pe = SCAN["fov_mm"] * lines / n
                ok = te0 + (lines // 2) * esp <= t["te_window_ms"][1] + 1e-9             # the echo train fits the window
                ok &= df_max * esp * 1e-3 * fov_pe <= how["displacement"] * t["displacement_max_mm"]
                idx = np.flatnonzero(ok)
                def at(k, pe_dir=pe_dir, pol=pol, lines=lines, fov_pe=fov_pe):
                    rp = dict(pe_dir=pe_dir, pe_polarity=pol, matrix=n, matrix_pe=lines, fov_mm=SCAN["fov_mm"], fov_pe_mm=fov_pe, phase_fov=lines / n,
                              bandwidth_hz_per_px=float(bw[k]), echo_spacing_ms=float(esp[k]), centre_y_mm=0.0, centre_x_mm=0.0,
                              te_ms=max(float(te0[k] + (lines // 2) * esp[k]), t["te_window_ms"][0]))
                    rp["centre_y_mm" if pe_dir == "y" else "centre_x_mm"] = best_centre_mm(f, rp)
                    return rp
                if not len(idx) or wrap_clearance_mm(f, at(idx[-1])) < how["wrap_mm"]:
                    continue                                             # even the shortest echo spacing leaves the object wrapped
                lo, hi = 0, len(idx) - 1                                 # the displaced object shrinks back with the bandwidth: bisect
                while lo < hi:
                    mid = (lo + hi) // 2
                    lo, hi = (lo, mid) if wrap_clearance_mm(f, at(idx[mid])) >= how["wrap_mm"] else (mid + 1, hi)
                for k in idx[lo:]:
                    rp = at(k)
                    if max(_ghost_fast(f, i, rp) for i in range(len(f["regions"]))) > how["ghost"] * t["ghost_max"]:
                        break                                            # the ghost grows with the bandwidth: nothing above passes
                    if f["fat"] and fat_clearance_mm(f, rp) < how["fat_mm"]:
                        continue
                    out.append(rp)
                    break
    return out


def candidates(world: dict, task: dict, how: dict, f: dict | None = None) -> list:
    """``geometries`` as the console realises them: realised protocols with one average, at the shortest echo time
    inside the window."""
    exam, t, out = world["console"], task, []
    for g in geometries(world, task, how, f):
        req = dict(SCAN, matrix=g["matrix"], pe_dir=g["pe_dir"], pe_polarity=g["pe_polarity"], phase_fov=g["phase_fov"], bandwidth_hz_per_px=g["bandwidth_hz_per_px"],
                   centre_y_mm=g["centre_y_mm"], centre_x_mm=g["centre_x_mm"])
        pl = console.plan(req, exam)
        if pl["accept"]:
            pl = console.plan(dict(req, te_ms=max(pl["timing"]["te_min_ms"], t["te_window_ms"][0])), exam)
        if pl["accept"] and pl["realised"]["te_ms"] <= t["te_window_ms"][1]:
            out.append(pl["realised"])
    return out


def signal_model(scans: M.Scans, task: dict, f: dict, rp0: dict):
    """One noise-free scan of ``rp0``; returns a function that predicts the mean magnitude of the weakest marked region
    for any protocol of the same matrix: the scanned value carried along the echo time with the region's T2* and
    corrected for the cos factor the gradient delay puts on the image."""
    img = np.abs(scans.image(rp0))
    ref = {r["id"]: float(img[M.mask(rp0, r)].mean()) for r in f["regions"]}

    def cos_factor(r, rp):
        m = M.mask(rp, r)
        y, x = M.core.pixel_positions(m.shape, M.core.fov_yx(rp), M.core.centre_yx(rp))
        c = (x[None, :] if rp["pe_dir"] == "y" else y[:, None]) + np.zeros(m.shape)
        return float(np.abs(np.cos(ghost_phase(c[m], f["delay_s"], rp))).mean())
    base = {r["id"]: cos_factor(r, rp0) for r in f["regions"]}
    return lambda rp: min(ref[r["id"]] * math.exp(-(rp["te_ms"] - rp0["te_ms"]) / r["t2s_ms"]) * cos_factor(r, rp) / base[r["id"]] for r in f["regions"])


def relative_snr(f: dict, g: dict) -> float:
    """SNR of a geometry of ``geometries`` up to a factor common to all of them: the square root of the acquisition time
    of its lines, times the decay of the faster-decaying region at its echo time."""
    return math.sqrt(g["matrix_pe"] / g["bandwidth_hz_per_px"]) * math.exp(-g["te_ms"] / min(r["t2s_ms"] for r in f["regions"]))


def averages_for(snr1: float, snr_min: float) -> int:
    return int(math.ceil((snr_min / snr1) ** 2 - 1e-9))


def ladder(exam: dict, rp: dict, te_window) -> list:
    """The geometry of ``rp`` at every bandwidth of ``BW`` the console accepts and whose echo train fits the echo-time
    window, lowest first (realised protocols at the shortest echo time inside the window)."""
    out = []
    for bw in BW:
        req = dict(SCAN, matrix=rp["matrix"], pe_dir=rp["pe_dir"], pe_polarity=rp["pe_polarity"], phase_fov=rp["matrix_pe"] / rp["matrix"], bandwidth_hz_per_px=float(bw),
                   centre_y_mm=rp["centre_y_mm"], centre_x_mm=rp["centre_x_mm"])
        pl = console.plan(req, exam)
        if pl["accept"] and pl["timing"]["te_min_ms"] <= te_window[1]:
            out.append(console.plan(dict(req, te_ms=max(pl["timing"]["te_min_ms"], te_window[0])), exam)["realised"])
    return out


def search(world: dict, task: dict, scans: M.Scans | None = None, strict: bool = False, f: dict | None = None, model=None, how: dict | None = None) -> dict:
    """The candidates with their predicted one-average SNR, best first: dict(snr1=, candidates=[dict(snr1, realised)]).
    ``strict``: under the margins of ``STRICT`` (the SNR is then the predicted one divided by its margin); ``how``: other margins."""
    how = how or (STRICT if strict else LOOSE)
    scans = scans or M.Scans(world)
    f = f or facts(world, task)
    found = candidates(world, task, how, f)
    if not found:
        return dict(snr1=None, candidates=[], model=model)
    model = model or signal_model(scans, task, f, found[0])
    noise = scans.engine().noise_sd
    rows = sorted((dict(snr1=model(rp) / noise(dict(rp, averages=1)) / how["snr"], realised=rp) for rp in found), key=lambda r: -r["snr1"])
    return dict(snr1=rows[0]["snr1"], candidates=rows, model=model)


def fields_of(rp: dict, averages: int) -> dict:
    return dict(pe_dir=rp["pe_dir"], pe_polarity=rp["pe_polarity"], bandwidth_hz_per_px=rp["bandwidth_hz_per_px"], matrix=rp["matrix"],
                phase_fov=rp["matrix_pe"] / rp["matrix"], centre_pe_mm=centre_pe(rp), te_ms=rp["te_ms"], averages=int(averages))


def evaluate(task: dict, fields: dict, world: dict, scans: M.Scans) -> dict:
    """Verifier side: every measure of one delivered protocol on the hidden world (``metrics``), without the verdicts."""
    pl = console.plan(protocol(fields), world["console"])
    rp = pl["realised"]
    by_id = {r["id"]: r for r in world["regions"]}
    out = dict(scan_s=pl["scan_s"], realised=rp, te_ms=rp["te_ms"], pixel_mm=rp["fov_mm"] / rp["matrix"], wrap=M.wrap(scans, rp),
               displacement_mm={}, ghost={}, snr={}, fat={})
    has_fat = any(c["fat"] for c in world["tissues"].values())
    for r in task["regions"]:
        out["displacement_mm"][r["id"]] = abs(M.displacement_mm(scans, rp, by_id[r["id"]]["cls"]))
        out["ghost"][r["id"]] = M.ghost(scans, rp, r)
        out["snr"][r["id"]] = M.snr(scans, rp, r)
        out["fat"][r["id"]] = M.fat(scans, rp, r) if has_fat else 0.0
    return out


def failures(task: dict, m: dict) -> list:
    """The requirements a measured protocol misses, in the order of the task text (without the scan time)."""
    bad = []
    if not task["te_window_ms"][0] - 1e-9 <= m["te_ms"] <= task["te_window_ms"][1] + 1e-9:
        bad.append("echo_time")
    if m["pixel_mm"] > task["pixel_max_mm"] + 1e-9:
        bad.append("pixel")
    if max(m["displacement_mm"].values()) > task["displacement_max_mm"]:
        bad.append("displacement")
    if max(m["ghost"].values()) > task["ghost_max"]:
        bad.append("ghost")
    if m["wrap"] > WRAP_LIMIT:
        bad.append("wrap")
    if max(m["fat"].values()) > FAT_LIMIT:
        bad.append("fat")
    if min(m["snr"].values()) < task["snr_min"]:
        bad.append("snr")
    return bad


def least(cfg: dict, world: dict, scans: M.Scans | None = None) -> dict:
    """The fewest averages of an instance and a protocol that reaches them and passes the verifier's own measures:
    dict(averages, fields, measured, rejected). Candidates are tried in the order of their predicted SNR; the averages of
    each are the fewest with which the measured SNR reaches the floor. A candidate is skipped only when even
    ``MODEL_TOL`` more SNR than predicted would not need fewer averages than the best protocol found so far.
    ``rejected``: candidates the closed forms accepted and the measures did not."""
    t = cfg["task"]
    scans = scans or M.Scans(world)
    best, rejected = None, []
    for row in search(world, t, scans)["candidates"]:
        hope = averages_for(row["snr1"] * MODEL_TOL, t["snr_min"])
        if hope > AVERAGES_MAX or (best is not None and hope >= best["averages"]):
            continue
        fields = fields_of(row["realised"], max(hope, 1))
        m = evaluate(t, fields, world, scans)
        n = max(averages_for(min(m["snr"].values()) * math.sqrt(1.0 / fields["averages"]), t["snr_min"]), 1)
        if n != fields["averages"]:                                      # the measured SNR decides the averages
            fields = dict(fields, averages=n)
            m = evaluate(t, fields, world, scans)
        bad = failures(t, m)
        if bad or n > AVERAGES_MAX:
            rejected.append(dict(fields=fields, fails=bad or ["averages"]))
        elif best is None or n < best["averages"]:
            best = dict(averages=n, fields=fields, measured=m)
    return dict(best or dict(averages=None, fields=None, measured=None), rejected=rejected)


@O.timed
def solve(cfg: dict, world: dict | None = None) -> dict:
    """The optimum of an instance (the contract is in ``mrgym.envs.oracle``): ``value`` is the fewest averages,
    ``argmin`` a deliverable that reaches them; ``certificate``: that deliverable with one average less (when it has
    two or more), with the bandwidth halved and with the console's default matrix, each with the check it fails."""
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    scans = M.Scans(world)
    t = cfg["task"]
    best = least(cfg, world, scans)
    certificate = []
    if best["fields"]:
        tries = [dict(best["fields"], bandwidth_hz_per_px=max(best["fields"]["bandwidth_hz_per_px"] / 2.0, 500.0), te_ms=t["te_window_ms"][1])]
        if best["averages"] >= 2:
            tries.append(dict(best["fields"], averages=best["averages"] - 1))
        for fields in tries:
            pl = console.plan(protocol(fields), world["console"])
            bad = failures(t, evaluate(t, fields, world, scans)) if pl["accept"] else ["deliverable_valid"]
            if bad:
                certificate.append(dict(deliverable=fields, fails=bad[0]))
    n = matrix_of(t["pixel_max_mm"])
    size = 2 * (2 if any(c["fat"] for c in world["tissues"].values()) else 1) * len(phase_fovs(n)) * len(BW)
    return dict(value=None if best["averages"] is None else float(best["averages"]), argmin=best["fields"], method="exhaustive",
                grid=dict(axes=dict(pe_dir=["y", "x"], pe_polarity=[1, -1], phase_fov=dict(start=PHASE_FOV_MIN, stop=1.0, points=len(phase_fovs(n))),
                                    bandwidth_hz_per_px=dict(start=float(BW[0]), stop=float(BW[-1]), points=len(BW)), matrix=n,
                                    te_ms="the console minimum inside the window", averages="the fewest that reach the SNR floor"),
                          size=size, evaluated=size),
                certificate=certificate, rejected=best["rejected"], measured=best["measured"])

