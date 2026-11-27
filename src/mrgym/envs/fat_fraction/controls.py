"""Control policies of ``Scanner/FatFraction``. Every policy plays through the agent's interface only (``info`` /
``plan`` / ``scan`` / ``budget``) and returns the deliverable. None reads the hidden world.

The signal model of the policies that fit, per disc, at echo time t (ms) after the excitation:

    s(t) = (A + B c(t)) exp(-R t) exp(2 pi i psi t)        A, B complex; c(t) = ``task.spectrum(t)``, the six peaks

with one decay rate R for water and fat and the off-resonance psi of the disc. ``fit_complex`` finds psi within the
told bound (so that water cannot be taken for fat, which sits 200 Hz away) and R on a grid, A and B by linear least
squares, and returns |B| / (|A| + |B|). It uses the complex mean of a region (no noise floor) and does not assume
which of the two is larger.

Must pass
  reference      level 1: the stated method, to the letter (``stated``). Otherwise: the stated scan (six echoes, flip
                 angle 3 degrees) and ``fit_complex``. One fixed scan and one fixed estimator
Must fail
  prior          no scan; the middle of the told range in every disc
  fixed_recipe   one token scan, then one constant fraction for every disc, tuned on other worlds
  two_point      the stated scan; the textbook formula (IP - OP) / (2 IP) on the first two echoes: single-peak fat, no decay
  two_point_high_flip   the same formula at a flip angle of 20 degrees, where the T1 weighting offsets part of the other
                 two errors: the cancellation depends on the hidden T1 and T2* of each disc
  one_image      the opposed-phase echo alone: the fraction from how dark a disc is against the brightest one
  brute_force    eight echoes; the fraction from the depth of the beat of the magnitude, (max - min) / (max + min). No model
  wrong_units    the reference's fraction as a number between 0 and 1
Ablations (the reference with one step removed; required where the step decides)
  high_flip      the reference at a flip angle of 20 degrees (T1 weights the echoes): every level
  single_peak    the reference's fit with one fat line at -3.4 ppm: every level
  magnitude_only the stated method (magnitudes, fat taken as the smaller part) where fat may be the larger part: level 3
Reported
  open_loop      the reference: it is a fixed scan with a fixed estimator at every level
  no_decay       the reference's fit with R = 0
"""
from __future__ import annotations

import math

import numpy as np

from . import task

REFERENCE = "reference"
MUST_FAIL = ("prior", "fixed_recipe", "two_point", "two_point_high_flip", "one_image", "brute_force", "wrong_units")
ABLATIONS = {"high_flip": (1, 2, 3), "single_peak": (1, 2, 3), "magnitude_only": (3,)}
REPORTED = ("open_loop", "no_decay")
DESCRIPTIONS = dict(
    reference="six echoes at a flip angle of 3 degrees; level 1: the stated arithmetic; else a fit of water, fat (six peaks), decay and off-resonance to the complex echoes",
    prior="no scan; the middle of the told range in every disc",
    fixed_recipe="one token scan, then one constant fraction tuned on other worlds",
    two_point="the textbook formula (IP - OP) / (2 IP) on two echoes at a flip angle of 3 degrees",
    two_point_high_flip="the textbook formula at a flip angle of 20 degrees, where T1 offsets part of its error",
    one_image="the opposed-phase echo alone, read against the brightest disc",
    brute_force="eight echoes; the depth of the beat of the magnitude, no model",
    wrong_units="the reference's fraction as a number between 0 and 1",
    high_flip="the reference at a flip angle of 20 degrees",
    single_peak="the reference's fit with one fat line at -3.4 ppm",
    magnitude_only="the stated method (magnitudes; fat taken as the smaller part)",
    open_loop="the reference (one fixed scan, one fixed estimator)",
    no_decay="the reference's fit without the decay between the echoes",
)
SCAN = dict(task.METHOD)
HIGH_FLIP_DEG = 20.0
PSI_MARGIN_HZ = 20.0                        # the fit searches the off-resonance this far beyond the told bound
R_MAX_PER_MS = 0.15                         # largest decay rate of the fit (T2* 6.7 ms)


def _task(sc):
    return sc.info()["task"]


def _ids(t):
    return [r["id"] for r in t["regions"]]


def _read(sc, t, protocol):
    """One scan: (echo times in ms, {disc: mean magnitudes}, {disc: complex means}), one value per echo."""
    r = sc.scan(protocol, rois=t["regions"])
    rows = {row["id"]: row for row in r["roi_readouts"]}
    mag = {i: np.array(rows[i]["mean_magnitude"], float) for i in _ids(t)}
    cplx = {i: np.array(rows[i]["complex_mean_magnitude"], float) * np.exp(1j * np.array(rows[i]["complex_mean_phase_rad"], float)) for i in _ids(t)}
    return np.array(r["realised"]["echo_times_ms"], float), mag, cplx


def stated_pdff(te, m) -> float:
    """The arithmetic of level 1 on six magnitudes (opposed, in, opposed, in, ...): percent."""
    c1, c2, c4 = (float(task.spectrum(te[k]).real) for k in (0, 1, 3))
    rate = math.log(m[1] / m[3]) / (te[3] - te[1])                        # step 3: decay from two in-phase echoes, first guess
    for again in (True, False):
        in_phase = m[1] * math.exp(rate * (te[1] - te[0]))                # step 4: the in-phase signal at the first echo time
        fat = (in_phase - m[0]) / (c2 - c1)
        water = in_phase - c2 * fat
        if again:                                                         # step 5: the fat vector shrinks between t2 and t4; take that out
            rate = math.log(m[1] * (water + c4 * fat) / (m[3] * (water + c2 * fat))) / (te[3] - te[1])
    return float(np.clip(100.0 * fat / (water + fat), 0.0, 100.0))


def two_point_pdff(m) -> float:
    """The textbook formula on an opposed-phase and an in-phase magnitude: percent."""
    return float(np.clip(100.0 * (m[1] - m[0]) / (2.0 * m[1]), 0.0, 100.0))


def fit_complex(te, d, psi_max_hz, single_peak=False, decay=True) -> dict:
    """Least squares of (A + B c(t)) exp(-R t) exp(2 pi i psi t) to the complex echoes ``d`` at ``te`` (ms).
    Returns dict(pdff percent, psi_hz, rate_per_ms)."""
    te, d = np.asarray(te, float), np.asarray(d, complex)
    c = np.exp(2j * np.pi * (-3.4 * task.GAMMA_MHZ_PER_T * task.B0_T) * te * 1e-3) if single_peak else task.spectrum(te)

    def search(psi, rate):
        """Residual and (A, B) on the grid psi x rate."""
        e2 = np.exp(-2.0 * rate[:, None] * te[None, :])                                  # [rate, echo]
        uu, vv, uv = e2.sum(1), (e2 * np.abs(c) ** 2).sum(1), (e2 * c).sum(1)             # the Gram matrix does not depend on psi
        w = np.exp(-rate[None, :, None] * te - 2j * np.pi * psi[:, None, None] * te * 1e-3) * d   # conj(u) d, [psi, rate, echo]
        ud, vd = w.sum(2), (w * np.conj(c)).sum(2)
        det = uu * vv - np.abs(uv) ** 2
        a = (vv * ud - uv * vd) / det
        b = (uu * vd - np.conj(uv) * ud) / det
        res = np.sum(np.abs(d) ** 2) - (np.conj(a) * ud + np.conj(b) * vd).real
        k = np.unravel_index(int(np.argmin(res)), res.shape)
        return float(psi[k[0]]), float(rate[k[1]]), a[k], b[k]

    rates = np.linspace(0.0, R_MAX_PER_MS, 31) if decay else np.zeros(1)
    psi, rate, a, b = search(np.arange(-psi_max_hz, psi_max_hz + 1e-9, 2.0), rates)
    for step_psi, step_rate in ((0.2, 0.0005), (0.02, 0.00005)):
        fine = np.clip(rate + step_rate * np.arange(-10, 11), 0.0, R_MAX_PER_MS) if decay else np.zeros(1)
        psi, rate, a, b = search(psi + step_psi * np.arange(-10, 11), fine)
    return dict(pdff=float(100.0 * abs(b) / (abs(a) + abs(b))), psi_hz=psi, rate_per_ms=rate)


def _fit(sc, flip_deg=SCAN["flip_deg"], **options):
    t = _task(sc)
    te, _, cplx = _read(sc, t, dict(SCAN, flip_deg=flip_deg))
    psi_max = t["off_resonance_max_hz"] + PSI_MARGIN_HZ
    return dict(pdff_percent={i: round(fit_complex(te, cplx[i], psi_max, **options)["pdff"], 2) for i in _ids(t)})


def stated(sc):
    t = _task(sc)
    te, mag, _ = _read(sc, t, SCAN)
    return dict(pdff_percent={i: round(stated_pdff(te, mag[i]), 2) for i in _ids(t)})


def reference(sc):
    return stated(sc) if "method" in _task(sc).get("hints", ()) else _fit(sc)


def open_loop(sc):
    return reference(sc)


def prior(sc):
    t = _task(sc)
    return dict(pdff_percent={i: round(0.5 * sum(t["pdff_range_percent"]), 1) for i in _ids(t)})


TOKEN_SCAN = dict(sequence="spgr", flip_deg=10.0, tr_ms=20.0, matrix=32, partial_fourier=0.6)


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        _read(sc, t, TOKEN_SCAN)
        return dict(pdff_percent={i: float(recipe["pdff_percent"]) for i in _ids(t)})
    return policy


DEFAULT_RECIPE = dict(pdff_percent=25.0)


def two_point(sc):
    t = _task(sc)
    _, mag, _ = _read(sc, t, SCAN)
    return dict(pdff_percent={i: round(two_point_pdff(mag[i]), 2) for i in _ids(t)})


def two_point_high_flip(sc):
    t = _task(sc)
    _, mag, _ = _read(sc, t, dict(SCAN, flip_deg=HIGH_FLIP_DEG))
    return dict(pdff_percent={i: round(two_point_pdff(mag[i]), 2) for i in _ids(t)})


def one_image(sc):
    t = _task(sc)
    te, mag, _ = _read(sc, t, dict(SCAN, n_echoes=1))
    top = max(float(m[0]) for m in mag.values())
    depth = 1.0 - float(task.spectrum(te[0]).real)                        # a disc of pure fat would read this much below pure water
    return dict(pdff_percent={i: round(float(np.clip(100.0 * (1.0 - m[0] / top) / depth, 0.0, 100.0)), 2) for i, m in mag.items()})


def brute_force(sc):
    t = _task(sc)
    _, mag, _ = _read(sc, t, dict(SCAN, n_echoes=8))
    out = {}
    for i, m in mag.items():
        depth = (m.max() - m.min()) / (m.max() + m.min())
        out[i] = round(float(np.clip(100.0 * depth / (1.0 + depth), 0.0, 100.0)), 2)
    return dict(pdff_percent=out)


def wrong_units(sc):
    return dict(pdff_percent={i: v / 100.0 for i, v in reference(sc)["pdff_percent"].items()})


def high_flip(sc):
    if "method" in _task(sc).get("hints", ()):
        t = _task(sc)
        te, mag, _ = _read(sc, t, dict(SCAN, flip_deg=HIGH_FLIP_DEG))
        return dict(pdff_percent={i: round(stated_pdff(te, mag[i]), 2) for i in _ids(t)})
    return _fit(sc, flip_deg=HIGH_FLIP_DEG)


def single_peak(sc):
    return _fit(sc, single_peak=True)


def magnitude_only(sc):
    return stated(sc)


def no_decay(sc):
    return _fit(sc, decay=False)


def policies(recipe=None):
    return dict(reference=reference, prior=prior, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE), two_point=two_point,
                two_point_high_flip=two_point_high_flip, one_image=one_image, brute_force=brute_force, wrong_units=wrong_units,
                high_flip=high_flip, single_peak=single_peak, magnitude_only=magnitude_only, open_loop=open_loop, no_decay=no_decay)


def tune(cfgs):
    """The strongest answer that ignores the object, on other worlds: the constant fraction with the highest share of
    discs inside the pass window, among the true fractions of those worlds and a grid over the told range."""
    truth = np.array([v for c in cfgs for v in task.true_pdff_percent(c).values()])
    lo, hi = cfgs[0]["task"]["pdff_range_percent"]
    cand = np.concatenate([truth, np.linspace(lo, hi, 181)])
    rate = [float(np.mean(np.abs(truth - v) <= task.TOLERANCE_POINTS)) for v in cand]
    best = int(np.argmax(rate))
    return dict(pdff_percent=round(float(cand[best]), 2), tuned_region_pass_rate=round(rate[best], 3))
