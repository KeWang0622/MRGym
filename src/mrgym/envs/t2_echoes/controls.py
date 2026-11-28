"""Control policies of ``Scanner/T2Echoes``. Every policy plays through the agent's interface only (``info`` / ``plan``
/ ``scan`` / ``budget``) and returns the deliverable. None reads the hidden world, and none calls the signal core: the
model fit is ``model.train``, the recursion that level 1 prints as a text for ``calc``.

The echoes of a region are ``|complex_mean|`` of the images of one ``mese`` scan. Every policy that fits keeps the
echoes from the first down to the last that is at least ``min_share`` of the largest (at least ``min_echoes``): below
that a short-T2 vial shows what the surrounding gel leaks into the region.

Must pass
  reference      one train with the ``sharp`` pulse; per region the echo-train model fitted with amplitude, T2 and
                 the refocusing angle free
Must fail
  prior          no scan; the geometric mean of the told T2 range in every region
  fixed_recipe   one token scan, then one constant T2 for every region, tuned on other worlds
  mono_exp       one train with the console's default pulse; T2 from a straight line through ln(signal) of the kept echoes
  single_echoes  the reference of ``Scanner/T2Decay`` (a series of single-echo spin echoes): this budget pays one scan of it
  wrong_units    the reference's T2 in seconds (refused by the console of the task: not a T2 in milliseconds)
Ablations (the reference with one step removed; required where the step decides)
  line_fit       the reference's train, a straight line through ln(signal) instead of the model: every level
Reported
  open_loop      the reference: it is one scan with a fixed estimator at every level
  drop_first     the reference's train, a straight line through ln(signal) without the first echo
  even_echoes    the reference's train, a straight line through ln(signal) of echoes 2, 4, 6, ...
  default_pulse  the reference's fit on a train with the console's default pulse (the angle falls off across more of the slice)
"""
from __future__ import annotations

import math

import numpy as np
import scipy.optimize

from ..t2_decay import controls as single_echo
from . import model, task

REFERENCE = "reference"
MUST_FAIL = ("prior", "fixed_recipe", "mono_exp", "single_echoes", "wrong_units")
ABLATIONS = {"line_fit": (1, 2, 3)}
REPORTED = ("open_loop", "drop_first", "even_echoes", "default_pulse")
DESCRIPTIONS = dict(
    reference="one echo train with the sharp pulse; the echo-train model fitted per region with amplitude, T2 and refocusing angle free",
    prior="no scan; the geometric mean of the told T2 range in every region",
    fixed_recipe="one token scan, then one constant T2 tuned on other worlds",
    mono_exp="one echo train with the default pulse; a straight line through ln(signal) of the echoes",
    single_echoes="the reference of Scanner/T2Decay: a series of single-echo spin echoes, which the budget does not pay",
    wrong_units="the reference's T2 in seconds",
    line_fit="the reference's train; a straight line through ln(signal) instead of the echo-train model",
    open_loop="the reference (one scan, a fixed estimator)",
    drop_first="the reference's train; a straight line through ln(signal) without the first echo",
    even_echoes="the reference's train; a straight line through ln(signal) of the even echoes",
    default_pulse="the reference's fit on a train with the console's default pulse",
)
M = task.METHOD
TOKEN = dict(sequence="mese", n_echoes=2, tr_ms=300.0, matrix=32, partial_fourier=0.6)


def _task(sc):
    return sc.info()["task"]


def _ids(t):
    return [r["id"] for r in t["regions"]]


def _protocol(t, **change):
    """The train of the method."""
    return dict(M["scan"], **change)


def _read(sc, t, protocol):
    """One scan: (echo times in ms, {region id: echoes})."""
    r = sc.scan(protocol, rois=t["regions"])
    return np.array(r["realised"]["echo_times_ms"]), {row["id"]: np.array(row["complex_mean_magnitude"], float) for row in r["roi_readouts"]}


def kept(s) -> int:
    """How many echoes enter a fit: from the first down to the last that is at least ``min_share`` of the largest."""
    s = np.asarray(s, float)
    below = np.flatnonzero(s < M["min_share"] * s.max())
    return int(min(len(s), max(M["min_echoes"], below[0] if len(below) else len(s))))


def fit_model(te, s) -> float:
    """T2 of one region: the echo-train model fitted to the kept echoes, amplitude, T2 and refocusing angle free."""
    return fit_parameters(te, s)[1]


def fit_parameters(te, s) -> tuple:
    """(amplitude, T2 in ms, refocusing angle in degrees) of the echo-train model fitted to the kept echoes of one region."""
    n = kept(s)
    y, esp = np.asarray(s[:n], float), float(te[0])
    lo, hi = np.array([0.0, M["t2_bounds_ms"][0], M["theta_bounds_deg"][0]]), np.array([np.inf, M["t2_bounds_ms"][1], M["theta_bounds_deg"][1]])
    best = None
    for t2_0, theta_0 in (M["start"], (30.0, 120.0), (250.0, 160.0)):      # the fit of the method, and two more starts
        r = scipy.optimize.least_squares(lambda p: p[0] * model.train(n, esp, p[1], p[2]) - y, [y[0], t2_0, theta_0], bounds=(lo, hi))
        best = r if best is None or r.cost < best.cost else best
    return tuple(float(v) for v in best.x)


def fit_line(te, s, sel=slice(None)) -> float:
    """T2 from a straight line through ln(signal) of the kept echoes (``sel``: which of them)."""
    n = kept(s)
    te, s = np.asarray(te[:n], float)[sel], np.asarray(s[:n], float)[sel]
    slope = np.polyfit(te, np.log(np.maximum(s, 1e-12)), 1)[0]
    return float(-1.0 / slope) if slope < 0 else task.T2_VALID_MS[1]


def _answer(t, te, echoes, fit):
    return dict(t2_ms={i: round(float(np.clip(fit(te, echoes[i]), *task.T2_VALID_MS)), 2) for i in _ids(t)})


def reference(sc):
    t = _task(sc)
    return _answer(t, *_read(sc, t, _protocol(t)), fit_model)


def open_loop(sc):
    return reference(sc)


def line_fit(sc):
    t = _task(sc)
    return _answer(t, *_read(sc, t, _protocol(t)), fit_line)


def drop_first(sc):
    t = _task(sc)
    return _answer(t, *_read(sc, t, _protocol(t)), lambda te, s: fit_line(te, s, slice(1, None)))


def even_echoes(sc):
    t = _task(sc)
    return _answer(t, *_read(sc, t, _protocol(t)), lambda te, s: fit_line(te, s, slice(1, None, 2)))


def default_pulse(sc):
    t = _task(sc)
    return _answer(t, *_read(sc, t, _protocol(t, rf_pulse="normal")), fit_model)


def mono_exp(sc):
    t = _task(sc)
    return _answer(t, *_read(sc, t, _protocol(t, rf_pulse="normal")), fit_line)


def single_echoes(sc):
    return single_echo.reference(sc)



def prior(sc):
    t = _task(sc)
    return dict(t2_ms={i: round(math.sqrt(t["t2_range_ms"][0] * t["t2_range_ms"][1]), 1) for i in _ids(t)})


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        sc.scan(dict(TOKEN), rois=t["regions"])
        return dict(t2_ms={i: float(recipe["t2_ms"]) for i in _ids(t)})
    return policy


DEFAULT_RECIPE = dict(t2_ms=89.0)


def wrong_units(sc):
    return dict(t2_ms={i: v / 1000.0 for i, v in reference(sc)["t2_ms"].items()})


def policies(recipe=None):
    return dict(reference=reference, prior=prior, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE), mono_exp=mono_exp,
                single_echoes=single_echoes, wrong_units=wrong_units, line_fit=line_fit, open_loop=open_loop,
                drop_first=drop_first, even_echoes=even_echoes, default_pulse=default_pulse)


def tune(cfgs):
    """The strongest answer that ignores the object, on other worlds: the constant T2 with the highest share of
    regions inside the pass window, among the true T2 of those worlds and the points of a fine grid over the told range."""
    truth = np.array([v for c in cfgs for v in task.true_t2_ms(c).values()])
    lo, hi = cfgs[0]["task"]["t2_range_ms"]
    cand = np.concatenate([truth, np.geomspace(lo, hi, 200)])
    width = math.log(1.0 + task.TOLERANCE)
    rate = [float(np.mean(np.abs(np.log(truth / v)) <= width)) for v in cand]
    best = int(np.argmax(rate))
    return dict(t2_ms=round(float(cand[best]), 2), tuned_region_pass_rate=round(rate[best], 3))
