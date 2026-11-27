"""Control policies of ``Scanner/T2Decay``. Every policy plays through the agent's interface only (``info`` / ``plan``
/ ``scan`` / ``budget``) and returns the deliverable. None reads the hidden world.

The estimator of every policy that fits: a straight line through ln(mean magnitude) against the echo time, T2 = -1 /
slope, over the two shortest echoes and those of the others that stand at least ``floor_factor`` times above the noise
floor (the mean magnitude of a circle that holds no object) and at least ``min_share`` of the region's first echo.

Must pass
  reference      a fixed list of scans with the recovery held fixed (tr_ms = TR0 + ``task.RECOVERY_SHARE`` x te_ms in
                 every scan). Where the method of level 1 fits the budget it is played as stated. Otherwise: told T2
                 range within a factor 6, three echo times on a 48 matrix; a wider range, four echo times spread
                 over it on a 64 matrix; both at the longest TR0 the budget pays
Must fail
  prior          no scan; the geometric mean of the told T2 range in every region (a textbook value)
  fixed_recipe   one token scan, then one constant T2 for every region, tuned on other worlds
  one_image      one scan at te_ms 80: T2 from the brightness alone, as if every vial had the proton density and the
                 recovery of the brightest one
  brute_force    a ladder of echo times at one repetition time; delivers the first echo time at which the signal is
                 below 1 / e of the first echo's. No fit
  wrong_units    the reference's T2 in seconds (refused by the console of the task: not a T2 in milliseconds)
Ablations (the reference with one step removed; required where the step decides)
  fixed_tr       the same echo times at one repetition time (the mean of the reference's): levels 2 and 3
  no_floor       the fit through every echo, none left out: level 3
Reported
  open_loop      the reference: it is a fixed list of scans with a fixed estimator at every level
  adaptive       a wide range of T2 only (elsewhere the reference): two echo times first, two more placed from what
                 they show. Says whether choosing the next scan from the previous result buys anything
  ideal_pulses   as the reference with tr_ms = TR0 + te_ms: the rule for ideal pulses (the recovery lasts TR - TE)
  coarse         as the reference on a 32 matrix with partial_fourier 0.6 at the repetition time that image pays
  short_echoes   as the reference with the echo times 12, 20 and 30 ms: too close for the noise
"""
from __future__ import annotations

import math

import numpy as np

from . import task

REFERENCE = "reference"
MUST_FAIL = ("prior", "fixed_recipe", "one_image", "brute_force", "wrong_units")
ABLATIONS = {"fixed_tr": (2, 3), "no_floor": (3,)}
REPORTED = ("open_loop", "adaptive", "ideal_pulses", "coarse", "short_echoes")
DESCRIPTIONS = dict(
    reference="echo times with the recovery held fixed (tr_ms = TR0 + 0.75 x te_ms), a line through ln(signal), echoes at the noise floor left out",
    prior="no scan; the geometric mean of the told T2 range in every region",
    fixed_recipe="one token scan, then one constant T2 tuned on other worlds",
    one_image="one scan at te_ms 80; T2 from the brightness alone, the brightest region taken as the reference",
    brute_force="a ladder of echo times at one repetition time; the first echo time with the signal below 1 / e of the first echo, no fit",
    wrong_units="the reference's T2 in seconds",
    fixed_tr="the reference's echo times at one repetition time (the recovery differs between the echoes)",
    no_floor="the reference's scans, the line fitted through every echo (none left out at the noise floor)",
    open_loop="the reference (a fixed list of scans, a fixed estimator)",
    adaptive="level 3: two echo times first, two more placed from what they show (elsewhere the reference)",
    ideal_pulses="the reference with tr_ms = TR0 + te_ms (the rule for ideal pulses)",
    coarse="the reference on a 32 matrix with partial_fourier 0.6",
    short_echoes="the reference with the echo times 12, 20 and 30 ms",
)
M = task.METHOD
FINE = dict(sequence="se", matrix=M["matrix"], partial_fourier=M["partial_fourier"])
FAST = dict(sequence="se", matrix=48, partial_fourier=0.75)
COARSE = dict(sequence="se", matrix=32, partial_fourier=0.6)
NARROW_TE_MS = (12.0, 55.0, 150.0)          # three echo times for T2 within 40 to 200 ms
NARROW_RANGE = 6.0                          # told hi / lo up to which one list of echo times serves every vial
WIDE_TE_MS = (8.5, 32.0, 100.0, 300.0)      # four echo times for T2 within 15 to 400 ms (8.5: about the console minimum)
WIDE_FIRST_MS = WIDE_TE_MS[:2]              # the two echo times ``adaptive`` starts with
TR0_MIN_MS, TR0_MAX_MS = 300.0, M["tr0_ms"]
USE = 0.97                                  # share of the remaining scan time a planned list may take


def _task(sc):
    return sc.info()["task"]


def _ids(t):
    return [r["id"] for r in t["regions"]]


def _read(sc, t, protocol):
    """One scan: (realised te_ms, {region id: mean magnitude}, noise floor)."""
    r = sc.scan(protocol, rois=t["regions"] + [M["background"]])
    rows = {row["id"]: float(row["mean_magnitude"][0]) for row in r["roi_readouts"]}
    return float(r["realised"]["te_ms"]), {i: rows[i] for i in _ids(t)}, rows[M["background"]["id"]]


def _cycle_s(sc, geom):
    """(seconds per scan that do not depend on tr_ms, seconds per millisecond of tr_ms) of a spin echo with ``geom``."""
    a, b = (sc.plan(dict(geom, tr_ms=tr, te_ms=12.0))["scan_s"] for tr in (1000.0, 2000.0))
    slope = (b - a) / 1000.0
    return a - 1000.0 * slope, slope


def _tr0(sc, geom, te_ms, share=task.RECOVERY_SHARE, left=None):
    """The longest TR0 (10 ms steps, at most the stated one) at which the scans ``te_ms`` with tr_ms = TR0 + share x
    te_ms fit ``left`` seconds (default: ``USE`` of what remains)."""
    fixed, slope = _cycle_s(sc, geom)
    left = USE * sc.budget()["remaining_s"] if left is None else left
    tr0 = ((left - len(te_ms) * fixed) / slope - share * sum(te_ms)) / len(te_ms)
    return float(np.clip(math.floor(tr0 / 10.0) * 10.0, TR0_MIN_MS, TR0_MAX_MS))


def fit_t2(te, s, floor, drop=True):
    """T2 from echo times ``te`` and signals ``s`` (ascending te): the line through ln(s) over the kept echoes."""
    te, s = np.asarray(te, float), np.asarray(s, float)
    keep = np.ones(len(te), bool)
    if drop:
        keep = (s >= M["floor_factor"] * floor) & (s >= M["min_share"] * s[0])
        keep[:2] = True
    slope = np.polyfit(te[keep], np.log(np.maximum(s[keep], 1e-12)), 1)[0]
    return float(np.clip(-1.0 / slope, task.T2_VALID_MS[0], task.T2_VALID_MS[1])) if slope < 0 else task.T2_VALID_MS[1]


def _answer(t, scans, drop=True):
    """The deliverable from a list of ``_read`` results."""
    scans = sorted(scans, key=lambda r: r[0])
    floor = float(np.mean([r[2] for r in scans]))
    return dict(t2_ms={i: round(fit_t2([r[0] for r in scans], [r[1][i] for r in scans], floor, drop), 2) for i in _ids(t)})


def _play(sc, t, geom, te_ms, tr0, share=task.RECOVERY_SHARE):
    return [_read(sc, t, dict(geom, te_ms=te, tr_ms=round(tr0 + share * te, 1))) for te in te_ms]


def _stated_fits(sc):
    """Whether the method of level 1 fits what is left of the scan time."""
    cost = sum(sc.plan(dict(FINE, te_ms=te, tr_ms=M["tr0_ms"] + task.RECOVERY_SHARE * te))["scan_s"] for te in M["te_ms"])
    return cost <= sc.budget()["remaining_s"]


def _narrow(t):
    lo, hi = t["t2_range_ms"]
    return hi / lo <= NARROW_RANGE


def _fixed_list(sc, t):
    """The scans of a fixed list: (geometry, echo times, TR0)."""
    if _narrow(t) and _stated_fits(sc):
        return FINE, M["te_ms"], M["tr0_ms"]
    if _narrow(t):
        return FAST, NARROW_TE_MS, _tr0(sc, FAST, NARROW_TE_MS)
    return FINE, WIDE_TE_MS, _tr0(sc, FINE, WIDE_TE_MS)


def _adaptive(sc, t):
    """A wide range of T2: two short echo times, then two more placed where the first two say the longest and the
    middle T2 of the marked regions decay by about 1 / e. Returns the scans."""
    te_max, share = WIDE_TE_MS[-1], task.RECOVERY_SHARE
    tr0 = _tr0(sc, FINE, WIDE_FIRST_MS + (te_max, te_max))
    scans = _play(sc, t, FINE, WIDE_FIRST_MS, tr0)
    first = sorted(_answer(t, scans)["t2_ms"].values())
    late = float(np.clip(1.2 * first[-1], 90.0, te_max))
    mid = float(np.clip(1.2 * first[len(first) // 2], 55.0, 0.6 * late))
    return scans + [_read(sc, t, dict(FINE, te_ms=round(te, 1), tr_ms=round(tr0 + share * te, 1))) for te in (mid, late)]


def reference(sc):
    t = _task(sc)
    return _answer(t, _play(sc, t, *_fixed_list(sc, t)))


def open_loop(sc):
    return reference(sc)


def adaptive(sc):
    t = _task(sc)
    return reference(sc) if _narrow(t) else _answer(t, _adaptive(sc, t))


def prior(sc):
    t = _task(sc)
    return dict(t2_ms={i: round(math.sqrt(t["t2_range_ms"][0] * t["t2_range_ms"][1]), 1) for i in _ids(t)})


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        _read(sc, t, dict(COARSE, te_ms=12.0, tr_ms=300.0))
        return dict(t2_ms={i: float(recipe["t2_ms"]) for i in _ids(t)})
    return policy


DEFAULT_RECIPE = dict(t2_ms=89.0)


def one_image(sc):
    t = _task(sc)
    te, s, _ = _read(sc, t, dict(FAST, te_ms=80.0, tr_ms=_tr0(sc, FAST, (80.0,), share=0.0)))
    top, hi = max(s.values()), t["t2_range_ms"][1]
    return dict(t2_ms={i: round(float(np.clip(te / (te / hi + math.log(top / v)), *task.T2_VALID_MS)), 2) for i, v in s.items()})


def brute_force(sc):
    t = _task(sc)
    te_ms = M["te_ms"] if _narrow(t) else WIDE_TE_MS
    scans = _play(sc, t, COARSE, te_ms, _tr0(sc, COARSE, te_ms, share=0.0), share=0.0)
    out = {}
    for i in _ids(t):
        below = [r[0] for r in scans[1:] if r[1][i] < math.exp(-1.0) * scans[0][1][i]]
        out[i] = round(below[0] if below else scans[-1][0], 2)
    return dict(t2_ms=out)


def wrong_units(sc):
    return dict(t2_ms={i: v / 1000.0 for i, v in reference(sc)["t2_ms"].items()})


def fixed_tr(sc):
    t = _task(sc)
    geom, te_ms, tr0 = _fixed_list(sc, t)
    tr = tr0 + task.RECOVERY_SHARE * float(np.mean(te_ms))                 # the same scan time as the reference
    return _answer(t, [_read(sc, t, dict(geom, te_ms=te, tr_ms=round(tr, 1))) for te in te_ms])


def no_floor(sc):
    t = _task(sc)
    return _answer(t, _play(sc, t, *_fixed_list(sc, t)), drop=False)


def ideal_pulses(sc):
    t = _task(sc)
    geom, te_ms, _ = _fixed_list(sc, t)
    tr0 = M["tr0_ms"] if _narrow(t) and _stated_fits(sc) else _tr0(sc, geom, te_ms, share=1.0)
    return _answer(t, _play(sc, t, geom, te_ms, tr0, share=1.0))


def coarse(sc):
    t = _task(sc)
    te_ms = _fixed_list(sc, t)[1]
    return _answer(t, _play(sc, t, COARSE, te_ms, _tr0(sc, COARSE, te_ms)))


def short_echoes(sc):
    t = _task(sc)
    geom, te_ms = _fixed_list(sc, t)[0], (12.0, 20.0, 30.0)
    return _answer(t, _play(sc, t, geom, te_ms, _tr0(sc, geom, te_ms)))


def policies(recipe=None):
    return dict(reference=reference, prior=prior, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE), one_image=one_image,
                brute_force=brute_force, wrong_units=wrong_units, fixed_tr=fixed_tr, no_floor=no_floor, open_loop=open_loop,
                adaptive=adaptive, ideal_pulses=ideal_pulses, coarse=coarse, short_echoes=short_echoes)


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
