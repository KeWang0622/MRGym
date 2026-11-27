"""Control policies of ``Scanner/TransmitAdjust``. Every policy plays through the agent's interface only (``info`` /
``plan`` / ``scan`` / ``budget``) and returns the deliverable. None reads the hidden world.

The estimator, written here on the agent's side (datasheet, afi): the dual-TR scan returns the echo of the short
interval (image 1) and of the long one (image 2); for repetition times much shorter than T1 and ideal spoiling
r = S2 / S1 = (1 + n cos a) / (n + cos a) with n = TR2 / TR1, so a = arccos((r n - 1) / (n - r)) and the factor is
a / nominal angle. The sign of S2 relative to S1 is read from the two phase images (above about 100 degrees S2 is
negative).

Must pass
  reference         the non-selective dual-TR scan with the strongest spoiler and its phase increment
                    (``task.METHOD_SCAN``), with as much of k-space (partial Fourier) as the budget pays; the formula
Must fail
  prior             no scan; 1.0 in every region
  fixed_recipe      one token scan, then the best answer that ignores the object: a constant, times a bowl around the
                    isocentre evaluated at the (told) position of each region; both tuned on other worlds
  wrong_units       the reference's factors in percent
  afi_2d            the plausible shortcut: the same dual-TR scan slice-selective (2D, a twentieth of the scan time)
                    with the same formula
  flip_sweep        2D gradient echoes at five nominal flip angles; the angle of the largest signal is taken for 90
                    degrees actual. No model: at an affordable repetition time the signal peaks at the Ernst angle
Ablation (the reference with one step removed; it must lose at the level it names)
  untrimmed         the reference without fitting the scan to the budget: the level-1 protocol as stated (level 3,
                    where the budget does not pay it)
Reported
  open_loop         one scan list for every instance of every level (the protocol with the smallest part of k-space
                    the reference uses) and the same formula
  default_spoiling  the reference with the console's default spoiler moment and RF phase increment: the factor reads
                    high in materials with a long T2, beyond the tolerance in a part of the instances
"""
from __future__ import annotations

import numpy as np

from . import task

REFERENCE = "reference"
MUST_FAIL = ("prior", "fixed_recipe", "wrong_units", "afi_2d", "flip_sweep")
ABLATIONS = {"untrimmed": (3,)}
REPORTED = ("open_loop", "default_spoiling")
DESCRIPTIONS = dict(
    reference="non-selective dual-TR scan with strong spoiling, cut to the budget by partial Fourier; the dual-TR formula",
    prior="no scan; 1.0 in every region",
    fixed_recipe="one token scan, then a constant times a bowl around the isocentre, tuned on other worlds",
    wrong_units="the reference's factors in percent",
    afi_2d="the same dual-TR scan slice-selective (2D) with the same formula",
    flip_sweep="2D gradient echoes at five nominal flip angles; the largest signal is taken for 90 degrees actual",
    default_spoiling="the reference with the console's default spoiler moment and RF phase increment",
    untrimmed="the level-1 protocol as stated, whatever the budget",
    open_loop="a fixed scan list (the reference's protocol at its smallest partial Fourier) and the same formula",
)
PARTIAL_FOURIER = (1.0, 0.8, 0.7, 0.6)      # tried in this order; the first that the budget pays is played
SWEEP = dict(sequence="spgr", tr_ms=100.0, matrix=32, bandwidth_hz_per_px=500.0)
SWEEP_FLIPS = (30.0, 45.0, 60.0, 75.0, 90.0)
BOWL_MM = 72.0                              # length scale of the bowl of ``fixed_recipe``


def _task(sc):
    return sc.info()["task"]


def _ids(t):
    return [r["id"] for r in t["regions"]]


def flip_ratio(s1, s2, n, nominal_deg):
    """The dual-TR formula: actual / nominal flip angle from the signed signals of the short (``s1``) and the long
    (``s2``) interval, n = TR2 / TR1."""
    r = np.asarray(s2, float) / np.asarray(s1, float)
    return np.degrees(np.arccos(np.clip((r * n - 1.0) / (n - r), -1.0, 1.0))) / nominal_deg


def _signed(resp):
    """(S1, S2) per region of a dual-TR scan, S2 signed by the phase difference of the two images."""
    mag = np.array([r["complex_mean_magnitude"] for r in resp["roi_readouts"]], float).T
    ph = np.array([r["complex_mean_phase_rad"] for r in resp["roi_readouts"]], float).T
    return mag[0], mag[1] * np.sign(np.cos(ph[1] - ph[0]))


def _answer(ids, values):
    lo, hi = task.RATIO_VALID
    return {"flip_ratio": {i: round(float(np.clip(v, lo, hi)), 4) for i, v in zip(ids, values)}}


def _afi(sc, protocol):
    """One dual-TR scan and the formula on the marked regions."""
    t = _task(sc)
    r = sc.scan(protocol, t["regions"])
    return _answer([row["id"] for row in r["roi_readouts"]], flip_ratio(*_signed(r), r["realised"]["tr_ratio"], r["realised"]["flip_deg"]))


def _fit_budget(sc, protocol):
    """The protocol with the largest part of k-space the budget pays (the last candidate if none fits: the scan is
    then refused and the episode fails)."""
    left = sc.budget()["remaining_s"]
    for pf in PARTIAL_FOURIER:
        p = sc.plan(dict(protocol, partial_fourier=pf))
        if p["accept"] and p["scan_s"] <= left:
            break
    return dict(protocol, partial_fourier=pf)


def reference(sc):
    return _afi(sc, _fit_budget(sc, task.METHOD_SCAN))


def open_loop(sc):
    return _afi(sc, dict(task.METHOD_SCAN, partial_fourier=PARTIAL_FOURIER[-1]))


def untrimmed(sc):
    return _afi(sc, dict(task.METHOD_SCAN))


def default_spoiling(sc):
    weak = {k: v for k, v in task.METHOD_SCAN.items() if k not in ("spoiler_mt_ms_per_m", "rf_phase_increment_deg")}
    return _afi(sc, _fit_budget(sc, weak))


def afi_2d(sc):
    return _afi(sc, dict(task.METHOD_SCAN, excitation="selective"))


def flip_sweep(sc):
    t = _task(sc)
    signal = np.array([[row["complex_mean_magnitude"][0] for row in sc.scan(dict(SWEEP, flip_deg=f), t["regions"])["roi_readouts"]]
                       for f in SWEEP_FLIPS])
    return _answer(_ids(t), [90.0 / SWEEP_FLIPS[int(k)] for k in np.argmax(signal, axis=0)])


def prior(sc):
    t = _task(sc)
    return _answer(_ids(t), [1.0] * len(t["regions"]))


def _token_scan(sc):
    sc.scan(dict(sequence="spgr", matrix=48, tr_ms=11.0, flip_deg=8.0))


def _bowl(recipe, region):
    """The answer of ``fixed_recipe`` for one region: a constant times a bowl around the isocentre at its position."""
    rho2 = (region["y_mm"] ** 2 + region["x_mm"] ** 2) / BOWL_MM ** 2
    return recipe["scale"] * (1.0 + recipe["bowl"] * (0.5 - rho2))


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        _token_scan(sc)
        return _answer(_ids(t), [_bowl(recipe, r) for r in t["regions"]])
    return policy


DEFAULT_RECIPE = dict(scale=1.0, bowl=0.0)


def wrong_units(sc):
    return {"flip_ratio": {i: round(100.0 * v, 2) for i, v in reference(sc)["flip_ratio"].items()}}


def policies(recipe=None):
    return dict(reference=reference, prior=prior, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE), wrong_units=wrong_units,
                afi_2d=afi_2d, flip_sweep=flip_sweep, default_spoiling=default_spoiling, untrimmed=untrimmed, open_loop=open_loop)


def tune(cfgs):
    """The strongest answer that ignores the object, on other worlds: a constant factor times a bowl around the
    isocentre evaluated at each region's told position (the transmit pattern of these objects is a bowl), both on a
    grid, by the share of worlds whose every region is within the tolerance; ties by the share of regions."""
    truths = [(c["task"]["regions"], task.truth(c)) for c in cfgs]
    tol = cfgs[0]["task"]["tolerance"]
    best = None
    for scale in np.linspace(task.RATIO_BOUND[0] + 0.2, task.RATIO_BOUND[1] - 0.3, 41):
        for bowl in np.linspace(-0.3, 0.3, 13):
            recipe = dict(scale=float(scale), bowl=float(bowl))
            hit = [[abs(_bowl(recipe, r) / tr[r["id"]] - 1.0) <= tol for r in regions] for regions, tr in truths]
            key = (sum(all(h) for h in hit) / len(hit), float(np.mean([v for h in hit for v in h])))
            if best is None or key > best[0]:
                best = (key, recipe)
    (worlds, regions), recipe = best
    return dict(scale=round(recipe["scale"], 4), bowl=round(recipe["bowl"], 4), tuned_pass_rate=round(worlds, 3), tuned_region_pass_rate=round(regions, 3))
