"""Control policies of ``Scanner/FrequencyAdjust``. Every policy plays through the agent's interface only (``info`` /
``plan`` / ``scan`` / ``budget``) and returns the deliverable. None reads the hidden world.

The estimator, written here on the agent's side: two gradient echoes dTE apart differ in phase by 2 pi x offset x dTE,
so  offset = (phase difference, in turns) / dTE  modulo 1 / dTE. A ladder of echo-time differences removes the
"modulo": the shortest one covers the whole range the task states, and each longer one is unwrapped with the
estimate before it and replaces it.

Must pass
  reference       spgr with three echoes at the console's shortest spacing, read on the marked regions. When the
                  task states offsets (and, at level 3, oil lines) beyond the range of that spacing, one more scan of
                  one echo 0.3 ms earlier, which gives an echo-time difference that covers +-1667 Hz. Ladder: 0.3 ms
                  (if needed), one spacing, two spacings. Level 3: a region whose coarse frequency lies more than half
                  the water-fat shift below the highest region is oil and is named
Must fail
  prior           no scan; 0 Hz in every region
  fixed_recipe    one token scan, then one constant offset for every region, tuned on other worlds
  wrong_units     the reference's offsets as phase slopes in rad/s (not divided by 2 pi)
  one_echo_sweep  three scans of one echo at three echo times; phase / (2 pi TE) of each, averaged: no model of the
                  phase an image has at TE = 0 (the receive coil's)
Ablations (the reference with one step removed; each must lose at the levels it names)
  single_spacing  the method stated at level 1 at every level: two echoes 2 ms apart, no ladder (levels 2 and 3, where
                  the offset wraps that spacing)
  lock_on_fat     the reference without the species step: a number for every region (level 3)
Reported
  open_loop       the two scans at every level and the same estimator. No scan of the reference depends on a
                  measurement (it depends on what the task states), so this is the reference with its scan list fixed
"""
from __future__ import annotations

import math

import numpy as np

from . import task

REFERENCE = "reference"
MUST_FAIL = ("prior", "fixed_recipe", "wrong_units", "one_echo_sweep")
ABLATIONS = {"single_spacing": (2, 3), "lock_on_fat": (3,)}
REPORTED = ("open_loop",)
DESCRIPTIONS = dict(
    reference="multi-echo spgr, phase differences on a ladder of echo-time differences (0.3 ms to two echo spacings); oil named by its shift",
    prior="no scan; 0 Hz in every region",
    fixed_recipe="one token scan, then one constant offset for every region, tuned on other worlds",
    wrong_units="the reference's offsets in rad/s (the phase slope, not divided by 2 pi)",
    one_echo_sweep="three single-echo scans; phase / (2 pi TE) of each, averaged (the receive phase is ignored)",
    single_spacing="two echoes 2 ms apart, no ladder: the method stated at level 1",
    lock_on_fat="the reference without the species step: a number for every region",
    open_loop="both scans at every level and the reference's estimator (a fixed scan list)",
)
GAMMA_MHZ_PER_T = 42.577478518
PROBE = dict(sequence="spgr", tr_ms=30.0, flip_deg=15.0, matrix=64, bandwidth_hz_per_px=1000.0)
SHORT_MS = 0.3                              # echo-time step between the two scans: +-1 / (2 x 0.3 ms) = +-1667 Hz without a wrap
N_ECHOES = 3
RANGE_MARGIN = 1.1                          # the range a spacing covers must exceed the stated bound by this factor
OIL_SPREAD_HZ = 60.0                        # how far the apparent frequency of oil may lie from its main line (several lines)


def _task(sc):
    return sc.info()["task"]


def _ids(t):
    return [r["id"] for r in t["regions"]]


def _phasors(resp) -> dict:
    """{region id: complex mean of every image of the scan}."""
    return {r["id"]: np.array(r["complex_mean_magnitude"], float) * np.exp(1j * np.array(r["complex_mean_phase_rad"], float))
            for r in resp["roi_readouts"]}


def fat_shift_hz(field_t: float) -> float:
    """Distance of the main fat line below water, Hz (datasheet: 3.4 ppm)."""
    return task.FAT_PPM * GAMMA_MHZ_PER_T * field_t


def ladder(times_ms, z) -> list:
    """Offsets in Hz from the complex signals ``z`` of one region at ``times_ms`` (increasing): one estimate per echo
    after the first, each from the phase difference to the first echo, unwrapped with the estimate before it. The
    first estimate is right modulo 1 / (times[1] - times[0]); the caller makes that difference short enough."""
    t = (np.asarray(times_ms, float) - times_ms[0]) * 1e-3
    out = []
    for k in range(1, len(t)):
        turns = float(np.angle(z[k] * np.conj(z[0]))) / (2 * math.pi)                   # in (-0.5, 0.5]
        whole = round(out[-1] * t[k] - turns) if out else 0
        out.append((turns + whole) / t[k])
    return out


def _measure(sc, both=None):
    """The scans and, per region, the ladder: (task, {id: [coarse ... fine] offsets in Hz}). ``both``: play the
    single-echo scan too (None: when the stated range needs it)."""
    card = sc.info()
    t = card["task"]
    te0 = sc.plan(dict(PROBE))["realised"]["te_ms"]
    multi = dict(PROBE, te_ms=round(te0 + SHORT_MS, 2), n_echoes=N_ECHOES)
    spacing_ms = sc.plan(multi)["realised"]["echo_spacing_ms"]
    reach = t["offset_bound_hz"] * RANGE_MARGIN + (fat_shift_hz(card["field_strength_t"]) + OIL_SPREAD_HZ if t["oil_possible"] else 0.0)
    if both is None:
        both = reach >= 0.5e3 / spacing_ms
    first = sc.scan(dict(PROBE, te_ms=te0), t["regions"]) if both else None
    b = sc.scan(multi, t["regions"])
    times, z = list(b["realised"]["echo_times_ms"]), _phasors(b)
    if first is not None:
        times, z0 = [first["realised"]["echo_times_ms"][0]] + times, _phasors(first)
        z = {i: np.concatenate([z0[i], z[i]]) for i in z}
    out = {}
    for i in _ids(t):
        if first is None:
            out[i] = ladder(times, z[i])
        else:                                                           # the 0.3 ms step spans two scans; the fine steps stay within one
            coarse = ladder(times[:2], z[i][:2])[0]
            fine = np.asarray(times[1:], float) - times[1]
            turns = [float(np.angle(z[i][k + 1] * np.conj(z[i][1]))) / (2 * math.pi) for k in (1, 2)]
            est = [coarse]
            for k, d in zip((1, 2), turns):
                est.append((d + round(est[-1] * fine[k] * 1e-3 - d)) / (fine[k] * 1e-3))
            out[i] = est
    return card, out


def _species(card, est: dict) -> dict:
    """{id: True where the region is oil}: its coarse frequency lies more than half the water-fat shift below the
    highest region's. At least one marked region holds water (the task says so), and water is the higher line."""
    top = max(v[0] for v in est.values())
    return {i: v[0] < top - 0.5 * fat_shift_hz(card["field_strength_t"]) for i, v in est.items()}


def _answer(card, est: dict, species=True) -> dict:
    oil = _species(card, est) if species and card["task"]["oil_possible"] else {}
    lo, hi = task.OFFSET_VALID_HZ
    return {"offset_hz": {i: task.OIL if oil.get(i) else round(float(np.clip(v[-1], lo, hi)), 2) for i, v in est.items()}}


def reference(sc):
    return _answer(*_measure(sc))


def open_loop(sc):
    return _answer(*_measure(sc, both=True))


def lock_on_fat(sc):
    return _answer(*_measure(sc), species=False)


def single_spacing(sc):
    card = sc.info()
    t = card["task"]
    r = sc.scan(dict(task.METHOD_SCAN), t["regions"])
    return _answer(card, {i: ladder(r["realised"]["echo_times_ms"], z) for i, z in _phasors(r).items()})


def one_echo_sweep(sc):
    t = _task(sc)
    te0 = sc.plan(dict(PROBE))["realised"]["te_ms"]
    est = {i: [] for i in _ids(t)}
    for k in range(3):
        r = sc.scan(dict(PROBE, te_ms=round(te0 + k, 2)), t["regions"])
        te = r["realised"]["echo_times_ms"][0] * 1e-3
        for i, z in _phasors(r).items():
            est[i].append(float(np.angle(z[0])) / (2 * math.pi * te))
    return {"offset_hz": {i: round(float(np.mean(v)), 2) for i, v in est.items()}}


def prior(sc):
    return {"offset_hz": {i: 0.0 for i in _ids(_task(sc))}}


def _token_scan(sc):
    sc.scan(dict(sequence="spgr", matrix=48, tr_ms=11.0, flip_deg=8.0))


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        _token_scan(sc)
        return {"offset_hz": {i: float(recipe["offset_hz"]) for i in _ids(t)}}
    return policy


DEFAULT_RECIPE = dict(offset_hz=0.0)


def wrong_units(sc):
    return {"offset_hz": {i: v if v == task.OIL else round(2 * math.pi * v, 2) for i, v in reference(sc)["offset_hz"].items()}}


def policies(recipe=None):
    return dict(reference=reference, prior=prior, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE), wrong_units=wrong_units,
                one_echo_sweep=one_echo_sweep, single_spacing=single_spacing, lock_on_fat=lock_on_fat, open_loop=open_loop)


def tune(cfgs):
    """The strongest answer that ignores the object, on other worlds: the one constant offset that brings the most
    regions of water within the tolerance (the verifier's own rule, per region), and the share of worlds it passes.
    Nothing the task states varies with the answer (the bound and the tolerance are constants of a level)."""
    truths = [task.truth(c) for c in cfgs]
    tol, bound = cfgs[0]["task"]["tolerance_hz"], cfgs[0]["task"]["offset_bound_hz"]
    values = np.array([t["offset_hz"] for tr in truths for t in tr.values() if not t["oil"]])
    grid = np.arange(-bound, bound + 0.5, 1.0)
    rate = [float(np.mean(np.abs(c - values) <= tol)) for c in grid]
    best = float(grid[int(np.argmax(rate))])
    passed = sum(all(not t["oil"] and abs(best - t["offset_hz"]) <= tol for t in tr.values()) for tr in truths)
    return dict(offset_hz=best, tuned_region_pass_rate=round(max(rate), 3), tuned_pass_rate=round(passed / len(truths), 3))
