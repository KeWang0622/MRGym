"""Control policies of ``Scanner/ErnstFlip``. Every policy plays through the agent's interface only (``info`` / ``plan``
/ ``scan`` / ``budget``) and returns the deliverable. None reads the hidden world. What a policy may use: the task
card, the instrument datasheet and the public physics of ``model.py`` (the ideal-spoiling steady state summed over the
slice profile of the documented pulse).

Must pass
  reference        two scans of the target sequence at the target's repetition time, at a low and a high flip angle
                   (the two angles level 1 states). Level 1: the stated method to the letter (one average, the ratio
                   of the two read-outs, linear interpolation in the printed table). Levels 2 and 3: averages that
                   fill the budget, and per region a fit of  M x (2D signal curve at T1)  to the two read-outs against
                   the NOMINAL angle; the answer is the maximum of the fitted curve. A transmit scale needs no extra
                   step: the scans sweep the nominal angle of the target itself
Must fail
  prior            no scan; the best angle of a material in the middle of the told T1 range, for every region
  fixed_recipe     one token scan, then the best answer that ignores the object: one constant angle, or the best
                   angle of one constant T1 at the told repetition time, tuned on other worlds
  brute_force      a sweep without a model: the two scans the budget pays for, at the pair of angles that is right for
                   the most vials of the told T1 range; per region the scanned angle with the larger signal
  epi_sweep        the cheapest sweep the console sells: six single-shot scans at their shortest repetition time
                   (50 ms; the target's is 12 to 30 ms); per region the scanned angle with the largest signal
  wrong_units      the reference's angles in radians
Ablations (the reference with one piece removed; each must lose at the levels it names)
  textbook         the reference's scans and fit give T1; the answer is the textbook angle arccos(exp(-TR / T1)) of
                   that T1, the best angle when every spin sees one flip angle (every level: the target is a 2D scan)
  ignores_profile  the same two scans fitted with the one-angle Ernst formula instead of the 2D curve; the answer is
                   the maximum of that fitted curve (every level)
Reported
  open_loop        a fixed scan list (the two scans, one average) and a fixed estimator (the 2D fit). If it passes
                   within 25 points of the reference the level is a measurement task
"""
from __future__ import annotations

import math

import numpy as np

from . import model, task

REFERENCE = "reference"
MUST_FAIL = ("prior", "fixed_recipe", "brute_force", "epi_sweep", "wrong_units")
ABLATIONS = {"textbook": (1, 2, 3), "ignores_profile": (1, 2, 3)}
REPORTED = ("open_loop",)
DESCRIPTIONS = dict(
    reference="two scans of the target at a low and a high flip angle; level 1: the stated table; else a fit of the 2D signal curve",
    prior="no scan; the best angle of a material in the middle of the told T1 range",
    fixed_recipe="one token scan, then one constant angle or the best angle of one constant T1, tuned on other worlds",
    brute_force="two scans at the pair of angles that suits most vials; per region the scanned angle with the larger signal",
    epi_sweep="six single-shot scans at 50 ms repetition time; per region the scanned angle with the largest signal",
    wrong_units="the reference's angles in radians",
    textbook="the reference's T1, then the textbook angle arccos(exp(-TR / T1)) (ignores the slice profile)",
    ignores_profile="the reference's scans fitted with the one-angle Ernst formula; the maximum of that curve",
    open_loop="a fixed scan list (two scans, one average) and a fixed estimator (the 2D fit)",
)
PROBE = dict(sequence="spgr", matrix=32)                    # the target sequence; the matrix of a probe does not change the answer
T1_MID_MS = math.sqrt(task.T1_RANGE_MS[0] * task.T1_RANGE_MS[1])
BRUTE_OFFSETS = (-0.3, 0.0)     # ln of the flip angle around the middle of the plausible range: the pair that passes most vials on tuning worlds
EPI = dict(sequence="epi", matrix=32, tr_ms=50.0)
EPI_SCANS = 6
MAX_AVERAGES = 16


def _task(sc):
    return sc.info()["task"]


def _ids(t):
    return [r["id"] for r in t["regions"]]


def _clip(flip):
    return float(np.clip(flip, task.FLIP_MIN_DEG, task.FLIP_MAX_DEG))


def _answer(ids, flips):
    return dict(flip_deg={i: round(_clip(v), 2) for i, v in zip(ids, flips)})


def _read(resp):
    """{region id: |complex mean|} of the first image of a scan."""
    return {row["id"]: float(row["complex_mean_magnitude"][0]) for row in resp["roi_readouts"]}


def _averages(sc, protocol, n_scans):
    """The averages with which ``n_scans`` scans of ``protocol`` fill what is left of the budget (at least 1)."""
    p = sc.plan(dict(protocol, averages=1))
    fixed, per = p["scan_s"] - p["scan_s_parts"]["acquisition_s"], p["scan_s_parts"]["acquisition_s"]
    room = sc.budget()["remaining_s"] / n_scans - fixed - 0.05
    return int(np.clip(math.floor(room / per), 1, MAX_AVERAGES))


def _two_scans(sc, t, fill_budget):
    """The two scans of the stated method: (flip angles, {region id: [signal low, signal high]})."""
    tr = t["target_scan"]["tr_ms"]
    flips = t["method"]["flips_deg"] if t.get("method") else task.probe_flips(tr)
    avg = _averages(sc, dict(PROBE, tr_ms=tr, flip_deg=flips[1]), 2) if fill_budget else 1
    reads = [_read(sc.scan(dict(PROBE, tr_ms=tr, flip_deg=f, averages=avg), rois=t["regions"])) for f in flips]
    return flips, {i: [r[i] for r in reads] for i in _ids(t)}


def _by_table(t, signals):
    """Level 1, to the letter: r = S(high) / S(low), linear interpolation in the printed table."""
    r, a = (np.array(c) for c in zip(*t["method"]["table"]))
    return [float(np.interp(s[1] / s[0], r, a)) for s in signals.values()]


def _by_fit(t, flips, signals, profile=True, answer="maximum"):
    """Per region: fit of the signal curve to the two read-outs, then the maximum of the fitted curve (``answer``
    "maximum") or the textbook angle of the fitted T1 ("ernst")."""
    tr, out = t["target_scan"]["tr_ms"], []
    for s in signals.values():
        t1 = model.fit(flips, s, tr, profile)[0]
        out.append(model.optimum_deg(tr, t1) if profile and answer == "maximum" else model.ernst_angle_deg(tr, t1))
    return out


def reference(sc):
    t = _task(sc)
    stated = bool(t.get("method"))
    flips, signals = _two_scans(sc, t, fill_budget=not stated)
    return _answer(_ids(t), _by_table(t, signals) if stated else _by_fit(t, flips, signals))


def open_loop(sc):
    t = _task(sc)
    flips, signals = _two_scans(sc, t, fill_budget=False)
    return _answer(_ids(t), _by_fit(t, flips, signals))


def textbook(sc):
    t = _task(sc)
    flips, signals = _two_scans(sc, t, fill_budget=True)
    return _answer(_ids(t), _by_fit(t, flips, signals, answer="ernst"))


def ignores_profile(sc):
    t = _task(sc)
    flips, signals = _two_scans(sc, t, fill_budget=True)
    return _answer(_ids(t), _by_fit(t, flips, signals, profile=False))


def _centre(t):
    return model.optimum_deg(t["target_scan"]["tr_ms"], T1_MID_MS)


def prior(sc):
    t = _task(sc)
    return _answer(_ids(t), [_centre(t)] * len(t["regions"]))


def _token_scan(sc, t):
    sc.scan(dict(PROBE, tr_ms=t["target_scan"]["tr_ms"], flip_deg=10.0))


def recipe_flip(recipe, tr_ms) -> float:
    """The answer of a recipe for a repetition time: a constant angle, or the best angle of a constant T1."""
    return float(recipe["flip_deg"]) if recipe.get("flip_deg") else model.optimum_deg(tr_ms, float(recipe["t1_ms"]))


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        _token_scan(sc, t)
        return _answer(_ids(t), [recipe_flip(recipe, t["target_scan"]["tr_ms"])] * len(t["regions"]))
    return policy


DEFAULT_RECIPE = dict(t1_ms=round(T1_MID_MS, 1))


def _largest(t, flips, reads):
    """Per region the scanned angle with the largest read-out."""
    return _answer(_ids(t), [flips[int(np.argmax([r[i] for r in reads]))] for i in _ids(t)])


def brute_force(sc):
    t = _task(sc)
    tr = t["target_scan"]["tr_ms"]
    flips = [round(_clip(_centre(t) * math.exp(d)), 1) for d in BRUTE_OFFSETS]
    return _largest(t, flips, [_read(sc.scan(dict(PROBE, tr_ms=tr, flip_deg=f), rois=t["regions"])) for f in flips])


def epi_sweep(sc):
    """Single-shot scans cost 3 s of adjustments and three repetitions, so six of them fit the budget. Their steady
    state is that of their own repetition time (50 ms at the least), not of the target's."""
    t = _task(sc)
    tr = t["target_scan"]["tr_ms"]
    lo, hi = (model.optimum_deg(tr, v) for v in reversed(task.T1_RANGE_MS))
    flips = [round(_clip(v), 1) for v in np.geomspace(lo, hi, EPI_SCANS)]
    reads = []
    for f in flips:
        p = sc.plan(dict(EPI, flip_deg=f))
        if not (p["accept"] and p["scan_s"] <= sc.budget()["remaining_s"]):
            break
        reads.append(_read(sc.scan(dict(EPI, flip_deg=f), rois=t["regions"])))
    if not reads:
        return prior(sc)
    return _largest(t, flips[:len(reads)], reads)


def wrong_units(sc):
    return dict(flip_deg={i: round(math.radians(v), 4) for i, v in reference(sc)["flip_deg"].items()})


def policies(recipe=None):
    return dict(reference=reference, prior=prior, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE), brute_force=brute_force,
                epi_sweep=epi_sweep, wrong_units=wrong_units, textbook=textbook, ignores_profile=ignores_profile, open_loop=open_loop)


def tune(cfgs):
    """The strongest answer that ignores the object, on other worlds: one constant angle, or the best angle of one
    constant T1 at the told repetition time, by the share of worlds on which every region passes (ties: the share of
    regions). The verifier's own rule, evaluated on the signal curve of every region's material."""
    from ...scanner import kspace
    from ...scanner import world as W
    rows = []                                                            # per world: (tr, [normalised curve of a region])
    for c in cfgs:
        w = W.make_world(c["seed"], c["world_spec"])
        e = kspace.Engine(w)
        rows.append((c["task"]["target_scan"]["tr_ms"], [v / v.max() for v in task.signal_curves(c, w, e).values()]))
    need, grid = cfgs[0]["task"]["pass_fraction"], model.FLIP_GRID_DEG
    cands = [dict(flip_deg=float(a)) for a in np.arange(3.0, 80.5, 1.0)] + [dict(t1_ms=round(float(v), 1)) for v in np.geomspace(100.0, 4000.0, 40)]
    best = None
    for cand in cands:
        ok = [[float(np.interp(_clip(recipe_flip(cand, tr)), grid, v)) >= need for v in curves] for tr, curves in rows]
        rate = (float(np.mean([all(o) for o in ok])), float(np.mean([x for o in ok for x in o])))
        if best is None or rate > best[0]:
            best = (rate, cand)
    return dict(best[1], tuned_pass_rate=round(best[0][0], 3), tuned_region_pass_rate=round(best[0][1], 3))
