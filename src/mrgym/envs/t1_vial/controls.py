"""Control policies of ``Scanner/T1Vial``. Every policy plays through the agent's interface only (``info`` / ``plan`` /
``scan`` / ``budget``) and returns the deliverable. None reads the hidden world.

The estimator of the reference is the three-parameter recovery curve at a fixed repetition time,
    S(TI) = a - b exp(-TI / T1),
fitted to SIGNED region read-outs: the sign of a read-out is that of its phase relative to the same region in the scan
with the longest inversion time, where every material is past its null. Nothing else is modelled: the repetition time,
the inversion efficiency and the slice profile change a and b only (``task`` module docstring).

Every scan list below is fixed per level: no scan depends on a measurement. The lists (``DESIGN``) come from an offline
study over the told T1 ranges: the Cramer-Rao bound of T1 for three or four inversion times at the repetition time the
budget pays for, and the leak of the surrounding gel into a region mean, which a fit turns into an error of T1 that
grows when two inversion times sit on the plateau of a short T1.

Must pass
  reference        inversion recovery with the level's scan list (full Fourier). Level 1: the stated method to the
                   letter (three scans, the ratio q, bisection). Levels 2 and 3: a least-squares fit of the
                   three-parameter curve. Level 2: three inversion times at the repetition time the budget pays for
                   (900 ms), spread from the shortest to the longest. Level 3: four inversion times from 10 to 3480 ms
                   at matrix 48, where the gel around a vial leaks half as much into the region mean of a short T1
Must fail
  prior            no scan; the geometric mean of the told T1 range in every region
  fixed_recipe     one token scan, then one constant T1 for every region, tuned on other worlds
  brute_force      a sweep without a model: inversion times on a grid, as many as the budget pays for at a repetition
                   time of 700 ms; T1 = (inversion time of the smallest magnitude) / ln 2
  wrong_units      the reference's numbers in seconds
Ablations (the reference with one piece removed; each must lose at the levels it names)
  unsigned         the reference's scans fitted to magnitudes, without the sign (every level)
  full_recovery    the reference's scans, signed, fitted with the textbook two-parameter curve M (1 - 2 exp(-TI / T1)):
                   it assumes a repetition time of many T1 and a complete inversion (levels 1 and 2)
  bunched_tis      the reference's repetition time with three inversion times around the expected nulls (150, 300,
                   600 ms) instead of spread over the recovery (level 2)
  level1_scans     the three scans level 1 states, at every level (levels 2 and 3: over the budget of level 2, and
                   blind to the short and the long T1 of level 3)
Reported
  open_loop        the level's fixed scan list and the least-squares fit: what the reference is at levels 2 and 3.
                   If it passes within 25 points of the reference the level is a measurement task
"""
from __future__ import annotations

import math

import numpy as np

from . import task

REFERENCE = "reference"
MUST_FAIL = ("prior", "fixed_recipe", "brute_force", "wrong_units")
ABLATIONS = {"unsigned": (1, 2, 3), "full_recovery": (1, 2), "bunched_tis": (2,), "level1_scans": (2, 3)}
REPORTED = ("open_loop",)
DESCRIPTIONS = dict(
    reference="inversion recovery, three or four inversion times spread over the recovery, signed three-parameter fit",
    prior="no scan; the geometric mean of the told T1 range in every region",
    fixed_recipe="one token scan, then one constant T1 for every region, tuned on other worlds",
    brute_force="inversion times on a grid; T1 = (inversion time of the smallest magnitude) / ln 2",
    wrong_units="the reference's numbers in seconds",
    unsigned="the reference's scans fitted to magnitudes, without the sign",
    full_recovery="the reference's scans fitted with M (1 - 2 exp(-TI / T1)) (assumes full recovery and a complete inversion)",
    bunched_tis="three inversion times around the expected nulls instead of spread over the recovery",
    level1_scans="the three scans level 1 states, at every level",
    open_loop="the level's fixed scan list and the least-squares fit of the three-parameter curve",
)
PROBE = dict(task.PROBE)
# Scan list per T1 range and budget: (repetition time in ms, inversion times in ms, matrix). Chosen offline (module docstring).
DESIGN = {1: (task.STATED["tr_ms"], task.STATED["ti_ms"], PROBE["matrix"]), 2: (900.0, (15.0, 400.0, 875.0), 32),
          3: (3500.0, (10.0, 100.0, 900.0, 3480.0), 48)}
BUNCHED_TI_MS = (150.0, 300.0, 600.0)
BRUTE_TR_MS = 700.0
T1_GRID_MS = np.geomspace(15.0, 15000.0, 1200)


def _info(sc):
    return sc.info()


def _ids(t):
    return [r["id"] for r in t["regions"]]


def _level(t) -> int:
    """The level, from what the task card states (the policies are given no level): the stated method, the spread."""
    return 1 if t.get("method") else 3 if t.get("min_spread") else 2


def _answer(ids, t1):
    return {"t1_ms": {k: round(float(np.clip(v, task.T1_VALID_MS[0] + 1.0, task.T1_VALID_MS[1] - 1.0)), 1) for k, v in zip(ids, t1)}}


def _prior_answer(t):
    return _answer(_ids(t), [math.sqrt(t["t1_range_ms"][0] * t["t1_range_ms"][1])] * len(t["regions"]))


def _scan(sc, t, tr_ms, ti_ms, matrix=PROBE["matrix"]):
    """One inversion-recovery scan: {region id: complex mean}."""
    r = sc.scan(dict(PROBE, matrix=matrix, tr_ms=tr_ms, ti_ms=ti_ms), rois=t["regions"])
    return {row["id"]: row["complex_mean_magnitude"][0] * np.exp(1j * row["complex_mean_phase_rad"][0]) for row in r["roi_readouts"]}


def _measure(sc, t, tr_ms, tis, matrix=PROBE["matrix"]):
    """The scans of a list: {region id: (signed read-outs, magnitudes)}, in the order of ``tis``. The sign of a read-out
    is that of its phase relative to the scan with the longest inversion time (positive there)."""
    reads = [_scan(sc, t, tr_ms, ti, matrix) for ti in tis]
    last = reads[int(np.argmax(tis))]
    return {i: (np.array([float((r[i] * np.conj(last[i] / abs(last[i]))).real) for r in reads]), np.array([abs(r[i]) for r in reads]))
            for i in _ids(t)}


def solve_three(tis, s) -> float:
    """Level 1, to the letter: the T1 at which (E2 - E3) / (E1 - E2), E_k = exp(-TI_k / T1), equals the measured
    q = (S3 - S2) / (S2 - S1); bisection between 50 and 5000 ms (the ratio grows with T1)."""
    t1, t2, t3 = tis
    q = (s[2] - s[1]) / (s[1] - s[0]) if s[1] != s[0] else float("inf")

    def f(t):
        return (math.exp(-t2 / t) - math.exp(-t3 / t)) / (math.exp(-t1 / t) - math.exp(-t2 / t))
    lo, hi = 50.0, 5000.0
    if not q > f(lo):
        return lo
    if not q < f(hi):
        return hi
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if f(mid) < q else (lo, mid)
    return math.sqrt(lo * hi)


def fit_three_parameters(tis, s) -> float:
    """Least squares of s = a - b exp(-TI / T1) over T1 on a grid (a and b are linear, both positive). Returns T1."""
    ti, s = np.asarray(tis, float), np.asarray(s, float)
    e = np.exp(-ti[None, :] / T1_GRID_MS[:, None])
    n, se, see, ss, ses = len(ti), e.sum(1), (e * e).sum(1), s.sum(), e @ s
    det = n * see - se * se
    a, b = (see * ss - se * ses) / det, (se * ss - n * ses) / det
    sse = ((a[:, None] - b[:, None] * e - s[None, :]) ** 2).sum(1)
    sse = np.where((a > 0) & (b > 0), sse, np.inf)
    return float(T1_GRID_MS[int(np.argmin(sse))])


def fit_full_recovery(tis, s) -> float:
    """Least squares of the textbook curve s = M (1 - 2 exp(-TI / T1)) over T1 on a grid (M is linear). Returns T1."""
    ti, s = np.asarray(tis, float), np.asarray(s, float)
    m = 1.0 - 2.0 * np.exp(-ti[None, :] / T1_GRID_MS[:, None])
    scale = (m @ s) / (m * m).sum(1)
    sse = ((scale[:, None] * m - s[None, :]) ** 2).sum(1)
    return float(T1_GRID_MS[int(np.argmin(np.where(scale > 0, sse, np.inf)))])


def _play(sc, design=None, estimator=fit_three_parameters, signed=True):
    t = _info(sc)["task"]
    tr, tis, matrix = design or DESIGN[_level(t)]
    data = _measure(sc, t, tr, tis, matrix)
    return _answer(_ids(t), [estimator(tis, data[i][0] if signed else data[i][1]) for i in _ids(t)])


def reference(sc):
    t = _info(sc)["task"]
    if t.get("method"):                                                  # level 1: the stated scans and the stated equation
        m = t["method"]
        data = _measure(sc, t, m["protocol"]["tr_ms"], m["ti_ms"])
        return _answer(_ids(t), [solve_three(m["ti_ms"], data[i][0]) for i in _ids(t)])
    return _play(sc)


def open_loop(sc):
    return _play(sc)


def unsigned(sc):
    return _play(sc, signed=False)


def full_recovery(sc):
    return _play(sc, estimator=fit_full_recovery)


def bunched_tis(sc):
    t = _info(sc)["task"]
    return _play(sc, design=(DESIGN[_level(t)][0], BUNCHED_TI_MS, DESIGN[_level(t)][2]))


def level1_scans(sc):
    return _play(sc, design=DESIGN[1])


def prior(sc):
    return _prior_answer(_info(sc)["task"])


def _token_scan(sc):
    sc.scan(dict(PROBE, tr_ms=300.0, ti_ms=100.0))


def fixed_recipe(recipe):
    def policy(sc):
        t = _info(sc)["task"]
        _token_scan(sc)
        return _answer(_ids(t), [recipe["t1_ms"]] * len(t["regions"]))
    return policy


DEFAULT_RECIPE = dict(t1_ms=round(math.sqrt(task.NARROW_MS[0] * task.NARROW_MS[1]), 1))


def brute_force(sc):
    """The null search: no fit, no model of the recovery between repetitions or of the inversion efficiency."""
    t = _info(sc)["task"]
    p = sc.plan(dict(PROBE, tr_ms=BRUTE_TR_MS, ti_ms=50.0))
    n = int(min(max(math.floor(sc.budget()["remaining_s"] / p["scan_s"]), 0), 12))
    if n < 2:
        return _prior_answer(t)
    tis = [round(float(v), 1) for v in np.geomspace(20.0, BRUTE_TR_MS - p["realised"]["te_ms"] - 15.0, n)]
    reads = [_scan(sc, t, BRUTE_TR_MS, ti) for ti in tis]
    return _answer(_ids(t), [tis[int(np.argmin([abs(r[i]) for r in reads]))] / math.log(2.0) for i in _ids(t)])


def wrong_units(sc):
    return {"t1_ms": {k: v / 1000.0 for k, v in reference(sc)["t1_ms"].items()}}


def policies(recipe=None):
    return dict(reference=reference, prior=prior, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE), brute_force=brute_force,
                wrong_units=wrong_units, unsigned=unsigned, full_recovery=full_recovery, bunched_tis=bunched_tis,
                level1_scans=level1_scans, open_loop=open_loop)


def tune(cfgs):
    """The strongest answer that ignores the object, on other worlds: the one constant T1 that brings the most regions
    within the tolerance (the verifier's own rule, per region), and the share of worlds it passes."""
    truths = [task.true_t1_ms(c) for c in cfgs]
    limit = math.log(1.0 + cfgs[0]["task"]["tolerance"])
    values = np.array([v for t in truths for v in t.values()])
    lo, hi = cfgs[0]["task"]["t1_range_ms"]
    best = max(np.geomspace(lo, hi, 200), key=lambda c: float(np.mean(np.abs(np.log(c / values)) <= limit)))
    passed = sum(all(abs(math.log(best / v)) <= limit for v in t.values()) for t in truths)
    return dict(t1_ms=round(float(best), 1), tuned_region_pass_rate=round(float(np.mean(np.abs(np.log(best / values)) <= limit)), 3),
                tuned_pass_rate=round(passed / len(truths), 3))
