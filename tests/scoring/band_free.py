"""Scoring table of ``Scanner/BandFree-v0`` (format: ``tests/scoring/__init__.py``).

Pass rule (task statement), for each of the two images:
  1  every marked region keeps at least 80 % of its signal at its best phase cycle, at the present centre frequency and
     at every drift of the centre frequency up to 20 Hz either way; with two acquisitions, in one and the same of them
  2  two acquisitions only when no single acquisition at 400 Hz per pixel or less is band-free
  3  bandwidth at most 1.15 times the lowest band-free bandwidth with the same number of acquisitions
  (the comparison protocols of rules 2 and 3 stay band-free up to a drift of 26 Hz)

Where the expected values come from. Not from the task module and not from the engine's signal code: from the
family's independent oracle (``mrgym.envs.band_free.oracle``), which reads the true field at the pixels of every
region and the true material, writes the balanced-SSFP steady state down in closed form and integrates it over the
slice with the 3 x 3 rotation cascade of ``mrgym.scanner.bloch``; it shares no code with ``task`` or with the
engine's ``rf`` / ``kspace``. Every deliverable of the table is built from the oracle's sets of band-free phase
cycles with a margin on the side the row needs, so that the verdict follows from the rule and not from a tie:

  best             the oracle's own protocols: just above the lowest band-free bandwidth, where the drift margin is widest
  half a turn      the same bandwidth with the phase cycle moved by 180 degrees: the regions sit in the band
  edge of a band   a phase cycle that the oracle finds band-free without drift and lost at a drift of 14 Hz
  too wide         2.2 times the lowest band-free bandwidth with a phase cycle that is band-free there up to 29 Hz
  needless pair    two phase cycles at the lowest bandwidth of a pair where one acquisition is band-free below 400
  one of two       level 3, the image with the oil region, which needs two acquisitions: the first of the oracle's two
                   phase cycles alone
"""
import functools

import numpy as np

from mrgym.envs.band_free import oracle
from scoring import Row

ENV_ID = "Scanner/BandFree-v0"
INSTANCES = {"L2": dict(level=2, seed=2), "L3": dict(level=3, seed=7)}
KEEP_MIN, DRIFT_HZ, REFERENCE_DRIFT_HZ, SLACK, SINGLE_UP_TO = 0.8, 20.0, 26.0, 1.15, 400.0     # the numbers of the task statement
LOWEST_RTOL = 0.06              # verifier (2 % grid, amplitudes) against oracle (1 % grid, closed form): lowest band-free bandwidths
TOP = 600.0                     # the top of the allowed bandwidths
OTHER_RTOL = 0.15               # ... of the kind that does not decide the image (a pair exists first at an isolated point of the grid)


@functools.cache
def _solved(key):
    case = _CASES[key]
    return oracle.solve(case.cfg, case.world), oracle.points(case.cfg, case.world)


_CASES = {}


def solved(case):
    key = (case.level, case.seed)
    _CASES[key] = case
    return _solved(key)


def _pts(case, image):
    return [solved(case)[1][i] for i in case.cfg["task"]["images"][image]]


def best(case):
    return {name: dict(s["answer"]) for name, s in solved(case)[0]["images"].items()}


def _with(case, image, **change):
    a = best(case)
    a[image] = dict(a[image], **change)
    return a


def half_a_turn(*images):
    def deliverable(case):
        a = best(case)
        for image in images:
            a[image] = dict(a[image], phase_cycle_deg=[(p + 180.0) % 360.0 for p in a[image]["phase_cycle_deg"]])
        return a
    return deliverable


def edge_of_a_band(image):
    """Band-free now, lost within the drift: the oracle's sets at no drift and at a drift of 14 Hz (inside the 20 Hz)."""
    def deliverable(case):
        bw = best(case)[image]["bandwidth_hz_per_px"]
        now = oracle.allowed(case.cfg, case.world, _pts(case, image), bw, 2.0).all(0)
        later = oracle.allowed(case.cfg, case.world, _pts(case, image), bw, 14.0).all(0)
        edge = np.flatnonzero(now & ~later)
        assert len(edge), "no phase cycle at the edge of a band on this instance"
        return _with(case, image, phase_cycle_deg=[float(edge[len(edge) // 2] * oracle.STEP_DEG)])
    return deliverable


def too_wide(image, factor=2.2):
    def deliverable(case):
        s = solved(case)[0]["images"][image]
        bw = round(factor * s["single_bandwidth"], 1)
        ok = oracle.allowed(case.cfg, case.world, _pts(case, image), bw, REFERENCE_DRIFT_HZ + 3.0).all(0)
        assert ok.any() and bw <= TOP
        return _with(case, image, bandwidth_hz_per_px=bw, phase_cycle_deg=[float(oracle._middle(ok) * oracle.STEP_DEG)])
    return deliverable


def needless_pair(image):
    def deliverable(case):
        s = solved(case)[0]["images"][image]
        assert s["single_bandwidth"] <= 0.8 * SINGLE_UP_TO and s["pair_bandwidth"] is not None
        for factor in (1.06, 1.03, 1.09, 1.12):
            bw = round(factor * s["pair_bandwidth"], 1)
            pair = oracle._pair(oracle.allowed(case.cfg, case.world, _pts(case, image), bw, REFERENCE_DRIFT_HZ + 2.0))
            if pair is not None:
                return _with(case, image, bandwidth_hz_per_px=bw, phase_cycle_deg=[float(i * oracle.STEP_DEG) for i in pair])
        raise AssertionError("no band-free pair near the lowest pair bandwidth on this instance")
    return deliverable


def one_of_two(image):
    def deliverable(case):
        a = best(case)
        assert len(a[image]["phase_cycle_deg"]) == 2, "this row needs an image with two acquisitions"
        return _with(case, image, phase_cycle_deg=a[image]["phase_cycle_deg"][:1])
    return deliverable


def agrees_with_the_oracle(case, score):
    """The verifier's lowest band-free bandwidths are the oracle's (two routes, two grids); it counts the acquisitions
    the oracle counts; every check is the rule applied to the numbers it reports."""
    truth, answer = solved(case)[0]["images"], score["answer"]
    assert set(score["images"]) == set(truth) == set(answer)
    for name, img in score["images"].items():
        low, o = img["lowest_bandwidth"], truth[name]
        for kind, n_acq in (("single_bandwidth", 1), ("pair_bandwidth", 2)):     # the kind that decides this image, and the other one
            if low[kind] is not None and o[kind] is not None:
                rtol = LOWEST_RTOL if (n_acq == 2) == o["needs_two"] else OTHER_RTOL
                assert abs(low[kind] / o[kind] - 1.0) <= rtol, (name, kind, low[kind], o[kind])
        needs_two = low["single_bandwidth"] is None or low["single_bandwidth"] > SINGLE_UP_TO
        assert needs_two == o["needs_two"], name
        n = len(answer[name]["phase_cycle_deg"])
        assert img["acquisitions"] == n and img["bandwidth_hz_per_px"] == answer[name]["bandwidth_hz_per_px"]
        worst = min(min(k["now"], k["drift"]) for k in img["keep"].values())
        assert img["keep_worst"] == worst and score["checks"][f"band_free_{name}"] is (worst >= KEEP_MIN)
        assert score["checks"][f"acquisitions_needed_{name}"] is (n == 1 or needs_two)
        lowest = low["single_bandwidth"] if n == 1 else low["pair_bandwidth"]
        assert score["checks"][f"bandwidth_low_{name}"] is (lowest is None or answer[name]["bandwidth_hz_per_px"] <= SLACK * lowest * (1 + 1e-9))
        assert set(img["keep"]) == set(case.cfg["task"]["images"][name])
    assert score["keep_worst"] == min(img["keep_worst"] for img in score["images"].values())
    assert score["error"] == round(1.0 - score["keep_worst"], 5)
    assert set(score["checks"]) == {"deliverable_valid"} | {f"{c}_{n}" for n in truth for c in ("band_free", "acquisitions_needed", "bandwidth_low")}


def lost_within_the_drift(case, score):
    """The edge row: fine at the present centre frequency (the scans), lost before the drift reaches 20 Hz (the amplitudes)."""
    agrees_with_the_oracle(case, score)
    keep = score["images"]["A"]["keep"]
    assert min(k["now"] for k in keep.values()) >= KEEP_MIN > min(k["drift"] for k in keep.values())


def not_evaluated(case, score):
    assert score["checks"] == dict(deliverable_valid=False) and score["error"] is None and "images" not in score


SHAPE = 'the deliverable is {"A": ..., "B": ...} with {"bandwidth_hz_per_px": <number>, "phase_cycle_deg": [<one or two numbers>]} for each image'
ROWS = [
    # ---- level 2: both images band-free with one acquisition
    Row("L2", "the oracle's protocols", best, passed=True, expect=agrees_with_the_oracle),
    Row("L2", "image A half a turn of the phase cycle away", half_a_turn("A"), passed=False, failing=("band_free_A",), reason="image A: R",
        expect=agrees_with_the_oracle),
    Row("L2", "image B half a turn away", half_a_turn("B"), passed=False, failing=("band_free_B",), reason="image B: R", expect=agrees_with_the_oracle),
    Row("L2", "both images half a turn away", half_a_turn("A", "B"), passed=False, failing=("band_free_A", "band_free_B"), reason="; image B: R"),
    Row("L2", "image A at the edge of a band: fine now, lost within the drift", edge_of_a_band("A"), passed=False, failing=("band_free_A",),
        reason="within a drift of 20 Hz", expect=lost_within_the_drift),
    Row("L2", "image A band-free at 2.2 times the lowest bandwidth", too_wide("A"), passed=False, failing=("bandwidth_low_A",),
        reason="image A: a band-free protocol exists at", expect=agrees_with_the_oracle),
    Row("L2", "image B with two acquisitions where one is band-free", needless_pair("B"), passed=False, failing=("acquisitions_needed_B",),
        reason="image B: one acquisition is band-free at", expect=agrees_with_the_oracle),
    Row("L2", "integers are numbers", lambda c: {n: dict(a, phase_cycle_deg=[int(round(p)) for p in a["phase_cycle_deg"]]) for n, a in best(c).items()},
        passed=True),
    # ---- level 3: one region of image B holds oil; no single acquisition is band-free for it at any allowed bandwidth
    Row("L3", "the oracle's protocols: two acquisitions for image B", best, passed=True, expect=agrees_with_the_oracle),
    Row("L3", "one of the two acquisitions of image B alone", one_of_two("B"), passed=False, failing=("band_free_B",), reason="image B: R",
        expect=agrees_with_the_oracle),
    Row("L3", "image A half a turn away", half_a_turn("A"), passed=False, failing=("band_free_A",), reason="image A: R", expect=agrees_with_the_oracle),
    # ---- deliverables the console cannot evaluate: refused, the episode continues
    Row("L2", "the bandwidth in kHz per pixel", lambda c: _with(c, "A", bandwidth_hz_per_px=best(c)["A"]["bandwidth_hz_per_px"] / 1000.0),
        refused=True, reason="image A: bandwidth_hz_per_px=", expect=not_evaluated),
    Row("L2", "a bandwidth above the allowed range", lambda c: _with(c, "B", bandwidth_hz_per_px=1200.0), refused=True,
        reason="image B: bandwidth_hz_per_px=1200 is outside the allowed range 60 to 600 Hz per pixel"),
    Row("L2", "a phase cycle beyond 360 degrees", lambda c: _with(c, "B", phase_cycle_deg=[400.0]), refused=True,
        reason="image B: phase_cycle_deg must lie between 0 and 360 degrees"),
    Row("L2", "three phase cycles", lambda c: _with(c, "A", phase_cycle_deg=[10.0, 130.0, 250.0]), refused=True, reason=SHAPE),
    Row("L2", "the phase cycle as a number, not a list", lambda c: _with(c, "A", phase_cycle_deg=180.0), refused=True, reason=SHAPE),
    Row("L2", "a bandwidth that is not a number", lambda c: _with(c, "A", bandwidth_hz_per_px="low"), refused=True, reason=SHAPE),
    Row("L2", "one image missing", lambda c: {"A": best(c)["A"]}, refused=True, reason=SHAPE),
    Row("L2", "a third image", lambda c: dict(best(c), C=best(c)["A"]), refused=True, reason=SHAPE),
    Row("L2", "one protocol for both, not per image", lambda c: best(c)["A"], refused=True, reason=SHAPE),
    Row("L2", "nothing", None, refused=True, reason=SHAPE, expect=not_evaluated),
]
