"""Scoring table of ``Scanner/FatSuppress-v0`` (format: ``tests/scoring/__init__.py``).

Pass rule (task statement), for every marked region, on the target spin echo with the delivered method:
  1  suppressed: its signal is at most 5 % of its signal in the same spin echo without suppression
  2  no water signal given away: ``stir`` passes only where fat saturation leaves more than 3 % of the region's signal

The expected values are derived here from the hidden truth, not taken from the task module and not from the engine's
signal code:

  fat saturation   engine 0.3.0. The pulse multiplies the longitudinal magnetisation of every line of the fat spectrum
                   (datasheet: six lines) by q = 1 - ``fatsat_eff`` x band, 7 ms before every excitation: the band is
                   ``fatsat_bw_hz`` wide around the main fat line moved by ``fatsat_offset_hz``, with an edge of 8 Hz
                   (0.5 (1 - tanh((|off| - bw / 2) / 8))); ``off`` is the distance of that line from the centre of the
                   pulse where the fat of that line that appears in the region really is: the image shows it displaced
                   along the readout by (off-resonance + its chemical shift) / 300 Hz per pixel. With exact 90 and 180
                   degree pulses the 90 degree pulse leaves no Mz, m of it has recovered when the saturation pulse is
                   played, and 1 - Es grows back in the 7 ms after it whatever q is: a line keeps
                   (q m Es + 1 - Es) / (m Es + 1 - Es) of its signal. The region's value is the sum over the lines of
                   share x the mean of that over a 1 mm grid. The lines outside the band keep everything, so no band
                   of this family brings a region below the 5 % of the pass rule: fat saturation passes nowhere.
  inversion        the ratio of the inversion-recovery echo to the plain spin echo of a 2D slice: the pulse chain of
                   ``tests/scoring/null_tissue.py`` (the documented selective pulse as a cascade of rotations, the
                   refocusing slice 1.5 times as thick, the longitudinal magnetisation walked through the repetition
                   until it repeats), with the T1 of this vial's fat, the hidden inversion efficiency and the flip
                   angle at the region. The null is where that ratio changes sign (bisection).

The verifier's numbers are compared with these: what fat saturation leaves within 0.03 + 15 % (the verifier scans a
2D slice with its profile, and the edge of the band crosses some regions), the inversion ratio within ``RATIO_ATOL``.
Rows are placed away from the thresholds, so the verdicts follow from the rule.

On engine 0.2.0 a saturated region kept 1 - ``fatsat_eff`` of its signal (0.2 to 2.5 %), so the best answer mixed the
two methods and the second pass rule decided rows. The rows that needed a region where fat saturation is enough are
gone: rule 2 cannot fail on this engine with the thresholds of the task statement (``tests/test_fat_suppress.py``
checks the rule itself with other thresholds).
"""
import math

import numpy as np

from scoring import Row
from scoring import null_tissue as chain

ENV_ID = "Scanner/FatSuppress-v0"
INSTANCES = {"L1": dict(level=1, seed=0), "L3": dict(level=3, seed=2)}
FAT_MAX, FAT_SAT_SUFFICIENT = 0.05, 0.03                # the two thresholds of the task statement
SWITCH = 0.04                                            # between them both methods pass
TE_MS = 8.4                                              # console minimum of the target scan (matrix 128, 300 Hz/px), as in null_tissue
PIXEL_MM, BANDWIDTH_HZ_PER_PX = 240.0 / 128, 300.0      # the target geometry of the task statement
HZ_PER_PPM = 1e-6 * 42.577478518e6 * 3.0                # at 3 T
FAT_PPM = -3.4                                           # the main fat line against water: the nominal centre of the pulse
FAT_LINES = ((-3.80, 0.087), (-3.40, 0.694), (-2.60, 0.128), (-1.94, 0.004), (-0.39, 0.039), (0.60, 0.048))   # ppm, share (datasheet)
SAT_DELAY_MS = 7.0                                       # centre of the saturation pulse to centre of the excitation (datasheet)
EDGE_HZ = 8.0
RATIO_ATOL = 4e-3
NEAREST, FAILS = "R2", "R3"                              # L1: the region where fat saturation leaves least, and one where it clearly fails
L3_TEXTBOOK_FAILS = ("R1", "R3", "R4", "R5", "R6")                     # L3: the regions where ln 2 x T1 leaves more than 5 % (all of them)


def ids(case):
    return [r["id"] for r in case.cfg["task"]["regions"]]


def fat_sat_left(case, region):
    """What fat saturation leaves of a region's signal (hidden truth; see the module docstring)."""
    w, sy = case.world, case.world["system"]
    n, d = w["n"], w["extent_mm"] / w["n"]
    r = next(r for r in case.cfg["task"]["regions"] if r["id"] == region)
    g = np.arange(-r["radius_mm"], r["radius_mm"] + 1e-9, 1.0)
    yy, xx = np.meshgrid(g, g, indexing="ij")
    inside = yy ** 2 + xx ** 2 <= r["radius_mm"] ** 2
    y, x0 = r["y_mm"] + yy[inside], r["x_mm"] + xx[inside]

    def off_resonance(x):
        i = np.clip(np.rint(y / d).astype(int) + n // 2, 0, n - 1)
        j = np.clip(np.rint(x / d).astype(int) + n // 2, 0, n - 1)
        return w["fields"]["b0_hz"][i, j] + sy["cf_offset_hz"]
    t1, tr = case.truth[region]["t1_ms"], case.cfg["task"]["target_scan"]["tr_ms"]
    es = math.exp(-SAT_DELAY_MS / t1)
    m = 1.0 - (2.0 - math.exp(-TE_MS / 2 / t1)) * math.exp(-(tr - TE_MS / 2 - SAT_DELAY_MS) / t1)      # recovered before the saturation pulse
    left = 0.0
    for ppm, share in FAT_LINES:
        x = x0
        for _ in range(3):                                               # where the fat of this line seen at this pixel really is
            x = x0 + (off_resonance(x) + ppm * HZ_PER_PPM) / BANDWIDTH_HZ_PER_PX * PIXEL_MM
        off = off_resonance(x) + (ppm - FAT_PPM) * HZ_PER_PPM - sy["fatsat_offset_hz"]
        q = 1.0 - sy["fatsat_eff"] * 0.5 * (1.0 - np.tanh((np.abs(off) - sy["fatsat_bw_hz"] / 2) / EDGE_HZ))
        left += share * float(np.mean((q * m * es + 1 - es) / (m * es + 1 - es)))
    return left / sum(share for _, share in FAT_LINES)


def stir_ratio(case, region, ti):
    """Signed signal of ``region`` at inversion time ``ti`` over its signal in the plain spin echo (hidden truth)."""
    t, tr = case.truth[region], case.cfg["task"]["target_scan"]["tr_ms"]
    a = math.pi / 2 * t["b1"]
    return chain.slice_signal(a, tr, TE_MS, t["t1_ms"], ti, case.world["system"]["inv_eff"]) / chain.slice_signal(a, tr, TE_MS, t["t1_ms"])


def null_ms(case, region):
    lo, hi = 20.0, 1000.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if stir_ratio(case, region, mid) < 0 else (lo, mid)
    return round(0.5 * (lo + hi), 2)


def best(case):
    """Fat saturation where it leaves less than the middle of the two thresholds, inversion recovery at the null elsewhere."""
    return {i: dict(method="fat_sat") if fat_sat_left(case, i) < SWITCH else dict(method="stir", ti_ms=null_ms(case, i)) for i in ids(case)}


def with_(case, **change):
    a = best(case)
    for region, entry in change.items():
        a[region] = entry(case, region) if callable(entry) else entry
    return a


def stir_at(factor):
    return lambda case, region: dict(method="stir", ti_ms=round(factor * null_ms(case, region), 1))


def textbook(case):
    """ln 2 x T1 wherever the best answer uses inversion recovery: right only for a complete inversion."""
    return {i: e if e["method"] == "fat_sat" else dict(method="stir", ti_ms=round(math.log(2.0) * case.truth[i]["t1_ms"], 1))
            for i, e in best(case).items()}


def agrees_with_the_truth(case, score):
    """The verifier's two numbers per region are the ones derived here; every check is the rule applied to them; the
    reported error is the largest delivered residual."""
    answer = score["answer"]
    assert set(score["residual"]) == set(ids(case)) == set(answer)
    for i in ids(case):
        left, got = fat_sat_left(case, i), score["residual"][i]
        assert abs(got["fat_sat"] - left) <= 0.03 + 0.15 * left, (i, got, left)
        if answer[i]["method"] == "stir":
            want = abs(stir_ratio(case, i, answer[i]["ti_ms"]))
            assert abs(got["delivered"] - want) <= RATIO_ATOL, (i, got, want)
            assert (got["delivered"] <= FAT_MAX) == (want <= FAT_MAX)
        else:
            assert got["delivered"] == got["fat_sat"]
        assert score["checks"][f"suppressed_{i}"] is (got["delivered"] <= FAT_MAX)
        assert score["checks"][f"method_{i}"] is (answer[i]["method"] == "fat_sat" or got["fat_sat"] > FAT_SAT_SUFFICIENT)
    assert score["error"] == round(max(r["delivered"] for r in score["residual"].values()), 5) and score["fat_max"] == FAT_MAX
    assert score["methods"] == {i: e["method"] for i, e in answer.items()}
    assert set(score["checks"]) == {"deliverable_valid"} | {f"{c}_{i}" for i in ids(case) for c in ("suppressed", "method")}


def fat_saturation_is_enough_nowhere(case, score):
    """The premise of the level-1 rows on engine 0.3.0: even where the pulse sits on the main lines (the named region),
    the lines outside its band keep more than a fifth of the signal; in the other named region it misses altogether.
    The best answer is inversion recovery in every region."""
    agrees_with_the_truth(case, score)
    left = {i: fat_sat_left(case, i) for i in ids(case)}
    assert min(left, key=left.get) == NEAREST and 0.15 < left[NEAREST] < 0.27 and left[FAILS] > 0.30
    assert all(v > FAT_MAX for v in left.values()) and all(e["method"] == "stir" for e in score["answer"].values())


def not_evaluated(case, score):
    assert score["checks"] == dict(deliverable_valid=False) and score["error"] is None and "residual" not in score


SHAPE = "the deliverable is {"
ROWS = [
    # ---- level 1: an efficient pulse of the nominal band and a nearly complete inversion
    Row("L1", "inversion recovery at the null in every region: fat saturation is enough nowhere", best, passed=True,
        expect=fat_saturation_is_enough_nowhere),
    Row("L1", "the null 3 % late: about 2 % left", lambda c: with_(c, **{FAILS: stir_at(1.03)}), passed=True, expect=agrees_with_the_truth),
    Row("L1", "fat saturation where its band holds the main lines: the lines outside it stay", lambda c: with_(c, **{NEAREST: dict(method="fat_sat")}),
        passed=False, failing=(f"suppressed_{NEAREST}",), reason=f"{NEAREST} keeps", expect=agrees_with_the_truth),
    Row("L1", "fat saturation where it leaves a third and more", lambda c: with_(c, **{FAILS: dict(method="fat_sat")}), passed=False,
        failing=(f"suppressed_{FAILS}",), reason=f"{FAILS} keeps", expect=agrees_with_the_truth),
    Row("L1", "the null missed by a quarter", lambda c: with_(c, **{FAILS: stir_at(1.25)}), passed=False, failing=(f"suppressed_{FAILS}",),
        reason=f"{FAILS} keeps", expect=agrees_with_the_truth),
    Row("L1", "the null 20 % early", lambda c: with_(c, **{FAILS: stir_at(0.80)}), passed=False, failing=(f"suppressed_{FAILS}",),
        reason=f"{FAILS} keeps", expect=agrees_with_the_truth),
    Row("L1", "both mistakes at once", lambda c: with_(c, **{NEAREST: dict(method="fat_sat"), FAILS: stir_at(1.25)}), passed=False,
        failing=(f"suppressed_{NEAREST}", f"suppressed_{FAILS}"), reason="keeps", expect=agrees_with_the_truth),
    # ---- level 3: an incomplete inversion
    Row("L3", "inversion recovery at the null in every region", best, passed=True, expect=agrees_with_the_truth),
    Row("L3", "the textbook null ln 2 x T1 ignores the incomplete inversion", textbook, passed=False,
        failing=tuple(f"suppressed_{i}" for i in L3_TEXTBOOK_FAILS), reason="keeps", expect=agrees_with_the_truth),
    # ---- deliverables the console cannot evaluate: refused, the episode continues
    Row("L1", "the inversion time in seconds", lambda c: with_(c, **{FAILS: lambda case, r: dict(method="stir", ti_ms=null_ms(case, r) / 1000.0)}),
        refused=True, reason=f"the console rejects the target scan of {FAILS}: ti_ms=", expect=not_evaluated),
    Row("L1", "inversion recovery without an inversion time", lambda c: with_(c, **{FAILS: dict(method="stir")}), refused=True,
        reason=f"{FAILS}: stir needs ti_ms, a number in milliseconds"),
    Row("L1", "fat saturation with an inversion time", lambda c: with_(c, **{NEAREST: dict(method="fat_sat", ti_ms=230.0)}), refused=True,
        reason=f"{NEAREST}: fat_sat has no inversion time"),
    Row("L1", "an inversion time the repetition time has no room for", lambda c: with_(c, **{FAILS: dict(method="stir", ti_ms=5000.0)}), refused=True,
        reason=f"the console rejects the target scan of {FAILS}: "),
    Row("L1", "a method the console does not have", lambda c: with_(c, **{FAILS: dict(method="dixon")}), refused=True, reason=SHAPE),
    Row("L1", "the method as a word, not an entry", lambda c: with_(c, **{NEAREST: "fat_sat"}), refused=True, reason=SHAPE),
    Row("L1", "a region is missing", lambda c: {i: e for i, e in best(c).items() if i != FAILS}, refused=True, reason=SHAPE),
    Row("L1", "a region too many", lambda c: dict(best(c), R99=dict(method="fat_sat")), refused=True, reason=SHAPE),
    Row("L1", "one method for all, not per region", dict(method="fat_sat"), refused=True, reason=SHAPE),
    Row("L1", "nothing", None, refused=True, reason=SHAPE, expect=not_evaluated),
]
