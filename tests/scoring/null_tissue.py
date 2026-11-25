"""Scoring table of ``Scanner/NullTissue-v0`` (format: ``tests/scoring/__init__.py``).

Pass rule (task statement): for every marked region, in the target inversion-recovery scan with the delivered inversion
time, the signal is at most 5 % of the signal of the same scan without the inversion pulse.

The expected ratio is derived here, not taken from the task module and not from the engine's signal code.

Slice-aware, and why. The target scan is fixed by the task (``sequence ir`` at the console defaults) and the
deliverable holds inversion times only, so the verifier always scores the default excitation: slice-selective
(engine 0.2.0). A row cannot ask for ``excitation: nonselective``; a hard-pulse oracle would describe a scan the task
never scores. The oracle therefore integrates over the slice profile, with its own implementation of the documented
pulse (docs/scanner.md, "Slice selection"):

  pulse       128 hard sub-pulses with the amplitudes of a sinc of time-bandwidth product 4 under a Hann window, summing
              to the flip angle at the slice centre (a = 90 deg x B1); precession about z between them in proportion to
              the height in the slice; for the excitation a rephasing lobe of half the selection area. Simulated here
              as a cascade of 3 x 3 rotations of the magnetisation vector (the engine uses Cayley-Klein parameters).
  refocusing  the same shape at twice the angle on a slice 1.5 times as thick, between crushers: the echo keeps the
              share sin^2(beta / 2) of the transverse magnetisation and Mz is multiplied by cos(beta), beta the
              rotation angle at that height.
  column      at every height the longitudinal magnetisation is walked through the chain
                  excitation - TE/2 - refocusing - (TR - TI - TE/2) - inversion (Mz -> -k Mz, not selective) - TI
              for many repetitions until it repeats (a plain iteration, no closed form).
  signal      the sum over heights (40 per slice thickness out to 3.5 thicknesses; the engine uses 24) of
              Mz before the excitation x transverse magnetisation the excitation makes x echo share.

The ratio of the scan with the inversion to the scan without it is compared with the verifier's. They agree within
``RATIO_ATOL``: the transmit field varies by a fraction of a percent across a region and the two quadratures differ
(measured on these instances on engine 0.2.0: at most 1.2e-3).
"""
import functools
import math

import numpy as np
import pytest

from scoring import Row

ENV_ID = "Scanner/NullTissue-v0"
INSTANCES = {"L0": dict(level=0, seed=2), "L1": dict(level=1, seed=0), "L2": dict(level=2, seed=0), "L3": dict(level=3, seed=1)}
NULL_MAX = 0.05                 # the pass threshold of the task statement
RATIO_ATOL = 2.5e-3
TE_MS = 8.4                     # console minimum of the target scan (matrix 128, 300 Hz/px), asserted in tests/test_null_tissue.py
TBW, SUB_PULSES, REFOCUS_WIDTH = 4.0, 128, 1.5      # the console's default pulse ("normal"), asserted there too
HEIGHTS = np.arange(-140, 141) / 40.0               # slice thicknesses: 40 per thickness out to 0.5 + 12 / TBW = 3.5


def _pulse(alpha, zeta, rephase):
    """Magnetisation (mx, my, mz) that the windowed-sinc pulse of centre angle ``alpha`` (rad) makes of Mz = 1 at heights
    ``zeta`` (slice thicknesses); ``rephase``: the share of the selection area that is rewound afterwards."""
    t = (np.arange(SUB_PULSES) + 0.5) / SUB_PULSES - 0.5
    amp = np.sinc(TBW * t) * 0.5 * (1.0 + np.cos(2 * np.pi * t))
    amp = amp / amp.sum()
    step = 2 * np.pi * TBW * zeta / SUB_PULSES                           # precession per sub-pulse at each height
    mx, my, mz = np.zeros_like(zeta), np.zeros_like(zeta), np.ones_like(zeta)

    def about_z(mx, my, phi):
        return mx * np.cos(phi) + my * np.sin(phi), my * np.cos(phi) - mx * np.sin(phi)
    for w in amp:
        mx, my = about_z(mx, my, step / 2)
        c, sn = math.cos(w * alpha), math.sin(w * alpha)
        my, mz = my * c + mz * sn, mz * c - my * sn                      # the sub-pulse turns the vector about x
        mx, my = about_z(mx, my, step / 2)
    mx, my = about_z(mx, my, -rephase * step * SUB_PULSES)
    return mx, my, mz


@functools.lru_cache(maxsize=64)
def _slice(a):
    """Per height: cos of the excitation angle, the transverse magnetisation along the direction of the slice centre,
    cos of the refocusing angle."""
    mx, my, mz = _pulse(a, HEIGHTS, 0.5)
    centre = len(HEIGHTS) // 2
    norm = math.hypot(mx[centre], my[centre])
    along = (mx * mx[centre] + my * my[centre]) / norm
    cos_beta = _pulse(2 * a, HEIGHTS / REFOCUS_WIDTH, 0.0)[2]
    return mz, along, cos_beta


def slice_signal(a, tr, te, t1, ti=None, kappa=1.0, repetitions=60):
    """Signal of the slice per unit thickness for a flip angle ``a`` (rad) at its centre. ``ti`` None: no inversion pulse."""
    cos_theta, along, cos_beta = _slice(round(float(a), 12))
    m = np.ones_like(HEIGHTS)                                            # Mz just before the excitation, at every height
    for _ in range(repetitions):
        m = m * cos_theta
        m = 1.0 + (m - 1.0) * math.exp(-(te / 2) / t1)
        m = m * cos_beta
        if ti is None:
            m = 1.0 + (m - 1.0) * math.exp(-(tr - te / 2) / t1)
        else:
            m = 1.0 + (m - 1.0) * math.exp(-(tr - ti - te / 2) / t1)
            m = -kappa * m
            m = 1.0 + (m - 1.0) * math.exp(-ti / t1)
    columns = m * along * (1.0 - cos_beta) / 2.0                         # echo share sin^2(beta / 2) = (1 - cos beta) / 2
    return float((columns[:-1] + columns[1:]).sum() / 2.0 * (HEIGHTS[1] - HEIGHTS[0]))


def ratio(case, region, ti, kappa=None):
    """Signed signal of ``region`` at inversion time ``ti`` over its signal without the inversion (hidden truth)."""
    t, tr = case.truth[region], case.cfg["task"]["target_scan"]["tr_ms"]
    a = math.pi / 2 * t["b1"]
    k = case.world["system"]["inv_eff"] if kappa is None else kappa
    return slice_signal(a, tr, TE_MS, t["t1_ms"], ti, k) / slice_signal(a, tr, TE_MS, t["t1_ms"])


def null_ms(case, region, kappa=None):
    """The inversion time at which ``ratio`` changes sign (bisection)."""
    lo, hi = 6.5, case.cfg["task"]["target_scan"]["tr_ms"] - TE_MS - 30.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ratio(case, region, mid, kappa) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def ids(case):
    return [r["id"] for r in case.cfg["task"]["regions"]]


def at(factor=1.0, kappa=None, **per_region):
    """Deliverable: every region at ``factor`` times its null; ``R1=0.9`` overrides the factor of one region."""
    return lambda case: {"ti_ms": {i: round(null_ms(case, i, kappa) * per_region.get(i, factor), 2) for i in ids(case)}}


def textbook(case):
    """TI = ln 2 x T1: right only for a complete inversion and a repetition time much longer than T1."""
    return {"ti_ms": {i: round(math.log(2.0) * case.truth[i]["t1_ms"], 2) for i in ids(case)}}


def agrees_with_the_chain(case, score):
    """The verifier's ratio of every region is the ratio of the pulse chain; the largest one is the reported error; the
    checks are one per region plus the validity of the deliverable; the diagnostic names what each failing region keeps."""
    answer = score["answer"]
    assert list(score["null_ratio"]) == list(answer["ti_ms"])
    for i in ids(case):
        want = abs(ratio(case, i, answer["ti_ms"][i]))
        assert abs(score["null_ratio"][i] - want) <= RATIO_ATOL, (i, score["null_ratio"][i], want)
        assert (score["null_ratio"][i] <= NULL_MAX) == (want <= NULL_MAX)
    assert score["error"] == max(score["null_ratio"].values()) and score["null_max"] == NULL_MAX
    assert score["checks"] == dict(deliverable_valid=True, **{f"nulled_{i}": v <= NULL_MAX for i, v in score["null_ratio"].items()})
    assert score["diagnostic"] == "; ".join(f"{i} keeps {100 * v:.1f} %" for i, v in score["null_ratio"].items() if v > NULL_MAX)
    # diagnostics for trainers (never paid): how many regions are nulled, and the documented progress of the answer
    left = [v for v in score["null_ratio"].values() if v > NULL_MAX]
    n = len(score["null_ratio"])
    assert score["distance"] == dict(regions=n, nulled=n - len(left), worst_ratio=max(score["null_ratio"].values()))
    assert score["progress"] == pytest.approx(progress_by_hand(score["null_ratio"]), abs=1e-12)
    assert (score["progress"] == 1.0) == (not left) and 0.0 <= score["progress"] <= 1.0


def progress_by_hand(ratios):
    """Module docstring of the task: with k of n regions nulled and c the mean closeness of the others, closeness =
    ln(0.5 / ratio) / ln(0.5 / 0.05) clipped to [0, 0.999], progress = 0.5 (k + c) / n; 1 when every region is nulled."""
    left = [v for v in ratios.values() if v > NULL_MAX]
    if not left:
        return 1.0
    close = [min(max(math.log(0.5 / v) / math.log(0.5 / NULL_MAX), 0.0), 1.0) for v in left]
    return 0.5 * (len(ratios) - len(left) + min(sum(close) / len(close), 0.999)) / len(ratios)


def not_evaluated(case, score):
    """A deliverable the console cannot evaluate: one failed check, no error value, nothing scanned."""
    assert score["checks"] == dict(deliverable_valid=False) and score["error"] is None and "null_ratio" not in score


def with_ti(case, **change):
    a = at()(case)
    a["ti_ms"].update(change)
    return a


CONSOLE = "the console rejects the target scan with the inversion time of "
SHAPE = 'the deliverable is {"ti_ms": {'
ROWS = [
    # ---- level 1: long repetition time, two regions
    Row("L1", "every region at its null", at(), passed=True, expect=agrees_with_the_chain),
    Row("L1", "3 % early and late: about 2 % signal left", at(R1=0.97, R2=1.03), passed=True, expect=agrees_with_the_chain),
    Row("L1", "one region 10 % early: about 7 % left", at(R1=0.90), passed=False, failing=("nulled_R1",), reason="R1 keeps",
        expect=agrees_with_the_chain),
    Row("L1", "the other region 10 % late", at(R2=1.10), passed=False, failing=("nulled_R2",), reason="R2 keeps",
        expect=agrees_with_the_chain),
    Row("L1", "both regions 10 % off", at(0.90), passed=False, failing=("nulled_R1", "nulled_R2"), reason="; R2 keeps",
        expect=agrees_with_the_chain),
    Row("L1", "the textbook rule is close enough at a long repetition time", textbook, passed=True, expect=agrees_with_the_chain),
    Row("L1", "integers are numbers", lambda c: {"ti_ms": {i: int(round(v)) for i, v in at()(c)["ti_ms"].items()}}, passed=True),
    # ---- level 0 (practice: the world of level 1 with hints): the same verifier
    Row("L0", "every region at its null", at(), passed=True, expect=agrees_with_the_chain),
    Row("L0", "both regions 10 % off", at(1.10), passed=False, failing=("nulled_R1", "nulled_R2"), reason="; R2 keeps", expect=agrees_with_the_chain),
    # ---- level 2: repetition time comparable to T1, three regions
    Row("L2", "every region at its null", at(), passed=True, expect=agrees_with_the_chain),
    Row("L2", "the textbook rule ignores the repetition time: wrong for the two long T1", textbook, passed=False,
        failing=("nulled_R1", "nulled_R3"), reason="; R3 keeps", expect=agrees_with_the_chain),
    # ---- level 3: incomplete inversion
    Row("L3", "every region at its null", at(), passed=True, expect=agrees_with_the_chain),
    Row("L3", "the textbook rule: wrong for the two long T1", textbook, passed=False, failing=("nulled_R2", "nulled_R3"),
        reason="R2 keeps", expect=agrees_with_the_chain),
    Row("L3", "one region 10 % late", at(R3=1.10), passed=False, failing=("nulled_R3",), reason="R3 keeps", expect=agrees_with_the_chain),
    # ---- deliverables the console cannot evaluate: refused, the episode continues
    Row("L1", "the right numbers in seconds", lambda c: {"ti_ms": {i: v / 1000.0 for i, v in at()(c)["ti_ms"].items()}},
        refused=True, reason=CONSOLE + "R1: ti_ms=", expect=not_evaluated),
    Row("L1", "an inversion time the repetition time has no room for", lambda c: with_ti(c, R2=5995.0),
        refused=True, reason=CONSOLE + "R2: tr_ms=6000.0 is below the minimum"),
    Row("L1", "below the shortest inversion time of the console", lambda c: with_ti(c, R1=5.0),
        refused=True, reason=CONSOLE + "R1: ti_ms=5.0 is below the minimum 6.04 ms"),
    Row("L1", "a value that is not a number", lambda c: with_ti(c, R2="soon"), refused=True, reason=CONSOLE + "R2: ti_ms must be a finite number"),
    Row("L1", "a region is missing", lambda c: {"ti_ms": {"R1": 300.0}}, refused=True, reason=SHAPE + '"R1": <number>, "R2": <number>}}'),
    Row("L1", "a region too many", lambda c: with_ti(c, R3=300.0), refused=True, reason=SHAPE),
    Row("L1", "a second field", lambda c: dict(at()(c), units="ms"), refused=True, reason=SHAPE),
    Row("L1", "the numbers without the field", lambda c: at()(c)["ti_ms"], refused=True, reason=SHAPE),
    Row("L1", "a list", [300.0, 800.0], refused=True, reason=SHAPE),
    Row("L1", "nothing", None, refused=True, reason=SHAPE, expect=not_evaluated),
    Row("L1", "the field is not an object", {"ti_ms": 300.0}, refused=True, reason=SHAPE),
]
