"""Scoring table of ``Scanner/ProtocolToSpec-v0`` (format: ``tests/scoring/__init__.py``).

Pass rule (task statement), for each of the two images, on a noise-free scan of the delivered protocol: the T1 contrast
ratio is at least 4, the T2 contrast ratio at least 1.5 and the SNR of the named region at least 20 with the noise SD of
one pixel of this scanner; and the scan times of the two images together are within the stated limit.

The expected values are derived here, not taken from the task module:

    scan time   adjustments 30 s + TR x (1 dummy cycle + 128 lines x averages), TR 400 ms (the console's rule, written out)
    signal      of a region, up to a factor common to all regions: proton density x receive magnitude at its centre x
                the longitudinal magnetisation before the excitation x sin^3(a) x exp(-TE / T2), a = 90 deg x B1. The
                magnetisation is found by walking it through the documented chain
                    excitation (a) - TE/2 - refocusing (2a) - (TR - TI - TE/2) - inversion (Mz -> -k Mz) - TI
                for many repetitions (a plain iteration: no closed form shared with ``mrgym.scanner.signal``)
    noise       SD of one pixel = sigma0 x sqrt(2 x 128 x bandwidth / averages) x sqrt(lines x 2 x 128) / (2 x FOV x FOV):
                receiver noise per sample summed by the reconstruction (readout oversampling 2), written out

Hard pulse: the chain has one flip angle per position, the scanner integrates over the slice profile. Ratios of two
regions agree with the scanner within ``RATIO_RTOL`` and absolute signals, times ``PROFILE``, within ``SIGNAL_RTOL``.
The T1 contrast compares a region with one near its null, where a relative error means nothing: there the residual
signal of the nulled region, as a share of the brighter one (1 / ratio), agrees within ``NULL_ATOL``. All three are
measured on these instances and asserted by ``agrees_with_the_chain``. They are not predictions: each was set after
the comparison was seen on these two instances (largest differences found: 8.1 % on a T2 ratio, 7.5 % on an SNR, 0.02
on a residual), and the first guesses (4 %, 8 % with a profile factor of 0.80) were wrong. On another instance they may
not hold; every row keeps a larger distance
from the thresholds than that, which ``far_from_the_thresholds`` asserts on the chain's own numbers.
"""
import math

from scoring import Row

ENV_ID = "Scanner/ProtocolToSpec-v0"
INSTANCES = {"TI": dict(level=1, seed=19), "TE": dict(level=2, seed=0)}       # the first one is found on the first world of its seed
T1_RATIO, T2_RATIO, SNR_MIN = 4.0, 1.5, 20.0                            # the specification of the task statement
TR_MS, TI0_MS, TE0_MS, LINES, FOV_MM, BANDWIDTH, SLICE_MM, PRICE_S = 400.0, 80.0, 12.0, 128, 240.0, 200.0, 5.0, 30.0
PROFILE = 0.82                      # slice-integrated signal of the selective pulses / hard-pulse signal (measured here: 0.78 to 0.86)
RATIO_RTOL, SIGNAL_RTOL, NULL_ATOL = 0.09, 0.09, 0.03
CLEAR, CLEAR_SNR = 1.08, 1.12       # every row is at least this factor away from each threshold, by the chain (SNR: the larger one)


def scan_s(averages):
    return PRICE_S + TR_MS * (1 + LINES * averages) / 1000.0


def averages_total(case):
    """As many averages, both images together, as the stated limit of the examination allows."""
    n = 2
    while 2 * scan_s(0) + (scan_s(n + 1) - scan_s(0)) <= case.cfg["task"]["scan_time_max_s"]:
        n += 1
    return n


def mz_before_excitation(a, te, ti, t1, kappa, repetitions=80):
    m = 1.0
    for _ in range(repetitions):
        m = m * math.cos(a)
        m = 1.0 + (m - 1.0) * math.exp(-(te / 2) / t1)
        m = m * math.cos(2 * a)
        m = 1.0 + (m - 1.0) * math.exp(-(TR_MS - ti - te / 2) / t1)
        m = -kappa * m
        m = 1.0 + (m - 1.0) * math.exp(-ti / t1)
    return m


def receive(case, region):
    r = next(q for q in case.cfg["task"]["regions"] if q["id"] == region)
    n, d = case.world["n"], case.world["extent_mm"] / case.world["n"]
    return abs(case.world["fields"]["rx"][int(round(r["y_mm"] / d)) + n // 2, int(round(r["x_mm"] / d)) + n // 2])


def signal(case, region, ti, te):
    """Signed signal of a region by the chain (the scanner's units when multiplied by ``PROFILE``)."""
    t = case.truth[region]
    a = math.pi / 2 * t["b1"]
    return (t["pd"] * receive(case, region) * SLICE_MM * mz_before_excitation(a, te, ti, t["t1_ms"], case.world["system"]["inv_eff"])
            * math.sin(a) ** 3 * math.exp(-te / t["t2_ms"]))


def noise_sd(case, averages):
    k = 2 * LINES                                                       # samples along the readout (oversampling 2), image 128 x 128
    return case.world["system"]["sigma0"] * math.sqrt(k * BANDWIDTH / averages) * math.sqrt(LINES * k) / (2 * FOV_MM * FOV_MM)


def numbers(case, name, fields):
    """What the chain gives for one image: the two ratios, the SNR and the scan time."""
    q = case.cfg["task"]["images"][name]
    s = lambda region: abs(signal(case, region, fields["ti_ms"], fields["te_ms"]))   # noqa: E731
    return dict(t1_ratio=s(q["t1_contrast"]["brighter"]) / s(q["t1_contrast"]["darker"]),
                t2_ratio=s(q["t2_contrast"]["brighter"]) / s(q["t2_contrast"]["darker"]),
                snr=PROFILE * s(q["snr"]["region"]) / noise_sd(case, fields["averages"]), scan_s=scan_s(fields["averages"]))


def null_ms(case, region, te=TE0_MS):
    """The inversion time at which the chain's signal of a region changes sign (bisection)."""
    lo, hi = 8.0, 300.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if signal(case, region, mid, te) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def clear(case, name, fields):
    """Whether the chain passes every requirement of an image by ``CLEAR``."""
    v = numbers(case, name, fields)
    return 1 / v["t1_ratio"] <= 1 / T1_RATIO - NULL_ATOL - 0.01 and v["t2_ratio"] >= CLEAR * T2_RATIO and v["snr"] >= CLEAR_SNR * SNR_MIN


def good(case, name, averages):
    """A protocol of an image with ``averages`` that the chain passes by ``CLEAR``, or None: the first inversion time
    (the null of the region to be nulled, then steps of 6 % around it, then the standard one) and the first echo time
    on a 2 ms raster at which the chain clears every requirement."""
    q = case.cfg["task"]["images"][name]
    null = null_ms(case, q["t1_contrast"]["darker"])
    for ti in [null * f for f in (1.0, 0.94, 1.06, 0.88, 1.12, 0.82)] + [TI0_MS]:
        for te in [TE0_MS + 2.0 * k for k in range(55)]:
            fields = dict(ti_ms=round(ti, 2), te_ms=te, averages=averages)
            if clear(case, name, fields):
                return fields
    return None


def both(**change):
    """Deliverable: a good protocol of each image, the averages the limit allows shared out by the first split for which
    the chain clears both; ``A=dict(...)`` / ``B=dict(...)`` change fields of one (a callable is called with the fields)."""
    def deliverable(case):
        total = averages_total(case)
        out = next(dict(A=a, B=b) for n in range(1, total) for a, b in [(good(case, "A", n), good(case, "B", total - n))] if a and b)
        for name in out:
            for k, v in change.get(name, {}).items():
                out[name][k] = v(out[name]) if callable(v) else v
        return out
    return deliverable


def one_more_average(fields):
    return fields["averages"] + 1


def agrees_with_the_chain(case, score):
    """The verifier's ratios, SNR and scan time of each image against the chain; the checks are the pass rule applied
    to them; ``error`` is minus the smallest logarithmic slack."""
    slacks, total, limit = [], 0.0, case.cfg["task"]["scan_time_max_s"]
    for name, got in score["images"].items():
        want = numbers(case, name, score["answer"][name])
        assert got["scan_s"] == want["scan_s"]
        total += got["scan_s"]
        assert abs(1 / got["t1_ratio"] - 1 / want["t1_ratio"]) <= NULL_ATOL, (name, got["t1_ratio"], want["t1_ratio"])
        for key, rtol in (("t2_ratio", RATIO_RTOL), ("snr", SIGNAL_RTOL)):
            assert abs(got[key] / want[key] - 1.0) <= rtol, (name, key, got[key], want[key])
        rule = {f"t1_contrast_{name}": got["t1_ratio"] >= T1_RATIO, f"t2_contrast_{name}": got["t2_ratio"] >= T2_RATIO,
                f"snr_{name}": got["snr"] >= SNR_MIN}
        assert {k: score["checks"][k] for k in rule} == rule
        slacks += [math.log(got["t1_ratio"] / T1_RATIO), math.log(got["t2_ratio"] / T2_RATIO), math.log(got["snr"] / SNR_MIN)]
    assert score["checks"]["time"] is (total <= limit) and score["scan_s"] == total
    assert score["error"] == -min(slacks + [math.log(limit / total)]) and score["checks"]["deliverable_valid"] is True


def far_from_the_thresholds(case, score):
    """The row does not rest on the accuracy of the chain: each of its numbers is ``CLEAR`` away from its threshold."""
    agrees_with_the_chain(case, score)
    for name in score["images"]:
        v = numbers(case, name, score["answer"][name])
        assert abs(1 / v["t1_ratio"] - 1 / T1_RATIO) >= NULL_ATOL + 0.01, (name, v["t1_ratio"])
        for value, need, by in ((v["t2_ratio"], T2_RATIO, CLEAR), (v["snr"], SNR_MIN, CLEAR_SNR)):
            assert value >= by * need or value <= need / by, (name, value, need)


def not_evaluated(case, score):
    assert score["checks"] == dict(deliverable_valid=False) and score["error"] is None and "images" not in score


CONSOLE = "the console rejects the scan of image "
SHAPE = 'the deliverable is {"A": {"ti_ms": <number>, "te_ms": <number>, "averages": <integer>}, "B": {'
ROWS = [
    # ---- level 1: the T1 contrast fails at the standard protocol; the inversion time has to move
    Row("TI", "each image at the null of its region, the averages the limit pays for shared out", both(), passed=True, expect=far_from_the_thresholds),
    Row("TI", "one average more than the limit of the examination pays for", both(A=dict(averages=one_more_average)), passed=False,
        failing=("time",), reason="scan time of the examination", expect=far_from_the_thresholds),
    Row("TI", "integers written as numbers are numbers", lambda c: {k: dict(v, te_ms=12) for k, v in both()(c).items()}, passed=True),
    # ---- level 2: the T2 contrast fails at the standard protocol; the echo time has to move
    Row("TE", "each image at an echo time the chain clears, the averages shared out", both(), passed=True,
        expect=far_from_the_thresholds),
    Row("TE", "image B left at the standard echo time", both(B=dict(te_ms=TE0_MS)), passed=False, failing=("t2_contrast_B",),
        reason="image B: T2 contrast", expect=agrees_with_the_chain),
    Row("TE", "one average more in image B than the limit pays for", both(B=dict(averages=one_more_average)), passed=False,
        failing=("time",), reason="scan time of the examination", expect=far_from_the_thresholds),
    # ---- deliverables the console cannot evaluate: refused, the episode continues
    Row("TI", "an inversion time the repetition time has no room for", both(B=dict(ti_ms=395.0)), refused=True,
        reason=CONSOLE + "B: tr_ms=400.0 is below the minimum"),
    Row("TI", "the right times in seconds", lambda c: {k: dict(v, ti_ms=v["ti_ms"] / 1000.0, te_ms=v["te_ms"] / 1000.0) for k, v in both()(c).items()},
        refused=True, reason=CONSOLE + "A: ", expect=not_evaluated),
    Row("TI", "averages that are not an integer", both(A=dict(averages=2.5)), refused=True, reason=CONSOLE + "A: averages must be an integer"),
    Row("TI", "no averages at all", both(A=dict(averages=0)), refused=True, reason=CONSOLE + "A: averages=0 outside"),
    Row("TI", "a field of the fixed protocol", lambda c: {k: dict(v, tr_ms=400.0) for k, v in both()(c).items()}, refused=True, reason=SHAPE),
    Row("TI", "an image is missing", lambda c: {"A": both()(c)["A"]}, refused=True, reason=SHAPE),
    Row("TI", "a third image", lambda c: dict(both()(c), C=both()(c)["A"]), refused=True, reason=SHAPE),
    Row("TI", "a list", [80.0, 12.0, 1], refused=True, reason=SHAPE),
    Row("TI", "nothing", None, refused=True, reason=SHAPE, expect=not_evaluated),
]
