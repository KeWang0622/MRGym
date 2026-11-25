"""Control policies of ``Scanner/T1Transmit``. Every policy plays through the agent's interface only (``info`` /
``plan`` / ``scan`` / ``budget``) and returns the deliverable. None reads the hidden world. What a policy may use: the
task card, the instrument datasheet and its own physics.

The physics here is the AGENT-SIDE helper physics, written again in this file on purpose (the scanner serves the EPG
steady state integrated over the slice; sharing its code would let errors cancel):
  * ideal-spoiling closed forms of the gradient echo and of the dual-TR scan;
  * the slice profile of the selective pulse, from what the datasheet states about it (a Hann-windowed sinc of 128
    samples with a time-bandwidth product, a rephasing lobe of half the selection area): a cascade of small rotations
    with precession in between, tabulated once per pulse over the flip angle at the slice centre. A 2D signal is the
    sum of the closed form over that profile.
The protocol constants of the reference (spoiler moments, repetition times, RF phase increment of the dual-TR scan)
come from an offline study of the documented spoiling mechanism over the public priors.

Must pass
  reference        chooses by the budget how to handle the slice profile: if the non-selective 3D acquisition fits,
                   it is played and fitted with the one-angle closed forms; otherwise 2D scans, fitted with the
                   closed forms summed over the slice profile. Either way: B1 map (dual-TR, signed with the phase
                   images), two gradient echoes, the budget filled with averages, joint fit of (M0, T1, B1) per
                   region. No scan depends on a measurement: only the averages depend on the budget
Must fail
  prior            no scan; the geometric mean of the told T1 range in every region
  fixed_recipe     one token scan, then the best answer that ignores the object: one constant T1 for every region,
                   tuned on other worlds
  wrong_units      the reference's numbers in seconds
Ablations (the reference with one piece removed; each must lose at the levels it names)
  ignores_profile  2D scans fitted with the one-angle closed forms, the model of ``Scanner/T1Transmit-v0`` (levels 2
                   and 3, where the budget leaves no room for the 3D acquisition)
  blind_vfa        two gradient echoes and no B1 map: the textbook variable flip angle method (every level)
Reported
  open_loop        a fixed scan list (the 2D protocol of the reference with the averages that fit the smallest budget
                   of any level) and a fixed estimator (the profile-aware joint fit). If it passes within 25 points of
                   the reference the family is a measurement task
  profile_2d       the 2D route with the slice-profile model at every level, whatever the budget
  nonselective_3d  the 3D route at every level; where it does not fit the budget, the prior answer after a token scan
  ir_only          inversion recovery only, as much as fits (the textbook gold standard, priced out by the budget)
"""
from __future__ import annotations

import functools
import math

import numpy as np

from . import task

REFERENCE = "reference"
MUST_FAIL = ("prior", "fixed_recipe", "wrong_units")
ABLATIONS = {"ignores_profile": (2, 3), "blind_vfa": (1, 2, 3)}
REPORTED = ("open_loop", "profile_2d", "nonselective_3d", "ir_only")
DESCRIPTIONS = dict(
    reference="B1 map, two gradient echoes, joint fit; the slice profile is avoided (3D) where the budget allows, else modelled",
    prior="no scan; the geometric mean of the told T1 range in every region",
    fixed_recipe="one token scan, then one constant T1 for every region, tuned on other worlds",
    wrong_units="the reference's numbers in seconds",
    ignores_profile="2D scans fitted with the one-angle closed forms (the v0 reference)",
    blind_vfa="two gradient echoes at nominal flip angles, no B1 map (confounder-blind)",
    open_loop="a fixed scan list and a fixed estimator: the 2D protocol with fixed averages, the profile-aware fit",
    profile_2d="2D scans with the slice-profile model at every level",
    nonselective_3d="the non-selective 3D acquisition at every level; the prior answer where it does not fit the budget",
    ir_only="inversion recovery only, as much as fits",
)
D2R = math.pi / 180.0
T1_PRIOR_MS = math.sqrt(task.T1_RANGE_MS[0] * task.T1_RANGE_MS[1])


# ------------------------------------------------------------------------------------------------ agent-side physics
def spgr_ideal(a, tr, t1):
    e1 = np.exp(-tr / t1)
    return np.sin(a) * (1 - e1) / (1 - e1 * np.cos(a))


def afi_ideal(a, tr1, tr2, t1):
    e1, e2, c = np.exp(-tr1 / t1), np.exp(-tr2 / t1), np.cos(a)
    den = 1 - e1 * e2 * c * c
    return np.sin(a) * (1 - e2 + (1 - e1) * e2 * c) / den, np.sin(a) * (1 - e1 + (1 - e2) * e1 * c) / den


PULSE_SAMPLES = 128
PULSE_TBW = dict(fast=2.0, normal=4.0, sharp=8.0)           # datasheet section 1
ALPHA_GRID = np.linspace(0.0, 170.0 * D2R, 341)             # flip angle at the slice centre on which a profile is tabulated


@functools.lru_cache(maxsize=4)
def slice_profile(tbw: float):
    """The profile of the selective pulse with time-bandwidth product ``tbw``, from the datasheet's description:
    (theta [alpha, z], weight [alpha, z]). ``theta`` is the flip angle at height z for a flip angle alpha at the slice
    centre (``ALPHA_GRID``); ``weight`` is the quadrature weight of z times the cosine of the transverse phase left
    after the rephasing lobe (the two halves of the slice added), per slice thickness.

    The pulse is ``PULSE_SAMPLES`` hard sub-pulses about x with the amplitudes of a Hann-windowed sinc; between them
    the spin at height z precesses about z in proportion to z. Each step is a 2 x 2 unitary matrix; the product is
    accumulated for every (alpha, z) at once."""
    t = (np.arange(PULSE_SAMPLES) + 0.5) / PULSE_SAMPLES - 0.5
    amp = np.sinc(tbw * t) * 0.5 * (1 + np.cos(2 * np.pi * t))
    amp /= amp.sum()
    z = np.linspace(0.0, 0.5 + 12.0 / tbw, int(round(6 * tbw * (0.5 + 12.0 / tbw))) + 1)          # in slice thicknesses, z >= 0
    wz = np.full(len(z), 2 * (z[1] - z[0]))
    wz[[0, -1]] /= 2
    alpha = ALPHA_GRID[:, None]
    step = 2 * np.pi * tbw * z[None, :] / PULSE_SAMPLES                                         # precession between two sub-pulses
    half = np.exp(-0.25j * step)
    a, b = np.ones((len(ALPHA_GRID), len(z)), complex), np.zeros((len(ALPHA_GRID), len(z)), complex)
    for w in amp:
        a, b = half * a, np.conj(half) * b
        c, s = np.cos(w * alpha / 2), -1j * np.sin(w * alpha / 2)
        a, b = c * a - np.conj(s) * b, s * a + c * b
        a, b = half * a, np.conj(half) * b
    back = np.exp(0.25j * step * PULSE_SAMPLES)                                                 # rephasing lobe: half the selection area
    a, b = back * a, np.conj(back) * b
    theta = 2 * np.arcsin(np.clip(np.abs(b), 0.0, 1.0))
    phase = 1j * np.exp(1j * np.angle(2 * np.conj(a) * b))                                      # +1 at the slice centre
    return theta, wz * phase.real


def _through_slice(alpha, tbw):
    """(theta, weight) [..., z] for flip angles ``alpha`` (rad, any shape) at the slice centre, by linear interpolation
    in the table of ``slice_profile``. ``tbw`` None: a hard pulse, one angle per voxel."""
    alpha = np.asarray(alpha, float)
    if tbw is None:
        return alpha[..., None], np.ones(alpha.shape + (1,))
    theta, weight = slice_profile(float(tbw))
    x = np.clip(alpha / (ALPHA_GRID[1] - ALPHA_GRID[0]), 0.0, len(ALPHA_GRID) - 1.000001)
    i = x.astype(int)
    f = (x - i)[..., None]
    return theta[i] * (1 - f) + theta[i + 1] * f, weight[i] * (1 - f) + weight[i + 1] * f


def _model(meas, t1, b1):
    """Signals of the measurements ``meas`` for T1 and B1 arrays of one shape: [n_signals, ...]."""
    rows = []
    for m in meas:
        theta, weight = _through_slice(b1 * m["flip"], m["tbw"])
        if m["kind"] == "spgr":
            rows.append((weight * spgr_ideal(theta, m["tr"], t1[..., None])).sum(-1))
        else:
            rows.extend((weight * s).sum(-1) for s in afi_ideal(theta, m["tr"], m["tr2"], t1[..., None]))
    return np.stack(np.broadcast_arrays(*rows))


def fit_region(meas, y, w, b1_fixed=None):
    """Weighted least squares of (M0, T1, B1) to the values ``y`` of one region; M0 is linear, (T1, B1) by a zooming
    grid. Returns (t1_ms, b1, m0)."""
    t1g = np.geomspace(150.0, 5000.0, 72)
    b1g = np.array([b1_fixed]) if b1_fixed is not None else np.linspace(0.45, 1.75, 53)
    w2 = w * w
    for _ in range(5):
        tt, bb = np.meshgrid(t1g, b1g, indexing="ij")
        m = _model(meas, tt, bb)
        num = np.einsum("n,nij->ij", w2 * y, m)
        den = np.einsum("n,nij->ij", w2, m * m)
        sse = -num * num / np.maximum(den, 1e-300)
        i, j = np.unravel_index(np.argmin(sse), sse.shape)
        best = (float(t1g[i]), float(b1g[j]), float(num[i, j] / max(den[i, j], 1e-300)))
        t1g = np.geomspace(t1g[max(i - 1, 0)], t1g[min(i + 1, len(t1g) - 1)], 13)
        if b1_fixed is None:
            b1g = np.linspace(b1g[max(j - 1, 0)], b1g[min(j + 1, len(b1g) - 1)], 13)
    return best


# ------------------------------------------------------------------------------------------------ helpers on the agent's interface
def _ids(card):
    return [r["id"] for r in card["task"]["regions"]]


def _values(resp):
    """Region signals [n_images, n_regions] from the complex-mean read-out (no noise floor)."""
    return np.array([r["complex_mean_magnitude"] for r in resp["roi_readouts"]], float).T


def _afi_signed(resp):
    """Dual-TR signals [2, n_regions] with the sign of image 2 relative to image 1 taken from the phase images: above
    about 100 degrees actual flip the second signal is negative, and magnitudes alone would fold the B1 estimate."""
    v = _values(resp)
    ph = np.array([r["complex_mean_phase_rad"] for r in resp["roi_readouts"]], float).T
    return np.stack([v[0], v[1] * np.sign(np.cos(ph[1] - ph[0]))])


def _answer(ids, t1):
    return {"t1_ms": {k: float(np.clip(v, task.T1_VALID_MS[0] + 1.0, task.T1_VALID_MS[1] - 1.0)) for k, v in zip(ids, t1)}}


def _prior_answer(card):
    return _answer(_ids(card), [T1_PRIOR_MS] * len(card["task"]["regions"]))


def _cost(sc, protocol):
    """(fixed seconds, seconds per average) of a protocol, or None if the console rejects it."""
    p = sc.plan(dict(protocol, averages=1))
    if not p["accept"]:
        return None
    return p["scan_s"] - p["scan_s_parts"]["acquisition_s"], p["scan_s_parts"]["acquisition_s"]


# The object is uniform through the slice (datasheet), so the thickest slice gives the most signal per scan second.
COMMON = dict(matrix=64, bandwidth_hz_per_px=200.0, fov_mm=240.0, slice_mm=10.0)
# From the offline spoiling study (strongest spoiler that fits the repetition time; RF phase increment scanned):
# dual-TR B1 error <= 0.2 % and two-angle T1 error <= 1.5 % over the public priors of T1, T2 and diffusion.
AFI = dict(COMMON, sequence="afi", tr_ms=30.0, tr_ratio=5, spoiler_mt_ms_per_m=585.0, rf_phase_increment_deg=45.0)
SPGR = dict(COMMON, sequence="spgr", tr_ms=20.0, spoiler_mt_ms_per_m=340.0)
AFI_FLIP, FLIPS = 70.0, (4.0, 20.0)        # nominal flip angles, degrees: of the dual-TR scan and of the two gradient echoes
AFI_SHARE = 0.75                            # of the acquisition time: the B1 map limits the precision of T1
MODE_2D, MODE_3D = dict(rf_pulse="normal"), dict(excitation="nonselective")
OPEN_LOOP_BUDGET_S = 45.0                   # the fixed scan list is sized for this budget: it fits every instance of every level


def _allocate(budget, cost_afi, cost_spgr, n_spgr, share=AFI_SHARE):
    """Averages (afi, spgr each) that fill ``budget`` with ``share`` of the acquisition time on the B1 map, or None."""
    acq = budget - cost_afi[0] - n_spgr * cost_spgr[0] - 0.05
    if acq < cost_afi[1] + n_spgr * cost_spgr[1]:
        return None
    n_a = int(np.clip(math.floor(share * acq / cost_afi[1]), 1, 64))
    n_s = int(np.clip(math.floor((acq - n_a * cost_afi[1]) / (n_spgr * cost_spgr[1])), 1, 64))
    return n_a, n_s


def _plan(sc, mode, budget=None):
    """Averages of the scan list in ``mode`` that fill ``budget`` (default: what is left), or None when it does not fit."""
    ca, cs = _cost(sc, dict(AFI, **mode, flip_deg=AFI_FLIP)), _cost(sc, dict(SPGR, **mode, flip_deg=FLIPS[0]))
    if ca is None or cs is None:
        return None
    return _allocate(sc.budget()["remaining_s"] if budget is None else budget, ca, cs, len(FLIPS))


def _pipeline(sc, n_afi, n_spgr, mode=MODE_2D, b1_fixed=None, model_profile=True):
    """B1 map + two gradient echoes + joint fit, at the nominal flip angles ``AFI_FLIP`` and ``FLIPS``. ``mode``: the
    excitation (``MODE_2D`` or ``MODE_3D``). ``model_profile`` False: fit as if every voxel saw one flip angle.
    ``n_afi`` 0: no B1 map."""
    card = sc.info()
    rois, ids = card["task"]["regions"], _ids(card)
    tbw = PULSE_TBW[mode["rf_pulse"]] if model_profile and mode.get("excitation") != "nonselective" else None
    meas, ys, ws = [], [], []
    if n_afi:
        r = sc.scan(dict(AFI, **mode, flip_deg=AFI_FLIP, averages=n_afi), rois)
        v = _afi_signed(r)
        meas.append(dict(kind="afi", flip=AFI_FLIP * D2R, tr=r["realised"]["tr_ms"], tr2=r["realised"]["tr2_ms"], tbw=tbw))
        ys += [v[0], v[1]]
        ws += [math.sqrt(n_afi)] * 2
    for f in FLIPS:
        r = sc.scan(dict(SPGR, **mode, flip_deg=f, averages=n_spgr), rois)
        meas.append(dict(kind="spgr", flip=f * D2R, tr=r["realised"]["tr_ms"], tbw=tbw))
        ys.append(_values(r)[0])
        ws.append(math.sqrt(n_spgr))
    y, w = np.array(ys), np.array(ws)
    return _answer(ids, [fit_region(meas, y[:, i], w, b1_fixed)[0] for i in range(len(ids))])


def _token_scan(sc):
    sc.scan(dict(sequence="spgr", matrix=48, tr_ms=11.0, flip_deg=8.0))


# ------------------------------------------------------------------------------------------------ the policies
def profile_2d(sc):
    """2D: map B1, two gradient echoes, the budget filled with averages, a joint fit with the slice profile in the model."""
    alloc = _plan(sc, MODE_2D)
    if alloc is None:
        return _prior_answer(sc.info())
    return _pipeline(sc, *alloc, mode=MODE_2D)


def nonselective_3d(sc):
    """The same with a non-selective 3D acquisition: one flip angle per voxel, every partition paid for."""
    alloc = _plan(sc, MODE_3D)
    if alloc is None:
        _token_scan(sc)
        return _prior_answer(sc.info())
    return _pipeline(sc, *alloc, mode=MODE_3D)


def reference(sc):
    """3D where the budget has room for it (no slice profile to model), otherwise 2D with the profile in the model."""
    return nonselective_3d(sc) if _plan(sc, MODE_3D) is not None else profile_2d(sc)


def ignores_profile(sc):
    alloc = _plan(sc, MODE_2D)
    if alloc is None:
        return _prior_answer(sc.info())
    return _pipeline(sc, *alloc, mode=MODE_2D, model_profile=False)


def blind_vfa(sc):
    cs = _cost(sc, dict(SPGR, **MODE_2D, flip_deg=FLIPS[0]))
    n = int(np.clip(math.floor((sc.budget()["remaining_s"] - len(FLIPS) * cs[0] - 0.05) / (len(FLIPS) * cs[1])), 1, 64))
    return _pipeline(sc, 0, n, mode=MODE_2D, b1_fixed=1.0, model_profile=False)


def open_loop(sc):
    """One scan list for every instance of every level: the 2D protocol with the averages that fit ``OPEN_LOOP_BUDGET_S``.
    Neither a measurement nor the budget of the instance changes a scan."""
    return _pipeline(sc, *_plan(sc, MODE_2D, OPEN_LOOP_BUDGET_S), mode=MODE_2D)


def prior(sc):
    return _prior_answer(sc.info())


def fixed_recipe(recipe):
    def policy(sc):
        card = sc.info()
        _token_scan(sc)
        return _answer(_ids(card), [recipe["t1_ms"]] * len(card["task"]["regions"]))
    return policy


DEFAULT_RECIPE = dict(t1_ms=round(T1_PRIOR_MS, 1))


def wrong_units(sc):
    return {"t1_ms": {k: v / 1000.0 for k, v in reference(sc)["t1_ms"].items()}}


def fit_ir(ti, y):
    """Three-parameter fit  s = a - b exp(-TI / T1)  to magnitude data with unknown polarity (every sign pattern that
    switches once is tried; a and b are linear)."""
    ti, y = np.asarray(ti, float), np.asarray(y, float)
    order = np.argsort(ti)
    ti, y = ti[order], y[order]
    t1g = np.geomspace(100.0, 6000.0, 600)
    e = np.exp(-ti[None, :] / t1g[:, None])                 # [grid, n]
    n, se, see = len(ti), e.sum(1), (e * e).sum(1)
    det = n * see - se * se
    best = (np.inf, T1_PRIOR_MS)
    for k in range(len(ti) + 1):                           # the first k points are negative
        s = np.where(np.arange(len(ti)) < k, -y, y)
        ss, ses = s.sum(), e @ s
        a = (see * ss - se * ses) / det
        b = (se * ss - n * ses) / det
        sse = ((a[:, None] - b[:, None] * e - s[None, :]) ** 2).sum(1)
        sse = np.where((a > 0) & (b > a), sse, np.inf)
        i = int(np.argmin(sse))
        if sse[i] < best[0]:
            best = (float(sse[i]), float(t1g[i]))
    return best[1]


def ir_only(sc):
    """Inversion recovery at the cheapest useful settings; needs at least three inversion times for its fit."""
    card = sc.info()
    rois, ids = card["task"]["regions"], _ids(card)
    budget = sc.budget()["remaining_s"]
    for matrix, tr, tis in ((64, 6000.0, (60.0, 250.0, 600.0, 1300.0, 2800.0, 5500.0)), (64, 3000.0, (60.0, 300.0, 800.0, 1800.0)),
                            (48, 2500.0, (60.0, 500.0, 1500.0)), (48, 1000.0, (40.0, 300.0, 900.0))):
        p = sc.plan(dict(sequence="ir", matrix=matrix, tr_ms=tr, ti_ms=tis[0]))
        if p["accept"] and p["scan_s"] * len(tis) <= budget:
            data = [_values(sc.scan(dict(sequence="ir", matrix=matrix, tr_ms=tr, ti_ms=t), rois))[0] for t in tis]
            y = np.array(data)
            return _answer(ids, [fit_ir(tis, y[:, i]) for i in range(len(ids))])
    p = sc.plan(dict(sequence="ir", matrix=48, tr_ms=1000.0, ti_ms=300.0))
    if p["accept"] and p["scan_s"] <= budget:              # one inversion time fits: it is in the log, and useless
        sc.scan(dict(sequence="ir", matrix=48, tr_ms=1000.0, ti_ms=300.0), rois)
    return _prior_answer(card)


def policies(recipe=None):
    return dict(reference=reference, prior=prior, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE), wrong_units=wrong_units,
                ignores_profile=ignores_profile, blind_vfa=blind_vfa, open_loop=open_loop, profile_2d=profile_2d,
                nonselective_3d=nonselective_3d, ir_only=ir_only)


def tune(cfgs):
    """The strongest answer that ignores the object, on other worlds: the one constant T1 that brings the most regions
    within the tolerance (the verifier's own rule, per region), and the share of worlds it passes."""
    truths = [task.true_t1_ms(c) for c in cfgs]
    limit = math.log(1.0 + cfgs[0]["task"]["tolerance"])
    values = np.array([v for t in truths for v in t.values()])
    best = max(np.geomspace(task.T1_RANGE_MS[0], task.T1_RANGE_MS[1], 200), key=lambda c: float(np.mean(np.abs(np.log(c / values)) <= limit)))
    passed = sum(all(abs(math.log(best / v)) <= limit for v in t.values()) for t in truths)
    return dict(t1_ms=round(float(best), 1), tuned_region_pass_rate=round(float(np.mean(np.abs(np.log(best / values)) <= limit)), 3),
                tuned_pass_rate=round(passed / len(truths), 3))
