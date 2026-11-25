"""Metamorphic relations: what must stay equal, or change in a known way, when the input changes in a known way.

A simulator has no oracle for a new phantom or protocol; closed forms (``mrgym validate``) cover a few points. A relation
between the outputs of two related inputs needs no oracle and holds everywhere. Twenty relations, R1 to R20; the table
is ``RELATIONS`` below. Every test records the deviation it measured next to its tolerance. Tolerances were set from the
deviations measured on engine 0.2.0 (numpy 2.2.6 / scipy 1.15.3): about ten times the deviation where the engine sets
it, 1e-12 where only rounding does (measured 1e-14 or less), 6 % for noise statistics; with
``MRGYM_METAMORPHIC_TABLE=<path>`` the table is written as JSON for the audit report. Three tolerances are set by the
engine, not by rounding: R1 (the partial-volume weights are float32, so proton density is linear to 3e-7), R7 and R8
(material weights below 1e-4 are dropped, a step that phase oversampling and an axis swap see at 1e-4).

Excitation (third column of ``RELATIONS``). Since engine 0.2.0 the default excitation is slice-selective: the signal of
a voxel is an integral over the slice profile. ``excitation: nonselective`` keeps the hard pulse (one flip angle per
voxel). What each relation covers:
  both         the relation is run on slice-selective scans (the default) and on hard-pulse scans: R1, R2, R3, R4, R9, R10
               (the protocols ``se3d`` / ``spgr3d`` are the hard-pulse ones)
  hard pulse   it is a statement about the hard-pulse closed forms or needs them as one side: the first half of R4 (the
               closed forms of ``signal`` and ``epg``; marked ``hard_pulse``, ``pytest -m hard_pulse``), R19 (an ideal
               rectangular profile gives the hard-pulse image) and R20 (at a small flip angle the slice integral is the
               hard-pulse signal)
  -            independent of how the flip angle is produced (noise, price, geometry of the read-out, the reward)
R9 no longer assumes a hard pulse: the engine's profile leaves a real amplitude at the slice centre's phase, and the
relation is run with both excitations.
"""
import json
import math
import os

import numpy as np
import pytest

import scoring
from conftest import legacy_task
from mrgym.env import derive_seed
from mrgym.envs.nowrap import task as NW1
from mrgym.envs.t1_transmit import task as T11
from mrgym.scanner import console, epg, kspace, session, tissue
from mrgym.scanner import signal as S
from mrgym.scanner import world as W
from scoring import null_tissue as NT_TABLE

NW, _, NW_GONE = legacy_task("Scanner/NoWrap-v0", "mrgym.envs.nowrap")
T1, _, T1_GONE = legacy_task("Scanner/T1Transmit-v0", "mrgym.envs.t1_transmit")

RELATIONS = {
    "R1": ("proton density of every material x a", "k-space x a", "both"),
    "R2": ("phantom = A + B (disjoint)", "k-space(A + B) = k-space(A) + k-space(B)", "both"),
    "R3": ("transmit gain x g, nominal flip / g", "identical image", "both"),
    "R4": ("every time x a (TR, TE, TI, T1, T2, T2')", "signal models unchanged, closed forms and through the slice profile", "both"),
    "R5": ("averages 1 -> 4", "noise-free image identical; noise SD / 2", "-"),
    "R6": ("bandwidth x 4", "noise SD x 2", "-"),
    "R7": ("phase oversampling 0 -> 1, object inside the field of view", "image unchanged; acquisition time x 2", "-"),
    "R8": ("phase-encode axis y <-> x, square field of view, no readout effects", "identical image", "-"),
    "R9": ("real object on resonance", "S(-k) = conj S(k) up to one global phase", "both"),
    "R10": ("object moved by 4 pixels; object and field of view moved together", "image moved by 4 pixels; image unchanged", "both"),
    "R11": ("readout polarity 1 -> -1, fat; water off resonance by the fat frequency", "fat displacement changes sign, same size; "
            "water is displaced like fat", "-"),
    "R12": ("field 1.5 T -> 3 T, same bandwidth, fat", "fat displacement x 2", "-"),
    "R13": ("scan B after another scan A, or after a different one", "noise-free B, realised protocol, cost identical; noisy B "
            "depends only on its index", "-"),
    "R14": ("noise draw 0 -> 1", "difference has mean 0 and SD sqrt(2) x the analytic noise SD; clean part identical", "-"),
    "R15": ("circle C contains circle c", "n_C mean_C - n_c mean_c = sum over the ring; |complex mean| <= mean |S|", "-"),
    "R16": ("averages x k; partial Fourier, phase field of view, oversampling raised", "acquisition time x k; scan time never falls", "-"),
    "R17": ("the same deliverable written another way; more scans before the answer", "reward unchanged", "-"),
    "R18": ("regions listed in another order; every value moved further from the truth", "reward unchanged; reward never rises", "-"),
    "R19": ("slice profile -> an ideal rectangle", "the image of hard pulses (excitation: nonselective), same timing", "hard pulse"),
    "R20": ("flip angle -> 0 at a long repetition time", "slice-selective image -> effective thickness x hard-pulse image; "
            "remainder ~ flip^2", "hard pulse"),
}
MEASURED: dict = {}
N, EXTENT = 256, 256.0                       # master grid of the hand-made worlds: 1 mm
GEO = dict(matrix=64, fov_mm=240.0)          # 3.75 mm pixels
PROTOCOLS = dict(
    se=dict(GEO, sequence="se", tr_ms=500.0), ir=dict(GEO, sequence="ir", tr_ms=1500.0, ti_ms=300.0),
    spgr=dict(GEO, sequence="spgr", tr_ms=30.0, flip_deg=24.0), afi=dict(GEO, sequence="afi", tr_ms=25.0, flip_deg=50.0),
    bssfp=dict(GEO, sequence="bssfp", tr_ms=8.0, flip_deg=40.0), epi=dict(GEO, sequence="epi", tr_ms=200.0, flip_deg=60.0))
PROTOCOLS.update(se3d=dict(PROTOCOLS["se"], excitation="nonselective"), spgr3d=dict(PROTOCOLS["spgr"], excitation="nonselective"))
EXAM = dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=6.0, b0_t=1.5)


@pytest.fixture(scope="module", autouse=True)
def _table():
    yield
    path = os.environ.get("MRGYM_METAMORPHIC_TABLE")
    if path:
        rows = [dict(relation=k, transformation=RELATIONS[k][0], expected=RELATIONS[k][1], excitation=RELATIONS[k][2],
                     checks=MEASURED.get(k, [])) for k in RELATIONS]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)


def within(relation, what, measured, tolerance):
    """Assert ``measured <= tolerance`` and record both."""
    MEASURED.setdefault(relation, []).append(dict(what=what, tolerance=tolerance, measured=float(measured)))
    assert measured <= tolerance, f"{relation} {what}: deviation {measured:.3g} > tolerance {tolerance:.3g}"


def exact(relation, what, ok):
    MEASURED.setdefault(relation, []).append(dict(what=what, tolerance="exact", measured=0.0 if ok else "differs"))
    assert ok, f"{relation} {what}: not identical"


def disc(cy, cx, r):
    return lambda y, x: np.hypot(y - cy, x - cx) - r


TWO = [("wm", disc(-30.0, -20.0, 28.0)), ("gm", disc(35.0, 25.0, 22.0))]


def world(shapes=TWO, b0_t=3.0, tissues=None, edge=None, **system):
    kw = {} if edge is None else dict(edge=edge)
    return W.flat_world(W.shapes_phantom(shapes, b0_t=b0_t, n=N, extent_mm=EXTENT, tissues=tissues, **kw), **system)


def realised(w, protocol):
    pl = console.plan(protocol, w["console"])
    assert pl["accept"], pl["violations"]
    return pl["realised"]


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(b)))


def image(w, protocol, effects=None):
    e = kspace.Engine(w, effects=effects)
    return e.acquire(realised(w, protocol), None)


# ---------------------------------------------------------------------------------------------- linearity
@pytest.mark.parametrize("seq", sorted(PROTOCOLS))
def test_r1_proton_density_scales_the_kspace(seq):
    base = kspace.Engine(world()).clean_kspace(realised(world(), PROTOCOLS[seq]))
    for a in (0.5, 3.0):
        tab = tissue.table(3.0)
        scaled = world(tissues={k: dict(tab[k], pd=tab[k]["pd"] * a) for k in ("wm", "gm")})
        ks = kspace.Engine(scaled).clean_kspace(realised(scaled, PROTOCOLS[seq]))
        within("R1", f"{seq}, a = {a}", rel(ks, a * base), 3e-7)         # the partial-volume weights are float32


@pytest.mark.parametrize("seq", sorted(PROTOCOLS))
def test_r2_two_disjoint_objects_add(seq):
    both, a, b = world(), world(TWO[:1]), world(TWO[1:])
    ks = [kspace.Engine(w).clean_kspace(realised(w, PROTOCOLS[seq])) for w in (both, a, b)]
    assert rel(ks[1], ks[0]) > 0.1 and rel(ks[2], ks[0]) > 0.1          # each part matters
    within("R2", seq, rel(ks[1] + ks[2], ks[0]), 1e-12)


@pytest.mark.parametrize("seq", ["spgr", "afi", "bssfp", "epi", "spgr3d"])
def test_r3_transmit_gain_and_nominal_flip_angle_enter_as_a_product(seq):
    base = image(world(), PROTOCOLS[seq])
    for g in (0.8, 1.25):
        got = image(world(tx_gain=g), dict(PROTOCOLS[seq], flip_deg=PROTOCOLS[seq]["flip_deg"] / g))
        within("R3", f"{seq}, g = {g}", rel(got, base), 1e-12)
        assert rel(image(world(tx_gain=g), PROTOCOLS[seq]), base) > 1e-3   # the gain alone does change the image


def test_r4_scaling_every_time_changes_no_signal_through_the_slice_profile():
    """The pulse is instantaneous in the model at every height of the slice, so the slice-integrated signal the engine
    serves scales like the closed forms."""
    a = np.deg2rad(np.array([35.0, 90.0, 104.0]))
    w = world()
    cls = dict(tissue.table(3.0)["gm"])
    for seq in ("se", "ir", "spgr", "afi"):
        rp = realised(w, PROTOCOLS[seq])
        want = np.array(kspace.slice_signal(rp, cls, a, rp["rf_tbw"], inv_eff=0.9))
        assert np.abs(want).max() > 1e-3
        for s in (0.5, 3.0):
            times = {k: s * rp[k] for k in ("tr_ms", "te_ms", "ti_ms", "tr2_ms") if rp.get(k) is not None}
            scaled = {k: (s * v if k in ("t1_ms", "t2_ms", "t2s_ms", "t2p_ms") else v) for k, v in cls.items()}
            scaled["adc_um2_ms"] = cls["adc_um2_ms"] / s                 # diffusion under the spoiler acts as D x TR: kept
            got = np.array(kspace.slice_signal(dict(rp, **times), scaled, a, rp["rf_tbw"], inv_eff=0.9))
            within("R4", f"slice-selective {seq}, a = {s}", rel(got, want), 1e-12)


@pytest.mark.hard_pulse
def test_r4_scaling_every_time_changes_no_signal():
    a = np.deg2rad(np.array([12.0, 47.0, 88.0]))
    tr, te, ti, t1, t2 = 600.0, 14.0, 180.0, 900.0, 80.0
    for s in (0.5, 3.0):
        pairs = dict(
            spgr=(S.spgr(a, tr, t1), S.spgr(a, s * tr, s * t1)),
            se=(S.se(a, tr, te, t1), S.se(a, s * tr, s * te, s * t1)),
            ir_se=(S.ir_se(a, tr, te, ti, t1, 0.93), S.ir_se(a, s * tr, s * te, s * ti, s * t1, 0.93)),
            ir_gre=(S.ir_gre(a, ti, tr, t1, 0.93), S.ir_gre(a, s * ti, s * tr, s * t1, 0.93)),
            afi=(np.array(S.afi(a, 20.0, 100.0, t1)), np.array(S.afi(a, s * 20.0, s * 100.0, s * t1))),
            bssfp=(S.bssfp(a, 0.7, 6.0, t1, t2), S.bssfp(a, 0.7, s * 6.0, s * t1, s * t2)),
            epg_spgr=(epg.spgr(a, 20.0, t1, t2), epg.spgr(a, s * 20.0, s * t1, s * t2)),
            epg_afi=(np.array(epg.afi(a, 20.0, 5, t1, t2)), np.array(epg.afi(a, s * 20.0, 5, s * t1, s * t2))))
        for name, (want, got) in pairs.items():
            within("R4", f"{name}, a = {s}", rel(got, want), 1e-12)
        cls = dict(t2_ms=t2, t2s_ms=40.0, t2p_ms=80.0, fat=False)
        scaled = {k: (s * v if k != "fat" else v) for k, v in cls.items()}
        tau = np.linspace(-2e-3, 2e-3, 9)
        for echo in ("gre", "se"):
            want = S.time_function(cls, tau, tau, te, echo, 3.0, tissue.FAT_PEAKS)
            got = S.time_function(scaled, s * tau, s * tau, s * te, echo, 3.0, tissue.FAT_PEAKS)
            within("R4", f"decay through the readout ({echo}), a = {s}", rel(got, want), 1e-12)


# ---------------------------------------------------------------------------------------------- noise
def _noise(w, protocol, seed):
    e = kspace.Engine(w)
    rp = realised(w, protocol)
    clean = e.acquire(rp, None)
    return (e.acquire(rp, np.random.default_rng(seed)) - clean)[0], clean, e.noise_sd(rp)


def test_r5_averaging_leaves_the_image_and_halves_the_noise():
    w = world(sigma0=0.02)
    p = dict(sequence="se", tr_ms=500.0, matrix=96, fov_mm=240.0)
    n1, clean1, sd1 = _noise(w, p, seed=7)
    n4, clean4, sd4 = _noise(w, dict(p, averages=4), seed=8)             # independent draws: a statement about statistics
    assert n1.size >= 4000
    exact("R5", "noise-free image, averages 1 and 4", np.array_equal(clean1, clean4))
    within("R5", "noise SD ratio vs 2", abs(n1.real.std() / n4.real.std() - 2.0) / 2.0, 0.06)
    within("R5", "noise SD vs the analytic Engine.noise_sd", abs(n1.real.std() / sd1 - 1.0), 0.06)
    assert sd1 / sd4 == pytest.approx(2.0, rel=1e-12)


def test_r6_four_times_the_bandwidth_doubles_the_noise():
    w = world(sigma0=0.02)
    p = dict(sequence="se", tr_ms=500.0, matrix=96, fov_mm=240.0, bandwidth_hz_per_px=150.0)
    lo, _, sd_lo = _noise(w, p, seed=9)
    hi, _, sd_hi = _noise(w, dict(p, bandwidth_hz_per_px=600.0), seed=10)
    within("R6", "noise SD ratio vs 2", abs(hi.real.std() / lo.real.std() - 2.0) / 2.0, 0.06)
    within("R6", "imaginary part: noise SD ratio vs 2", abs(hi.imag.std() / lo.imag.std() - 2.0) / 2.0, 0.06)
    assert sd_hi / sd_lo == pytest.approx(2.0, rel=1e-12)


# ---------------------------------------------------------------------------------------------- geometry
# Edges 6 mm wide, so truncation ringing is small, and the whole object inside every field of view used below. What is
# left (1e-4) is the engine's support threshold: material weights below 1e-4 are dropped, a step of that size.
SMOOTH = [("wm", disc(-12.0, 8.0, 25.0)), ("gm", disc(20.0, -15.0, 10.0))]
SMOOTH_EDGE = 6.0


def test_r7_phase_oversampling_changes_the_price_not_the_image():
    w = world(SMOOTH, edge=SMOOTH_EDGE)
    p = dict(PROTOCOLS["se"], phase_fov=0.75)
    plain, twice = console.plan(p, w["console"]), console.plan(dict(p, phase_oversampling=1.0), w["console"])
    assert twice["realised"]["lines"] == 2 * plain["realised"]["lines"]
    within("R7", "image", rel(image(w, dict(p, phase_oversampling=1.0)), image(w, p)), 1e-3)
    within("R7", "acquisition seconds vs twice", abs(twice["scan_s_parts"]["acquisition_s"] - 2 * plain["scan_s_parts"]["acquisition_s"]), 1e-9)
    same = {k: twice["scan_s_parts"][k] == plain["scan_s_parts"][k] for k in ("overhead_s", "dummy_s")}
    exact("R7", "overhead and dummy time", all(same.values()))


def test_r8_swapping_the_phase_encode_axis_gives_the_same_image():
    w = world(SMOOTH, edge=SMOOTH_EDGE)
    y = image(w, dict(PROTOCOLS["se"], pe_dir="y"), effects=frozenset())
    x = image(w, dict(PROTOCOLS["se"], pe_dir="x"), effects=frozenset())
    assert rel(x[0], y[0].T) > 0.1                                       # the object is not symmetric: a transpose would show
    within("R8", "se, no readout effects", rel(x, y), 1e-3)


@pytest.mark.parametrize("seq", ["se", "se3d"])
def test_r9_a_real_object_has_a_hermitian_kspace(seq):
    w = world([("wm", disc(-18.0, 26.0, 30.0))])
    e = kspace.Engine(w, effects=frozenset())
    ks = e.clean_kspace(realised(w, PROTOCOLS[seq]))[0]
    centre = ks[ks.shape[0] // 2, ks.shape[1] // 2]
    ks = ks * np.conj(centre) / abs(centre)                              # one global phase divided out
    a = ks[1:, 1:]                                                       # index i is k = i - n // 2: -k is index n - i
    assert np.abs(a.imag).max() > 0.1 * np.abs(a).max()                  # off-centre object: the k-space is not real
    within("R9", f"{seq}: S(-k) vs conj S(k)", rel(a[::-1, ::-1], np.conj(a)), 1e-12)


def test_r10_moving_the_object_moves_the_image_and_moving_the_field_of_view_with_it_changes_nothing():
    shift_mm, shift_px = 15.0, 4                                         # 4 pixels of 3.75 mm = 15 points of the master grid
    along_x = [(name, disc(-30.0, -20.0 + shift_mm, 28.0) if name == "wm" else disc(35.0, 25.0 + shift_mm, 22.0)) for name, _ in TWO]
    along_y = [(name, disc(-30.0 + shift_mm, -20.0, 28.0) if name == "wm" else disc(35.0 + shift_mm, 25.0, 22.0)) for name, _ in TWO]
    for seq in ("se", "spgr", "epi", "se3d"):
        base = image(world(), PROTOCOLS[seq])
        got = image(world(along_x), PROTOCOLS[seq])
        assert rel(got, base) > 0.1
        # the readout is oversampled and cropped: what enters at the border is new, everything else has moved
        within("R10", f"{seq}: moved along the readout, on the overlap", rel(got[..., shift_px:], base[..., :-shift_px]), 1e-12)
        # the phase direction is periodic: the image rolls
        within("R10", f"{seq}: moved along the phase direction", rel(image(world(along_y), PROTOCOLS[seq]), np.roll(base, shift_px, axis=1)), 1e-12)
        with_fov = image(world(along_x), dict(PROTOCOLS[seq], centre_x_mm=shift_mm))
        within("R10", f"{seq}: object and field of view moved together", rel(with_fov, base), 1e-12)


def _displacement_px(material, b0_t, cf_offset_hz=0.0, **protocol):
    """Magnitude-weighted centroid along x of a centred disc (soft edge), in pixels from the centre of the image."""
    w = world([(material, disc(0.0, 0.0, 30.0))], b0_t=b0_t, edge=SMOOTH_EDGE, cf_offset_hz=cf_offset_hz)
    img = np.abs(image(w, dict(sequence="se", tr_ms=500.0, matrix=96, fov_mm=240.0, bandwidth_hz_per_px=100.0, **protocol))[0])
    x = np.arange(img.shape[1]) - img.shape[1] // 2
    return float((img.sum(0) * x).sum() / img.sum())


def _fat_displacement_px(b0_t, **protocol):
    return _displacement_px("subcut_fat", b0_t, **protocol)


def test_r11_reversing_the_readout_reverses_the_fat_displacement():
    plus, minus = _fat_displacement_px(1.5), _fat_displacement_px(1.5, ro_polarity=-1)
    mean_ppm = sum(ppm * amp for ppm, amp in tissue.FAT_PEAKS) / sum(amp for _, amp in tissue.FAT_PEAKS)
    expected = -mean_ppm * 1e-6 * 42.577478518e6 * 1.5 / 100.0           # amplitude-weighted fat frequency at 100 Hz per pixel
    assert 1.9 < expected < 2.0                                          # fat moves towards +readout (kspace.py docstring)
    within("R11", "displacement vs the mean fat frequency over the bandwidth, pixels", abs(plus - expected), 0.01)
    within("R11", "sum of the displacements at the two polarities, pixels", abs(plus + minus), 0.01)
    # two mechanisms, one law: water that is off resonance by the mean fat frequency (a field offset, displaced on the
    # master grid) lands where fat lands (a spectrum, displaced through the readout time)
    water = _displacement_px("muscle", 1.5, cf_offset_hz=mean_ppm * 1e-6 * 42.577478518e6 * 1.5)
    within("R11", "off-resonant water vs fat at the same frequency, pixels", abs(water - plus), 0.01)


def test_r12_doubling_the_field_doubles_the_fat_displacement():
    low, high = _fat_displacement_px(1.5), _fat_displacement_px(3.0)
    within("R12", "displacement at 3 T minus twice that at 1.5 T, pixels", abs(high - 2 * low), 0.01)


# ---------------------------------------------------------------------------------------------- the session
def _cfg(**kw):
    return dict(task_id="metamorphic", seed="000102030405060708090a0b0c0d0e0f", world_spec=dict(kind="qa", n=128), budget_s=600.0,
                max_requests=24, task=dict(regions=[]), **kw)


A_SCAN = dict(sequence="spgr", tr_ms=20.0, flip_deg=15.0, matrix=48)
A_OTHER = dict(sequence="epi", tr_ms=100.0, matrix=32)
B_SCAN = dict(sequence="se", tr_ms=300.0, matrix=96, fov_mm=400.0)


def _play(cfg, *protocols):
    s = session.ScannerSession(cfg)
    out = [s.scan({"protocol": p}) for p in protocols]
    return s, out


def test_r13_an_earlier_scan_does_not_change_a_later_one():
    alone, (b0,) = _play(_cfg(), B_SCAN)
    after_a, (_, b1) = _play(_cfg(), A_SCAN, B_SCAN)
    after_other, (_, b2) = _play(_cfg(), A_OTHER, B_SCAN)
    for other in (b1, b2):
        exact("R13", "realised protocol and cost of B", other[0]["realised"] == b0[0]["realised"] and other[0]["cost_s"] == b0[0]["cost_s"])
    rp = b0[0]["realised"]
    clean = [s.engine.acquire(rp, None) for s in (alone, after_a, after_other)]
    exact("R13", "noise-free B", np.array_equal(clean[0], clean[1]) and np.array_equal(clean[0], clean[2]))
    exact("R13", "noisy B after two different scans (same scan index)", b1[0]["data_sha256"] == b2[0]["data_sha256"])
    assert b0[0]["data_sha256"] != b1[0]["data_sha256"]                  # another index draws other noise


def test_r14_another_noise_draw_changes_the_noise_and_nothing_else():
    s0, (r0,) = _play(_cfg(), B_SCAN)
    s1, (r1,) = _play(_cfg(noise_draw=1), B_SCAN)
    rp = r0[0]["realised"]
    exact("R14", "noise-free image and realised protocol", np.array_equal(s0.engine.acquire(rp, None), s1.engine.acquire(rp, None))
          and r0[0]["realised"] == r1[0]["realised"])
    img = [r[1]["magnitude"][0].astype(float) * np.exp(1j * r[1]["phase"][0].astype(float)) for r in (r0, r1)]
    d = img[1] - img[0]
    sd = s0.engine.noise_sd(rp)
    assert d.size >= 4000 and sd > 0
    within("R14", "SD of the difference vs sqrt(2) x analytic SD", abs(d.real.std() / (math.sqrt(2) * sd) - 1.0), 0.06)
    within("R14", "mean of the difference, in standard errors", abs(d.real.mean()) / (math.sqrt(2) * sd / math.sqrt(d.size)), 4.0)


def test_r15_read_outs_of_nested_circles_are_consistent_with_the_ring_between_them():
    s, (r,) = _play(_cfg(), B_SCAN)
    rp = r[0]["realised"]
    data = s.engine.acquire(rp, np.random.default_rng(3))
    big, small = dict(id="C", y_mm=5.0, x_mm=-10.0, radius_mm=70.0), dict(id="c", y_mm=0.0, x_mm=-20.0, radius_mm=30.0)
    rows = {x["id"]: x for x in session.roi_readouts(data, [big, small], session.fov_yx(rp), session.centre_yx(rp))}
    y, x = session.pixel_positions(data.shape, session.fov_yx(rp))
    inside = lambda c: (y[:, None] - c["y_mm"]) ** 2 + (x[None, :] - c["x_mm"]) ** 2 <= c["radius_mm"] ** 2   # noqa: E731
    ring = inside(big) & ~inside(small)
    assert rows["C"]["n_pixels"] - rows["c"]["n_pixels"] == int(ring.sum()) > 500
    total = lambda q: q["n_pixels"] * q["mean_magnitude"][0]             # noqa: E731
    want = float(np.abs(data[0][ring]).sum())
    within("R15", "n_C mean_C - n_c mean_c vs the sum over the ring", abs(total(rows["C"]) - total(rows["c"]) - want) / want, 1e-8)
    cm = lambda q: q["n_pixels"] * q["complex_mean_magnitude"][0] * np.exp(1j * q["complex_mean_phase_rad"][0])   # noqa: E731
    want_c = data[0][ring].sum()
    within("R15", "the same for the complex mean", abs(cm(rows["C"]) - cm(rows["c"]) - want_c) / abs(want_c), 1e-8)
    exact("R15", "|complex mean| <= mean |S|", all(q["complex_mean_magnitude"][0] <= q["mean_magnitude"][0] for q in rows.values()))


# ---------------------------------------------------------------------------------------------- price
PRICED = dict(se=dict(sequence="se", tr_ms=500.0), spgr=dict(sequence="spgr", tr_ms=20.0, flip_deg=15.0),
              epi=dict(sequence="epi", tr_ms=400.0))


def _plan(**p):
    pl = console.plan(p, EXAM)
    assert pl["accept"], pl["violations"]
    return pl


@pytest.mark.parametrize("seq", sorted(PRICED))
def test_r16_price_relations(seq):
    base = _plan(**PRICED[seq])
    for k in (2, 4, 7):
        more = _plan(**PRICED[seq], averages=k)
        within("R16", f"{seq}: acquisition seconds at {k} averages vs {k} x", abs(more["scan_s_parts"]["acquisition_s"] - k * base["scan_s_parts"]["acquisition_s"]), 1e-9)
        exact("R16", f"{seq}: overhead and dummy time at {k} averages", more["scan_s_parts"]["dummy_s"] == base["scan_s_parts"]["dummy_s"]
              and more["scan_s_parts"]["overhead_s"] == base["scan_s_parts"]["overhead_s"])
    for name, values in (("partial_fourier", (0.55, 0.65, 0.75, 0.9, 1.0)), ("phase_fov", (0.25, 0.5, 0.75, 1.0)),
                         ("phase_oversampling", (0.0, 0.25, 0.5, 1.0))):
        t = [_plan(**PRICED[seq], **{name: v})["scan_s"] for v in values]
        exact("R16", f"{seq}: scan time never falls with {name}", all(b >= a for a, b in zip(t, t[1:])))
        if seq != "epi":                                                # one line per cycle: more lines cost more
            assert all(b > a for a, b in zip(t, t[1:])), (name, t)
        else:                                                           # single shot: the price does not depend on the lines
            assert len(set(t)) == 1


def test_r16_spin_echo_price_table_by_hand():
    """scan_s = 3.0 (overhead) + 1 dummy cycle + TR x lines x averages; lines = ceil(matrix x phase_fov x (1 + os) x pf)."""
    table = [(dict(), 128, 67.5), (dict(partial_fourier=0.55), 71, 39.0), (dict(partial_fourier=0.75), 96, 51.5),
             (dict(phase_oversampling=0.5), 192, 99.5), (dict(averages=4), 128, 259.5), (dict(phase_fov=0.5), 64, 35.5),
             (dict(phase_fov=0.5, partial_fourier=0.75, phase_oversampling=0.5, averages=4), 72, 147.5)]
    for extra, lines, seconds in table:
        pl = _plan(sequence="se", tr_ms=500.0, matrix=128, **extra)
        assert (pl["realised"]["lines"], pl["scan_s"]) == (lines, seconds), extra
        assert seconds == 3.0 + 0.5 + 0.5 * lines * extra.get("averages", 1)
    epi = _plan(sequence="epi", tr_ms=200.0, averages=3)
    assert epi["scan_s"] == pytest.approx(3.0 + 0.2 * (3 + 2), abs=1e-9)   # overhead + TR x (averages + 2 dummy shots)
    # hard pulses are a 3D acquisition: the acquisition (not the overhead or the dummy cycle) times ceil(160 mm slab / slice)
    slab = _plan(sequence="se", tr_ms=500.0, matrix=128, excitation="nonselective", slice_mm=10.0)
    assert slab["realised"]["partitions"] == 16 and slab["scan_s"] == 3.0 + 0.5 + 0.5 * 128 * 16


# ---------------------------------------------------------------------------------------------- the reward
def test_r17_the_same_deliverable_written_another_way_earns_the_same():
    case = scoring.case(NT_TABLE, "L1")
    answer = NT_TABLE.at()(case)
    as_text = {"ti_ms": {k: float(f"{v:.2f}") for k, v in answer["ti_ms"].items()}}
    a, b = (case.task.score(case.cfg, x, world=case.world, engine=case.engine) for x in (answer, as_text))
    exact("R17", "NullTissue: the same numbers re-parsed from text", a["passed"] is True and a == b)


@pytest.mark.skipif(NW is None, reason=NW_GONE)
def test_r17_nowrap_v0_the_same_geometry_written_another_way_and_more_scans_earn_the_same():
    cfg = NW.make_instance(derive_seed("Scanner/NoWrap-v0", 1, "train", 3, environ={}), 1, task_id="metamorphic")
    w = W.make_world(cfg["seed"], cfg["world_spec"])
    e = kspace.Engine(w)
    geo = dict(pe_dir="x", fov_mm=132.0, phase_fov=1.0, matrix=66, phase_oversampling=0.0)       # 66 lines: partial credit
    base = NW.score(cfg, geo, 1, True, w, e)
    assert 0.0 < base["reward"] < 1.0
    other = NW.score(cfg, dict(geo, phase_fov=0.9999), 1, True, w, e)    # rounds to the same 66 lines
    exact("R17", "NoWrap-v0: phase_fov 1.0 and 0.9999 (same lines)", other["lines"] == base["lines"] and other["reward"] == base["reward"])
    no_os = {k: v for k, v in geo.items() if k != "phase_oversampling"}
    exact("R17", "NoWrap-v0: oversampling 0 given or left out", NW.score(cfg, no_os, 1, True, w, e)["reward"] == base["reward"])
    exact("R17", "NoWrap-v0: 1 scan or 9 scans before the answer", NW.score(cfg, geo, 9, True, w, e)["reward"] == base["reward"])


def test_r18_the_order_of_the_regions_does_not_matter_and_a_worse_answer_never_earns_more():
    case = scoring.case(NT_TABLE, "L1")
    score = lambda ans: case.task.score(case.cfg, ans, world=case.world, engine=case.engine)       # noqa: E731
    good = NT_TABLE.at()(case)
    turned = {"ti_ms": dict(reversed(list(good["ti_ms"].items())))}
    exact("R18", "NullTissue: regions in reverse order", score(turned)["passed"] is True and score(turned)["null_ratio"] == score(good)["null_ratio"])
    for side in (-1, 1):
        ladder = [score(NT_TABLE.at(1.0 + side * step)(case)) for step in (0.0, 0.03, 0.06, 0.12, 0.3)]
        errors, passes, progress = ([s[k] for s in ladder] for k in ("error", "passed", "progress"))
        exact("R18", f"NullTissue: further from the null ({'late' if side > 0 else 'early'}) leaves more signal, never turns a fail "
              "into a pass and never raises the progress",
              all(y > x for x, y in zip(errors, errors[1:])) and all(not y or x for x, y in zip(passes, passes[1:]))
              and passes[0] and not passes[-1] and all(y <= x for x, y in zip(progress, progress[1:])) and progress[0] == 1.0)


@pytest.mark.skipif(T1 is None, reason=T1_GONE)
def test_r18_t1transmit_v0_the_order_of_the_regions_does_not_matter_and_a_worse_answer_never_earns_more():
    cfg = T1.make_instance(derive_seed("Scanner/T1Transmit-v0", 2, "train", 5, environ={}), 2, task_id="metamorphic")
    w = W.make_world(cfg["seed"], cfg["world_spec"], layout_only=True)
    truth = {k: v["t1_ms"] for k, v in W.truth(w).items()}
    answer = lambda f, order: {"units": "ms", "t1_ms": {k: truth[k] * f for k in order}}       # noqa: E731
    ids = sorted(truth)
    a, b = (T1.score(cfg, answer(1.12, order), 0.05, 1, True, w) for order in (ids, ids[::-1]))
    exact("R18", "T1Transmit-v0: regions in reverse order", a["reward"] == b["reward"] and 0.0 < a["reward"] < 1.0)
    rewards = [T1.score(cfg, answer(f, ids), 0.05, 1, True, w)["reward"] for f in (1.0, 1.04, 1.08, 1.15, 1.3, 1.6, 2.5)]
    exact("R18", "T1Transmit-v0: reward never rises with the error", all(y <= x for x, y in zip(rewards, rewards[1:])) and rewards[0] == 1.0
          and rewards[-1] == 0.0)


# ---------------------------------------------------------------------------------------------- the slice profile
def _same_timing(w, protocol):
    """``protocol`` as (slice-selective, hard-pulse) realised protocols with one echo time (the longer minimum)."""
    hard = dict(protocol, excitation="nonselective")
    if "te_ms" in console.SEQUENCES[protocol["sequence"]]:
        te = max(realised(w, protocol)["te_ms"], realised(w, hard)["te_ms"])
        protocol, hard = dict(protocol, te_ms=te), dict(hard, te_ms=te)
    return realised(w, protocol), realised(w, hard)


@pytest.mark.hard_pulse
@pytest.mark.parametrize("seq", ["se", "ir", "spgr", "afi", "bssfp", "epi"])
def test_r19_an_ideal_rectangular_slice_profile_gives_the_image_of_hard_pulses(seq):
    w = world()
    selective, hard = _same_timing(w, PROTOCOLS[seq])
    assert hard["partitions"] > 1 and selective["partitions"] == 1 and selective["te_ms"] == hard["te_ms"]
    want = kspace.Engine(w).acquire(hard, None)
    within("R19", f"{seq}: rectangular profile vs hard pulses", rel(kspace.Engine(w, rf=dict(profile="rect")).acquire(selective, None), want), 1e-12)
    assert rel(kspace.Engine(w).acquire(selective, None), want) > 0.05   # the real profile does change the image


@pytest.mark.hard_pulse
def test_r20_at_a_small_flip_angle_the_slice_integral_is_the_hard_pulse_signal():
    """Small-tip limit: the profile is the Fourier transform of the pulse, so its integral is the pulse's centre
    amplitude: ``thickness`` slice thicknesses of hard-pulse signal, computed here from the documented pulse (128
    samples of a sinc of time-bandwidth 4 under a Hann window, summing to 1). The remainder falls with the square of
    the flip angle."""
    t = (np.arange(128) + 0.5) / 128 - 0.5
    thickness = 128 / (4.0 * (np.sinc(4.0 * t) * 0.5 * (1 + np.cos(2 * np.pi * t))).sum())
    assert 0.98 < thickness < 0.99
    w, left = world(), {}
    for flip in (1.0, 2.0, 4.0):
        selective, hard = _same_timing(w, dict(GEO, sequence="spgr", tr_ms=4000.0, flip_deg=flip))
        e = kspace.Engine(w)
        left[flip] = rel(e.acquire(selective, None), thickness * e.acquire(hard, None))
    within("R20", "spgr at 1 degree, TR 4 s: slice-selective vs effective thickness x hard pulses", left[1.0], 1e-4)
    within("R20", "the remainder at 2 degrees over 4 x the remainder at 1 degree, minus 1", abs(left[2.0] / (4 * left[1.0]) - 1.0), 0.1)
    within("R20", "the remainder at 4 degrees over 16 x the remainder at 1 degree, minus 1", abs(left[4.0] / (16 * left[1.0]) - 1.0), 0.1)

# ---- the two migrated families (v1, family contract)
def test_r17_nowrap_v1_the_same_geometry_written_another_way_earns_the_same():
    cfg = NW1.make_instance(derive_seed("Scanner/NoWrap-v1", 2, "train", 3, environ={}), 2, task_id="metamorphic")
    w = W.make_world(cfg["seed"], cfg["world_spec"])
    e = kspace.Engine(w)
    geo = dict(pe_dir="x", fov_mm=260.0, phase_fov=1.0, matrix=130, centre_y_mm=0.0, centre_x_mm=0.0, rotation_deg=0.0,
               phase_oversampling=0.0)                                   # holds any object: every check passes but the scan time
    base = NW1.score(cfg, geo, w, e)
    assert [k for k, ok in base["checks"].items() if not ok] == ["scan_time_within_factor"] and base["lines"] == 130
    same = lambda a, b: (a["lines"], a["error"], a["checks"], a["passed"]) == (b["lines"], b["error"], b["checks"], b["passed"])   # noqa: E731
    other = NW1.score(cfg, dict(geo, phase_fov=0.9999), w, e)             # rounds to the same 130 lines
    exact("R17", "NoWrap: phase_fov 1.0 and 0.9999 (same lines)", same(other, base))
    no_os = {k: v for k, v in geo.items() if k != "phase_oversampling"}
    exact("R17", "NoWrap: oversampling 0 given or left out", NW1.score(cfg, no_os, w, e) == base)
    exact("R17", "NoWrap: phase along x, or phase along y after a quarter turn", same(NW1.score(cfg, dict(geo, pe_dir="y", rotation_deg=90.0), w, e), base))


def test_r18_t1transmit_v1_the_order_of_the_regions_does_not_matter_and_a_worse_answer_never_passes():
    cfg = T11.make_instance(derive_seed("Scanner/T1Transmit-v1", 2, "train", 5, environ={}), 2, task_id="metamorphic")
    w = W.make_world(cfg["seed"], cfg["world_spec"], layout_only=True)
    truth = {k: v["t1_ms"] for k, v in W.truth(w).items()}
    answer = lambda f, order: {"t1_ms": {k: truth[k] * f for k in order}}       # noqa: E731
    ids = sorted(truth)
    a, b = (T11.score(cfg, answer(1.12, order), w) for order in (ids, ids[::-1]))
    exact("R18", "T1Transmit: regions in reverse order", a["passed"] is False and b["passed"] is False and a["error"] == b["error"]
          and a["checks"] == b["checks"])
    passes = [T11.score(cfg, answer(f, ids), w)["passed"] for f in (1.0, 1.04, 1.08, 1.15, 1.3, 1.6, 2.5)]
    exact("R18", "T1Transmit: a worse answer never passes where a better one fails", passes == [True, True, True, False, False, False, False])


def test_every_relation_was_exercised():
    assert sorted(MEASURED, key=lambda k: int(k[1:])) == list(RELATIONS) and len(RELATIONS) == 20
