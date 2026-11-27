"""Engine 0.3.0: four behaviours that contradicted textbook practice on 0.2.0, and four adjustment parameters.

Every test of the first block states the textbook behaviour with a number and a tolerance and fails on engine 0.2.0
(the number 0.2.0 served stands in the comment above the assertion). The expected numbers are computed here, in this
file, from closed forms and from a Bloch iteration written for the purpose; nothing is taken from the engine.

  P1  partial Fourier costs signal-to-noise: the console synthesises the missing lines (homodyne detection), so the
      noise of a pixel rises and the resolution stays; the header's relative SNR is what a region measures
  P2  a spectrally selective pulse saturates the fat lines inside its band, not the whole six-peak spectrum
  P3  saturation before every excitation drives a steady state that depends on TR, T1 and the flip angle
  P4  the slice gradient is calibrated per pulse: the full width at half maximum of the profile is ``slice_mm``

  B1  (found after the audit) the RF phase advance of ``bssfp`` moves the dark bands towards higher frequency: the
      steady state depends on how far a spin runs ahead of the pulse, 2 pi df TR - phase cycle
  E1  (found while the above were corrected) echo-planar partial Fourier acquires the number of lines the realised
      protocol counts, with either blip polarity

The second block: ``cf_shift_hz``, ``tx_scale``, ``shim_x_hz_per_mm``, ``shim_y_hz_per_mm``. Each compensates a hidden
state of the system; the right setting brings the measured quantity to its ideal value.
"""
import math

import numpy as np
import pytest

from conftest import episode_cfg
from mrgym.scanner import console, kspace, rf, session, tissue
from mrgym.scanner import world as W

GAMMA = console.GAMMA_HZ_PER_T
SHORT_T1 = dict(pd=1.0, t1_ms=250.0, t2_ms=80.0, t2s_ms=60.0, t2p_ms=240.0, adc_um2_ms=0.0, fat=False)     # relaxed at TR 5 s
STILL_FAT = dict(tissue.table(3.0)["subcut_fat"], adc_um2_ms=0.0)                                           # no diffusion: the Bloch iteration has none


def disc(name, r=60.0):
    return (name, lambda y, x: np.hypot(y, x) - r)


def flat(shapes, b0_t=3.0, n=256, tissues=None, **system):
    return W.flat_world(W.shapes_phantom(shapes, b0_t=b0_t, n=n, extent_mm=256.0, tissues=tissues), **system)


def realised(world, protocol):
    pl = console.plan(protocol, world["console"])
    assert pl["accept"], pl["violations"]
    return pl["realised"]


def centre(img, k=6):
    ny, nx = img.shape[-2:]
    return img[..., ny // 2 - k:ny // 2 + k, nx // 2 - k:nx // 2 + k]


def centre_mean(img, k=6):
    return float(np.abs(centre(img, k)).mean())


# ================================================================================================== P1 partial Fourier
PF = dict(sequence="se", tr_ms=500, te_ms=20, matrix=64, fov_mm=256)


def homodyne_snr_factor(lines_full, lines):
    """SNR of a homodyne image relative to the fully sampled one (Noll, Nishimura & Macovski 1991, step weighting).
    Lines -n .. +n are acquired on both sides and weigh 1; the lines above +n exist on one side only and weigh 2 (they
    stand in for their missing conjugates); the real part after phase correction is the image. The noise variance of
    that real part is proportional to the sum of the squared weights, the signal is unchanged."""
    n = lines - lines_full // 2
    return math.sqrt(lines_full / ((2 * n + 1) + 4 * (lines_full // 2 - 1 - n)))


def test_p1_partial_fourier_saves_time_and_costs_snr_in_the_header():
    """0.2.0: relative SNR 5.66 / 6.53 / 7.54 for partial_fourier 1 / 0.75 / 0.55 (it rose while the scan got shorter)."""
    exam = dict(b0_t=3.0)
    full, three_quarter, least = (console.plan(dict(PF, partial_fourier=f), exam) for f in (1.0, 0.75, 0.55))
    assert full["scan_s"] > three_quarter["scan_s"] > least["scan_s"]
    assert [p["realised"]["lines"] for p in (full, three_quarter, least)] == [64, 48, 36]
    for pl in (three_quarter, least):
        want = homodyne_snr_factor(64, pl["realised"]["lines"])                  # 0.830 and 0.740: about sqrt(0.75), sqrt(0.55)
        assert pl["derived"]["relative_snr"] / full["derived"]["relative_snr"] == pytest.approx(want, rel=2e-3)
        assert pl["derived"]["relative_snr"] < full["derived"]["relative_snr"]
    assert homodyne_snr_factor(64, 48) == pytest.approx(math.sqrt(0.75), abs=0.04) and homodyne_snr_factor(64, 36) == pytest.approx(math.sqrt(0.55), abs=0.01)


def measured_snr(engine, rp, pairs=4, half=16):
    """SNR of a uniform region by the difference method (NEMA MS 1): mean of the magnitude over SD of the difference of
    two acquisitions / sqrt(2), averaged over ``pairs`` pairs."""
    sig, var = [], []
    for k in range(pairs):
        a, b = (np.abs(centre(engine.acquire(rp, np.random.default_rng(100 * k + j))[0], half)) for j in (1, 2))
        sig.append(0.5 * (a.mean() + b.mean()))
        var.append((a - b).var() / 2)
    return float(np.mean(sig) / math.sqrt(np.mean(var)))


def test_p1_measured_snr_falls_with_partial_fourier_as_the_header_says():
    """0.2.0: measured SNR of a region 256 / 289 / 316 for partial_fourier 1 / 0.75 / 0.55."""
    w = flat([disc("gm")], sigma0=0.02)
    eng = kspace.Engine(w)
    base = dict(PF, matrix=128)
    plans = {f: console.plan(dict(base, partial_fourier=f), w["console"]) for f in (1.0, 0.75, 0.55)}
    snr = {f: measured_snr(eng, pl["realised"]) for f, pl in plans.items()}
    for f in (0.75, 0.55):
        header = plans[f]["derived"]["relative_snr"] / plans[1.0]["derived"]["relative_snr"]
        textbook = homodyne_snr_factor(128, plans[f]["realised"]["lines"])
        assert snr[f] / snr[1.0] == pytest.approx(textbook, rel=0.05), (f, snr)
        assert snr[f] / snr[1.0] == pytest.approx(header, rel=0.05)
        assert eng.noise_sd(plans[f]["realised"]) / eng.noise_sd(plans[1.0]["realised"]) == pytest.approx(1 / textbook, rel=1e-9)


def test_p1_the_point_spread_of_a_partial_fourier_image_is_that_of_the_full_one():
    """A real object without phase: conjugate synthesis is exact, so the image is the fully sampled one (up to the one
    unpaired line at the edge of k-space). 0.2.0 zero-filled: the edge of the disc was blurred along the phase axis
    (largest difference to the full image 30 % of the plateau)."""
    w = flat([disc("gm", r=50.0)])
    eng = kspace.Engine(w)
    full = eng.acquire(realised(w, dict(PF, partial_fourier=1.0)))[0]
    part = eng.acquire(realised(w, dict(PF, partial_fourier=0.6)))[0]
    diff = np.abs(part - full)
    assert diff.max() < 0.04 * np.abs(full).max() and math.sqrt((diff ** 2).mean()) < 0.005 * np.abs(full).max()
    # along the phase axis the edge is one pixel wide, as in the full image (zero filling spread it over three)
    for img in (full, part):
        col = np.abs(img[:, 32])
        assert np.abs(np.diff(col)).max() > 0.75 * col.max()


def test_p1_a_fully_sampled_scan_is_reconstructed_by_the_plain_inverse_fft():
    w = flat([disc("gm")], sigma0=0.02)
    eng = kspace.Engine(w)
    rp = realised(w, PF)
    ks = eng.acquire_kspace(rp, np.random.default_rng(3))[0]
    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ks))) * ks.size / ((2 * rp["fov_mm"]) * rp["fov_pe_mm"])
    assert np.array_equal(eng.recon(ks, rp), img[:, 32:96])


# ================================================================================================== P2 fat saturation, per line
def outside_band(b0_t, half_width_ppm=0.65, centre_ppm=tissue.FAT_PPM):
    total = sum(a for _, a in tissue.FAT_PEAKS)
    return sum(a for ppm, a in tissue.FAT_PEAKS if abs(ppm - centre_ppm) > half_width_ppm) / total


def band(off_hz, bw_hz, edge_hz=8.0):
    """The band of the saturation pulse as the datasheet states it: 1 inside ``bw_hz``, 0 outside, edges 8 Hz soft."""
    return 0.5 * (1 - math.tanh((abs(off_hz) - bw_hz / 2) / edge_hz))


def fat_left(b0_t, centre_ppm, inside_left):
    """Share of the six-line fat signal that a perfect pulse centred at ``centre_ppm`` leaves when a saturated line
    keeps ``inside_left`` of its signal (the lines are in phase: the centre of a spin echo)."""
    f0, total = 1e-6 * GAMMA * b0_t, sum(a for _, a in tissue.FAT_PEAKS)
    hit = [(a / total, band((ppm - centre_ppm) * f0, 1.3 * f0)) for ppm, a in tissue.FAT_PEAKS]
    return sum(a * (1 - h * (1 - inside_left)) for a, h in hit)


def se_saturated(q, tr, te, t1, delay):
    """Spin echo with ideal 90 and 180 degree pulses and a saturation Mz -> q Mz ``delay`` ms before every excitation:
    longitudinal magnetisation at the excitation, relative to equilibrium. The 90 degree pulse leaves no Mz, so the
    recovery starts from zero in every repetition."""
    e = lambda t: math.exp(-t / t1)                                             # noqa: E731
    before = 1 - (2 - e(te / 2)) * e(tr - te / 2 - delay)                         # just before the saturation pulse
    return q * before * e(delay) + 1 - e(delay)


@pytest.mark.parametrize("b0_t", [1.5, 3.0])
def test_p2_a_perfect_fat_saturation_pulse_leaves_the_lines_outside_its_band(b0_t):
    """0.2.0: fat left 0.0000 with an efficiency of 1. Textbook: 21.9 % of the six-peak fat model (Hernando 2017) lies
    outside a band of 1.3 ppm around the main peak; what is inside recovers for the time between the saturation and
    the excitation."""
    assert outside_band(b0_t) == pytest.approx(0.219, abs=5e-4)
    w = flat([disc("subcut_fat")], b0_t=b0_t, fatsat_eff=1.0, fatsat_bw_hz=1.3e-6 * GAMMA * b0_t)
    eng = kspace.Engine(w)
    p = dict(sequence="se", tr_ms=2000, te_ms=20, matrix=64, fov_mm=256, excitation="nonselective")
    plain, sat = realised(w, p), realised(w, dict(p, fat_sat=True))
    left = centre_mean(eng.acquire(sat)) / centre_mean(eng.acquire(plain))
    t1 = w["tissues"]["subcut_fat"]["t1_ms"]
    inside = se_saturated(0.0, 2000.0, 20.0, t1, 7.0) / se_saturated(1.0, 2000.0, 20.0, t1, 7.0)      # 1.8 %: 7 ms of recovery
    textbook = outside_band(b0_t) + (1 - outside_band(b0_t)) * inside             # 0.233: a band with sharp edges
    assert left == pytest.approx(textbook, abs=0.012), (left, textbook)           # at 1.5 T the soft edge reaches the line at -2.6 ppm
    assert left == pytest.approx(fat_left(b0_t, tissue.FAT_PPM, inside), abs=1e-3)
    assert sat["fatsat_delay_ms"] == pytest.approx(7.0)                          # half the 8 ms pulse + 2 ms + half the 2 ms excitation
    # the default (slice-selective) scan: the same lines stay
    left_2d = centre_mean(eng.acquire(realised(w, dict(p, excitation="selective", fat_sat=True)))) / centre_mean(eng.acquire(realised(w, dict(p, excitation="selective"))))
    assert left_2d == pytest.approx(textbook, abs=0.015)


def test_p2_a_pulse_moved_onto_water_saturates_water_and_the_fat_lines_next_to_it():
    """1.5 T, the pulse shifted by +3.4 ppm (217 Hz) onto the water line. Water goes; of the fat only the two lines next
    to water (-0.39 and +0.60 ppm, 8.7 %) are inside the band. 0.2.0: fat left 1.0000."""
    b0_t = 1.5
    shift = -tissue.FAT_PPM * 1e-6 * GAMMA * b0_t
    near_water = 1 - outside_band(b0_t, centre_ppm=0.0)
    assert near_water == pytest.approx(0.087, abs=5e-4)
    p = dict(sequence="se", tr_ms=2000, te_ms=20, matrix=64, fov_mm=256, excitation="nonselective")
    left = {}
    for name in ("subcut_fat", "gm"):
        w = flat([disc(name)], b0_t=b0_t, fatsat_eff=1.0, fatsat_bw_hz=1.3e-6 * GAMMA * b0_t)
        eng = kspace.Engine(w)
        left[name] = centre_mean(eng.acquire(realised(w, dict(p, fat_sat=True, fatsat_offset_hz=shift)))) / centre_mean(eng.acquire(realised(w, p)))
    t1 = tissue.table(b0_t)["subcut_fat"]["t1_ms"]
    inside = se_saturated(0.0, 2000.0, 20.0, t1, 7.0) / se_saturated(1.0, 2000.0, 20.0, t1, 7.0)
    assert left["subcut_fat"] == pytest.approx(1 - near_water * (1 - inside), abs=0.02)            # 0.915 for a band with sharp edges
    assert left["subcut_fat"] == pytest.approx(fat_left(b0_t, 0.0, inside), abs=1e-3)              # 0.930: the line at +0.60 ppm sits on the soft edge
    assert left["gm"] < 0.02                                                                       # what recovers in 7 ms of a T1 of 1.2 s


# ================================================================================================== P3 driven steady state
def bloch_rf_spoiled(flip, tr, t1, t2, q=1.0, delay=0.0, phi_deg=117.0, n_iso=256):
    """RF-spoiled gradient echo by time stepping, written for this test: ``n_iso`` isochromats that precess by
    2 pi m / n_iso per repetition (one spoiler cycle across the voxel), a rotation by ``flip`` about an axis in the
    plane turned by the quadratic phase, T1 and T2 relaxation, and ``delay`` ms before every pulse a saturation
    Mz -> q Mz. Returns abs(mean transverse magnetisation) right after the last pulse."""
    n_tr = int(math.ceil(12 * t1 / tr)) + 30
    th = 2 * np.pi * np.arange(n_iso) / n_iso
    m = np.zeros((3, n_iso))
    m[2] = 1.0
    e2, e_rec, e_del = math.exp(-tr / t2), math.exp(-(tr - delay) / t1), math.exp(-delay / t1)
    phi, out = math.radians(phi_deg), 0.0
    for j in range(n_tr):
        ph = phi * j * (j + 1) / 2
        ax = np.array([math.cos(ph), math.sin(ph), 0.0])
        k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        rot = np.eye(3) + math.sin(flip) * k + (1 - math.cos(flip)) * (k @ k)     # Rodrigues: right-handed rotation about ax
        m = rot @ m
        out = abs(np.mean(m[0] + 1j * m[1]))
        mx, my = e2 * (m[0] * np.cos(th) - m[1] * np.sin(th)), e2 * (m[0] * np.sin(th) + m[1] * np.cos(th))
        mz = 1 + (m[2] - 1) * e_rec
        mz = 1 + (q * mz - 1) * e_del
        m = np.stack([mx, my, mz])
    return out


def test_p3_the_fat_saturation_factor_depends_on_tr_and_flip_angle():
    """0.2.0: fat with / without saturation = 0.1000 at every TR and flip angle (efficiency 0.9). Textbook: the
    longitudinal magnetisation of fat is driven by the saturation in every repetition and recovers only for the time
    between the saturation pulse and the excitation; the ratio is small at short TR and low flip angle and approaches
    (1 - efficiency) + recovery at long TR."""
    w = flat([disc("subcut_fat")], tissues=dict(subcut_fat=STILL_FAT), fatsat_eff=0.9)
    eng = kspace.Engine(w, fat_model="single")                                  # one line, inside the band: the steady state alone
    t1, t2 = STILL_FAT["t1_ms"], STILL_FAT["t2_ms"]
    got, want, ideal = {}, {}, {}
    for tr, fa in ((30, 10), (30, 60), (500, 60), (4000, 90)):
        p = dict(sequence="spgr", tr_ms=tr, flip_deg=fa, matrix=64, fov_mm=256, excitation="nonselective")
        sat, delay = realised(w, dict(p, fat_sat=True)), 6.5                     # half the 8 ms pulse + 2 ms + half the 1 ms hard pulse
        got[tr, fa] = centre_mean(eng.acquire(sat)) / centre_mean(eng.acquire(realised(w, p)))
        a = math.radians(fa)
        want[tr, fa] = bloch_rf_spoiled(a, tr, t1, t2, q=0.1, delay=delay) / bloch_rf_spoiled(a, tr, t1, t2)
        e_s, e_r, e1 = math.exp(-delay / t1), math.exp(-(tr - delay) / t1), math.exp(-tr / t1)
        ideal[tr, fa] = ((1 - e_s) + 0.1 * e_s * (1 - e_r)) / (1 - 0.1 * e1 * math.cos(a)) * (1 - e1 * math.cos(a)) / (1 - e1)
    for key in got:
        assert got[key] == pytest.approx(want[key], rel=0.01), (key, got[key], want[key])
    assert realised(w, dict(p, fat_sat=True))["fatsat_delay_ms"] == pytest.approx(6.5)
    # the closed form under ideal spoiling, (1 - Es + q Es (1 - Er)) (1 - E1 cos a) / ((1 - q E1 cos a) (1 - E1)), says the same
    # to a few percent of the ratio
    assert [round(ideal[k], 3) for k in ((30, 10), (30, 60), (500, 60), (4000, 90))] == [0.030, 0.170, 0.106, 0.115]
    for key in got:
        assert got[key] == pytest.approx(ideal[key], rel=0.06), key
    assert got[30, 10] < 0.3 * got[4000, 90] and got[30, 60] > 1.4 * got[4000, 90]


def test_p3_spin_echo_and_inversion_recovery_carry_the_saturation_in_their_steady_state():
    """Hard pulses, exact 90 and 180 degrees: the closed form of ``se_saturated``. At TR 60 ms the saturated fat is a
    larger share of the (small) unsaturated signal than at TR 2000 ms. 0.2.0: 0.1000 at both."""
    w = flat([disc("subcut_fat")], fatsat_eff=0.9)
    eng = kspace.Engine(w, fat_model="single")
    t1 = w["tissues"]["subcut_fat"]["t1_ms"]
    for tr in (60.0, 2000.0):
        p = dict(sequence="se", tr_ms=tr, te_ms=12.0, matrix=64, fov_mm=256, excitation="nonselective")
        got = centre_mean(eng.acquire(realised(w, dict(p, fat_sat=True)))) / centre_mean(eng.acquire(realised(w, p)))
        want = se_saturated(0.1, tr, 12.0, t1, 7.0) / se_saturated(1.0, tr, 12.0, t1, 7.0)
        assert got == pytest.approx(want, rel=1e-6), (tr, got, want)
    assert se_saturated(0.1, 60.0, 12.0, t1, 7.0) / se_saturated(1.0, 60.0, 12.0, t1, 7.0) > 0.2 > se_saturated(0.1, 2000.0, 12.0, t1, 7.0)


def test_p3_water_outside_the_band_is_untouched_by_fat_saturation():
    w = flat([disc("gm")], fatsat_eff=0.9)
    eng = kspace.Engine(w)
    for p in (dict(sequence="spgr", tr_ms=30, flip_deg=20), dict(sequence="se", tr_ms=500), dict(sequence="ir", tr_ms=2000, ti_ms=400),
              dict(sequence="epi", tr_ms=2000, fat_sat=False)):
        p = dict(p, matrix=64, fov_mm=256)
        on, off = eng.acquire(realised(w, dict(p, fat_sat=True)))[0], eng.acquire(realised(w, p))[0]
        assert np.abs(on - off).max() <= 1e-9 * np.abs(off).max(), p["sequence"]


# ================================================================================================== P4 slice calibration
def fwhm(alpha_deg, tbw, window="hann"):
    z = np.linspace(-3.0, 3.0, 12001)
    m = np.sin(rf.excitation(np.deg2rad(alpha_deg), tbw, z, window)[0])
    half = z[m >= 0.5 * m.max()]
    return float(half.max() - half.min())


@pytest.mark.parametrize("name", list(rf.PULSES))
def test_p4_every_pulse_excites_the_prescribed_thickness(name):
    """0.2.0: full width at half maximum of the 1 degree profile 1.196 (fast), 0.994 (normal), 0.998 (sharp) x slice_mm."""
    tbw = rf.PULSES[name][1]
    assert fwhm(1.0, tbw) == pytest.approx(1.0, abs=0.02)
    assert fwhm(1.0, tbw, "hamming") == pytest.approx(1.0, abs=0.02)             # the perturbation pulses are calibrated the same way


def test_p4_the_profile_differences_between_the_pulses_stay():
    wide = {name: fwhm(90.0, tbw) for name, (_, tbw) in rf.PULSES.items()}       # the profile broadens with the flip angle, most for the fast pulse
    assert wide["fast"] > wide["normal"] > wide["sharp"] > 1.0 and wide["fast"] == pytest.approx(1.17, abs=0.02)
    # transition band: share of the small-flip-angle signal that comes from outside the prescribed slice
    z = np.linspace(-6.0, 6.0, 24001)
    outside = {}
    for name, (_, tbw) in rf.PULSES.items():
        th, ph = rf.excitation(np.deg2rad(1.0), tbw, z)
        s = np.sin(th) * ph.real
        outside[name] = float(s[np.abs(z) > 0.5].sum() / s.sum())
    assert outside["fast"] > outside["normal"] > outside["sharp"] and outside["fast"] > 0.20 and outside["sharp"] < 0.08


def test_p4_the_served_signal_is_that_of_the_prescribed_slice_and_the_header_agrees():
    """Low flip angle, long TR: the signal is proportional to the excited thickness. 0.2.0: 1.225 (fast), 0.976
    (normal), 0.968 (sharp) x the hard-pulse value under one and the same relative SNR in the header."""
    w = flat([disc("gm")])
    eng = kspace.Engine(w)
    p = dict(sequence="spgr", tr_ms=3000, flip_deg=3, matrix=64, fov_mm=256)
    hard = centre_mean(eng.acquire(realised(w, dict(p, excitation="nonselective"))))
    snr = set()
    for name in rf.PULSES:
        pl = console.plan(dict(p, rf_pulse=name), w["console"])
        assert centre_mean(eng.acquire(pl["realised"])) / hard == pytest.approx(1.0, abs=0.035), name
        snr.add(pl["derived"]["relative_snr"])
    assert len(snr) == 1


def test_p4_the_slice_gradient_is_the_calibrated_one():
    exam = dict(b0_t=3.0)
    for name, (dur, tbw) in rf.PULSES.items():
        k = rf.fwhm_factor(tbw)
        pl = console.plan(dict(sequence="spgr", tr_ms=30, flip_deg=10, rf_pulse=name, slice_mm=5.0), exam)
        assert pl["timing"]["slice_gradient_mt_m"] == pytest.approx(k * tbw / (dur * 1e-3 * GAMMA * 5e-3) * 1e3, abs=1e-3)
    assert [round(rf.fwhm_factor(t), 4) for t in (2.0, 4.0, 8.0)] == [1.1966, 0.9944, 0.9997]
    # through-slice dephasing follows the calibrated profile: c cycles across the prescribed thickness are c / k cycles
    # across the nominal bandwidth of the gradient, so the first zero of the fast pulse lies at 1.197 cycles, not at 1
    k = rf.fwhm_factor(2.0)
    assert rf.dephasing_factor(np.array([1.0]), 2.0)[0] == pytest.approx(np.sinc(1 / k) * 0.5 * (1 + math.cos(math.pi / k)), abs=1e-12)
    assert rf.dephasing_factor(np.array([1.0]), 2.0)[0] > 0.01 and rf.dephasing_factor(np.array([k]), 2.0)[0] == pytest.approx(0.0, abs=1e-12)


# ================================================================================================== B1 the phase cycle of bssfp
def bssfp_pulse_by_pulse(flip_deg, tr_ms, t1_ms, t2_ms, df_hz, phase_cycle_deg):
    """Balanced SSFP, hard pulses, written out: pulse j turns the magnetisation by the flip angle about the axis at the
    angle j x phase cycle from x (counted the way a faster spin precesses); between two pulses the spin gains
    2 pi df TR of phase and relaxes. Returns the transverse magnetisation right after a pulse, as the receiver sees
    it when it follows the phase of that pulse: (Mx + i My) exp(-i j phase cycle) / i, which is +sin(flip) for one
    pulse from equilibrium."""
    a, psi, step = math.radians(flip_deg), 2 * math.pi * df_hz * tr_ms * 1e-3, math.radians(phase_cycle_deg)
    e1, e2 = math.exp(-tr_ms / t1_ms), math.exp(-tr_ms / t2_ms)
    m, out = np.array([0.0, 0.0, 1.0]), 0j
    for j in range(int(14 * t1_ms / tr_ms) + 20):
        ax = np.array([math.cos(j * step), math.sin(j * step), 0.0])
        # Rodrigues rotation by -a about the axis: a pulse about +x takes Mz to +My
        m = m * math.cos(a) - np.cross(ax, m) * math.sin(a) + ax * np.dot(ax, m) * (1 - math.cos(a))
        out = (m[0] + 1j * m[1]) * np.exp(-1j * j * step) / 1j
        m = np.array([e2 * (m[0] * math.cos(psi) - m[1] * math.sin(psi)), e2 * (m[0] * math.sin(psi) + m[1] * math.cos(psi)), 1 + (m[2] - 1) * e1])
    return out


def test_b1_the_dark_band_of_bssfp_moves_up_in_frequency_with_the_rf_phase_advance():
    """Textbook: the dark band is where the spin keeps step with the pulse phase, 2 pi df TR = phase cycle (mod 2 pi).
    With a phase cycle of 90 degrees at TR 5 ms that is df = +1 / (4 TR) = +50 Hz. Engine 0.2.0 put it at -50 Hz."""
    tr, t1, t2, flip = 5.0, 800.0, 80.0, 40.0
    f = np.arange(-99.0, 100.0, 1.0)
    bloch = np.array([abs(bssfp_pulse_by_pulse(flip, tr, t1, t2, v, 90.0)) for v in f])
    assert f[np.argmin(bloch)] == 50.0
    # the engine, in a field that rises by 2 Hz per mm along x: the band sits at x = +25 mm less the readout shift
    ph = W.shapes_phantom([("wm", lambda y, x: np.maximum(np.abs(y) - 40.0, np.abs(x) - 110.0))])
    world = W.flat_world(ph)
    n, d = ph["n"], ph["extent_mm"] / ph["n"]
    world["fields"]["b0_hz"] = 2.0 * np.broadcast_to((np.arange(n) - n // 2) * d, (n, n)).copy()
    engine = kspace.Engine(world)
    image = np.abs(engine.scan(dict(sequence="bssfp", tr_ms=tr, flip_deg=flip, fov_mm=256, matrix=256, bandwidth_hz_per_px=1000,
                                    phase_cycle_deg=90.0))["images"][0])
    x = np.arange(256) - 128.0
    prof = image[118:138].mean(0)
    near = np.abs(x) < 60
    assert x[near][np.argmin(prof[near])] == pytest.approx(50.0 / 2.0 - 50.0 / 1000.0, abs=1.0)        # 0.2.0: -25 mm


@pytest.mark.parametrize("phase_cycle", [0.0, 37.0, 90.0, 180.0, 270.0])
def test_b1_the_served_bssfp_signal_is_the_pulse_by_pulse_steady_state(phase_cycle):
    """The complex signal of a 3D (hard-pulse) scan at off-resonances on both sides, against the iteration above with
    the echo phase exp(i pi df TR) of TE = TR / 2. 0.2.0 agreed at 0 and 180 degrees only (0.19 M0 off at 90 and 270 degrees, df = -60 Hz)."""
    tr, flip = 6.0, 50.0
    world = flat([disc("wm")], b0_t=1.5)
    cls = world["tissues"]["wm"]
    for df in (-60.0, -20.0, 0.0, 35.0, 70.0):
        w = dict(world, system=dict(world["system"], cf_offset_hz=df))
        engine = kspace.Engine(w)
        rp = realised(w, dict(sequence="bssfp", tr_ms=tr, flip_deg=flip, matrix=64, fov_mm=256, phase_cycle_deg=phase_cycle, excitation="nonselective"))
        i = int(np.argmax(w["weights"][0][engine.sup[0]]))
        served = complex(np.asarray(engine.amplitudes(rp, 0)[0])[i] / (w["weights"][0][engine.sup[0]][i] * cls["pd"]))
        want = bssfp_pulse_by_pulse(flip, rp["tr_ms"], cls["t1_ms"], cls["t2_ms"], df, phase_cycle) * np.exp(1j * math.pi * df * rp["tr_ms"] * 1e-3)
        assert abs(served - want) < 1e-5, (df, served, want)


def test_b1_validation_rows_for_the_phase_cycle_and_for_the_slice():
    """S18: every phase cycle against the separately written simulation. R15: the served slice integral treats every
    column as a rotation about the pulse axis; against isochromats turned by their whole balanced pulse it is within
    1 % of the largest signal over the half of the period around the pass band (the difference inside the dark band is
    declared in docs/physics/limits.md, block 11)."""
    from mrgym import validate
    rows = {r["id"]: r for r in validate.check_bssfp_phase_cycle()}
    assert rows["S18"]["ok"] and rows["S18"]["value"] < 1e-6 and rows["R15"]["ok"] and rows["R15"]["value"] < 0.01
    assert all(r["ok"] for r in validate.check_bssfp_bands())


# ================================================================================================== the adjustments
ADJUST = dict(cf_shift_hz=(-500.0, 500.0, 0.0), tx_scale=(0.5, 2.0, 1.0), shim_x_hz_per_mm=(-2.0, 2.0, 0.0), shim_y_hz_per_mm=(-2.0, 2.0, 0.0))
EVERY = dict(spgr=dict(tr_ms=30, flip_deg=20), afi=dict(tr_ms=30, flip_deg=50), se=dict(tr_ms=500), ir=dict(tr_ms=2000, ti_ms=400),
             bssfp=dict(tr_ms=8, flip_deg=40), epi=dict(tr_ms=2000))


def two_echo(world, engine, rois, **adjust):
    """Off-resonance (Hz) in every circle from the phase between two gradient echoes 2 ms apart."""
    rp = realised(world, dict(sequence="spgr", tr_ms=40, flip_deg=15, matrix=64, fov_mm=256, n_echoes=2, echo_spacing_ms=2.0,
                              bandwidth_hz_per_px=1500, **adjust))
    rows = session.roi_readouts(engine.acquire(rp), rois, session.fov_yx(rp))
    dte = (rp["echo_times_ms"][1] - rp["echo_times_ms"][0]) * 1e-3
    return {r["id"]: float(np.angle(np.exp(1j * (r["complex_mean_phase_rad"][1] - r["complex_mean_phase_rad"][0]))) / (2 * np.pi * dte)) for r in rows}


@pytest.mark.parametrize("sequence", list(EVERY))
def test_every_sequence_takes_the_four_adjustments_and_the_defaults_change_nothing(sequence):
    exam = dict(b0_t=3.0)
    base = dict(EVERY[sequence], sequence=sequence, matrix=64)
    plain = console.plan(base, exam)
    for name, (lo, hi, default) in ADJUST.items():
        assert console.SEQUENCES[sequence][name][:2] == (lo, hi) and console.SEQUENCES[sequence][name][3] == default
        assert plain["realised"][name] == default
        assert console.plan(dict(base, **{name: default}), exam)["realised"] == plain["realised"]
        for bad, nearest in ((lo - 0.1, lo), (hi + 0.1, hi)):
            v = console.plan(dict(base, **{name: bad}), exam)["violations"][0]
            assert (v["code"], v["parameter"], v["nearest_valid"]) == ("OUT_OF_RANGE", name, nearest)
        assert name in console.PARAMETER_UNITS and name in console.PARAMETER_MEANING
    moved = console.plan(dict(base, cf_shift_hz=25.0, shim_x_hz_per_mm=0.2, shim_y_hz_per_mm=-0.1), exam)
    assert moved["accept"] and moved["scan_s"] == plain["scan_s"] and moved["timing"] == plain["timing"]      # they cost nothing
    assert "cf +25 Hz" in moved["header"] and "shim y -0.10 x +0.20 Hz/mm" in moved["header"] and "ADJ" not in plain["header"]
    assert console.request_of(moved["realised"])["cf_shift_hz"] == 25.0


def test_defaults_serve_the_data_of_a_protocol_without_adjustments(qa_world, qa_engine):
    for sequence, p in EVERY.items():
        rp = realised(qa_world, dict(p, sequence=sequence, matrix=64))
        again = realised(qa_world, dict(p, sequence=sequence, matrix=64, cf_shift_hz=0.0, tx_scale=1.0, shim_x_hz_per_mm=0.0, shim_y_hz_per_mm=0.0))
        assert rp == again
        bare = {k: v for k, v in rp.items() if k not in ADJUST}                  # a realised protocol written before the parameters existed
        assert np.array_equal(qa_engine.acquire(bare), qa_engine.acquire(rp)), sequence


def test_cf_shift_brings_the_measured_off_resonance_to_zero():
    """Hidden: the error of the frequency adjustment (``cf_offset_hz``). Measured: +37 Hz. With ``cf_shift_hz`` = the
    measured value the same measurement reads 0."""
    w = flat([disc("gm")], cf_offset_hz=37.0)
    eng = kspace.Engine(w)
    roi = [dict(id="c", y_mm=0.0, x_mm=0.0, radius_mm=20.0)]
    before = two_echo(w, eng, roi)["c"]
    assert before == pytest.approx(37.0, abs=0.2)
    assert two_echo(w, eng, roi, cf_shift_hz=round(before, 2))["c"] == pytest.approx(0.0, abs=0.2)


@pytest.mark.parametrize("sequence", list(EVERY))
def test_cf_shift_acts_on_the_whole_acquisition(sequence):
    """A system 150 Hz off, shifted back by 150 Hz, serves the image of a system that is on resonance: displacement,
    echo phase, bSSFP bands and the band of the fat-saturation pulse move together."""
    shapes = [disc("gm"), ("subcut_fat", lambda y, x: np.hypot(y - 20.0, x + 15.0) - 12.0)]
    p = dict(EVERY[sequence], sequence=sequence, matrix=64, fov_mm=256)
    if "fat_sat" in console.SEQUENCES[sequence]:
        p["fat_sat"] = True
    on, off = flat(shapes, fatsat_eff=0.95), flat(shapes, cf_offset_hz=150.0, fatsat_eff=0.95)
    want = kspace.Engine(on).acquire(realised(on, p))
    eng = kspace.Engine(off)
    assert np.abs(eng.acquire(realised(off, dict(p, cf_shift_hz=150.0))) - want).max() <= 1e-9 * np.abs(want).max()
    assert np.abs(eng.acquire(realised(off, p)) - want).max() > 0.02 * np.abs(want).max()


def test_tx_scale_brings_a_nominal_90_degree_pulse_to_90_degrees():
    """Hidden: the transmit gain (0.8 here). Measured by the double-angle method on a relaxed, non-selective scan:
    cos(actual) = S(2a) / (2 S(a)). With ``tx_scale`` = nominal / actual the nominal angle is the actual one."""
    w = flat([disc("gm")], tissues=dict(gm=SHORT_T1), tx_gain=0.8)
    eng = kspace.Engine(w)
    p = dict(sequence="spgr", tr_ms=5000, matrix=64, fov_mm=256, excitation="nonselective")

    def actual_deg(nominal, **kw):
        s1, s2 = (centre_mean(eng.acquire(realised(w, dict(p, flip_deg=f, **kw)))) for f in (nominal, 2 * nominal))
        return math.degrees(math.acos(s2 / (2 * s1)))
    before = actual_deg(45.0)
    assert before == pytest.approx(36.0, abs=0.05)
    scale = round(45.0 / before, 4)
    assert actual_deg(45.0, tx_scale=scale) == pytest.approx(45.0, abs=0.05)
    # a 90 degree nominal pulse is now a 90 degree pulse: the signal of the relaxed scan is at its maximum
    s = {f: centre_mean(eng.acquire(realised(w, dict(p, flip_deg=f, tx_scale=scale)))) for f in (80.0, 90.0)}
    s90 = centre_mean(eng.acquire(realised(w, dict(p, flip_deg=90.0))))
    assert s[90.0] > s[80.0] and s90 / s[90.0] == pytest.approx(math.sin(math.radians(72.0)), abs=1e-3)
    # the calibrated system is the system with the right gain, for the selective spin echo too (90 and 180 degrees scale)
    right = flat([disc("gm")], tissues=dict(gm=SHORT_T1))
    se = dict(sequence="se", tr_ms=600, matrix=64, fov_mm=256)
    want = kspace.Engine(right).acquire(realised(right, se))
    assert np.abs(eng.acquire(realised(w, dict(se, tx_scale=1.25))) - want).max() <= 1e-9 * np.abs(want).max()


def test_tx_scale_is_paid_in_rf_power_and_peak_b1():
    exam = dict(b0_t=3.0, b1rms_limit_ut=3.0)
    p = dict(sequence="bssfp", tr_ms=6.0, flip_deg=40.0, matrix=64)
    one, more = console.plan(p, exam), console.plan(dict(p, tx_scale=1.3), exam)
    assert more["limits"]["rf_power_percent"] == pytest.approx(1.3 ** 2 * one["limits"]["rf_power_percent"], rel=2e-3)
    # the refocusing pulse of a spin echo at twice the amplitude would exceed the peak B1: it is lengthened, as any pulse is
    se = [console.plan(dict(sequence="se", tr_ms=500, tx_scale=s), exam) for s in (1.0, 2.0)]
    assert se[1]["timing"]["rf_180_ms"] > se[0]["timing"]["rf_180_ms"] and se[1]["accept"]


def test_linear_shims_bring_the_field_of_a_region_to_zero():
    """Hidden: a linear field (0.40 Hz/mm along x, -0.25 Hz/mm along y) and 12 Hz. Measured at five circles; the shim
    terms of opposite sign and the frequency shift leave no off-resonance in any of them."""
    w = flat([disc("gm", r=80.0)])
    y, x = W.grid(w)
    w["fields"]["b0_hz"] = 12.0 + 0.40 * x - 0.25 * y
    eng = kspace.Engine(w)
    rois = [dict(id=k, y_mm=yy, x_mm=xx, radius_mm=8.0) for k, (yy, xx) in dict(c=(0, 0), n=(-40, 0), s=(40, 0), w=(0, -40), e=(0, 40)).items()]
    f = two_echo(w, eng, rois)
    gx, gy = (f["e"] - f["w"]) / 80.0, (f["s"] - f["n"]) / 80.0
    assert (gx, gy, f["c"]) == (pytest.approx(0.40, abs=0.01), pytest.approx(-0.25, abs=0.01), pytest.approx(12.0, abs=0.3))
    after = two_echo(w, eng, rois, shim_x_hz_per_mm=round(-gx, 3), shim_y_hz_per_mm=round(-gy, 3), cf_shift_hz=round(f["c"], 1))
    assert max(abs(v) for v in after.values()) < 0.5, after
    # the shimmed scan is the scan of a homogeneous system (here bSSFP, whose bands follow the field)
    even = flat([disc("gm", r=80.0)])
    p = dict(sequence="bssfp", tr_ms=8.0, flip_deg=40.0, matrix=64, fov_mm=256)
    want = kspace.Engine(even).acquire(realised(even, p))
    banded = eng.acquire(realised(w, p))
    shimmed = eng.acquire(realised(w, dict(p, shim_x_hz_per_mm=-0.40, shim_y_hz_per_mm=0.25, cf_shift_hz=12.0)))
    assert np.abs(shimmed - want).max() <= 1e-9 * np.abs(want).max() and np.abs(banded - want).max() > 0.2 * np.abs(want).max()


def test_shims_are_terms_of_the_patient_frame_whatever_the_field_of_view(qa_world, qa_engine):
    """A turned field of view sees the same shimmed field: the read-out of a region does not change with the rotation."""
    roi = [dict(id="c", y_mm=20.0, x_mm=-30.0, radius_mm=8.0)]
    p = dict(sequence="spgr", tr_ms=40, flip_deg=15, matrix=64, n_echoes=2, echo_spacing_ms=2.0, bandwidth_hz_per_px=1500,
             shim_x_hz_per_mm=0.3, shim_y_hz_per_mm=-0.2)
    out = []
    for rot in (0.0, 30.0):
        rp = realised(qa_world, dict(p, rotation_deg=rot))
        r = session.roi_readouts(qa_engine.acquire(rp), roi, session.fov_yx(rp), session.centre_yx(rp), session.rotation_of(rp))[0]
        out.append(float(np.angle(np.exp(1j * (r["complex_mean_phase_rad"][1] - r["complex_mean_phase_rad"][0]))) / (2 * np.pi * 2e-3)))
    plain = realised(qa_world, {k: v for k, v in p.items() if not k.startswith("shim")})
    r0 = session.roi_readouts(qa_engine.acquire(plain), roi, session.fov_yx(plain))[0]
    f0 = float(np.angle(np.exp(1j * (r0["complex_mean_phase_rad"][1] - r0["complex_mean_phase_rad"][0]))) / (2 * np.pi * 2e-3))
    assert out[0] == pytest.approx(out[1], abs=0.5)
    assert out[0] - f0 == pytest.approx(0.3 * -30.0 - 0.2 * 20.0, abs=0.5)        # -13 Hz: the two terms at the centre of the circle


def test_the_scanner_log_stores_the_adjustments_and_replays_them(qa_engine):
    s = session.ScannerSession(episode_cfg(), engine=qa_engine)
    p = dict(sequence="spgr", tr_ms=20.0, flip_deg=20.0, matrix=64, cf_shift_hz=-12.5, tx_scale=1.1, shim_x_hz_per_mm=0.05, shim_y_hz_per_mm=-0.07)
    plain = s.plan({"protocol": {k: v for k, v in p.items() if k not in ADJUST}})
    planned = s.plan({"protocol": p})
    assert planned["scan_s"] == plain["scan_s"]
    assert {k: planned["realised"][k] for k in ADJUST} == {k: p[k] for k in ADJUST}
    s.scan({"protocol": p})
    logged = [e for e in s.events if e["type"] == "scan"][0]["realised"]
    assert {k: logged[k] for k in ADJUST} == {k: p[k] for k in ADJUST}
    assert session.replay(s.cfg, s.events, head=s.receipt())["ok"]


# ================================================================================================== what is written about it
def test_the_datasheet_states_the_numbers_of_this_engine():
    """The numbers the agent is told about the corrected behaviours are computed here, not typed there."""
    from mrgym.envs import scanner_env
    md = " ".join(scanner_env.datasheet().split())
    g_lim = 0.8 * 40e-3
    thin = {name: rf.fwhm_factor(tbw) * tbw / (dur * 1e-3 * GAMMA * g_lim) * 1e3 for name, (dur, tbw) in rf.PULSES.items()}
    assert f"cannot go below {thin['normal']:.2f} mm (`fast`: {thin['fast']:.2f} mm, `sharp`: {thin['sharp']:.2f} mm)" in md
    assert f"the thinnest selective slice is {thin['normal']:.2f} mm (`fast`: {thin['fast']:.2f} mm, `sharp`: {thin['sharp']:.2f} mm)" in md
    k = {name: rf.fwhm_factor(tbw) for name, (_, tbw) in rf.PULSES.items()}
    assert f"k = {k['fast']:.3f} (`fast`), {k['normal']:.3f} (`normal`), {k['sharp']:.3f} (`sharp`)" in md
    g5 = [k[name] * tbw / (dur * 1e-3 * GAMMA * 5e-3) * 1e3 for name, (dur, tbw) in rf.PULSES.items()]
    assert "{:.1f}, {:.1f} and {:.1f} mT/m for a 5 mm slice".format(*g5) in md
    z = np.linspace(-6.0, 6.0, 24001)
    outside = []
    for _, tbw in rf.PULSES.values():
        th, ph = rf.excitation(np.deg2rad(1.0), tbw, z)
        s = np.sin(th) * ph.real
        outside.append(100 * s[np.abs(z) > 0.5].sum() / s.sum())
    assert "(about {:.0f} %, {:.0f} % and {:.0f} % of a small-flip-angle signal for `fast`, `normal` and `sharp`)".format(*outside) in md
    factor = lambda f: math.sqrt(128 / console.noise_lines(128, math.ceil(128 * f)))     # noqa: E731
    assert f"(x {factor(0.75):.2f} at 0.75, x {factor(0.55):.2f} at 0.55 for 128 lines)" in md and "sqrt((3 L - 2 l - 3) / L)" in md
    assert console.noise_lines(128, 96) == 3 * 128 - 2 * 96 - 3
    total = sum(a for _, a in tissue.FAT_PEAKS)
    ppm = ", ".join(f"{p:+.2f}" if p > 0 else f"{p:.2f}" for p, _ in tissue.FAT_PEAKS[:-1]) + f" and {tissue.FAT_PEAKS[-1][0]:+.2f} ppm"
    share = ", ".join(f"{100 * a / total:.1f}" for _, a in tissue.FAT_PEAKS[:-1]) + f" and {100 * tissue.FAT_PEAKS[-1][1] / total:.1f} %"
    assert f"({ppm} from water with {share} of the signal)" in md
    assert f"holds the first two, {100 * (1 - outside_band(3.0)):.0f} %" in md
    sat = console.plan(dict(sequence="se", tr_ms=500, fat_sat=True), dict(b0_t=3.0))["realised"]
    assert sat["fatsat_delay_ms"] == 7.0 and "7 ms before the centre of a 2 ms excitation" in md
    for name, (lo, hi, _) in ADJUST.items():
        assert f"`{name}`" in md and f"{lo:g} to {hi:g}" in md
    assert "`cf_shift_hz` | -500 to 500 | 0 |" in md and "`tx_scale` | 0.5 to 2 | 1 |" in md and "`shim_y_hz_per_mm`, `shim_x_hz_per_mm` | -2 to 2 | 0 |" in md
    assert "zero-filled" not in md
    assert "with a mean of 0.80 to 0.83 times the noise SD of a pixel for fractions up to 0.75" in md      # row K37 of mrgym validate measures it


def test_the_agent_finds_the_adjustments_behind_scanner_info_and_in_the_plan():
    import mrgym
    env = mrgym.make("Scanner/Explore-v0", seed=0)
    env.reset()
    step = lambda name, **kw: env.step({"name": name, "arguments": kw}).observation            # noqa: E731
    topic = step("scanner_info", topic="adjustments")
    assert not topic.is_error and all(k in topic.text for k in ADJUST) and "has to be measured" in topic.text
    assert "adjustments" in step("scanner_info", topic="physics").text
    geometry = step("scanner_info", topic="geometry").text
    assert all(f"{k}, " in geometry for k in ADJUST)
    one = step("scanner_info", parameter="tx_scale")
    assert not one.is_error and "0.5" in one.text and "2" in one.text
    p = dict(sequence="spgr", tr_ms=20, flip_deg=15, matrix=64)
    plain, moved = step("plan", **p), step("plan", **p, cf_shift_hz=-30, tx_scale=1.2, shim_x_hz_per_mm=0.1)
    assert "cf_shift_hz" not in plain.text and "tx_scale" not in plain.text
    assert " · cf_shift_hz -30 · tx_scale 1.2 · shim_x_hz_per_mm 0.1" in moved.text and "shim_y_hz_per_mm" not in moved.text
    assert moved.data["scan_s"] == plain.data["scan_s"]
    refused = step("plan", **p, shim="auto")
    assert refused.is_error and "shim_y_hz_per_mm" in refused.text and "cf_shift_hz" in refused.text      # the word of real consoles points at the parameters


# ================================================================================================== E1 echo-planar line count
@pytest.mark.parametrize("pe_polarity", [1, -1])
def test_epi_partial_fourier_acquires_the_lines_the_protocol_counts(pe_polarity):
    """The realised protocol states ``lines``; the header's scan time and the noise of a pixel are computed from it.
    The receiver must deliver exactly that many lines. Engine 0.2.0 took every line whose order in the echo train was
    at or after the first counted one, and with ``pe_polarity`` -1 that included the one line of the lattice that has
    no conjugate partner: one line more than counted (``lines`` + 1). Neither polarity acquires that line now."""
    world = flat([disc("a")], tissues=dict(a=SHORT_T1))
    eng = kspace.Engine(world)
    for matrix, fraction, phase_fov in ((64, 0.75, 1.0), (64, 0.55, 1.0), (96, 0.9, 1.0), (64, 0.75, 0.75), (48, 0.6, 1.0)):
        rp = realised(world, dict(sequence="epi", tr_ms=2000, matrix=matrix, partial_fourier=fraction, phase_fov=phase_fov, pe_polarity=pe_polarity,
                                  fat_sat=False))
        k_pe = rp["lines_full"]
        assert rp["lines"] < k_pe
        delivered = np.abs(eng.acquire_kspace(rp)[0]).sum(1) > 0
        assert int(delivered.sum()) == rp["lines"] == int(eng._acquired(rp).sum())
        assert not delivered[0]                                         # the line without a conjugate partner (index -k_pe / 2)
        line = np.arange(k_pe) - k_pe // 2
        old_rule = pe_polarity * line >= -(rp["lines"] - k_pe // 2)     # the mask of engine 0.2.0
        assert int(old_rule.sum()) == rp["lines"] + (1 if pe_polarity == -1 else 0)
        # the acquired lines are those of the other polarity, mirrored about the centre line
        twin = dict(rp, pe_polarity=-pe_polarity)
        assert np.array_equal(eng._acquired(rp)[1:], eng._acquired(twin)[1:][::-1])


def test_a_stored_gate_of_another_engine_claims_nothing():
    """Every gate stored in the package was measured on some engine. One measured on another engine than the installed
    one must not read as an admission: the id is not registered as admitted, its status line does not say ADMITTED,
    and its card says that the gate is not stored on this engine."""
    import mrgym
    from mrgym import cards, check, gate
    stale = []
    for env_id in mrgym.list_envs():
        stored = gate.load(env_id)
        if not stored or stored["engine"] == mrgym.ENGINE_ID:
            continue
        stale.append(env_id)
        spec, card = mrgym.spec(env_id), mrgym.card(env_id)
        assert spec.kind != "admitted", env_id
        assert "ADMITTED" not in card["status"] and stored["engine"] in card["status"] and mrgym.ENGINE_ID in card["status"], card["status"]
        assert check.claim_evidence(spec, stored, mrgym.ENGINE_ID)[0]
        md = cards.markdown(card)
        assert f"**Gate not stored on this engine** (`{mrgym.ENGINE_ID}`)" in md and "The table below is that older result." in md
    fresh = [i for i in mrgym.list_envs() if (gate.load(i) or {}).get("engine") == mrgym.ENGINE_ID]
    assert stale or fresh                                               # the test reads the package as it is shipped


def test_an_adjustment_is_not_a_field_of_a_delivered_examination_protocol():
    """``Scanner/ArtefactClinic-v0`` takes a whole protocol as its deliverable and locks, among others, the flip angle.
    An adjustment in it is refused (the episode goes on): ``tx_scale`` would be the locked flip angle by another name."""
    import mrgym
    from mrgym.envs.artefact_clinic import task
    env = mrgym.make("Scanner/ArtefactClinic-v0", level=1, seed=3)
    env.reset()
    cfg, exam = env.episode_config(), env.hidden_world()["console"]
    current = dict(cfg["task"]["current_protocol"])
    answer = dict(causes={f["id"]: [] for f in cfg["task"]["findings"]}, protocol=current)
    assert task.check(cfg, answer, exam) is None
    for name, value in (("tx_scale", 1.3), ("cf_shift_hz", 40.0), ("shim_x_hz_per_mm", 0.2)):
        why = task.check(cfg, dict(answer, protocol=dict(current, **{name: value})), exam)
        assert why and name in why and "leave it out" in why
        assert task.check(cfg, dict(answer, protocol=dict(current, **{name: mrgym.scanner.console.ADJUSTMENTS[name]})), exam) is None
    step = env.step({"name": "submit", "arguments": dict(answer, protocol=dict(current, tx_scale=1.3))})
    assert step.observation.is_error and not step.terminated and "tx_scale" in step.observation.text
