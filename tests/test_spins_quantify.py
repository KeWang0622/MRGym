"""Fitting, Cramer-Rao bounds, water referencing, line width and signal-to-noise ratio."""
import math

import numpy as np
import pytest

from mrgym.spins import evolve, metabolites, quantify, sequences

DWELL, N = 0.5, 512


@pytest.fixture(scope="module")
def basis():
    seq = sequences.press(30.0)
    return np.array([sequences.simulate(metabolites.get(n), seq).fid(N, DWELL, 150.0) for n in ("naa", "creatine", "choline")])


def test_model_is_linear_in_the_amplitudes_and_carries_shift_width_phase(basis):
    y = quantify.model(basis, DWELL, [1.0, 2.0, 0.5], 3.0, 2.0, 0.3)
    t = 1e-3 * DWELL * np.arange(N)
    ref = np.exp(0.3j + (2j * np.pi * 3.0 - np.pi * 2.0) * t) * (basis[0] + 2 * basis[1] + 0.5 * basis[2])
    assert np.abs(y - ref).max() < 1e-12
    assert quantify.model(basis[0], DWELL, [2.0]) == pytest.approx(2 * basis[0])


def test_fit_recovers_parameters_and_respects_fixed_ones(basis):
    y = quantify.model(basis, DWELL, [10.0, 7.0, 2.0], -1.5, 1.0, -0.2)
    fit = quantify.fit(y, basis, DWELL)
    assert fit.success and fit.amplitudes == pytest.approx([10.0, 7.0, 2.0], abs=1e-7)
    assert (fit.shift_hz, fit.width_hz, fit.phase_rad) == pytest.approx((-1.5, 1.0, -0.2), abs=1e-8)
    assert fit.residual_sd < 1e-8 and np.abs(fit.fitted - y).max() < 1e-7
    held = quantify.fit(y, basis, DWELL, start=(-1.5, 1.0, -0.2), fixed=("shift", "width", "phase"))
    assert held.amplitudes == pytest.approx([10.0, 7.0, 2.0], abs=1e-9)
    wrong = quantify.fit(y, basis, DWELL, start=(0.0, 0.0, 0.0), fixed=("shift", "width", "phase"))
    assert np.abs(wrong.amplitudes - [10.0, 7.0, 2.0]).max() > 0.1
    with pytest.raises(ValueError):
        quantify.fit(y[:-1], basis, DWELL)
    with pytest.raises(ValueError):
        quantify.fit(y, basis, DWELL, band_hz=(0.0, 1.0))


def test_residual_sd_estimates_the_noise(basis):
    rng = np.random.default_rng(5)
    y = quantify.add_noise(quantify.model(basis, DWELL, [10.0, 7.0, 2.0]), 0.3, rng)
    assert quantify.fit(y, basis, DWELL).residual_sd == pytest.approx(0.3, rel=0.08)


def test_bounds_shrink_with_noise_and_grow_with_free_parameters(basis):
    amps = [10.0, 7.0, 2.0]
    a = quantify.crb(basis, DWELL, amps, 0.5)
    b = quantify.crb(basis, DWELL, amps, 0.25)
    assert a["amplitudes"] == pytest.approx(2 * b["amplitudes"])
    fixed = quantify.crb(basis, DWELL, amps, 0.5, fixed=("shift", "width", "phase"))
    assert np.all(fixed["amplitudes"] <= a["amplitudes"] + 1e-12) and fixed["shift_hz"] == 0.0
    # with everything else known the bound of one amplitude is sigma / norm of its basis decay (orthogonal case)
    alone = quantify.crb(basis[:1], DWELL, [10.0], 0.5, fixed=("shift", "width", "phase"))
    assert alone["amplitudes"][0] == pytest.approx(0.5 / np.linalg.norm(basis[0]))
    band = quantify.crb(basis, DWELL, amps, 0.5, band_hz=(-400.0, -150.0))
    assert np.all(band["amplitudes"] >= a["amplitudes"] - 1e-12)
    assert quantify.fisher(basis, DWELL, amps, 0.5).shape == (6, 6)


def test_water_referenced_concentration():
    comp = {k: dict(volume=v, **metabolites.WATER_3T[k]) for k, v in (("gm", 0.6), ("wm", 0.4), ("csf", 0.0))}
    pure = dict(w=dict(volume=1.0, water=1.0, t1_ms=1e-3, t2_ms=1e12))
    # a voxel of pure water without relaxation: 2 protons of water against 3 of a metabolite at 10 mmol/kg
    s_w, s_m = 2 * quantify.WATER_MOLAL_MM, 3 * 10.0
    assert quantify.water_referenced(s_m, s_w, 3, 30.0, 2000.0, 1e-3, 1e12, pure) == pytest.approx(10.0)
    # more CSF with the same signals means less tissue for the same metabolite signal: a higher concentration
    a = quantify.water_referenced(1.0, 1e4, 3, 30.0, 2000.0, 1300.0, 160.0, comp)
    comp2 = {k: dict(volume=v, **metabolites.WATER_3T[k]) for k, v in (("gm", 0.45), ("wm", 0.3), ("csf", 0.25))}
    assert quantify.water_referenced(1.0, 1e4, 3, 30.0, 2000.0, 1300.0, 160.0, comp2) > a
    assert quantify.relaxation_factor(30.0, 2000.0, 1300.0, 160.0) == pytest.approx(math.exp(-30 / 160) * (1 - math.exp(-2000 / 1300)))
    assert quantify.WATER_MOLAL_MM == pytest.approx(1e6 / 18.015, rel=1e-4)
    with pytest.raises(ValueError):
        quantify.water_referenced(1.0, 1.0, 3, 30.0, 2000.0, 1300.0, 160.0, dict(csf=dict(volume=1.0, **metabolites.WATER_3T["csf"])))
    with pytest.raises(ValueError):
        quantify.water_referenced(1.0, 1.0, 3, 30.0, 2000.0, 1300.0, 160.0, dict(gm=dict(volume=0.0, **metabolites.WATER_3T["gm"])))


def test_linewidth_and_snr_of_a_known_line():
    ln = evolve.Lines(np.array([0.0]), np.array([1.0 + 0j]), np.array([0]), ("g",))
    f, s = evolve.spectrum(ln.fid(8192, 0.5, 100.0), 0.5)
    assert quantify.linewidth_hz(f, s.real, (-20, 20)) == pytest.approx(1e3 / (np.pi * 100.0), rel=5e-3)
    with pytest.raises(ValueError):
        quantify.linewidth_hz(f[4000:4100], np.ones(100), (-20, 20))
    with pytest.raises(ValueError):
        quantify.snr(f, s.real, (-20, 20), (500.0, 501.0))
    rng = np.random.default_rng(11)
    noisy = quantify.add_noise(ln.fid(8192, 0.5, 100.0), 0.01, rng)
    f, s = evolve.spectrum(noisy, 0.5, lb_hz=1.0)
    want = (1.0 / (1e3 / 100.0 + np.pi * 1.0)) / quantify.spectral_noise_sd(0.01, 8192, 0.5, lb_hz=1.0)
    assert quantify.snr(f, s.real, (-20, 20), (300.0, 900.0)) == pytest.approx(want, rel=0.06)
    # a slope under the noise region is removed before the standard deviation is taken
    tilted = s.real + 1e-4 * f
    assert quantify.snr(f, tilted, (-20, 20), (300.0, 900.0)) == pytest.approx(quantify.snr(f, s.real, (-20, 20), (300.0, 900.0)), rel=0.02)
    assert quantify.snr(f, tilted, (-20, 20), (300.0, 900.0), detrend=False) < 0.5 * want
