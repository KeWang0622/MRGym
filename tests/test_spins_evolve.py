"""Propagation over a block list, ensembles, lines and spectra."""
import math

import numpy as np
import pytest

from mrgym.design import pulses, rotation
from mrgym.spins import evolve, operators

AX = operators.SpinSystem("ax", (1.3, 4.1), ((0, 1, 7.0),), groups=("a", "x"))
ONE = operators.SpinSystem("w", (4.9,), ())


def lines(system, blocks, **args):
    return evolve.run(operators.compile(system, 3.0, 4.7, **args), blocks)[0]


def test_sign_and_phase_of_the_signal_follow_the_design_layer():
    """90 about +x: Mxy = -i, and the signal i Mxy is real and positive; a positive offset gains positive phase."""
    c = operators.compile(ONE, 3.0, 4.7)
    sec = c.sectors[0]
    rho = evolve.sandwich(evolve.hard_propagator(sec, c, math.pi / 2, 0.0), sec.fz())
    mxy, mz = evolve.magnetisation(sec, c, rho, 0)
    assert mxy == pytest.approx(-1j) and mz == pytest.approx(0.0, abs=1e-15)
    assert (mxy, mz) == pytest.approx(rotation.apply(*rotation.axis_angle((1, 0, 0), math.pi / 2)))
    got = lines(ONE, (evolve.rf(90.0), evolve.acquire()))
    f = c.freq_hz[0]
    assert f > 0 and got.freq_hz == pytest.approx([f]) and got.amp == pytest.approx([1.0])
    assert got.fid(3, 1.0) == pytest.approx(np.exp(2e-3j * np.pi * f * np.arange(3)))
    # pulse about +y: Mxy = +1; receiver phase of 90 degrees takes it back to +1 after the +i of the detection
    assert lines(ONE, (evolve.rf(90.0, 90.0), evolve.acquire(90.0))).amp == pytest.approx([1.0])
    assert lines(ONE, (evolve.rf(30.0), evolve.acquire())).amp == pytest.approx([0.5])


def test_block_constructors_refuse_what_cannot_be_played():
    with pytest.raises(ValueError):
        evolve.delay(-1.0)
    with pytest.raises(ValueError):
        evolve.rf(90.0, pathway="echo")
    with pytest.raises(ValueError):
        evolve.rf_pulse([1.0, 2.0], 0.1, model="exact")
    with pytest.raises(ValueError):
        evolve.rf_pulse([1.0, 2.0], 0.1, axis="x")
    with pytest.raises(ValueError):
        evolve.rf_pulse([[1.0, 2.0]], 0.1)
    with pytest.raises(ValueError):
        evolve.run(operators.compile(ONE), (evolve.Block("jump"),))
    assert evolve.delay(-1e-12).ms == 0.0 and evolve.spoil().orders == (0,)
    assert evolve.run(operators.compile(ONE), (evolve.rf(90.0),)) == []


def test_a_refocusing_pulse_between_crushers_keeps_b_squared_of_the_design_layer():
    """Echo of an uncoupled spin after an imperfect 180 about y between equal crushers: -b^2 conj(Mxy)."""
    for flip in (180.0, 150.0, 90.0):
        a, b = rotation.axis_angle((0, 1, 0), math.radians(flip))
        got = lines(ONE, (evolve.rf(90.0), evolve.delay(9.0), evolve.rf(flip, 90.0, pathway="refocus"), evolve.delay(9.0), evolve.acquire())).total()
        assert got == pytest.approx(1j * (-b * b * np.conj(-1j)))
        assert got == pytest.approx(math.sin(math.radians(flip) / 2) ** 2)


def test_explicit_gradient_ensemble_needs_enough_members():
    """Two members cannot tell order +1 from -1 (their phases differ by a whole turn): the selection fails with 2 and
    is exact with 4."""
    echo = math.sin(math.radians(120.0) / 2) ** 2

    def seq(n):
        return (evolve.rf(90.0), evolve.delay(5.0), evolve.gradient(1, "c", n), evolve.rf(120.0, 90.0), evolve.gradient(1, "c", n), evolve.delay(5.0), evolve.acquire())

    assert lines(ONE, seq(4)).total() == pytest.approx(echo)
    assert abs(lines(ONE, seq(2)).total() - echo) > 0.05


def test_an_axis_cannot_be_reopened_with_another_size_and_average_closes_it():
    c = operators.compile(ONE)
    with pytest.raises(ValueError):
        evolve.run(c, (evolve.rf(90.0), evolve.gradient(1, "c", 8), evolve.gradient(1, "c", 4), evolve.acquire()))
    # after average the axis may be opened again with any size; a gradient without a partner kills the signal
    got = evolve.run(c, (evolve.rf(90.0), evolve.gradient(1, "c", 8), evolve.average("c"), evolve.gradient(0, "c", 4), evolve.acquire()))[0]
    assert abs(got.total()) < 1e-12 or got.freq_hz.size == 0


def test_two_axes_at_once_are_independent():
    """A slice ensemble open across a crusher ensemble: the result is the product of the two selections."""
    p = pulses.sinc(180.0, 3.0, 4.0, thickness_mm=10.0, n=150)
    pos = np.linspace(-12.0, 12.0, 25)
    w = np.full(25, 1.0 / 10.0)
    c = operators.compile(ONE, 3.0, ONE.shifts_ppm[0])
    fast = evolve.run(c, (evolve.rf(90.0), evolve.rf_pulse(p.b1_ut, p.dt_ms, p.g_mt_m, 90.0, None, "x", pos, w, "refocus"), evolve.acquire()))[0].total()
    slow = evolve.run(c, (evolve.rf(90.0), evolve.gradient(1, "c", 4), evolve.rf_pulse(p.b1_ut, p.dt_ms, p.g_mt_m, 90.0, None, "x", pos, w, None, close=False),
                          evolve.gradient(1, "c", 4), evolve.average("x"), evolve.acquire()))[0].total()
    b = rotation.rotate(p.b1_ut * 1j, p.dt_ms, p.g_mt_m, pos)[1]
    assert fast == pytest.approx(slow) and fast == pytest.approx(1j * np.sum(w * (-b * b) * 1j))


def test_slice_selective_excitation_gives_the_profile_of_the_design_layer():
    p = pulses.sinc(90.0, 2.0, 4.0, thickness_mm=5.0, n=128)
    pos = np.linspace(-6.0, 6.0, 49)
    c = operators.compile(ONE, 3.0, ONE.shifts_ppm[0])
    for model in evolve.MODELS:
        got = evolve.run(c, (evolve.rf_design(p, axis="z", pos_mm=pos, model=model), evolve.acquire()))[0].total()
        a, b = p.rotate(pos)
        assert got == pytest.approx(1j * np.sum(np.gradient(pos) * 2 * np.conj(a) * b), abs=1e-6)


def test_a_pulse_off_the_reference_is_referred_to_its_centre():
    """An ideal-like 180 played 300 Hz away from a spin on the reference does nothing but a small Bloch-Siegert phase;
    a spin on the carrier is refocused without a phase, wherever the carrier is."""
    g = np.full(200, 1.0)
    b1 = g / g.sum() * math.pi / (rotation.RAD_PER_UT_MS * 0.01)                    # 2 ms rectangular 180
    for carrier in (4.9, 3.0, 1.0):
        s = operators.SpinSystem("s", (carrier,), ())
        seq = (evolve.rf(90.0), evolve.delay(6.0), evolve.rf_pulse(b1, 0.01, None, 90.0, carrier, pathway="refocus", model="full"), evolve.delay(6.0), evolve.acquire())
        assert lines(s, seq).total() == pytest.approx(1.0, abs=1e-9)
        assert lines(s, seq[:2] + (evolve.rf_pulse(b1, 0.01, None, 90.0, carrier, pathway="refocus"),) + seq[3:]).total() == pytest.approx(1.0, abs=1e-9)


def test_lines_fid_equals_the_sum_of_exponentials_and_merge_keeps_the_signal():
    rng = np.random.default_rng(3)
    n_lines = 700
    ln = evolve.Lines(rng.uniform(-900, 900, n_lines), rng.standard_normal(n_lines) + 1j * rng.standard_normal(n_lines),
                      rng.integers(0, 2, n_lines), ("a", "b"))
    t = 1e-3 * (2.0 + 0.4 * np.arange(300))
    t2 = {"a": 50.0, "b": 200.0}
    rate = np.array([1e3 / t2[g] for g in ln.groups])[ln.group]
    ref = (ln.amp[:, None] * np.exp((2j * np.pi * ln.freq_hz - rate)[:, None] * t[None, :])).sum(axis=0)
    assert np.abs(ln.fid(300, 0.4, t2, t0_ms=2.0) - ref).max() < 1e-10
    env = np.exp(-np.pi * 3.0 * t - (np.pi * 4.0 * t) ** 2 / (4 * math.log(2)))
    assert np.abs(ln.fid(300, 0.4, t2, 3.0, 4.0, t0_ms=2.0) - ref * env).max() < 1e-10
    dup = evolve.concat([ln, ln.weighted(2.0), ln.weighted({"a": 0.0, "b": 0.0})])
    merged = evolve.merge(dup)
    assert merged.freq_hz.size == n_lines and np.abs(merged.fid(64, 0.4) - 3 * ln.fid(64, 0.4)).max() < 1e-9
    assert ln.shifted(5.0).freq_hz == pytest.approx(ln.freq_hz + 5.0)
    assert ln.window(-100, 100).freq_hz.size == np.sum(np.abs(ln.freq_hz) <= 100)
    assert ln.total("a") + ln.total("b") == pytest.approx(ln.total())
    empty = evolve.Lines(np.zeros(0), np.zeros(0, complex), np.zeros(0, int), ())
    assert np.all(empty.fid(8, 1.0) == 0) and evolve.merge(empty) is empty


def test_spectrum_axis_and_phasing():
    dwell, n = 0.5, 4096
    f0 = 205 * 1e3 / (n * dwell)                                        # on a point of the frequency grid
    ln = evolve.Lines(np.array([f0]), np.array([2.0 + 0j]), np.array([0]), ("g",))
    f, s = evolve.spectrum(ln.fid(n, dwell, 100.0), dwell)
    assert f[0] < f[-1] and f[np.argmax(s.real)] == pytest.approx(f0, abs=1e-9)
    ppm, s2 = evolve.spectrum(ln.fid(n, dwell, 100.0), dwell, b0_t=3.0, ref_ppm=4.7)
    assert ppm[np.argmax(s2.real)] == pytest.approx(4.7 + f0 / operators.larmor_hz_per_ppm(3.0), abs=1e-9)
    # a constant phase and a delay of the first sample are removed by phase0 and phase1
    k = int(np.argmax(s.real))
    turned = ln.weighted(np.exp(0.7j)).fid(n, dwell, 100.0, t0_ms=0.9)
    _, bad = evolve.spectrum(turned, dwell)
    _, good = evolve.spectrum(turned, dwell, phase0_deg=math.degrees(0.7), phase1_ms=0.9)
    assert abs(bad[k].imag) > 0.2 * abs(bad[k]) and abs(good[k].imag) < 0.01 * abs(good[k])
    # zero filling keeps the scale; apodisation adds its width
    f4, s4 = evolve.spectrum(ln.fid(n, dwell, 100.0), dwell, zero_fill=4, lb_hz=2.0)
    assert f4.size == 4 * n and s4.real.max() == pytest.approx(2.0 / (1e3 / 100.0 + np.pi * 2.0), rel=2e-3)
    with pytest.raises(ValueError):
        evolve.lorentzian_spectrum(ln, f)


def test_final_state_needs_one_sector():
    with pytest.raises(ValueError):
        evolve.final_state(operators.compile(operators.SpinSystem("m", (1.0,), (), counts=(3,))), ())
    c = operators.compile(AX)
    assert np.allclose(evolve.final_state(c, ()), c.sectors[0].fz())


def test_weak_and_strong_coupling_agree_far_from_degeneracy_and_differ_near_it():
    """357 Hz apart with J = 7 Hz the two differ in first order of J / shift difference (0.02); 13 Hz apart they
    have nothing in common."""
    seq = (evolve.rf(90.0), evolve.delay(40.0), evolve.rf(180.0, 90.0, pathway="refocus"), evolve.delay(40.0), evolve.acquire())
    far = [lines(AX, seq, coupling=c).total("a") for c in operators.COUPLINGS]
    near_system = operators.SpinSystem("ab", (2.0, 2.1), ((0, 1, 7.0),), groups=("a", "x"))
    near = [lines(near_system, seq, coupling=c).total("a") for c in operators.COUPLINGS]
    assert abs(far[0] - far[1]) < 0.05 and abs(near[0] - near[1]) > 0.3


def test_b1_scale_scales_the_flip_angle_in_both_models():
    n = 100
    b1 = np.full(n, math.pi / 2 / (rotation.RAD_PER_UT_MS * 0.01 * n))                # 1 ms rectangular 90
    c = operators.compile(operators.SpinSystem("s", (4.7,), ()), 3.0, 4.7)
    for model in evolve.MODELS:
        for scale in (0.5, 1.0, 4 / 3):
            got = evolve.run(c, (evolve.rf_pulse(b1, 0.01, model=model, b1_scale=scale), evolve.acquire()))[0].total()
            assert got == pytest.approx(math.sin(scale * math.pi / 2)), (model, scale)
