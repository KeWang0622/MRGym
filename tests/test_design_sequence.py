"""The block list and the phase graph: construction, columns, truncation, continuation."""
import math

import numpy as np
import pytest

from mrgym.design import pulses
from mrgym.design import sequence as S
from mrgym.design.units import RAD_PER_UT_MS


def test_blocks_are_checked():
    with pytest.raises(ValueError):
        S.gradient(0.5)
    with pytest.raises(ValueError):
        S.delay(-1.0)
    with pytest.raises(ValueError):
        S.gradient(1, ms=-1.0)
    with pytest.raises(ValueError):
        S.rf_rotation(1.0, 0.5)
    with pytest.raises(ValueError):
        S.run([("rf", 90, 0)], 1000.0, 100.0)
    with pytest.raises(ValueError):
        S.run([S.rf(90)], 0.0, 100.0)
    with pytest.raises(ValueError):
        S.bd_of_b(100.0, 1.0, 0)
    with pytest.raises(ValueError):
        S.run([S.rf(np.array([10.0, 20.0])), S.readout()], np.array([1.0, 2.0, 3.0]), 100.0)


def test_readouts_come_back_in_order_with_their_labels():
    r = S.run([S.rf(30), S.readout("a"), S.delay(10.0), S.readout(7), S.readout()], 1000.0, 50.0)
    assert r.index == ("a", 7, None) and r.signal.shape == (3,)
    assert abs(r.signal[0] - math.sin(math.radians(30))) < 1e-15 and abs(r.signal[1] - r.signal[0] * math.exp(-0.2)) < 1e-15
    assert S.run([S.rf(30)], 1000.0, 50.0).signal.shape == (0,)


def test_phase_conventions():
    """Pulse about x: F0 = -i sin(a), reported real and positive; pulse phase p turns it by p; receiver phase takes it out;
    a positive off-resonance gains positive phase."""
    raw = S.run([S.rf(40), S.readout()], 1e3, 1e2, demodulation=1.0).signal[0]
    assert abs(raw + 1j * math.sin(math.radians(40))) < 1e-15
    assert abs(S.run([S.rf(40, 25), S.readout()], 1e3, 1e2).signal[0] - math.sin(math.radians(40)) * np.exp(1j * math.radians(25))) < 1e-15
    assert abs(S.run([S.rf(40, 25), S.readout(None, 25)], 1e3, 1e2).signal[0] - math.sin(math.radians(40))) < 1e-15
    s = S.run([S.rf(40), S.delay(2.0), S.readout()], 1e9, 1e9, df_hz=50.0).signal[0]
    assert abs(np.angle(s) - 2 * math.pi * 50.0 * 2e-3) < 1e-12


def test_columns_and_weights():
    flips = np.array([10.0, 30.0, 90.0])
    per = S.run([S.rf(flips), S.readout()], 1000.0, 80.0).signal
    assert per.shape == (1, 3) and np.allclose(per[0], np.sin(np.radians(flips)))
    w = np.array([0.2, 0.3, 0.5])
    assert abs(S.run([S.rf(flips), S.readout()], 1000.0, 80.0, weights=w).signal[0] - np.sin(np.radians(flips)) @ w) < 1e-15
    t1 = np.array([300.0, 1000.0])
    sat = S.run([S.rf(90), S.spoil(), S.delay(500.0), S.rf(90), S.readout()], t1, 80.0).signal
    assert np.allclose(sat[0], 1 - np.exp(-500.0 / t1))
    assert np.allclose(S.run([S.rf(60), S.readout()], 1e3, 1e2, b1_scale=np.array([0.5, 1.0])).signal[0], np.sin(np.radians([30.0, 60.0])))


def test_b1_scale_applies_to_flip_blocks_only():
    a, b = math.cos(0.4), -1j * math.sin(0.4)
    fixed = S.run([S.rf_rotation(a, b), S.readout()], 1e3, 1e2, b1_scale=0.5).signal[0]
    scaled = S.run([S.rf(math.degrees(0.8)), S.readout()], 1e3, 1e2, b1_scale=0.5).signal[0]
    assert abs(fixed - math.sin(0.8)) < 1e-15 and abs(scaled - math.sin(0.4)) < 1e-15


def test_a_designed_pulse_as_a_block():
    hard = pulses.control_points([8.0], 0.5)
    flip = RAD_PER_UT_MS * 8.0 * 0.5
    assert abs(S.run([S.rf_pulse(hard), S.readout()], 1e3, 1e2).signal[0] - math.sin(flip)) < 1e-12
    assert abs(S.run([S.rf_pulse(hard, b1_scale=0.5, phase_deg=90), S.readout()], 1e3, 1e2).signal[0] - 1j * math.sin(flip / 2)) < 1e-12
    p = pulses.sinc(20.0, 2.0, 4.0, 5.0)
    z = np.linspace(-15, 15, 601)
    r = S.run([S.rf_pulse(p, z, weights=np.full(601, 0.05 / 5.0)), S.readout()], 1e3, 1e2)
    assert abs(r.signal[0].imag) < 1e-12 and 0.3 < r.signal[0].real < 0.36            # about sin(20 deg) per unit thickness


def test_a_run_can_be_continued_from_its_state():
    blocks = [S.rf(70, 10), S.gradient(2), S.delay(5.0), S.rf(130, -30), S.gradient(-3), S.delay(4.0), S.rf(40), S.gradient(1), S.readout()]
    whole = S.run(blocks, 900.0, 60.0, df_hz=12.0)
    first = S.run(blocks[:4], 900.0, 60.0, df_hz=12.0)
    rest = S.run(blocks[4:], 900.0, 60.0, df_hz=12.0, state=(first.fp, first.fm, first.z))
    assert np.abs(rest.signal - whole.signal).max() < 1e-15


def test_truncation_is_exact_when_no_reachable_order_is_dropped_and_an_approximation_otherwise():
    blocks = [S.rf(90, 90)]
    for i in range(30):
        blocks += [S.delay(2.5), S.gradient(1), S.rf(140, 0), S.gradient(1), S.delay(2.5), S.readout(i)]
    full = S.run(blocks, 1000.0, 100.0)
    assert full.fp.shape[0] == 61
    assert np.abs(S.run(blocks, 1000.0, 100.0, kmax=60).signal - full.signal).max() == 0.0
    assert np.abs(S.run(blocks, 1000.0, 100.0, kmax=31).signal - full.signal).max() < 1e-12     # orders above 30 cannot return in time
    assert np.abs(S.run(blocks, 1000.0, 100.0, kmax=4).signal - full.signal).max() > 1e-4


def test_opposite_dephasing_cancels_and_a_large_step_is_handled():
    plain = S.run([S.rf(50), S.readout()], 1e3, 1e2).signal[0]
    assert abs(S.run([S.rf(50), S.gradient(5), S.gradient(-5), S.readout()], 1e3, 1e2).signal[0] - plain) < 1e-15
    assert abs(S.run([S.rf(50), S.gradient(-7), S.gradient(3), S.gradient(4), S.readout()], 1e3, 1e2).signal[0] - plain) < 1e-15
    assert abs(S.run([S.rf(50), S.gradient(3), S.readout()], 1e3, 1e2).signal[0]) < 1e-15
    assert abs(S.run([S.rf(50), S.gradient(9), S.gradient(-9), S.readout()], 1e3, 1e2, kmax=3).signal[0]) < 1e-15   # the order was dropped


def test_spoil_keeps_the_longitudinal_orders():
    r = S.run([S.rf(60), S.gradient(1), S.rf(60), S.spoil()], 1e3, 1e2)
    assert np.all(r.fp == 0) and abs(r.z[1, 0]) > 0.1
