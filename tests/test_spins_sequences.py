"""Sequence builders, timing, relaxation factors."""
import math

import numpy as np
import pytest

from mrgym.design import pulses
from mrgym.spins import evolve, metabolites, operators, sequences

ONE = operators.SpinSystem("w", (4.2,), ())
AX = operators.SpinSystem("ax", (1.3, 4.1), ((0, 1, 7.0),), groups=("a", "x"))


def duration(seq):
    return sum(b.ms for b in seq.blocks if b.kind == "delay") + sum(float(b.dt_ms.sum()) for b in seq.blocks if b.kind == "rf_pulse")


def test_every_sequence_lasts_its_echo_time_with_and_without_shaped_pulses():
    p = pulses.slr(180.0, 4.0, 5.0, thickness_mm=15.0, kind="refocusing", n=100)
    afp = pulses.sech(25.0, 6.0, 1.6, 4.0, thickness_mm=15.0, n=120)
    ed = sequences.gaussian(180.0, 14.0)
    cases = [(sequences.spin_echo(40.0), 40.0), (sequences.spin_echo(40.0, refocus=p), 40.0), (sequences.press(35.0), 35.0),
             (sequences.press(te1_ms=11.0, te2_ms=24.0, refocus=p), 35.0), (sequences.steam(20.0, 10.0), 30.0), (sequences.slaser(60.0), 60.0),
             (sequences.slaser(60.0, afp=afp), 60.0), (sequences.mega_press(68.0, 1.9), 68.0), (sequences.mega_press(68.0, 1.9, edit=ed, refocus=p), 68.0),
             (sequences.mega_press(68.0, None, refocus=p), 68.0)]
    for seq, total in cases:
        assert duration(seq) == pytest.approx(total), seq.name
        assert seq.transverse_ms + seq.longitudinal_ms == pytest.approx(total)


def test_pulses_that_do_not_fit_are_refused():
    p = pulses.slr(180.0, 8.0, 5.0, thickness_mm=15.0, kind="refocusing", n=100)
    with pytest.raises(ValueError):
        sequences.press(12.0, refocus=p)
    with pytest.raises(ValueError):
        sequences.press()
    with pytest.raises(ValueError):
        sequences.mega_press(68.0, 1.9, te1_ms=40.0)
    with pytest.raises(ValueError):
        sequences.mega_press(30.0, 1.9, edit=sequences.gaussian(180.0, 20.0))
    with pytest.raises(ValueError):
        sequences.slice_positions(sequences.gaussian(180.0, 10.0))


def test_an_uncoupled_spin_gives_one_in_every_echo_sequence_and_half_in_steam():
    for seq in (sequences.pulse_acquire(), sequences.spin_echo(30.0), sequences.press(30.0), sequences.press(te1_ms=8.0, te2_ms=50.0),
                sequences.slaser(40.0), sequences.mega_press(68.0, 1.9), sequences.mega_press(68.0, None)):
        assert sequences.simulate(ONE, seq).total() == pytest.approx(1.0), seq.name
    assert sequences.simulate(ONE, sequences.steam(20.0, 10.0)).total() == pytest.approx(0.5)
    assert sequences.simulate(ONE, sequences.steam_explicit(20.0, 10.0)).total() == pytest.approx(0.5)
    assert sequences.simulate(ONE, sequences.pulse_acquire(30.0)).total() == pytest.approx(0.5)


def test_mega_press_without_editing_pulses_is_press():
    a = sequences.simulate(AX, sequences.mega_press(68.0, None, te1_ms=20.0)).fid(16, 1.0)
    b = sequences.simulate(AX, sequences.press(te1_ms=20.0, te2_ms=48.0)).fid(16, 1.0)
    assert np.abs(a - b).max() < 1e-12
    # an editing pulse far from every site changes nothing either
    c = sequences.simulate(AX, sequences.mega_press(68.0, 7.5, te1_ms=20.0)).fid(16, 1.0)
    assert np.abs(a - c).max() < 1e-12


def test_gaussian_pulse_inverts_on_resonance_and_reports_its_width():
    g = sequences.gaussian(180.0, 14.0)
    assert g.flip_deg() == pytest.approx(180.0, abs=1e-6) and g.kind == "inversion"
    assert 60.0 < g.bandwidth_hz < 120.0
    half = sequences.gaussian(180.0, 28.0)
    assert half.bandwidth_hz == pytest.approx(g.bandwidth_hz / 2, rel=1e-3)


def test_a_narrow_editing_pulse_edits_and_a_wide_refocusing_pulse_keeps_the_singlet():
    gaba = metabolites.get("gaba")
    hz = operators.larmor_hz_per_ppm(3.0)
    band = ((3.0128 - 0.3 - 4.7) * hz, (3.0128 + 0.3 - 4.7) * hz)
    ed = sequences.gaussian(180.0, 14.0)
    on = sequences.simulate(gaba, sequences.mega_press(68.0, 1.9, edit=ed)).window(*band).total().real
    off = sequences.simulate(gaba, sequences.mega_press(68.0, 7.46, edit=ed)).window(*band).total().real
    assert on > 1.5 and abs(off) < 0.4
    cr = metabolites.get("creatine")
    # the creatine singlet, 144 Hz from the editing pulse, loses about 1 % to the skirt of the Gaussian
    assert sequences.simulate(cr, sequences.mega_press(68.0, 1.9, edit=ed)).total("cr_ch3").real == pytest.approx(3.0, abs=0.05)
    assert sequences.simulate(cr, sequences.mega_press(68.0, 7.46, edit=ed)).total("cr_ch3").real == pytest.approx(3.0, abs=1e-3)


def test_slice_selective_press_converges_with_the_number_of_positions():
    """More positions over the same extent change nothing. A larger extent is a larger sample: the stop band of an
    equiripple pulse refocuses 1 % of what lies outside the slice, so the signal changes by at most that."""
    lac = metabolites.get("lactate")
    p = pulses.slr(180.0, 5.0, 6.0, thickness_mm=20.0, kind="refocusing", n=250)
    got = [sequences.simulate(lac, sequences.press(144.0, refocus=p, carrier_ppm=1.3142, n_pos=n, extent=e)).window(-460, -410).total()
           for n, e in ((32, 2.5), (128, 2.5), (128, 4.0))]
    assert abs(got[0] - got[1]) < 2e-3 and abs(got[1] - got[2]) < 3 * 0.01 * 1.5
    assert -1.4 < got[1].real < -1.0                                    # of -3 with ideal pulses


def test_recovery_and_relaxation_factors():
    assert sequences.recovery(1e9, 1000.0) == pytest.approx(1.0)
    assert sequences.recovery(1500.0, 1000.0, (), 0.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        sequences.recovery(100.0, 1000.0, (200.0,))
    seq = sequences.steam(20.0, 30.0)
    f = sequences.relaxation_factors(seq, {"a": (1200.0, 90.0)}, None, ("a", "x"))
    assert f["a"] == pytest.approx(math.exp(-20.0 / 90.0 - 30.0 / 1200.0)) and f["x"] == 1.0
    f = sequences.relaxation_factors(sequences.spin_echo(30.0), {"a": (1200.0, 90.0)}, 2000.0)
    assert f["a"] == pytest.approx(math.exp(-30.0 / 90.0) * (1 - 2 * math.exp(-(2000.0 - 15.0) / 1200.0) + math.exp(-2000.0 / 1200.0)))
    lines = sequences.simulate(AX, sequences.spin_echo(30.0), relaxation={"a": (1200.0, 90.0), "x": (900.0, 60.0)}, tr_ms=2000.0)
    bare = sequences.simulate(AX, sequences.spin_echo(30.0))
    assert lines.total("x") / bare.total("x") == pytest.approx(math.exp(-0.5) * sequences.recovery(2000.0, 900.0, (15.0,)))


def test_simulate_takes_systems_metabolites_and_lists():
    naa, cr = metabolites.get("naa"), metabolites.get("creatine")
    both = sequences.simulate([naa, cr], sequences.pulse_acquire())
    assert both.total() == pytest.approx(naa.protons + cr.protons)
    assert set(both.groups) == {"naa_acetyl", "naa_aspartyl", "cr_ch3", "cr_ch2"}
    with pytest.raises(ValueError):
        sequences.simulate(naa, sequences.Sequence("x", (evolve.rf(90.0),)))
    assert sequences.simulate_2d(cr, sequences.jres(20.0, 3, 10.0), 8, 1.0).shape == (3, 8)
    assert len(sequences.cosy(4, 1.0, gradient_selected=False)[0].blocks) == 4


def test_field_strength_changes_shift_dispersion_only():
    """Lactate doublet: the splitting is J at 1.5 and at 3 T; the distance to the quartet doubles."""
    for b0 in (1.5, 3.0):
        ln = sequences.simulate(metabolites.get("lactate"), sequences.pulse_acquire(), b0_t=b0, coupling="weak")
        f = np.sort(ln.freq_hz)
        assert f[1] - f[0] == pytest.approx(6.933, abs=1e-9)
        assert (f[2:].mean() - f[:2].mean()) == pytest.approx((4.0974 - 1.3142) * operators.larmor_hz_per_ppm(b0), abs=1e-9)


def test_pulses_from_a_bandwidth():
    p = sequences.refocusing_pulse(1200.0, 20.0)
    assert p.bandwidth_hz == pytest.approx(1200.0) and p.duration_ms == pytest.approx(5.0) and p.kind == "refocusing"
    assert p.flip_deg() > 168.0                                            # a pass-band ripple of 1 % in |b|^2 allows 168.5 degrees
    afp = sequences.adiabatic_pulse(3000.0, 8.0, 20.0)
    assert afp.bandwidth_hz == pytest.approx(3000.0) and afp.call["truncation"] == pytest.approx(0.01)
    assert afp.peak_b1_ut == pytest.approx(1.1 * 1500.0 / 42.577478518, rel=1e-3)             # gamma B1 = 1.1 x bandwidth / 2 (no sample at the peak)
    a, b = afp.rotate(np.array([-6.0, 0.0, 6.0]))
    assert np.all(np.abs(a) ** 2 - np.abs(b) ** 2 < -0.98)                                      # inverted inside the slice
    weak = afp.replace(b1_ut=afp.b1_ut * 0.25)
    a, b = weak.rotate(np.array([0.0]))
    assert abs(a[0]) ** 2 - abs(b[0]) ** 2 > -0.5                                               # at a quarter of that B1 it is not


def test_the_signal_of_a_slice_selective_voxel_grows_with_the_extent_of_the_sample():
    """The numbers of the module docstring: a singlet on the carrier under PRESS with the default 1.2 kHz pulse."""
    p = sequences.refocusing_pulse(1200.0, 20.0, raster_ms=0.02)
    one = operators.SpinSystem("s", (3.0,), ())
    got = [sequences.simulate(one, sequences.press(30.0, refocus=p, carrier_ppm=3.0, n_pos=int(32 * e / 2.5), extent=e)).total().real for e in (2.5, 8.0)]
    assert got[0] == pytest.approx(1.03, abs=0.01) and got[1] == pytest.approx(1.08, abs=0.01)


def test_steam_recovery_runs_from_the_third_pulse():
    f = sequences.relaxation_factors(sequences.steam(30.0, 100.0), {"w": (1350.0, 110.0)}, 1000.0)["w"]
    assert f == pytest.approx(math.exp(-30.0 / 110.0 - 100.0 / 1350.0) * (1 - math.exp(-(1000.0 - 115.0) / 1350.0)))
    assert sequences.recovery(1000.0, 1350.0, (), 90.0, (115.0,)) == pytest.approx(1 - math.exp(-885.0 / 1350.0))
    assert sequences.press(te1_ms=20.0, te2_ms=40.0).inversions_ms == (10.0, 40.0) and sequences.slaser(64.0).inversions_ms == (8.0, 24.0, 40.0, 56.0)


def test_gaussian_bandwidth_is_where_the_inversion_is_half():
    from mrgym.design import rotation
    g = sequences.gaussian(180.0, 14.0)
    a, b = rotation.rotate(g.b1_ut, g.dt_ms, None, None, np.array([0.0, g.bandwidth_hz / 2]))
    mz = np.abs(a) ** 2 - np.abs(b) ** 2
    assert mz[1] == pytest.approx(0.5 * (mz[0] + 1.0), abs=1e-4)
