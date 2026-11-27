"""Pulse generators: the pulse object, argument checks, records and the rules the docstrings state."""
import math

import numpy as np
import pytest

from mrgym.design import profile, pulses
from mrgym.design.units import HZ_PER_MT_M_MM, RAD_PER_UT_MS


def test_a_pulse_is_read_only_and_replace_makes_a_new_one():
    p = pulses.sinc(90.0, 2.0, 4.0, 5.0)
    with pytest.raises(ValueError):
        p.b1_ut[0] = 1.0
    with pytest.raises(AttributeError):
        p.kind = "inversion"
    q = p.replace(kind="saturation")
    assert q.kind == "saturation" and p.kind == "excitation" and q.b1_ut is not p.b1_ut


def test_record_fields_and_units():
    p = pulses.sinc(30.0, 1.0, 2.0, 4.0)
    rec = p.record(tr_ms=10.0)
    assert set(rec) == {"kind", "samples", "duration_ms", "peak_b1_ut", "energy_ut2_ms", "tr_ms", "b1rms_ut", "flip_deg", "bandwidth_hz",
                        "thickness_mm", "gradient_mt_m", "rephase_mt_ms_m"}
    assert rec["samples"] == 100 and rec["duration_ms"] == 1.0 and rec["bandwidth_hz"] == 2000.0
    assert abs(rec["b1rms_ut"] - math.sqrt(rec["energy_ut2_ms"] / 10.0)) < 1e-6
    assert abs(rec["gradient_mt_m"] - 2000.0 / (HZ_PER_MT_M_MM * 4.0)) < 1e-6
    assert abs(rec["rephase_mt_ms_m"] + 0.5 * rec["gradient_mt_m"] * 1.0) < 1e-5
    assert abs(p.b1rms_ut(10.0, pulses_per_tr=4) - 2 * p.b1rms_ut(10.0)) < 1e-12
    with pytest.raises(ValueError):
        p.b1rms_ut(0.5)


def test_duration_must_sit_on_the_raster_unless_n_is_given():
    with pytest.raises(ValueError, match="raster"):
        pulses.sinc(90.0, 1.003, 4.0)
    assert pulses.sinc(90.0, 1.003, 4.0, n=128).n == 128
    with pytest.raises(ValueError):
        pulses.sinc(90.0, 100.0, 4.0)                    # 10,000 samples
    with pytest.raises(ValueError):
        pulses.sinc(90.0, 1.0, 4.0, window="kaiser")
    with pytest.raises(ValueError):
        pulses.sinc(90.0, 1.0, 4.0, thickness_mm=0.0)


def test_only_an_excitation_gets_a_rephasing_lobe():
    assert pulses.sinc(180.0, 3.0, 4.0, 5.0, kind="refocusing").rephase_mt_ms_m == 0.0
    assert pulses.slr(180.0, 4.0, 8.0, 5.0, kind="inversion").rephase_mt_ms_m == 0.0
    assert pulses.slr(90.0, 4.0, 8.0, 5.0).rephase_mt_ms_m < 0.0


@pytest.mark.parametrize("kind, flip", [("excitation", 90.0), ("refocusing", 180.0), ("inversion", 180.0), ("saturation", 90.0)])
def test_slr_designs_are_real_symmetric_and_record_their_design(kind, flip):
    p = pulses.slr(flip, 2.56, 6.0, 5.0, kind=kind, n=256)
    assert np.abs(p.b1_ut.imag).max() < 1e-9 * p.peak_b1_ut
    assert np.abs(p.b1_ut - p.b1_ut[::-1]).max() < 1e-6 * p.peak_b1_ut
    d = p.call["design"]
    assert d["inverse_error"] < 1e-9 and 0 < d["transition_fraction"] < 1
    assert d["filter_pass_ripple"] <= d["filter_pass_target"] * 1.001 and d["filter_stop_ripple"] <= d["filter_stop_target"] * 1.001
    assert abs(0.5 * (d["pass_edge_hz"] + d["stop_edge_hz"]) - p.bandwidth_hz / 2) < 1e-6


def test_slr_refuses_what_it_cannot_design():
    with pytest.raises(ValueError, match="cannot hold"):
        pulses.slr(90.0, 2.0, 1.0)                        # time-bandwidth 1 with 1 % ripples
    with pytest.raises(ValueError, match="odd"):
        pulses.slr(90.0, 2.0, 6.0, filter="ls")
    with pytest.raises(ValueError):
        pulses.slr(90.0, 2.0, 6.0, filter="kaiser")
    with pytest.raises(ValueError):
        pulses.slr(200.0, 2.0, 6.0)
    with pytest.raises(ValueError):
        pulses.slr(90.0, 2.0, 6.0, d1=0.0)
    with pytest.raises(ValueError, match="too few"):
        pulses.slr(90.0, 0.12, 12.0, n=12)
    with pytest.raises(ValueError, match="below 1"):
        pulses.min_phase_a(np.array([0.7, 0.7]))


def test_least_squares_filter_has_smooth_ends_and_equiripple_can_have_end_samples():
    pm = pulses.slr(90.0, 2.55, 4.0, kind="saturation", n=255)
    ls = pulses.slr(90.0, 2.55, 4.0, kind="saturation", filter="ls", n=255)
    assert np.argmax(np.abs(pm.b1_ut)) in (0, 1, 253, 254) and pm.peak_b1_ut > 2 * abs(pm.b1_ut[127])     # the peak sits at the ends
    assert np.argmax(np.abs(ls.b1_ut)) == 127 and abs(ls.b1_ut[0]) < 0.1 * ls.peak_b1_ut and ls.peak_b1_ut < 0.6 * pm.peak_b1_ut


def test_dinf_is_the_published_fit_at_one_point():
    # 1 % and 1 %: (5.309e-3 * 4 - 7.114e-2 * 2 - 0.4761) * (-2) + (-2.66e-3 * 4 + 0.5941 * 2 - 0.4278)
    assert abs(pulses.dinf(0.01, 0.01) - ((5.309e-3 * 4 - 7.114e-2 * 2 - 0.4761) * -2 + (-2.66e-3 * 4 + 0.5941 * 2 - 0.4278))) < 1e-12
    assert 1.9 < pulses.dinf(0.01, 0.01) < 2.0


def test_sech_parameters():
    p = pulses.sech(12.0, 10.0, 1.06, 5.0, thickness_mm=5.0)
    assert p.kind == "inversion" and abs(p.bandwidth_hz - 1e3 * 5.0 * 1.06 / math.pi) < 1e-9
    assert abs(p.call["truncation"] - 1 / math.cosh(5.3)) < 1e-12
    assert abs(np.abs(p.b1_ut).max() - 12.0) < 1e-3 and abs(p.g_mt_m[0] - p.bandwidth_hz / (HZ_PER_MT_M_MM * 5.0)) < 1e-12
    inst = np.diff(np.unwrap(np.angle(p.b1_ut))) / p.dt_ms[0]                 # rad/ms: the frequency sweep -mu beta tanh(beta t)
    t = (np.arange(1, p.n)) * p.dt_ms[0] - 5.0
    assert np.abs(inst + 5.0 * 1.06 * np.tanh(1.06 * t)).max() < 1e-3
    assert abs(float(pulses.sech_mz(3.0, 1.0, 0.0)) - math.cos(math.pi * RAD_PER_UT_MS * 3.0)) < 1e-12      # mu = 0: area pi x
    with pytest.raises(ValueError):
        pulses.sech(0.0, 10.0, 1.0, 5.0)


def test_control_points_limits_and_rule():
    with pytest.raises(ValueError):
        pulses.control_points(np.ones(65), 1.0)
    with pytest.raises(ValueError):
        pulses.control_points([], 1.0)
    with pytest.raises(ValueError):
        pulses.control_points([1.0, 2.0], 1.0, bandwidth_hz=1000.0)
    with pytest.raises(ValueError):
        pulses.control_points([1.0, 2.0], 1.0, interp="quadratic")
    p = pulses.control_points([0.0, 10.0, 0.0], 1.0, 2000.0, 5.0, n=100, rephase_fraction=0.52)
    assert abs(p.b1_ut[49] - 9.9) < 1e-12 and abs(p.b1_ut[0] - 0.1) < 1e-12        # centres of the samples
    assert abs(p.rephase_mt_ms_m + 0.52 * p.g_mt_m[0]) < 1e-12
    assert p.call["points_ut"] == [[0.0, 0.0], [10.0, 0.0], [0.0, 0.0]]
    assert pulses.control_points(np.ones(64) * (1 + 1j), 0.64).n == 64


def test_verse_needs_a_gradient_and_keeps_the_sub_pulse_angles():
    with pytest.raises(ValueError):
        pulses.verse(pulses.sinc(90.0, 2.0, 4.0), 6.0, 20.0)
    with pytest.raises(ValueError):
        pulses.verse(pulses.sinc(90.0, 2.0, 4.0, 5.0), 0.0, 20.0)
    base = pulses.sinc(90.0, 2.0, 4.0, 5.0)
    fast = pulses.verse(base, 6.0, 20.0, 120.0)
    assert np.allclose(fast.b1_ut * fast.dt_ms, base.b1_ut * base.dt_ms) and np.allclose(fast.g_mt_m * fast.dt_ms, base.g_mt_m * base.dt_ms)
    assert fast.duration_ms < base.duration_ms and fast.energy_ut2_ms < base.energy_ut2_ms
    assert fast.rephase_mt_ms_m == base.rephase_mt_ms_m and fast.call["of"] == base.call
    assert fast.duration_ms >= profile.variable_rate_floor_ms(base, 6.0)
    with pytest.raises(ValueError):
        pulses.verse(fast, 6.0, 20.0, 120.0)             # the slew pass needs a constant raster


def test_inverse_slr_of_a_single_sample():
    theta = pulses.inverse_slr(np.array([-1j * math.sin(0.4)]), np.array([math.cos(0.4)]))
    assert abs(theta[0] - 0.8) < 1e-14


def test_verse_with_a_slew_limit_is_feasible_and_says_when_it_is_not_the_fastest():
    """The two-pass rule is the fastest schedule while every rate stays above sqrt(c / 2); below it the rule holds the
    cap at its minimum and is conservative (pinned here on a two-sample case with c = 1)."""
    base = pulses.sinc(90.0, 2.0, 4.0, 5.0)
    for peak, slew in ((6.0, 120.0), (3.0, 40.0), (11.0, 15.0)):
        fast = pulses.verse(base, peak, 30.0, slew)
        centres = np.cumsum(fast.dt_ms) - 0.5 * fast.dt_ms
        assert (np.abs(np.diff(fast.g_mt_m)) / np.diff(centres)).max() <= slew * (1 + 1e-9)
        assert fast.peak_b1_ut <= peak * (1 + 1e-12) and fast.g_mt_m.max() <= 30.0 * (1 + 1e-12)
        assert fast.duration_ms >= pulses.verse(base, peak, 30.0).duration_ms
    assert abs(pulses._slew_cap(0.1, 1.0) - pulses._slew_cap(np.sqrt(0.5), 1.0)) < 1e-15       # held at the minimum of the cap
    r = pulses._slew_cap(2.0, 1.0)
    assert abs((r - 2.0) - 0.5 * (1 / r + 1 / 2.0)) < 1e-12                                     # the stated rule, with equality


def test_ripples_larger_than_the_profile_are_refused():
    with pytest.raises(ValueError, match="too large"):
        pulses.slr(1.0, 2.0, 6.0)                         # |Mxy| of a 1 degree pulse is 0.017; ripples of 0.01 of M0 mean nothing
    assert pulses.slr(1.0, 2.0, 6.0, d1=1e-4, d2=1e-4).flip_deg() < 1.1
    assert pulses.slr(5.0, 2.0, 6.0).peak_b1_ut > 0
