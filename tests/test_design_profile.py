"""Profile figures, the mask check and the bounds: behaviour at the edges of their definitions."""
import math

import numpy as np
import pytest

from mrgym.design import profile, pulses
from mrgym.design.units import RAD_PER_UT_MS


def test_figures_of_a_generated_pulse_have_the_documented_fields():
    p = pulses.slr(90.0, 4.0, 8.0, 5.0)
    f = profile.figures(p)
    assert {"fwhm_mm", "passband_ripple", "passband_mean", "stopband_ripple", "transition_width_mm", "out_of_slice_fraction", "flip", "phase"} <= set(f)
    assert abs(f["fwhm_mm"] - 5.0) < 0.1 and f["transition_fraction"] == p.call["design"]["transition_fraction"]
    assert f["stopband_ripple"] <= 0.0101 and abs(f["passband_mean"] - 1.0) < 0.01
    assert 75.0 < f["flip"]["mean_deg"] < 100.0 and 80.0 < f["flip"]["centre_deg"] < 100.0   # the mean includes the two half transitions
    assert f["flip"]["percentiles_deg"][5] <= f["flip"]["percentiles_deg"][50] <= f["flip"]["percentiles_deg"][95]
    assert "phase" not in profile.figures(pulses.slr(180.0, 4.0, 8.0, 5.0, kind="inversion"))
    with pytest.raises(ValueError):
        profile.figures(pulses.sinc(90.0, 1.0, 2.0))


def test_response_kinds():
    a, b = np.array([math.cos(0.3)]), np.array([-1j * math.sin(0.3)])
    assert abs(profile.response("excitation", a, b)[0] + 1j * math.sin(0.6)) < 1e-15
    assert abs(profile.response("refocusing", a, b)[0] - math.sin(0.3) ** 2) < 1e-15
    assert abs(profile.response("inversion", a, b)[0] - math.sin(0.3) ** 2) < 1e-15
    with pytest.raises(ValueError):
        profile.response("spoiler", a, b)


def test_rephasing_fraction_shows_in_the_phase_slope():
    p = pulses.sinc(30.0, 2.0, 4.0, 5.0)
    z = profile.sample_positions(5.0)
    good = profile.phase_figures(z, profile.response("excitation", *p.rotate(z)), 5.0, 0.3)
    none = profile.phase_figures(z, profile.response("excitation", *p.rotate(z, rephase=False)), 5.0, 0.3)
    assert abs(good["slope_rad_per_mm"]) < 0.05 and good["refocused_fraction"] > 0.999
    assert abs(none["slope_rad_per_mm"]) > 1.0 and none["refocused_fraction"] < 0.5
    with pytest.raises(ValueError):
        profile.phase_figures(z, np.abs(z), 5.0)


def test_figures_need_usable_positions():
    z = np.linspace(-1, 1, 11)
    with pytest.raises(ValueError):
        profile.fwhm(z[::-1], np.ones(11))
    with pytest.raises(ValueError):
        profile.passband_ripple(z, np.ones(11), 5.0, 0.2)             # no position in the stop band
    assert math.isnan(profile.fwhm(z, np.ones(11)))                      # never falls to half
    assert math.isnan(profile.transition_width(z, np.ones(11)))


def test_mask_names_the_first_violated_band_in_list_order():
    z = np.linspace(-10, 10, 401)
    r = (np.abs(z) < 2.5) * 1.0
    bands = profile.standard_mask(5.0, 0.2, 1.0, 0.05, 0.02)
    assert profile.check_mask(z, r, bands)["passed"]
    both = r * 0.8 + 0.05                                # too low in the slice and too high outside
    assert profile.check_mask(z, both, bands)["first_violation"] == "passband"
    assert profile.check_mask(z, both, bands[::-1])["first_violation"] == "stopband_high"
    assert set(profile.check_mask(z, both, bands)["margins"]) == {"passband", "stopband_low", "stopband_high"}


def test_duration_bound_moves_the_right_way():
    base = profile.duration_bound_ms(90.0, 2000.0, 15.0, 0.01)
    assert base["bound_ms"] == max(base["hard_pulse_ms"], base["energy_ms"])
    assert abs(base["hard_pulse_ms"] - (math.pi / 2) / (RAD_PER_UT_MS * 15.0)) < 1e-12
    assert profile.duration_bound_ms(120.0, 2000.0, 15.0, 0.01)["bound_ms"] > base["bound_ms"]
    assert profile.duration_bound_ms(90.0, 2000.0, 10.0, 0.01)["bound_ms"] > base["bound_ms"]
    assert profile.duration_bound_ms(90.0, 8000.0, 15.0, 0.01)["energy_ms"] > base["energy_ms"]
    assert math.isinf(profile.duration_bound_ms(180.0, 1000.0, 15.0, 0.01)["energy_ms"])       # an exact inversion of a band
    assert profile.duration_bound_ms(180.0, 0.0, 15.0, 0.01)["bound_ms"] == profile.duration_bound_ms(180.0, 0.0, 15.0, 0.01)["hard_pulse_ms"]
    for bad in (dict(flip_min_deg=0.0), dict(b1_peak_ut=0.0), dict(passband_hz=-1.0), dict(passband_hz=2e5)):
        args = dict(flip_min_deg=90.0, passband_hz=2000.0, b1_peak_ut=15.0, raster_ms=0.01)
        args.update(bad)
        with pytest.raises(ValueError):
            profile.duration_bound_ms(**args)


def test_energy_floor_needs_the_class_it_is_derived_for():
    base = pulses.sinc(90.0, 2.0, 4.0, 5.0)
    with pytest.raises(ValueError):
        profile.pulse_energy_floor_ut2_ms(pulses.verse(base, 6.0, 20.0))
    assert base.energy_ut2_ms >= profile.pulse_energy_floor_ut2_ms(base)


def test_jensen_integral_handles_a_zero_of_a_on_a_grid_point():
    """An exact 180 degree pulse has a = 0 on resonance, a grid point of the quadrature; the integral must not move with
    the number of points and must equal -sum ln cos(theta_j / 2) (a minimum-phase A)."""
    p = pulses.sinc(180.0, 2.0, 4.0, 5.0, kind="refocusing")
    exact = -np.sum(np.log(np.cos(np.abs(RAD_PER_UT_MS * p.b1_ut * p.dt_ms) / 2)))
    values = [profile.jensen_integral(p, points) for points in (1 << 14, 1 << 16, 1 << 18)]
    assert max(abs(v / exact - 1) for v in values) < 1e-4
    assert all(v <= exact * (1 + 1e-12) for v in values)                 # the midpoint rule errs on the low side


def test_the_energy_floor_is_empty_when_one_raster_sample_can_turn_by_180_degrees():
    """90x 180y 90x in three samples inverts every frequency: no floor from the profile can forbid it."""
    dt = 0.01
    unit = math.pi / (RAD_PER_UT_MS * dt)
    p = pulses.Pulse(np.array([0.5 * unit, 1j * unit, 0.5 * unit]), dt, 0.0)
    _, b = p.rotate(None, np.linspace(-20000, 20000, 9))
    assert np.abs(np.abs(b) - 1).max() < 1e-12
    bound = profile.duration_bound_ms(180.0, 40000.0, p.peak_b1_ut, dt)
    assert bound["energy_ms"] == 0.0 and bound["bound_ms"] <= p.duration_ms
    assert math.isinf(profile.duration_bound_ms(180.0, 40000.0, 0.5 * p.peak_b1_ut, dt)["bound_ms"])
