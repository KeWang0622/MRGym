"""``mrgym.design.bound``: the duration bound of a flip-angle mask is a theorem for hard sub-pulses under one constant
gradient. These tests hold the code against the statements of docs/physics/design.md, section 4a, with computations
that share no code with it (decimal arithmetic, a linear programme, the forward Shinnar-Le Roux transform)."""
import dataclasses
import math
import os

import numpy as np
import pytest

from mrgym.design import bound, hardware, pulses, validate
from mrgym.design.units import HZ_PER_MT_M_MM, RAD_PER_UT_MS

SPEC = dict(thickness_mm=5.0, transition=0.3, flip_deg=30.0, tol_in_deg=1.2, max_out_deg=0.9)
WIDE = dict(thickness_mm=6.0, transition=0.4, flip_deg=40.0, tol_in_deg=2.0, max_out_deg=1.6)
SPLIT = hardware.Chain(model="split", guard=True)


def _minphase(spec, n, g=32.0):
    t = round(n * 0.01, 6)
    tbw = math.floor(HZ_PER_MT_M_MM * g * spec["thickness_mm"] * t * 1e-3 * 1e6) / 1e6
    return pulses.minphase(spec["flip_deg"], t, tbw, spec["thickness_mm"], spec["transition"], spec["tol_in_deg"], spec["max_out_deg"])


def test_the_mask_in_the_units_of_the_proof():
    m = bound.Mask.of(SPEC)
    assert (m.z_in, m.z_out) == (1.75, 3.25)
    assert m.lo == pytest.approx(math.sin(math.radians(28.8) / 2) ** 2) and m.hi == pytest.approx(math.sin(math.radians(31.2) / 2) ** 2)
    assert m.stop == pytest.approx(math.sin(math.radians(0.9) / 2) ** 2)
    p = m.on_circle(0.05, 0.06)                                          # weakened for a cell: the pass edge of its lower end, the stop edge of its upper end
    assert (p.a, p.b) == (0.05 * 1.75, 0.06 * 3.25)
    assert bound.precession(32.0) == pytest.approx(2 * math.pi * 42.577478518 * 32.0 * 0.01e-3)
    with pytest.raises(ValueError):
        bound.Mask.of(dict(SPEC, max_out_deg=29.0))                      # the outside may not be allowed more than the slice needs


@pytest.mark.parametrize("spec", [SPEC, WIDE])
def test_a_certificate_verifies_in_decimal_arithmetic_and_fails_when_it_is_spoiled(spec):
    b = bound.duration_bound(spec, 32.0)
    assert b["cells"] and b["limited_by"] == "gradient"
    for lo, hi, c in b["cells"][-2:]:
        exact = validate.certificate_margin_decimal(c)                   # 50 digits, its own cosine, no rounding budget
        assert exact > bound.MARGIN and bound.check(c) <= exact and exact - bound.check(c) < 1e-9
        assert c.n == b["samples"] - 1 and c.psi.size == c.n + 1 and abs(np.abs(c.weight).sum() - 1) < 1e-12
        assert np.all(np.sign(c.weight[1:]) == -np.sign(c.weight[:-1]))  # the weights of an alternation
        inside = c.psi <= lo * b["mask"].z_in
        assert np.array_equal(inside, c.in_slice) and np.all(c.psi[~inside] >= hi * b["mask"].z_out)
        spoiled = c.weight.copy()
        spoiled[3] *= 1.01                                               # no longer a null vector: the residual eats the margin
        assert bound.check(dataclasses.replace(c, weight=spoiled)) < 0
        loose = dataclasses.replace(c, upper=np.where(c.in_slice, c.upper, 5 * c.upper))    # a mask five times as lenient outside
        assert bound.check(loose) < bound.check(c)
        assert bound.check(dataclasses.replace(c, psi=c.psi[:-1])) == -math.inf


@pytest.mark.parametrize("spec", [SPEC, WIDE])
def test_the_cells_cover_every_gradient_up_to_the_limit(spec):
    b = bound.duration_bound(spec, 32.0)
    m, cells = b["mask"], b["cells"]
    assert cells[0][0] <= m.slope_precession(b["samples"] - 1) and cells[-1][1] == pytest.approx(bound.precession(32.0))
    assert all(nxt[0] == pytest.approx(prev[1]) for prev, nxt in zip(cells, cells[1:]))
    # below the first cell the slope of the flip angle forbids the transition
    n, u = b["samples"] - 1, 0.999 * m.slope_precession(b["samples"] - 1)
    assert (n - 1) * u * (m.z_out - m.z_in) < m.theta_lo - m.theta_out


@pytest.mark.parametrize("spec", [SPEC, WIDE])
def test_a_linear_programme_agrees_with_the_bound(spec):
    b = bound.duration_bound(spec, 32.0)
    for lo, hi, c in b["cells"]:
        assert not validate._lp_feasible(b["samples"] - 1, b["mask"].on_circle(lo, hi), c.psi)
    easier = bound.Mask.of(dict(spec, tol_in_deg=1.001 * spec["tol_in_deg"], max_out_deg=1.001 * spec["max_out_deg"]))
    assert validate._lp_feasible(b["samples"], easier.on_circle(bound.precession(32.0)))


@pytest.mark.parametrize("spec", [SPEC, WIDE])
def test_the_bound_is_reached_by_the_minimum_phase_pulse_and_not_beaten(spec):
    b = bound.duration_bound(spec, 32.0)
    assert b["samples"] == b["found_samples"]
    p = _minphase(spec, b["samples"])
    assert p.n == b["samples"] and all(bound.mask_met(p, spec)) and p.call["design"]["use"] <= 1.0
    assert p.call["design"]["inverse_error"] < 1e-9                      # the pulse has the polynomial it was built from
    a, bb = hardware.rotate(p, SPLIT, np.linspace(-b["mask"].z_in, b["mask"].z_in, 201))
    assert pulses.refocused(2 * np.conj(a) * bb, np.linspace(-b["mask"].z_in, b["mask"].z_in, 201)) > 0.95
    for n in range(b["samples"] - 6, b["samples"]):                      # shorter: the mask is missed at a position the verifier looks at
        short = _minphase(spec, n)
        assert short.call["design"]["use"] > 1.0 and not all(bound.mask_met(short, spec, SPLIT, b["cells"]))
    for g in (31.0, 24.0, 12.0):                                         # a weaker gradient never helps
        assert not all(bound.mask_met(_minphase(spec, b["samples"] - 1, g), spec, SPLIT, b["cells"]))


def test_the_slope_of_the_flip_angle_and_jensens_inequality():
    rng = np.random.default_rng(11)
    for n in (6, 20, 48):
        theta = rng.uniform(0, 0.7, n) * np.exp(2j * np.pi * rng.uniform(size=n))
        psi = np.linspace(-np.pi, np.pi, 20001)
        a_c, b_c = pulses.forward_slr(theta)
        big_b, big_a = np.polyval(b_c[::-1], np.exp(1j * psi)), np.polyval(a_c[::-1], np.exp(1j * psi))
        assert np.abs(np.abs(big_a) ** 2 + np.abs(big_b) ** 2 - 1).max() < 1e-12
        flip = 2 * np.arcsin(np.clip(np.abs(big_b), 0, 1))
        assert np.abs(np.diff(flip) / np.diff(psi)).max() <= (n - 1) * (1 + 1e-6)
        left = np.sum(-2 * np.log(np.cos(np.abs(theta) / 2)))
        right = -np.mean(np.log(1 - np.abs(big_b[:-1]) ** 2))             # -(1 / 2 pi) int ln(1 - P)
        assert left >= right - 1e-6
        assert a_c[0].real == pytest.approx(np.prod(np.cos(np.abs(theta) / 2)))
    t = np.linspace(0.01, 3.0, 300)                                      # g(t) / t^2 increases: what bounds the energy from the angles
    assert np.all(np.diff(-2 * np.log(np.cos(t / 2)) / t ** 2) > 0)


def test_the_gradient_cap_holds_for_built_pulses_and_scales_with_the_transmit_scale():
    m = bound.Mask.of(SPEC)
    for g in (32.0, 20.0, 9.0):
        p = _minphase(SPEC, bound.duration_bound(SPEC, g)["samples"], g)
        u = bound.precession(float(p.g_mt_m[0]))
        assert u <= bound.gradient_cap(p.n, m, 1.0, p.peak_b1_ut * 1.000001, p.energy_ut2_ms * 1.000001)
        assert u > bound.gradient_cap(p.n, m, 1.0, 40.0, 0.4 * p.energy_ut2_ms)        # far less energy: this gradient is excluded
        theta = RAD_PER_UT_MS * np.abs(p.b1_ut) * 0.01
        assert np.sum(-2 * np.log(np.cos(theta / 2))) <= p.n * bound._g(theta.max()) * (1 + 1e-12)
    assert bound.gradient_cap(60, m) == math.inf
    cap = [bound.gradient_cap(60, m, s, 40.0, 30.0) for s in (0.6, 1.2)]
    assert 4.0 <= cap[1] / cap[0] < 4.2                                 # energy on the nominal samples: the cap (plus a constant) grows with the square of the scale
    assert bound.gradient_cap(61, m, 0.6, 40.0, 30.0) > cap[0] and bound.gradient_cap(1, m, 1.0, 40.0, 30.0) > 0
    with pytest.raises(ValueError):
        bound.gradient_cap(60, m, energy_ut2_ms=30.0)
    assert bound._g_integral(1.0) <= 1.0 ** 3 / 12 + 1.0 ** 5 / 480 + 1e-3 and bound._g_integral(1.0) > 0.08   # below the series of the integral


def test_rf_limits_lengthen_the_bound_and_the_rule_is_monotone():
    free = bound.duration_bound(SPEC, 32.0)
    p = _minphase(SPEC, free["samples"])
    same = bound.duration_bound(SPEC, 32.0, 1.0, 40.0, 2 * p.energy_ut2_ms)
    assert same["samples"] == free["samples"] and same["limited_by"] == "gradient"
    previous = free["samples"]
    for scale in (1.0, 0.8, 0.6):                                        # the same nominal limits on scanners that deliver less
        b = bound.duration_bound(SPEC, 32.0, scale, 40.0, 0.6 * p.energy_ut2_ms)
        assert b["samples"] >= previous and b["limited_by"] == "rf" and b["gradient_cap_mt_m"] < 32.0
        assert b["cells"][-1][1] == pytest.approx(b["cap"]) and b["cap"] < bound.precession(32.0)
        previous = b["samples"]
    assert previous > 1.5 * free["samples"]
    assert bound.duration_bound(SPEC, 16.0)["samples"] >= 2 * free["samples"] - 2      # half the gradient: about twice the samples


def test_positions_of_the_pass_rule():
    b = bound.duration_bound(SPEC, 32.0)
    u = bound.precession(32.0)
    z_in, z_out = bound.positions(SPEC, 56, u)
    assert z_in.min() == -1.75 and z_in.max() == 1.75 and np.diff(np.sort(z_in)).max() <= 2 * bound.SLACK_RAD / (55 * u) * (1 + 1e-9)
    assert np.abs(z_out).min() == 3.25 and np.abs(z_out).max() == pytest.approx(math.pi / u) and np.diff(np.sort(z_out[z_out > 0])).max() <= 0.05 + 1e-12
    more_in, more_out = bound.positions(SPEC, 52, 0.999 * u, b["cells"])
    cell = b["cells"][-1][2]
    extra = np.concatenate([cell.psi / (0.999 * u), -cell.psi / (0.999 * u)])
    assert all(np.abs(np.concatenate([more_in, more_out]) - z).min() < 1e-12 for z in extra)
    assert np.abs(more_in).max() <= 1.75 and np.abs(more_out).min() >= 3.25 and np.abs(more_out).max() <= math.pi / (0.999 * u) + 1e-12


def test_the_generator_states_what_it_uses_and_refuses_what_cannot_exist():
    p = _minphase(SPEC, 80)
    assert p.call["design"]["use"] < 0.9 and p.kind == "excitation" and np.ptp(p.g_mt_m) == 0
    assert 1.0 < _minphase(SPEC, 50).call["design"]["use"] < _minphase(SPEC, 40).call["design"]["use"]     # too short: built for wider tolerances, and says so
    assert p.rephase_mt_ms_m == pytest.approx(-p.call["design"]["rephase_fraction"] * p.g_mt_m[0] * p.duration_ms)
    shifted = pulses.minphase(30.0, 0.8, 4.0, 5.0, 0.3, 1.2, 0.9, freq_hz=250.0)
    plain = pulses.minphase(30.0, 0.8, 4.0, 5.0, 0.3, 1.2, 0.9)
    z = np.linspace(-6, 6, 241)
    moved = hardware.flip_deg(shifted, hardware.Chain(df_hz=250.0, model="split"), z)       # the slice follows the spins that precess 250 Hz faster
    assert np.abs(moved - hardware.flip_deg(plain, SPLIT, z)).max() < 1e-9
    assert np.abs(hardware.flip_deg(shifted, SPLIT, z) - hardware.flip_deg(plain, SPLIT, z - 250.0 / (HZ_PER_MT_M_MM * plain.g_mt_m[0]))).max() < 1e-9
    with pytest.raises(ValueError):
        pulses.minphase(30.0, 0.05, 4.0, 5.0, 0.3, 1.2, 0.9)
    with pytest.raises(ValueError):
        bound.equiripple_b(40, SPEC, 0.99 * math.pi / 3.25 + 0.02)       # the stop band would start beyond half the repeat distance


def test_a_late_plateau_tilts_the_phase_and_changes_no_flip_angle():
    p = _minphase(SPEC, 60)
    z = np.linspace(-1.75, 1.75, 101)
    late = hardware.Chain(grad_delay_us=80.0, model="split", guard=True)
    a0, b0 = hardware.rotate(p, SPLIT, z)
    a1, b1 = hardware.rotate(p, late, z)
    assert np.abs(np.abs(b1) - np.abs(b0)).max() < 1e-13
    tilt = np.angle((2 * np.conj(a1) * b1) / (2 * np.conj(a0) * b0))
    assert np.abs(tilt - 2 * np.pi * HZ_PER_MT_M_MM * p.g_mt_m[0] * 80e-6 * z).max() < 1e-9
    fixed = pulses.minphase(30.0, 0.6, p.call["tbw"], 5.0, 0.3, 1.2, 0.9, rephase_fraction=p.call["rephase_fraction"] + 0.08 / 0.6)
    a2, b2 = hardware.rotate(fixed, late, z)
    assert np.abs(2 * np.conj(a2) * b2 - 2 * np.conj(a0) * b0).max() < 1e-9             # rephasing fraction + delay / duration takes it back
    # without the guard (the chain of -v0) a constant gradient is not changed by a delay at all
    a3, b3 = hardware.rotate(p, hardware.Chain(grad_delay_us=80.0, model="split"), z)
    assert np.abs(a3 - a0).max() < 1e-13 and hardware.Chain().model == "piecewise" and not hardware.Chain().guard


def test_the_document_states_the_tightness_that_the_command_prints():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "physics", "design.md")
    if not os.path.exists(path):
        pytest.skip("the tests are not running from a checkout of the repository")
    with open(path, encoding="utf-8") as f:
        block = f.read().split("<!-- tightness: begin -->")[1].split("<!-- tightness: end -->")[0]
    rows = bound.tightness()
    assert " ".join(block.split()) == bound.tightness_text(rows)
    assert len(rows) == 54 and min(r["ratio"] for r in rows) >= 1.0 and max(r["ratio"] for r in rows) <= 1.03
    assert all(r["samples"] <= r["found_samples"] <= r["samples"] + 1 for r in rows)
