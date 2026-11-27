"""Closed-form fields of the layered cylinder."""
import math

import numpy as np
import pytest

from mrgym.em import analytic
from mrgym.em.units import MU0


def test_refused_geometries():
    with pytest.raises(ValueError):
        analytic.LayeredCylinder(64e6, [0.09, 0.08], [50, 5], [0.5, 0.05])
    with pytest.raises(ValueError):
        analytic.LayeredCylinder(64e6, [0.09], [50, 5], [0.5])
    with pytest.raises(ValueError):
        analytic.LayeredCylinder(64e6, [0.15], [50], [0.5], source_radius_m=0.14)
    with pytest.raises(ValueError):
        analytic.LayeredCylinder(64e6, [0.09], [50], [0.5], source_radius_m=0.14, shield_radius_m=0.13)
    with pytest.raises(ValueError):
        analytic.sheet_modes("circular")
    with pytest.raises(ValueError):
        analytic.rung_currents(8, "circular")
    with pytest.raises(ValueError):                                          # Bessel functions out of range are refused, not returned
        analytic.LayeredCylinder(1e5, source_radius_m=0.14).radial(120)


@pytest.mark.parametrize("shield", [None, 0.2])
def test_fields_are_continuous_across_every_interface(shield):
    cyl = analytic.LayeredCylinder(298e6, [0.05, 0.08], [52.0, 5.6], [0.55, 0.04], 0.14, shield)
    modes = analytic.birdcage_modes(0.14, 16, "linear", n_max=30)
    phi = np.linspace(0, 2 * math.pi, 17)[:-1]
    for radius in (0.05, 0.08):
        inner = cyl.fields(modes, (radius - 1e-9) * np.sin(phi), (radius - 1e-9) * np.cos(phi))
        outer = cyl.fields(modes, (radius + 1e-9) * np.sin(phi), (radius + 1e-9) * np.cos(phi))
        for k in ("ez", "bx", "by"):
            assert np.abs(inner[k] - outer[k]).max() < 1e-6 * np.abs(inner[k]).max(), (radius, k)


def test_reciprocity_of_the_series_between_two_radii():
    """Ez at a point of radius 0.16 from a line current at radius 0.12 equals Ez at the latter from the former."""
    body = ([0.09], [60.0], [0.7])
    p1, p2 = (0.12, 0.4), (0.16, 2.1)                                        # (radius, angle)
    one = analytic.LayeredCylinder(128e6, *body, source_radius_m=p1[0])
    two = analytic.LayeredCylinder(128e6, *body, source_radius_m=p2[0])
    e12 = one.fields(analytic.line_modes(p1[0], p1[1], 1.0, 60), p2[0] * math.sin(p2[1]), p2[0] * math.cos(p2[1]))["ez"]
    e21 = two.fields(analytic.line_modes(p2[0], p2[1], 1.0, 60), p1[0] * math.sin(p1[1]), p1[0] * math.cos(p1[1]))["ez"]
    assert abs(e12 - e21) < 1e-7 * abs(e12)


def test_sense_of_rotation_of_the_three_drives():
    cyl = analytic.LayeredCylinder(1e5, source_radius_m=0.14)
    at = (np.zeros(1), np.zeros(1))
    quad = cyl.fields(analytic.sheet_modes("quadrature"), *at)
    rev = cyl.fields(analytic.sheet_modes("reverse"), *at)
    lin = cyl.fields(analytic.sheet_modes("linear"), *at)
    assert abs(quad["b1p"][0]) == pytest.approx(MU0 / 2, rel=1e-5) and abs(quad["b1m"][0]) < 1e-6 * MU0
    assert abs(rev["b1m"][0]) == pytest.approx(MU0 / 2, rel=1e-5) and abs(rev["b1p"][0]) < 1e-6 * MU0
    assert abs(lin["b1p"][0]) == pytest.approx(MU0 / 4, rel=1e-5) and abs(lin["b1m"][0]) == pytest.approx(MU0 / 4, rel=1e-5)
    # K0 cos(phi): current up at +x, down at -x, so the field on the axis points along -y
    assert lin["by"][0].real == pytest.approx(-MU0 / 2, rel=1e-5) and abs(lin["bx"][0]) < 1e-6 * MU0
    # a sheet phase turns B1+ by the same angle
    turned = cyl.fields(analytic.sheet_modes("quadrature", phi0=0.3), *at)
    assert np.angle(turned["b1p"][0] / quad["b1p"][0]) == pytest.approx(0.3, abs=1e-9)


def test_no_field_behind_the_shield_and_none_on_it():
    cyl = analytic.LayeredCylinder(128e6, [0.09], [60.0], [0.7], 0.14, 0.19)
    modes = analytic.birdcage_modes(0.14, 16, n_max=30)
    assert abs(cyl.fields(modes, np.array([0.0]), np.array([0.19 - 1e-12]))["ez"][0]) < 1e-6 * abs(cyl.fields(modes, np.array([0.0]), np.array([0.15]))["ez"][0])
    assert cyl.fields(modes, np.array([0.0]), np.array([0.25]))["ez"][0] == 0
    assert cyl.radiated_power(modes) == 0.0


def test_disc_fraction_adds_up_to_the_area_of_the_disc():
    h = 0.003
    y = (np.arange(81) - 40) * h
    yy, xx = np.meshgrid(y + 0.0007, y - 0.0011, indexing="ij")             # a grid that is not centred on the disc
    frac = analytic.disc_fraction(yy, xx, h, 0.085)
    assert 0 <= frac.min() and frac.max() <= 1
    assert frac.sum() * h * h == pytest.approx(math.pi * 0.085 ** 2, rel=2e-5)
    cyl = analytic.LayeredCylinder(64e6, [0.05, 0.085], [50.0, 5.0], [0.5, 0.05])
    eps, sig = cyl.cell_maps(yy, xx, h)
    area = math.pi * np.array([0.05 ** 2, 0.085 ** 2 - 0.05 ** 2])
    assert (sig.sum() * h * h) == pytest.approx(area @ np.array([0.5, 0.05]), rel=1e-4)
    assert ((eps - 1).sum() * h * h) == pytest.approx(area @ np.array([49.0, 4.0]), rel=1e-4)


def test_mode_helpers():
    m = analytic.add_modes({1: 1.0, 2: 2.0}, {2: 1.0j, -1: 3.0})
    assert m == {1: 1.0, 2: 2.0 + 1.0j, -1: 3.0}
    cur = analytic.rung_currents(8, "quadrature", 2.0)
    assert np.allclose(np.abs(cur), 2.0) and np.angle(cur[1]) == pytest.approx(-math.pi / 4)
    assert np.allclose(analytic.rung_currents(8, "linear"), np.cos(2 * math.pi * np.arange(8) / 8))
    # a birdcage of N rungs has only the orders -1 + m N
    modes = analytic.birdcage_modes(0.14, 8, "quadrature", n_max=20)
    big = {n for n, v in modes.items() if abs(v) > 1e-12}
    assert big == {-17, -9, -1, 7, 15}
