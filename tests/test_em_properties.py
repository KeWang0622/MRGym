"""Tissue permittivity, conductivity and density of the RF field model."""
import numpy as np
import pytest

from mrgym.em import properties, units
from mrgym.scanner import tissue, world

SEED = "000102030405060708090a0b0c0d0e0f"


def test_the_model_reproduces_the_fetched_calculator_values():
    for row, by_f in properties.CHECK_VALUES.items():
        for mhz, (eps, sig, lam, depth) in by_f.items():
            e, s = properties.cole_cole(row, mhz * 1e6)
            assert e == pytest.approx(eps, rel=1e-4) and s == pytest.approx(sig, rel=1e-4), (row, mhz)
            assert float(units.wavelength_m(mhz * 1e6, e, s)) == pytest.approx(lam, rel=1e-4)
            assert float(units.skin_depth_m(mhz * 1e6, e, s)) == pytest.approx(depth, rel=1e-4)


def test_every_engine_class_has_a_row_and_every_row_is_a_known_one():
    assert set(tissue.NAMES) <= set(properties.CLASS_ROWS)
    for name, (row, density, tag, _note) in properties.CLASS_ROWS.items():
        assert row in properties.COLE_COLE, name
        assert tag in ("F", "f", "C") and 800 < density < 2000
        assert (tag == "F") <= (row in properties.CHECK_VALUES), name          # an [F] row has a tabulated check value


def test_dispersion_runs_the_right_way_between_64_and_298_mhz():
    for row in properties.COLE_COLE:
        e = [properties.cole_cole(row, f)[0] for f in (64e6, 128e6, 298e6)]
        s = [properties.cole_cole(row, f)[1] for f in (64e6, 128e6, 298e6)]
        assert e[0] > e[1] > e[2] > 1, row
        assert 0 < s[0] < s[1] < s[2], row


def test_wavenumber_conventions():
    k = units.wavenumber(128e6, 60.0, 0.7)
    assert k.real > 0 > k.imag                                               # exp(-i k r) decays for exp(+i w t)
    assert float(units.wavelength_m(64e6)) == pytest.approx(units.C0 / 64e6)
    assert np.isinf(units.skin_depth_m(64e6, 80.0, 0.0))
    assert units.EPS0 == pytest.approx(8.8541878128e-12, rel=1e-9)
    assert units.larmor_hz(3.0) == pytest.approx(127.73e6, rel=1e-4)


def test_classes_of_test_objects_and_unknown_classes():
    assert properties.tissue("vial3", 128e6)["row"] == "phantom_fill"
    assert properties.tissue("vial3", 128e6, fat=True)["row"] == "oil"
    assert properties.tissue("fill", 64e6, kind="acr")["sigma"] == properties.PHANTOM["acr_fill"][1]
    assert properties.tissue("fill", 64e6, kind="nist_t1")["row"] == "phantom_fill"
    with pytest.raises(KeyError):
        properties.tissue("unobtainium", 64e6)
    assert set(properties.table(64e6)) == set(properties.CLASS_ROWS)


@pytest.mark.parametrize("kind", ["brain", "abdomen", "qa", "acr", "nist_t1"])
def test_maps_of_every_world_kind(kind):
    w = world.make_world(SEED, dict(kind=kind, b0_t=3.0, n=160))
    eps, sig, rho = properties.world_maps(w, 128e6)
    assert eps.shape == sig.shape == rho.shape == (160, 160)
    assert eps.min() >= 1 - 1e-6 and sig.min() >= 0 and rho.min() >= 0
    body = np.asarray(w["masks"]["body"])
    assert sig[body].mean() > 0.05 and rho[body].mean() > 800
    corner = (slice(0, 4), slice(0, 4))
    assert np.allclose(eps[corner], 1, atol=1e-5) and np.allclose(sig[corner], 0, atol=1e-6)


def test_overrides_change_only_the_named_class():
    w = world.make_world(SEED, dict(kind="brain", b0_t=3.0, n=160))
    base = properties.world_maps(w, 128e6)
    new = properties.world_maps(w, 128e6, overrides=dict(csf=dict(sigma=4.0)))
    k = w["names"].index("csf")
    touched = w["weights"][k] > 0
    assert np.array_equal(base[0], new[0]) and np.array_equal(base[2], new[2])
    assert np.array_equal(base[1][~touched], new[1][~touched]) and (new[1][touched] >= base[1][touched]).all() and new[1].max() > base[1].max()
