"""The inverse relation: electrical properties from B1+."""
import math

import numpy as np
import pytest

from mrgym.em import ept
from mrgym.em.units import wavenumber


def plane_wave(f, eps_r, sigma, h, n=81, angle=0.4):
    k = wavenumber(f, eps_r, sigma)
    y = (np.arange(n) - n // 2) * h
    yy, xx = np.meshgrid(y, y, indexing="ij")
    return np.exp(-1j * k * (math.cos(angle) * xx + math.sin(angle) * yy))


@pytest.mark.parametrize("f, eps_r, sigma", [(64e6, 80.0, 0.5), (128e6, 60.0, 0.7), (298e6, 50.0, 0.6)])
def test_a_decaying_plane_wave_gives_back_its_medium(f, eps_r, sigma):
    """Independent of the cylinder: exp(-i k . r) obeys the Helmholtz equation of the medium."""
    b = plane_wave(f, eps_r, sigma, 0.002)
    for m in (1, 2, 3):
        got = ept.helmholtz(b, f, 0.002, m)
        inner = (slice(m, -m), slice(m, -m))
        assert np.abs(got["eps_r"][inner] / eps_r - 1).max() < 0.02
        assert np.abs(got["sigma"][inner] / sigma - 1).max() < 0.02
    # a plane wave has uniform |B1+| along the wave fronts and a linear phase: the phase-only estimate misses the loss term
    assert np.abs(ept.sigma_from_phase(np.angle(b), f, 0.002, 2)[10:-10, 10:-10]).max() < 0.05 * sigma


def test_the_error_of_the_estimate_falls_with_the_grid_as_h_squared():
    f, eps_r, sigma = 298e6, 50.0, 0.6
    errs = []
    for h in (0.004, 0.002, 0.001):
        got = ept.helmholtz(plane_wave(f, eps_r, sigma, h, n=41), f, h, 1)
        errs.append(np.abs(got["sigma"][2:-2, 2:-2] / sigma - 1).max())
    assert 3.5 < errs[0] / errs[1] < 4.5 and 3.5 < errs[1] / errs[2] < 4.5


def test_kernel_properties():
    for m in (1, 2, 4):
        k = ept.laplacian_kernel(m)
        assert k.shape == (2 * m + 1, 2 * m + 1)
        assert abs(k.sum()) < 1e-12                                          # a constant has no Laplacian
        assert np.allclose(k, k.T) and np.allclose(k, k[::-1]) and np.allclose(k, k[:, ::-1])
        v, u = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1), indexing="ij")
        assert np.sum(k * v * v) == pytest.approx(2.0) and np.sum(k * u * u) == pytest.approx(2.0) and abs(np.sum(k * u * v)) < 1e-12
    with pytest.raises(ValueError):
        ept.laplacian_kernel(0)


def test_a_larger_window_averages_noise():
    rng = np.random.default_rng(4)
    noise = rng.normal(size=(200, 200))
    sd = [ept.laplacian(noise, 1.0, m)[10:-10, 10:-10].std() for m in (1, 2, 4)]
    assert sd[0] > 3 * sd[1] > 9 * sd[2]
    # the kernel's own noise gain: the root of the sum of its squares
    assert sd[1] == pytest.approx(math.sqrt(np.sum(ept.laplacian_kernel(2) ** 2)), rel=0.05)


def test_valid_region_and_unwrap():
    mask = np.zeros((30, 30), bool)
    mask[5:25, 5:25] = True
    v = ept.valid_region(mask, 2)
    assert v.sum() == 16 * 16 and v[7:23, 7:23].all()
    y, x = np.meshgrid(np.arange(40), np.arange(40), indexing="ij")
    phase = 0.4 * x + 0.3 * y - 5.0
    wrapped = np.angle(np.exp(1j * phase))
    un = ept.unwrap(wrapped)
    assert np.allclose(un - un[20, 20], phase - phase[20, 20])
    assert np.allclose(ept.transceive_phase(np.exp(0.3j) * np.ones(3), np.exp(0.5j) * np.ones(3)), 0.8)
