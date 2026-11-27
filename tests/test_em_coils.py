"""Coils as line currents, reception, noise matrix, g-factor."""
import math

import numpy as np
import pytest

from mrgym.em import coils, solver
from mrgym.em.units import K_BOLTZMANN


def test_geometry_of_the_coils():
    (ch,) = coils.birdcage(0.14, 16)
    assert len(ch) == 16 and all(math.hypot(y, x) == pytest.approx(0.14) for y, x, _ in ch)
    assert ch[0][:2] == (0.0, 0.14) and ch[4][0] == pytest.approx(0.14)      # angles run from +x towards +y
    assert abs(sum(c for _, _, c in ch)) < 1e-12                             # mode 1 carries no net current
    i_port, q_port = coils.birdcage_ports(0.14, 16)
    quad = [a[2] - 1j * b[2] for a, b in zip(i_port, q_port)]
    assert np.allclose(quad, [c for _, _, c in ch])
    lines = coils.array(0.14, 8, "line")
    loops = coils.array(0.14, 8, "loop")
    assert len(lines) == len(loops) == 8 and all(len(c) == 1 for c in lines) and all(len(c) == 2 for c in loops)
    assert all(abs(c[0][2] + c[1][2]) == 0 for c in loops)                   # a loop carries no net current
    width = math.atan2(loops[0][1][0], loops[0][1][1]) - math.atan2(loops[0][0][0], loops[0][0][1])
    assert width == pytest.approx(0.8 * 2 * math.pi / 8)
    with pytest.raises(ValueError):
        coils.array(0.14, 8, "patch")
    assert np.allclose(coils.mode_weights(4), [1, -1j, -1, 1j]) and np.allclose(coils.mode_weights(4, sense=-1), [1, 1j, -1, -1j])


def test_quadrature_makes_b1_plus_and_reverse_makes_b1_minus():
    h = 0.004
    n = 2 * int(round(0.2 / h)) + 1
    s = solver.Solver(64e6, np.ones((n, n)), np.zeros((n, n)), h, n_pml=0, shield_radius_m=0.19)
    quad = s.fields(coils.birdcage(0.14, 16, "quadrature"))
    rev = s.fields(coils.birdcage(0.14, 16, "reverse"))
    lin = s.fields(coils.birdcage(0.14, 16, "linear"))
    at = (0.0, 0.0)
    assert abs(s.sample(quad["b1m"][0], *at)) < 1e-3 * abs(s.sample(quad["b1p"][0], *at))
    assert abs(s.sample(rev["b1p"][0], *at)) < 1e-3 * abs(s.sample(rev["b1m"][0], *at))
    assert abs(s.sample(lin["b1p"][0], *at)) == pytest.approx(abs(s.sample(lin["b1m"][0], *at)), rel=1e-6)
    assert abs(s.sample(lin["b1p"][0], *at)) == pytest.approx(abs(s.sample(quad["b1p"][0], *at)) / 2, rel=1e-6)
    # a line element on the +x side: the field at the centre points along -y; a loop there: along the radius
    line = s.fields(coils.array(0.14, 8, "line")[:1])
    loop = s.fields(coils.array(0.14, 8, "loop")[:1])
    assert abs(s.sample(line["bx"][0], *at)) < 1e-6 * abs(s.sample(line["by"][0], *at)) and s.sample(line["by"][0], *at).real < 0
    assert abs(s.sample(loop["by"][0], *at)) < 1e-6 * abs(s.sample(loop["bx"][0], *at))
    # receive sensitivity of the reverse mode equals in size the transmit field of the quadrature mode (mirror image)
    assert abs(s.sample(coils.receive_maps(rev)[0], *at)) == pytest.approx(abs(s.sample(quad["b1p"][0], *at)), rel=1e-9)


def test_noise_matrices_and_combinations():
    rng = np.random.default_rng(1)
    g = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    psi = g @ g.conj().T + 0.1 * np.eye(5)
    s = rng.normal(size=(5, 7)) + 1j * rng.normal(size=(5, 7))
    opt = coils.snr_optimal(s, psi)
    want = np.sqrt(np.real(np.einsum("ik,ij,jk->k", np.conj(s), np.linalg.inv(psi), s)))
    assert np.allclose(opt, want)
    assert coils.snr_optimal(s.reshape(5, 7, 1), psi).shape == (7, 1)
    for k in range(7):
        w = np.linalg.solve(psi, s[:, k])
        assert coils.snr_combined(s[:, k:k + 1], psi, w)[0] == pytest.approx(opt[k])
        assert coils.snr_combined(s[:, k:k + 1], psi, rng.normal(size=5) + 0j)[0] <= opt[k] * (1 + 1e-12)
    assert np.allclose(coils.noise_covariance(psi, 300.0, 1000.0), 4 * K_BOLTZMANN * 300.0 * 1000.0 * psi)
    with pytest.raises(np.linalg.LinAlgError):
        coils.snr_optimal(s, -psi)                                           # a noise matrix has to be positive definite


def test_g_factor():
    psi = np.eye(3)
    same = np.ones((3, 2), complex)
    assert np.isinf(coils.g_factor(same, psi)).all()                         # two places with the same sensitivities cannot be told apart
    orth = np.array([[1, 0], [0, 1], [0, 0]], complex)
    assert np.allclose(coils.g_factor(orth, psi), 1.0)
    s = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], complex)               # 45 degrees between the two columns
    assert np.allclose(coils.g_factor(s, psi), math.sqrt(2.0))
    rx = np.zeros((2, 8, 3), complex)
    rx[0, :4], rx[1, 4:] = 1.0, 1.0                                          # channel 0 sees the upper half, channel 1 the lower
    mask = np.ones((8, 3), bool)
    mask[0, 0] = False
    g = coils.g_map(rx, np.eye(2), 2, mask)
    assert g.shape == (8, 3) and g[0, 0] == 0 and np.allclose(g[mask], 1.0)
    with pytest.raises(ValueError):
        coils.g_map(rx[:, :7], np.eye(2), 2, mask[:7])
