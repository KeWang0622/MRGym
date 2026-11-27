"""Spin operators, Hamiltonians and equivalent groups."""
import math

import numpy as np
import pytest

from mrgym.design import rotation
from mrgym.design.units import GAMMA_HZ_PER_T
from mrgym.spins import operators


def test_a_spin_system_checks_its_description():
    with pytest.raises(ValueError):
        operators.SpinSystem("x", ())
    with pytest.raises(ValueError):
        operators.SpinSystem("x", (1.0, 2.0), ((0, 0, 7.0),))
    with pytest.raises(ValueError):
        operators.SpinSystem("x", (1.0, 2.0), ((0, 1, 7.0), (1, 0, 7.0)))
    with pytest.raises(ValueError):
        operators.SpinSystem("x", (1.0, 2.0), ((0, 2, 7.0),))
    with pytest.raises(ValueError):
        operators.SpinSystem("x", (1.0,), (), counts=(0,))
    with pytest.raises(ValueError):
        operators.SpinSystem("x", (1.0, float("nan")), ())
    with pytest.raises(ValueError):
        operators.SpinSystem("x", (1.0,), (), counts=(operators.MAX_SPINS + 1,))
    s = operators.SpinSystem("x", (1.0, 2.0), ((1, 0, 7.0),), counts=(3, 1))
    assert s.j_hz == ((0, 1, 7.0),) and s.n_spins == 4 and s.n_sites == 2 and s.groups == ("x", "x")


def test_frequencies_follow_the_convention_of_the_engine():
    """A larger chemical shift is a higher frequency; fat (1.3 ppm) is 3.4 ppm x gamma x B0 below water (4.7 ppm)."""
    f = operators.frequencies_hz((1.3, 4.7), 3.0, 4.7)
    assert f[1] == 0 and f[0] == pytest.approx(-3.4e-6 * GAMMA_HZ_PER_T * 3.0)
    assert operators.larmor_hz_per_ppm(1.5) == pytest.approx(63.866, abs=1e-3)


def test_sectors_of_equivalent_groups_fill_the_whole_space():
    for counts in ((3, 1), (2, 2, 2), (9,), (3, 2, 1, 1)):
        c = operators.compile(operators.SpinSystem("s", tuple(1.0 + k for k in range(len(counts))), (), counts=counts))
        assert sum(sec.weight * sec.dim for sec in c.sectors) == 2 ** sum(counts)
        assert c.norm == 2.0 ** (sum(counts) - 2)
    c = operators.compile(operators.SpinSystem("ch3", (1.0,), (), counts=(3,)))
    assert sorted((sec.js, sec.weight) for sec in c.sectors) == [((0.5,), 2), ((1.5,), 1)]


def test_expand_gives_one_site_per_proton_with_all_couplings():
    s = operators.SpinSystem("s", (1.0, 2.0), ((0, 1, 7.0),), counts=(3, 2)).expand(intra_j_hz=-12.0)
    assert s.n_sites == 5 and s.counts == (1,) * 5
    assert sum(1 for p in s.j_hz if p[2] == 7.0) == 6 and sum(1 for p in s.j_hz if p[2] == -12.0) == 3 + 1


def test_a_sector_above_the_limit_is_refused():
    s = operators.SpinSystem("s", tuple(1.0 + 0.3 * k for k in range(9)), ())
    with pytest.raises(ValueError):
        operators.compile(s)
    with pytest.raises(ValueError):
        operators.compile(operators.SpinSystem("s", (1.0, 2.0), ((0, 1, 7.0),)), coupling="medium")
    with pytest.raises(ValueError):
        operators.spin_matrices(0.75)


def test_wigner_matrix_of_spin_one_half_is_the_spinor_of_the_design_layer():
    a, b = rotation.axis_angle((0.3, -0.5, 0.8), 1.234)
    q = operators.wigner(0.5, a, b)
    assert np.allclose(q, [[a, -np.conj(b)], [b, np.conj(a)]])
    assert operators.wigner(0.0, a, b).shape == (1, 1)
    stack = operators.wigner(1.5, np.array([a, 1.0]), np.array([b, 0.0]))
    assert stack.shape == (2, 4, 4) and np.allclose(stack[1], np.eye(4))


def test_the_rotation_of_a_methyl_group_is_the_product_of_three_proton_rotations():
    """Total-spin sectors: the trace of the rotation over all sectors equals the trace of the three-fold product."""
    a, b = rotation.axis_angle((1.0, 2.0, -0.5), 0.9)
    q = np.array([[a, -np.conj(b)], [b, np.conj(a)]])
    full = np.kron(np.kron(q, q), q)
    c = operators.compile(operators.SpinSystem("ch3", (1.0,), (), counts=(3,)))
    tr = sum(sec.weight * np.trace(sec.rotation(np.array([a]), np.array([b]))) for sec in c.sectors)
    assert tr == pytest.approx(np.trace(full))


def test_eigenvectors_have_a_definite_m_even_when_levels_cross():
    """Two uncoupled protons at the same shift: degenerate levels in different M blocks must not mix."""
    c = operators.compile(operators.SpinSystem("s", (2.0, 2.0, 4.7), ((0, 1, 5.0),)))
    sec = c.sectors[0]
    fz = sec.vec @ np.diag(sec.m) @ sec.vec.T
    assert np.allclose(fz, np.diag(sec.iz_product.sum(axis=0)), atol=1e-13)
    assert set(np.unique(sec.order)) == {-3, -2, -1, 0, 1, 2, 3}
    assert np.allclose(sec.site_operator(0, "z"), sec.to_eigen(np.diag(sec.iz_product[0])), atol=1e-13)
    assert math.isclose(np.trace(sec.fz() @ sec.fz()).real, 3 * c.norm)
