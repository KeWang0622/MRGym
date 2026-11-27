"""The general RF rotation: shapes, argument checks and the algebra of Cayley-Klein parameters."""
import numpy as np
import pytest

from mrgym.design import rotation as R
from mrgym.design.units import GAMMA_HZ_PER_T, RAD_PER_UT_MS


def test_gamma_is_the_engine_value():
    from mrgym.scanner import console
    assert GAMMA_HZ_PER_T == console.GAMMA_HZ_PER_T


def test_isochromat_arrays_broadcast_to_a_grid():
    b1 = np.full(20, 5.0 + 0j)
    z, df, sc = np.linspace(-5, 5, 7)[:, None, None], np.array([0.0, 100.0])[None, :, None], np.array([0.8, 1.0, 1.2])[None, None, :]
    a, b = R.rotate(b1, 0.01, 10.0, z, df, sc)
    assert a.shape == b.shape == (7, 2, 3)
    a1, b1_ = R.rotate(b1, 0.01, 10.0, z[3, 0, 0], 100.0, 1.2)
    assert a1.shape == () and abs(a1 - a[3, 1, 2]) < 1e-14 and abs(b1_ - b[3, 1, 2]) < 1e-14


def test_three_axes_need_three_components():
    b1, g = np.ones(4, complex), np.ones((4, 3))
    assert R.rotate(b1, 0.01, g, np.zeros((5, 3)))[0].shape == (5,)
    with pytest.raises(ValueError):
        R.rotate(b1, 0.01, g, np.zeros(5))
    with pytest.raises(ValueError):
        R.rotate(b1, 0.01, np.ones((3, 3)), np.zeros((5, 3)))
    with pytest.raises(ValueError):
        R.rotate(b1, 0.01, np.ones((4, 4)), np.zeros((5, 4)))


@pytest.mark.parametrize("bad", [dict(b1_ut=[]), dict(dt_ms=0.0), dict(dt_ms=-1.0), dict(b1_ut=[np.nan]), dict(model="exact")])
def test_bad_arguments_are_refused(bad):
    args = dict(b1_ut=[1.0, 2.0], dt_ms=0.01)
    args.update(bad)
    with pytest.raises(ValueError):
        R.rotate(**args)


def test_no_pulse_is_no_rotation_and_a_gradient_alone_is_a_phase():
    a, b = R.rotate(np.zeros(10), 0.1, 3.0, np.array([0.0, 2.0]))
    assert np.allclose(b, 0) and abs(a[0] - 1) < 1e-15
    mxy, mz = R.apply(a, b, 1.0, 0.3)
    angle = 2 * np.pi * GAMMA_HZ_PER_T * 3e-3 * 2e-3 * 1e-3            # gamma G z t: positive for positive G and z
    assert abs(mxy[1] - np.exp(1j * angle)) < 1e-12 and np.allclose(mz, 0.3)


def test_sense_of_rotation_matches_the_engine_convention():
    """A real positive B1 turns +z towards -y; a phase of 90 degrees turns it towards +x."""
    flip = np.pi / 2
    a, b = R.rotate([flip / (RAD_PER_UT_MS * 0.5)], 0.5)
    assert abs(R.apply(a, b)[0] + 1j) < 1e-12
    a, b = R.rotate([1j * flip / (RAD_PER_UT_MS * 0.5)], 0.5)
    assert abs(R.apply(a, b)[0] - 1) < 1e-12
    assert np.allclose(R.matrix(*R.axis_angle([1, 0, 0], flip)) @ [0, 0, 1], [0, -1, 0])


def test_compose_inverse_and_matrix_agree():
    rng = np.random.default_rng(0)
    r1, r2 = R.axis_angle(rng.normal(size=3), 1.1), R.axis_angle(rng.normal(size=3), -2.3)
    both = R.compose(*r1, *r2)
    assert np.allclose(R.matrix(*both), R.matrix(*r2) @ R.matrix(*r1))
    ident = R.compose(*both, *R.inverse(*both))
    assert abs(ident[0] - 1) < 1e-14 and abs(ident[1]) < 1e-14
    assert abs(np.linalg.det(R.matrix(*both)) - 1) < 1e-14
    assert abs(R.flip_angle(R.axis_angle([0, 1, 0], 0.7)[1]) - 0.7) < 1e-14


def test_relaxation_alone_and_full_decay():
    mxy, mz = R.bloch(np.zeros(100), 0.1, t1_ms=50.0, t2_ms=5.0, mxy0=0.4j, mz0=-1.0)
    assert abs(mxy - 0.4j * np.exp(-10.0 / 5.0)) < 1e-13 and abs(mz - (1 - 2 * np.exp(-10.0 / 50.0))) < 1e-13
    mxy, mz = R.bloch(np.full(50, 3.0), 0.02, t1_ms=np.array([300.0, 1e9]), t2_ms=np.array([20.0, 1e9]))
    assert mxy.shape == (2,) and abs(mxy[0]) < abs(mxy[1])
    with pytest.raises(ValueError):
        R.bloch([1.0], 0.1, t1_ms=0.0)


def test_equilibrium_magnetisation_per_isochromat():
    m_eq = np.array([[1.0, 0.5], [0.2, 0.0]])
    mxy, mz = R.bloch(np.zeros(10), 1.0, t1_ms=4.0, t2_ms=1.0, mz0=0.0, m_eq=m_eq)
    assert mz.shape == (2, 2) and np.allclose(mz, m_eq * (1 - np.exp(-10.0 / 4.0)))
