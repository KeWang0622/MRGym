"""The finite-difference solver: refusals, the grid, linearity, SAR maps, the world adaptor."""
import math

import numpy as np
import pytest

from mrgym.em import analytic, coils, solver
from mrgym.scanner import world

SEED = "000102030405060708090a0b0c0d0e0f"


def empty(n=61, h=0.004, f=128e6, **kw):
    return solver.Solver(f, np.ones((n, n)), np.zeros((n, n)), h, **kw)


def test_refused_inputs():
    with pytest.raises(ValueError):
        solver.Solver(128e6, np.ones((10, 12)), np.zeros((10, 12)), 0.004)
    body = np.ones((61, 61))
    body[2, 30] = 50.0
    with pytest.raises(ValueError, match="rim"):
        solver.Solver(128e6, body, np.zeros((61, 61)), 0.004, n_pml=8)
    with pytest.raises(ValueError, match="shield"):
        solver.Solver(128e6, body, np.zeros((61, 61)), 0.004, n_pml=0, shield_radius_m=0.1)
    with pytest.raises(ValueError, match="absorbing layer"):
        empty(f=5e6, n_pml=16)
    with pytest.raises(ValueError, match="absorbing layer"):
        empty(f=20e6, n_pml=16)
    empty(f=20e6, n_pml=24, n=81)                                            # the thicker layer is accepted
    empty(f=1e5, n_pml=0, shield_radius_m=0.1)                               # and any frequency inside a shield
    s = empty(n_pml=8)
    with pytest.raises(ValueError, match="outside the physical part"):
        s.solve([[(0.11, 0.0, 1.0)]])
    sh = empty(n_pml=0, shield_radius_m=0.1)
    with pytest.raises(ValueError, match="shield"):
        sh.solve([[(0.099, 0.0, 1.0)]])
    with pytest.raises(ValueError, match="rim"):
        s.box(0.2)
    with pytest.raises(ValueError):
        solver.embed([np.ones((10, 10))], 0.001, 3, 0.02, 4, [1.0])


def test_the_matrix_is_symmetric_with_layer_and_with_shield():
    for kw in (dict(n_pml=8), dict(n_pml=0, shield_radius_m=0.1)):
        a = empty(**kw)._matrix()
        assert abs(a - a.T).max() == 0


def test_linearity_and_superposition():
    n, h = 81, 0.004
    y = (np.arange(n) - n // 2) * h
    yy, xx = np.meshgrid(y, y, indexing="ij")
    cyl = analytic.LayeredCylinder(128e6, [0.06], [60.0], [0.7], 0.1)
    eps, sig = cyl.cell_maps(yy, xx, h)
    s = solver.Solver(128e6, eps, sig, h, n_pml=10)
    a, b = [(0.1, 0.0, 1.0)], [(0.0, -0.1, 0.5j)]
    f = s.fields([a, b, a + b, [(0.1, 0.0, 2.0)], []])
    for k in ("ez", "b1p", "bx"):
        assert np.allclose(f[k][2], f[k][0] + f[k][1], rtol=0, atol=1e-12 * np.abs(f[k][2]).max())
        assert np.allclose(f[k][3], 2 * f[k][0], rtol=0, atol=1e-12 * np.abs(f[k][3]).max())
        assert not f[k][4].any()                                             # no current, no field
    w = np.array([0.3 - 0.2j, 1.1j])
    both = solver.combine(dict((k, f[k][:2]) for k in ("ez", "bx", "by", "b1p", "b1m")), w)
    direct = s.fields([[(0.1, 0.0, w[0]), (0.0, -0.1, 0.5j * w[1])]])
    for k in ("ez", "b1p", "b1m"):
        assert np.allclose(both[k], direct[k][0], rtol=0, atol=1e-12 * np.abs(direct[k][0]).max())
    # rotating components are what the definitions say
    assert np.allclose(f["b1p"], (f["bx"] + 1j * f["by"]) / 2) and np.allclose(f["b1m"], np.conj(f["bx"] - 1j * f["by"]) / 2)


def test_a_quarter_turn_of_the_problem_turns_the_field():
    """The grid has the symmetry of the square: a source moved from +x to +y gives the transposed field."""
    s = empty(n=61, n_pml=10)
    ez = s.solve([[(0.0, 0.05, 1.0)], [(0.05, 0.0, 1.0)]])
    assert np.allclose(ez[0], ez[1].T, rtol=0, atol=1e-12 * np.abs(ez).max())


def test_coarsening_keeps_objects_where_they_are():
    """A block of the master grid (640 points of 0.4 mm) has the same centre of mass on the solver grid, for every
    coarsening factor: no half-pixel shift."""
    n, d = 640, 0.4e-3
    a = np.zeros((n, n))
    a[400:440, 240:320] = 1.0                                                # edges on multiples of 40: whole blocks for every factor
    cy = (np.arange(400, 440).mean() - n // 2) * d
    cx = (np.arange(240, 320).mean() - n // 2) * d
    for factor in (1, 2, 4, 5, 8):
        (b,), h, origin, pad = solver.embed([a], d, factor, 0.15, 6, [0.0])
        assert h == pytest.approx(d * factor)
        m = b.shape[0]
        y = (np.arange(m) - m // 2) * h + origin[0]
        x = (np.arange(m) - m // 2) * h + origin[1]
        assert b.sum() * h * h == pytest.approx(a.sum() * d * d)
        assert (b.sum(axis=1) @ y) / b.sum() == pytest.approx(cy, abs=1e-9)
        assert (b.sum(axis=0) @ x) / b.sum() == pytest.approx(cx, abs=1e-9)
        assert y[6] <= -0.15 + h and y[-7] >= 0.15 - h                       # the physical part reaches the asked half-width


def test_the_way_back_to_the_master_grid_is_exact_for_a_plane():
    w = world.make_world(SEED, dict(kind="qa", b0_t=3.0, n=160))
    wm = solver.WorldModel(w, 128e6, factor=4, reach_m=0.16, n_pml=8)
    s = wm.solver
    plane = 2.0 + 30.0 * s.y - 12.0 * s.x + 1j * (5.0 * s.x)
    back = wm.to_world(plane)
    yy, xx = world.grid(w)
    want = 2.0 + 30.0 * yy * 1e-3 - 12.0 * xx * 1e-3 + 1j * (5.0 * xx * 1e-3)
    assert back.shape == (160, 160) and np.abs(back - want).max() < 1e-9
    # the masks sit where the object is
    body = np.asarray(w["masks"]["body"], float)
    cy = (body.sum(axis=1) @ yy[:, 0]) / body.sum() * 1e-3
    got = (wm.masks["body"].sum(axis=1) @ s.y[:, 0]) / wm.masks["body"].sum()
    assert got == pytest.approx(cy, abs=0.3 * s.h)


@pytest.mark.parametrize("kind, reach, radius", [("brain", 0.17, 0.14), ("abdomen", 0.32, 0.30), ("acr", 0.17, 0.14), ("nist_t1", 0.17, 0.14)])
def test_world_models_of_every_kind(kind, reach, radius):
    w = world.make_world(SEED, dict(kind=kind, b0_t=3.0, n=320))
    keep = {k: (v.copy() if isinstance(v, np.ndarray) else None) for k, v in w["fields"].items()}
    weights = w["weights"].copy()
    wm = solver.WorldModel(w, 128e6, factor=4, reach_m=reach, n_pml=10)
    f = wm.solver.fields(coils.birdcage(radius, 16))
    organ = wm.masks["organ"]
    assert organ.any() and np.isfinite(f["b1p"]).all()
    b = np.abs(f["b1p"][0][organ])
    assert b.min() > 0 and b.max() / b.min() < 6
    assert np.abs(f["b1m"][0][organ]).mean() < b.mean()                      # quadrature drive: mostly B1+
    # the world is read, not changed
    assert np.array_equal(weights, w["weights"])
    for k, v in keep.items():
        if v is not None:
            assert np.array_equal(v, w["fields"][k])
    # the same world gives the same field, bit for bit
    again = solver.WorldModel(w, 128e6, factor=4, reach_m=reach, n_pml=10).solver.fields(coils.birdcage(radius, 16))
    assert np.array_equal(again["b1p"], f["b1p"])


def test_maps_with_jumps_come_back_without_overshoot():
    """The default way back (bilinear) keeps a SAR map inside its own range; the cubic one, meant for the smooth
    field maps, does not."""
    w = world.make_world(SEED, dict(kind="brain", b0_t=3.0, n=320))
    wm = solver.WorldModel(w, 128e6, factor=4, n_pml=10)
    s = wm.solver
    ez = s.fields(coils.birdcage(0.14, 16))["ez"][0]
    sar = s.point_sar(ez, wm.density)
    back = wm.to_world(sar)
    assert back.min() >= 0 and back.max() <= sar.max() * (1 + 1e-12)
    assert wm.to_world(sar, order=3).min() < 0
    # the masks: tissue is every pixel with mass and contains the body mask; SAR has a value only where there is tissue
    assert (wm.masks["tissue"] | ~wm.masks["body"]).all() and wm.masks["tissue"].sum() > wm.masks["body"].sum()
    assert np.array_equal(s.sar_support(wm.density), wm.masks["tissue"]) and not sar[~wm.masks["tissue"]].any()
    support = s.sar_support(wm.density, solver.TEN_GRAM_SIDE_M)
    avg = s.averaged_sar(ez, wm.density)
    assert not avg[~support].any() and (avg[support] > 0).all()


def test_a_shield_replaces_the_absorbing_layer_in_a_world_model():
    w = world.make_world(SEED, dict(kind="qa", b0_t=3.0, n=160))
    wm = solver.WorldModel(w, 128e6, factor=4, shield_radius_m=0.19)
    assert wm.solver.n_pml == 0 and wm.solver.shield == 0.19
    f = wm.solver.fields(coils.birdcage(0.14, 16))
    outside = np.hypot(wm.solver.y, wm.solver.x) > 0.19 + wm.solver.h
    assert not f["ez"][0][outside].any()
    assert wm.solver.source_power(f["ez"], f["channels"]) == pytest.approx(wm.solver.absorbed_power(f["ez"][0]), rel=1e-9)


def test_sar_maps():
    n, h = 61, 0.002
    s = solver.Solver(64e6, np.full((n, n), 50.0), np.full((n, n), 0.5), h, n_pml=0)
    rho = np.full((n, n), 1000.0)
    rho[:, :20] = 0.0                                                        # air on the left
    rng = np.random.default_rng(2)
    ez = rng.normal(size=(3, n, n)) + 1j * rng.normal(size=(3, n, n))
    point = s.point_sar(ez[0], rho)
    assert not point[:, :20].any() and np.allclose(point[:, 20:], 0.25 * np.abs(ez[0][:, 20:]) ** 2 / 1000.0)
    kernel = s.averaging_kernel()
    assert len(kernel) == 11 and kernel.sum() * h == pytest.approx(solver.TEN_GRAM_SIDE_M) and (kernel[1:-1] == 1).all()
    assert np.allclose(s.averaging_kernel(0.004), [0.5, 1.0, 0.5]) and np.allclose(s.averaging_kernel(1e-6), [1.0])
    assert np.allclose(s.averaging_kernel(0.006), [1.0, 1.0, 1.0])
    avg = s.averaged_sar(ez[0], rho)
    assert not avg[:, :15].any()                                             # squares that are mostly air have no value
    assert avg[30, 20] > 0                                                   # a square that is half full has one
    one = s.averaged_sar(ez[0], rho, side_m=1e-6)
    assert np.allclose(one, point)
    # a square of three whole pixels, by hand
    three = s.averaged_sar(ez[0], rho, side_m=0.006)
    pd = 0.25 * np.abs(ez[0]) ** 2
    assert three[30, 40] == pytest.approx(pd[29:32, 39:42].sum() / rho[29:32, 39:42].sum())
    mask = np.zeros((n, n), bool)
    mask[10:50, 18:40] = True
    w = np.array([0.5, -1.0j, 0.2 + 0.1j])
    total = np.tensordot(w, ez, 1)
    for side in (None, solver.TEN_GRAM_SIDE_M):
        q = s.sar_matrices(ez, rho, mask, side)
        assert q.shape == (mask.sum(), 3, 3) and np.allclose(q, np.conj(np.swapaxes(q, 1, 2)))
        want = s.point_sar(total, rho) if side is None else s.averaged_sar(total, rho)
        got = np.real(np.einsum("i,pij,j->p", np.conj(w), q, w))
        assert np.allclose(got, want[mask], rtol=1e-10, atol=1e-12 * want.max())
    r = s.power_matrix(ez)
    assert np.allclose(r, r.conj().T) and np.real(np.conj(w) @ r @ w) / 2 == pytest.approx(s.absorbed_power(total))


def test_quasi_static_field_of_a_small_shielded_birdcage():
    """An independent closed form on a coarse grid: mu0 N I / (4 pi R) (1 - (R / S)^2)."""
    h = 0.004
    n = 2 * int(round(0.2 / h)) + 1
    s = solver.Solver(1e5, np.ones((n, n)), np.zeros((n, n)), h, n_pml=0, shield_radius_m=0.19)
    f = s.fields(coils.birdcage(0.14, 16))
    want = analytic.quasi_static_b1(n_rungs=16, source_radius_m=0.14, shield_radius_m=0.19)
    assert abs(s.sample(f["b1p"][0], 0.0, 0.0)) == pytest.approx(want, rel=2e-4)
    assert abs(s.sample(f["b1m"][0], 0.0, 0.0)) < 1e-3 * want
    assert math.isclose(want, 4e-7 * math.pi * 16 / (4 * math.pi * 0.14) * (1 - (0.14 / 0.19) ** 2), rel_tol=1e-9)
