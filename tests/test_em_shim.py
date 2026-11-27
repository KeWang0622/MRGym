"""RF shimming on synthetic maps: guarantees, limits, gradients."""
import numpy as np
import pytest

from mrgym.em import shim


def problem(seed=0, n_pix=300, nc=6):
    """Smooth synthetic channel maps on a line of pixels, a power matrix and local-SAR matrices."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, n_pix)[:, None]
    centre = np.linspace(-1.2, 1.2, nc)[None, :]
    a = np.exp(-(x - centre) ** 2 / 0.8) * np.exp(1j * (2.0 * x * centre + rng.uniform(0, 6.28, nc)[None, :])) * 1e-7
    e = rng.normal(size=(40, nc)) + 1j * rng.normal(size=(40, nc))
    q = np.einsum("pi,pj->pij", np.conj(e), e) * 0.02
    r = q.sum(axis=0) * 3.0
    return a, r, q, np.ones(nc, complex)


def test_scaled_reference_is_the_best_scale_of_the_start():
    a, r, q, w0 = problem()
    t = 1e-6
    ref = shim.scaled_reference(a, w0, t)
    f0 = shim.objective(a, ref, t)
    for s in (0.9, 0.99, 1.01, 1.1):
        assert shim.objective(a, s * ref, t) > f0
    lim = 0.5 * shim.peak_sar(q, ref)[0]
    capped = shim.scaled_reference(a, w0, t, q=q, sar_max=lim)
    assert shim.peak_sar(q, capped)[0] == pytest.approx(lim) and np.allclose(capped / ref, np.sqrt(0.5))
    capped = shim.scaled_reference(a, w0, t, r=r, p_max=0.25 * shim.power(r, ref))
    assert np.allclose(capped / ref, 0.5)


@pytest.mark.parametrize("kind", ["full", "phase"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_never_worse_than_the_start_and_inside_the_limits(kind, seed):
    a, r, q, w0 = problem(seed)
    t = 1e-6
    ref = shim.scaled_reference(a, w0, t)
    p0, s0 = shim.power(r, ref), shim.peak_sar(q, ref)[0]
    for lim in ({}, dict(r=r, p_max=0.6 * p0), dict(q=q, sar_max=0.5 * s0), dict(r=r, p_max=0.9 * p0, q=q, sar_max=0.4 * s0)):
        res = shim.solve(a, t, w0, kind=kind, **lim)
        start = shim.objective(a, shim.scaled_reference(a, w0, t, **lim), t)
        assert res["start_f"] == pytest.approx(start)
        assert res["f"] <= start * (1 + 1e-12) and res["feasible"] and not res["fallback"]
        assert res["f"] == pytest.approx(shim.objective(a, res["w"], t))
        assert res["raw_f"] <= start * (1 + 1e-12) and 0 <= res["raw_excess"] < 1e-6      # what the optimiser itself delivered
        if "p_max" in lim:
            assert shim.power(r, res["w"]) <= lim["p_max"] * (1 + 1e-9)
        if "sar_max" in lim:
            assert shim.quad(q, res["w"]).max() <= lim["sar_max"] * (1 + 1e-9)
        assert res["kkt"] < 1e-5
        if kind == "phase":
            assert np.allclose(np.abs(res["w"]), np.abs(res["w"][0]))            # one amplifier: equal magnitudes


def test_a_tighter_limit_costs_uniformity_and_the_curve_is_monotone():
    a, _, q, w0 = problem(3)
    t = 1e-6
    s0 = shim.peak_sar(q, shim.scaled_reference(a, w0, t))[0]
    limits = [1.2 * s0, 0.3 * s0, 0.6 * s0, 0.15 * s0]
    curve = shim.tradeoff(a, t, w0, q, limits)
    assert [c["sar_max"] for c in curve] == limits
    by_limit = sorted(curve, key=lambda c: c["sar_max"])
    fs = [c["f"] for c in by_limit]
    assert all(b <= a_ * (1 + 1e-12) for a_, b in zip(fs, fs[1:])) and fs[0] > 1.5 * fs[-1]
    assert all(c["peak_sar"] <= c["sar_max"] * (1 + 1e-9) for c in curve)


def test_more_starts_never_hurt():
    a, _, _, w0 = problem(4)
    one = shim.solve(a, 1e-6, w0)
    many = shim.solve(a, 1e-6, w0, starts=shim.random_starts(w0, 5, seed=1))
    assert many["f"] <= one["f"] * (1 + 1e-12) and many["start_f"] == one["start_f"]
    assert len(shim.random_starts(w0, 5)) == 5 and np.allclose(shim.random_starts(w0, 2, 7)[1], shim.random_starts(w0, 2, 7)[1])


def test_gradient_of_the_objective_in_both_parametrisations():
    a, _, _, _ = problem(5)
    rng = np.random.default_rng(0)
    w = (rng.normal(size=6) + 1j * rng.normal(size=6)) * 3.0
    t = 1e-6
    f, g = shim._f_grad(a, w, t)
    assert f == pytest.approx(shim.objective(a, w, t))
    d = 1e-6
    for i in range(6):
        for step in (d, 1j * d):
            num = (shim.objective(a, w + step * np.eye(6)[i], t) - shim.objective(a, w - step * np.eye(6)[i], t)) / (2 * d)
            want = 2 * (g[i].real if step == d else g[i].imag)
            assert num == pytest.approx(want, rel=1e-5, abs=1e-9)
        # a phase turn of channel i: dw = i w_i dtheta
        turn = np.exp(1j * d * np.eye(6)[i])
        num = (shim.objective(a, w * turn, t) - shim.objective(a, w / turn, t)) / (2 * d)
        assert num == pytest.approx(2 * np.real(np.conj(g[i]) * 1j * w[i]), rel=1e-5, abs=1e-9)


def test_refusals_and_small_helpers():
    a, r, q, w0 = problem()
    with pytest.raises(ValueError):
        shim.solve(a, 1e-6, w0, kind="amplitude")
    u = shim.uniformity(a, w0)
    assert set(u) == {"mean", "cv", "min_over_mean", "max_over_mean"} and u["min_over_mean"] <= 1 <= u["max_over_mean"]
    b1p = np.arange(24, dtype=complex).reshape(2, 3, 4)
    mask = np.zeros((3, 4), bool)
    mask[1, 2] = mask[2, 0] = True
    assert np.array_equal(shim.system_matrix(b1p, mask), np.array([[6, 18], [8, 20]], complex))
    assert shim.quad(q, w0).shape == (40,) and shim.power(r, w0) == pytest.approx(shim.quad(r, w0) / 2)
    lim = shim.IEC_LIMITS_W_PER_KG
    assert lim["normal"]["head"] == 3.2 and lim["normal"]["whole_body"] == 2.0 and lim["first_level"]["whole_body"] == 4.0
    assert lim["normal"]["local_head_trunk_10g"] == 10.0 and lim["first_level"]["local_head_trunk_10g"] == 20.0
    assert shim.kkt_residual(a, shim.scaled_reference(a, w0, 1e-6), 1e-6) > 1e-2   # the start is not a stationary point
