"""The three evaluators of a block list agree: isochromat ensemble, phase graph, periodic solve.

Seeded random block lists (other seeds than ``mrgym.core.validate``), and the properties the evaluators promise:
continuation from a state, exact pruning, truncation, the pulse frame, the reduced periodic method."""
import numpy as np
import pytest

from mrgym.core import bloch, epg, periodic
from mrgym.core import blocks as B
from mrgym.core import library as L
from mrgym.core import operators as O
from mrgym.core.validate import _random_blocks, _random_spins


@pytest.mark.parametrize("seed", range(8))
def test_ensemble_and_phase_graph_agree_on_random_lists(seed):
    rng = np.random.default_rng(1000 + seed)
    sp = _random_spins(rng, 3)
    axes = 1 + seed % 2
    seq = _random_blocks(rng, 20, axes=axes)
    e, r = epg.run(seq, sp), bloch.run(seq, sp)
    assert np.abs(e.signal - r.signal).max() < 1e-10
    mxy, mz = epg.magnetisation(e.state, r.state["x"])
    assert np.abs(mxy - (r.state["m"][0] + 1j * r.state["m"][1])[:, 0]).max() < 1e-10
    assert np.abs(mz - r.state["m"][2][:, 0]).max() < 1e-10
    # Z_-k = conj(Z_k) is kept, and the pruned run gives the same samples
    flip = (slice(None, None, -1),) * axes
    assert np.abs(e.state.z - np.conj(e.state.z[flip])).max() < 1e-12
    assert np.abs(epg.run(seq, sp, prune=True, keep_state=False).signal - e.signal).max() < 1e-12


@pytest.mark.parametrize("seed", range(4))
def test_a_run_continues_from_a_state(seed):
    rng = np.random.default_rng(2000 + seed)
    sp = _random_spins(rng, 3)
    seq = _random_blocks(rng, 24)
    whole = epg.run(seq, sp)
    first = epg.run(seq[:20], sp)
    second = epg.run(seq[20:], sp, state=first.state)
    assert np.abs(np.concatenate([first.signal, second.signal]) - whole.signal).max() < 1e-12
    assert np.abs(epg.magnetisation(second.state, [0.13, -0.4])[0] - epg.magnetisation(whole.state, [0.13, -0.4])[0]).max() < 1e-12
    ens = bloch.run(seq[:20], sp, n_sub=B.compile_ops(seq).reach[0] + 1)
    ens2 = bloch.run(seq[20:], sp, n_sub=B.compile_ops(seq).reach[0] + 1, start=ens.state)
    assert np.abs(ens2.signal - second.signal).max() < 1e-10
    with pytest.raises(ValueError):
        epg.run([B.gradient(0.5)], sp, state=first.state)


def test_truncation_drops_only_high_orders_and_converges():
    sp = O.spins([900.0, 400.0], [70.0, 30.0])
    seq = L.spgr(8.0, 25.0, repetitions=300)
    full = epg.run(seq, sp, frame=True, keep_state=False).signal[-1]
    errs = [np.abs(epg.run(seq, sp, kmax=k, frame=True, keep_state=False).signal[-1] - full).max() for k in (4, 16, 64)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-8
    # the rule of the periodic solve is the engine's: ceil(-ln(tol) T2 / TR) + 2 orders
    from mrgym.scanner import epg as engine_epg
    for t2, tr in ((80.0, 10.0), (2000.0, 5.0), (40.0, 30.0)):
        prog, _ = periodic.period(L.spgr(tr, 20.0))
        assert O.auto_kmax(prog, O.spins(1000.0, t2))[0] == engine_epg.n_states(tr, t2) - 1
    assert O.auto_kmax(periodic.period(L.afi(20.0, 5, 60.0))[0], O.spins(1000.0, 80.0))[0] == engine_epg.n_states(20.0, 80.0) - 1
    assert O.auto_kmax(periodic.period(L.bssfp(5.0, 40.0))[0], sp) == (0,)
    assert O.auto_kmax(B.compile_ops([B.repeat([B.rf(30), B.gradient(1), B.delay(5.0), B.rf(20)], 1)]), sp) is not None
    assert O.auto_kmax(B.compile_ops([B.repeat([B.rf(30), B.gradient(1), B.rf(20), B.delay(5.0)], 1)]), sp) is None
    with pytest.raises(ValueError):
        periodic.steady_state([B.repeat([B.rf(30), B.gradient(1), B.rf(20), B.delay(5.0)], 1)], sp)


@pytest.mark.parametrize("seed", range(4))
def test_the_periodic_state_is_a_fixed_point(seed):
    rng = np.random.default_rng(3000 + seed)
    sp = _random_spins(rng, 3, relax=(30.0, 80.0, 10.0, 30.0))
    body = [B.rf(flip_table="a"), B.adc("x"), B.delay(rng.uniform(0.5, 2)), B.rf(rng.uniform(5, 50), 40.0), B.adc("y"),
            B.gradient(2, ms=rng.uniform(1, 3)), B.gradient(-1, ms=1.0)]
    rep = [B.repeat(body, 1, phase_inc_deg=rng.uniform(-180, 180), spoil_inc_deg=float(rng.choice([0.0, 117.0])),
                    tables=dict(a=tuple(rng.uniform(10, 90, 1 + seed % 3))))]
    fixed = periodic.steady_state(rep, sp, kmax=12, method="matrix")
    n = fixed.signal.shape[0]
    again = epg.run(rep, sp, kmax=12, frame=True, repetitions=n // 2, state=fixed.state)          # one more period from the fixed point
    assert np.abs(again.signal - fixed.signal).max() < 1e-11
    assert np.abs(again.state.f - fixed.state.f).max() < 1e-11 and np.abs(again.state.z - fixed.state.z).max() < 1e-11
    stepped, periodic_samples = periodic.approach(rep, sp, 200, kmax=12)
    assert stepped.shape == (200, n, 3) and np.abs(stepped[-1] - periodic_samples).max() < 1e-10
    assert np.abs(stepped[0] - periodic_samples).max() > 1e-3                                       # the transient is there
    # an ensemble in the laboratory frame ends on the same samples (to what 48 isochromats and 23 orders resolve)
    lab = bloch.run([B.repeat(body, 240 * (n // 2), rep[0].phase_inc_deg, rep[0].spoil_inc_deg, dict(rep[0].tables))], sp, n_sub=48)
    assert np.abs(lab.signal[-n:] - periodic.steady_state(rep, sp, kmax=23, method="matrix").signal).max() < 1e-5


def test_the_reduced_method_and_where_it_applies():
    sp = O.spins([900.0, 400.0], [70.0, 30.0])
    seq = L.fse(700.0, 9.0, 6, [170, 150, 130, 110, 100, 100])
    red, mat = periodic.steady_state(seq, sp), periodic.steady_state(seq, sp, kmax=12, method="matrix")
    assert np.abs(red.signal - mat.signal).max() < 1e-12 and red.labels == mat.labels
    assert red.state.kmax == (6,) and np.abs(red.state.z[6] - mat.state.z[12]).max() < 1e-12
    with pytest.raises(ValueError):
        periodic.steady_state(L.spgr(10.0, 20.0), sp, method="reduced")
    with pytest.raises(ValueError):
        periodic.steady_state(seq, sp, method="other")
    with pytest.raises(ValueError):
        periodic.approach([B.rf(90), B.delay(5.0)], sp, 3)
    # two pools: the fixed point has two numbers per column
    spm = O.spins([900.0, 400.0], [70.0, 30.0], pool="mt", f_b=0.12, k_ab_per_s=3.0, t1b_ms=800.0, lineshape="gaussian")
    body = [B.rf(90, 0, energy_ut2_ms=50.0), B.delay(5.0), B.gradient(1), B.rf(160, 90, energy_ut2_ms=200.0), B.gradient(1), B.delay(5.0), B.adc("e"),
            B.delay(300.0), B.crush()]
    red, mat = periodic.steady_state([B.repeat(body, 1)], spm), periodic.steady_state([B.repeat(body, 1)], spm, kmax=2, method="matrix")
    assert np.abs(red.signal - mat.signal).max() < 1e-12 and np.abs(red.state.zb - mat.state.zb[1:4]).max() < 1e-12


def test_spins_and_columns():
    sp = O.grid([dict(t1_ms=800.0, t2_ms=60.0, adc_um2_ms=1.0), dict(t1_ms=1500.0, t2_ms=90.0, adc_um2_ms=2.0)], zeta=[-0.5, 0.0, 0.5], b1=[0.9, 1.1])
    assert sp.shape == (2, 3, 2) and sp.n == 12 and list(sp.column[:6]) == [0, 0, 1, 1, 2, 2]
    assert O.take(sp, np.arange(4)).n == 4
    for bad in (lambda: O.spins(-1.0, 50.0), lambda: O.spins(1000.0, 50.0, pool="mt"), lambda: O.spins(1000.0, 50.0, pool="other"),
                lambda: O.grid([dict(t1_ms=1.0, t2_ms=1.0, adc_um2_ms=1.0), dict(t1_ms=1.0, t2_ms=1.0, f_b=0.1)]),
                lambda: O.lineshape("superlorentzian", 0.0, 0.012), lambda: O.lineshape("other", 1.0, 0.012)):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(ValueError):          # an rf block with 2 slice columns asked for the third
        epg.run([B.rf(ck=[(1, 0, 0, 0), (1, 0, 0, 0)])], sp)
    z2, w2 = O.slice_columns(4.0)
    z1, w1 = O.slice_columns(4.0, two_sided=False)
    assert z2.size == 2 * z1.size - 1 == 169 and w2.sum() == pytest.approx(w1.sum()) and z1.size == 85
    x, w = O.lorentzian_spread(1001)
    assert w.sum() == pytest.approx(1.0) and x[500] == pytest.approx(0.0, abs=1e-12)
    for _ in range(3):                       # the closed 2 x 2 exponential against scipy, real and complex, also nearly degenerate
        import scipy.linalg
        rng = np.random.default_rng(5)
        m = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)) * (_ > 0)
        if _ == 2:
            m = np.array([[-1.0, 1e-9], [1e-9, -1.0 + 1e-9]], complex)
        assert np.abs(np.array(O.exp2(m[0, 0], m[0, 1], m[1, 0], m[1, 1], 0.7)).reshape(2, 2) - scipy.linalg.expm(0.7 * m)).max() < 1e-12


def test_a_designed_pulse_is_a_block():
    """A pulse of the design layer as slice columns, and the same pulse as the console's selective block. The design
    layer takes the thickness its gradient spans with the nominal bandwidth of the pulse; the console's slice is the
    width at half maximum of the profile (engine 0.3.0), ``slice_calibration`` nominal bandwidths wide."""
    from mrgym.design import pulses
    tbw, flip, thick = 4.0, 35.0, 5.0
    zeta, w = O.slice_columns(tbw)
    assert O.slice_calibration(tbw) == pytest.approx(0.994, abs=1e-3) and O.slice_calibration(2.0) == pytest.approx(1.197, abs=1e-3)
    pulse = pulses.sinc(flip, 2.0, tbw, thick / O.slice_calibration(tbw), n=128)
    sp = O.grid([dict(t1_ms=900.0, t2_ms=70.0)], zeta)
    a = periodic.steady_state([B.repeat([B.rf_designed(pulse, zeta * thick), B.adc("e"), B.gradient(1, ms=12.0)], 1, spoil_inc_deg=117.0)], sp).signal[0]
    b = periodic.steady_state(L.spgr(12.0, flip, selective=B.Selective(tbw)), sp).signal[0]
    assert np.abs(a - b).max() < 1e-12 and abs(np.sum(w * a)) > 0.01


def test_a_sample_away_from_zero_dephasing_in_an_rf_spoiled_repeat():
    """Laboratory frame and pulse frame return the same samples wherever the adc sits; in the periodic state the
    magnitude of such a sample repeats and its phase advances by +phi C / n per repetition."""
    sp = O.spins([60.0, 45.0], [25.0, 15.0])
    inner = B.repeat([B.gradient(1, ms=1.0), B.adc("in")], 2)
    for body, c_over_n in (([B.rf(30), B.gradient(2, ms=2.0), B.adc("off"), B.gradient(-1, ms=2.0)], 2.0),
                           ([B.rf(40), inner, B.gradient(-1, ms=1.0)], None)):
        rep = [B.repeat(body, 9, phase_inc_deg=20.0, spoil_inc_deg=50.0)]
        lab, fr = epg.run(rep, sp), epg.run(rep, sp, frame=True)
        assert np.abs(lab.signal).max() > 0.02 and np.abs(lab.signal - fr.signal).max() < 1e-12
        if c_over_n is not None:
            fixed = periodic.steady_state(rep, sp, kmax=40)
            assert not fixed.program.ops[2].at_zero
            late = epg.run(rep, sp, kmax=40, repetitions=400, keep_state=False).signal[-2:]
            assert np.abs(np.abs(late) - np.abs(fixed.signal[0])).max() < 1e-10
            assert np.abs(late[1] / late[0] - np.exp(1j * np.radians(50.0) * c_over_n)).max() < 1e-9


@pytest.mark.parametrize("pool", [None, "mt", "cx"])
def test_diffusion_shape_moments_and_two_axes_in_stepping_and_in_the_matrix_method(pool):
    """Diffusion with a shaped lobe, a second gradient axis and a truncated lattice: the phase graph stepped to
    convergence and the matrix method give the same periodic samples, also with two pools."""
    kw = dict() if pool is None else dict(pool=pool, f_b=0.12, k_ab_per_s=40.0, t1b_ms=50.0, t2b_ms=0.012 if pool == "mt" else 15.0, lineshape="gaussian",
                                          shift_b_hz=80.0)
    sp = O.spins([60.0, 40.0], [25.0, 12.0], adc_um2_ms=[1.0, 2.5], df_hz=[5.0, -20.0], **kw)
    body = [B.rf(35, 0, energy_ut2_ms=300.0), B.adc("a"), B.delay(0.7), B.rf(20, 70), B.adc("b"), B.gradient((2, 0), ms=2.0, shape=(0.4, 0.22)),
            B.gradient((-1, 1), ms=1.5), B.delay(0.5)]
    seq = B.sequence([B.repeat(body, 1, phase_inc_deg=25.0, spoil_inc_deg=117.0)], cycle_mm=(0.03, 0.05))
    kmax = (9, 7)
    fixed = periodic.steady_state(seq, sp, kmax=kmax, method="matrix")
    stepped = epg.run(seq, sp, kmax=kmax, frame=True, repetitions=260, keep_state=False).signal[-2:]
    assert np.abs(stepped - fixed.signal).max() < 1e-11
    plain = periodic.steady_state(B.sequence(seq.blocks), sp, kmax=kmax, method="matrix").signal
    assert np.abs(plain - fixed.signal).max() > 1e-3                        # diffusion matters in this test
    with pytest.raises(ValueError):                                         # the grid integration takes constant gradients and one pool
        bloch.run(seq, O.spins(60.0, 25.0, adc_um2_ms=1.0), diffusion_substeps=4)
