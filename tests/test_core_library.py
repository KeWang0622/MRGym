"""The library: every sequence runs in the three evaluators, and the served ones give the engine's numbers."""
import math

import numpy as np
import pytest

from mrgym.core import bloch, epg, periodic
from mrgym.core import blocks as B
from mrgym.core import library as L
from mrgym.core import operators as O
from mrgym.core.validate import _library_cases
from mrgym.scanner import console, kspace
from mrgym.scanner import signal as S
from mrgym.scanner import world as W

SP = O.spins([150.0, 90.0], [40.0, 25.0], adc_um2_ms=1.0)        # short T1: stepping reaches the periodic state quickly


@pytest.mark.parametrize("name,fn,args", _library_cases(), ids=[c[0] for c in _library_cases()])
def test_every_library_sequence_steps_to_its_periodic_state(name, fn, args):
    seq = fn(*args)
    prog, reps = periodic.period(seq)
    assert reps is not None and prog.period_ms > 0
    sp = SP if "mt" not in name else O.spins(SP.t1_ms, SP.t2_ms, pool="mt", f_b=0.1, k_ab_per_s=4.0, t1b_ms=100.0)
    fixed = periodic.steady_state(seq, sp, kmax=None if O.auto_kmax(prog, sp) is not None or periodic._reducible(prog) else 16)
    assert fixed.labels == prog.labels and np.all(np.isfinite(fixed.signal)) and np.abs(fixed.signal).max() > 1e-4
    n_per = int(math.ceil(25 * 150.0 / prog.period_ms)) + 2
    kmax = fixed.state.kmax if fixed.state is not None else O.auto_kmax(prog, sp)
    stepped = epg.run(seq, sp, kmax=kmax, frame=True, repetitions=reps * n_per, keep_state=False)
    assert np.abs(stepped.signal[-fixed.signal.shape[0]:] - fixed.signal).max() < 1e-9, name


def test_one_shot_of_every_new_sequence_in_the_ensemble():
    sp = O.spins([900.0, 300.0], [80.0, 30.0], df_hz=[7.0, -15.0])
    for name, fn in L.NEW.items():
        args = dict(fse=(3000.0, 8.0, 8, 140.0), mese=(2000.0, 10.0, 6, 160.0), megre=(40.0, 20.0, (3.0, 9.0)), spgr_lobe=(10.0, 20.0, 3.0),
                    dess=(15.0, 30.0), mprage=(300.0, 6, 8.0, 12.0, 200.0), t2prep=(1000.0, 40.0), satrec=(800.0, 300.0),
                    dwse=(3000.0, 80.0, 30.0, 20.0, 40.0, 1.0), steam=(3000.0, 40.0, 100.0, 30.0, 10.0, 1.0), mt_spgr=(30.0, 10.0, 2000.0, 2000.0))[name]
        seq = fn(*args, repetitions=3)
        cols = sp if name != "mt_spgr" else O.spins(sp.t1_ms, sp.t2_ms, pool="mt", f_b=0.1, k_ab_per_s=4.0, t1b_ms=900.0)
        e, r = epg.run(seq, cols), bloch.run(seq, cols)
        assert np.abs(e.signal - r.signal).max() < 1e-10, name


@pytest.mark.parametrize("proto", [
    dict(sequence="spgr", tr_ms=9.0, flip_deg=25.0, excitation="nonselective"), dict(sequence="afi", tr_ms=20.0, flip_deg=55.0, excitation="nonselective"),
    dict(sequence="epi", tr_ms=1500.0, fat_sat=False, excitation="nonselective"), dict(sequence="se", tr_ms=450.0, excitation="nonselective"),
    dict(sequence="ir", tr_ms=1800.0, ti_ms=300.0, excitation="nonselective"), dict(sequence="bssfp", tr_ms=6.0, flip_deg=45.0, phase_cycle_deg=180.0, excitation="nonselective"),
    dict(sequence="bssfp", tr_ms=6.0, flip_deg=45.0, phase_cycle_deg=75.0, excitation="nonselective")],
    ids=["spgr", "afi", "epi", "se", "ir", "bssfp", "bssfp-cycle-75"])
def test_served_block_lists_give_the_engine_amplitudes_of_a_world(proto):
    """Hard pulse, one voxel per material of a small phantom world: the block list of the realised protocol against
    ``Engine.amplitudes`` (which interpolates over the flip angle: 1e-6)."""
    w = W.make_world("7c" * 16, dict(kind="qa"))
    eng = kspace.Engine(w)
    rp = console.plan(dict(proto, matrix=64), w["console"])["realised"]
    seq, labels = L.served(rp, inv_eff=w["system"]["inv_eff"])
    for c, name in enumerate(w["names"]):
        cls, sup = w["tissues"][name], eng.sup[c]
        i = int(np.flatnonzero(w["weights"][c][sup] > 0.5)[7])
        served = [v[i] / (w["weights"][c][sup][i] * cls["pd"]) for v in eng.amplitudes(rp, c)]
        sp = O.spins(cls["t1_ms"], cls["t2_ms"], b1=eng.b1[sup][i], adc_um2_ms=cls["adc_um2_ms"], df_hz=eng.df[sup][i] if rp["sequence"] == "bssfp" else 0.0)
        got = periodic.steady_state(seq, sp)
        decay = math.exp(-rp["te_ms"] / cls["t2_ms"]) if rp["sequence"] in ("se", "ir", "bssfp") else 1.0
        for lab, ref in zip(labels, served):
            assert abs(got.of(lab)[0, 0] / decay - ref) < 1e-6 * max(abs(ref), 1e-3), (name, lab)


def test_the_idealised_blocks_are_the_engine_model():
    """Crushed refocusing and ideal spoiling are the engine's idealisations; with real gradients the signal differs."""
    sp = O.spins(900.0, 80.0, b1=0.85)
    ideal = periodic.steady_state(L.se(400.0, 20.0), sp).signal[0, 0]
    assert ideal == pytest.approx(S.se(0.85 * math.pi / 2, 400.0, 20.0, 900.0) * math.exp(-20.0 / 80.0), rel=1e-12)
    real = epg.run(L.se(400.0, 20.0, crusher_cycles=1, repetitions=40), sp, kmax=40).signal[-1, 0]
    assert abs(real / ideal - 1) > 0.02                         # the stimulated echoes of earlier repetitions (limits.md, 8a)
    with pytest.raises(ValueError):
        L.spgr(10.0, 20.0, te_ms=3.0)                           # the RF-spoiled served model samples right after the pulse
    with pytest.raises(ValueError):
        L.served(dict(sequence="radial"))
    for bad in (lambda: L.afi(20.0, 1.5, 60.0), lambda: L.fse(100.0, 10.0, 16), lambda: L.megre(20.0, 10.0, (5.0, 4.0)),
                lambda: L.spgr_lobe(5.0, 10.0, 4.5, spoiler_ms=1.0), lambda: L.dwse(3000.0, 40.0, 30.0, 20.0, 40.0), lambda: L.steam(100.0, 80.0, 50.0, 30.0, 10.0),
                lambda: L.diffusion_lobe(30.0, 5.0, 6.0), lambda: L.diffusion_waveform("gre", 80.0, 30.0, 20.0)):
        with pytest.raises(ValueError):
            bad()


def test_selective_models_and_the_balanced_slice_gradient():
    """For an unbalanced sequence the rotation of a column equals a hard pulse of its flip angle times its phase; a
    balanced sequence needs half of the rewinding area before the pulse."""
    cls = dict(t1_ms=700.0, t2_ms=60.0)
    zeta, w = O.slice_columns(4.0)
    sp = O.grid([cls], zeta)
    rot = periodic.steady_state(L.spgr(12.0, 35.0, selective=B.Selective(4.0)), sp).signal[0]
    sel = B.Selective(4.0, model="flip")
    flp = periodic.steady_state(L.spgr(12.0, 35.0, selective=sel), sp).signal[0] * O.column_phase(sel, 35.0, sp)
    assert np.abs(rot - flp).max() < 1e-12
    assert L.bssfp(6.0, 40.0, selective=B.Selective(4.0)).blocks[0].body[0].selective.prephase == 0.5
    # the balanced rotation is symmetric: its axis has no y component, so b is imaginary at every column
    a, b = O.selective_rotation(B.Selective(4.0, prephase=0.5), math.radians(60.0), zeta)
    assert np.abs(b.real).max() < 1e-12 and np.abs(a.imag).max() > 0.05


def test_diffusion_programs():
    from mrgym.design import gradients
    lobe = L.diffusion_lobe(30.0, 20.0, 1.0)
    assert lobe["duration"] == 21.0 and lobe["area"] == pytest.approx(600.0)
    seq = L.dwse(3000.0, 80.0, 30.0, 20.0, 40.0, 1.0)
    t, g, refocus = L.diffusion_waveform("se", 80.0, 30.0, 20.0, 40.0, ramp_ms=1.0)
    b = gradients.b_value(t, g, refocus)
    sig = epg.run(seq, O.spins(1e15, 1e15, adc_um2_ms=[0.0, 1.0, 2.0])).signal[0].real
    assert sig[0] == pytest.approx(1.0) and np.allclose(-np.log(sig[1:]) / np.array([1e-3, 2e-3]), b, rtol=1e-9)
