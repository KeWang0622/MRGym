"""Saturation recovery, and a preparation in front of the fast spin echo, on a console that lists them.

Textbook statements:
  satrec   After a complete saturation the longitudinal magnetisation regrows from zero: the signal of a pulse of angle a
           a time TS later is sin a (1 - exp(-TS / T1)) exp(-TE / T2*), whatever TR and whatever a (nothing of the last
           repetition is left). T1 follows from a series of TS without any assumption about the flip angle.
  fse      With an inversion a time TI before the excitation, and exact pulses, every echo is that of the train times
           Mz(TI) = 1 - (1 + q Mz_before) exp(-TI / T1), Mz_before being what the train and the rest of the repetition left
           (q: inversion efficiency). The signal is signed and crosses zero at the null: a fluid is nulled at
           TI = T1 ln(2 / (1 + exp(-(TR - train) / T1) ...)), which for TR >> T1 is T1 ln 2 (FLAIR).
           With a saturation instead, Mz(TS) = 1 - exp(-TS / T1) whatever the train left.
"""
import math

import numpy as np
import pytest

from mrgym.scanner import console, kspace, served
from mrgym.scanner import world as W

HARD = dict(excitation="nonselective")
TIS = dict(a=dict(pd=1.0, t1_ms=900.0, t2_ms=80.0, t2s_ms=60.0, t2p_ms=240.0, adc_um2_ms=0.0, fat=False),
           b=dict(pd=0.9, t1_ms=2500.0, t2_ms=60.0, t2s_ms=40.0, t2p_ms=120.0, adc_um2_ms=0.0, fat=False))


def _disc(r, cy=0.0, cx=0.0):
    return lambda y, x: np.hypot(y - cy, x - cx) - r


@pytest.fixture(scope="module")
def world():
    w = W.flat_world(W.shapes_phantom([("a", _disc(30.0, 0.0, -40.0)), ("b", _disc(30.0, 0.0, 40.0))], b0_t=3.0, n=192, tissues=TIS))
    w["console"].update(sequences="all", min_pixel_mm=0.5)
    return w


def _unit(eng, rp, name):
    c = eng.w["names"].index(name)
    scale = eng.w["weights"][c][eng.sup[c]] * eng.w["tissues"][name]["pd"]
    i = int(np.argmax(scale))
    return np.array([a[i] / scale[i] for a in eng.amplitudes(rp, c)])


def test_saturation_recovery_does_not_depend_on_tr_or_on_what_came_before(world):
    eng = kspace.Engine(world)
    for ts in (50.0, 400.0, 1500.0):
        for tr, flip in ((ts + 40.0, 90.0), (ts + 900.0, 35.0)):
            rp = console.plan(dict(sequence="satrec", tr_ms=tr, ts_ms=ts, flip_deg=flip, matrix=32, **HARD), world["console"])
            assert rp["accept"], rp["violations"]
            for name in ("a", "b"):
                got = _unit(eng, rp["realised"], name)[0]
                assert abs(got - math.sin(math.radians(flip)) * (1 - math.exp(-ts / TIS[name]["t1_ms"]))) < 1e-12


def test_t1_from_a_saturation_recovery_series_of_images(world):
    eng = kspace.Engine(world)
    ts = np.array([100.0, 300.0, 700.0, 1500.0, 4000.0, 12000.0])
    means = []
    for t in ts:
        out = eng.scan(dict(sequence="satrec", tr_ms=float(t) + 30.0, ts_ms=float(t), matrix=32))        # slice-selective, the default
        rp = out["plan"]["realised"]
        yy = (np.arange(rp["matrix_pe"]) - rp["matrix_pe"] // 2) * rp["fov_pe_mm"] / rp["matrix_pe"]
        xx = (np.arange(rp["matrix"]) - rp["matrix"] // 2) * rp["fov_mm"] / rp["matrix"]
        means.append([abs(out["images"][0][np.hypot(yy[:, None], xx[None, :] - cx) < 15].mean()) for cx in (-40.0, 40.0)])
    means = np.array(means)
    for k, name in enumerate(("a", "b")):
        ratio = means[:-1, k] / means[-1, k] * (1 - np.exp(-ts[-1] / TIS[name]["t1_ms"]))      # the last point is nearly recovered
        t1 = -ts[:-1] / np.log(1 - ratio)
        assert np.abs(t1 / TIS[name]["t1_ms"] - 1).max() < 2e-3       # the region mean holds the truncation ringing of the other disc (6e-4 here)


def test_inversion_in_front_of_an_echo_train_exact_pulses(world):
    eng = kspace.Engine(world)
    q = world["system"]["inv_eff"]
    for tr, ti, n, esp in ((6000.0, 1700.0, 8, 10.0), (2500.0, 300.0, 4, 12.0)):
        rp = console.plan(dict(sequence="fse", tr_ms=tr, etl=n, echo_spacing_ms=esp, prep="inversion", ti_ms=ti, matrix=32, **HARD), world["console"])
        assert rp["accept"], rp["violations"]
        rp = rp["realised"]
        for name in ("a", "b"):
            t1, t2 = TIS[name]["t1_ms"], TIS[name]["t2_ms"]
            # one number through the repetition: 0 after the 90, inverted by every 180, recovery until the inversion
            u = 1.0                                                     # u = 1 - Mz
            u = 2 - u * math.exp(-esp / (2 * t1))
            for _ in range(n - 1):
                u = 2 - u * math.exp(-esp / t1)
            u = u * math.exp(-(tr - ti - (n - 0.5) * esp) / t1)         # at the inversion
            mz = 1 - (1 + q * (1 - u)) * math.exp(-ti / t1)
            want = mz * np.exp(-np.arange(1, n + 1) * esp / t2)
            assert np.abs(_unit(eng, rp, name) - want).max() < 1e-12


def test_a_fluid_is_nulled_at_t1_ln2_when_tr_is_long(world):
    eng = kspace.Engine(world)
    t1 = TIS["b"]["t1_ms"]
    lo = _unit(eng, console.plan(dict(sequence="fse", tr_ms=20000.0, etl=4, prep="inversion", ti_ms=round(t1 * math.log(2) - 40.0, 2), matrix=32, **HARD),
                                 world["console"])["realised"], "b")[0]
    hi = _unit(eng, console.plan(dict(sequence="fse", tr_ms=20000.0, etl=4, prep="inversion", ti_ms=round(t1 * math.log(2) + 40.0, 2), matrix=32, **HARD),
                                 world["console"])["realised"], "b")[0]
    assert lo.real < 0 < hi.real and abs(lo.real + hi.real) < 0.02 * abs(hi.real - lo.real)


def test_saturation_in_front_of_an_echo_train_forgets_the_last_repetition(world):
    eng = kspace.Engine(world)
    for tr in (700.0, 5000.0):
        rp = console.plan(dict(sequence="fse", tr_ms=tr, etl=4, echo_spacing_ms=10.0, prep="saturation", ti_ms=500.0, matrix=32, **HARD), world["console"])["realised"]
        for name in ("a", "b"):
            want = (1 - math.exp(-500.0 / TIS[name]["t1_ms"])) * np.exp(-np.arange(1, 5) * 10.0 / TIS[name]["t2_ms"])
            assert np.abs(_unit(eng, rp, name) - want).max() < 1e-12


def test_refusals_and_defaults_of_the_preparation(world):
    card = world["console"]
    assert console.plan(dict(sequence="fse", tr_ms=3000.0, etl=8), card)["realised"]["prep"] == "none"
    bad = console.plan(dict(sequence="fse", tr_ms=3000.0, etl=8, prep="inversion"), card)                 # needs its time
    assert not bad["accept"] and bad["violations"][0]["parameter"] == "ti_ms"
    bad = console.plan(dict(sequence="fse", tr_ms=3000.0, etl=8, ti_ms=500.0), card)                      # a time without a preparation
    assert not bad["accept"] and bad["violations"][0]["parameter"] == "ti_ms"
    bad = console.plan(dict(sequence="fse", tr_ms=600.0, etl=8, prep="inversion", ti_ms=560.0), card)     # does not fit into TR
    assert not bad["accept"] and bad["violations"][0]["code"] == "TR_BELOW_MINIMUM"
    bad = console.plan(dict(sequence="satrec", tr_ms=300.0, ts_ms=1.0), card)
    assert not bad["accept"] and bad["violations"][0]["parameter"] == "ts_ms" and bad["violations"][0]["minimum"] > 1.0
    assert bad["violations"][0]["code"] == "TI_NOT_REALISABLE"            # the code of a preparation time the console cannot realise (no new code)
    none = console.plan(dict(sequence="mese", tr_ms=1500.0, n_echoes=4), dict(card, system=dict(crusher_ms=0.0)))     # a console without crusher lobes
    assert none["accept"] and console.REGISTRY["mese"].blocks(none["realised"])[0].cycle_mm is None
    assert console.PROTOCOL_DEPENDENT == ("tr_ms", "te_ms", "ti_ms", "echo_spacing_ms")       # read by the tool layer of every console: unchanged
    ok = console.plan(dict(sequence="satrec", tr_ms=300.0, ts_ms=bad["violations"][0]["minimum"]), card)
    assert ok["accept"] and " TS " in ok["header"]
    assert console.plan(dict(sequence="satrec", tr_ms=300.0, ts_ms=100.0), dict(card, sequences=None))["violations"][0]["code"] == "UNKNOWN_SEQUENCE"


def test_two_pool_numbers_are_additive_sourced_and_read_by_no_existing_sequence():
    """Magnetisation transfer: bound-pool fraction, exchange rate and bound-pool T2 per tissue, from the table of one
    paper that was read in full. They are a table of their own: ``tissue.table`` and every served amplitude are what
    they were."""
    from mrgym.scanner import tissue
    mt = tissue.mt_table(3.0)
    assert set(mt) == {"wm", "gm", "muscle", "liver", "cord", "blood", "cartilage"} and set(tissue.MT_NONE) == {"csf", "bowel_fluid", "scalp_fat", "subcut_fat", "visceral_fat"}
    wm, gm = mt["wm"], mt["gm"]
    assert (wm["f_b"], wm["k_ab_per_s"], wm["t2b_ms"], wm["lineshape"]) == (0.139, pytest.approx(0.139 * 23.0), 0.010, "superlorentzian")
    assert gm["f_b"] == 0.050 and gm["k_ab_per_s"] == pytest.approx(0.050 * 40.0) and mt["blood"]["lineshape"] == "lorentzian"
    assert tissue.mt_table(1.5)["wm"]["f_b"] == 0.152 and "cartilage" not in tissue.mt_table(1.5)       # not measured at 1.5 T in the source
    assert all(row["tag"] == "F" and "Stanisz 2005" in row["source"] for row in mt.values())
    assert all(set(v) == {"pd", "t1_ms", "t2_ms", "t2s_ms", "t2p_ms", "adc_um2_ms", "fat"} for v in tissue.table(3.0).values())
    # the numbers go straight into the two-pool spins of the core: the saturation of a white-matter-like tissue is larger than that of grey matter
    from mrgym.core import library, operators, periodic
    ratio = {}
    for name in ("wm", "gm"):
        t = tissue.table(3.0)[name]
        sp = operators.spins(t["t1_ms"], t["t2_ms"], pool="mt", **{k: mt[name][k] for k in ("f_b", "k_ab_per_s", "t1b_ms", "t2b_ms")}, lineshape=mt[name]["lineshape"])
        on = periodic.steady_state(library.mt_spgr(30.0, 10.0, 3000.0, 2000.0), sp).signal[0, 0].real
        off = periodic.steady_state(library.mt_spgr(30.0, 10.0, 0.0, 2000.0), sp).signal[0, 0].real
        ratio[name] = 1 - on / off
    assert 0.05 < ratio["gm"] < ratio["wm"] < 0.6


def test_the_slice_integral_keeps_its_mirror_symmetry_with_gratings_and_loses_it_with_transverse_states():
    """The table integrates z >= 0 and takes the mirror column as the complex conjugate. That holds for what is served
    (the gratings along z carried, the transverse magnetisation dephased at the end of the repetition). It does NOT hold
    once transverse states are carried past a spoiler on the crusher axis (found by the independent review: for a
    fluid the two sides differ by a third of the echo), which is one reason they are not served on resonance only."""
    from mrgym.core import blocks as B
    from mrgym.core import operators as O
    from mrgym.core import periodic
    fluid = dict(pd=1.0, t1_ms=4300.0, t2_ms=2000.0, t2s_ms=1000.0, t2p_ms=2000.0, adc_um2_ms=3.0, fat=False)
    tis = dict(pd=1.0, t1_ms=900.0, t2_ms=80.0, t2s_ms=60.0, t2p_ms=240.0, adc_um2_ms=0.8, fat=False)
    w = W.flat_world(W.shapes_phantom([("f", _disc(30.0, 0.0, -40.0)), ("t", _disc(30.0, 0.0, 40.0))], b0_t=3.0, n=96, tissues=dict(f=fluid, t=tis)))
    w["console"].update(sequences="all", min_pixel_mm=0.5)
    eng = kspace.Engine(w)
    for prot in (dict(sequence="mese", tr_ms=500.0, n_echoes=4, echo_spacing_ms=10.0, matrix=32), dict(sequence="mese", tr_ms=2500.0, n_echoes=4, matrix=32),
                 dict(sequence="dwse", tr_ms=400.0, b_value=0.0, matrix=32)):
        rp = console.plan(prot, w["console"])["realised"]
        tab = served.table(eng, rp, nodes=1)
        tbw, window = eng.pulse(rp)
        seq, labels = console.REGISTRY[rp["sequence"]].blocks(rp, selective=B.Selective(tbw, window))
        zeta, wz = O.slice_columns(tbw, two_sided=True, window=window)
        half = zeta.size // 2
        for i, c in enumerate(tab["classes"]):
            cls = w["tissues"][w["names"][c]]
            sp = O.grid([served._tissue(cls)], zeta, tab["gn"])
            orders, transverse = served.carry_of(console.REGISTRY[rp["sequence"]].blocks(rp)[0], served._tissue(cls))
            assert transverse is False
            sig = periodic.steady_state(seq, sp, method="carry", carry=orders, carry_transverse=False).signal.reshape(len(labels), zeta.size)
            both = (sig * wz[None, :]).sum(1)                           # the integral over both sides of the slice
            assert np.abs(tab["a"][i, 0] - both).max() < 1e-6 * np.abs(both).max(), (prot["sequence"], w["names"][c])
            assert np.abs(sig[:, half + 5] - np.conj(sig[:, half - 5])).max() < 1e-12 * np.abs(sig).max()
    # the same train with the transverse states kept (the repetition without the ideal spoiler at its end): no mirror symmetry for the fluid
    rp = console.plan(dict(sequence="mese", tr_ms=500.0, n_echoes=4, echo_spacing_ms=10.0, matrix=32), w["console"])["realised"]
    seq, labels = console.REGISTRY["mese"].blocks(rp, selective=B.Selective(*eng.pulse(rp)))
    kept = B.sequence([B.repeat(seq.blocks[0].body[:-1], 1)], cycle_mm=seq.cycle_mm)
    sig = periodic.steady_state(kept, O.grid([served._tissue(fluid)], zeta, (1.0,)), method="carry", carry=(12,)).signal
    assert np.abs(sig[:, half + 5] - np.conj(sig[:, half - 5])).max() > 0.02 * np.abs(sig).max()
