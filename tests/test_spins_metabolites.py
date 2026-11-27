"""The metabolite basis: what it holds, and that the engine's fat spectrum is in it unchanged."""
import numpy as np
import pytest

from mrgym.scanner import signal, tissue
from mrgym.spins import metabolites, operators, sequences


def test_the_basis_holds_the_required_entries_with_their_proton_counts():
    want = dict(water=2, lactate=4, alanine=4, naa=7, creatine=5, choline=13, gaba=6, gaba_near2013=6, glutamate=5, glutamine=5,
                myo_inositol=6, lipid_pair=5, triglyceride=1)
    assert {m.name: round(m.protons, 12) for m in metabolites.BASIS} == want
    assert "gaba_near2013" not in metabolites.DEFAULT and len(metabolites.basis()) == len(want) - 1
    assert metabolites.get("naa").groups == ("naa_acetyl", "naa_aspartyl")
    assert [m.name for m in metabolites.basis(("naa", "water"))] == ["naa", "water"]
    with pytest.raises(KeyError):
        metabolites.get("unobtainium")


def test_every_entry_has_a_source_and_a_tag():
    for m in metabolites.BASIS:
        assert m.source and m.tag in ("F", "E", "M"), m.name
        for s in m.systems:
            assert all(0.5 < ppm < 8.0 for ppm in s.shifts_ppm) and all(abs(j) < 20 for _, _, j in s.j_hz), m.name
    assert [m.name for m in metabolites.BASIS if m.tag == "M"] == ["lipid_pair"]
    assert any("lipid_pair" in item for item in metabolites.NOT_VERIFIED)


def test_the_two_gaba_sets_describe_the_same_molecule():
    a, b = metabolites.get("gaba").systems[0], metabolites.get("gaba_near2013").systems[0]
    assert sorted(np.round(a.shifts_ppm, 2)) == sorted(np.round(b.shifts_ppm, 2))
    assert len(a.j_hz) == len(b.j_hz) == 11
    # no coupling between the 2.28 ppm and the 3.01 ppm protons (four bonds apart) in either set
    for s in (a, b):
        assert not [p for p in s.j_hz if {round(s.shifts_ppm[p[0]], 1), round(s.shifts_ppm[p[1]], 1)} == {2.3, 3.0}]


def test_the_engine_fat_spectrum_is_the_triglyceride_entry():
    assert metabolites.ENGINE_FAT_PEAKS == tissue.FAT_PEAKS and metabolites.WATER_PPM + tissue.FAT_PPM == pytest.approx(1.3)
    t_ms = 0.9 * np.arange(40)
    for b0 in (1.5, 3.0):
        fid = sequences.simulate(metabolites.get("triglyceride"), sequences.pulse_acquire(), b0_t=b0).fid(40, 0.9)
        assert np.abs(fid - signal.fat_spectrum(1e-3 * t_ms, b0, tissue.FAT_PEAKS)).max() < 1e-12
    # in-phase and opposed-phase times of the main peak follow from the same numbers
    df = abs(tissue.FAT_PPM) * operators.larmor_hz_per_ppm(3.0)
    main = sequences.simulate(metabolites.get("triglyceride"), sequences.pulse_acquire()).window(-df - 1, -df + 1)
    assert main.fid(1, 1.0, t0_ms=1e3 / df)[0].real == pytest.approx(main.total().real)
    assert main.fid(1, 1.0, t0_ms=0.5e3 / df)[0].real == pytest.approx(-main.total().real)


def test_lactate_and_lipid_overlap_at_1_3_ppm_and_differ_in_their_echo_time_behaviour():
    """What an echo-time task rests on: at TE = 1 / J the lactate doublet is inverted, the fat methylene is not."""
    hz = operators.larmor_hz_per_ppm(3.0)
    band = ((1.3 - 0.15 - 4.7) * hz, (1.3 + 0.15 - 4.7) * hz)
    te = 1e3 / 6.933
    lac = [sequences.simulate(metabolites.get("lactate"), sequences.press(t)).window(*band).total().real for t in (30.0, te, 2 * te)]
    fat = [sequences.simulate(metabolites.get("triglyceride"), sequences.press(t)).window(*band).total().real for t in (30.0, te, 2 * te)]
    assert lac[1] == pytest.approx(-3.0, abs=0.02) and lac[2] == pytest.approx(3.0, abs=0.05) and lac[0] > 2.0
    assert fat == pytest.approx([0.693 / 0.999] * 3, abs=1e-9)                 # the six amplitudes of the engine sum to 0.999


def test_table_lists_every_network():
    table = metabolites.table_markdown().splitlines()
    assert len(table) == 2 + sum(len(m.systems) for m in metabolites.BASIS)
    assert any("6.933" in row and "lactate" in row for row in table)


def test_the_reviewed_table_is_pinned():
    """Every shift, coupling and pair of the basis was compared with the fetched copies by a second reader; a change
    of any number or text of the table changes this digest and must be made on purpose."""
    import hashlib
    assert hashlib.sha256(metabolites.table_markdown().encode()).hexdigest() == "690a06f6fcc788a34ffa50b27bb3dc8493a01f931918d535cb62edd9e79df41e"
    assert metabolites.WATER_3T == dict(gm=dict(water=0.78, t1_ms=1331.0, t2_ms=110.0), wm=dict(water=0.65, t1_ms=832.0, t2_ms=79.2),
                                        csf=dict(water=0.97, t1_ms=3817.0, t2_ms=503.0))
