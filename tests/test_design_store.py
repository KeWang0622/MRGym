"""The design store: content-derived ids, immutability, where it writes, and the length of what an agent reads."""
import json
import os

import numpy as np
import pytest

from mrgym.design import gradients, pulses, sequence, store

PROGRAM = dict(x=[(0, 0), (0.3, 30), (1.3, 30), (1.6, 0)], z=[(0.5, 0), (0.6, 45), (0.9, 0)])


def designs():
    base = pulses.sinc(90.0, 2.0, 4.0, 5.0)
    return [base, pulses.sinc(30.0, 1.0, 2.0), pulses.slr(180.0, 4.0, 8.0, 5.0, kind="refocusing"), pulses.sech(12.0, 10.0, 1.06, 5.0, 5.0),
            pulses.control_points((np.arange(64) + 1j * np.arange(64)[::-1]) * 0.1, 2.0, 3000.0, 5.0, interp="cubic"),
            pulses.verse(base, 6.0, 20.0, 120.0), gradients.program(**PROGRAM),
            gradients.program(x=[(i * 0.1, (-1) ** i * 20.0) for i in range(32)], y=[(0, 0), (1, 1)], z=[(0, 0), (2, 2)]),
            [sequence.rf(90, 90)] + sequence.repeat([sequence.delay(5), sequence.gradient(1), sequence.rf(150), sequence.gradient(1),
                                                    sequence.delay(5), sequence.readout()], 200)]


def test_ids_come_from_the_content():
    st = store.DesignStore()
    ids = [st.put(d) for d in designs()]
    assert len(set(ids)) == len(ids) and all(len(i) == store.ID_LENGTH and int(i, 16) >= 0 for i in ids)
    assert [st.put(d) for d in designs()] == ids and len(st) == len(ids)            # the same designs again: nothing new
    assert st.put(gradients.program(**PROGRAM)) == "7b92b0354a8c"                  # pinned: the canonical serialisation
    other = dict(PROGRAM, z=[(0.5, 0), (0.6, 45.000001), (0.9, 0)])
    assert st.put(gradients.program(**other)) != "7b92b0354a8c"
    assert st.put(pulses.sinc(90.0, 2.0, 4.0, 5.0), tr_ms=500.0) == ids[0]          # the record is not part of the identity


def test_the_last_bits_of_a_float_do_not_change_the_id():
    st = store.DesignStore()
    p = pulses.sinc(90.0, 2.0, 4.0, 5.0)
    assert st.put(p) == st.put(p.replace(b1_ut=p.b1_ut * (1 + 1e-15)))
    assert st.put(p) != st.put(p.replace(b1_ut=p.b1_ut * (1 + 1e-9)))
    assert store.canonical(0.1 + 0.2) == 0.3 and store.canonical(3.0) == 3 and store.canonical(1 + 2j) == [1, 2]
    assert store.serialise(dict(b=1, a=[1.5, None])) == '{"a":[1.5,null],"b":1}'
    with pytest.raises(TypeError):
        store.canonical(object())


def test_records_are_immutable_and_hold_only_what_is_documented():
    st = store.DesignStore()
    ident = st.put(pulses.sinc(90.0, 2.0, 4.0, 5.0), tr_ms=500.0)
    rec = st.get(ident)
    assert set(rec) == {"id", "kind", "design", "call", "record"} and rec["id"] == ident and rec["kind"] == "pulse"
    assert rec["call"]["generator"] == "sinc" and rec["record"]["tr_ms"] == 500.0
    assert {"duration_ms", "peak_b1_ut", "energy_ut2_ms", "limits"} <= set(rec["record"]) and rec["record"]["limits"]["ok"] is True
    rec["record"]["peak_b1_ut"] = 0.0
    rec["design"]["b1_ut"][0] = [9, 9]
    assert st.get(ident)["record"]["peak_b1_ut"] > 11 and st.get(ident)["design"]["b1_ut"][0] != [9, 9]
    st.put(pulses.sinc(90.0, 2.0, 4.0, 5.0), tr_ms=50.0)                              # same design, other TR: the first record stays
    assert st.get(ident)["record"]["tr_ms"] == 500.0
    with pytest.raises(KeyError):
        st.get("0" * 12)
    with pytest.raises(TypeError):
        st.put("a pulse")
    with pytest.raises(ValueError):
        st.pulse(st.put(gradients.program(**PROGRAM)))


def test_a_stored_pulse_comes_back():
    st = store.DesignStore()
    p = pulses.verse(pulses.sinc(90.0, 2.0, 4.0, 5.0), 6.0, 20.0)
    q = st.pulse(st.put(p))
    assert np.abs(q.b1_ut - p.b1_ut).max() < 1e-9 and np.abs(q.dt_ms - p.dt_ms).max() < 1e-12 and q.kind == p.kind
    assert q.call["generator"] == "verse" and st.put(q) == st.put(p)


def test_two_stores_share_nothing_and_nothing_is_written_without_a_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a, b = store.DesignStore(), store.DesignStore()
    ident = a.put(pulses.sinc(90.0, 2.0, 4.0, 5.0))
    assert ident in a and ident not in b and len(b) == 0
    assert os.listdir(tmp_path) == []


def test_a_directory_holds_one_file_per_design_and_is_checked_on_reading(tmp_path):
    where = tmp_path / "designs"
    st = store.DesignStore(str(where))
    ids = sorted(st.put(d) for d in designs()[:3])
    assert sorted(os.listdir(where)) == [i + ".json" for i in ids] and os.listdir(tmp_path) == ["designs"]
    again = store.DesignStore(str(where))
    assert again.ids() == ids and again.summary(ids[0]) == st.summary(ids[0])
    path = where / (ids[0] + ".json")
    rec = json.loads(path.read_text())
    rec["design"]["dt_ms"][0] = 0.02
    path.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="changed"):
        store.DesignStore(str(where))


def test_summaries_are_short_and_say_whether_the_limits_hold():
    st = store.DesignStore()
    for d in designs():
        text = st.summary(st.put(d, tr_ms=500.0) if isinstance(d, pulses.Pulse) else st.put(d))
        assert store.count_words(text) <= store.MAX_WORDS and len(text) <= store.MAX_CHARS, text
        assert text.startswith("design ") and ("within limits" in text or "VIOLATES" in text)
    assert "VIOLATES B1_PEAK_LIMIT" in st.summary(st.put(pulses.slr(180.0, 4.0, 8.0, 5.0, kind="refocusing")))
    assert "VIOLATES GRADIENT_LIMIT, SLEW_LIMIT, STIMULATION_LIMIT" in st.summary(st.put(gradients.program(**PROGRAM)))
    assert "sech(" in st.summary(st.put(pulses.sech(12.0, 10.0, 1.06, 5.0, 5.0)))
    assert store.count_words("a  b\nc") == 3


def test_the_limits_are_those_of_the_store():
    strict = store.DesignStore(hardware=gradients.Hardware(b1_peak_ut=10.0))
    assert "VIOLATES B1_PEAK_LIMIT" in strict.summary(strict.put(pulses.sinc(90.0, 2.0, 4.0, 5.0)))


def test_rounding_noise_around_zero_does_not_change_the_id():
    """An array is rounded on the scale of its largest entry: an imaginary part of 1e-16 next to a real part of 10
    is noise."""
    st = store.DesignStore()
    p = pulses.slr(90.0, 2.56, 6.0, 5.0, n=256)
    assert st.put(p) == st.put(p.replace(b1_ut=p.b1_ut + 1e-18j)) == st.put(p.replace(b1_ut=p.b1_ut.real + 0j))
    assert st.put(p) != st.put(p.replace(b1_ut=p.b1_ut + 1e-9))


def test_single_rotations_and_designed_pulses_in_a_block_list_are_stored_and_counted():
    st = store.DesignStore()
    one = st.put([sequence.rf_rotation(np.cos(0.5), -1j * np.sin(0.5)), sequence.readout()])
    rec = st.get(one)["record"]
    assert rec["rf_pulses"] == 1 and abs(rec["peak_flip_deg"] - np.degrees(1.0)) < 1e-3
    two = st.put([sequence.rf_pulse(pulses.sinc(90.0, 2.0, 4.0, 5.0)), sequence.readout()])
    assert abs(st.get(two)["record"]["peak_flip_deg"] - 90.0) < 1e-3 and "largest flip 90 deg" in st.summary(two)
    cols = st.put([sequence.rf_pulse(pulses.sinc(90.0, 2.0, 4.0, 5.0), np.linspace(-5, 5, 11)), sequence.readout()])
    assert st.get(cols)["record"]["peak_flip_deg"] > 85.0


def test_summary_survives_odd_designs_and_never_cuts_the_verdict():
    st = store.DesignStore()
    p = pulses.sinc(90.0, 1.0, 8.0, 1.0)                                   # over the gradient and the B1 limit
    for _ in range(9):
        p = pulses.verse(p, 60.0, 400.0)
    text = st.summary(st.put(p))
    assert len(text) <= store.MAX_CHARS and store.count_words(text) <= store.MAX_WORDS and text.endswith(".") and "VIOLATES" in text
    assert text.rstrip(".").split("VIOLATES ")[1].split(", ") == st.get(st.put(p))["record"]["limits"]["violations"]
    bare = pulses.Pulse(np.ones(10) * 3.0, 0.01, 5.0, thickness_mm=4.0)   # a thickness, no stated bandwidth
    assert "bandwidth not stated" in st.summary(st.put(bare))
    assert "0.01 ms" in st.summary(st.put(dict(x=[(0, 0)])))               # a program of one vertex is one raster step of nothing
