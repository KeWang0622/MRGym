"""The vocabulary of the signal core: blocks are plain data, and a block list compiles to what it says."""
import json
import math

import numpy as np
import pytest

from mrgym.core import blocks as B
from mrgym.core import library as L
from mrgym.design import gradients, pulses
from mrgym.design import sequence as design_sequence


def test_blocks_are_checked():
    for bad in (lambda: B.delay(-1.0), lambda: B.gradient(1, ms=-1.0), lambda: B.gradient((1, 2, 3, 4)), lambda: B.gradient(1, shape=(0.9, 0.5)),
                lambda: B.rf(90, selective=B.Selective(4.0), ck=[(1, 0, 0, 0)]), lambda: B.rf(ck=[(1, 0, 0.5, 0)]), lambda: B.rf(30, offset_hz=1000.0),
                lambda: B.rf(energy_ut2_ms=-1.0), lambda: B.repeat([B.rf(flip_table="a")], 3), lambda: B.repeat([B.rf(90)], 2.5),
                lambda: B.repeat([B.rf(90)], 2, tables=dict(a=())), lambda: B.sequence([("rf", 90)]), lambda: B.sequence([B.rf(90)], cycle_mm=0.0),
                lambda: B.Selective(0.0), lambda: B.Selective(4.0, window="kaiser"), lambda: B.Selective(4.0, model="other"),
                lambda: B.delay(float("nan")), lambda: B.from_data(dict(kind="warp"))):
        with pytest.raises(ValueError):
            bad()


def test_blocks_are_plain_hashable_data():
    a = B.repeat([B.rf(30, 10, selective=B.Selective(4.0)), B.adc("x"), B.gradient((1, 0.5), ms=3.0)], 4, spoil_inc_deg=117.0, tables=dict(t=(1, 2)))
    b = B.repeat([B.rf(30, 10, selective=B.Selective(4.0)), B.adc("x"), B.gradient((1, 0.5), ms=3.0)], 4, spoil_inc_deg=117.0, tables=dict(t=(1, 2)))
    assert a == b and hash(a) == hash(b) and len({a, b}) == 1
    assert a != B.repeat(a.body, 5, spoil_inc_deg=117.0, tables=dict(t=(1, 2)))
    with pytest.raises(AttributeError):
        a.times = 7
    seq = B.sequence([a, B.crush(), B.mz(-0.9), B.spoil(), B.delay(2.0)], cycle_mm=(0.2, 0.3), name="x")
    text = json.dumps(B.to_data(seq))
    back = B.from_data(json.loads(text))
    assert back == seq and B.digest(back) == B.digest(seq) and len(B.digest(seq)) == 16
    assert B.digest(seq) != B.digest(B.sequence(seq.blocks, cycle_mm=(0.2, 0.31), name="x"))
    assert list(seq) == list(seq.blocks) and len(seq) == 5


def test_every_library_sequence_is_data():
    from mrgym.core.validate import _library_cases
    seen = set()
    for name, fn, args in _library_cases():
        seq = fn(*args)
        assert isinstance(seq, B.Sequence) and len(seq.blocks) == 1 and seq.blocks[0].kind == "repeat", name
        assert B.from_data(json.loads(json.dumps(B.to_data(seq)))) == seq, name
        seen.add(B.digest(seq))
    assert len(seen) == len(_library_cases())


def test_compilation_phases_tables_and_labels():
    rep = B.repeat([B.rf(flip_table="a", phase_deg=5.0), B.adc("s", phase_deg=1.0), B.gradient(1, ms=2.0)], 4, phase_inc_deg=10.0, spoil_inc_deg=50.0,
                   tables=dict(a=(20.0, 40.0)))
    prog = B.compile_ops([B.rf(90), rep, B.adc("end")])
    rf = [op for op in prog.ops if op.kind == "rf"]
    assert [op.flip_deg for op in rf] == [90.0, 20.0, 40.0, 20.0, 40.0]
    assert [op.phase_deg for op in rf[1:]] == pytest.approx([5.0 + 10 * j + 50 * j * (j + 1) / 2 for j in range(4)])
    adc = [op for op in prog.ops if op.kind == "adc"]
    assert [op.phase_deg for op in adc[:4]] == pytest.approx([1.0 + 10 * j + 50 * j * (j + 1) / 2 for j in range(4)])
    assert prog.labels == (("s", 0), ("s", 1), ("s", 2), ("s", 3), "end")
    assert prog.reach == (4,) and prog.unit == (1.0,) and prog.period_ms == pytest.approx(8.0)
    # the pulse frame: base phases, a frame operation after every repetition, and one that turns back at the end
    fr = B.compile_ops([B.rf(90), rep, B.adc("end")], frame=True)
    assert [op.phase_deg for op in fr.ops if op.kind == "rf"][1:] == [5.0] * 4
    frames = [op for op in fr.ops if op.kind == "frame"]
    assert len(frames) == 5 and frames[0].turn == pytest.approx(-math.radians(60.0)) and frames[0].shift == pytest.approx((math.radians(50.0),))
    assert frames[-1].turn == pytest.approx(math.radians(10 * 4 + 50 * 4 * 5 / 2)) and frames[-1].shift == pytest.approx((-4 * math.radians(50.0),))
    one = B.compile_ops([rep], frame=True, repetitions=2)
    assert sum(op.kind == "frame" for op in one.ops) == 2 and one.labels == (("s", 0), ("s", 1))
    nested = B.compile_ops([B.repeat([B.rf(90), B.repeat([B.rf(180), B.adc(("echo", "a"))], 2)], 2)])
    assert nested.labels == (("echo", "a", 0, 0), ("echo", "a", 0, 1), ("echo", "a", 1, 0), ("echo", "a", 1, 1))


def test_the_pulse_frame_refuses_what_it_cannot_do():
    with pytest.raises(ValueError):          # an rf at non-zero accumulated dephasing
        B.compile_ops([B.repeat([B.rf(30), B.gradient(1), B.rf(30), B.gradient(1)], 3, spoil_inc_deg=117.0)], frame=True)
    with pytest.raises(ValueError):          # no net dephasing
        B.compile_ops([B.repeat([B.rf(30), B.gradient(1), B.gradient(-1)], 3, spoil_inc_deg=117.0)], frame=True)
    with pytest.raises(ValueError):          # fractional cycles with RF spoiling
        B.compile_ops([B.repeat([B.rf(30), B.gradient(0.5)], 3, spoil_inc_deg=117.0)], frame=True)
    B.compile_ops([B.repeat([B.rf(30), B.gradient(1), B.rf(30), B.gradient(1)], 3, phase_inc_deg=180.0)], frame=True)      # linear: always
    with pytest.raises(ValueError):
        B.compile_ops([B.rf(90), B.delay(1.0)], repetitions=3)


def test_the_lattice_of_fractional_and_multi_axis_dephasing():
    assert B.lattice([B.gradient(2), B.gradient(-3)]) == (1, (1,))
    axes, unit = B.lattice([B.gradient((0.5, 1)), B.gradient((0.75, 0)), B.repeat([B.gradient((0, 2, 0.1))], 2)])
    assert axes == 3 and [float(u) for u in unit] == [0.25, 1.0, 0.1]
    prog = B.compile_ops([B.gradient((0.5, 1)), B.gradient((-0.75, 0), ms=1.0)])
    assert [op.n for op in prog.ops] == [(2, 1), (-3, 0)] and prog.reach == (5, 1)
    with pytest.raises(ValueError):          # three axes of dephasing, two cycle lengths
        B.compile_ops(B.sequence([B.gradient((1, 1, 1))], cycle_mm=(1.0, 2.0)))
    assert B.compile_ops(B.sequence([B.gradient((1, 1))], cycle_mm=0.5)).cycle_mm == (0.5, 0.5)


def test_lobe_shape_and_b_increment_against_the_gradient_layer():
    """The shape moments of a trapezoid and the b-value of a block are those of ``mrgym.design.gradients.b_value``."""
    area, dur, (m1, m2) = B.lobe_shape([0.0, 4.0], [10.0, 10.0])
    assert (area, dur) == (40.0, 4.0) and (m1, m2) == pytest.approx(B.LINEAR)
    t, g = np.array([0.0, 0.7, 5.0, 5.7]), np.array([0.0, 28.0, 28.0, 0.0])
    area, dur, shape = B.lobe_shape(t, g)
    assert area == pytest.approx(28.0 * 5.0) and dur == pytest.approx(5.7) and shape[0] == pytest.approx(0.5)
    lobe = L.diffusion_lobe(28.0, 5.0, 0.7)
    b_alone = B.b_increment(0.0, 1.0, lobe["duration"], lobe["cycle_mm"], lobe["shape"])
    assert b_alone == pytest.approx(gradients.b_value(t, g), rel=1e-12)
    # entering at order k: the lobe followed by its mirror image (a state at -1 going to 0) and a gap in between
    b_pair = b_alone + B.b_increment(1.0, 0.0, 20.0, lobe["cycle_mm"]) + B.b_increment(-1.0, 1.0, lobe["duration"], lobe["cycle_mm"], lobe["shape"])
    wave_t = np.concatenate([t, [25.7], 25.7 + t[1:]])
    wave_g = np.concatenate([g, [0.0], g[1:]])
    assert b_pair == pytest.approx(gradients.b_value(wave_t, wave_g, refocus_ms=(15.0,)), rel=1e-12)
    with pytest.raises(ValueError):          # a bipolar pair has no area: it is two lobes
        B.lobe_shape([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, -1.0, 0.0])


def test_design_blocks_convert():
    S = design_sequence
    pulse = pulses.sinc(60.0, 2.0, 4.0, 5.0, n=64)
    pos = np.linspace(-6.0, 6.0, 9)
    out = B.from_design([S.rf(30, 20), S.delay(3.0), S.gradient(-2, ms=1.0), S.spoil(), S.rf_pulse(pulse, pos), S.readout("x", 15.0)])
    assert [b.kind for b in out] == ["rf", "delay", "gradient", "spoil", "rf", "adc"]
    assert out[0].flip_deg == pytest.approx(30.0) and out[2].cycles == (-2.0,) and len(out[4].ck) == 9 and out[5].label == "x"
    a, b = pulse.rotate(pos)
    assert B.rf_designed(pulse, pos).ck == out[4].ck
    assert np.allclose(np.array(out[4].ck)[:, 0] + 1j * np.array(out[4].ck)[:, 1], a)
    with pytest.raises(ValueError):
        B.from_design([S.rf(np.array([10.0, 20.0]))])


def test_echo_of_line():
    echo, te, shots = L.echo_of_line(128, 16, 8.0, 80.0)
    assert shots == 8 and te == 80.0 and echo[64] == 9 and sorted(set(echo)) == list(range(16))
    assert all(np.sum(echo == e) == 8 for e in range(16))
    echo, te, shots = L.echo_of_line(96, 8, 10.0, 0.0, order="centric")
    assert te == 10.0 and echo[48] == 0 and 7 in (echo[0], echo[-1]) and sorted(set(echo)) == list(range(8))
    assert np.all(np.abs(np.diff(echo)) <= 2)
    with pytest.raises(ValueError):
        L.echo_of_line(64, 8, 10.0, 40.0, order="spiral")
