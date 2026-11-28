"""Periodic motion in the sequences served from the block-list core.

The engine's model of motion is a modulation of k-space lines (docs/physics/limits.md, block 7): every line of the
moving class is multiplied by 1 + depth cos(2 pi f t + phase) (a pulsing signal) or by exp(-2 pi i k . d(t)) (a rigid
displacement d(t) = amp cos(2 pi f t + phase)), t being the time at which the line is acquired. For a sequence with
one line per repetition that is what `se` does. In an echo train several lines share one excitation: line p is read
in shot (p mod shots) by the echo that the sampling plan assigns to it, at

    t(p, average a) = ((p mod shots) x averages + a) x TR + echo time of that echo,

so two lines of one shot are an echo spacing apart and two neighbouring lines of one band a repetition apart. The
train itself is that of the object at rest: motion does not change the echo amplitudes.
"""
import hashlib

import numpy as np
import pytest

from mrgym.scanner import console, kspace, served
from mrgym.scanner import world as W

PHASE_KEY = 7


def _world(motion, **spec):
    return W.make_world("c1" * 16, dict(dict(kind="qa", n=128, console_sequences="all", sigma0=0.0), motion=motion, **spec))


def _phase(mv, eng, rp):
    key = kspace.physics_key(rp, averages=True)
    h = hashlib.sha256(repr((mv["phase_key"], key)).encode()).digest()
    return 2 * np.pi * int.from_bytes(h[:4], "big") / 2 ** 32


@pytest.mark.parametrize("prot", [dict(sequence="fse", tr_ms=900.0, etl=6, echo_order="linear", te_ms=40.0, matrix=48, averages=2),
                                  dict(sequence="fse", tr_ms=700.0, etl=4, matrix=32, partial_fourier=0.75, pe_dir="x")])
def test_an_echo_train_line_is_modulated_at_the_time_of_its_echo(prot):
    w = _world([dict(kind="pulse", cls="*", f_hz=1.3, depth=0.4, phase_key=PHASE_KEY)])
    still = kspace.Engine(dict(w, motion=[]))
    moved = kspace.Engine(w)
    rp = console.plan(prot, w["console"])["realised"]
    a, b = still.clean_kspace(rp)[0], moved.clean_kspace(rp)[0]
    plan = console.sampling(rp)
    lines = np.asarray(plan.lines[0])
    acquired = np.flatnonzero(lines >= 0)
    shot = np.arange(acquired.size) % plan.shots                        # the lines of a band are read in successive shots
    phase = _phase(w["motion"][0], moved, rp)
    factor = np.ones(rp["lines_full"])
    for i, p in enumerate(acquired):
        t = ((shot[i] * rp["averages"] + np.arange(rp["averages"])) * rp["tr_ms"] + (lines[p] + 1) * rp["echo_spacing_ms"]) * 1e-3
        factor[p] = 1 + 0.4 * np.mean(np.cos(2 * np.pi * 1.3 * t + phase))
    assert np.abs(b - factor[:, None] * a)[acquired].max() < 1e-12 * np.abs(a).max()
    assert np.abs(b - a).max() > 0.05 * np.abs(a).max()


def test_every_image_of_a_multi_echo_scan_is_modulated_at_its_own_echo_time():
    w = _world([dict(kind="pulse", cls="*", f_hz=0.8, depth=0.3, phase_key=PHASE_KEY)])
    still, moved = kspace.Engine(dict(w, motion=[])), kspace.Engine(w)
    for prot in (dict(sequence="mese", tr_ms=600.0, n_echoes=4, echo_spacing_ms=20.0, matrix=32),
                 dict(sequence="megre", tr_ms=120.0, flip_deg=20.0, n_echoes=3, matrix=32),
                 dict(sequence="dwse", tr_ms=500.0, b_value=300.0, matrix=32)):
        rp = console.plan(prot, w["console"])["realised"]
        a, b = still.clean_kspace(rp), moved.clean_kspace(rp)
        phase = _phase(w["motion"][0], moved, rp)
        line = np.arange(rp["lines_full"]) - rp["lines_full"] // 2 + (rp["lines"] - rp["lines_full"] // 2)
        for i, te in enumerate(rp["echo_times_ms"]):
            t = (line * rp["tr_ms"] + te) * 1e-3
            assert np.abs(b[i] - (1 + 0.3 * np.cos(2 * np.pi * 0.8 * t + phase))[:, None] * a[i]).max() < 1e-12 * np.abs(a[i]).max(), prot["sequence"]


def test_a_frozen_displacement_moves_one_class_of_an_echo_train_image():
    """f = 0: the class is displaced by amp x cos(phase) along y for the whole scan. Its k-space is that of the class at
    rest times exp(-2 pi i k_y d) on every line; the other classes do not move."""
    w = _world([dict(kind="shift", cls="vial0", f_hz=0.0, amp_mm=8.0, dir="y", phase_key=PHASE_KEY)])
    still, moved = kspace.Engine(dict(w, motion=[])), kspace.Engine(w)
    rp = console.plan(dict(sequence="fse", tr_ms=1500.0, etl=8, matrix=64), w["console"])["realised"]
    a, b = still.clean_kspace(rp)[0], moved.clean_kspace(rp)[0]
    shift = 8.0 * np.cos(_phase(w["motion"][0], moved, rp))            # mm along y
    assert abs(shift) > 1.0
    c = w["names"].index("vial0")
    only = served.form_lines(still, rp, served.table(still, rp), console.sampling(rp), only=c)[0]     # the k-space of the class alone
    ky = (np.arange(rp["lines_full"]) - rp["lines_full"] // 2) / (rp["fov_pe_mm"] * rp["lines_full"] / rp["matrix_pe"])
    want = a + (np.exp(-2j * np.pi * ky * shift)[:, None] - 1) * only
    assert np.abs(b - want).max() < 1e-12 * np.abs(a).max()
    assert np.abs(b - a).max() > 1e-3 * np.abs(a).max()
