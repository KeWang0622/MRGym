"""The engine version is part of every number MR Gym reports: the same seed and protocol give the same data only on
the same engine, and a recorded log replays only on the engine that wrote it.

The version is a constant that is bumped by hand (``ENGINE_VERSION`` in ``src/mrgym/scanner/session.py``). This file
pins it together with three reference numbers of the engine, so a change of the physics cannot slip in without a
bump. The reference numbers below must change only together with ``ENGINE_VERSION``: if a test here fails, either
revert the change, or bump the version, record the new numbers, and add an "Engine" line to CHANGELOG.md.

Reference case: seed ``000102030405060708090a0b0c0d0e0f``, world spec ``{"kind": "qa"}``, the noise-free image of
the protocol ``{"sequence": "se", "tr_ms": 500, "matrix": 64}`` (slice-selective, the default since 0.2.0). On 0.3.0
the three numbers are those of 0.2.0 divided by 0.99443, the calibration of the slice gradient of the default pulse
(mean 0.5995635694482917, maximum 2.286886738607616, pixel 1.8942632136377284 on 0.2.0).

The same protocol with ``excitation: nonselective`` is the hard-pulse model of engine 0.1.0; its three numbers are
kept below as recorded on 0.1.0 and must not move at all (``mrgym validate`` row R6 asserts the same).
"""
import numpy as np
import pytest

import mrgym
from mrgym.scanner import kspace
from mrgym.scanner import world as W

ENGINE_VERSION = "0.3.0"
SEED = "000102030405060708090a0b0c0d0e0f"
WORLD_SPEC = dict(kind="qa")
PROTOCOL = dict(sequence="se", tr_ms=500, matrix=64)
# recorded with ENGINE_VERSION above (numpy 2.2.6, scipy 1.15.3, one thread)
MEAN_MAGNITUDE = 0.6029202321250108
MAX_MAGNITUDE = 2.299689897025717
MAGNITUDE_AT_0_32_32 = 1.9048682653004751
RTOL = 1e-6
# recorded with engine 0.1.0: the hard-pulse image, served since 0.2.0 by excitation = nonselective
HARD_PULSE_0_1_0 = dict(mean=0.7259219837862718, max=2.778571927919207, pixel=2.3098207026200184)


@pytest.fixture(scope="module")
def magnitude():
    world = W.make_world(SEED, WORLD_SPEC)
    return np.abs(kspace.Engine(world).scan(PROTOCOL)["images"])


def test_engine_id_is_the_recorded_one():
    assert mrgym.ENGINE_VERSION == ENGINE_VERSION
    assert mrgym.ENGINE_ID == "mrgym-engine-" + ENGINE_VERSION == "mrgym-engine-0.3.0"


def test_reference_image_has_the_recorded_shape(magnitude):
    assert magnitude.shape == (1, 64, 64)


def test_reference_numbers_of_the_engine(magnitude):
    got = dict(mean=float(magnitude.mean()), max=float(magnitude.max()), pixel=float(magnitude[0, 32, 32]))
    want = dict(mean=MEAN_MAGNITUDE, max=MAX_MAGNITUDE, pixel=MAGNITUDE_AT_0_32_32)
    for key in want:
        assert got[key] == pytest.approx(want[key], rel=RTOL), (
            f"{key} magnitude changed: {want[key]!r} -> {got[key]!r}. Scan data moved, so ENGINE_VERSION must be "
            f"bumped together with these numbers (see the docstring of this file).")


def test_the_step_from_0_2_0_is_the_calibration_of_the_default_pulse(magnitude):
    """The reference image has no fat saturation and full sampling: of the four corrections of 0.3.0 only the slice
    calibration touches it, as one factor on the whole image (the profile is the same, the slice 0.56 % thinner in
    units of the nominal bandwidth, so every column through the slice counts 1 / 0.99443 times as much)."""
    from mrgym.scanner import rf
    old = dict(mean=0.5995635694482917, max=2.286886738607616, pixel=1.8942632136377284)
    got = dict(mean=float(magnitude.mean()), max=float(magnitude.max()), pixel=float(magnitude[0, 32, 32]))
    for key in old:
        assert got[key] * rf.fwhm_factor(4.0) == pytest.approx(old[key], rel=2e-6), key


def test_the_noise_free_scan_is_deterministic(magnitude):
    again = np.abs(kspace.Engine(W.make_world(SEED, WORLD_SPEC)).scan(PROTOCOL)["images"])
    assert np.array_equal(again, magnitude)


def test_the_hard_pulse_mode_is_the_engine_of_version_0_1_0():
    world = W.make_world(SEED, WORLD_SPEC)
    m = np.abs(kspace.Engine(world).scan(dict(PROTOCOL, excitation="nonselective"))["images"])
    got = dict(mean=float(m.mean()), max=float(m.max()), pixel=float(m[0, 32, 32]))
    for key, want in HARD_PULSE_0_1_0.items():
        assert got[key] == pytest.approx(want, rel=1e-9), key
