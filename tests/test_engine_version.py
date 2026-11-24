"""The engine version is part of every number MR Gym reports: the same seed and protocol give the same data only on
the same engine, and a recorded log replays only on the engine that wrote it.

The version is a constant that is bumped by hand (``ENGINE_VERSION`` in ``src/mrgym/scanner/session.py``). This file
pins it together with three reference numbers of the engine, so a change of the physics cannot slip in without a
bump. The reference numbers below must change only together with ``ENGINE_VERSION``: if a test here fails, either
revert the change, or bump the version, record the new numbers, and add an "Engine" line to CHANGELOG.md.

Reference case: seed ``000102030405060708090a0b0c0d0e0f``, world spec ``{"kind": "qa"}``, the noise-free image of
the protocol ``{"sequence": "se", "tr_ms": 500, "matrix": 64}``.
"""
import numpy as np
import pytest

import mrgym
from mrgym.scanner import kspace
from mrgym.scanner import world as W

ENGINE_VERSION = "0.1.0"
SEED = "000102030405060708090a0b0c0d0e0f"
WORLD_SPEC = dict(kind="qa")
PROTOCOL = dict(sequence="se", tr_ms=500, matrix=64)
# recorded with ENGINE_VERSION above (numpy 2.2.6, scipy 1.15.3, one thread)
MEAN_MAGNITUDE = 0.7259219837862718
MAX_MAGNITUDE = 2.778571927919207
MAGNITUDE_AT_0_32_32 = 2.3098207026200184
RTOL = 1e-6


@pytest.fixture(scope="module")
def magnitude():
    world = W.make_world(SEED, WORLD_SPEC)
    return np.abs(kspace.Engine(world).scan(PROTOCOL)["images"])


def test_engine_id_is_the_recorded_one():
    assert mrgym.ENGINE_VERSION == ENGINE_VERSION
    assert mrgym.ENGINE_ID == "mrgym-engine-" + ENGINE_VERSION == "mrgym-engine-0.1.0"


def test_reference_image_has_the_recorded_shape(magnitude):
    assert magnitude.shape == (1, 64, 64)


def test_reference_numbers_of_the_engine(magnitude):
    got = dict(mean=float(magnitude.mean()), max=float(magnitude.max()), pixel=float(magnitude[0, 32, 32]))
    want = dict(mean=MEAN_MAGNITUDE, max=MAX_MAGNITUDE, pixel=MAGNITUDE_AT_0_32_32)
    for key in want:
        assert got[key] == pytest.approx(want[key], rel=RTOL), (
            f"{key} magnitude changed: {want[key]!r} -> {got[key]!r}. Scan data moved, so ENGINE_VERSION must be "
            f"bumped together with these numbers (see the docstring of this file).")


def test_the_noise_free_scan_is_deterministic(magnitude):
    again = np.abs(kspace.Engine(W.make_world(SEED, WORLD_SPEC)).scan(PROTOCOL)["images"])
    assert np.array_equal(again, magnitude)
