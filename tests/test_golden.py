"""Golden digests of the engine (``tests/golden/engine_digests.json``; contents: ``tests/golden_cases.py``).

Two modes, printed in the header of the test run. On the stack the file was recorded on every image and every noisy
scan is compared bit for bit (sha256). On any other numpy / scipy / machine the 8 x 8 block fingerprints must agree
within ``REPLAY_RTOL`` (NumPy promises no bit-identical results across builds), and prices must still be equal.

A failure means one of two things, and the message says which:
  * ``ENGINE_VERSION`` was bumped and the file was not regenerated  ->  ``python tools/audit/golden.py --write``
  * scan data or prices moved under an unchanged ``ENGINE_VERSION``  ->  bump it, then regenerate
"""
import pytest

import golden_cases as G
import mrgym
from mrgym.scanner import session

GOLDEN = G.load(G.ENGINE_FILE)
MODE = G.mode(GOLDEN["stack"])[0] if GOLDEN else None
SILENT = ("scan data moved while ENGINE_VERSION is still {v}: bump ENGINE_VERSION in src/mrgym/scanner/session.py, then run "
          "`python tools/audit/golden.py --write`")


def _same(got: dict, want: dict, what: str):
    assert got["shape"] == want["shape"], f"{what}: shape {got['shape']} != {want['shape']}; " + SILENT.format(v=GOLDEN["engine"])
    if MODE == "bit-exact":
        assert got["sha256"] == want["sha256"], f"{what}: " + SILENT.format(v=GOLDEN["engine"])
    else:
        assert session.fingerprints_close(got["fingerprint"], want["fingerprint"]), f"{what} (tolerance mode): " + SILENT.format(v=GOLDEN["engine"])


def test_the_file_was_recorded_on_this_engine_version():
    assert GOLDEN, "tests/golden/engine_digests.json is missing: run `python tools/audit/golden.py --write`"
    assert GOLDEN["engine"] == mrgym.ENGINE_VERSION, (
        f"ENGINE_VERSION is {mrgym.ENGINE_VERSION}, the golden digests were recorded on {GOLDEN['engine']}: "
        "run `python tools/audit/golden.py --write`")
    assert MODE in ("bit-exact", "tolerance")


def test_the_file_covers_every_sequence_phantom_and_field():
    assert sorted(GOLDEN["images"]) == sorted(G.image_keys()) and len(G.image_keys()) == 6 * 3 * 2 + len(G.VARIANTS) + len(G.VARIANTS_0_3)
    assert {v.get("excitation") for v in G.VARIANTS.values()} >= {"nonselective"} and len(G.VARIANTS) >= 6 + 3
    # engine 0.3.0: fat saturation on a phantom that holds fat, partial Fourier, each of the four adjustments
    said = [set(v) for v in G.VARIANTS_0_3.values()]
    assert any("fat_sat" in v for v in said) and any("partial_fourier" in v for v in said)
    assert set(mrgym.scanner.console.ADJUSTMENTS) <= set().union(*said)
    assert any(c["fat"] for c in mrgym.scanner.world.make_world(G.SEED, dict(G.WORLD_SPECS["qa_oil"], n=64))["tissues"].values())
    assert set(G.PROTOCOLS) == set(mrgym.scanner.console.SEQUENCES)
    assert len(GOLDEN["noisy"]) == len(G.PROTOCOLS) + 2 and len(GOLDEN["prices"]) == len(G.PRICE_GRID)


@pytest.mark.parametrize("key", G.image_keys())
def test_noise_free_image(key):
    _same(G.image_case(key), GOLDEN["images"][key], key)


def test_noisy_scans_and_the_log_chain():
    got = G.noisy_session()
    assert list(got) == list(GOLDEN["noisy"]) or sorted(got) == sorted(GOLDEN["noisy"])
    for key in got:
        if key != "chain_head":
            _same(got[key], GOLDEN["noisy"][key], "noisy " + key)
    assert got["0:spgr"]["sha256"] != GOLDEN["images"]["qa/3.0T/spgr"]["sha256"]          # noise was added
    assert got["2:se"]["sha256"] != got["6:se"]["sha256"]                                 # a repeated scan draws new noise
    if MODE == "bit-exact":
        assert got["chain_head"] == GOLDEN["noisy"]["chain_head"], "the scanner log changed: " + SILENT.format(v=GOLDEN["engine"])


def test_prices_are_equal_on_every_stack():
    assert G.prices() == GOLDEN["prices"], "a scan costs something else: " + SILENT.format(v=GOLDEN["engine"])


def test_a_changed_image_is_detected_in_both_modes():
    """The comparison itself: one block 0.1 % brighter must fail, in tolerance mode too."""
    want = GOLDEN["images"]["qa/3.0T/se"]
    moved = dict(want, sha256="0" * 64, fingerprint=[[v * (1.001 if i == 27 else 1.0) for i, v in enumerate(row)] for row in want["fingerprint"]])
    with pytest.raises(AssertionError, match="bump ENGINE_VERSION"):
        _same(moved, want, "moved")
    assert not session.fingerprints_close(moved["fingerprint"], want["fingerprint"])
    assert session.fingerprints_close([[v * (1 + 1e-6) for v in row] for row in want["fingerprint"]], want["fingerprint"])
