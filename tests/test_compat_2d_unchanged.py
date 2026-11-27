"""A 2D exam is what it was before the third dimension arrived (``tests/golden/compat_2d.json``; ``tests/compat_cases.py``).

Every package that adds volume worlds, slices, receive arrays or parameters for them must keep this file green: for
every 2D world kind and every console sequence the world itself, the protocol schema, the scanner card, the whole
``plan`` output, the header line and the noise-free image equal the values stored on ``v1`` before the first such
package; the parameters a 2D exam refuses keep their refusal; the golden engine digests and the id fingerprints
that existed then are still the stored ones. New kinds, new ids and new golden sections may be added next to them.

Two modes, as in ``tests/test_golden.py``: on the numerical stack the baseline was recorded on, arrays are compared
bit for bit (sha256); on any other, by their 8 x 8 block fingerprints within ``REPLAY_RTOL``. Text, schemas, plans
and prices are equal on every stack.

A failure means one of three things, and the message says which:
  * a 3D package changed a 2D exam: undo that (the new behaviour belongs to volume or array exams only)
  * ``ENGINE_VERSION`` was bumped on purpose: ``python tests/compat_cases.py --write --rebase``
  * an id was re-fingerprinted on purpose (it was never released): the same command
"""
import json
import os

import pytest

import compat_cases as C
import golden_cases as G
import mrgym
from mrgym.scanner import console, session
from mrgym.scanner import world as W

BASE = C.load()
MODE = G.mode(BASE["stack"])[0] if BASE else None
MOVED = ("a 2D exam changed against the baseline recorded on v1 ({commit}, engine {engine}). If a volume or array package did this, "
         "undo it: new behaviour belongs to the new exams only. If the engine version was bumped on purpose: "
         "`python tests/compat_cases.py --write --rebase`.")


def _why():
    return MOVED.format(**BASE["recorded"])


def _same_array(got: dict, want: dict, key: str, what: str):
    if MODE == "bit-exact":
        assert got[key] == want[key], f"{what}: {key} differs; " + _why()
    else:
        assert session.fingerprints_close(got["fingerprint"], want["fingerprint"]), f"{what} (tolerance mode): " + _why()


def test_the_baseline_exists_and_was_recorded_on_this_engine_version():
    assert BASE, "tests/golden/compat_2d.json is missing: it is recorded once, on v1 (python tests/compat_cases.py --write)"
    assert BASE["engine"] == mrgym.ENGINE_VERSION, (
        f"ENGINE_VERSION is {mrgym.ENGINE_VERSION}, the 2D baseline was recorded on {BASE['engine']}: after a deliberate bump run "
        "`python tests/compat_cases.py --write --rebase`")
    assert BASE["recorded"]["source_tree_clean"] and MODE in ("bit-exact", "tolerance")


def test_the_baseline_covers_every_2d_kind_and_every_console_sequence():
    assert set(C.PROTOCOLS) == set(console.SEQUENCES), "a console sequence has no protocol in tests/compat_cases.py"
    assert set(C.KINDS_2D) <= set(W.KINDS) and sorted(BASE["scans"]) == sorted(C.scan_keys()) and len(BASE["scans"]) == 5 * 6
    assert set(BASE["worlds"]) == set(BASE["exams"]) == set(C.KINDS_2D)
    # a world kind that is not in the baseline must be a volume kind (it branches before the 2D code): nothing else may be added silently
    try:
        from mrgym.scanner import volume
        volumes = set(volume.KINDS)
    except ImportError:                                                 # the tree before the first 3D package
        volumes = set()
    assert set(W.KINDS) - set(C.KINDS_2D) <= volumes
    assert set(BASE["refusals"]) == set(console.NOT_SIMULATED), "the list of refused parameters changed for a 2D exam; " + _why()


@pytest.mark.parametrize("kind", C.KINDS_2D)
def test_world_of_a_2d_kind_is_unchanged(kind):
    got, want = C.world_case(kind), BASE["worlds"][kind]
    assert set(want) <= set(got), f"world {kind}: {sorted(set(want) - set(got))} is gone from the world; " + _why()
    for key in want:                                                     # a key may be added to a world; what was there stays
        if key == "fingerprint":
            continue
        if key == "weights" or key.startswith(("field.", "mask.")):
            _same_array(got, want, key, f"world {kind}")
        else:
            assert got[key] == want[key], f"world {kind}: {key} differs; " + _why()


@pytest.mark.parametrize("kind", C.KINDS_2D)
def test_schema_and_scanner_card_of_a_2d_exam_are_unchanged(kind):
    got, want = C.exam_case(kind), BASE["exams"][kind]
    moved = [k for k in want if got.get(k) != want[k]]
    assert not moved, f"exam on {kind}: {', '.join(moved)} differ(s); " + _why()


@pytest.mark.parametrize("key", C.scan_keys())
def test_plan_header_and_image_of_a_2d_scan_are_unchanged(key):
    kind, seq = key.split("/")
    got, want = C.scan_case(kind, seq), BASE["scans"][key]
    assert got["header"] == want["header"], f"{key}: header\n  {got['header']}\ninstead of\n  {want['header']}\n" + _why()
    assert got["plan"] == want["plan"], f"{key}: the plan output differs (the header is equal); " + _why()
    assert got["shape"] == want["shape"], f"{key}: image shape; " + _why()
    _same_array(got, want, "image", key)


def test_a_2d_exam_still_refuses_the_parameters_it_does_not_simulate():
    got = C.refusals()
    moved = sorted(k for k in BASE["refusals"] if got.get(k) != BASE["refusals"][k])
    assert not moved, f"plan answers something else to {', '.join(moved)} on a 2D exam; " + _why()
    for name in ("slices", "slice_gap_mm", "slice_position_mm", "acceleration"):           # the words the 3D packages will add
        pl = console.plan(dict(C.PROTOCOLS["se"], **{name: 2}))
        assert not pl["accept"] and pl["violations"][0]["code"] == "NOT_SIMULATED", name


def test_the_golden_engine_digests_that_existed_are_the_stored_ones():
    now = C.stored_golden()
    moved = sorted(k for k, v in BASE["golden"].items() if now.get(k) != v)
    assert not moved, (f"{len(moved)} entr(ies) of tests/golden/engine_digests.json were rewritten or removed, e.g. {moved[:4]}: sections may be "
                       "added (volume, array), the 2D entries stay; " + _why())
    assert G.load(G.ENGINE_FILE)["engine"] == BASE["engine"]


def test_the_id_fingerprints_that_existed_are_the_stored_ones():
    now = C.stored_fingerprints()
    moved = sorted(k for k, v in BASE["fingerprints"].items() if now.get(k) != v)
    assert not moved, (f"the stored fingerprint of {', '.join(moved)} changed or is gone. The tool schema and the headers are part of every id "
                       "fingerprint: a parameter that exists only for volume or array exams must not appear in a 2D exam. " + _why())


def test_the_comparison_itself_detects_a_change(monkeypatch):
    """One character in a header, one bit of an image, one more schema property: each is seen."""
    key = "qa/se"
    want = BASE["scans"][key]
    assert C.sha(dict(a=1)) != C.sha(dict(a=1, slices=1))
    with pytest.raises(AssertionError, match="2D exam changed"):
        _same_array(dict(want, image="0" * 64, fingerprint=[[v * (1.001 if i == 5 else 1.0) for i, v in enumerate(row)] for row in want["fingerprint"]]),
                    want, "image", key)
    real = console.protocol_schema

    def with_slices(exam=None, brief=False):
        out = json.loads(json.dumps(real(exam, brief)))
        out["properties"]["slices"] = {"type": "integer", "description": "number of slices"}
        return out

    monkeypatch.setattr(console, "protocol_schema", with_slices)
    got = C.exam_case("qa")
    assert got["schema"] != BASE["exams"]["qa"]["schema"] and got["schema_brief"] != BASE["exams"]["qa"]["schema_brief"]
    diff, added, gone = C.compare(BASE, dict(BASE, exams=dict(BASE["exams"], qa=got)))
    assert diff == ["exams.qa.schema", "exams.qa.schema_brief"] and not added and not gone


def test_the_baseline_file_is_what_the_tool_writes():
    """Nothing in the file is typed by hand: it is the dump of a dict, and its size stays small."""
    with open(C.FILE, encoding="utf-8") as f:
        text = f.read()
    assert json.loads(text) == BASE and os.path.getsize(C.FILE) < 120_000
