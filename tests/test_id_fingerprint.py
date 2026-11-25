"""An environment id is a promise that rewards under it are comparable. The fingerprint of an id hashes everything
that can move a reward (``mrgym.compat.fingerprint``); a fingerprint that changes under an unchanged id fails here."""
import json

import pytest

import mrgym
from mrgym import tools as T
from mrgym.compat import fingerprint as F

ID = mrgym.list_envs()[0]


@pytest.fixture(scope="module")
def current():
    """The parts of one id, computed once; the mechanism tests below replace ``F.parts`` with edited copies of it."""
    return F.parts(ID)


def test_every_registered_id_has_the_fingerprint_that_is_stored_for_it():
    problems = F.check()
    assert not problems, "\n".join(problems)


def test_the_stored_file_names_every_part_and_no_id_is_frozen_before_a_release():
    stored = F.load()
    assert stored["schema"] == F.SCHEMA and set(mrgym.list_envs()) <= set(stored["ids"])
    for env_id, row in stored["ids"].items():
        assert F.fingerprint(row["parts"]) == row["fingerprint"], env_id
        assert {"registration", "engine", "rules", "scoring_source"} <= set(row["parts"])
        if env_id in mrgym.registry:
            for level in mrgym.spec(env_id).levels:
                assert {f"L{level}.statement", f"L{level}.tools", f"L{level}.transcript"} <= set(row["parts"])


def test_one_character_in_a_tool_description_moves_the_tools_part_only(monkeypatch):
    level = mrgym.spec(ID).default_level
    before = F.level_parts(ID, level)
    original = T.ToolSpec.to

    def edited(self, fmt="openai"):
        out = original(self, fmt)
        if fmt == "openai":
            out["function"]["description"] += "."
        return out

    monkeypatch.setattr(T.ToolSpec, "to", edited)
    after = F.level_parts(ID, level)
    assert [k for k in before if before[k] != after[k]] == [f"L{level}.tools"]


def test_a_changed_part_fails_the_check_and_is_named(monkeypatch, tmp_path, current):
    file = str(tmp_path / "fingerprints.json")
    monkeypatch.setattr(F, "parts", lambda env_id: dict(current))
    assert F.write([ID], file) == ([ID], [])
    assert not F.check([ID], file)
    monkeypatch.setattr(F, "parts", lambda env_id: dict(current, **{"L1.tools": "0" * 64}))
    problems = F.check([ID], file)
    assert len(problems) == 1 and "parts that moved: L1.tools." in problems[0] and "needs a new id" in problems[0]
    assert "--write" in problems[0]                                  # not frozen: the fingerprint may be stored again


def test_a_frozen_id_cannot_be_rewritten_and_a_vanished_id_is_reported(monkeypatch, tmp_path, current):
    file = str(tmp_path / "fingerprints.json")
    monkeypatch.setattr(F, "parts", lambda env_id: dict(current))
    F.write([ID], file)
    assert F.freeze([ID], file) == []
    assert F.write([ID], file) == ([], [])                           # unchanged: nothing to do, nothing refused
    monkeypatch.setattr(F, "parts", lambda env_id: dict(current, engine="0" * 64))
    written, refused = F.write([ID], file)
    assert not written and len(refused) == 1 and "frozen" in refused[0]
    problem = F.check([ID], file)[0]
    assert "FROZEN" in problem and "engine" in problem and "--write" not in problem
    assert F.freeze([ID], file)                                      # a changed id cannot be frozen either
    data = json.load(open(file))
    data["ids"]["Scanner/GoneForTheTest-v0"] = data["ids"][ID]
    json.dump(data, open(file, "w"))
    monkeypatch.setattr(mrgym, "list_envs", lambda namespace=None: [])
    assert any("Scanner/GoneForTheTest-v0" in p and "neither registered nor retired" in p for p in F.check(None, file))


def test_comments_and_docstrings_do_not_move_the_source_digest_and_code_does(tmp_path, monkeypatch):
    pkg = tmp_path / "fp_probe_mod.py"
    monkeypatch.syspath_prepend(str(tmp_path))

    def digest(text):
        import importlib
        import linecache
        import sys
        pkg.write_text(text)
        sys.modules.pop("fp_probe_mod", None)
        importlib.invalidate_caches()
        linecache.clearcache()
        return F.source_digest(["fp_probe_mod"])

    base = digest('"""doc"""\nTOL = 0.05\n\n\ndef score(x):\n    """doc"""\n    return x <= TOL\n')
    assert digest('"""another doc"""\nTOL = 0.05   # a comment\n\ndef score(x):\n    return x <= TOL\n') == base
    assert digest('"""doc"""\nTOL = 0.06\n\n\ndef score(x):\n    """doc"""\n    return x <= TOL\n') != base


@pytest.mark.parametrize("field", ["max_steps", "reward_threshold"])
def test_registration_fields_are_part_of_the_fingerprint(field, monkeypatch):
    import dataclasses
    before = F.static_parts(ID)
    spec = mrgym.spec(ID)
    monkeypatch.setitem(mrgym.registry, ID, dataclasses.replace(spec, **{field: getattr(spec, field) + 1}))
    after = F.static_parts(ID)
    assert [k for k in before if before[k] != after[k]] == ["registration"]
