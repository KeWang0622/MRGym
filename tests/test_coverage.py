"""``mrgym coverage``: the competency map as data, the computed column, the thresholds."""
import json
import pathlib

import mrgym
from mrgym import cli, coverage

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_the_map_holds_the_lines_of_the_outline():
    t = coverage.table()
    counts = t["counts"]
    assert {g: c["outline"] for g, c in counts.items()} == dict(parameter=18, option=11, artefact=14, qc=7)
    assert all(c["rows"] == c["outline"] == c["now"] + c["partial"] + c["absent"] for c in counts.values())
    ids = coverage.ids()
    assert len(ids) == len(set(ids)) and all(i.split(".")[0] in coverage.GROUPS for i in ids)
    assert {r["engine"] for r in t["rows"]} == {"now", "partial", "absent", "n/a"}
    assert [r["id"] for r in t["rows"] if r["engine"] == "n/a"] == ["qc.equipment_inspection"]
    numbers = [r["outline"] for r in t["rows"] if r["id"] not in coverage.EXTRA]
    assert numbers == ([f"2.A.{i}" for i in range(1, 19)] + [f"2.C.{i}" for i in range(1, 12)] + [f"1.C.{i}" for i in range(1, 15)]
                       + [f"1.D.{i}" for i in range(1, 9)])
    assert coverage.THRESHOLDS == dict(parameter=15, artefact=10, qc=6)


def test_the_computed_column_comes_from_the_registry_and_the_families():
    t = coverage.table()
    dec = coverage.declared()
    assert set(dec) == {i for i in mrgym.list_envs() if mrgym.spec(i).kind != "sandbox" and mrgym.spec(i).family}
    for env_id in ("Scanner/NullTissue-v0", "Scanner/SNRSpec-v0", "Scanner/PhantomQA-v0"):
        assert dec[env_id] and set(dec[env_id]) <= set(coverage.ids())
    rows = {r["id"]: r for r in t["rows"]}
    assert rows["parameter.ti"]["environments"] == ["Scanner/FatSuppress-v0", "Scanner/NullTissue-v0"] and "Scanner/SNRSpec-v0" in rows["qc.snr"]["environments"]
    assert rows["qc.geometric_accuracy"]["environments"] == ["Scanner/PhantomQA-v0"] and rows["parameter.etl"]["environments"] == []
    for g, c in t["counts"].items():
        assert c["exercised"] == sum(bool(r["environments"]) for r in t["rows"] if r["group"] == g and r["id"] not in coverage.EXTRA and r["engine"] != "n/a")
    checks = {r["check"] for r in t["rows"] if r["check"] and not r["check"].startswith("mrgym")}
    assert checks <= coverage.validate_ids()                             # every named row of `mrgym validate` exists
    assert all(r["check_exists"] for r in t["rows"] if r["check"])


def test_assert_enforces_what_the_table_states(capsys, monkeypatch):
    t = coverage.table()
    assert cli.main(["coverage"]) == 0
    text = capsys.readouterr().out
    assert "exercised by a scored environment" in text and "(enforced by --assert)" in text
    assert cli.main(["coverage", "--assert"]) == (0 if t["ok"] else 1)
    capsys.readouterr()
    assert cli.main(["coverage", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["counts"] == t["counts"]
    need = {g: f"{coverage.GROUP_TITLES[g]}: {t['counts'][g]['exercised']} of" for g, n in coverage.THRESHOLDS.items() if t["counts"][g]["exercised"] < n}
    assert all(any(p.startswith(prefix) for p in t["problems"]) for prefix in need.values())
    monkeypatch.setattr(coverage, "THRESHOLDS", dict(parameter=0))
    monkeypatch.setattr(coverage, "declared", lambda: {"Scanner/PhantomQA-v0": ("qc.ghosting", "qc.made_up")})
    problems = coverage.table()["problems"]
    assert any("'qc.made_up', which is not a row of the map" in p for p in problems) and not any("threshold" in p for p in problems)


def test_the_page_under_docs_is_the_output_of_the_command():
    page = (ROOT / "docs" / "community" / "competency_map.md").read_text(encoding="utf-8")
    assert page == coverage.doc_page(coverage.table()), "regenerate: mrgym coverage --write docs/community/competency_map.md"
