"""The conformance registry and its arithmetic (``conformance/``). The checkers themselves run in ``make conformance``;
these tests keep the registry well-formed and the scoring rules honest, and cost nothing."""
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = os.path.join(ROOT, "conformance")
pytestmark = pytest.mark.skipif(not os.path.isdir(CONF), reason="needs a checkout: conformance/ is not part of the wheel")


def _load(name):
    sys.path.insert(0, CONF)
    try:
        spec = importlib.util.spec_from_file_location(f"conformance_{name}", os.path.join(CONF, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(CONF)


@pytest.fixture(scope="module")
def run():
    return _load("run")


@pytest.fixture(scope="module")
def items(run):
    return run.load_items()


def test_the_registry_has_108_items_in_11_themes_with_stable_unique_ids(run, items):
    assert len(items) == 108 and len({i["id"] for i in items}) == 108
    assert {i["theme"] for i in items} == set(run.THEMES) and len(run.THEMES) == 11
    for i in items:
        assert i["id"][0] == i["theme"] and i["id"][1:].isdigit() and i["item"] and i["source"], i["id"]
        assert i["tier"] in run.TIERS, i["id"]


def test_every_item_is_a_command_with_an_expected_result_or_a_documented_action(run, items):
    kinds = {"check", "tests", "tests_matching", "cmd", "files", "native", "evidence", "online"}
    for i in items:
        c = i["check"]
        assert c["kind"] in kinds, i["id"]
        if c["kind"] == "native":
            assert callable(run.K.registry(c["fn"])), i["id"]
        if c["kind"] in ("evidence", "online"):
            assert i.get("action"), f"{i['id']}: an item no command can decide must say what to do"
        if c["kind"] == "tests":
            assert c["names"] and all(n.startswith("test_") for n in c["names"]), i["id"]
        if i["tier"] is not None:
            assert i.get("action"), f"{i['id']}: an item with a tier names the change that flips it"


def test_named_tests_exist_in_the_test_suite(items):
    source = ""
    for name in os.listdir(os.path.join(ROOT, "tests")):
        if name.startswith("test_") and name.endswith(".py"):
            source += open(os.path.join(ROOT, "tests", name), encoding="utf-8").read()
    wanted = {n for i in items for c in (i["check"], i["check"].get("then") or {}) if c.get("kind") == "tests" for n in c["names"]}
    missing = sorted(n for n in wanted if f"def {n}(" not in source)
    assert not missing, f"conformance items name tests that do not exist: {missing}"


def test_score_is_pass_plus_half_partial_over_all_items_and_a_skip_is_never_a_pass(run):
    results = [dict(theme="A", status="PASS"), dict(theme="A", status="PARTIAL"), dict(theme="B", status="FAIL"), dict(theme="B", status="PASS")]
    rows, tot = run.scorecard(results)
    assert tot == dict(n=4, PASS=2, PARTIAL=1, FAIL=1, points=2.5)
    assert {r["theme"]: r["percent"] for r in rows if r["n"]} == {"A": 75.0, "B": 50.0}
    assert set(run.VALUE) == {"PASS", "PARTIAL", "FAIL"}                 # there is no status that counts without a check
    assert run.TARGET == 0.90


def test_a_checker_that_crashes_is_a_failed_item_with_the_reason(run):
    class Boom:
        online = False
    status, detail = run.run_check(Boom(), dict(id="X1", check=dict(kind="check", lines=["x"])))
    assert status == "FAIL" and "could not run" in detail


def test_evidence_needs_every_key_and_a_named_person_where_the_item_is_about_human_judgement(run, tmp_path, monkeypatch):
    K = run.K
    monkeypatch.setattr(K, "ROOT", str(tmp_path))
    os.makedirs(tmp_path / "conformance" / "evidence")
    path = tmp_path / "conformance" / "evidence" / "B12.json"
    assert K.kind_evidence(None, dict(action="label 50 rollouts"), "B12")[0] == "FAIL"
    ev = dict(claim="50 rollouts labelled", n=50, result="pass", who="a coding agent", date="2026-11-25", artefact_sha256="0" * 64)
    path.write_text(json.dumps(ev))
    status, detail = K.kind_evidence(None, {}, "B12")
    assert status == "FAIL" and "named person" in detail
    path.write_text(json.dumps(dict(ev, who="Claude", who_is="person")))
    assert K.kind_evidence(None, {}, "B12")[0] == "FAIL"                # a model's name is not a person
    path.write_text(json.dumps(dict(ev, who="A. Physicist", who_is="person")))
    assert K.kind_evidence(None, {}, "B12")[0] == "PASS"
    del ev["artefact_sha256"]
    path.write_text(json.dumps(dict(ev, who="A. Physicist", who_is="person")))
    assert K.kind_evidence(None, {}, "B12")[0] == "FAIL"
    assert set(K.PERSON_ITEMS) == {"B12", "B13", "D9", "F8", "I5"}


def test_no_evidence_file_is_shipped_for_an_item_about_human_judgement_without_a_person(run):
    for item_id in run.K.PERSON_ITEMS:
        path = os.path.join(CONF, "evidence", f"{item_id}.json")
        if os.path.exists(path):
            assert json.load(open(path)).get("who_is") == "person"


def test_the_stored_report_matches_its_own_arithmetic_and_the_generated_readme_block(run):
    path = os.path.join(CONF, "report.json")
    if not os.path.exists(path):
        pytest.skip("no stored report")
    report = json.load(open(path))
    rows, tot = run.scorecard(report["items"])
    assert len(report["items"]) == 108 and tot == report["total"]
    assert report["percent"] == round(100 * tot["points"] / tot["n"], 1)
    assert run.write_readme(report, check=True), "integrations/README.md: the generated block is stale; run `make conformance`"
    base = os.path.join(CONF, "baseline.json")
    if os.path.exists(base):
        assert set(json.load(open(base))["pass"]) <= {r["id"] for r in report["items"] if r["status"] == "PASS"}
