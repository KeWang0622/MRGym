"""Cards, ``mrgym list`` and the generated index in the world without per-instance admission.

  * a card shows the stored gate table (schema ``mrgym-gate/2``: reference, must-fail policies with bounds, ablations,
    reported policies, the verdict per level), or says that no gate is stored on this engine
  * a gate measured on another engine is shown as an older result, not as evidence
  * ``mrgym list``: kind, levels of the claim, practice levels, the stored verdicts, the retired ids
  * the generated index and table carry kinds and verdicts, and no count of admitted instances anywhere

The gate result used here (``tests/data/gate/``) is a development run of ``Scanner/NoWrap-v1`` on 60 instances per
level. It is a fixture of the format: it is not stored with the package and backs no claim.
"""
import copy
import importlib.util
import json
import os
import subprocess

import pytest

import mrgym
from mrgym import cards, check, cli, gate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOWRAP, T1 = "Scanner/NoWrap-v1", "Scanner/T1Transmit-v1"


@pytest.fixture(autouse=True)
def nowrap_as_a_candidate(monkeypatch):
    """These tests describe an id whose gate is not stored yet (or stored from a development run): the registry entry of
    ``Scanner/NoWrap-v1`` is set back to ``candidate`` here, whatever the package registers it as today."""
    import dataclasses
    monkeypatch.setitem(mrgym.registry, NOWRAP, dataclasses.replace(mrgym.spec(NOWRAP), kind="candidate"))


@pytest.fixture
def stored(monkeypatch):
    """Makes ``gate.load(NOWRAP)`` return the fixture (on the current engine); returns the dict for edits."""
    with open(os.path.join(ROOT, "tests", "data", "gate", "Scanner-NoWrap-v1.gate.json"), encoding="utf-8") as f:
        g = json.load(f)
    g["engine"] = mrgym.ENGINE_ID
    load = gate.load
    monkeypatch.setattr(gate, "load", lambda env_id: g if env_id == NOWRAP else load(env_id))
    return g


def test_a_card_without_a_stored_gate_says_so(monkeypatch):
    monkeypatch.setattr(gate, "load", lambda env_id: None)
    for env_id, kind in ((NOWRAP, "candidate"), (T1, "practice")):
        card = mrgym.card(env_id)
        assert card["kind"] == kind and card["gate"] is None and cards.lint(card) == []
        assert card["status"] == f"{kind.upper()}: the gate has not been run (no stored result)"
        md = cards.markdown(card)
        assert f"**Gate not stored on this engine** (`{mrgym.ENGINE_ID}`). `mrgym gate {env_id} --n 300 --write` measures and stores it." in md
        assert all(f"| `{c['name']}` | {c['must']} | {c['what']} |" in md for c in card["controls"])     # the policies, without numbers
        assert "instances admitted" not in md and "Admitted instances" not in md
    assert "Kind of task: **measurement**" in cards.markdown(mrgym.card(T1)) and "Registered as practice" in cards.markdown(mrgym.card(T1))
    md = cards.markdown(mrgym.card(NOWRAP))
    assert "| 0 | no: practice level |" in md and "| 1 | no: practice level |" in md and "| 2 | yes |" in md and "| 3 | yes |" in md


def test_the_card_shows_the_stored_gate_table(stored):
    card = mrgym.card(NOWRAP)
    assert card["gate"] is stored and cards.lint(card) == [] and cards.gate_problems(card) == []
    assert card["status"].startswith("PRACTICE: level(s) 3 admitted; fixed_recipe#1 passes 3 of 60")
    md = cards.markdown(card)
    assert cards.GATE_NOT_STORED not in md.lower()
    assert "`mrgym gate Scanner/NoWrap-v1 --n 60 --write` (split gate, seeds 0..59;" in md
    assert "| Policy | Must | Level 2 | Level 3 | What it does |" in md
    row = lambda name: next(ln for ln in md.splitlines() if ln.startswith(f"| `{name}` |"))       # noqa: E731
    assert row("reference").startswith("| `reference` | pass | 60/60 (100.0 %; lower bound 95.1 %) | 60/60 (100.0 %; lower bound 95.1 %) |")
    # the recipe tuned on the set of other worlds that passes most often stands for the policy
    assert row("fixed_recipe").startswith("| `fixed_recipe` | fail | 3/60 (5.0 %; upper bound 12.4 %) NOT CLEARED | 1/60 (1.7 %; upper bound 7.7 %) |")
    assert row("prior").startswith("| `prior` | fail | 0/60 (0.0 %; upper bound 4.9 %) | 0/60 (0.0 %; upper bound 4.9 %) |")
    assert row("no_turn").startswith("| `no_turn` | ablation | 60/60 (100.0 %; 0.0 points below the reference, lower bound 0.0) not required here | "
                                     "21/60 (35.0 %; 65.0 points below the reference, lower bound 54.6) |")
    assert row("open_loop").startswith("| `open_loop` | reported | 60/60 (100.0 %) | 60/60 (100.0 %) |")
    assert ("Verdict per level: level 2 **practice** (fixed_recipe#1 passes 3 of 60 (upper bound 12.4 % > 10 %)); level 3 **admitted**.") in md
    assert "Blind ceiling (the must-fail policy that passes most often): level 2 `fixed_recipe#1` 5.0 % (upper bound 12.4 %); level 3" in md
    for c in card["controls"]:                                           # every listed policy has its cells
        assert row(c["name"]).count("/60") == 2, c["name"]


def test_a_control_without_a_stored_row_is_a_card_problem(stored):
    for lv in stored["levels"].values():
        lv["reported"] = [r for r in lv["reported"] if r["policy"] != "safe_max"]
    problems = cards.lint(mrgym.card(NOWRAP))
    assert problems == ["control safe_max has no stored result at level 2", "control safe_max has no stored result at level 3"]


def test_a_gate_from_another_engine_is_shown_as_an_older_result_not_as_evidence(stored):
    stored["engine"] = "mrgym-engine-0.1.0"
    card = mrgym.card(NOWRAP)
    assert cards.gate_problems(card) == [f"the stored gate was measured on engine mrgym-engine-0.1.0, not on {mrgym.ENGINE_ID}"]
    md = cards.markdown(card)
    assert f"**Gate not stored on this engine** (`{mrgym.ENGINE_ID}`): the stored gate was measured on engine mrgym-engine-0.1.0" in md
    assert "The table below is that older result." in md and "on engine `mrgym-engine-0.1.0`" in md
    row = next(r for r in cli.list_rows() if r["id"] == NOWRAP)
    assert row["gate"] == {"2": None, "3": None}
    spec = mrgym.spec(NOWRAP)
    admitted = type(spec)(**{**spec.__dict__, "kind": "admitted"})
    assert check.claim_evidence(admitted, stored, mrgym.ENGINE_ID)[0] is False


def test_a_missing_claim_level_is_named(stored):
    del stored["levels"]["3"]
    assert cards.gate_problems(mrgym.card(NOWRAP)) == ["the stored gate has no result for level(s) 3"]
    assert "Gate not stored on this engine" in cards.markdown(mrgym.card(NOWRAP))


def test_an_admitted_card_needs_its_gate(monkeypatch):
    monkeypatch.setattr(gate, "load", lambda env_id: None)
    card = dict(mrgym.card(NOWRAP), kind="admitted")
    assert cards.lint(card) == ["listed as admitted without a stored gate result (run `mrgym gate <id> --n 300 --write`)"]
    assert cards.lint(dict(card, kind="graded")) == ["kind must be one of ('sandbox', 'practice', 'admitted', 'candidate'), not 'graded'"]
    two = copy.deepcopy(dict(card, kind="candidate"))
    two["controls"].append(dict(name="second", must="pass", what="x"))
    assert "exactly one reference policy (a control that must pass) is needed" in cards.lint(two)
    assert cards.lint({k: v for k, v in card.items() if k != "gate"}) == ["missing field: gate"]
    assert cards.claim_levels(card) == [2, 3] and cards.slug(NOWRAP) == "scanner-nowrap-v1"


# ---------------------------------------------------------------------------------------------- mrgym list
def test_list_prints_kind_claim_levels_and_the_retired_ids(capsys, stored):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out.splitlines()
    line = next(i for i, ln in enumerate(out) if ln.startswith(NOWRAP))
    assert out[line].split()[:6] == [NOWRAP, "candidate", "levels", "0,1,2,3", "(claim:", "2,3)"] and "PRACTICE: level(s) 3 admitted" in out[line + 1]
    assert any(ln.split()[:4] == [T1, "practice", "levels", "1,2,3"] for ln in out)
    assert any(ln.split()[:3] == ["Scanner/Explore-v0", "sandbox", "levels"] for ln in out)
    assert [ln for ln in out if " retired " in ln] == [
        "Scanner/NoWrap-v0        retired   last served by mrgym 0.1.0; successor Scanner/NoWrap-v1",
        "Scanner/T1Transmit-v0    retired   last served by mrgym 0.1.0; successor Scanner/T1Transmit-v1"]
    assert cli.main(["list", "--json"]) == 0
    rows = {r["id"]: r for r in json.loads(capsys.readouterr().out)}
    assert set(rows) == set(mrgym.list_envs()) and all(r["kind"] in cards.KINDS for r in rows.values())
    assert rows[NOWRAP] == dict(id=NOWRAP, kind="candidate", levels=[0, 1, 2, 3], claim_levels=[2, 3], practice_levels=[0, 1],
                                title="The fastest geometry without wrap-around", status=rows[NOWRAP]["status"], task_kind="measurement",
                                gate={"2": "practice", "3": "admitted"})
    assert rows["Scanner/Explore-v0"]["gate"] is None and rows["Scanner/Explore-v0"]["claim_levels"] == []
    assert cli.main(["list", "--json", "--retired"]) == 0
    both = json.loads(capsys.readouterr().out)
    assert [r["id"] for r in both["environments"]] == mrgym.list_envs()
    assert both["retired"] == [dict(id="Scanner/NoWrap-v0", last_version="0.1.0", successor=NOWRAP),
                               dict(id="Scanner/T1Transmit-v0", last_version="0.1.0", successor=T1)]


def test_check_has_no_admission_flag_and_gate_is_its_own_command():
    with pytest.raises(SystemExit):
        cli.main(["check", NOWRAP, "--" + "admission"])
    with pytest.raises(SystemExit):
        cli.main(["gate", "--help"])


# ---------------------------------------------------------------------------------------------- generated files
def _generate():
    spec = importlib.util.spec_from_file_location("mrgym_generate", os.path.join(ROOT, "tools", "generate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_cards_index_and_table(stored):
    files, table = _generate().env_cards()
    ids = mrgym.list_envs()
    assert set(files) == {"envs/index.json", "envs/README.md"} | {f"envs/{cards.slug(i)}/README.md" for i in ids}
    index = json.loads(files["envs/index.json"])
    assert set(index) == {"environments", "retired"} and [e["id"] for e in index["environments"]] == ids
    assert [r["id"] for r in index["retired"]] == ["Scanner/NoWrap-v0", "Scanner/T1Transmit-v0"]
    entry = next(e for e in index["environments"] if e["id"] == NOWRAP)
    assert entry["gate"] == dict(engine=mrgym.ENGINE_ID, instances_per_level=60, verdicts={"2": "practice", "3": "admitted"})
    assert (entry["kind"], entry["claim_levels"], entry["practice_levels"], entry["task_kind"]) == ("candidate", [2, 3], [0, 1], "measurement")
    assert all("admitted_instances" not in e and "instances" not in e for e in index["environments"])
    lines = table.splitlines()
    assert lines[0] == "| Environment | Given | Deliver | Scored by | Kind | Levels of the claim | Stored gate |"
    assert next(ln for ln in lines if NOWRAP in ln).endswith("| candidate | 2, 3 (practice: 0, 1) | practice: level(s) 3 admitted; fixed_recipe#1 passes 3 of 60 "
                                                              "(upper bound 12.4 % > 10 %) |")
    assert next(ln for ln in lines if "Explore" in ln).endswith("| sandbox | - | no reward |")
    assert "## Retired ids" in files["envs/README.md"] and "`pip install mrgym==0.1.0` reproduces it. Successor: `Scanner/NoWrap-v1`." in files["envs/README.md"]
    assert "scanner-nowrap-v0" not in "".join(files)


def test_no_trace_of_the_per_instance_admission_is_left():
    """The acceptance search of the work package, kept as a test. The pattern is assembled here so that this file
    does not match itself."""
    pattern = "\\|".join(["admitted" + "_seeds", "instance" + "_list", "admission" + ".run", "--" + "admission"])
    out = subprocess.run(["grep", "-rn", "--exclude-dir=__pycache__", pattern, "src", "tools", "tests", "docs", "README.md"],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 1 and out.stdout == "", out.stdout[:2000]
    assert not os.path.exists(os.path.join(ROOT, "src", "mrgym", "admission.py"))
    results = os.listdir(os.path.join(ROOT, "src", "mrgym", "envs", "results"))
    assert not [n for n in results if n.endswith("-v0.json") and "NullTissue" not in n]      # the stored results of engine 0.1.0 are gone
