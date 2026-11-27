"""The map (``mrgym.physics_map``): one authored table of MRI physics and of where each task stands, and what is
generated from it: ``docs/tasks.md``, ``mrgym map``, the grouping of ``mrgym list``, the line on every card, the task
table for the front page. The map can file a family elsewhere without moving a fingerprint."""
import json
import pathlib
import re

import pytest

import mrgym
from mrgym import cards, cli
from mrgym import physics_map as M
from mrgym.compat import fingerprint as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCORED = [i for i in mrgym.list_envs() if mrgym.spec(i).family and mrgym.spec(i).kind != "sandbox"]
ID = {mrgym.spec(i).name: i for i in sorted(SCORED, key=lambda i: mrgym.spec(i).version)}       # family name -> its newest id; ids are never typed here


# ------------------------------------------------------------------ the table
def test_the_map_has_no_problem():
    assert M.validate(str(ROOT)) == []
    assert [s.letter for s in M.SUBJECTS] == ["A", "B", "C", "D"] and [k.number for k in M.KINDS] == [1, 2, 3, 4, 5]
    assert [k.name for k in M.KINDS] == ["measure", "set", "trade", "request", "design"]
    ids = [i.id for i in M.ITEMS]
    assert len(ids) == len(set(ids)) and len(ids) >= 80
    for s in M.SUBJECTS:                                                # ids are numbered once and without a gap; a new item takes the next number
        assert {i.id for i in M.ITEMS if i.id[0] == s.letter} == {f"{s.letter}{n}" for n in range(1, 1 + sum(i[0] == s.letter for i in ids))}
        assert len(M.BLOCKS[s.letter]) >= 2
    assert {i.status for i in M.ITEMS} == set(M.STATUSES) == {"formed", "partial", "computed", "absent"}       # no status says "cannot be formed"
    assert all(i.reason for i in M.ITEMS if i.status != "formed") and not any(i.reason for i in M.ITEMS if i.status == "formed")
    assert all(i.outlines is None or 1 <= i.outlines <= len(M.OUTLINES) for i in M.ITEMS) and len(M.OUTLINES) == 10


def test_every_formed_item_names_checks_that_exist():
    shown = [i for i in M.ITEMS if i.status in ("formed", "partial", "computed")]
    assert shown and all(i.evidence for i in shown)
    missing = [(i.id, e) for i in shown for e in i.evidence if M.evidence_exists(e, str(ROOT)) is not True]
    assert not missing, missing
    assert not any(i.evidence for i in M.ITEMS if i.status == "absent")
    forms = {"validate" if M._VALIDATE_ROW.match(e) else e.split("/")[0].split(" ")[0].split(":")[0] for i in shown for e in i.evidence}
    # the kinds of check, all in use: the rows of ``mrgym validate``, those of every validated package, tests, examples, commands
    # and the rows of the comparison with measured data
    assert forms == {"validate", *M.EVIDENCE_PACKAGES, "tests", "examples", "mrgym", "measured"} and list(M.EVIDENCE_PACKAGES) == ["design", "core", "spins", "em", "array", "fields3d"]
    for prefix in M.EVIDENCE_PACKAGES:
        cited = {e for i in M.ITEMS for e in i.evidence if e.startswith(prefix + ":")}
        assert cited and cited <= {f"{prefix}:{r}" for r in M.package_ids(prefix)} and all(M.package_row(e)[0] == prefix for e in cited), prefix
    assert len(M.package_ids("design")) > 60
    # the ids read from the source of ``mrgym.core.validate``, ``mrgym.spins.validate`` and ``mrgym.em.validate`` are the rows
    # of the tables in docs/physics/, which tests/test_core_validate.py, tests/test_spins_validate.py and tests/test_em_validate.py
    # hold equal to what the commands print; the receive arrays and the fields in 3D do not write every id out in their source
    # (a counter, a loop variable), so theirs are read from those tables (tests/test_array.py, tests/test_fields3d.py)
    for prefix, rows in (("core", 81), ("spins", 69), ("em", 82), ("array", 76), ("fields3d", 54)):
        table = (ROOT / "docs" / M.EVIDENCE_PACKAGES[prefix][2]).read_text(encoding="utf-8").split(M._TABLE_MARKS[0])[1].split(M._TABLE_MARKS[1])[0]
        assert M.package_ids(prefix) == set(re.findall(r"^\| ([A-Z]+\d+[a-z]?\d?) \|", table, re.M)) and len(M.package_ids(prefix)) == rows, prefix
    assert [p for p, line in M.EVIDENCE_PACKAGES.items() if line[1] is M._ON_THE_PAGE] == ["array", "fields3d"]
    source = {p: __import__(M.EVIDENCE_PACKAGES[p][0], fromlist=["x"]).__file__ for p in ("array", "fields3d")}
    assert 'f"P{len(rows) + 1}"' in pathlib.Path(source["array"]).read_text(encoding="utf-8")        # why: ids made by a counter
    assert 'id_ + "b"' in pathlib.Path(source["fields3d"]).read_text(encoding="utf-8")               # and from a loop variable
    assert M.evidence_exists("em:E3a2", str(ROOT)) is True and M.evidence_exists("array:P44", str(ROOT)) is True and M.evidence_exists("fields3d:F24b", str(ROOT)) is True
    assert M.package_ids("array", "/nonexistent") is None and M.evidence_exists("array:P3", "/nonexistent") is None      # no checkout, no page: not looked up
    assert M.evidence_exists("em:S10", "/nonexistent") is True                                       # an id of the source needs no checkout
    assert M.evidence_exists("core:N1", str(ROOT)) is True and M.evidence_exists("core:E9", str(ROOT)) is False     # E9 is a row of the design layer only
    assert M.evidence_exists("spins:S7", str(ROOT)) is True and M.evidence_exists("spins:N1", str(ROOT)) is False   # N1 is a row of the signal core only
    assert M.evidence_exists("spins:M4c", str(ROOT)) is True            # a row that the source names in a loop, not as ``_row("M4c"``
    assert M._evidence(("K6 K22", "design:E3 core:N1b", "spins:S7 K6", "mrgym qa acr")) == ("K6", "K22", "design:E3", "core:N1b", "spins:S7", "K6", "mrgym qa acr")
    for wrong in ("K999", "design:Z9", "core:Z9", "core:N99", "spins:Z9", "flow:P1", "em:Z9", "em:P44", "array:P99", "array:S10", "fields3d:F99", "fields3d:P3",
                  "tests/test_scanner.py::test_that_does_not_exist", "tests/no_such_file.py::test_x",
                  "examples/textbook/99_nothing.py", "mrgym qa acr --fault no_such_fault"):
        assert M.evidence_exists(wrong, str(ROOT)) is False, wrong
    assert M.evidence_exists("tests/test_scanner.py::test_plan_never_depends_on_the_world", "/nonexistent") is None     # no checkout: cannot say


def test_every_scored_family_has_one_row_and_one_cell():
    assert set(M.registered()) == set(M.FAMILIES) == {mrgym.spec(i).name for i in SCORED}
    assert not set(M.FAMILIES) & set(M.PLANNED)
    by_id = M.items()
    for env_id in SCORED:
        name, row = M.family_of(env_id)
        letter, block, number = M.cell(row)
        assert name == mrgym.spec(env_id).name and letter == row.isolates[0] and block == by_id[row.isolates].block and number == row.kind
        assert by_id[row.isolates].status in M.SERVED and row.isolates not in row.uses
        assert all(by_id[i].status in M.SERVED for i in row.uses)       # a task asks only for what the scanner serves
    assert M.family_of("Scanner/Explore-v0") is None and M.family_of("not an id") is None and M.card_line("Scanner/Explore-v0") is None
    for row in list(M.FAMILIES.values()) + list(M.PLANNED.values()):
        if row.kind >= 3:
            assert len({(i[0], by_id[i].block) for i in row.uses}) >= 2


def _patched(monkeypatch, **changes):
    for name, value in changes.items():
        monkeypatch.setattr(M, name, value)
    return M.validate(str(ROOT))


def test_validate_names_what_is_wrong(monkeypatch):
    item = M.items()

    def with_item(old, **fields):
        return tuple(i._replace(**fields) if i.id == old else i for i in M.ITEMS)
    with monkeypatch.context() as m:
        assert any("A1 rests on Z9, which is not an item" in p for p in _patched(m, ITEMS=with_item("A1", rests_on=("Z9",))))
    with monkeypatch.context() as m:
        problems = _patched(m, ITEMS=with_item("A1", rests_on=("A3",)))
        assert any("form a cycle" in p for p in problems) and any("A1 stands before A3" in p for p in problems)
    with monkeypatch.context() as m:
        assert any("A1: the check 'K999' does not exist" in p for p in _patched(m, ITEMS=with_item("A1", evidence=("K999",))))
    with monkeypatch.context() as m:
        assert any("A1: a reason is given exactly when" in p for p in _patched(m, ITEMS=with_item("A1", status="absent")))
    absent = next(i.id for i in M.ITEMS if i.status == "absent")
    with monkeypatch.context() as m:
        problems = _patched(m, FAMILIES=dict(M.FAMILIES, NoWrap=M.FAMILIES["NoWrap"]._replace(isolates=absent)))
        assert any(f"NoWrap names {absent}, which the scanner does not serve (absent)" in p for p in problems)
    computed = next(i.id for i in M.ITEMS if i.status == "computed")    # computed in the package, not served: no task can ask for it either
    with monkeypatch.context() as m:
        problems = _patched(m, FAMILIES=dict(M.FAMILIES, NoWrap=M.FAMILIES["NoWrap"]._replace(uses=(computed,))))
        assert any(f"NoWrap names {computed}, which the scanner does not serve (computed)" in p for p in problems)
    with monkeypatch.context() as m:
        assert any(f"{computed}: evidence is named exactly when" in p for p in _patched(m, ITEMS=with_item(computed, evidence=())))
    with monkeypatch.context() as m:
        one_block = tuple(i for i in M.FAMILIES["ArtefactClinic"].uses if item[i].block == "artefacts")
        problems = _patched(m, FAMILIES=dict(M.FAMILIES, ArtefactClinic=M.FAMILIES["ArtefactClinic"]._replace(uses=one_block)))
        assert any("ArtefactClinic is of kind 3 and uses items of 1 block(s)" in p for p in problems)
    with monkeypatch.context() as m:
        rest = {k: v for k, v in M.FAMILIES.items() if k != "SNRSpec"}
        problems = _patched(m, FAMILIES=rest)
        assert any(f"{ID['SNRSpec']}: the family SNRSpec is registered and has no row in FAMILIES" in p for p in problems)
        assert any("ProtocolToSpec builds on SNRSpec, which has no row" in p for p in problems)
    with monkeypatch.context() as m:
        problems = _patched(m, FAMILIES=dict(M.FAMILIES, NullTissue=M.FAMILIES["NullTissue"]._replace(builds_on=("FatSuppress",))))
        assert any("prerequisite families form a cycle" in p for p in problems)
    with monkeypatch.context() as m:
        assert any("NoWrap: kind 6" in p for p in _patched(m, FAMILIES=dict(M.FAMILIES, NoWrap=M.FAMILIES["NoWrap"]._replace(kind=6))))


def test_a_family_that_lands_moves_one_line_from_planned_to_families(monkeypatch):
    name = next(iter(M.PLANNED))
    env_id = f"Scanner/{name}-v0"
    reg = dict(M.registered(), **{name: [env_id]})
    monkeypatch.setattr(M, "registered", lambda: reg)
    assert M.validate(str(ROOT)) == [f"{env_id}: the family {name} is registered and has no row in FAMILIES: move its line from PLANNED"]
    monkeypatch.setattr(M, "FAMILIES", dict(M.FAMILIES, **{name: M.PLANNED[name]}))
    monkeypatch.setattr(M, "PLANNED", {k: v for k, v in M.PLANNED.items() if k != name})
    assert M.validate(str(ROOT)) == []
    monkeypatch.setattr(M, "registered", lambda: {k: v for k, v in reg.items() if k != name})
    assert M.validate(str(ROOT)) == [f"{name} has a row in FAMILIES and no registered environment: its line belongs in PLANNED"]


# ------------------------------------------------------------------ counts
def test_coverage_counts_are_recounts_of_the_table():
    c = M.coverage()
    isolated = {f.isolates for f in M.FAMILIES.values()}
    used = isolated | {i for f in M.FAMILIES.values() for i in f.uses}
    assert c["families"] == len(M.FAMILIES) and c["total"]["listed"] == len(M.ITEMS)
    for letter, n in c["subjects"].items():
        mine = [i for i in M.ITEMS if i.id[0] == letter]
        assert n["listed"] == len(mine) == n["formed"] + n["partial"] + n["computed"] + n["absent"]
        assert n["isolated"] == sum(i.id in isolated for i in mine) and n["used"] == sum(i.id in used for i in mine) >= n["isolated"]
        assert set(n["blocks"]) == set(M.BLOCKS[letter])
        for key in ("listed", "formed", "partial", "computed", "absent", "isolated", "used"):
            assert n[key] == sum(b[key] for b in n["blocks"].values())
            assert c["total"][key] == sum(s[key] for s in c["subjects"].values())
    assert c["total"]["isolated"] == len(isolated) <= len(M.FAMILIES)
    more = M.coverage(planned=True)
    assert more["families"] == len(M.FAMILIES) + len(M.PLANNED) and more["total"]["isolated"] >= c["total"]["isolated"]
    assert more["total"]["listed"] == c["total"]["listed"] and more["total"]["formed"] == c["total"]["formed"]
    taken = {f.isolates for f in M.FAMILIES.values()}
    assert all(i.status == "formed" and i.id not in taken for i in M.not_isolated())
    assert [M.dependents()[i.id] for i in M.not_isolated()] == sorted((M.dependents()[i.id] for i in M.not_isolated()), reverse=True)
    assert all(i.status in ("computed", "absent") for i in M.missing()) and M.missing()[0].outlines == max(i.outlines or 0 for i in M.missing())


def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def test_the_page_under_docs_is_current_and_its_counts_are_those_of_coverage():
    page = (ROOT / "docs" / "tasks.md").read_text(encoding="utf-8")
    assert page == M.doc_page(), "regenerate: python tools/generate.py"
    assert page.startswith("<!-- Generated by tools/generate.py")
    c = M.coverage()
    lines = page.splitlines()
    start = lines.index("| Subject | Listed | Formed | Isolated by a task | Used by a task |")
    rows = [_cells(ln) for ln in lines[start + 2:start + 3 + len(M.SUBJECTS)]]
    for s, row in zip(M.SUBJECTS, rows):
        n = c["subjects"][s.letter]
        assert row[0].startswith(f"**{s.letter}** {s.title}")
        assert [int(row[1]), int(row[2].split()[0]), int(row[3]), int(row[4])] == [n["listed"], n["formed"], n["isolated"], n["used"]]
        assert ("in part" in row[2]) == bool(n["partial"]) and (not n["partial"] or f"+{n['partial']} in part" in row[2])
    t = c["total"]
    assert rows[-1][0] == "All" and [int(rows[-1][1]), int(rows[-1][2].split()[0]), int(rows[-1][3]), int(rows[-1][4])] == [
        t["listed"], t["formed"], t["isolated"], t["used"]]
    assert (f"The list holds {t['listed']} pieces" in page and f"The scanner forms {t['formed']} of them and {t['partial']} more in part. The other "
            f"{t['computed'] + t['absent']} are not formed yet: {t['computed']} of them the package already computes and checks" in page)
    assert "no image of its sequences has been compared pixel by pixel with a scan of the same object." in page and "(physics/measured.md)" in page
    # the grid: every built family is linked in its cell, every planned one stands in italics in its cell
    grid = {_cells(ln)[0][2]: _cells(ln)[1:] for ln in lines if re.match(r"^\| \*\*[A-D]\*\* ", ln) and ln.count("|") == 2 + len(M.KINDS)}
    assert set(grid) == {"A", "B", "C", "D"}
    for name, row in M.FAMILIES.items():
        letter, _, number = M.cell(row)
        assert f"[{name}](../envs/" in grid[letter][number - 1]
    for name, row in M.PLANNED.items():
        letter, _, number = M.cell(row)
        assert f"*{name}*" in grid[letter][number - 1]
    # every item once in its subject table, every reason once in its list
    for i in M.ITEMS:
        assert sum(ln.startswith(f"| {i.id} | {i.name} | ") for ln in lines) == (1 if i.status == "formed" else 2), i.id
    def ids_under(part):
        return [ln.split(" | ")[0][2:] for ln in part.splitlines() if re.match(r"^\| [A-D]\d+ \|", ln)]
    assert ids_under(page.split("## Formed in part")[1].split("\n## ")[0]) == [i.id for i in M.ITEMS if i.status == "partial"]
    # one list of what is not formed yet, grouped by the piece of engine work each item waits for, with the count of each piece
    waits = M.waiting()
    last = page.split("\n## Not formed yet\n")[1].split("\n## ")[0]
    assert sorted(ids_under(last)) == sorted(i.id for i in M.ITEMS if i.status in ("computed", "absent")) and "out of reach" not in page.lower()
    counts = {_cells(ln)[0].strip("*"): int(_cells(ln)[2]) for ln in last.split("\n### ")[0].splitlines() if re.match(r"^\| .* \| \d+ \|$", ln)}
    assert counts == {name or M.ORDINARY: len(ids) for name, ids in waits.items()}
    groups = {part.split("\n")[0]: ids_under(part) for part in last.split("\n### ")[1:]}
    assert groups == {(name or M.ORDINARY)[0].upper() + (name or M.ORDINARY)[1:]: ids for name, ids in waits.items() if ids}
    n_piece = sum(len(v) for k, v in waits.items() if k)
    pieces = sum(bool(v) for k, v in waits.items() if k)                 # the pieces that an item which is not formed waits for
    assert f"{n_piece + len(waits[''])} items. {n_piece} of them wait for one of {pieces} pieces of engine work; the other {len(waits[''])} take" in last
    # "computed ... and checks it against a closed form or a second method": every computed item names a row of a validated
    # package, and the list links the table of the checks, where those packages are named once
    assert f"[the checks](#{M._anchor(M.CHECKS_HEADING)}) name the rows." in last and f"\n## {M.CHECKS_HEADING}\n" in page
    assert all(any(M.package_row(e) for e in i.evidence) for i in M.ITEMS if i.status == "computed")
    for i in M.ITEMS:                                                   # "computed" is said beside the item, in the same list
        if i.status in ("computed", "absent"):
            row = next(_cells(ln) for ln in last.splitlines() if ln.startswith(f"| {i.id} | "))
            assert row[2] == ("computed" if i.status == "computed" else "-") and row[3].lower() == i.reason.lower(), i.id


def test_what_is_not_formed_waits_for_one_of_seven_pieces_or_for_ordinary_work():
    names = [name for name, _ in M.ENABLERS]
    assert names == ["3D world", "receive array", "spin systems", "field model", "core in the console", "realistic anatomy", "measured data"]
    # the signal core exists as a library (``mrgym.core``): what its piece still asks for is serving it, and the page says so
    what = dict(M.ENABLERS)["core in the console"]
    assert "exists as a library" in what and "still to do" in what and "block-list core" not in M.doc_page() + M.text_table(M.table(str(ROOT)))
    # so do the spin systems (``mrgym.spins``): what is left is the voxel and the sequence
    what = dict(M.ENABLERS)["spin systems"]
    assert "exists as a library" in what and "still to do: a voxel that holds metabolites in the scanner world and a spectroscopy sequence in the console" in what
    # so do the volume world with its fields, the receive arrays and the RF field model: each piece says what exists and what is left
    for name, exists, left in (("3D world", "exist as libraries", "the slice engine that scans the volume"),
                               ("receive array", "exist as a library", "the console serving them"),
                               ("field model", "exist as a library, in two dimensions", "the scanner imaging and pricing RF with these fields")):
        what = dict(M.ENABLERS)[name]
        assert exists in what and f"still to do: {left}" in what, name
    assert "still to do: a scan of the same object with the console's sequences, compared pixel by pixel" in dict(M.ENABLERS)["measured data"]
    page = M.doc_page()
    assert all(f"| **{name}** | {what} | {len(M.waiting()[name])} |" in page for name, what in M.ENABLERS)       # the page and the command show the same list
    assert all(re.search(rf"^\s+{len(M.waiting()[name])}  {re.escape(name)}$", M.text_table(M.table(str(ROOT))), re.M) for name, _ in M.ENABLERS)
    waits = M.waiting()
    assert list(waits) == names + [""] and all(waits[name] for name in names[:-1])               # every piece unlocks something that is listed
    c33 = M.items()["C33"]                                                                       # measured data has arrived in part: its one item is formed in part
    assert waits["measured data"] == [] and (c33.status, c33.needs) == ("partial", "measured data") and all(e.startswith("measured:M") for e in c33.evidence)
    stored = json.loads((ROOT / "docs" / "physics" / "measured" / "results.json").read_text(encoding="utf-8"))["rungs"]
    assert {e.split(":")[1] for e in c33.evidence} == {r["rung"].replace(" ", "") for r in stored if r["rows"]}      # every rung of the stored table, and no other
    assert "M3b" in {r["rung"] for r in stored if "design layer" in r["title"]} and "with the design layer" in c33.reason      # the rung whose title names the design layer
    by_id = M.items()
    assert sorted(i for ids in waits.values() for i in ids) == sorted(i.id for i in M.ITEMS if i.status in ("computed", "absent"))
    assert all(by_id[i].needs == name for name, ids in waits.items() for i in ids)
    assert all(not i.needs for i in M.ITEMS if i.status == "formed") and all(i.needs in names + [""] for i in M.ITEMS)
    assert [i.id for i in M.ITEMS if "compared pixel by pixel with a scan of the same object" in i.reason] == ["C33"]
    for status, piece in (("formed", "3D world"), ("absent", "a fourth dimension")):
        patched = tuple(i._replace(needs=piece) if i.id == ("A1" if status == "formed" else waits[""][0]) else i for i in M.ITEMS)
        with pytest.MonkeyPatch.context() as m:
            m.setattr(M, "ITEMS", patched)
            assert any("needs is one of" in p for p in M.validate(str(ROOT))), status


def test_what_the_signal_core_computes_is_computed_or_stays_in_part():
    """Two exchanging pools are computed by the core and served by no sequence; what the console serves in part stays
    "in part" with the core's rows beside it; what no row of the core covers stays "not yet"."""
    by_id = M.items()
    assert by_id["A20"].status == "computed" and {"core:X6", "core:X6b", "core:M2", "core:M3"} <= set(by_id["A20"].evidence)
    assert "one pool of protons" in by_id["A20"].reason and "chemical exchange between its evaluators only" in by_id["A20"].reason
    for i in ("A13", "A14", "A15", "A21"):                              # computed before, by the design layer or a formula: further rows
        assert by_id[i].status == "computed" and any(e.startswith("core:") and M.package_row(e) for e in by_id[i].evidence), i
    assert by_id["C20"].status == "partial" and any(e.startswith("core:") and M.package_row(e) for e in by_id["C20"].evidence)
    assert "the console does not serve these yet" in by_id["C20"].reason
    for i in ("A19", "A22", "B19"):                                     # spatial saturation, flow, the order of the lines
        assert by_id[i].status == "absent" and not by_id[i].evidence, i
    cell = M._evidence_text(by_id["A21"].evidence)
    assert cell == "`python -m mrgym.design.validate` G5, G6, E10; `python -m mrgym.core.validate` X7, N11, N11c, N12"
    assert f"| A21 | {cell} |" in M.doc_page() and "`python -m mrgym.core.validate` ([core](physics/core.md))" in M.doc_page()


def test_validated_packages_are_one_table(monkeypatch):
    """``EVIDENCE_PACKAGES`` is the one place that knows a validated package: the pattern of a row, the existence check,
    the cell of an item and the sentence above the checks follow from a line of it."""
    page = M.doc_page()
    legend = page.split(f"\n## {M.CHECKS_HEADING}\n")[1].split("\n| |")[0]
    for prefix, (_, _, doc, command) in M.EVIDENCE_PACKAGES.items():
        ids = M.package_ids(prefix)
        assert ids and all(M.package_row(f"{prefix}:{r}") == (prefix, r) for r in ids), prefix
        shown = (ROOT / "docs" / doc).read_text(encoding="utf-8")
        assert all(re.search(rf"^\| {r} \|", shown, re.M) for r in ids), prefix                    # every row that can be cited is on its page
        assert f"of `{command}` ([{prefix}]({doc}))" in legend and page.count(f"({doc})") == 1     # named once, with its command
    assert legend.strip() == M._row_sources() + ", tests, textbook examples that assert their result, commands."
    assert M._row_sources().startswith("Rows of `mrgym validate` ([validation](physics/validation.md)), of `python -m mrgym.design.validate` ")
    assert M.package_row("K6") is None and M.package_row("design:e3") is None and M.package_row("flow:P1") is None and M.package_problems(str(ROOT)) == []
    # a new package is one line: here the rows of the spin systems under a second prefix
    line = ("mrgym.spins.validate", M._ALSO_IN_A_LOOP, "physics/spins.md", "python -m mrgym.flow.validate")
    with monkeypatch.context() as m:
        m.setattr(M, "EVIDENCE_PACKAGES", dict(M.EVIDENCE_PACKAGES, flow=line))
        m.setattr(M, "ITEMS", tuple(i._replace(evidence=M._evidence(("K18 flow:S7 flow:M4c core:N1",))) if i.id == "A2" else i for i in M.ITEMS))
        assert M.package_row("flow:S7") == ("flow", "S7") and M.evidence_exists("flow:S7", str(ROOT)) is True and M.evidence_exists("flow:N1", str(ROOT)) is False
        assert M.items()["A2"].evidence == ("K18", "flow:S7", "flow:M4c", "core:N1") and M.validate(str(ROOT)) == []
        assert M._evidence_text(M.items()["A2"].evidence) == "`mrgym validate` K18; `python -m mrgym.core.validate` N1; `python -m mrgym.flow.validate` S7, M4c"
        assert " and of `python -m mrgym.flow.validate` ([flow](physics/spins.md)), tests, " in M.doc_page()
    # a package is listed when it is merged, not before; and a line that finds nothing, or shows nothing, is refused
    for wrong, said in ((("mrgym.no_such_package.validate",) + line[1:], "which cannot be imported: list a package when it is merged"),
                        ((line[0], r'_no_row\("([A-Z]+)"') + line[2:], "finds no row id in the source of mrgym.spins.validate"),
                        (line[:2] + ("physics/no_such_page.md", line[3]), "the page docs/physics/no_such_page.md of 'flow' does not exist"),
                        # ids read from the page: a page without the two marks of a validation table shows none
                        ((line[0], M._ON_THE_PAGE, "physics/conventions.md", line[3]), "finds no row id in the table of docs/physics/conventions.md")):
        with monkeypatch.context() as m:
            m.setattr(M, "EVIDENCE_PACKAGES", dict(M.EVIDENCE_PACKAGES, flow=wrong))
            problems = M.validate(str(ROOT))
            assert len(problems) == 1 and problems[0].startswith("EVIDENCE_PACKAGES: ") and said in problems[0], problems
            assert M.evidence_exists("flow:S7", str(ROOT)) is (wrong[0] == line[0] and wrong[1] == line[1])


def test_what_the_spin_systems_compute_is_computed_and_not_served():
    """Single-voxel spectroscopy, J-coupling with editing, and quantification with a basis are computed by ``mrgym.spins``
    and checked there; no console sequence reaches them, and the reasons say what is missing."""
    by_id = M.items()
    assert [i.id for i in M.ITEMS if i.block == "contrast beyond relaxation"][-3:] == ["A28", "A31", "A32"]     # new items take the next numbers
    assert by_id["A29"].name.startswith("Relaxation-time mapping") and by_id["A30"].name.startswith("Fat fraction")     # and no id moves
    for i in ("A28", "A31", "A32"):
        assert by_id[i].status == "computed" and by_id[i].needs == "spin systems" and by_id[i].reason.startswith("the spin-system package "), i
        assert by_id[i].evidence and all(M.package_row(e) and M.package_row(e)[0] == "spins" for e in by_id[i].evidence), i
        assert M.tasks_of(i, planned=True) == ([], [])                  # no task, built or in construction, asks for it
    assert M.waiting()["spin systems"] == ["A28", "A31", "A32"]
    assert by_id["A28"].name == "Spectroscopy: single voxel and spectroscopic imaging" and by_id["A28"].outlines == 8
    assert {"spins:S3", "spins:S5", "spins:S6"} <= set(by_id["A28"].evidence)                      # PRESS, STEAM, semi-LASER
    assert {"spins:S1", "spins:Q1", "spins:Q4"} <= set(by_id["A28"].evidence)                      # a coupled spin, a fit, a concentration
    for missing in ("the scanner world has no voxel that holds metabolites", "the console has no spectroscopy sequence", "spectroscopic imaging is not computed"):
        assert missing in by_id["A28"].reason, missing
    assert {"spins:E6", "spins:S7", "spins:S7b", "spins:S8"} <= set(by_id["A31"].evidence) and "no console sequence acquires a spectrum" in by_id["A31"].reason
    assert {"spins:Q1", "spins:Q2", "spins:Q3", "spins:Q4"} <= set(by_id["A32"].evidence) and "the scanner returns no spectrum" in by_id["A32"].reason
    # rows that show only that amplitudes add up to the number of protons, or how long a simulation takes, are not evidence
    cited = {e for i in M.ITEMS for e in i.evidence if e.startswith("spins:")}
    assert not cited & {"spins:M2", "spins:M2b", "spins:M4", "spins:M4b", "spins:M4c", "spins:M4d"}
    cell = "`python -m mrgym.spins.validate` S1, S3, S4, S5, S6, Q1, Q4"
    assert M._evidence_text(by_id["A28"].evidence) == cell and f"| A28 | {cell} |" in M.doc_page()


def _page_rows(prefix):
    """{row id: the cells of its row} of the validation table on the page of a package."""
    text = (ROOT / "docs" / M.EVIDENCE_PACKAGES[prefix][2]).read_text(encoding="utf-8").split(M._TABLE_MARKS[0])[1].split(M._TABLE_MARKS[1])[0]
    return {_cells(ln)[0]: _cells(ln) for ln in text.splitlines() if re.match(r"^\| [A-Z]+\d+[a-z]?\d? \|", ln)}


def test_what_the_field_model_computes_is_computed_and_not_served():
    """``mrgym.em``: B1+ and B1- from Maxwell's equations, SAR in a body, RF shimming under limits and electrical
    properties are computed on the slice; the scanner still images with declared patterns and prices RF as B1+rms."""
    by_id = M.items()
    assert [i.id for i in M.ITEMS if i.block == "RF chain"][-3:] == ["C14", "C34", "C36"]              # new items take the next numbers
    assert [i.id for i in M.ITEMS if i.block == "operating limits"] == ["C22", "C23", "C35", "C24", "C25", "C26", "C27"]
    assert by_id["C28"].name.startswith("Phantom test: geometric accuracy") and by_id["C33"].name.startswith("Agreement with measurement")     # and no id moves
    for i in ("C23", "C34", "C35", "C36"):
        assert by_id[i].status == "computed" and by_id[i].needs == "field model" and by_id[i].reason.startswith("the field model "), i
        assert by_id[i].evidence and all(M.package_row(e) and M.package_row(e)[0] == "em" for e in by_id[i].evidence), i
        assert M.tasks_of(i, planned=True) == ([], [])                  # no task, built or in construction, asks for it
        assert "two dimensions" in by_id[i].reason or "two-dimensional" in by_id[i].reason or i == "C36", i      # its own stated limit
    assert M.waiting()["field model"] == ["C34", "C36", "C23", "C35", "C26"]
    assert by_id["C23"].outlines == 9 and "no number of it is a safety figure" in by_id["C23"].reason and "The console still prices RF as B1+rms only" in by_id["C23"].reason
    assert {"em:S13", "em:S13b", "em:S13c", "em:A10", "em:S7"} <= set(by_id["C23"].evidence)          # averaged SAR, the law of the frequency, the power balance
    assert {"em:A4", "em:A11b", "em:S3b", "em:S10", "em:S10c", "em:S11"} <= set(by_id["C34"].evidence)   # B1+ and B1-, the wavelength, central brightening
    assert "the scanner still images with declared transmit and receive patterns" in by_id["C34"].reason
    assert {"em:H1", "em:H3", "em:H3c", "em:H3e", "em:H5b", "em:H5c"} <= set(by_id["C35"].evidence) and by_id["C35"].rests_on == ("C23", "C34")
    assert "spatially selective parallel transmission) are not built" in by_id["C35"].reason and "one transmit channel" in by_id["C35"].reason
    assert {"em:E1", "em:E1b", "em:E3a"} <= set(by_id["C36"].evidence) and "without noise" in by_id["C36"].reason
    # what the console serves in part stays "in part", with the rows of the field model beside it
    assert by_id["C22"].status == "partial" and {"C1", "C5", "em:A10", "em:S12"} == set(by_id["C22"].evidence) and "it has no figure in W/kg" in by_id["C22"].reason
    assert by_id["C6"].status == by_id["C10"].status == "formed"        # the declared patterns are served
    # what the field model does not reach keeps its status: a device (still this piece: three dimensions, conductors), and the
    # field of the magnet outside the bore, which is not an RF field and waits for no piece of the list
    assert (by_id["C26"].status, by_id["C26"].needs) == ("absent", "field model") and "one component, along z" in by_id["C26"].reason
    assert (by_id["C27"].status, by_id["C27"].needs) == ("absent", "") and "the field of the magnet itself" in by_id["C27"].reason
    # not evidence: a row that states a time, one that only shows that another row can fail, one that agrees with a published size
    # within a factor, one that holds by construction
    cited = {e for i in M.ITEMS for e in i.evidence if e.startswith("em:")}
    assert not cited & {"em:S14", "em:C4b", "em:S10b", "em:C6c", "em:H5"}
    rows = _page_rows("em")
    assert all(rows[e[3:]][-1] == "pass" for e in cited)
    cell = "`python -m mrgym.em.validate` E1, E1b, E2, E3, E3a, E3a2, E3b, E4"
    assert M._evidence_text(by_id["C36"].evidence) == cell and f"| C36 | {cell} |" in M.doc_page()


def test_what_the_array_library_computes_is_computed_and_not_served():
    """``mrgym.scanner.array`` and ``recon``: coil elements with their noise covariance, the combination of the channels,
    SENSE and the g-factor are computed; the console has one receive channel. GRAPPA is not built."""
    by_id = M.items()
    for i in ("C11", "B22"):
        assert by_id[i].status == "computed" and by_id[i].needs == "receive array" and by_id[i].reason.startswith("the array library "), i
        assert {M.package_row(e)[0] for e in by_id[i].evidence} == {"array", "em"} and "one receive channel" in by_id[i].reason, i
        assert M.tasks_of(i, planned=True) == ([], [])
    assert M.waiting()["receive array"] == ["B22", "C11"] and by_id["C10"].status == "formed"
    assert "noise covariance" in by_id["C11"].reason and "the two are not joined" in by_id["C11"].reason and "no coil to choose" in by_id["C11"].reason
    assert "(GRAPPA) and the acceleration of several slices at once are not built" in by_id["B22"].reason and "no acceleration factor" in by_id["B22"].reason
    # a row that is reported without a tolerance has no verdict, and a row that states a time or a memory is not physics: not cited
    rows, cited = _page_rows("array"), {e[6:] for i in M.ITEMS for e in i.evidence if e.startswith("array:")}
    assert cited and all(rows[r][-1] == "pass" for r in cited) and not any(r.startswith("T") for r in cited)
    assert not cited & {"C1", "C3", "P5", "P10", "P36"}                  # an order that theory fixes between two closed forms; a value recorded for the prototype
    assert sum(r[-1] != "pass" for r in rows.values()) == 34            # the rows the command calls "reported"
    # the rows of SENSE are numbered by a counter: a row added before them would move every number, so what each cited one says is held here
    said = dict(P1="residual of the unfolding when the data are the maps times an object", P2="the same by conjugate gradients", P3="g of two pixels folded onto each other",
                P4="two channels, each seeing one of the two folded pixels", P6="R = 2: g measured by", P7="R = 2: SNR_full / SNR_R / (g sqrt(R))",
                P11="R = 3: g measured by", P12="R = 3: SNR_full / SNR_R / (g sqrt(R))", P16="8 loops on a ring, R = 2: g measured by",
                P17="8 loops on a ring, R = 2: SNR_full / SNR_R / (g sqrt(R))", P21="8 loops on a ring, R = 3: g measured by",
                P22="8 loops on a ring, R = 3: SNR_full / SNR_R / (g sqrt(R))", P33="R = 2: mean g with the phase direction along y / along x",
                P34="R = 3: mean g with the phase direction along y / along x")
    assert {r for r in cited if r.startswith("P")} == set(said) and all(text in rows[r][1] for r, text in said.items())
    assert {"array:N1", "array:C7", "array:C9", "array:C11", "em:C4", "em:C5"} <= set(by_id["C11"].evidence)      # covariance, combination, the optimum
    assert {"em:C6", "em:C6b"} <= set(by_id["B22"].evidence)            # the g-factor of maps from Maxwell's equations
    assert f"| B22 | `python -m mrgym.em.validate` C6, C6b; `python -m mrgym.scanner.array_checks` {', '.join(said)} |" in M.doc_page()


def test_the_volume_world_and_the_fields_in_3d_exist_and_nothing_is_scanned_in_3d():
    """``volume``: a seeded 3D world, of which no scan is formed: what needs a slice in a volume stays "not yet" and says
    that the slice engine is the missing part. ``fields3d``: the susceptibility field and the shim are computed."""
    by_id = M.items()
    for i in ("B4", "B5", "C30", "D19", "D20", "A24"):                  # several slices, slice position, slice tests, cross-talk, metal, inflow
        assert (by_id[i].status, by_id[i].needs) == ("absent", "3D world") and not by_id[i].evidence and "slice" in by_id[i].reason, i
    for i in ("B4", "B5", "C30", "D19"):
        assert "exists as a library" in by_id[i].reason or "as a library" in by_id[i].reason, i
        assert "slice engine" in by_id[i].reason, i
    assert "no scan is formed of them" in by_id["B5"].reason and "ramps and crossed wedges" in by_id["C30"].reason
    assert not any(i.reason in ("a scan is one slice", "no test object has structure through the slice") for i in M.ITEMS)
    # shimming: computed to second order, with the engine's own first-order shim as one of its rows
    c4 = by_id["C4"]
    assert (c4.status, c4.needs) == ("computed", "") and c4.evidence == tuple(f"fields3d:F{n}" for n in range(33, 39))
    assert "first or second order" in c4.reason and "the console has no shim setting" in c4.reason
    assert "A first-order setting is ordinary work on the present engine; a second-order one waits for the slice engine on the 3D world" in c4.reason
    # the field of an object from its susceptibility: served for a few air spheres, computed for any distribution
    c5 = by_id["C5"]
    assert c5.status == "partial" and "R11" in c5.evidence and {"fields3d:F1", "fields3d:F6", "fields3d:F12", "fields3d:F15"} <= set(c5.evidence)
    assert "tests/test_fields3d.py::test_the_seeded_volume_world_end_to_end" in c5.evidence and "no scan uses that field yet" in c5.reason
    # susceptibility mapping: the forward model and a reference inversion, and the list of what is not there
    a26 = by_id["A26"]
    assert (a26.status, a26.needs) == ("computed", "3D world") and a26.evidence == ("fields3d:F1", "fields3d:F43", "fields3d:F44")
    assert a26.reason.startswith("what exists, in the 3D field library: the forward model") and "one reference inversion, truncated division in k-space" in a26.reason
    for missing in ("removal of the background field", "masks", "unwrapping of the phase", "any regularised inversion", "a scan that gives the phase of a volume"):
        assert missing in a26.reason.split("What does not: ")[1], missing
    assert by_id["A25"].status == "partial" and "no scan is formed of it" in by_id["A25"].reason
    assert M.waiting()["3D world"] == ["A24", "A26", "B4", "B5", "C30", "D19", "D20"] and "C4" in M.waiting()[""]
    cited = {e[9:] for i in M.ITEMS for e in i.evidence if e.startswith("fields3d:")}
    assert not cited & {"F45", "F16c", "F23", "F24b", "F25b", "F32b", "F3", "F42"}       # a time, the rows that show a plainer method fail, a figure of the specification
    rows = _page_rows("fields3d")
    assert all(rows[r][-1] == "pass" for r in cited)


def test_a_family_that_uses_an_item_is_not_contradicted_by_what_the_item_says_is_missing():
    """Gradient non-linearity is formed in part (one declared term) and ``GradientCal`` has it at its last level."""
    by_id = M.items()
    c17 = by_id["C17"]
    assert (c17.status, c17.needs) == ("partial", "") and "C17" in M.FAMILIES["GradientCal"].uses and M.tasks_of("C17") == ([], ["GradientCal"])
    assert c17.evidence == ("tests/test_gradient_cal.py::test_under_non_linear_gradients_the_edge_of_the_disc_is_not_at_the_scale_of_the_isocentre",)
    assert "one declared term" in c17.reason and "no correction of the distortion" in c17.reason
    from mrgym.envs.gradient_cal import task
    assert task.LEVELS[3]["nonlinear"] and not any(task.LEVELS[k]["nonlinear"] for k in (0, 1, 2))
    used = {f.isolates for f in M.FAMILIES.values()} | {i for f in M.FAMILIES.values() for i in f.uses}
    for i in M.ITEMS:                                                   # no item that a task takes says that no task has it
        if i.id in used:
            assert not re.search(r"no task (has|asks|measures|uses|states)", i.reason), i.id


def test_the_page_uses_plain_words():
    """Outside the names of commands, tests and files (in backticks), no word of the workshop."""
    page = re.sub(r"`[^`]*`", "", M.doc_page()).lower()
    for word in ("gate", "admitted", "must-fail", "must_fail", "candidate", "ablation", "oracle", "fingerprint", "cannot", "out of reach"):
        assert word not in page, word


# ------------------------------------------------------------------ what else is generated
def test_every_card_carries_its_place_on_the_map():
    for env_id in SCORED:
        line = M.card_line(env_id)
        name, row = M.family_of(env_id)
        letter, block, number = M.cell(row)
        assert line.startswith(f"**{letter} · {M.subject(letter).title}** · {block} · kind {number}, {M.kind(number).name} · isolates {row.isolates}: ")
        md = cards.markdown(mrgym.card(env_id)).splitlines()
        assert md[1].startswith(f"# `{env_id}`") and md[3] == line and md[5].startswith("**Status:**")
        on_disk = (ROOT / "envs" / cards.slug(env_id) / "README.md").read_text(encoding="utf-8").splitlines()
        assert on_disk[3] == line, f"regenerate: python tools/generate.py ({env_id})"
    sandbox = cards.markdown(mrgym.card("Scanner/Explore-v0")).splitlines()
    assert sandbox[3].startswith("**Status:**") and "isolates" not in "\n".join(sandbox[:6])


def test_list_is_grouped_by_subject_and_flat_on_request(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out.splitlines()
    heads = [ln for ln in out if re.match(r"^[A-D\-?]  ", ln)]
    assert heads == [f"{s.letter}  {s.title}" for s in M.SUBJECTS] + ["-  Sandbox"]
    seen = None
    for ln in out:
        if ln in heads:
            seen = ln[0]
        elif ln.startswith("Scanner/") and " retired " not in ln:
            env_id = ln.split()[0]
            found = M.family_of(env_id)
            if found:
                letter, _, number = M.cell(found[1])
                assert seen == letter and f" {number} {M.kind(number).name} " in ln, ln
            else:
                assert seen == "-" and env_id == "Scanner/Explore-v0"
    assert [ln.split()[0] for ln in out if ln.startswith("Scanner/") and " retired " not in ln] == [i for _, _, rows in M.groups() for i, _ in rows]
    assert sorted(i for _, _, rows in M.groups() for i, _ in rows) == mrgym.list_envs()
    assert cli.main(["list", "--flat"]) == 0
    flat = capsys.readouterr().out.splitlines()
    assert not [ln for ln in flat if re.match(r"^[A-D\-?]  ", ln)]
    assert [ln.split()[0] for ln in flat if ln.startswith("Scanner/") and " retired " not in ln] == mrgym.list_envs()
    spec = mrgym.spec(ID["SNRSpec"])
    row = next(ln for ln in flat if ln.startswith(spec.id))
    assert row.split()[:4] == [spec.id, spec.kind, "levels", ",".join(map(str, spec.levels))]
    assert row.split()[-len(mrgym.card(spec.id)["title"].split()):] == mrgym.card(spec.id)["title"].split()
    assert f" {M.FAMILIES['SNRSpec'].kind} {M.kind(M.FAMILIES['SNRSpec'].kind).name} " not in row    # no kind of task in the flat list
    assert f" {M.FAMILIES['SNRSpec'].kind} {M.kind(M.FAMILIES['SNRSpec'].kind).name} " in next(ln for ln in out if ln.startswith(spec.id))
    assert [ln for ln in flat if " retired " in ln] == [ln for ln in out if " retired " in ln] and len([ln for ln in flat if " retired " in ln]) == 2


def test_an_id_without_a_row_is_listed_apart(monkeypatch, capsys):
    monkeypatch.setattr(M, "FAMILIES", {k: v for k, v in M.FAMILIES.items() if k != "SNRSpec"})
    assert M.groups()[-1] == ("?", "Not on the map (no row in mrgym.physics_map)", [(ID["SNRSpec"], None)])
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[out.index("?  Not on the map (no row in mrgym.physics_map)") + 1].split()[0] == ID["SNRSpec"]
    assert ID["SNRSpec"] in M.readme_table().splitlines()[-1]           # on the front page it comes last, without a kind


def test_the_map_command(capsys):
    assert cli.main(["map"]) == 0
    text = capsys.readouterr().out
    assert all(f"{k.number} {k.name}" in text for k in M.KINDS) and all(f"{s.letter}  {s.title}" in text for s in M.SUBJECTS)
    assert all(name in text for name in M.FAMILIES) and all(f"({name})" in text for name in M.PLANNED)
    c = M.coverage()["total"]
    last = next(ln for ln in text.splitlines() if ln.startswith("all "))
    assert [int(x) for x in last.split()[1:]] == [c["listed"], c["formed"], c["partial"], c["computed"], c["absent"], c["isolated"], c["used"]]
    assert all(re.search(rf"^\s+{len(ids)}  {re.escape(name or M.ORDINARY)}$", text, re.M) for name, ids in M.waiting().items())
    assert "A1 " not in text and "the map is consistent with the registry; " in text
    assert cli.main(["map", "--items", "--assert"]) == 0
    listed = capsys.readouterr().out
    assert all(re.search(rf"^\s+{i.id}\s", listed, re.M) for i in M.ITEMS)
    assert cli.main(["map", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] and M.table(str(ROOT))["unchecked"] == [] and data["coverage"] == json.loads(json.dumps(M.coverage())) and len(data["items"]) == len(M.ITEMS)
    assert {f["name"]: f["ids"] for f in data["families"]} == M.registered() and all(not f["ids"] and not f["built"] for f in data["planned"])


def test_map_writes_the_page_and_refuses_a_map_with_a_problem(monkeypatch, capsys, tmp_path):
    assert cli.main(["map", "--write", str(tmp_path / "tasks.md")]) == 0
    assert (tmp_path / "tasks.md").read_text(encoding="utf-8") == M.doc_page()
    monkeypatch.setattr(M, "FAMILIES", {k: v for k, v in M.FAMILIES.items() if k != "SNRSpec"})
    assert cli.main(["map", "--assert"]) == 1 and "problems: 2" in capsys.readouterr().out
    assert cli.main(["map"]) == 0                                       # without --assert the problems are printed, not enforced
    assert cli.main(["map", "--write", str(tmp_path / "other.md")]) == 1 and not (tmp_path / "other.md").exists()


def test_the_task_table_for_the_front_page():
    """One table: the sandbox, then a bold row per subject and its tasks by kind; Task, What you do, Kind, Status."""
    words = dict(admitted="ready", candidate="experimental", practice="experimental", sandbox="no score")
    what = {ID["NoWrap"]: "a plain phrase"}
    lines = M.readme_table(what, words).splitlines()
    assert lines[:2] == ["| Task | What you do | Kind | Status |", "|---|---|---|---|"] and all(ln.count("|") == 5 for ln in lines)
    assert "Scanner/Explore-v0" in lines[2] and _cells(lines[2])[2:] == ["-", "no score"]             # the sandbox: first, outside the subjects
    heads = [ln for ln in lines if ln.startswith("| **")]
    assert heads == [f"| **{s.letter} · {s.title}**: {s.line} | | | |" for s in M.SUBJECTS]
    rows, letter, placed = {}, None, {}
    for ln in lines[3:]:
        if ln.startswith("| **"):
            letter = ln[4]
        else:
            env_id = _cells(ln)[0].split("`")[1]
            rows[env_id], placed[env_id] = _cells(ln), letter
    assert sorted(rows) == SCORED
    for env_id, row in rows.items():
        spec, (_, family) = mrgym.spec(env_id), M.family_of(env_id)
        assert row[0] == f"[`{env_id}`](envs/{cards.slug(env_id)}/)" and row[1] == what.get(env_id, mrgym.card(env_id)["title"])
        assert row[2] == M.kind(family.kind).name and row[3] == words[spec.kind] and placed[env_id] == M.cell(family)[0]
    for s in M.SUBJECTS:                                                # inside a subject: by kind of task, then in the order of the words
        mine = [(M.family_of(i)[1].kind, list(words).index(mrgym.spec(i).kind)) for i in rows if placed[i] == s.letter]
        assert mine == sorted(mine)
    plain = M.readme_table().splitlines()                               # without phrases and words: the card title and the registry's kind
    assert [_cells(ln)[1:] for ln in plain if ID["SNRSpec"] in ln] == [[mrgym.card(ID["SNRSpec"])["title"], M.kind(M.FAMILIES["SNRSpec"].kind).name,
                                                                        mrgym.spec(ID["SNRSpec"]).kind]]


# ------------------------------------------------------------------ the map cannot move a reward
def test_the_map_is_outside_every_fingerprint():
    for env_id in mrgym.list_envs():
        modules = F.scoring_modules(mrgym.spec(env_id))
        assert "mrgym.physics_map" not in modules and "mrgym.cards" not in modules
        for name in modules:
            assert "physics_map" not in pathlib.Path(__import__(name, fromlist=["x"]).__file__).read_text(encoding="utf-8"), name
    stored = F.load()["ids"]
    for env_id in mrgym.list_envs():                                    # the parts that need no episode, for every id
        assert {k: stored[env_id]["parts"][k] for k in ("registration", "engine", "rules", "scoring_source")} == F.static_parts(env_id), env_id


@pytest.mark.parametrize("family", ["NoWrap"])
def test_filing_a_family_elsewhere_moves_no_fingerprint(monkeypatch, family):
    env_id = ID[family]
    name, row = M.family_of(env_id)
    before = (M.card_line(env_id), M.cell(row), cards.markdown(mrgym.card(env_id)))
    assert F.check([env_id]) == []
    moved = row._replace(kind=3, isolates="D7", uses=("B9", "B15", "D1"))
    monkeypatch.setattr(M, "FAMILIES", dict(M.FAMILIES, **{name: moved}))
    assert M.validate(str(ROOT)) == []                                  # the new place is a valid one
    after = (M.card_line(env_id), M.cell(moved), cards.markdown(mrgym.card(env_id)))
    assert after[1] == ("D", "artefacts", 3) != before[1] and after[0] != before[0] and after[2] != before[2]
    assert "| D · artefacts | 3 trade | D7 |" in M.doc_page().split("## The tasks and what they build on")[1].split(f"[{name}]")[1].split("\n")[0]
    assert F.check([env_id]) == []                                      # same id, same fingerprint: statement, tools, episode, rules, code
    assert F.static_parts(env_id) == {k: F.load()["ids"][env_id]["parts"][k] for k in ("registration", "engine", "rules", "scoring_source")}
