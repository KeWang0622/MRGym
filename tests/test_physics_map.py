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
    assert forms == {"validate", *M.EVIDENCE_PACKAGES, "tests", "examples", "mrgym"} and list(M.EVIDENCE_PACKAGES)[:3] == ["design", "core", "spins"]
    for prefix in M.EVIDENCE_PACKAGES:
        cited = {e for i in M.ITEMS for e in i.evidence if e.startswith(prefix + ":")}
        assert cited and cited <= {f"{prefix}:{r}" for r in M.package_ids(prefix)} and all(M.package_row(e)[0] == prefix for e in cited), prefix
    assert len(M.package_ids("design")) > 60
    # the ids read from the source of ``mrgym.core.validate`` and of ``mrgym.spins.validate`` are the rows of the tables in
    # docs/physics/, which tests/test_core_validate.py and tests/test_spins_validate.py hold equal to what the commands print
    for prefix, rows in (("core", 81), ("spins", 69)):
        table = (ROOT / "docs" / M.EVIDENCE_PACKAGES[prefix][2]).read_text(encoding="utf-8").split("<!-- validation table: begin -->")[1].split("<!-- validation table: end -->")[0]
        assert M.package_ids(prefix) == set(re.findall(r"^\| ([A-Z]+\d+[a-z]?) \|", table, re.M)) and len(M.package_ids(prefix)) == rows, prefix
    assert M.evidence_exists("core:N1", str(ROOT)) is True and M.evidence_exists("core:E9", str(ROOT)) is False     # E9 is a row of the design layer only
    assert M.evidence_exists("spins:S7", str(ROOT)) is True and M.evidence_exists("spins:N1", str(ROOT)) is False   # N1 is a row of the signal core only
    assert M.evidence_exists("spins:M4c", str(ROOT)) is True            # a row that the source names in a loop, not as ``_row("M4c"``
    assert M._evidence(("K6 K22", "design:E3 core:N1b", "spins:S7 K6", "mrgym qa acr")) == ("K6", "K22", "design:E3", "core:N1b", "spins:S7", "K6", "mrgym qa acr")
    for wrong in ("K999", "design:Z9", "core:Z9", "core:N99", "spins:Z9", "em:P1", "tests/test_scanner.py::test_that_does_not_exist", "tests/no_such_file.py::test_x",
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
    assert "No image of this engine has been compared with a measurement on a scanner yet." in page
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
    assert f"{n_piece + len(waits[''])} items. {n_piece} of them wait for one of {len(M.ENABLERS)} pieces of engine work; the other {len(waits[''])} take" in last
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
    waits = M.waiting()
    assert list(waits) == names + [""] and all(waits[name] for name in names)                    # every piece unlocks something that is listed
    by_id = M.items()
    assert sorted(i for ids in waits.values() for i in ids) == sorted(i.id for i in M.ITEMS if i.status in ("computed", "absent"))
    assert all(by_id[i].needs == name for name, ids in waits.items() for i in ids)
    assert all(not i.needs for i in M.ITEMS if i.status == "formed") and all(i.needs in names + [""] for i in M.ITEMS)
    assert waits["measured data"] == [i.id for i in M.ITEMS if "compared with a measurement on a scanner" in i.reason]
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
    assert M.package_row("K6") is None and M.package_row("design:e3") is None and M.package_row("em:P1") is None and M.package_problems(str(ROOT)) == []
    # a new package is one line: here the rows of the spin systems under a second prefix
    line = ("mrgym.spins.validate", M._ALSO_IN_A_LOOP, "physics/spins.md", "python -m mrgym.em.validate")
    with monkeypatch.context() as m:
        m.setattr(M, "EVIDENCE_PACKAGES", dict(M.EVIDENCE_PACKAGES, em=line))
        m.setattr(M, "ITEMS", tuple(i._replace(evidence=M._evidence(("K18 em:S7 em:M4c core:N1",))) if i.id == "A2" else i for i in M.ITEMS))
        assert M.package_row("em:S7") == ("em", "S7") and M.evidence_exists("em:S7", str(ROOT)) is True and M.evidence_exists("em:N1", str(ROOT)) is False
        assert M.items()["A2"].evidence == ("K18", "em:S7", "em:M4c", "core:N1") and M.validate(str(ROOT)) == []
        assert M._evidence_text(M.items()["A2"].evidence) == "`mrgym validate` K18; `python -m mrgym.core.validate` N1; `python -m mrgym.em.validate` S7, M4c"
        assert " and of `python -m mrgym.em.validate` ([em](physics/spins.md)), tests, " in M.doc_page()
    # a package is listed when it is merged, not before; and a line that finds nothing, or shows nothing, is refused
    for wrong, said in ((("mrgym.no_such_package.validate",) + line[1:], "which cannot be imported: list a package when it is merged"),
                        ((line[0], r'_no_row\("([A-Z]+)"') + line[2:], "finds no row id in the source of mrgym.spins.validate"),
                        (line[:2] + ("physics/no_such_page.md", line[3]), "the page docs/physics/no_such_page.md of 'em' does not exist")):
        with monkeypatch.context() as m:
            m.setattr(M, "EVIDENCE_PACKAGES", dict(M.EVIDENCE_PACKAGES, em=wrong))
            problems = M.validate(str(ROOT))
            assert len(problems) == 1 and problems[0].startswith("EVIDENCE_PACKAGES: ") and said in problems[0], problems
            assert M.evidence_exists("em:S7", str(ROOT)) is (wrong[0] == line[0] and wrong[1] == line[1])


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
