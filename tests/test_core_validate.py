"""The validation table of the signal core: every row passes, and docs/physics/core.md shows the same rows."""
import os
import re

import pytest

from mrgym.core import validate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cache = {}


def rows(group):
    if group not in _cache:
        _cache[group] = validate.CHECKS[group]()
    return _cache[group]


@pytest.mark.parametrize("group", list(validate.CHECKS))
def test_every_row_passes(group):
    bad = [(r["id"], r["measured"], r["tolerance"]) for r in rows(group) if not r["ok"]]
    assert not bad, bad
    assert len(rows(group)) >= 15


def test_row_ids_are_unique_and_groups_known():
    ids = [r["id"] for g in validate.CHECKS for r in rows(g)]
    assert len(ids) == len(set(ids))
    assert set(validate.CHECKS) == set(validate.GROUPS)


def test_every_served_sequence_and_every_new_sequence_has_rows():
    from mrgym.core import library
    text = " ".join(r["quantity"] for g in ("engine", "sequences") for r in rows(g))
    for name in list(library.SERVED) + list(library.NEW):
        assert re.search(rf"\b{name}\b", text), name
    # at least two rows (independent checks) for every sequence the scanner does not serve
    seq = " ".join(r["quantity"] + " " + r["reference"] for r in rows("sequences"))
    for name, words in dict(fse=("fse",), mese=("mese",), megre=("megre",), spgr_lobe=("spgr_lobe", "RF spoiling", "increment 0"),
                            dess=("dess", "the same two samples"), mprage=("mprage", "one-pulse limit"), t2prep=("t2prep",), satrec=("satrec", "T1 / 8"),
                            dwse=("dwse", "b-value of those programs", "Bloch-Torrey"), steam=("steam", "the same sequences without diffusion"),
                            mt_spgr=("mt_spgr", "two pools", "MT saturation")).items():
        assert sum(seq.count(w) for w in words) >= 2, name


def test_the_document_shows_the_table_that_the_command_prints():
    """Quantity, reference, tolerance and verdict of every row are those of ``python -m mrgym.core.validate --markdown``;
    the error column is rounding noise (or a measured size) and may differ between machines."""
    path = os.path.join(ROOT, "docs", "physics", "core.md")
    if not os.path.exists(path):
        pytest.skip("the tests are not running from a checkout of the repository")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    block = text.split("<!-- validation table: begin -->")[1].split("<!-- validation table: end -->")[0]
    number = re.compile(r"\d[\d.,]*(e[+-]?\d+)?")

    def strip(md):
        out = []
        for line in md.strip().splitlines():
            cells = [c.strip() for c in line.split("|")]
            if len(cells) == 8 and re.fullmatch(r"[A-Z]\d+[a-z]*", cells[1]):
                # measured sizes quoted inside a quantity or a reference are machine dependent in their last digits
                out.append((cells[1], number.sub("#", cells[2]), number.sub("#", cells[3]), cells[5], cells[6]))
        return out

    made = strip(validate.markdown_table([r for g in validate.CHECKS for r in rows(g)]))
    assert strip(block) == made
    assert made and all(row[4] == "pass" for row in made)


def test_a_failing_row_is_reported_as_failing():
    assert not validate._row("X1", "agreement", "q", "r", 2e-3, 1e-3)["ok"]
    assert not validate._row("X2", "agreement", "q", "r", 0.5, 1.0, kind="min")["ok"]
    assert not validate._row("X3", "agreement", "q", "r", float("nan"), 1.0)["ok"]
    assert "FAIL" in validate.markdown_table([validate._row("X1", "agreement", "q", "r", 2e-3, 1e-3)])


def test_the_speed_table_has_a_row_per_library_sequence():
    import inspect
    src = inspect.getsource(validate.speed_table)
    from mrgym.core import library
    for name in list(library.SERVED) + list(library.NEW):
        assert f"L.{name}(" in src, name


def test_the_example_of_the_document_runs_as_written():
    path = os.path.join(ROOT, "docs", "physics", "core.md")
    if not os.path.exists(path):
        pytest.skip("the tests are not running from a checkout of the repository")
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("    from mrgym.core import"))
    code = []
    for line in lines[start:]:
        if line and not line.startswith("    "):
            break
        code.append(line[4:])
    scope = {}
    exec("\n".join(code), scope)                                        # noqa: S102
    assert scope["echoes"].shape == (16, 2 * 169) and scope["per_tissue"].shape == (16, 2) and scope["first_tr"].shape == scope["echoes"].shape
    assert 0.2 < scope["per_tissue"][0].real.min() < 1.0
