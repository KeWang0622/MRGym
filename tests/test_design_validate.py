"""The validation table of the design layer: every row passes, and docs/physics/design.md shows the same rows."""
import os
import re

import pytest

from mrgym.design import validate

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
    assert len(rows(group)) >= 8


def test_row_ids_are_unique_and_groups_known():
    ids = [r["id"] for g in validate.CHECKS for r in rows(g)]
    assert len(ids) == len(set(ids))
    assert set(validate.CHECKS) == set(validate.GROUPS)


def test_the_document_shows_the_table_that_the_command_prints():
    """Quantity, reference, tolerance and verdict of every row are those of ``python -m mrgym.design.validate
    --markdown``; the error column is rounding noise and may differ between machines."""
    path = os.path.join(ROOT, "docs", "physics", "design.md")
    if not os.path.exists(path):
        pytest.skip("the tests are not running from a checkout of the repository")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    block = text.split("<!-- validation table: begin -->")[1].split("<!-- validation table: end -->")[0]

    def strip(md):
        out = []
        for line in md.strip().splitlines():
            cells = [c.strip() for c in line.split("|")]
            if len(cells) == 8 and re.fullmatch(r"[A-Z]\d+[a-z]*", cells[1]):
                out.append((cells[1], cells[2], cells[3], cells[5], cells[6]))
        return out

    made = strip(validate.markdown_table([r for g in validate.CHECKS for r in rows(g)]))
    assert strip(block) == made
    assert made and all(row[4] == "pass" for row in made)


def test_a_failing_row_is_reported_as_failing():
    assert not validate._row("X1", "rotation", "q", "r", 2e-3, 1e-3)["ok"]
    assert not validate._row("X2", "rotation", "q", "r", 0.5, 1.0, kind="min")["ok"]
    assert not validate._row("X3", "rotation", "q", "r", float("nan"), 1.0)["ok"]
    assert "FAIL" in validate.markdown_table([validate._row("X1", "rotation", "q", "r", 2e-3, 1e-3)])
