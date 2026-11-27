"""The validation table of the RF field model: every row passes, and docs/physics/em.md shows what the commands print."""
import os
import re

import pytest

from mrgym.em import validate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cache = {}


def rows(group):
    if group not in _cache:
        _cache[group] = validate.CHECKS[group]()
    return _cache[group]


def page():
    path = os.path.join(ROOT, "docs", "physics", "em.md")
    if not os.path.exists(path):
        pytest.skip("the tests are not running from a checkout of the repository")
    with open(path, encoding="utf-8") as f:
        return f.read()


def block(text, name):
    return text.split(f"<!-- {name}: begin -->")[1].split(f"<!-- {name}: end -->")[0].strip()


@pytest.mark.parametrize("group", list(validate.CHECKS))
def test_every_row_passes(group):
    bad = [(r["id"], r["measured"], r["tolerance"]) for r in rows(group) if not r["ok"]]
    assert not bad, bad
    assert len(rows(group)) >= 4


def test_row_ids_are_unique_and_groups_known():
    ids = [r["id"] for g in validate.CHECKS for r in rows(g)]
    assert len(ids) == len(set(ids))
    assert set(validate.CHECKS) == set(validate.GROUPS)


def test_the_document_shows_the_table_that_the_command_prints():
    """Quantity, reference, tolerance and verdict of every row are those of ``python -m mrgym.em.validate --markdown``;
    the error column may differ between machines."""

    def strip(md):
        out = []
        for line in md.strip().splitlines():
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)]          # a bar inside a cell is escaped
            if len(cells) > 1 and re.fullmatch(r"[A-Z]\d+[a-z]*\d*", cells[1]):
                assert len(cells) == 8, line
                out.append((cells[1], cells[2], cells[3], cells[5], cells[6]))
        return out

    all_rows = [r for g in validate.CHECKS for r in rows(g)]
    made = strip(validate.markdown_table(all_rows))
    assert len(made) == len(all_rows)                                        # every row is a table row of six cells
    assert strip(block(page(), "validation table")) == made
    assert made and all(row[4] == "pass" for row in made)


def test_the_document_shows_the_tissue_table_of_the_code():
    assert block(page(), "tissue table") == validate.tissue_markdown().strip()


def test_the_document_shows_the_demonstration_the_command_prints():
    """The same lines with the same numbers, each within 2 % (three printed digits; the solve is deterministic, a
    different linear-algebra library may move a last digit)."""
    number = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")
    want, got = validate.demonstration_markdown().strip().splitlines(), block(page(), "demonstration").splitlines()
    assert len(want) == len(got)
    for a, b in zip(want, got):
        assert number.sub("#", a) == number.sub("#", b)
        for x, y in zip(number.findall(a), number.findall(b)):
            assert float(x) == pytest.approx(float(y), rel=0.02, abs=0.006), (a, b)


def test_the_cost_block_has_the_columns_of_the_estimate():
    """Timings depend on the machine: only the shape of the block is pinned, and the step counts, which do not."""
    recs = validate.cost_3d(repeat=1)
    text = block(page(), "cost")
    assert text.startswith("```text") and text.endswith("```")
    lines = text.splitlines()[1:-1]
    assert len(lines) == 1 + len(recs) and lines[0] == validate.cost_3d_text(recs).splitlines()[0]
    for r, line in zip(recs, lines[1:]):
        assert f"{r['steps']:,}" in line and f"{r['cells']:,}" in line


def test_a_failing_row_is_reported_as_failing():
    assert not validate._row("X1", "solver", "q", "r", 2e-3, 1e-3)["ok"]
    assert not validate._row("X2", "solver", "q", "r", 0.5, 1.0, kind="min")["ok"]
    assert not validate._row("X3", "solver", "q", "r", float("nan"), 1.0)["ok"]
    assert "FAIL" in validate.markdown_table([validate._row("X1", "solver", "q", "r", 2e-3, 1e-3)])


def test_fill_document_replaces_the_blocks_and_nothing_else():
    text = "head\n" + "".join(f"<!-- {n}: begin -->\nold\n<!-- {n}: end -->\nbetween {i}\n" for i, n in enumerate(validate.BLOCKS))
    out = validate.fill_document(text, rows=[validate._row("X1", "solver", "q", "r", 0.0, 1.0)])
    assert out.startswith("head\n") and "old" not in out
    assert all(f"between {i}\n" in out for i in range(len(validate.BLOCKS)))
    assert "| X1 | q | r | 0 | ≤ 1 | pass |" in out
    assert "| X2 | \\|B1+\\| | r | 0 | ≤ 1 | pass |" in validate.markdown_table([validate._row("X2", "solver", "|B1+|", "r", 0.0, 1.0)])
