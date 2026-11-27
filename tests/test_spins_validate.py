"""The validation table of the spin simulator: every row passes, and docs/physics/spins.md shows the same rows."""
import os
import re

import pytest

from mrgym.spins import metabolites, validate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cache = {}


def rows(group):
    if group not in _cache:
        _cache[group] = validate.CHECKS[group]()
    return _cache[group]


def page():
    path = os.path.join(ROOT, "docs", "physics", "spins.md")
    if not os.path.exists(path):
        pytest.skip("the tests are not running from a checkout of the repository")
    with open(path, encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("group", list(validate.CHECKS))
def test_every_row_passes(group):
    bad = [(r["id"], r["measured"], r["tolerance"]) for r in rows(group) if not r["ok"]]
    assert not bad, bad
    assert len(rows(group)) >= 6


def test_row_ids_are_unique_and_groups_known():
    ids = [r["id"] for g in validate.CHECKS for r in rows(g)]
    assert len(ids) == len(set(ids))
    assert set(validate.CHECKS) == set(validate.GROUPS)


def _strip(md):
    """Id, quantity, reference, tolerance and verdict of every row. Numbers inside the texts are replaced by '#':
    some rows print what they measured, and the last digit of a measurement may differ between machines."""
    out = []
    for line in md.strip().splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) == 8 and re.fullmatch(r"[A-Z]\d+[a-z]*", cells[1]):
            out.append((cells[1], re.sub(r"-?\d+\.\d+", "#", cells[2]), re.sub(r"-?\d+\.\d+", "#", cells[3]), cells[5], cells[6]))
    return out


def test_the_document_shows_the_table_that_the_command_prints():
    block = page().split("<!-- validation table: begin -->")[1].split("<!-- validation table: end -->")[0]
    made = _strip(validate.markdown_table([r for g in validate.CHECKS for r in rows(g)]))
    assert _strip(block) == made
    assert len(made) >= 60 and all(row[4] == "pass" for row in made)


def test_the_document_shows_the_metabolite_table_of_the_code():
    block = page().split("<!-- metabolite table: begin -->")[1].split("<!-- metabolite table: end -->")[0]
    assert block.strip() == metabolites.table_markdown().strip()
    for item in metabolites.NOT_VERIFIED:
        assert item in page()


def test_a_failing_row_is_reported_as_failing():
    assert not validate._row("X1", "operators", "q", "r", 2e-3, 1e-3)["ok"]
    assert not validate._row("X2", "operators", "q", "r", 0.5, 1.0, kind="min")["ok"]
    assert not validate._row("X3", "operators", "q", "r", float("nan"), 1.0)["ok"]
    assert "FAIL" in validate.markdown_table([validate._row("X1", "operators", "q", "r", 2e-3, 1e-3)])
    assert validate.main(["nothing-of-that-name", "operators"]) == 0


def test_a_wrong_sign_convention_would_be_caught():
    """The independent propagator with the opposite sense of precession does not reproduce the package."""
    import numpy as np

    from mrgym.spins import evolve, operators, sequences
    s = operators.SpinSystem("ab", (2.0, 3.1), ((0, 1, 8.0),))
    c = operators.compile(s, 3.0, 4.7)
    fid = evolve.run(c, sequences.pulse_acquire().blocks)[0].fid(8, 0.5)
    right, wrong = validate._Brute(c.freq_hz, s.j_hz), validate._Brute(-c.freq_hz, s.j_hz)
    for br, same in ((right, True), (wrong, False)):
        rho = validate._apply(br.pulse(np.pi / 2), br.fz.astype(complex))
        ref = []
        for _ in range(8):
            ref.append(br.signal(rho))
            rho = validate._apply(br.free(0.5), rho)
        assert (np.abs(fid - np.array(ref)).max() < 1e-12) == same
