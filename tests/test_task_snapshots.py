"""Pins of what a task says and what its constants are (``tests/golden/task_snapshots.json``).

Scoring constants live at module level, where a mutation tool does not reach; statements, schemas and cards are text
that no formula predicts. Both are pinned here against the stored snapshot of every registered id that has a task
module. The snapshot is written by ``python tools/audit/golden.py --write`` and reviewed as a diff: a changed snapshot
of a released id needs a new ``-vN`` (CONTRIBUTING.md, "Golden data").

The stored episode of the reference policy is compared in ``tests/test_determinism.py``.

An id that is not in the file yet shows as an expected failure here and fails ``make audit``.
"""
import pytest

import golden_cases as G
import mrgym

GOLDEN = G.load(G.TASK_FILE) or dict(ids={}, shared={}, engine=None)
IDS = G.task_ids()
REGENERATE = "; if the change is intended run `python tools/audit/golden.py --write` and review the diff"


def _stored(env_id):
    if env_id not in GOLDEN["ids"]:
        pytest.xfail(f"{env_id} has no stored snapshot: run `python tools/audit/golden.py --write`")
    assert "error" not in GOLDEN["ids"][env_id], f"the stored snapshot of {env_id} records a failure: {GOLDEN['ids'][env_id].get('error')}"
    return GOLDEN["ids"][env_id]


@pytest.fixture(scope="module")
def static():
    cache = {}
    return lambda env_id: cache.setdefault(env_id, G.task_static(env_id))


def test_the_file_exists_and_names_this_engine():
    assert GOLDEN["engine"] == mrgym.ENGINE_VERSION, "run `python tools/audit/golden.py --write`"
    # a stored id that was retired since is not an error of this suite: `golden.py --check` (and `make audit`) lists it as gone
    assert set(GOLDEN["ids"]) & set(mrgym.registry) or not IDS, "no stored snapshot matches a registered id" + REGENERATE


def test_shared_scoring_constants():
    assert G.shared_constants() == GOLDEN["shared"], "a constant of the gate, the family rule, the log or the console moved" + REGENERATE


@pytest.mark.parametrize("env_id", IDS)
def test_module_constants(env_id, static):
    want = _stored(env_id)
    assert static(env_id)["constants"] == want["constants"], f"a scoring constant of {env_id} moved" + REGENERATE


@pytest.mark.parametrize("env_id", IDS)
def test_instances_statements_and_schemas(env_id, static):
    want, got = _stored(env_id), static(env_id)
    assert sorted(got["levels"]) == sorted(want["levels"])
    for level, row in got["levels"].items():
        for key in row:
            assert row[key] == want["levels"][level][key], f"{env_id} level {level}: {key} changed" + REGENERATE
    assert got.get("submit_doc") == want.get("submit_doc")


@pytest.mark.parametrize("env_id", IDS)
def test_card(env_id, static):
    want, got = _stored(env_id), static(env_id)
    assert got.get("card") == want.get("card"), f"the card of {env_id} changed" + REGENERATE
