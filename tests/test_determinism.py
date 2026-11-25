"""Same seed, same episode, however the process is set up: two episodes at once in threads, a fresh interpreter
with another hash seed, four BLAS threads instead of one.

The reference is the stored digest of one reference-policy episode per id (``tests/golden/task_snapshots.json``: the
sha256 of every observation text, the head of the scanner log's hash chain, the reward), which a plain serial run
must reproduce first. On a stack other than the recorded one only the reward and the verdict
are compared (see ``tests/test_golden.py`` for the two modes).
"""
import concurrent.futures
import json
import os
import subprocess
import sys

import pytest

import golden_cases as G

SNAPSHOT = G.load(G.TASK_FILE) or dict(ids={})
ENGINE = G.load(G.ENGINE_FILE)
IDS = [i for i in G.task_ids() if i in SNAPSHOT["ids"]]
BIT_EXACT = bool(ENGINE) and G.mode(ENGINE["stack"])[0] == "bit-exact"


def _same(got: dict, env_id: str, how: str):
    want = SNAPSHOT["ids"][env_id]["episode"]
    keys = list(want) if BIT_EXACT else ["reward", "passed", "policy_error"]
    assert {k: got[k] for k in keys} == {k: want[k] for k in keys}, f"{env_id} differs when run {how}"


@pytest.mark.parametrize("env_id", G.task_ids())
def test_the_reference_episode_is_the_stored_one(env_id):
    if env_id not in SNAPSHOT["ids"]:
        pytest.xfail(f"{env_id} has no stored snapshot: run `python tools/audit/golden.py --write`")
    want, got = SNAPSHOT["ids"][env_id]["episode"], G.episode(env_id)
    assert (got["reward"], got["passed"], got["policy_error"]) == (want["reward"], want["passed"], want["policy_error"])
    if BIT_EXACT:
        assert got == want, (f"the reference episode of {env_id} changed (engine, task or reference policy); if the change is "
                             "intended run `python tools/audit/golden.py --write` and review the diff")


@pytest.mark.skipif(not IDS, reason="no stored episode: python tools/audit/golden.py --write")
def test_episodes_in_two_threads_at_once_equal_the_serial_ones():
    """Two at a time: every episode holds a hidden world, an engine and its scans (a quarter of a gigabyte)."""
    jobs = (IDS * 4)[:max(4, len(IDS) + 1)]                              # every id, and at least one id twice
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results = list(pool.map(G.episode, jobs))
    for env_id, got in zip(jobs, results):
        _same(got, env_id, "in a thread next to another")


@pytest.mark.skipif(not IDS, reason="no stored episode: python tools/audit/golden.py --write")
def test_a_fresh_interpreter_with_another_hash_seed_and_four_blas_threads_gives_the_same_episodes():
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); import golden_cases as G; "
            "print(json.dumps({i: G.episode(i) for i in sys.argv[2:]}))")
    env = dict(os.environ, PYTHONHASHSEED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    out = subprocess.run([sys.executable, "-c", code, os.path.dirname(os.path.abspath(__file__)), *IDS], env=env, capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    results = json.loads(out.stdout.strip().splitlines()[-1])
    for env_id in IDS:
        _same(results[env_id], env_id, "in a fresh interpreter with PYTHONHASHSEED=1 and 4 BLAS threads")
