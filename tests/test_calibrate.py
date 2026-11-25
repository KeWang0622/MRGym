"""``mrgym calibrate`` (``mrgym.calibrate``), end to end against the stub endpoint on loopback. No model is called.

What is shown
  * the row of a level from episode records: pass rate, interval over instances, mixed-group share, ICC, outcome counts,
    band; the entry rung is the hardest level in the 20 - 60 % band with at least 80 % mixed groups
  * against the stub: a model that replays the reference is ``mastered`` on every level (no entry rung, exit code 1), a
    model that answers without scanning is ``dead``; the turn cap is in what the model reads; the run resumes
  * storing: rows replace older rows of the same model and level; a stub model is never stored as package data
  * a card says "trainable by" only through the stored calibration (nothing is typed)
"""
import json
import os

import numpy as np
import pytest

import mrgym
from mrgym import calibrate, learn
from mrgym import cli as mrgym_cli
from mrgym.agents import runner, stub_server

NULL = "Scanner/NullTissue-v0"


def _records(level, groups):
    return [dict(level=level, seed=s, passed=bool(x), stop_reason="submitted", status="completed")
            for s, g in enumerate(groups) for x in g]


def test_bands_and_mixed_share():
    assert [learn.band(p) for p in (0.0, 0.019, 0.02, 0.199, 0.2, 0.6, 0.79, 0.8, 1.0)] == \
        ["dead", "dead", "frontier", "frontier", "entry", "comfortable", "comfortable", "mastered", "mastered"]
    assert learn.mixed_share(0.5) == pytest.approx(1 - 2 * 0.5 ** 8) and learn.mixed_share(0.0) == 0.0 == learn.mixed_share(1.0)
    assert learn.mixed_share(0.18) >= 0.79 and learn.mixed_share(0.02) < 0.15      # the band table of docs/TRAINING.md


def test_level_row_from_records():
    rng = np.random.default_rng(0)
    groups = [(rng.random(8) < 0.4).astype(int).tolist() for _ in range(64)]
    row = calibrate.level_row(_records(1, groups), 1)
    flat = [x for g in groups for x in g]
    assert row["n_instances"] == 64 and row["n_episodes"] == 512 and row["k"] == 8 and row["passed"] == sum(flat)
    assert row["pass"] == round(float(np.mean(flat)), 4) and row["ci95"][0] < row["pass"] < row["ci95"][1]
    assert 0.05 < row["ci95"][1] - row["ci95"][0] < 0.12                # about +- 4 points at 64 x 8 without clustering
    assert row["mixed_group_share"] == round(float(np.mean([0 < sum(g) < 8 for g in groups])), 4) and row["band"] == "entry"
    assert abs(row["icc"]) < 0.08 and row["outcome_counts"] == {"check_failed": 512 - sum(flat), "pass": sum(flat)}
    assert calibrate.level_row(_records(1, groups), 1) == row            # the bootstrap is seeded
    # clustered passes: the same rate, a wider interval, fewer mixed groups, a high ICC
    clustered = [[1] * 8] * 26 + [[0] * 8] * 38
    c = calibrate.level_row(_records(2, clustered), 2)
    assert c["icc"] == 1.0 and c["mixed_group_share"] == 0.0 and c["ci95"][1] - c["ci95"][0] > row["ci95"][1] - row["ci95"][0]
    assert calibrate.level_row([], 3) == dict(level=3, n_instances=0, n_episodes=0)
    assert calibrate.outcome_of(dict(outcome="some_nulled", passed=False)) == "some_nulled"
    assert calibrate.outcome_of(dict(passed=False, stop_reason="truncated")) == "no_submission_cap"
    assert calibrate.outcome_of(dict(passed=False, stop_reason="rejected_call_limit")) == "no_submission_rejected"


def test_decide_the_entry_rung():
    def row(level, p, mixed):
        return dict(level=level, n_episodes=512, mixed_group_share=mixed, **{"pass": p})
    d = calibrate.decide([row(0, 0.72, 0.9), row(1, 0.41, 0.93), row(2, 0.22, 0.81), row(3, 0.03, 0.2)])
    assert d["in_band"] == [1, 2] and d["entry_rung"] == 2 and "level 2" in d["advice"]
    d = calibrate.decide([row(1, 0.41, 0.93), row(2, 0.22, 0.6)])       # in band but too few mixed groups
    assert d["in_band"] == [1, 2] and d["entry_rung"] == 1
    assert calibrate.decide([row(0, 0.01, 0.05), row(1, 0.0, 0.0)])["entry_rung"] is None
    assert "easier level" in calibrate.decide([row(0, 0.01, 0.05), row(1, 0.0, 0.0)])["advice"]
    assert "harder level" in calibrate.decide([row(0, 0.95, 0.3), row(1, 0.9, 0.5)])["advice"]
    assert "gap" in calibrate.decide([row(0, 0.95, 0.3), row(1, 0.05, 0.3)])["advice"]
    assert calibrate.decide([dict(level=1, n_episodes=0)])["entry_rung"] is None


@pytest.fixture(scope="module")
def stub():
    server, url = stub_server.start(seeds=[9000, 9001], env_ids=[NULL], levels=[1])
    yield url
    server.shutdown()


def test_calibrate_against_the_stub_endpoint(stub, tmp_path):
    out = str(tmp_path / "cal")
    kw = dict(levels=[1], n=2, episodes=2, max_turns=20, workers=2, quiet=True)
    ref = calibrate.run(NULL, "stub:reference", stub, out, **kw)
    row = ref["calibration"][0]
    assert ref["schema"] == calibrate.SCHEMA and ref["infra_errors"] == 0 and ref["settings"]["seeds"] == "9000-9001"
    assert (row["n_instances"], row["n_episodes"], row["k"], row["pass"], row["band"]) == (2, 4, 2, 1.0, "mastered")
    assert row["outcome_counts"] == {"pass": 4} and row["mixed_group_share"] == 0.0 and row["ci95"] == [1.0, 1.0]
    assert row["model"] == "stub:reference" and row["temperature"] == 1.0 and row["max_turns"] == 20 and row["engine"] == mrgym.ENGINE_ID
    assert ref["in_band"] == [] and ref["entry_rung"] is None and not row["entry_rung"] and "harder level" in ref["advice"]
    # what the model read: the cap as the environment's own tool-call limit; what was sent: the training temperature
    episode = os.path.join(out, "episodes", runner.slug("stub:reference"), runner.env_slug(NULL), "L1", "s9000", "e0")
    with open(os.path.join(episode, "messages.json"), encoding="utf-8") as f:
        tool_texts = [m["content"] for m in json.load(f)["messages"] if m["role"] == "tool"]
    assert "calls 1/20" in tool_texts[0].splitlines()[-1] and "turn" not in tool_texts[0].splitlines()[-1]
    with open(os.path.join(episode, "record.json"), encoding="utf-8") as f:
        rec = json.load(f)
    assert rec["params_sent"]["temperature"] == 1.0 and rec["outcome"] == "pass" and rec["progress"] == 1.0 and rec["checks"]["measured"]
    again = calibrate.run(NULL, "stub:reference", stub, out, **kw)       # resumes: nothing is played twice
    assert again["calibration"][0] == dict(row, date=again["calibration"][0]["date"]) and again["run_id"] == ref["run_id"]
    prior = calibrate.run(NULL, "stub:prior", stub, out, **kw)
    p = prior["calibration"][0]
    assert p["pass"] == 0.0 and p["band"] == "dead" and p["outcome_counts"] == {"no_measurement": 4} and "easier level" in prior["advice"]
    assert "mastered" in calibrate.text_table(ref) and "dead" in calibrate.text_table(prior)
    # storing
    with pytest.raises(ValueError):
        calibrate.write(ref)                                             # a stub model is not package data
    results = str(tmp_path / "results")
    path = calibrate.write(ref, results)
    calibrate.write(prior, results)
    calibrate.write(ref, results)                                        # replaces its own rows
    stored = learn.load(NULL, "calibration", results)
    assert path.endswith("Scanner-NullTissue-v0.calibration.json") and len(stored["calibration"]) == 2
    assert {r["model"] for r in stored["calibration"]} == {"stub:reference", "stub:prior"}


def test_calibrate_command(stub, tmp_path, capsys):
    argv = ["calibrate", NULL, "--model", "stub:prior", "--base-url", stub, "--levels", "1", "--n", "2", "--episodes", "2",
            "--max-turns", "6", "--workers", "2", "--out", str(tmp_path / "cal"), "--write", "--results-dir", str(tmp_path / "res")]
    assert mrgym_cli.main(argv) == 1                                     # no entry rung: a non-zero exit
    text = capsys.readouterr().out
    assert "calibration of stub:prior" in text and "entry rung" in text and "written:" in text
    assert learn.load(NULL, "calibration", str(tmp_path / "res"))["calibration"][0]["band"] == "dead"


def test_calibration_claim_needs_the_stored_file():
    """Rule: a card names a model as able to train on an environment only through a stored calibration."""
    for env_id in mrgym.list_envs():
        card = mrgym.card(env_id)
        field = card.get("learnability")
        if field is None:
            continue
        stored = learn.load(env_id, "calibration")
        assert field == learn.stored(env_id)                             # the card field is the stored content, nothing else
        entry = sorted({r["model"] for r in (stored or {}).get("calibration", []) if r.get("entry_rung")})
        assert field["trainable_by"] == entry
        if stored is None:
            assert field["trainable_by"] == [] and field["calibration"] == []
        for r in field["calibration"]:
            assert not r["model"].startswith("stub:") and r["engine"] == mrgym.ENGINE_ID
            if r.get("entry_rung"):
                assert learn.ENTRY_BAND[0] <= r["pass"] <= learn.ENTRY_BAND[1] and r["mixed_group_share"] >= learn.MIXED_MIN
