"""The interface-tax metric: recorded calls replayed on the current interface, and the tax of a run folder.

The corpus under ``tests/data`` holds every tool call of 128 recorded episodes of an open 9B model on the v0 interface
(the arguments as the model wrote them). 329 of its 1,462 calls were lost to the interface when they were recorded.
The interface was designed with this corpus in view, so the test is a regression test: the tax stays below 3 %, and
no call that was accepted then is refused now for how it is written.
"""
import collections
import json
import os

import pytest

from mrgym import canary, cli
from mrgym import interface as I
from mrgym.agents import iface_tax

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "u03_interface_calls.jsonl")
# The second corpus: every call of 368 episodes of an open 27B model with reasoning off, on seven environments. A tenth
# of its calls were lost to the interface it was recorded on, most of them ``calc`` text the old grammar refused.
SECOND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "c03_interface_calls.jsonl")
SECOND_MAX = 0.05


@pytest.fixture(scope="module")
def result():
    return iface_tax.replay(CORPUS, draws=2000)


def test_tax_corpus_is_what_its_header_says():
    header, calls = iface_tax.read_corpus(CORPUS)
    assert (header["episodes"], header["calls"], header["lost_when_recorded"], sum(header["turns"])) == (128, 1462, 329, 1341)
    assert len(calls) == 1462 and sum(c["lost"] for c in calls) == 329 and len(header["turns"]) == 128
    assert all(set(c) == {"episode", "turn", "name", "arguments", "recorded", "lost"} for c in calls)
    assert all(isinstance(v, str) for c in calls for v in c["arguments"].values())          # as written: every value is text
    assert all(0 <= c["turn"] < header["turns"][c["episode"]] for c in calls)
    assert {c["name"] for c in calls} == {"plan", "scan", "scanner_info", "view", "read_regions", "calc"}


def test_tax_of_the_recorded_calls_is_below_three_percent(result):
    was, now = result["recorded"], result["now"]
    # the id the corpus was recorded on is retired: its successor has the same scanner tools, and the corpus holds no
    # call to the delivering tool (the recording kept answered calls, and a delivery ends the episode), so every call compares
    assert (result["recorded_on"], result["env_id"], result["not_comparable"]) == ("Scanner/NoWrap-v0", "Scanner/NoWrap-v1", 0)
    assert (was["lost_calls"], was["calls"], was["lost_turns"], was["turns"]) == (329, 1462, 316, 1341)       # as recorded: 22.5 % and 23.6 %
    assert now["calls"] == 1462 and now["turns"] == 1341 and now["episodes"] == 128
    assert now["call_tax"] <= 0.03 and now["turn_tax"] <= 0.03 and now["lost_calls"] <= 16
    assert max(now["interval95"]["calls"][1], now["interval95"]["turns"][1]) <= 0.04                       # the upper bound of the bootstrap
    assert now["interval95"]["calls"][0] <= now["call_tax"] <= now["interval95"]["calls"][1]
    assert sum(b["calls"] for b in result["became"]) == 329
    # what is still lost is what must not be accepted: misspelt names, parameters of another sequence, text that is not arithmetic
    assert {r["type"] for r in result["residual"]} <= {"unknown_parameter", "not_applicable", "malformed_call"}
    assert sum(r["calls"] for r in result["residual"]) == now["lost_calls"]
    assert all(r["parameter"] not in ("protocol", "regions", "te_ms") or r["type"] == "not_applicable" for r in result["residual"])
    # the two losses that made up most of the tax are gone: te_ms below the minimum, and the spellings of regions
    became = {(b["recorded"], b["now"].split(" (")[0]): b["calls"] for b in result["became"]}
    assert not [k for k in became if k[0] in ("console/TE_BELOW_MINIMUM", "invalid/regions", "invalid/protocol_not_object") and k[1] == "still lost"]
    assert became[("console/TE_BELOW_MINIMUM", "accepted")] > 100 and became[("invalid/regions", "accepted")] > 50


def test_tax_no_call_that_was_accepted_is_refused_now(result):
    x = result["accepted_then_refused_now"]
    assert x["checked"] > 1000 and x["refused"] == 0, x["examples"]


@pytest.fixture(scope="module")
def second():
    return iface_tax.replay(SECOND, draws=1000)


def test_tax_second_corpus_is_what_its_header_says():
    header, calls = iface_tax.read_corpus(SECOND)
    assert (header["episodes"], header["calls"], header["lost_when_recorded"], len(calls)) == (368, 10148, 1063, 10148)
    assert header["canary"] == canary.CANARY and header["agent"] == "qwen3.8-27b"                       # calls only, and marked
    assert len(header["instances"]) == len(header["turns"]) == 368 and sum(c["lost"] for c in calls) == 1063
    assert len({tuple(x) for x in header["instances"]}) == 46 and len({x[0] for x in header["instances"]}) == 7
    assert all(set(c) == set(iface_tax.CALL_KEYS) and 0 <= c["turn"] < header["turns"][c["episode"]] for c in calls)
    assert {c["name"] for c in calls} == {"scanner_info", "status", "plan", "scan", "read_regions", "view", "profile", "calc", "submit"}
    assert not [c for c in calls if c["lost"] and c["recorded"] == "ok"] and {c["recorded"].split("/")[-1] for c in calls if c["lost"]} == {"shape", "vocabulary"}
    with open(SECOND, encoding="utf-8") as f:                              # nothing of the conversations: no task text, no answer of a tool
        text = f.read()
    assert "Limits. " not in text and "scan_index 0 ·" not in text and '"role"' not in text


def test_tax_of_the_second_corpus_is_at_most_five_percent(second):
    was, now = second["recorded"], second["now"]
    assert (second["instances"], len(second["environments"]), second["not_comparable"]) == (46, 7, 0)
    assert (was["lost_calls"], was["calls"], was["turns"]) == (1063, 10148, 7092) and was["call_tax"] > 0.10 and was["turn_tax"] > 0.13
    assert now["calls"] == 10148 and now["turns"] == 7092 and now["episodes"] == 368
    assert now["call_tax"] <= SECOND_MAX and now["turn_tax"] <= SECOND_MAX
    assert max(now["interval95"]["calls"][1], now["interval95"]["turns"][1]) <= SECOND_MAX + 0.01
    assert sum(b["calls"] for b in second["became"]) == 1063 and sum(r["calls"] for r in second["residual"]) == now["lost_calls"]
    # what is still lost: text that is not Python, Python the calculator does not have, a name the tool does not have
    assert {r["type"] for r in second["residual"]} <= {"malformed_call", "invalid_value", "unknown_parameter", "not_applicable"}
    assert sum(r["calls"] for r in second["residual"] if r["tool"] != "calc") <= 30
    became = collections.Counter()
    for b in second["became"]:
        became[(b["recorded"], b["now"].split(" (")[0])] += b["calls"]
    assert became[("malformed_call/shape", "accepted")] + became[("malformed_call/shape", "refused by the instrument")] > 500
    assert became[("invalid_value/shape", "still lost")] < 40


def test_tax_second_corpus_no_call_that_was_accepted_is_refused_now(second):
    x = second["accepted_then_refused_now"]
    assert x["checked"] > 8000 and x["refused"] == 0, x["examples"]


def test_tax_report_and_exit_code(result, tmp_path, capsys):
    text = iface_tax.report(result)
    assert f"calls {result['now']['lost_calls']:>4} of 1462" in text and "as recorded" in text and "regression test" in text
    header, calls = iface_tax.read_corpus(CORPUS)
    small = tmp_path / "small.jsonl"                                     # four episodes of the corpus
    with open(small, "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(header, episodes=4, turns=header["turns"][:4])) + "\n")
        f.writelines(json.dumps(c) + "\n" for c in calls if c["episode"] < 4)
    assert cli.main(["iface-tax", "--replay", str(small), "--max", "0.03", "--draws", "200"]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "interface tax of the recorded calls" in out
    lost = [dict(episode=0, turn=0, name="plan", arguments={"sequence": "se", "tr_ms": "500", "te_mm": "9"}, recorded="invalid/other", lost=True)]
    bad = tmp_path / "bad.jsonl"
    with open(bad, "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(env_id="Scanner/NoWrap-v0", episodes=1, turns=[1])) + "\n" + json.dumps(lost[0]) + "\n")
    assert cli.main(["iface-tax", "--replay", str(bad), "--max", "0.03", "--draws", "0"]) == 1
    assert "FAIL" in capsys.readouterr().out
    assert cli.main(["iface-tax", "--replay", str(bad), "--json", "--draws", "0"]) == 0
    assert json.loads(capsys.readouterr().out)["residual"][0]["type"] == "unknown_parameter"
    # a delivery written for the retired id meets the deliverable of its successor: the fields it lacks are a refusal of
    # the task, not interface tax, and the call is counted as not comparable
    old = dict(episode=0, turn=0, name="submit", arguments={"pe_dir": "y", "fov_mm": "240", "phase_fov": "1", "matrix": "120"}, recorded="invalid/other", lost=True)
    moved = tmp_path / "moved.jsonl"
    with open(moved, "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(env_id="Scanner/NoWrap-v0", episodes=1, turns=[1])) + "\n" + json.dumps(old) + "\n")
    got = iface_tax.replay(str(moved), draws=0)
    assert got["not_comparable"] == 1 and got["now"]["lost_calls"] == 0 and got["became"][0]["now"] == "refused by the instrument (invalid_value)"
    assert "1 of 1 calls (those to the delivering tool) are not comparable" in iface_tax.report(got)
    with pytest.raises(SystemExit):
        cli.main(["iface-tax"])


def test_tax_of_a_run_folder_comes_from_its_records(tmp_path, capsys):
    """``--dir``: the runner writes an ``interface`` block per model episode; the tax of an agent is their sum."""
    for k, (shape, vocabulary, lost_turns) in enumerate([(0, 0, 0), (1, 2, 2), (0, 1, 0)]):
        d = tmp_path / "episodes" / "some-model" / "Scanner-NoWrap-v0" / "L1" / f"s{k}" / "e0"
        d.mkdir(parents=True)
        rec = dict(agent="some-model", status="completed", turns=10, interface=dict(
            calls=12, executed=12 - shape - vocabulary, rejected=shape + vocabulary, shape=shape, vocabulary=vocabulary, task_refusals=1,
            coerced=3, internal_errors=0, lost_turns=lost_turns, idle_turns=0, cut_replies=k))
        (d / "record.json").write_text(json.dumps(rec))
    table = iface_tax.from_records(str(tmp_path), draws=500)
    row = table["some-model"]
    assert (row["calls"], row["lost_calls"], row["turns"], row["lost_turns"], row["episodes"]) == (36, 4, 30, 2, 3)
    assert row["call_tax"] == pytest.approx(4 / 36) and row["turn_tax"] == pytest.approx(2 / 30)
    assert (row["task_refusals"], row["coerced"], row["cut_replies"]) == (3, 9, 3)
    assert cli.main(["iface-tax", "--dir", str(tmp_path), "--max", "0.03", "--draws", "100"]) == 1
    out = capsys.readouterr().out
    assert "interface-limited" in out and "FAIL" in out
    assert set(I.INTERFACE_CLASSES) == {"shape", "vocabulary"}
