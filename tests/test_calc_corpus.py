"""Recorded ``calc`` calls of an open model, replayed: a small sample of the corpus that ``tools/audit/calc_corpus.py``
replays in full (``--write-sample`` wrote ``tests/data/calc_corpus_sample.jsonl``: whole episodes, because the names of
one call are there for the next). The replay gives the stored outcome of every call, twice the same, and what is
still refused is the model's own slip."""
import hashlib
import importlib.util
import json
import os
import sys

import pytest

from mrgym import calc as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "audit", "calc_corpus.py")
SAMPLE = os.path.join(ROOT, "tests", "data", "calc_corpus_sample.jsonl")
pytestmark = pytest.mark.skipif(not os.path.isfile(TOOL), reason="tools/audit is not next to the tests (installed-wheel run outside a checkout)")


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("audit_tool_calc_corpus", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def episodes():
    with open(SAMPLE) as f:
        return [json.loads(line) for line in f if line.strip()]


def replay(tool, episodes):
    out = []
    for episode in episodes:
        names = {}
        for call in episode["calls"]:
            try:
                shown, error = C.render(C.calc(call["text"], names)), None
            except C._CALC_ERRORS as e:
                shown, error = None, e
            said = shown if error is None else str(error)[:200]
            out.append((tool.classify(error), hashlib.sha256((said or "").encode()).hexdigest()[:16]))
    return out


def test_the_sample_replays_as_stored_and_the_same_twice(tool, episodes):
    stored = [(call["cls"], call["shown"]) for episode in episodes for call in episode["calls"]]
    assert 60 <= len(stored) <= 150 and len(episodes) >= 12
    first = replay(tool, episodes)
    assert first == replay(tool, episodes)                               # no clock, no random number, no state between episodes
    differ = [(k, a[0], b[0]) for k, (a, b) in enumerate(zip(first, stored)) if a != b]
    assert not differ, differ[:5]         # after a deliberate change of calc: python tools/audit/calc_corpus.py --refresh-sample <this file>


def test_what_the_sample_still_refuses_is_the_models_own_slip(tool, episodes):
    calls = [call for episode in episodes for call in episode["calls"]]
    before = sum(call["was_refused"] for call in calls)
    after = [call["cls"] for call in calls if call["cls"] != "ok"]
    assert before >= 20 and len(after) <= before // 2                    # the sample is drawn from episodes with a refusal
    assert all(cls.startswith(tool.SLIPS) for cls in after), sorted(set(after))
    lifted = [call for call in calls if call["was_refused"] and call["cls"] == "ok"]
    assert sum(call["was_limit"] for call in lifted) >= 3                # programs the old step limit stopped, finishing now


def test_the_reading_of_a_recorded_call(tool):
    assert tool.text_of('{"expressions": "a = 1"}') == "a = 1" and tool.text_of('{"expressions": ["a = 1", "a"]}') == "a = 1\na"
    assert tool.text_of("{}") is None and tool.text_of('{"express": "1"}') is None and tool.text_of("not json") is None
    assert tool.classify(None) == "ok" and tool.classify(SyntaxError("x")) == "not Python"
    assert tool.classify(C.CalcError("unknown name 'x'", "unknown")) == "unknown name"
    assert tool.classify(C._Stop("stopped after 5 steps")) == "step limit"
    assert tool.classify(C.CalcError("index 5 is out of range: the list has 3 element(s)", "value")).startswith("agent: ")
    assert tool.classify(C.CalcError("no sets in calc: use a list", "unsupported")).startswith("calc has no: ")
    assert tool.same_answer("3.5\n[calls 4/48]", "3.5") and not tool.same_answer("3.5\n[calls 4/48]", "3.6")
    assert tool.same_answer("[calls 4/48]", "") and tool.same_answer("calc: nothing to evaluate\n[calls 4/48]", "")
