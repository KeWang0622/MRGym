"""The rows of the scoring tables (``tests/scoring/``) through ``reset`` / ``step``, for every family.

  gates     the two gates of ``family.py`` in every combination, through ``FamilyEnv.finish``
  episodes  a refusal does not end the episode; a pass without a scan earns 0; a pass after a scan earns exactly 1.0;
            a fail earns exactly 0.0; the agent is not shown the score unless ``reveal_score`` is set
"""
import itertools

import pytest

import mrgym
import scoring
from mrgym import policies
from mrgym.envs import family as F

MODULES = scoring.registered_modules()
FAMILIES = [pytest.param(m, id=name) for name, m in MODULES.items()]
CHEAP_SCAN = dict(sequence="epi", matrix=32, tr_ms=50.0)        # 3.15 s: fits every budget


def _episode(module, key, scans=1, **kw):
    env = mrgym.make(module.ENV_ID, **module.INSTANCES[key], **kw)
    env.reset()
    for _ in range(scans):
        policies.ToolScanner(env).scan(CHEAP_SCAN)
    return env


def _submit(env, answer):
    return env.step({"name": "submit", "arguments": answer})


def _refused_through_step(module, key, case):
    """The rows the verifier refuses that are also refused as a call. A deliverable whose shape has one reading (the
    numbers of a nested field written without the field) is refused by ``task.check`` on the verifier's side, and
    repaired by the interface before it gets there (rule C15): such a row is delivered, not refused."""
    env = _episode(module, key, scans=0)
    rows = [r for r in module.ROWS if r.refused and r.instance == key]

    def repaired(row):
        if not isinstance(case.answer(row), dict):
            return False
        _, _, coercions, problems = env.admit({"name": "submit", "arguments": case.answer(row)})
        return not problems and "C15" in {c["rule"] for c in coercions}
    moved = [r for r in rows if repaired(r)]
    env.close()
    return [r for r in rows if r not in moved], moved


# ---------------------------------------------------------------------------------------------- the gates of family.py
@pytest.mark.parametrize("module", FAMILIES)
def test_gates_in_every_combination(module):
    """reward = 1 exactly when the deliverable passes, the log replays within budget and shows a scan."""
    good, case = scoring.first_row(module, passed=True)
    bad = next(r for r in module.ROWS if r.passed is False and r.instance == good.instance)
    env = _episode(module, good.instance, scans=0)
    for answer, passes in ((case.answer(good), True), (case.answer(bad), False)):
        for ok, over, scans in itertools.product((True, False), (False, True), (0, 1, 3)):
            reward, score, text = env.finish(answer, dict(ok=ok, over_budget=over, scans=scans))
            log_gate, measured = ok and not over, scans > 0
            want = 1.0 if (passes and log_gate and measured) else 0.0
            assert type(reward) is float and reward == want == score["reward"], (passes, ok, over, scans)
            assert score["passed"] is (want == 1.0)
            assert score["checks"]["log_replays_within_budget"] is log_gate and score["checks"]["measured"] is measured
            assert set(F.GATES) <= set(score["checks"]) and all(type(v) is bool for v in score["checks"].values())
            gates_failing = [g for g, v in (("log_replays_within_budget", log_gate), ("measured", measured)) if not v]
            assert scoring.failing(score) == sorted(gates_failing + ([] if passes else list(bad.failing)))
            assert text == "deliverable received; the episode is over"   # nothing about the score unless asked for
    env.close()


# ---------------------------------------------------------------------------------------------- through reset / step
@pytest.mark.parametrize("module", FAMILIES)
def test_a_refused_deliverable_does_not_end_the_episode_and_a_pass_without_a_scan_earns_nothing(module):
    """Every refused row of the table, each as the first call of a fresh episode (an episode allows only a few refusals)."""
    good, case = scoring.first_row(module, passed=True)
    reached_the_task_check = 0
    refused, repaired = _refused_through_step(module, good.instance, case)
    for row in repaired:                                                 # written flat: read as what it is, and delivered
        env = _episode(module, good.instance, scans=0)
        assert "C15" in {c["rule"] for c in env.admit({"name": "submit", "arguments": case.answer(row)})[2]}, row.name
        obs, reward, terminated, truncated, info = _submit(env, case.answer(row))
        assert terminated and not obs.is_error and info["submitted"] is True and reward == 0.0, row.name     # no scan: no reward
        env.close()
    assert len(refused) >= 2 and len(repaired) <= 1
    for k, row in enumerate(refused):
        env = _episode(module, good.instance, scans=0)
        before = env.state()
        obs, reward, terminated, truncated, info = _submit(env, case.answer(row))
        assert obs.is_error and not terminated and not truncated and reward == 0.0 and set(info) <= {"step"}, row.name
        err = obs.data["error"]
        assert obs.data["episode_continues"] is True and obs.data["charged"]["scan_time_s"] == 0.0, row.name
        if err["type"] == "invalid_deliverable":                         # past the interface: refused by task.check
            reached_the_task_check += 1
            assert row.reason in err["message"] and err["message"].endswith("; nothing was delivered")
        now = env.state()
        assert now["done"] is False and now["reward"] is None and now["tool_calls"] + now["rejected_calls"] == 1
        counters = ("tool_calls", "tool_calls_left", "rejected_calls")
        assert {key: now[key] for key in now if key not in counters} == {key: before[key] for key in before if key not in counters}
        if k >= 2:                                                       # what follows is the same for every row: twice is enough
            env.close()
            continue
        obs, reward, terminated, truncated, info = _submit(env, case.answer(good))      # the episode goes on: a pass, but no scan
        assert terminated and not truncated and not obs.is_error and type(reward) is float and reward == 0.0, row.name
        assert info["score"]["passed"] is False and scoring.failing(info["score"]) == ["measured"] and info["submitted"] is True
        assert info["replay"]["ok"] is True and info["replay"]["scans"] == 0
        with pytest.raises(mrgym.ResetNeeded):
            _submit(env, case.answer(good))
        env.close()
    if not reached_the_task_check:                                       # the schema of the terminal tool refuses every such row first:
        import warnings  # task.check is then reached by the verifier side only
        warnings.warn(f"{module.__name__}: no refused row gets past the interface; task.check is not exercised through step", stacklevel=1)


@pytest.mark.parametrize("module", FAMILIES)
def test_refusals_are_limited_and_the_limit_ends_the_episode_without_a_reward(module):
    """An episode that runs out of refused deliverables is truncated: reward 0, nothing submitted, a stated reason. A
    deliverable the console cannot evaluate, or one refused for how it is written, is a rejected call; one refused for
    what it asks (a region missing, a value out of its range: class task) is a tool call. Whichever limit comes first
    ends the episode."""
    good, case = scoring.first_row(module, passed=True)
    rows = _refused_through_step(module, good.instance, case)[0]
    env = _episode(module, good.instance)
    steps, before = [], env.state()["tool_calls"] + env.state()["rejected_calls"]        # the calls of the scan the episode starts with
    for k in range(env.max_steps + env.max_rejected_calls):
        obs, reward, terminated, truncated, info = _submit(env, case.answer(rows[k % len(rows)]))
        assert obs.is_error and not terminated and reward == 0.0
        steps.append(truncated)
        if truncated:
            break
    state = env.state()
    assert steps[-1] is True and not any(steps[:-1]) and len(steps) == state["tool_calls"] + state["rejected_calls"] - before
    assert info["submitted"] is False and obs.data["episode_continues"] is False
    by_rejected = state["rejected_calls"] >= env.max_rejected_calls or env._episode["invalid_deliverables"] >= env.max_invalid_deliverables
    assert info["reason"] == ("rejected_call_limit" if by_rejected else "max_steps") and (by_rejected or state["tool_calls"] == env.max_steps)
    assert info["interface"]["rejected"] == info["interface"]["shape"] + info["interface"]["vocabulary"] + env._episode["invalid_deliverables"]
    assert env.state()["done"] is True and env.state()["reward"] == 0.0
    env.close()


@pytest.mark.parametrize("module", FAMILIES)
def test_a_pass_after_a_scan_earns_one_and_the_agent_is_not_shown_the_score(module):
    good, case = scoring.first_row(module, passed=True)
    env = _episode(module, good.instance)
    used = env.state()["scan_time_used_s"]
    assert used > 0
    obs, reward, terminated, truncated, info = _submit(env, case.answer(good))
    assert terminated and not truncated and type(reward) is float and reward == 1.0
    assert info["score"]["passed"] is True and info["score"]["reward"] == 1.0 and scoring.failing(info["score"]) == []
    assert set(F.GATES) <= set(info["score"]["checks"])
    assert info["replay"]["ok"] and info["replay"]["scans"] == 1 and not info["replay"]["over_budget"]
    assert info["replay"]["used_s"] == info["budget"]["used_s"] == used   # the replayed log is the live episode
    assert obs.data == {"received": True}
    assert obs.text.splitlines()[0] == "deliverable received; the episode is over" and "reward" not in obs.text
    assert env.state()["reward"] == 1.0
    env.close()


@pytest.mark.parametrize("module", FAMILIES)
def test_a_fail_after_a_scan_earns_zero_and_reveal_score_says_why(module):
    bad, case = scoring.first_row(module, passed=False)
    env = _episode(module, bad.instance, reveal_score=True)
    obs, reward, terminated, truncated, info = _submit(env, case.answer(bad))
    assert terminated and type(reward) is float and reward == 0.0
    assert scoring.failing(info["score"]) == sorted(bad.failing) and info["score"]["reward"] == 0.0
    lines = obs.text.splitlines()
    assert lines[0] == "deliverable received; the episode is over"
    assert lines[1] == "reward 0 · failed: " + ", ".join(k for k, ok in info["score"]["checks"].items() if not ok)
    assert lines[2] == info["score"]["diagnostic"] and bad.reason in lines[2]
    assert obs.data == {"received": True, "score": info["score"]}
    env.close()
    good = next(r for r in module.ROWS if r.passed and r.instance == bad.instance)
    env = _episode(module, good.instance, reveal_score=True)
    obs, reward, *_ = _submit(env, case.answer(good))
    assert reward == 1.0 and obs.text.splitlines()[1] == "reward 1 · every check passed"
    assert len(obs.text.splitlines()) == 3                               # header, verdict, status line: no diagnostic
    env.close()
