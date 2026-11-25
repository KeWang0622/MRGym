"""The training-facing side of a family environment (``mrgym.learn.hooks``, ``mrgym.learn.shaping``). No model is called.

What is shown
  (a) schema: on random-policy episodes of every level, each step's info validates, the episode has exactly one
      outcome of the closed taxonomy, at its last step and only there, and the ladder flags form a prefix
  (b) shaping: the potential-based step rewards of an episode sum to its terminal reward to 1e-12, for a pass, a
      failure, a refused deliverable, truncation by the tool-call limit, by the turn cap and by the harness
  (c) groups: the same seed and calls give identical observations; another ``noise_nonce`` gives the same task and the
      same verdict on an answer, and other pixel noise
  (d) a cap on turns, when set, is a limit like the others: the Limits block, ``info["limits"]``, the ``status`` tool and
      the status line state it; against the stub endpoint with ``max_turns=5`` the episode ends after turn 5 and the
      last status line reads ``turn 5 of 5``. Without it the tool-call limit is the cap, as the interface states it
  (f) error classes and counts are the interface's own (``mrgym.interface``): nothing is classed or counted twice
  (e) level 0: the hints are in the statement, the first probe is in the log and charged, the gate does not claim it
"""
import re

import numpy as np
import pytest

import mrgym
from mrgym import interface as I
from mrgym import policies
from mrgym.agents import chat, runner, stub_server
from mrgym.envs import family as F
from mrgym.envs.null_tissue import task
from mrgym.learn import hooks
from mrgym.learn.shaping import Shaped

NULL = "Scanner/NullTissue-v0"
CHEAP = {"sequence": "spgr", "matrix": 32, "tr_ms": 12.0, "flip_deg": 8.0}
PROBE = {"sequence": "ir", "matrix": 32, "partial_fourier": 0.6, "tr_ms": 400.0}


def _call(name, **arguments):
    return {"name": name, "arguments": arguments}


def _random_call(rng, env):
    ids = [r["id"] for r in env.episode_config()["task"]["regions"]]
    kind = rng.choice(["scan", "probe", "plan", "bad_plan", "format", "view", "overspend", "bad_submit", "submit"],
                      p=[0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.08, 0.12])
    if kind == "scan":
        return _call("scan", **CHEAP)
    if kind == "probe":
        return _call("scan", **dict(PROBE, ti_ms=float(rng.choice([80.0, 150.0, 300.0]))), regions=["all"])
    if kind == "plan":
        return _call("plan", **CHEAP)
    if kind == "bad_plan":
        return _call("plan", sequence="spgr", tr_ms=1.0)
    if kind == "format":
        return {"name": "scan", "arguments": "{not json"}
    if kind == "view":
        return _call("view", scan_index=0)
    if kind == "overspend":
        return _call("scan", sequence="ir", tr_ms=6000.0, ti_ms=500.0, matrix=256)
    if kind == "bad_submit":
        return _call("submit", ti_ms={ids[0]: 1e9})
    return _call("submit", ti_ms={i: float(rng.uniform(100.0, 900.0)) for i in ids})


# ------------------------------------------------------------------ (a) schema, outcome, ladder
@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_schema_one_outcome_and_a_prefix_ladder_on_random_episodes(level):
    rng, seen = np.random.default_rng(100 + level), set()
    for episode in range(5):
        cap = [None, 4, None, 3, None][episode]
        env = mrgym.make(NULL, level=level, seed=f"api:{episode}", max_turns=cap, max_steps=[48, 48, 6, 48, 48][episode])
        _, info = env.reset()
        assert info["group"]["env_id"] == NULL and info["group"]["level"] == level and info["max_turns"] == cap
        assert env.outcomes() == F.outcomes(task)
        final, n_final = None, 0
        for turn in range(12):
            note = env.note_turn()
            if note["truncated"]:
                final = note
                break
            r = env.step(_random_call(rng, env))
            problems = hooks.validate(r.info, env.outcomes(), final=r.done)
            assert not problems, problems
            assert r.info["step"]["turn"] == turn + 1 and (r.reward == 0.0 or r.terminated)
            if r.done:
                final, n_final = r.info, n_final + 1
                break
        if final is None:
            final = env.end_without_submission("stopped" if episode % 2 else "length")
        assert final["outcome"] in env.outcomes() and n_final <= 1
        assert env.training_info()["outcome"] == final["outcome"] and env.state()["done"]
        flags = [final["ladder"][k] for k in hooks.LADDER]
        assert all(a or not b for a, b in zip(flags, flags[1:]))        # each flag implies the ones before it
        assert final["reward"] == float(final["passed"]) == float(final["outcome"] == "pass")
        assert (final["truncated_by"] is None) == final["ladder"]["submitted"]
        counts = final["counts"]
        assert counts["turns"] <= (cap or 12) and counts["executed"] <= env.max_steps and tuple(counts) == hooks.COUNT_KEYS
        assert {k: counts[k] for k in hooks.INTERFACE_COUNTS} == env.interface_counts()     # the interface's numbers, not a second count
        assert counts["calls"] == counts["executed"] + counts["rejected"] and counts["rejected"] <= env.limits()["max_rejected_calls"]
        seen.add(final["outcome"])
        with pytest.raises(mrgym.ResetNeeded):
            env.step(_call("scanner_info"))
    assert len(seen) >= 2, seen                                          # the random policy ends in more than one way


def test_each_way_an_episode_ends_has_its_outcome():
    def fresh(**kw):
        env = mrgym.make(NULL, level=1, seed=11, **kw)
        env.reset()
        return env, [r["id"] for r in env.episode_config()["task"]["regions"]]

    env, ids = fresh(max_turns=2)                                        # the visible turn cap
    assert env.note_turn()["turns_left"] == 1 and env.note_turn()["turns_left"] == 0
    end = env.note_turn()
    assert end["truncated"] and end["outcome"] == "no_submission_cap" and end["truncated_by"] == "max_turns" and end["reward"] == 0.0
    with pytest.raises(mrgym.ResetNeeded):
        env.note_turn()
    env, ids = fresh(max_steps=2)                                        # the tool-call limit
    env.step(_call("scanner_info"))
    r = env.step(_call("scanner_info"))
    assert r.truncated and r.info["outcome"] == "no_submission_cap" and r.info["truncated_by"] == "max_steps"
    env, ids = fresh()                                                   # the interface's limit on rejected calls
    for _ in range(env.limits()["max_rejected_calls"]):
        r = env.step(_call("nope"))
    assert r.truncated and r.info["reason"] == "rejected_call_limit" and r.info["truncated_by"] == "rejected_call_limit"
    assert r.info["outcome"] == "no_submission_rejected" and r.info["counts"]["vocabulary"] == r.info["counts"]["rejected"] == 8
    assert r.info["counts"]["executed"] == 0 and r.info["step"]["rejected_calls_left"] == 0
    for reason, name in (("stopped", "no_submission_stopped"), ("length", "no_submission_length")):
        env, ids = fresh()
        env.note_length_cut()
        end = env.end_without_submission(reason)
        assert end["outcome"] == name and end["counts"]["length_cuts"] == 1 and env.end_without_submission(reason) == end
    env, ids = fresh()                                                   # a deliverable with a region missing: a rejected call of class task
    r = env.step(_call("submit", ti_ms={ids[0]: 400.0}))
    assert not r.done and (r.info["step"]["error_class"], r.info["step"]["error_type"], r.info["step"]["counted"]) == ("task", "invalid_value", False)
    end = env.end_without_submission("stopped")
    assert end["outcome"] == "no_submission_stopped" and end["counts"]["refused_deliverables"] == 0 and end["counts"]["rejected"] == 1
    env = mrgym.make(NULL, level=2, seed=11)                             # a deliverable the console refuses, then nothing
    env.reset()
    t = env.episode_config()["task"]
    refusable = _call("submit", ti_ms={x["id"]: t["target_scan"]["tr_ms"] - 1.0 for x in t["regions"]})
    r = env.step(refusable)
    assert not r.done and (r.info["step"]["error_class"], r.info["step"]["error_type"]) == ("task", "invalid_deliverable")
    end = env.end_without_submission("stopped")
    assert end["outcome"] == "invalid_deliverable" and end["counts"]["refused_deliverables"] == 1 and not end["ladder"]["submitted"]
    env = mrgym.make(NULL, level=2, seed=11)                             # ... or refused until the interface ends the episode
    env.reset()
    for _ in range(env.limits()["max_invalid_deliverables"]):
        r = env.step(refusable)
    assert r.truncated and r.info["truncated_by"] == "rejected_call_limit" and r.info["outcome"] == "invalid_deliverable"
    assert r.info["counts"]["refused_deliverables"] == 3 == r.info["counts"]["task_refusals"]
    env, ids = fresh()                                                   # delivered without a measurement
    r = env.step(_call("submit", ti_ms={i: 400.0 for i in ids}))
    assert r.terminated and r.info["outcome"] == "no_measurement" and r.info["progress"] == 0.0
    assert r.info["ladder"] == dict(submitted=True, valid=True, measured=False, in_budget=False, passed=False)
    env, ids = fresh()                                                   # measured, delivered, wrong
    assert env.step(_call("scan", **CHEAP)).info["step"]["cost_s"] > 0
    r = env.step(_call("submit", ti_ms={i: 25.0 for i in ids}))
    assert r.terminated and r.info["outcome"] in task.OUTCOMES and r.info["ladder"]["in_budget"] and not r.info["passed"]
    assert r.info["distance"]["regions"] == len(ids) and r.info["counts"]["scans"] == 1
    for not_the_harness in ("max_steps", "rejected_call_limit", "other"):
        with pytest.raises(mrgym.Error):
            fresh()[0].end_without_submission(not_the_harness)
    with pytest.raises(mrgym.Error):
        mrgym.make(NULL, level=1, seed=0, max_turns=0)


def test_error_classes_and_counts_are_the_interfaces():
    env = mrgym.make(NULL, level=2, seed=5)
    env.reset()
    steps = [env.step({"name": "scan", "arguments": "{not json"}), env.step(_call("nope")), env.step(_call("plan", sequence="spgr", tr_ms=1.0)),
             env.step(_call("scan", sequence="ir", tr_ms=6000.0, ti_ms=500.0, matrix=256)), env.step(_call("view", scan_index=7)),
             env.step(_call("scan", protocol=CHEAP)), env.step(_call("scan", **CHEAP))]
    got = [(s.info["step"]["error_class"], s.info["step"]["error_type"], s.info["step"]["counted"]) for s in steps]
    assert got == [("shape", "malformed_call", False), ("vocabulary", "unknown_tool", False), ("task", "invalid_value", False),
                   ("task", "budget", True), ("task", "not_found", True), (None, None, True), (None, None, True)]
    for s in steps:                                                      # the same class and charge as the observation itself states
        e = s.observation.data.get("error")
        assert s.info["step"]["error_class"] == I.error_class(s.observation.data) and s.info["step"]["ok"] == (e is None)
        assert s.info["step"]["counted"] == I.counts_as_tool_call(s.observation.data) and not hooks.validate(s.info, env.outcomes(), final=False)
    last = steps[-1].info["step"]
    assert (last["tool_calls_left"], last["rejected_calls_left"], last["turns_left"]) == (env.max_steps - 4, 8 - 3, None)
    assert last["tool_calls_left"] == env.state()["tool_calls_left"] and last["cost_s"] > 0
    end = env.end_without_submission("stopped")["counts"]
    assert {k: end[k] for k in hooks.INTERFACE_COUNTS} == env.interface_counts() == dict(
        calls=7, executed=4, rejected=3, shape=1, vocabulary=1, task_refusals=3, coerced=1, internal_errors=0)
    assert (end["scans"], end["refused_deliverables"], end["turns"]) == (2, 0, 0)


# ------------------------------------------------------------------ (b) shaping
def _shaped_episode(level, seed, play, **kw):
    env = Shaped(mrgym.make(NULL, level=level, seed=seed, **kw))
    env.reset()
    return env, play(env)


def test_shaped_step_rewards_sum_to_the_terminal_reward():
    def run_policy(name):
        def play(env):
            answer = policies.controls_of(env).policies()[name](policies.ToolScanner(env))
            return env.step(_call("submit", **answer)).reward
        return play

    def probes_then(end):
        def play(env):
            ids = [r["id"] for r in env.episode_config()["task"]["regions"]]
            for ti in (80.0, 150.0, 300.0, 300.0):
                env.note_turn()
                r = env.step(_call("scan", **dict(PROBE, ti_ms=ti), regions=["all"]))
                assert not r.done and r.info["shaping"]["step_reward"] in (0.0, pytest.approx(0.1))
            return end(env, ids)
        return play

    cases = [
        ("pass", 2, 3, run_policy("reference"), {}, 1.0),
        ("failure", 2, 3, run_policy("textbook"), {}, 0.0),
        ("no measurement", 1, 3, run_policy("prior"), {}, 0.0),
        ("level 0, the probe acquired at reset earns nothing", 0, 3, run_policy("reference"), {}, 1.0),
        ("unreadable deliverable, then stopped", 1, 4, probes_then(lambda env, ids: (env.step(_call("submit", ti_ms={i: 1e9 for i in ids})),
                                                                     env.end_without_submission("stopped"))[1]["reward"]), {}, 0.0),
        ("tool-call limit", 1, 4, probes_then(lambda env, ids: [env.step(_call("scanner_info")) for _ in range(2)][-1].reward),
         dict(max_steps=6), 0.0),
        ("turn cap", 1, 4, probes_then(lambda env, ids: env.note_turn()["reward"]), dict(max_turns=4), 0.0),
        ("reply cut", 1, 4, probes_then(lambda env, ids: env.end_without_submission("length")["reward"]), {}, 0.0),
    ]
    for label, level, seed, play, kw, expected in cases:
        env, terminal = _shaped_episode(level, seed, play, **kw)
        assert env.state()["done"], label
        paid = env.rewards
        assert abs(sum(paid) - env.training_info()["reward"]) < 1e-12, (label, paid)
        assert env.training_info()["reward"] == expected, label
        if label not in ("no measurement",):
            assert any(abs(x) > 0.05 for x in paid[:-1]), label          # something was paid on the way, and taken back


def test_a_bonus_that_is_never_taken_back_is_caught_by_the_same_sum():
    """What the sum protects against: a per-scan bonus pays without the outcome."""
    env = mrgym.make(NULL, level=1, seed=4)
    env.reset()
    bonus = [0.1 * (not env.step(_call("scan", **CHEAP)).observation.is_error) for _ in range(3)]
    assert abs(sum(bonus) + env.end_without_submission("stopped")["reward"] - 0.0) > 0.2


# ------------------------------------------------------------------ (c) groups and noise
def test_same_seed_same_observations_and_a_noise_nonce_changes_only_the_noise():
    calls = [_call("scan", **dict(PROBE, ti_ms=200.0), regions=["all"]), _call("view", scan_index=0), _call("plan", **CHEAP)]

    def play(nonce=None):
        env = mrgym.make(NULL, level=2, seed=21)
        _, info = env.reset(options=None if nonce is None else {"noise_nonce": nonce})
        obs = [env.step(c).observation for c in calls]
        return env, info, obs

    a, info_a, obs_a = play()
    b, info_b, obs_b = play()
    c, info_c, obs_c = play(7)
    assert [o.text for o in obs_a] == [o.text for o in obs_b] and info_a["group"] == info_b["group"]
    assert info_a["group"]["id"] == info_c["group"]["id"] and info_c["group"]["noise_nonce"] == 7 and info_a["group"]["noise_nonce"] == 0
    assert a.episode_config()["task"] == c.episode_config()["task"]      # the same task card
    assert obs_a[0].data["realised"] == obs_c[0].data["realised"] and obs_a[0].data["cost_s"] == obs_c[0].data["cost_s"]
    assert obs_a[0].data["data_sha256"] != obs_c[0].data["data_sha256"]  # other pixel noise
    answer = dict(ti_ms={r["id"]: 400.0 for r in a.episode_config()["task"]["regions"]})
    ends = [e.step(_call("submit", **answer)) for e in (a, c)]
    assert ends[0].info["score"]["null_ratio"] == ends[1].info["score"]["null_ratio"]    # the verifier does not depend on it
    assert all(e.info["replay"]["ok"] for e in ends)
    with pytest.raises(mrgym.Error):
        mrgym.make(NULL, level=2, seed=21).reset(options={"noise_nonce": -1})
    with pytest.raises(mrgym.Error):
        mrgym.make(NULL, level=2, seed=21).reset(options={"unknown": 1})


# ------------------------------------------------------------------ (d) the turn cap
def test_a_cap_on_turns_is_stated_like_every_other_limit_and_only_when_set():
    env = mrgym.make(NULL, level=1, seed=0)
    obs, info = env.reset()
    assert "turn" not in obs.text.splitlines()[-1] and "turn" not in obs.status and "max_turns" not in info["limits"] and "Turns:" not in obs.text
    env = mrgym.make(NULL, level=1, seed=0, max_turns=3)
    obs, info = env.reset()
    assert info["limits"]["max_turns"] == 3 == obs.data["limits"]["max_turns"] == env.limits()["max_turns"] == info["max_turns"]
    assert "Turns: 3 (a turn is one reply of yours" in obs.text and obs.text.splitlines()[-1].endswith("· turn 0 of 3]")
    assert obs.status["max_turns"] == 3 and env.state()["turn"] == 0
    env.note_turn()
    assert env.step(_call("scanner_info")).observation.text.splitlines()[-1].endswith("· turn 1 of 3]")
    env.note_turn()                                                      # one turn is left: the line says what to do with it
    step = env.step(_call("status"))
    assert step.observation.text.splitlines()[-1].endswith("· turn 2 of 3, submit in your next reply or the episode scores 0]")
    assert step.observation.data["limits"]["max_turns"] == 3 and step.info["step"]["turns_left"] == 1
    env.note_turn()
    assert env.step(_call("scanner_info")).observation.text.splitlines()[-1].endswith("· turn 3 of 3, no turn left]")
    end = env.note_turn()
    assert end["truncated"] and end["outcome"] == "no_submission_cap" and env.state()["turn"] == 3


def test_against_the_stub_the_episode_ends_at_the_turn_cap():
    server, url = stub_server.start(seeds=[0], env_ids=[NULL], levels=[1])
    try:
        client = chat.ChatClient(url, "stub:reference", None, {})
        env = mrgym.make(NULL, level=1, seed=0, max_turns=5)
        played = runner.play_model(env, client)
        tool_texts = [m["content"] for m in played["messages"] if m["role"] == "tool"]
        # the reply that would be turn 6 is discarded: the environment has truncated the episode
        assert played["stop_reason"] == "truncated" and played["turns"] == 6 and played["final"] is None and env.state()["done"]
        assert [re.search(r"turn (\d) of 5", t.splitlines()[-1])[1] for t in tool_texts] == ["1", "2", "3", "4", "5"]
        assert tool_texts[-1].splitlines()[-1].endswith("turn 5 of 5, no turn left]") and "next reply or the episode scores 0" in tool_texts[-2]
        end = env.training_info()
        assert end["outcome"] == "no_submission_cap" and end["counts"]["turns"] == 5 and end["truncated_by"] == "max_turns"
        out = runner._outcome(env, played["final"], played["stop_reason"])
        assert out["outcome"] == "no_submission_cap" and not out["passed"] and out["counts"]["executed"] == 5
        # the cap of the runner and of `mrgym calibrate` is the environment's tool-call limit: stated by the interface, no hook needed
        env = runner.make_env(NULL, 1, 0, dict(max_turns=4))
        played = runner.play_model(env, client)
        assert played["stop_reason"] == "truncated" and played["final"].info["reason"] == "max_steps" and env.max_steps == 4
        assert played["final"].info["outcome"] == "no_submission_cap" and played["final"].info["truncated_by"] == "max_steps"
        assert "calls 4/4" in played["final"].observation.text.splitlines()[-1]
        # without a cap in the way the same model passes, and the record of the runner carries the outcome
        env = mrgym.make(NULL, level=1, seed=0)
        played = runner.play_model(env, client)
        out = runner._outcome(env, played["final"], played["stop_reason"])
        assert played["stop_reason"] == "submitted" and out["passed"] and out["outcome"] == "pass" and out["progress"] == 1.0
        assert out["ladder"]["passed"] and out["checks"]["measured"] and out["group"].startswith(NULL)
        assert out["counts"]["turns"] == played["turns"] and out["counts"]["shape"] == out["counts"]["vocabulary"] == 0
    finally:
        server.shutdown()


# ------------------------------------------------------------------ (e) level 0
def test_level_0_states_the_procedure_and_acquires_the_first_probe():
    env = mrgym.make(NULL, level=0, seed=2)
    obs, _ = env.reset()
    log = env.scanner_log()
    assert "How to do it" in obs.text and "The first probe is already acquired: scan 0" in obs.text
    assert len(log) == 1 and log[0]["type"] == "scan" and log[0]["realised"]["ti_ms"] == task.FIRST_PROBE["ti_ms"]
    assert obs.status["scans"] == 1 and obs.status["scan_time_used_s"] == log[0]["cost_s"] > 0 and obs.status["tool_calls"] == 0
    assert env.episode_config()["task"]["hints"] == ["rule", "first_measurement"]
    for r in env.episode_config()["task"]["regions"]:                    # the read-out in the text is the one in the log
        assert f"  {r['id']:4s} |complex mean|" in obs.text
    assert not env.step(_call("view", scan_index=0)).observation.is_error      # the probe is a stored scan like any other
    other = mrgym.make(NULL, level=1, seed=2)
    text = other.reset()[0].text
    assert "How to do it" not in text and "hints" not in other.episode_config()["task"] and other.scanner_log() == []
    ref = policies.run(mrgym.make(NULL, level=0, seed=2), "reference")
    assert ref.reward == 1.0 and ref.info["replay"]["ok"] and ref.info["budget"]["used_s"] <= task.LEVELS[0]["budget_s"]


def test_level_0_is_registered_and_outside_the_gate_claim():
    import inspect

    from mrgym import gate
    assert mrgym.spec(NULL).levels == (0, 1, 2, 3) and task.LEVELS[0]["kind"] == "practice"
    assert mrgym.card(NULL)["practice_levels"] == [0]
    assert all(lv.get("kind") != "practice" for k, lv in task.LEVELS.items() if k != 0)
    assert "practice" in inspect.getsource(gate.run)                     # the default level list of the gate leaves it out
    passes = {"reference": np.ones(4, bool), "prior": np.zeros(4, bool), "fixed_recipe#0": np.zeros(4, bool),
              "brute_force": np.zeros(4, bool), "wrong_units": np.zeros(4, bool), "textbook": np.zeros(4, bool)}
    assert gate.decide(passes, policies.controls_of(mrgym.spec(NULL)), 0, 4)["ablations"][0]["required"] is False
