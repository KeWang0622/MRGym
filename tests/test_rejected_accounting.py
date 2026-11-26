"""What a refused call is charged as: one rule, and the counters that follow from it.

    A call refused with class ``shape`` or ``vocabulary`` (for how it is written) is a rejected call (``max_rejected_calls``).
    A call refused with class ``task`` (for what it asks) is a tool call (``max_steps``), whatever its type.
    ``invalid_deliverable``, ``episode_over`` and ``internal_error`` are rejected calls whatever their class.
    A refusal with problems of several classes is a rejected call when any of them is ``shape`` or ``vocabulary``.
    No refusal is charged scan time.

    (a) calculator run-time errors never end an episode as rejected calls; they are tool calls, up to ``max_steps``
    (b) eight malformed calls (unknown tool, unknown parameter, wrong type) still end it with ``rejected_call_limit``
    (c) a refusal with problems of several classes
    (d) the counters of ``info["interface"]`` / the status add up; the identity is stated in ``_check_counters``
    (e) the limit on refused deliverables is unchanged
    (f) the table, the Limits block and the status line say the rule
"""
import pytest

import mrgym
from mrgym import interface as I

NOWRAP, NULL = "Scanner/NoWrap-v1", "Scanner/NullTissue-v0"
CENTRED = {"centre_y_mm": 0, "centre_x_mm": 0, "rotation_deg": 0}
BAD_GEOMETRY = dict(pe_dir="y", fov_mm=130, phase_fov=0.5, matrix=65, **CENTRED)        # the console cannot evaluate it: invalid_deliverable
RUNTIME = ["x = undefined_name",                                         # an unknown name
           "v = [1, 2, 3]\nprint(v[7])",                                 # an index out of range
           "d = {'a': 1}\nprint(d['b'])",                                # a key that is not there
           "print(brentq(lambda t: t * t + 1, 0, 1))"]                   # no sign change
MALFORMED = [{"name": "budget", "arguments": {}},                                               # unknown tool
             {"name": "plan", "arguments": {"sequence": "se", "tr_ms": 500, "shim": "auto"}},   # unknown parameter
             {"name": "plan", "arguments": {"sequence": "se", "tr_ms": "soon"}}]                # wrong type


def make(env_id=NULL, level=1, seed=0, **kw):
    env = mrgym.make(env_id, level=level, seed=seed, **kw)
    env.reset()
    return env


def call(env, name, **arguments):
    return env.step({"name": name, "arguments": arguments})


def _check_counters(env, answered: int, refused_deliverables: int):
    """The identity of the counters, at any point of an episode.

        calls     == executed + rejected
        rejected  == shape + vocabulary + refused deliverables + internal_errors
        executed  == answered (no error) + task_refusals - refused deliverables

    ``task_refusals`` counts every refusal of class ``task``; of those only a refused deliverable (``invalid_deliverable``)
    is a rejected call. The status tool and ``env.state()`` give the same numbers as ``interface_counts``."""
    c, s = env.interface_counts(), env.state()
    assert c["calls"] == c["executed"] + c["rejected"]
    assert c["rejected"] == c["shape"] + c["vocabulary"] + refused_deliverables + c["internal_errors"]
    assert c["executed"] == answered + c["task_refusals"] - refused_deliverables
    assert (s["tool_calls"], s["rejected_calls"]) == (c["executed"], c["rejected"])
    assert env._episode["invalid_deliverables"] == refused_deliverables
    return c


# ------------------------------------------------------------------------------------------------ the rule itself
@pytest.mark.parametrize("problems, counted", [
    ([I.problem("invalid_value", "expressions", "unknown name 'x'", "task")], True),
    ([I.problem("invalid_value", "expressions", "calc has no import", "shape")], False),
    ([I.problem("invalid_value", "pe_dir", "not one of y, x", "vocabulary")], False),
    ([I.problem("invalid_value", "fov_mm", "above the maximum", "task"), I.problem("invalid_value", "tr_ms", "expected a number", "shape")], False),
    ([I.problem("invalid_value", "fov_mm", "above the maximum", "task"), I.problem("unknown_parameter", "shim", "no parameter shim")], False),
    ([I.problem("invalid_value", "fov_mm", "above the maximum", "task"), I.problem("not_found", "scan_index", "no scan 7")], True),
    ([I.problem("console_limit", "tr_ms", "below the minimum")], True),
    ([I.problem("budget", None, "no scan time left")], True),
    ([I.problem("not_found", "scan_index", "no scan 7")], True),
    ([I.problem("malformed_call", None, "not JSON")], False),
    ([I.problem("unknown_tool", None, "no tool")], False),
    ([I.problem("not_applicable", "ti_ms", "se has no ti_ms")], False),
    ([I.problem("invalid_deliverable", "matrix", "odd")], False),
    ([I.problem("episode_over", None, "over")], False),
    ([I.problem("internal_error", None, "a fault")], False),
])
def test_the_class_of_a_refusal_decides_what_it_is_charged_as(problems, counted):
    data = I.error_data(problems)
    cls = data["error"]["class"]
    assert I.counts_as_tool_call(data) is counted
    assert counted == (cls == "task" and data["error"]["type"] not in I.REJECTED_TYPES)
    assert (cls in I.INTERFACE_CLASSES) == any(p["class"] in I.INTERFACE_CLASSES for p in problems)      # any shape / vocabulary problem: rejected
    assert I.counts_as_tool_call({"values": []}) and I.counts_as_tool_call({}) and I.counts_as_tool_call(None)   # an answer is a tool call
    assert not I.counts_as_tool_call({"error": {"type": "no_such_type", "class": "task"}})                # outside the closed vocabulary


def test_the_table_states_the_rule_for_every_type():
    assert I.REJECTED_TYPES == ("invalid_deliverable", "episode_over", "internal_error") and I.INTERFACE_CLASSES == ("shape", "vocabulary")
    for type, (cls, counted) in I.ERRORS.items():
        if cls is None:                                                  # invalid_value: each problem carries its class, the class decides
            assert type == "invalid_value" and counted is None
            for c in I.CLASSES[:3]:
                assert I.counts_as_tool_call(I.error_data([I.problem(type, "p", "m", c)])) == (c == "task")
            continue
        assert counted == (cls == "task" and type not in I.REJECTED_TYPES), type
        assert I.counts_as_tool_call(I.error_data([I.problem(type, None, "m")])) == counted, type
    # the class decides whatever the type: a type of the vocabulary classes, said by the instrument, is a tool call
    assert I.counts_as_tool_call(I.error_data([I.problem("not_applicable", "p", "m", "task")]))
    assert not I.counts_as_tool_call(I.error_data([I.problem("not_found", "p", "m", "vocabulary")]))


# ------------------------------------------------------------------------------------------------ (a)
def test_calculator_run_time_errors_are_tool_calls_and_never_reach_the_rejected_limit():
    env = make(max_steps=12)
    assert env.max_rejected_calls == 8
    used = env.state()["scan_time_used_s"]
    for k in range(1, 12):                                               # eight in a row and three more: the episode continues
        step = call(env, "calc", expressions=RUNTIME[k % len(RUNTIME)])
        obs, err = step.observation, step.observation.data["error"]
        assert obs.is_error and (err["type"], err["class"], err["parameter"]) == ("invalid_value", "task", "expressions"), obs.text
        assert not step.done and obs.data["episode_continues"] and obs.data["charged"] == dict(scan_time_s=0.0, tool_calls=1, rejected_calls=0)
        assert (env.state()["tool_calls"], env.state()["rejected_calls"], env.state()["scan_time_used_s"]) == (k, 0, used)
        # the status line shows the call among the calls, and says nothing of a rejected call
        assert obs.text.splitlines()[-1].startswith(f"[calls {k}/12") and "rejected" not in obs.text.splitlines()[-1]
    step = call(env, "calc", expressions=RUNTIME[0])                     # the twelfth: the tool-call limit, as for any tool call
    assert step.truncated and not step.terminated and step.reward == 0.0 and step.info["reason"] == "max_steps"
    assert step.info["interface"] == dict(calls=12, executed=12, rejected=0, shape=0, vocabulary=0, task_refusals=12, coerced=0, internal_errors=0)


def test_what_the_calculator_cannot_read_is_still_a_rejected_call():
    env = make()
    for k, (text, type) in enumerate([("import os", "invalid_value"), ("x = = 3", "malformed_call"), ("class A: pass", "invalid_value")], 1):
        obs = call(env, "calc", expressions=text).observation
        assert (obs.data["error"]["type"], obs.data["error"]["class"]) == (type, "shape"), obs.text
        assert obs.data["charged"] == dict(scan_time_s=0.0, tool_calls=0, rejected_calls=1) and env.state()["rejected_calls"] == k
        assert f"rejected {k}/8 (nothing charged)" in obs.text.splitlines()[-1]
    assert env.state()["tool_calls"] == 0


# ------------------------------------------------------------------------------------------------ (b)
def test_eight_malformed_calls_end_the_episode_at_the_rejected_call_limit():
    env = make()
    steps = [env.step(MALFORMED[k % len(MALFORMED)]) for k in range(8)]
    assert [s.observation.data["error"]["type"] for s in steps[:3]] == ["unknown_tool", "unknown_parameter", "invalid_value"]
    assert [s.observation.data["error"]["class"] for s in steps[:3]] == ["vocabulary", "vocabulary", "shape"]
    assert [s.done for s in steps] == [False] * 7 + [True] and steps[-1].truncated and steps[-1].reward == 0.0
    assert steps[-1].info["reason"] == "rejected_call_limit" and steps[-1].info["submitted"] is False
    assert "episode truncated: 8 calls rejected" in steps[-1].observation.text and "rejected 8/8 (nothing charged)" in steps[-1].observation.text
    assert steps[-1].info["interface"] == dict(calls=8, executed=0, rejected=8, shape=2, vocabulary=6, task_refusals=0, coerced=0, internal_errors=0)
    with pytest.raises(mrgym.ResetNeeded):
        call(env, "status")


def test_refusals_of_the_instrument_between_malformed_calls_do_not_bring_the_limit_closer():
    env = make()
    for k in range(7):                                                   # seven rejected calls, a refusal of the instrument after each
        assert not env.step(MALFORMED[k % len(MALFORMED)]).done
        assert not call(env, "calc", expressions=RUNTIME[k % len(RUNTIME)]).done
        assert not call(env, "plan", sequence="se").done                 # a required parameter is missing: class task
    assert (env.state()["tool_calls"], env.state()["rejected_calls"]) == (14, 7)
    last = env.step(MALFORMED[0])
    assert last.truncated and last.info["reason"] == "rejected_call_limit"
    assert last.info["interface"] == dict(calls=22, executed=14, rejected=8, shape=2, vocabulary=6, task_refusals=14, coerced=0, internal_errors=0)


# ------------------------------------------------------------------------------------------------ (c)
def test_a_refusal_with_problems_of_several_classes():
    env = make(NOWRAP)
    # out of range (task) next to a wrong type (shape), both of type invalid_value: the call could not be taken as written
    mixed = call(env, "plan", sequence="se", tr_ms="soon", fov_mm=600).observation
    err = mixed.data["error"]
    assert {p["parameter"]: p["type"] for p in err["problems"]} == {"tr_ms": "invalid_value", "fov_mm": "invalid_value"}
    assert (err["type"], err["class"], err["parameter"]) == ("invalid_value", "shape", "tr_ms")
    assert mixed.data["charged"] == dict(scan_time_s=0.0, tool_calls=0, rejected_calls=1)
    # out of range (task) next to an unknown parameter (vocabulary)
    mixed = call(env, "plan", sequence="se", tr_ms=500, fov_mm=600, shim=1).observation
    err = mixed.data["error"]
    assert {p["parameter"]: p["type"] for p in err["problems"]} == {"shim": "unknown_parameter", "fov_mm": "invalid_value"}
    assert err["class"] == "vocabulary" and mixed.data["charged"] == dict(scan_time_s=0.0, tool_calls=0, rejected_calls=1)
    # two problems of class task: the instrument said no twice, one tool call
    both = call(env, "plan", sequence="se", tr_ms="0.5 s", fov_mm=600).observation
    err = both.data["error"]
    assert {p["parameter"] for p in err["problems"]} == {"tr_ms", "fov_mm"} and err["class"] == "task"
    assert both.data["charged"] == dict(scan_time_s=0.0, tool_calls=1, rejected_calls=0)
    assert (env.state()["tool_calls"], env.state()["rejected_calls"]) == (1, 2)
    assert env.interface_counts() == dict(calls=3, executed=1, rejected=2, shape=1, vocabulary=1, task_refusals=1, coerced=0, internal_errors=0)


# ------------------------------------------------------------------------------------------------ (d)
def test_the_counters_add_up_at_every_step(monkeypatch):
    env = make(NOWRAP, max_steps=40, max_rejected_calls=12)
    script = [   # (call, error type or None, error class or None)
        ({"name": "status", "arguments": {}}, None, None),
        ({"name": "calc", "arguments": {"expressions": "a = 3\nprint(a * 2)"}}, None, None),
        ({"name": "calc", "arguments": {"expressions": "print(b)"}}, "invalid_value", "task"),
        ({"name": "calc", "arguments": {"expressions": "import os"}}, "invalid_value", "shape"),
        ({"name": "calc", "arguments": {"expressions": "a = = 3"}}, "malformed_call", "shape"),
        ({"name": "budget", "arguments": {}}, "unknown_tool", "vocabulary"),
        ({"name": "plan", "arguments": {"sequence": "se", "tr_ms": 500, "shim": 1}}, "unknown_parameter", "vocabulary"),
        ({"name": "plan", "arguments": {"sequence": "se", "tr_ms": 500, "ti_ms": 300}}, "not_applicable", "vocabulary"),
        ({"name": "plan", "arguments": {"sequence": "se", "tr_ms": 500, "pe_dir": "ROW"}}, "invalid_value", "vocabulary"),
        ({"name": "plan", "arguments": {"sequence": "se"}}, "invalid_value", "task"),                       # missing required
        ({"name": "plan", "arguments": {"sequence": "se", "tr_ms": "0.5 s"}}, "invalid_value", "task"),      # another unit
        ({"name": "plan", "arguments": {"sequence": "se", "tr_ms": 500, "fov_mm": 600}}, "invalid_value", "task"),    # out of range
        ({"name": "plan", "arguments": {"sequence": "se", "tr_ms": 50, "te_ms": 80}}, "console_limit", "task"),
        ({"name": "view", "arguments": {"scan_index": 7}}, "not_found", "task"),
        ({"name": "read_regions", "arguments": {"circles": [{"id": "a", "y_mm": 0, "x_mm": 0, "radius_mm": 5},
                                                             {"id": "a", "y_mm": 9, "x_mm": 0, "radius_mm": 5}]}}, None, None),   # set below
        ({"name": "submit", "arguments": BAD_GEOMETRY}, "invalid_deliverable", "task"),
        ({"name": "submit", "arguments": dict(pe_dir="y", fov_mm=240, matrix=120, **CENTRED)}, "invalid_value", "task"),   # a field is missing
        ({"name": "plan", "arguments": {"sequence": "se", "tr_ms": 500}}, None, None),
    ]
    answered = refused_deliverables = 0
    seen = set()
    for k, (tool_call, type, cls) in enumerate(script):
        if tool_call["name"] == "read_regions":                          # a read-out the scanner cannot take: two circles with one id
            assert not call(env, "scan", sequence="spgr", tr_ms=12, flip_deg=10, matrix=48).observation.is_error
            answered += 1
            type, cls = "invalid_value", "task"
        before = env.state()
        step = env.step(tool_call)
        obs, err = step.observation, step.observation.data.get("error")
        assert not step.done, obs.text
        assert (err and (err["type"], err["class"])) == ((type, cls) if type else None), (k, obs.text)
        answered += not obs.is_error
        refused_deliverables += type == "invalid_deliverable"
        counted = type is None or (cls == "task" and type != "invalid_deliverable")
        after = env.state()
        assert (after["tool_calls"] - before["tool_calls"], after["rejected_calls"] - before["rejected_calls"]) == (int(counted), int(not counted)), k
        assert after["scan_time_used_s"] == before["scan_time_used_s"] or not obs.is_error                # no refusal is charged scan time
        if obs.is_error:
            assert obs.data["charged"] == dict(scan_time_s=0.0, tool_calls=int(counted), rejected_calls=int(not counted)), k
            assert ("rejected" in obs.text.splitlines()[-1]) == (not counted), obs.text
            seen.add((type, cls))
        _check_counters(env, answered, refused_deliverables)
    assert seen == {("invalid_value", "task"), ("invalid_value", "shape"), ("invalid_value", "vocabulary"), ("malformed_call", "shape"),
                    ("unknown_tool", "vocabulary"), ("unknown_parameter", "vocabulary"), ("not_applicable", "vocabulary"),
                    ("console_limit", "task"), ("not_found", "task"), ("invalid_deliverable", "task")}
    monkeypatch.setattr(env, "_tool_status", lambda a: 1 / 0)             # a fault of the environment: a rejected call, counted for the trainer
    fault = call(env, "status").observation
    assert fault.data["error"]["type"] == "internal_error" and fault.data["charged"] == dict(scan_time_s=0.0, tool_calls=0, rejected_calls=1)
    monkeypatch.undo()
    c = _check_counters(env, answered, refused_deliverables)
    assert c == dict(calls=20, executed=12, rejected=8, shape=2, vocabulary=4, task_refusals=9, coerced=0, internal_errors=1)
    status = call(env, "status").observation                             # the status tool prints the same counters
    answered += 1
    c = _check_counters(env, answered, refused_deliverables)
    assert f"tool calls: {c['executed']} of 40 used" in status.text and f"rejected calls: {c['rejected']} of 12; refused submit calls: 1 of 3" in status.text
    assert status.data["used"]["tool_calls"] == c["executed"] and status.data["used"]["rejected_calls"] == c["rejected"]
    assert status.data["used"]["invalid_deliverables"] == refused_deliverables == 1


# ------------------------------------------------------------------------------------------------ (e)
def test_the_limit_on_refused_deliverables_is_unchanged():
    assert I.MAX_INVALID_DELIVERABLES == 3 and I.MAX_REJECTED_CALLS == 8
    env = make(NOWRAP)
    assert env.limits()["max_invalid_deliverables"] == 3 and env.limits()["max_rejected_calls"] == 8
    steps = []
    for k in range(3):                                                   # refusals of the instrument in between change nothing
        assert not call(env, "calc", expressions=RUNTIME[k]).done
        steps.append(call(env, "submit", **BAD_GEOMETRY))
        obs = steps[-1].observation
        assert obs.data["error"]["type"] == "invalid_deliverable" and obs.data["charged"] == dict(scan_time_s=0.0, tool_calls=0, rejected_calls=1)
        assert (env.state()["tool_calls"], env.state()["rejected_calls"]) == (k + 1, k + 1)
    assert [s.done for s in steps] == [False, False, True] and steps[-1].truncated and steps[-1].reward == 0.0
    assert steps[-1].info["reason"] == "rejected_call_limit" and "episode truncated: 3 submissions refused" in steps[-1].observation.text
    assert steps[-1].info["interface"] == dict(calls=6, executed=3, rejected=3, shape=0, vocabulary=0, task_refusals=6, coerced=0, internal_errors=0)
    _check_counters(env, answered=0, refused_deliverables=3)


# ------------------------------------------------------------------------------------------------ (f)
@pytest.mark.parametrize("env_id", [NULL, NOWRAP])
def test_the_task_text_states_the_rule_in_the_words_of_the_interface(env_id):
    env = mrgym.make(env_id, level=1, seed=0)
    obs, info = env.reset()
    limits = next(line for line in obs.text.splitlines() if line.startswith("Limits. "))
    assert f"Tool calls: {env.max_steps} ({I.TOOL_CALLS_TOLD})." in limits
    assert f"Rejected calls: 8 ({I.REJECTED_CALLS_TOLD}), of them at most 3 refused `submit` calls." in limits
    assert "reaches no tool" not in obs.text and "executed call" not in obs.text       # the two sentences the old accounting was told in
    assert I.TOOL_CALLS_TOLD == "every call counts once unless rejected" and "refused for how it is written" in I.REJECTED_CALLS_TOLD
    for words in ("refused for how it is written", "refused for what it asks", "whatever its type"):      # the module says the same rule
        assert words in " ".join(I.__doc__.split()), words
    assert "refused for how they are written" in " ".join(mrgym.env.__doc__.split()) and "refused for what it asks" in " ".join(mrgym.env.__doc__.split())
    overview = call(env, "scanner_info", topic="overview").observation.text
    assert "every executed call" not in overview
