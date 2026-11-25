"""The ``reset`` / ``step`` / ``submit`` contract as a Hypothesis rule-based state machine, for every registered id.

Rules send schema-valid calls (generated from the tool schemas of the episode), near-valid calls (one value replaced, a
key dropped or added, the arguments as JSON text), arbitrary JSON in the four call shapes, a cheap scan, and calls after
the end. After every step:

  I1   ``step`` does not raise (``ResetNeeded`` after the end, and only then)
  I2   the result is (Observation, float, bool, bool, dict); the reward is finite and in [0, 1]; for a family built on
       ``envs/family.py`` it is exactly 0.0 or 1.0
  I3   before the end the reward is 0.0 and ``info`` carries nothing about the outcome: it is empty, or holds the
       per-step record of the training hooks (``info["step"]``) and nothing else
  I4   the budget is monotone: scan time used never decreases and never exceeds the budget; counters never decrease;
       every step is either a tool call or a rejected call, never both and never neither
  I5   an error observation charges no scan time and stores no scan
  I6   ``terminated`` and ``truncated`` are never both true; afterwards ``step`` raises ``ResetNeeded`` and ``state()``
       does not move
  I7   a refused deliverable (an error observation from the terminal tool) does not terminate the episode; an episode
       that ends without a deliverable is truncated, with reward 0 and a stated reason
  I8   nothing returned before the end contains the world seed
  I9   replay equals live: at the terminal step the verifier's replay of the sealed log is anchored to its head and
       gives the live scan seconds and scan count; at teardown a replay of the log against the session's receipt does
  I10  same seed, same episode: at teardown the calls are replayed on a new environment of the same seed and every
       text, data payload, reward and flag is equal (the random episode handle is the one thing that differs)

Size: profile ``dev`` (plain ``pytest``), ``ci`` (``make audit``: HYPOTHESIS_PROFILE=ci), ``nightly``; see conftest.py.

Memory. An episode holds a hidden world, an engine with its cache of noise-free k-space, and the stored images of
every scan; this machine holds two of them (the episode and its twin) per example and Hypothesis runs hundreds of
examples in one process. Three things keep that bounded (measured: under 0.5 GB for the ``ci`` profile):
  * every generated call passes through ``bounded``: a ``matrix`` above 64, more than 2 echoes or averages are cut
    down before the call is sent. A valid random protocol is accepted far more often since the arguments are flat and
    the canonicaliser is tolerant; a 512-matrix, 8-echo scan is 100 MB of k-space and images, per scan, per engine
  * the machine keeps digests of the results, not the observations
  * teardown closes both environments, drops every reference and collects garbage, also when an invariant failed

``ANSWERED`` lists inputs on which ``step`` once raised (an MCP-shaped call whose ``params`` is not an object; nesting
2,000 levels deep). They are answered with an error observation since ``step`` was made total and stay as regression
cases.
"""
import copy
import gc
import hashlib
import json
import math

import pytest

from conftest import scan_call

hypothesis = pytest.importorskip("hypothesis")
hj = pytest.importorskip("hypothesis_jsonschema")
from hypothesis import strategies as st  # noqa: E402
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule  # noqa: E402

import mrgym  # noqa: E402
from mrgym.scanner import session  # noqa: E402

IDS = tuple(mrgym.list_envs())
FAMILY_IDS = tuple(i for i in IDS if mrgym.spec(i).entry_point == "mrgym.envs.family:FamilyEnv")
CHEAP = {"sequence": "epi", "matrix": 32, "tr_ms": 50}
LIMITS = dict(matrix=64, n_echoes=2, averages=2)                # what ``bounded`` allows a generated call to ask for
END_REASONS = ("max_steps", "rejected_call_limit")


def _nested(depth):
    out = {}
    for _ in range(depth):
        out = {"a": out}
    return out


# ---- inputs on which step() once raised: (label, a function that builds the call). Built per test: a structure 2,000
# levels deep cannot be copied by ``copy.deepcopy`` (it recurses), and nothing here needs a copy.
ANSWERED = [
    ("mcp params is a list", lambda: {"method": "tools/call", "params": [1, 2], "id": 1}),
    ("mcp params is a string", lambda: {"method": "tools/call", "params": "x"}),
    ("mcp params is a number", lambda: {"method": "tools/call", "params": 1}),
    ("protocol nested 2000 deep", lambda: {"name": "plan", "arguments": {"protocol": _nested(2000)}}),
    ("arguments nested 2000 deep", lambda: {"name": "plan", "arguments": _nested(2000)}),
    ("circles nested 2000 deep", lambda: {"name": "scan", "arguments": dict(CHEAP, circles=_nested(2000))}),
    ("arguments as JSON text nested 5000 deep", lambda: {"name": "plan", "arguments": '{"sequence": ' + "[" * 5000 + "]" * 5000 + "}"}),
]


def _answered_cases():
    """Every input on the first family id; the first input of each kind (MCP shape, nesting) on every other id."""
    home = (FAMILY_IDS or IDS)[0]
    first = {"mcp": ANSWERED[0][0], "nested": ANSWERED[3][0]}
    return [pytest.param(env_id, label, build, id=f"{label} [{env_id}]") for env_id in IDS for label, build in ANSWERED
            if env_id == home or label in first.values()]


@pytest.mark.parametrize("env_id,label,build", _answered_cases())
def test_step_answers_an_input_that_once_raised(env_id, label, build):
    env = mrgym.make(env_id, seed=0)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(build())
    assert obs.is_error and reward == 0.0 and not terminated and not truncated
    assert obs.data["error"]["type"] == "malformed_call" and obs.data["episode_continues"] is True
    assert obs.status["rejected_calls"] == 1 and obs.status["tool_calls"] == 0 and obs.status["scan_time_used_s"] == 0.0
    assert set(info) <= {"step"}
    env.close()


# ---------------------------------------------------------------------------------------------- strategies
json_leaf = st.one_of(st.none(), st.booleans(), st.integers(-10 ** 20, 10 ** 20), st.floats(allow_nan=True, allow_infinity=True),
                      st.text(max_size=12))
json_any = st.recursive(json_leaf, lambda c: st.one_of(st.lists(c, max_size=4), st.dictionaries(st.text(max_size=8), c, max_size=4)),
                        max_leaves=12)


def bounded(call, depth=0):
    """The same call with the size of any scan it asks for cut down (see "Memory" in the module docstring). Works on
    every shape a rule sends: a dict at any nesting, or arguments given as JSON text."""
    if depth > 8:
        return call
    if isinstance(call, dict):
        return {k: (min(v, LIMITS[k]) if k in LIMITS and type(v) in (int, float) and math.isfinite(v) else bounded(v, depth + 1))
                for k, v in call.items()}
    if isinstance(call, list):
        return [bounded(v, depth + 1) for v in call]
    if isinstance(call, str) and any(k in call for k in LIMITS) and len(call) < 4000:
        try:
            parsed = json.loads(call)
        except ValueError:
            return call
        return json.dumps(bounded(parsed, depth + 1)) if isinstance(parsed, (dict, list)) else call
    return call


def _schemas(env):
    return {t["name"]: t["parameters"] for t in env.tools("neutral")}


@st.composite
def valid_call(draw, schemas):
    name = draw(st.sampled_from(sorted(schemas)))
    return {"name": name, "arguments": draw(hj.from_schema(schemas[name]))}


@st.composite
def near_valid_call(draw, schemas):
    """A schema-valid call with one value replaced by arbitrary JSON, one key dropped or added, or the arguments as text."""
    call = draw(valid_call(schemas))
    args = call["arguments"]
    how = draw(st.sampled_from(("replace", "drop", "add", "stringify")))
    if how == "stringify":
        return {"name": call["name"], "arguments": json.dumps(args)}
    if how == "add" or not args:
        args[draw(st.text(max_size=8))] = draw(json_any)
    else:
        key = draw(st.sampled_from(sorted(args)))
        if how == "drop":
            del args[key]
        else:
            args[key] = draw(json_any)
    return call


hostile_call = st.one_of(
    json_any,
    st.fixed_dictionaries({"name": st.one_of(st.text(max_size=12), json_any), "arguments": json_any}),
    st.fixed_dictionaries({"method": st.just("tools/call"), "params": json_any}),
    st.fixed_dictionaries({"type": st.just("function"), "function": json_any}),
    st.fixed_dictionaries({"type": st.just("tool_use"), "name": st.text(max_size=12), "input": json_any}),
)


def _json(x):
    return json.dumps(x, sort_keys=True, default=str)


def _digest(result):
    """What the twin must reproduce of a step: text, data, reward and flags (the observation itself is not kept)."""
    obs, reward, terminated, truncated, _ = result
    return (hashlib.sha256(obs.text.encode()).hexdigest(), hashlib.sha256(_json(obs.data).encode()).hexdigest(), reward, terminated, truncated)


def _public(info):
    """The reset info without what is random per episode by design (the episode handle and the group id built on it)."""
    return {k: v for k, v in info.items() if k not in ("episode_id", "group")}


# ---------------------------------------------------------------------------------------------- the machine
class Episode(RuleBasedStateMachine):
    env = twin = None

    @initialize(env_id=st.sampled_from(IDS), seed=st.integers(0, 3))
    def start(self, env_id, seed):
        self.env_id, self.seed = env_id, seed
        self.env = mrgym.make(env_id, seed=seed)
        obs0, self.info0 = self.env.reset()
        self.text0 = obs0.text
        self.schemas = _schemas(self.env)
        self.terminal = next(t.name for t in self.env._tool_specs() if t.terminal)
        self.calls, self.digests, self.over = [], [], False
        self.prev = dict(obs0.status)
        self.secrets = [str(self.env.episode_config()["seed"])]
        assert not any(x in obs0.text + _json(obs0.data) for x in self.secrets)

    def _step(self, call):
        call = bounded(call)
        sent = copy.deepcopy(call)
        res = self.env.step(call)                                           # I1: any exception fails the test
        self.calls.append(sent)
        self.digests.append(_digest(res))
        obs, reward, terminated, truncated, info = res
        assert isinstance(obs, mrgym.Observation) and type(reward) is float and math.isfinite(reward) and 0.0 <= reward <= 1.0   # I2
        assert type(terminated) is bool and type(truncated) is bool and isinstance(info, dict)
        assert not (terminated and truncated)                               # I6
        if self.env_id in FAMILY_IDS:
            assert reward in (0.0, 1.0)                                     # I2: no partial credit
        if not (terminated or truncated):
            assert reward == 0.0 and set(info) <= {"step"}                  # I3
        s, p = obs.status, self.prev
        assert s["scan_time_used_s"] >= p["scan_time_used_s"]              # I4
        assert s["scan_time_used_s"] <= s["scan_time_budget_s"] + 1e-6
        assert s["scans"] >= p["scans"] and s["scanner_requests"] >= p["scanner_requests"]
        assert abs(s["scan_time_left_s"] - (s["scan_time_budget_s"] - s["scan_time_used_s"])) < 1e-6
        assert s["tool_calls"] + s["rejected_calls"] == len(self.calls)
        assert (s["tool_calls"] - p["tool_calls"], s["rejected_calls"] - p["rejected_calls"]) in ((1, 0), (0, 1))
        assert s["tool_calls"] <= s["max_tool_calls"] and s["rejected_calls"] <= s["max_rejected_calls"]
        if s["rejected_calls"] > p["rejected_calls"]:
            assert obs.is_error                                             # only a refused call is a rejected call
        if obs.is_error:                                                    # I5
            assert s["scan_time_used_s"] == p["scan_time_used_s"] and s["scans"] == p["scans"]
            assert obs.data["charged"]["scan_time_s"] == 0.0 and obs.data["episode_continues"] is (not truncated)
        if obs.is_error and obs.tool == self.terminal:
            assert not terminated                                           # I7: a refused deliverable does not end the episode
        if terminated:
            assert obs.tool == self.terminal and not obs.is_error and info.get("submitted") is True
            rep = info["replay"]                                            # I9: the verifier's replay is the live episode
            assert rep["used_s"] == s["scan_time_used_s"] and rep["scans"] == s["scans"] and info["budget"]["used_s"] == s["scan_time_used_s"]
            assert rep["ok"] and rep["integrity"] == "anchored" and rep["sealed"] and not rep["over_budget"]
            if reward > 0:
                assert rep["scans"] > 0 and info["score"]["passed"] is True
        if truncated:                                                       # I7
            assert reward == 0.0 and info["submitted"] is False and info["reason"] in END_REASONS
        if not (terminated or truncated):                                   # I8
            blob = obs.text + _json(obs.data) + _json(self.env.state())
            assert not any(x in blob for x in self.secrets)
        self.prev, self.over = dict(s), terminated or truncated

    @precondition(lambda self: not self.over)
    @rule(data=st.data())
    def valid(self, data):
        self._step(data.draw(valid_call(self.schemas)))

    @precondition(lambda self: not self.over)
    @rule(data=st.data())
    def near_valid(self, data):
        self._step(data.draw(near_valid_call(self.schemas)))

    @precondition(lambda self: not self.over)
    @rule(call=hostile_call)
    def hostile(self, call):
        self._step(call)

    @precondition(lambda self: not self.over)
    @rule()
    def cheap_scan(self):
        self._step(scan_call(self.env, CHEAP))

    @precondition(lambda self: self.over)
    @rule(call=hostile_call)
    def after_the_end(self, call):                                          # I6
        frozen = self.env.state()
        with pytest.raises(mrgym.ResetNeeded):
            self.env.step(call)
        assert self.env.state() == frozen

    @invariant()
    def state_is_public(self):
        if self.env is not None:
            now = self.env.state()
            assert now["env_id"] == self.env_id and (now["reward"] is None) == (not self.over) and now["done"] is self.over

    def _check_replay_and_twin(self):
        live = self.env.state()
        if not self.over:                                                   # I9 (a finished episode was checked at its last step)
            sess = self.env._session                                        # trainer side: the receipt is the session's
            rep = session.replay(self.env.episode_config(), self.env.scanner_log(), head=sess.receipt(with_key=True))
            assert rep["ok"] and rep["integrity"] == "anchored" and not rep["over_budget"] and rep["mode"] == "bit-exact"
            assert rep["used_s"] == live["scan_time_used_s"] and rep["scans"] == live["scans"]
        self.twin = mrgym.make(self.env_id, seed=self.seed)                 # I10
        obs, info = self.twin.reset()
        assert obs.text == self.text0 and _public(info) == _public(self.info0)
        for call, want in zip(self.calls, self.digests):
            assert _digest(self.twin.step(copy.deepcopy(call))) == want
        now = self.twin.state()
        assert {k: v for k, v in now.items() if k != "episode_id"} == {k: v for k, v in live.items() if k != "episode_id"}

    def teardown(self):
        try:
            if self.env is not None:
                self._check_replay_and_twin()
        finally:                                                            # free both episodes, also after a failed invariant
            for e in (self.env, self.twin):
                if e is not None:
                    e.close()
            self.env = self.twin = None
            self.calls, self.digests = [], []
            gc.collect()


TestEpisodeContract = Episode.TestCase


# ---------------------------------------------------------------------------------------------- what the machine relies on
@pytest.mark.parametrize("env_id", IDS)
def test_the_cheap_scan_of_the_machine_is_accepted(env_id):
    """If the scan tool changes shape and this call is refused, the machine would silently stop covering scans."""
    env = mrgym.make(env_id, seed=0)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(scan_call(env, CHEAP))
    assert not obs.is_error and obs.status["scans"] == 1 and obs.status["scan_time_used_s"] == pytest.approx(3.15)
    assert (reward, terminated, truncated) == (0.0, False, False) and set(info) <= {"step"}
    env.close()


def test_bounded_cuts_the_size_of_a_scan_and_nothing_else():
    big = {"name": "scan", "arguments": {"sequence": "spgr", "matrix": 512, "n_echoes": 8, "averages": 64, "tr_ms": 5000, "flip_deg": 90.0}}
    assert bounded(big) == {"name": "scan", "arguments": {"sequence": "spgr", "matrix": 64, "n_echoes": 2, "averages": 2, "tr_ms": 5000, "flip_deg": 90.0}}
    assert bounded({"method": "tools/call", "params": {"name": "scan", "arguments": {"matrix": 300.0}}})["params"]["arguments"]["matrix"] == 64
    assert json.loads(bounded({"name": "scan", "arguments": json.dumps({"matrix": 256, "x": [1]})})["arguments"]) == {"matrix": 64, "x": [1]}
    for same in (None, 3, "matrix", [1, {"matrix": "512"}], {"matrix": True}, {"matrix": 32}, {"matrix": float("inf")},
                 {"name": 1, "arguments": "{bad json matrix"}):
        assert bounded(same) == same                                     # not a finite size: left for the interface to refuse
    assert math.isnan(bounded({"matrix": float("nan")})["matrix"])
