"""The environment surface: registry, make / reset / tools / step, seeds, wire formats, adapters, cards, the gate."""
import json
import re

import pytest

import mrgym
from mrgym import admission, cards, check, policies, render
from mrgym import tools as T
from mrgym.adapters import chat, mcp
from mrgym.env import derive_seed

NOWRAP, T1, EXPLORE = "Scanner/NoWrap-v0", "Scanner/T1Transmit-v0", "Scanner/Explore-v0"
LOCALISER = {"sequence": "spgr", "tr_ms": 10.0, "flip_deg": 10.0, "fov_mm": 288.0, "matrix": 96}


# ------------------------------------------------------------------ registry
def test_registry_lists_the_three_ids_with_their_kind():
    assert mrgym.list_envs() == [EXPLORE, NOWRAP, T1] == mrgym.list_envs("Scanner")
    assert [mrgym.spec(i).kind for i in mrgym.list_envs()] == ["sandbox", "admitted", "practice"]
    assert mrgym.spec(T1).levels == (1, 2, 3) and mrgym.spec(NOWRAP).version == 0


@pytest.mark.parametrize("bad, text", [("Scanner/NoWrap-v7", "registered versions of Scanner/NoWrap: v0"), ("nowrap", "malformed"),
                                       ("Scanner/Nope-v0", "is not registered")])
def test_unknown_ids_name_the_fix(bad, text):
    with pytest.raises(mrgym.UnknownEnv, match=re.escape(text)):
        mrgym.make(bad)


def test_make_rejects_unknown_levels_and_modes():
    with pytest.raises(mrgym.Error):
        mrgym.make(NOWRAP, level=9)
    with pytest.raises(mrgym.Error):
        mrgym.make(NOWRAP, obs="audio")
    with pytest.raises(mrgym.Error):
        mrgym.make(EXPLORE, phantom="knee")
    with pytest.raises(mrgym.Error):
        mrgym.register("scanner/bad", "x:y", "x:z")


# ------------------------------------------------------------------ seeds
def test_integer_seeds_map_to_secret_world_seeds():
    a = derive_seed(NOWRAP, 1, "train", 0, environ={})
    assert a == derive_seed(NOWRAP, 1, "train", "0", environ={}) and re.fullmatch(r"[0-9a-f]{32}", a)
    assert len({a, derive_seed(NOWRAP, 2, "train", 0, environ={}), derive_seed(NOWRAP, 1, "train", 1, environ={}),
                derive_seed(T1, 1, "train", 0, environ={}), derive_seed(NOWRAP, 1, "train", 0, environ={"MRGYM_SECRET": "s"})}) == 5
    assert derive_seed(NOWRAP, 1, "train", "ab" * 16) == "ab" * 16          # an operator's own 128-bit seed is used as is
    assert derive_seed(NOWRAP, 1, "train", None) != derive_seed(NOWRAP, 1, "train", None)
    with pytest.raises(mrgym.Error, match="MRGYM_SECRET"):
        derive_seed(NOWRAP, 1, "heldout", 0, environ={})
    assert derive_seed(NOWRAP, 1, "heldout", 0, environ={"MRGYM_SECRET": "s"}) != a


# ------------------------------------------------------------------ episode contract
def test_reset_step_contract_and_three_call_shapes():
    env = mrgym.make(NOWRAP, level=1, seed=3)
    with pytest.raises(mrgym.ResetNeeded):
        env.step({"name": "scanner_info", "arguments": {}})
    obs, info = env.reset()
    assert "# Prescribe" in obs.text and info["tools"] == ["scanner_info", "plan", "scan", "read_regions", "view", "submit"]
    assert info["env_id"] == NOWRAP and obs.status["scan_time_left_s"] == 30.0
    plan = {"protocol": LOCALISER}
    shapes = [{"name": "plan", "arguments": plan},
              {"id": "call_1", "type": "function", "function": {"name": "plan", "arguments": json.dumps(plan)}},
              {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "plan", "arguments": plan}},
              {"type": "tool_use", "id": "tu_1", "name": "plan", "input": plan}]
    texts = set()
    for call in shapes:
        obs, reward, terminated, truncated, info = env.step(call)
        assert not obs.is_error and reward == 0.0 and not terminated and not truncated and info == {}
        texts.add(obs.text.split("\n[")[0])
    assert len(texts) == 1                                               # the four shapes are one call
    step = env.step({"name": "scan", "arguments": plan})
    assert step.observation.data["cost_s"] > 0 and env.state()["scans"] == 1 and len(step.observation.text) < 1500
    end = env.step({"name": "submit", "arguments": {"pe_dir": "y", "fov_mm": 260.0, "phase_fov": 1.0, "matrix": 130}})
    assert end.terminated and not end.truncated and 0.0 <= end.reward <= 1.0 and end.info["replay"]["ok"]
    assert "reward" not in end.observation.text                          # the score is for the trainer unless reveal_score
    with pytest.raises(mrgym.ResetNeeded):
        env.step({"name": "scanner_info", "arguments": {}})
    assert env.state()["done"] and env.state()["reward"] == end.reward
    env.reset()
    assert env.state()["scans"] == 0 and not env.state()["done"]         # reset starts a new episode on a new session


BAD_CALLS = [
    ("nope", {}, "invalid_tool_call"), ("scan", "{not json", "invalid_tool_call"), ("scan", {"protocol": LOCALISER, "seed": 1}, "invalid_tool_call"),
    ("scan", {}, "invalid_tool_call"), ("scan", {"protocol": dict(LOCALISER, tr_ms=3.0)}, "invalid_request"),
    ("scan", {"protocol": dict(LOCALISER, seed=4)}, "invalid_request"), ("scan", {"protocol": dict(LOCALISER, averages=64)}, "budget_exhausted"),
    ("read_regions", {"scan": 5, "regions": [{"y_mm": 0, "x_mm": 0, "radius_mm": 5}]}, "invalid_request"),
    ("scan", {"protocol": LOCALISER, "regions": [{"y_mm": 0, "x_mm": 0, "radius_mm": 5000}]}, "invalid_tool_call"),
    ("view", {"scan": 0, "kind": "truth"}, "invalid_tool_call"),
]


@pytest.mark.parametrize("name, arguments, kind", BAD_CALLS)
def test_bad_calls_are_observations_and_cost_nothing(name, arguments, kind):
    env = mrgym.make(NOWRAP, level=1, seed=3)
    env.reset()
    step = env.step({"name": name, "arguments": arguments})
    assert step.observation.is_error and not step.done and step.reward == 0.0
    assert step.observation.data["error"]["type"] == kind and "nothing was charged" in step.observation.text
    assert env.state()["scan_time_used_s"] == 0.0


def test_rejected_plan_is_an_error_observation_with_typed_violations():
    env = mrgym.make(NOWRAP, level=1, seed=3)
    env.reset()
    obs = env.step({"name": "plan", "arguments": {"protocol": dict(LOCALISER, tr_ms=3.0)}}).observation
    assert obs.is_error and obs.data["violations"][0]["code"] == "TR_BELOW_MINIMUM" and "TR_BELOW_MINIMUM" in obs.text


def test_max_steps_truncates_with_reward_zero():
    env = mrgym.make(NOWRAP, level=1, seed=3, max_steps=2)
    env.reset()
    env.step({"name": "scanner_info", "arguments": {}})
    step = env.step({"name": "scanner_info", "arguments": {"topic": "physics"}})
    assert step.truncated and not step.terminated and step.reward == 0.0 and "Signal mechanism" in step.observation.text


# ------------------------------------------------------------------ observations
def test_text_and_vision_carry_the_same_numbers_and_images_are_requested():
    out = {}
    for mode in ("text", "vision"):
        env = mrgym.make(T1, level=2, seed=1, obs=mode)
        env.reset()
        scan = env.step({"name": "scan", "arguments": {"protocol": {"sequence": "spgr", "tr_ms": 12.0, "flip_deg": 8.0, "matrix": 64}}}).observation
        view = env.step({"name": "view", "arguments": {"scan": 0}}).observation
        out[mode] = (scan, view)
    assert out["text"][0].text == out["vision"][0].text and out["text"][0].data == out["vision"][0].data
    assert not out["text"][0].images and not out["vision"][0].images    # a scan never pushes an image
    assert len(out["text"][0].data["roi_readouts"]) >= 4                 # the task's regions are read out by default
    png = out["vision"][1].images[0]["png"]
    assert render.png_size(png)[0] >= 384 and out["vision"][1].to_openai("c")[1]["content"][0]["type"] == "image_url"
    assert out["vision"][1].to_anthropic("u")["content"][1]["type"] == "image" and out["vision"][1].to_mcp()["content"][1]["mimeType"] == "image/png"
    grid = out["text"][1].data["digit_map"]
    assert len(grid) == 32 and set("".join(grid)) <= set("0123456789") and "extent of pixels" in out["text"][1].text


def test_tools_in_three_wire_formats_and_schema_generated_from_the_console():
    from mrgym.scanner import console
    env = mrgym.make(T1, level=1, seed=0)
    env.reset()
    oa, mc, an = env.tools("openai"), env.tools("mcp"), env.tools("anthropic")
    assert [t["function"]["name"] for t in oa] == [t["name"] for t in mc] == [t["name"] for t in an]
    proto = oa[2]["function"]["parameters"]["properties"]["protocol"]
    assert proto == console.protocol_schema() and proto["properties"]["sequence"]["enum"] == sorted(console.SEQUENCES)
    assert not [p for t in env._tool_specs() for p in T.check_schema(t.parameters, t.name)]
    submit = oa[-1]["function"]["parameters"]["properties"]["t1_ms"]
    assert submit["required"] == [r["id"] for r in env._regions]        # tools are a function of the episode
    assert mc[1]["annotations"]["readOnlyHint"] and not mc[2]["annotations"]["readOnlyHint"]
    with pytest.raises(ValueError):
        T.ToolSpec("reset", "x", {"type": "object"})


# ------------------------------------------------------------------ policies through the tools
@pytest.mark.parametrize("env_id, level", [(NOWRAP, 1), (NOWRAP, 2), (T1, 2)])
def test_control_policies_through_the_tools_match_the_scanner_interface(env_id, level):
    """The reference policy played through ``env.step`` leaves the same scanner log and answer as the same policy
    on the scanner interface directly: the tool surface adds nothing and loses nothing."""
    import importlib
    spec = mrgym.spec(env_id)
    controls = importlib.import_module(spec.family + ".controls")
    env = mrgym.make(env_id, level=level, seed=5, tolerance=0.05) if env_id == T1 else mrgym.make(env_id, level=level, seed=5)
    r = policies.run(env, "reference")
    assert r.info["policy_error"] is None and r.terminated and r.info["replay"]["ok"]
    answer, sess, err = controls.run_policy(env.episode_config(), controls.reference, 0, env._session.engine)
    assert err is None and sess.used == pytest.approx(r.info["budget"]["used_s"]) and sess.n_scans == r.info["budget"]["scans"]
    assert [e["data_sha256"] for e in sess.events if e["type"] == "scan"] == [e["data_sha256"] for e in env.scanner_log() if e["type"] == "scan"]
    assert r.reward >= 0.9
    floor = policies.run(mrgym.make(env_id, level=level, seed=5), "prior")
    assert floor.reward == 0.0 and not floor.info["score"]["gate_measured"]
    unit = policies.run(mrgym.make(env_id, level=level, seed=5), "wrong_units")
    assert unit.reward == 0.0 and not unit.info["score"]["gate_deliverable"]


def test_a_failing_policy_is_reported_not_hidden():
    def broken(sc):
        raise RuntimeError("policy bug")
    r = policies.run(mrgym.make(NOWRAP, level=1, seed=0), broken)
    assert r.info["policy_error"] == "RuntimeError: policy bug" and r.reward == 0.0


def test_same_seed_same_episode_other_seed_other_world():
    runs = []
    for seed in (4, 4, 5):
        env = mrgym.make(NOWRAP, level=2, seed=seed)
        r = policies.run(env, "reference")
        runs.append((env._session.chain, [e["text"] for e in env.transcript()], r.reward))
    assert runs[0] == runs[1] and runs[0][0] != runs[2][0]


# ------------------------------------------------------------------ sandbox
def test_explore_is_a_scanner_without_a_reward():
    env = mrgym.make(EXPLORE, phantom="brain", field_t=1.5, lesion="ms", seed=2)
    obs, info = env.reset()
    assert "Nothing is asked" in obs.text and info["tools"][-1] == "finish"
    s = env.step({"name": "scan", "arguments": {"protocol": {"sequence": "se", "tr_ms": 500.0, "matrix": 64}}})
    assert "SE 1.5T" in s.observation.text and not s.observation.is_error
    end = env.step({"name": "finish", "arguments": {}})
    assert end.terminated and end.reward == 0.0 and end.info["score"]["passed"] is None
    assert mrgym.spec(EXPLORE).reward_range == (0.0, 0.0)


# ------------------------------------------------------------------ adapters in the core (standard library only)
def test_mcp_handler():
    env = mrgym.make(NOWRAP, level=1, seed=0)
    env.reset()
    assert mcp.handle(env, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})["result"]["serverInfo"]["title"] == NOWRAP
    names = [t["name"] for t in mcp.handle(env, {"id": 2, "method": "tools/list"})["result"]["tools"]]
    assert "submit" in names and not set(names) & T.RESERVED
    ok = mcp.handle(env, {"id": 3, "method": "tools/call", "params": {"name": "plan", "arguments": {"protocol": LOCALISER}}})["result"]
    assert not ok["isError"] and ok["structuredContent"]["accept"]
    bad = mcp.handle(env, {"id": 4, "method": "tools/call", "params": {"name": "plan", "arguments": {"protocol": {"sequence": "x"}}}})["result"]
    assert bad["isError"]
    assert mcp.handle(env, {"id": 5, "method": "resources/list"})["error"]["code"] == -32601


def test_chat_rollout_with_a_scripted_stand_in():
    calls = iter([("scan", {"protocol": LOCALISER}), ("submit", {"pe_dir": "y", "fov_mm": 260.0, "phase_fov": 1.0, "matrix": 130})])

    def policy_fn(messages, tools):
        name, args = next(calls)
        return {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{len(messages)}", "type": "function",
                                                                      "function": {"name": name, "arguments": json.dumps(args)}}]}
    messages, step, out = chat.rollout(mrgym.make(NOWRAP, level=1, seed=0), policy_fn, system="You operate a scanner.")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert out["reason"] == "submitted" and step.terminated and out["reward"] == step.reward


# ------------------------------------------------------------------ cards and the gate
@pytest.mark.parametrize("env_id", [EXPLORE, NOWRAP, T1])
def test_card_is_complete_and_renders(env_id):
    card = mrgym.card(env_id)
    assert not cards.lint(card), cards.lint(card)
    md = cards.markdown(card)
    assert md.startswith("<!-- Generated") and f"`{env_id}`" in md and card["status"] in md


def test_an_id_listed_as_admitted_has_admitted_instances_on_this_engine():
    """The gate for publishing an id. ``kind`` in the registry is a claim; the stored control-policy run is the
    evidence. An environment the gate rejects must be listed as practice."""
    for env_id in mrgym.list_envs():
        spec, measured = mrgym.spec(env_id), admission.load(env_id)
        if spec.kind == "sandbox":
            assert measured is None
            continue
        assert measured is not None, f"{env_id}: no stored control-policy results"
        assert measured["engine"] == mrgym.ENGINE_ID, f"{env_id}: results were measured on {measured['engine']}"
        k, n = admission.counts(measured)
        assert n >= 24 * len(spec.levels)
        if spec.kind == "admitted":
            assert k > 0 and all(lv["controls"]["reference"]["pass_rate"] >= 0.9 for lv in measured["levels"].values())
            assert all(lv["controls"]["prior"]["pass_rate"] == 0 and lv["controls"]["wrong_units"]["pass_rate"] == 0
                       for lv in measured["levels"].values())
        else:
            assert k < 0.5 * n, f"{env_id} is listed as practice but the gate admits {k} of {n}: list it as admitted"


@pytest.mark.parametrize("env_id", [EXPLORE, NOWRAP, T1])
def test_check_passes_for_every_registered_id(env_id):
    res = check.check_env(env_id)
    assert not [(name, detail) for name, ok, detail in res if not ok]
    assert len(res) >= 16
