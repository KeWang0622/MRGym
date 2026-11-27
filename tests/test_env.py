"""The environment surface: registry, make / reset / tools / step, seeds, wire formats, adapters, cards, and the claim
of an id (its ``kind``) against the stored gate result."""
import json
import re

import pytest

import mrgym
from mrgym import cards, check, gate, policies, render
from mrgym import interface as I
from mrgym import tools as T
from mrgym.adapters import chat, mcp
from mrgym.env import derive_seed

NOWRAP, T1, EXPLORE = "Scanner/NoWrap-v1", "Scanner/T1Transmit-v1", "Scanner/Explore-v0"
RETIRED = {"Scanner/NoWrap-v0": NOWRAP, "Scanner/T1Transmit-v0": T1}
SAFE_GEOMETRY = {"pe_dir": "y", "fov_mm": 260.0, "phase_fov": 1.0, "matrix": 130, "centre_y_mm": 0.0, "centre_x_mm": 0.0, "rotation_deg": 0.0}
LOCALISER = {"sequence": "spgr", "tr_ms": 11.0, "flip_deg": 10.0, "fov_mm": 288.0, "matrix": 96}


# ------------------------------------------------------------------ registry
def test_registry_lists_the_ids_with_their_kind():
    assert {EXPLORE, NOWRAP, T1} <= set(mrgym.list_envs()) and mrgym.list_envs() == mrgym.list_envs("Scanner")
    assert [mrgym.spec(i).kind for i in (EXPLORE, NOWRAP, T1)] == ["sandbox", "candidate", "practice"]      # NoWrap: no gate on engine 0.3.0 yet
    assert all(mrgym.spec(i).kind in cards.KINDS for i in mrgym.list_envs())
    assert mrgym.spec(T1).levels == (1, 2, 3) and mrgym.spec(NOWRAP).levels == (0, 1, 2, 3) and mrgym.spec(NOWRAP).version == 1
    assert gate.claim_levels(mrgym.spec(NOWRAP)) == [2, 3] and gate.claim_levels(mrgym.spec(T1)) == [1, 2, 3]
    assert all(mrgym.spec(i).entry_point == "mrgym.envs.family:FamilyEnv" for i in mrgym.list_envs() if mrgym.spec(i).family)


def test_the_two_v0_ids_are_retired_and_name_their_successor():
    """One break to ``-v1``: the old ids are not served, are never reused, and say which package version reproduces them."""
    from mrgym.registry import retired
    assert {i: s for i, (_, s) in retired.items()} == RETIRED and all(v == "0.1.0" for v, _ in retired.values())
    for old, new in RETIRED.items():
        assert old not in mrgym.list_envs() and new in mrgym.list_envs()
        for call in (mrgym.make, mrgym.spec, mrgym.card):
            with pytest.raises(mrgym.RetiredEnv, match=re.escape(f"{old} is retired: mrgym 0.1.0 is the last package version")) as e:
                call(old)
            assert "pip install mrgym==0.1.0" in str(e.value) and f"Successor: {new}." in str(e.value)
        with pytest.raises(mrgym.Error, match="never reused"):
            mrgym.register(old, "mrgym.envs.family:FamilyEnv", "mrgym.envs.nowrap.task:card")


def test_the_per_instance_admission_machinery_is_gone():
    import importlib.util

    import mrgym.envs.nowrap.controls as nw
    import mrgym.envs.t1_transmit.controls as t1
    assert importlib.util.find_spec("mrgym.admission") is None
    assert all(importlib.util.find_spec(f"mrgym.envs.{f}.env") is None for f in ("nowrap", "t1_transmit"))
    for controls in (nw, t1):
        assert not hasattr(controls, "admit") and not hasattr(controls, "run_policy") and not hasattr(controls, "ADMISSION_CONTROLS")
        assert all(callable(fn) for fn in controls.policies().values())          # no (function, budget scale) entries
    info = mrgym.make(NOWRAP, level=2, seed=0).reset()[1]
    assert "admitted" not in info and info["task_id"].startswith("nowrap-")
    assert "admitted" not in mrgym.make(EXPLORE).reset()[1]


@pytest.mark.parametrize("bad, text", [("Scanner/NoWrap-v7", "registered versions of Scanner/NoWrap: v1"), ("nowrap", "malformed"),
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
    secret = {"MRGYM_SECRET": "5a" * 32}
    a = derive_seed(NOWRAP, 1, "train", 0, environ={})
    assert a == derive_seed(NOWRAP, 1, "train", "0", environ={}) and re.fullmatch(r"[0-9a-f]{32}", a)
    assert len({a, derive_seed(NOWRAP, 2, "train", 0, environ={}), derive_seed(NOWRAP, 1, "train", 1, environ={}),
                derive_seed(T1, 1, "train", 0, environ={}), derive_seed(NOWRAP, 1, "gate", 0, environ={})}) == 5
    assert derive_seed(NOWRAP, 1, "train", 0, environ=secret) == a           # a secret never changes a public split
    assert derive_seed(NOWRAP, 1, "train", 0, environ={"MRGYM_SECRET": "x"}) == a      # not even a malformed one
    assert derive_seed(NOWRAP, 1, "train", "ab" * 16) == "ab" * 16          # an operator's own 128-bit seed is used as is
    assert derive_seed(NOWRAP, 1, "train", None) != derive_seed(NOWRAP, 1, "train", None)
    with pytest.raises(mrgym.Error, match="MRGYM_SECRET"):
        derive_seed(NOWRAP, 1, "heldout", 0, environ={})
    with pytest.raises(mrgym.Error, match="hexadecimal"):                    # a guessable secret is refused, not used
        derive_seed(NOWRAP, 1, "heldout", 0, environ={"MRGYM_SECRET": "s"})
    assert derive_seed(NOWRAP, 1, "heldout", 0, environ=secret) != a


# ------------------------------------------------------------------ episode contract
def test_reset_step_contract_and_three_call_shapes():
    env = mrgym.make(NOWRAP, level=1, seed=3)
    with pytest.raises(mrgym.ResetNeeded):
        env.step({"name": "scanner_info", "arguments": {}})
    obs, info = env.reset()
    assert "# Prescribe" in obs.text and info["tools"] == ["scanner_info", "status", "plan", "scan", "read_regions", "view", "profile", "calc", "submit"]
    assert info["env_id"] == NOWRAP and obs.status["scan_time_left_s"] == 30.0
    plan = {"protocol": LOCALISER}
    shapes = [{"name": "plan", "arguments": plan},
              {"id": "call_1", "type": "function", "function": {"name": "plan", "arguments": json.dumps(plan)}},
              {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "plan", "arguments": plan}},
              {"type": "tool_use", "id": "tu_1", "name": "plan", "input": plan}]
    texts = set()
    for call in shapes:
        obs, reward, terminated, truncated, info = env.step(call)
        assert not obs.is_error and reward == 0.0 and not terminated and not truncated and list(info) == ["step"]     # the trainer's per-step record
        texts.add(obs.text.split("\n[")[0])
    assert len(texts) == 1                                               # the four shapes are one call
    step = env.step({"name": "scan", "arguments": plan})
    assert step.observation.data["cost_s"] > 0 and env.state()["scans"] == 1 and len(step.observation.text) < 1500
    end = env.step({"name": "submit", "arguments": SAFE_GEOMETRY})
    assert end.terminated and not end.truncated and end.reward == 0.0 and end.info["replay"]["ok"]      # safe and too slow
    assert [k for k, ok in end.info["score"]["checks"].items() if not ok] == ["scan_time_within_factor"]
    assert "reward" not in end.observation.text                          # the score is for the trainer unless reveal_score
    with pytest.raises(mrgym.ResetNeeded):
        env.step({"name": "scanner_info", "arguments": {}})
    assert env.state()["done"] and env.state()["reward"] == end.reward
    env.reset()
    assert env.state()["scans"] == 0 and not env.state()["done"]         # reset starts a new episode on a new session


BAD_CALLS = [
    ("nope", {}, "unknown_tool"), ("scan", "{not json", "malformed_call"), ("scan", {"protocol": LOCALISER, "seed": 1}, "unknown_parameter"),
    ("scan", {}, "invalid_value"), ("scan", {"protocol": dict(LOCALISER, tr_ms=3.0)}, "console_limit"),
    ("scan", {"protocol": dict(LOCALISER, seed=4)}, "unknown_parameter"), ("scan", {"protocol": dict(LOCALISER, averages=64)}, "budget"),
    ("read_regions", {"scan": 5, "regions": [{"y_mm": 0, "x_mm": 0, "radius_mm": 5}]}, "not_found"),
    ("scan", {"protocol": LOCALISER, "regions": [{"y_mm": 0, "x_mm": 0, "radius_mm": 5000}]}, "invalid_value"),
    ("view", {"scan": 0, "kind": "truth"}, "invalid_value"),
]


@pytest.mark.parametrize("name, arguments, kind", BAD_CALLS)
def test_bad_calls_are_observations_and_cost_nothing(name, arguments, kind):
    env = mrgym.make(NOWRAP, level=1, seed=3)
    env.reset()
    step = env.step({"name": name, "arguments": arguments})
    assert step.observation.is_error and not step.done and step.reward == 0.0
    data = step.observation.data
    assert data["error"]["type"] == kind and data["charged"]["scan_time_s"] == 0 and data["episode_continues"]
    refused = not I.counts_as_tool_call(data)                           # the status line says what the refusal cost
    assert ("(nothing charged)" in step.observation.text.splitlines()[-1]) == refused and env.state()["scan_time_used_s"] == 0.0


def test_rejected_plan_is_an_error_observation_with_the_console_code_and_a_fix():
    env = mrgym.make(NOWRAP, level=1, seed=3)
    env.reset()
    obs = env.step({"name": "plan", "arguments": {"protocol": dict(LOCALISER, tr_ms=3.0)}}).observation
    err = obs.data["error"]
    assert obs.is_error and err["type"] == "console_limit" and err["code"] == "TR_BELOW_MINIMUM" and err["parameter"] == "tr_ms"
    assert err["fix"] == {"tr_ms": err["nearest_valid"]} and "fix: " in obs.text
    again = env.step({"name": "plan", "arguments": dict(LOCALISER, **err["fix"])}).observation
    assert not again.is_error and again.text.startswith("plan ok")


def test_max_steps_truncates_with_reward_zero():
    env = mrgym.make(NOWRAP, level=1, seed=3, max_steps=2)
    env.reset()
    env.step({"name": "scanner_info", "arguments": {}})
    step = env.step({"name": "scanner_info", "arguments": {"topic": "physics"}})
    assert step.truncated and not step.terminated and step.reward == 0.0 and "signal mechanisms, one topic each" in step.observation.text


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
    assert len(grid) == 16 and set("".join(grid)) <= set("0123456789")   # 16 blocks unless ``size`` asks for more
    for mode in out:                                                     # a view shows the picture and measures nothing in it
        assert "extent" not in out[mode][1].text and "bounding_box" not in out[mode][1].data


def test_tools_in_three_wire_formats_and_schema_generated_from_the_console():
    from mrgym.scanner import console
    env = mrgym.make(T1, level=1, seed=0)
    env.reset()
    oa, mc, an = env.tools("openai"), env.tools("mcp"), env.tools("anthropic")
    assert [t["function"]["name"] for t in oa] == [t["name"] for t in mc] == [t["name"] for t in an]
    by_name = {t["function"]["name"]: t["function"] for t in oa}
    plan = by_name["plan"]["parameters"]                                 # the protocol parameters are the arguments of plan
    generated = console.protocol_schema(env._session.exam)              # names, types and enums are the console's; plan adds ``verbose``
    assert list(plan["properties"]) == list(generated["properties"]) + ["verbose"] and plan["required"] == generated["required"]
    assert all(plan["properties"][k].get(key) == v.get(key) for k, v in generated["properties"].items() for key in ("type", "enum", "anyOf"))
    assert plan["properties"]["sequence"]["enum"] == sorted(console.SEQUENCES)
    assert set(by_name["scan"]["parameters"]["properties"]) == (set(plan["properties"]) - {"verbose"}) | {"regions", "circles"}
    assert not [p for t in env._tool_specs() for p in T.check_tool(t)]
    submit = oa[-1]["function"]["parameters"]["properties"]["t1_ms"]
    assert submit["required"] == [r["id"] for r in env._regions]        # tools are a function of the episode
    hints = {t["name"]: t["annotations"]["readOnlyHint"] for t in mc}
    assert hints["plan"] and hints["status"] and not hints["scan"] and not hints["submit"]
    with pytest.raises(ValueError):
        T.ToolSpec("reset", "x", {"type": "object"})


# ------------------------------------------------------------------ policies through the tools
@pytest.mark.parametrize("env_id, level", [(NOWRAP, 2), (NOWRAP, 3), (T1, 2)])
def test_control_policies_through_the_tools_match_the_scanner_interface(env_id, level):
    """The reference policy played through ``env.step`` leaves the same scanner log and answer as the same policy
    on the scanner interface directly: the tool surface adds nothing and loses nothing."""
    import importlib

    from mrgym.scanner import session
    spec = mrgym.spec(env_id)
    controls = importlib.import_module(spec.family + ".controls")
    env = mrgym.make(env_id, level=level, seed=5)
    r = policies.run(env, "reference")
    assert r.info["policy_error"] is None and r.terminated and r.info["replay"]["ok"]
    sess = session.ScannerSession(env.episode_config(), engine=env._session.engine)
    answer = controls.reference(session.LocalScanner(sess))
    assert sess.used == pytest.approx(r.info["budget"]["used_s"]) and sess.n_scans == r.info["budget"]["scans"]
    assert [e["data_sha256"] for e in sess.events if e["type"] == "scan"] == [e["data_sha256"] for e in env.scanner_log() if e["type"] == "scan"]
    assert r.reward == 1.0 and r.info["score"]["passed"] and all(r.info["score"]["checks"].values())
    assert env.task.score(env.episode_config(), answer, world=env.hidden_world(), engine=env._session.engine)["passed"]
    floor = policies.run(mrgym.make(env_id, level=level, seed=5), "prior")
    assert floor.reward == 0.0 and not floor.info["score"]["checks"]["measured"]
    unit = policies.run(mrgym.make(env_id, level=level, seed=5), "wrong_units")
    assert unit.reward == 0.0 and not unit.info["score"]["checks"]["deliverable_valid"] and unit.info["reason"] == "deliverable_refused"


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
    assert "SE 1.5T" in s.observation.data["header"] and "te_ms " in s.observation.text and not s.observation.is_error
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
    calls = iter([("scan", {"protocol": LOCALISER}), ("submit", SAFE_GEOMETRY)])

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


def _stored(env_id, verdicts, engine=None, schema=gate.SCHEMA):
    return dict(schema=schema, env_id=env_id, engine=engine or mrgym.ENGINE_ID, levels={str(lv): dict(verdict=v) for lv, v in verdicts.items()})


def test_the_claim_of_every_registered_id_agrees_with_its_stored_gate():
    """``kind`` in the registry is a claim; the stored gate result is the evidence. There is no verdict per instance."""
    for env_id in mrgym.list_envs():
        spec, stored = mrgym.spec(env_id), gate.load(env_id)
        ok, why = check.claim_evidence(spec, stored, mrgym.ENGINE_ID)
        assert ok, f"{env_id}: listed as {spec.kind}, {why}"
        if spec.kind == "sandbox":
            assert stored is None
        if spec.kind == "admitted":
            assert stored["engine"] == mrgym.ENGINE_ID and {lv["verdict"] for lv in stored["levels"].values()} == {"admitted"}


def test_claim_against_evidence_by_kind():
    spec = mrgym.spec(NOWRAP)                                             # claim levels 2 and 3; levels 0 and 1 are practice
    here, all_in = mrgym.ENGINE_ID, {2: "admitted", 3: "admitted"}
    as_kind = lambda kind: type(spec)(**{**spec.__dict__, "kind": kind})   # noqa: E731
    ev = lambda kind, stored: check.claim_evidence(as_kind(kind), stored, here)   # noqa: E731
    # candidate: nothing is required of the gate
    assert ev("candidate", None) == (True, "no gate stored on this engine")
    assert ev("candidate", _stored(NOWRAP, all_in, engine="mrgym-engine-0.1.0")) == (True, "no gate stored on this engine")
    assert ev("candidate", _stored(NOWRAP, all_in)) == (True, "level 2 admitted, level 3 admitted")
    # admitted: stored, on this engine, every claim level admitted
    assert ev("admitted", _stored(NOWRAP, all_in)) == (True, "level 2 admitted, level 3 admitted")
    assert ev("admitted", _stored(NOWRAP, {0: "practice", 1: "practice", 2: "admitted", 3: "admitted"}))[0]      # practice levels are outside the claim
    assert ev("admitted", None) == (False, "no stored gate result")
    assert ev("admitted", _stored(NOWRAP, all_in, engine="mrgym-engine-0.1.0")) == (
        False, f"the stored gate was measured on mrgym-engine-0.1.0, the engine is {here}")
    assert ev("admitted", _stored(NOWRAP, all_in, schema="mrgym-admission/1"))[0] is False
    assert ev("admitted", _stored(NOWRAP, {2: "admitted", 3: "practice"})) == (False, "level 2 admitted, level 3 practice")
    assert ev("admitted", _stored(NOWRAP, {2: "admitted"})) == (False, "level 2 admitted, level 3 not run")
    # practice: a stored gate has to say why: a rejected level, or a fixed protocol that passes like the reference
    assert ev("practice", None)[0] and ev("practice", _stored(NOWRAP, {2: "admitted", 3: "practice"}))[0]
    ok, why = ev("practice", _stored(NOWRAP, all_in))
    assert not ok and "not a practice id" in why
    rows = lambda ref, fixed: dict(verdict="admitted", reference=dict(rate=ref), reported=[dict(policy="open_loop", rate=fixed)])   # noqa: E731
    near = dict(_stored(NOWRAP, all_in), levels={"2": rows(0.98, 0.97), "3": rows(0.97, 0.80)})
    assert check.open_loop_gap(near, ["2", "3"]) == pytest.approx(0.17)
    assert ev("practice", near) == (True, "level 2 admitted, level 3 admitted; a fixed protocol (open_loop) passes within 17 points of the "
                                          "reference: a measurement task")
    far = dict(near, levels={"2": rows(0.98, 0.97), "3": rows(0.97, 0.60)})
    assert ev("practice", far)[0] is False and check.open_loop_gap(_stored(NOWRAP, all_in), ["2", "3"]) is None
    assert check.claim_evidence(mrgym.spec(EXPLORE), None, here) == (True, "no reward")


def test_check_fails_for_an_admitted_id_without_its_gate_and_passes_for_a_candidate(monkeypatch):
    spec = mrgym.spec(NOWRAP)
    monkeypatch.setattr(gate, "load", lambda env_id: None)                              # as if no gate had been stored
    monkeypatch.setitem(mrgym.registry, NOWRAP, type(spec)(**{**spec.__dict__, "kind": "candidate"}))
    assert [r for r in check.check_env(NOWRAP) if not r[1]] == []                       # a candidate needs no stored gate
    monkeypatch.setitem(mrgym.registry, NOWRAP, type(spec)(**{**spec.__dict__, "kind": "admitted"}))
    bad = [(name, detail) for name, ok, detail in check.check_env(NOWRAP) if not ok]
    assert ("listed as admitted: the claim agrees with the stored gate result", "no stored gate result") in bad
    assert any(name == "environment card is complete" and "without a stored gate result" in detail for name, detail in bad)


def test_family_contract_problems_are_named():
    assert check.contract_problems(mrgym.spec(NOWRAP)) == [] and check.contract_problems(mrgym.spec(T1)) == []
    import mrgym.envs.nowrap.controls as controls
    spec = mrgym.spec(NOWRAP)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(controls, "policies", lambda recipe=None: dict(reference=(controls.reference, 1.0)))
        problems = check.contract_problems(spec)
    assert "policy prior is listed and not in policies()" in problems
    assert any("is not a callable of the scanner" in p for p in problems)


@pytest.mark.parametrize("env_id", mrgym.list_envs())
def test_check_passes_for_every_registered_id(env_id):
    res = check.check_env(env_id)
    assert not [(name, detail) for name, ok, detail in res if not ok]
    assert len(res) >= 18 and res[-1][0] == f"listed as {mrgym.spec(env_id).kind}: the claim agrees with the stored gate result"


def test_verify_command_scores_a_served_episode(tmp_path, capsys):
    """The service path end to end without HTTP: the service's episode writes the log and its head, ``mrgym verify``
    replays the log against the head and scores the deliverable with the environment's own code."""
    import importlib

    from mrgym import cli
    from mrgym.scanner import service, session
    svc = service.Service(str(tmp_path / "log"), str(tmp_path / "state"), public_worlds_ok=True)
    spec = dict(env_id=NOWRAP, level=1, split="train", seed=5)
    ep = svc.create(spec)
    controls = importlib.import_module("mrgym.envs.nowrap.controls")
    answer = controls.reference(session.LocalScanner(ep.session))
    assert svc.submit(ep, answer) == {"received": True}
    svc.close()
    (tmp_path / "episode.json").write_text(json.dumps(spec))
    (tmp_path / "answer.json").write_text(json.dumps(answer))
    args = ["verify", "--episode-file", str(tmp_path / "episode.json"), "--log-dir", str(tmp_path / "log" / ep.handle),
            "--head", str(tmp_path / "state" / f"{ep.handle}.head"), "--answer", str(tmp_path / "answer.json")]
    assert cli.main(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["reward"] == svc.report(ep)["reward"] == 1.0 and out["replay"]["ok"] and out["replay"]["mode"] == "bit-exact"
    assert out["replay"]["sealed"] and out["replay"]["integrity"] == "anchored"


def test_typed_arguments_may_arrive_as_strings():
    """Tool-call formats that write every parameter as text must reach the scanner; malformed text must still be refused."""
    import json

    import mrgym

    proto = {"sequence": "spgr", "tr_ms": 11, "flip_deg": 10, "fov_mm": 288, "matrix": 160, "averages": 2}
    as_text = dict(proto, tr_ms="11", matrix="160")
    for args in ({"protocol": json.dumps(proto)}, {"protocol": as_text}, {"protocol": json.dumps(as_text)}):
        env = mrgym.make(NOWRAP, level=1, seed=0)
        env.reset()
        obs, *_ = env.step({"name": "plan", "arguments": args})
        assert not obs.is_error and obs.text.startswith("plan ok"), obs.text[:120]
    for args in ({"protocol": "{'sequence': 'spgr'}"}, {"protocol": "spgr"}, {"protocol": dict(proto, tr_ms="ten")}):
        env = mrgym.make(NOWRAP, level=1, seed=0)
        env.reset()
        obs, *_ = env.step({"name": "plan", "arguments": args})
        assert obs.is_error and obs.text.startswith("error invalid_value"), obs.text[:120]


def test_step_never_raises_on_malformed_numbers_or_shapes():
    """NaN, Infinity, a 320-digit integer, or a list where a single value belongs: an error observation, not an exception."""
    import mrgym

    proto = {"sequence": "spgr", "tr_ms": 11, "flip_deg": 10, "fov_mm": 288, "matrix": 160, "averages": 2}
    calls = [
        {"name": "plan", "arguments": {"protocol": dict(proto, tr_ms=float("nan"))}},
        {"name": "scan", "arguments": {"protocol": dict(proto, fov_mm=float("inf"))}},
        {"name": "plan", "arguments": '{"protocol": {"sequence": "spgr", "tr_ms": NaN}}'},
        {"name": "plan", "arguments": '{"protocol": {"sequence": "spgr", "tr_ms": Infinity}}'},
        {"name": "plan", "arguments": {"protocol": dict(proto, matrix=int("9" * 320))}},
        {"name": "plan", "arguments": {"protocol": dict(proto, sequence=["spgr"])}},
        {"name": "scan", "arguments": {"protocol": dict(proto, sequence={"a": 1})}},
        {"name": "plan", "arguments": {"protocol": dict(proto, tr_ms=[10])}},
        {"name": "scan", "arguments": {"protocol": proto, "regions": [{"y_mm": float("nan"), "x_mm": 0, "radius_mm": 5}]}},
    ]
    for env_id in (NOWRAP, T1, EXPLORE):
        for call in calls:
            env = mrgym.make(env_id, level=1, seed=0)
            env.reset()
            obs, reward, terminated, truncated, info = env.step(call)
            assert obs.is_error and not terminated, (env_id, call, obs.text[:100])
            obs, *_ = env.step({"name": "plan", "arguments": {"protocol": proto}})     # the episode is still usable
            assert not obs.is_error or obs.data["error"]["type"] == "console_limit", (env_id, obs.text[:100])


def test_step_survives_hostile_calls_after_a_scan():
    """Integer-valued floats are integers; calls that cannot be executed come back as errors."""
    import mrgym

    proto = {"sequence": "spgr", "tr_ms": 11, "flip_deg": 10, "fov_mm": 288, "matrix": 160, "averages": 2}
    hostile = [
        {"name": "view", "arguments": {"scan": 0, "image": 0.0}}, {"name": "view", "arguments": {"scan": 0, "size": 32.0}},
        {"name": "view", "arguments": {"scan": 0.0}}, {"name": "view", "arguments": {"scan": 99}},
        {"name": "view", "arguments": {"scan": 0, "image": 7}}, {"name": "view", "arguments": {"scan": 0, "size": 100000}},
        {"name": "read_regions", "arguments": {"scan": 0, "regions": [{"y_mm": 9e5, "x_mm": 0, "radius_mm": 5}]}},
        {"name": "plan", "arguments": {"protocol": dict(proto, matrix=160.0, averages=2.0)}},
        {"name": "", "arguments": {}}, {"name": "plan", "arguments": [1, 2]}, "plan", None,
    ]
    for env_id in (NOWRAP, EXPLORE):
        for call in hostile:
            env = mrgym.make(env_id, level=1, seed=0)
            env.reset()
            env.step({"name": "scan", "arguments": {"protocol": proto}})
            obs, reward, terminated, truncated, info = env.step(call)          # must not raise
            assert not terminated and reward == 0.0, (env_id, call)
    env = mrgym.make(NOWRAP, level=1, seed=0)
    env.reset()
    env.step({"name": "scan", "arguments": {"protocol": proto}})
    obs, *_ = env.step({"name": "view", "arguments": {"scan": 0, "image": 0.0}})
    assert not obs.is_error, obs.text[:120]


def test_replaying_an_instance_with_a_log_dir_starts_a_fresh_episode(tmp_path):
    """The log directory is per episode, not per instance: no inherited scans, budget or 'measured' gate."""
    import os

    import mrgym

    proto = {"sequence": "spgr", "tr_ms": 11, "flip_deg": 10, "fov_mm": 288, "matrix": 160, "averages": 2}
    spent = []
    for _ in range(3):
        env = mrgym.make(NOWRAP, level=1, seed=7, log_dir=str(tmp_path))
        env.reset()
        assert env._session.n_scans == 0 and env._session.used == 0.0
        env.step({"name": "scan", "arguments": {"protocol": proto}})
        spent.append(env._session.used)
        env.reset()                                                      # the same object, played again
        assert env._session.n_scans == 0 and env._session.used == 0.0
    assert len(set(spent)) == 1 and spent[0] > 0
    assert len(os.listdir(tmp_path)) == 6


def test_a_hidden_value_is_served_when_it_stands_as_a_number_not_when_its_digits_do():
    """``mrgym check`` looks for the hidden values of an episode in everything the agent received. A value is found
    where it stands as a number of its own; its digits inside another number are not the value. (A plain substring
    search reported the 150 ms of the ``Scanner/NoiseSNR-v0`` phantom in a read-out of 0.0121506732 once the scan
    data of engine 0.3.0 held those digits.)"""
    served = check.served
    assert served("150", '"t1_ms": 150.0') and served("150", "T1 150 ms") and served("150", "[150]") and served("150", "150")
    assert not served("150", "0.0121506732") and not served("150", "2.251500048") and not served("150", "1500") and not served("150", "0.150")
    assert served("0.0221475", "noise 0.0221475 per sample") and served("0.0221475", "0.02214753") and not served("0.0221475", "10.0221475")
    assert served("-4.12836", "offset -4.12836 Hz") and served("ab" * 16, "seed " + "ab" * 16) and not served("0.982253", "x")
