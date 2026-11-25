"""Adapter parity: the same seed and the same scripted policy give the same reward, the same observations and the
same scanner-log head through the native API and through each adapter, and every adapter serves the tool list of
``env.tools`` unchanged.

The ids come from the registry (``mrgym.list_envs()``), so a new family is covered without a change here. An id whose
scripted reference does not pass natively is still compared call for call; only the claim "the reference passes" is
skipped for it, with the reason. The Harbor path (the scanner service, its HTTP client and the separate verifier) needs
nothing beyond the package and always runs; it is compared on the reward and the scanner-log head. The Gymnasium, OpenEnv, verifiers and Inspect paths need their
frameworks, whose pins cannot share one environment; each is skipped where its framework is not installed, and
``make conformance`` runs this file once in each framework's environment.
"""
import asyncio
import functools
import json
import os
import sys

import pytest

import mrgym
from mrgym import compat
from mrgym import tools as T
from mrgym.compat import export, harbor
from mrgym.env import derive_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTEGRATIONS = os.path.join(ROOT, "integrations")
IDS = mrgym.list_envs()
# Seeds per id and adapter. One by default, so the test suite stays fast; `make conformance` sets MRGYM_PARITY_SEEDS=3.
SEEDS = tuple(range(int(os.environ.get("MRGYM_PARITY_SEEDS", "1"))))
KEYS = ("reward", "done", "observations", "text_sha256", "log_head")


@functools.cache
def scripted(env_id, seed):
    """The native episode: the id's scripted reference policy (the probe policy for an id without controls)."""
    return compat.scripted_episode(env_id, mrgym.spec(env_id).default_level, seed)


def native_tools(env_id, seed):
    env = mrgym.make(env_id, seed=seed)
    env.reset()
    out = env.tools("neutral")
    env.close()
    return out


def same(native, got):
    return {k: (native[k], got[k]) for k in KEYS if native[k] != got[k]}


def needs_passing_reference(env_id, seed=0):
    """Skip a test whose claim is "the reference passes": an id whose scripted reference does not pass natively on this
    engine cannot show it through any adapter. (Parity itself, equal traces for equal calls, is tested for every id.)"""
    ep = scripted(env_id, seed)
    if ep["policy"] == "probe":
        pytest.skip(f"{env_id} has no scripted reference policy")
    if not ep["passed"]:
        pytest.skip(f"the scripted reference of {env_id} does not pass natively on {mrgym.ENGINE_ID} at seed {seed} "
                    f"(reward {ep['reward']:.3f}{', ' + ep['policy_error'] if ep['policy_error'] else ''}): nothing to show through an adapter")


def test_the_registry_has_ids_with_a_scripted_episode_each():
    assert IDS
    for env_id in IDS:
        ep = scripted(env_id, 0)
        assert ep["calls"] and ep["done"], env_id                        # every id has an episode that ends, to replay


@pytest.mark.parametrize("env_id", IDS)
def test_the_reference_passes_natively(env_id):
    needs_passing_reference(env_id)
    ep = scripted(env_id, 0)
    assert ep["policy_error"] is None and ep["reward"] >= mrgym.spec(env_id).reward_threshold


@pytest.mark.parametrize("env_id", IDS)
def test_a_native_replay_of_the_recorded_calls_is_the_same_episode(env_id):
    ep = scripted(env_id, 0)
    assert not same(ep, compat.replay(env_id, mrgym.spec(env_id).default_level, 0, ep["calls"]))


# ------------------------------------------------------------------------------------------------ Harbor (always runs)
SCORED = [i for i in IDS if mrgym.spec(i).kind != "sandbox"]


def harbor_task(root, env_id, seed):
    """A Harbor task on the world of the public training seed ``seed`` (its raw world seed in the episode file)."""
    level = mrgym.spec(env_id).default_level
    ep = dict(export.new_episode(env_id, level), seed=derive_seed(env_id, level, "train", seed))
    token = export.new_token()
    task = os.path.join(str(root), export.task_dirname(ep, token))
    export._write(task, export.render_task(ep, token))
    return task, ep


@functools.cache
def harbor_trial(env_id, seed):
    """One exported task played both ways: (native episode on the task's world, oracle report, no-op report)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        task, ep = harbor_task(tmp, env_id, seed)
        native = compat.scripted_episode(env_id, ep["level"], ep["seed"], split=ep["split"])
        return native, harbor.local(task, policy="reference"), harbor.local(task, policy="nop") if seed == SEEDS[0] else None


@pytest.mark.parametrize("env_id", SCORED)
@pytest.mark.parametrize("seed", SEEDS)
def test_harbor_service_and_verifier_match_the_native_api(env_id, seed):
    """The scripted reference through the scanner service, the HTTP client and the separate verifier gives the reward
    and the scanner-log head of the same policy played natively through the tools."""
    native, report, floor = harbor_trial(env_id, seed)
    assert report["log_replays"] and report["sealed"], report["problems"]
    assert (report["reward"], report["log_head"], report["passed"]) == (native["reward"], native["log_head"], native["passed"])
    if floor is not None:                                                # no call and no deliverable: the floor
        assert floor["reward"] == mrgym.spec(env_id).reward_range[0] and not floor["submitted"] and not floor["passed"] and not floor["sealed"]


@pytest.mark.parametrize("env_id", SCORED)
def test_harbor_oracle_earns_the_threshold_where_the_native_reference_passes(env_id):
    native, report, _ = harbor_trial(env_id, SEEDS[0])
    if not native["passed"]:
        pytest.skip(f"the scripted reference of {env_id} does not pass natively on {mrgym.ENGINE_ID} for this instance (reward "
                    f"{native['reward']:.3f}); the Harbor path gives the same reward and log head, the oracle claim is not shown")
    assert report["passed"] and report["reward"] >= mrgym.spec(env_id).reward_threshold


@pytest.mark.parametrize("env_id", SCORED)
def test_harbor_service_serves_the_native_statement_and_deliverable_schema(tmp_path, env_id):
    task, ep = harbor_task(tmp_path / "task", env_id, 0)
    with harbor.LocalService(ep, str(tmp_path / "log"), str(tmp_path / "state")) as srv:
        served = harbor.HttpSession(srv.url, srv.token)._call("GET", "/v1/task")
    env = mrgym.make(env_id, level=ep["level"], split=ep["split"], seed=ep["seed"])
    obs, _ = env.reset()
    terminal = next(t for t in env.tools("neutral") if t["terminal"])
    assert served["statement"] == obs.text and served["submit"] == compat.plain(terminal["parameters"])
    assert open(os.path.join(task, "instruction.md")).read().startswith(obs.text.rstrip())


# ------------------------------------------------------------------------------------------------ Gymnasium
@pytest.mark.parametrize("env_id", IDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_gymnasium_adapter_matches_the_native_api(env_id, seed):
    gym = pytest.importorskip("gymnasium")
    import mrgym.compat.gymnasium  # noqa: F401  registers the ids
    native = scripted(env_id, seed)
    env = gym.make(env_id).unwrapped
    text, info = env.reset(seed=seed)
    assert info["tools"] == compat.plain(env.mrgym_env.tools("openai"))
    assert [t["function"]["name"] for t in info["tools"]] == [t["name"] for t in native_tools(env_id, seed)]
    texts, reward, done = [text], 0.0, False
    for call in native["calls"]:
        text, reward, terminated, truncated, info = env.step(json.dumps(call))
        texts.append(text)
        done = terminated or truncated
        if done:
            break
    assert not same(native, compat.trace(reward, done, texts, env.mrgym_env))
    env.close()


# ------------------------------------------------------------------------------------------------ OpenEnv
@pytest.mark.parametrize("env_id", IDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_openenv_environment_matches_the_native_api(env_id, seed):
    pytest.importorskip("openenv")
    if not os.path.isdir(INTEGRATIONS):
        pytest.skip("needs a checkout: integrations/ is not part of the wheel")
    sys.path.insert(0, os.path.join(INTEGRATIONS, "openenv", "mrgym_scanner_env"))
    from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
    from server.mrgym_environment import make_environment_class
    native = scripted(env_id, seed)
    env = make_environment_class(IDS[0])()
    o = env.reset(seed=seed, env_id=env_id, level=mrgym.spec(env_id).default_level)
    listed = env.step(ListToolsAction()).tools
    mcp = env.mrgym_env.tools("mcp")
    assert [(t.name, t.description, t.input_schema) for t in listed] == [(t["name"], t["description"], t["inputSchema"]) for t in mcp]
    texts, reward, done = [o.metadata["text"]], 0.0, False
    for call in native["calls"]:
        o = env.step(CallToolAction(tool_name=call["name"], arguments=call.get("arguments") or {}))
        texts.append(o.result["content"][0]["text"])
        reward, done = o.reward, o.done
        if done:
            break
    assert not same(native, compat.trace(reward, done, texts, env.mrgym_env))
    assert env.state.env_id == env_id and env.state.done == native["done"]


# ------------------------------------------------------------------------------------------------ verifiers
@pytest.mark.parametrize("env_id", IDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_verifiers_toolset_matches_the_native_api(env_id, seed):
    vf = pytest.importorskip("verifiers.v1")
    taskset = pytest.importorskip("mrgym_scanner.taskset")
    native = scripted(env_id, seed)
    level = mrgym.spec(env_id).default_level

    async def play():
        ts = taskset.ScannerToolset(vf.ToolsetConfig())
        await ts.setup_task(taskset.ScannerData(idx=0, name="parity", prompt="", env_id=env_id, level=level, seed=seed))
        listed = await ts.list_tools()
        for call in native["calls"]:
            result = await ts.call_tool(call["name"], call.get("arguments") or {})
            assert result.content[0].text == ts.episode.texts[-1]
            if ts.episode.done:
                break
        return ts, listed

    ts, listed = asyncio.run(play())
    mcp = ts.episode.tools()
    assert [(t.name, t.description, t.input_schema) for t in listed] == [(t["name"], t["description"], t["inputSchema"]) for t in mcp]
    assert not same(native, ts.episode.trace())
    assert (ts.state.done, ts.state.reward) == (native["done"], native["reward"])        # scoring reads typed state


@pytest.mark.parametrize("env_id", IDS)
def test_verifiers_taskset_yields_this_id_and_keeps_the_seed_out_of_the_prompt(env_id):
    pytest.importorskip("verifiers.v1")
    taskset = pytest.importorskip("mrgym_scanner.taskset")
    ts = taskset.MRGymScannerTaskset(taskset.ScannerConfig(env_id=env_id, first_seed=41))
    task = next(iter(ts.load()))
    env = mrgym.make(env_id, seed=41)
    obs, info = env.reset()
    assert task.data.env_id == env_id and task.data.prompt == obs.text and compat.instance_label(info) in task.data.name
    assert "41" not in task.data.name.split(":")[-1] and env.episode_config()["seed"] not in task.data.name


# ------------------------------------------------------------------------------------------------ Inspect
@pytest.mark.parametrize("env_id", IDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_inspect_task_matches_the_native_api(tmp_path, env_id, seed):
    pytest.importorskip("inspect_ai")
    if not os.path.isdir(INTEGRATIONS):
        pytest.skip("needs a checkout: integrations/ is not part of the wheel")
    sys.path.insert(0, os.path.join(INTEGRATIONS, "inspect"))
    import mock_run
    level = mrgym.spec(env_id).default_level
    schemas = {t["name"]: t["parameters"] for t in native_tools(env_id, seed)}
    # Inspect answers a schema violation itself, so only schema-valid calls reach the environment (see mock_run.py)
    calls = [c for c in scripted(env_id, seed)["calls"] if c["name"] in schemas and not T.validate(c.get("arguments", {}), schemas[c["name"]])]
    native = compat.replay(env_id, level, seed, calls)
    got = mock_run.play(env_id, level, seed, calls, str(tmp_path))
    assert got.get("status") == "success", got
    assert not same(native, got)
    floor = mock_run.play(env_id, level, seed, [], str(tmp_path))
    assert floor["reward"] == mrgym.spec(env_id).reward_range[0] and not floor["done"]
