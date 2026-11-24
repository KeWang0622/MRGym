"""The agent runner, end to end against the stub endpoint on loopback. No model is called.

What is shown
  * a stub that replays the reference policy as tool calls reproduces the scripted reference episode exactly:
    reward, pass, error, scans, scan seconds and the head of the scanner's hash-chained log; also with the text and
    the tool call in different ``choices[]``, without a usage block, after two malformed turns, through a flaky
    endpoint, and for a model that rejects sampling parameters
  * the no-scan answer earns reward 0
  * malformed arguments, unknown tools, prose, empty replies, call floods and overspending are scored episodes with
    the right stop reason and reward 0, never a crash
  * a dead endpoint gives ``infra_error`` records that are left out of the rates and played again on resume
  * a finished run starts nothing when repeated; other settings in the same folder are refused
  * the world seed is in no saved file and no request
  * ``mrgym results --check`` passes on a fresh build and fails after a record is edited
"""
import glob
import json
import os
import re
import socket

import pytest

import mrgym
from mrgym import cli, results
from mrgym.agents import chat, runner, stub_server, summarize
from mrgym.env import derive_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOWRAP, T1 = "Scanner/NoWrap-v0", "Scanner/T1Transmit-v0"
SEEDS = [0, 1]
FAST = dict(backoff_s=0.01, max_retries=3, quiet=True)
FAILURES = dict(malformed_json="turn_cap", unknown_tool="turn_cap", text_only="no_tool_calls", empty="no_tool_calls",
                flood="truncated", bad_args="submitted", overspend="submitted")


@pytest.fixture(scope="module")
def stub():
    httpd, url = stub_server.start(seeds=SEEDS)
    yield url
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture(scope="module")
def replay(stub, tmp_path_factory):
    """Scripted reference and prior, and the stub replaying both, on the same instances of both environments."""
    out = str(tmp_path_factory.mktemp("replay"))
    agents = dict(policies=["reference", "prior"], models=["stub:reference", "stub:prior"], base_url=stub, workers=2, **FAST)
    first = runner.run(out, [NOWRAP], levels=[1, 2], seeds=SEEDS, **agents)
    second = runner.run(out, [T1], levels=[2], seeds=[0], **agents)
    return out, first["started"] + second["started"], first["infra_errors"] + second["infra_errors"]


@pytest.fixture(scope="module")
def variants(stub, tmp_path_factory):
    """One instance, every failure behaviour and every endpoint habit of the stub."""
    out = str(tmp_path_factory.mktemp("variants"))
    models = ["stub:" + k for k in FAILURES] + ["stub:reference", "stub:reference+split+nousage", "stub:recover+flaky", "stub:reference+noparams"]
    res = runner.run(out, [NOWRAP], levels=[1], seeds=[0], models=models, policies=["reference"], base_url=stub, max_turns=12,
                     temperature=0.0, max_tokens=2000, workers=3, **FAST)
    return out, res


def by_agent(out):
    table = {}
    for r in summarize.load(out):
        table.setdefault(r["agent"], {})[(r["env_id"], r["level"], r["seed"], r["episode"])] = r
    return table


def one(out, agent):
    (rec,) = by_agent(out)[agent].values()
    return rec


SAME = ("reward", "passed", "error", "scans", "scan_s_used", "scan_s_budget", "scanner_requests", "gates", "task_id", "admitted")


# ------------------------------------------------------------------ the reference, replayed as tool calls
def test_reference_replay_reproduces_the_scripted_episode(replay):
    out, started, infra = replay
    assert (started, infra) == (20, 0)
    table = by_agent(out)
    assert set(table) == {"reference", "prior", "stub:reference", "stub:prior"}
    assert len(table["reference"]) == 5 and set(table["reference"]) == set(table["stub:reference"])
    for key, ref in table["reference"].items():
        got = table["stub:reference"][key]
        assert (ref["agent_kind"], got["agent_kind"], got["model"]) == ("scripted", "stub", "stub:reference")
        assert ref["status"] == got["status"] == "completed" and ref["stop_reason"] == got["stop_reason"] == "submitted"
        for field in SAME:
            assert got[field] == ref[field], (key, field)
        assert got["log"]["head"] == ref["log"]["head"] and ref["log"]["head"] is not None, key
        assert got["log"]["events"] == ref["log"]["events"] and got["log"]["replays"] and ref["log"]["replays"]
        assert ref["passed"] and ref["reward"] == 1.0 and ref["scans"] >= 1, key
        assert got["turns"] == got["tool_calls"] == ref["tool_calls"] and got["invalid_calls"] == 0
        assert got["tokens_in"] > 0 and got["tokens_out"] > 0 and not got["tokens_estimated"]
        assert ref["turns"] is None and ref["tokens_in"] is None
        assert (got["engine"], got["package_version"]) == (mrgym.ENGINE_ID, mrgym.__version__)


def test_prior_answer_earns_nothing(replay):
    table = by_agent(replay[0])
    for agent in ("prior", "stub:prior"):
        assert len(table[agent]) == 5
        for rec in table[agent].values():
            assert rec["status"] == "completed" and rec["stop_reason"] == "submitted"
            assert rec["reward"] == 0.0 and not rec["passed"] and rec["scans"] == 0 and rec["scan_s_used"] == 0
            assert rec["gates"]["deliverable"] and not rec["gates"]["measured"]
    for key, rec in table["prior"].items():
        assert table["stub:prior"][key]["error"] == rec["error"]


def test_endpoint_habits_do_not_change_the_episode(variants):
    out, res = variants
    assert res["infra_errors"] == 0
    ref = one(out, "reference")
    for agent in ("stub:reference", "stub:reference+split+nousage", "stub:recover+flaky", "stub:reference+noparams"):
        rec = one(out, agent)
        assert rec["status"] == "completed" and rec["stop_reason"] == "submitted", agent
        for field in SAME:
            assert rec[field] == ref[field], (agent, field)
        assert rec["log"]["head"] == ref["log"]["head"], agent
    split = one(out, "stub:reference+split+nousage")           # the tool call sat in choices[1]; no usage block
    assert split["turns"] == split["tool_calls"] == ref["tool_calls"]
    assert split["tokens_estimated"] and split["tokens_in"] > 0
    recover = one(out, "stub:recover+flaky")                    # two bad turns, one HTTP 500, one body that is not JSON
    assert recover["invalid_calls"] == 2 and recover["turns"] == ref["tool_calls"] + 2
    assert recover["env_tool_calls"] == ref["tool_calls"] + 2


def test_rejected_sampling_parameters_are_dropped_and_reported(variants):
    out, _ = variants
    refused = one(out, "stub:reference+noparams")
    assert refused["params_dropped"] == ["temperature", "max_tokens"] and refused["params_sent"] == {"max_completion_tokens": 2000}
    accepted = one(out, "stub:reference")
    assert accepted["params_sent"] == {"temperature": 0.0, "max_tokens": 2000} and accepted["params_dropped"] == []
    assert accepted["settings"]["max_turns"] == 12 and accepted["settings"]["calc"] is True


# ------------------------------------------------------------------ agent failures are scored episodes
@pytest.mark.parametrize("behaviour", sorted(FAILURES))
def test_agent_failure_is_a_scored_episode(variants, behaviour):
    rec = one(variants[0], "stub:" + behaviour)
    assert rec["status"] == "completed" and rec["stop_reason"] == FAILURES[behaviour]
    assert rec["reward"] == 0.0 and not rec["passed"]
    if behaviour in ("malformed_json", "unknown_tool"):
        assert rec["turns"] == 12 and rec["invalid_calls"] == 12 and rec["scans"] == 0
    if behaviour in ("text_only", "empty"):
        assert rec["turns"] == runner.IDLE_TURNS and rec["tool_calls"] == 0
    if behaviour == "flood":                                    # the environment's own limit of tool calls ends it
        assert rec["env_tool_calls"] == mrgym.spec(NOWRAP).max_steps and rec["turns"] == 4 and rec["tool_calls"] == 120
    if behaviour == "bad_args":
        assert rec["turns"] == 10 and rec["invalid_calls"] == 7 and not rec["gates"]["deliverable"]
    if behaviour == "overspend":                                # the scanner refused; the budget held
        assert rec["scans"] >= 1 and rec["scan_s_used"] <= rec["scan_s_budget"] and rec["log"]["replays"]
        assert rec["gates"]["measured"] and rec["gates"]["deliverable"]


def test_calc_is_arithmetic_only():
    assert runner.calc("a = [1, 2, 3]; mean(a) * 2; a[-1] ^ 2; max(1, 5)") == [("a", [1.0, 2.0, 3.0]), (None, 4.0), (None, 9.0), (None, 5.0)]
    for text in ("__import__('os').system('id')", "open('x')", "a.b", "[x for x in [1]]", "'text'", "lambda: 1", "import os"):
        with pytest.raises((ValueError, SyntaxError)):
            runner.calc(text)
    text, ok = runner.calc_call('{"expressions": "sqrt(2)"}')
    assert ok and "1.414214" in text
    text, ok = runner.calc_call('{"expressions": "sqrt(')
    assert not ok and text.startswith("error: invalid_tool_call")


def test_merge_choices_reads_every_choice():
    call = {"id": "c1", "type": "function", "function": {"name": "scan", "arguments": "{}"}}
    text, calls, finish = chat.merge_choices({"choices": [
        {"finish_reason": "stop", "message": {"role": "assistant", "content": "Scanning."}},
        {"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [call, call]}},
        {"finish_reason": "tool_calls", "message": {"role": "assistant", "function_call": {"name": "plan", "arguments": {"a": 1}}}}]})
    assert text == "Scanning." and [c["name"] for c in calls] == ["scan", "plan"] and finish == ["stop", "tool_calls", "tool_calls"]
    assert calls[1]["arguments"] == '{"a": 1}' and len({c["id"] for c in calls}) == 2
    echoed = chat.assistant_message("", [dict(id="x", name="bad name?", arguments="{cut")])
    assert echoed["tool_calls"][0]["function"] == {"name": "bad_name_", "arguments": "{}"} and echoed["content"] is None


# ------------------------------------------------------------------ endpoint failures, resume
def test_dead_endpoint_is_an_infra_error_and_is_played_again(stub, tmp_path):
    out = str(tmp_path)
    with socket.socket() as s:                                  # a loopback port with nothing behind it
        s.bind(("127.0.0.1", 0))
        closed = f"http://127.0.0.1:{s.getsockname()[1]}/v1"
    kw = dict(levels=[1], seeds=SEEDS, models=["stub:reference"], policies=["reference"], backoff_s=0.01, max_retries=1, quiet=True)
    res = runner.run(out, [NOWRAP], base_url=closed, **kw)
    assert (res["started"], res["infra_errors"]) == (4, 2)
    dead = by_agent(out)["stub:reference"]
    assert all(r["status"] == "infra_error" and r["stop_reason"] == "endpoint_error" and "reward" not in r for r in dead.values())
    row = next(r for r in summarize.summarise(summarize.load(out))["rows"] if r["agent"] == "stub:reference")
    assert (row["episodes"], row["passed"], row["infra_errors"], row["pass_rate"]) == (0, 0, 2, None)
    assert "| stub:reference | stub | 0 | 0/0 | - |" in results.table_markdown(out)
    always_500 = runner.run(out, [NOWRAP], levels=[1], seeds=[0], models=["stub:reference+dead"], base_url=stub,
                            backoff_s=0.01, max_retries=1, quiet=True)
    assert always_500["infra_errors"] == 1 and "HTTP 500" in always_500["records"][0]["infra_error"]
    again = runner.run(out, [NOWRAP], base_url=stub, **kw)       # the same command against a live endpoint
    assert (again["started"], again["already_finished"], again["infra_errors"]) == (2, 2, 0)
    row = next(r for r in summarize.summarise(summarize.load(out))["rows"] if r["agent"] == "stub:reference")
    assert (row["episodes"], row["passed"], row["infra_errors"]) == (2, 2, 0)


def test_resume_of_a_finished_run_starts_nothing(replay, stub):
    out, started, _ = replay
    agents = dict(policies=["reference", "prior"], models=["stub:reference", "stub:prior"], base_url=stub, **FAST)
    again = runner.run(out, [NOWRAP], levels=[1, 2], seeds=SEEDS, **agents)
    assert (again["started"], again["already_finished"], again["records"]) == (0, 16, [])
    assert len(runner._read(os.path.join(out, "run_config.json"))["invocations"]) == 2
    with pytest.raises(mrgym.Error, match="other settings"):
        runner.run(out, [NOWRAP], levels=[1], seeds=[0], models=["stub:reference"], base_url=stub, max_turns=7, **FAST)
    with pytest.raises(mrgym.Error, match="no environment of this run"):
        runner.run(out, [NOWRAP], seeds=[0], policies=["blind_vfa"], quiet=True)


def test_scripted_policy_plays_once_per_instance_and_skips_what_the_budget_forbids(tmp_path):
    res = runner.run(str(tmp_path), [T1], levels=[3], seeds=[0], policies=["all"], episodes=3, records_only=True, quiet=True)
    names = {r["agent"] for r in res["records"]}
    assert res["started"] == len(names) == len(res["records"]) and {"reference", "prior", "fixed_recipe"} <= names
    assert "ir_unpriced" not in names                           # needs 40 times the budget of the environment
    assert all(r["episode"] == 0 and r["recipe_source"] == "admission" for r in res["records"])
    files = {os.path.basename(p) for p in glob.glob(os.path.join(str(tmp_path), "episodes", "*", "*", "*", "*", "*", "*"))}
    assert files == {"record.json"}


def test_command_line_runs_a_model_and_a_policy_into_one_folder(stub, tmp_path, capsys):
    out = os.path.join(str(tmp_path), "run")
    argv = ["run-agent", "--env", NOWRAP, "--levels", "1", "--seeds", "0", "--model", "stub:prior", "--policy", "prior",
            "--base-url", stub, "--max-turns", "5", "--no-calc", "--record-bundles", "--backoff-s", "0.01", "--out", out]
    assert cli.main(argv) == 0
    assert "2 episode(s) to run, 0 already finished" in capsys.readouterr().out
    assert cli.main(argv) == 0
    assert "0 episode(s) to run, 2 already finished" in capsys.readouterr().out
    config = runner._read(os.path.join(out, "run_config.json"))
    assert len(config["invocations"]) == 1 and config["invocations"][0]["command"] == (
        f"mrgym run-agent --env {NOWRAP} --levels 1 --seeds 0 --policy prior --model stub:prior --base-url {stub} "
        f"--api-key-env OPENAI_API_KEY --max-turns 5 --no-calc --record-bundles --workers 1 --out {out}")
    seed_hex = derive_seed(NOWRAP, 1, "train", 0)
    for agent, kind in (("stub_prior", "stub"), ("policy-prior", "scripted")):
        folder = os.path.join(out, "episodes", agent, "Scanner-NoWrap-v0", "L1", "s0", "e0")
        rec = runner._read(os.path.join(folder, "record.json"))
        assert rec["agent_kind"] == kind and rec["status"] == "completed" and rec["bundle"] is True and rec["reward"] == 0.0
        bundle = runner._read(os.path.join(folder, "episode_bundle.json"))
        assert bundle["agent"]["name"] == rec["agent"] and bundle["result"]["reward"] == 0.0
        assert bundle["agent"]["language_model"] is False
        with open(os.path.join(folder, "episode_bundle.json"), encoding="utf-8") as f:
            assert seed_hex not in f.read()
    saved = runner._read(os.path.join(out, "episodes", "stub_prior", "Scanner-NoWrap-v0", "L1", "s0", "e0", "messages.json"))
    assert all(t["function"]["name"] != "calc" for t in saved["tools"]) and "`calc`" not in saved["messages"][0]["content"]
    events = runner._read(os.path.join(out, "episodes", "stub_prior", "Scanner-NoWrap-v0", "L1", "s0", "e0", "episode_bundle.json"))["agent_events"]
    assert [(e["role"], e["step"]) for e in events] == [("assistant", 1), ("assistant", 2)]


# ------------------------------------------------------------------ the world seed stays on the runner's side
def test_world_seed_is_in_no_saved_file(replay, variants):
    checked = 0
    for out in (replay[0], variants[0]):
        public = [os.path.join(out, "run_config.json")]
        for path in glob.glob(os.path.join(out, "episodes", "*", "*", "*", "*", "*", "record.json")):
            rec = runner._read(path)
            seed_hex = derive_seed(rec["env_id"], rec["level"], "train", rec["seed"])
            assert re.fullmatch("[0-9a-f]{32}", seed_hex)
            names = sorted(os.listdir(os.path.dirname(path)))
            assert names == (["record.json", "scanner_log.jsonl"] if rec["agent_kind"] == "scripted"
                             else ["messages.json", "record.json", "scanner_log.jsonl"])
            for name in names:
                with open(os.path.join(os.path.dirname(path), name), encoding="utf-8") as f:
                    assert seed_hex not in f.read(), (path, name)
                checked += 1
            for name in public:
                with open(name, encoding="utf-8") as f:
                    assert seed_hex not in f.read()
    assert checked > 60
    saved = runner._read(os.path.join(replay[0], "episodes", "stub_reference", "Scanner-T1Transmit-v0", "L2", "s0", "e0", "messages.json"))
    assert set(saved) == {"tools", "messages"} and saved["messages"][-1]["role"] == "tool"
    assert [m["role"] for m in saved["messages"][:3]] == ["system", "user", "assistant"]


def test_a_request_that_would_carry_the_seed_is_never_sent(stub, tmp_path, monkeypatch):
    leak = lambda env, text, *a, **k: [{"role": "user", "content": text + " " + env.episode_config()["seed"]}]   # noqa: E731
    monkeypatch.setattr(runner, "first_messages", leak)
    res = runner.run(str(tmp_path), [NOWRAP], levels=[1], seeds=[0], models=["stub:reference"], base_url=stub, **FAST)
    (rec,) = res["records"]
    assert rec["status"] == "infra_error" and rec["stop_reason"] == "runner_error" and "world seed" in rec["infra_error"]
    folder = os.path.join(str(tmp_path), "episodes", "stub_reference", "Scanner-NoWrap-v0", "L1", "s0", "e0")
    assert sorted(os.listdir(folder)) == ["record.json", "scanner_log.jsonl"]       # the conversation was not saved either
    with open(os.path.join(folder, "record.json"), encoding="utf-8") as f:
        assert derive_seed(NOWRAP, 1, "train", 0) not in f.read()


def test_dry_run_shows_the_first_request_and_calls_nothing(capsys, tmp_path):
    out = os.path.join(str(tmp_path), "never_made")
    assert cli.main(["run-agent", "--env", f"{NOWRAP},{T1}", "--levels", "2", "--seeds", "1", "--model", "some-model",
                     "--max-turns", "9", "--dry-run", "--out", out]) == 0
    text = capsys.readouterr().out
    assert not os.path.exists(out) and text.count("nothing was called") == 2
    (item, other) = runner.dry_run([NOWRAP, T1], levels=[2], seeds=[1], model="some-model", max_turns=9)
    req = item["request"]
    assert json.dumps(req, indent=1) in text and set(req) == {"model", "messages", "tools"}
    assert [m["role"] for m in req["messages"]] == ["system", "user"] and "at most 9 turns" in req["messages"][0]["content"]
    env = mrgym.make(NOWRAP, level=2, seed=1)
    assert req["messages"][1]["content"] == env.reset()[0].text
    assert req["tools"][:-1] == env.tools("openai") and req["tools"][-1]["function"]["name"] == "calc"
    assert all(t["function"]["name"] != "calc" for t in runner.dry_run([NOWRAP], seeds=[1], calc=False)[0]["request"]["tools"])
    assert other["env_id"] == T1 and derive_seed(T1, 2, "train", 1) not in text


# ------------------------------------------------------------------ the table
def test_wilson_interval_and_rates():
    assert summarize.wilson(0, 0) == (0.0, 1.0)
    lo, hi = summarize.wilson(10, 20)
    assert abs(lo - 0.2993) < 1e-3 and abs(hi - 0.7007) < 1e-3
    lo, hi = summarize.wilson(24, 24)
    assert abs(lo - 0.8620) < 1e-3 and hi == 1.0
    assert summarize.rate(3, 24) == "3/24 (12%)" and summarize.rate(0, 0) == "0/0"
    assert runner.parse_seeds("0-3,8") == [0, 1, 2, 3, 8] and runner.seeds_text([8, 0, 1, 2, 3]) == "0-3,8"


def test_results_check_passes_on_a_fresh_build_and_fails_after_an_edit(replay, capsys):
    out = replay[0]
    assert cli.main(["results", "--dir", out, "--quiet"]) == 0
    assert cli.main(["results", "--dir", out, "--check"]) == 0
    with open(os.path.join(out, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    assert readme.count(results.START) == readme.count(results.END) == 1
    assert results.dateline(out) in readme and results.table_markdown(out) in readme
    assert "No language model has been run on these instances yet." in readme
    assert "| reference | scripted | 2 | 2/2 (100%) | 34-100 % | 1.000 |" in readme
    assert "| stub:prior | stub | 2 | 0/2 (0%) | 0-66 % | 0.000 |" in readme
    stored = runner._read(os.path.join(out, "results.json"))
    assert stored["records"] == stored["completed"] == 20 and stored["infra_errors"] == 0 and stored["seeds"] == "0-1"
    gate = stored["admitted_instances"]                                  # the stored verdict of the admission gate, per record
    assert set(gate) == {NOWRAP, T1} and gate[T1] == {"2": 0} and all(0 <= n <= 2 for n in gate[NOWRAP].values())
    assert f"**`{NOWRAP}`, level 1** (2 instances; the admission gate admits {gate[NOWRAP]['1']} of them)" in readme
    row = next(r for r in stored["rows"] if (r["env_id"], r["level"], r["agent"]) == (NOWRAP, 1, "reference"))
    assert row["admitted_passed"] == row["admitted_episodes"] == gate[NOWRAP]["1"]
    assert [r["agent"] for r in stored["rows"] if (r["env_id"], r["level"]) == (NOWRAP, 1)] == ["prior", "reference", "stub:prior", "stub:reference"]

    path = glob.glob(os.path.join(out, "episodes", "policy-reference", "Scanner-NoWrap-v0", "L1", "s0", "e0", "record.json"))[0]
    with open(path, encoding="utf-8") as f:
        original = f.read()
    rec = json.loads(original)
    rec.update(passed=False, reward=0.0)
    runner.write_json(path, rec, indent=1)
    assert cli.main(["results", "--dir", out, "--check"]) == 1          # the stored table no longer matches the records
    assert set(results.check(out)) == {"results.json", "README.md"}
    with open(os.path.join(out, "README.md"), encoding="utf-8") as f:
        assert f.read() == readme                                        # --check wrote nothing
    with open(path, "w", encoding="utf-8") as f:
        f.write(original)
    assert cli.main(["results", "--dir", out, "--check"]) == 0
    with open(os.path.join(out, "README.md"), "a", encoding="utf-8") as f:
        f.write("\nA note outside the markers.\n")
    assert cli.main(["results", "--dir", out, "--quiet"]) == 0           # text outside the markers is kept
    with open(os.path.join(out, "README.md"), encoding="utf-8") as f:
        assert f.read().endswith("A note outside the markers.\n")
    capsys.readouterr()


def test_committed_results_match_their_records():
    """The results folder of the repository: the stored table and JSON are a fresh build from the records, and the
    page says so in plain words for as long as no language model has been run."""
    folder = os.path.join(ROOT, "results")
    if not os.path.isdir(os.path.join(folder, "episodes")):
        pytest.skip("no results folder in this checkout")
    assert results.check(folder) == []
    stored = runner._read(os.path.join(folder, "results.json"))
    assert stored["records"] == len(glob.glob(os.path.join(folder, summarize.PATTERN))) > 0
    assert stored["agents"]["stub"] == [], "rows of the test endpoint do not belong in the published results"
    with open(os.path.join(folder, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    assert ("No language model has been run on these instances yet." in readme) == (stored["agents"]["model"] == [])


def test_example_compiles():
    path = os.path.join(ROOT, "examples", "run_llm_agent.py")
    if not os.path.exists(path):
        pytest.skip("examples are not part of this checkout")
    with open(path, encoding="utf-8") as f:
        compile(f.read(), path, "exec")
