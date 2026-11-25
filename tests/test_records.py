"""Episode records of schema 2, the stored conversations, the trajectory export, the frozen manifests and the
published / pilot tiers of the results table.

What is shown
  * a model episode stores the whole conversation: prompt, every model turn (with its reasoning text when the
    endpoint returns it), every tool call and result, usage and latency per request; the record carries its hash,
    the harness id and configuration, the sampling settings as sent, the delivered answer, the outcome
  * ``--records-only`` does not drop the conversation of a model
  * the export writes one JSON line per episode with messages, deliverable, harness id and the canary string
  * records of schema 1 (the 1,400 stored ones) are read, are pilot rows, and are in no published table or comparison
  * a synthetic run of the published design passes ``mrgym results --check``; a record without its conversation, a
    missing plan, a short design, another seed list or settings that were not sent each fail it, with the rule
  * the planned comparisons are the paired differences a few lines of numpy give
  * the stored manifests are frozen
"""
import copy
import glob
import json
import os

import numpy as np
import pytest
from scipy import stats as st

import synth
from mrgym import canary, cli, export, gate, manifest, record, results
from mrgym.agents import chat, runner, stub_server, summarize
from mrgym.stats import lint, power

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOWRAP = "Scanner/NoWrap-v1"
FAST = dict(backoff_s=0.01, max_retries=3, quiet=True)
MODELS = ["stub:reference", "stub:prior", "stub:text_only"]


@pytest.fixture(scope="module")
def stub():
    httpd, url = stub_server.start(seeds=[0, 1])
    yield url
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture(scope="module")
def run(stub, tmp_path_factory):
    """24 stub episodes (3 agents x 2 levels x 2 seeds x 2 episodes) and the scripted reference."""
    out = str(tmp_path_factory.mktemp("records"))
    res = runner.run(out, [NOWRAP], levels=[1, 2], seeds=[0, 1], episodes=2, models=MODELS, policies=["reference"], base_url=stub,
                     temperature=1.0, max_tokens=2000, max_turns=20, workers=2, command="mrgym run-agent --test", **FAST)
    assert (res["started"], res["infra_errors"]) == (28, 0)
    return out


def rules(violations) -> set:
    return {v["rule"] for v in violations}


# ------------------------------------------------------------------ what a record carries
def test_model_record_carries_everything_that_makes_it_checkable(run):
    recs = [r for r in summarize.load(run) if r["agent"] == "stub:reference"]
    assert len(recs) == 8
    for r in recs:
        assert r["schema"] == record.EPISODE_SCHEMA == runner.RECORD_SCHEMA and r["record_schema"] == 2 and "tier" not in r
        assert record.missing(r) == []
        assert r["harness"]["kind"] == "chat-tools" and r["harness"]["settings"]["max_turns"] == 20 and "submit" in r["harness"]["tools"]
        assert r["harness_id"] == record.harness_id(r["harness"]) and len(r["harness_id"]) == 16
        assert r["sampler"] == dict(requested=dict(temperature=1.0, max_tokens=2000), sent=dict(temperature=1.0, max_tokens=2000), dropped=[])
        assert r["max_turns"] == 20 and r["provider"].startswith("127.0.0.1:") and r["adapter_sha256"] is None
        assert r["group"] == dict(env_id=NOWRAP, level=r["level"], split="train", task_id=r["task_id"])
        assert r["deliverable"]["pe_dir"] in ("x", "y") and r["outcome"] == "pass" and r["ladder"] == dict(submitted=True, valid=True, measured=True, in_budget=True, passed=True)
        assert r["tokens_in"] > 0 and r["tokens_out"] > 0 and r["wall_s"] > 0 and r["model_s"] > 0 and r["requests"] == r["turns"]
        assert r["command"] == "mrgym run-agent --test" and r["plan_sha256"] is None and r["canary"] == canary.CANARY
        t = r["transcript"]
        assert t["complete"] and t["assistant_turns"] == r["turns"] and t["tool_results"] == r["tool_calls"]
        assert record.transcript_problem(r, os.path.join(run, r["_dir"])) is None
    assert len({r["harness_id"] for r in recs}) == 1 and "temperature" not in recs[0]["harness"]["settings"]      # sampling is in ``sampler``
    idle = [r for r in summarize.load(run) if r["agent"] == "stub:text_only"]
    assert all(r["deliverable"] is None and r["outcome"] == "no_delivery:no_tool_calls" and not r["ladder"]["submitted"] for r in idle)
    assert all(record.missing(r) == [] for r in idle)                       # no delivery is a result, not a gap in the record
    prior = [r for r in summarize.load(run) if r["agent"] == "stub:prior"]
    # the outcome and the ladder come from the family environment now (``mrgym.learn.hooks``): the first failed stage
    assert all(r["outcome"] == "no_measurement" and r["ladder"] == dict(submitted=True, valid=True, measured=False, in_budget=False, passed=False)
               for r in prior)


def test_stored_conversation_is_complete(run):
    r = next(r for r in summarize.load(run) if r["agent"] == "stub:reference")
    with open(os.path.join(run, r["_dir"], "messages.json"), encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["schema"] == record.TRANSCRIPT_SCHEMA and doc["canary"] == canary.CANARY and doc["harness_id"] == r["harness_id"] and doc["complete"]
    roles = [m["role"] for m in doc["messages"]]
    assert roles[:2] == ["system", "user"] and roles.count("assistant") == r["turns"] and roles.count("tool") == r["tool_calls"]
    assert [t["turn"] for t in doc["turns"]] == list(range(1, r["turns"] + 1))
    assert all(isinstance(t["usage"].get("prompt_tokens"), int) and t["seconds"] >= 0 and t["attempts"] >= 1 for t in doc["turns"])
    assert sum(t["usage"]["prompt_tokens"] for t in doc["turns"]) == r["tokens_in"]
    last = [m for m in doc["messages"] if m["role"] == "assistant"][-1]
    assert last["turn"] == r["turns"] and json.loads(last["tool_calls"][0]["function"]["arguments"]) == r["deliverable"]
    assert {t["function"]["name"] for t in doc["tools"]} == set(r["harness"]["tools"])
    sent = json.dumps([m for m in doc["messages"] if m["role"] in ("system", "user", "tool")])
    assert canary.CANARY not in sent and "CANARY" not in sent               # the marker is never in text the agent reads
    assert os.path.exists(os.path.join(run, r["_dir"], "scanner_log.jsonl"))


def test_scripted_record_has_no_conversation_and_misses_nothing(run):
    recs = [r for r in summarize.load(run) if r["agent_kind"] == "scripted"]
    assert len(recs) == 4
    for r in recs:
        tuned = str(r["level"]) in (gate.load(r["env_id"]) or {}).get("levels", {})     # a stored gate result carries the tuned recipe
        assert r["schema"] == record.EPISODE_SCHEMA and r["transcript"] is None
        assert r["harness"] == dict(kind="scripted", policy="reference", recipe="tuned" if tuned else "default", max_steps=r["max_steps"])
        assert record.missing(r) == [] and r["deliverable"] and r["outcome"] == "pass" and r["sampler"] is None
        assert os.path.exists(os.path.join(run, r["_dir"], "scanner_log.jsonl"))


def test_reasoning_text_and_reasoning_tokens_are_kept(stub, tmp_path, monkeypatch):
    post, thought = chat.ChatClient._post, "The object is wider than tall, so phase encoding goes along y."

    def with_reasoning(self, body):
        raw = post(self, body)
        raw["choices"][0]["message"]["reasoning_content"] = thought
        raw.setdefault("usage", {})["completion_tokens_details"] = {"reasoning_tokens": 17}
        return raw

    monkeypatch.setattr(chat.ChatClient, "_post", with_reasoning)
    out = str(tmp_path)
    runner.run(out, [NOWRAP], levels=[1], seeds=[0], models=["stub:reference"], base_url=stub, records_only=True, **FAST)
    (r,) = summarize.load(out)
    assert r["status"] == "completed" and r["tokens_reasoning"] == 17 * r["turns"] and r["transcript"]["reasoning_chars"] == len(thought) * r["turns"]
    with open(os.path.join(out, r["_dir"], "messages.json"), encoding="utf-8") as f:     # kept although --records-only was given
        doc = json.load(f)
    assistants = [m for m in doc["messages"] if m["role"] == "assistant"]
    assert all(m["reasoning"] == thought for m in assistants) and doc["turns"][0]["reasoning"] == assistants[0]["reasoning"]
    assert record.transcript_problem(r, os.path.join(out, r["_dir"])) is None


def test_reasoning_is_read_from_the_fields_endpoints_use():
    msg = lambda **kw: {"choices": [{"message": dict(content="ok", **kw)}]}             # noqa: E731
    assert record.reasoning_of(msg(reasoning_content="a")) == "a" and record.reasoning_of(msg(reasoning="b")) == "b"
    assert record.reasoning_of(msg(reasoning={"content": "c"})) == "c" and record.reasoning_of(msg()) == ""
    parts = {"choices": [{"message": {"content": [{"type": "thinking", "thinking": "d"}, {"type": "text", "text": "ok"}]}}]}
    assert record.reasoning_of(parts) == "d" and record.reasoning_of({}) == "" and record.reasoning_of(None) == ""


def test_a_tampered_or_missing_conversation_is_found(run, tmp_path):
    r = next(r for r in summarize.load(run) if r["agent"] == "stub:reference")
    d = str(tmp_path)
    with open(os.path.join(run, r["_dir"], "messages.json"), encoding="utf-8") as f:
        text = f.read()
    assert record.transcript_problem(r, d) == "messages.json is missing"
    with open(os.path.join(d, "messages.json"), "w", encoding="utf-8") as f:
        f.write(text.replace("Step 1", "Step one"))
    assert "sha256" in record.transcript_problem(r, d)
    assert record.transcript_problem(dict(r, transcript=None), d) == "no transcript in the record"
    gone = dict(r, transcript=None)
    assert "transcript" in record.missing(gone)
    cut = dict(r, transcript=dict(r["transcript"], assistant_turns=r["turns"] - 1))
    assert record.missing(cut) == ["transcript (incomplete)"]
    old = record.upgrade({k: v for k, v in r.items() if k in ("schema", "agent", "agent_kind", "status", "stop_reason", "turns")} | {"schema": "mrgym-episode/1"})
    assert old["tier"] == "pilot" and old["record_schema"] == 1 and {"transcript", "harness_id", "deliverable", "outcome"} <= set(record.missing(old))


# ------------------------------------------------------------------ export
def test_export_writes_one_line_per_episode_with_the_canary(run, tmp_path, capsys):
    path = str(tmp_path / "episodes.jsonl")
    assert cli.main(["export", "--dir", run, "--out", path]) == 0
    with open(path, encoding="utf-8") as f:
        text = f.read()
    rows = [json.loads(line) for line in text.splitlines()]
    assert len(rows) == 24 and text.count(canary.CANARY) == 24 and all(row["canary"] == canary.CANARY for row in rows)
    for row in rows:
        assert {"id", "env_id", "level", "split", "task_id", "harness_id", "messages", "tools", "reward", "passed", "terminated",
                "truncated", "deliverable", "gates", "tokens", "canary"} <= set(row)
        called = set()
        for m in row["messages"]:                                           # every tool message answers a call
            if m["role"] == "assistant":
                called.update(c["id"] for c in m.get("tool_calls") or [])
            elif m["role"] == "tool":
                assert m["tool_call_id"] in called
        assert (row["deliverable"] is not None) == row["terminated"] and row["terminated"] != row["truncated"]
        assert row["harness_id"] and row["tokens"]["in"] > 0 and row["split"] == "train"
    assert sum(row["terminated"] for row in rows) == 16 and sum(row["passed"] for row in rows) == 8
    assert len(list(export.lines(run, include_scripted=True))) == 28
    sft = list(export.lines(run, "sft"))
    assert len(sft) == 8 and all(set(x) == {"messages", "tools"} for x in sft)
    assert all(set(m) <= set(export.STANDARD_KEYS) for x in sft for m in x["messages"]) and canary.CANARY not in json.dumps(sft)
    capsys.readouterr()
    assert cli.main(["export", "--dir", run, "--agent", "stub:prior"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 8


def test_sft_export_refuses_evaluation_content(run, tmp_path):
    import shutil
    out = str(tmp_path / "copy")
    shutil.copytree(run, out)
    path = glob.glob(os.path.join(out, "episodes", "stub_reference", "*", "L1", "s0", "e0", "record.json"))[0]
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(rec, split="test-iid"), f)
    with pytest.raises(Exception, match="evaluation content"):
        list(export.lines(out, "sft"))
    assert len(list(export.lines(out, "rl"))) == 24


def test_canary_is_one_constant():
    assert canary.CANARY.startswith("MRGYM-CANARY GUID ") and len(canary.CANARY.split()[-1]) == 36
    assert record.CANARY == canary.CANARY and canary.stamp({})["canary"] == canary.CANARY
    assert "26b5c67b" not in canary.CANARY                                   # not the string of another benchmark
    with open(os.path.join(ROOT, "results", "results.json"), encoding="utf-8") as f:
        assert json.load(f)["canary"] == canary.CANARY


# ------------------------------------------------------------------ tiers
def test_small_runs_are_pilot_rows(run, capsys):
    s = results._summary(run)
    assert s["schema"] == "mrgym-results/2" and s["published_records"] == 0 and s["comparisons"] == []
    assert all(r["tier"] == "pilot" and r["tier_reason"] == "< 100 instances" and r["record_schemas"] == [2] for r in s["rows"])
    row = next(r for r in s["rows"] if r["agent"] == "stub:reference")
    assert (row["instances"], row["episodes"], row["episodes_per_instance"], row["passed"]) == (4, 8, [2, 2], 8)
    assert row["pass_rate"] == 1.0 and row["one_sided_bound"] == pytest.approx(0.05 ** 0.25, abs=1e-6) and row["ci_method"] in lint.CLUSTERED
    assert row["pass_ci95_beta_binomial"] is not None and row["retest_agreement"] == 1.0 and row["pass_all_k"] == 1.0
    assert row["sampling"] == dict(temperature=1.0, max_tokens=2000, max_turns=20) and row["records_missing"] == {} and row["transcripts"] == 8
    assert row["tokens_in_total"] > 0 and row["wall_s_total"] > 0 and row["cost_usd"] is None and row["seeds"] == "L1: 0-1; L2: 0-1"
    assert cli.main(["results", "--dir", run, "--quiet"]) == 0 and cli.main(["results", "--dir", run, "--check"]) == 0
    assert "(0 published, 28 pilot)" in capsys.readouterr().out
    with open(os.path.join(run, "README.md"), encoding="utf-8") as f:
        text = f.read()
    assert "No row is published yet" in text and "### Pilot rows (not published)" in text and "8/8 (100.0 %; " in text and "one-sided bound >= " in text


def test_stored_records_of_schema_1_are_pilot_and_not_published():
    directory = os.path.join(ROOT, "results")
    recs = summarize.load(directory)
    assert len(recs) == 1400 and all(r["record_schema"] == 1 and r["tier"] == "pilot" for r in recs)
    with open(os.path.join(directory, "results.json"), encoding="utf-8") as f:
        s = json.load(f)
    assert s["schema"] == "mrgym-results/2" and (s["published_records"], s["pilot_records"]) == (0, 1400)
    assert s["rows"] and all(r["tier"] == "pilot" and r["tier_reason"] == "schema 1" for r in s["rows"]) and s["comparisons"] == []
    assert s["reliability"]["cells"] == []
    drift, broken, fresh = results.audit(directory)
    assert drift == [] and broken == [] and fresh == s
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        block = f.read().split("<!-- RESULTS:START -->")[1].split("<!-- RESULTS:END -->")[0]
    assert "No row is published yet" in block and not any(m in block for m in s["agents"]["model"])   # pilot rows are absent from the README table


# ------------------------------------------------------------------ a run of the published design
@pytest.fixture(scope="module")
def published(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("published"))
    recs = synth.write(out)
    assert cli.main(["results", "--dir", out, "--quiet"]) == 0
    return out, recs


def test_published_run_passes_the_check(published, capsys):
    out, recs = published
    assert cli.main(["results", "--dir", out, "--check"]) == 0
    assert "(3000 published, 0 pilot); no reporting rule is broken" in capsys.readouterr().out
    with open(os.path.join(out, "results.json"), encoding="utf-8") as f:
        s = json.load(f)
    assert all(r["tier"] == "published" and r["instances"] == 300 for r in s["rows"]) and len(s["rows"]) == 8
    assert {r["seed_set_sha256"] for r in s["rows"]} == {synth.report_hash()} == set(s["plan"]["design"]["seed_set_sha256"].values())
    assert s["plan_sha256"] == summarize.file_sha256(out, "analysis_plan.json") and len(s["comparisons"]) == 3
    assert s["floors"][synth.ENV] == dict(reference="reference", no_scan="prior", must_fail=["prior", "fixed_recipe", "brute_force", "wrong_units"], ablations=["textbook"])
    model = next(r for r in s["rows"] if r["agent"] == "model-a")
    assert (model["episodes"], model["episodes_per_instance"], model["tokens_in_total"], model["wall_s_total"]) == (600, [2, 2], 540000, 1200.0)
    assert model["ci_method"] == "wilson-kg-t" and model["n_eff"] < 600 and model["icc"] > 0 and 0 < model["pass_all_k"] < model["pass_rate"]
    assert model["regret_median"] is not None and model["regret_negative_passing"] == 0
    with open(os.path.join(out, "README.md"), encoding="utf-8") as f:
        text = f.read()
    assert "(300 instances, levels 1, 2, 3 as strata; models play 2 episodes per instance" in text
    assert f"| model-a | model | {model['passed']}/600 ({100 * model['pass_rate']:.1f} %; " in text and "### Pilot rows" not in text
    assert "| NullTissue-v0:model-a-vs-model-b | model-a | model-b | 300 |" in text and "ICC (95 % interval)" in text


def test_missing_conversation_fails_the_check(published, capsys):
    out, recs = published
    path = os.path.join(out, next(r["_dir"] for r in recs if r["agent"] == "model-b"), "messages.json")
    os.rename(path, path + ".away")
    try:
        assert cli.main(["results", "--dir", out, "--check"]) == 1
        text = capsys.readouterr().out
        assert "R13" in text and "messages.json is missing" in text and "record.json:1:" in text
    finally:
        os.rename(path + ".away", path)
    assert cli.main(["results", "--dir", out, "--check"]) == 0


def edited(change, plan="default", **kw):
    """Rules broken by the synthetic run after ``change(records)``."""
    p = synth.plan_with_hash() if plan == "default" else plan
    recs = synth.records(plan_sha=None if p is None else p["sha256"], **kw)
    recs = change(recs) or recs
    return lint.lint_results(synth.summary(recs, p))


def test_the_synthetic_run_breaks_no_rule():
    assert edited(lambda recs: None) == []


def test_a_published_row_whose_records_lack_a_field_fails():
    def drop(recs):
        for field, seed in (("transcript", "report:1"), ("harness_id", "report:2"), ("outcome", "report:3"), ("command", "report:4"),
                            ("sampler", "report:5"), ("tokens_in", "report:6")):
            for r in recs:
                if r["agent"] == "model-a" and r["episode"] == 0 and r["seed"] == seed:
                    r[field] = None
    found = edited(drop)                                                   # three records of 600 per field are enough
    assert rules(found) == {"R13"} and len(found) == 1 and "model-a" in found[0]["message"]
    for field in ("transcript", "harness_id", "outcome", "command", "sampler", "tokens_in"):
        assert f"{field} (3)" in found[0]["message"]
    everywhere = edited(lambda recs: [r.update(command=None) for r in recs if r["agent"] == "model-a"][:0])
    assert rules(everywhere) == {"R5", "R13"}                              # no record of the row names its command
    assert rules(edited(lambda recs: [r.pop("deliverable") for r in recs if r["agent"] == "model-b"][:0])) == {"R13"}


def test_design_minimum_seed_list_and_settings():
    assert rules(edited(lambda recs: None, per_level=80)) == {"D1", "R4"}                 # 240 instances: short, and not the planned seed set
    assert rules(edited(lambda recs: None, episodes=1)) == {"D1"}                         # one episode per instance for a model
    other = edited(lambda recs: [r.update(seed=r["seed"].replace("report", "other")) for r in recs if r["agent"] == "model-b"][:0])
    assert rules(other) == {"R4", "R11"} and any("differs from the other rows" in v["message"] for v in other)     # and its comparisons cannot be computed

    def not_sent(recs):
        for r in recs:
            if r["agent"] == "model-a":
                r.update(params_sent={}, sampler=dict(requested=dict(temperature=None, max_tokens=None), sent={}, dropped=[]))
    found = edited(not_sent)
    assert rules(found) == {"R5"} and "temperature as sent" in found[0]["message"] and "max_tokens as sent" in found[0]["message"]

    def lost(recs):
        for r in recs[:40]:
            r.update(status="infra_error", stop_reason="endpoint_error")
    assert "R10" in rules(edited(lost))
    assert "R12" in rules(edited(lambda recs: [r.update(tokens_in=None, tokens_out=None) for r in recs if r["agent"] == "model-a"][:0]))


def test_floors_oracle_and_plan_are_required():
    assert rules(edited(lambda recs: [r for r in recs if r["agent"] != "reference"])) == {"R8"}
    assert rules(edited(lambda recs: [r for r in recs if r["agent"] != "textbook"])) == {"R8"}
    no_oracle = lint.lint_results(synth.summary(with_oracle=False))
    assert rules(no_oracle) == {"R8"} and "regret" in no_oracle[0]["message"]
    assert rules(edited(lambda recs: None, plan=None)) == {"R11"}                           # no plan
    late = dict(synth.plan_with_hash(), registered_utc="2026-12-03T00:00:00Z")              # the plan is younger than the records
    assert rules(edited(lambda recs: None, plan=late)) == {"R11"}
    assert rules(edited(lambda recs: [r.update(plan_sha256="0" * 64) for r in recs if r["agent"] == "model-a"][:0])) == {"R11"}
    extra = synth.plan_with_hash(models=["model-a", "model-b", "model-c"])                  # a planned model that was not run
    found = edited(lambda recs: None, plan=extra)
    assert rules(found) == {"R11"} and "not computed" in found[0]["message"] and "model-c" in found[0]["message"]
    zero = synth.summary()
    next(r for r in zero["rows"] if r["agent"] == "prior")["one_sided_bound"] = None
    assert rules(lint.lint_results(zero)) == {"R6"}


def test_pilot_rows_never_enter_a_comparison():
    plan = synth.plan_with_hash(models=["model-a", "model-b", "old-model"])
    recs = synth.records(plan_sha=plan["sha256"])
    old = [record.upgrade(dict({k: v for k, v in r.items() if k not in ("transcript", "harness_id", "deliverable", "tier", "record_schema")},
                               schema="mrgym-episode/1", agent="old-model", model="old-model")) for r in recs if r["agent"] == "model-a"]
    s = synth.summary(recs + old, plan)
    row = next(r for r in s["rows"] if r["agent"] == "old-model")
    assert row["tier"] == "pilot" and row["tier_reason"] == "schema 1" and row["instances"] == 300
    assert all("old-model" not in (c["a"], c["b"]) for c in s["comparisons"]) and len(s["comparisons"]) == 3
    assert all(c["agent"] != "old-model" for c in s["reliability"]["cells"])
    found = lint.lint_results(s)
    assert rules(found) == {"R11"} and "old-model" in found[0]["message"]            # the planned family is incomplete, and says so
    text = summarize.markdown(s)
    published, pilot = text.split("### Pilot rows (not published)")
    assert "old-model" not in published and "| old-model | model | 300 |" in pilot


def test_comparisons_are_what_a_few_lines_of_numpy_give():
    recs = synth.records()
    s = synth.summary(recs)
    means, level = {}, {}
    for r in recs:
        means.setdefault(r["agent"], {}).setdefault((r["level"], r["seed"]), []).append(float(r["passed"]))
        level[(r["level"], r["seed"])] = r["level"]
    keys = sorted(level)
    ps = []
    for c in s["comparisons"]:
        d = np.array([np.mean(means[c["a"]][k]) - np.mean(means[c["b"]][k]) for k in keys])
        lv = np.array([level[k] for k in keys])
        strata = sorted(set(lv))
        diff = np.mean([d[lv == h].mean() for h in strata])
        se = np.sqrt(sum(d[lv == h].var(ddof=1) / (lv == h).sum() for h in strata)) / len(strata)
        dof = len(d) - len(strata)
        half = st.t.ppf(0.975, dof) * se
        p = st.t.sf(diff / se, dof) if c["alternative"] == "greater" else 2 * st.t.sf(abs(diff / se), dof)
        ps.append(p)
        a, b = np.array([np.mean(means[c["a"]][k]) for k in keys]), np.array([np.mean(means[c["b"]][k]) for k in keys])
        assert c["instances"] == 300 and c["test"] == "paired-t"
        assert c["diff"] == pytest.approx(diff, abs=1e-9) and c["ci95"] == pytest.approx([diff - half, diff + half], abs=1e-9)
        assert c["se"] == pytest.approx(se, abs=1e-9) and c["p"] == pytest.approx(p, rel=1e-9, abs=1e-300)
        assert c["corr"] == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-9)
        assert (c["wins"], c["losses"], c["ties"]) == (int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum()))
    order = np.argsort(ps)
    holm, run_max = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, (len(ps) - rank) * ps[i])
        holm[i] = min(1.0, run_max)
    assert [c["p_holm"] for c in s["comparisons"]] == pytest.approx(holm.tolist(), rel=1e-9, abs=1e-300)
    assert [c["id"] for c in s["comparisons"]] == s["plan"]["comparisons"]
    assert [c["alternative"] for c in s["comparisons"]] == ["two-sided", "greater", "greater"]


def test_cost_needs_a_dated_price_list():
    recs = synth.records()
    plan = synth.plan_with_hash()
    assert all(r["cost_usd"] is None for r in summarize.summarise(recs, plan)["rows"])
    prices = dict(prices=[dict(model="model-a", date="2026-12-01", usd_per_million_in=3.0, usd_per_million_out=15.0, source="https://example.invalid/pricing")])
    rows = {r["agent"]: r for r in summarize.summarise(recs, plan, prices)["rows"]}
    assert rows["model-a"]["cost_usd"] == pytest.approx((540000 * 3.0 + 24000 * 15.0) / 1e6) and rows["model-a"]["price"]["date"] == "2026-12-01"
    assert rows["model-b"]["cost_usd"] is None and rows["reference"]["cost_usd"] is None


def test_oracle_values_are_stored_per_instance_and_give_the_regret_column(tmp_path, monkeypatch, capsys):
    from mrgym.envs import oracle as O
    out = str(tmp_path)
    synth.write(out, synth.records(per_level=1, models={"model-a": 0.5}), with_oracle=False)
    seen = []

    def solve(cfg, world=None):
        seen.append((cfg["level"], cfg["task_id"], world is not None))
        return dict(value=0.002, argmin={}, method="exhaustive")

    fake = type("Oracle", (), dict(solve=staticmethod(solve), REGRET_KIND="difference"))
    monkeypatch.setattr(O, "load", lambda env: fake)
    assert cli.main(["results", "--dir", out, "--oracle"]) == 0 and "oracle.json" in capsys.readouterr().out
    stored = summarize.read_json(out, summarize.ORACLE_FILE)[synth.ENV]
    assert stored["instances"] == {"L1:report:0": 0.002, "L2:report:0": 0.002, "L3:report:0": 0.002} and stored["kind"] == "difference"
    assert sorted(lv for lv, _, _ in seen) == [1, 2, 3] and all(w for _, _, w in seen) and len({t for _, t, _ in seen}) == 3
    results.write_oracle(out)
    assert len(seen) == 3                                                    # instances already stored are not searched again
    row = next(r for r in results._summary(out)["rows"] if r["agent"] == "reference")
    assert row["regret_median"] is not None and row["regret_median"] == pytest.approx(row["error_median"] - 0.002)


def test_plan_is_registered_once_and_before_the_run(published, tmp_path, capsys):
    out, _ = published
    assert cli.main(["results", "--dir", out, "--register-plan", "--models", "model-a"]) == 1          # the run has records: too late
    assert "registered before the run, once" in capsys.readouterr().out
    fresh = str(tmp_path / "new")
    assert cli.main(["results", "--dir", fresh, "--register-plan", "--models", "m1,m2", "--env", synth.ENV]) == 0
    plan = summarize.load_plan(fresh)
    assert plan["schema"] == "mrgym-analysis-plan/1" and plan["correction"] == "holm" and plan["alpha"] == 0.05
    assert plan["design"]["instances"] == {synth.ENV: 300} and plan["design"]["episodes"] == 2 and plan["design"]["suite"] == "report"
    assert [(c["a"], c["b"], c["alternative"]) for c in plan["comparisons"]] == [("m1", "m2", "two-sided"), ("m1", "fixed_recipe", "greater"), ("m2", "fixed_recipe", "greater")]
    assert plan["mde_points"][synth.ENV] == round(100 * power.mde_paired(300, 2, comparisons=3), 1) == 9.7 and plan["stop_rules"]["looks"] == 1
    assert {"registered_utc", "primary_endpoint", "exclusions", "stop_rules", "sha256"} <= set(plan)


def test_stat_spans_are_filled_from_results():
    s = synth.summary()
    text = ("The reference passes <!-- STAT:row:Scanner/NullTissue-v0:reference -->typed<!-- /STAT --> on "
            "<!-- STAT:row:Scanner/NullTissue-v0:reference:instances -->0<!-- /STAT --> instances; model-a against model-b: "
            "<!-- STAT:comparison:NullTissue-v0:model-a-vs-model-b -->x<!-- /STAT -->; <!-- STAT:published_records -->0<!-- /STAT --> records.")
    out = results.fill_stat_spans(text, s)
    ref = next(r for r in s["rows"] if r["agent"] == "reference")
    assert f"-->{summarize.cell_text(ref)}<!--" in out and "-->300<!--" in out and "-->3000<!--" in out and "points (" in out and "typed" not in out
    assert results.fill_stat_spans(out, s) == out and lint.lint_text(out, s) == []
    with pytest.raises(KeyError, match="no such row"):
        results.fill_stat_spans("<!-- STAT:row:Scanner/NullTissue-v0:nobody -->1<!-- /STAT -->", s)


# ------------------------------------------------------------------ manifests
def test_stored_manifests_are_frozen():
    directory = os.path.join(ROOT, "results", "manifests")
    sizes = dict(report=300, validation=200, smoke=12)
    sizes["public-test"] = 100
    seen = {}
    for suite, per_env in sizes.items():
        assert manifest.drift(suite, directory) is None, suite
        m = manifest.load(suite, directory)
        assert m["schema"] == "mrgym-manifest/1" and m["canary"] == canary.CANARY and m["instances"] == per_env * len(m["environments"]) == len(m["rows"])
        keys = [(r["env_id"], r["level"], r["split"], r["seed"]) for r in m["rows"]]
        assert len(set(keys)) == len(keys) and all(r["seed"].startswith(m["seed_prefix"] + ":") for r in m["rows"])
        seen[suite] = set(keys)
        for env_id, e in m["environments"].items():
            part = [r for r in m["rows"] if r["env_id"] == env_id]
            assert e["seed_set_sha256"] == summarize.seed_set_sha256(env_id, {(r["level"], r["split"], r["seed"]) for r in part})
            assert max(e["levels"].values()) - min(e["levels"].values()) <= 1                 # equal allocation over the levels
    assert not seen["report"] & seen["validation"] and not seen["report"] & seen["smoke"]
    report = manifest.load("report", directory)
    assert all(r["task_id"] and r["split"] == "train" for r in report["rows"]) and len({r["task_id"] for r in report["rows"]}) == 300
    assert not any(str(r["seed"]).startswith(("gate:", "tune:")) for r in report["rows"])
    secret = manifest.load("public-test", directory)
    assert secret["task_ids"].startswith("omitted") and all(r["task_id"] is None and r["split"] == "test-iid" for r in secret["rows"])


def test_manifest_builds_are_deterministic_and_keyed(capsys):
    a, b = manifest.build("smoke", [synth.ENV]), manifest.build("smoke", [synth.ENV])
    assert manifest.text(a) == manifest.text(b) and [r["level"] for r in a["rows"]] == [1] * 4 + [2] * 4 + [3] * 4
    key = {"MRGYM_SECRET": "5a" * 32}
    keyed = manifest.build("public-test", [synth.ENV], environ=key)
    assert keyed["task_ids"] == "listed" and len({r["task_id"] for r in keyed["rows"]}) == 100
    assert {r["task_id"] for r in keyed["rows"]} != {r["task_id"] for r in manifest.build("public-test", [synth.ENV], environ={"MRGYM_SECRET": "6b" * 32})["rows"]}
    assert synth.report_hash() == manifest.build("report", [synth.ENV])["environments"][synth.ENV]["seed_set_sha256"]
    assert cli.main(["manifest", "report", "--env", synth.ENV]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 300
    assert cli.main(["manifest", "list"]) == 0 and "public-test" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["manifest", "nonsense"])


def test_seed_text_and_hash_helpers():
    assert summarize.seeds_text([(1, 0), (1, 1), (1, 2), (2, 5), (2, 7)]) == "L1: 0-2; L2: 5,7"
    assert summarize.seeds_text([(1, "report:0"), (1, "report:1"), (1, "report:3"), (2, "odd")]) == "L1: report:0-1,3; L2: odd"
    a = summarize.seed_set_sha256("E", {(1, "train", 0), (1, "train", 1)})
    assert a == summarize.seed_set_sha256("E", [(1, "train", 1), (1, "train", 0)]) != summarize.seed_set_sha256("E", {(1, "train", 0)})
    assert copy.deepcopy(synth.summary()) == synth.summary()                 # a pure function of the records
