"""The episode recorder and the one-file viewer."""
import json
import re

import numpy as np
import pytest

import mrgym
from mrgym import cli, policies, record, viewer
from mrgym.env import derive_seed

ENV_ID, LEVEL, SEED = "Scanner/NoWrap-v0", 2, 0
HEX32 = re.compile(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])")


@pytest.fixture(scope="module")
def bundle():
    return record.record(ENV_ID, policy="reference", level=LEVEL, seed=SEED)


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def test_bundle_layout(bundle):
    assert bundle["format"] == "mrgym-episode-bundle/1"
    assert set(bundle) == {"format", "package", "env", "episode", "agent", "task", "steps", "scans", "scanner_log",
                           "result", "agent_events", "hidden"}
    assert bundle["package"] == dict(name="mrgym", version=mrgym.__version__, engine=mrgym.ENGINE_ID)
    assert bundle["env"]["id"] == ENV_ID and bundle["env"]["level"] == LEVEL
    assert bundle["agent"]["kind"] == "scripted" and bundle["agent"]["name"] == "reference"
    assert bundle["agent_events"] is None
    steps = bundle["steps"]
    assert steps[0]["call"] is None and steps[0]["text"].startswith("# Prescribe")
    assert [s["step"] for s in steps] == list(range(len(steps)))
    assert [s["call"]["name"] for s in steps[1:]] == ["scanner_info", "scan", "view", "submit"]
    for s in steps:
        assert {"call", "text", "data", "is_error", "status", "reward"} <= set(s)
        assert s["status"]["scan_time_budget_s"] == bundle["task"]["budget_s"]
    r = bundle["result"]
    assert r["reward"] == 1.0 and r["terminated"] and not r["truncated"] and r["submitted"]
    assert r["score"]["passed"] and r["replay"]["ok"] and r["answer"] == steps[-1]["call"]["arguments"]
    assert r["delivered"]["fov_yx_mm"] and r["delivered"]["header"].startswith("SE ")
    assert len(bundle["scanner_log"]) == r["replay"]["events"] == 1
    assert bundle["scanner_log"][0]["type"] == "scan"


def test_bundle_holds_no_seed(bundle):
    text = json.dumps(bundle)
    seed_hex = derive_seed(ENV_ID, LEVEL, "train", SEED)
    assert seed_hex not in text
    assert "seed" not in set(_keys(bundle))
    # a 32-digit run of decimal digits is a row of a digit map; anything with a hex letter would be a seed-like token
    assert [t for t in HEX32.findall(text) if not t.isdigit()] == []


def test_bundle_is_strict_json(bundle, tmp_path):
    path = record.save(bundle, str(tmp_path / "episode_bundle.json"))
    def no_constant(name):
        raise AssertionError(f"{name} is not JSON")

    with open(path, encoding="utf-8") as f:
        assert json.load(f, parse_constant=no_constant) == json.loads(json.dumps(bundle))
    assert record.load(path) == json.loads(json.dumps(bundle))
    assert record.load(str(tmp_path)) == record.load(path)
    (tmp_path / "other.json").write_text('{"format": "something-else/1"}')
    with pytest.raises(mrgym.Error):
        record.load(str(tmp_path / "other.json"))


def test_images_decode_to_what_the_scanner_stored():
    env = mrgym.make(ENV_ID, level=LEVEL, seed=SEED, reveal_score=True)
    result = policies.run(env, "reference")
    b = record.bundle_from_env(env, result, {"kind": "scripted", "name": "reference"})
    assert len(b["scans"]) == len(env._scanner.scans) == 1
    for s in b["scans"]:
        stored = env._scanner.scans[s["index"]]
        mag, phase = record.scan_arrays(b, s["index"])
        assert list(mag.shape) == s["shape"] == list(stored["magnitude"].shape) and mag.dtype == np.float32
        assert np.array_equal(mag, stored["magnitude"]) and np.array_equal(phase, stored["phase"])
        assert s["fov_yx_mm"] == list(stored["fov_yx_mm"]) and s["step"] == 2
    env.close()
    with pytest.raises(mrgym.Error):
        record.bundle_from_env(env, result, {"kind": "scripted", "name": "reference"})


def test_hidden_block(bundle):
    h = bundle["hidden"]
    assert {"note", "maps", "classes", "system", "object_extent_mm", "regions", "rescan"} <= set(h)
    n = h["maps"]["shape"]
    assert n[0] == n[1] <= 160
    for name in ("pd", "t1_ms", "t2_ms", "b1", "b0_hz"):
        a = record.hidden_map(bundle, name)
        assert a.shape == tuple(n) and np.isfinite(a).sum() > 100
    pd, t1 = record.hidden_map(bundle, "pd"), record.hidden_map(bundle, "t1_ms")
    assert np.isfinite(pd).all() and np.isnan(t1).any()                 # T1 is undefined outside the object
    ext = h["object_extent_mm"]
    assert ext["y"][0] < 0 < ext["y"][1] and ext["x"][0] < 0 < ext["x"][1]
    fy, fx = bundle["result"]["delivered"]["fov_yx_mm"]
    assert fy / 2 >= max(map(abs, ext["y"])) - 0.5 and fx / 2 >= max(map(abs, ext["x"])) - 0.5   # the reference covers it
    rescan = record.decode_f32(h["rescan"]["magnitude_f32_b64"], h["rescan"]["shape"])
    assert rescan.shape[0] == 1 and rescan.max() > 0
    # nothing hidden outside the hidden block and the verifier's result
    public = {k: v for k, v in bundle.items() if k not in ("hidden", "result")}
    assert "tx_gain" not in json.dumps(public)


def test_failing_policy_on_the_same_world(bundle):
    other = record.record(ENV_ID, policy="size_only", level=LEVEL, seed=SEED)
    assert other["episode"]["task_id"] == bundle["episode"]["task_id"]
    assert other["hidden"]["maps"]["pd_f32_b64"] == bundle["hidden"]["maps"]["pd_f32_b64"]
    assert other["result"]["reward"] == 0.0 and not other["result"]["score"]["passed"]


def test_agent_events_and_truncation():
    env = mrgym.make(ENV_ID, level=1, seed=3, max_steps=2)
    env.reset()
    env.step({"name": "scanner_info", "arguments": {}})
    last = env.step({"name": "plan", "arguments": {"protocol": {"sequence": "se", "tr_ms": 500}}})
    assert last.truncated
    events = [{"step": 1, "role": "assistant", "content": "I read the instrument card first."},
              {"step": 2, "role": "assistant", "content": "", "tool_calls": [{"name": "plan"}]}]
    b = record.bundle_from_env(env, last, {"kind": "model", "name": "some-model"}, agent_events=events)
    assert b["agent_events"] == events and b["agent"]["kind"] == "model"
    assert b["result"]["truncated"] and not b["result"]["terminated"] and b["result"]["reward"] == 0.0
    assert b["result"]["answer"] is None and b["scans"] == [] and "rescan" not in b["hidden"]
    html = viewer.render([b])
    assert "I read the instrument card first." in html


def test_record_refuses_unknown_policy_and_sandbox():
    with pytest.raises(mrgym.Error):
        record.record(ENV_ID, policy="no_such_policy")
    with pytest.raises(mrgym.Error):
        record.record("Scanner/Explore-v0")


def test_viewer_is_one_self_contained_file(bundle, tmp_path):
    out = viewer.build([bundle, bundle], str(tmp_path / "sub" / "viewer.html"))
    assert [p.name for p in (tmp_path / "sub").iterdir()] == ["viewer.html"]
    html = open(out, encoding="utf-8").read()
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert "http://" not in html and "https://" not in html
    assert "src=" not in html and "<link" not in html and "@import" not in html and "url(" not in html
    m = re.search(r'<script id="episodes" type="application/json">(.*?)</script>', html, re.S)
    embedded = json.loads(m.group(1))
    assert embedded == json.loads(json.dumps([bundle, bundle]))
    assert "<" not in m.group(1)
    assert viewer.SLOT not in html
    with pytest.raises(mrgym.Error):
        viewer.build([{"not": "a bundle"}], str(tmp_path / "bad.html"))


def test_cli_record_and_view(tmp_path, capsys):
    out = tmp_path / "episode"
    assert cli.main(["record", ENV_ID, "--policy", "reference", "--level", str(LEVEL), "--seed", str(SEED), "--out", str(out)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    path = out / "episode_bundle.json"
    assert path.is_file() and lines[0] == str(path)
    assert len(lines) == 2 and "reward 1.000" in lines[1] and ENV_ID in lines[1]
    html = tmp_path / "viewer.html"
    assert cli.main(["view", str(path), str(out), "--out", str(html)]) == 0
    assert html.is_file() and str(html) in capsys.readouterr().out
    text = html.read_text(encoding="utf-8")
    assert "http://" not in text and "https://" not in text
    assert text.count(record.load(str(path))["episode"]["id"]) == 2
