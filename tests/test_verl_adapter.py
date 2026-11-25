"""The verl integration (``integrations/verl``) without verl, a GPU or a model: everything runs against stand-in
endpoints on loopback.

What is shown
  * ``mrgym_verl.episode.Episode`` is ``mrgym.agents.runner.play_model`` cut at the model call: the same conversation
    and the same result for scripted behaviours (a pass, the no-scan answer, malformed turns, prose, call floods)
  * the reward handed to a trainer is the environment's paid reward (1 for a pass, else 0); ``signal="progress"`` is
    labelled as such and leaves the paid reward untouched; there is no turn cap
  * every episode is stored as a record of the runner's schema with the whole conversation, the per-step ``info`` and
    one outcome of the environment's taxonomy; a reply cut by the token limit ends as ``no_submission_length``
  * instance rows: training rows are placeholders served by the curriculum hook (one instance per group, a seed never
    issued twice, also after the hook is rebuilt from its state file); dev rows are another split; test rows are the
    secret split and cannot be played without a secret
  * the stand-in trainer (``mrgym_verl.sim``) trains, checkpoints, is killed, restores and continues; the stop rules
    fire on a run without reward and stay quiet on a run that learns; the readout pairs base and trained
  * the three node entry scripts share one bootstrap block, name no bucket, and ``calibrate.sh`` resumes after a kill
    (the full simulations with their outputs are in integrations/verl/SIMULATION.md)

The parts that import verl (``agent_loop``, ``checkpoint``, ``dryrun``) are exercised by ``test_verl_adapter_verl.py``,
which is skipped where verl is not installed.
"""
import glob
import json
import os
import re
import signal
import subprocess
import sys

import pytest

import mrgym
from mrgym import record as mrecord
from mrgym.agents import chat, runner, stub_server
from mrgym.learn import hooks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERL = os.path.join(ROOT, "integrations", "verl")
sys.path.insert(0, VERL)

from mrgym_verl import calsum, instances, mixstub, readout, smokecheck, stoprules, store  # noqa: E402
from mrgym_verl.episode import Episode, episode_dir, play_chat  # noqa: E402

NOWRAP = "Scanner/NoWrap-v1"
FAST = dict(backoff_s=0.01, max_retries=2)


@pytest.fixture(scope="module")
def stub():
    httpd, url = stub_server.start(seeds=[0, 1], env_ids=[NOWRAP], levels=[2])
    yield url
    httpd.shutdown()


@pytest.fixture(scope="module")
def mix():
    httpd, url = mixstub.start(0, [(NOWRAP, 2)], {"train": list(range(10000, 10006)), "dev": [0]})
    yield url
    httpd.shutdown()


def play(url, model, seed=0, **kw):
    ep = Episode(NOWRAP, 2, seed, max_reply_tokens=4096, **kw)
    result = play_chat(ep, chat.ChatClient(url, model, None, dict(max_tokens=4096), **FAST))
    return ep, result


# ------------------------------------------------------------------ the episode is the runner's
@pytest.mark.parametrize("kind", ["reference", "prior", "recover", "text_only", "malformed_json", "flood"])
def test_episode_equals_runner(stub, tmp_path, kind):
    model = "stub:" + kind
    runner.run(str(tmp_path / "r"), [NOWRAP], levels=[2], seeds=[0], models=[model], base_url=stub, max_tokens=4096, quiet=True, **FAST)
    d = episode_dir(str(tmp_path / "r"), model, NOWRAP, 2, 0, 0)
    with open(os.path.join(d, "record.json")) as f:
        ref = json.load(f)
    with open(os.path.join(d, mrecord.TRANSCRIPT_FILE)) as f:
        ref_doc = json.load(f)
    ep, result = play(stub, model)
    rec = ep.write_record(episode_dir(str(tmp_path / "a"), model, NOWRAP, 2, 0, 0), agent=model, model=model, sampler=dict(max_tokens=4096),
                          base_url=stub, training=dict(phase="train", step=1, uid="g", session=0))
    final = ep.env.training_info()
    ep.close()
    strip = lambda ms: [{k: v for k, v in m.items() if k not in ("turn", "reasoning")} for m in ms]   # noqa: E731
    assert json.loads(json.dumps(ep.messages)) == strip(ref_doc["messages"]) and json.loads(json.dumps(ep.tools)) == ref_doc["tools"]
    for key in ("reward", "passed", "stop_reason", "outcome", "ladder", "turns", "tool_calls", "invalid_calls", "by_tool", "interface", "scans",
                "scan_s_used", "deliverable", "checks", "max_steps", "max_turns", "task_id", "split", "tokens_in", "tokens_out"):
        assert rec.get(key) == ref.get(key), key
    assert rec["log"]["head"] == ref["log"]["head"]
    assert rec["harness"]["loop"] == "mrgym_verl.episode.Episode" and rec["harness_id"] != ref["harness_id"]
    assert mrecord.missing(rec) == [] and mrecord.transcript_problem(rec, episode_dir(str(tmp_path / "a"), model, NOWRAP, 2, 0, 0)) is None
    # the trainer's block: paid reward, the environment's own outcome, the per-step info
    tr = rec["training"]
    assert tr["signal_mode"] == "pass" and tr["signal"] == tr["paid_reward"] == result["reward"] == (1.0 if result["passed"] else 0.0)
    assert tr["outcome"] == final["outcome"] and tr["outcome"] in ep.env.outcomes()
    assert all(tuple(s) == hooks.STEP_KEYS for s in tr["steps"]) and len(tr["steps"]) == rec["interface"]["calls"]
    assert sum(s["counted"] for s in tr["steps"]) == rec["env_tool_calls"]
    if tr["steps"]:
        assert hooks.validate(dict(final, step=tr["steps"][-1]), ep.env.outcomes()) == []


def test_reward_is_the_pass_and_progress_is_labelled(stub):
    ep, passed = play(stub, "stub:reference")
    ep.close()
    assert passed["passed"] and passed["reward"] == passed["signal"] == 1.0 and passed["outcome"] == "pass"
    ep, failed = play(stub, "stub:overspend")
    ep.close()
    assert not failed["passed"] and failed["reward"] == failed["signal"] == 0.0 and failed["submitted"]
    ep, prog = play(stub, "stub:overspend", signal="progress")
    ep.close()
    assert prog["signal_mode"] == "progress" and prog["reward"] == 0.0 and prog["signal"] == prog["progress"] and 0.0 <= prog["signal"] < 1.0
    with pytest.raises(ValueError):
        Episode(NOWRAP, 2, 0, signal="env_plus_pass")


def test_no_turn_cap_only_the_environments_limit(stub):
    ep = Episode(NOWRAP, 2, 0)
    ep.start()
    assert ep.env.max_turns is None and ep.env.max_steps == mrgym.spec(NOWRAP).max_steps
    assert ep.reset_info["max_turns"] is None and "max_turns" not in ep.reset_info["limits"]
    ep.close()
    ep, result = play(stub, "stub:flood")                                # 30 calls per reply: only the environment's limit ends it
    ep.close()
    assert result["outcome"] == "no_submission_cap" and result["truncated_by"] == "max_steps" and result["counts"]["calls"] >= mrgym.spec(NOWRAP).max_steps
    ep = Episode(NOWRAP, 2, 0, tool_call_limit=6)                        # an explicit limit is the environment's own, stated in the task text
    messages, _ = ep.start()
    assert ep.env.max_steps == 6 and "6" in json.dumps(ep.reset_info["limits"])
    ep.close()


def test_cut_reply_ends_with_one_outcome(tmp_path):
    ep = Episode(NOWRAP, 2, 0, max_reply_tokens=64)
    messages, _ = ep.start()
    assert "64" in messages[1]["content"]                               # the reply limit is stated in the task text
    ep.abort("length", reply=dict(raw_text="Let me think", prompt_tokens=5000, completion_tokens=64))
    result = ep.result()
    assert ep.done and result["outcome"] == "no_submission_length" and result["reward"] == 0.0 and result["cut_replies"] == 1
    assert result["turns"] == 0 and result["tokens_out"] == 64
    rec = ep.write_record(episode_dir(str(tmp_path), "t", NOWRAP, 2, 0, 0), agent="t", model="m", base_url="inproc://trainer",
                          training=dict(phase="train", step=1, uid="g", session=0))
    ep.close()
    assert rec["training"]["aborted"] == "length" and rec["training"]["cut_reply"]["raw_text"] == "Let me think"
    assert rec["training"]["outcome"] == "no_submission_length" and mrecord.missing(rec) == []
    with pytest.raises(RuntimeError):
        ep.assistant_turn("more", [])


def test_world_seed_never_in_a_record(stub, tmp_path):
    ep, _ = play(stub, "stub:reference", seed=1)
    seed_hex = runner.world_seed(ep.env)
    ep.write_record(str(tmp_path / "e"), agent="t", model="stub:reference", training=dict(phase="train", step=1, uid="g", session=0))
    ep.close()
    for path in glob.glob(str(tmp_path / "e" / "*")) + glob.glob(str(tmp_path / "run_config.json")):
        with open(path) as f:
            assert seed_hex not in f.read(), path


# ------------------------------------------------------------------ instances and the curriculum hook
def test_rows_and_splits(tmp_path, monkeypatch):
    out = str(tmp_path / "data")
    assert instances.main(["--out", out, "--arms", f"{NOWRAP}:2,3", "--train-rows", "40", "--dev-seeds", "0-3", "--test",
                           f"{NOWRAP}:2 Scanner/SNRSpec-v0:1", "--test-seeds", "0-4", "--format", "jsonl"]) == 0
    train, dev, test = (instances.read_rows(os.path.join(out, f"{n}.jsonl")) for n in ("train", "dev", "test"))
    assert len(train) == 40 and all(r["extra_info"]["curriculum"] and "seed" not in r["extra_info"] for r in train)
    assert {r["extra_info"]["split"] for r in dev} == {"dev"} and len(dev) == 8
    assert {r["extra_info"]["split"] for r in test} == {"test-iid"} and len(test) == 10
    assert {r["agent_name"] for r in train + dev + test} == {"mrgym_agent"}
    assert all("Scanner" not in json.dumps(r["prompt"]) for r in train + dev + test)       # a row is an index, never task content
    # dev worlds are not training worlds; the secret split refuses to run without a secret
    a, b = Episode(NOWRAP, 2, 0, "train"), Episode(NOWRAP, 2, 0, "dev")
    assert a.start()[0][1]["content"] != b.start()[0][1]["content"] or runner.world_seed(a.env) != runner.world_seed(b.env)
    assert runner.world_seed(a.env) != runner.world_seed(b.env)
    a.close()
    b.close()
    monkeypatch.delenv("MRGYM_SECRET_FILE", raising=False)
    monkeypatch.delenv("MRGYM_SECRET", raising=False)
    with pytest.raises(mrgym.Error):
        Episode(NOWRAP, 2, 0, "test-iid").start()
    key = tmp_path / "key"
    key.write_text("ab" * 32 + "\n")
    key.chmod(0o600)
    monkeypatch.setenv("MRGYM_SECRET_FILE", str(key))
    ep = Episode(NOWRAP, 2, 0, "test-iid")
    ep.start()
    assert runner.world_seed(ep.env) not in (runner.world_seed(a.env), runner.world_seed(b.env))
    ep.close()
    with pytest.raises(ValueError):
        instances.parse_arms(f"{NOWRAP}:9")
    with pytest.raises(ValueError):
        instances.parse_arms(f"{NOWRAP}:2 {NOWRAP}:2")
    assert instances.arms_text(instances.parse_arms(f"{NOWRAP}:2,3 Scanner/SNRSpec-v0:1")) == f"{NOWRAP}:2,3 Scanner/SNRSpec-v0:1"


def test_hook_groups_seeds_and_restart(tmp_path):
    import asyncio
    cfg = str(tmp_path / "curriculum.json")
    state = str(tmp_path / "ckpt" / "curriculum_state.json")
    instances.write_config(cfg, [(NOWRAP, 2), (NOWRAP, 3)], group_size=2, groups_per_step=2, state_path=state)
    hook = instances.Hook(json.load(open(cfg)))
    a1, a2, b = hook.assign("A"), hook.assign("A"), hook.assign("B")
    assert a1 == a2 and a1[2] != b[2] and a1[3] == 1 and a1[0] == NOWRAP and a1[1] in (2, 3)
    for uid, rewards in (("A", (1.0, 0.0)), ("B", (0.0, 0.0))):
        for r in rewards:
            hook.report(uid, r, bool(r))
    assert hook.info()["steps"] == 1 and hook.assign("C")[3] == 2          # two groups closed the step
    hook.report("unknown-group", 1.0, True)                               # a group of before a restart: ignored
    issued = hook.info()["issued"]
    again = instances.Hook(json.load(open(cfg)))                          # the process died; the state file survives
    assert again.restored == dict(steps=1, issued=issued, dropped=1)
    seen = {a1[2], b[2], hook.assign("C")[2]}
    fresh = again.assign("D")
    assert fresh[2] not in seen and again.info()["issued"] == issued + 1
    assert instances.snapshot_state(str(tmp_path), cfg) and os.path.exists(tmp_path / "curriculum_state.json")
    # the process-local client (outside Ray) is one object per configuration file
    c1, c2 = instances.client(cfg), instances.client(cfg)
    assert not c1.remote and asyncio.run(c1.assign("E")) == asyncio.run(c2.assign("E"))


def test_curriculum_drop_pending():
    from mrgym.curriculum import Curriculum
    cur = Curriculum([(NOWRAP, 2)])
    rows = cur.sample(3)
    cur.update(rows[0]["group_id"], [1.0, 0.0])
    assert cur.drop_pending() == 3 and cur.drop_pending() == 0
    assert cur.sample(1)[0]["seed"] == rows[-1]["seed"] + 1                # seeds stay issued
    with pytest.raises(KeyError):
        cur.update(rows[1]["group_id"], [1.0])


# ------------------------------------------------------------------ store, stop rules, readout
def test_store_directory_and_checkpoints(tmp_path):
    src, dest = tmp_path / "ckpt" / "global_step_4", str(tmp_path / "remote")
    (src / "actor").mkdir(parents=True)
    (src / "actor" / "adapter.bin").write_bytes(b"x" * 10)
    (src / "half.tmp").write_text("never uploaded")
    assert store.latest_checkpoint(dest) is None and store.restore_checkpoint(dest, str(tmp_path / "fresh")) is None
    latest = store.publish_checkpoint(str(src), dest, 4)
    assert latest["global_step"] == 4 and latest["files"] == 1 and store.Store(dest).ls() == ["LATEST", "global_step_4/actor/adapter.bin"]
    assert store.restore_checkpoint(dest, str(tmp_path / "fresh")) == 4
    assert (tmp_path / "fresh" / "global_step_4" / "actor" / "adapter.bin").read_bytes() == b"x" * 10
    assert (tmp_path / "fresh" / "latest_checkpointed_iteration.txt").read_text() == "4"
    seen = {}
    assert store.up(store.Store(dest), str(src), "again", seen) == 1 and store.up(store.Store(dest), str(src), "again", seen) == 0
    assert store.main(["exists", dest + "/LATEST"]) == 0 and store.main(["exists", dest + "/NOPE"]) == 1


def _rows(step, passes, phase="train", group=4, attempt="a", submitted=True, aborted=None, seed0=0):
    out = []
    for i, p in enumerate(passes):
        out.append(dict(phase=phase, step=step, attempt=attempt, uid=f"{phase}{step}g{i // group}", session=i % group, env_id=NOWRAP, level=2,
                        seed=seed0 + i // (group if phase == "train" else 1), split="train" if phase == "train" else "dev", passed=bool(p),
                        signal=float(p), submitted=submitted, aborted=aborted, cut_replies=int(aborted == "length"), interface_tax=0.0,
                        tokens_out=100, context_tokens=7000, step_source="trainer", tool_calls=4))
    return out


def test_stop_rules():
    cfg = dict(episodes_per_step=8, group_size=4, no_reward_steps=3, harness_steps=4, window=2, collapse_steps=3, base_rate=0.5, looks=4,
               futility_from_step=4, min_look_instances=8)
    quiet = sum((_rows(s, [1, 0, 0, 1, 0, 1, 1, 0]) for s in range(1, 6)), [])
    assert not stoprules.decide(quiet, cfg)["stop"] and len(stoprules.decide(quiet, cfg)["steps"]) == 5
    dead = sum((_rows(s, [0] * 8) for s in range(1, 4)), [])
    d = stoprules.decide(dead, cfg)
    assert d["stop"] and d["rule"] == "no_reward" and "no reward" in d["reason"]
    assert not stoprules.decide(dead[:16] + dead[16:20], cfg)["stop"]      # the third step is still being played
    low = sum((_rows(s, [1, 0, 0, 0, 0, 0, 0, 0]) for s in range(1, 5)), [])
    assert stoprules.decide(low, cfg)["rule"] == "harness"                 # 12.5 % against a calibrated 50 %
    flat = sum((_rows(s, [1, 1, 1, 1, 0, 0, 0, 0]) for s in range(1, 5)), [])
    assert stoprules.decide(flat, cfg)["rule"] == "harness"                # every group all-pass or all-fail
    silent = quiet + sum((_rows(s, [1, 0, 0, 1, 0, 1, 1, 0], submitted=False) for s in (6, 7)), [])
    assert stoprules.decide(silent, cfg)["rule"] == "not_delivering"
    cut = quiet + sum((_rows(s, [1, 0, 0, 1, 0, 1, 1, 0], aborted="length") for s in (6, 7)), [])
    assert stoprules.decide(cut, cfg)["rule"] == "cut_replies"
    fall = quiet + sum((_rows(s, [1, 0, 0, 0, 0, 0, 0, 0]) for s in (6, 7, 8)), [])
    assert stoprules.decide(fall, cfg)["rule"] == "collapse"
    # a replayed step after a preemption: the newest attempt counts, once
    replay = quiet + _rows(5, [1, 1, 0, 1, 0, 1, 1, 0], attempt="b")
    assert [s["episodes"] for s in stoprules.decide(replay, cfg)["steps"]] == [8] * 5
    # sequential test on the dev instances: no gain by the middle of the run is futility; a gain is recorded as efficacy
    base = _rows(0, [1, 0] * 8, phase="dev")
    same = quiet + base + _rows(4, [1, 0] * 8, phase="dev")
    d = stoprules.decide(same, cfg)
    assert d["rule"] == "futility" and d["dev"][-1]["look"]["instances"] == 16
    better = quiet + base + _rows(4, [1, 1] * 8, phase="dev")
    d = stoprules.decide(better, cfg)
    assert not d["stop"] and d["dev"][-1]["look"]["efficacy"] and d["dev"][-1]["look"]["diff"] == 0.5
    done = better + _rows(5, [1] * 16, phase="dev") + _rows(6, [1] * 16, phase="dev")
    assert stoprules.decide(done, cfg)["rule"] == "mastered"
    assert "decision: STOP no_reward" in stoprules.text_table(stoprules.decide(dead, cfg))


def test_smokecheck_names_the_cause():
    ok_rows = _rows(1, [1, 0, 0, 1, 0, 1, 1, 0])
    log = "step:1 - critic/score/mean:0.5 - actor/grad_norm:0.1 - timing_s/step:300.5\n"
    assert smokecheck.check(log, ok_rows, 1)[0]
    assert smokecheck.check(log, ok_rows, 2)[1].startswith("the log has no step 2")
    assert smokecheck.check("torch.OutOfMemoryError: CUDA out of memory", [], 1)[1].startswith("OOM on load")
    assert smokecheck.check("CUDA out of memory", ok_rows, 1)[1].startswith("OOM on the first backward")
    assert smokecheck.check(log + "CUDA out of memory", ok_rows, 2)[1].startswith("OOM after a completed update")
    mute = [dict(r, tool_calls=0) for r in ok_rows]
    assert smokecheck.check(log, mute, 1)[1].startswith("tool-call parser mismatch")
    over = [dict(r, aborted="length", tool_calls=0) for r in ok_rows[:2]] + ok_rows[2:]
    assert smokecheck.check(log, over, 1)[1].startswith("context overflow")
    assert smokecheck.check(log, [dict(r, aborted="length", tool_calls=0) for r in ok_rows], 1)[1].startswith("context overflow")   # cut before any call
    relabelled = [dict(r, step_source="curriculum") for r in ok_rows]
    assert smokecheck.check(log, relabelled, 1)[1].startswith("step labels")


def test_readout_rule():
    def rows(env, passes, step):
        return [dict(phase="test", step=step, attempt="a", env_id=env, level=1, split="test-iid", seed=i // 2, passed=bool(p)) for i, p in enumerate(passes)]
    base = rows("A", [0, 0, 1, 0] * 20, 0) + rows("B", [1, 0, 0, 0] * 20, 0) + rows("U", [1, 0] * 40, 0)
    trained = rows("A", [1, 1, 1, 0] * 20, 9) + rows("B", [1, 1, 1, 0] * 20, 9) + rows("U", [1, 0] * 40, 9)
    r = readout.readout(base, trained, untrained=["U"])
    by = {e["env_id"]: e for e in r["environments"]}
    assert by["A"]["paired"]["diff"] == pytest.approx(0.5) and by["A"]["meets_rule"] and by["B"]["meets_rule"]
    assert by["U"]["role"] == "untrained" and by["U"]["paired"]["diff"] == 0.0 and by["U"]["meets_rule"]
    assert r["verdict"]["it_trained"] and r["verdict"]["trained_environments_meeting_rule"] == 2 and "p_holm" in by["A"]["paired"]
    worse = rows("A", [1, 1, 1, 0] * 20, 9) + rows("B", [1, 1, 1, 0] * 20, 9) + rows("U", [0, 0] * 40, 9)
    assert not readout.readout(base, worse, untrained=["U"])["verdict"]["it_trained"]              # the untrained environment dropped
    one = rows("A", [1, 1, 1, 0] * 20, 9) + rows("B", [1, 0, 0, 0] * 20, 9) + rows("U", [1, 0] * 40, 9)
    assert not readout.readout(base, one, untrained=["U"])["verdict"]["it_trained"]                # one environment is not two
    none = readout.readout(base[:160], trained[:160])
    assert none["verdict"]["it_trained"] is False and none["verdict"]["note"]                      # no untrained environment: no verdict
    assert "it trained: True" in readout.text_table(r)


# ------------------------------------------------------------------ the stand-in trainer: train, die, restore, stop
def _sim(tmp_path, mix, **env):
    data, run = str(tmp_path / "data"), str(tmp_path / "run")
    if not os.path.exists(os.path.join(data, "dataset.json")):
        instances.main(["--out", data, "--arms", f"{NOWRAP}:2", "--train-rows", "8", "--dev-seeds", "0", "--format", "jsonl"])
        instances.write_config(os.path.join(run, "curriculum.json"), [(NOWRAP, 2)], group_size=4, groups_per_step=1,
                               state_path=os.path.join(run, "ckpt", "curriculum_state.json"))
    full = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "src"), VERL]), OMP_NUM_THREADS="1", DATA_DIR=data,
                CKPT_DIR=os.path.join(run, "ckpt"), RECORD_DIR=os.path.join(run, "records"), MRGYM_CKPT_DEST=os.path.join(run, "remote"),
                MRGYM_CURRICULUM=os.path.join(run, "curriculum.json"), SIM_BASE_URLS=mix, PROMPTS_PER_STEP="1", GROUP_SIZE="4", SAVE_FREQ="1",
                TEST_FREQ="2", VAL_N="1", SIM_IDLE_SHARE="0", SIM_WORKERS="1", RUN_NAME="simtest")
    full.update({k: str(v) for k, v in env.items()})
    return subprocess.run([sys.executable, "-m", "mrgym_verl.sim"], env=full, capture_output=True, text=True, timeout=120), run


def test_sim_trains_dies_restores(tmp_path, mix):
    p, run = _sim(tmp_path, mix, TOTAL_STEPS=3, SIM_DIE_AT_STEP=3, MRGYM_ATTEMPT="a")
    assert p.returncode == -signal.SIGKILL and "step:2 - " in p.stdout and "killed after the rollouts of step 3" in p.stdout
    assert store.latest_checkpoint(os.path.join(run, "remote"))["global_step"] == 2
    with open(os.path.join(run, "remote", "global_step_2", "curriculum_state.json")) as f:
        issued_at_2 = json.load(f)["curriculum"]["issued"]
    import shutil
    shutil.rmtree(os.path.join(run, "ckpt"))                              # the node's disk is gone
    assert store.restore_checkpoint(os.path.join(run, "remote"), os.path.join(run, "ckpt")) == 2
    shutil.copyfile(os.path.join(run, "ckpt", "global_step_2", "curriculum_state.json"), os.path.join(run, "ckpt", "curriculum_state.json"))
    p, _ = _sim(tmp_path, mix, TOTAL_STEPS=4, MRGYM_ATTEMPT="b")
    assert p.returncode == 0, p.stderr[-2000:]
    assert re.search(r"Resuming from .*global_step_2", p.stdout) and "step:4 - " in p.stdout and "step:1 - " not in p.stdout
    rows = stoprules.load(os.path.join(run, "records", "episodes.jsonl"))
    train = [r for r in rows if r["phase"] == "train"]
    assert {r["step"] for r in train} == {1, 2, 3, 4} and all(r["step_source"] == "trainer" for r in train)
    assert {r["phase"] for r in rows} == {"train", "dev"} and {r["split"] for r in rows if r["phase"] == "dev"} == {"dev"}
    assert all(r["seed"] >= 10000 for r in train) and len({r["uid"] for r in train}) == 5          # step 3 was played twice
    # the replayed step 3 plays the instance the checkpoint's sampler issued next, and the table counts it once
    step3 = {r["attempt"]: r["seed"] for r in train if r["step"] == 3}
    assert step3["a"] == step3["b"] == 10000 + issued_at_2
    table = stoprules.decide(rows, dict(episodes_per_step=4, group_size=4, no_reward_steps=99, harness_steps=99, window=99))
    assert [s["episodes"] for s in table["steps"]] == [4, 4, 4, 4]
    # every logged episode is a stored record with its conversation; the paid reward is the signal
    paths = glob.glob(os.path.join(run, "records", "*", "step_*", "episodes", "*", "*", "*", "*", "*", "record.json"))
    assert len(paths) == len(rows)
    for path in paths:
        with open(path) as f:
            rec = json.load(f)
        assert mrecord.missing(rec) == [] and mrecord.transcript_problem(rec, os.path.dirname(path)) is None
        assert rec["training"]["signal"] == rec["reward"] == float(rec["passed"]) and rec["split"] in ("train", "dev")


def test_sim_without_reward_is_stopped_by_the_rule(tmp_path, mix):
    p, run = _sim(tmp_path, mix, TOTAL_STEPS=3, SIM_THETA0=-20, VAL_BEFORE_TRAIN=0, TEST_FREQ=99)
    assert p.returncode == 0, p.stderr[-2000:]
    d = stoprules.decide(stoprules.load(os.path.join(run, "records", "episodes.jsonl")), dict(episodes_per_step=4, group_size=4, no_reward_steps=3))
    assert d["stop"] and d["rule"] == "no_reward"


# ------------------------------------------------------------------ the node scripts
ENTRIES = ("calibrate.sh", "train_smoke.sh", "train.sh")


def _text(name):
    with open(os.path.join(VERL, "node", name)) as f:
        return f.read()


def test_entry_scripts_share_the_bootstrap_and_name_nothing():
    blocks = []
    for name in ENTRIES:
        text = _text(name)
        start, end = text.index("# ---------------------------------------------------------------- bootstrap"), text.index("# ---------------------------------------------------------------- end of bootstrap")
        blocks.append(text[start:end])
        head = text[:start]
        assert "parameters: environment variables" in head and head.count("set -uo pipefail") == 1
        assert subprocess.run(["bash", "-n", os.path.join(VERL, "node", name)]).returncode == 0
    assert blocks[0] == blocks[1] == blocks[2]
    for path in glob.glob(os.path.join(VERL, "**", "*"), recursive=True):
        if os.path.isfile(path) and not path.endswith((".pyc", ".md")):
            with open(path, errors="replace") as f:
                text = f.read()
            assert not re.search(r"s3://[a-z0-9][a-z0-9.-]*[a-z0-9]/", text.replace("s3://bucket/", "").replace("s3://b/", "")), f"{path} names a bucket"
            assert "python -m uv" not in text.replace("never as `python -m uv`", "").replace("`python -m uv pip install`", ""), path
    lib = _text("lib.sh")
    assert "UV_PYTHON_PREFERENCE=only-managed" in lib and "uv.find_uv_bin()" in lib      # the lessons of the earlier jobs


@pytest.mark.parametrize("name", ENTRIES)
def test_entry_script_refuses_to_start_without_its_inputs(name, tmp_path):
    p = subprocess.run(["bash", os.path.join(VERL, "node", name)], env=dict(PATH=os.environ["PATH"], MRGYM_STUB="1"), capture_output=True, text=True, timeout=30)
    assert p.returncode == 2 and "RUN_ID, OUT_URI and REPO_SRC are required" in p.stdout


def _leftovers(marker: str) -> list:
    """Processes that still name ``marker`` (a temporary folder of a test) on their command line."""
    out = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True).stdout.splitlines()
    return [x for x in out if marker in x and " ps " not in x]


def test_calibrate_script_resumes_after_a_kill(tmp_path):
    env = dict(PATH=os.environ["PATH"], HOME=os.environ.get("HOME", "/tmp"), MRGYM_STUB="1", STUB_SKIP_VENV="1", STUB_PYTHON=sys.executable,
               RUN_ID="c1", OUT_URI=str(tmp_path / "out"), REPO_SRC=ROOT, WORK_ROOT=str(tmp_path / "work"), HOLD_S="0", HEARTBEAT_S="0",
               SYNC_EVERY="1", MODEL_HF="stand-in/none", MODEL_NAME="mix:0.5:0.0", CAL_PLAN=f"{NOWRAP}:2", CAL_N="4", CAL_K="4", CAL_WORKERS="1",
               CAL_PROCS="2", SELFTEST="0", OMP_NUM_THREADS="1", SUMMARY_EVERY="0")
    script = os.path.join(VERL, "node", "calibrate.sh")
    first = subprocess.run(["bash", script], env=dict(env, SIM_DIE_AT="calibrate", SIM_DIE_AFTER_S="3"), capture_output=True, text=True, timeout=110)
    assert first.returncode == -signal.SIGKILL, first.stdout[-1500:]
    assert not _leftovers(str(tmp_path))                                  # the kill took every child with it
    run = tmp_path / "out" / "runs" / "c1"
    uploaded = glob.glob(str(run / "records" / "cal" / "episodes" / "*" / "*" / "*" / "*" / "*" / "record.json"))
    assert not (run / "DONE.txt").exists() and not (run / "FAILED.txt").exists()
    import shutil
    shutil.rmtree(tmp_path / "work")                                      # a fresh node
    second = subprocess.run(["bash", script], env=env, capture_output=True, text=True, timeout=110)
    assert second.returncode == 0, second.stdout[-2500:]
    assert f"records already present: {len(uploaded)}" in second.stdout and "calibration complete: 16 episode record(s)" in second.stdout
    with open(run / "calibration_summary.json") as f:
        s = json.load(f)
    assert s["complete"] and s["episodes"] == 16 and s["cells"][0]["n_instances"] == 4 and s["cells"][0]["k"] == 4
    with open(run / "train_arms.json") as f:
        arms = json.load(f)
    assert arms["complete"] and set(arms) >= {"train_arms", "base_rate", "rule", "cells"}
    assert not _leftovers(str(tmp_path))


def test_calibrate_uses_the_environments_own_limit(mix, tmp_path):
    from mrgym import calibrate
    r = calibrate.run(NOWRAP, "stub:reference", mix, str(tmp_path / "cal"), levels=[2], n=1, episodes=1, first_seed=10000, max_turns=0, quiet=True)
    assert r["calibration"][0]["max_turns"] == mrgym.spec(NOWRAP).max_steps and r["settings"]["max_turns"] == mrgym.spec(NOWRAP).max_steps
    rec = calibrate.read_records(str(tmp_path / "cal"), NOWRAP, "stub:reference")[0]
    assert rec["max_steps"] == mrgym.spec(NOWRAP).max_steps and rec["settings"]["max_turns"] is None
    s = calsum.summarize(str(tmp_path / "cal"), "stub:reference", [(NOWRAP, 2)], 1, 1, 10000, gpus=8, concurrency=1)
    assert s["complete"] and s["cells"][0]["stream_tokens_est"]["episodes"] == 1 and s["gpu_s_per_episode"] > 0


# ------------------------------------------------------------------ the pre-registration of the 27B run
def _plan_27b():
    with open(os.path.join(ROOT, "docs", "experiments", "27b", "plan.json")) as f:
        return json.load(f)


def _env_27b():
    env = {}
    with open(os.path.join(ROOT, "docs", "experiments", "27b", "overrides.env")) as f:
        for text in f:
            text = text.split("#")[0].strip()
            if "=" in text:
                k, v = text.split("=", 1)
                env[k] = v.strip()
    return env


def test_27b_tables_are_the_computed_ones():
    import importlib.util
    spec = importlib.util.spec_from_file_location("compute_27b", os.path.join(ROOT, "docs", "experiments", "27b", "compute.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0
    plan = _plan_27b()
    mem = {m["card"]: m for m in module.memory(plan)}
    assert mem["H100"]["tokens_fit"][0] < mem["H100"]["cap"] < mem["H100"]["tokens_fit"][1]        # on 80 GB cards the smoke decides
    assert mem["H200"]["verdict"] == mem["B200"]["verdict"] == "fits on paper"
    assert abs(mem["H100"]["rollout_weights"] * 4 - plan["hardware"]["model_dims"]["weights_gb_bf16"]) < 1e-9
    c = module.compute(plan)
    assert c["per_step"]["episodes"] == 128 and c["designs"][0]["total"]["low"] < c["designs"][0]["total"]["high"] < c["designs"][1]["total"]["high"]


def test_27b_plan_agrees_with_the_settings_and_the_code():
    plan, env = _plan_27b(), _env_27b()
    a, s, r = plan["algorithm"], plan["stop_rules"], plan["readout"]
    assert plan["status"].startswith("not launched") and plan["environments"]["train_cells"] is None       # filled by the calibration only
    assert (int(env["GROUP_SIZE"]), int(env["PROMPTS_PER_STEP"]), int(env["TOTAL_STEPS"])) == (a["group_size"], a["groups_per_step"], a["steps"])
    assert float(env["LR"]) == a["lr"] and int(env["WARMUP_STEPS"]) == a["warmup_steps"] and env["KL_COEF"] == "0" and env["SIGNAL"] == "pass"
    assert (int(env["LORA_RANK"]), int(env["LORA_ALPHA"])) == (plan["model"]["lora_rank"], plan["model"]["lora_alpha"])
    assert (int(env["MAX_PROMPT_LEN"]), int(env["MAX_RESPONSE_LEN"]), int(env["MAX_TOKENS_PER_TURN"])) == (a["max_prompt_len"], a["max_response_len"], a["max_tokens_per_turn"])
    assert int(env["TOKENS_PER_GPU"]) == a["max_prompt_len"] + a["max_response_len"] + 2048
    assert (int(env["SAVE_FREQ"]), int(env["TEST_FREQ"])) == (a["save_every_steps"], a["dev_every_steps"]) and int(env["LOOKS"]) == a["steps"] // a["dev_every_steps"]
    assert int(env["ROLLOUT_TP"]) == plan["hardware"]["rollout"]["tensor_parallel"] and float(env["GPU_MEM_UTIL"]) == plan["hardware"]["rollout"]["gpu_memory_utilization"]
    for key, name in (("no_reward_steps", "NO_REWARD_STEPS"), ("no_reward_min", "NO_REWARD_MIN"), ("harness_steps", "HARNESS_STEPS"), ("mixed_min", "MIXED_MIN"),
                      ("submit_min", "SUBMIT_MIN"), ("cut_max", "CUT_MAX"), ("window", "WINDOW"), ("collapse_steps", "COLLAPSE_STEPS"), ("alpha", "ALPHA"),
                      ("looks", "LOOKS"), ("mde", "MDE"), ("futility_from_step", "FUTILITY_FROM_STEP"), ("mastered", "MASTERED")):
        assert float(env[name]) == float(s[key]), name
        assert key in stoprules.DEFAULTS
    assert (float(env["GAIN"]), float(env["LOWER"]), float(env["HARM"]), int(env["MIN_ENVS"])) == (r["gain"], r["lower"], r["harm"], r["min_envs"])
    # the candidate cells exist, are admitted and are not practice levels; the calibration rule is the package's
    from mrgym import learn
    for cell in plan["environments"]["candidates_for_training"]:
        for env_id, level in instances.parse_arms(cell):
            assert mrgym.spec(env_id).kind == "admitted" and level in mrgym.spec(env_id).levels
    assert learn.ENTRY_BAND == (0.20, 0.60) and learn.MIXED_MIN == 0.80 and learn.GROUP_SIZE == a["group_size"]
    assert plan["curriculum"]["first_seed"] == 10_000 > 9000 + plan["evaluation"]["calibration"]["instances"]
    assert "8357f1e" in plan["algorithm"]["name"]
    with open(os.path.join(ROOT, "docs", "experiments", "PREREG-27b.md")) as f:
        doc = f.read()
    assert "not launched" in doc and "+15 points" in doc and "at least 2" in doc and "[0.20, 0.60]" in doc
    import mrgym_verl
    assert mrgym_verl.PINNED_VERL.startswith("8357f1e") and mrgym_verl.PINNED_VERL in _text("lib.sh")


# ------------------------------------------------------------------ several runner processes on one run folder
def test_two_runner_processes_share_a_run_folder(tmp_path, mix):
    """Two processes used to share one temporary file name for ``run_config.json``; the second rename raised
    FileNotFoundError and the process died before it played an episode."""
    out = str(tmp_path / "run")
    os.makedirs(out)
    code = ("import sys\nfrom mrgym.agents import runner\n"
            "for i in range(150):\n    runner._note_invocation(sys.argv[1], dict(utc=runner.now_utc(), command=f'p{sys.argv[2]} {i}'))\n")
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"), OMP_NUM_THREADS="1")
    procs = [subprocess.Popen([sys.executable, "-c", code, out, str(k)], env=env, stderr=subprocess.PIPE, text=True) for k in range(2)]
    errors = [p.communicate(timeout=60)[1] for p in procs]
    assert [p.returncode for p in procs] == [0, 0], errors
    with open(os.path.join(out, "run_config.json")) as f:
        commands = [x["command"] for x in json.load(f)["invocations"]]
    assert len(commands) == 300 and len(set(commands)) == 300            # under the lock no invocation is lost
    assert not glob.glob(os.path.join(out, "*.tmp"))
    # and the real thing: two `mrgym calibrate` processes with disjoint seeds, one records folder
    cal = str(tmp_path / "cal")
    args = [sys.executable, "-m", "mrgym.cli", "calibrate", NOWRAP, "--model", "stub:reference", "--base-url", mix, "--levels", "2", "--n", "1", "--episodes", "2",
            "--max-turns", "0", "--workers", "2", "--out", cal, "--json"]
    procs = [subprocess.Popen(args + ["--first-seed", str(10000 + k)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for k in range(2)]
    outs = [p.communicate(timeout=120) for p in procs]
    assert [p.returncode for p in procs] == [1, 1], outs                  # 1: no entry rung (the reference passes); a crash would be 2 or a trace
    assert all(json.loads(o)["calibration"][0]["n_episodes"] in (2, 4) and json.loads(o)["infra_errors"] == 0 for o, _ in outs), outs
    with open(os.path.join(cal, "run_config.json")) as f:
        assert len(json.load(f)["invocations"]) == 2
    s = calsum.summarize(cal, "stub:reference", [(NOWRAP, 2)], 2, 2, 10000)
    assert s["complete"] and s["episodes"] == 4


def test_a_restart_adds_episodes_to_a_look_of_the_same_weights_and_replaces_a_look_of_other_weights():
    def ep(attempt, phase, step, seed, passed):
        return dict(attempt=attempt, phase=phase, step=step, env_id="Scanner/NoWrap-v1", level=1, split="dev", seed=seed, passed=passed)
    rows = [ep("a", "dev", 0, s, False) for s in range(4)]                   # attempt a: base look, one step, preempted
    rows += [ep("a", "train", 1, 100 + s, False) for s in range(4)]
    rows += [ep("b", "dev", 0, s, s == 0) for s in range(4)]                 # attempt b restored nothing: trains from step 1 again
    rows += [ep("b", "train", k, 100 + s, True) for k in (1, 2) for s in range(4)]
    rows += [ep("b", "dev", 2, s, True) for s in range(4)]                   # look at step 2, checkpoint, preempted
    rows += [ep("c", "dev", 2, s, s < 2) for s in range(4)]                  # attempt c restored step 2 and evaluated it again
    looks = stoprules._looks(rows, "dev")
    count = lambda step, attempt: sum(1 for r in looks if r["step"] == step and r["attempt"] == attempt)   # noqa: E731
    assert (count(0, "a"), count(0, "b"), count(2, "b"), count(2, "c")) == (4, 4, 4, 4)       # c has not trained: step 2 is its first look
    rows += [ep("c", "train", 3, 100 + s, True) for s in range(4)]           # c trains on from step 3: the same weights, still pooled
    assert sum(1 for r in stoprules._looks(rows, "dev") if r["step"] == 2) == 8
    table = stoprules.dev_table(rows, dict(stoprules.DEFAULTS, looks=2))
    assert [(d["step"], d["episodes"]) for d in table] == [(0, 8), (2, 8)] and abs(table[1]["pass_rate"] - 0.75) < 1e-9
    other = rows + [ep("d", "train", 2, 100 + s, True) for s in range(4)] + [ep("d", "dev", 2, s, False) for s in range(4)]
    kept = [r for r in stoprules._looks(other, "dev") if r["step"] == 2]     # d restored step 1 and trained step 2 again: other weights
    assert {r["attempt"] for r in kept} == {"d"} and len(kept) == 4
