"""``mrgym/envs/family.py``: the reward rule, the environment class and the helpers a verifier uses.

The environment class is driven with ``tests/stub_family`` (milliseconds per episode); the real families go through the
same assertions in ``tests/test_scoring_table.py``.
"""
import hashlib
import itertools

import numpy as np
import pytest

import mrgym
from conftest import scan_call
from mrgym import keys, policies
from mrgym.env import derive_seed
from mrgym.envs import family as F
from mrgym.scanner import console, session
from stub_family import task as stub_task

SUFFIX = " Ends the episode; a deliverable the console cannot evaluate is refused and the episode continues."


def _number(env_id, level, seed):
    return stub_task.number(derive_seed(env_id, level, "train", seed, environ={}), level)


def _env(env_id, seed=0, scans=1, **kw):
    env = mrgym.make(env_id, seed=seed, **kw)
    obs, info = env.reset()
    for _ in range(scans):
        policies.ToolScanner(env).scan(stub_task.CHEAP_SCAN)
    return env, obs, info


# ---------------------------------------------------------------------------------------------- the rule
def test_the_gates_and_the_reward_rule_are_the_documented_ones():
    assert F.GATES == ("log_replays_within_budget", "measured")
    assert F.reward_of(True) == 1.0 and F.reward_of(False) == 0.0
    assert type(F.reward_of(True)) is float and type(F.reward_of(False)) is float


def _score(passed, progress=None, **checks):
    return dict(passed=passed, checks=checks, **({} if progress is None else dict(progress=progress)))


def test_progress_is_one_for_a_pass_zero_without_a_valid_measured_answer_and_below_one_otherwise():
    """The documented rule ("Progress and outcome" in family.py), row by row."""
    ok = dict(log_replays_within_budget=True, measured=True)
    assert (F.PROGRESS_FAIL_MAX, F.PROGRESS_SUP, F.GATE_OUTCOMES) == (0.5, 0.999, ("no_measurement", "log_or_budget"))
    assert F.progress_of(_score(True, 0.2, a=True, **ok)) == 1.0                          # a pass is 1, whatever the family says
    assert F.progress_of(_score(False, 0.9, deliverable_valid=False)) == 0.0             # nothing valid was delivered
    assert F.progress_of(_score(False, 0.9, a=True, log_replays_within_budget=True, measured=False)) == 0.0
    assert F.progress_of(_score(False, 0.9, a=True, log_replays_within_budget=False, measured=True)) == 0.0
    # the family's own value, kept inside [0, 0.999]
    assert F.progress_of(_score(False, 0.4, a=False, **ok)) == 0.4
    assert F.progress_of(_score(False, 1.0, a=False, **ok)) == 0.999 and F.progress_of(_score(False, 7, a=False, **ok)) == 0.999
    assert F.progress_of(_score(False, -0.3, a=False, **ok)) == 0.0 and F.progress_of(_score(False, 0.0, a=False, **ok)) == 0.0
    # the default: 0.5 x the share of the task's own checks that pass before the first failure, in their order
    assert F.progress_of(_score(False, a=True, b=True, c=False, d=True, **ok)) == 0.5 * 2 / 4
    assert F.progress_of(_score(False, a=False, b=True, **ok)) == 0.0
    assert F.progress_of(_score(False, deliverable_valid=True, a=True, b=False, **ok)) == 0.5 * 2 / 3
    assert F.progress_of(_score(False, **ok)) == 0.0                                     # no check of its own: nothing to show
    assert all(type(F.progress_of(_score(False, p, a=False, **ok))) is float for p in (0.4, 1, 0))
    assert F.progress_of(dict(passed=False, checks=dict(a=True, b=False))) == 0.25       # gates not recorded: taken as met
    assert F.progress_of(_score(False, a=True, **ok)) == 0.5                             # one check, met, yet not passed: the cap


def test_outcome_is_the_first_failed_stage():
    ok = dict(log_replays_within_budget=True, measured=True)
    task = type("Task", (), dict(OUTCOMES=("far", "near"), outcome=staticmethod(lambda score: "near" if score["checks"].get("b", True) else "far")))
    assert F.outcome_of(_score(True, **ok), task) == "pass"
    assert F.outcome_of(_score(False, deliverable_valid=False, measured=False, log_replays_within_budget=False), task) == "invalid_deliverable"
    assert F.outcome_of(_score(False, a=False, measured=False, log_replays_within_budget=False), task) == "no_measurement"
    assert F.outcome_of(_score(False, a=False, measured=True, log_replays_within_budget=False), task) == "log_or_budget"
    assert F.outcome_of(_score(False, a=False, **ok), task) == "near"                    # the family's own label
    assert F.outcome_of(_score(False, a=False, b=False, **ok), task) == "far"            # ... decided on the score it is given
    assert F.outcome_of(_score(False, a=False, **ok)) == "check_failed"                  # a family without labels
    assert F.outcome_of(dict(passed=False, checks=dict(a=False)), task) == "near"        # gates not recorded: taken as met
    assert F.outcomes(task)[-3:] == ("far", "near", "pass") and F.outcomes()[-2:] == ("check_failed", "pass")
    assert F.outcomes(task)[-5:-3] == ("no_measurement", "log_or_budget") and "invalid_deliverable" in F.outcomes(task)
    assert len(set(F.outcomes(task))) == len(F.outcomes(task))


def test_finish_reports_progress_and_outcome_and_never_pays_them(stub_env):
    n = _number(stub_env, 1, 0)
    env, _, _ = _env(stub_env, scans=0)
    rep = dict(ok=True, over_budget=False, scans=1)
    reward, score, _ = env.finish(dict(value=n), rep)
    assert (reward, score["progress"], score["outcome"]) == (1.0, 1.0, "pass")
    reward, score, _ = env.finish(dict(value=n + 1), rep)                # two checks of its own, the second fails: 0.5 x 1 / 2
    assert (reward, score["progress"], score["outcome"]) == (0.0, 0.25, "check_failed")
    reward, score, _ = env.finish(dict(value=n), dict(rep, scans=0))
    assert (reward, score["progress"], score["outcome"]) == (0.0, 0.0, "no_measurement")
    reward, score, _ = env.finish(dict(value=n), dict(rep, ok=False))
    assert (reward, score["progress"], score["outcome"]) == (0.0, 0.0, "log_or_budget")
    assert env.outcomes() == F.outcomes(env.task) and env.outcomes()[-2:] == ("check_failed", "pass")
    env.close()


def test_a_noise_nonce_selects_another_noise_stream_of_the_same_world(stub_env):
    env = mrgym.make(stub_env, seed=4)
    env.reset()
    plain = env.episode_config()
    env.reset(options={"noise_nonce": 3})
    other = env.episode_config()
    assert "noise_draw" not in plain and other["noise_draw"] == 3
    assert {k: v for k, v in other.items() if k not in ("noise_draw",)} == plain
    env.close()


# ---------------------------------------------------------------------------------------------- the environment class
def test_the_family_supplies_instance_statement_and_the_submit_tool(stub_env):
    env, obs, info = _env(stub_env, seed=5, scans=0, level=2)
    n = _number(stub_env, 2, 5)
    cfg = env.episode_config()
    assert cfg["level"] == 2 and cfg["task"]["number"] == n and cfg["task_id"] == info["task_id"] != "stub-token"
    assert cfg["task_id"] == keys.instance_label("stub", derive_seed(stub_env, 2, "train", 5, environ={}), "train")   # the trainer's label
    assert obs.text.startswith(f"# Deliver the number {n}\n") and "Scan time: 20 s" in obs.text
    assert info["tools"][-1] == "submit" and {"scanner_info", "plan", "scan", "read_regions", "view"} <= set(info["tools"][:-1])
    spec = env.terminal_tool()
    assert (spec.name, spec.terminal, spec.cost, spec.counted) == ("submit", True, "free", False)
    assert spec.description == stub_task.SUBMIT_DOC + SUFFIX
    assert spec.parameters == stub_task.submit_schema(cfg)
    assert env.tools("neutral")[-1]["name"] == "submit"
    env.close()


def test_reward_is_one_only_for_a_pass_with_a_measurement_on_a_log_that_replays(stub_env):
    for level, seed in ((1, 0), (2, 3)):
        n = _number(stub_env, level, seed)
        env, _, _ = _env(stub_env, seed=seed, scans=0, level=level)
        for value, ok, over, scans in itertools.product((n, n + 1), (True, False), (False, True), (0, 2)):
            reward, score, text = env.finish(dict(value=value), dict(ok=ok, over_budget=over, scans=scans))
            want = 1.0 if (value == n and ok and not over and scans > 0) else 0.0
            assert type(reward) is float and reward == want and score["reward"] == want and score["passed"] is (want == 1.0)
            assert score["checks"] == dict(deliverable_valid=True, right_number=value == n,
                                           log_replays_within_budget=ok and not over, measured=scans > 0)
            assert all(type(v) is bool for v in score["checks"].values())
            assert score["error"] == float(value - n) and score["diagnostic"] == ("" if value == n else "off by 1")
            assert text == "deliverable received; the episode is over"
        env.close()


def test_reveal_score_adds_the_verdict_and_the_diagnostic_to_the_text(stub_env):
    n = _number(stub_env, 1, 0)
    env, _, _ = _env(stub_env, scans=0, reveal_score=True)
    rep = dict(ok=True, over_budget=False, scans=1)
    assert env.finish(dict(value=n), rep)[2] == "deliverable received; the episode is over\nreward 1 · every check passed"
    assert env.finish(dict(value=n + 2), rep)[2] == ("deliverable received; the episode is over\n"
                                                    "reward 0 · failed: right_number\noff by 2")
    assert env.finish(dict(value=n), dict(rep, scans=0))[2] == "deliverable received; the episode is over\nreward 0 · failed: measured"
    text = env.finish(dict(value=n + 2), dict(ok=False, over_budget=False, scans=0))[2]
    assert text.splitlines()[1] == "reward 0 · failed: right_number, log_replays_within_budget, measured"
    env.close()


def test_a_task_that_reports_no_checks_still_gets_the_two_gates(stub_env, monkeypatch):
    env, _, _ = _env(stub_env, scans=0)
    monkeypatch.setattr(env.task, "score", lambda cfg, a, world=None, engine=None: dict(passed=True, diagnostic=""))
    reward, score, _ = env.finish(dict(value=1), dict(ok=True, over_budget=False, scans=1))
    assert reward == 1.0 and score["checks"] == dict(log_replays_within_budget=True, measured=True)
    reward, score, _ = env.finish(dict(value=1), dict(ok=True, over_budget=False, scans=0))
    assert reward == 0.0 and score["passed"] is False
    # a task whose own verdict is "failed" is not rescued by the gates
    monkeypatch.setattr(env.task, "score", lambda cfg, a, world=None, engine=None: dict(passed=False, checks=dict(x=True), diagnostic=""))
    assert env.finish(dict(value=1), dict(ok=True, over_budget=False, scans=1))[0] == 0.0
    env.close()


def test_the_verifier_gets_the_hidden_world_and_engine_of_the_episode(stub_env, monkeypatch):
    env, _, _ = _env(stub_env, scans=0)
    seen = {}

    def score(cfg, answer, world=None, engine=None):
        seen.update(cfg=cfg, answer=answer, world=world, engine=engine)
        return F.result(dict(ok=True))
    monkeypatch.setattr(env.task, "score", score)
    env.finish(dict(value=7), dict(ok=True, over_budget=False, scans=1))
    assert seen["cfg"] == env.episode_config() and seen["answer"] == dict(value=7)
    assert seen["world"] is env.hidden_world() and seen["engine"].world is env.hidden_world()
    env.close()


def test_a_family_that_takes_the_split_is_told_it(stub_env, monkeypatch):
    plain = stub_task.make_instance

    def make_instance(seed, level, task_id=None, split="train"):
        cfg = plain(seed, level, task_id)
        return dict(cfg, task=dict(cfg["task"], split=split))
    monkeypatch.setattr(stub_task, "make_instance", make_instance)
    for split in ("train", "dev"):
        env = mrgym.make(stub_env, seed=5, split=split)
        _, info = env.reset()
        cfg = env.episode_config()
        assert cfg["task"]["split"] == split and cfg["seed"] == derive_seed(stub_env, 1, split, 5, environ={})
        assert cfg["task_id"] == info["task_id"] == keys.instance_label("stub", cfg["seed"], split)
        env.close()


def test_a_family_with_prefill_acquires_its_scans_once_per_episode_and_states_them(stub_env, monkeypatch):
    calls = []

    def prefill(cfg, scanner):
        calls.append((cfg["task"]["number"], scanner))
        return [scanner.scan(dict(stub_task.CHEAP_SCAN))]
    monkeypatch.setattr(stub_task, "prefill", prefill, raising=False)
    monkeypatch.setattr(stub_task, "statement", lambda cfg, given=None: f"# number {cfg['task']['number']}; given scan "
                        f"{given[0]['scan_index'] if given else None} for {given[0]['cost_s'] if given else 0} s")
    env = mrgym.make(stub_env, seed=0)
    obs, _ = env.reset()
    n = _number(stub_env, 1, 0)
    assert obs.text.startswith(f"# number {n}; given scan 0 for 3.15 s\n")
    assert len(calls) == 1 and calls[0][0] == n and calls[0][1].session is env._session
    assert obs.status["scans"] == 1 and obs.status["scan_time_used_s"] == pytest.approx(3.15)       # logged and charged
    assert env.task_text() == f"# number {n}; given scan 0 for 3.15 s" and len(calls) == 1           # not acquired again
    obs, reward, terminated, _, info = env.step({"name": "submit", "arguments": {"value": n}})
    assert terminated and reward == 1.0 and info["replay"]["scans"] == 1     # the prefilled scan is the measurement
    env.reset(seed=1)                                                    # a new episode acquires its own
    assert len(calls) == 2 and calls[1][0] == _number(stub_env, 1, 1) and calls[1][1].session is env._session
    env.close()


def test_a_family_with_its_own_outcome_labels_gets_them(stub_env, monkeypatch):
    monkeypatch.setattr(stub_task, "OUTCOMES", ("far", "near"), raising=False)
    monkeypatch.setattr(stub_task, "outcome", lambda score: "near" if score["error"] <= 1 else "far", raising=False)
    n = _number(stub_env, 1, 0)
    env, _, _ = _env(stub_env, scans=0)
    rep = dict(ok=True, over_budget=False, scans=1)
    assert env.finish(dict(value=n + 1), rep)[1]["outcome"] == "near" and env.finish(dict(value=n + 5), rep)[1]["outcome"] == "far"
    assert env.outcomes()[-3:] == ("far", "near", "pass") == F.outcomes(stub_task)[-3:]
    env.close()


def test_a_refused_deliverable_is_an_error_observation_and_the_episode_continues(stub_env):
    n = _number(stub_env, 1, 0)
    env, _, _ = _env(stub_env)
    used = env.state()["scan_time_used_s"]
    obs, reward, terminated, truncated, info = env.step({"name": "submit", "arguments": {"value": -3}})
    assert (obs.is_error, reward, terminated, truncated) == (True, 0.0, False, False)
    assert set(info) == {"step"} and info["step"]["error_type"] == "invalid_deliverable" and info["step"]["counted"] is False
    err = obs.data["error"]
    assert err["type"] == "invalid_deliverable" and err["message"] == "the number is not negative; nothing was delivered"
    assert obs.data["episode_continues"] is True and obs.data["charged"] == dict(scan_time_s=0.0, tool_calls=0, rejected_calls=1)
    assert obs.text.splitlines()[0] == "error invalid_deliverable · the number is not negative; nothing was delivered"
    assert obs.tool == "submit" and env.state()["done"] is False and env.state()["scan_time_used_s"] == used
    obs, reward, terminated, truncated, info = env.step({"name": "submit", "arguments": {"value": n}})
    assert (obs.is_error, reward, terminated, truncated) == (False, 1.0, True, False)
    assert info["score"]["passed"] is True and info["submitted"] is True and obs.data == {"received": True}
    env.close()


def test_the_check_sees_the_console_card_of_the_episode(stub_env, monkeypatch):
    env, _, _ = _env(stub_env, scans=0)
    seen = []
    monkeypatch.setattr(env.task, "check", lambda cfg, a, exam: seen.append((cfg, a, exam)) or "no")
    obs = env.step({"name": "submit", "arguments": {"value": 1}})[0]
    assert obs.is_error and obs.data["error"]["message"] == "no; nothing was delivered"
    assert seen == [(env.episode_config(), {"value": 1}, env.hidden_world()["console"])]
    env.close()


def test_a_scan_the_remaining_budget_cannot_pay_is_refused_and_costs_nothing(stub_env):
    env, _, _ = _env(stub_env, scans=0)                                  # 20 s budget; a cheap scan costs 3.15 s
    sc = policies.ToolScanner(env)
    for k in range(1, 7):
        sc.scan(stub_task.CHEAP_SCAN)
        assert env.state()["scan_time_used_s"] == pytest.approx(3.15 * k) and env.state()["scans"] == k
    obs, reward, terminated, truncated, info = env.step(scan_call(env, stub_task.CHEAP_SCAN))
    assert obs.is_error and obs.data["error"]["type"] == "budget" and (reward, terminated, truncated) == (0.0, False, False)
    assert info["step"]["counted"] is True and info["step"]["cost_s"] == 0.0     # a refusal by the scanner is a tool call that costs no time
    now = env.state()                                                    # 18.9 s used: the seventh scan (22.05 s) is over the budget
    assert now["scan_time_used_s"] == pytest.approx(18.9) and now["scans"] == 6 and now["scan_time_left_s"] == pytest.approx(1.1)
    obs, reward, terminated, _, info = env.step({"name": "submit", "arguments": {"value": _number(stub_env, 1, 0)}})
    assert terminated and reward == 1.0 and info["replay"]["used_s"] == pytest.approx(18.9) and not info["replay"]["over_budget"]
    env.close()


# ---------------------------------------------------------------------------------------------- helpers of a task module
def test_reseed_is_the_seed_itself_then_a_hash_of_seed_and_attempt(seed):
    assert F.reseed(seed, 0) == seed
    for k in (1, 2, 17):
        assert F.reseed(seed, k) == hashlib.sha256(f"{seed}|attempt|{k}".encode()).hexdigest()[:32]
    assert len({F.reseed(seed, k) for k in range(5)}) == 5


def test_public_regions_and_the_region_table():
    regions = [dict(id="R1", cls="vial3", y_mm=-12.5, x_mm=40.0, radius_mm=11.5, t1_ms=900.0),
               dict(id="R12", cls="vial0", y_mm=3.456, x_mm=-100.0, radius_mm=8.0)]
    public = F.public_regions(regions)
    assert public == [dict(id="R1", y_mm=-12.5, x_mm=40.0, radius_mm=11.5), dict(id="R12", y_mm=3.456, x_mm=-100.0, radius_mm=8.0)]
    assert F.public_regions([]) == []
    assert F.region_table(public) == "  R1: y -12.5, x 40, radius 11.5\n  R12: y 3.46, x -100, radius 8"      # mm, as the task text says once
    assert F.region_table([]) == ""


def test_result_is_the_conjunction_of_its_checks():
    r = F.result(dict(a=1, b=np.bool_(True)), "fine", error=0.25, extra=[1])
    assert r == dict(passed=True, checks=dict(a=True, b=True), diagnostic="fine", error=0.25, extra=[1])
    assert all(type(v) is bool for v in r["checks"].values()) and type(r["passed"]) is bool
    r = F.result(dict(a=True, b=0))
    assert r == dict(passed=False, checks=dict(a=True, b=False), diagnostic="")
    assert F.result({})["passed"] is True


def test_noise_free_read_out_is_the_mean_over_the_pixels_inside_the_circle(qa_engine, qa_world):
    regions = F.public_regions(qa_world["regions"][:2]) + [dict(id="off", y_mm=31.0, x_mm=-17.0, radius_mm=9.0)]
    rp = console.plan(dict(sequence="spgr", tr_ms=30.0, flip_deg=20.0, matrix=64, n_echoes=2, centre_y_mm=10.0, phase_fov=0.75,
                           pe_dir="x"), qa_world["console"])["realised"]
    img, rows = F.noise_free(qa_engine, rp, regions)
    assert img.shape == (2, 64, 48) and np.array_equal(img, qa_engine.acquire(rp, None))
    assert list(rows) == [r["id"] for r in regions]
    # the documented geometry: pixel i of an n-point axis is centred at centre + (i - n // 2) * field of view / n
    y = 10.0 + (np.arange(64) - 32) * (240.0 / 64)
    x = (np.arange(48) - 24) * (180.0 / 48)
    for image in (0, 1):
        got = F.region_complex(qa_engine, rp, regions, image=image)
        assert list(got) == [r["id"] for r in regions]
        for r in regions:
            mask = (y[:, None] - r["y_mm"]) ** 2 + (x[None, :] - r["x_mm"]) ** 2 <= r["radius_mm"] ** 2
            want = img[image][mask].mean()
            assert rows[r["id"]]["n_pixels"] == int(mask.sum()) > 10
            assert got[r["id"]] == pytest.approx(want, rel=1e-7)
            assert rows[r["id"]]["mean_magnitude"][image] == pytest.approx(np.abs(img[image][mask]).mean(), rel=1e-8)
    first = F.region_complex(qa_engine, rp, regions)
    assert first == F.region_complex(qa_engine, rp, regions, image=0) != F.region_complex(qa_engine, rp, regions, image=1)


def test_noise_free_read_out_follows_a_turned_field_of_view(qa_engine, qa_world):
    regions = F.public_regions(qa_world["regions"][:3])
    plain = console.plan(dict(sequence="se", tr_ms=300.0, matrix=64), qa_world["console"])["realised"]
    turned = console.plan(dict(sequence="se", tr_ms=300.0, matrix=64, rotation_deg=35.0), qa_world["console"])["realised"]
    a, b = F.region_complex(qa_engine, plain, regions), F.region_complex(qa_engine, turned, regions)
    for r in regions:                                                    # the same material is read out, wherever the image puts it
        assert abs(b[r["id"]]) == pytest.approx(abs(a[r["id"]]), rel=0.03), r["id"]
    img, rows = F.noise_free(qa_engine, turned, regions)
    wrong = session.roi_readouts(img, regions, session.fov_yx(turned), session.centre_yx(turned))       # circles not turned with it
    assert any(abs(w["complex_mean_magnitude"][0] - rows[w["id"]]["complex_mean_magnitude"][0]) > 0.05 * abs(a[w["id"]]) for w in wrong)
