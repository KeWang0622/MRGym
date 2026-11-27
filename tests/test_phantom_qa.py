"""``Scanner/PhantomQA-v0``: the instance generator, the task text, the control policies through the tools.

The reward as a table is ``tests/scoring/phantom_qa.py``; the gate (``mrgym gate Scanner/PhantomQA-v0``) is run by
hand, its development result is shipped as ``gate_dev.json``."""
import json

import pytest

import mrgym
from mrgym import policies, qa
from mrgym.env import derive_seed
from mrgym.envs.phantom_qa import controls, task
from mrgym.scanner import console

ENV_ID = "Scanner/PhantomQA-v0"


def _cfg(level, seed):
    return task.make_instance(derive_seed(ENV_ID, level, "train", seed, environ={}), level, task_id="t")


def test_the_family_is_registered_with_three_levels():
    spec = mrgym.spec(ENV_ID)
    assert spec.kind in ("candidate", "admitted") and (spec.levels, spec.family, spec.max_steps) == ((1, 2, 3), "mrgym.envs.phantom_qa", 64)
    assert task.ENV_ID == ENV_ID and set(task.LEVELS) == {1, 2, 3}
    assert {f: v["test"] for f, v in task.FAULTS.items()} == {f: t for t, fs in task.GROUPS.items() for f in fs}
    assert all(len(fs) >= 2 for fs in task.GROUPS.values())             # no test names its fault
    assert task.TESTS == tuple(task.GROUPS) and set(task.COMPETENCIES) <= set(__import__("mrgym.coverage", fromlist=["x"]).ids())


def test_the_standard_scan_is_the_acr_t1_series_and_its_price_is_the_budget_base():
    cfg = _cfg(1, 2)
    proto = task.standard_protocol(cfg["task"])
    assert {k: proto[k] for k in qa.SERIES["t1"]} == qa.SERIES["t1"] and proto["pe_dir"] in ("y", "x")
    pl = console.plan(proto, dict(fov_mm=250.0, min_pixel_mm=0.8, b1rms_limit_ut=6.0, b0_t=1.5))
    assert pl["accept"] and pl["scan_s"] == task.STANDARD_S
    extra = {lv: d["budget_s"] - task.STANDARD_S for lv, d in task.LEVELS.items()}
    swap = console.plan(dict(controls.SWAP, pe_dir="x"), dict(fov_mm=250.0, min_pixel_mm=0.8, b1rms_limit_ut=6.0, b0_t=1.5))["scan_s"]
    flip = console.plan(dict(controls.FLIP_MAP), dict(fov_mm=250.0, min_pixel_mm=0.8, b1rms_limit_ut=6.0, b0_t=1.5))["scan_s"]
    assert extra[1] > extra[2] >= swap + flip and max(swap, flip) <= extra[3] < swap + flip     # level 3: one of the two, not both


def test_an_instance_is_a_function_of_the_seed_and_keeps_the_world_seed():
    seed = derive_seed(ENV_ID, 3, "train", 3, environ={})
    a, b = task.make_instance(seed, 3, task_id="t"), task.make_instance(seed, 3, task_id="t")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True) and a["seed"] == seed
    assert a["world_spec"]["kind"] == "acr" and a["world_spec"]["acr"] == dict(slice=7) and a["budget_s"] == task.LEVELS[3]["budget_s"]
    assert len(a["hidden"]["faults"]) == 2 and len({task.FAULTS[f]["test"] for f in a["hidden"]["faults"]}) == 2
    public = json.dumps(a["task"]) + task.statement(a) + json.dumps(task.submit_schema(a))
    assert seed not in public and "hidden" not in a["task"] and all(str(round(v, 2)) not in public for v in a["hidden"]["measured"].values() if v > 1)
    for f in task.FAULTS:                                                # the statement lists every fault with the test it moves
        assert f in task.statement(a)
    assert task._draw(seed, 3, 0)["faults"] == task._draw(seed, 3, 5)["faults"]     # a redraw keeps the question
    assert task._draw(seed, 3, 0)["size"] != task._draw(seed, 3, 5)["size"]


def test_verdicts_margins_and_tolerances():
    ok = dict(diameter_y_mm=190.4, diameter_x_mm=189.2, piu_percent=95.0, ghosting_ratio=0.002)
    assert task.verdicts(ok, 1.5) == dict(geometric_accuracy=False, intensity_uniformity=False, ghosting=False)
    assert task.verdicts(dict(ok, diameter_x_mm=193.1), 1.5)["geometric_accuracy"] and task.verdicts(dict(ok, diameter_y_mm=186.9), 3.0)["geometric_accuracy"]
    assert task.verdicts(dict(ok, piu_percent=84.9), 1.5)["intensity_uniformity"] and not task.verdicts(dict(ok, piu_percent=84.9), 3.0)["intensity_uniformity"]
    assert task.verdicts(dict(ok, piu_percent=79.9), 3.0)["intensity_uniformity"] and task.verdicts(dict(ok, ghosting_ratio=0.031), 1.5)["ghosting"]
    assert task.clear_of_limits(ok, 1.5) and not task.clear_of_limits(dict(ok, diameter_x_mm=192.0), 1.5)
    assert not task.clear_of_limits(dict(ok, piu_percent=87.0), 1.5) and task.clear_of_limits(dict(ok, piu_percent=87.0), 3.0)
    assert not task.clear_of_limits(dict(ok, ghosting_ratio=0.03), 1.5) and task.clear_of_limits(dict(ok, ghosting_ratio=0.05), 1.5)
    assert task.within("diameter_x_mm", 190.6, 190.0) and not task.within("diameter_x_mm", 190.7, 190.0)
    assert task.within("piu_percent", 93.0, 90.0) and not task.within("piu_percent", 93.2, 90.0)
    assert task.within("ghosting_ratio", 0.006, 0.0) and not task.within("ghosting_ratio", 0.007, 0.0)
    assert task.within("ghosting_ratio", 0.115, 0.1) and not task.within("ghosting_ratio", 0.116, 0.1)
    for key in ("diameter_mm", "piu_percent"):                           # a value inside its tolerance keeps its verdict
        assert task.MARGIN[key] > task.TOLERANCE[key]


@pytest.mark.parametrize("level,seed", [(1, 2), (2, 5), (3, 3)])
def test_the_reference_passes_through_the_tools_and_the_shortcuts_do_not(level, seed):
    played = {}
    for name in ("reference", "brute_force", "wrong_units", "no_discrimination", "prior"):
        env = mrgym.make(ENV_ID, level=level, seed=seed)
        r = policies.run(env, name)
        played[name] = (r.reward, r.info["score"], r.info["policy_error"], env.state()["scan_time_used_s"], env.episode_config())
        env.close()
    reward, score, error, used, cfg = played["reference"]
    assert reward == 1.0 and error is None and score["error"] < 0.7 and used <= cfg["budget_s"]
    assert score["failed_tests"] == sorted({task.FAULTS[f]["test"] for f in cfg["hidden"]["faults"]})
    for name in ("brute_force", "wrong_units", "prior"):
        assert played[name][0] == 0.0 and played[name][2] is None, name
    assert played["brute_force"][1]["checks"]["faults"] is False and all(v for k, v in played["brute_force"][1]["checks"].items() if k != "faults")
    assert played["prior"][1]["checks"]["measured"] is False and played["prior"][3] == 0.0
    assert not any(played["wrong_units"][1]["checks"][f"measured_diameter_{a}_mm"] for a in "yx")
    assert played["no_discrimination"][2] is None and played["no_discrimination"][3] == task.STANDARD_S     # no second scan


def test_tune_gives_the_most_frequent_answer_and_the_controls_are_described():
    cfgs = [_cfg(1, k) for k in (0, 1, 2, 6)]
    r = controls.tune(cfgs)
    assert r["faults"] == ["b1_transmit"] and r["failed_tests"] == ["intensity_uniformity"] and r["tuned_share"] == 0.75
    assert set(r["measured"]) == set(task.QUANTITIES) and sorted(r["order"]) == ["flip_map", "swap"]
    names = [controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED]
    assert set(names) == set(controls.policies()) == set(controls.DESCRIPTIONS)
    assert controls.MUST_FAIL == ("prior", "fixed_recipe", "brute_force", "wrong_units") and controls.REPORTED == ("open_loop",)
    assert controls.ABLATIONS == {"no_discrimination": (1, 2, 3), "single_fault": (3,)}


def test_the_label_of_a_level_is_read_from_a_gate_result():
    level = lambda ref, ol: dict(n=60, reference=dict(rate=ref), reported=[dict(policy="open_loop", rate=ol)])     # noqa: E731
    k = controls.task_kind(dict(command="c", engine="e", date_utc="d", levels={"1": level(0.98, 0.95), "3": level(0.97, 0.60)}))
    assert k["levels"]["1"]["kind"] == "measurement" and k["levels"]["3"]["kind"] == "decision" and k["levels"]["3"]["gap_points"] == 37.0
    stored = controls.stored_gate()
    assert stored is not None and stored["env_id"] == ENV_ID and stored["engine"] == mrgym.ENGINE_ID, "ship gate_dev.json of the current engine"
    card = task.card()
    assert card["id"] == ENV_ID and set(card["task_kind"]["levels"]) == {"1", "2", "3"} and card["competencies"] == list(task.COMPETENCIES)
    assert all(v["kind"] in ("measurement", "decision") for v in card["task_kind"]["levels"].values())
