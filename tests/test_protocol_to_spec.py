"""``Scanner/ProtocolToSpec-v0``: the instances, the oracle, the verifier, the budget arithmetic and the controls."""
import json
import math

import numpy as np
import pytest

import mrgym
from mrgym import policies
from mrgym.envs import oracle as O
from mrgym.envs.protocol_to_spec import controls, oracle, task
from mrgym.scanner import console, kspace
from mrgym.scanner import world as W

ENV = "Scanner/ProtocolToSpec-v0"
LEVEL_OF = {"ti": 1, "te": 2}                                           # the level on which each field is the one that moves


@pytest.fixture(scope="module", params=["ti", "te"])
def instance(request):
    """One instance per field that moves, with its hidden world, engine and oracle solution (verifier side)."""
    env = mrgym.make(ENV, level=LEVEL_OF[request.param], seed="test:1")
    env.reset()
    cfg = env.episode_config()
    world = W.make_world(cfg["seed"], cfg["world_spec"])
    return request.param, cfg, world, kspace.Engine(world), oracle.solve(cfg, world)


def _run(level, seed, name):
    env = mrgym.make(ENV, level=level, seed=seed)
    result = policies.run(env, name)
    env.close()
    return result


# ------------------------------------------------------------------ instances
def test_the_specification_is_the_same_for_every_instance_and_the_noise_is_hidden(instance):
    knob, cfg, world, _, _ = instance
    t = cfg["task"]
    for q in t["images"].values():
        assert (q["t1_contrast"]["min_ratio"], q["t2_contrast"]["min_ratio"], q["snr"]["min"]) == (task.T1_RATIO, task.T2_RATIO, task.SNR_MIN)
        assert world["tissues"][_cls(world, q["t2_contrast"]["brighter"])]["t2_ms"] > world["tissues"][_cls(world, q["t2_contrast"]["darker"])]["t2_ms"]
    public = json.dumps(t) + task.statement(cfg)
    assert "sigma0" not in public and "centre frequency, transmit gain, shim" in task.statement(cfg) and t["scan_time_max_s"] > 0
    assert world["console"]["system"] == task.CONSOLE and cfg["budget_s"] == task.BUDGET_S
    seed = mrgym.env.derive_seed(ENV, cfg["level"], "train", "test:1", environ={})       # the instance is a function of its seed
    again = task.make_instance(seed, cfg["level"], "x")
    assert (again["seed"], again["world_spec"], again["task"]) == (cfg["seed"], cfg["world_spec"], cfg["task"])
    assert json.dumps(again, sort_keys=True) == json.dumps(task.make_instance(seed, cfg["level"], "x"), sort_keys=True)


def _cls(world, region):
    return next(r["cls"] for r in world["regions"] if r["id"] == region)


def test_one_field_has_to_move_and_the_other_cannot_repair_the_image(instance):
    """The feasible protocols of each image: a range of the field that moves at the standard value of the other field,
    ``WINDOW_OK`` wide, and nothing with the field that moves left at its standard value."""
    knob, cfg, world, _, solution = instance
    for w in solution["windows"].values():
        along, across = (w["ti_ms"], w["te_ms"]) if knob == "ti" else (w["te_ms"], w["ti_ms"])
        assert across is None and task.WINDOW_OK[0] - 0.05 <= along[1] / along[0] <= task.WINDOW_OK[1] + 0.6     # at the oracle's averages
        assert not along[0] <= oracle.STANDARD[f"{knob}_ms"] <= along[1]
    assert controls.moving_field(cfg) == knob


# ------------------------------------------------------------------ oracle and verifier
def test_the_oracle_is_independent_of_the_control_policies():
    assert O.independence_problems("mrgym.envs.protocol_to_spec") == []


def test_the_oracle_answer_passes_the_verifier_and_the_standard_protocol_does_not(instance):
    knob, cfg, world, engine, solution = instance
    assert solution["value"] < 0 and solution["method"] == "exhaustive"
    score = task.score(cfg, solution["argmin"], world, engine)
    assert score["passed"] and score["error"] == pytest.approx(solution["value"], abs=0.02)      # the model against the engine scan
    standard = task.score(cfg, {k: dict(oracle.STANDARD) for k in task.IMAGES}, world, engine)
    failed = {k for k, ok in standard["checks"].items() if not ok}
    binding = "t1_contrast" if knob == "ti" else "t2_contrast"
    assert {f"{binding}_A", f"{binding}_B"} <= failed and standard["error"] > 0
    assert not any(k.startswith("t2_contrast" if knob == "ti" else "t1_contrast") for k in failed)


def test_the_certificate_protocols_fail_the_check_they_name(instance):
    _, cfg, world, engine, solution = instance
    assert solution["certificate"]
    for c in solution["certificate"]:
        answer = dict(solution["argmin"], **c["deliverable"])
        assert task.score(cfg, answer, world, engine)["checks"][c["fails"]] is False


def test_the_signal_model_of_the_oracle_matches_noise_free_scans(instance):
    _, cfg, world, engine, _ = instance
    model = oracle.Model(world, engine)
    regions = cfg["task"]["regions"]
    for ti, te in ((oracle.TI_GRID[30], oracle.TE_GRID[0]), (oracle.TI_GRID[150], oracle.TE_GRID[40]), (oracle.TI_GRID[220], oracle.TE_GRID[70])):
        rp = console.plan(oracle.protocol(dict(ti_ms=ti, te_ms=te, averages=1)), world["console"])["realised"]
        scan = {i: abs(v) for i, v in task.F.region_complex(engine, rp, regions).items()}
        top = max(scan.values())
        for i, v in scan.items():
            k, j = int(np.argmin(np.abs(oracle.TE_GRID - te))), int(np.argmin(np.abs(oracle.TI_GRID - ti)))
            assert abs(abs(model.sig[i][k, j]) - v) <= 0.015 * top, (i, ti, te)


def test_the_pass_rule_at_its_boundaries():
    assert task.meets(4.0, 4.0) and not task.meets(3.999, 4.0)
    assert task.in_time(120.0, 120.0) and not task.in_time(120.01, 120.0)


def test_a_deliverable_the_console_cannot_run_is_refused(instance):
    _, cfg, world, _, solution = instance
    good = solution["argmin"]
    assert task.check(cfg, good, world["console"]) is None
    for bad in (dict(good, A=dict(good["A"], ti_ms=good["A"]["ti_ms"] / 1000.0)), dict(good, A=dict(good["A"], ti_ms=395.0)),
                {"A": good["A"]}, dict(good, A=dict(good["A"], tr_ms=400.0)), dict(good, B=dict(good["B"], averages=2.5)), None):
        assert isinstance(task.check(cfg, bad, world["console"]), str)


# ------------------------------------------------------------------ the budget
def test_a_console_can_price_every_scan(qa_world):
    """The engine change this family rests on: ``exam["system"]`` (world spec ``console_system``) replaces constants of
    the console for one scanner; here the overhead before every scan. Other worlds have no such entry."""
    exam, p = qa_world["console"], dict(sequence="se", tr_ms=300.0)
    assert "system" not in exam
    priced = console.plan(p, dict(exam, system=dict(overhead_s=task.PRICE_S)))
    assert priced["scan_s"] - console.plan(p, exam)["scan_s"] == pytest.approx(task.PRICE_S - console.SYSTEM["overhead_s"])
    env = mrgym.make(ENV, level=1, seed="test:1")
    env.reset()
    assert policies.ToolScanner(env).info()["cost"]["overhead_s_per_scan"] == task.PRICE_S
    env.close()


def test_the_budget_pays_for_two_scans_and_measuring_everything_needs_three():
    env = mrgym.make(ENV, level=3, seed="test:budget")
    env.reset()
    sc = policies.ToolScanner(env)
    cost = controls.battery_cost(sc)
    assert len(cost["scans"]) == 3 and cost["ratio"] > 1.25              # three scans at the reference's resolution do not fit
    assert cost["scans"]["standard"] + cost["scans"]["echo_time"] <= cost["budget_s"]
    t = controls._task(sc)
    first = sc.plan(controls._protocol(t))["scan_s"]
    sc.scan(controls._protocol(t), t["regions"])
    assert first - sc.plan(controls._protocol(t, te_ms=60.0))["scan_s"] == pytest.approx(task.CONSOLE["overhead_s"] - task.CONSOLE["repeat_overhead_s"])
    assert sc.plan(dict(controls._protocol(t), fov_mm=380.0))["scan_s"] == pytest.approx(first)   # another field of view: adjusted again
    env.close()


def test_measuring_everything_is_refused_by_the_budget():
    result = _run(1, "test:1", "characterise_all")
    assert result.reward == 0.0 and "BudgetExhausted" in result.info["policy_error"]


# ------------------------------------------------------------------ the controls
def test_the_fits_recover_what_the_model_was_given():
    ti = np.array([80.0, 160.0])
    for t1 in (60.0, 180.0, 420.0, 760.0):
        y = 3.0 * controls.ir_shape(ti, 12.0, t1, 400.0)
        phase = np.exp(0.7j)
        scans = [dict(ti=float(a), te=12.0, tr=400.0, tbw=4.0, z={"R1": complex(b) * phase}) for a, b in zip(ti, y)]
        fit = controls._fit_ti(*scans)["R1"]
        assert fit["t1"] == pytest.approx(t1, rel=0.02) and fit["amp"] == pytest.approx(3.0, rel=0.02)
    a, b = dict(te=12.0, z={"R1": 2.0 + 0j}), dict(te=60.0, z={"R1": 2.0 * math.exp(-48.0 / 80.0) + 0j})
    assert controls._fit_te(a, b)["R1"] == pytest.approx(1.0 / 80.0)


@pytest.mark.parametrize("knob", ["ti", "te"])
def test_the_reference_passes_and_the_fixed_second_scan_of_the_other_level_does_not(knob):
    level, wrong = LEVEL_OF[knob], {"ti": "always_te", "te": "always_ti"}[knob]
    good = _run(level, "test:1", "reference")
    assert good.reward == 1.0 and good.info["policy_error"] is None and good.info["budget"]["used_s"] <= task.BUDGET_S
    assert _run(level, "test:1", {"ti": "always_ti", "te": "always_te"}[knob]).reward == 1.0
    assert _run(level, "test:1", wrong).reward == 0.0


def test_answers_that_ignore_the_object_fail():
    assert _run(1, "test:1", "prior").info["score"]["checks"]["measured"] is False
    refused = _run(1, "test:1", "wrong_units")
    assert refused.reward == 0.0 and refused.info["submitted"] is False and "below the minimum" in refused.info["refused"]
    env = mrgym.make(ENV, level=1, seed="test:1")
    env.reset()
    answers = controls.blind_answers(env.episode_config()["task"])
    assert len(answers) == 20 and all(set(a) == set(task.IMAGES) for a in answers.values())
    env.close()


def test_the_card_rule_of_the_open_loop_is_tuned_on_the_tuning_split():
    rule = controls.card_table(1, 3)
    assert rule["majority"] == "ti" and rule["worlds"] == 3 and rule["accuracy"] == 1.0 and rule["threshold"] is None


def test_the_kind_of_task_is_read_from_a_gate_result():
    def level(ref, open_loop):
        return dict(n=60, reference=dict(rate=ref), reported=[], ablations=[dict(policy="always_ti", rate=open_loop), dict(policy="open_loop_all", rate=0.1)])
    stored = dict(engine="x", levels={"1": level(0.95, 0.95), "2": level(0.97, 0.95), "3": level(0.95, 0.5)})
    kind = task.task_kind(stored)
    assert kind["kind"] == "decision" and kind["levels"] == {"1": "measurement", "2": "measurement", "3": "decision"}
    stored["levels"]["3"] = level(0.95, 0.8)
    assert task.task_kind(stored)["kind"] == "measurement"
    assert set(controls.ABLATIONS) | set(controls.MUST_FAIL) | set(controls.REPORTED) | {controls.REFERENCE} == set(controls.policies())
    assert controls.LADDER[3] == controls.REFERENCE and task.card()["id"] == ENV
