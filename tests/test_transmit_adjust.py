"""``Scanner/TransmitAdjust-v0``: what the scoring table (``tests/scoring/transmit_adjust.py``) does not cover.

  * the central formula, checked independently: a pulse-by-pulse iteration of the dual-TR steady state against the
    closed form of the signal and against the ratio formula the policies invert
  * which measurement is valid on this scanner: with a known transmit gain the non-selective scan with strong
    spoiling returns it, the slice-selective scan reads far too low, the default spoiling reads high for a long T2
  * an instance is a function of the seed, also from threads; what is told holds; what is hidden is not in the text
  * the reference passes within the budget, every shortcut fails, the ablation fails where its step matters
  * a refused deliverable does not end the episode
"""
import concurrent.futures
import json
import math
import re

import numpy as np
import pytest

import mrgym
from mrgym import coverage, policies
from mrgym.env import derive_seed
from mrgym.envs.transmit_adjust import controls, task
from mrgym.scanner import console, kspace, session
from mrgym.scanner import signal as S
from mrgym.scanner import world as W

ENV_ID = "Scanner/TransmitAdjust-v0"
GEL = dict(pd=1.0, t1_ms=1000.0, t2_ms=100.0, t2s_ms=60.0, t2p_ms=150.0, adc_um2_ms=1.5, fat=False)
LONG_T2 = dict(GEL, t1_ms=900.0, t2_ms=190.0, t2s_ms=100.0, t2p_ms=211.0, adc_um2_ms=0.8)
CIRCLE = [dict(id="A", y_mm=0.0, x_mm=0.0, radius_mm=12.0)]
EXAM = dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=3.0, b0_t=3.0)


def _instance(level, seed, split="train"):
    return task.make_instance(derive_seed(ENV_ID, level, split, seed, environ={}), level, task_id="t")


# ---------------------------------------------------------------------------------------------- the physics, independently
def _dual_tr_steady_state(a, tr1, tr2, t1, cycles=4000):
    """Signals (S1, S2) of the dual-TR sequence by following Mz pulse by pulse (ideal spoiling: every pulse acts on
    the longitudinal magnetisation alone and leaves Mz cos a; the signal is Mz sin a): no closed form used."""
    mz = 1.0
    for _ in range(cycles):
        s1 = mz * math.sin(a)                                            # the pulse that starts the short interval
        mz = 1.0 + (mz * math.cos(a) - 1.0) * math.exp(-tr1 / t1)
        s2 = mz * math.sin(a)                                            # the pulse that starts the long interval
        mz = 1.0 + (mz * math.cos(a) - 1.0) * math.exp(-tr2 / t1)
    return s1, s2


def test_the_dual_tr_formula_against_a_pulse_by_pulse_iteration():
    n = 5
    for deg in (25.0, 45.0, 60.0, 75.0, 95.0, 120.0):
        a = math.radians(deg)
        # the closed form the scanner's ideal-spoiling limit uses
        s1, s2 = _dual_tr_steady_state(a, 25.0, 125.0, 1000.0)
        assert S.afi(a, 25.0, 125.0, 1000.0) == pytest.approx((s1, s2), rel=1e-9)
        # repetition times much shorter than T1: r = (1 + n cos a) / (n + cos a), and the policies' inversion returns a
        s1, s2 = _dual_tr_steady_state(a, 0.0025, 0.0125, 1000.0)
        assert s2 / s1 == pytest.approx((1 + n * math.cos(a)) / (n + math.cos(a)), abs=1e-4)
        assert float(controls.flip_ratio(s1, s2, n, 60.0)) * 60.0 == pytest.approx(deg, abs=0.02)
    # at the repetition times of the stated protocol the approximation reads a little low, more for a short T1
    for t1, worst in ((2500.0, 0.005), (1000.0, 0.012), (300.0, 0.035)):
        low = [float(controls.flip_ratio(*_dual_tr_steady_state(math.radians(d), 25.0, 125.0, t1), n, d)) for d in (30.0, 60.0, 90.0)]
        assert all(1.0 - worst < v < 1.0 for v in low), (t1, low)
    assert controls.flip_ratio(1.0, -0.2, 5, 60.0) > controls.flip_ratio(1.0, 0.2, 5, 60.0)      # a negative S2 is a larger angle


def _factor(material, gain, protocol):
    """The dual-TR formula on a noise-free scan of a uniform disc with transmit gain ``gain``, relative to ``gain``."""
    ph = W.shapes_phantom([("gel", lambda y, x: np.hypot(y, x) - 80.0)], b0_t=3.0, n=256, tissues=dict(gel=material))
    world = W.flat_world(ph, tx_gain=gain)
    pl = console.plan(protocol, world["console"])
    assert pl["accept"], pl["violations"]
    rp = pl["realised"]
    row = session.roi_readouts(kspace.Engine(world).acquire(rp, None), CIRCLE, session.fov_yx(rp), session.centre_yx(rp), 0.0)[0]
    return float(controls.flip_ratio(*controls._signed(dict(roi_readouts=[row])), rp["tr_ratio"], rp["flip_deg"])[0]) / gain


def test_which_measurement_returns_the_transmit_gain_on_this_scanner():
    """The content of the task, on worlds with a known gain. The non-selective scan with strong spoiling is right
    within 1 %; the same scan slice-selective reads a quarter to 40 % low (still more than a tenth with the sharpest
    pulse); with the
    console's default spoiler the non-selective scan reads high in a material with a long T2."""
    default = {k: v for k, v in task.METHOD_SCAN.items() if k not in ("spoiler_mt_ms_per_m", "rf_phase_increment_deg")}
    for gain in (0.75, 1.25):
        for material in (GEL, LONG_T2):
            assert _factor(material, gain, dict(task.METHOD_SCAN)) == pytest.approx(1.0, abs=0.01)
            assert _factor(material, gain, dict(task.METHOD_SCAN, partial_fourier=controls.PARTIAL_FOURIER[-1])) == pytest.approx(1.0, abs=0.01)
        assert 0.60 < _factor(GEL, gain, dict(task.METHOD_SCAN, excitation="selective")) < 0.75
        assert 0.78 < _factor(GEL, gain, dict(task.METHOD_SCAN, excitation="selective", rf_pulse="sharp", tr_ms=30.0)) < 0.90
    assert _factor(LONG_T2, 0.75, default) > 1.0 + task.TOLERANCE                       # beyond the tolerance
    assert 1.0 < _factor(GEL, 1.25, default) < 1.0 + task.TOLERANCE                     # inside it: the bias depends on the material


# ---------------------------------------------------------------------------------------------- instances
def test_an_instance_is_a_function_of_the_seed_also_from_threads():
    jobs = [(lv, s) for lv in task.LEVELS for s in range(4)]
    serial = [json.dumps(_instance(*j), sort_keys=True) for j in jobs]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        threaded = list(pool.map(lambda j: json.dumps(_instance(*j), sort_keys=True), jobs * 2))
    assert threaded == serial * 2
    a, b = (task.make_instance("0f" * 16, 2)["task_id"] for _ in range(2))
    assert re.fullmatch(r"transmitadjust-[0-9a-f]{12}", a) and a != b


def test_what_the_task_tells_holds_on_instances():
    """Every factor lies inside the told bound, away from 1 (the transmitter is miscalibrated) and inside the range in
    which the dual-TR signal ratio stays positive at the stated flip angle; the regions of an instance differ."""
    assert task.TOLERANCE == 0.04 and task.RATIO_BOUND == (0.4, 1.7)
    spread = {lv: [] for lv in task.LEVELS}
    for level in task.LEVELS:
        for seed in range(12):
            cfg = _instance(level, seed, "dev")
            values = list(task.truth(cfg).values())
            assert 5 <= len(values) <= 7                                    # every vial is marked
            assert task.RATIO_BOUND[0] + 0.05 < min(values) and max(values) < task.RATIO_BOUND[1] - 0.05
            assert max(values) * task.METHOD_SCAN["flip_deg"] < 100.0
            spread[level].append(max(values) / min(values) - 1.0)
            assert cfg["task"]["tolerance"] == 0.04 and cfg["task"]["ratio_bound"] == [0.4, 1.7]
    assert all(np.median(v) > 2 * task.TOLERANCE for v in spread.values())               # one value for every region does not pass
    assert np.median(spread[3]) > np.median(spread[1])
    world = W.make_world(cfg["seed"], cfg["world_spec"])                                 # the truth is the region mean of world.b1_at's product
    for r in cfg["task"]["regions"]:
        assert task.truth(cfg, world)[r["id"]] == pytest.approx(W.b1_at(world, r["y_mm"], r["x_mm"]), rel=0.01)


def test_the_statement_states_the_rule_and_nothing_hidden():
    for level in task.LEVELS:
        cfg = _instance(level, 3)
        text, truth = task.statement(cfg), task.truth(cfg)
        world = W.make_world(cfg["seed"], cfg["world_spec"])
        public = text + json.dumps(cfg["task"])
        for hidden in [world["system"]["tx_gain"]] + list(truth.values()):
            assert f"{hidden:.3f}" not in public
        assert "cls" not in json.dumps(cfg["task"]) and "seed" not in cfg["task"]
        assert "within 4 % of the true factor" in text and "(between 0.96 x and 1.04 x the true factor)" in text
        assert "factor = actual flip angle / nominal flip angle" in text and "between 0.4 and 1.7" in text
        assert ("How to do it" in text) is (level == 1) and ("nonselective" in text) is (level == 1)
        assert text.count("An answer in the wrong unit is wrong.") == 1 and "The scanner's log must show that you measured." in text
        assert task.submit_schema(cfg)["properties"]["flip_ratio"]["required"] == [r["id"] for r in cfg["task"]["regions"]]
    hint = task.hint_text(_instance(1, 0))
    for word in ("excitation nonselective", "tr_ms 25", "tr_ratio 5", "flip_deg 60", "spoiler_mt_ms_per_m 585", "rf_phase_increment_deg 45",
                 "arccos((r x n - 1) / (n - r))", "/ 60 degrees"):
        assert word in hint


def test_the_budget_pays_the_stated_protocol_at_levels_1_and_2_and_not_at_level_3():
    cost = {pf: console.plan(dict(task.METHOD_SCAN, partial_fourier=pf), EXAM)["scan_s"] for pf in controls.PARTIAL_FOURIER}
    assert console.plan(dict(task.METHOD_SCAN), EXAM)["realised"]["partitions"] == 16
    assert cost[1.0] <= task.LEVELS[1]["budget_s"] == task.LEVELS[2]["budget_s"] < 2 * min(cost.values())    # once, not twice
    assert cost[1.0] > task.LEVELS[3]["budget_s"] >= cost[0.7] > cost[0.6]
    assert console.plan(dict(task.METHOD_SCAN, excitation="selective"), EXAM)["scan_s"] < cost[1.0] / 5     # the tempting 2D scan


# ---------------------------------------------------------------------------------------------- policies
def _play(level, seed, name, recipe=None):
    env = mrgym.make(ENV_ID, level=level, seed=seed)
    r = policies.run(env, name, recipe=recipe)
    env.close()
    return r


@pytest.mark.parametrize("level", sorted(task.LEVELS))
def test_the_reference_passes_within_the_budget_and_every_shortcut_fails(level):
    for seed in (0, 1):
        r = _play(level, seed, "reference")
        s = r.info["score"]
        assert r.reward == 1.0 and s["outcome"] == "pass" and s["error"] < 0.85 * task.TOLERANCE, s
        assert r.info["budget"]["used_s"] <= task.LEVELS[level]["budget_s"] and r.info["budget"]["scans"] == 1 and r.info["policy_error"] is None
    for name in controls.MUST_FAIL:
        r = _play(level, 0, name)
        assert r.reward == 0.0 and r.info["policy_error"] is None, name
    r = _play(level, 0, "afi_2d")                                        # the 2D shortcut: every region far too low
    assert r.info["score"]["outcome"] == "no_region_within" and all(-0.45 < m < -0.20 for m in r.info["score"]["per_region"].values())
    assert _play(level, 0, "wrong_units").info["reason"] == "deliverable_refused"
    assert _play(level, 0, "open_loop").reward == 1.0


def test_the_ablation_fails_where_its_step_matters():
    assert controls.ABLATIONS == {"untrimmed": (3,)} and controls.REPORTED == ("open_loop", "default_spoiling")
    assert _play(1, 0, "untrimmed").reward == 1.0                        # the stated protocol, where the budget pays it
    r = _play(3, 0, "untrimmed")
    assert r.reward == 0.0 and r.info["policy_error"].startswith("BudgetExhausted") and r.info["score"]["outcome"] == "no_measurement"


def test_the_blind_recipe_is_a_constant_times_a_bowl_at_the_told_positions():
    cfgs = [_instance(2, f"0:{i}", "tune") for i in range(4)]
    recipe = controls.tune(cfgs)
    assert set(recipe) == {"scale", "bowl", "tuned_pass_rate", "tuned_region_pass_rate"}
    hits = [abs(controls._bowl(recipe, r) / task.truth(c)[r["id"]] - 1.0) <= task.TOLERANCE for c in cfgs for r in c["task"]["regions"]]
    assert recipe["tuned_region_pass_rate"] == round(float(np.mean(hits)), 3) and recipe["tuned_region_pass_rate"] > 0.0
    assert controls._bowl(dict(scale=0.9, bowl=0.2), dict(y_mm=0.0, x_mm=0.0)) == pytest.approx(0.9 * 1.1)
    assert controls._bowl(dict(scale=0.9, bowl=0.2), dict(y_mm=72.0, x_mm=0.0)) == pytest.approx(0.9 * 0.9)
    assert _play(2, 0, "fixed_recipe", recipe).reward == 0.0
    assert set(controls.policies()) == {controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED} == set(controls.DESCRIPTIONS)


def test_a_refused_deliverable_does_not_end_the_episode():
    env = mrgym.make(ENV_ID, level=1, seed=0)
    env.reset()
    sc = policies.ToolScanner(env)
    answer = controls.reference(sc)
    used = sc.budget()["used_s"]
    percent = {"flip_ratio": {i: 100.0 * v for i, v in answer["flip_ratio"].items()}}
    for bad in (percent, {"flip_ratio": {"R1": 1.0}}, {"flip_ratio": dict(answer["flip_ratio"], R1="high")}):
        obs, reward, terminated, truncated, _ = env.step({"name": "submit", "arguments": bad})
        assert obs.is_error and reward == 0.0 and not terminated and not truncated and sc.budget()["used_s"] == used
    obs, reward, terminated, _, info = env.step({"name": "submit", "arguments": answer})
    assert terminated and reward == 1.0 and info["score"]["outcome"] == "pass"
    env.close()


def test_outcome_and_competencies():
    assert task.OUTCOMES == ("no_region_within", "some_regions_within")
    ok = dict(deliverable_valid=True)
    assert task.outcome(dict(checks=dict(ok, ratio_within_tolerance_R1=False, ratio_within_tolerance_R2=False))) == "no_region_within"
    assert task.outcome(dict(checks=dict(ok, ratio_within_tolerance_R1=False, ratio_within_tolerance_R2=True))) == "some_regions_within"
    assert set(task.COMPETENCIES) <= set(coverage.ids()) and "qc.transmit_gain" in task.COMPETENCIES
