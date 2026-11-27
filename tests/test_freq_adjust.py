"""``Scanner/FrequencyAdjust-v0``: what the scoring table (``tests/scoring/freq_adjust.py``) does not cover.

  * the central formula, checked independently: the phase difference of two echoes gives the offset modulo
    1 / (echo-time difference), on a closed form and on the engine with a known centre-frequency error
  * the line of oil lies where the reference's species step expects it
  * an instance is a function of the seed, also from threads; what is told holds; what is hidden is not in the text
  * the reference passes, every shortcut fails, each ablation fails where its step matters
  * a refused deliverable does not end the episode
"""
import concurrent.futures
import json
import math
import re

import numpy as np
import pytest

import mrgym
from mrgym import policies
from mrgym.env import derive_seed
from mrgym.envs.freq_adjust import controls, task
from mrgym.scanner import console, kspace, session
from mrgym.scanner import world as W

ENV_ID = "Scanner/FrequencyAdjust-v0"
GEL = dict(pd=1.0, t1_ms=1000.0, t2_ms=100.0, t2s_ms=60.0, t2p_ms=150.0, adc_um2_ms=1.5, fat=False)
CIRCLES = [dict(id="A", y_mm=0.0, x_mm=0.0, radius_mm=12.0), dict(id="B", y_mm=30.0, x_mm=-20.0, radius_mm=12.0)]


def _instance(level, seed, split="train"):
    return task.make_instance(derive_seed(ENV_ID, level, split, seed, environ={}), level, task_id="t")


def _disc(tissue="gel", **system):
    """A disc of one material in a perfectly uniform field, with a chosen state of the system (no noise)."""
    ph = W.shapes_phantom([(tissue, lambda y, x: np.hypot(y, x) - 80.0)], b0_t=3.0, n=256, tissues=dict(gel=GEL))
    return W.flat_world(ph, **system)


def _read(world, protocol):
    """Noise-free complex means of ``CIRCLES`` for a protocol: (realised protocol, {id: one value per echo})."""
    pl = console.plan(protocol, world["console"])
    assert pl["accept"], pl["violations"]
    rp = pl["realised"]
    rows = session.roi_readouts(kspace.Engine(world).acquire(rp, None), CIRCLES, session.fov_yx(rp), session.centre_yx(rp), 0.0)
    return rp, {r["id"]: np.array(r["complex_mean_magnitude"]) * np.exp(1j * np.array(r["complex_mean_phase_rad"])) for r in rows}


# ---------------------------------------------------------------------------------------------- the physics, independently
def test_two_echoes_give_the_offset_modulo_one_over_their_distance_closed_form():
    """s(t) = A exp(i phi0) exp(2 pi i f t) exp(-t / T2*): the receive phase phi0 and the decay drop out of the phase
    difference; the estimate is f for |f| < 1 / (2 dTE) and f - k / dTE beyond."""
    rng = np.random.default_rng(5)
    for f in (-249.0, -41.5, 0.0, 12.25, 249.0):                         # inside +-250 Hz at 2 ms
        t = np.array([2.2, 4.2])
        z = 0.7 * np.exp(1j * rng.uniform(0, 2 * np.pi)) * np.exp(2j * np.pi * f * t * 1e-3) * np.exp(-t / 45.0)
        assert controls.ladder(t, z) == pytest.approx([f], abs=1e-9)
    for f, alias in ((251.0, -249.0), (420.0, -80.0), (-600.0, -100.0), (1000.0, 0.0)):   # beyond: off by a multiple of 500 Hz
        t = np.array([2.2, 4.2])
        z = np.exp(2j * np.pi * f * t * 1e-3)
        assert controls.ladder(t, z) == pytest.approx([alias], abs=1e-9)
    # the ladder: 0.3 ms covers +-1667 Hz; each longer difference is unwrapped with the estimate before it
    for f in (-1500.0, -600.0, 37.0, 599.0, 1034.0):
        t = np.array([2.18, 2.48, 4.05, 5.62])
        z = np.exp(1j * 0.4) * np.exp(2j * np.pi * f * t * 1e-3)
        assert controls.ladder(t, z) == pytest.approx([f, f, f], abs=1e-6)
    assert controls.ladder([1.0, 1.3], np.exp(2j * np.pi * 1700.0 * np.array([1.0, 1.3]) * 1e-3)) == pytest.approx([1700.0 - 1e3 / 0.3], abs=1e-6)


def test_the_engine_puts_the_centre_frequency_error_into_the_echo_phase_as_the_formula_says():
    """A uniform disc with a known centre-frequency error: the stated two-echo method returns it (sign included),
    and wraps exactly where the closed form says. The engine's phase is exp(+2 pi i f TE) on the receive phase."""
    for f0, want in ((-180.0, -180.0), (37.5, 37.5), (240.0, 240.0), (420.0, 420.0 - 500.0)):
        rp, z = _read(_disc(cf_offset_hz=f0), dict(task.METHOD_SCAN))
        assert rp["echo_times_ms"][1] - rp["echo_times_ms"][0] == pytest.approx(2.0, abs=1e-9)
        for i in ("A", "B"):
            assert controls.ladder(rp["echo_times_ms"], z[i])[-1] == pytest.approx(want, abs=0.05)
    # the reference's two scans on the same world: the 0.3 ms step between them resolves 420 Hz and -900 Hz
    for f0 in (420.0, -900.0):
        world = _disc(cf_offset_hz=f0)
        te0 = console.plan(dict(controls.PROBE), world["console"])["realised"]["te_ms"]
        rp_a, z_a = _read(world, dict(controls.PROBE, te_ms=te0))
        rp_b, z_b = _read(world, dict(controls.PROBE, te_ms=te0 + controls.SHORT_MS, n_echoes=controls.N_ECHOES))
        times = rp_a["echo_times_ms"] + rp_b["echo_times_ms"]
        assert times[1] - times[0] == pytest.approx(controls.SHORT_MS, abs=1e-9)
        assert controls.ladder(times, np.concatenate([z_a["A"], z_b["A"]])) == pytest.approx([f0] * 3, abs=0.05)


def test_oil_reads_about_the_water_fat_shift_below_water():
    """What the species step of the reference relies on: on a disc of fat the coarse estimate lies between half the
    shift and the shift plus ``OIL_SPREAD_HZ`` below the water offset (the fat signal has several lines)."""
    shift = controls.fat_shift_hz(3.0)
    assert shift == pytest.approx(3.4e-6 * 42.577478518e6 * 3.0) and shift == pytest.approx(434.3, abs=0.1)
    world = _disc("subcut_fat", cf_offset_hz=100.0)
    te0 = console.plan(dict(controls.PROBE), world["console"])["realised"]["te_ms"]
    _, z_a = _read(world, dict(controls.PROBE, te_ms=te0))
    _, z_b = _read(world, dict(controls.PROBE, te_ms=te0 + controls.SHORT_MS))
    coarse = controls.ladder([te0, te0 + controls.SHORT_MS], np.concatenate([z_a["A"], z_b["A"]]))[0]
    assert 0.5 * shift + 50.0 < 100.0 - coarse < shift + controls.OIL_SPREAD_HZ


# ---------------------------------------------------------------------------------------------- instances
def test_an_instance_is_a_function_of_the_seed_also_from_threads():
    jobs = [(lv, s) for lv in task.LEVELS for s in range(4)]
    serial = [json.dumps(_instance(*j), sort_keys=True) for j in jobs]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        threaded = list(pool.map(lambda j: json.dumps(_instance(*j), sort_keys=True), jobs * 2))
    assert threaded == serial * 2
    a, b = (task.make_instance("0f" * 16, 2)["task_id"] for _ in range(2))
    assert re.fullmatch(r"freqadjust-[0-9a-f]{12}", a) and a != b


def test_what_the_task_tells_holds_on_instances():
    """The told bound contains every offset; level 1 never wraps the stated spacing; level 3 marks one or two regions
    of oil and at least one of water under ids that are not the vials' own."""
    assert task.TOLERANCE_HZ == 3.0 and [lv["n_regions"] for lv in task.LEVELS.values()] == [2, 3, 3]
    oil_ids, beyond = [], 0
    for level, lv in task.LEVELS.items():
        for seed in range(12):
            cfg = _instance(level, seed, "dev")
            t, truth = cfg["task"], task.truth(cfg)
            assert [r["id"] for r in t["regions"]] == [f"R{k + 1}" for k in range(lv["n_regions"])]
            assert max(abs(v["offset_hz"]) for v in truth.values()) < 0.95 * t["offset_bound_hz"] == 0.95 * lv["offset_bound_hz"]
            oil = [i for i, v in truth.items() if v["oil"]]
            assert (1 <= len(oil) <= 2 and len(oil) < len(truth)) if level == 3 else not oil
            assert t["oil_possible"] is (level == 3) and t["tolerance_hz"] == 3.0
            oil_ids += oil
            beyond += level == 2 and max(abs(v["offset_hz"]) for v in truth.values()) > 250.0
    assert task.LEVELS[1]["offset_bound_hz"] < 0.5e3 / task.METHOD_SCAN["echo_spacing_ms"] < task.LEVELS[2]["offset_bound_hz"]
    assert beyond >= 3                                                   # level 2: a good part of the instances wraps 2 ms
    assert set(oil_ids) == {"R1", "R2", "R3"}                            # the oil is not at a fixed id


def test_the_statement_states_the_rule_and_nothing_hidden():
    for level in task.LEVELS:
        cfg = _instance(level, 3)
        text, truth = task.statement(cfg), task.truth(cfg)
        world = W.make_world(cfg["seed"], cfg["world_spec"])
        public = text + json.dumps(cfg["task"])
        for hidden in [world["system"]["cf_offset_hz"]] + [v["offset_hz"] for v in truth.values()]:
            assert f"{hidden:.1f}" not in public and f"{hidden:.0f} Hz" not in public
        assert "cls" not in json.dumps(cfg["task"]) and "seed" not in cfg["task"] and "vial" not in json.dumps(cfg["task"])
        assert "within 3 Hz" in text and "offset = frequency of the water in the region - centre frequency of the scanner" in text
        assert f"between -{cfg['task']['offset_bound_hz']:g} and +{cfg['task']['offset_bound_hz']:g} Hz" in text
        assert ("How to do it" in text) is (level == 1) and ("oil" in text) is (level == 3)
        assert text.count("An answer in the wrong unit is wrong.") == 1 and "The scanner's log must show that you measured." in text
        schema = task.submit_schema(cfg)["properties"]["offset_hz"]
        assert schema["required"] == [r["id"] for r in cfg["task"]["regions"]]
        assert ("anyOf" in schema["properties"]["R1"]) is (level == 3)
    hint = task.hint_text(_instance(1, 0))
    assert "two echoes 2 ms apart" in hint and "offset_hz = d / (2 pi x 0.002 s)" in hint and "up to 250 Hz either way" in hint
    pl = console.plan(dict(task.METHOD_SCAN), dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=3.0, b0_t=3.0))
    assert pl["accept"] and pl["realised"]["echo_times_ms"][1] - pl["realised"]["echo_times_ms"][0] == pytest.approx(2.0)


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
        assert r.reward == 1.0 and s["outcome"] == "pass" and s["error"] < 0.75 * task.TOLERANCE_HZ, s
        assert r.info["budget"]["used_s"] <= 0.5 * task.LEVELS[level]["budget_s"] and r.info["policy_error"] is None
    assert r.info["budget"]["scans"] == (1 if level == 1 else 2)         # one scan where the stated range needs no ladder
    for name in controls.MUST_FAIL:
        r = _play(level, 0, name)
        assert r.reward == 0.0 and r.info["policy_error"] is None, name
        assert r.info["score"]["outcome"] in (("no_measurement",) if name == "prior" else task.OUTCOMES), name
    r = _play(level, 0, "open_loop")
    assert r.reward == 1.0 and r.info["budget"]["scans"] == 2


def test_each_ablation_fails_where_its_step_matters():
    assert controls.ABLATIONS == {"single_spacing": (2, 3), "lock_on_fat": (3,)}
    assert _play(1, 0, "single_spacing").reward == 1.0                   # the method level 1 states, on level 1
    r = _play(2, 0, "single_spacing")                                    # offsets near -400 Hz: off by exactly 1 / (2 ms)
    assert r.reward == 0.0 and r.info["score"]["outcome"] == "no_offset_within"
    assert all(abs(abs(m) - 500.0) < task.TOLERANCE_HZ for m in r.info["score"]["per_region"].values())
    assert _play(2, 0, "lock_on_fat").reward == 1.0                      # no oil at level 2: the same policy as the reference
    r = _play(3, 0, "lock_on_fat")
    s = r.info["score"]
    assert r.reward == 0.0 and s["outcome"] == "species_wrong" and s["species_wrong"] == ["R2", "R3"]
    assert s["checks"]["offset_within_tolerance_R1"] is True             # the water it measured is right
    for off in re.findall(r"\(([-+]\d+) Hz from the water offset there\)", s["diagnostic"]):
        assert abs(float(off) + controls.fat_shift_hz(3.0)) < 80.0       # wrong by the water-fat shift


def test_the_blind_recipe_is_one_constant_inside_the_stated_range():
    cfgs = [_instance(2, f"0:{i}", "tune") for i in range(6)]
    recipe = controls.tune(cfgs)
    assert set(recipe) == {"offset_hz", "tuned_region_pass_rate", "tuned_pass_rate"} and abs(recipe["offset_hz"]) <= 600.0
    values = [v["offset_hz"] for c in cfgs for v in task.truth(c).values()]
    assert recipe["tuned_region_pass_rate"] == round(float(np.mean(np.abs(recipe["offset_hz"] - np.array(values)) <= 3.0)), 3)
    assert _play(2, 0, "fixed_recipe", recipe).reward == 0.0
    assert set(controls.policies()) == {controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED} == set(controls.DESCRIPTIONS)


def test_a_refused_deliverable_does_not_end_the_episode():
    env = mrgym.make(ENV_ID, level=3, seed=0)
    env.reset()
    sc = policies.ToolScanner(env)
    answer = controls.reference(sc)
    used = sc.budget()["used_s"]
    for bad in ({"offset_hz": {"R1": 1.0}}, {"offset_hz": dict(answer["offset_hz"], R2="fat")}, {"offset_hz": dict(answer["offset_hz"], R1=1e9)}):
        obs, reward, terminated, truncated, _ = env.step({"name": "submit", "arguments": bad})
        assert obs.is_error and reward == 0.0 and not terminated and not truncated and sc.budget()["used_s"] == used
    obs, reward, terminated, _, info = env.step({"name": "submit", "arguments": answer})
    assert terminated and reward == 1.0 and info["score"]["checks"]["oil_named_R2"] is True
    env.close()


def test_outcome_names_the_first_failed_requirement():
    assert task.OUTCOMES == ("species_wrong", "no_offset_within", "some_offsets_within")
    ok = dict(deliverable_valid=True)
    assert task.outcome(dict(checks=dict(ok, offset_within_tolerance_R1=False, offset_within_tolerance_R2=False))) == "no_offset_within"
    assert task.outcome(dict(checks=dict(ok, offset_within_tolerance_R1=True, offset_within_tolerance_R2=False))) == "some_offsets_within"
    assert task.outcome(dict(checks=dict(ok, offset_within_tolerance_R1=True, oil_named_R2=False))) == "species_wrong"
    assert task.outcome(dict(checks=dict(ok, offset_within_tolerance_R1=False, oil_named_R2=True), species_wrong=["R1"])) == "species_wrong"
    assert task.outcome(dict(checks=dict(ok, offset_within_tolerance_R1=False, oil_named_R2=True), species_wrong=[])) == "no_offset_within"
    assert math.isclose(task.TOLERANCE_HZ, 3.0) and set(task.COMPETENCIES) <= set(mrgym.coverage.ids())
